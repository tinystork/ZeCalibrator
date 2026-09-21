"""Private auto-route facade for the Standard UX (NOT part of the public contract).

The Standard workflow expresses only user intent ("use the compatible masters I
supplied"); ZeCalibrator resolves the scientific calibration route automatically.
This private module (underscore-prefixed, never listed in ``api.v1.__all__``)
wraps the application-layer resolver (``zecalibrator.application.routes``) and
mirrors ``calibrate_batch`` with per-light auto-route, so the GUI worker can use
it while importing only ``zecalibrator.api.v1`` (the public boundary).

The resolver itself remains application-layer; it is NOT exported from
``zecalibrator.api.v1.__init__``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from zecalibrator.application.library import (
    LibrarySnapshot,
    light_constraints_from_sensor_metadata,
)
from zecalibrator.application.routes import resolve_route as _resolve_route
from zecalibrator.core.plans import MatchPolicy

__all__ = ["auto_route_batch", "light_constraints_from_sensor_metadata", "resolve_route"]


def resolve_route(light, snapshot: LibrarySnapshot, policy: MatchPolicy):
    """Wrap the application-layer auto-route resolver (metadata-only)."""
    return _resolve_route(light, snapshot, policy)


# Outcome vocabulary the batch/presentation layer already understands.
_AUTO_ROUTE_TO_MATCH = {
    "READY": "MATCHED",
    "NEEDS_ATTENTION": "NO_MATCH",
    "AMBIGUOUS": "AMBIGUOUS",
}


def auto_route_batch(
    frames: Iterable,
    library,
    policy: MatchPolicy,
    options=None,
    *,
    cancel=None,
    progress=None,
):
    """Auto-route batch: per light, inspect -> auto-route -> calibrate -> write.

    Mirrors ``calibrate_batch`` (same ``BatchItem`` stream, transactional output
    writer and last-written manifest) but resolves each light's scientific route
    automatically instead of taking an explicit ``CalibrationRequest``. Each
    light may therefore resolve to a different route; a light with no READY route
    is reported as a ``FAILED`` item (never silently routed).
    """
    from zecalibrator import _version
    from zecalibrator.api.v1.batch import _write_output
    from zecalibrator.api.v1.calibration import _calibrate_frame_impl, _PreparedContextSlot
    from zecalibrator.api.v1.frames import _decode_and_inspect
    from zecalibrator.api.v1.models import (
        BatchItem,
        BatchOptions,
        ExecutionOptions,
        _identity_to_dict,
    )
    from zecalibrator.api.v1.errors import InvalidRequestError
    from zecalibrator.application.batch import (
        BATCH_OPERATION_ID,
        batch_disposition,
        derive_batch_status,
        emit_batch_progress,
        frame_display_id,
        new_batch_id,
        ordered_frames,
    )
    from zecalibrator.application.cancellation import CancellationToken, OperationCancelled
    from zecalibrator.core.plans import (
        PROVENANCE_SCHEMA_VERSION,
        SCIENCE_CONTRACT_VERSION,
    )
    from zecalibrator.io.output_writer import NoClobberViolation
    # The manifest builder/writer are reached through the ``zecalibrator.api.v1.batch``
    # module attributes at call time so the production/test seam stays patchable.
    import zecalibrator.api.v1.batch as _batch

    if options is None:
        options = BatchOptions()
    if not isinstance(options, BatchOptions):
        raise InvalidRequestError("options must be a BatchOptions")
    frame_list = ordered_frames(frames)

    token = cancel if cancel is not None else CancellationToken()
    from zecalibrator.api.v1 import _io

    obs = _io.normalize_progress(progress, BATCH_OPERATION_ID)

    total = len(frame_list)
    batch_id = options.batch_id or new_batch_id()
    destination = options.destination

    manifest_inputs = []
    manifest_items = []
    cancelled = False
    context_slot = _PreparedContextSlot()

    def _process(idx, frame):
        nonlocal cancelled
        try:
            inspection_result, decoded_light = _decode_and_inspect(frame, token=token, obs=None)
        except OperationCancelled:
            raise
        except InvalidRequestError as exc:
            return BatchItem(index=idx, disposition="FAILED", input_identity=None,
                             plan_id=None, reason_code="INVALID_SOURCE", reason_details=str(exc))
        except Exception as exc:  # noqa: BLE001
            return BatchItem(index=idx, disposition="FAILED", input_identity=None,
                             plan_id=None, reason_code="INSPECT_ERROR", reason_details=str(exc))
        if inspection_result.operation_status == "CANCELLED":
            raise OperationCancelled()
        if inspection_result.operation_status == "FAILED":
            return BatchItem(index=idx, disposition="FAILED", input_identity=None,
                             plan_id=None,
                             reason_code=inspection_result.reason_code or "INSPECT_FAILED",
                             reason_details=inspection_result.details)
        inspection = inspection_result.inspection

        if inspection.domain_finding != "raw":
            return BatchItem(index=idx, disposition="SKIPPED",
                             input_identity=inspection.identity, plan_id=None,
                             reason_code="NON_RAW_DOMAIN",
                             reason_details=f"domain_finding={inspection.domain_finding}")

        try:
            light = light_constraints_from_sensor_metadata(inspection.metadata)
            resolution = _resolve_route(light, library.snapshot, policy)
        except OperationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            return BatchItem(index=idx, disposition="FAILED",
                             input_identity=inspection.identity, plan_id=None,
                             reason_code="RESOLVE_ERROR", reason_details=str(exc))

        plan = resolution.plan
        if plan is None:
            return BatchItem(
                index=idx, disposition="FAILED", input_identity=inspection.identity,
                plan_id=None, reason_code=_AUTO_ROUTE_TO_MATCH[resolution.outcome],
                reason_details="; ".join(r.code for r in resolution.reasons),
            )

        try:
            result = _calibrate_frame_impl(frame, plan, ExecutionOptions(), token=token, obs=None, slot=context_slot, decoded_light=decoded_light)
        except OperationCancelled:
            raise
        except InvalidRequestError as exc:
            return BatchItem(index=idx, disposition="FAILED",
                             input_identity=inspection.identity, plan_id=plan.plan_id,
                             reason_code="PLAN_SOURCE_MISMATCH", reason_details=str(exc))
        except Exception as exc:  # noqa: BLE001
            return BatchItem(index=idx, disposition="FAILED",
                             input_identity=inspection.identity, plan_id=plan.plan_id,
                             reason_code="CALIBRATE_ERROR", reason_details=str(exc))

        if result.status == "CANCELLED":
            raise OperationCancelled()

        disposition = batch_disposition(result.status)
        if result.status == "FAILED":
            return BatchItem(index=idx, disposition="FAILED",
                             input_identity=inspection.identity, plan_id=plan.plan_id,
                             reason_code=result.reason_code, warnings=result.warnings)

        if destination is None:
            return BatchItem(index=idx, disposition=disposition,
                             input_identity=inspection.identity, plan_id=plan.plan_id,
                             result=result, output=None, warnings=result.warnings)

        try:
            output = _write_output(result, inspection.identity, plan.plan_id, destination, token)
        except OperationCancelled:
            raise
        except NoClobberViolation as exc:
            return BatchItem(index=idx, disposition="FAILED",
                             input_identity=inspection.identity, plan_id=plan.plan_id,
                             reason_code="OUTPUT_COLLISION", reason_details=str(exc),
                             warnings=result.warnings)
        except Exception as exc:  # noqa: BLE001
            return BatchItem(index=idx, disposition="FAILED",
                             input_identity=inspection.identity, plan_id=plan.plan_id,
                             reason_code="OUTPUT_ERROR", reason_details=str(exc),
                             warnings=result.warnings)
        return BatchItem(index=idx, disposition=disposition,
                         input_identity=inspection.identity, plan_id=plan.plan_id,
                         result=None, output=output, warnings=result.warnings)

    emit_batch_progress(obs, "batch_start", 0, total)
    try:
        for idx, frame in enumerate(frame_list):
            token.raise_if_cancelled()
            frame_id = frame_display_id(frame)
            emit_batch_progress(obs, "frame_start", idx, total, frame_id=frame_id)

            item = _process(idx, frame)

            manifest_inputs.append({"index": idx, "identity": _identity_to_dict(item.input_identity)})
            manifest_items.append(item.to_dict())
            emit_batch_progress(obs, "frame_complete", idx + 1, total, frame_id=frame_id)
            yield item

        emit_batch_progress(obs, "complete", total, total)
    except OperationCancelled:
        cancelled = True
        raise
    except GeneratorExit:
        cancelled = True
        raise
    finally:
        if destination is not None:
            batch_status = derive_batch_status(
                [m["disposition"] for m in manifest_items], cancelled=cancelled, total=total
            )
            manifest = _batch.build_batch_manifest(
                batch_id=batch_id,
                operation_id=BATCH_OPERATION_ID,
                api_version="1.0",
                product_version=_version.__version__,
                science_contract=SCIENCE_CONTRACT_VERSION,
                matching_policy=policy.version,
                provenance_schema=PROVENANCE_SCHEMA_VERSION,
                decoder_version="1.0",
                commit_state="CANCELLED" if cancelled else "COMMITTED",
                batch_status=batch_status,
                destination=destination,
                inputs=manifest_inputs,
                items=manifest_items,
            )
            try:
                _batch.write_batch_manifest(_batch.manifest_path(destination, batch_id), manifest)
            except (OSError, _batch.BatchManifestError) as exc:
                if not cancelled:
                    if isinstance(exc, _batch.BatchManifestError):
                        raise
                    raise _batch.BatchManifestError(f"failed to write batch manifest: {exc}") from exc
