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

import os
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
    collision_decision=None,
    bpm_preview=None,
    bpm_run_wide=None,
):
    """Auto-route batch: per light, inspect -> auto-route -> calibrate -> write.

    Mirrors ``calibrate_batch`` (same ``BatchItem`` stream, transactional output
    writer and last-written manifest) but resolves each light's scientific route
    automatically instead of taking an explicit ``CalibrationRequest``. Each
    light may therefore resolve to a different route; a light with no READY route
    is reported as a ``FAILED`` item (never silently routed).
    """
    from zecalibrator import _version
    from zecalibrator.api.v1.batch import _CalibratedOne, _materialize_one, _write_output
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
    decided_policy = None
    context_slot = _PreparedContextSlot()
    run_wide = bpm_run_wide is not None

    def _process(idx, frame, frame_id):
        nonlocal decided_policy
        replace_existing = False
        try:
            inspection_result, decoded_light = _decode_and_inspect(frame, token=token, obs=None)
        except OperationCancelled:
            raise
        except InvalidRequestError as exc:
            return _CalibratedOne(index=idx, disposition="FAILED", input_identity=None,
                                  plan_id=None, result=None, plan=None, warnings=(),
                                  reason_code="INVALID_SOURCE", reason_details=str(exc),
                                  composition=None), replace_existing
        except Exception as exc:  # noqa: BLE001
            return _CalibratedOne(index=idx, disposition="FAILED", input_identity=None,
                                  plan_id=None, result=None, plan=None, warnings=(),
                                  reason_code="INSPECT_ERROR", reason_details=str(exc),
                                  composition=None), replace_existing
        if inspection_result.operation_status == "CANCELLED":
            raise OperationCancelled()
        if inspection_result.operation_status == "FAILED":
            return _CalibratedOne(index=idx, disposition="FAILED", input_identity=None,
                                  plan_id=None, result=None, plan=None, warnings=(),
                                  reason_code=inspection_result.reason_code or "INSPECT_FAILED",
                                  reason_details=inspection_result.details,
                                  composition=None), replace_existing
        inspection = inspection_result.inspection

        if inspection.domain_finding != "raw":
            return _CalibratedOne(index=idx, disposition="SKIPPED",
                                  input_identity=inspection.identity, plan_id=None,
                                  result=None, plan=None, warnings=(),
                                  reason_code="NON_RAW_DOMAIN",
                                  reason_details=f"domain_finding={inspection.domain_finding}",
                                  composition=None), replace_existing

        try:
            light = light_constraints_from_sensor_metadata(inspection.metadata)
            resolution = _resolve_route(light, library.snapshot, policy)
        except OperationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            return _CalibratedOne(index=idx, disposition="FAILED",
                                  input_identity=inspection.identity, plan_id=None,
                                  result=None, plan=None, warnings=(),
                                  reason_code="RESOLVE_ERROR", reason_details=str(exc),
                                  composition=None), replace_existing

        plan = resolution.plan
        if plan is None:
            return _CalibratedOne(index=idx, disposition="FAILED",
                                  input_identity=inspection.identity, plan_id=None,
                                  result=None, plan=None, warnings=(),
                                  reason_code=_AUTO_ROUTE_TO_MATCH[resolution.outcome],
                                  reason_details="; ".join(r.code for r in resolution.reasons),
                                  composition=None), replace_existing

        if destination is not None and collision_decision is not None:
            planned = _batch._output_path_for(inspection.identity, plan.plan_id, destination)
            if os.path.exists(planned):
                if decided_policy is None:
                    decided_policy = collision_decision(planned)
                policy_value = decided_policy
                if policy_value == "skip":
                    return _CalibratedOne(index=idx, disposition="SKIPPED",
                                          input_identity=inspection.identity,
                                          plan_id=plan.plan_id, result=None, plan=plan,
                                          warnings=(), reason_code="DESTINATION_EXISTS",
                                          reason_details=f"output already exists: {planned}",
                                          composition=None), replace_existing
                if policy_value != "overwrite":
                    # "cancel" or any malformed/garbled decision fails safe to
                    # cancel (overwrite/skip are never selected implicitly).
                    raise OperationCancelled()
                # policy_value == "overwrite" -> proceed with transactional replace
                replace_existing = True

        try:
            result = _calibrate_frame_impl(frame, plan, ExecutionOptions(), token=token, obs=None, slot=context_slot, decoded_light=decoded_light, bpm_preview=bpm_preview)
        except OperationCancelled:
            raise
        except InvalidRequestError as exc:
            return _CalibratedOne(index=idx, disposition="FAILED",
                                  input_identity=inspection.identity, plan_id=plan.plan_id,
                                  result=None, plan=plan, warnings=(),
                                  reason_code="PLAN_SOURCE_MISMATCH", reason_details=str(exc),
                                  composition=None), replace_existing
        except Exception as exc:  # noqa: BLE001
            return _CalibratedOne(index=idx, disposition="FAILED",
                                  input_identity=inspection.identity, plan_id=plan.plan_id,
                                  result=None, plan=plan, warnings=(),
                                  reason_code="CALIBRATE_ERROR", reason_details=str(exc),
                                  composition=None), replace_existing

        if result.status == "CANCELLED":
            raise OperationCancelled()

        disposition = batch_disposition(result.status)
        if result.status == "FAILED":
            return _CalibratedOne(index=idx, disposition="FAILED",
                                  input_identity=inspection.identity, plan_id=plan.plan_id,
                                  result=None, plan=plan, warnings=result.warnings,
                                  reason_code=result.reason_code, reason_details="",
                                  composition=None), replace_existing

        # LOT 3: record the successful engine calibration into the run-wide seam.
        if bpm_run_wide is not None:
            bpm_run_wide.record(frame_id, result._engine, plan.light_constraints)

        return _CalibratedOne(index=idx, disposition=disposition,
                              input_identity=inspection.identity, plan_id=plan.plan_id,
                              result=result, plan=plan, warnings=result.warnings,
                              reason_code=None, reason_details="",
                              composition=dict(plan.composition.to_dict()) if plan.composition is not None else None), replace_existing

    emit_batch_progress(obs, "batch_start", 0, total)
    try:
        if run_wide:
            # LOT 3 two-time structure: calibrate every frame first, then apply
            # the run-wide BPM plan once, then write corrected outputs.
            staged: list = []
            for idx, frame in enumerate(frame_list):
                token.raise_if_cancelled()
                frame_id = frame_display_id(frame)
                emit_batch_progress(obs, "frame_start", idx, total, frame_id=frame_id)
                one, replace_existing = _process(idx, frame, frame_id)
                staged.append((frame_id, one, replace_existing))
                emit_batch_progress(obs, "frame_complete", idx + 1, total, frame_id=frame_id)
            bpm_run_wide.finalize()
            for frame_id, one, replace_existing in staged:
                token.raise_if_cancelled()
                prepared = bpm_run_wide.prepared(frame_id) if one.result is not None else None
                item = _materialize_one(one, destination, token, replace_existing=replace_existing, prepared=prepared)
                manifest_inputs.append({"index": one.index, "identity": _identity_to_dict(item.input_identity)})
                manifest_items.append(item.to_dict())
                yield item
        else:
            for idx, frame in enumerate(frame_list):
                token.raise_if_cancelled()
                frame_id = frame_display_id(frame)
                emit_batch_progress(obs, "frame_start", idx, total, frame_id=frame_id)

                one, replace_existing = _process(idx, frame, frame_id)
                item = _materialize_one(one, destination, token, replace_existing=replace_existing)

                manifest_inputs.append({"index": one.index, "identity": _identity_to_dict(item.input_identity)})
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
