"""Public batch capability: ``calibrate_batch`` and ``index_library``.

``calibrate_batch`` is a bounded iterator over per-item results that reuses the
frozen G5 unitary science (``inspect_frame`` / ``resolve_calibration`` /
``calibrate_frame``) so batch science is always unitary G5 science — master
selection, plan, DQ, float32 arithmetic, flat normalization, validation,
matching, temperature and geometry are never changed. It only adds
orchestration: deterministic ordering, per-item dispositions, cooperative
cancellation checkpoints and monotonic progress, plus (standalone mode) the
transactional output writer and last-committed batch manifest.

``index_library`` is the public library-indexing path (scan root + build/publish
a versioned index) required by the CLI ``index`` command. Descriptors are built
from FITS headers + explicit evidence-backed :class:`MasterImportSpec`
declarations (SYNTH-BASE-1), never a real header->descriptor importer.

The ``calibrate_batch`` capability is advertised in ``CAPABILITIES`` since its
G6 acceptance (admitted via ZC-P6-M2-BATCH-CAPABILITY-ADMISSION).
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
from typing import Iterable, Mapping, Optional

from zecalibrator import _version
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
from zecalibrator.core.digests import canonical_json, sha256_hex
from zecalibrator.core.plans import SCIENCE_CONTRACT_VERSION
from zecalibrator.io.batch_manifest import (
    BatchManifestError,
    build_batch_manifest,
    manifest_path,
    write_batch_manifest,
)
from zecalibrator.io.master_source import FilesystemSource
from zecalibrator.io.output_writer import (
    NoClobberViolation,
    write_standalone_output,
)

from . import _io
from .calibration import calibrate_frame
from .errors import InvalidRequestError, LibraryClosedError
from .frames import inspect_frame
from .matching import resolve_calibration
from .models import (
    ArrayFrameSource,
    BatchItem,
    BatchOptions,
    BatchOutputRecord,
    CalibrationRequest,
    ExecutionOptions,
    FitsFrameSource,
    IndexLibraryResult,
    LibraryHandle,
    LibrarySpec,
    MasterImportSpec,
    MatchPolicy,
    PROVENANCE_SCHEMA,
    _identity_to_dict,
)

INDEX_OPERATION_ID = "zecalibrator-index-library"

_DECODER_VERSION = "1.0"
_PROCESSING_SOURCES = ("synthetic_fixture", "user_import", "observed")


def _validate_batch_args(frames, request, library, policy, options):
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Iterable):
        raise InvalidRequestError("frames must be an iterable of FrameSource")
    if not isinstance(request, CalibrationRequest):
        raise InvalidRequestError("request must be a CalibrationRequest")
    if not isinstance(library, LibraryHandle):
        raise InvalidRequestError("library must be a LibraryHandle")
    if not isinstance(policy, MatchPolicy):
        raise InvalidRequestError("policy must be a MatchPolicy")
    if options is None:
        options = BatchOptions()
    if not isinstance(options, BatchOptions):
        raise InvalidRequestError("options must be a BatchOptions")
    return options


def _process_one(idx, frame, request, library, policy, destination, token):
    """Process one frame into a :class:`BatchItem` (raises on cancellation)."""
    try:
        inspection_result = inspect_frame(frame, cancel=token)
    except OperationCancelled:
        raise
    except InvalidRequestError as exc:
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=None, plan_id=None,
            reason_code="INVALID_SOURCE", reason_details=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - per-item operational failure
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=None, plan_id=None,
            reason_code="INSPECT_ERROR", reason_details=str(exc),
        )

    if inspection_result.operation_status == "CANCELLED":
        raise OperationCancelled()
    if inspection_result.operation_status == "FAILED":
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=None, plan_id=None,
            reason_code=inspection_result.reason_code or "INSPECT_FAILED",
            reason_details=inspection_result.details,
        )
    inspection = inspection_result.inspection

    if inspection.domain_finding != "raw":
        # SKIPPED is reserved: a frame intentionally not processed (non-raw /
        # domain-unsupported). The current strict G5 decoder rejects non-raw
        # FITS at inspect (FAILED) and ``ArrayFrameSource`` requires a raw
        # domain, so this branch has no live witness today; it is retained for
        # the frozen status set (ARCHITECTURE §3.3).
        return BatchItem(
            index=idx, disposition="SKIPPED", input_identity=inspection.identity,
            plan_id=None, reason_code="NON_RAW_DOMAIN",
            reason_details=f"domain_finding={inspection.domain_finding}",
        )

    try:
        resolve_result = resolve_calibration(
            inspection, request, library, policy, cancel=token
        )
    except OperationCancelled:
        raise
    except LibraryClosedError as exc:
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=None, reason_code="LIBRARY_CLOSED", reason_details=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=None, reason_code="RESOLVE_ERROR", reason_details=str(exc),
        )

    if resolve_result.operation_status == "CANCELLED":
        raise OperationCancelled()
    if resolve_result.operation_status == "FAILED":
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=None, reason_code=resolve_result.reason_code or "RESOLVE_FAILED",
            reason_details=resolve_result.details,
        )
    if resolve_result.outcome != "MATCHED" or resolve_result.plan is None:
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=None, reason_code=resolve_result.outcome or "NO_MATCH",
            reason_details=resolve_result.details,
        )
    plan = resolve_result.plan

    try:
        result = calibrate_frame(frame, plan, ExecutionOptions(), cancel=token)
    except OperationCancelled:
        raise
    except InvalidRequestError as exc:
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=plan.plan_id, reason_code="PLAN_SOURCE_MISMATCH",
            reason_details=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=plan.plan_id, reason_code="CALIBRATE_ERROR",
            reason_details=str(exc),
        )

    if result.status == "CANCELLED":
        raise OperationCancelled()

    disposition = batch_disposition(result.status)

    if result.status == "FAILED":
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=plan.plan_id, reason_code=result.reason_code,
            warnings=result.warnings,
        )

    if destination is None:
        return BatchItem(
            index=idx, disposition=disposition, input_identity=inspection.identity,
            plan_id=plan.plan_id, result=result, output=None,
            warnings=result.warnings,
        )

    try:
        output = _write_output(result, inspection.identity, plan.plan_id, destination, token)
    except OperationCancelled:
        raise
    except NoClobberViolation as exc:
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=plan.plan_id, reason_code="OUTPUT_COLLISION",
            reason_details=str(exc), warnings=result.warnings,
        )
    except Exception as exc:  # noqa: BLE001
        return BatchItem(
            index=idx, disposition="FAILED", input_identity=inspection.identity,
            plan_id=plan.plan_id, reason_code="OUTPUT_ERROR",
            reason_details=str(exc), warnings=result.warnings,
        )
    return BatchItem(
        index=idx, disposition=disposition, input_identity=inspection.identity,
        plan_id=plan.plan_id, result=None, output=output,
        warnings=result.warnings,
    )


def _write_output(result, input_identity, plan_id, destination, token):
    from zecalibrator.io.output_writer import output_logical_id

    logical_id = output_logical_id(_identity_to_dict(input_identity), plan_id)
    header_fields = {
        "ZECALCAL": "zecalibrator",
        "HIERARCH ZECALSCHEMA": PROVENANCE_SCHEMA,
        "HIERARCH ZECALPLAN": plan_id[:16],
        "HIERARCH ZECALSTAT": result.status,
    }
    record = write_standalone_output(
        result.data,
        result.mask,
        result.provenance.to_dict(),
        input_identity=_identity_to_dict(input_identity),
        plan_id=plan_id,
        destination=destination,
        status=result.status,
        header_fields=header_fields,
        check_cancelled=token.raise_if_cancelled,
    )
    return BatchOutputRecord(
        path=record.path,
        logical_id=record.logical_id,
        science_digest=record.science_digest,
        whole_file_sha256=record.whole_file_sha256,
        size_bytes=record.size_bytes,
        committed=record.committed,
    )


def _batch_generator(frames, request, library, policy, options, token, obs):
    total = len(frames)
    batch_id = options.batch_id or new_batch_id()
    destination = os.fspath(options.destination) if options.destination is not None else None

    manifest_inputs = []
    manifest_items = []
    cancelled = False

    emit_batch_progress(obs, "batch_start", 0, total)
    try:
        for idx, frame in enumerate(frames):
            token.raise_if_cancelled()
            frame_id = frame_display_id(frame)
            emit_batch_progress(obs, "frame_start", idx, total, frame_id=frame_id)

            item = _process_one(idx, frame, request, library, policy, destination, token)

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
            manifest = build_batch_manifest(
                batch_id=batch_id,
                operation_id=BATCH_OPERATION_ID,
                api_version="1.0",
                product_version=_version.__version__,
                science_contract=SCIENCE_CONTRACT_VERSION,
                matching_policy=policy.version,
                provenance_schema=PROVENANCE_SCHEMA,
                decoder_version=_DECODER_VERSION,
                commit_state="CANCELLED" if cancelled else "COMMITTED",
                batch_status=batch_status,
                destination=destination,
                inputs=manifest_inputs,
                items=manifest_items,
            )
            try:
                write_batch_manifest(manifest_path(destination, batch_id), manifest)
            except (OSError, BatchManifestError) as exc:
                # A manifest-write failure is a typed, structured batch failure:
                # never a bare private OSError across the public boundary, and
                # never a discarded already-yielded item stream (the items were
                # already produced; only the manifest is missing).
                if not cancelled:
                    if isinstance(exc, BatchManifestError):
                        raise
                    raise BatchManifestError(
                        f"failed to write batch manifest: {exc}"
                    ) from exc


def calibrate_batch(
    frames,
    request,
    library,
    policy,
    options=None,
    *,
    cancel=None,
    progress=None,
):
    """Return a bounded iterator over per-item batch results.

    Validation (bad ``frames``/``request``/``library``/``policy``/``options``)
    raises :class:`InvalidRequestError` eagerly, including a standalone
    ``destination`` that is not an existing directory (never auto-created).
    Anticipated per-item operational failures are yielded as ``FAILED`` /
    ``SKIPPED`` dispositions; cooperative cancellation raises
    :class:`OperationCancelled` (no committed partial output for the in-flight
    item). Standalone mode commits each output transactionally and writes the
    batch manifest last; a manifest-write failure raises the typed
    :class:`BatchManifestError` (never a bare ``OSError``) after the per-item
    results were already yielded. A pre-cancelled standalone batch still writes
    a zero-item ``CANCELLED`` manifest (truthful partial record).
    """
    options = _validate_batch_args(frames, request, library, policy, options)
    frame_list = ordered_frames(frames)
    for frame in frame_list:
        if not isinstance(frame, (FitsFrameSource, ArrayFrameSource)):
            raise InvalidRequestError("every frame must be a FitsFrameSource or ArrayFrameSource")

    token = cancel if cancel is not None else CancellationToken()
    obs = _io.normalize_progress(progress, BATCH_OPERATION_ID)
    return _batch_generator(frame_list, request, library, policy, options, token, obs)


# ---------------------------------------------------------------------------
# index_library — public library-indexing path
# ---------------------------------------------------------------------------
def _to_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "to_mapping"):
        return _to_jsonable(value.to_mapping())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(
            {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        )
    return str(value)


def _spec_to_index_dict(spec: MasterImportSpec) -> dict:
    return {
        "path": spec.path,
        "master_type": spec.master_type,
        "hdu": spec.hdu,
        "mask_path": spec.mask_path,
        "bias_state": spec.bias_state,
        "flat_form": spec.flat_form,
        "normalization_algorithm": spec.normalization_algorithm,
        "normalization_scalars": _to_jsonable(spec.normalization_scalars),
        "normalization_provenance": _to_jsonable(spec.normalization_provenance),
        "validity_evidence": _to_jsonable(spec.validity_evidence),
        "processing_provenance": _to_jsonable(spec.processing_provenance),
        "declaration": _to_jsonable(spec.declaration),
    }


def _default_revision(imports) -> str:
    payload = canonical_json(
        [_spec_to_index_dict(s) for s in sorted(imports, key=lambda s: s.path)]
    )
    return sha256_hex(payload.encode("utf-8"))[:16]


def _fits_shape(source: FilesystemSource, path: str, hdu) -> tuple:
    cards = source.read_header(path, hdu=hdu)
    header = {c.keyword: c.value for c in cards}
    naxis1 = int(header.get("NAXIS1", 0))
    naxis2 = int(header.get("NAXIS2", 0))
    if naxis1 < 1 or naxis2 < 1:
        raise ValueError(f"FITS {path} has no supported 2D plane (NAXIS1/NAXIS2)")
    return (naxis2, naxis1)


def _descriptor_from_spec(
    spec: MasterImportSpec, *, shape, content_sha256, size_bytes, mask_identity
):
    from zecalibrator.core.descriptors import (
        Acquisition,
        AcquisitionProfileEvidence,
        DetectorIdentity,
        MasterDescriptor,
        ProcessingProvenance,
        ValidityEvidence,
    )
    from zecalibrator.core.geometry import Geometry

    decl = spec.declaration

    geometry = Geometry(
        shape=shape,
        sensor_dimensions=decl.sensor_dimensions,
        binning=decl.binning,
        roi_origin=decl.roi_origin,
        roi_extent=shape,
        orientation=decl.orientation,
        cfa_phase=decl.cfa_phase,
    )

    detector = DetectorIdentity(
        detector_instance_id=decl.detector_instance_id or "unknown",
        detector_model=decl.detector_model,
        serial=None,
    )

    acquisition = Acquisition(
        gain=decl.gain,
        offset=decl.offset,
        readout_mode=decl.readout_mode,
        adc_mode=decl.adc_mode,
        temperature_c=decl.temperature_c,
        exposure_s=decl.exposure_s,
        saturation_limit_adu=decl.saturation_limit_adu,
        saturation_evidence=decl.saturation_evidence,
    )

    if spec.bias_state is not None:
        bias_state = spec.bias_state
    elif spec.master_type in ("bias", "flat"):
        bias_state = "not_applicable"
    else:
        bias_state = "unknown"

    flat_form = spec.flat_form if spec.master_type == "flat" else None
    if spec.master_type == "flat" and flat_form == "normalized_response":
        pixel_domain = "normalized_response"
        physical_units = "dimensionless"
    else:
        pixel_domain = "sensor_adu"
        physical_units = "ADU"

    proc_source = decl.source if decl.source in _PROCESSING_SOURCES else "user_import"
    if spec.processing_provenance is not None:
        processing = spec.processing_provenance
    else:
        processing = ProcessingProvenance(
            source=proc_source,
            acquisition_profile=AcquisitionProfileEvidence(
                source=decl.source,
                identity=decl.identity,
                version=decl.version,
                bias_exposure_max_s=decl.bias_exposure_max_s,
                short_flat_profile=decl.short_flat_profile,
            ),
        )

    validity = spec.validity_evidence or ValidityEvidence(
        saturation_limit_known=(decl.saturation_evidence == "qualified"),
    )

    return MasterDescriptor(
        master_type=spec.master_type,
        pixel_domain=pixel_domain,
        physical_units=physical_units,
        bias_state=bias_state,
        geometry=geometry,
        detector=detector,
        acquisition=acquisition,
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=spec.hdu,
        mask_identity=mask_identity,
        processing_provenance=processing,
        validity_evidence=validity,
        flat_form=flat_form,
        normalization_algorithm=spec.normalization_algorithm,
        normalization_scalars=spec.normalization_scalars,
        optical_train_id=decl.optical_train_id,
        filter=decl.filter,
    )


def _build_candidate(path, spec, source, mask_path):
    from zecalibrator.core.descriptors import DescriptorSnapshot
    from zecalibrator.core.plans import Candidate, FitsFileLocator, MaskPayloadLocator

    loc = FitsFileLocator(path=path, hdu=spec.hdu)
    ident = source.image_identity(loc)
    mask_loc = MaskPayloadLocator(path=mask_path)
    mask_identity = source.mask_identity(mask_loc)
    shape = _fits_shape(source, path, spec.hdu)
    desc = _descriptor_from_spec(
        spec, shape=shape, content_sha256=ident.content_sha256,
        size_bytes=ident.size_bytes, mask_identity=mask_identity,
    )
    return Candidate(
        candidate_id=desc.descriptor_id,
        descriptor=desc,
        descriptor_snapshot=DescriptorSnapshot(desc),
        locators=(loc,),
        mask_locator=mask_loc,
    )


def index_library(
    spec: LibrarySpec,
    imports: Iterable[MasterImportSpec],
    *,
    revision: Optional[str] = None,
    cancel=None,
    progress=None,
) -> IndexLibraryResult:
    """Scan a library root and build/publish a versioned index.

    Descriptors are built from FITS headers + explicit evidence-backed
    :class:`MasterImportSpec` declarations; no real header->descriptor inference
    is performed. ``imports`` entries with a relative ``path`` are resolved
    against ``spec.root``. When ``revision`` is omitted, the default revision
    label is derived from the *specs* (paths/declarations), not from file
    content; the index itself still stores per-candidate content hashes (so two
    runs over changed bytes with unchanged specs share a label while each
    candidate keeps its own content identity).
    """
    if not isinstance(spec, LibrarySpec):
        raise InvalidRequestError("spec must be a LibrarySpec")
    if isinstance(imports, (str, bytes)) or not isinstance(imports, Iterable):
        raise InvalidRequestError("imports must be an iterable of MasterImportSpec")
    import_list = list(imports)
    for imp in import_list:
        if not isinstance(imp, MasterImportSpec):
            raise InvalidRequestError("every import must be a MasterImportSpec")

    token = cancel if cancel is not None else CancellationToken()
    obs = _io.normalize_progress(progress, INDEX_OPERATION_ID)

    if token.is_cancelled():
        return IndexLibraryResult(operation_status="CANCELLED", reason_code="CANCELLED")

    root = os.path.abspath(os.fspath(spec.root))
    spec_by_path = {}
    mask_by_path = {}
    for imp in import_list:
        if imp.mask_path is None:
            raise InvalidRequestError(f"MasterImportSpec {imp.path!r} requires a mask_path")
        p = imp.path
        if not os.path.isabs(p):
            p = os.path.join(root, p)
        p = os.path.abspath(p)
        m = imp.mask_path
        if not os.path.isabs(m):
            m = os.path.join(root, m)
        mask_by_path[p] = os.path.abspath(m)
        spec_by_path[p] = imp
    paths = sorted(spec_by_path.keys())

    rev = revision or _default_revision(import_list)
    source = FilesystemSource()

    _io.emit_progress(obs, INDEX_OPERATION_ID, "start", 0, 2)
    try:
        from zecalibrator.io.library_index import LibraryIndex, LibraryIndexError

        token.raise_if_cancelled()
        index = LibraryIndex(spec.resolved_index_path).open(initialize=True)
        try:
            result = index.scan_and_publish(
                paths,
                build_candidate=lambda p: _build_candidate(p, spec_by_path[p], source, mask_by_path[p]),
                source=source,
                revision=rev,
                cancel=token,
                progress=None,
            )
        finally:
            index.close()
    except OperationCancelled:
        return IndexLibraryResult(operation_status="CANCELLED", reason_code="CANCELLED")
    except LibraryIndexError as exc:
        return IndexLibraryResult(
            operation_status="FAILED", reason_code="LIBRARY_ERROR", details=str(exc)
        )
    except (sqlite3.Error, OSError, ValueError) as exc:
        return IndexLibraryResult(
            operation_status="FAILED", reason_code="INDEX_FAILED", details=str(exc)
        )

    status = result.status
    _io.emit_progress(obs, INDEX_OPERATION_ID, "complete", 2, 2)
    return IndexLibraryResult(
        operation_status=status,
        revision=result.revision if status == "COMPLETED" else None,
        candidate_count=result.candidate_count,
        diagnostics=result.diagnostics,
    )


__all__ = ["BatchManifestError", "calibrate_batch", "index_library"]
