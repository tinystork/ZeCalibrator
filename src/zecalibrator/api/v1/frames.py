"""Public frame inspection: ``FrameSource`` decoding and ``inspect_frame``.

``inspect_frame`` wraps the strict G3 FITS decoder (or an already-decoded array
source) into an immutable :class:`~zecalibrator.api.v1.models.FrameInspection`
without any pixel mutation. The FITS bytes are read once, hashed, and decoded
from the same bytes (no unchecked reopen between hash and decode).
"""

from __future__ import annotations

from typing import Optional

from zecalibrator.application.cancellation import CancellationToken, OperationCancelled

from . import _io
from .errors import DecodeError, InvalidRequestError, PrecisionRefusalError, SourceError
from .models import (
    ArrayFrameSource,
    ArrayInputIdentity,
    FitsFrameSource,
    FitsInputIdentity,
    FrameInspection,
    FrameSource,
    InspectResult,
    InputIdentity,
)

OPERATION_ID = "zecalibrator-inspect"


def _domain_finding(metadata) -> str:
    value = getattr(metadata, "raw_domain_declaration", None)
    return value if value in ("raw", "processed", "unknown") else "unsupported"


def inspect_frame(
    source: FrameSource,
    *,
    declaration=None,
    cancel=None,
    progress=None,
) -> InspectResult:
    """Decode ``source`` into an immutable :class:`FrameInspection`.

    ``declaration`` (for FITS sources) overrides ``source.declaration`` when
    provided. A bad source raises :class:`InvalidRequestError`; anticipated
    operational failures (decode refusal, source read failure, cancellation,
    shape/precision/digest mismatch) return a structured :class:`InspectResult`
    with no inspection.
    """
    if not isinstance(source, (FitsFrameSource, ArrayFrameSource)):
        raise InvalidRequestError("source must be a FitsFrameSource or ArrayFrameSource")

    token = cancel or CancellationToken()
    obs = _io.normalize_progress(progress, OPERATION_ID)

    if token.is_cancelled():
        return InspectResult(operation_status="CANCELLED", reason_code="CANCELLED")

    if isinstance(source, ArrayFrameSource):
        try:
            decoded = _io.array_source_to_decoded(source, cancel=token)
            actual_digest = _io.verify_identity_digest(
                source.identity, decoded.data, decoded.mask
            )
        except OperationCancelled:
            return InspectResult(operation_status="CANCELLED", reason_code="CANCELLED")
        except PrecisionRefusalError as exc:
            return InspectResult(operation_status="FAILED", reason_code="PRECISION_REFUSAL", details=str(exc))
        except (ValueError, InvalidRequestError) as exc:
            return InspectResult(operation_status="FAILED", reason_code="INVALID_SOURCE", details=str(exc))

        resolved_identity: InputIdentity = ArrayInputIdentity(
            caller_logical_id=source.identity.caller_logical_id,
            decoded_digest=actual_digest,
        )
        inspection = FrameInspection(
            metadata=source.metadata,
            identity=resolved_identity,
            domain_finding=_domain_finding(source.metadata),
            hdu=None,
            shape=tuple(decoded.data.shape),
            warnings=tuple(source.metadata.warnings),
            roi_extent_evidence=None,
        )
        _io.emit_progress(obs, OPERATION_ID, "complete", 1, 1)
        return InspectResult(operation_status="COMPLETED", inspection=inspection)

    # FitsFrameSource
    effective_declaration = declaration if declaration is not None else source.declaration
    if effective_declaration is None:
        # R3D-E F1: a Standard light supplied as "Image to calibrate" carries no
        # import declaration; build the standard_light_contract so the strict
        # decoder's raw-domain evidence requirement is satisfied without
        # weakening its processed-history/units/structural checks.
        effective_declaration = _io.standard_light_contract()
    try:
        data = _io.read_bytes(source.path, cancel=token)
        whole_fits_sha256 = _io.sha256_bytes(data)
        decoded = _io.decode_fits_from_bytes(
            data, source.hdu, effective_declaration, cancel=token, progress=obs
        )
        metadata = _io.apply_roi_extent(decoded.metadata, source.roi_extent, decoded.data.shape)
    except OperationCancelled:
        return InspectResult(operation_status="CANCELLED", reason_code="CANCELLED")
    except DecodeError as exc:
        return InspectResult(operation_status="FAILED", reason_code=exc.reason_code, details=str(exc))
    except PrecisionRefusalError as exc:
        return InspectResult(operation_status="FAILED", reason_code="PRECISION_REFUSAL", details=str(exc))
    except (OSError, SourceError, ValueError) as exc:
        return InspectResult(operation_status="FAILED", reason_code="SOURCE_ERROR", details=str(exc))

    decoded_digest = _io.verify_identity_digest(
        FitsInputIdentity(path=str(source.path), hdu=decoded.hdu, decoded_digest=""),
        decoded.data,
        decoded.mask,
    )
    identity: InputIdentity = FitsInputIdentity(
        path=str(source.path),
        hdu=decoded.hdu,
        whole_fits_sha256=whole_fits_sha256,
        decoded_digest=decoded_digest,
    )
    inspection = FrameInspection(
        metadata=metadata,
        identity=identity,
        domain_finding=_domain_finding(metadata),
        hdu=decoded.hdu,
        shape=tuple(decoded.data.shape),
        warnings=tuple(metadata.warnings),
        roi_extent_evidence=source.roi_extent,
    )
    _io.emit_progress(obs, OPERATION_ID, "complete", 1, 1)
    return InspectResult(operation_status="COMPLETED", inspection=inspection)


__all__ = ["inspect_frame"]
