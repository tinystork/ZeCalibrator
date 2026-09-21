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
    FitsFrameSource,
    FrameInspection,
    FrameSource,
    InspectResult,
)

OPERATION_ID = "zecalibrator-inspect"


def _decode_and_inspect(source, *, declaration=None, token, obs):
    """Decode once and return ``(InspectResult, carrier)`` (private P8-A4 seam).

    Identical to :func:`inspect_frame` in behaviour and reason codes, but also
    returns the private :class:`~zecalibrator.api.v1._io.DecodedLight` carrier so
    the batch/auto-route path can hand the SAME decoded snapshot and identity to
    calibration. ``carrier`` is ``None`` whenever the result is not ``COMPLETED``.
    """
    if not isinstance(source, (FitsFrameSource, ArrayFrameSource)):
        raise InvalidRequestError("source must be a FitsFrameSource or ArrayFrameSource")

    if token.is_cancelled():
        return InspectResult(operation_status="CANCELLED", reason_code="CANCELLED"), None

    if isinstance(source, ArrayFrameSource):
        try:
            carrier = _io._decode_light_once(source, token=token)
        except OperationCancelled:
            return InspectResult(operation_status="CANCELLED", reason_code="CANCELLED"), None
        except PrecisionRefusalError as exc:
            return InspectResult(operation_status="FAILED", reason_code="PRECISION_REFUSAL", details=str(exc)), None
        except (ValueError, InvalidRequestError) as exc:
            return InspectResult(operation_status="FAILED", reason_code="INVALID_SOURCE", details=str(exc)), None

        inspection = FrameInspection(
            metadata=source.metadata,
            identity=carrier.identity,
            domain_finding=carrier.domain_finding,
            hdu=None,
            shape=tuple(carrier.frame.data.shape),
            warnings=carrier.warnings,
            roi_extent_evidence=None,
        )
        _io.emit_progress(obs, OPERATION_ID, "complete", 1, 1)
        return InspectResult(operation_status="COMPLETED", inspection=inspection), carrier

    # FitsFrameSource
    try:
        carrier = _io._decode_light_once(
            source, token=token, declaration=declaration, progress=obs
        )
    except OperationCancelled:
        return InspectResult(operation_status="CANCELLED", reason_code="CANCELLED"), None
    except DecodeError as exc:
        return InspectResult(operation_status="FAILED", reason_code=exc.reason_code, details=str(exc)), None
    except PrecisionRefusalError as exc:
        return InspectResult(operation_status="FAILED", reason_code="PRECISION_REFUSAL", details=str(exc)), None
    except (OSError, SourceError, ValueError) as exc:
        return InspectResult(operation_status="FAILED", reason_code="SOURCE_ERROR", details=str(exc)), None

    inspection = FrameInspection(
        metadata=carrier.frame.metadata,
        identity=carrier.identity,
        domain_finding=carrier.domain_finding,
        hdu=carrier.frame.hdu,
        shape=tuple(carrier.frame.data.shape),
        warnings=carrier.warnings,
        roi_extent_evidence=carrier.roi_extent_evidence,
    )
    _io.emit_progress(obs, OPERATION_ID, "complete", 1, 1)
    return InspectResult(operation_status="COMPLETED", inspection=inspection), carrier


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
    result, _carrier = _decode_and_inspect(source, declaration=declaration, token=token, obs=obs)
    return result


__all__ = ["inspect_frame"]
