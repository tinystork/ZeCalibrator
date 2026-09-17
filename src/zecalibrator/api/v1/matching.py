"""Public matching and validation: ``resolve_calibration``, ``validate_binding``
and ``validate_plan``.

``resolve_calibration`` adapts a :class:`FrameInspection` into the pure G4
matching constraints and returns a structured :class:`ResolveResult`.
``validate_binding`` / ``validate_plan`` are standalone, detached boundaries
re-exported unchanged from the G4 library layer (no handle required; a source
adapter supplies the real byte identities).

.. note::
    ``validate_plan`` / ``validate_binding`` (re-exported unchanged from G4)
    verify byte/identity (image content, size, mask identity) but do **not**
    enforce the facade semantic gate (role->master_type map and hashed-HDU
    binding). ``calibrate_frame`` performs that stricter semantic validation
    before execution; a plan that ``validate_plan`` reports VALID may still be
    refused as ``PLAN_INVALID`` by ``calibrate_frame`` on role/HDU grounds.
"""

from __future__ import annotations

from typing import Mapping, Optional

from zecalibrator.application import library as _alib
from zecalibrator.application.cancellation import CancellationToken, OperationCancelled

from . import _io
from .errors import InvalidRequestError
from .models import (
    CalibrationRequest,
    FrameInspection,
    LibraryHandle,
    MatchPolicy,
    ResolveResult,
)

OPERATION_ID = "zecalibrator-resolve"


def resolve_calibration(
    frame: FrameInspection,
    request: CalibrationRequest,
    library: LibraryHandle,
    policy: MatchPolicy,
    *,
    manual_selection: Optional[Mapping[str, str]] = None,
    cancel=None,
    progress=None,
) -> ResolveResult:
    """Run pure metadata-only matching against the library (selection only).

    Returns ``ResolveResult.operation_status == "COMPLETED"`` with a
    :class:`DecisionEnvelope` whose ``outcome`` is ``MATCHED`` / ``NO_MATCH`` /
    ``AMBIGUOUS``. A closed handle raises :class:`LibraryClosedError` (the same
    class the re-exported handle raises).
    """
    if not isinstance(frame, FrameInspection):
        raise InvalidRequestError("frame must be a FrameInspection")
    if not isinstance(request, CalibrationRequest):
        raise InvalidRequestError("request must be a CalibrationRequest")
    if not isinstance(library, LibraryHandle):
        raise InvalidRequestError("library must be a LibraryHandle")
    if not isinstance(policy, MatchPolicy):
        raise InvalidRequestError("policy must be a MatchPolicy")

    token = cancel or CancellationToken()
    obs = _io.normalize_progress(progress, OPERATION_ID)

    if token.is_cancelled():
        return ResolveResult(operation_status="CANCELLED", reason_code="CANCELLED")

    _io.emit_progress(obs, OPERATION_ID, "start", 0, 2)
    try:
        light = _alib.light_constraints_from_sensor_metadata(frame.metadata)
    except _alib.MetadataAdapterError as exc:
        return ResolveResult(
            operation_status="FAILED", reason_code="METADATA_ADAPTER", details=str(exc)
        )

    token.raise_if_cancelled()
    try:
        decision = library.resolve(
            light,
            request,
            policy,
            manual_selection=manual_selection,
            input_identity=frame.identity,
        )
    except OperationCancelled:
        return ResolveResult(operation_status="CANCELLED", reason_code="CANCELLED")

    _io.emit_progress(obs, OPERATION_ID, "complete", 2, 2)
    return ResolveResult(operation_status="COMPLETED", decision=decision)


# Standalone, detached validation (no library handle): re-exported unchanged.
validate_binding = _alib.validate_binding
validate_plan = _alib.validate_plan


__all__ = ["resolve_calibration", "validate_binding", "validate_plan"]
