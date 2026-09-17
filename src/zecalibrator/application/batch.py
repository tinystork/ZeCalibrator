"""Bounded batch orchestration (Phase 6): ordering, dispositions, progress.

This module provides the *orchestration* layer for ``calibrate_batch`` without
importing ``zecalibrator.api.v1`` (dependency direction: application -> io/core,
never application -> api). The public facade in ``api.v1.batch`` wires these
helpers to the frozen G5 science (``inspect_frame`` / ``resolve_calibration`` /
``calibrate_frame``) so batch science is always unitary G5 science.

Responsibilities:
* deterministic ordered input list (caller order, never sorted/mutated);
* batch/operation identity generation;
* status -> batch disposition mapping and batch-status derivation;
* monotonic progress emission with an optional per-frame id;
* cancellation checkpointing between frames and before commits.
"""

from __future__ import annotations

import uuid
from typing import Iterable, Optional, Sequence

BATCH_OPERATION_ID = "zecalibrator-calibrate-batch"

DISPOSITION_COMPLETED = "COMPLETED"
DISPOSITION_COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
DISPOSITION_SKIPPED = "SKIPPED"
DISPOSITION_FAILED = "FAILED"
DISPOSITION_CANCELLED = "CANCELLED"

_BATCH_STATUSES = ("COMPLETED", "PARTIAL", "CANCELLED")


def new_batch_id() -> str:
    """Return a fresh batch id (128-bit hex). Overridable via ``BatchOptions``."""
    return uuid.uuid4().hex


def ordered_frames(frames: Iterable) -> list:
    """Materialize the ordered frame-source list (deterministic caller order).

    A string/bytes is refused (never iterated per-character); a non-iterable is
    refused. The resulting list is a shallow copy of the caller's order and is
    never sorted.
    """
    if isinstance(frames, (str, bytes)):
        raise TypeError("frames must be an iterable of FrameSource, not a string")
    try:
        return list(frames)
    except TypeError as exc:
        raise TypeError("frames must be an iterable of FrameSource") from exc


def batch_disposition(result_status: str) -> str:
    """Map a ``calibrate_frame`` result status to a batch item disposition.

    ``CANCELLED`` is *not* a disposition produced here: the facade raises
    ``OperationCancelled`` for a cancelled frame (no committed partial output).
    """
    if result_status == "COMPLETED":
        return DISPOSITION_COMPLETED
    if result_status == "COMPLETED_WITH_WARNINGS":
        return DISPOSITION_COMPLETED_WITH_WARNINGS
    if result_status == "FAILED":
        return DISPOSITION_FAILED
    raise ValueError(f"unsupported result status {result_status!r}")


def derive_batch_status(
    dispositions: Sequence[str], *, cancelled: bool, total: int = 0
) -> str:
    """Derive the batch-level status from item dispositions.

    ``CANCELLED`` when the batch was cancelled; ``PARTIAL`` when any item failed;
    ``COMPLETED`` otherwise (including all-skipped: skipping is not an error).
    """
    if cancelled:
        return "CANCELLED"
    if any(d == DISPOSITION_FAILED for d in dispositions):
        return "PARTIAL"
    return "COMPLETED"


def emit_batch_progress(obs, phase: str, completed: int, total: int, frame_id: Optional[str] = None) -> None:
    """Emit one bounded batch progress event (observer failures isolated)."""
    if obs is None:
        return
    from zecalibrator.application.cancellation import ProgressEvent

    report = getattr(obs, "report", None)
    if report is None and callable(obs):
        report = obs
    if report is None:
        return
    try:
        report(
            ProgressEvent(
                operation_id=BATCH_OPERATION_ID,
                phase=phase,
                completed=completed,
                total=total,
                unit="frames",
                frame_id=frame_id,
            )
        )
    except Exception:
        return


def frame_display_id(frame) -> str:
    """A stable per-frame display id for progress events (no fabricated FITS id)."""
    path = getattr(frame, "path", None)
    if path is not None:
        return str(path)
    identity = getattr(frame, "identity", None)
    if identity is not None and getattr(identity, "caller_logical_id", None):
        return str(identity.caller_logical_id)
    return "<frame>"


__all__ = [
    "BATCH_OPERATION_ID",
    "DISPOSITION_CANCELLED",
    "DISPOSITION_COMPLETED",
    "DISPOSITION_COMPLETED_WITH_WARNINGS",
    "DISPOSITION_FAILED",
    "DISPOSITION_SKIPPED",
    "batch_disposition",
    "derive_batch_status",
    "emit_batch_progress",
    "frame_display_id",
    "new_batch_id",
    "ordered_frames",
]
