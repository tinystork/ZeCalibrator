"""Qt-free cooperative cancellation and progress (ARCHITECTURE §3.7).

Cancellation is a separate, bounded, cooperative mechanism: it is checked
between tiles / before expensive steps, never via unsafe thread kill, and never
changes arithmetic. User callbacks run outside library locks; a failing
observer must not alter arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


class OperationCancelled(Exception):
    """Raised internally when a cooperative cancellation token is set."""


@dataclass(frozen=True)
class ProgressEvent:
    """Immutable, monotonically ordered progress event."""

    operation_id: str
    phase: str
    completed: int
    total: Optional[int]
    unit: str
    frame_id: Optional[str] = None


class CancellationToken:
    """Thread-safe, Qt-free cooperative cancellation token."""

    def __init__(self) -> None:
        self._cancelled = False
        self._lock = Lock()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise OperationCancelled()


class ProgressObserver:
    """Bounded progress sink that isolates observer exceptions.

    A failing observer never changes arithmetic and never surfaces as an
    operation failure (ARCHITECTURE §3.7); the failure is recorded and the
    arithmetic continues.
    """

    def __init__(self, callback=None, *, operation_id: str = "zecalibrator-op") -> None:
        self._callback = callback
        self._operation_id = operation_id
        self._errors: list[Exception] = []

    def report(self, event: ProgressEvent) -> None:
        if self._callback is None:
            return
        try:
            self._callback(event)
        except Exception as exc:  # noqa: BLE001 - isolate observer failures
            self._errors.append(exc)

    @property
    def observer_errors(self) -> tuple[Exception, ...]:
        return tuple(self._errors)


__all__ = [
    "CancellationToken",
    "OperationCancelled",
    "ProgressEvent",
    "ProgressObserver",
]
