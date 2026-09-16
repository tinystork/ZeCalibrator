"""Typed, stable exception hierarchy for the ZeCalibrator engine.

These errors distinguish *parameter/request* failures (which raise) from
*anticipated operational* outcomes (which are returned as structured results,
e.g. ``CalibrationResult(status="FAILED"/"CANCELLED")``). Reason codes stay
aligned with the frozen G1 registry in ``research/phase1/cases.json``.
"""

from __future__ import annotations


class ZeCalibratorError(Exception):
    """Base class for all ZeCalibrator structured errors."""


class InvalidRequestError(ZeCalibratorError, ValueError):
    """A malformed or unsupported request/mode was supplied."""


class GeometryMismatchError(ZeCalibratorError):
    """Explicit master geometry/acquisition is incompatible with the light.

    This is a *validation* failure against an explicitly supplied master, never
    automatic matching. ``reason_code`` is one of the frozen G1 reason codes.
    """

    def __init__(self, reason_code: str, fields: tuple[str, ...] = ()) -> None:
        self.reason_code = reason_code
        self.fields = tuple(fields)
        suffix = f": {', '.join(fields)}" if fields else ""
        super().__init__(f"{reason_code}{suffix}")


class PrecisionRefusalError(ZeCalibratorError):
    """Decoded values exceed the documented float32 precision envelope."""


class DecodeError(ZeCalibratorError):
    """A strict FITS raw-decode rejection (malformed/conflict/ambiguity/domain)."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)


__all__ = [
    "DecodeError",
    "GeometryMismatchError",
    "InvalidRequestError",
    "PrecisionRefusalError",
    "ZeCalibratorError",
]
