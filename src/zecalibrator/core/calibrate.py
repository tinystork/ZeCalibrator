"""CPU calibration pipeline: float32 frame arithmetic + DQ semantics (SCIENCE §7-§9).

Frame arithmetic is float32 (ARCHITECTURE §6); float64 is used only for decode/
statistical intermediates and as an *independent reference* to report the
actual error. Produces a signed native contiguous float32 calibrated plane plus
a uint16 DQ mask and exact counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from zecalibrator.core.dq import (
    ADDITIVE_INVALID,
    ARITH_NONFINITE,
    CountSummary,
    FLAT_INVALID,
    INPUT_INVALID,
    SATURATED,
    validate_mask,
)
from zecalibrator.core.equations import DEFAULT_FLOOR, additive_numerator, final_plane
from zecalibrator.core.errors import InvalidRequestError
from zecalibrator.core.precision import PrecisionInfo, measure_float32_error

STATUS_COMPLETED = "COMPLETED"
STATUS_COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class FrameQuality:
    """Frame-level quality diagnostic (SCIENCE §9.3)."""

    saturation_evidence: str  # "qualified" | "unknown"


@dataclass(frozen=True)
class CalibrationResult:
    """In-memory result of a single-frame CPU calibration.

    ``data`` and ``mask`` are freshly allocated outputs owned by this result;
    caller arrays are never mutated. ``data``/``mask`` are ``None`` on ``FAILED``
    (all-invalid) and ``CANCELLED``. ``scalars`` is a read-only mapping.
    """

    status: str
    data: Optional[np.ndarray]  # native contiguous float32, signed
    mask: Optional[np.ndarray]  # uint16 DQ reason mask
    counts: Optional[CountSummary]
    frame_quality: FrameQuality
    precision: Optional[PrecisionInfo]
    scalars: Mapping[str, Optional[float]]
    warnings: tuple[str, ...] = ()
    reason_code: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        object.__setattr__(self, "warnings", tuple(self.warnings))


def _f32_checked(arr: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float32, returning (array, overflow_invalid bool mask).

    A finite float64 input whose magnitude overflows float32 is reported as
    invalid (never silently returned as Inf).
    """
    a64 = np.asarray(arr, dtype=np.float64)
    if a64.ndim != 2:
        raise InvalidRequestError(f"{name} must be a 2D plane")
    with np.errstate(over="ignore", invalid="ignore"):
        a32 = np.ascontiguousarray(a64.astype(np.float32))
    overflow = np.isfinite(a64) & ~np.isfinite(a32)
    return a32, overflow


def _master_mask(mask: Optional[np.ndarray], shape: tuple[int, int], name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint16)
    m = validate_mask(mask)
    if m.shape != shape:
        raise InvalidRequestError(f"{name} mask shape {m.shape} vs {shape}")
    return m


def _require_masters(additive_mode: str, kwargs: dict) -> None:
    required = {
        "control": (),
        "bias_only": ("bias",),
        "dark_incl_bias": ("dark_inc",),
        "dark_bias_removed": ("bias", "dark_removed"),
    }
    if additive_mode not in required:
        raise InvalidRequestError(f"unsupported additive_mode: {additive_mode!r}")
    for name in required[additive_mode]:
        if kwargs.get(name) is None:
            raise InvalidRequestError(
                f"additive_mode {additive_mode!r} requires master {name!r}"
            )


def _float64_reference(
    L: np.ndarray,
    additive_mode: str,
    flat_mode: str,
    bias=None,
    dark_inc=None,
    dark_removed=None,
    flat_response=None,
) -> np.ndarray:
    """Independent float64 reference of the same arithmetic (for error report)."""
    L64 = L.astype(np.float64)
    if additive_mode == "control":
        A = L64
    elif additive_mode == "bias_only":
        A = L64 - bias.astype(np.float64)
    elif additive_mode == "dark_incl_bias":
        A = L64 - dark_inc.astype(np.float64)
    elif additive_mode == "dark_bias_removed":
        A = L64 - bias.astype(np.float64) - dark_removed.astype(np.float64)
    else:
        raise InvalidRequestError(f"unsupported additive_mode: {additive_mode!r}")
    if flat_mode == "apply":
        return A / flat_response.astype(np.float64)
    return A


def calibrate_light(
    light: np.ndarray,
    *,
    additive_mode: str,
    flat_mode: str = "none",
    input_mask: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    bias_mask: Optional[np.ndarray] = None,
    dark_inc: Optional[np.ndarray] = None,
    dark_inc_mask: Optional[np.ndarray] = None,
    dark_removed: Optional[np.ndarray] = None,
    dark_removed_mask: Optional[np.ndarray] = None,
    flat_response: Optional[np.ndarray] = None,
    flat_valid: Optional[np.ndarray] = None,
    saturation_limit: Optional[float] = None,
    floor: float = DEFAULT_FLOOR,
) -> CalibrationResult:
    """Calibrate a decoded light plane and return a structured result.

    ``light`` is a float32 physical-ADU plane. Master arrays are float32 decoded
    planes; their masks are uint16 reason masks (any nonzero == invalid). When
    ``flat_mode == "apply"``, ``flat_response`` is an already-normalized
    dimensionless response and ``flat_valid`` a boolean validity mask; the
    ``R > floor`` division floor is enforced here too (never clamped).
    """
    if flat_mode not in ("none", "apply"):
        raise InvalidRequestError(f"unsupported flat_mode: {flat_mode!r}")

    L, light_overflow = _f32_checked(light, "light")
    shape = L.shape

    _require_masters(additive_mode, {"bias": bias, "dark_inc": dark_inc, "dark_removed": dark_removed})

    in_mask = _master_mask(input_mask, shape, "input")

    # Convert masters to float32 (frame arithmetic) with overflow detection.
    m_bias = m_bias_of = m_dinc = m_dinc_of = m_drem = m_drem_of = None
    if bias is not None:
        m_bias, m_bias_of = _f32_checked(bias, "bias")
    if dark_inc is not None:
        m_dinc, m_dinc_of = _f32_checked(dark_inc, "dark_inc")
    if dark_removed is not None:
        m_drem, m_drem_of = _f32_checked(dark_removed, "dark_removed")

    A = additive_numerator(
        L, additive_mode, bias=m_bias, dark_inc=m_dinc, dark_removed=m_drem
    )

    additive_invalid = np.zeros(shape, dtype=bool)
    if additive_mode == "bias_only":
        additive_invalid |= _master_mask(bias_mask, shape, "bias") != 0
        additive_invalid |= m_bias_of
    elif additive_mode == "dark_incl_bias":
        additive_invalid |= _master_mask(dark_inc_mask, shape, "dark_inc") != 0
        additive_invalid |= m_dinc_of
    elif additive_mode == "dark_bias_removed":
        additive_invalid |= _master_mask(bias_mask, shape, "bias") != 0
        additive_invalid |= _master_mask(dark_removed_mask, shape, "dark_removed") != 0
        additive_invalid |= m_bias_of | m_drem_of

    flat_invalid = np.zeros(shape, dtype=bool)
    R = None
    if flat_mode == "apply":
        if flat_response is None:
            raise InvalidRequestError("flat_mode 'apply' requires flat_response")
        R, r_overflow = _f32_checked(flat_response, "flat_response")
        if R.shape != shape:
            raise InvalidRequestError(f"flat response shape {R.shape} vs {shape}")
        # The frozen R > floor division floor applies to supplied responses too.
        flat_invalid = ~(np.isfinite(R) & (R > np.float32(floor)))
        flat_invalid |= r_overflow
        if flat_valid is not None:
            fv = np.asarray(flat_valid, dtype=bool)
            if fv.shape != shape:
                raise InvalidRequestError(f"flat_valid shape {fv.shape} vs {shape}")
            flat_invalid |= ~fv
        C = final_plane(A, R)
    else:
        C = final_plane(A, None)

    # Build the DQ mask (union of contributing reasons).
    mask = in_mask.copy()
    mask[light_overflow] |= INPUT_INVALID
    mask[additive_invalid] |= ADDITIVE_INVALID
    mask[flat_invalid] |= FLAT_INVALID

    saturation_evidence = "unknown"
    if saturation_limit is not None:
        saturation_evidence = "qualified"
        sat = L >= np.float32(saturation_limit)
        mask[sat] |= SATURATED

    # Arithmetic-produced non-finite (from otherwise-valid inputs) is canonicalized
    # to NaN with ARITH_NONFINITE, never returned as Inf with DQ0.
    nonfinite = ~np.isfinite(C)
    arith_nonfinite = nonfinite & (mask == 0)
    mask[arith_nonfinite] |= ARITH_NONFINITE

    C = np.asarray(C, dtype=np.float32)
    C[mask != 0] = np.nan

    counts = CountSummary.from_mask(mask)
    all_invalid = counts.invalid_count == counts.total

    # Independent float64 reference for the reported precision envelope.
    reference = _float64_reference(
        L, additive_mode, flat_mode,
        bias=m_bias, dark_inc=m_dinc, dark_removed=m_drem, flat_response=R,
    )
    precision = measure_float32_error(C, reference)

    if all_invalid:
        return CalibrationResult(
            status=STATUS_FAILED,
            data=None,
            mask=None,
            counts=counts,
            frame_quality=FrameQuality(saturation_evidence=saturation_evidence),
            precision=precision,
            scalars={},
            reason_code="ALL_INVALID",
        )

    data = np.ascontiguousarray(C.astype(np.float32))
    out_mask = np.ascontiguousarray(mask.astype(np.uint16))

    warnings: list[str] = []
    if saturation_evidence == "unknown":
        warnings.append("saturation limit unknown; SATURATED bit not evaluated")
    if counts.invalid_count > 0:
        warnings.append("partial invalidity present; see mask and counts")

    if counts.invalid_count > 0 or saturation_evidence == "unknown":
        status = STATUS_COMPLETED_WITH_WARNINGS
    else:
        status = STATUS_COMPLETED

    return CalibrationResult(
        status=status,
        data=data,
        mask=out_mask,
        counts=counts,
        frame_quality=FrameQuality(saturation_evidence=saturation_evidence),
        precision=precision,
        scalars={},
        warnings=tuple(warnings),
    )


__all__ = [
    "CalibrationResult",
    "FrameQuality",
    "STATUS_CANCELLED",
    "STATUS_COMPLETED",
    "STATUS_COMPLETED_WITH_WARNINGS",
    "STATUS_FAILED",
    "calibrate_light",
]
