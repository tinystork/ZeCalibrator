"""Documented float32 precision envelope (SCIENCE §2.5, §7.6).

float32 exactly represents integer ADU through ``2**24`` (16777216). Larger
integer ranges require an explicit error-bound witness or refusal; there is no
silent precision-loss claim and no implicit ADC-depth inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# IEEE-754 fact: float32 represents all integers with |v| <= 2**24 exactly.
FLOAT32_EXACT_INT_BOUND = 2**24
# Round-to-nearest relative error for non-integer float32 conversion.
FLOAT32_REL_EPS = np.finfo(np.float32).eps  # 2**-23
FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class PrecisionInfo:
    """Measured float64 -> float32 round-off envelope for a decoded/calibrated
    plane. ``max_abs_error`` / ``max_rel_error`` are measured against the
    float64 reference, never a loosened tolerance.
    """

    max_abs_error: float
    max_rel_error: float
    exact_integer_bound: int = FLOAT32_EXACT_INT_BOUND


def measure_float32_roundoff(reference: np.ndarray) -> PrecisionInfo:
    """Measure float64->float32 round-off against an independent float64
    reference. Non-finite reference entries are excluded from the error metric.
    """
    ref = np.asarray(reference, dtype=np.float64)
    finite = np.isfinite(ref)
    if not np.any(finite):
        return PrecisionInfo(max_abs_error=0.0, max_rel_error=0.0)
    f32 = ref.astype(np.float32).astype(np.float64)
    with np.errstate(invalid="ignore"):
        abs_err = np.abs(f32 - ref)
    max_abs = float(np.max(abs_err[finite]))
    denom = np.abs(ref[finite])
    # Relative error is meaningful only where the reference is non-zero.
    nonzero = denom > 0.0
    if np.any(nonzero):
        rel = abs_err[finite][nonzero] / denom[nonzero]
        max_rel = float(np.max(rel))
    else:
        max_rel = 0.0
    return PrecisionInfo(max_abs_error=max_abs, max_rel_error=max_rel)


def measure_float32_error(computed_f32: np.ndarray, reference_f64: np.ndarray) -> PrecisionInfo:
    """Measure the error of a float32 *frame-arithmetic* result against an
    independent float64 reference (same inputs, float64 arithmetic). Non-finite
    entries on either side are excluded from the metric.
    """
    computed = np.asarray(computed_f32, dtype=np.float64)
    ref = np.asarray(reference_f64, dtype=np.float64)
    finite = np.isfinite(ref) & np.isfinite(computed)
    if not np.any(finite):
        return PrecisionInfo(max_abs_error=0.0, max_rel_error=0.0)
    with np.errstate(invalid="ignore"):
        abs_err = np.abs(computed - ref)
    max_abs = float(np.max(abs_err[finite]))
    denom = np.abs(ref[finite])
    nonzero = denom > 0.0
    if np.any(nonzero):
        rel = abs_err[finite][nonzero] / denom[nonzero]
        max_rel = float(np.max(rel))
    else:
        max_rel = 0.0
    return PrecisionInfo(max_abs_error=max_abs, max_rel_error=max_rel)


def integer_exceeds_float32_exact_range(value: float) -> bool:
    """True when an integer-valued float64 is not exactly representable in
    float32 (|v| > 2**24)."""
    v = float(value)
    if not np.isfinite(v):
        return False
    if v != round(v):
        return False
    return abs(v) > FLOAT32_EXACT_INT_BOUND


def exceeds_float32_magnitude(value: float) -> bool:
    """True when a finite float64 magnitude overflows the float32 range
    (|v| > float32 max)."""
    v = float(value)
    if not np.isfinite(v):
        return False
    return abs(v) > FLOAT32_MAX


def float32_guard_provable_safe(stored_dtype, bscale: float, bzero: float) -> bool:
    """True when the storage representation proves that every *finite* decoded
    value satisfies both precision limits (``|v| <= 2**24``, hence also
    ``|v| <= float32 max``), so the value-scanning precision guard checks
    (C1 magnitude / C2 exact-integer) can be skipped.

    This is an acceptance-only (sound) static proof: it returns ``True`` only
    when safety is proven from the representation alone. It never *refuses* a
    frame from a representation bound. When ``False``, the caller must fall back
    to the vectorised value predicate. Byte-order invariant: the bound is
    computed from ``np.iinfo(dtype).min/.max`` on the (possibly non-native)
    dtype, never from an itemsize attribute of ``np.iinfo`` (which has none).
    """
    dt = np.dtype(stored_dtype)
    bs = float(bscale)
    bz = float(bzero)
    if bs == 0.0:
        # physical is constant = bzero wherever stored is finite, for any
        # storage dtype (integer or floating).
        bound = abs(bz)
    elif dt.kind in ("i", "u"):
        info = np.iinfo(dt)
        maxabs_stored = float(max(abs(int(info.min)), abs(int(info.max))))
        bound = abs(bs) * maxabs_stored + abs(bz)
    else:
        # Floating storage with a nonzero scale has an unbounded value domain:
        # no representation bound can prove the limit unreachable.
        return False
    return bool(np.isfinite(bound)) and bound <= float(FLOAT32_EXACT_INT_BOUND)


def exceeds_float32_exact_bound(values) -> bool:
    """Vectorised twin of :func:`integer_exceeds_float32_exact_range`: ``True``
    when any finite, integer-valued entry exceeds the float32 exact-integer
    bound (``2**24``). Non-finite and non-integer-valued entries are ignored,
    matching the scalar predicate exactly.
    """
    vals = np.asarray(values)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return False
    f = vals[finite]
    return bool(
        np.any((f == np.rint(f)) & (np.abs(f) > np.float64(FLOAT32_EXACT_INT_BOUND)))
    )


__all__ = [
    "FLOAT32_EXACT_INT_BOUND",
    "FLOAT32_MAX",
    "FLOAT32_REL_EPS",
    "PrecisionInfo",
    "exceeds_float32_magnitude",
    "integer_exceeds_float32_exact_range",
    "measure_float32_error",
    "measure_float32_roundoff",
]
