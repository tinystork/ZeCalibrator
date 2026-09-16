"""Data-quality (DQ) reason bits and count semantics (SCIENCE §9).

The five v1 reason bits are all *invalid-reason* bits: any nonzero mask means
the output sample is ``NaN``. There is no per-pixel "missing evidence" bit and
no quality-flag-only bit; unknown saturation quality is a frame-level
diagnostic, never a sixth blanket invalid bit (SCIENCE §9.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

INPUT_INVALID = 0x0001  # stored-space BLANK / NaN / ±Inf, or decode failure
ADDITIVE_INVALID = 0x0002  # contributing bias/dark master sample invalid
FLAT_INVALID = 0x0004  # flat response non-positive / non-finite / ≤ floor
SATURATED = 0x0008  # sample at/above a qualified saturation limit
ARITH_NONFINITE = 0x0010  # arithmetic produced NaN/±Inf from otherwise-valid inputs

# Bits 5..15 are reserved and MUST be zero in v1.
RESERVED_MASK = 0xFFE0

DQ_NAMES: tuple[tuple[str, int], ...] = (
    ("INPUT_INVALID", INPUT_INVALID),
    ("ADDITIVE_INVALID", ADDITIVE_INVALID),
    ("FLAT_INVALID", FLAT_INVALID),
    ("SATURATED", SATURATED),
    ("ARITH_NONFINITE", ARITH_NONFINITE),
)

_BIT_BY_NAME: Mapping[str, int] = MappingProxyType(dict(DQ_NAMES))


@dataclass(frozen=True)
class CountSummary:
    """Exact DQ counts (SCIENCE §9.2).

    ``invalid_count = total - valid_count``. ``per_bit`` counts overlap
    independently: a sample contributes to every bit it has set, so
    ``invalid_count`` is *not* the sum of the per-bit counts.
    """

    total: int
    valid_count: int
    invalid_count: int
    per_bit: Mapping[str, int]

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "CountSummary":
        m = np.asarray(mask, dtype=np.uint16)
        total = int(m.size)
        valid = int(np.count_nonzero(m == 0))
        invalid = total - valid
        per_bit = {name: int(np.count_nonzero(m & bit)) for name, bit in DQ_NAMES}
        return cls(
            total=total,
            valid_count=valid,
            invalid_count=invalid,
            per_bit=MappingProxyType(per_bit),
        )


def bit_by_name(name: str) -> int:
    """Return the reason bit for a canonical DQ name."""
    try:
        return _BIT_BY_NAME[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown DQ reason name: {name!r}") from exc


def validate_mask(mask: np.ndarray) -> np.ndarray:
    """Validate a DQ mask: integer dtype and no reserved bits set (SCIENCE §9.1).

    Returns a uint16 copy. Rejects (rather than silently truncates) masks that
    carry reserved bits 5..15, which MUST be 0 in v1.
    """
    m = np.asarray(mask)
    if not np.issubdtype(m.dtype, np.integer):
        raise ValueError(f"DQ mask must be integer dtype, got {m.dtype}")
    as_u64 = m.astype(np.uint64)
    reserved = as_u64 & np.uint64(RESERVED_MASK)
    if np.any(reserved != 0):
        raise ValueError("DQ mask contains reserved bits (5..15) which must be 0 in v1")
    return m.astype(np.uint16, copy=True)


__all__ = [
    "ADDITIVE_INVALID",
    "ARITH_NONFINITE",
    "CountSummary",
    "DQ_NAMES",
    "FLAT_INVALID",
    "INPUT_INVALID",
    "RESERVED_MASK",
    "SATURATED",
    "bit_by_name",
    "validate_mask",
]
