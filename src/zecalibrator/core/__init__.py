"""ZeCalibrator scientific core (Phase 3 / G3 scope).

Pure, headless calibration primitives: metadata value objects, geometry,
data-quality semantics, flat/CFA normalization equations, and the CPU
calibration pipeline. This package performs **no** filesystem I/O, no FITS
parsing, no Qt/ZeAlfie/ZSSS/CuPy import, and no GPU work.

The public bootstrap surface (``zecalibrator.api.v1``) remains unchanged at G3:
these modules are the internal scientific engine, not advertised capabilities.
"""

from __future__ import annotations

from zecalibrator.core.dq import (
    ADDITIVE_INVALID,
    ARITH_NONFINITE,
    DQ_NAMES,
    FLAT_INVALID,
    INPUT_INVALID,
    SATURATED,
)
from zecalibrator.core.geometry import CFA_PHASES, Geometry

__all__ = [
    "ADDITIVE_INVALID",
    "ARITH_NONFINITE",
    "CFA_PHASES",
    "DQ_NAMES",
    "FLAT_INVALID",
    "Geometry",
    "INPUT_INVALID",
    "SATURATED",
]
