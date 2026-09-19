"""Exact geometry contract (SCIENCE §4, ARCHITECTURE §3.3).

Geometry is an *exact* contract: shape, binning, ROI origin/extent, orientation
and CFA phase. Same shape does not imply same ROI; translation by even one pixel
changes CFA phase; orientation heuristics and crop/transpose/resample are
forbidden inside calibration.

Unknown fields are represented as ``None`` — never invented defaults. A
``None`` field does **not** match another ``None`` (SCIENCE §6.4: unknown ≠
unknown).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Explicitly enumerated CFA phases (no ellipsis). "mono" is an explicit mono
# sensor, distinct from ``None`` (unknown CFA, never silently treated as mono).
CFA_PHASES: tuple[str, ...] = ("GRBG", "RGGB", "BGGR", "GBRG", "mono")

# Sensor-level 2x2 Bayer tile labels, ordered by parity index
# ``(y % 2) * 2 + (x % 2)`` == 0..3 (even/even, even/odd, odd/even, odd/odd).
# The two greens are labelled distinctly (G1, G2) per SCIENCE §7.5.
_PHASE_LABELS: dict[str, tuple[str, str, str, str]] = {
    "GRBG": ("G1", "R", "B", "G2"),
    "RGGB": ("R", "G1", "G2", "B"),
    "BGGR": ("B", "G1", "G2", "R"),
    "GBRG": ("G1", "B", "R", "G2"),
}

CFA_PLANES: tuple[str, ...] = ("G1", "R", "B", "G2")
MONO_PLANES: tuple[str, ...] = ("mono",)


@dataclass(frozen=True)
class Geometry:
    """Immutable sensor/array geometry.

    ``shape`` is the array/ROI ``(ny, nx)`` (always known from the data).
    ``sensor_dimensions`` (full frame), ``binning``, ``roi_origin`` (``(oy, ox)``
    binned-pixel), ``roi_extent``, ``orientation`` and ``cfa_phase`` are
    ``None`` when unknown — never invented defaults.
    """

    shape: tuple[int, int]
    sensor_dimensions: Optional[tuple[int, int]] = None
    binning: Optional[tuple[int, int]] = None
    roi_origin: Optional[tuple[int, int]] = None
    roi_extent: Optional[tuple[int, int]] = None
    orientation: Optional[str] = None  # v1 supports "identity" only, when declared
    cfa_phase: Optional[str] = None  # None = unknown; "mono" = explicit mono

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "sensor_dimensions", _freeze_tuple(self.sensor_dimensions))
        object.__setattr__(self, "binning", _freeze_tuple(self.binning))
        object.__setattr__(self, "roi_origin", _freeze_tuple(self.roi_origin))
        object.__setattr__(self, "roi_extent", _freeze_tuple(self.roi_extent))


def _freeze_tuple(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise ValueError(f"geometry tuple field must be a tuple/list or None, got {type(value)!r}")


def is_bayer_phase(cfa_phase: Optional[str]) -> bool:
    """Return ``True`` when ``cfa_phase`` denotes a Bayer CFA sensor.

    This is the shared CFA-conditional decision predicate for the relation
    matcher and its execution-time mirror (R3B): for a Bayer sensor a one-pixel
    ROI translation flips the parity phase, so ``orientation``/``roi_origin``/
    ``roi_extent`` are *necessary*; for an explicit mono sensor (``"mono"``) —
    where CFA does not apply — those fields degrade to disambiguators.
    ``None`` (unknown) is not a Bayer phase.
    """
    return cfa_phase in ("GRBG", "RGGB", "BGGR", "GBRG")


def _phase_labels(cfa_phase: str) -> tuple[str, str, str, str]:
    try:
        return _PHASE_LABELS[cfa_phase]
    except KeyError as exc:
        raise ValueError(f"unsupported CFA phase: {cfa_phase!r}") from exc


def sensor_plane(y: int, x: int, cfa_phase: str) -> str:
    """Return the CFA plane label at *sensor* coordinates ``(y, x)``."""
    labels = _phase_labels(cfa_phase)
    return labels[(y % 2) * 2 + (x % 2)]


def array_plane(
    y: int, x: int, cfa_phase: str, roi_origin: tuple[int, int] = (0, 0)
) -> str:
    """Return the CFA plane label at *array* coordinates under an ROI origin.

    The ROI origin (``(oy, ox)``) shifts the sensor parity: array ``(y, x)``
    corresponds to sensor ``(y + oy, x + ox)`` (SCIENCE §7.5).
    """
    oy, ox = roi_origin
    return sensor_plane(y + oy, x + ox, cfa_phase)


def plane_index_array(
    shape: tuple[int, int], roi_origin: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Return an int32 array of parity indices ``(y+oy)%2*2 + (x+ox)%2``."""
    oy, ox = roi_origin
    yy, xx = np.indices(shape, dtype=np.int64)
    return (((yy + oy) % 2) * 2 + ((xx + ox) % 2)).astype(np.int32)


def plane_label_array(
    shape: tuple[int, int],
    cfa_phase: str,
    roi_origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Return a string array of CFA plane labels for the whole array."""
    labels = _phase_labels(cfa_phase)
    idx = plane_index_array(shape, roi_origin)
    out = np.empty(shape, dtype=object)
    for i, label in enumerate(labels):
        out[idx == i] = label
    return out


def geometry_matches(light: Geometry, master: Geometry) -> tuple[str, ...]:
    """Return the reason codes for every exact incompatibility (empty == compatible).

    Exact comparison; ``None`` (unknown) never matches ``None`` or a known value.
    ``shape`` must be exactly equal. Every other field must be known on *both*
    sides and exactly equal; a missing/unknown required field yields
    ``MISSING_REQUIRED_FIELD``.
    """
    reasons: list[str] = []

    if light.shape != master.shape:
        reasons.append("GEOMETRY_MISMATCH")

    for field, code in (
        ("sensor_dimensions", "GEOMETRY_MISMATCH"),
        ("binning", "BINNING_MISMATCH"),
        ("orientation", "GEOMETRY_MISMATCH"),
    ):
        lv = getattr(light, field)
        mv = getattr(master, field)
        if lv is None or mv is None:
            reasons.append("MISSING_REQUIRED_FIELD")
        elif lv != mv:
            reasons.append(code)

    # roi_extent is optional (redundant with shape); checked only when both known.
    if light.roi_extent is not None and master.roi_extent is not None:
        if light.roi_extent != master.roi_extent:
            reasons.append("GEOMETRY_MISMATCH")

    lroi = light.roi_origin
    mroi = master.roi_origin
    if lroi is None or mroi is None:
        reasons.append("MISSING_REQUIRED_FIELD")
    elif lroi != mroi:
        reasons.append("ROI_ORIGIN_MISMATCH")
        reasons.append("CFA_PHASE_MISMATCH")

    lcfa = light.cfa_phase
    mcfa = master.cfa_phase
    if lcfa is None or mcfa is None:
        reasons.append("MISSING_REQUIRED_FIELD")
    elif lcfa != mcfa:
        reasons.append("CFA_PHASE_MISMATCH")

    return tuple(dict.fromkeys(reasons))


__all__ = [
    "CFA_PHASES",
    "CFA_PLANES",
    "Geometry",
    "MONO_PLANES",
    "array_plane",
    "geometry_matches",
    "is_bayer_phase",
    "plane_index_array",
    "plane_label_array",
    "sensor_plane",
]
