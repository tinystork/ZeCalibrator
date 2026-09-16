"""Pure calibration equations, flat normalization and quality screen (SCIENCE §7-§8).

Frame arithmetic is **float32** (ARCHITECTURE §6): additive subtraction and
response division operate on float32 planes. Decode and statistical
intermediates (the exact flat median) use float64. No I/O, no Qt, no state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from zecalibrator.core.errors import InvalidRequestError
from zecalibrator.core.geometry import MONO_PLANES, plane_label_array

ADDITIVE_MODES: tuple[str, ...] = (
    "control",
    "bias_only",
    "dark_incl_bias",
    "dark_bias_removed",
)

FLAT_PREP_MODES: tuple[str, ...] = (
    "flat_dark_incl_bias",
    "flat_dark_bias_removed",
    "bias_only_flat",
    "already_normalized",
)

DEFAULT_FLOOR = 1e-6
DEFAULT_QUALITY_THRESHOLD = 90.0  # adopted owner policy (SCIENCE §8.3)


def _as_f32(a: np.ndarray, name: str) -> np.ndarray:
    if a is None:
        raise InvalidRequestError(f"required master is missing: {name}")
    arr = np.asarray(a, dtype=np.float32)
    if arr.ndim != 2:
        raise InvalidRequestError(f"{name} must be a 2D plane")
    return arr


def _require_same_shape(*arrays: np.ndarray) -> tuple[int, int]:
    if not arrays:
        raise InvalidRequestError("no arrays supplied")
    shape = arrays[0].shape
    for a in arrays[1:]:
        if a.shape != shape:
            raise InvalidRequestError(
                f"broadcasting shape mismatch: {a.shape} vs {shape}"
            )
    return shape


def additive_numerator(
    light: np.ndarray,
    additive_mode: str,
    *,
    bias: Optional[np.ndarray] = None,
    dark_inc: Optional[np.ndarray] = None,
    dark_removed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the additive-corrected numerator ``A`` (float32 frame arithmetic).

    Exactly one coherent branch is chosen; no silent downgrade and no double
    bias subtraction (SCIENCE §7.2, §7.8).
    """
    L = _as_f32(light, "light")
    if additive_mode == "control":
        return L.copy()
    if additive_mode == "bias_only":
        B = _as_f32(bias, "bias")
        _require_same_shape(L, B)
        return L - B
    if additive_mode == "dark_incl_bias":
        D = _as_f32(dark_inc, "dark_inc")
        _require_same_shape(L, D)
        return L - D
    if additive_mode == "dark_bias_removed":
        B = _as_f32(bias, "bias")
        D = _as_f32(dark_removed, "dark_removed")
        _require_same_shape(L, B, D)
        return L - B - D
    raise InvalidRequestError(f"unsupported additive_mode: {additive_mode!r}")


def flat_correction(
    finc: np.ndarray,
    prep_mode: str,
    *,
    flat_dark_inc: Optional[np.ndarray] = None,
    flat_dark_removed: Optional[np.ndarray] = None,
    bias_flat: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Return the corrected-but-unnormalized flat ``Fcorr`` (float32).

    Returns ``None`` for ``already_normalized`` (consume ``R`` directly, never
    subtract again). The flat-dark corrects the *flat*, never the light.
    """
    F = _as_f32(finc, "finc")
    if prep_mode == "flat_dark_incl_bias":
        FD = _as_f32(flat_dark_inc, "flat_dark_inc")
        _require_same_shape(F, FD)
        return F - FD
    if prep_mode == "flat_dark_bias_removed":
        B = _as_f32(bias_flat, "bias_flat")
        FD = _as_f32(flat_dark_removed, "flat_dark_removed")
        _require_same_shape(F, B, FD)
        return F - B - FD
    if prep_mode == "bias_only_flat":
        B = _as_f32(bias_flat, "bias_flat")
        _require_same_shape(F, B)
        return F - B
    if prep_mode == "already_normalized":
        return None
    raise InvalidRequestError(f"unsupported flat prep_mode: {prep_mode!r}")


@dataclass(frozen=True)
class FlatNormalization:
    """Result of normalizing a corrected flat to a dimensionless response."""

    R: np.ndarray  # float32 dimensionless response
    scalars: Mapping[str, Optional[float]]  # per-plane exact float64 median
    valid: np.ndarray  # bool: response valid (finite, >0, >floor, contributors valid)
    plane_counts: Mapping[str, int]  # per-plane valid population count
    plane_totals: Mapping[str, int]  # per-plane geometric total
    plane_ratios: Mapping[str, float]  # valid/total per plane (percent)
    usable: bool
    unusable_planes: tuple[str, ...]
    empty_planes: tuple[str, ...]
    cfa_phase: str
    roi_origin: tuple[int, int]
    floor: float
    quality_threshold: float

    def __post_init__(self) -> None:
        for name in ("scalars", "plane_counts", "plane_totals", "plane_ratios"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "unusable_planes", tuple(self.unusable_planes))
        object.__setattr__(self, "empty_planes", tuple(self.empty_planes))


def _to_f32(v: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(v, dtype=np.float32))


def normalize_flat_response(
    fcorr: np.ndarray,
    *,
    cfa_phase: str,
    roi_origin: tuple[int, int] = (0, 0),
    saturation_samples: Optional[np.ndarray] = None,
    saturation_limit: Optional[float] = None,
    fcorr_invalid: Optional[np.ndarray] = None,
    floor: float = DEFAULT_FLOOR,
) -> FlatNormalization:
    """Normalize a corrected flat to ``R = Fcorr / s[plane]`` (float32 response).

    Mono uses one exact median; Bayer CFA uses four distinct parity planes
    (G1/R/B/G2) with the frozen sensor phase + ROI origin. The population is
    computed once, globally (never a per-tile median). Saturation is evaluated
    on the *original acquisition samples* (``saturation_samples``), not the
    corrected flat. Contributor invalidity (``fcorr_invalid``) propagates to an
    invalid response (FLAT_INVALID), never merely excluded from the median.
    """
    F = _as_f32(fcorr, "fcorr")
    shape = F.shape

    if cfa_phase == "mono":
        planes = MONO_PLANES
    elif cfa_phase in ("GRBG", "RGGB", "BGGR", "GBRG"):
        planes = ("G1", "R", "B", "G2")
    else:
        raise InvalidRequestError(f"unsupported cfa_phase for normalization: {cfa_phase!r}")

    inv = (
        np.asarray(fcorr_invalid, dtype=bool)
        if fcorr_invalid is not None
        else np.zeros(shape, dtype=bool)
    )
    if inv.shape != shape:
        raise InvalidRequestError(f"fcorr_invalid shape {inv.shape} vs {shape}")

    sat = (
        np.asarray(saturation_samples, dtype=np.float32)
        if saturation_samples is not None
        else F
    )
    if sat.shape != shape:
        raise InvalidRequestError(f"saturation_samples shape {sat.shape} vs {shape}")

    # Canonical valid population: finite, strictly positive, non-saturated
    # (evaluated on original acquisition), not invalid via the flat's own
    # additive correction.
    population = np.isfinite(F) & (F > 0) & ~inv
    if saturation_limit is not None:
        population &= sat < float(saturation_limit)

    labels = (
        np.full(shape, "mono", dtype=object)
        if cfa_phase == "mono"
        else plane_label_array(shape, cfa_phase, roi_origin)
    )

    scalars: dict[str, Optional[float]] = {}
    R = np.full(shape, np.nan, dtype=np.float32)
    for plane in planes:
        plane_sel = labels == plane
        pop_plane = population & plane_sel
        vals = F[pop_plane]
        if vals.size == 0:
            scalars[plane] = None
            continue
        # Statistical intermediate: exact float64 median.
        s = float(np.median(vals.astype(np.float64)))
        scalars[plane] = s
        # Frame arithmetic: float32 division.
        R[plane_sel] = F[plane_sel] / np.float32(s)

    valid = np.isfinite(R) & (R > 0) & (R > np.float32(floor)) & ~inv
    if saturation_limit is not None:
        valid &= sat < float(saturation_limit)

    # Per-plane counts/ratios use the §7.4 canonical *median population* (before
    # response-floor exclusion); the R > floor guard is an independent response
    # validity condition, not a quality numerator (SCIENCE §8.3).
    plane_counts: dict[str, int] = {}
    plane_totals: dict[str, int] = {}
    plane_ratios: dict[str, float] = {}
    empty_planes: list[str] = []
    for plane in planes:
        plane_sel = labels == plane
        total = int(np.count_nonzero(plane_sel))
        pop_count = int(np.count_nonzero(population & plane_sel))
        plane_counts[plane] = pop_count
        plane_totals[plane] = total
        plane_ratios[plane] = (100.0 * pop_count / total) if total else 0.0
        if pop_count == 0:
            empty_planes.append(plane)

    # >=90% valid-per-plane product screen (separate G1/G2); zero valid samples
    # in any required plane makes the flat unusable globally.
    unusable = [p for p in planes if plane_ratios[p] < DEFAULT_QUALITY_THRESHOLD]
    usable = len(unusable) == 0

    return FlatNormalization(
        R=R,
        scalars=scalars,
        valid=valid,
        plane_counts=plane_counts,
        plane_totals=plane_totals,
        plane_ratios=plane_ratios,
        usable=usable,
        unusable_planes=tuple(unusable),
        empty_planes=tuple(empty_planes),
        cfa_phase=cfa_phase,
        roi_origin=roi_origin,
        floor=float(floor),
        quality_threshold=float(DEFAULT_QUALITY_THRESHOLD),
    )


def final_plane(A: np.ndarray, R: Optional[np.ndarray] = None) -> np.ndarray:
    """Return ``C = A / R`` (float32) when a flat is requested, else ``C = A``."""
    a = _to_f32(A)
    if R is None:
        return a
    r = _to_f32(R)
    _require_same_shape(a, r)
    return a / r


__all__ = [
    "ADDITIVE_MODES",
    "DEFAULT_FLOOR",
    "DEFAULT_QUALITY_THRESHOLD",
    "FLAT_PREP_MODES",
    "FlatNormalization",
    "additive_numerator",
    "final_plane",
    "flat_correction",
    "normalize_flat_response",
]
