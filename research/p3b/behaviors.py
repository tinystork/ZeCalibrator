"""Deterministic physical behaviour of synthetic sites (LOT1).

The generator must be able to *materialise* any declared catalogue class
(:mod:`research.p3b.catalog`). This module maps each of the 20 class names to a
deterministic function of ``(frame_type, frame_ordinal, rng, sensor, site)``.

Scope note: this is the *infrastructure* for materialising the 20 classes, not
the full adversarial matrix (that is a later lot). Only a representative subset
is exercised by the LOT1 tests; the remaining classes still materialise
deterministically but their amplitudes/states are NOT adversarially tuned.

All randomness flows from the single seeded ``rng`` passed in by the generator
(never a global RNG), so identical seeds produce identical bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import SensorSpec, SiteSpec
from .parameters import (
    CENSORED_EXCESS_ADU,
    COSMIC_RAY_AMPLITUDE_ADU,
    DUST_SCALE_ADU,
    HOT_PIXEL_AMPLITUDE_ADU,
    INTERMITTENT_OFF_VALUE_ADU,
    INTERMITTENT_ON_VALUE_ADU,
    NEAR_SATURATION_VALUE_ADU,
    NOISE_EXTREME_SIGMA_ADU,
    OPTICAL_RING_AMPLITUDE_ADU,
    STAR_AMPLITUDE_ADU,
    STAR_DRIFT_PX_PER_FRAME,
    STAR_SIGMA_PX,
    TRANSIENT_AMPLITUDE_ADU,
    WEAK_ANOMALY_AMPLITUDE_ADU,
    get_param,
)


@dataclass(frozen=True)
class SitePatch:
    """A small deterministic ADU footprint to add into the frame.

    ``values`` is a 2-D array (odd dimensions); it is added centred at
    ``(cy, cx)``.
    """

    cy: int
    cx: int
    values: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("SitePatch.values must be 2-D")
        object.__setattr__(self, "values", arr)


def _point(value: float, y: int, x: int) -> SitePatch:
    return SitePatch(y, x, np.array([[float(value)]], dtype=np.float64))


def _gaussian_patch(amplitude: float, sigma: float, cy: float, cx: float, radius: int) -> SitePatch:
    ys = np.arange(-radius, radius + 1, dtype=np.float64)
    xs = ys.copy()
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    r2 = (X * X + Y * Y) / (2.0 * sigma * sigma)
    values = amplitude * np.exp(-r2)
    return SitePatch(int(round(cy)), int(round(cx)), values)


def _temp_factor(temperature_c: float) -> float:
    coeff = get_param("TEMPERATURE_COEFFICIENT_PER_C")
    ref = get_param("REFERENCE_TEMPERATURE_C")
    return 1.0 + coeff * (float(temperature_c) - ref)


def evaluate_site(
    site: SiteSpec,
    frame_type: str,
    frame_ordinal: int,
    rng: np.random.Generator,
    sensor: SensorSpec,
) -> SitePatch:
    """Return the site's deterministic contribution for one frame.

    Dispatches on ``site.cfa_class``. ``frame_ordinal`` is the frame ordinal
    within its group/sequence. ``rng`` is the single seeded generator.
    """
    cls = site.cfa_class
    y, x = site.y, site.x
    hard = site.hard_limit_adu if site.hard_limit_adu is not None else sensor.saturation_limit_adu

    # --- calibration-frame (dark) behaviours -------------------------------
    if frame_type == "dark":
        if cls == "STABLE_ANOMALY_CORRECTED_BY_DARK":
            # Hot pixel present in the dark too -> a representative dark corrects it.
            amp = site.param("amplitude_adu", HOT_PIXEL_AMPLITUDE_ADU)
            return _point(float(amp), y, x)
        if cls == "SIGN_CHANGING_POST_DARK":
            # Present stably in the dark (the residual flips sign around it).
            amp = site.param("amplitude_adu", HOT_PIXEL_AMPLITUDE_ADU)
            return _point(float(amp), y, x)
        # All other classes are light/flat-only; dark has no contribution.
        return _point(0.0, y, x)

    # --- bias frames: no site contribution ---------------------------------
    if frame_type == "bias":
        return _point(0.0, y, x)

    # --- flat frames --------------------------------------------------------
    if frame_type == "flat":
        if cls in ("FLAT_STRUCTURE", "DUST_OR_VIGNETTING", "OPTICAL_STRUCTURE"):
            return _optical_structure_patch(site, sensor, rng)
        return _point(0.0, y, x)

    # --- light frames (the general case) -----------------------------------
    return _light_contribution(site, frame_ordinal, rng, sensor, hard)


def _optical_structure_patch(site: SiteSpec, sensor: SensorSpec, rng: np.random.Generator) -> SitePatch:
    """A static ring/vignetting structure (light and/or flat)."""
    cls = site.cfa_class
    radius = int(site.param("radius_px", 6))
    if cls == "DUST_OR_VIGNETTING":
        amp = site.param("amplitude_adu", DUST_SCALE_ADU)
        ys = np.arange(-radius, radius + 1, dtype=np.float64)
        xs = ys.copy()
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        r = np.sqrt(X * X + Y * Y)
        # Darkening toward the centre of the structure (a dust mote).
        values = -amp * np.exp(-(r ** 2) / (2.0 * (radius / 2.0) ** 2))
        return SitePatch(site.y, site.x, values)
    # OPTICAL_STRUCTURE / FLAT_STRUCTURE: a bright ring.
    amp = site.param("amplitude_adu", OPTICAL_RING_AMPLITUDE_ADU)
    ys = np.arange(-radius, radius + 1, dtype=np.float64)
    xs = ys.copy()
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    r = np.sqrt(X * X + Y * Y)
    values = amp * np.exp(-((r - radius / 2.0) ** 2) / (2.0 * 1.0 ** 2))
    return SitePatch(site.y, site.x, values)


def _light_contribution(
    site: SiteSpec,
    frame_ordinal: int,
    rng: np.random.Generator,
    sensor: SensorSpec,
    hard: float,
) -> SitePatch:
    cls = site.cfa_class
    y, x = site.y, site.x
    n = int(frame_ordinal)

    if cls == "NORMAL":
        return _point(0.0, y, x)

    if cls == "STABLE_ANOMALY_CORRECTED_BY_DARK":
        amp = site.param("amplitude_adu", HOT_PIXEL_AMPLITUDE_ADU)
        return _point(float(amp), y, x)

    if cls == "STABLE_ANOMALY_WITH_MISMATCHED_DARK":
        amp = site.param("amplitude_adu", HOT_PIXEL_AMPLITUDE_ADU)
        return _point(float(amp), y, x)

    if cls == "INTERMITTENT_TWO_STATE":
        on = site.param("on_value_adu", INTERMITTENT_ON_VALUE_ADU)
        off = site.param("off_value_adu", INTERMITTENT_OFF_VALUE_ADU)
        period = int(site.param("period_frames", get_param("INTERMITTENT_PERIOD_FRAMES")))
        value = on if (n % period) == 0 else off
        return _point(float(value), y, x)

    if cls == "INTERMITTENT_MULTI_STATE":
        levels = site.param("levels_adu", (0.0, 1200.0, 2500.0))
        return _point(float(levels[n % len(levels)]), y, x)

    if cls == "INTERMITTENT_CONTINUOUS":
        start = float(site.param("start_adu", 0.0))
        slope = float(site.param("slope_adu_per_frame", 300.0))
        return _point(start + slope * n, y, x)

    if cls == "RARE_HIGH_STATE":
        on = float(site.param("on_value_adu", INTERMITTENT_ON_VALUE_ADU))
        off = float(site.param("off_value_adu", INTERMITTENT_OFF_VALUE_ADU))
        rare = site.param("rare_ordinals", (5,))
        return _point(on if n in rare else off, y, x)

    if cls == "RARE_LOW_STATE":
        on = float(site.param("on_value_adu", INTERMITTENT_ON_VALUE_ADU))
        off = float(site.param("off_value_adu", INTERMITTENT_OFF_VALUE_ADU))
        rare = site.param("rare_ordinals", (5,))
        return _point(off if n in rare else on, y, x)

    if cls == "SIGN_CHANGING_POST_DARK":
        amp = float(site.param("amplitude_adu", HOT_PIXEL_AMPLITUDE_ADU))
        value = amp if (n % 2) == 0 else -amp
        return _point(value, y, x)

    if cls == "CENSORED_ANOMALY":
        # Would-be value exceeds the hard limit; the observed value is censored
        # at the hard limit (the generator clips to the hard limit too).
        return _point(float(hard), y, x)

    if cls == "SINGLE_TRANSIENT":
        frame = int(site.param("transient_ordinal", 3))
        amp = float(site.param("amplitude_adu", TRANSIENT_AMPLITUDE_ADU))
        return _point(amp if n == frame else 0.0, y, x)

    if cls == "STAR_CROSSING_SITE":
        amp = float(site.param("amplitude_adu", STAR_AMPLITUDE_ADU))
        sigma = float(site.param("sigma_px", STAR_SIGMA_PX))
        drift = float(site.param("drift_px_per_frame", STAR_DRIFT_PX_PER_FRAME))
        mid = int(site.param("crossing_ordinal", 3))
        # The star centroid moves along +x; its PSF crosses the site.
        cx = float(x) + drift * (n - mid)
        cy = float(y)
        radius = int(site.param("radius_px", int(np.ceil(3 * sigma))))
        return _gaussian_patch(amp, sigma, cy, cx, radius)

    if cls == "UNDERSAMPLED_STAR_CORE":
        amp = float(site.param("amplitude_adu", STAR_AMPLITUDE_ADU))
        sigma = float(site.param("sigma_px", 0.4))
        return _gaussian_patch(amp, sigma, float(y), float(x), 1)

    if cls == "COSMIC_RAY":
        frame = int(site.param("transient_ordinal", 3))
        amp = float(site.param("amplitude_adu", COSMIC_RAY_AMPLITUDE_ADU))
        return _point(amp if n == frame else 0.0, y, x)

    if cls == "NOISE_EXTREME":
        sigma = float(site.param("sigma_adu", NOISE_EXTREME_SIGMA_ADU))
        return _point(float(rng.normal(0.0, sigma)), y, x)

    if cls in ("FLAT_STRUCTURE", "DUST_OR_VIGNETTING", "OPTICAL_STRUCTURE"):
        # In the light these appear too (the flat carries them as well).
        return _optical_structure_patch(site, sensor, rng)

    if cls == "NEAR_SATURATION":
        value = float(site.param("value_adu", NEAR_SATURATION_VALUE_ADU))
        return _point(value, y, x)

    if cls == "AMBIGUOUS_INSUFFICIENT_EVIDENCE":
        amp = float(site.param("amplitude_adu", WEAK_ANOMALY_AMPLITUDE_ADU))
        return _point(amp, y, x)

    raise ValueError(f"no behaviour for class {cls!r}")


__all__ = ["SitePatch", "evaluate_site"]
