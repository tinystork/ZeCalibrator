"""Tagged numeric constants for the P3B synthetic-corpus harness.

Marking discipline (LOT1 requirement 10):

* every numeric constant used to *fabricate* a scenario is marked
  ``SYNTHETIC_GENERATOR_PARAMETER``;
* every numeric constant used only to *visualise / analyse* is marked
  ``EXPLORATORY``;
* **no** constant is presented or named ``PRODUCT THRESHOLD`` — there are no
  product thresholds in this corpus (all numeric budgets are deferred to a later
  owner gate, ``SCIENCE_CONTRACT.md`` §13.10).

Constants are registered through :func:`param`, which records their value and
kind in a module-level registry so tests can audit the marking discipline
programmatically (see ``tests/p3b/test_parameters.py``).
"""

from __future__ import annotations

SYNTHETIC_GENERATOR_PARAMETER = "SYNTHETIC_GENERATOR_PARAMETER"
EXPLORATORY = "EXPLORATORY"

# name -> (value, kind). Immutable to the outside world via all_params().
_PARAMS: dict[str, tuple[object, str]] = {}


def param(name: str, value, kind: str = SYNTHETIC_GENERATOR_PARAMETER):
    """Register (and return) a named constant with its marking kind."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"parameter name must be a non-empty string, got {name!r}")
    if name in _PARAMS:
        raise RuntimeError(f"parameter {name!r} registered twice")
    _PARAMS[name] = (value, kind)
    return value


def get_param(name: str):
    """Return the value of a registered parameter."""
    try:
        return _PARAMS[name][0]
    except KeyError as exc:
        raise KeyError(f"unknown parameter {name!r}") from exc


def param_kind(name: str) -> str:
    """Return the marking kind of a registered parameter."""
    try:
        return _PARAMS[name][1]
    except KeyError as exc:
        raise KeyError(f"unknown parameter {name!r}") from exc


def all_params() -> dict[str, tuple[object, str]]:
    """Return a copy of the whole parameter registry (name -> (value, kind))."""
    return dict(_PARAMS)


# ---------------------------------------------------------------------------
# Fabrication constants — SYNTHETIC_GENERATOR_PARAMETER
# ---------------------------------------------------------------------------

# Default sensor/array geometry: small on purpose (LOT1 keeps everything fast;
# order of 64x64 .. 128x128). (ny, nx).
DEFAULT_SHAPE = param("DEFAULT_SHAPE", (96, 128))

DEFAULT_SEED = param("DEFAULT_SEED", 0)

# Acquisition facts (declared, not inferred — SCIENCE §5).
DEFAULT_GAIN = param("DEFAULT_GAIN", 80.0)
DEFAULT_OFFSET_ADU = param("DEFAULT_OFFSET_ADU", 800.0)
DEFAULT_SATURATION_LIMIT_ADU = param("DEFAULT_SATURATION_LIMIT_ADU", 60000.0)
DEFAULT_EXPOSURE_S = param("DEFAULT_EXPOSURE_S", 10.0)
DEFAULT_TEMPERATURE_C = param("DEFAULT_TEMPERATURE_C", 20.0)

# Temperature-dependent dark model (linear, deterministic; not a physical law).
REFERENCE_TEMPERATURE_C = param("REFERENCE_TEMPERATURE_C", 20.0)
TEMPERATURE_COEFFICIENT_PER_C = param("TEMPERATURE_COEFFICIENT_PER_C", 0.03)

# Signal pedestals / rates (physical ADU).
DARK_CURRENT_ADU_PER_S = param("DARK_CURRENT_ADU_PER_S", 0.4)
SKY_BACKGROUND_ADU_PER_S = param("SKY_BACKGROUND_ADU_PER_S", 60.0)
FLAT_PEDESTAL_ADU = param("FLAT_PEDESTAL_ADU", 20000.0)
READ_NOISE_ADU = param("READ_NOISE_ADU", 5.0)

# FITS storage convention (observed on the real corpus, applied exactly once).
FITS_BSCALE = param("FITS_BSCALE", 1.0)
FITS_BZERO = param("FITS_BZERO", 32768.0)

# Site physical behaviours (per-class defaults; per-site overrides via SiteSpec).
HOT_PIXEL_AMPLITUDE_ADU = param("HOT_PIXEL_AMPLITUDE_ADU", 1500.0)
INTERMITTENT_ON_VALUE_ADU = param("INTERMITTENT_ON_VALUE_ADU", 2500.0)
INTERMITTENT_OFF_VALUE_ADU = param("INTERMITTENT_OFF_VALUE_ADU", 0.0)
INTERMITTENT_PERIOD_FRAMES = param("INTERMITTENT_PERIOD_FRAMES", 2)
TRANSIENT_AMPLITUDE_ADU = param("TRANSIENT_AMPLITUDE_ADU", 5000.0)
COSMIC_RAY_AMPLITUDE_ADU = param("COSMIC_RAY_AMPLITUDE_ADU", 30000.0)
STAR_AMPLITUDE_ADU = param("STAR_AMPLITUDE_ADU", 4000.0)
STAR_SIGMA_PX = param("STAR_SIGMA_PX", 1.5)
STAR_DRIFT_PX_PER_FRAME = param("STAR_DRIFT_PX_PER_FRAME", 1.0)
CENSORED_EXCESS_ADU = param("CENSORED_EXCESS_ADU", 90000.0)
NEAR_SATURATION_VALUE_ADU = param("NEAR_SATURATION_VALUE_ADU", 55000.0)
NOISE_EXTREME_SIGMA_ADU = param("NOISE_EXTREME_SIGMA_ADU", 200.0)
WEAK_ANOMALY_AMPLITUDE_ADU = param("WEAK_ANOMALY_AMPLITUDE_ADU", 60.0)
DUST_SCALE_ADU = param("DUST_SCALE_ADU", 800.0)
OPTICAL_RING_AMPLITUDE_ADU = param("OPTICAL_RING_AMPLITUDE_ADU", 1200.0)

# ---------------------------------------------------------------------------
# Visualisation / analysis constants — EXPLORATORY
# ---------------------------------------------------------------------------

# Quick-look stretch percentiles for inspecting a generated frame. Used only by
# :func:`research.p3b.generator.quicklook_stats` — never by generation.
QUICKLOOK_PERCENTILE_LOW = param("QUICKLOOK_PERCENTILE_LOW", 5.0, EXPLORATORY)
QUICKLOOK_PERCENTILE_HIGH = param("QUICKLOOK_PERCENTILE_HIGH", 95.0, EXPLORATORY)
