"""P3C-4 LOT 2 — temporally-faithful adversarial corpus (truth side).

Research-only, internal, non-public. This module is the **corpus side** of the
P3C-4 LOT 2 fix: it materialises the adversarial couples that make temporal
discrimination *possible*, by giving **celestial confounders a distinct temporal
evolution** — they follow the sky — while **sensor defects stay fixed in sensor
coordinates**. It does **not** modify ``research/p3b`` and it implements **no
inference rule** (that is LOT 3).

Why a separate layer (option (b) of the LOT 2 mission)
-------------------------------------------------------

The LOT 1 diagnosis (SCIENCE §32, §44) established that the P3B generator
materialises ``UNDERSAMPLED_STAR_CORE`` as a patch at a **fixed sensor
coordinate** in every epoch/group — so a celestial confounder is rendered
identically to a fixed sensor defect, and *no temporal evidence can tell them
apart by principle*. The oracle already declares the wanted physics
(``persisted = NO`` for the star core); the generator rendered the opposite.

The P3B generator is **frozen** (P3B accepted; P3C qualification campaign
asserts the current behaviour), so this module does not edit it. Instead it
provides a **sky-offset-aware materialiser** that reconciles the generated data
with the declared truth: a sky structure is placed at ``(x, y) + sky_offset``,
where ``sky_offset`` is a deterministic dither per epoch, so its signature at a
*fixed* sensor coordinate departs across epochs; a sensor defect ignores the sky
offset and stays fixed.

This module is the **truth side** (like ``development_corpus`` /
``qualification_corpus``): it reads class names and declared structure, and it
may import ``research.p3b.model``. It is deliberately **not** re-exported by
``research.p3c4/__init__.py`` so that importing the P3C-4 *contract* package
still pulls no P3B truth into ``sys.modules`` (the LOT 1 anti-leak guard is
unchanged).

The twenty temporal behaviours (LOT 4 — full catalogue coverage)
----------------------------------------------------------------

LOT 2 covered four behaviours. LOT 4 extends the same principle — *sensor defect
→ fixed in sensor coordinates; sky/optical structure → follows the sky* — to
**all 20 catalogue classes**, so §35 (the five celestial confounders, separately
materialisable) and §36 (the five rare/intermittent classes, separately
materialisable) are applicable. The three families:

* **Sensor defects (fixed coordinate, stationary)** — ``TEMPORAL_NORMAL``,
  ``TEMPORAL_STABLE_CORRECTED_BY_DARK``, ``TEMPORAL_FIXED_DEFECT``,
  ``TEMPORAL_INTERMITTENT_TWO_STATE``, ``TEMPORAL_INTERMITTENT_MULTI_STATE``,
  ``TEMPORAL_INTERMITTENT_CONTINUOUS``, ``TEMPORAL_LOW_OCCUPANCY_INTERMITTENT``
  (RARE_HIGH), ``TEMPORAL_RARE_LOW_STATE``, ``TEMPORAL_SIGN_CHANGING_POST_DARK``,
  ``TEMPORAL_CENSORED_ANOMALY``, ``TEMPORAL_NOISE_EXTREME``,
  ``TEMPORAL_NEAR_SATURATION``, ``TEMPORAL_AMBIGUOUS_INSUFFICIENT_EVIDENCE``.
  The site stays at ``(y, x)`` in every frame; ``persisted`` is the declared
  sensor-defect truth. Censored stays censored (§14); near-saturation stays a
  high-value anomaly *below* the limit.
* **Sky/optical confounders (follow the sky)** — ``TEMPORAL_STAR_CORE``,
  ``TEMPORAL_STAR_CROSSING_SITE``, ``TEMPORAL_OPTICAL_STRUCTURE``,
  ``TEMPORAL_FLAT_STRUCTURE``, ``TEMPORAL_DUST_OR_VIGNETTING``. Their centroid
  shifts by :func:`sky_offset` per epoch (and the star-crossing site also drifts
  per frame), so the signature at the fixed coordinate **departs/drifts** — the
  temporal non-stationarity §35 requires. Their amplitude stays plausible (not
  collapsed), so the *spatial* similarity to a sensor defect stays high; only
  the *displacement* differs.
* **Transients (single frame)** — ``TEMPORAL_SINGLE_EXCURSION``,
  ``TEMPORAL_COSMIC_RAY``. Declared truth ``persisted = NO``,
  ``transient_only = YES``.

The §17 couple A/B remains the principal adversarial pair: their **spatial**
features (amplitude, morphology) are deliberately identical; only the **temporal**
behaviour differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

from research.p3b.model import (
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SensorSpec,
    SiteSpec,
)
from research.p3b.parameters import READ_NOISE_ADU, SKY_BACKGROUND_ADU_PER_S

# ---------------------------------------------------------------------------
# Temporal-behaviour vocabulary (the four adversarial behaviours)
# ---------------------------------------------------------------------------

TEMPORAL_FIXED_DEFECT = "fixed_sensor_defect"
TEMPORAL_STAR_CORE = "undersampled_star_core"
TEMPORAL_LOW_OCCUPANCY_INTERMITTENT = "low_occupancy_intermittent"
TEMPORAL_SINGLE_EXCURSION = "single_excursion"

# LOT 4 — the remaining 16 classes (full catalogue coverage).
TEMPORAL_NORMAL = "normal"
TEMPORAL_STABLE_CORRECTED_BY_DARK = "stable_anomaly_corrected_by_dark"
TEMPORAL_INTERMITTENT_TWO_STATE = "intermittent_two_state"
TEMPORAL_INTERMITTENT_MULTI_STATE = "intermittent_multi_state"
TEMPORAL_INTERMITTENT_CONTINUOUS = "intermittent_continuous"
TEMPORAL_RARE_LOW_STATE = "rare_low_state"
TEMPORAL_SIGN_CHANGING_POST_DARK = "sign_changing_post_dark"
TEMPORAL_CENSORED_ANOMALY = "censored_anomaly"
TEMPORAL_OPTICAL_STRUCTURE = "optical_structure"
TEMPORAL_STAR_CROSSING_SITE = "star_crossing_site"
TEMPORAL_COSMIC_RAY = "cosmic_ray"
TEMPORAL_NOISE_EXTREME = "noise_extreme"
TEMPORAL_FLAT_STRUCTURE = "flat_structure"
TEMPORAL_DUST_OR_VIGNETTING = "dust_or_vignetting"
TEMPORAL_NEAR_SATURATION = "near_saturation"
TEMPORAL_AMBIGUOUS_INSUFFICIENT_EVIDENCE = "ambiguous_insufficient_evidence"

TEMPORAL_KINDS: Tuple[str, ...] = (
    TEMPORAL_FIXED_DEFECT,
    TEMPORAL_STAR_CORE,
    TEMPORAL_LOW_OCCUPANCY_INTERMITTENT,
    TEMPORAL_SINGLE_EXCURSION,
    TEMPORAL_NORMAL,
    TEMPORAL_STABLE_CORRECTED_BY_DARK,
    TEMPORAL_INTERMITTENT_TWO_STATE,
    TEMPORAL_INTERMITTENT_MULTI_STATE,
    TEMPORAL_INTERMITTENT_CONTINUOUS,
    TEMPORAL_RARE_LOW_STATE,
    TEMPORAL_SIGN_CHANGING_POST_DARK,
    TEMPORAL_CENSORED_ANOMALY,
    TEMPORAL_OPTICAL_STRUCTURE,
    TEMPORAL_STAR_CROSSING_SITE,
    TEMPORAL_COSMIC_RAY,
    TEMPORAL_NOISE_EXTREME,
    TEMPORAL_FLAT_STRUCTURE,
    TEMPORAL_DUST_OR_VIGNETTING,
    TEMPORAL_NEAR_SATURATION,
    TEMPORAL_AMBIGUOUS_INSUFFICIENT_EVIDENCE,
)

# Declared class (truth) carried by each behaviour. The class only supplies the
# declared-facts / oracle truth; the *rendered* behaviour below is what this
# module controls (sky-following vs fixed).
KIND_TO_CLASS: Mapping[str, str] = {
    TEMPORAL_FIXED_DEFECT: "STABLE_ANOMALY_WITH_MISMATCHED_DARK",
    TEMPORAL_STAR_CORE: "UNDERSAMPLED_STAR_CORE",
    TEMPORAL_LOW_OCCUPANCY_INTERMITTENT: "RARE_HIGH_STATE",
    TEMPORAL_SINGLE_EXCURSION: "SINGLE_TRANSIENT",
    TEMPORAL_NORMAL: "NORMAL",
    TEMPORAL_STABLE_CORRECTED_BY_DARK: "STABLE_ANOMALY_CORRECTED_BY_DARK",
    TEMPORAL_INTERMITTENT_TWO_STATE: "INTERMITTENT_TWO_STATE",
    TEMPORAL_INTERMITTENT_MULTI_STATE: "INTERMITTENT_MULTI_STATE",
    TEMPORAL_INTERMITTENT_CONTINUOUS: "INTERMITTENT_CONTINUOUS",
    TEMPORAL_RARE_LOW_STATE: "RARE_LOW_STATE",
    TEMPORAL_SIGN_CHANGING_POST_DARK: "SIGN_CHANGING_POST_DARK",
    TEMPORAL_CENSORED_ANOMALY: "CENSORED_ANOMALY",
    TEMPORAL_OPTICAL_STRUCTURE: "OPTICAL_STRUCTURE",
    TEMPORAL_STAR_CROSSING_SITE: "STAR_CROSSING_SITE",
    TEMPORAL_COSMIC_RAY: "COSMIC_RAY",
    TEMPORAL_NOISE_EXTREME: "NOISE_EXTREME",
    TEMPORAL_FLAT_STRUCTURE: "FLAT_STRUCTURE",
    TEMPORAL_DUST_OR_VIGNETTING: "DUST_OR_VIGNETTING",
    TEMPORAL_NEAR_SATURATION: "NEAR_SATURATION",
    TEMPORAL_AMBIGUOUS_INSUFFICIENT_EVIDENCE: "AMBIGUOUS_INSUFFICIENT_EVIDENCE",
}

# Reverse map: declared class -> temporal kind (used by the QUALIFICATION-2
# builder and by coverage tests).
CLASS_TO_KIND: Mapping[str, str] = {cls: kind for kind, cls in KIND_TO_CLASS.items()}

# The five §35 celestial confounders (temporally NON-stationary — departure).
CONFOUNDER_KINDS: Tuple[str, ...] = (
    TEMPORAL_STAR_CORE,
    TEMPORAL_STAR_CROSSING_SITE,
    TEMPORAL_OPTICAL_STRUCTURE,
    TEMPORAL_FLAT_STRUCTURE,
    TEMPORAL_DUST_OR_VIGNETTING,
)

# The five §36 rare/intermittent sensor-defect classes (fixed coordinate,
# variable occupancy).
INTERMITTENT_KINDS: Tuple[str, ...] = (
    TEMPORAL_LOW_OCCUPANCY_INTERMITTENT,
    TEMPORAL_RARE_LOW_STATE,
    TEMPORAL_INTERMITTENT_TWO_STATE,
    TEMPORAL_INTERMITTENT_MULTI_STATE,
    TEMPORAL_INTERMITTENT_CONTINUOUS,
)

# ---------------------------------------------------------------------------
# Numeric constants (corpus generator parameters — none is a product threshold)
# ---------------------------------------------------------------------------

#: Per-epoch sky dither (x-direction, pixels). A real telescope dithers the
#: pointing between acquisition epochs, so a sky structure lands on a different
#: sensor pixel each epoch. 6 px moves a sub-pixel (sigma ~0.4) star core
#: entirely off the site coordinate.
SKY_DITHER_PX = 6

#: Matched amplitude of the §17 adversarial couple (defect == star core).
PAIR_AMPLITUDE_ADU = 4000.0

#: Undersampled star-core PSF sigma (sub-pixel; energy concentrated in ~1 px).
STAR_CORE_SIGMA_PX = 0.4

#: Patch radius of the matched point-like profile (1 -> 3x3 footprint).
PATCH_RADIUS_PX = 1

#: §18 low-occupancy intermittent ON value (matches the P3B intermittent scale).
LOW_OCCUPANCY_ON_ADU = 2500.0

#: §19 single excursion amplitude — huge (near but below the 60000 hard limit),
#: to prove that persistence is independent of amplitude.
SINGLE_EXCURSION_ADU = 50000.0

#: Stable additive anomaly (hot-pixel scale) for STABLE_ANOMALY_CORRECTED_BY_DARK.
HOT_PIXEL_AMPLITUDE_ADU = 1500.0

#: Intermittent ON/OFF values (matches the P3B intermittent scale).
INTERMITTENT_ON_ADU = 2500.0
INTERMITTENT_OFF_ADU = 0.0
#: INTERMITTENT_CONTINUOUS ramp slope (ADU per frame ordinal).
INTERMITTENT_CONTINUOUS_SLOPE_ADU = 300.0
#: INTERMITTENT_MULTI_STATE deterministic levels.
INTERMITTENT_MULTI_LEVELS = (0.0, 1200.0, 2500.0)
#: SIGN_CHANGING_POST_DARK residual magnitude (flips sign around the dark).
SIGN_CHANGING_AMPLITUDE_ADU = 1500.0
#: CENSORED_ANOMALY rendered value — equals the acquisition hard limit.
CENSORED_ANOMALY_ADU = 60000.0
#: COSMIC_RAY single-frame hit amplitude.
COSMIC_RAY_AMPLITUDE_ADU = 30000.0
#: NOISE_EXTREME per-frame noise sigma at the site.
NOISE_EXTREME_SIGMA_ADU = 200.0
#: NEAR_SATURATION value — high but strictly below the 60000 hard limit.
NEAR_SATURATION_ADU = 55000.0
#: AMBIGUOUS_INSUFFICIENT_EVIDENCE weak-anomaly amplitude.
WEAK_ANOMALY_AMPLITUDE_ADU = 60.0

# --- Sky/optical confounder profiles (LOT 4). Amplitudes stay plausible (not
# --- collapsed) so spatial similarity to a sensor defect stays high; only the
# --- centroid displacement differs. -------------------------------------------------

#: Optical-structure amplitude (a soft, extended optical halo / ring).
OPTICAL_STRUCTURE_AMPLITUDE_ADU = 1200.0
OPTICAL_STRUCTURE_SIGMA_PX = 2.0
#: Flat-structure amplitude (a broader flat-field structure).
FLAT_STRUCTURE_AMPLITUDE_ADU = 1200.0
FLAT_STRUCTURE_SIGMA_PX = 2.5
#: Dust/vignetting amplitude — a *negative* (darkening) Gaussian dip.
DUST_OR_VIGNETTING_AMPLITUDE_ADU = -800.0
DUST_OR_VIGNETTING_SIGMA_PX = 2.0
#: Star-crossing-site PSF (a moving star of the same amplitude scale as §17).
STAR_CROSSING_AMPLITUDE_ADU = 4000.0
STAR_CROSSING_SIGMA_PX = 1.5
STAR_CROSSING_DRIFT_PX_PER_FRAME = 1.0
STAR_CROSSING_MID_ORDINAL = 3

#: Patch radius for the wider sky structures (enough to hold a ~2 sigma PSF).
STRUCTURE_PATCH_RADIUS_PX = 6

# Default acquisition structure (mirrors the P3B DEVELOPMENT corpus).
EPOCHS = 2
GROUPS_PER_EPOCH = 2
FRAMES_PER_GROUP = 6

DEFAULT_SITE_X = 8
DEFAULT_SITE_Y = 8


def _build_sensor() -> SensorSpec:
    return SensorSpec(
        instance_id="SYNTH-DET-0001",
        model="SYNTH-S50",
        shape=(96, 128),
        cfa_pattern="GRBG",
        binning=(1, 1),
        roi_origin=(0, 0),
        gain=80.0,
        offset_adu=800.0,
        saturation_limit_adu=60000.0,
        filter="NONE",
    )


# ---------------------------------------------------------------------------
# Sky-following geometry
# ---------------------------------------------------------------------------


def sky_offset(epoch_index: int, group_index: int = 0) -> Tuple[float, float]:
    """Deterministic sky dither ``(dy, dx)`` for a celestial confounder.

    The sky pointing dithers by ``SKY_DITHER_PX`` per epoch (along +x). Within an
    epoch the pointing is fixed, so ``group_index`` does not change the offset.
    A sensor defect ignores this offset (it is fixed in sensor coordinates).
    """
    return (0.0, float(epoch_index) * float(SKY_DITHER_PX))


# ---------------------------------------------------------------------------
# Shared point-like spatial profile (identical for the §17 couple)
# ---------------------------------------------------------------------------


def gaussian_values(amplitude: float, sigma: float, radius: int) -> np.ndarray:
    """A 2-D Gaussian patch (odd side ``2*radius+1``), peak at the centre.

    ``amplitude`` is the peak ADU; ``sigma`` is the PSF sigma in pixels. This is
    the **single** spatial profile used by both the fixed defect and the star
    core, so the §17 couple is spatially identical by construction.
    """
    ys = np.arange(-radius, radius + 1, dtype=np.float64)
    xs = ys.copy()
    y_grid, x_grid = np.meshgrid(ys, xs, indexing="ij")
    r2 = (x_grid * x_grid + y_grid * y_grid) / (2.0 * sigma * sigma)
    return amplitude * np.exp(-r2)


def _point(value: float) -> np.ndarray:
    return np.array([[float(value)]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Temporal behaviour — the deterministic site contribution per light frame
# ---------------------------------------------------------------------------


def site_contribution(
    temporal_kind: str,
    y: int,
    x: int,
    epoch_index: int,
    group_index: int,
    frame_ordinal: int,
    *,
    rng: "np.random.Generator | None" = None,
) -> Tuple[float, float, np.ndarray]:
    """Return ``(cy, cx, values)`` — the site's deterministic patch for one frame.

    ``cy``/``cx`` is the patch centre (sensor coordinates); ``values`` is the 2-D
    ADU footprint to add centred there. Celestial confounders shift ``cy``/``cx``
    by :func:`sky_offset`; sensor defects do not. ``rng`` is used only by
    ``TEMPORAL_NOISE_EXTREME`` (a single deterministic noise draw per frame).
    """
    # --- sensor defects: fixed coordinate, stationary ------------------------
    if temporal_kind == TEMPORAL_FIXED_DEFECT:
        cy, cx = float(y), float(x)
        values = gaussian_values(PAIR_AMPLITUDE_ADU, STAR_CORE_SIGMA_PX, PATCH_RADIUS_PX)
        return cy, cx, values

    if temporal_kind == TEMPORAL_STABLE_CORRECTED_BY_DARK:
        # A stable additive hot pixel (present in light AND dark — the dark
        # corrects it). Fixed coordinate, stationary.
        return float(y), float(x), _point(HOT_PIXEL_AMPLITUDE_ADU)

    if temporal_kind == TEMPORAL_NORMAL:
        return float(y), float(x), _point(0.0)

    if temporal_kind == TEMPORAL_INTERMITTENT_TWO_STATE:
        # Toggles deterministically between two levels (period 2 frames).
        value = INTERMITTENT_ON_ADU if (frame_ordinal % 2) == 0 else INTERMITTENT_OFF_ADU
        return float(y), float(x), _point(value)

    if temporal_kind == TEMPORAL_INTERMITTENT_MULTI_STATE:
        value = INTERMITTENT_MULTI_LEVELS[frame_ordinal % len(INTERMITTENT_MULTI_LEVELS)]
        return float(y), float(x), _point(value)

    if temporal_kind == TEMPORAL_INTERMITTENT_CONTINUOUS:
        # A deterministic ramp across frames (continuous drift, fixed coordinate).
        value = INTERMITTENT_CONTINUOUS_SLOPE_ADU * float(frame_ordinal)
        return float(y), float(x), _point(value)

    if temporal_kind == TEMPORAL_LOW_OCCUPANCY_INTERMITTENT:
        # ON for the whole of one group (epoch 0, group 0); OFF everywhere else.
        on = epoch_index == 0 and group_index == 0
        value = LOW_OCCUPANCY_ON_ADU if on else 0.0
        return float(y), float(x), _point(value)

    if temporal_kind == TEMPORAL_RARE_LOW_STATE:
        # Usually high, rarely (deterministically) low: OFF for one group.
        off = epoch_index == 0 and group_index == 0
        value = INTERMITTENT_OFF_ADU if off else LOW_OCCUPANCY_ON_ADU
        return float(y), float(x), _point(value)

    if temporal_kind == TEMPORAL_SIGN_CHANGING_POST_DARK:
        # Residual flips sign around the dark (fixed coordinate, conflicting).
        value = SIGN_CHANGING_AMPLITUDE_ADU if (frame_ordinal % 2) == 0 else -SIGN_CHANGING_AMPLITUDE_ADU
        return float(y), float(x), _point(value)

    if temporal_kind == TEMPORAL_CENSORED_ANOMALY:
        # Hits the acquisition hard limit every frame (censored, §14).
        return float(y), float(x), _point(CENSORED_ANOMALY_ADU)

    if temporal_kind == TEMPORAL_NOISE_EXTREME:
        # Extreme per-frame noise at the site (fixed coordinate).
        r = rng if rng is not None else np.random.default_rng(0)
        value = float(r.normal(0.0, NOISE_EXTREME_SIGMA_ADU))
        return float(y), float(x), _point(value)

    if temporal_kind == TEMPORAL_NEAR_SATURATION:
        # A high-value anomaly persistently below the hard limit (never censored).
        return float(y), float(x), _point(NEAR_SATURATION_ADU)

    if temporal_kind == TEMPORAL_AMBIGUOUS_INSUFFICIENT_EVIDENCE:
        # A weak anomaly at the fixed coordinate.
        return float(y), float(x), _point(WEAK_ANOMALY_AMPLITUDE_ADU)

    # --- transients: single frame, never repeated -----------------------------
    if temporal_kind == TEMPORAL_SINGLE_EXCURSION:
        # Exactly one frame: the first frame of the first group.
        on = epoch_index == 0 and group_index == 0 and frame_ordinal == 0
        value = SINGLE_EXCURSION_ADU if on else 0.0
        return float(y), float(x), _point(value)

    if temporal_kind == TEMPORAL_COSMIC_RAY:
        # A bright cosmic-ray hit in a single frame.
        on = epoch_index == 0 and group_index == 0 and frame_ordinal == 3
        value = COSMIC_RAY_AMPLITUDE_ADU if on else 0.0
        return float(y), float(x), _point(value)

    # --- sky/optical confounders: follow the sky ------------------------------
    if temporal_kind == TEMPORAL_STAR_CORE:
        dy, dx = sky_offset(epoch_index, group_index)
        cy, cx = float(y) + dy, float(x) + dx
        values = gaussian_values(PAIR_AMPLITUDE_ADU, STAR_CORE_SIGMA_PX, PATCH_RADIUS_PX)
        return cy, cx, values

    if temporal_kind == TEMPORAL_STAR_CROSSING_SITE:
        # A moving star PSF crossing the site: drifts per frame AND follows the
        # sky per epoch (departure + drift — §35).
        dy, dx = sky_offset(epoch_index, group_index)
        drift = STAR_CROSSING_DRIFT_PX_PER_FRAME * (frame_ordinal - STAR_CROSSING_MID_ORDINAL)
        cy = float(y) + dy
        cx = float(x) + dx + drift
        values = gaussian_values(
            STAR_CROSSING_AMPLITUDE_ADU, STAR_CROSSING_SIGMA_PX, STRUCTURE_PATCH_RADIUS_PX
        )
        return cy, cx, values

    if temporal_kind == TEMPORAL_OPTICAL_STRUCTURE:
        dy, dx = sky_offset(epoch_index, group_index)
        cy, cx = float(y) + dy, float(x) + dx
        values = gaussian_values(
            OPTICAL_STRUCTURE_AMPLITUDE_ADU, OPTICAL_STRUCTURE_SIGMA_PX, STRUCTURE_PATCH_RADIUS_PX
        )
        return cy, cx, values

    if temporal_kind == TEMPORAL_FLAT_STRUCTURE:
        dy, dx = sky_offset(epoch_index, group_index)
        cy, cx = float(y) + dy, float(x) + dx
        values = gaussian_values(
            FLAT_STRUCTURE_AMPLITUDE_ADU, FLAT_STRUCTURE_SIGMA_PX, STRUCTURE_PATCH_RADIUS_PX
        )
        return cy, cx, values

    if temporal_kind == TEMPORAL_DUST_OR_VIGNETTING:
        # A dust/vignetting dip: negative Gaussian (darkening) that follows the sky.
        dy, dx = sky_offset(epoch_index, group_index)
        cy, cx = float(y) + dy, float(x) + dx
        values = gaussian_values(
            DUST_OR_VIGNETTING_AMPLITUDE_ADU, DUST_OR_VIGNETTING_SIGMA_PX, STRUCTURE_PATCH_RADIUS_PX
        )
        return cy, cx, values

    raise ValueError(f"unknown temporal kind {temporal_kind!r}")


# ---------------------------------------------------------------------------
# Scenario construction (structure + declared truth)
# ---------------------------------------------------------------------------


def build_scenario(
    temporal_kind: str,
    *,
    epochs: int = EPOCHS,
    groups_per_epoch: int = GROUPS_PER_EPOCH,
    frames_per_group: int = FRAMES_PER_GROUP,
    seed: int = 0,
    site_x: int = DEFAULT_SITE_X,
    site_y: int = DEFAULT_SITE_Y,
) -> ScenarioSpec:
    """Build a coherent light-only :class:`ScenarioSpec` for ``temporal_kind``.

    The site is declared with the class in :data:`KIND_TO_CLASS` (so the declared
    facts / oracle carry the wanted truth). Deterministic: same ``(kind, seed,
    structure)`` ⇒ same scenario. Light frames only: the temporal behaviours are
    all light-frame behaviours.
    """
    if temporal_kind not in KIND_TO_CLASS:
        raise ValueError(f"unknown temporal kind {temporal_kind!r}")
    sensor = _build_sensor()
    class_name = KIND_TO_CLASS[temporal_kind]

    counter = 0
    epoch_specs = []
    for e in range(epochs):
        groups = []
        for g in range(groups_per_epoch):
            frames = tuple(
                FrameSpec(
                    frame_id=f"light{counter + i}",
                    frame_type="light",
                    epoch_id=f"e{e}",
                    group_id=f"g{e}_{g}",
                    ordinal=i,
                )
                for i in range(frames_per_group)
            )
            counter += frames_per_group
            groups.append(GroupSpec(group_id=f"g{e}_{g}", frames=frames))
        epoch_specs.append(EpochSpec(epoch_id=f"e{e}", groups=tuple(groups)))

    site = SiteSpec(
        site_id="site0",
        x=site_x,
        y=site_y,
        cfa_class=class_name,
        censored=(class_name == "CENSORED_ANOMALY"),
        hard_limit_adu=(sensor.saturation_limit_adu if class_name == "CENSORED_ANOMALY" else None),
    )
    return ScenarioSpec(
        name=f"p3c4:{temporal_kind}:{seed}",
        seed=seed,
        sensor=sensor,
        epochs=tuple(epoch_specs),
        sites=(site,),
    )


# ---------------------------------------------------------------------------
# Materialisation (physical uint16 ADU arrays, sky-offset aware)
# ---------------------------------------------------------------------------


def _base_float(sensor: SensorSpec, exposure_s: float, rng: np.random.Generator) -> np.ndarray:
    ny, nx = sensor.shape
    base = np.full((ny, nx), float(sensor.offset_adu), dtype=np.float64)
    base += SKY_BACKGROUND_ADU_PER_S * float(exposure_s)
    base += rng.normal(0.0, READ_NOISE_ADU, size=(ny, nx))
    return base


def _add_patch(frame: np.ndarray, cy: float, cx: float, values: np.ndarray) -> None:
    h, w = values.shape
    y0 = int(round(cy)) - h // 2
    x0 = int(round(cx)) - w // 2
    ny, nx = frame.shape
    fy0 = max(0, y0)
    fx0 = max(0, x0)
    fy1 = min(ny, y0 + h)
    fx1 = min(nx, x0 + w)
    if fy1 <= fy0 or fx1 <= fx0:
        return
    frame[fy0:fy1, fx0:fx1] += values[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0]


def _quantize(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0.0, 65535.0).astype(np.uint16)


def materialise_frames(
    scenario: ScenarioSpec,
    temporal_kind: str,
    *,
    seed: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Render the physical ADU frames for ``scenario`` under ``temporal_kind``.

    Returns ``{frame_id: uint16 array}`` (the same convention as
    ``generate_corpus.frame_arrays``). The single site is rendered by
    :func:`site_contribution` with the epoch/group index threaded through, so
    celestial confounders follow the sky and sensor defects stay fixed.
    Deterministic given ``seed`` (defaults to ``scenario.seed``).
    """
    if temporal_kind not in KIND_TO_CLASS:
        raise ValueError(f"unknown temporal kind {temporal_kind!r}")
    seed = scenario.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    sensor = scenario.sensor
    site = scenario.sites[0]

    out: dict[str, np.ndarray] = {}
    for epoch_index, epoch in enumerate(scenario.epochs):
        for group_index, group in enumerate(epoch.groups):
            for frame in group.frames:
                arr = _base_float(sensor, frame.exposure_s, rng)
                cy, cx, values = site_contribution(
                    temporal_kind, site.y, site.x, epoch_index, group_index, frame.ordinal,
                    rng=rng,
                )
                _add_patch(arr, cy, cx, values)
                # Censoring (§14): a censored site's value is clipped to its hard
                # limit, mirroring research.p3b.generator._render_frame.
                if site.censored:
                    limit = (
                        site.hard_limit_adu
                        if site.hard_limit_adu is not None
                        else sensor.saturation_limit_adu
                    )
                    if arr[site.y, site.x] > limit:
                        arr[site.y, site.x] = limit
                out[frame.frame_id] = _quantize(arr)
    return out


# ---------------------------------------------------------------------------
# Measurement helpers (temporal + spatial, used by the tests)
# ---------------------------------------------------------------------------


def fixed_coordinate_series(
    frame_arrays: Mapping[str, np.ndarray],
    light_frames: Tuple[FrameSpec, ...],
    x: int,
    y: int,
) -> np.ndarray:
    """The site value at sensor ``(x, y)`` for each light frame, in canonical order.

    ``light_frames`` must be the ordered light frames (e.g. ``scenario.light_frames()``).
    """
    return np.asarray(
        [float(frame_arrays[f.frame_id][y, x]) for f in light_frames],
        dtype=np.float64,
    )


def local_window(frame: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """The ``(2*radius+1)``-square window around sensor ``(x, y)`` (bounds-clipped)."""
    ny, nx = frame.shape
    y0 = max(0, y - radius)
    y1 = min(ny, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(nx, x + radius + 1)
    return frame[y0:y1, x0:x1].astype(np.float64)


@dataclass(frozen=True)
class SpatialSimilarity:
    """Explicit local spatial similarity between two patches.

    ``amplitude_ratio`` is ``min/max`` of the two peak |amplitudes| (1.0 ==
    identical amplitude); ``shape_correlation`` is the cosine similarity of the
    two flattened profiles (1.0 == identical morphology). Both are unit-free and
    independent of any product threshold.
    """

    peak_a: float
    peak_b: float
    amplitude_ratio: float
    shape_correlation: float


def profile_similarity(a: np.ndarray, b: np.ndarray) -> SpatialSimilarity:
    """Amplitude + shape similarity of two equal-shaped 2-D profiles."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"profile shapes differ: {a.shape} vs {b.shape}")

    a_flat = a.ravel()
    b_flat = b.ravel()
    peak_a = float(np.max(np.abs(a_flat)))
    peak_b = float(np.max(np.abs(b_flat)))
    peak_max = max(peak_a, peak_b)
    amplitude_ratio = (min(peak_a, peak_b) / peak_max) if peak_max > 0.0 else 1.0

    na = float(np.linalg.norm(a_flat))
    nb = float(np.linalg.norm(b_flat))
    if na == 0.0 or nb == 0.0:
        shape_correlation = 1.0 if na == nb else 0.0
    else:
        shape_correlation = float(np.dot(a_flat, b_flat) / (na * nb))

    return SpatialSimilarity(
        peak_a=peak_a,
        peak_b=peak_b,
        amplitude_ratio=amplitude_ratio,
        shape_correlation=shape_correlation,
    )


__all__ = [
    "TEMPORAL_FIXED_DEFECT",
    "TEMPORAL_STAR_CORE",
    "TEMPORAL_LOW_OCCUPANCY_INTERMITTENT",
    "TEMPORAL_SINGLE_EXCURSION",
    "TEMPORAL_NORMAL",
    "TEMPORAL_STABLE_CORRECTED_BY_DARK",
    "TEMPORAL_INTERMITTENT_TWO_STATE",
    "TEMPORAL_INTERMITTENT_MULTI_STATE",
    "TEMPORAL_INTERMITTENT_CONTINUOUS",
    "TEMPORAL_RARE_LOW_STATE",
    "TEMPORAL_SIGN_CHANGING_POST_DARK",
    "TEMPORAL_CENSORED_ANOMALY",
    "TEMPORAL_OPTICAL_STRUCTURE",
    "TEMPORAL_STAR_CROSSING_SITE",
    "TEMPORAL_COSMIC_RAY",
    "TEMPORAL_NOISE_EXTREME",
    "TEMPORAL_FLAT_STRUCTURE",
    "TEMPORAL_DUST_OR_VIGNETTING",
    "TEMPORAL_NEAR_SATURATION",
    "TEMPORAL_AMBIGUOUS_INSUFFICIENT_EVIDENCE",
    "TEMPORAL_KINDS",
    "KIND_TO_CLASS",
    "CLASS_TO_KIND",
    "CONFOUNDER_KINDS",
    "INTERMITTENT_KINDS",
    "SKY_DITHER_PX",
    "PAIR_AMPLITUDE_ADU",
    "STAR_CORE_SIGMA_PX",
    "PATCH_RADIUS_PX",
    "LOW_OCCUPANCY_ON_ADU",
    "SINGLE_EXCURSION_ADU",
    "HOT_PIXEL_AMPLITUDE_ADU",
    "INTERMITTENT_ON_ADU",
    "INTERMITTENT_OFF_ADU",
    "INTERMITTENT_CONTINUOUS_SLOPE_ADU",
    "INTERMITTENT_MULTI_LEVELS",
    "SIGN_CHANGING_AMPLITUDE_ADU",
    "CENSORED_ANOMALY_ADU",
    "COSMIC_RAY_AMPLITUDE_ADU",
    "NOISE_EXTREME_SIGMA_ADU",
    "NEAR_SATURATION_ADU",
    "WEAK_ANOMALY_AMPLITUDE_ADU",
    "OPTICAL_STRUCTURE_AMPLITUDE_ADU",
    "OPTICAL_STRUCTURE_SIGMA_PX",
    "FLAT_STRUCTURE_AMPLITUDE_ADU",
    "FLAT_STRUCTURE_SIGMA_PX",
    "DUST_OR_VIGNETTING_AMPLITUDE_ADU",
    "DUST_OR_VIGNETTING_SIGMA_PX",
    "STAR_CROSSING_AMPLITUDE_ADU",
    "STAR_CROSSING_SIGMA_PX",
    "STAR_CROSSING_DRIFT_PX_PER_FRAME",
    "STAR_CROSSING_MID_ORDINAL",
    "STRUCTURE_PATCH_RADIUS_PX",
    "EPOCHS",
    "GROUPS_PER_EPOCH",
    "FRAMES_PER_GROUP",
    "DEFAULT_SITE_X",
    "DEFAULT_SITE_Y",
    "SpatialSimilarity",
    "sky_offset",
    "gaussian_values",
    "site_contribution",
    "build_scenario",
    "materialise_frames",
    "fixed_coordinate_series",
    "local_window",
    "profile_similarity",
]
