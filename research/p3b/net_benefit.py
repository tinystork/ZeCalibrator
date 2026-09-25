"""Net-benefit harness (LOT6) — paired measurement, never a verdict.

Internal, non-public, never imported by ``zecalibrator.api.v1``.

Measures, as a **paired couple**, two branches over a synthetic scenario
(``docs/SCIENCE_CONTRACT.md`` §13.9):

* **branch A** = ordinary representative calibration alone;
* **branch B** = the *identical* pipeline plus targeted reconstruction of the
  qualified site (via the frozen research witness operator).

The only scientific difference is reconstruction ON/OFF. The observable is the
same on both sides: same geometry, same post-calibration CFA, same local
reference, same aggregation method.

What the harness measures (and the invariant each one carries):

* **residual reduction at the site** — branch A vs B, in ADU and as a fraction
  of the baseline residual, with the mean centred stamp (profile) and the
  fraction of that profile carried by the centre (never a bare number, lesson
  P2I/S1).
* **collateral degradation** — local background bias, local noise, nearby-star
  integrity *with the published distance* (lesson P2I/S2), CFA structure,
  photometric impact, new-artifact creation, and untouched-domain invariance
  (which must be 0 outside the site neighbourhood).
* **localization** — the B − A difference must be concentrated at the site, not
  spread globally; measured with a spatial witness (shifted positions).

Boundary (enforced here, verified by ``tests/p3b/test_net_benefit.py``):

* the reconstruction operator is the frozen research witness ONLY
  (``RESEARCH_WITNESS_ONLY``); no operator is searched, no neighbourhood size is
  tuned;
* the *decision to apply* passes through the frozen LOT3 ``PreparationPlan``
  (uniform per site over the whole run); no per-frame variation;
* features are consumed through ``compute_features`` (the censoring-filtered
  path); the harness never calls a ``features.py`` primitive on an unfiltered
  censored series;
* **no threshold, no acceptance rule**: the harness measures and produces
  distributions; it chooses no operating point, maximises nothing, imposes no
  ordering, and produces **no acceptance verdict** — the acceptance rule is
  DEFERRED to the owner (HUMAN_GATE);
* a censored measurement never supports a quantitative inference (SCIENCE
  §13.4): censored frames are excluded from every quantitative aggregate, and a
  censored site yields no reduction / no collateral deltas.

Every numeric constant is registered through :func:`research.p3b.parameters.param`
and marked ``EXPLORATORY`` (analysis) — never a product threshold. The only
scientific numeric constants are unit conversions; the remaining ``0``/``1`` are
structural counters / indices / bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from .declared_facts import build_declared_facts, evidence_packet
from .features import compute_features
from .generator import cfa_plane_label
from .metrics import CollateralDegradation, NetBenefitPair
from .model import ScenarioSpec
from .parameters import EXPLORATORY, param
from .preparation_plan import (
    DONORS_AVAILABLE,
    DONORS_UNAVAILABLE,
    RUN_WIDE_ELIGIBLE,
    GeometryBinding,
    SiteQualification,
    preflight,
)
from .qualification_policy import ACTION_ELIGIBLE, NET_BENEFIT_ESTABLISHED, YES, evaluate
from .reconstruction import (
    OPERATOR_VERSION,
    RESEARCH_WITNESS_ONLY,
    donor_offsets_available,
    reconstruct_site_pixel,
)

# ---------------------------------------------------------------------------
# Tagged numeric constants — EXPLORATORY (analysis, never a product threshold)
# ---------------------------------------------------------------------------

# Radius (Chebyshev) of the mean centred stamp used for the residual profile and
# for the centre-carrying fraction. An analysis parameter, not a product rule.
PROFILE_STAMP_RADIUS = param("NET_BENEFIT_PROFILE_STAMP_RADIUS", 3, EXPLORATORY)

# Radius (Chebyshev) of the same-CFA local window used as the *paired* local
# reference for the residual (site value minus same-CFA local median). Analysis
# only; never a product rule.
LOCAL_REFERENCE_RADIUS = param("NET_BENEFIT_LOCAL_REFERENCE_RADIUS", 3, EXPLORATORY)

# Same-CFA annulus [inner, outer] (Chebyshev) around the site used for the
# collateral background-bias / noise controls. Analysis only.
COLLATERAL_ANNULUS = param("NET_BENEFIT_COLLATERAL_ANNULUS", (4, 8), EXPLORATORY)

# Witness offsets for the localization (spatial) control: the B − A effect at
# the site must exceed the same measure at these shifted positions.
LOCALIZATION_OFFSETS = param(
    "NET_BENEFIT_LOCALIZATION_OFFSETS",
    ((0, 8), (0, -8), (8, 0), (-8, 0), (12, 12), (-12, -12)),
    EXPLORATORY,
)


# ---------------------------------------------------------------------------
# Result value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Localization:
    """Spatial concentration of the B − A difference.

    ``site_magnitude`` is the mean |B − A| at the site over frames;
    ``shifted_magnitudes`` is the same measure at shifted witness positions
    (aligned with ``shifted_offsets``). ``concentration`` is the site's share of
    the total magnitude across the site plus the shifted positions: 1.0 when the
    effect is fully concentrated at the site (all shifted positions 0).
    """

    site_magnitude: Optional[float]  # ADU (mean |B-A| at site)
    shifted_offsets: Tuple[Tuple[int, int], ...]
    shifted_magnitudes: Tuple[float, ...]  # ADU
    concentration: Optional[float]  # dimensionless, in [0, 1]


@dataclass(frozen=True)
class SiteNetBenefit:
    """The measured net benefit for ONE site (paired couple + controls)."""

    site_id: str
    x: int
    y: int
    cfa_plane: str
    run_wide_applicability: str  # from the frozen LOT3 plan
    reconstructed: bool  # whether branch B reconstructed this site
    censored: bool  # site declared censored (no quantitative inference)
    # LOT5 typed pair: reduction (ADU + fraction) + collateral degradation.
    pair: NetBenefitPair
    # LOT6 enrichment — the profile, never a bare number (lesson P2I/S1).
    baseline_residual_adu: Optional[float]
    residual_after_adu: Optional[float]
    mean_profile_a: Tuple[Tuple[float, ...], ...]
    mean_profile_b: Tuple[Tuple[float, ...], ...]
    profile_radius: int
    center_fraction_a: Optional[float]
    center_fraction_b: Optional[float]
    n_frames: int  # non-censored frames used in the means
    # LOT6 localization (spatial witness).
    localization: Localization


@dataclass(frozen=True)
class NetBenefitHarnessResult:
    """The full harness result over a scenario."""

    scenario_name: str
    seed: int
    operator_id: str
    operator_version: str
    sites: Tuple[SiteNetBenefit, ...]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def representative_dark(scenario, frame_arrays) -> np.ndarray:
    """Median of the scenario's dark frames — the representative additive reference.

    A convenience for building the ``dark_reference`` the harness consumes. It
    medians the materialised dark frames (physical ADU) and never chooses or
    validates a calibration (that is upstream, LOT1/LOT2).
    """
    dark_ids = [f.frame_id for f in scenario.all_frames() if f.frame_type == "dark"]
    if not dark_ids:
        raise ValueError("scenario has no dark frames to build a representative dark")
    stack = np.stack([frame_arrays[fid] for fid in dark_ids], axis=0).astype(np.float64)
    return np.median(stack, axis=0)


def _light_ids(scenario) -> Tuple[str, ...]:
    return tuple(f.frame_id for f in scenario.light_frames())


def _same_cfa_local_median(frame, y, x, pattern, radius) -> float:
    """Median same-CFA neighbour value (site excluded); NaN with no neighbour.

    The *paired* local reference for the post-calibration residual — a geometric
    same-CFA local median, not a ``features.py`` primitive and never fed a raw
    censored series (censoring is handled by the caller, which skips censored
    frames before this is used in a mean).
    """
    ny, nx = frame.shape
    target = cfa_plane_label(y, x, pattern)
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and cfa_plane_label(yy, xx, pattern) == target:
                vals.append(float(frame[yy, xx]))
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _extract_stamp(frame, y, x, radius, ny, nx) -> np.ndarray:
    """Centred stamp around ``(y, x)`` with NaN where out of bounds."""
    out = np.full((2 * radius + 1, 2 * radius + 1), np.nan, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx:
                out[dy + radius, dx + radius] = frame[yy, xx]
    return out


def _profile_to_tuple(arr: Optional[np.ndarray]) -> Tuple[Tuple[float, ...], ...]:
    if arr is None:
        return ()
    return tuple(tuple(float(v) for v in row) for row in arr)


def _mean_residual_profile(cfa, light_ids, site, sensor, censored_mask):
    """Mean centred stamp of ``cfa − local same-CFA reference`` over non-censored frames.

    Returns ``None`` when no non-censored frame provides a finite reference.
    """
    pattern = sensor.cfa_pattern
    ny, nx = sensor.shape
    r = PROFILE_STAMP_RADIUS
    profiles = []
    for i, fid in enumerate(light_ids):
        if censored_mask[i]:
            continue
        frame = cfa[fid]
        ref = _same_cfa_local_median(frame, site.y, site.x, pattern, LOCAL_REFERENCE_RADIUS)
        if not math.isfinite(ref):
            continue
        stamp = _extract_stamp(frame - ref, site.y, site.x, r, ny, nx)
        profiles.append(stamp)
    if not profiles:
        return None
    return np.nanmean(np.stack(profiles, axis=0), axis=0)


def _center_fraction(profile: Optional[np.ndarray]) -> Optional[float]:
    """Fraction of the profile's total |residual| carried by the centre pixel."""
    if profile is None:
        return None
    total = float(np.nansum(np.abs(profile)))
    if total == 0.0:
        return None
    r = profile.shape[0] // 2
    return abs(float(profile[r, r])) / total


def _site_delta_map(cfa_a, cfa_b, light_ids, site, sensor, reconstructed) -> np.ndarray:
    """Isolate one site's reconstruction effect: mean (B − A) at the site pixel.

    A non-reconstructed site contributes no change anywhere; a reconstructed site
    contributes only at its own pixel (the frozen operator modifies one pixel).
    This per-site isolation is what makes collateral degradation and localization
    mean "the effect of *this* site's reconstruction", not another site's.
    """
    ny, nx = sensor.shape
    delta = np.zeros((ny, nx), dtype=np.float64)
    if not reconstructed:
        return delta
    acc = 0.0
    for fid in light_ids:
        acc += cfa_b[fid][site.y, site.x] - cfa_a[fid][site.y, site.x]
    delta[site.y, site.x] = acc / len(light_ids)
    return delta


def _chebyshev(dy, dx) -> int:
    return abs(dy) if abs(dy) >= abs(dx) else abs(dx)


def _collateral(delta_mean, site, sensor, nearby_star) -> CollateralDegradation:
    pattern = sensor.cfa_pattern
    ny, nx = sensor.shape
    site_plane = cfa_plane_label(site.y, site.x, pattern)
    r_in, r_out = COLLATERAL_ANNULUS

    # Local background: same-CFA pixels in the annulus around the site.
    annulus = []
    for dy in range(-r_out, r_out + 1):
        for dx in range(-r_out, r_out + 1):
            cheb = _chebyshev(dy, dx)
            if cheb < r_in or cheb > r_out:
                continue
            yy, xx = site.y + dy, site.x + dx
            if 0 <= yy < ny and 0 <= xx < nx and cfa_plane_label(yy, xx, pattern) == site_plane:
                annulus.append(float(delta_mean[yy, xx]))
    annulus = np.asarray(annulus, dtype=np.float64)
    if annulus.size == 0:
        local_background_bias = None
        local_noise = None
    else:
        med = float(np.median(annulus))
        local_background_bias = med
        local_noise = float(np.median(np.abs(annulus - med)))

    # Photometric impact: the net ADU change over the whole frame (signed).
    photometric_impact = float(np.sum(delta_mean))

    # Untouched domain: the reconstruction effect outside the site's own pixel.
    outside = []
    for yy in range(ny):
        for xx in range(nx):
            if yy == site.y and xx == site.x:
                continue
            outside.append(float(delta_mean[yy, xx]))
    outside = np.asarray(outside, dtype=np.float64)
    untouched_domain_invariance = float(np.abs(outside).max()) if outside.size else 0.0
    new_artifact_created = bool(np.any(outside != 0.0))
    cfa_structure_preserved = not new_artifact_created

    # Nearby-star integrity (published with the distance, lesson P2I/S2).
    if nearby_star is not None:
        sy, sx = nearby_star
        nearby_star_peak_delta = float(delta_mean[sy, sx])
        nearby_star_core_delta = float(delta_mean[sy, sx])
        nearby_star_distance_px = float(np.hypot(sy - site.y, sx - site.x))
    else:
        nearby_star_peak_delta = None
        nearby_star_core_delta = None
        nearby_star_distance_px = None

    return CollateralDegradation(
        local_background_bias=local_background_bias,
        local_noise=local_noise,
        nearby_star_peak_delta=nearby_star_peak_delta,
        nearby_star_core_delta=nearby_star_core_delta,
        nearby_star_distance_px=nearby_star_distance_px,
        cfa_structure_preserved=cfa_structure_preserved,
        photometric_impact=photometric_impact,
        new_artifact_created=new_artifact_created,
        untouched_domain_invariance=untouched_domain_invariance,
    )


def _localization(delta_mean, site, sensor, reconstructed) -> Localization:
    ny, nx = sensor.shape
    if not reconstructed:
        return Localization(None, LOCALIZATION_OFFSETS, tuple(0.0 for _ in LOCALIZATION_OFFSETS), None)
    site_magnitude = abs(float(delta_mean[site.y, site.x]))
    shifted = []
    for dy, dx in LOCALIZATION_OFFSETS:
        yy, xx = site.y + dy, site.x + dx
        if 0 <= yy < ny and 0 <= xx < nx:
            shifted.append(abs(float(delta_mean[yy, xx])))
        else:
            shifted.append(0.0)
    total = site_magnitude + sum(shifted)
    concentration = site_magnitude / total if total > 0.0 else None
    return Localization(site_magnitude, LOCALIZATION_OFFSETS, tuple(shifted), concentration)


def _default_donor_availability(scenario):
    """Default donor-availability fact: all 36 frozen donors in-bounds (geometry only)."""
    sensor = scenario.sensor
    pattern = sensor.cfa_pattern
    by_id = {s.site_id: s for s in scenario.sites}

    def avail(site_id: str, frame_id: str) -> str:
        site = by_id[site_id]
        return (
            DONORS_AVAILABLE
            if donor_offsets_available(sensor.shape, site.y, site.x, pattern)
            else DONORS_UNAVAILABLE
        )

    return avail


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def run_net_benefit_harness(
    scenario: ScenarioSpec,
    frame_arrays: Mapping[str, np.ndarray],
    *,
    dark_reference: np.ndarray,
    donor_availability: Optional[Callable[[str, str], str]] = None,
    nearby_star: Optional[Tuple[int, int]] = None,
    operator_id: str = RESEARCH_WITNESS_ONLY,
) -> NetBenefitHarnessResult:
    """Run the paired net-benefit measurement over ``scenario``.

    ``frame_arrays`` maps ``frame_id`` → physical-ADU array (from
    :func:`research.p3b.generator.generate_corpus`). ``dark_reference`` is the
    representative additive dark (same shape). ``donor_availability`` overrides
    the default geometric donor fact for the LOT3 preflight (used to exercise
    run-wide atomicity). ``nearby_star`` is an optional ``(y, x)`` of a nearby
    star whose integrity the collateral controls verify.

    Pure and deterministic: a function of the inputs only.
    """
    if operator_id != RESEARCH_WITNESS_ONLY:
        raise ValueError(
            f"unknown reconstruction operator {operator_id!r}; only "
            f"{RESEARCH_WITNESS_ONLY!r} exists in the net-benefit harness"
        )

    sensor = scenario.sensor
    pattern = sensor.cfa_pattern
    light_ids = _light_ids(scenario)
    if not light_ids:
        raise ValueError("net-benefit harness needs at least one light frame")

    dark = np.asarray(dark_reference, dtype=np.float64)

    # Branch A — ordinary calibration: post-calibration CFA (dark subtraction).
    cfa_a = {fid: frame_arrays[fid].astype(np.float64) - dark for fid in light_ids}

    # Frozen LOT3 plan (the only decision to apply; uniform per site over the run).
    declared = build_declared_facts(scenario)
    site_by_id = {s.site_id: s for s in scenario.sites}
    site_quals = []
    for site, sf in zip(scenario.sites, declared.sites):
        packet = evidence_packet(sf, declared)
        # Measurement-run input: a declared reconstruction candidate is marked
        # net-benefit-established *so the harness can measure it*. This is not a
        # verdict — the harness reports the measurement, it does not accept it.
        if site.expected_action_state == ACTION_ELIGIBLE and sf.censored_measurement_present != YES:
            packet = replace(packet, net_benefit_established=NET_BENEFIT_ESTABLISHED)
        decision = evaluate(packet)
        site_quals.append(
            SiteQualification(
                site_id=site.site_id,
                x=site.x,
                y=site.y,
                decision=decision,
                operator_id=RESEARCH_WITNESS_ONLY,
                operator_version=OPERATOR_VERSION,
            )
        )
    if donor_availability is None:
        donor_availability = _default_donor_availability(scenario)
    plan = preflight(
        run_id=f"net-benefit:{scenario.name}",
        profile_revision="p3b-lot6-research-witness",
        calibration_identity="dark-median",
        geometry=GeometryBinding(
            shape=sensor.shape,
            cfa_pattern=pattern,
            binning=sensor.binning,
            roi_origin=sensor.roi_origin,
        ),
        frame_ids=tuple(light_ids),
        site_qualifications=tuple(site_quals),
        donor_availability=donor_availability,
    )

    # Branch B — identical pipeline + reconstruction of ELIGIBLE sites (uniform).
    cfa_b = {fid: cfa_a[fid].copy() for fid in light_ids}
    for entry in plan.sites():
        if entry.run_wide_applicability == RUN_WIDE_ELIGIBLE:
            for fid in light_ids:
                cfa_b[fid][entry.y, entry.x] = reconstruct_site_pixel(
                    cfa_a[fid], entry.y, entry.x, pattern, operator_id
                )

    # Features are consumed through compute_features (the censoring-filtered
    # path): its ``post_calibration_residual`` is ``light − dark`` at the site,
    # NaN exactly where the acquisition was censored.
    feats_a = compute_features(scenario, frame_arrays, dark_reference=dark)
    sf_a_by_id = {sf.site_id: sf for sf in feats_a.sites}

    sites_out = []
    for entry in plan.sites():
        site = site_by_id[entry.site_id]
        sf_a = sf_a_by_id[entry.site_id]
        reconstructed = entry.run_wide_applicability == RUN_WIDE_ELIGIBLE
        censored = bool(site.censored)

        # Censored-frame set, from the filtered path (NaN == censored).
        post_res_a = sf_a.post_calibration_residual or ()
        censored_mask = tuple(not math.isfinite(v) for v in post_res_a)

        # Residual reduction — post-calibration local residual, paired on the
        # same local reference, censored frames excluded (no censored inference).
        residuals = []
        for i, fid in enumerate(light_ids):
            if censored_mask[i]:
                continue
            ref = _same_cfa_local_median(cfa_a[fid], site.y, site.x, pattern, LOCAL_REFERENCE_RADIUS)
            if not math.isfinite(ref):
                continue
            ra = post_res_a[i] - ref
            rb = cfa_b[fid][site.y, site.x] - ref  # same local reference (paired)
            residuals.append((ra, rb))
        n = len(residuals)
        if n == 0:
            baseline = residual_after = reduction = fraction = None
        else:
            baseline = sum(a for a, _b in residuals) / n
            residual_after = sum(b for _a, b in residuals) / n
            reduction = baseline - residual_after
            fraction = reduction / baseline if baseline != 0.0 else None

        prof_a = _mean_residual_profile(cfa_a, light_ids, site, sensor, censored_mask)
        prof_b = _mean_residual_profile(cfa_b, light_ids, site, sensor, censored_mask)

        delta_site = _site_delta_map(cfa_a, cfa_b, light_ids, site, sensor, reconstructed)

        sites_out.append(
            SiteNetBenefit(
                site_id=entry.site_id,
                x=site.x,
                y=site.y,
                cfa_plane=cfa_plane_label(site.y, site.x, pattern),
                run_wide_applicability=entry.run_wide_applicability,
                reconstructed=reconstructed,
                censored=censored,
                pair=NetBenefitPair(
                    residual_reduction_at_site=reduction,
                    residual_reduction_fraction=fraction,
                    collateral_degradation=_collateral(delta_site, site, sensor, nearby_star),
                ),
                baseline_residual_adu=baseline,
                residual_after_adu=residual_after,
                mean_profile_a=_profile_to_tuple(prof_a),
                mean_profile_b=_profile_to_tuple(prof_b),
                profile_radius=PROFILE_STAMP_RADIUS,
                center_fraction_a=_center_fraction(prof_a),
                center_fraction_b=_center_fraction(prof_b),
                n_frames=n,
                localization=_localization(delta_site, site, sensor, reconstructed),
            )
        )

    return NetBenefitHarnessResult(
        scenario_name=scenario.name,
        seed=scenario.seed,
        operator_id=RESEARCH_WITNESS_ONLY,
        operator_version=OPERATOR_VERSION,
        sites=tuple(sites_out),
    )


__all__ = [
    "COLLATERAL_ANNULUS",
    "LOCAL_REFERENCE_RADIUS",
    "LOCALIZATION_OFFSETS",
    "PROFILE_STAMP_RADIUS",
    "Localization",
    "NetBenefitHarnessResult",
    "SiteNetBenefit",
    "representative_dark",
    "run_net_benefit_harness",
]
