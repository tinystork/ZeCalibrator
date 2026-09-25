"""P3C-4 LOT 4 — corpus coverage tests: all 20 classes, temporal fidelity (§35/§36).

These tests prove, at the **corpus/data** level (no inference rule is touched),
that the P3C-4 corpus layer now covers the **full 20-class catalogue** with the
correct temporal behaviour, so the campaign's §35 (five celestial confounders,
separately materialisable) and §36 (five rare/intermittent classes, separately
materialisable) are applicable:

* every one of the 20 catalogue classes is materialisable (build + materialise);
* the five §35 confounders are temporally NON-stationary — their centroid
  departs/drifts across epochs (measurable at the fixed coordinate);
* the five §36 intermittents remain temporally coherent with a **sensor defect**
  — fixed coordinate, variable occupancy (no departure);
* STAR_CROSSING_SITE departs AND drifts;
* CENSORED_ANOMALY contributes no quantitative value (every frame censored);
* determinism: same seeds ⇒ same bytes, for every kind.

The trap (§16/§17) is also checked: the confounders keep a **plausible, stable
amplitude** across epochs — the temporal difference is the *displacement*, never
an amplitude collapse.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from research.p3b.catalog import CLASS_NAMES
from research.p3c4.corpus import (
    CLASS_TO_KIND,
    CONFOUNDER_KINDS,
    INTERMITTENT_KINDS,
    KIND_TO_CLASS,
    TEMPORAL_CENSORED_ANOMALY,
    TEMPORAL_STAR_CROSSING_SITE,
    TEMPORAL_KINDS,
    build_scenario,
    fixed_coordinate_series,
    local_window,
    materialise_frames,
    sky_offset,
)
from research.p3c4.temporal_features import compute_temporal_features


def _series(kind: str, *, seed: int = 0):
    scenario = build_scenario(kind, seed=seed)
    frames = materialise_frames(scenario, kind, seed=seed)
    light = scenario.light_frames()
    series = fixed_coordinate_series(frames, light, scenario.sites[0].x, scenario.sites[0].y)
    return series, scenario, frames


def _epoch_means(series: np.ndarray) -> tuple[float, float]:
    half = len(series) // 2
    return float(np.mean(series[:half])), float(np.mean(series[half:]))


def _temporal_features(kind: str, *, seed: int = 0):
    scenario = build_scenario(kind, seed=seed)
    frames = materialise_frames(scenario, kind, seed=seed)
    site = scenario.sites[0]
    tf = compute_temporal_features(
        frames,
        scenario.light_frames(),
        site.x,
        site.y,
        cfa_pattern=scenario.sensor.cfa_pattern,
        hard_limit=scenario.sensor.saturation_limit_adu,
    )
    return scenario, frames, tf


# ---------------------------------------------------------------------------
# 1) The full 20-class catalogue is materialisable
# ---------------------------------------------------------------------------


def test_kind_to_class_covers_all_20_catalogue_classes():
    assert set(KIND_TO_CLASS.values()) == set(CLASS_NAMES)
    assert len(KIND_TO_CLASS) == len(CLASS_NAMES) == 20
    assert set(CLASS_TO_KIND) == set(CLASS_NAMES)


def test_every_class_is_materialisable():
    for kind in TEMPORAL_KINDS:
        scenario = build_scenario(kind, seed=0)
        frames = materialise_frames(scenario, kind, seed=0)
        assert len(frames) == 24, kind
        assert scenario.sites[0].cfa_class == KIND_TO_CLASS[kind]
        for fid, arr in frames.items():
            assert arr.shape == (96, 128), kind
            assert arr.dtype == np.uint16, kind


def test_confounder_and_intermittent_kind_sets_partition_correctly():
    assert len(CONFOUNDER_KINDS) == 5
    assert len(INTERMITTENT_KINDS) == 5
    assert set(CONFOUNDER_KINDS).isdisjoint(set(INTERMITTENT_KINDS))
    # The five confounders are exactly the §35 celestial classes.
    assert {KIND_TO_CLASS[k] for k in CONFOUNDER_KINDS} == {
        "UNDERSAMPLED_STAR_CORE",
        "STAR_CROSSING_SITE",
        "OPTICAL_STRUCTURE",
        "FLAT_STRUCTURE",
        "DUST_OR_VIGNETTING",
    }
    # The five intermittents are exactly the §36 classes.
    assert {KIND_TO_CLASS[k] for k in INTERMITTENT_KINDS} == {
        "RARE_HIGH_STATE",
        "RARE_LOW_STATE",
        "INTERMITTENT_TWO_STATE",
        "INTERMITTENT_MULTI_STATE",
        "INTERMITTENT_CONTINUOUS",
    }


# ---------------------------------------------------------------------------
# 2) The five §35 confounders are temporally NON-stationary (departure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", CONFOUNDER_KINDS)
def test_confounder_departs_the_fixed_coordinate_across_epochs(kind):
    series, _, _ = _series(kind)
    e0, e1 = _epoch_means(series)
    # A sky structure is dithered SKY_DITHER_PX px per epoch, so its value at the
    # fixed coordinate measurably departs between epochs (the wide structures
    # depart by a smaller but still measurable margin than the point star core).
    departure = abs(e0 - e1)
    assert departure > 200.0, (
        f"{kind}: no measurable departure ({e0:.1f} vs {e1:.1f})"
    )


@pytest.mark.parametrize("kind", CONFOUNDER_KINDS)
def test_confounder_keeps_stable_amplitude_no_collapse(kind):
    # Trap (§16/§17): the confounder must NOT become trivially easy by collapsing
    # its amplitude. Its contrast (peak |deviation| from the local background)
    # must stay plausible in BOTH epochs — only the *location* changes. A dust
    # mote is a *darkening* dip, so measure contrast, not absolute peak.
    scenario = build_scenario(kind, seed=0)
    frames = materialise_frames(scenario, kind, seed=0)
    light = scenario.light_frames()

    def contrast(frame_id):
        arr = frames[frame_id].astype(np.float64)
        return float(np.max(np.abs(arr - np.median(arr))))

    c0 = max(contrast(f.frame_id) for f in light[:12])
    c1 = max(contrast(f.frame_id) for f in light[12:])
    # Both epochs must contain a genuine structure (well above read noise ~5 ADU).
    assert c0 > 400.0, f"{kind}: epoch-0 amplitude collapsed ({c0})"
    assert c1 > 400.0, f"{kind}: epoch-1 amplitude collapsed ({c1})"
    # Amplitude must not halve between epochs (displacement, not collapse).
    assert min(c0, c1) / max(c0, c1) > 0.5, (
        f"{kind}: amplitude collapsed across epochs ({c0} vs {c1})"
    )


def test_star_crossing_site_departs_and_drifts():
    # STAR_CROSSING_SITE must show both departure (across epochs) and drift
    # (within an epoch, per frame).
    series, scenario, frames = _series(TEMPORAL_STAR_CROSSING_SITE)
    e0, e1 = _epoch_means(series)
    assert abs(e0 - e1) > 200.0, f"star crossing did not depart: {e0} vs {e1}"
    # Within the first group (6 frames), the moving PSF drifts, so the fixed
    # coordinate sees a changing value (not a flat run).
    light = scenario.light_frames()
    first_group = light[:6]
    vals = [float(frames[f.frame_id][scenario.sites[0].y, scenario.sites[0].x]) for f in first_group]
    assert float(np.std(vals)) > 100.0, f"star crossing did not drift within a group: {vals}"


def test_star_crossing_site_uses_sky_offset_per_epoch():
    # The sky_offset mechanism applies to STAR_CROSSING_SITE (per-epoch dither).
    assert sky_offset(0, 0) == (0.0, 0.0)
    assert sky_offset(1, 0)[1] == 6.0


# ---------------------------------------------------------------------------
# 3) The five §36 intermittents stay coherent with a sensor defect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", INTERMITTENT_KINDS)
def test_intermittent_is_fixed_coordinate_variable_occupancy(kind):
    series, _, _ = _series(kind)
    # Variable occupancy: the series spans more than a single level (it toggles /
    # ramps / is rarely-ON/OFF), so it is NOT a flat run.
    assert float(np.std(series)) > 100.0, f"{kind}: not variable occupancy"
    # Fixed coordinate: unlike a sky confounder, an intermittent does NOT depart
    # — its patch centre never moves with the sky (deterministic, checked at the
    # contribution level, not via a fragile peak-finder).
    from research.p3c4.corpus import site_contribution

    centres = set()
    for epoch in (0, 1):
        for group in (0, 1):
            cy, cx, _ = site_contribution(kind, 8, 8, epoch, group, 0)
            centres.add((cy, cx))
    assert centres == {(8.0, 8.0)}, f"{kind}: centre moved with the sky: {centres}"


@pytest.mark.parametrize("kind", INTERMITTENT_KINDS)
def test_intermittent_declared_truth_is_persisted(kind):
    from research.p3c.oracle import sensor_evidence_truth

    class_name = KIND_TO_CLASS[kind]
    truth = {t.fact: t.value for t in sensor_evidence_truth(class_name)}
    assert truth["persisted_at_same_sensor_coord"] == "YES", class_name


# ---------------------------------------------------------------------------
# 4) CENSORED_ANOMALY — no quantitative contribution
# ---------------------------------------------------------------------------


def test_censored_anomaly_every_frame_is_censored():
    _, scenario, frames = _series(TEMPORAL_CENSORED_ANOMALY)
    site = scenario.sites[0]
    assert scenario.sites[0].censored is True
    light = scenario.light_frames()
    for f in light:
        assert frames[f.frame_id][site.y, site.x] == 60000.0


def test_censored_anomaly_contributes_no_quantitative_value():
    _, _, tf = _temporal_features(TEMPORAL_CENSORED_ANOMALY)
    assert all(tf.frame_censored)
    for i, censored in enumerate(tf.frame_censored):
        if censored:
            assert math.isnan(tf.frame_coord_residual[i])


# ---------------------------------------------------------------------------
# 5) Determinism — same seeds ⇒ same bytes, for every kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", TEMPORAL_KINDS)
def test_materialise_is_deterministic_for_every_kind(kind):
    sc = build_scenario(kind, seed=7)
    a = materialise_frames(sc, kind, seed=7)
    b = materialise_frames(sc, kind, seed=7)
    assert set(a) == set(b)
    for fid in a:
        assert np.array_equal(a[fid], b[fid]), f"non-deterministic frame {fid} for {kind}"


__all__ = []
