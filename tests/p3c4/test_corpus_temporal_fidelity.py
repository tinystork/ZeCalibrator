"""P3C-4 LOT 2 — corpus temporal-fidelity tests (the discrimination became *possible*).

These tests prove, at the **corpus/data** level (no inference rule is touched —
that is LOT 3), that the P3C-4 corpus layer gives celestial confounders a
distinct temporal evolution while keeping their spatial similarity to a fixed
sensor defect high. Concretely:

* §17-A (fixed sensor defect): the residual at the fixed coordinate is
  temporally **stationary / recurrent** across epochs.
* §17-B (undersampled star core): the signature at the fixed coordinate is
  **not reproduced** — it departs as the star follows the sky.
* §17 spatial similarity: A and B have the **same local amplitude and
  morphology** (an explicit measure), so the temporal difference is not an
  artefact of a grossly different confounder.
* §18 (low-occupancy intermittent): a genuine sensor defect ON in a single
  group (a run of frames) is **not** temporally identical to a single transient.
* §19 (single excursion): one enormous excursion, once, is **not** assimilable
  to a recurrence — regardless of amplitude.

The declared truth (oracle) is reconciled with the rendered data: the celestial
confounder now *renders* non-persistent (it moves), matching its declared
``persisted = NO``.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.p3c.oracle import sensor_evidence_truth
from research.p3c4.corpus import (
    LOW_OCCUPANCY_ON_ADU,
    PAIR_AMPLITUDE_ADU,
    SINGLE_EXCURSION_ADU,
    SKY_DITHER_PX,
    TEMPORAL_FIXED_DEFECT,
    TEMPORAL_LOW_OCCUPANCY_INTERMITTENT,
    TEMPORAL_SINGLE_EXCURSION,
    TEMPORAL_STAR_CORE,
    build_scenario,
    fixed_coordinate_series,
    local_window,
    materialise_frames,
    profile_similarity,
    sky_offset,
)


def _truth(class_name: str) -> dict:
    return {t.fact: t.value for t in sensor_evidence_truth(class_name)}


def _series(temporal_kind: str, *, seed: int = 0) -> tuple[np.ndarray, object]:
    scenario = build_scenario(temporal_kind, seed=seed)
    frames = materialise_frames(scenario, temporal_kind, seed=seed)
    light = scenario.light_frames()
    series = fixed_coordinate_series(frames, light, scenario.sites[0].x, scenario.sites[0].y)
    return series, scenario


def _epoch_means(series: np.ndarray, frames_per_epoch: int) -> tuple[float, float]:
    # 24 frames, 12 per epoch (2 groups x 6 frames), in canonical order.
    half = len(series) // 2
    return float(np.mean(series[:half])), float(np.mean(series[half:]))


# ---------------------------------------------------------------------------
# §17-A — the fixed sensor defect is temporally stationary at the coordinate
# ---------------------------------------------------------------------------


def test_fixed_defect_is_temporally_stationary_across_epochs():
    series, _ = _series(TEMPORAL_FIXED_DEFECT)
    assert len(series) == 24
    # A constant ~4000 ADU signal on a ~1400 ADU base: the series is flat.
    assert float(np.std(series)) < 50.0, f"fixed defect is not stationary: std={np.std(series)}"
    e0, e1 = _epoch_means(series, 12)
    assert abs(e0 - e1) < 50.0, f"fixed defect drifted across epochs: {e0} vs {e1}"


def test_fixed_defect_signal_matches_declared_amplitude():
    series, _ = _series(TEMPORAL_FIXED_DEFECT)
    # base (offset 800 + sky 600) + matched amplitude 4000.
    base = 800.0 + 600.0
    assert float(np.mean(series)) == pytest.approx(base + PAIR_AMPLITUDE_ADU, abs=20.0)


def test_fixed_defect_truth_is_persisted_yes():
    assert _truth("STABLE_ANOMALY_WITH_MISMATCHED_DARK")["persisted_at_same_sensor_coord"] == "YES"


# ---------------------------------------------------------------------------
# §17-B — the star core follows the sky and departs the fixed coordinate
# ---------------------------------------------------------------------------


def test_star_core_departs_the_fixed_coordinate_across_epochs():
    series, _ = _series(TEMPORAL_STAR_CORE)
    e0, e1 = _epoch_means(series, 12)
    # In epoch 0 the star is at the site; in epoch 1 it has dithered away.
    assert e0 - e1 > 2000.0, f"star core did not depart: epoch0={e0} epoch1={e1}"


def test_star_core_truth_is_persisted_no():
    assert _truth("UNDERSAMPLED_STAR_CORE")["persisted_at_same_sensor_coord"] == "NO"


def test_sky_offset_dithers_per_epoch_and_is_deterministic():
    assert sky_offset(0, 0) == (0.0, 0.0)
    assert sky_offset(1, 0) == (0.0, float(SKY_DITHER_PX))
    assert sky_offset(2, 0) == (0.0, float(2 * SKY_DITHER_PX))
    # Within an epoch the pointing is fixed (group_index does not move the sky).
    assert sky_offset(1, 1) == sky_offset(1, 0)


# ---------------------------------------------------------------------------
# §17 — spatial similarity: A and B are locally near-identical (the trap avoided)
# ---------------------------------------------------------------------------


def test_pair_17_spatial_similarity_is_high():
    # Same local amplitude + morphology in the epoch where both are at the
    # coordinate (epoch 0). Different seeds => independent read-noise draws, so
    # the comparison is not a byte-identity tautology.
    sc_a = build_scenario(TEMPORAL_FIXED_DEFECT, seed=0)
    sc_b = build_scenario(TEMPORAL_STAR_CORE, seed=1)
    fr_a = materialise_frames(sc_a, TEMPORAL_FIXED_DEFECT, seed=0)
    fr_b = materialise_frames(sc_b, TEMPORAL_STAR_CORE, seed=1)

    x, y = sc_a.sites[0].x, sc_a.sites[0].y
    # First light frame of epoch 0 (both at the coordinate).
    frame_a = fr_a["light0"]
    frame_b = fr_b["light0"]

    def _signal(frame):
        window = local_window(frame, x, y, radius=2)
        return window - float(np.median(window))

    sim = profile_similarity(_signal(frame_a), _signal(frame_b))
    assert sim.peak_a > 2000.0 and sim.peak_b > 2000.0, "no point source present"
    assert sim.amplitude_ratio > 0.95, f"amplitudes differ: {sim.amplitude_ratio}"
    assert sim.shape_correlation > 0.98, f"morphology differs: {sim.shape_correlation}"


def test_pair_17_discrimination_is_temporal_not_spatial():
    # The couple's spatial similarity is high (previous test) AND their temporal
    # behaviour differs — i.e. the discrimination is not a gross spatial
    # difference, it is purely temporal.
    series_a, _ = _series(TEMPORAL_FIXED_DEFECT, seed=0)
    series_b, _ = _series(TEMPORAL_STAR_CORE, seed=0)
    # A stationary, B departs.
    assert float(np.std(series_a)) < 50.0
    e0_b, e1_b = _epoch_means(series_b, 12)
    assert e0_b - e1_b > 2000.0


# ---------------------------------------------------------------------------
# §18 — low-occupancy intermittent vs §19 — single huge excursion
# ---------------------------------------------------------------------------


def _occupancy(series: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(series > threshold))


def test_low_occupancy_intermittent_is_a_run_not_a_single_frame():
    # A genuine intermittent sensor defect, ON for one whole group (6 frames).
    series, scenario = _series(TEMPORAL_LOW_OCCUPANCY_INTERMITTENT)
    # base ~1400, ON ~3900: a signal threshold of 2000 separates ON from OFF.
    on_count = _occupancy(series, 2000.0)
    assert on_count == 6, f"expected 6 ON frames (one group), got {on_count}"
    # All ON frames belong to the same group (epoch 0, group 0) -> a run.
    light = scenario.light_frames()
    on_frames = [f for f, v in zip(light, series) if v > 2000.0]
    keys = {(f.epoch_id, f.group_id) for f in on_frames}
    assert keys == {("e0", "g0_0")}, f"ON frames spread over unexpected groups: {keys}"


def test_low_occupancy_intermittent_truth_is_persisted_yes():
    assert _truth("RARE_HIGH_STATE")["persisted_at_same_sensor_coord"] == "YES"


def test_single_excursion_occurs_exactly_once():
    series, _ = _series(TEMPORAL_SINGLE_EXCURSION)
    on_count = _occupancy(series, 2000.0)
    assert on_count == 1, f"expected exactly 1 excursion frame, got {on_count}"
    # The excursion is enormous (independent of amplitude) — well above any
    # reasonable anomaly threshold.
    assert float(np.max(series)) > SINGLE_EXCURSION_ADU * 0.9


def test_single_excursion_truth_is_transient_not_persisted():
    truth = _truth("SINGLE_TRANSIENT")
    assert truth["persisted_at_same_sensor_coord"] == "NO"
    assert truth["transient_only"] == "YES"


def test_low_occupancy_intermittent_is_not_temporally_identical_to_single_excursion():
    # §18 (a 6-frame run of a recurring defect) and §19 (one huge excursion) are
    # temporally distinct: different occupancy, and only §18 recurs.
    series_18, _ = _series(TEMPORAL_LOW_OCCUPANCY_INTERMITTENT)
    series_19, _ = _series(TEMPORAL_SINGLE_EXCURSION)
    assert _occupancy(series_18, 2000.0) == 6
    assert _occupancy(series_19, 2000.0) == 1
    # The §19 series is not a recurrence: its signal appears in exactly one frame.
    assert not np.array_equal(series_18, series_19)


def test_single_excursion_is_not_a_recurrence_regardless_of_amplitude():
    # §19: no matter how large the amplitude, a single occurrence never recurs.
    series, _ = _series(TEMPORAL_SINGLE_EXCURSION)
    # One and only one frame carries the signal (occupancy == 1, not a run).
    assert _occupancy(series, 2000.0) == 1
    assert float(np.max(series)) >= 50000.0  # "énorme" — near saturation


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_materialise_is_deterministic():
    sc = build_scenario(TEMPORAL_STAR_CORE, seed=7)
    a = materialise_frames(sc, TEMPORAL_STAR_CORE, seed=7)
    b = materialise_frames(sc, TEMPORAL_STAR_CORE, seed=7)
    assert set(a) == set(b)
    for fid in a:
        assert np.array_equal(a[fid], b[fid]), f"non-deterministic frame {fid}"
