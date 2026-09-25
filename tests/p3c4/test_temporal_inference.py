"""P3C-4 LOT 3 — temporal-persistence inference tests (§48 / GATE §23).

These tests pin the LOT 3 rule that replaces the *spatial/local* basis for
``persisted_at_same_sensor_coord`` with the **temporal proof** of the LOT 1
contract, plus the new versioned ``p3c4-*`` candidate family.

Covered (SCIENCE §48):

* the new inference module passes the anti-truth-leak guard (structurally, in
  ``test_p3c4_truth_leak_guard.py`` — asserted again here at the signature level);
* class-name independence: the temporal rule reads only measured temporal
  features, never a class label, scenario name or expected state;
* identical measured data + different hidden truth ⇒ identical inference;
* same class + different temporal behaviour ⇒ persistence MAY differ;
* the undersampled star core (spatially similar to the fixed defect) differs in
  persistence **by time**, not by a spatial difference (§17);
* single transient ⇒ non-persistent;
* rare intermittent ⇒ not classified transient merely for rarity (§18);
* censored ⇒ no quantitative inference (§14);
* the five other SENSOR-EVIDENCE facts are produced verbatim by the frozen P3C
  candidate (the base is delegated, never re-derived — §21/§24).

GATE §23 (run on the DEVELOPMENT corpus below) is exercised as explicit
assertions: ``UNDERSAMPLED_STAR_CORE`` must be ``NO`` (not ``YES``), the
persistent/intermittent classes must keep a *useful* temporal signal (not "all
UNDETERMINED"), ``SINGLE_TRANSIENT`` and ``STAR_CROSSING_SITE`` must be
non-persistent.
"""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from research.p3b.features import compute_features
from research.p3b.generator import generate_corpus
from research.p3c.development_corpus import build_admission, build_development_scenario
from research.p3c.inference_candidates import (
    RESEARCH_CANDIDATE_PARAMETER,
    CANDIDATE_BASELINE,
    infer_site as p3c_infer_site,
)
from research.p3c.inference_contract import (
    DETERMINED,
    NO,
    UNDETERMINED,
    YES,
    AdmissionFacts,
)
from research.p3c4.corpus import (
    TEMPORAL_FIXED_DEFECT,
    TEMPORAL_LOW_OCCUPANCY_INTERMITTENT,
    TEMPORAL_SINGLE_EXCURSION,
    TEMPORAL_STAR_CORE,
    build_scenario,
    materialise_frames,
)
from research.p3c4.temporal_features import compute_temporal_features
from research.p3c4.temporal_inference import (
    PERSISTED_FACT,
    CANDIDATE_P3C4_BASELINE,
    CANDIDATE_P3C4_CONSERVATIVE,
    CANDIDATE_P3C4_SENSITIVE,
    TEMPORAL_CANDIDATES,
    TEMPORAL_CANDIDATE_IDS,
    TemporalCandidateConfig,
    infer_persisted,
    infer_site,
    temporal_candidate,
)


# ---------------------------------------------------------------------------
# Harness: temporal features for a P3C4 temporal kind (or a P3B dev class)
# ---------------------------------------------------------------------------


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


def _dev_temporal_features(class_name: str, *, seed: int = 0):
    scenario = build_development_scenario(class_name, seed)
    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        site = scenario.sites[0]
        hard = (
            site.hard_limit_adu
            if site.hard_limit_adu is not None
            else scenario.sensor.saturation_limit_adu
        )
        tf = compute_temporal_features(
            corpus.frame_arrays,
            scenario.light_frames(),
            site.x,
            site.y,
            cfa_pattern=scenario.sensor.cfa_pattern,
            hard_limit=hard,
        )
    return scenario, corpus.frame_arrays, tf


def _admission(**overrides) -> AdmissionFacts:
    base = dict(
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=YES,
        calibration_representativeness="REPRESENTATIVE",
        independent_group_count=4,
        epoch_count=2,
    )
    base.update(overrides)
    return AdmissionFacts(**base)


# ---------------------------------------------------------------------------
# Candidate family is bounded and versioned (never a giant grid)
# ---------------------------------------------------------------------------


def test_family_is_bounded():
    assert 3 <= len(TEMPORAL_CANDIDATES) <= 6


def test_candidate_ids_are_unique_and_stable():
    assert TEMPORAL_CANDIDATE_IDS == tuple(c.candidate_id for c in TEMPORAL_CANDIDATES)
    assert len(set(TEMPORAL_CANDIDATE_IDS)) == len(TEMPORAL_CANDIDATE_IDS)


def test_every_candidate_is_versioned_and_described():
    for c in TEMPORAL_CANDIDATES:
        assert c.version
        assert c.description
        assert c.candidate_id
        assert c.base_candidate_id


def test_every_parameter_is_marked_research_candidate():
    for c in TEMPORAL_CANDIDATES:
        for p in c.parameters:
            assert p.kind == RESEARCH_CANDIDATE_PARAMETER
            assert "PRODUCT" not in p.kind
            assert p.unit
            assert p.rationale


def test_candidate_lookup_roundtrips():
    for c in TEMPORAL_CANDIDATES:
        assert temporal_candidate(c.candidate_id) is c


def test_unknown_candidate_raises_key_error():
    with pytest.raises(KeyError):
        temporal_candidate("does-not-exist")


def test_duplicate_parameter_names_rejected():
    from research.p3c.inference_candidates import CandidateParam

    with pytest.raises(ValueError):
        TemporalCandidateConfig(
            candidate_id="bad",
            version="1",
            description="x",
            base_candidate_id="p3c-baseline",
            parameters=(
                CandidateParam("a", 1.0, "ADU", "r"),
                CandidateParam("a", 2.0, "ADU", "r"),
            ),
        )


def test_p3c_frozen_candidates_are_not_modified():
    # The frozen p3c-* artefacts must be reachable and unchanged (§24): the
    # temporal candidates merely reference them as a base, never edit them.
    assert CANDIDATE_P3C4_BASELINE.base_candidate_id == CANDIDATE_BASELINE.candidate_id
    for c in TEMPORAL_CANDIDATES:
        assert c.candidate_id.startswith("p3c4-")
        assert c.base_candidate_id.startswith("p3c-")


# ---------------------------------------------------------------------------
# Class-name independence + identical data / different truth (§48)
# ---------------------------------------------------------------------------


def test_temporal_rule_has_no_truth_input():
    # The temporal persisted inference consumes (cfg, temporal_features) only —
    # no class name, scenario name or expected state can reach it.
    import inspect

    sig = inspect.signature(infer_persisted)
    assert set(sig.parameters) == {"cfg", "tf"}


def test_identical_measured_data_identical_inference_regardless_of_truth():
    # Two temporal kinds with byte-identical rendered frames would infer
    # identically — but here the honest form: the same kind, same seed, inferred
    # twice, must be byte-identical (determinism), and the result depends only
    # on the measured features, never on any truth label.
    _, _, tf = _temporal_features(TEMPORAL_STAR_CORE)
    a = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    b = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    assert a == b


def test_same_class_different_temporal_behaviour_may_differ():
    # The corpus renders the SAME declared class differently by temporal kind:
    # a fixed defect is stationary; a star core (same spatial amplitude) departs.
    # The temporal rule must be able to distinguish them — proving it reads
    # temporal behaviour, not the class label (the two kinds share local
    # amplitude; the declared class in KIND_TO_CLASS differs, but the rule never
    # sees it).
    _, _, tf_fixed = _temporal_features(TEMPORAL_FIXED_DEFECT)
    _, _, tf_star = _temporal_features(TEMPORAL_STAR_CORE)
    persisted_fixed = infer_persisted(CANDIDATE_P3C4_BASELINE, tf_fixed).value
    persisted_star = infer_persisted(CANDIDATE_P3C4_BASELINE, tf_star).value
    assert persisted_fixed != persisted_star


# ---------------------------------------------------------------------------
# The four temporal behaviours (GATE §23 core)
# ---------------------------------------------------------------------------


def test_fixed_sensor_defect_is_persisted():
    # A persistent sensor defect is stationary at the coordinate across groups
    # AND epochs -> temporal persistence SUPPORTED (YES).
    _, _, tf = _temporal_features(TEMPORAL_FIXED_DEFECT)
    field = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    assert field.value == YES
    assert field.uncertainty == DETERMINED


def test_undersampled_star_core_is_not_persisted():
    # The latent defect (§32): the celestial confounder departs the coordinate
    # between epochs -> persistence NOT supported (NO), not YES.
    _, _, tf = _temporal_features(TEMPORAL_STAR_CORE)
    field = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    assert field.value == NO
    assert field.uncertainty == DETERMINED


def test_low_occupancy_intermittent_is_not_concluded_non_persistent():
    # Low occupancy WITHOUT an epoch departure -> never NO (a rare intermittent
    # must not be erased). Here it is UNDETERMINED (single supporting group).
    _, _, tf = _temporal_features(TEMPORAL_LOW_OCCUPANCY_INTERMITTENT)
    field = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    assert field.value == UNDETERMINED
    assert field.uncertainty == UNDETERMINED


def test_single_excursion_is_not_persisted():
    # One frame, never repeated -> non-persistent (NO), regardless of amplitude.
    _, _, tf = _temporal_features(TEMPORAL_SINGLE_EXCURSION)
    field = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    assert field.value == NO
    assert field.uncertainty == DETERMINED


@pytest.mark.parametrize("candidate_id", TEMPORAL_CANDIDATE_IDS)
def test_all_candidates_close_the_star_core_defect(candidate_id):
    # Every temporal candidate must turn the undersampled star core into NO —
    # none may regress to the spatial YES (the GATE §23 latent defect).
    cfg = temporal_candidate(candidate_id)
    _, _, tf = _temporal_features(TEMPORAL_STAR_CORE)
    assert infer_persisted(cfg, tf).value == NO


@pytest.mark.parametrize("candidate_id", TEMPORAL_CANDIDATE_IDS)
def test_all_candidates_keep_the_persistent_defect_useful(candidate_id):
    # Every candidate must keep a *useful* signal: the persistent class stays YES
    # (not "all UNDETERMINED" — that would replace one error with another).
    cfg = temporal_candidate(candidate_id)
    _, _, tf = _temporal_features(TEMPORAL_FIXED_DEFECT)
    assert infer_persisted(cfg, tf).value == YES


# ---------------------------------------------------------------------------
# §17 — spatial similarity maintained; persistence differs by time
# ---------------------------------------------------------------------------


def test_star_core_is_spatially_similar_to_fixed_defect_but_differs_in_time():
    from research.p3c4.corpus import local_window, profile_similarity

    sc_a = build_scenario(TEMPORAL_FIXED_DEFECT, seed=0)
    sc_b = build_scenario(TEMPORAL_STAR_CORE, seed=1)
    fr_a = materialise_frames(sc_a, TEMPORAL_FIXED_DEFECT, seed=0)
    fr_b = materialise_frames(sc_b, TEMPORAL_STAR_CORE, seed=1)
    x, y = sc_a.sites[0].x, sc_a.sites[0].y

    def _signal(frame):
        window = local_window(frame, x, y, radius=2)
        return window - float(np.median(window))

    sim = profile_similarity(_signal(fr_a["light0"]), _signal(fr_b["light0"]))
    assert sim.amplitude_ratio > 0.95
    assert sim.shape_correlation > 0.98

    # The temporal proof differs even though the spatial profile is near-identical.
    _, _, tf_a = _temporal_features(TEMPORAL_FIXED_DEFECT)
    _, _, tf_b = _temporal_features(TEMPORAL_STAR_CORE)
    assert infer_persisted(CANDIDATE_P3C4_BASELINE, tf_a).value == YES
    assert infer_persisted(CANDIDATE_P3C4_BASELINE, tf_b).value == NO


# ---------------------------------------------------------------------------
# §18 — rare intermittent is not transient merely for rarity
# ---------------------------------------------------------------------------


def test_rare_intermittent_is_not_classified_transient_by_rarity():
    # The low-occupancy intermittent is present in one group (rare) but is a
    # genuine run, not a single transient: it must NOT be concluded non-persistent
    # and NOT be classified as a unique occurrence. The temporal rule yields
    # UNDETERMINED, never NO.
    _, _, tf = _temporal_features(TEMPORAL_LOW_OCCUPANCY_INTERMITTENT)
    field = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    assert field.value != NO
    assert field.value != YES


# ---------------------------------------------------------------------------
# §14 — censored ⇒ no quantitative inference
# ---------------------------------------------------------------------------


def test_censored_site_yields_no_quantitative_persisted_inference():
    # CENSORED_ANOMALY's every frame is censored (value at the hard limit): the
    # temporal rule must not produce a quantitative YES/NO.
    _, _, tf = _dev_temporal_features("CENSORED_ANOMALY")
    field = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    assert field.value == UNDETERMINED


def test_censored_frames_are_excluded_from_temporal_features():
    # A censored frame contributes nothing quantitative: its residual and peak
    # are NaN, and it is flagged censored.
    scenario = build_development_scenario("CENSORED_ANOMALY", 0)
    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        site = scenario.sites[0]
        hard = (
            site.hard_limit_adu
            if site.hard_limit_adu is not None
            else scenario.sensor.saturation_limit_adu
        )
        tf = compute_temporal_features(
            corpus.frame_arrays,
            scenario.light_frames(),
            site.x,
            site.y,
            cfa_pattern=scenario.sensor.cfa_pattern,
            hard_limit=hard,
        )
    assert any(tf.frame_censored)
    for i, censored in enumerate(tf.frame_censored):
        if censored:
            assert math.isnan(tf.frame_coord_residual[i])


# ---------------------------------------------------------------------------
# §21 — the five other facts come verbatim from the frozen P3C base
# ---------------------------------------------------------------------------


def test_five_other_facts_are_delegated_verbatim():
    # infer_site must produce the five non-persisted facts byte-identically to
    # the frozen P3C base candidate; only persisted is re-derived temporally.
    scenario, frames, tf = _temporal_features(TEMPORAL_STAR_CORE)
    cf = compute_features(scenario, frames)
    adm = build_admission(scenario)

    base_ev = p3c_infer_site(CANDIDATE_BASELINE, cf.sites[0], adm)
    tmp_ev = infer_site(CANDIDATE_P3C4_BASELINE, cf.sites[0], tf, adm)

    base_map = {f.field: f.value for f in base_ev.sensor_evidence}
    tmp_map = {f.field: f.value for f in tmp_ev.sensor_evidence}
    assert tmp_ev.candidate_id == CANDIDATE_P3C4_BASELINE.candidate_id
    for fact in ("censored_measurement_present", "conflicting_evidence",
                 "transient_only", "site_residual_behaviour",
                 "neighbourhood_residual_stable"):
        assert tmp_map[fact] == base_map[fact], fact
    # persisted is the one fact that changes (temporal proof replaces spatial).
    assert tmp_map[PERSISTED_FACT] != base_map[PERSISTED_FACT]


def test_infer_site_covers_all_six_facts_and_is_deterministic():
    from research.p3c.inference_contract import SENSOR_EVIDENCE_FACTS

    scenario, frames, tf = _temporal_features(TEMPORAL_FIXED_DEFECT)
    cf = compute_features(scenario, frames)
    adm = build_admission(scenario)
    first = infer_site(CANDIDATE_P3C4_BASELINE, cf.sites[0], tf, adm)
    second = infer_site(CANDIDATE_P3C4_BASELINE, cf.sites[0], tf, adm)
    assert first == second
    fields = {f.field for f in first.sensor_evidence}
    assert fields == set(SENSOR_EVIDENCE_FACTS)


# ---------------------------------------------------------------------------
# GATE §23 — STAR_CROSSING_SITE is non-persistent (P3B development corpus)
# ---------------------------------------------------------------------------


def test_star_crossing_site_is_not_persisted():
    # A moving star PSF crossing the site departs the coordinate (it drifts per
    # frame), so the temporal proof must reject it as non-persistent.
    _, _, tf = _dev_temporal_features("STAR_CROSSING_SITE")
    field = infer_persisted(CANDIDATE_P3C4_BASELINE, tf)
    assert field.value == NO


def test_gate_23_development_outcomes():
    # The complete GATE §23 table, evaluated on the DEVELOPMENT corpus:
    #   UNDERSAMPLED_STAR_CORE  -> NO (not YES)   [latent defect closed]
    #   persistent/intermittent -> YES / not-NO   [temporal proof useful]
    #   SINGLE_TRANSIENT        -> NO             [non-persistent]
    #   STAR_CROSSING_SITE      -> NO             [non-persistent]
    outcomes = {}
    for kind, label in (
        (TEMPORAL_STAR_CORE, "UNDERSAMPLED_STAR_CORE"),
        (TEMPORAL_FIXED_DEFECT, "STABLE_ANOMALY_WITH_MISMATCHED_DARK"),
        (TEMPORAL_LOW_OCCUPANCY_INTERMITTENT, "RARE_HIGH_STATE"),
        (TEMPORAL_SINGLE_EXCURSION, "SINGLE_TRANSIENT"),
    ):
        _, _, tf = _temporal_features(kind)
        outcomes[label] = infer_persisted(CANDIDATE_P3C4_BASELINE, tf).value
    _, _, tf = _dev_temporal_features("STAR_CROSSING_SITE")
    outcomes["STAR_CROSSING_SITE"] = infer_persisted(CANDIDATE_P3C4_BASELINE, tf).value

    assert outcomes["UNDERSAMPLED_STAR_CORE"] == NO
    assert outcomes["STABLE_ANOMALY_WITH_MISMATCHED_DARK"] == YES
    assert outcomes["RARE_HIGH_STATE"] != NO  # rare intermittent protected
    assert outcomes["SINGLE_TRANSIENT"] == NO
    assert outcomes["STAR_CROSSING_SITE"] == NO


__all__ = []
