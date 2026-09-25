"""P3C-2 tests — the bounded inference-candidate family (transparency, determinism, contract conformance)."""

from __future__ import annotations

import pytest

from research.p3c.inference_candidates import (
    RESEARCH_CANDIDATE_PARAMETER,
    SENSOR_EVIDENCE_ORDER,
    CANDIDATE_BASELINE,
    CANDIDATE_CONSERVATIVE,
    CANDIDATE_SENSITIVE,
    CANDIDATES,
    CANDIDATE_IDS,
    CandidateParam,
    InferenceCandidateConfig,
    candidate,
    infer_site,
)
from research.p3c.inference_contract import (
    DETERMINED,
    NO,
    RESIDUAL_INDETERMINATE,
    RESIDUAL_NONE,
    RESIDUAL_SYSTEMATIC_STABLE,
    RESIDUAL_VARIABLE,
    UNDETERMINED,
    YES,
    SENSOR_EVIDENCE_FACTS,
    AdmissionFacts,
    InferredEvidence,
)


# ---------------------------------------------------------------------------
# The family is bounded and versioned (never a giant grid)
# ---------------------------------------------------------------------------


def test_family_is_bounded():
    assert 3 <= len(CANDIDATES) <= 6


def test_candidate_ids_are_unique_and_stable():
    assert CANDIDATE_IDS == tuple(c.candidate_id for c in CANDIDATES)
    assert len(set(CANDIDATE_IDS)) == len(CANDIDATE_IDS)


def test_every_candidate_is_versioned_and_described():
    for c in CANDIDATES:
        assert c.version
        assert c.description
        assert c.candidate_id


def test_every_parameter_is_marked_research_candidate():
    # Every numeric constant is a RESEARCH CANDIDATE PARAMETER — never a product
    # threshold, never an untagged magic number.
    for c in CANDIDATES:
        for p in c.parameters:
            assert isinstance(p, CandidateParam)
            assert p.kind == RESEARCH_CANDIDATE_PARAMETER
            assert "PRODUCT" not in p.kind
            assert p.unit
            assert p.rationale


def test_candidate_lookup_roundtrips():
    for c in CANDIDATES:
        assert candidate(c.candidate_id) is c


def test_unknown_candidate_raises_key_error():
    with pytest.raises(KeyError):
        candidate("does-not-exist")


def test_duplicate_parameter_names_rejected():
    with pytest.raises(ValueError):
        InferenceCandidateConfig(
            candidate_id="bad",
            version="1",
            description="x",
            parameters=(
                CandidateParam("a", 1.0, "ADU", "r"),
                CandidateParam("a", 2.0, "ADU", "r"),
            ),
        )


# ---------------------------------------------------------------------------
# Determinism and purity
# ---------------------------------------------------------------------------


def test_infer_site_is_deterministic_given_features_and_admission(features_factory):
    features = features_factory("INTERMITTENT_TWO_STATE")
    adm = _admission()
    first = infer_site(CANDIDATE_BASELINE, features, adm)
    second = infer_site(CANDIDATE_BASELINE, features, adm)
    assert first == second


def test_different_candidates_can_disagree(features_factory):
    # The candidates are distinct hypotheses; on a borderline weak anomaly they
    # need not all agree (a research family, not a frozen detector).
    features = features_factory("AMBIGUOUS_INSUFFICIENT_EVIDENCE")
    adm = _admission()
    results = {c.candidate_id: infer_site(c, features, adm) for c in CANDIDATES}
    # At least the candidate IDs are stamped; results are valid InferredEvidence.
    for ev in results.values():
        assert isinstance(ev, InferredEvidence)


# ---------------------------------------------------------------------------
# Output shape: exactly the SENSOR-EVIDENCE facts, no score, no net benefit
# ---------------------------------------------------------------------------


def _field_map(ev: InferredEvidence):
    return {f.field: f for f in ev.sensor_evidence}


def test_output_covers_exactly_the_six_sensor_evidence_facts(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("NORMAL"), _admission())
    fields = _field_map(ev)
    assert set(fields) == set(SENSOR_EVIDENCE_FACTS)
    assert tuple(fields) == SENSOR_EVIDENCE_ORDER


def test_no_score_or_confidence_anywhere(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("INTERMITTENT_TWO_STATE"), _admission())
    for f in ev.sensor_evidence:
        for attr in ("score", "confidence", "probability"):
            assert not hasattr(f, attr)
        assert isinstance(f.value, str) and not isinstance(f.value, bool)
        assert f.uncertainty in (DETERMINED, UNDETERMINED)


def test_net_benefit_is_never_predicted(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("INTERMITTENT_TWO_STATE"), _admission())
    assert ev.net_benefit_established == UNDETERMINED


def test_candidate_id_is_stamped(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("NORMAL"), _admission())
    assert ev.candidate_id == CANDIDATE_BASELINE.candidate_id


# ---------------------------------------------------------------------------
# Censoring (§19) — never a quantitative residual from a censored sample
# ---------------------------------------------------------------------------


def test_censored_sample_produces_indeterminate_residual(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("CENSORED_ANOMALY"), _admission())
    fmap = _field_map(ev)
    assert fmap["censored_measurement_present"].value == YES
    assert fmap["site_residual_behaviour"].value == RESIDUAL_INDETERMINATE
    assert fmap["site_residual_behaviour"].uncertainty == UNDETERMINED


@pytest.mark.parametrize("candidate_id", CANDIDATE_IDS)
def test_no_candidate_derives_a_quantitative_residual_from_censored(
    features_factory, candidate_id
):
    # Every candidate must respect §19: a censored site never yields
    # SYSTEMATIC_STABLE / VARIABLE (that would raise CensoredInferenceError at
    # construction anyway, so reaching here without raising is the assertion).
    ev = infer_site(candidate(candidate_id), features_factory("CENSORED_ANOMALY"), _admission())
    fmap = _field_map(ev)
    assert fmap["site_residual_behaviour"].value not in (
        RESIDUAL_SYSTEMATIC_STABLE,
        RESIDUAL_VARIABLE,
    )


# ---------------------------------------------------------------------------
# Honest UNDETERMINED (§21) — insufficient evidence is explicit, never a default
# ---------------------------------------------------------------------------


def test_normal_site_is_not_forced_to_yes_or_no(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("NORMAL"), _admission())
    fmap = _field_map(ev)
    # A normal site has no residual; persistence is genuinely undecidable.
    assert fmap["site_residual_behaviour"].value == RESIDUAL_NONE
    assert fmap["persisted_at_same_sensor_coord"].value == UNDETERMINED
    assert fmap["persisted_at_same_sensor_coord"].uncertainty == UNDETERMINED


def test_undetermined_fields_carry_explicit_uncertainty(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("NORMAL"), _admission())
    for f in ev.sensor_evidence:
        if f.value in (UNDETERMINED, RESIDUAL_INDETERMINATE):
            assert f.uncertainty == UNDETERMINED


def test_every_field_carries_source_and_feature_refs(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("SIGN_CHANGING_POST_DARK"), _admission())
    for f in ev.sensor_evidence:
        assert f.source
        assert isinstance(f.feature_refs, tuple)


# ---------------------------------------------------------------------------
# The classes the rules are supposed to separate, at a smoke level
# ---------------------------------------------------------------------------


def test_sign_changing_conflicts(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("SIGN_CHANGING_POST_DARK"), _admission())
    fmap = _field_map(ev)
    assert fmap["conflicting_evidence"].value == YES


def test_star_crossing_is_not_a_sensor_site(features_factory):
    # A moving star is a sky structure: not persisted at the sensor coordinate,
    # and the neighbourhood disagreement is global.
    ev = infer_site(CANDIDATE_BASELINE, features_factory("STAR_CROSSING_SITE"), _admission())
    fmap = _field_map(ev)
    assert fmap["persisted_at_same_sensor_coord"].value == NO
    assert fmap["neighbourhood_residual_stable"].value == NO


def test_single_transient_is_transient_only(features_factory):
    ev = infer_site(CANDIDATE_BASELINE, features_factory("SINGLE_TRANSIENT"), _admission())
    fmap = _field_map(ev)
    assert fmap["transient_only"].value == YES
    assert fmap["persisted_at_same_sensor_coord"].value == NO


# ---------------------------------------------------------------------------
# Admission facts traverse untouched (never inferred)
# ---------------------------------------------------------------------------


def test_admission_is_passed_through_verbatim(features_factory):
    adm = AdmissionFacts(
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=YES,
        calibration_representativeness="REPRESENTATIVE",
        independent_group_count=4,
        epoch_count=2,
    )
    ev = infer_site(CANDIDATE_BASELINE, features_factory("NORMAL"), adm)
    assert ev.admission is adm


def _admission() -> AdmissionFacts:
    return AdmissionFacts(
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=YES,
        calibration_representativeness="REPRESENTATIVE",
        independent_group_count=4,
        epoch_count=2,
    )
