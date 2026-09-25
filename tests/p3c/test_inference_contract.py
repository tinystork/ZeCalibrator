"""P3C-1 tests: the inference contract (typed result, fact separation, UNDETERMINED)."""

from __future__ import annotations

import pytest

from research.p3c.inference_contract import (
    DETERMINED,
    NO,
    RESIDUAL_INDETERMINATE,
    RESIDUAL_NONE,
    RESIDUAL_SYSTEMATIC_STABLE,
    RESIDUAL_VARIABLE,
    UNDETERMINED,
    YES,
    EXOGENOUS_FACTS,
    NON_INFERABLE_FACTS,
    SENSOR_EVIDENCE_FACTS,
    AdmissionFacts,
    CensoredInferenceError,
    ExogenousFactError,
    InferredField,
    InferredEvidence,
    NonInferableFactError,
    UnknownFactError,
    build_inferred_evidence,
    undetermined_value_for,
)


def _field(name, value, uncertainty=DETERMINED, source="test:measured_feature", refs=()):
    return InferredField(
        field=name, value=value, uncertainty=uncertainty, source=source, feature_refs=tuple(refs)
    )


# ---------------------------------------------------------------------------
# Fact categories are structurally separated (§17)
# ---------------------------------------------------------------------------


def test_two_categories_are_exactly_the_expected_fact_sets():
    assert set(EXOGENOUS_FACTS) == {
        "sensor_identity_resolved",
        "geometry_compatible",
        "calibration_present",
        "calibration_representativeness",
        "independent_group_count",
        "epoch_count",
    }
    assert set(SENSOR_EVIDENCE_FACTS) == {
        "persisted_at_same_sensor_coord",
        "site_residual_behaviour",
        "neighbourhood_residual_stable",
        "transient_only",
        "conflicting_evidence",
        "censored_measurement_present",
    }
    assert NON_INFERABLE_FACTS == ("net_benefit_established",)


def test_categories_are_disjoint():
    assert not set(EXOGENOUS_FACTS) & set(SENSOR_EVIDENCE_FACTS)
    assert not set(EXOGENOUS_FACTS) & set(NON_INFERABLE_FACTS)
    assert not set(SENSOR_EVIDENCE_FACTS) & set(NON_INFERABLE_FACTS)


# ---------------------------------------------------------------------------
# An exogenous fact cannot be overwritten by a pixel inference (§2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", EXOGENOUS_FACTS)
def test_exogenous_fact_cannot_be_inferred(field):
    with pytest.raises(ExogenousFactError):
        _field(field, YES)


def test_exogenous_error_names_the_field():
    with pytest.raises(ExogenousFactError) as exc:
        _field("epoch_count", "3")
    assert exc.value.field == "epoch_count"


def test_exogenous_facts_are_frozen_pass_through():
    adm = AdmissionFacts(
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=YES,
        calibration_representativeness="REPRESENTATIVE",
        independent_group_count=3,
        epoch_count=2,
    )
    with pytest.raises(Exception):
        adm.sensor_identity_resolved = NO  # frozen dataclass


def test_admission_rejects_bad_values():
    with pytest.raises(ValueError):
        AdmissionFacts(sensor_identity_resolved="maybe")
    with pytest.raises(ValueError):
        AdmissionFacts(independent_group_count=-1)
    with pytest.raises(ValueError):
        AdmissionFacts(epoch_count=True)


# ---------------------------------------------------------------------------
# net_benefit_established is NOT predicted (§18)
# ---------------------------------------------------------------------------


def test_net_benefit_is_not_inferable():
    with pytest.raises(NonInferableFactError):
        _field("net_benefit_established", YES)


def test_net_benefit_is_always_undetermined_on_result():
    ev = InferredEvidence(candidate_id="c1", admission=AdmissionFacts())
    assert ev.net_benefit_established == UNDETERMINED
    # even with a full sensor-evidence set, it is never ESTABLISHED by inference
    full = build_inferred_evidence(
        "c1",
        AdmissionFacts(),
        (
            _field("persisted_at_same_sensor_coord", YES),
            _field("site_residual_behaviour", RESIDUAL_SYSTEMATIC_STABLE),
            _field("neighbourhood_residual_stable", YES),
            _field("transient_only", NO),
            _field("conflicting_evidence", NO),
            _field("censored_measurement_present", NO),
        ),
    )
    assert full.net_benefit_established == UNDETERMINED


# ---------------------------------------------------------------------------
# UNDETERMINED is a normal, explicit output (§4, §21)
# ---------------------------------------------------------------------------


def test_unknown_stays_unknown_explicit():
    f = _field("transient_only", UNDETERMINED, uncertainty=UNDETERMINED)
    assert f.value == UNDETERMINED
    assert f.uncertainty == UNDETERMINED


def test_undetermined_value_requires_explicit_uncertainty():
    # A field carrying the "insufficient evidence" value must be marked
    # UNDETERMINED explicitly — never silently defaulted.
    with pytest.raises(ValueError):
        _field("transient_only", UNDETERMINED, uncertainty=DETERMINED)


def test_determined_value_cannot_claim_undetermined_uncertainty():
    with pytest.raises(ValueError):
        _field("transient_only", YES, uncertainty=UNDETERMINED)


def test_residual_uses_structured_undetermined_state():
    assert undetermined_value_for("site_residual_behaviour") == RESIDUAL_INDETERMINATE
    f = _field("site_residual_behaviour", RESIDUAL_INDETERMINATE, uncertainty=UNDETERMINED)
    assert f.value == RESIDUAL_INDETERMINATE


def test_missing_field_is_none_not_implicit_yes_no():
    ev = InferredEvidence(candidate_id="c1", admission=AdmissionFacts())
    assert ev.field("transient_only") is None  # "not inferred", never a silent YES/NO


# ---------------------------------------------------------------------------
# Censoring (§19): censored != high-value anomaly; no quantitative state
# ---------------------------------------------------------------------------


def test_censored_sample_forbids_characterised_residual():
    with pytest.raises(CensoredInferenceError):
        build_inferred_evidence(
            "c1",
            AdmissionFacts(),
            (
                _field("censored_measurement_present", YES),
                _field("site_residual_behaviour", RESIDUAL_SYSTEMATIC_STABLE),
            ),
        )
    with pytest.raises(CensoredInferenceError):
        build_inferred_evidence(
            "c1",
            AdmissionFacts(),
            (
                _field("censored_measurement_present", YES),
                _field("site_residual_behaviour", RESIDUAL_VARIABLE),
            ),
        )


def test_censored_sample_allows_non_quantitative_residual():
    # NONE / INDETERMINATE are not quantitative states, so they are allowed.
    build_inferred_evidence(
        "c1",
        AdmissionFacts(),
        (
            _field("censored_measurement_present", YES),
            _field("site_residual_behaviour", RESIDUAL_NONE),
        ),
    )
    build_inferred_evidence(
        "c1",
        AdmissionFacts(),
        (
            _field("censored_measurement_present", YES),
            _field("site_residual_behaviour", RESIDUAL_INDETERMINATE, uncertainty=UNDETERMINED),
        ),
    )


# ---------------------------------------------------------------------------
# Provenance is machine-visible (§1)
# ---------------------------------------------------------------------------


def test_every_inferred_field_carries_machine_visible_provenance():
    f = _field(
        "persisted_at_same_sensor_coord",
        YES,
        source="measured_feature:same_cfa_local_residual",
        refs=("same_cfa_local_residual", "temporal_value_series"),
    )
    assert f.source == "measured_feature:same_cfa_local_residual"
    assert f.feature_refs == ("same_cfa_local_residual", "temporal_value_series")
    assert f.uncertainty == DETERMINED


def test_candidate_id_is_required_and_visible():
    ev = InferredEvidence(candidate_id="cand-17", admission=AdmissionFacts())
    assert ev.candidate_id == "cand-17"
    with pytest.raises(ValueError):
        InferredEvidence(candidate_id="", admission=AdmissionFacts())


def test_duplicate_fields_are_rejected():
    with pytest.raises(ValueError):
        build_inferred_evidence(
            "c1",
            AdmissionFacts(),
            (_field("transient_only", NO), _field("transient_only", NO)),
        )


# ---------------------------------------------------------------------------
# No magic boolean, no confidence score (§1)
# ---------------------------------------------------------------------------


def test_values_are_strings_not_booleans_and_there_is_no_score_field():
    f = _field("transient_only", NO)
    assert isinstance(f.value, str)
    assert not isinstance(f.value, bool)
    # The field model has no numeric score / probability attribute at all.
    assert not hasattr(f, "score")
    assert not hasattr(f, "confidence")
    assert not hasattr(f, "probability")
    # No float is carried anywhere in the field dataclass.
    for attr in ("field", "value", "source", "uncertainty"):
        assert not isinstance(getattr(f, attr), float)


def test_unknown_field_name_raises_typed_error():
    with pytest.raises(UnknownFactError):
        _field("not_a_fact", YES)
