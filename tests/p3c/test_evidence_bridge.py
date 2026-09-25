"""P3C-2 — evidence bridge (inference → existing LOT2 policy) tests.

Includes the fail-closed safety-fact convention (rework-1 M1): an UNDETERMINED
safety fact must lead the existing policy to ABSTAIN, never to proceed.
"""

from __future__ import annotations

import pytest

from research.p3c.evidence_bridge import ADAPTER_VERSION, _safety_fact, to_evidence_packet
from research.p3c.inference_candidates import CANDIDATE_BASELINE, infer_site
from research.p3c.inference_contract import (
    NO,
    UNDETERMINED,
    YES,
    AdmissionFacts,
    InferredEvidence,
    InferredField,
)
from research.p3b.qualification_policy import (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    RC_EVIDENCE_CENSORED,
    RC_INCONSISTENT_EVIDENCE,
    RC_TRANSIENT_ONLY,
    EvidencePacket,
    evaluate,
)


def _admission(**overrides):
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


def _full_evidence(**overrides) -> InferredEvidence:
    """A benign, otherwise-eligible evidence set, with one fact overridable."""
    facts = {
        "persisted_at_same_sensor_coord": YES,
        "site_residual_behaviour": "NONE",
        "neighbourhood_residual_stable": YES,
        "transient_only": NO,
        "conflicting_evidence": NO,
        "censored_measurement_present": NO,
    }
    facts.update(overrides)
    fields = []
    for name, value in facts.items():
        uncertainty = "UNDETERMINED" if value in (UNDETERMINED, "INDETERMINATE") else "DETERMINED"
        fields.append(InferredField(name, value, "test", uncertainty))
    return InferredEvidence(
        candidate_id="c1", admission=_admission(), sensor_evidence=tuple(fields)
    )


def test_adapter_version_is_bumped():
    # rework-1 M1: the convention changed (fail-closed), so the version must too.
    assert ADAPTER_VERSION == "p3c-evidence-bridge-2"


def test_exogenous_facts_traverse_verbatim():
    adm = _admission()
    ev = InferredEvidence(candidate_id="c1", admission=adm)
    packet = to_evidence_packet(ev)
    assert isinstance(packet, EvidencePacket)
    assert packet.sensor_identity_resolved == adm.sensor_identity_resolved
    assert packet.geometry_compatible == adm.geometry_compatible
    assert packet.calibration_present == adm.calibration_present
    assert packet.calibration_representativeness == adm.calibration_representativeness
    assert packet.independent_group_count == adm.independent_group_count
    assert packet.epoch_count == adm.epoch_count


def test_sensor_evidence_mapped_by_name():
    ev = InferredEvidence(
        candidate_id="c1",
        admission=_admission(),
        sensor_evidence=(
            InferredField("persisted_at_same_sensor_coord", YES, "s", "DETERMINED"),
            InferredField("site_residual_behaviour", "SYSTEMATIC_STABLE", "s", "DETERMINED"),
            InferredField("neighbourhood_residual_stable", YES, "s", "DETERMINED"),
            InferredField("transient_only", NO, "s", "DETERMINED"),
            InferredField("conflicting_evidence", NO, "s", "DETERMINED"),
            InferredField("censored_measurement_present", NO, "s", "DETERMINED"),
        ),
    )
    packet = to_evidence_packet(ev)
    assert packet.persisted_at_same_sensor_coord == YES
    assert packet.site_residual_behaviour == "SYSTEMATIC_STABLE"
    assert packet.neighbourhood_residual_stable == YES
    assert packet.transient_only == NO
    assert packet.conflicting_evidence == NO
    assert packet.censored_measurement_present == NO


def test_net_benefit_is_never_forced():
    ev = InferredEvidence(candidate_id="c1", admission=_admission())
    packet = to_evidence_packet(ev)
    assert packet.net_benefit_established == UNDETERMINED


def test_persisted_basis_names_the_candidate():
    ev = InferredEvidence(candidate_id="cand-7", admission=_admission())
    packet = to_evidence_packet(ev)
    assert "cand-7" in packet.persisted_basis


def test_inferred_then_bridged_reaches_policy(features_factory):
    features = features_factory("INTERMITTENT_TWO_STATE")
    ev = infer_site(CANDIDATE_BASELINE, features, _admission())
    packet = to_evidence_packet(ev)
    assert isinstance(packet, EvidencePacket)
    # Every field of the packet is a fact string (no numeric amplitude/score).
    for field in (
        "site_residual_behaviour",
        "persisted_at_same_sensor_coord",
        "neighbourhood_residual_stable",
        "transient_only",
        "conflicting_evidence",
        "censored_measurement_present",
    ):
        assert isinstance(getattr(packet, field), str)


# ---------------------------------------------------------------------------
# rework-1 M1 — fail-closed safety facts
# ---------------------------------------------------------------------------

_SAFETY_FACTS = (
    "censored_measurement_present",
    "conflicting_evidence",
    "transient_only",
)


@pytest.mark.parametrize("field", _SAFETY_FACTS)
def test_safety_fact_undetermined_maps_to_yes_not_no(field):
    # Direct witness: under the OLD convention this returned NO (permissive).
    # The fail-closed convention must return YES (abstain).
    assert _safety_fact(field, UNDETERMINED) == YES
    assert _safety_fact(field, NO) == NO
    assert _safety_fact(field, YES) == YES


def test_witness_old_no_mapping_would_not_abstain_on_safety_guard():
    # A test that FAILS under the old UNDETERMINED→NO convention: with
    # transient_only UNDETERMINED, the old bridge passed transient_only=NO and
    # the policy proceeded past the P4 guard (here to NO_ACTION via residual
    # NONE); the fail-closed bridge must instead trigger the transient guard.
    ev = _full_evidence(transient_only=UNDETERMINED)
    decision = evaluate(to_evidence_packet(ev))
    assert decision.action == ACTION_ABSTAIN_INCONSISTENT
    assert RC_TRANSIENT_ONLY in decision.reason_codes


@pytest.mark.parametrize(
    "override, expected_action, expected_code",
    [
        ("censored_measurement_present", ACTION_ABSTAIN_CENSORED, RC_EVIDENCE_CENSORED),
        ("conflicting_evidence", ACTION_ABSTAIN_INCONSISTENT, RC_INCONSISTENT_EVIDENCE),
        ("transient_only", ACTION_ABSTAIN_INCONSISTENT, RC_TRANSIENT_ONLY),
    ],
)
def test_undetermined_safety_fact_forces_policy_abstention(
    override, expected_action, expected_code
):
    # Each safety fact, left UNDETERMINED, must make the EXISTING policy abstain
    # at that fact's guard — never pass it.
    ev = _full_evidence(**{override: UNDETERMINED})
    decision = evaluate(to_evidence_packet(ev))
    assert decision.action == expected_action
    assert expected_code in decision.reason_codes


def test_the_three_safety_facts_remain_binary_in_the_packet():
    # The fix does not touch qualification_policy: the packet facts stay binary.
    ev = _full_evidence(transient_only=UNDETERMINED)
    packet = to_evidence_packet(ev)
    for field in _SAFETY_FACTS:
        assert getattr(packet, field) in (YES, NO)
