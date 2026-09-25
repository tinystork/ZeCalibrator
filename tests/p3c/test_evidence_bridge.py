"""P3C-2 — evidence bridge (inference → existing LOT2 policy) tests."""

from __future__ import annotations

from research.p3c.evidence_bridge import ADAPTER_VERSION, to_evidence_packet
from research.p3c.inference_candidates import CANDIDATE_BASELINE, infer_site
from research.p3c.inference_contract import (
    NO,
    UNDETERMINED,
    YES,
    AdmissionFacts,
    InferredEvidence,
    InferredField,
)
from research.p3b.qualification_policy import EvidencePacket


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


def test_adapter_version_is_declared():
    assert ADAPTER_VERSION


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


def test_binary_fact_undetermined_maps_to_no():
    # The inference contract allows UNDETERMINED on transient_only; the LOT2
    # packet is binary. The documented adapter convention maps UNDETERMINED→NO.
    ev = InferredEvidence(
        candidate_id="c1",
        admission=_admission(),
        sensor_evidence=(InferredField("transient_only", UNDETERMINED, "s", "UNDETERMINED"),),
    )
    packet = to_evidence_packet(ev)
    assert packet.transient_only == NO


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
