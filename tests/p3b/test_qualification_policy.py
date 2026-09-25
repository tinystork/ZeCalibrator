"""LOT2 tests: qualification policy engine (properties 1-9 + ladder + projection).

Scoped, fast: no corpus generation, no FITS, no numpy in the engine under test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from research.p3b.qualification_policy import (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
    ACTION_LADDER,
    ACTION_NO_ACTION,
    ACTION_REQUALIFY,
    EPISTEMIC_CHARACTERISED_STABLE,
    EPISTEMIC_CENSORED,
    EPISTEMIC_UNKNOWN,
    MISSION_NAMES,
    MISSION_NAME_PROJECTIONS,
    NET_BENEFIT_ESTABLISHED,
    NET_BENEFIT_NOT_ESTABLISHED,
    NOT_REPRESENTATIVE,
    NO,
    RC_CALIBRATION_NOT_REPRESENTATIVE,
    RC_EVIDENCE_CENSORED,
    RC_INSUFFICIENT_EPOCHS,
    RC_INSUFFICIENT_INDEPENDENT_GROUPS,
    RC_NET_BENEFIT_NOT_ESTABLISHED,
    RC_NO_ACTION_NEEDED_AFTER_CALIBRATION,
    RC_TRANSIENT_ONLY,
    REPRESENTATIVE,
    RESIDUAL_INDETERMINATE,
    RESIDUAL_NONE,
    RESIDUAL_SYSTEMATIC_STABLE,
    RESIDUAL_VARIABLE,
    UNDETERMINED,
    YES,
    CensoredInferenceError,
    EvidencePacket,
    QualificationDecision,
    evaluate,
    project_mission_name,
)

_QP_PATH = Path(__file__).resolve().parents[2] / "research" / "p3b" / "qualification_policy.py"


# ---------------------------------------------------------------------------
# Hand-built packet helpers (a "fully good" baseline + overrides)
# ---------------------------------------------------------------------------

def make_packet(**overrides) -> EvidencePacket:
    """A complete, fully-qualified 'good' dossier (eligible when untouched)."""
    base = dict(
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=YES,
        calibration_representativeness=REPRESENTATIVE,
        persisted_at_same_sensor_coord=YES,
        independent_group_count=2,
        epoch_count=2,
        censored_measurement_present=NO,
        site_residual_behaviour=RESIDUAL_SYSTEMATIC_STABLE,
        neighbourhood_residual_stable=YES,
        transient_only=NO,
        conflicting_evidence=NO,
        net_benefit_established=NET_BENEFIT_ESTABLISHED,
    )
    base.update(overrides)
    return EvidencePacket(**base)


# ---------------------------------------------------------------------------
# Property 8 — pure function of facts: no numpy/astropy, no FITS, no file I/O
# ---------------------------------------------------------------------------

def test_engine_module_imports_no_numpy_astropy_fits_and_no_file_io():
    source = _QP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    banned = [name for name in imported if any(t in name.lower() for t in ("numpy", "astropy", "fits"))]
    assert banned == [], f"engine imports forbidden modules: {banned}"

    # No file access: no open() call, no Path()/read_text/read_bytes/write_*.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open", "engine must not call open()"
        if isinstance(node, ast.Attribute) and node.attr in (
            "read_text", "read_bytes", "write_text", "write_bytes",
        ):
            raise AssertionError(f"engine must not access files via {node.attr}()")


def test_engine_executes_on_hand_built_packet():
    p = make_packet()
    d = evaluate(p)
    assert isinstance(d, QualificationDecision)
    assert d.action == ACTION_ELIGIBLE


# ---------------------------------------------------------------------------
# Property 1 — no detected -> corrected shortcut
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "override",
    [
        {"calibration_representativeness": NOT_REPRESENTATIVE},
        {"calibration_representativeness": UNDETERMINED},
        {"censored_measurement_present": YES, "site_residual_behaviour": RESIDUAL_INDETERMINATE,
         "net_benefit_established": NET_BENEFIT_NOT_ESTABLISHED},
        {"conflicting_evidence": YES},
        {"transient_only": YES},
        {"sensor_identity_resolved": NO},
        {"geometry_compatible": NO},
        {"geometry_compatible": UNDETERMINED},
        {"calibration_present": NO},
        {"epoch_count": 1},
        {"independent_group_count": 1},
        {"net_benefit_established": NET_BENEFIT_NOT_ESTABLISHED},
        {"net_benefit_established": UNDETERMINED},
        {"neighbourhood_residual_stable": NO},
        {"neighbourhood_residual_stable": UNDETERMINED},
        {"site_residual_behaviour": RESIDUAL_NONE},
        {"site_residual_behaviour": RESIDUAL_INDETERMINATE},
    ],
)
def test_no_single_missing_qualification_fact_yields_eligible(override):
    # Breaking ANY one required fact of the complete dossier must never yield
    # ELIGIBLE — there is no shortcut from "residual present" to "reconstruct".
    d = evaluate(make_packet(**override))
    assert d.action != ACTION_ELIGIBLE


def test_detection_signal_alone_never_eligible():
    # A packet carrying only a residual signal and nothing else qualified.
    p = make_packet(
        sensor_identity_resolved=NO,
        geometry_compatible=NO,
        calibration_present=NO,
        calibration_representativeness=UNDETERMINED,
        independent_group_count=0,
        epoch_count=0,
        net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED,
        neighbourhood_residual_stable=UNDETERMINED,
    )
    d = evaluate(p)
    assert d.action != ACTION_ELIGIBLE
    assert d.action.startswith("ABSTAIN")


# ---------------------------------------------------------------------------
# Property 2 — representativeness dominates (old inadequate dark)
# ---------------------------------------------------------------------------

def test_not_representative_forces_requalify_even_when_everything_else_is_good():
    d = evaluate(make_packet(calibration_representativeness=NOT_REPRESENTATIVE))
    assert d.action == ACTION_REQUALIFY
    assert d.reason_codes == (RC_CALIBRATION_NOT_REPRESENTATIVE,)
    assert d.representativeness == NOT_REPRESENTATIVE


def test_requalify_never_reconstructs():
    # "vieux dark inadéquat": even with net benefit established and full
    # independence, the decision is requalify, never eligible.
    d = evaluate(
        make_packet(
            calibration_representativeness=NOT_REPRESENTATIVE,
            net_benefit_established=NET_BENEFIT_ESTABLISHED,
        )
    )
    assert d.action == ACTION_REQUALIFY
    assert d.action != ACTION_ELIGIBLE


# ---------------------------------------------------------------------------
# Property 3 — censoring
# ---------------------------------------------------------------------------

def test_censored_evidence_abstains_censored():
    d = evaluate(
        make_packet(
            censored_measurement_present=YES,
            site_residual_behaviour=RESIDUAL_INDETERMINATE,
            net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED,
        )
    )
    assert d.action == ACTION_ABSTAIN_CENSORED
    assert d.reason_codes == (RC_EVIDENCE_CENSORED,)
    assert d.epistemic_state == EPISTEMIC_CENSORED


def test_censored_site_cannot_claim_characterised_residual():
    with pytest.raises(CensoredInferenceError):
        evaluate(
            make_packet(
                censored_measurement_present=YES,
                site_residual_behaviour=RESIDUAL_VARIABLE,
            )
        )


def test_censored_site_cannot_claim_established_net_benefit():
    with pytest.raises(CensoredInferenceError):
        evaluate(
            make_packet(
                censored_measurement_present=YES,
                site_residual_behaviour=RESIDUAL_INDETERMINATE,
                net_benefit_established=NET_BENEFIT_ESTABLISHED,
            )
        )


# ---------------------------------------------------------------------------
# Property 4 — independence (distinct codes, never merged)
# ---------------------------------------------------------------------------

def test_insufficient_epochs_distinct_code():
    d = evaluate(make_packet(epoch_count=1))
    assert d.action == ACTION_ABSTAIN_INSUFFICIENT
    assert d.reason_codes == (RC_INSUFFICIENT_EPOCHS,)


def test_insufficient_independent_groups_distinct_code():
    d = evaluate(make_packet(independent_group_count=1))
    assert d.action == ACTION_ABSTAIN_INSUFFICIENT
    assert d.reason_codes == (RC_INSUFFICIENT_INDEPENDENT_GROUPS,)


def test_both_insufficient_yields_both_distinct_codes_in_order():
    d = evaluate(make_packet(epoch_count=1, independent_group_count=1))
    assert d.action == ACTION_ABSTAIN_INSUFFICIENT
    assert d.reason_codes == (RC_INSUFFICIENT_EPOCHS, RC_INSUFFICIENT_INDEPENDENT_GROUPS)


def test_independence_codes_are_distinct_tokens():
    assert RC_INSUFFICIENT_EPOCHS != RC_INSUFFICIENT_INDEPENDENT_GROUPS


# ---------------------------------------------------------------------------
# Property 5 — knowledge is not action
# ---------------------------------------------------------------------------

def test_known_site_corrected_by_representative_calibration_is_no_action():
    d = evaluate(make_packet(site_residual_behaviour=RESIDUAL_NONE))
    assert d.action == ACTION_NO_ACTION
    assert d.reason_codes == (RC_NO_ACTION_NEEDED_AFTER_CALIBRATION,)
    # the site stays known, never "removed"
    assert d.epistemic_state == EPISTEMIC_CHARACTERISED_STABLE


def test_normal_site_is_no_action_but_unknown():
    d = evaluate(
        make_packet(site_residual_behaviour=RESIDUAL_NONE, persisted_at_same_sensor_coord=NO)
    )
    assert d.action == ACTION_NO_ACTION
    assert d.epistemic_state == EPISTEMIC_UNKNOWN


# ---------------------------------------------------------------------------
# Property 6 — eligibility is not obligation (net benefit consumed, not computed)
# ---------------------------------------------------------------------------

def test_eligible_requires_net_benefit_established():
    d = evaluate(make_packet(net_benefit_established=NET_BENEFIT_ESTABLISHED))
    assert d.action == ACTION_ELIGIBLE


def test_missing_net_benefit_yields_net_benefit_not_established_abstention():
    d = evaluate(make_packet(net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED))
    assert d.action == ACTION_ABSTAIN_INSUFFICIENT
    assert d.reason_codes == (RC_NET_BENEFIT_NOT_ESTABLISHED,)


def test_undetermined_net_benefit_also_abstains():
    d = evaluate(make_packet(net_benefit_established=UNDETERMINED))
    assert d.action == ACTION_ABSTAIN_INSUFFICIENT
    assert d.reason_codes == (RC_NET_BENEFIT_NOT_ESTABLISHED,)


# ---------------------------------------------------------------------------
# Property 7 — transient only
# ---------------------------------------------------------------------------

def test_transient_only_abstains_with_transient_code():
    d = evaluate(make_packet(transient_only=YES))
    assert d.action == ACTION_ABSTAIN_INCONSISTENT
    assert d.reason_codes == (RC_TRANSIENT_ONLY,)
    assert d.action != ACTION_ELIGIBLE


# ---------------------------------------------------------------------------
# Property 9 — non-boolean decision: non-empty reason codes + exact summary
# ---------------------------------------------------------------------------

def test_reason_codes_never_empty():
    # A representative sample of packets: every decision carries reason codes.
    packets = [
        make_packet(),  # eligible
        make_packet(site_residual_behaviour=RESIDUAL_NONE),  # no action
        make_packet(calibration_representativeness=NOT_REPRESENTATIVE),  # requalify
        make_packet(censored_measurement_present=YES, site_residual_behaviour=RESIDUAL_INDETERMINATE,
                    net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED),  # censored
        make_packet(conflicting_evidence=YES),  # inconsistent
        make_packet(epoch_count=1),  # insufficient
        make_packet(sensor_identity_resolved=NO),  # identity unresolved
    ]
    for p in packets:
        d = evaluate(p)
        assert d.reason_codes, "reason_codes must never be empty"


def test_unused_fact_does_not_change_decision():
    a = make_packet(persisted_basis="site H1 stable across epochs")
    b = make_packet(persisted_basis="completely different factual basis")
    assert evaluate(a) == evaluate(b)


def test_unused_count_does_not_change_decision():
    a = make_packet(censored_measurement_count=0)
    b = make_packet(censored_measurement_count=99)
    assert evaluate(a) == evaluate(b)


def test_used_fact_appears_in_evidence_summary():
    d = evaluate(make_packet(net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED))
    fields = {f.field for f in d.evidence_summary}
    assert "net_benefit_established" in fields
    assert "site_residual_behaviour" in fields


def test_unused_fact_absent_from_evidence_summary():
    d = evaluate(make_packet(persisted_basis="some basis", censored_measurement_count=5))
    fields = {f.field for f in d.evidence_summary}
    assert "persisted_basis" not in fields
    assert "censored_measurement_count" not in fields


def test_evidence_summary_captures_representativeness_axis():
    d = evaluate(make_packet(calibration_representativeness=NOT_REPRESENTATIVE))
    assert d.representativeness == NOT_REPRESENTATIVE
    fields = {f.field for f in d.evidence_summary}
    assert "calibration_representativeness" in fields


# ---------------------------------------------------------------------------
# Action ladder ordering (explicitly required by the mission)
# ---------------------------------------------------------------------------

def test_ladder_is_exactly_the_ordered_scale():
    assert ACTION_LADDER == (
        ACTION_REQUALIFY,
        ACTION_ABSTAIN_CENSORED,
        ACTION_ABSTAIN_INCONSISTENT,
        ACTION_ABSTAIN_INSUFFICIENT,
        ACTION_ELIGIBLE,
        ACTION_NO_ACTION,
    )


def test_requalify_beats_abstain_censored_when_both_apply():
    # NOT_REPRESENTATIVE + censored: requalify (priority 1) wins over censored (priority 2).
    d = evaluate(
        make_packet(
            calibration_representativeness=NOT_REPRESENTATIVE,
            censored_measurement_present=YES,
            site_residual_behaviour=RESIDUAL_INDETERMINATE,
            net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED,
        )
    )
    assert d.action == ACTION_REQUALIFY


def test_abstain_censored_beats_abstain_insufficient_evidence():
    # censored + insufficient evidence: censored (priority 2) wins (priority 4).
    d = evaluate(
        make_packet(
            censored_measurement_present=YES,
            site_residual_behaviour=RESIDUAL_INDETERMINATE,
            epoch_count=1,
            independent_group_count=1,
            net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED,
        )
    )
    assert d.action == ACTION_ABSTAIN_CENSORED


# ---------------------------------------------------------------------------
# Projection to the 8 mission-brief names (total + reachable)
# ---------------------------------------------------------------------------

def test_projection_table_covers_exactly_the_eight_mission_names():
    assert set(MISSION_NAME_PROJECTIONS) == set(MISSION_NAMES)


def _decision_for(name: str) -> QualificationDecision:
    """One canonical packet -> decision that reaches each of the 8 names."""
    if name == "UNKNOWN":
        return evaluate(make_packet(site_residual_behaviour=RESIDUAL_NONE, persisted_at_same_sensor_coord=NO))
    if name == "KNOWN_STABLE":
        return evaluate(make_packet(site_residual_behaviour=RESIDUAL_NONE))
    if name == "KNOWN_INTERMITTENT_CANDIDATE":
        return evaluate(make_packet(site_residual_behaviour=RESIDUAL_VARIABLE))
    if name == "CENSORED":
        return evaluate(make_packet(censored_measurement_present=YES,
                                    site_residual_behaviour=RESIDUAL_INDETERMINATE,
                                    net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED))
    if name == "TRANSIENT_OR_UNRESOLVED":
        return evaluate(make_packet(transient_only=YES))
    if name == "CALIBRATION_NOT_REPRESENTATIVE":
        return evaluate(make_packet(calibration_representativeness=NOT_REPRESENTATIVE))
    if name == "NO_ACTION_REQUIRED":
        return evaluate(make_packet(site_residual_behaviour=RESIDUAL_NONE))
    if name == "QUALIFIED_FOR_RECONSTRUCTION":
        return evaluate(make_packet())
    raise AssertionError(name)


def test_each_mission_name_is_reachable_by_projection():
    for name in MISSION_NAMES:
        d = _decision_for(name)
        assert MISSION_NAME_PROJECTIONS[name](d), f"name {name!r} not reachable"


def test_projection_function_is_total():
    # Cover every epistemic state and every action: the function must always
    # return one of the 8 names.
    decisions = [
        evaluate(make_packet()),  # CHARACTERISED_STABLE + ELIGIBLE
        evaluate(make_packet(site_residual_behaviour=RESIDUAL_NONE)),  # NO_ACTION
        evaluate(make_packet(site_residual_behaviour=RESIDUAL_NONE, persisted_at_same_sensor_coord=NO)),  # UNKNOWN
        evaluate(make_packet(site_residual_behaviour=RESIDUAL_VARIABLE)),  # CHARACTERISED_INTERMITTENT
        evaluate(make_packet(site_residual_behaviour=RESIDUAL_VARIABLE, epoch_count=1)),  # OBSERVED_CANDIDATE
        evaluate(make_packet(transient_only=YES)),  # TRANSIENT_OR_UNRESOLVED
        evaluate(make_packet(censored_measurement_present=YES, site_residual_behaviour=RESIDUAL_INDETERMINATE,
                             net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED)),  # CENSORED
        evaluate(make_packet(calibration_representativeness=NOT_REPRESENTATIVE)),  # REQUALIFY
    ]
    for d in decisions:
        assert project_mission_name(d) in MISSION_NAMES


def test_projection_function_documented_mapping():
    assert project_mission_name(evaluate(make_packet())) == "QUALIFIED_FOR_RECONSTRUCTION"
    assert project_mission_name(evaluate(make_packet(site_residual_behaviour=RESIDUAL_NONE))) == "NO_ACTION_REQUIRED"
    assert project_mission_name(evaluate(make_packet(calibration_representativeness=NOT_REPRESENTATIVE))) == "CALIBRATION_NOT_REPRESENTATIVE"
    assert project_mission_name(evaluate(make_packet(censored_measurement_present=YES,
                                                     site_residual_behaviour=RESIDUAL_INDETERMINATE,
                                                     net_benefit_established=NET_BENEFIT_NOT_ESTABLISHED))) == "CENSORED"
    assert project_mission_name(evaluate(make_packet(transient_only=YES))) == "TRANSIENT_OR_UNRESOLVED"
