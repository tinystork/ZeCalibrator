"""LOT5 tests: metrics contract (units, universes, hand-computed values, properties).

Covered requirements (mission ZC-SENSOR-P3B-LOT5-METRICS-CONTRACT):

1. explicit units / universes: every metric declares universe + unit, and the
   three FPR variants are distinct and non-addable;
2. known value: >= 4 metrics hand-computed on a small hand-built set (including
   ``site_level_false_promotion_rate`` and ``unsafe_action_rate``);
3. abstention != failure: a detected site that correctly abstains counts in
   ``correct_abstention_rate`` and NOT in a detection error;
4. censure invariant: 0 quantitative inferences from a censored measurement;
5. no threshold selection (AST + execution);
6. independence recurrence: ``group_level_repeatability`` cannot be improved by
   adding frames to the same sequence;
7. determinism: same inputs -> same metrics.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from research.p3b.metrics import (
    ABSTENTION_CAUSES,
    METRIC_ACTION_ELIGIBILITY_RECALL,
    METRIC_ACTION_LEVEL_FALSE_APPLICATION_RATE,
    METRIC_ABSTENTION_RATE,
    METRIC_CALIBRATION_REPRESENTATIVENESS_FAILURE_RATE,
    METRIC_CANDIDATE_RECALL,
    METRIC_CORRECT_ABSTENTION_RATE,
    METRIC_FALSE_CANDIDATE_RATE,
    METRIC_GROUP_LEVEL_REPEATABILITY,
    METRIC_QUALIFICATION_RECALL,
    METRIC_REPRESENTATIVENESS_UNDETERMINED_RATE,
    METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE,
    METRIC_UNSAFE_ACTION_RATE,
    UNIVERSE_DATA,
    UNIVERSE_DECISION,
    build_outcomes,
    candidate_operating_points,
    compare_to_truth,
    compute_metrics,
    confusion_matrix,
    is_reconstruction_target,
    metrics_for_scenario,
)
from research.p3b.model import (
    AggregateSpec,
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SiteSpec,
    compute_bookkeeping,
)
from research.p3b.qualification_policy import (
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
    ACTION_NO_ACTION,
    ACTION_REQUALIFY,
    EPISTEMIC_CHARACTERISED_INTERMITTENT,
    EPISTEMIC_CHARACTERISED_STABLE,
    EPISTEMIC_CENSORED,
    EPISTEMIC_OBSERVED_CANDIDATE,
    EPISTEMIC_UNKNOWN,
    NET_BENEFIT_ESTABLISHED,
    NOT_REPRESENTATIVE,
    REPRESENTATIVE,
    UNDETERMINED,
    YES,
    EvidencePacket,
    QualificationDecision,
    evaluate,
)
from tests.p3b.conftest import make_sensor

METRICS_PATH = Path(__file__).resolve().parents[2] / "research" / "p3b" / "metrics.py"


# ---------------------------------------------------------------------------
# Helpers: hand-built outcomes and a "good" packet baseline
# ---------------------------------------------------------------------------


def make_packet(**overrides) -> EvidencePacket:
    base = dict(
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=YES,
        calibration_representativeness=REPRESENTATIVE,
        persisted_at_same_sensor_coord=YES,
        independent_group_count=2,
        epoch_count=2,
        censored_measurement_present="NO",
        site_residual_behaviour="VARIABLE",
        neighbourhood_residual_stable=YES,
        transient_only="NO",
        conflicting_evidence="NO",
        net_benefit_established=NET_BENEFIT_ESTABLISHED,
    )
    base.update(overrides)
    return EvidencePacket(**base)


def _outcome(
    site_id,
    expected_action=ACTION_NO_ACTION,
    expected_qualification="UNQUALIFIED",
    *,
    decision=None,
    applied=None,
    packet=None,
    censored=False,
    cfa_class="NORMAL",
):
    from research.p3b.metrics import SiteOutcome

    if decision is None:
        decision = evaluate(make_packet())
    return SiteOutcome(
        site_id=site_id,
        x=0,
        y=0,
        cfa_class=cfa_class,
        expected_qualification_state=expected_qualification,
        expected_action_state=expected_action,
        censored=censored,
        decision=decision,
        packet=packet,
        applied=applied,
    )


def _target_decision():
    """A decision whose action is ELIGIBLE (promoted)."""
    return evaluate(make_packet())  # fully good -> ELIGIBLE


def _abstain_decision():
    """A decision whose action is ABSTAIN_INSUFFICIENT (net benefit not established)."""
    return evaluate(make_packet(net_benefit_established="NOT_ESTABLISHED"))


# ---------------------------------------------------------------------------
# Requirement 1 — explicit units / universes; three FPR variants distinct
# ---------------------------------------------------------------------------


def test_every_metric_declares_universe_and_unit():
    from research.p3b.metrics import MetricValue

    # Build a small scenario so the report is populated.
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="STABLE_ANOMALY_WITH_MISMATCHED_DARK")
    scenario = _two_group_scenario(sites=(site,))
    report = metrics_for_scenario(scenario)
    for name, mv in report.metrics.items():
        assert isinstance(mv, MetricValue)
        assert mv.universe in (UNIVERSE_DATA, UNIVERSE_DECISION, "truth"), name
        assert isinstance(mv.unit, str) and mv.unit, name


def test_three_fpr_variants_are_distinct_and_non_addable():
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="INTERMITTENT_TWO_STATE")
    scenario = _two_group_scenario(sites=(site,))
    report = metrics_for_scenario(scenario, net_benefit_established=NET_BENEFIT_ESTABLISHED)

    promotion = report.metrics[METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE]
    candidate = report.metrics[METRIC_FALSE_CANDIDATE_RATE]
    application = report.metrics[METRIC_ACTION_LEVEL_FALSE_APPLICATION_RATE]

    # Distinct universes: promotion/application are DECISION-universe (per
    # site), candidate is DATA-universe (per frame) — never addable.
    assert promotion.universe == UNIVERSE_DECISION
    assert application.universe == UNIVERSE_DECISION
    assert candidate.universe == UNIVERSE_DATA
    assert promotion.universe != candidate.universe
    assert application.universe != candidate.universe

    # Distinct units even within the same universe.
    assert promotion.unit != application.unit

    # The three metric *names* are distinct and non-commutable.
    names = {promotion.name, candidate.name, application.name}
    assert len(names) == 3


def test_false_candidate_rate_and_promotion_do_not_share_denominator():
    # §30: "10 false candidates / million pixels" is NOT "10 sites faussement
    # promus / million de sites". The two metrics must expose different
    # denominators (frames vs negative sites), so they cannot be summed.
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="INTERMITTENT_TWO_STATE")
    scenario = _two_group_scenario(sites=(site,))
    report = metrics_for_scenario(scenario, net_benefit_established=NET_BENEFIT_ESTABLISHED)
    promotion = report.metrics[METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE]
    candidate = report.metrics[METRIC_FALSE_CANDIDATE_RATE]
    assert "site" in promotion.unit
    assert "frame" in candidate.unit


# ---------------------------------------------------------------------------
# Requirement 2 — known value: hand-computed on a hand-built set
# ---------------------------------------------------------------------------


def _hand_scenario(sites=(), groups=2, epochs=2, frames_per_group=3):
    sensor = make_sensor(shape=(16, 16))
    epoch_specs = []
    counter = 0
    for e in range(epochs):
        group_specs = []
        for g in range(groups):
            fr = tuple(
                FrameSpec(
                    frame_id=f"f{counter + i}", frame_type="light",
                    epoch_id=f"e{e}", group_id=f"g{e}_{g}", ordinal=i,
                )
                for i in range(frames_per_group)
            )
            counter += frames_per_group
            group_specs.append(GroupSpec(group_id=f"g{e}_{g}", frames=fr))
        epoch_specs.append(EpochSpec(epoch_id=f"e{e}", groups=tuple(group_specs)))
    return ScenarioSpec(
        name="hand", seed=0, sensor=sensor,
        epochs=tuple(epoch_specs), sites=tuple(sites),
    )


def _two_group_scenario(sites=()):
    return _hand_scenario(sites=sites, groups=2, epochs=2, frames_per_group=3)


def test_hand_computed_metrics():
    # A hand-built set with a KNOWN composition:
    #   t0: target, promoted   -> TP
    #   t1: target, abstained  -> FN (detected, correctly... no: a target that
    #                              abstains is a recall miss at eligibility)
    #   n0: non-target, promoted -> FP (false promotion)
    #   n1: non-target, abstained -> correct abstention
    #   n2: non-target, no_action -> true negative
    outcomes = (
        _outcome("t0", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_target_decision()),
        _outcome("t1", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_abstain_decision()),
        _outcome("n0", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
                 decision=_target_decision()),
        _outcome("n1", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
                 decision=_abstain_decision()),
        _outcome("n2", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
                 decision=evaluate(make_packet(site_residual_behaviour="NONE"))),
    )

    report = compute_metrics(outcomes, light_frame_count=6, megapixels_per_frame=1.0)

    # negatives = n0, n1, n2 (3). targets = t0, t1 (2).
    assert report.negative_count == 3
    assert report.target_count == 2

    # site_level_false_promotion_rate = false promotions / negatives = 1 / 3.
    assert report.metrics[METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE].value == pytest.approx(1 / 3)

    # confusion: TP=1 (t0), FN=1 (t1), FP=1 (n0), TN=2 (n1, n2).
    cm = report.confusion
    assert (cm.true_positive, cm.false_negative, cm.false_positive, cm.true_negative) == (1, 1, 1, 2)

    # action_eligibility_recall = targets promoted / targets = 1 / 2.
    assert report.metrics[METRIC_ACTION_ELIGIBILITY_RECALL].value == pytest.approx(1 / 2)

    # qualification_recall = targets qualified / targets. Both t0 and t1 are
    # characterised (CHARACTERISED_INTERMITTENT): t1 is *qualified* but abstains
    # at the ACTION stage (net benefit not established). That is exactly the
    # detect-then-abstain separation: qualification = 1.0, eligibility = 0.5.
    assert report.metrics[METRIC_QUALIFICATION_RECALL].value == pytest.approx(1.0)
    # candidate_recall = targets detected / targets = 1.0 (both observed).
    assert report.metrics[METRIC_CANDIDATE_RECALL].value == pytest.approx(1.0)


def test_hand_computed_unsafe_action_rate():
    # unsafe_action_rate = false corrective actions / negatives.
    # Two negatives; one actually reconstructed (applied=True).
    applied_negative = _outcome(
        "n0", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
        decision=_target_decision(), applied=True,
    )
    untouched_negative = _outcome(
        "n1", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
        decision=_abstain_decision(), applied=False,
    )
    outcomes = (applied_negative, untouched_negative)
    report = compute_metrics(outcomes, light_frame_count=6, megapixels_per_frame=1.0)

    # unsafe_action_rate = 1 / 2.
    assert report.metrics[METRIC_UNSAFE_ACTION_RATE].value == pytest.approx(0.5)
    # action_level_false_application_rate = false applications / negative promoted.
    # one negative promoted (n0), one applied (n0) -> 1 / 1.
    assert report.metrics[METRIC_ACTION_LEVEL_FALSE_APPLICATION_RATE].value == pytest.approx(1.0)


def test_false_candidate_rate_hand_computed():
    # false_candidate_rate = healthy-but-detected / frames.
    # n0 is UNQUALIFIED (healthy) but its decision is OBSERVED_CANDIDATE (detected).
    detected_healthy = _outcome(
        "n0", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
        decision=evaluate(make_packet(net_benefit_established="NOT_ESTABLISHED",
                                      site_residual_behaviour="INDETERMINATE")),
    )
    outcomes = (detected_healthy,)
    report = compute_metrics(outcomes, light_frame_count=4, megapixels_per_frame=1.0)
    assert report.metrics[METRIC_FALSE_CANDIDATE_RATE].value == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# Requirement 3 — abstention != failure
# ---------------------------------------------------------------------------


def test_detected_but_correctly_abstained_is_not_a_detection_error():
    # A *detected* non-target site that correctly abstains must count in
    # correct_abstention_rate and must NOT count as a false promotion (and the
    # detection itself is not a "detection error" on the recall side).
    detected_abstained = _outcome(
        "n0", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
        decision=_abstain_decision(),  # ABSTAIN_INSUFFICIENT -> abstained
    )
    # Make the site detected by using a packet whose epistemic is observed.
    outcomes = (detected_abstained,)
    report = compute_metrics(outcomes, light_frame_count=4, megapixels_per_frame=1.0)

    # correct_abstention_rate = correct abstentions / abstentions = 1 / 1.
    assert report.metrics[METRIC_CORRECT_ABSTENTION_RATE].value == pytest.approx(1.0)
    # It was NOT promoted -> false promotion count is 0.
    assert report.confusion.false_positive == 0
    assert report.metrics[METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE].value == 0.0


def test_abstention_rate_and_decomposition():
    outcomes = (
        _outcome("a", expected_action=ACTION_NO_ACTION, decision=_abstain_decision()),
        _outcome("b", expected_action=ACTION_NO_ACTION,
                 decision=evaluate(make_packet(net_benefit_established="NOT_ESTABLISHED"))),
    )
    report = compute_metrics(outcomes, light_frame_count=4, megapixels_per_frame=1.0)
    assert report.metrics[METRIC_ABSTENTION_RATE].value == pytest.approx(1.0)
    # Both abstain via NET_BENEFIT_NOT_ESTABLISHED.
    assert report.abstention_decomposition["NET_BENEFIT_NOT_ESTABLISHED"] == 2
    assert report.abstention_decomposition["CENSORED"] == 0


# ---------------------------------------------------------------------------
# Requirement 4 — censure invariant (0 quantitative inference)
# ---------------------------------------------------------------------------


def test_censored_mis_inference_count_is_zero():
    # A censored site whose decision is ABSTAIN_CENSORED (no characterised
    # residual, no eligible) -> mis-inference count 0.
    censored = _outcome(
        "c0", expected_action="ABSTAIN_CENSORED", expected_qualification="CENSORED",
        censored=True,
        decision=evaluate(make_packet(
            censored_measurement_present=YES,
            site_residual_behaviour="INDETERMINATE",
            net_benefit_established="NOT_ESTABLISHED",
        )),
    )
    report = compute_metrics((censored,), light_frame_count=4, megapixels_per_frame=1.0)
    assert report.censored_evaluation_count == 1
    assert report.censored_mis_inference_count == 0


def test_representativeness_rates_hand_computed():
    not_rep = _outcome(
        "n0", expected_action=ACTION_REQUALIFY, expected_qualification="QUALIFIED_PERSISTENT",
        decision=evaluate(make_packet(calibration_representativeness=NOT_REPRESENTATIVE)),
    )
    ok = _outcome("n1", expected_action=ACTION_NO_ACTION, decision=_abstain_decision())
    report = compute_metrics((not_rep, ok), light_frame_count=4, megapixels_per_frame=1.0)
    assert report.metrics[METRIC_CALIBRATION_REPRESENTATIVENESS_FAILURE_RATE].value == pytest.approx(0.5)
    assert report.metrics[METRIC_REPRESENTATIVENESS_UNDETERMINED_RATE].value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Requirement 5 — no threshold selection (AST + execution)
# ---------------------------------------------------------------------------


def test_metrics_module_selects_no_optimum_ast():
    tree = ast.parse(METRICS_PATH.read_text())
    forbidden_calls = {"argmax", "argmin"}
    forbidden_names = {"max", "min", "sorted", "argmax", "argmin"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                raise AssertionError(f"metrics.py selects an optimum via .{node.func.attr}")
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                raise AssertionError(f"metrics.py selects via {node.func.id}(...)")
        if isinstance(node, ast.Attribute) and node.attr in forbidden_calls:
            raise AssertionError(f"metrics.py references .{node.attr}")


def test_operating_points_are_described_not_selected():
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="INTERMITTENT_TWO_STATE")
    scenario = _two_group_scenario(sites=(site,))
    points = candidate_operating_points(scenario)
    # Every point is a description; none is flagged "best"/"selected".
    assert len(points) == 3
    for p in points:
        assert p.description.startswith("net_benefit_established=")
        assert "best" not in p.description.lower()
    # No single point is returned as "the" operating point.
    assert not hasattr(points, "selected")
    assert len({p.net_benefit_setting for p in points}) == 3


def test_metrics_module_has_no_untagged_numeric_literals():
    tree = ast.parse(METRICS_PATH.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)):
                if sub.value not in (0, 1, 2):
                    offenders.append((node.name, sub.value, sub.lineno))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# Requirement 6 — independence recurrence (20 frames != 20 groups)
# ---------------------------------------------------------------------------


def _scenario_with(group_frames, groups, epochs=2):
    sensor = make_sensor(shape=(16, 16))
    epoch_specs = []
    counter = 0
    for e in range(epochs):
        gs = []
        for g in range(groups):
            fr = tuple(
                FrameSpec(frame_id=f"f{counter + i}", frame_type="light",
                          epoch_id=f"e{e}", group_id=f"g{e}_{g}", ordinal=i)
                for i in range(group_frames)
            )
            counter += group_frames
            gs.append(GroupSpec(group_id=f"g{e}_{g}", frames=fr))
        epoch_specs.append(EpochSpec(epoch_id=f"e{e}", groups=tuple(gs)))
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="INTERMITTENT_TWO_STATE")
    return ScenarioSpec(name="rec", seed=0, sensor=sensor, epochs=tuple(epoch_specs), sites=(site,))


def test_group_level_repeatability_not_improved_by_more_frames_same_group():
    # 2 epochs x 2 groups x 3 frames: qualified (2 groups, 2 epochs).
    few = metrics_for_scenario(
        _scenario_with(group_frames=3, groups=2), net_benefit_established=NET_BENEFIT_ESTABLISHED
    )
    # Same groups but 30 frames per group: independence is UNCHANGED -> same metric.
    many = metrics_for_scenario(
        _scenario_with(group_frames=30, groups=2), net_benefit_established=NET_BENEFIT_ESTABLISHED
    )
    assert few.metrics[METRIC_GROUP_LEVEL_REPEATABILITY].value == \
        many.metrics[METRIC_GROUP_LEVEL_REPEATABILITY].value


def test_group_level_repeatability_improves_only_with_more_groups():
    # 2 epochs x 1 group = 2 independent groups (the minimum). Removing one
    # group leaves 1 -> qualification breaks -> repeatability 0.
    two = metrics_for_scenario(
        _scenario_with(group_frames=3, groups=1, epochs=2),
        net_benefit_established=NET_BENEFIT_ESTABLISHED,
    )
    # 3 epochs x 1 group = 3 independent groups. Removing one leaves 2 ->
    # qualification reproduces -> repeatability 1.0.
    three = metrics_for_scenario(
        _scenario_with(group_frames=3, groups=1, epochs=3),
        net_benefit_established=NET_BENEFIT_ESTABLISHED,
    )
    assert two.metrics[METRIC_GROUP_LEVEL_REPEATABILITY].value == 0.0
    assert three.metrics[METRIC_GROUP_LEVEL_REPEATABILITY].value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Requirement 7 — determinism
# ---------------------------------------------------------------------------


def test_same_inputs_same_metrics():
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="INTERMITTENT_TWO_STATE")
    scenario = _two_group_scenario(sites=(site,))
    r1 = metrics_for_scenario(scenario, net_benefit_established=NET_BENEFIT_ESTABLISHED)
    r2 = metrics_for_scenario(scenario, net_benefit_established=NET_BENEFIT_ESTABLISHED)
    assert r1 == r2
    assert r1.metrics == r2.metrics


# ---------------------------------------------------------------------------
# Rework-1 (F1): the light-frame denominator must never be the LOT1 frame_count
# ---------------------------------------------------------------------------


def test_compute_metrics_has_no_frame_count_parameter():
    # The trap is a *naming* trap: a parameter named ``frame_count`` collides
    # with LOT1 ``compute_bookkeeping().frame_count`` (raw frames + aggregates,
    # including darks/bias/masters), which is NOT the data-universe denominator.
    # The renamed API makes the wrong counter impossible to pass.
    params = list(inspect.signature(compute_metrics).parameters)
    assert "light_frame_count" in params
    assert "frame_count" not in params


def test_metrics_for_scenario_derives_light_frame_denominator():
    # A scenario with 4 light frames + 5 darks + 1 master: LOT1 frame_count
    # (10) differs from observation_count (4). metrics_for_scenario must derive
    # the denominator from observation_count, so false_candidate_rate uses 4.
    sensor = make_sensor(shape=(16, 16))
    darks = tuple(
        FrameSpec(frame_id=f"d{i}", frame_type="dark", epoch_id="e0", group_id="gd", ordinal=i)
        for i in range(5)
    )
    lights = tuple(
        FrameSpec(frame_id=f"l{i}", frame_type="light", epoch_id=f"e{i // 2}",
                  group_id=f"g{i}", ordinal=0)
        for i in range(4)
    )
    scenario = ScenarioSpec(
        name="trap", seed=0, sensor=sensor,
        epochs=(
            EpochSpec("e0", (GroupSpec("gd", darks), GroupSpec("g0", (lights[0],)), GroupSpec("g1", (lights[1],)))),
            EpochSpec("e1", (GroupSpec("g2", (lights[2],)), GroupSpec("g3", (lights[3],)))),
        ),
        aggregates=(AggregateSpec(
            aggregate_id="master", method="median",
            constituent_frame_ids=tuple(f.frame_id for f in darks),
        ),),
        sites=(SiteSpec(site_id="s0", x=8, y=8, cfa_class="NOISE_EXTREME"),),
    )
    bk = compute_bookkeeping(scenario)
    assert bk.observation_count == 4
    assert bk.frame_count == 10  # 4 lights + 5 darks + 1 master
    assert bk.frame_count != bk.observation_count

    # NOISE_EXTREME is UNQUALIFIED (healthy) but detected -> exactly one false
    # candidate. The correct denominator is observation_count (4), never
    # frame_count (10).
    report = metrics_for_scenario(scenario)
    rate = report.metrics[METRIC_FALSE_CANDIDATE_RATE].value
    assert rate == pytest.approx(1 / 4)
    # The trap, rendered visible: had the denominator been frame_count, the
    # value would have been 1 / 10, a different (wrong) universe.
    assert rate != pytest.approx(1 / 10)


def test_compute_metrics_denominator_changes_the_rate():
    # The same numerator with two different light-frame denominators yields two
    # different rates — the caller-supplied denominator *is* the data universe.
    detected_healthy = _outcome(
        "n0", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
        decision=evaluate(make_packet(net_benefit_established="NOT_ESTABLISHED",
                                      site_residual_behaviour="INDETERMINATE")),
    )
    small = compute_metrics((detected_healthy,), light_frame_count=2, megapixels_per_frame=1.0)
    large = compute_metrics((detected_healthy,), light_frame_count=4, megapixels_per_frame=1.0)
    assert small.metrics[METRIC_FALSE_CANDIDATE_RATE].value == pytest.approx(1 / 2)
    assert large.metrics[METRIC_FALSE_CANDIDATE_RATE].value == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# Cross-check (descriptive)
# ---------------------------------------------------------------------------


def test_compare_to_truth_reports_deviations_descriptively():
    outcomes = (
        _outcome("t0", expected_action=ACTION_ELIGIBLE, decision=_target_decision()),
        _outcome("n0", expected_action=ACTION_NO_ACTION, decision=_target_decision()),
    )
    rows = compare_to_truth(outcomes)
    by_id = {r.site_id: r for r in rows}
    assert by_id["t0"].action_agrees is True
    assert by_id["n0"].action_agrees is False


def test_censored_site_is_never_established_in_sweep():
    # The operating-point sweep must not establish net benefit on a censored site.
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="CENSORED_ANOMALY", censored=True)
    scenario = _two_group_scenario(sites=(site,))
    for p in candidate_operating_points(scenario):
        assert p.confusion.false_positive == 0


# ---------------------------------------------------------------------------
# abstained_target_count (ZC-SENSOR-P3B1-CLOSURE, part B)
# ---------------------------------------------------------------------------
#
# A raw counter (not a metric, not a score) beside ``target_count``: the number
# of reconstruction targets (ground truth) whose policy result is an ABSTAIN_*
# action. A target that was *detected* but abstains is counted here; a healthy
# negative that abstains is NOT (it is not a reconstruction target).


def test_abstained_target_count_zero_with_no_targets():
    outcomes = (
        _outcome("n0", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
                 decision=_abstain_decision()),
    )
    report = compute_metrics(outcomes, light_frame_count=4, megapixels_per_frame=1.0)
    assert report.target_count == 0
    assert report.abstained_target_count == 0


def test_abstained_target_count_zero_with_no_abstained_targets():
    # One target, promoted (not abstained) -> abstained_target_count == 0.
    outcomes = (
        _outcome("t0", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_target_decision()),
    )
    report = compute_metrics(outcomes, light_frame_count=4, megapixels_per_frame=1.0)
    assert report.target_count == 1
    assert report.abstained_target_count == 0


def test_abstained_target_count_hand_computed_four_targets():
    # 4 targets: 1 eligible, 3 abstained -> abstained_target_count == 3.
    outcomes = (
        _outcome("t0", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_target_decision()),
        _outcome("t1", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_abstain_decision()),
        _outcome("t2", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_abstain_decision()),
        _outcome("t3", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_abstain_decision()),
    )
    report = compute_metrics(outcomes, light_frame_count=8, megapixels_per_frame=1.0)
    assert report.target_count == 4
    assert report.abstained_target_count == 3


def test_detected_but_abstained_target_is_counted():
    # A target that IS detected (epistemic != UNKNOWN) but abstains is still a
    # reconstruction target whose policy result is ABSTAIN_* -> counted here.
    detected_abstained_target = _outcome(
        "t0", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
        decision=evaluate(make_packet(net_benefit_established="NOT_ESTABLISHED")),
    )
    # Sanity: the decision is detected (CHARACTERISED_INTERMITTENT) but abstained.
    assert detected_abstained_target.decision.epistemic_state == EPISTEMIC_CHARACTERISED_INTERMITTENT
    assert detected_abstained_target.decision.action == ACTION_ABSTAIN_INSUFFICIENT

    report = compute_metrics((detected_abstained_target,), light_frame_count=4, megapixels_per_frame=1.0)
    assert report.target_count == 1
    assert report.abstained_target_count == 1


def test_healthy_negative_abstained_is_not_counted():
    # A healthy (non-target) site that abstains must NOT be counted: it is not a
    # reconstruction target, so abstained_target_count stays 0.
    healthy_abstained = _outcome(
        "n0", expected_action=ACTION_NO_ACTION, expected_qualification="UNQUALIFIED",
        decision=_abstain_decision(),
    )
    report = compute_metrics((healthy_abstained,), light_frame_count=4, megapixels_per_frame=1.0)
    assert report.target_count == 0
    assert report.abstained_target_count == 0


def test_abstained_target_count_does_not_replace_existing_recalls():
    # The new counter is additive: the existing recall metrics and target_count
    # are unchanged by its presence.
    outcomes = (
        _outcome("t0", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_target_decision()),
        _outcome("t1", expected_action=ACTION_ELIGIBLE, expected_qualification="QUALIFIED_INTERMITTENT",
                 decision=_abstain_decision()),
    )
    report = compute_metrics(outcomes, light_frame_count=6, megapixels_per_frame=1.0)
    assert report.target_count == 2
    assert report.abstained_target_count == 1
    # Existing recalls unchanged: eligibility recall = 1/2 (only t0 promoted).
    assert report.metrics[METRIC_ACTION_ELIGIBILITY_RECALL].value == pytest.approx(1 / 2)
    # The counter is not present as a named metric (it is a raw field, no score).
    assert "abstained_target_count" not in report.metrics
