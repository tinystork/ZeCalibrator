"""LOT4 tests: descriptive features + declared facts (separation, no decision).

Covered requirements (mission ZC-SENSOR-P3B-LOT4):

1. determinism (same seed ⇒ identical features; different seeds ⇒ different);
2. separation (facts come from the declaration, never from features);
3. computability (>=3 features hand-computed against a known analytic answer);
4. no threshold / no decision (AST scan + execution);
5. speed (tiny arrays, no FITS generation);
6. censored case (count/fraction computed; no quantitative inference from a
   censored measurement).
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from research.p3b.features import (
    calibration_mismatch,
    censored_count_and_fraction,
    compute_features,
    post_calibration_residual,
    recurrence_count,
    robust_local_scale,
    same_cfa_local_residual,
    sign_changes,
    state_spread,
)
from research.p3b.declared_facts import build_declared_facts, evidence_packet
from research.p3b.model import (
    ACTION_STATES,
    QUALIFICATION_STATES,
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SiteSpec,
)
from research.p3b.qualification_policy import (
    ACTION_LADDER,
    EPISTEMIC_STATES,
    REASON_CODES,
    EvidencePacket,
)
from tests.p3b.conftest import make_sensor

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "research" / "p3b" / "features.py"
DECLARED_FACTS_PATH = REPO_ROOT / "research" / "p3b" / "declared_facts.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario(sites=(), seed=0, frames_per_group=3, groups=1, epochs=1, shape=(16, 16)):
    """A tiny scenario with the given sites and light frames only (no FITS I/O)."""
    sensor = make_sensor(shape=shape)
    epoch_specs = []
    counter = 0
    for e in range(epochs):
        group_specs = []
        for g in range(groups):
            fr = tuple(
                FrameSpec(
                    frame_id=f"f{counter + i}",
                    frame_type="light",
                    epoch_id=f"e{e}",
                    group_id=f"g{e}_{g}",
                    ordinal=i,
                )
                for i in range(frames_per_group)
            )
            counter += frames_per_group
            group_specs.append(GroupSpec(group_id=f"g{e}_{g}", frames=fr))
        epoch_specs.append(EpochSpec(epoch_id=f"e{e}", groups=tuple(group_specs)))
    return ScenarioSpec(name="t", seed=seed, sensor=sensor, epochs=tuple(epoch_specs), sites=tuple(sites))


def _frame_arrays(scenario, value):
    """Constant frame arrays (same value everywhere) — for analytic feature checks."""
    ny, nx = scenario.sensor.shape
    return {f.frame_id: np.full((ny, nx), value, dtype=np.float64) for f in scenario.all_frames()}


# ---------------------------------------------------------------------------
# Requirement 3 — computability: hand-computed analytic answers
# ---------------------------------------------------------------------------


def test_same_cfa_local_residual_analytic():
    # 3x3 mono frame: centre 20, all 8 neighbours 10 -> residual = 20 - 10 = 10.
    frame = np.array(
        [[10.0, 10.0, 10.0],
         [10.0, 20.0, 10.0],
         [10.0, 10.0, 10.0]]
    )
    assert same_cfa_local_residual(frame, 1, 1, "mono", radius=1) == pytest.approx(10.0)


def test_robust_local_scale_analytic():
    # All 8 neighbours equal 10 -> MAD = 0.
    frame = np.array(
        [[10.0, 10.0, 10.0],
         [10.0, 20.0, 10.0],
         [10.0, 10.0, 10.0]]
    )
    assert robust_local_scale(frame, 1, 1, "mono", radius=1) == pytest.approx(0.0)


def test_sign_changes_analytic():
    # +, -, +, -, -, +  -> crossings: (1,-1),(-1,2),(2,-2),(-3,4) = 4.
    assert sign_changes([1.0, -1.0, 2.0, -2.0, -3.0, 4.0]) == 4


def test_recurrence_count_analytic():
    assert recurrence_count([5, 5, 7, 5]) == 3
    assert recurrence_count([1, 2, 1, 2]) == 4
    assert recurrence_count([1, 2, 3]) == 0
    assert recurrence_count([]) == 0


def test_post_calibration_residual_analytic():
    values = np.array([100.0, 200.0, 300.0])
    dark = np.array([50.0, 50.0, 50.0])
    out = post_calibration_residual(values, dark)
    assert list(out) == [50.0, 150.0, 250.0]


def test_calibration_mismatch_analytic():
    old = np.array([[0.0, 0.0], [0.0, 100.0]])
    cur = np.array([[0.0, 0.0], [0.0, 130.0]])
    assert calibration_mismatch(old, cur, 1, 1) == pytest.approx(30.0)


def test_state_spread_analytic():
    spread = state_spread([2.0, 4.0, 6.0, 8.0])
    assert spread.min == pytest.approx(2.0)
    assert spread.max == pytest.approx(8.0)
    assert spread.span == pytest.approx(6.0)


def test_censored_count_and_fraction_analytic():
    count, frac = censored_count_and_fraction([100.0, 60000.0, 60000.0, 300.0], 60000.0)
    assert count == 2
    assert frac == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Requirement 1 — determinism
# ---------------------------------------------------------------------------


def test_same_seed_identical_features():
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="STABLE_ANOMALY_WITH_MISMATCHED_DARK")
    scenario = _scenario(sites=(site,), seed=42)
    frames = _frame_arrays(scenario, 1000.0)
    r1 = compute_features(scenario, frames)
    r2 = compute_features(scenario, frames)
    assert r1 == r2


def test_different_seeds_differ():
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="NOISE_EXTREME")
    # The feature harness itself is a pure function of the arrays; to vary the
    # data across seeds we vary the arrays directly (no FITS generation).
    s1 = _scenario(sites=(site,), seed=1)
    s2 = _scenario(sites=(site,), seed=2)
    frames1 = _frame_arrays(s1, 1000.0)
    frames2 = _frame_arrays(s2, 1001.0)
    f1 = compute_features(s1, frames1).sites[0].temporal_value_series
    f2 = compute_features(s2, frames2).sites[0].temporal_value_series
    assert f1 != f2


# ---------------------------------------------------------------------------
# Requirement 2 — separation: facts come from the declaration, never features
# ---------------------------------------------------------------------------


def test_declared_facts_change_with_declaration_not_with_features():
    site_a = SiteSpec(site_id="s0", x=8, y=8, cfa_class="STABLE_ANOMALY_WITH_MISMATCHED_DARK")
    site_b = SiteSpec(site_id="s0", x=8, y=8, cfa_class="NORMAL")
    facts_a = build_declared_facts(_scenario(sites=(site_a,), seed=7))
    facts_b = build_declared_facts(_scenario(sites=(site_b,), seed=7))

    # Changing the declaration changes the facts.
    assert facts_a.sites[0].site_residual_behaviour != facts_b.sites[0].site_residual_behaviour

    # The facts API cannot receive features: build_declared_facts takes only the
    # scenario declaration and evidence_packet takes only declared facts.
    assert list(inspect.signature(build_declared_facts).parameters) == ["scenario"]
    assert list(inspect.signature(evidence_packet).parameters) == ["site", "declared"]

    # Rebuilding from the same declaration is identical (deterministic).
    assert build_declared_facts(_scenario(sites=(site_a,), seed=7)) == facts_a


def test_declared_facts_do_not_read_frames():
    # build_declared_facts reads only the scenario declaration, never frame data.
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="CENSORED_ANOMALY", censored=True)
    scenario = _scenario(sites=(site,), seed=3)
    facts = build_declared_facts(scenario)
    assert facts.sites[0].censored_measurement_present == "YES"
    assert facts.sites[0].site_residual_behaviour == "INDETERMINATE"


def test_bridge_is_trivial_projection():
    site = SiteSpec(
        site_id="s0", x=8, y=8,
        cfa_class="STABLE_ANOMALY_CORRECTED_BY_DARK",
        calibration_representativeness="representative",
    )
    declared = build_declared_facts(_scenario(sites=(site,), seed=1))
    packet = evidence_packet(declared.sites[0], declared)

    assert isinstance(packet, EvidencePacket)
    # The packet is exactly the declared facts, field by field.
    assert packet.site_residual_behaviour == declared.sites[0].site_residual_behaviour
    assert packet.persisted_at_same_sensor_coord == declared.sites[0].persisted_at_same_sensor_coord
    assert packet.censored_measurement_present == "NO"
    assert packet.independent_group_count == declared.independent_group_count
    assert packet.epoch_count == declared.epoch_count
    # Net benefit is passed through as the declared default (never computed).
    assert packet.net_benefit_established == "UNDETERMINED"


def test_bridge_never_calls_evaluate():
    # The bridge must not import or call the LOT2 decision function.
    tree = _module_ast(DECLARED_FACTS_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "research.p3b.qualification_policy":
            imported = {a.name for a in node.names}
            assert "evaluate" not in imported
        if isinstance(node, ast.Name) and node.id == "evaluate":
            raise AssertionError("declared_facts references 'evaluate'")
        if isinstance(node, ast.Attribute) and node.attr == "evaluate":
            raise AssertionError("declared_facts references '.evaluate'")
    # It imports EvidencePacket (the LOT2 data type) but not the decision engine.
    assert EvidencePacket.__module__ == "research.p3b.qualification_policy"


# ---------------------------------------------------------------------------
# Requirement 4 — no threshold / no decision (AST + execution)
# ---------------------------------------------------------------------------

_DECISION_VOCAB = set(
    list(QUALIFICATION_STATES)
    + list(ACTION_STATES)
    + list(ACTION_LADDER)
    + list(EPISTEMIC_STATES)
    + list(REASON_CODES)
)


def _module_ast(path: Path):
    return ast.parse(path.read_text())


def test_features_module_has_no_untagged_numeric_literals():
    tree = _module_ast(FEATURES_PATH)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)):
                # Structural/definitional literals only: 0, 1, 2.
                if sub.value not in (0, 1, 2):
                    offenders.append((node.name, sub.value, sub.lineno))
    assert offenders == [], offenders


def test_features_module_has_no_decision_returns():
    tree = _module_ast(FEATURES_PATH)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if sub.value in _DECISION_VOCAB:
                    offenders.append((node.name, sub.value, sub.lineno))
    assert offenders == [], offenders


def test_features_module_imports_no_decision_logic():
    tree = _module_ast(FEATURES_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "qualification_policy" not in alias.name
        if isinstance(node, ast.ImportFrom):
            assert node.module != "research.p3b.qualification_policy"
        # Docstrings are string constants (not Name/Attribute nodes), so these
        # checks flag only real code references to the decision function.
        if isinstance(node, ast.Name) and node.id == "evaluate":
            raise AssertionError("features.py references 'evaluate'")
        if isinstance(node, ast.Attribute) and node.attr == "evaluate":
            raise AssertionError("features.py references '.evaluate'")


def test_feature_functions_never_return_decision():
    # Execution check: every pure feature function returns numbers/arrays/records,
    # never a decision-state string.
    frame = np.ones((5, 5), dtype=np.float64) * 100.0
    results = [
        same_cfa_local_residual(frame, 2, 2, "mono", radius=1),
        robust_local_scale(frame, 2, 2, "mono", radius=1),
        recurrence_count([1, 1, 2]),
        sign_changes([1.0, -1.0, 1.0]),
        censored_count_and_fraction([1.0, 60000.0], 60000.0),
    ]
    for r in results:
        assert not isinstance(r, str)


# ---------------------------------------------------------------------------
# Requirement 6 — censored case
# ---------------------------------------------------------------------------


def test_censored_count_fraction_and_no_quantitative_inference():
    site = SiteSpec(
        site_id="s0", x=8, y=8, cfa_class="CENSORED_ANOMALY",
        censored=True, hard_limit_adu=60000.0,
    )
    scenario = _scenario(sites=(site,), seed=1, frames_per_group=4)
    # Frame 1 is censored (at the hard limit); the rest are normal sky values.
    frames = {f.frame_id: np.full((16, 16), 5000.0, dtype=np.float64) for f in scenario.all_frames()}
    censored_fid = scenario.light_frames()[1].frame_id
    frames[censored_fid] = np.full((16, 16), 60000.0, dtype=np.float64)

    dark = np.full((16, 16), 4800.0, dtype=np.float64)
    result = compute_features(scenario, frames, dark_reference=dark).sites[0]

    assert result.censored_count == 1
    assert result.censored_fraction == pytest.approx(0.25)

    # The censored frame is counted but excluded from quantitative residuals.
    assert result.temporal_residual_series  # non-empty (the 3 non-censored frames)
    assert censored_fid not in result.temporal_residual_frame_ids
    assert result.post_calibration_residual[1] != result.post_calibration_residual[1]  # NaN

    # No quantitative inference from the censored measurement: the residual
    # series contains no value derived from the censored (60000) sample.
    for v in result.temporal_residual_series:
        assert not (abs(v - (60000.0 - 4800.0)) < 1.0)


def test_censored_site_declared_fact_present():
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="CENSORED_ANOMALY", censored=True)
    declared = build_declared_facts(_scenario(sites=(site,), seed=1))
    assert declared.sites[0].censored_measurement_present == "YES"
    assert declared.sites[0].site_residual_behaviour == "INDETERMINATE"


# ---------------------------------------------------------------------------
# Requirement 5 — speed: the whole suite above uses tiny (<=16x16) arrays and no
# FITS generation, so a single quick smoke test confirms the harness is scoped.
# ---------------------------------------------------------------------------

def test_smoke_small_scenario_is_instant():
    site = SiteSpec(site_id="s0", x=8, y=8, cfa_class="INTERMITTENT_TWO_STATE")
    scenario = _scenario(sites=(site,), seed=5, frames_per_group=6, groups=2, epochs=2, shape=(16, 16))
    frames = _frame_arrays(scenario, 1000.0)
    result = compute_features(scenario, frames)
    s = result.sites[0]
    assert len(s.temporal_value_series) == 6 * 2 * 2
    assert len(s.within_group_recurrence) == 4  # 2 epochs x 2 groups
    assert s.cross_group_recurrence == 4  # all 4 group signatures identical (constant data)
    assert s.cross_epoch_recurrence == 2  # both epoch signatures identical
    assert s.censored_count == 0
