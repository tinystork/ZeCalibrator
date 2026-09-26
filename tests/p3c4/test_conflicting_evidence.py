"""P3C-4 — corrected ``conflicting_evidence`` closure tests.

These tests pin the corrected, versioned ``conflicting_evidence`` semantics that
fix the QUALIFICATION-2 instrumentation artifact (65 ``YES``-where-``NO`` cells
on the frozen P3C rule). They cover:

* the corrected rule is versioned and truth-free (structurally + at runtime);
* the frozen ``p3c-*`` and ``p3c4-*`` candidates are **not** modified — the
  corrected candidates carry new ``*-cefix`` ids and their own provenance;
* gate §19:
    NORMAL                     -> not conflicting (noise crossing zero)
    INTERMITTENT_* (qualified) -> not conflicting (state change only)
    SIGN_CHANGING_POST_DARK    -> still able to express a REAL conflict
    CENSORED                   -> still abstention
    single transient           -> still transient / abstention
    dark mismatch              -> still REQUALIFY
* the ``conflicting_evidence`` fact matrix is corrected (65 -> 5 residual cells);
* ``qualification_recall`` is no longer 0.0 by instrumentation artifact.
"""

from __future__ import annotations

import ast
import importlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from research.p3c.development_corpus import build_admission, build_development_scenario
from research.p3c.inference_candidates import (
    CANDIDATE_BASELINE,
    CANDIDATE_CONSERVATIVE,
    CANDIDATE_SENSITIVE,
    CANDIDATES as P3C_CANDIDATES,
)
from research.p3c.inference_contract import NO, YES
from research.p3c.oracle import sensor_evidence_truth
from research.p3c4.conflicting_evidence import (
    CONFLICT_CONTRACT_VERSION,
    assess_conflicting_evidence,
    significant_sign_reversals,
)
from research.p3c4.conflicting_inference import (
    CANDIDATE_CEFIX_BASELINE,
    CONFLICTING_CANDIDATES,
    CONFLICTING_CANDIDATE_IDS,
    ConflictingCandidateConfig,
    conflicting_candidate,
    infer_conflicting,
    infer_site,
)
from research.p3c4.temporal_inference import TEMPORAL_CANDIDATES

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def closure_campaign():
    """Run the full corrected campaign once (session-scoped)."""
    from research.p3c4.conflicting_runner import run_conflicting_campaign

    return run_conflicting_campaign()


# ---------------------------------------------------------------------------
# Corrected rule: versioned, bounded, pure, deterministic
# ---------------------------------------------------------------------------


def test_rule_is_versioned():
    assert CONFLICT_CONTRACT_VERSION == "p3c4-conflicting-evidence-contract-1"


def test_rule_is_a_pure_function_of_residuals_and_params():
    import inspect

    sig = inspect.signature(assess_conflicting_evidence)
    assert set(sig.parameters) == {
        "residuals",
        "conflict_amplitude_floor",
        "min_conflict_reversals",
    }


def test_significant_reversal_ignores_zero_crossing_noise():
    # A residual that merely crosses zero around noise (OFF state / background)
    # contributes NO reversal: one side of each crossing sits inside the floor.
    residuals = [2500.0, -3.0, 2500.0, 5.0, 2500.0, -7.0]
    assert significant_sign_reversals(residuals, conflict_amplitude_floor=50.0) == 0


def test_significant_reversal_counts_genuine_sign_flip():
    residuals = [1500.0, -1500.0, 1500.0, -1500.0]
    assert significant_sign_reversals(residuals, conflict_amplitude_floor=50.0) == 3


def test_significant_reversal_drops_censored_nan():
    # A censored (NaN) frame is dropped, never treated as zero: it neither
    # creates a spurious reversal nor hides a genuine one.
    r = assess_conflicting_evidence(
        [1500.0, float("nan"), -1500.0],
        conflict_amplitude_floor=50.0,
        min_conflict_reversals=2,
    )
    assert r.valid_samples == 2
    assert r.sign_reversals == 1  # 1500 -> -1500 across the dropped NaN
    # A NaN between same-sign residuals creates no reversal at all.
    assert (
        significant_sign_reversals(
            [1500.0, float("nan"), 1500.0], conflict_amplitude_floor=50.0
        )
        == 0
    )


def test_assess_conflicting_evidence_is_binary_and_deterministic():
    a = assess_conflicting_evidence(
        [1500.0, -1500.0, 1500.0], conflict_amplitude_floor=50.0, min_conflict_reversals=2
    )
    b = assess_conflicting_evidence(
        [1500.0, -1500.0, 1500.0], conflict_amplitude_floor=50.0, min_conflict_reversals=2
    )
    assert a == b
    assert a.state == YES
    assert a.sign_reversals == 2
    assert a.valid_samples == 3

    absent = assess_conflicting_evidence(
        [9.0, -11.0, 3.0], conflict_amplitude_floor=50.0, min_conflict_reversals=2
    )
    assert absent.state == NO
    assert absent.sign_reversals == 0


def test_conflict_below_floor_is_documented_false_negative():
    # The detection floor is a documented blind band, pinned here (F1 rework-1):
    # a genuine *low-amplitude* sign reversal BELOW conflict_amplitude_floor is
    # NOT detected (state=NO, 0 reversals) — the rule does not claim the absence
    # of a conflict, it simply cannot see one under the floor. Just above the
    # floor the same reversal IS detected. This pins the boundary, it does not
    # hide the window.
    below = assess_conflicting_evidence(
        [45.0, -45.0, 45.0, -45.0], conflict_amplitude_floor=50.0, min_conflict_reversals=2
    )
    assert below.state == NO
    assert below.sign_reversals == 0  # ±45 is a real signal, not read noise — yet undetected

    above = assess_conflicting_evidence(
        [55.0, -55.0, 55.0, -55.0], conflict_amplitude_floor=50.0, min_conflict_reversals=2
    )
    assert above.state == YES
    assert above.sign_reversals == 3

    # The floor sits well above the corpus read noise (~4-5 ADU): a genuine
    # low-amplitude systematic reversal (e.g. ±40 ADU) falls in the blind band
    # [noise, floor), which is exactly the non-conservative window this test
    # documents. Removing it requires a relative floor (k x local sigma) — a
    # product-lot decision, not this lot.
    blind = assess_conflicting_evidence(
        [40.0, -40.0, 40.0, -40.0], conflict_amplitude_floor=50.0, min_conflict_reversals=2
    )
    assert blind.state == NO
    assert blind.sign_reversals == 0


# ---------------------------------------------------------------------------
# Corrected candidate family: versioned, bounded, marked research parameters
# ---------------------------------------------------------------------------


def test_corrected_family_is_bounded_and_versioned():
    assert 1 <= len(CONFLICTING_CANDIDATES) <= 6
    assert len(set(CONFLICTING_CANDIDATE_IDS)) == len(CONFLICTING_CANDIDATE_IDS)
    for c in CONFLICTING_CANDIDATES:
        assert c.candidate_id.startswith("p3c4-")
        assert c.candidate_id.endswith("-cefix")
        assert c.version
        assert c.description
        assert c.base_temporal_candidate_id


def test_every_conflict_parameter_is_marked_research():
    for c in CONFLICTING_CANDIDATES:
        for p in c.parameters:
            assert p.kind == "RESEARCH_CANDIDATE_PARAMETER"
            assert "PRODUCT" not in p.kind
            assert p.unit
            assert p.rationale


def test_corrected_candidates_reference_frozen_temporal_bases():
    base_ids = {c.candidate_id for c in TEMPORAL_CANDIDATES}
    for c in CONFLICTING_CANDIDATES:
        assert c.base_temporal_candidate_id in base_ids


def test_frozen_p3c_and_p3c4_candidates_are_unchanged():
    # The corrected semantics must never edit the frozen candidates: the p3c
    # family and the p3c4 temporal family keep their exact ids and parameter
    # sets; the corrected candidates are a separate, new family.
    assert {c.candidate_id for c in P3C_CANDIDATES} == {
        "p3c-baseline",
        "p3c-conservative",
        "p3c-sensitive",
    }
    assert {c.candidate_id for c in TEMPORAL_CANDIDATES} == {
        "p3c4-baseline",
        "p3c4-conservative",
        "p3c4-sensitive",
    }
    # The frozen p3c conflicting rule still uses raw sign crossings (unchanged).
    assert CANDIDATE_BASELINE.param("conflict_sign_changes") == 2.0
    assert CANDIDATE_CONSERVATIVE.param("conflict_sign_changes") == 2.0
    assert CANDIDATE_SENSITIVE.param("conflict_sign_changes") == 1.0


# ---------------------------------------------------------------------------
# Truth-leak guard for the corrected modules (contract + inference sides)
# ---------------------------------------------------------------------------

_CORRECTED_MODULES = (
    "research.p3c4.conflicting_evidence",
    "research.p3c4.conflicting_inference",
)

_FORBIDDEN_MODULES = (
    "research.p3b.declared_facts",
    "research.p3b.catalog",
    "research.p3b.fixtures",
    "research.p3b.metrics",
    "research.p3c.oracle",
)

_FORBIDDEN_SYMBOLS = (
    "_CLASS_FACTS",
    "cfa_class",
    "expected_",
    "scenario_name",
    "compare_to_truth",
)


@pytest.mark.parametrize("module", _CORRECTED_MODULES)
def test_corrected_modules_do_not_import_truth(module):
    mod = importlib.import_module(module)
    tree = ast.parse(Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            assert not any(
                name == f or name.startswith(f + ".") for f in _FORBIDDEN_MODULES
            ), f"{module} imports truth module {name!r}"


@pytest.mark.parametrize("module", _CORRECTED_MODULES)
def test_corrected_modules_have_no_truth_symbols(module):
    mod = importlib.import_module(module)
    tree = ast.parse(Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            tok = getattr(node, "id", None) or getattr(node, "attr", None)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            tok = node.name
        else:
            continue
        if tok:
            assert not any(s in tok for s in _FORBIDDEN_SYMBOLS), (
                f"{module} references forbidden symbol {tok!r}"
            )


def test_importing_corrected_inference_pulls_no_p3b_truth():
    code = (
        "import sys\n"
        "import research.p3c4.conflicting_inference\n"
        "leaks = sorted(m for m in sys.modules if m.startswith('research.p3b'))\n"
        "print('\\n'.join(leaks))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing the corrected inference pulled P3B truth: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# Gate §19 — NORMAL / INTERMITTENT_* are not conflicting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_id", CONFLICTING_CANDIDATE_IDS)
def test_normal_is_not_conflicting(candidate_id):
    # A normal site's residual is read noise crossing zero; it must NOT be
    # flagged conflicting. Its expected conflicting_evidence truth is NO.
    cfg = conflicting_candidate(candidate_id)
    for seed in (10, 11, 12, 13, 14):
        scenario = build_development_scenario("NORMAL", seed)
        with tempfile.TemporaryDirectory() as td:
            corpus = _generate(scenario, td)
            features, tf = _features(scenario, corpus)
            adm = build_admission(scenario)
        field = infer_conflicting(cfg, features)
        assert field.value == NO


@pytest.mark.parametrize("candidate_id", CONFLICTING_CANDIDATE_IDS)
@pytest.mark.parametrize(
    "class_name",
    (
        "INTERMITTENT_TWO_STATE",
        "INTERMITTENT_MULTI_STATE",
        "INTERMITTENT_CONTINUOUS",
        "RARE_LOW_STATE",
        "RARE_HIGH_STATE",
    ),
)
def test_intermittent_state_change_is_not_conflicting(candidate_id, class_name):
    # An intermittent site changes state (ON positive / OFF ~zero) but never
    # genuinely reverses the residual sign; it must NOT be flagged conflicting.
    cfg = conflicting_candidate(candidate_id)
    for seed in (10, 11):
        scenario = build_development_scenario(class_name, seed)
        with tempfile.TemporaryDirectory() as td:
            corpus = _generate(scenario, td)
            features, tf = _features(scenario, corpus)
            adm = build_admission(scenario)
        field = infer_conflicting(cfg, features)
        assert field.value == NO, class_name


# ---------------------------------------------------------------------------
# Gate §19 — SIGN_CHANGING_POST_DARK keeps a REAL conflict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_id", CONFLICTING_CANDIDATE_IDS)
def test_sign_changing_post_dark_is_conflicting(candidate_id):
    cfg = conflicting_candidate(candidate_id)
    for seed in (10, 11):
        scenario = build_development_scenario("SIGN_CHANGING_POST_DARK", seed)
        with tempfile.TemporaryDirectory() as td:
            corpus = _generate(scenario, td)
            features, tf = _features(scenario, corpus)
            adm = build_admission(scenario)
        field = infer_conflicting(cfg, features)
        assert field.value == YES


# ---------------------------------------------------------------------------
# Gate §19 — CENSORED still abstains; single transient still transient/abstain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_id", CONFLICTING_CANDIDATE_IDS)
def test_censored_still_abstains(candidate_id):
    from research.p3b.qualification_policy import ACTION_ABSTAIN_CENSORED
    from research.p3c.evidence_bridge import to_evidence_packet
    from research.p3b.qualification_policy import evaluate

    cfg = conflicting_candidate(candidate_id)
    scenario = build_development_scenario("CENSORED_ANOMALY", 10)
    with tempfile.TemporaryDirectory() as td:
        corpus = _generate(scenario, td)
        features, tf = _features(scenario, corpus)
        adm = build_admission(scenario)
    decision = evaluate(to_evidence_packet(infer_site(cfg, features, tf, adm)))
    assert decision.action == ACTION_ABSTAIN_CENSORED


@pytest.mark.parametrize("candidate_id", CONFLICTING_CANDIDATE_IDS)
def test_single_transient_stays_transient_or_abstain(candidate_id):
    from research.p3b.qualification_policy import (
        ACTION_ABSTAIN_CENSORED,
        ACTION_ABSTAIN_INCONSISTENT,
        ACTION_ABSTAIN_INSUFFICIENT,
        evaluate,
    )
    from research.p3c.evidence_bridge import to_evidence_packet

    abstain_actions = (
        ACTION_ABSTAIN_CENSORED,
        ACTION_ABSTAIN_INCONSISTENT,
        ACTION_ABSTAIN_INSUFFICIENT,
    )
    cfg = conflicting_candidate(candidate_id)
    scenario = build_development_scenario("SINGLE_TRANSIENT", 10)
    with tempfile.TemporaryDirectory() as td:
        corpus = _generate(scenario, td)
        features, tf = _features(scenario, corpus)
        adm = build_admission(scenario)
    decision = evaluate(to_evidence_packet(infer_site(cfg, features, tf, adm)))
    assert decision.action in abstain_actions


# ---------------------------------------------------------------------------
# Gate §19 — dark mismatch still REQUALIFY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_id", CONFLICTING_CANDIDATE_IDS)
def test_dark_mismatch_stays_requalify(candidate_id):
    from research.p3b.qualification_policy import ACTION_REQUALIFY, evaluate
    from research.p3c.evidence_bridge import to_evidence_packet

    cfg = conflicting_candidate(candidate_id)
    scenario = build_development_scenario("STABLE_ANOMALY_WITH_MISMATCHED_DARK", 10)
    with tempfile.TemporaryDirectory() as td:
        corpus = _generate(scenario, td)
        dark = np.median(
            np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0
        )
        from research.p3b.features import compute_features

        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
        features = cf.sites[0]
        tf = _temporal_features(scenario, corpus)
        adm = build_admission(scenario)
    decision = evaluate(to_evidence_packet(infer_site(cfg, features, tf, adm)))
    assert decision.action == ACTION_REQUALIFY


# ---------------------------------------------------------------------------
# Full-campaign effects: fact matrix corrected, recall no longer 0
# ---------------------------------------------------------------------------


def test_conflicting_fact_matrix_is_corrected(closure_campaign):
    results = closure_campaign
    for cid in CONFLICTING_CANDIDATE_IDS:
        matrix = results["fact_confusion_matrices"][cid]["conflicting_evidence"]
        # The 65 YES-where-NO cells of the frozen rule collapse to a small
        # residual (the extreme per-frame noise class — a distinct phenomenon,
        # out of this lot's scope). Truth YES is fully preserved (SIGN_CHANGING).
        assert matrix["YES"]["YES"] == 5
        wrong = matrix.get("NO", {}).get("YES", 0)
        assert wrong <= 5, f"{cid} still has {wrong} YES-where-NO conflict cells"
        assert matrix["NO"]["NO"] >= 90


def test_sign_changing_conflict_is_preserved_in_campaign(closure_campaign):
    results = closure_campaign
    for cid in CONFLICTING_CANDIDATE_IDS:
        by_class = results["conflicting_evidence_by_class"][cid]
        assert by_class["SIGN_CHANGING_POST_DARK"]["inferred"] == {"YES": 5}
        for cls in ("NORMAL", "INTERMITTENT_TWO_STATE", "INTERMITTENT_MULTI_STATE"):
            assert by_class[cls]["inferred"] == {"NO": 5}, cls


def test_qualification_recall_is_no_longer_zero_by_artifact(closure_campaign):
    results = closure_campaign
    for cand in results["candidates"]:
        recall = cand["qualification_recall"]
        assert recall is not None and recall > 0.0, cand["candidate_id"]
        # Never a silent maximisation claim: the recall is reported alongside the
        # target count and abstained-target count, not asserted to be optimal.
        assert cand["target_count"] == 25


def test_corrected_rule_reads_no_class_or_truth(closure_campaign):
    # The corrected inference consumes (cfg, features, temporal_features,
    # admission) only — never a class label or expected state.
    import inspect

    sig = inspect.signature(infer_site)
    assert set(sig.parameters) == {"cfg", "features", "temporal_features", "admission"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate(scenario, td):
    from research.p3b.generator import generate_corpus

    return generate_corpus(scenario, td)


def _features(scenario, corpus):
    from research.p3b.features import compute_features
    from research.p3c4.temporal_features import compute_temporal_features

    site = scenario.sites[0]
    hard = (
        site.hard_limit_adu
        if site.hard_limit_adu is not None
        else scenario.sensor.saturation_limit_adu
    )
    cf = compute_features(scenario, corpus.frame_arrays)
    tf = compute_temporal_features(
        corpus.frame_arrays,
        scenario.light_frames(),
        site.x,
        site.y,
        cfa_pattern=scenario.sensor.cfa_pattern,
        hard_limit=hard,
    )
    return cf.sites[0], tf


def _temporal_features(scenario, corpus):
    from research.p3c4.temporal_features import compute_temporal_features

    site = scenario.sites[0]
    hard = (
        site.hard_limit_adu
        if site.hard_limit_adu is not None
        else scenario.sensor.saturation_limit_adu
    )
    return compute_temporal_features(
        corpus.frame_arrays,
        scenario.light_frames(),
        site.x,
        site.y,
        cfa_pattern=scenario.sensor.cfa_pattern,
        hard_limit=hard,
    )


__all__ = []
