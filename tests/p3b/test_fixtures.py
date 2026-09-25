"""Tests for the P3B scenario fixture helper (research/test only).

Covered requirements (mission ZC-SENSOR-P3B1-CLOSURE, part A):

1. the 20 catalogue classes are instantiable;
2. same seed => identical fixture (scenario + declarations);
3. unknown class => typed refusal;
4. the helper never derives truth from a filename;
5. the helper exposes every non-pixel declaration (with provenance);
6. the ``default_expected_*`` are reachable when all required evidence is
   provided explicitly;
7. removing one required evidence => conservative abstention (not a promotion).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from research.p3b.catalog import CLASS_NAMES, UnknownClassError, resolve_class
from research.p3b.fixtures import (
    PROVENANCE_ACQUISITION,
    PROVENANCE_DECLARED,
    PROVENANCE_PIXELS,
    PROVENANCE_TRUTH,
    PROVENANCES,
    ScenarioFixture,
    evaluate_fixture,
    scenario_for_class,
)
from research.p3b.qualification_policy import (
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_PATH = REPO_ROOT / "research" / "p3b" / "fixtures.py"

# The non-pixel facts the policy consumes (every EvidencePacket field) — the
# helper must expose all of them, each with a provenance.
_PACKET_FIELDS = (
    "sensor_identity_resolved",
    "geometry_compatible",
    "calibration_present",
    "calibration_representativeness",
    "persisted_at_same_sensor_coord",
    "independent_group_count",
    "epoch_count",
    "censored_measurement_present",
    "site_residual_behaviour",
    "neighbourhood_residual_stable",
    "transient_only",
    "conflicting_evidence",
    "net_benefit_established",
    "persisted_basis",
    "censored_measurement_count",
)

# Classes whose catalogue ``default_expected_action_state`` is consistent with
# their declared facts (``declared_facts._CLASS_FACTS``): supplying the full
# declared evidence reaches the default.
_REACHABLE = {
    "NORMAL",
    "STABLE_ANOMALY_CORRECTED_BY_DARK",
    "STABLE_ANOMALY_WITH_MISMATCHED_DARK",
    "INTERMITTENT_TWO_STATE",
    "INTERMITTENT_MULTI_STATE",
    "INTERMITTENT_CONTINUOUS",
    "RARE_HIGH_STATE",
    "RARE_LOW_STATE",
    "SIGN_CHANGING_POST_DARK",
    "CENSORED_ANOMALY",
    "NOISE_EXTREME",
    "NEAR_SATURATION",
    "AMBIGUOUS_INSUFFICIENT_EVIDENCE",
}

# Classes whose catalogue default is *inconsistent* with their declared facts —
# a pre-existing catalogue/facts mismatch (out of scope, not fixed here). For
# these the policy fails safe (conservative abstention) even with full evidence,
# and the helper reports ``default_reachable=False`` rather than hiding it.
_UNREACHABLE = {
    "SINGLE_TRANSIENT",
    "OPTICAL_STRUCTURE",
    "STAR_CROSSING_SITE",
    "UNDERSAMPLED_STAR_CORE",
    "COSMIC_RAY",
    "FLAT_STRUCTURE",
    "DUST_OR_VIGNETTING",
}


def _abstain_actions() -> tuple:
    from research.p3b.qualification_policy import (
        ACTION_ABSTAIN_CENSORED,
        ACTION_ABSTAIN_INCONSISTENT,
        ACTION_ABSTAIN_INSUFFICIENT,
    )

    return (ACTION_ABSTAIN_CENSORED, ACTION_ABSTAIN_INCONSISTENT, ACTION_ABSTAIN_INSUFFICIENT)


# ---------------------------------------------------------------------------
# Requirement 1 — all 20 classes instantiable
# ---------------------------------------------------------------------------


def test_all_20_classes_instantiable():
    assert len(CLASS_NAMES) == 20
    for name in CLASS_NAMES:
        fx = scenario_for_class(name)
        assert isinstance(fx, ScenarioFixture)
        assert fx.class_name == name
        assert fx.scenario.sites[0].cfa_class == name
        # The scenario must carry the class's declared truth labels (verbatim).
        entry = resolve_class(name)
        assert fx.expected_truth.qualification_state == entry.default_expected_qualification_state
        assert fx.expected_truth.action_state == entry.default_expected_action_state


# ---------------------------------------------------------------------------
# Requirement 2 — determinism (same seed => identical scenario + declarations)
# ---------------------------------------------------------------------------


def test_same_seed_identical_fixture():
    for name in CLASS_NAMES:
        a = scenario_for_class(name, seed=7)
        b = scenario_for_class(name, seed=7)
        assert a == b, name
        assert a.scenario == b.scenario, name
        assert a.declared_evidence == b.declared_evidence, name
        assert a.expected_truth == b.expected_truth, name


def test_different_seed_changes_scenario_but_not_declarations():
    a = scenario_for_class("INTERMITTENT_TWO_STATE", seed=1)
    b = scenario_for_class("INTERMITTENT_TWO_STATE", seed=2)
    # The pixel scenario differs (different seed), but the declared evidence and
    # truth labels are seed-independent (they are declarations, not pixels).
    assert a.scenario.seed != b.scenario.seed
    assert a.declared_evidence == b.declared_evidence
    assert a.expected_truth == b.expected_truth


# ---------------------------------------------------------------------------
# Requirement 3 — unknown class => typed refusal
# ---------------------------------------------------------------------------


def test_unknown_class_raises_typed_error():
    with pytest.raises(UnknownClassError):
        scenario_for_class("NOT_A_CLASS")


def test_unknown_class_is_key_error_subtype():
    assert issubclass(UnknownClassError, KeyError)


# ---------------------------------------------------------------------------
# Requirement 4 — the helper never derives truth from a filename
# ---------------------------------------------------------------------------


def test_fixtures_module_has_no_filename_or_file_io():
    tree = ast.parse(FIXTURES_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in ("pathlib", "os", "glob"), alias.name
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("pathlib", "os", "glob"), node.module
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("open",), "fixtures must not call open()"
        if isinstance(node, ast.Attribute) and node.attr in (
            "read_text",
            "read_bytes",
            "write_text",
            "write_bytes",
        ):
            raise AssertionError(f"fixtures must not access files via {node.attr}()")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert ".fits" not in node.value, "fixtures must not reference FITS filenames"


def test_expected_truth_comes_from_catalogue_not_filename():
    # The truth labels are exactly the catalogue defaults for the class — never
    # derived from any filename, and never recomputed by the helper.
    for name in CLASS_NAMES:
        entry = resolve_class(name)
        fx = scenario_for_class(name)
        assert fx.expected_truth.qualification_state == entry.default_expected_qualification_state
        assert fx.expected_truth.action_state == entry.default_expected_action_state


def test_helper_signature_takes_only_name_and_seed():
    import inspect

    params = list(inspect.signature(scenario_for_class).parameters)
    assert "name" in params
    assert "seed" in params
    # No filename / path / file-object parameter can smuggle truth in.
    for p in params:
        assert p not in ("filename", "path", "file", "file_path", "manifest"), p


# ---------------------------------------------------------------------------
# Requirement 5 — expose every non-pixel declaration, each with provenance
# ---------------------------------------------------------------------------


def test_fixture_exposes_all_non_pixel_declarations():
    fx = scenario_for_class("INTERMITTENT_TWO_STATE")
    by_name = {d.name: d for d in fx.declared_evidence}
    for field in _PACKET_FIELDS:
        assert field in by_name, f"missing declaration {field!r}"
        assert by_name[field].provenance in PROVENANCES


def test_provenance_separates_the_four_sources():
    fx = scenario_for_class("INTERMITTENT_TWO_STATE")
    by_name = {d.name: d for d in fx.declared_evidence}

    # Independence bookkeeping and calibration presence come from the acquisition
    # structure (epochs / groups / frames).
    assert by_name["epoch_count"].provenance == PROVENANCE_ACQUISITION
    assert by_name["independent_group_count"].provenance == PROVENANCE_ACQUISITION
    assert by_name["calibration_present"].provenance == PROVENANCE_ACQUISITION
    # Net benefit and representativeness are external declared evidence.
    assert by_name["net_benefit_established"].provenance == PROVENANCE_DECLARED
    assert by_name["calibration_representativeness"].provenance == PROVENANCE_DECLARED
    # The class->fact translation is declared evidence, not pixels.
    assert by_name["site_residual_behaviour"].provenance == PROVENANCE_DECLARED

    # The truth labels are exposed under the truth provenance and are NOT part of
    # the policy's input facts.
    assert fx.expected_truth.qualification_state == "QUALIFIED_INTERMITTENT"
    assert PROVENANCE_TRUTH in PROVENANCES


def test_declared_evidence_provenance_is_complete_and_valid():
    for name in CLASS_NAMES:
        fx = scenario_for_class(name)
        for d in fx.declared_evidence:
            assert d.provenance in PROVENANCES, (name, d.name, d.provenance)


# ---------------------------------------------------------------------------
# Requirement 6 — default reachable when all required evidence provided
# ---------------------------------------------------------------------------


def test_reachable_action_matches_the_real_policy_decision():
    # The helper never invents the reachable action: it is the real LOT2 decision
    # on the declared facts. Re-running the policy on the same declarations
    # reproduces it exactly.
    for name in CLASS_NAMES:
        fx = scenario_for_class(name)
        assert fx.reachable_action == evaluate_fixture(fx).action, name


def test_consistent_classes_reach_their_default_with_full_evidence():
    assert len(_REACHABLE) == 13
    for name in _REACHABLE:
        fx = scenario_for_class(name)
        assert fx.default_reachable is True, name
        assert fx.reachable_action == fx.expected_truth.action_state, name


def test_inconsistent_classes_are_reported_honestly_and_fail_safe():
    # For the catalogue/facts-mismatch classes the default is NOT reachable; the
    # helper exposes that fact (default_reachable=False) and the actual outcome
    # is a conservative abstention — never a promotion beyond the default.
    assert len(_UNREACHABLE) == 7
    for name in _UNREACHABLE:
        fx = scenario_for_class(name)
        assert fx.default_reachable is False, name
        assert fx.reachable_action != fx.expected_truth.action_state, name
        assert fx.reachable_action in _abstain_actions(), name


def test_partition_is_exhaustive():
    assert _REACHABLE | _UNREACHABLE == set(CLASS_NAMES)
    assert _REACHABLE.isdisjoint(_UNREACHABLE)


# ---------------------------------------------------------------------------
# Requirement 7 — removing one required evidence => conservative abstention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("remove", ["net_benefit", "independence"])
@pytest.mark.parametrize(
    "name",
    [n for n in CLASS_NAMES if resolve_class(n).default_expected_action_state == ACTION_ELIGIBLE],
)
def test_removing_required_evidence_abstains_not_promotes(name, remove):
    # A reconstruction target needs BOTH net benefit established AND sufficient
    # independence. Removing either must fall back to a conservative abstention,
    # never to ELIGIBLE.
    fx = scenario_for_class(name)
    d = evaluate_fixture(fx, remove=remove)
    assert d.action != ACTION_ELIGIBLE, (name, remove)
    assert d.action in _abstain_actions(), (name, remove)


def test_intermittent_two_state_specifically():
    # The F1 measured case: INTERMITTENT_TWO_STATE reaches ELIGIBLE only with the
    # explicit declarations; dropping net benefit or independence abstains.
    fx = scenario_for_class("INTERMITTENT_TWO_STATE")
    assert fx.reachable_action == ACTION_ELIGIBLE
    assert evaluate_fixture(fx, remove="net_benefit").action == ACTION_ABSTAIN_INSUFFICIENT
    assert evaluate_fixture(fx, remove="independence").action == ACTION_ABSTAIN_INSUFFICIENT

