"""P3C-4 LOT 5 — QUALIFICATION-2 campaign tests (§33–§42, §52/§54).

These tests pin the QUALIFICATION-2 measurement: the headline safety metric
``sky_confounder_persistence_false_positive`` (§34), the mandatory per-class
decompositions (§35 confounders, §36 intermittents), the fact confusion matrices
(§39), the machine-readable promotion flags (§37/§38), the §40 minimal criterion,
and reproducibility — with **no** winner, ranking or bare ``false_promotion=0``.
"""

from __future__ import annotations

import json
from pathlib import Path

from research.p3c.inference_contract import SENSOR_EVIDENCE_FACTS
from research.p3c4.qualification2_runner import (
    HEADLINE_METRIC_NAMES,
    SAFETY_METRIC_SKY_CONFOUNDER,
    build_artifacts,
    run_qualification2_campaign,
)

RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "research" / "p3c4" / "qualification2_runner.py"
)

# The five §35 confounders and five §36 intermittents, as class names.
_CONFOUNDERS = {
    "UNDERSAMPLED_STAR_CORE",
    "STAR_CROSSING_SITE",
    "OPTICAL_STRUCTURE",
    "FLAT_STRUCTURE",
    "DUST_OR_VIGNETTING",
}
_INTERMITTENTS = {
    "RARE_HIGH_STATE",
    "RARE_LOW_STATE",
    "INTERMITTENT_TWO_STATE",
    "INTERMITTENT_MULTI_STATE",
    "INTERMITTENT_CONTINUOUS",
}


# ---------------------------------------------------------------------------
# Well-formed result structure
# ---------------------------------------------------------------------------


def test_campaign_runs_and_has_expected_shape(frozen_campaign):
    freeze, results = frozen_campaign
    assert results["campaign"]["scenario_count"] == 100
    assert results["campaign"]["class_count"] == 20
    assert results["campaign"]["light_frame_count"] == 2400
    assert len(results["candidates"]) == 3
    assert {c["candidate_id"] for c in results["candidates"]} == {
        "p3c4-baseline",
        "p3c4-conservative",
        "p3c4-sensitive",
    }


def test_every_candidate_has_the_headline_metrics(frozen_campaign):
    _, results = frozen_campaign
    for cand in results["candidates"]:
        for name in HEADLINE_METRIC_NAMES:
            assert name in cand["metrics"], f"{cand['candidate_id']} missing {name}"
        assert cand["abstained_target_count"] == cand["target_count"]


def test_campaign_carries_verified_assertions(frozen_campaign):
    # §54: the four assertions are carried as verified, not declared.
    _, results = frozen_campaign
    a = results["campaign"]["verified_assertions"]
    assert a == {
        "fresh_validation": True,
        "development_overlap": False,
        "qualification1_overlap": False,
        "holdout_used": False,
    }


# ---------------------------------------------------------------------------
# §34 headline safety metric — zero sky-confounder false positives
# ---------------------------------------------------------------------------


def test_sky_confounder_false_positive_is_zero(frozen_campaign):
    _, results = frozen_campaign
    for cand in results["candidates"]:
        sky = cand["sky_confounder_persistence_false_positive"]
        assert sky["metric"] == SAFETY_METRIC_SKY_CONFOUNDER
        assert sky["denominator"] == 25  # 5 confounders x 5 seeds
        assert sky["numerator"] == 0
        assert sky["value"] == 0.0


def test_confounder_decomposition_covers_all_five_classes(frozen_campaign):
    # §35 — separately, never averaged away.
    _, results = frozen_campaign
    for cand in results["candidates"]:
        decomp = cand["sky_confounder_persistence_false_positive"]["decomposition_by_class"]
        assert set(decomp) == _CONFOUNDERS
        for cls, count in decomp.items():
            assert count == 0, f"{cand['candidate_id']}/{cls} persisted=YES ({count})"


def test_intermittent_control_is_reported_separately(frozen_campaign):
    # §36 — separately, proving the stellar error was not closed by making all
    # persistence impossible.
    _, results = frozen_campaign
    for cand in results["candidates"]:
        control = cand["intermittent_control"]
        assert set(control) == _INTERMITTENTS
        # A rare low-occupancy intermittent is protected (UNDETERMINED), never NO.
        assert control["RARE_HIGH_STATE"]["NO"] == 0
        assert control["RARE_HIGH_STATE"]["UNDETERMINED"] == 5
        # The other intermittents keep a useful temporal signal (YES).
        for cls in _INTERMITTENTS - {"RARE_HIGH_STATE"}:
            assert control[cls]["YES"] == 5, f"{cand['candidate_id']}/{cls} lost its signal"


# ---------------------------------------------------------------------------
# §37/§38 — promotion not exercised, machine-readable
# ---------------------------------------------------------------------------


def test_promotion_stage_not_exercised(frozen_campaign):
    _, results = frozen_campaign
    for cand in results["candidates"]:
        p = cand["promotion"]
        assert p["promotion_stage_exercised"] is False
        assert p["promotion_safety_measured"] is False
        # Never a bare false_promotion = 0: it carries a NOT_EXERCISED status.
        fpr = p["site_level_false_promotion_rate"]
        assert fpr["status"] == "NOT_EXERCISED"
        assert fpr["value"] == 0.0


# ---------------------------------------------------------------------------
# Safety errors + §40 hard invariants
# ---------------------------------------------------------------------------


def test_no_safety_violation_and_no_hard_invariant(frozen_campaign):
    _, results = frozen_campaign
    for cand in results["candidates"]:
        for name, count in cand["safety_violations"].items():
            assert count == 0, f"{cand['candidate_id']} safety error {name}={count}"
        for name, count in cand["hard_invariant_violations"].items():
            assert count == 0
        assert cand["invalid"] is False


def test_section_40_minimal_criterion_is_satisfied(frozen_campaign):
    _, results = frozen_campaign
    s40 = results["campaign"]["section_40_status"]
    assert s40["persistence_yes_zero_for_undersampled_star_core"] is True
    assert s40["persistence_yes_zero_for_star_crossing_site"] is True
    assert s40["censored_quantitative_inference_zero"] is True
    assert s40["non_representative_dark_as_reconstruction_candidate_zero"] is True
    assert s40["truth_leak_structural"] is True
    assert s40["inter_frame_switching_structural"] is True
    assert s40["sealed"] is True
    assert s40["opening_recommendation"] == "NONE"


# ---------------------------------------------------------------------------
# Fact confusion matrices (§39)
# ---------------------------------------------------------------------------


def test_fact_matrices_cover_the_six_facts(frozen_campaign):
    _, results = frozen_campaign
    matrices = results["fact_confusion_matrices"]
    assert set(matrices) == {"p3c4-baseline", "p3c4-conservative", "p3c4-sensitive"}
    for cid, facts in matrices.items():
        assert set(facts) == set(SENSOR_EVIDENCE_FACTS)
        for fact, matrix in facts.items():
            assert matrix, f"{cid}/{fact} matrix is empty"
            total = sum(
                count
                for truth_value, inferred_counts in matrix.items()
                for count in inferred_counts.values()
            )
            assert total == 100, f"{cid}/{fact} totals {total}"


def test_persisted_fact_matrix_shows_no_confounder_false_positive(frozen_campaign):
    # §39: persisted is the central metric. On QUALIFICATION-2 the star core must
    # NO longer be misread as a sensor site (the latent defect closed).
    _, results = frozen_campaign
    for cid in ("p3c4-baseline", "p3c4-conservative", "p3c4-sensitive"):
        persisted = results["fact_confusion_matrices"][cid]["persisted_at_same_sensor_coord"]
        assert persisted.get("NO", {}).get("YES", 0) == 0, (
            f"{cid} must record zero sky-confounder persistence mis-reads"
        )


# ---------------------------------------------------------------------------
# Class breakdown (§31)
# ---------------------------------------------------------------------------


def test_class_breakdown_covers_all_20_classes(frozen_campaign):
    _, results = frozen_campaign
    for cid, breakdown in results["class_breakdown"].items():
        assert len(breakdown) == 20
        for cls, rec in breakdown.items():
            assert rec["seed_count"] == 5
            assert rec["action_counts"]


# ---------------------------------------------------------------------------
# No winner, no ranking (§39)
# ---------------------------------------------------------------------------


def test_no_winner_or_ranking_is_produced(frozen_campaign):
    freeze, results = frozen_campaign
    artifacts = build_artifacts(results, freeze)
    serialized = json.dumps(artifacts, sort_keys=True)
    for key in ("winner", "best", "rank", "ranking", "optimal", "chosen", "selected", "top"):
        assert key not in serialized.lower()


def test_runner_does_not_rank_candidates():
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "argmax" not in text
    assert "argmin" not in text


def test_runner_reuses_existing_policy_and_metrics_never_reimplements():
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "from research.p3b.qualification_policy import" in text
    assert "from research.p3b.metrics import" in text
    assert "evaluate(" in text
    assert "compute_metrics(" in text
    assert "def evaluate(" not in text
    assert "def compute_metrics(" not in text
    assert "_derive_action" not in text


# ---------------------------------------------------------------------------
# Reproducibility (§44)
# ---------------------------------------------------------------------------


def test_campaign_is_byte_reproducible(frozen_campaign):
    freeze, _ = frozen_campaign
    from research.p3c4.qualification2_corpus import build_qualification2_scenario

    subset = tuple(
        build_qualification2_scenario(cls, seed)
        for cls in ("NORMAL", "UNDERSAMPLED_STAR_CORE", "INTERMITTENT_TWO_STATE", "CENSORED_ANOMALY")
        for seed in (10, 11)
    )
    a = run_qualification2_campaign(freeze, scenarios=subset)
    b = run_qualification2_campaign(freeze, scenarios=subset)
    artifacts_a = build_artifacts(a, freeze)
    artifacts_b = build_artifacts(b, freeze)
    assert json.dumps(artifacts_a, sort_keys=True) == json.dumps(artifacts_b, sort_keys=True)


__all__ = []
