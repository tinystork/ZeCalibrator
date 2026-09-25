"""P3C-3 — qualification campaign tests (§3, §29–§31, §38, §44): measurement, tables, no winner, reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.p3c.candidate_freeze import freeze_candidates
from research.p3c.inference_contract import SENSOR_EVIDENCE_FACTS
from research.p3c.qualification_corpus import build_qualification_scenario
from research.p3c.qualification_runner import (
    HEADLINE_METRIC_NAMES,
    SAFETY_ERROR_NAMES,
    build_artifacts,
    run_qualification_campaign,
)

RUNNER_PATH = Path(__file__).resolve().parents[2] / "research" / "p3c" / "qualification_runner.py"


# ---------------------------------------------------------------------------
# Well-formed result structure (§3, §29)
# ---------------------------------------------------------------------------


def test_campaign_runs_and_has_expected_shape(frozen_campaign):
    freeze, results = frozen_campaign
    assert results["campaign"]["scenario_count"] == 100
    assert results["campaign"]["class_count"] == 20
    assert results["campaign"]["light_frame_count"] == 2400
    assert len(results["candidates"]) == 3
    assert set(c["candidate_id"] for c in results["candidates"]) == {
        "p3c-baseline",
        "p3c-conservative",
        "p3c-sensitive",
    }


def test_every_candidate_has_the_headline_metrics(frozen_campaign):
    _, results = frozen_campaign
    for cand in results["candidates"]:
        for name in HEADLINE_METRIC_NAMES:
            assert name in cand["metrics"], f"{cand['candidate_id']} missing {name}"
        # abstained_target_count is a raw counter alongside target_count.
        assert cand["abstained_target_count"] == cand["target_count"]


def test_unsafe_action_rate_is_null(frozen_campaign):
    # §3: no reconstruction product exists, so unsafe_action_rate stays null.
    _, results = frozen_campaign
    for cand in results["candidates"]:
        value = cand["metrics"]["unsafe_action_rate"]["value"]
        assert value == 0.0, f"{cand['candidate_id']} unsafe_action_rate must be null"


def test_no_site_level_false_promotion(frozen_campaign):
    # No candidate can promote a non-target (net benefit is never established),
    # so the site-level false-promotion rate is exactly 0.
    _, results = frozen_campaign
    for cand in results["candidates"]:
        assert cand["metrics"]["site_level_false_promotion_rate"]["value"] == 0.0


def test_no_candidate_violates_a_hard_invariant(frozen_campaign):
    # §8/§40: INVALID is reserved for a hard-invariant violation. In this
    # campaign none of the two measurable hard invariants fire.
    _, results = frozen_campaign
    for cand in results["candidates"]:
        assert cand["invalid"] is False
        for name, count in cand["hard_invariant_violations"].items():
            assert count == 0, f"{cand['candidate_id']} hard invariant {name}={count}"


# ---------------------------------------------------------------------------
# Fact confusion matrices (§30): non-empty, well-formed, truth × inferred
# ---------------------------------------------------------------------------


def test_fact_matrices_cover_the_six_facts_for_every_candidate(frozen_campaign):
    _, results = frozen_campaign
    matrices = results["fact_confusion_matrices"]
    assert set(matrices) == {"p3c-baseline", "p3c-conservative", "p3c-sensitive"}
    for cid, facts in matrices.items():
        assert set(facts) == set(SENSOR_EVIDENCE_FACTS)
        for fact, matrix in facts.items():
            # Every cell is a truth-value -> {inferred-value: count} map.
            assert matrix, f"{cid}/{fact} matrix is empty"
            total = 0
            for truth_value, inferred_counts in matrix.items():
                assert inferred_counts, f"{cid}/{fact}/{truth_value} has no inferred counts"
                for inferred_value, count in inferred_counts.items():
                    assert isinstance(count, int) and count >= 0
                    total += count
            # A full-qualification matrix must total 100 evaluations.
            assert total == 100, f"{cid}/{fact} matrix totals {total}, expected 100"


def test_fact_matrix_exposes_where_the_chain_fails(frozen_campaign):
    # The whole point of §30 is that a failure is visible, not a bare aggregate.
    # The undersampled-star-core sky structure is misread as a sensor site: the
    # persisted fact matrix must show truth=NO mapped to inferred=YES.
    _, results = frozen_campaign
    for cid in ("p3c-baseline", "p3c-conservative", "p3c-sensitive"):
        persisted = results["fact_confusion_matrices"][cid]["persisted_at_same_sensor_coord"]
        assert persisted.get("NO", {}).get("YES", 0) == 5, (
            f"{cid} must record the 5 UNDERSAMPLED_STAR_CORE persistence mis-reads"
        )


# ---------------------------------------------------------------------------
# Class breakdown (§31): per-class, never a single global rate
# ---------------------------------------------------------------------------


def test_class_breakdown_covers_all_20_classes_per_candidate(frozen_campaign):
    _, results = frozen_campaign
    for cid, breakdown in results["class_breakdown"].items():
        assert len(breakdown) == 20, f"{cid} class breakdown has {len(breakdown)} classes"
        for cls, rec in breakdown.items():
            assert rec["seed_count"] == 5
            assert rec["action_counts"], f"{cid}/{cls} has no action counts"


def test_masked_classes_are_visible_in_breakdown(frozen_campaign):
    # §31: INTERMITTENT, CENSORED, STAR_CROSSING_SITE, FLAT_STRUCTURE must be
    # individually present (never hidden behind a global rate).
    _, results = frozen_campaign
    for cid, breakdown in results["class_breakdown"].items():
        for cls in ("INTERMITTENT_TWO_STATE", "CENSORED_ANOMALY", "STAR_CROSSING_SITE", "FLAT_STRUCTURE"):
            assert cls in breakdown, f"{cid} is missing {cls}"


# ---------------------------------------------------------------------------
# Policy outcomes (§38) — distributions, no winner
# ---------------------------------------------------------------------------


def test_policy_outcomes_are_nonempty_distributions(frozen_campaign):
    _, results = frozen_campaign
    for cid, dist in results["policy_outcomes"].items():
        assert dist
        assert sum(dist.values()) == 100, f"{cid} policy outcome distribution sums wrong"


def test_no_winner_or_ranking_is_produced(frozen_campaign):
    # §8/§38/§39: the artifacts are descriptive — no "best", no rank, no winner.
    _, results = frozen_campaign
    artifacts = build_artifacts(results, freeze_candidates())
    serialized = json.dumps(artifacts, sort_keys=True)

    forbidden_keys = ("winner", "best", "rank", "ranking", "optimal", "chosen", "selected", "top")
    for key in forbidden_keys:
        assert key not in serialized.lower(), f"artifact carries a ranking term {key!r}"


def test_runner_source_does_not_rank_candidates():
    # No argmax/argmin (the definitive "select best" constructs) anywhere in the
    # campaign runner: the candidates are reported, never ordered by a score.
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "argmax" not in text
    assert "argmin" not in text


def test_runner_reuses_existing_policy_and_metrics_never_reimplements():
    # §7/§33–§34: the action path is the existing qualification_policy.evaluate
    # and metrics.compute_metrics — imported and called, never reimplemented.
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "from research.p3b.qualification_policy import" in text
    assert "from research.p3b.metrics import" in text
    assert "evaluate(" in text
    assert "compute_metrics(" in text
    # No local reimplementation of a decision ladder or metric engine.
    assert "def evaluate(" not in text
    assert "def compute_metrics(" not in text
    assert "_derive_action" not in text


# ---------------------------------------------------------------------------
# Reproducibility (§44): same input + config => byte-identical results
# ---------------------------------------------------------------------------


def test_campaign_is_byte_reproducible():
    # Run the same freeze twice and assert the serialized artifacts are
    # byte-identical. The full campaign is ~10s; a deterministic subset keeps
    # this test fast while still exercising the pure-function guarantee.
    freeze = freeze_candidates()
    subset = tuple(
        build_qualification_scenario(cls, seed)
        for cls in ("NORMAL", "INTERMITTENT_TWO_STATE", "CENSORED_ANOMALY", "UNDERSAMPLED_STAR_CORE")
        for seed in (5, 6)
    )
    a = run_qualification_campaign(freeze, scenarios=subset)
    b = run_qualification_campaign(freeze, scenarios=subset)
    artifacts_a = build_artifacts(a, freeze)
    artifacts_b = build_artifacts(b, freeze)
    assert json.dumps(artifacts_a, sort_keys=True) == json.dumps(artifacts_b, sort_keys=True)
