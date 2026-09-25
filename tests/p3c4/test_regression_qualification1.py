"""P3C-4 LOT 5 — QUALIFICATION-1 regression tests (§27/§53).

The QUALIFICATION-1 replay is a **known-data regression only**, never a
validation. It states its corpus, does not claim the bug is closed on the
strength of ``YES``, and points to QUALIFICATION-2 as the closure proof.
"""

from __future__ import annotations

from research.p3c4.regression_qualification1 import (
    KNOWN_DATA_REGRESSION_ONLY,
    REGRESSION_SCHEMA,
    REGRESSION_VERSION,
)


def test_regression_is_marked_known_data_only(qualification1_regression):
    r = qualification1_regression
    assert r["known_data_regression_only"] is True
    assert r["role"] == "KNOWN_DATA_REGRESSION_ONLY"
    assert KNOWN_DATA_REGRESSION_ONLY is True


def test_regression_has_stable_schema(qualification1_regression):
    r = qualification1_regression
    assert r["regression_schema"] == REGRESSION_SCHEMA
    assert r["regression_version"] == REGRESSION_VERSION


def test_regression_states_its_corpus(qualification1_regression):
    r = qualification1_regression
    assert "P3B original QUALIFICATION-1 corpus" in r["corpus"]
    assert r["seeds"] == [5, 6, 7, 8, 9]


def test_undersampled_star_core_is_yes_on_the_p3b_corpus(qualification1_regression):
    # On the P3B corpus the star core is rendered at a FIXED coordinate, so YES
    # is the correct answer for those non-physical data — reported, not hidden.
    r = qualification1_regression
    assert r["persisted_truth_by_class"]["UNDERSAMPLED_STAR_CORE"] == "NO"
    for cand in r["candidates"]:
        dist = cand["persisted_distribution_by_class"]["UNDERSAMPLED_STAR_CORE"]
        assert dist == {"YES": 5, "NO": 0, "UNDETERMINED": 0}, (
            f"{cand['candidate_id']} must infer YES for the fixed-coordinate star core"
        )


def test_star_crossing_site_is_still_rejected_on_the_p3b_corpus(qualification1_regression):
    r = qualification1_regression
    for cand in r["candidates"]:
        dist = cand["persisted_distribution_by_class"]["STAR_CROSSING_SITE"]
        assert dist["YES"] == 0
        assert dist["NO"] == 5


def test_regression_does_not_claim_closure(qualification1_regression):
    # The honest reading: YES is correct for the non-physical corpus; the closure
    # proof is QUALIFICATION-2, not this replay.
    r = qualification1_regression
    headline = r["headline"]
    assert headline["undersampled_star_core_inferred_persisted"]["YES"] == 5
    assert "does NOT claim the bug is closed" in headline["reading"]
    assert "QUALIFICATION-2" in headline["reading"]


def test_regression_reports_distributions_not_single_values(qualification1_regression):
    # Seed-dependent classes are surfaced as distributions, never collapsed.
    r = qualification1_regression
    for cand in r["candidates"]:
        for cls, dist in cand["persisted_distribution_by_class"].items():
            assert set(dist) == {"YES", "NO", "UNDETERMINED"}
            assert sum(dist.values()) == 5, f"{cand['candidate_id']}/{cls}"


__all__ = []
