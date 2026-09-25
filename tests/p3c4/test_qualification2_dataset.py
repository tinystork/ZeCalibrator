"""P3C-4 LOT 4 — QUALIFICATION-2 dataset tests (§28–§30).

These tests pin the QUALIFICATION-2 dataset: fresh seeds (10..14), disjoint from
DEVELOPMENT (0..4) and QUALIFICATION-1 (5..9); no duplicated signature with DEV
or QUALIF-1 (verified, not declared, via research.p3b.dataset_split); and the
manifest carries the four verified assertions.

The HOLDOUT seal is tested separately in
``tests/p3c4/test_qualification2_holdout_seal.py`` (structural: no holdout code
path in research/p3c4).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from research.p3b.dataset_split import (
    DATASET_DEVELOPMENT,
    DATASET_HOLDOUT,
    DATASET_QUALIFICATION,
    scenario_signature,
)
from research.p3c.development_corpus import DEVELOPMENT_SEEDS, development_scenarios
from research.p3c.qualification_corpus import QUALIFICATION_SEEDS, qualification_scenarios
from research.p3c4.qualification2_corpus import (
    QUALIFICATION2_CLASSES,
    QUALIFICATION2_ROLE,
    QUALIFICATION2_SEEDS,
    build_qualification2_manifest,
    build_qualification2_scenario,
    qualification2_scenarios,
)


# ---------------------------------------------------------------------------
# Seeds: fresh and disjoint
# ---------------------------------------------------------------------------


def test_qualification2_seeds_are_exactly_10_to_14():
    assert QUALIFICATION2_SEEDS == (10, 11, 12, 13, 14)


def test_qualification2_seeds_are_disjoint_from_development_and_qualification1():
    assert set(QUALIFICATION2_SEEDS).isdisjoint(set(DEVELOPMENT_SEEDS))
    assert set(QUALIFICATION2_SEEDS).isdisjoint(set(QUALIFICATION_SEEDS))
    # All three seed sets are pairwise disjoint.
    assert set(DEVELOPMENT_SEEDS).isdisjoint(set(QUALIFICATION_SEEDS))


def test_qualification2_covers_all_20_classes():
    scenarios = qualification2_scenarios()
    classes = {s.sites[0].cfa_class for s in scenarios}
    assert len(classes) == 20
    assert classes == set(QUALIFICATION2_CLASSES)
    assert len(scenarios) == 20 * len(QUALIFICATION2_SEEDS)


# ---------------------------------------------------------------------------
# Signature-based anti-leakage: no duplicated signature with DEV or QUALIF-1
# ---------------------------------------------------------------------------


def test_no_signature_overlaps_development_or_qualification1():
    # Compute the actual signature sets (research.p3b.dataset_split), not a
    # declaration: QUALIFICATION-2 must not share any scenario signature with
    # either the DEVELOPMENT or the QUALIFICATION-1 dataset.
    q2 = {scenario_signature(s) for s in qualification2_scenarios()}
    dev = {scenario_signature(s) for s in development_scenarios()}
    q1 = {scenario_signature(s) for s in qualification_scenarios()}

    assert len(q2) == 100, "QUALIFICATION-2 has duplicate internal signatures"
    assert q2.isdisjoint(dev), "QUALIFICATION-2 overlaps DEVELOPMENT"
    assert q2.isdisjoint(q1), "QUALIFICATION-2 overlaps QUALIFICATION-1"
    # Sanity: DEV and QUALIF-1 are also disjoint from each other.
    assert dev.isdisjoint(q1)


def test_signature_guard_refuses_a_collision():
    # The dataset_split guard is a typed refusal, not a warning: the SAME
    # scenario signature presented under two different roles raises.
    from research.p3b.dataset_split import DatasetRegistry, SignatureCollisionError

    scenario = build_qualification2_scenario("NORMAL", 10)
    registry = DatasetRegistry()
    registry.register(DATASET_DEVELOPMENT, [scenario])
    with pytest.raises(SignatureCollisionError):
        registry.register(DATASET_QUALIFICATION, [scenario])


# ---------------------------------------------------------------------------
# Manifest — role, seeds, structure, hashes, verified assertions
# ---------------------------------------------------------------------------


def test_manifest_role_seeds_and_structure():
    m = build_qualification2_manifest()
    assert m["dataset_role"] == QUALIFICATION2_ROLE
    assert m["seeds"] == list(QUALIFICATION2_SEEDS)
    assert m["structure"]["class_count"] == 20
    assert m["structure"]["scenario_count"] == 100
    assert m["structure"]["light_frames_per_scenario"] == 24
    assert m["manifest_hash"]


def test_manifest_assertions_are_verified():
    m = build_qualification2_manifest()
    a = m["verified_assertions"]
    assert a == {
        "fresh_validation": True,
        "development_overlap": False,
        "qualification1_overlap": False,
        "holdout_used": False,
    }


def test_manifest_is_deterministic():
    m1 = build_qualification2_manifest()
    m2 = build_qualification2_manifest()
    assert m1 == m2
    assert m1["manifest_hash"] == m2["manifest_hash"]


def test_manifest_role_is_not_holdout():
    m = build_qualification2_manifest()
    assert m["dataset_role"] != DATASET_HOLDOUT
    assert m["dataset_role"] not in (DATASET_DEVELOPMENT, DATASET_QUALIFICATION)


__all__ = []
