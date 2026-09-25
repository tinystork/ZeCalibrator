"""P3C-3 — candidate freeze tests (§28): freeze precedes evaluation, is immutable, and is reproducible."""

from __future__ import annotations

import pytest

from research.p3c.candidate_freeze import (
    FREEZE_SCHEMA,
    FREEZE_VERSION,
    CandidateFreeze,
    freeze_candidates,
    live_candidate_config_hashes,
    source_hashes,
)
from research.p3c.inference_candidates import CANDIDATES, CANDIDATE_IDS


# ---------------------------------------------------------------------------
# The freeze is a snapshot of the candidates + manifests, produced before eval
# ---------------------------------------------------------------------------


def test_freeze_is_a_candidatefreeze_with_stable_schema():
    freeze = freeze_candidates()
    assert isinstance(freeze, CandidateFreeze)
    assert freeze.schema == FREEZE_SCHEMA
    assert freeze.version == FREEZE_VERSION
    assert freeze.freeze_hash


def test_freeze_covers_exactly_the_three_frozen_candidates():
    freeze = freeze_candidates()
    ids = [c["candidate_id"] for c in freeze.candidates]
    assert ids == list(CANDIDATE_IDS)
    assert set(ids) == {"p3c-baseline", "p3c-conservative", "p3c-sensitive"}


def test_freeze_records_numerical_search_parameters():
    # §1: the freeze pins the numerical search parameters, not just names.
    freeze = freeze_candidates()
    for c in freeze.candidates:
        assert c["version"]
        assert c["description"]
        assert c["parameters"]
        for name, info in c["parameters"].items():
            assert isinstance(info["value"], (int, float))
            assert info["unit"]
            assert info["kind"] == "RESEARCH_CANDIDATE_PARAMETER"
            assert info["rationale"]


def test_freeze_records_source_code_hashes_and_manifest_hashes():
    freeze = freeze_candidates()
    # Source hashes of the inference-side modules.
    assert set(freeze.source_hashes) == {
        "inference_candidates.py",
        "evidence_bridge.py",
        "inference_contract.py",
    }
    for name, digest in freeze.source_hashes.items():
        assert len(digest) == 64  # SHA-256 hex
    # DEVELOPMENT + QUALIFICATION manifest hashes.
    assert len(freeze.development_manifest_hash) == 64
    assert len(freeze.qualification_manifest_hash) == 64
    assert freeze.development_manifest_hash != freeze.qualification_manifest_hash


def test_freeze_records_version_provenance():
    freeze = freeze_candidates()
    assert freeze.metrics_contract_version
    assert freeze.reason_code_version
    assert freeze.adapter_version
    assert freeze.candidate_schema_version


# ---------------------------------------------------------------------------
# The freeze is reproducible and does not depend on any evaluation
# ---------------------------------------------------------------------------


def test_freeze_is_reproducible():
    a = freeze_candidates()
    b = freeze_candidates()
    assert a.freeze_hash == b.freeze_hash
    assert a.to_dict() == b.to_dict()


def test_freeze_hash_covers_the_payload():
    # A change to any frozen field must change the hash.
    freeze = freeze_candidates()
    tampered = freeze.to_dict()
    tampered["candidates"][0]["parameters"]["local_anomaly_adu"]["value"] = 99999.0
    from research.p3c.candidate_freeze import _sha256_text, _canonical_json

    hash_payload = {k: v for k, v in tampered.items() if k != "freeze_hash"}
    recomputed = _sha256_text(_canonical_json(hash_payload))
    assert recomputed != freeze.freeze_hash


# ---------------------------------------------------------------------------
# Frozen parameters match the live candidates (no drift after freeze)
# ---------------------------------------------------------------------------


def test_live_candidate_config_hashes_are_computable():
    hashes = live_candidate_config_hashes()
    assert set(hashes) == set(CANDIDATE_IDS)
    for c in CANDIDATES:
        assert len(hashes[c.candidate_id]) == 64


def test_source_hashes_match_live_source_files():
    # The freeze's source hashes equal the current source bytes (the campaign is
    # only valid while the code under freeze is the code actually measured).
    assert source_hashes() == freeze_candidates().source_hashes
