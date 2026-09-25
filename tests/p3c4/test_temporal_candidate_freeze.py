"""P3C-4 LOT 5 — GEL-2 candidate freeze tests (§31/§32/§53).

The freeze must be produced before any QUALIFICATION-2 metric, pin the temporal
candidates byte-for-byte (configs, source hashes, manifest hashes, version
provenance), be reproducible, and refuse drift with a **typed** error — never a
warning.
"""

from __future__ import annotations

import pytest

from research.p3c4.candidate_freeze import (
    FREEZE_SCHEMA,
    FREEZE_VERSION,
    CandidateDriftError,
    CandidateFreeze,
    _assert_not_drifted,
    freeze_candidates,
    live_candidate_config_hashes,
    source_hashes,
)
from research.p3c4.temporal_inference import TEMPORAL_CANDIDATES, TEMPORAL_CANDIDATE_IDS


# ---------------------------------------------------------------------------
# The freeze is a snapshot produced before any evaluation
# ---------------------------------------------------------------------------


def test_freeze_is_a_candidatefreeze_with_stable_schema():
    freeze = freeze_candidates()
    assert isinstance(freeze, CandidateFreeze)
    assert freeze.schema == FREEZE_SCHEMA
    assert freeze.version == FREEZE_VERSION
    assert freeze.freeze_hash


def test_freeze_covers_exactly_the_three_temporal_candidates():
    freeze = freeze_candidates()
    ids = [c["candidate_id"] for c in freeze.candidates]
    assert ids == list(TEMPORAL_CANDIDATE_IDS)
    assert set(ids) == {"p3c4-baseline", "p3c4-conservative", "p3c4-sensitive"}


def test_freeze_records_numerical_search_parameters():
    freeze = freeze_candidates()
    for c in freeze.candidates:
        assert c["version"]
        assert c["description"]
        assert c["base_candidate_id"].startswith("p3c-")
        assert c["parameters"]
        for name, info in c["parameters"].items():
            assert isinstance(info["value"], (int, float))
            assert info["unit"]
            assert info["kind"] == "RESEARCH_CANDIDATE_PARAMETER"
            assert info["rationale"]
        # The two temporal parameters are pinned by value.
        assert c["parameters"]["presence_adu"]["value"] in (25.0, 50.0, 150.0)
        assert c["parameters"]["departure_offset_px"]["value"] in (1.0, 2.0)


def test_freeze_records_source_code_hashes():
    freeze = freeze_candidates()
    assert set(freeze.source_hashes) == {
        "temporal_evidence.py",
        "temporal_features.py",
        "temporal_inference.py",
        "inference_candidates.py",
        "evidence_bridge.py",
        "inference_contract.py",
    }
    for digest in freeze.source_hashes.values():
        assert len(digest) == 64  # SHA-256 hex


def test_freeze_records_manifest_hashes():
    freeze = freeze_candidates()
    assert len(freeze.development_manifest_hash) == 64
    assert len(freeze.qualification2_manifest_hash) == 64
    assert freeze.development_manifest_hash != freeze.qualification2_manifest_hash


def test_freeze_records_version_provenance():
    freeze = freeze_candidates()
    assert freeze.metrics_contract_version == "p3c4-qualification-metrics-1"
    assert freeze.temporal_contract_version == "p3c4-temporal-evidence-contract-1"
    assert freeze.reason_code_version
    assert freeze.adapter_version
    assert freeze.candidate_schema_version


def test_freeze_is_reproducible():
    a = freeze_candidates()
    b = freeze_candidates()
    assert a.freeze_hash == b.freeze_hash
    assert a.to_dict() == b.to_dict()


def test_freeze_hash_covers_the_payload():
    freeze = freeze_candidates()
    tampered = freeze.to_dict()
    tampered["candidates"][0]["parameters"]["presence_adu"]["value"] = 99999.0
    from research.p3c4.candidate_freeze import _canonical_json, _sha256_text

    hash_payload = {k: v for k, v in tampered.items() if k != "freeze_hash"}
    recomputed = _sha256_text(_canonical_json(hash_payload))
    assert recomputed != freeze.freeze_hash


# ---------------------------------------------------------------------------
# Live candidates match the freeze (no drift after freeze)
# ---------------------------------------------------------------------------


def test_live_candidate_config_hashes_are_computable():
    hashes = live_candidate_config_hashes()
    assert set(hashes) == set(TEMPORAL_CANDIDATE_IDS)
    for c in TEMPORAL_CANDIDATES:
        assert len(hashes[c.candidate_id]) == 64


def test_source_hashes_match_live_source_files():
    assert source_hashes() == freeze_candidates().source_hashes


def test_no_drift_is_accepted_when_unchanged():
    freeze = freeze_candidates()
    # Must not raise when the live modules match the freeze.
    _assert_not_drifted(freeze)


# ---------------------------------------------------------------------------
# Refusal of drift (§32) — typed error, never a warning
# ---------------------------------------------------------------------------


def test_candidate_set_drift_is_a_typed_error():
    freeze = freeze_candidates()
    tampered = freeze.to_dict()
    # Drop a candidate from the frozen snapshot -> the live set diverges.
    tampered["candidates"] = tampered["candidates"][:2]
    drifted = CandidateFreeze(
        freeze_hash="x",
        schema=freeze.schema,
        version=freeze.version,
        metrics_contract_version=freeze.metrics_contract_version,
        reason_code_version=freeze.reason_code_version,
        adapter_version=freeze.adapter_version,
        temporal_contract_version=freeze.temporal_contract_version,
        candidate_schema_version=freeze.candidate_schema_version,
        development_manifest_hash=freeze.development_manifest_hash,
        qualification2_manifest_hash=freeze.qualification2_manifest_hash,
        source_hashes=dict(freeze.source_hashes),
        candidates=tuple(tampered["candidates"]),
    )
    with pytest.raises(CandidateDriftError):
        _assert_not_drifted(drifted)


def test_candidate_config_drift_is_a_typed_error():
    freeze = freeze_candidates()
    tampered = freeze.to_dict()
    tampered["candidates"][0]["parameters"]["presence_adu"]["value"] = 12345.0
    drifted = CandidateFreeze(
        freeze_hash="x",
        schema=freeze.schema,
        version=freeze.version,
        metrics_contract_version=freeze.metrics_contract_version,
        reason_code_version=freeze.reason_code_version,
        adapter_version=freeze.adapter_version,
        temporal_contract_version=freeze.temporal_contract_version,
        candidate_schema_version=freeze.candidate_schema_version,
        development_manifest_hash=freeze.development_manifest_hash,
        qualification2_manifest_hash=freeze.qualification2_manifest_hash,
        source_hashes=dict(freeze.source_hashes),
        candidates=tuple(tampered["candidates"]),
    )
    with pytest.raises(CandidateDriftError):
        _assert_not_drifted(drifted)


def test_source_drift_is_a_typed_error():
    freeze = freeze_candidates()
    tampered_hashes = dict(freeze.source_hashes)
    tampered_hashes["temporal_inference.py"] = "0" * 64
    drifted = CandidateFreeze(
        freeze_hash="x",
        schema=freeze.schema,
        version=freeze.version,
        metrics_contract_version=freeze.metrics_contract_version,
        reason_code_version=freeze.reason_code_version,
        adapter_version=freeze.adapter_version,
        temporal_contract_version=freeze.temporal_contract_version,
        candidate_schema_version=freeze.candidate_schema_version,
        development_manifest_hash=freeze.development_manifest_hash,
        qualification2_manifest_hash=freeze.qualification2_manifest_hash,
        source_hashes=tampered_hashes,
        candidates=tuple(freeze.candidates),
    )
    with pytest.raises(CandidateDriftError):
        _assert_not_drifted(drifted)


def test_version_provenance_drift_is_a_typed_error():
    freeze = freeze_candidates()
    drifted = CandidateFreeze(
        freeze_hash="x",
        schema=freeze.schema,
        version=freeze.version,
        metrics_contract_version="tampered-metrics-version",
        reason_code_version=freeze.reason_code_version,
        adapter_version=freeze.adapter_version,
        temporal_contract_version=freeze.temporal_contract_version,
        candidate_schema_version=freeze.candidate_schema_version,
        development_manifest_hash=freeze.development_manifest_hash,
        qualification2_manifest_hash=freeze.qualification2_manifest_hash,
        source_hashes=dict(freeze.source_hashes),
        candidates=tuple(freeze.candidates),
    )
    with pytest.raises(CandidateDriftError):
        _assert_not_drifted(drifted)


__all__ = []
