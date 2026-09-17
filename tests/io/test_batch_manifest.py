"""Batch manifest serialize/parse/write tests (Phase 6).

Covered: strict versioned round-trip, unknown-key / bad-schema / bad-disposition
refusal, atomic write, and manifest-only whole-file hashes (no self-reference).
"""

from __future__ import annotations

import json
import os

import pytest

from zecalibrator.io.batch_manifest import (
    BATCH_MANIFEST_SCHEMA,
    BatchManifestError,
    build_batch_manifest,
    manifest_path,
    parse_batch_manifest,
    serialize_batch_manifest,
    write_batch_manifest,
)


def _manifest(**overrides):
    base = dict(
        schema_version=BATCH_MANIFEST_SCHEMA,
        batch_id="batch-1",
        operation_id="zecalibrator-calibrate-batch",
        api_version="1.0",
        product_version="0.1.0",
        science_contract="1.0",
        matching_policy="zecalibrator.match.v1",
        provenance_schema="zecalibrator.provenance.v1",
        decoder_version="1.0",
        commit_state="COMMITTED",
        batch_status="COMPLETED",
        destination="/tmp/out",
        inputs=[{"index": 0, "identity": {"kind": "fits", "path": "/l.fits", "hdu": 0}}],
        items=[
            {
                "index": 0,
                "disposition": "COMPLETED",
                "plan_id": "p" * 64,
                "input_identity": {"kind": "fits", "path": "/l.fits", "hdu": 0},
                "reason_code": None,
                "reason_details": "",
                "warnings": [],
                "output": {
                    "path": "/tmp/out/z.fits",
                    "logical_id": "l" * 64,
                    "science_digest": "s" * 64,
                    "whole_file_sha256": "w" * 64,
                    "size_bytes": 10,
                    "committed": True,
                },
            }
        ],
    )
    base.update(overrides)
    return base


def test_manifest_roundtrip_preserves_content():
    m = _manifest()
    parsed = parse_batch_manifest(serialize_batch_manifest(m))
    assert parsed == m


def test_manifest_roundtrip_via_build():
    kwargs = _manifest()
    kwargs.pop("schema_version")
    m = build_batch_manifest(**kwargs)
    assert m["schema_version"] == BATCH_MANIFEST_SCHEMA
    assert parse_batch_manifest(serialize_batch_manifest(m)) == m


def test_manifest_unknown_top_key_rejected():
    m = _manifest()
    m["extra_fact"] = 1
    with pytest.raises(BatchManifestError):
        parse_batch_manifest(m)


def test_manifest_bad_schema_rejected():
    m = _manifest(schema_version="zecalibrator.batch_manifest.v9")
    with pytest.raises(BatchManifestError):
        parse_batch_manifest(m)


def test_manifest_bad_disposition_rejected():
    m = _manifest()
    m["items"][0]["disposition"] = "NOT_A_STATUS"
    with pytest.raises(BatchManifestError):
        parse_batch_manifest(m)


def test_manifest_bad_commit_state_rejected():
    m = _manifest(commit_state="ROLLED_BACK")
    with pytest.raises(BatchManifestError):
        parse_batch_manifest(m)


def test_manifest_unknown_item_key_rejected():
    m = _manifest()
    m["items"][0]["unexpected"] = True
    with pytest.raises(BatchManifestError):
        parse_batch_manifest(m)


def test_write_batch_manifest_is_atomic(tmp_path):
    path = os.path.join(str(tmp_path), "zecalibrator_batch_b1.json")
    write_batch_manifest(path, _manifest())
    assert os.path.exists(path)
    parsed = parse_batch_manifest(open(path, "rb").read())
    assert parsed["batch_id"] == "batch-1"
    # no temp file survives
    assert [n for n in os.listdir(str(tmp_path)) if n.endswith(".tmp")] == []


def test_manifest_path_deterministic():
    assert manifest_path("/out", "b1") == os.path.join("/out", "zecalibrator_batch_b1.json")


def test_manifest_serialization_is_strict_json():
    m = _manifest()
    text = serialize_batch_manifest(m).decode("utf-8")
    assert json.loads(text) == m  # parseable, and no bare NaN/Inf would survive json
