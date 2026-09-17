"""Batch manifest serialize/parse/write (PROVENANCE §4, §5, §2.7).

A structured-JSON manifest that indexes a bounded batch run: batch/operation
identity, ordered input list, per-input plan id, per-item disposition, output
locator + science digest + final whole-file SHA-256, commit state, structured
error/cancel reason, and the schema/product/API/science/matching-policy
versions. It is written atomically **last** (temp + ``os.replace`` in the same
directory); it is never log-text dependent and never self-referenced.

The final whole-file SHA-256 recorded here is computed after output closure by
the writer, so no manifest field is an input to itself.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Mapping, Optional, Sequence

BATCH_MANIFEST_SCHEMA = "zecalibrator.batch_manifest.v1"

_COMMIT_STATES = ("COMMITTED", "CANCELLED")
_BATCH_STATUSES = ("COMPLETED", "PARTIAL", "CANCELLED")

_TOP_KEYS = frozenset({
    "schema_version",
    "batch_id",
    "operation_id",
    "api_version",
    "product_version",
    "science_contract",
    "matching_policy",
    "provenance_schema",
    "decoder_version",
    "commit_state",
    "batch_status",
    "destination",
    "inputs",
    "items",
})

_ITEM_KEYS = frozenset({
    "index",
    "disposition",
    "plan_id",
    "input_identity",
    "reason_code",
    "reason_details",
    "warnings",
    "output",
})

_OUTPUT_KEYS = frozenset({
    "path",
    "logical_id",
    "science_digest",
    "whole_file_sha256",
    "size_bytes",
    "committed",
})


class BatchManifestError(ValueError):
    """A malformed/unsupported batch manifest was supplied."""


def _reject_unknown(d: Mapping, allowed: frozenset, path: str) -> None:
    unknown = set(d.keys()) - allowed
    if unknown:
        raise BatchManifestError(f"unknown key(s) at {path}: {sorted(unknown)}")


def build_batch_manifest(
    *,
    batch_id: str,
    operation_id: str,
    api_version: str,
    product_version: str,
    science_contract: str,
    matching_policy: str,
    provenance_schema: str,
    decoder_version: str,
    commit_state: str,
    batch_status: str,
    destination: Optional[str],
    inputs: Sequence[Mapping[str, object]],
    items: Sequence[Mapping[str, object]],
) -> dict:
    """Build the canonical manifest dict (all values strict-JSON safe)."""
    if commit_state not in _COMMIT_STATES:
        raise BatchManifestError(f"invalid commit_state {commit_state!r}")
    if batch_status not in _BATCH_STATUSES:
        raise BatchManifestError(f"invalid batch_status {batch_status!r}")
    return {
        "schema_version": BATCH_MANIFEST_SCHEMA,
        "batch_id": batch_id,
        "operation_id": operation_id,
        "api_version": api_version,
        "product_version": product_version,
        "science_contract": science_contract,
        "matching_policy": matching_policy,
        "provenance_schema": provenance_schema,
        "decoder_version": decoder_version,
        "commit_state": commit_state,
        "batch_status": batch_status,
        "destination": destination,
        "inputs": [dict(i) for i in inputs],
        "items": [dict(i) for i in items],
    }


def serialize_batch_manifest(manifest: Mapping[str, object]) -> bytes:
    """Serialize a manifest to strict UTF-8 JSON bytes (no bare NaN/Inf)."""
    return json.dumps(
        dict(manifest), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _parse_text(data) -> dict:
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    elif isinstance(data, str):
        text = data
    else:
        raise BatchManifestError("manifest must be bytes or str")

    def _reject_constant(name: str) -> None:
        raise BatchManifestError(f"non-finite JSON constant {name!r} in manifest")

    value = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise BatchManifestError("manifest must decode to a JSON object")
    return value


def _validate_manifest(d: dict) -> dict:
    _reject_unknown(d, _TOP_KEYS, "manifest")
    if d.get("schema_version") != BATCH_MANIFEST_SCHEMA:
        raise BatchManifestError(f"unsupported manifest schema {d.get('schema_version')!r}")
    if not isinstance(d.get("batch_id"), str) or not d["batch_id"]:
        raise BatchManifestError("batch_id must be a non-empty string")
    if not isinstance(d.get("operation_id"), str) or not d["operation_id"]:
        raise BatchManifestError("operation_id must be a non-empty string")
    if d.get("commit_state") not in _COMMIT_STATES:
        raise BatchManifestError(f"invalid commit_state {d.get('commit_state')!r}")
    if d.get("batch_status") not in _BATCH_STATUSES:
        raise BatchManifestError(f"invalid batch_status {d.get('batch_status')!r}")
    for key in (
        "api_version", "product_version", "science_contract",
        "matching_policy", "provenance_schema", "decoder_version",
    ):
        if not isinstance(d.get(key), str) or not d[key]:
            raise BatchManifestError(f"{key} must be a non-empty string")
    if not isinstance(d.get("inputs"), list):
        raise BatchManifestError("inputs must be a list")
    if not isinstance(d.get("items"), list):
        raise BatchManifestError("items must be a list")
    for i, item in enumerate(d["items"]):
        _reject_unknown(item, _ITEM_KEYS, f"items[{i}]")
        if not isinstance(item.get("index"), int):
            raise BatchManifestError(f"items[{i}].index must be an int")
        if item.get("disposition") not in (
            "COMPLETED", "COMPLETED_WITH_WARNINGS", "SKIPPED", "FAILED", "CANCELLED",
        ):
            raise BatchManifestError(f"items[{i}].disposition invalid: {item.get('disposition')!r}")
        if item.get("output") is not None:
            out = item["output"]
            if not isinstance(out, dict):
                raise BatchManifestError(f"items[{i}].output must be an object or null")
            _reject_unknown(out, _OUTPUT_KEYS, f"items[{i}].output")
            for key in ("path", "logical_id", "science_digest", "whole_file_sha256"):
                if not isinstance(out.get(key), str) or not out[key]:
                    raise BatchManifestError(f"items[{i}].output.{key} must be a non-empty string")
            if not isinstance(out.get("size_bytes"), int):
                raise BatchManifestError(f"items[{i}].output.size_bytes must be an int")
    return d


def parse_batch_manifest(data) -> dict:
    """Parse and strictly validate a manifest (bytes/str/dict accepted)."""
    if isinstance(data, dict):
        d = dict(data)
    else:
        d = _parse_text(data)
    return _validate_manifest(d)


def write_batch_manifest(path: str, manifest: Mapping[str, object]) -> None:
    """Write a manifest atomically (temp + ``os.replace``) in its destination."""
    data = serialize_batch_manifest(manifest)
    parent = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(
        prefix=".zecalibrator-manifest-", suffix=".json.tmp", dir=parent
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def manifest_path(destination: str, batch_id: str) -> str:
    """Deterministic manifest path for a batch id inside the destination."""
    return os.path.join(destination, f"zecalibrator_batch_{batch_id}.json")


__all__ = [
    "BATCH_MANIFEST_SCHEMA",
    "BatchManifestError",
    "build_batch_manifest",
    "manifest_path",
    "parse_batch_manifest",
    "serialize_batch_manifest",
    "write_batch_manifest",
]
