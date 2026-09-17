"""Canonical identity digests — ``zecalibrator.digest.v1`` (PROVENANCE §2.3–§2.5).

This is the *production* implementation of the frozen G1 digest contract. It is
byte-identical in semantics to the pinned reference in
``research/phase1/contract_witness.py`` (the reference is a standalone research
script and is **not** imported at runtime). Three distinct acyclic digests are
provided, each over an explicit projection (an enumerated allowlist of field
paths), never a blacklist:

* ``science_digest`` — canonical byte serialization of the calibrated float32
  data plane and uint16 DQ mask only (§2.4).
* ``descriptor_digest`` — canonical JSON of the explicit descriptor projection
  (§2.3 item 2).
* ``plan_digest`` — canonical JSON of the explicit plan projection with each
  master binding reduced to its hashed identity fields (§2.3 item 3).

No digest is ever an input to itself and no output digest appears inside its own
output.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Mapping

import numpy as np

_DTYPE_TOKENS: Mapping[tuple[str, int], str] = {
    ("f", 4): "float32",
    ("f", 8): "float64",
    ("u", 2): "uint16",
    ("u", 1): "uint8",
    ("i", 2): "int16",
    ("u", 4): "uint32",
    ("i", 4): "int32",
    ("u", 8): "uint64",
    ("i", 8): "int64",
}

# Fixed big-endian quiet-NaN bit patterns (literal bytes), never derived from the
# host ``float('nan')`` encoding (PROVENANCE §2.4 item 3).
_F32_NAN = bytes([0x7F, 0xC0, 0x00, 0x00])
_F64_NAN = bytes([0x7F, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])


class DigestSerializationError(ValueError):
    """Raised when a value cannot be serialized into a canonical digest."""


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def dtype_token(dtype) -> str:
    d = np.dtype(dtype)
    try:
        return _DTYPE_TOKENS[(d.kind, d.itemsize)]
    except KeyError as exc:  # pragma: no cover - defensive
        raise DigestSerializationError(f"unsupported dtype for canonical digest: {dtype!r}") from exc


def canonical_array_bytes(arr) -> bytes:
    """Return ``CANONICAL(A)`` — the length-prefixed byte record (§2.4)."""
    a = np.asarray(arr).astype(np.dtype(arr.dtype).newbyteorder(">"))
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        a = a.copy()
        if a.dtype.itemsize == 4:
            nanval = np.frombuffer(_F32_NAN, dtype=">f4")[0]
        elif a.dtype.itemsize == 8:
            nanval = np.frombuffer(_F64_NAN, dtype=">f8")[0]
        else:
            raise DigestSerializationError("unsupported float size for canonical digest")
        a[~np.isfinite(a)] = nanval
        neg_zero = (a == 0.0) & np.signbit(a)
        if neg_zero.any():
            a[neg_zero] = 0.0
    record = (
        b"ZCALDIG1\n"
        + dtype_token(a.dtype).encode("ascii")
        + b"\n"
        + ",".join(str(s) for s in a.shape).encode("ascii")
        + b"\n"
        + a.tobytes()
    )
    return struct.pack(">Q", len(record)) + record


def science_digest(data, dq) -> str:
    """Science digest over the calibrated float32 data plane and uint16 DQ mask."""
    return sha256_hex(canonical_array_bytes(data) + canonical_array_bytes(dq))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON subset for descriptor/plan digests (PROVENANCE §2.5).

    Compact separators, sorted object keys, ``ensure_ascii=False`` strings,
    finite numbers only (floats via Python ``repr``), integers as decimal
    integers, lists/tuples preserve order, ``null`` for explicit unknown, and
    **no** trailing newline.
    """
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, np.integer)):
        return str(int(obj))
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            raise DigestSerializationError("non-finite value in canonical metadata JSON")
        return repr(v)
    if isinstance(obj, str):
        import json

        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canonical_json(x) for x in obj) + "]"
    if isinstance(obj, dict):
        import json

        items = sorted(obj.items(), key=lambda kv: kv[0])
        return "{" + ",".join(
            json.dumps(k, ensure_ascii=False) + ":" + canonical_json(v)
            for k, v in items
        ) + "}"
    raise DigestSerializationError(f"unsupported canonical JSON type: {type(obj)!r}")


def _deep_get(obj: Mapping[str, Any], path: str) -> Any:
    cur = obj
    for p in path.split("."):
        if isinstance(cur, Mapping) and p in cur:
            cur = cur[p]
        else:
            raise KeyError(path)
    return cur


def _project(obj: Mapping[str, Any], paths) -> dict:
    """Build a nested dict projection from an explicit list of dot-paths."""
    root: dict = {}
    for p in paths:
        parts = p.split(".")
        node = root
        for seg in parts[:-1]:
            node = node.setdefault(seg, {})
        node[parts[-1]] = _deep_get(obj, p)
    return root


# Explicit projections (allowlists), frozen at G1 (PROVENANCE §2.3). The
# descriptor projection is the complete acyclic field set whose digest is
# ``descriptor_id``.
DESCRIPTOR_PROJECTION: tuple[str, ...] = (
    "master_type",
    "bias_state",
    "pixel_domain",
    "physical_units",
    "flat_form",
    "normalization_algorithm",
    "normalization_scalars",
    "geometry",
    "detector.detector_instance_id",
    "detector.detector_model",
    "detector.serial",
    "acquisition.gain",
    "acquisition.offset",
    "acquisition.readout_mode",
    "acquisition.adc_mode",
    "acquisition.temperature_c",
    "acquisition.exposure_s",
    "acquisition.saturation_limit_adu",
    "acquisition.saturation_evidence",
    "optical_train_id",
    "filter",
    "content_sha256",
    "size_bytes",
    "hdu",
    "mask_identity",
    "processing_provenance",
    "validity_evidence",
)

PLAN_PROJECTION: tuple[str, ...] = (
    "request.additive_mode",
    "request.flat_mode",
    "light_constraints.geometry",
    "light_constraints.detector",
    "light_constraints.acquisition",
    "light_constraints.optical",
    "light_constraints.raw_domain_declaration",
    "policy_parameters.exposure_tolerance",
    "policy_parameters.temperature_tolerance",
    "policy_parameters.flat_quality_policy",
    "versions.science_contract",
    "versions.decoder",
    "versions.provenance_schema",
    "versions.matching_policy",
)

MASTER_BINDING_PROJECTION: tuple[str, ...] = (
    "descriptor_id",
    "content_sha256",
    "size_bytes",
    "hdu",
    "mask_identity",
)


def descriptor_digest(fields: Mapping[str, Any]) -> str:
    """Descriptor identity over the explicit descriptor projection."""
    return sha256_hex(canonical_json(_project(fields, DESCRIPTOR_PROJECTION)).encode("utf-8"))


def descriptor_snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    """Descriptor identity is the digest of the complete acyclic snapshot (same
    field set as ``descriptor_digest``); a tampered snapshot recomputes to a
    different digest and is therefore rejected."""
    return descriptor_digest(snapshot)


def plan_digest(fields: Mapping[str, Any]) -> str:
    """Plan identity over the explicit plan projection.

    Each master binding is projected to its hashed identity fields only (the
    retrieval locator is excluded); incidental execution/retrieval fields are
    never part of the digest.
    """
    proj = _project(fields, PLAN_PROJECTION)
    masters = fields.get("masters", {}) if isinstance(fields, Mapping) else {}
    proj["masters"] = {
        role: _project(binding, MASTER_BINDING_PROJECTION)
        for role, binding in sorted(masters.items())
    }
    return sha256_hex(canonical_json(proj).encode("utf-8"))


__all__ = [
    "DESCRIPTOR_PROJECTION",
    "DigestSerializationError",
    "MASTER_BINDING_PROJECTION",
    "PLAN_PROJECTION",
    "canonical_array_bytes",
    "canonical_json",
    "descriptor_digest",
    "descriptor_snapshot_digest",
    "dtype_token",
    "plan_digest",
    "science_digest",
    "sha256_hex",
]
