"""P3C-3 — candidate freeze (§28): freeze the candidates *before* any evaluation.

The freeze is a snapshot produced **before** any qualification metric is ever
computed. It pins, byte-for-byte:

* the three candidate identifiers, their descriptions (the "formula" in prose)
  and their numerical search parameters (``RESEARCH_CANDIDATE_PARAMETER``);
* the SHA-256 hashes of the inference-side source code (the modules that carry
  the candidate rules and the adapter convention);
* the DEVELOPMENT manifest hash and the QUALIFICATION manifest hash;
* the version of the metrics contract, the reason-code vocabulary and the
  evidence-bridge adapter convention.

After the freeze, no candidate may be modified on the strength of qualification
results. That immutability is **enforced structurally**, not promised: the
qualification runner re-derives the source hashes and the candidate-parameter
hashes from the live modules and refuses to measure if they diverge from the
freeze (see :class:`research.p3c.qualification_runner.CandidateDriftError`). A
new candidate would carry a new ``candidate_id`` / version and its own
origin — it could never be silently substituted into a frozen snapshot.

The freeze is a pure function of (candidate configs, source files, DEVELOPMENT
manifest, QUALIFICATION manifest) — it reads **no** qualification result and no
holdout, and is byte-reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

from research.p3b.qualification_policy import REASON_CODE_VERSION

from .development_corpus import build_development_manifest
from .evidence_bridge import ADAPTER_VERSION
from .inference_candidates import (
    CANDIDATES,
    SCHEMA_VERSION,
    InferenceCandidateConfig,
)
from .qualification_corpus import build_qualification_manifest

FREEZE_SCHEMA = "zecalibrator-p3c-candidate-freeze"
FREEZE_VERSION = 1

# Version of the qualification metrics contract (this campaign). Kept distinct
# from the reason-code version (qualification_policy) and the adapter version
# (evidence_bridge) so each provenance is versioned independently.
METRICS_CONTRACT_VERSION = "p3c-qualification-metrics-1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload) -> str:
    """Byte-stable JSON serialisation (key-sorted, compact, ASCII)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# Source-code hashing (§1 / §44): the exact inference-side code under freeze
# ---------------------------------------------------------------------------

import research.p3c.evidence_bridge as _eb  # noqa: E402
import research.p3c.inference_candidates as _ic  # noqa: E402
import research.p3c.inference_contract as _co  # noqa: E402

# Stable short names → absolute source paths of the inference-side modules. Only
# the inference-side code is hashed: the oracle (truth side) and the harness are
# deliberately *not* part of the candidate freeze identity.
_SOURCE_MODULES = {
    "inference_candidates.py": _ic,
    "evidence_bridge.py": _eb,
    "inference_contract.py": _co,
}


def source_hashes() -> Mapping[str, str]:
    """SHA-256 of each inference-side source file, keyed by short name."""
    out = {}
    for name, mod in sorted(_SOURCE_MODULES.items()):
        path = Path(mod.__file__)
        out[name] = _sha256_bytes(path.read_bytes())
    return out


def _candidate_config_hash(cfg: InferenceCandidateConfig) -> str:
    """SHA-256 of one candidate's canonical config (id, version, description, params)."""
    return _sha256_text(_canonical_json(cfg.as_dict()))


# ---------------------------------------------------------------------------
# The frozen snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateFreeze:
    """The immutable freeze snapshot + its canonical hash."""

    freeze_hash: str
    schema: str
    version: int
    metrics_contract_version: str
    reason_code_version: str
    adapter_version: str
    candidate_schema_version: str
    development_manifest_hash: str
    qualification_manifest_hash: str
    source_hashes: Mapping[str, str]
    candidates: Tuple[Mapping, ...]  # frozen candidate configs, in CANDIDATES order

    def to_dict(self) -> dict:
        return {
            "freeze_hash": self.freeze_hash,
            "freeze_schema": self.schema,
            "freeze_version": self.version,
            "metrics_contract_version": self.metrics_contract_version,
            "reason_code_version": self.reason_code_version,
            "adapter_version": self.adapter_version,
            "candidate_schema_version": self.candidate_schema_version,
            "development_manifest_hash": self.development_manifest_hash,
            "qualification_manifest_hash": self.qualification_manifest_hash,
            "source_hashes": dict(self.source_hashes),
            "candidates": [dict(c) for c in self.candidates],
        }


def _freeze_payload() -> dict:
    dev_manifest = build_development_manifest()
    qual_manifest = build_qualification_manifest()
    return {
        "freeze_schema": FREEZE_SCHEMA,
        "freeze_version": FREEZE_VERSION,
        "metrics_contract_version": METRICS_CONTRACT_VERSION,
        "reason_code_version": REASON_CODE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "candidate_schema_version": SCHEMA_VERSION,
        "development_manifest_hash": dev_manifest.manifest_hash,
        "qualification_manifest_hash": qual_manifest.manifest_hash,
        "source_hashes": source_hashes(),
        "candidates": [c.as_dict() for c in CANDIDATES],
    }


def freeze_candidates() -> CandidateFreeze:
    """Produce the freeze snapshot (no qualification evaluation is involved).

    Pure and reproducible: the same candidate configs, source files, and
    manifests always yield the same ``freeze_hash``. This function reads no
    qualification result and no holdout.
    """
    payload = _freeze_payload()
    # The hash covers every field except the hash itself.
    hash_payload = {k: v for k, v in payload.items() if k != "freeze_hash"}
    freeze_hash = _sha256_text(_canonical_json(hash_payload))
    return CandidateFreeze(
        freeze_hash=freeze_hash,
        schema=payload["freeze_schema"],
        version=payload["freeze_version"],
        metrics_contract_version=payload["metrics_contract_version"],
        reason_code_version=payload["reason_code_version"],
        adapter_version=payload["adapter_version"],
        candidate_schema_version=payload["candidate_schema_version"],
        development_manifest_hash=payload["development_manifest_hash"],
        qualification_manifest_hash=payload["qualification_manifest_hash"],
        source_hashes=payload["source_hashes"],
        candidates=tuple(payload["candidates"]),
    )


def live_candidate_config_hashes() -> Mapping[str, str]:
    """SHA-256 of each live candidate config (for the drift check)."""
    return {c.candidate_id: _candidate_config_hash(c) for c in CANDIDATES}


__all__ = [
    "FREEZE_SCHEMA",
    "FREEZE_VERSION",
    "METRICS_CONTRACT_VERSION",
    "CandidateFreeze",
    "freeze_candidates",
    "live_candidate_config_hashes",
    "source_hashes",
]
