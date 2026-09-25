"""P3C-4 LOT 5 — candidate freeze (GEL-2): freeze *before* any QUALIFICATION-2 metric.

Research-only, internal, non-public. This is the P3C-4 counterpart of
:mod:`research.p3c.candidate_freeze`, but it freezes the **temporal** candidate
family (``p3c4-baseline`` / ``p3c4-conservative`` / ``p3c4-sensitive``) and pins
the provenance of the temporal-persistence campaign. The freeze is produced
**before** a single QUALIFICATION-2 metric is computed and pins, byte-for-byte:

* the three temporal candidate identifiers, their descriptions and their
  ``RESEARCH_CANDIDATE_PARAMETER`` values (``presence_adu`` /
  ``departure_offset_px``) plus the frozen P3C base candidate each references;
* the SHA-256 hashes of the **inference-side** source code — the temporal
  contract, the temporal measurement and the temporal rule, plus the frozen P3C
  inference side those rules delegate to (the truth side and the harness are
  deliberately *not* part of the freeze identity, mirroring P3C);
* the DEVELOPMENT manifest hash and the QUALIFICATION-2 manifest hash;
* the version provenance: temporal contract version, evidence-bridge adapter
  version, metrics contract version, reason-code version, candidate schema
  version.

Refusal of drift (§32 / §53): after the freeze, any change — source code, the
**set** of candidates, a candidate's **config**, or the temporal rule — is a
**typed error** (:class:`CandidateDriftError`), never a warning. The
QUALIFICATION-2 runner re-derives the source hashes, the candidate configs and
the version provenance from the live modules and refuses to measure if anything
diverges from the freeze. A new candidate would carry a new ``candidate_id`` /
version and its own origin — it could never be silently substituted into a
frozen snapshot.

The freeze is a pure function of (temporal candidate configs, inference-side
source files, DEVELOPMENT manifest, QUALIFICATION-2 manifest, version
provenance) — it reads **no** qualification result and no holdout, and is
byte-reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

from research.p3b.qualification_policy import REASON_CODE_VERSION

from .qualification2_corpus import build_qualification2_manifest
from .temporal_evidence import TEMPORAL_CONTRACT_VERSION
from .temporal_inference import (
    SCHEMA_VERSION,
    TEMPORAL_CANDIDATES,
    TemporalCandidateConfig,
)

FREEZE_SCHEMA = "zecalibrator-p3c4-candidate-freeze"
FREEZE_VERSION = 1

# Version of the QUALIFICATION-2 metrics contract (this campaign). Kept distinct
# from the reason-code version (qualification_policy), the adapter version
# (evidence_bridge) and the temporal-contract version (temporal_evidence) so
# each provenance is versioned independently.
METRICS_CONTRACT_VERSION = "p3c4-qualification-metrics-1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload) -> str:
    """Byte-stable JSON serialisation (key-sorted, compact, ASCII)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# Source-code hashing (§31 / §44): the exact inference-side code under freeze.
#
# Only the inference side is hashed: the temporal contract, the temporal
# measurement, the temporal rule, and the frozen P3C inference side the rule
# delegates to. The truth side (corpus.py, qualification2_corpus.py, the oracle)
# and the harness are deliberately *not* part of the candidate freeze identity.
# ---------------------------------------------------------------------------

import research.p3c.evidence_bridge as _eb  # noqa: E402
import research.p3c.inference_candidates as _ic  # noqa: E402
import research.p3c.inference_contract as _co  # noqa: E402
import research.p3c4.temporal_evidence as _te  # noqa: E402
import research.p3c4.temporal_features as _tf  # noqa: E402
import research.p3c4.temporal_inference as _ti  # noqa: E402

# Stable short names → absolute source paths of the inference-side modules.
_SOURCE_MODULES = {
    "temporal_evidence.py": _te,
    "temporal_features.py": _tf,
    "temporal_inference.py": _ti,
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


def _candidate_config_hash(cfg: TemporalCandidateConfig) -> str:
    """SHA-256 of one candidate's canonical config (id, version, description, base, params)."""
    return _sha256_text(_canonical_json(cfg.as_dict()))


def _live_version_provenance() -> Mapping[str, str]:
    """The version provenance re-derived from the live modules (drift check)."""
    from research.p3c.evidence_bridge import ADAPTER_VERSION

    return {
        "temporal_contract_version": TEMPORAL_CONTRACT_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "metrics_contract_version": METRICS_CONTRACT_VERSION,
        "reason_code_version": REASON_CODE_VERSION,
        "candidate_schema_version": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# The frozen snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateFreeze:
    """The immutable P3C-4 freeze snapshot + its canonical hash."""

    freeze_hash: str
    schema: str
    version: int
    metrics_contract_version: str
    reason_code_version: str
    adapter_version: str
    temporal_contract_version: str
    candidate_schema_version: str
    development_manifest_hash: str
    qualification2_manifest_hash: str
    source_hashes: Mapping[str, str]
    candidates: Tuple[Mapping, ...]  # frozen temporal candidate configs, in TEMPORAL_CANDIDATES order

    def to_dict(self) -> dict:
        return {
            "freeze_hash": self.freeze_hash,
            "freeze_schema": self.schema,
            "freeze_version": self.version,
            "metrics_contract_version": self.metrics_contract_version,
            "reason_code_version": self.reason_code_version,
            "adapter_version": self.adapter_version,
            "temporal_contract_version": self.temporal_contract_version,
            "candidate_schema_version": self.candidate_schema_version,
            "development_manifest_hash": self.development_manifest_hash,
            "qualification2_manifest_hash": self.qualification2_manifest_hash,
            "source_hashes": dict(self.source_hashes),
            "candidates": [dict(c) for c in self.candidates],
        }


def _freeze_payload() -> dict:
    from research.p3c.development_corpus import build_development_manifest
    from research.p3c.evidence_bridge import ADAPTER_VERSION

    dev_manifest = build_development_manifest()
    q2_manifest = build_qualification2_manifest()
    return {
        "freeze_schema": FREEZE_SCHEMA,
        "freeze_version": FREEZE_VERSION,
        "metrics_contract_version": METRICS_CONTRACT_VERSION,
        "reason_code_version": REASON_CODE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "temporal_contract_version": TEMPORAL_CONTRACT_VERSION,
        "candidate_schema_version": SCHEMA_VERSION,
        "development_manifest_hash": dev_manifest.manifest_hash,
        "qualification2_manifest_hash": q2_manifest["manifest_hash"],
        "source_hashes": source_hashes(),
        "candidates": [c.as_dict() for c in TEMPORAL_CANDIDATES],
    }


def freeze_candidates() -> CandidateFreeze:
    """Produce the P3C-4 freeze snapshot (no QUALIFICATION-2 evaluation involved).

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
        temporal_contract_version=payload["temporal_contract_version"],
        candidate_schema_version=payload["candidate_schema_version"],
        development_manifest_hash=payload["development_manifest_hash"],
        qualification2_manifest_hash=payload["qualification2_manifest_hash"],
        source_hashes=payload["source_hashes"],
        candidates=tuple(payload["candidates"]),
    )


def live_candidate_config_hashes() -> Mapping[str, str]:
    """SHA-256 of each live temporal candidate config (for the drift check)."""
    return {c.candidate_id: _candidate_config_hash(c) for c in TEMPORAL_CANDIDATES}


# ---------------------------------------------------------------------------
# Drift refusal (§32) — typed error, never a warning
# ---------------------------------------------------------------------------


class CandidateDriftError(RuntimeError):
    """The live candidates/source differ from the frozen snapshot — refuse to measure.

    Raised by the QUALIFICATION-2 runner when the current source hashes, the
    candidate set, a candidate config, or the version provenance do not match
    the freeze. This is the structural enforcement of §32: no source edit, no
    candidate-set change, no candidate-config change and no temporal-rule change
    may occur after the freeze, because any such change makes the drift check
    fail and the campaign refuses to run.
    """


def _assert_not_drifted(freeze: CandidateFreeze) -> None:
    """Raise :class:`CandidateDriftError` if live modules differ from ``freeze``."""
    if dict(source_hashes()) != dict(freeze.source_hashes):
        raise CandidateDriftError(
            "inference-side source code differs from the frozen snapshot; "
            "refusing to measure (no source edit or temporal-rule change after freeze)"
        )
    frozen_by_id = {c["candidate_id"]: c for c in freeze.candidates}
    if set(frozen_by_id) != {c.candidate_id for c in TEMPORAL_CANDIDATES}:
        raise CandidateDriftError(
            "the frozen candidate set differs from the live candidate set"
        )
    for c in TEMPORAL_CANDIDATES:
        if c.as_dict() != frozen_by_id[c.candidate_id]:
            raise CandidateDriftError(
                f"candidate {c.candidate_id!r} config differs from the frozen "
                "snapshot; refusing to measure"
            )
    if dict(_live_version_provenance()) != {
        "temporal_contract_version": freeze.temporal_contract_version,
        "adapter_version": freeze.adapter_version,
        "metrics_contract_version": freeze.metrics_contract_version,
        "reason_code_version": freeze.reason_code_version,
        "candidate_schema_version": freeze.candidate_schema_version,
    }:
        raise CandidateDriftError(
            "version provenance differs from the frozen snapshot; refusing to measure"
        )


__all__ = [
    "FREEZE_SCHEMA",
    "FREEZE_VERSION",
    "METRICS_CONTRACT_VERSION",
    "CandidateFreeze",
    "CandidateDriftError",
    "freeze_candidates",
    "live_candidate_config_hashes",
    "source_hashes",
]
