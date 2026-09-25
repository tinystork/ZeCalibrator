"""P3C-4 — corrected ``conflicting_evidence`` inference (versioned candidate family).

Research-only, internal, non-public. This module adds a **new, versioned**
candidate family that corrects the frozen P3C ``conflicting_evidence`` semantics
while leaving the frozen ``p3c-*`` and ``p3c4-*`` candidates untouched. It is
the inference-side counterpart of :mod:`research.p3c4.conflicting_evidence`.

What it does and does not do:

* It re-derives **only** ``conflicting_evidence`` using the corrected rule
  (:func:`research.p3c4.conflicting_evidence.assess_conflicting_evidence`). The
  other five SENSOR-EVIDENCE facts — including the temporally re-derived
  ``persisted_at_same_sensor_coord`` — come **verbatim** from the existing
  ``p3c4-*`` temporal candidate named by ``base_temporal_candidate_id``.
* It introduces **no** product threshold and **no** new spatial SNR or
  dispersion rule. Its two parameters (``conflict_amplitude_floor``,
  ``min_conflict_reversals``) are ``RESEARCH_CANDIDATE_PARAMETER`` values that
  live only in these new ``*-cefix`` candidate configs.
* It never modifies ``research.p3c`` and never edits the frozen ``p3c4-*``
  candidates: the corrected candidates carry new ``candidate_id``s
  (``p3c4-*-cefix``) and their own provenance.

Boundary (SCIENCE §16 / ARCHITECTURE §18): this module reads **no truth**. It
imports only the corrected conflict contract, the inference contract, the
temporal inference side, and the P3C candidate parameter type (the inference
side). It never imports ``research.p3b.*`` truth modules or
``research.p3c.oracle``, never references a class label, scenario name or
expected state, and never touches a corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from research.p3c.inference_candidates import CandidateParam, candidate as p3c_candidate
from research.p3c.inference_contract import (
    DETERMINED,
    AdmissionFacts,
    InferredEvidence,
    InferredField,
    build_inferred_evidence,
)
from research.p3c4.conflicting_evidence import (
    CONFLICT_CONTRACT_VERSION,
    ConflictingEvidence,
    assess_conflicting_evidence,
)
from research.p3c4.temporal_inference import (
    infer_site as p3c4_infer_site,
    temporal_candidate as p3c4_temporal_candidate,
)

# Version of the whole corrected candidate-rule schema (fields/units of a config).
SCHEMA_VERSION = "p3c4-conflicting-inference-candidates-schema-1"

# The one SENSOR-EVIDENCE fact this module owns. The other five come verbatim
# from the p3c4 temporal candidate named by ``base_temporal_candidate_id``.
CONFLICTING_FACT = "conflicting_evidence"


def _p(name: str, value: float, unit: str, rationale: str) -> CandidateParam:
    return CandidateParam(name=name, value=value, unit=unit, rationale=rationale)


@dataclass(frozen=True)
class ConflictingCandidateConfig:
    """A versioned corrected-conflict candidate: a p3c4 temporal base + the
    corrected conflict parameters.

    ``base_temporal_candidate_id`` names the frozen ``p3c4-*`` temporal
    candidate whose five non-conflict facts (including temporally re-derived
    ``persisted_at_same_sensor_coord``) are reused unchanged. ``parameters`` are
    the new corrected-conflict decision parameters.
    """

    candidate_id: str
    version: str
    description: str
    base_temporal_candidate_id: str
    parameters: Tuple[CandidateParam, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not isinstance(self.candidate_id, str):
            raise ValueError("candidate_id must be a non-empty string")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names in {self.candidate_id!r}")

    def param(self, name: str) -> float:
        for p in self.parameters:
            if p.name == name:
                return p.value
        raise KeyError(f"unknown conflicting-evidence parameter {name!r}")

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "version": self.version,
            "description": self.description,
            "base_temporal_candidate_id": self.base_temporal_candidate_id,
            "parameters": {
                p.name: {
                    "value": p.value,
                    "unit": p.unit,
                    "kind": p.kind,
                    "rationale": p.rationale,
                }
                for p in self.parameters
            },
        }


# ---------------------------------------------------------------------------
# Corrected-conflict candidate parameters (RESEARCH_CANDIDATE_PARAMETER — never
# a product threshold, never a re-tune of the frozen P3C budgets).
#
#   conflict_amplitude_floor  — the residual magnitude (ADU) at/above which one
#                               side of a sign crossing counts as "meaningful".
#                               A crossing whose either side sits below the
#                               floor is a noise-driven zero crossing, not
#                               conflict. 50 ADU is ~10x read noise (~4-5 ADU)
#                               and below the weak-anomaly band (~60 ADU), while
#                               genuine post-calibration sign flips are hundreds
#                               to thousands ADU.
#   min_conflict_reversals    — the minimum significant sign reversals before
#                               conflict is declared. A single isolated reversal
#                               is not robust contradictory evidence; 2 mirrors
#                               the frozen baseline's ">=2 crossings => genuine
#                               contradictory evidence" convention.
#
# The SAME two values are applied to every base (not a grid): the correction is
# one physically-justified semantics applied uniformly to the three persistence
# variants of the frozen p3c4 family.
# ---------------------------------------------------------------------------

_CONFLICT_PARAMS = (
    _p("conflict_amplitude_floor", 50.0, "ADU",
       "above read-noise |residual| (~4-5 ADU) so a noise-driven zero crossing never counts as conflict; a genuine post-calibration sign flip has hundreds-to-thousands ADU magnitude"),
    _p("min_conflict_reversals", 2.0, "count",
       "a single isolated sign reversal is not robust contradictory evidence; the residual must reverse sign at least twice (mirrors the frozen baseline's >=2 strict-crossing convention)"),
)

CANDIDATE_CEFIX_BASELINE = ConflictingCandidateConfig(
    candidate_id="p3c4-baseline-cefix",
    version="1",
    description=(
        "Corrected conflicting_evidence: significant post-calibration residual "
        "sign reversals (>=2, each side >=50 ADU) replace the raw sign-crossing "
        "count. Other five facts verbatim from p3c4-baseline."
    ),
    base_temporal_candidate_id="p3c4-baseline",
    parameters=_CONFLICT_PARAMS,
)

CANDIDATE_CEFIX_CONSERVATIVE = ConflictingCandidateConfig(
    candidate_id="p3c4-conservative-cefix",
    version="1",
    description=(
        "Corrected conflicting_evidence on the p3c4-conservative base (presence "
        "bar 150 ADU); conflict semantics identical to the baseline-cefix."
    ),
    base_temporal_candidate_id="p3c4-conservative",
    parameters=_CONFLICT_PARAMS,
)

CANDIDATE_CEFIX_SENSITIVE = ConflictingCandidateConfig(
    candidate_id="p3c4-sensitive-cefix",
    version="1",
    description=(
        "Corrected conflicting_evidence on the p3c4-sensitive base (presence "
        "bar 25 ADU); conflict semantics identical to the baseline-cefix."
    ),
    base_temporal_candidate_id="p3c4-sensitive",
    parameters=_CONFLICT_PARAMS,
)

CONFLICTING_CANDIDATES: Tuple[ConflictingCandidateConfig, ...] = (
    CANDIDATE_CEFIX_BASELINE,
    CANDIDATE_CEFIX_CONSERVATIVE,
    CANDIDATE_CEFIX_SENSITIVE,
)

CONFLICTING_CANDIDATE_IDS: Tuple[str, ...] = tuple(
    c.candidate_id for c in CONFLICTING_CANDIDATES
)


def conflicting_candidate(candidate_id: str) -> ConflictingCandidateConfig:
    for c in CONFLICTING_CANDIDATES:
        if c.candidate_id == candidate_id:
            return c
    raise KeyError(f"unknown conflicting-evidence candidate {candidate_id!r}")


def infer_conflicting(
    cfg: ConflictingCandidateConfig, features
) -> InferredField:
    """Infer ``conflicting_evidence`` from the corrected rule.

    ``features`` is the LOT4 ``SiteFeatures`` (duck-typed): only the canonical
    residual series (``temporal_residual_series``) is read, never a truth label.
    """
    residuals = features.temporal_residual_series
    floor = cfg.param("conflict_amplitude_floor")
    min_rev = int(cfg.param("min_conflict_reversals"))
    result: ConflictingEvidence = assess_conflicting_evidence(
        residuals,
        conflict_amplitude_floor=floor,
        min_conflict_reversals=min_rev,
    )
    return InferredField(
        field=CONFLICTING_FACT,
        value=result.state,
        source=(
            "corrected conflicting evidence (significant post-calibration "
            "residual sign reversal, not a raw sign crossing)"
        ),
        uncertainty=DETERMINED,
        feature_refs=("temporal_residual_series",),
    )


def infer_site(
    cfg: ConflictingCandidateConfig,
    features,
    temporal_features,
    admission: AdmissionFacts,
) -> InferredEvidence:
    """Infer one site's SENSOR-EVIDENCE facts with corrected candidate ``cfg``.

    The five non-conflict facts come **verbatim** from the ``p3c4-*`` temporal
    candidate named by ``cfg.base_temporal_candidate_id`` (whose
    ``persisted_at_same_sensor_coord`` is itself temporally re-derived). Only
    ``conflicting_evidence`` is replaced by the corrected rule.

    Pure and deterministic: a function of ``(cfg, features, temporal_features,
    admission)`` only.
    """
    base_cfg = p3c4_temporal_candidate(cfg.base_temporal_candidate_id)
    base_evidence = p3c4_infer_site(base_cfg, features, temporal_features, admission)

    conflicting = infer_conflicting(cfg, features)

    fields = tuple(
        conflicting if f.field == CONFLICTING_FACT else f
        for f in base_evidence.sensor_evidence
    )
    return build_inferred_evidence(cfg.candidate_id, admission, fields)


__all__ = [
    "SCHEMA_VERSION",
    "CONFLICT_CONTRACT_VERSION",
    "CONFLICTING_FACT",
    "ConflictingCandidateConfig",
    "CANDIDATE_CEFIX_BASELINE",
    "CANDIDATE_CEFIX_CONSERVATIVE",
    "CANDIDATE_CEFIX_SENSITIVE",
    "CONFLICTING_CANDIDATES",
    "CONFLICTING_CANDIDATE_IDS",
    "conflicting_candidate",
    "infer_conflicting",
    "infer_site",
]
