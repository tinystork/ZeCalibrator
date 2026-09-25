"""P3C-4 — temporal-persistence evidence contract (research-only, non-packaged).

LOT 1 defines, before any inference code, what *temporal persistence evidence at
a fixed sensor coordinate* means. This package holds the contract
(:mod:`research.p3c4.temporal_evidence`), the structured result type, the allowed
measurement sources and the temporal evaluator's normative definition. It
implements **no inference rule** (no candidate, no threshold, no score) — that is
LOT 2.

It reads **no truth**: importing this package pulls no
``research.p3b.*`` module and no truth-side oracle into ``sys.modules`` — enforced
by ``tests/p3c4/test_truth_leak_guard.py``.
"""

from .temporal_evidence import (  # noqa: F401
    YES,
    NO,
    UNDETERMINED,
    ALLOWED_MEASUREMENT_SOURCES,
    MIN_SUPPORTING_GROUPS,
    MIN_SUPPORTING_EPOCHS,
    MIN_VALID_SAMPLES,
    TEMPORAL_CONTRACT_PARAMETERS,
    GroupSignature,
    TemporalSignature,
    TemporalPersistenceEvidence,
    assess_temporal_persistence,
    InvalidTemporalStateError,
    CensoredContributionError,
    DisallowedEvidenceSourceError,
)

__all__ = [
    "YES",
    "NO",
    "UNDETERMINED",
    "ALLOWED_MEASUREMENT_SOURCES",
    "MIN_SUPPORTING_GROUPS",
    "MIN_SUPPORTING_EPOCHS",
    "MIN_VALID_SAMPLES",
    "TEMPORAL_CONTRACT_PARAMETERS",
    "GroupSignature",
    "TemporalSignature",
    "TemporalPersistenceEvidence",
    "assess_temporal_persistence",
    "InvalidTemporalStateError",
    "CensoredContributionError",
    "DisallowedEvidenceSourceError",
]
