"""Versioned Bad Pixel Database product vocabulary.

This module promotes the science-contract vocabulary (``docs/SCIENCE_CONTRACT.md``
§13) into a *product* vocabulary that the ``zecalibrator.bpm`` package owns and
tests independently. Nothing here imports ``research/*``; the retained contracts
are re-expressed in product terms and versioned as ``zecalibrator.bpm.v1``.

Only vocabulary lives here — no thresholds, no budgets, no reconstruction
operator, no pixel application (all DEFERRED to later owner gates / P4-A2).
"""

from __future__ import annotations

#: The BPM storage/identity schema version. Distinct from the product version,
#: public API version, provenance schema, matching-policy version and science
#: contract version (ARCHITECTURE §2). Bumping this is a breaking change.
BPM_SCHEMA_VERSION = "zecalibrator.bpm.v1"

# ---------------------------------------------------------------------------
# Revision states — a revision is immutable; an update is a NEW revision.
# ---------------------------------------------------------------------------
REVISION_STATE_CANDIDATE = "candidate"
REVISION_STATE_QUALIFIED = "qualified"
REVISION_STATE_PROMOTED = "promoted"

REVISION_STATES: tuple[str, ...] = (
    REVISION_STATE_CANDIDATE,
    REVISION_STATE_QUALIFIED,
    REVISION_STATE_PROMOTED,
)

# ---------------------------------------------------------------------------
# Knowledge vs action — a BPM is NOT a list of (x, y).
# ---------------------------------------------------------------------------
# A site's *knowledge* state records whether it is merely known to the base or
# qualified (SCIENCE §13.2: DETECTION -> QUALIFICATION). "KNOWN" is a first-class
# state: a known site is retained in persistent knowledge even when no action is
# currently required or possible.
KNOWLEDGE_STATE_KNOWN = "KNOWN"
KNOWLEDGE_STATE_QUALIFIED = "QUALIFIED"

KNOWLEDGE_STATES: tuple[str, ...] = (
    KNOWLEDGE_STATE_KNOWN,
    KNOWLEDGE_STATE_QUALIFIED,
)

# A site's *action* state is the ordered ladder of SCIENCE §13.7, promoted
# verbatim. The first satisfied condition decides; the BPM stores the decided
# state, never re-derives it. Abstention is a valid first-class outcome.
ACTION_STATE_REQUALIFY_ADDITIVE_CALIBRATION = "REQUALIFY_ADDITIVE_CALIBRATION"
ACTION_STATE_ABSTAIN_CENSORED = "ABSTAIN_CENSORED"
ACTION_STATE_ABSTAIN_INCONSISTENT = "ABSTAIN_INCONSISTENT"
ACTION_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE = "ABSTAIN_INSUFFICIENT_EVIDENCE"
ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION = "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION"
ACTION_STATE_NO_ACTION_REQUIRED = "NO_ACTION_REQUIRED"

ACTION_STATE_LADDER: tuple[str, ...] = (
    ACTION_STATE_REQUALIFY_ADDITIVE_CALIBRATION,
    ACTION_STATE_ABSTAIN_CENSORED,
    ACTION_STATE_ABSTAIN_INCONSISTENT,
    ACTION_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
    ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION,
    ACTION_STATE_NO_ACTION_REQUIRED,
)

#: The single action state that may reach reconstruction (SCIENCE §13.2/§28).
ACTION_ELIGIBLE = ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION

ABSTAIN_STATES: frozenset[str] = frozenset({
    ACTION_STATE_ABSTAIN_CENSORED,
    ACTION_STATE_ABSTAIN_INCONSISTENT,
    ACTION_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
})

# ---------------------------------------------------------------------------
# Lookup outcomes (not errors — the disposition of a sensor/run against a base).
# ---------------------------------------------------------------------------
# A compatible *promoted* revision was deterministically selected (P4-A2 may use
# it; this lot only selects it).
OUTCOME_SELECTED = "SELECTED"
# Benign fallback: ordinary calibration, no BPM preparation. Never a fatal
# error. Reached when there is no base, no compatible profile, or only an
# unqualified profile.
OUTCOME_CALIBRATION_ONLY = "CALIBRATION_ONLY"
# The base exists but is corrupt / invalid / incompatible: a typed failure with
# provenance, never a silent "applied" and never a silent fallback.
OUTCOME_BASE_ERROR = "BASE_ERROR"

# Stable reason codes (surfaced in provenance; UI may map them to a non-blocking
# warning when ordinary calibration remains possible).
REASON_NO_PROFILE = "NO_PROFILE"  # no compatible revision exists (not fatal)
REASON_UNQUALIFIED_PROFILE = "UNQUALIFIED_PROFILE"  # compatible but not promoted
REASON_NO_BASE = "NO_BASE"  # no Bad Pixel Database at the configured root
REASON_BASE_CORRUPTED = "BASE_CORRUPTED"
REASON_BASE_INVALID = "BASE_INVALID"
REASON_BASE_INCOMPATIBLE = "BASE_INCOMPATIBLE"


def is_action_eligible(action_state: str) -> bool:
    """Return whether ``action_state`` permits reconstruction (SCIENCE §13.2)."""
    return action_state == ACTION_ELIGIBLE


__all__ = [
    "ABSTAIN_STATES",
    "ACTION_ELIGIBLE",
    "ACTION_STATE_ABSTAIN_CENSORED",
    "ACTION_STATE_ABSTAIN_INCONSISTENT",
    "ACTION_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE",
    "ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
    "ACTION_STATE_LADDER",
    "ACTION_STATE_NO_ACTION_REQUIRED",
    "ACTION_STATE_REQUALIFY_ADDITIVE_CALIBRATION",
    "BPM_SCHEMA_VERSION",
    "KNOWLEDGE_STATES",
    "KNOWLEDGE_STATE_KNOWN",
    "KNOWLEDGE_STATE_QUALIFIED",
    "OUTCOME_BASE_ERROR",
    "OUTCOME_CALIBRATION_ONLY",
    "OUTCOME_SELECTED",
    "REASON_BASE_CORRUPTED",
    "REASON_BASE_INCOMPATIBLE",
    "REASON_BASE_INVALID",
    "REASON_NO_BASE",
    "REASON_NO_PROFILE",
    "REASON_UNQUALIFIED_PROFILE",
    "REVISION_STATES",
    "REVISION_STATE_CANDIDATE",
    "REVISION_STATE_PROMOTED",
    "REVISION_STATE_QUALIFIED",
    "is_action_eligible",
]
