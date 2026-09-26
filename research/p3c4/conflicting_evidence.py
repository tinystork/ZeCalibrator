"""P3C-4 — corrected ``conflicting_evidence`` contract + rule (versioned).

Research-only, internal, non-public. This module corrects the frozen P3C
``conflicting_evidence`` semantics that the QUALIFICATION-2 campaign exposed
(65 sites inferred ``YES`` where the truth is ``NO``). The frozen rule inferred
``conflicting_evidence`` from a **raw strict sign-crossing count** of the
canonical residual. On a light-only corpus an intermittent site in its OFF state
has a residual ≈ 0, so read noise makes the series cross zero repeatedly, and
the rule reports "contradictory evidence" for a normal/intermittent site.

This module states the corrected definition, before any integration code exists:

* **A sign change is not, by itself, conflict.** ``noise crossing zero !=
  scientific evidence of conflict``. Conflict requires the post-calibration CFA
  residual to *genuinely reverse sign with meaningful magnitude on both sides* —
  a **significant sign reversal** — never a zero crossing through the noise
  floor.
* **An intermittent site that changes state is not, by itself, conflict.** A
  persistent intermittent naturally produces a positive residual in its ON state
  and a residual ≈ 0 (background) in its OFF state; only the read noise crosses
  zero. If the sensor persistence is otherwise established, that state change is
  *not* contradictory evidence.
* **Detection floor (a documented blind band).** This fact detects only the
  conflicts whose **both** sides exceed ``conflict_amplitude_floor``. Below it,
  the state is ``NO`` **by construction**, without claiming the absence of a
  conflict: a genuine low-amplitude sign reversal (e.g. ±40 ADU, well above the
  ~4-5 ADU read noise) is **not** detected. The floor is a
  ``RESEARCH_CANDIDATE_PARAMETER``; a relative floor (``k × local σ``) would
  remove the blind band but introduces a new measurement (a local scale and its
  window) that must be adopted as a product parameter — deferred, not of this lot.

The corrected rule consumes the canonical residual series (the post-calibration
CFA residual when a representative dark is supplied; the same-CFA local residual
fallback on a light-only corpus) and applies an amplitude floor so that only a
**significant** positive↔negative reversal counts.

Boundary: this module reads **no truth**. It is self-contained (Python standard
library only), consumes only a residual series, and never imports
``research.p3b.*`` or ``research.p3c.oracle``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

# Explicit value vocabulary — never a boolean, never a numeric score. The
# corrected rule is binary (YES / NO): conflict is a positive finding, absence
# of a significant sign reversal is the permissive NO (same binary shape as the
# frozen P3C rule for this fact).
YES = "YES"
NO = "NO"

_TERNARY = (YES, NO)

# Version of the corrected conflicting-evidence *contract* (the normative
# definition this module makes executable). It is bumped only when the
# contract's normative meaning changes (the significance gate, the reversal
# definition). It is not a metric version and not an adapter version — each
# provenance is versioned independently.
CONFLICT_CONTRACT_VERSION = "p3c4-conflicting-evidence-contract-1"

# Allowed measurement sources for the corrected rule: the only inputs it may
# consume. ``canonical_residual_series`` is the per-frame residual series whose
# sign carries the calibration-conflict meaning (post-calibration CFA residual
# when a dark is present, the local residual fallback otherwise).
ALLOWED_EVIDENCE_SOURCES: Tuple[str, ...] = (
    "canonical_residual_series",
    "valid_sample_count",
)


class InvalidConflictStateError(ValueError):
    """A conflict state is outside the explicit YES/NO vocabulary."""

    def __init__(self, value: object) -> None:
        self.value = value
        super().__init__(
            f"conflict state must be one of {_TERNARY!r}, got {value!r}"
        )


class DisallowedConflictSourceError(ValueError):
    """A result cites a measurement source outside the allowed contract."""

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(
            f"disallowed conflict measurement source {source!r}; allowed sources "
            f"are {ALLOWED_EVIDENCE_SOURCES!r}"
        )


@dataclass(frozen=True)
class ConflictingEvidence:
    """The typed result of the corrected conflicting-evidence rule.

    ``state`` is explicit (``YES`` / ``NO``) — never a boolean, never a score.
    ``sign_reversals`` is the measured count of *significant* sign reversals;
    ``valid_samples`` is the number of finite residuals examined.
    ``evidence_refs`` lists only allowed measurement sources.
    """

    state: str
    sign_reversals: int
    valid_samples: int
    evidence_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in _TERNARY:
            raise InvalidConflictStateError(self.state)
        for name in ("sign_reversals", "valid_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        for ref in self.evidence_refs:
            if ref not in ALLOWED_EVIDENCE_SOURCES:
                raise DisallowedConflictSourceError(ref)


def _finite(residuals: Iterable) -> Tuple[float, ...]:
    """Drop non-finite (censored / missing) entries from a residual series."""
    out = []
    for v in residuals:
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return tuple(out)


def significant_sign_reversals(
    residuals: Iterable, *, conflict_amplitude_floor: float
) -> int:
    """Count **significant** sign reversals in a residual series.

    A reversal is counted between two consecutive residuals ``a``, ``b`` of
    opposite sign **only when both have magnitude at/above
    ``conflict_amplitude_floor``**. This is the amplitude gate that separates a
    genuine positive↔negative post-calibration flip from a noise-driven zero
    crossing: a residual that merely passes through zero (read noise at an OFF
    state, or a normal background) contributes no reversal, because at least one
    side of the crossing sits inside the floor.

    Non-finite entries (censored frames) are dropped first, never invented.
    """
    arr = _finite(residuals)
    floor = float(conflict_amplitude_floor)
    reversals = 0
    for i in range(1, len(arr)):
        a = arr[i - 1]
        b = arr[i]
        if abs(a) >= floor and abs(b) >= floor and (a > 0) != (b > 0):
            reversals += 1
    return int(reversals)


def assess_conflicting_evidence(
    residuals: Iterable,
    *,
    conflict_amplitude_floor: float,
    min_conflict_reversals: int,
) -> ConflictingEvidence:
    """Evaluate the corrected ``conflicting_evidence`` fact from a residual series.

    This is the contract's normative definition made executable:

    * ``YES`` — the canonical residual demonstrated at least
      ``min_conflict_reversals`` **significant** sign reversals (both sides of
      each crossing at/above ``conflict_amplitude_floor``): genuine contradictory
      post-calibration evidence;
    * ``NO`` — otherwise. Absence of a significant sign reversal is the
      permissive ``NO``: a normal background, an intermittent state change, or a
      single-sign residual is **not** conflicting evidence, regardless of how
      many times read noise crosses zero.

    The rule never produces ``UNDETERMINED``: conflict is a positive finding and
    its absence is ``NO`` (the same binary shape the frozen P3C rule used for
    this fact, and the fail-closed bridge still abstains when the *other* safety
    facts are unresolved).
    """
    valid = _finite(residuals)
    reversals = significant_sign_reversals(
        residuals, conflict_amplitude_floor=conflict_amplitude_floor
    )
    state = YES if reversals >= int(min_conflict_reversals) else NO
    return ConflictingEvidence(
        state=state,
        sign_reversals=reversals,
        valid_samples=len(valid),
        evidence_refs=("canonical_residual_series", "valid_sample_count"),
    )


__all__ = [
    "YES",
    "NO",
    "CONFLICT_CONTRACT_VERSION",
    "ALLOWED_EVIDENCE_SOURCES",
    "ConflictingEvidence",
    "significant_sign_reversals",
    "assess_conflicting_evidence",
    "InvalidConflictStateError",
    "DisallowedConflictSourceError",
]
