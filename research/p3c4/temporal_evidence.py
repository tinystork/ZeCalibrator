"""P3C-4 LOT 1 — temporal-persistence evidence *contract* (definition, not inference).

Research-only, internal, non-public. This module defines, **before any inference
code exists**, what *temporal persistence evidence at a fixed sensor coordinate*
means. It is the contract that the LOT 2 inference integration must satisfy; it
implements **no inference rule** itself: no feature extraction, no spatial
threshold, no candidate, no score.

Boundary (SCIENCE §16 / ARCHITECTURE §18, enforced by
``tests/p3c4/test_truth_leak_guard.py``): this module reads **no truth**. It
never imports the declared-facts table, the catalogue, the fixtures, the metrics,
or the truth-side oracle, and it never references a class label, a scenario name,
any expected-state label, or a truth-comparison helper. It is self-contained
(Python standard library only), so importing it pulls no truth module into
``sys.modules``.

The defect this contract corrects (SCIENCE §32, P3C-4): the frozen P3C
candidates inferred ``persisted_at_same_sensor_coord = YES`` and
``neighbourhood_residual_stable = YES`` for a celestial confounder (an
undersampled star core) on a **spatial / local** basis — a local signal that
recurs across independent groups. Spatial locality is **not** temporal
persistence. This contract states the distinction so the LOT 2 inference cannot
quietly substitute one for the other.

-----------------------------------------------------------------------------
Normative definition
-----------------------------------------------------------------------------

Temporal persistence evidence at a sensor coordinate is evidence that the
site's residual behaviour **at a fixed sensor coordinate** recurs (reproduces a
consistent signature) across **independent groups** **and** across
**independent epochs**, measured **exclusively from valid (non-censored)
samples**.

Two facts are explicitly **not** temporal persistence evidence:

* **Independence metadata is not evidence (§5).** ``epoch_count`` and
  ``independent_group_count`` are acquisition-structure bookkeeping. ``epoch_count
  >= 2`` is a conservative v1 **persistent-qualification policy**, not a physical
  law and not a persistence proof. A corpus with two epochs does not, by itself,
  prove that any residual recurred across them. The temporal evaluator must
  measure *recurrence of the signature* across groups and epochs; it must never
  recopy ``epoch_count``.

* **Spatial locality is not temporal persistence.** "The signal is local to the
  site (neighbours stable)" is a spatial statement. A celestial structure
  (star, optical feature) can be spatially local and still be a sky structure,
  not a sensor defect. Temporal persistence is established by recurrence across
  epochs/groups at a fixed coordinate, not by locality.

-----------------------------------------------------------------------------
The three states
-----------------------------------------------------------------------------

``YES`` — positive temporal persistence: the residual signature recurred across
at least ``MIN_SUPPORTING_GROUPS`` independent groups **and** at least
``MIN_SUPPORTING_EPOCHS`` independent epochs, from valid samples only.

``NO`` — positive temporal non-persistence, **reserved** for a positively
demonstrated non-recurrence at the fixed sensor coordinate: the signature is
seen, then **moves / departs** the coordinate, or its signature changes in a way
incompatible with a fixed site. ``NO`` is **never** produced from a mere
counting insufficiency (missing groups, missing epochs, missing valid samples) —
that is always ``UNDETERMINED``. See "Reachability of ``NO``" below.

``UNDETERMINED`` — a first-class, normal outcome. Insufficient evidence — no or
too-few valid samples, all-censored evidence, not enough supporting groups or
supporting epochs — produces ``UNDETERMINED``, **never** ``YES`` and **never**
``NO``.

Reachability of ``NO`` (LOT 1 → LOT 3)
--------------------------------------

``NO`` is reserved for a **positively demonstrated** non-recurrence at the fixed
coordinate. LOT 1 could not demonstrate it: the movement/departure discriminator
requires a temporal evolution of the signature across frames, which LOT 1 did not
yet consume. LOT 3 supplies that discriminator via an optional
:class:`NonPersistenceEvidence`:

* ``departed`` — the signature was seen at the fixed coordinate and then *moved
  away* between epochs (a celestial confounder following the sky);
* ``unique_occurrence`` — the signature appeared in a single (or very few) valid
  frames and demonstrably never recurred (a single transient).

When neither is supplied (``non_persistence=None``) the evaluator behaves exactly
as LOT 1: every counting insufficiency falls to ``UNDETERMINED`` and ``NO`` is
never produced. A positive ``NO`` is never a counting shortfall and never a
censored-sample inference; ``UNDETERMINED`` is never overwritten.

-----------------------------------------------------------------------------
Censoring (§14)
-----------------------------------------------------------------------------

A censored sample **cannot contribute quantitatively** to a persistence claim.
It may count only as "a censored observation exists" (``censored_samples``).
A group whose samples are all censored has no measurable signature presence, so
its ``signature_present`` is ``UNDETERMINED`` and it can never be a supporting
group. No amplitude is ever invented from a censored sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Explicit value vocabulary — never a boolean, never a numeric score.
# ---------------------------------------------------------------------------

YES = "YES"
NO = "NO"
UNDETERMINED = "UNDETERMINED"

_TERNARY = (YES, NO, UNDETERMINED)

# ---------------------------------------------------------------------------
# Allowed measurement sources (§2). The ONLY inputs a temporal evaluator may
# consume. Anything else is out of contract.
# ---------------------------------------------------------------------------

ALLOWED_MEASUREMENT_SOURCES: Tuple[str, ...] = (
    "per_frame_site_residual",      # residual behaviour of the site, per frame
    "per_group_site_residual",      # residual behaviour aggregated per independent group
    "fixed_sensor_coordinate",      # the FIXED sensor coordinate (x, y) under test
    "group_identity",               # independent-group identity (epoch_id, group_id)
    "epoch_identity",               # independent-epoch identity
    "valid_sample_count",           # count of valid (non-censored) samples
    "censored_sample_existence",    # existence of censored observations (count only)
    "non_persistence_demonstrated", # positive movement/departure or unique-occurrence (LOT 3)
)

# ---------------------------------------------------------------------------
# Temporal contract parameters — every parameter here is **temporal**, i.e. a
# recurrence/sufficiency count across groups/epochs/samples. There is no spatial
# threshold (no ADU, no neighbourhood SNR, no dispersion bound, no radius).
# ---------------------------------------------------------------------------

#: minimum independent groups whose signature must recur for ``YES``.
MIN_SUPPORTING_GROUPS = 2
#: minimum independent epochs whose signature must recur for ``YES``.
MIN_SUPPORTING_EPOCHS = 2
#: minimum valid (non-censored) samples needed to establish any recurrence.
MIN_VALID_SAMPLES = 2

TEMPORAL_CONTRACT_PARAMETERS: Tuple[Tuple[str, object, str], ...] = (
    ("min_supporting_groups", MIN_SUPPORTING_GROUPS,
     "temporal: minimum independent groups whose measured signature must recur"),
    ("min_supporting_epochs", MIN_SUPPORTING_EPOCHS,
     "temporal: minimum independent epochs whose measured signature must recur"),
    ("min_valid_samples", MIN_VALID_SAMPLES,
     "temporal: minimum valid (non-censored) samples required to establish recurrence"),
)


class InvalidTemporalStateError(ValueError):
    """A temporal state is outside the explicit YES/NO/UNDETERMINED vocabulary."""

    def __init__(self, value: object) -> None:
        self.value = value
        super().__init__(
            f"temporal state must be one of {_TERNARY!r}, got {value!r}"
        )


class CensoredContributionError(ValueError):
    """A censored sample was asked to contribute quantitatively (§14)."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(
            f"censored sample cannot contribute quantitatively to persistence: {detail}"
        )


class DisallowedEvidenceSourceError(ValueError):
    """A result cites a measurement source outside the allowed contract (§2)."""

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(
            f"disallowed measurement source {source!r}; allowed sources are "
            f"{ALLOWED_MEASUREMENT_SOURCES!r}"
        )


# ---------------------------------------------------------------------------
# Structured temporal signature — never a boolean, never a bare count.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupSignature:
    """Measured temporal evidence for one independent group at a fixed coordinate.

    ``signature_present`` is a *measured* fact over valid samples only:
    ``YES`` (the residual signature is present), ``NO`` (measured absent) or
    ``UNDETERMINED`` (cannot be established — e.g. all samples censored).

    Censoring (§14): a group with no valid samples has no measurable presence,
    so ``signature_present`` must be ``UNDETERMINED`` and the group cannot be a
    supporting group.
    """

    group_id: str
    epoch_id: str
    valid_samples: int
    censored_samples: int
    signature_present: str

    def __post_init__(self) -> None:
        if self.signature_present not in _TERNARY:
            raise InvalidTemporalStateError(self.signature_present)
        for name in ("valid_samples", "censored_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")
        if not isinstance(self.epoch_id, str) or not self.epoch_id:
            raise ValueError("epoch_id must be a non-empty string")
        # Censoring: presence/absence is measurable only from valid samples.
        if self.valid_samples == 0 and self.signature_present != UNDETERMINED:
            raise CensoredContributionError(
                f"group {self.group_id!r} has no valid samples but claims "
                f"signature_present={self.signature_present!r}"
            )

    @property
    def supports(self) -> bool:
        """True iff this group contributes positive support to persistence.

        Requires a measured ``YES`` presence and at least one valid sample;
        censored-only groups never support.
        """
        return self.signature_present == YES and self.valid_samples >= 1


@dataclass(frozen=True)
class TemporalSignature:
    """The structured temporal evidence at one **fixed** sensor coordinate.

    Holds the fixed coordinate (the identity that must not move) and the
    measured per-group evidence. This is the raw temporal record the evaluator
    consumes — never a boolean and never a single threshold.
    """

    coordinate_x: int
    coordinate_y: int
    groups: Tuple[GroupSignature, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate_x", int(self.coordinate_x))
        object.__setattr__(self, "coordinate_y", int(self.coordinate_y))
        object.__setattr__(self, "groups", tuple(self.groups))
        seen = set()
        for g in self.groups:
            if not isinstance(g, GroupSignature):
                raise TypeError(f"groups must be GroupSignature, got {type(g)!r}")
            key = (g.epoch_id, g.group_id)
            if key in seen:
                raise ValueError(f"duplicate independent group {key!r}")
            seen.add(key)


# ---------------------------------------------------------------------------
# The typed result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalPersistenceEvidence:
    """The typed result of temporal-persistence evaluation at a fixed coordinate.

    ``state`` is explicit (``YES`` / ``NO`` / ``UNDETERMINED``) — never a boolean,
    never a score. The supporting/examined counts, valid/censored sample counts
    and the structured ``temporal_signature`` make the evidence auditable.
    ``evidence_refs`` lists only allowed measurement sources.
    """

    state: str
    groups_examined: int
    groups_supporting: int
    epochs_examined: int
    valid_samples: int
    censored_samples: int
    temporal_signature: TemporalSignature
    evidence_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in _TERNARY:
            raise InvalidTemporalStateError(self.state)
        for name in (
            "groups_examined",
            "groups_supporting",
            "epochs_examined",
            "valid_samples",
            "censored_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if not isinstance(self.temporal_signature, TemporalSignature):
            raise TypeError(
                f"temporal_signature must be TemporalSignature, got "
                f"{type(self.temporal_signature)!r}"
            )
        if self.groups_supporting > self.groups_examined:
            raise ValueError(
                f"groups_supporting ({self.groups_supporting}) exceeds "
                f"groups_examined ({self.groups_examined})"
            )
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        for ref in self.evidence_refs:
            if ref not in ALLOWED_MEASUREMENT_SOURCES:
                raise DisallowedEvidenceSourceError(ref)


# ---------------------------------------------------------------------------
# Positive non-persistence demonstration (LOT 3) — the movement/departure and
# unique-occurrence discriminators that make ``NO`` reachable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonPersistenceEvidence:
    """Positive non-persistence demonstration (LOT 3; makes ``NO`` reachable).

    ``NO`` is reserved for a *positively demonstrated* non-recurrence. LOT 3
    supplies the two discriminators that demonstrate it, measured exclusively
    over valid (non-censored) samples:

    * ``departed`` — ``YES`` when the signature was measured present at the fixed
      coordinate in an earlier epoch and then *moved away* between epochs (a
      celestial confounder following the sky), leaving the coordinate while a
      signature remains nearby.
    * ``unique_occurrence`` — ``YES`` when the signature appeared in a single (or
      very few) valid frames and demonstrably never recurred, with enough valid
      samples elsewhere to positively establish absence (a single transient). It
      is **not** reachable from mere rarity: a low-occupancy intermittent that
      recurs across a run is never ``unique_occurrence``.

    Both are positive findings over valid samples — never a counting
    insufficiency, never a censored-sample inference, never a promotion to
    ``YES``.
    """

    departed: str = UNDETERMINED
    unique_occurrence: str = UNDETERMINED

    def __post_init__(self) -> None:
        for name in ("departed", "unique_occurrence"):
            value = getattr(self, name)
            if value not in _TERNARY:
                raise InvalidTemporalStateError(value)

    @property
    def demonstrates_no(self) -> bool:
        """True iff a positive non-persistence was demonstrated."""
        return self.departed == YES or self.unique_occurrence == YES


# ---------------------------------------------------------------------------
# The temporal evaluator (the contract's normative definition, made executable)
# ---------------------------------------------------------------------------

_DEFAULT_EVIDENCE_REFS: Tuple[str, ...] = (
    "fixed_sensor_coordinate",
    "group_identity",
    "epoch_identity",
    "per_group_site_residual",
    "valid_sample_count",
    "censored_sample_existence",
)


def assess_temporal_persistence(
    signature: TemporalSignature,
    *,
    non_persistence: Optional[NonPersistenceEvidence] = None,
    min_supporting_groups: int = MIN_SUPPORTING_GROUPS,
    min_supporting_epochs: int = MIN_SUPPORTING_EPOCHS,
    min_valid_samples: int = MIN_VALID_SAMPLES,
) -> TemporalPersistenceEvidence:
    """Evaluate temporal persistence from a :class:`TemporalSignature`.

    This is the contract's normative definition made executable. It consumes
    **only** the measured temporal signature (fixed coordinate + per-group
    recurrence) and derives the three-state outcome:

    * ``UNDETERMINED`` when evidence is insufficient — too few valid samples,
      or fewer than ``min_supporting_groups`` supporting groups, or fewer than
      ``min_supporting_epochs`` supporting epochs. Insufficiency is **never**
      promoted to ``YES`` and never demoted to ``NO``;
    * ``YES`` when the signature recurs across at least ``min_supporting_groups``
      independent groups **and** at least ``min_supporting_epochs`` independent
      epochs, from valid samples only;
    * ``NO`` is reserved for a **positively demonstrated** non-recurrence: the
      signature is seen, then moves/departs the coordinate, or appears once and
      never recurs. This is supplied **only** via ``non_persistence`` (the
      movement/departure or unique-occurrence discriminator, LOT 3). A mere
      counting insufficiency never produces ``NO``.

    ``non_persistence`` defaults to ``None`` (pure LOT 1 behaviour: ``NO``
    unreachable). When supplied, a demonstrated departure or unique occurrence
    yields ``NO`` **before** any counting branch — but it never overrides the
    valid-sample floor: if ``valid_samples < min_valid_samples`` the result is
    still ``UNDETERMINED``, never ``NO`` and never ``YES``.

    It never reads ``epoch_count`` or any independence metadata: the recurrence
    is measured from the per-group signatures themselves, and ``epochs_examined``
    counts the epochs actually examined, not an external ``epoch_count``.
    """
    groups = signature.groups

    groups_examined = len(groups)
    epochs_examined = len({g.epoch_id for g in groups})
    valid_samples = sum(g.valid_samples for g in groups)
    censored_samples = sum(g.censored_samples for g in groups)

    supporting = [g for g in groups if g.supports]
    groups_supporting = len(supporting)
    epochs_supporting = len({g.epoch_id for g in supporting})

    demonstrated_no = (
        non_persistence is not None and non_persistence.demonstrates_no
    )

    # Order of decisions (never reordered):
    #   1. valid-sample floor — a counting insufficiency is UNDETERMINED, never
    #      YES and never NO (§"UNDETERMINED jamais écrasé").
    #   2. positively demonstrated departure / unique occurrence — NO (LOT 3).
    #   3. >= min groups AND >= min epochs of supporting signatures — YES.
    #   4. any other counting insufficiency — UNDETERMINED.
    if valid_samples < min_valid_samples:
        state = UNDETERMINED
    elif demonstrated_no:
        state = NO
    elif (
        groups_supporting >= min_supporting_groups
        and epochs_supporting >= min_supporting_epochs
    ):
        state = YES
    else:
        # Every remaining case is a counting insufficiency (missing groups,
        # missing epochs, missing valid samples). It is UNDETERMINED, never NO:
        # NO requires a positively demonstrated movement/departure or unique
        # occurrence, which the non_persistence discriminator alone supplies.
        state = UNDETERMINED

    evidence_refs = _DEFAULT_EVIDENCE_REFS
    if state == NO:
        evidence_refs = _DEFAULT_EVIDENCE_REFS + ("non_persistence_demonstrated",)

    return TemporalPersistenceEvidence(
        state=state,
        groups_examined=groups_examined,
        groups_supporting=groups_supporting,
        epochs_examined=epochs_examined,
        valid_samples=valid_samples,
        censored_samples=censored_samples,
        temporal_signature=signature,
        evidence_refs=evidence_refs,
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
    "NonPersistenceEvidence",
    "assess_temporal_persistence",
    "InvalidTemporalStateError",
    "CensoredContributionError",
    "DisallowedEvidenceSourceError",
]
