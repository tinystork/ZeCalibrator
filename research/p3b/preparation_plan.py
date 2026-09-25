"""Run preparation plan (LOT3) — frozen plan, preflight, execution semantics.

Internal, non-public, never imported by ``zecalibrator.api.v1``.

This lot demonstrates the **scientific atomicity** of a run (site × run,
policy P-A, ``SCIENCE_CONTRACT.md`` §13.8 / ``ARCHITECTURE.md`` §18.3): a site
that cannot be prepared honestly and uniformly over *all* frames of a run is
prepared on *no* frame of that run. It does **not** build the industrial
transactional FITS writer — that is a different atomicity (file output), already
served by the G6 mechanism (``src/zecalibrator/io/output_writer.py`` +
``src/zecalibrator/io/batch_manifest.py``); a future materialization stage will
reuse that substrate, not this module.

**This module MUST NOT be used as an output-transaction mechanism.** It freezes
the *decision* (what is prepared where) before any output exists; the moment a
result becomes visible on disk is a separate, file-output atomicity (G6), not a
``PreparationPlan`` concern.

Two atomicities are kept apart on purpose:

* **SCIENTIFIC ATOMICITY = site × run** (this lot). The run-wide decision is a
  single value per site (``ELIGIBLE`` / ``ABSTAIN`` / ``NO_ACTION`` /
  ``REQUALIFY``); there is no per-frame action slot, so
  ``apply / apply / skip / apply`` for one site inside a run is unrepresentable
  and can never be produced.
* **FILE OUTPUT TRANSACTION** (not this lot). When results become visible on
  disk — deferred; a later temp-dir + atomic commit/rename will suffice.

Preflight stays lightweight: it consumes *facts* (the LOT2 qualification
decision, plus a declared per-(frame, site) donor-availability fact that
synthesizes geometry / DQ / usable mask / saturation-censoring into a single
``AVAILABLE`` / ``UNAVAILABLE`` token), never full calibrated images. It retains
only a few bytes per (frame, site) and writes no final product.

Model-level invariant (rework F1): ``SitePlanEntry.run_wide_applicability`` and
``SitePlanEntry.abstention_reason`` are **derived** (``init=False``) from
``action_state`` + ``per_frame_donor_availability``; they are never caller-
supplied, so a frozen plan whose run-wide decision contradicts the P-A policy is
**not representable** — it cannot be constructed, not merely rejected.

Disciplines demonstrated here (each tested in ``tests/p3b/test_preparation_plan.py``):

* a site missing valid same-CFA donors on any one frame is ``ABSTAIN`` for the
  whole run (site × run atomicity, never ``N-1 reconstructed + 1 original``);
* a blocked site never disables another independently qualified site in the
  same run (atomicity is never global);
* the run plan is frozen before any output can be finalized (a non-frozen plan
  cannot finalize);
* a frozen plan is applied *exactly*: an observed fact that contradicts what the
  plan assumed raises ``PlanInvalidError`` (abort), never a silent per-frame
  switch;
* a frozen plan refuses mutation with a typed ``PlanFrozenError`` (including the
  internal site table, rework S3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from .qualification_policy import (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
    ACTION_NO_ACTION,
    ACTION_REQUALIFY,
    QualificationDecision,
)

# ---------------------------------------------------------------------------
# Per-(frame, site) donor availability facts (declared, never computed here)
# ---------------------------------------------------------------------------
#
# A donor is a same-CFA neighbour usable to reconstruct the site. Its
# availability on one frame is a *fact* synthesized upstream from geometry +
# DQ / usable mask + saturation-censoring; this module consumes that token and
# never re-derives it from pixels or from any numeric threshold.

DONORS_AVAILABLE = "AVAILABLE"
DONORS_UNAVAILABLE = "UNAVAILABLE"
DONOR_AVAILABILITY: Tuple[str, ...] = (DONORS_AVAILABLE, DONORS_UNAVAILABLE)

# ---------------------------------------------------------------------------
# Run-wide applicability vocabulary (single value per site, never per frame)
# ---------------------------------------------------------------------------

RUN_WIDE_ELIGIBLE = "ELIGIBLE"
RUN_WIDE_ABSTAIN = "ABSTAIN"
RUN_WIDE_NO_ACTION = "NO_ACTION"
RUN_WIDE_REQUALIFY = "REQUALIFY"
RUN_WIDE_APPLICABILITY: Tuple[str, ...] = (
    RUN_WIDE_ELIGIBLE,
    RUN_WIDE_ABSTAIN,
    RUN_WIDE_NO_ACTION,
    RUN_WIDE_REQUALIFY,
)

# Run-wide abstention reason for the one case this lot adds on top of LOT2:
# an ELIGIBLE site whose donors are not available on *every* frame of the run.
RC_DONORS_UNAVAILABLE_RUN_WIDE = "DONORS_UNAVAILABLE_RUN_WIDE"


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class PreparationPlanError(Exception):
    """Base class for all run-preparation-plan errors."""


class PlanFrozenError(PreparationPlanError):
    """A frozen plan was mutated (or an entry added after freezing)."""


class PlanNotFrozenError(PreparationPlanError):
    """An output was finalized (executed) from a plan that is not yet frozen."""


class PlanInvalidError(PreparationPlanError):
    """An observed fact contradicts the frozen plan during execution -> abort."""


class DuplicateSiteError(PreparationPlanError):
    """The same site id was declared more than once in one run."""


class EmptyFramesError(PreparationPlanError):
    """A run-wide decision was requested over zero frames (no donor facts)."""


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryBinding:
    """The geometry a plan is bound to (SCIENCE §4)."""

    shape: Tuple[int, int]
    cfa_pattern: str
    binning: Tuple[int, int] = (1, 1)
    roi_origin: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(v <= 0 for v in self.shape):
            raise ValueError(f"shape must be a positive (ny, nx), got {self.shape!r}")
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        object.__setattr__(self, "binning", tuple(int(v) for v in self.binning))
        object.__setattr__(self, "roi_origin", tuple(int(v) for v in self.roi_origin))


@dataclass(frozen=True)
class SiteQualification:
    """One site's coordinate bundled with its LOT2 qualification decision.

    The decision is *consumed verbatim* (never recomputed): this lot imports the
    LOT2 engine output and does not re-run qualification or add any threshold.
    """

    site_id: str
    x: int
    y: int
    decision: QualificationDecision
    operator_id: Optional[str] = None
    operator_version: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.site_id or not isinstance(self.site_id, str):
            raise ValueError(f"site_id must be a non-empty string, got {self.site_id!r}")
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))


@dataclass(frozen=True)
class SitePlanEntry:
    """One site's frozen, run-wide entry in a :class:`PreparationPlan`.

    ``run_wide_applicability`` and ``abstention_reason`` are **derived**
    (``init=False``) from ``action_state`` + ``per_frame_donor_availability``:
    they cannot be supplied by the caller, so an entry whose run-wide decision
    contradicts the P-A policy is *not representable*. The per-frame
    donor-availability tuple is the *assumed facts* retained for contradiction
    re-validation at execution (a few bytes per frame, never the calibrated
    image).
    """

    site_id: str
    x: int
    y: int
    qualification_state: str  # LOT2 axis 1 (epistemic), verbatim
    action_state: str  # LOT2 axis 3 (action), verbatim
    reason_codes: Tuple[str, ...]  # LOT2 reason codes, verbatim
    per_frame_donor_availability: Tuple[str, ...]  # assumed facts, per frame
    operator_id: Optional[str] = None
    operator_version: Optional[str] = None
    # Derived — never caller-supplied (rework F1).
    run_wide_applicability: str = field(init=False)
    abstention_reason: Optional[str] = field(init=False)

    def __post_init__(self) -> None:
        for token in self.per_frame_donor_availability:
            if token not in DONOR_AVAILABILITY:
                raise ValueError(f"invalid donor-availability token {token!r}")
        object.__setattr__(
            self, "per_frame_donor_availability", tuple(self.per_frame_donor_availability)
        )
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))
        run_wide, reason = derive_run_wide(
            self.action_state, self.per_frame_donor_availability
        )
        object.__setattr__(self, "run_wide_applicability", run_wide)
        object.__setattr__(self, "abstention_reason", reason)


@dataclass(frozen=True)
class AppliedSiteRecord:
    """One site's outcome on one frame of the execution pass."""

    site_id: str
    applied: bool


@dataclass(frozen=True)
class FrameExecution:
    """The per-site outcomes for one frame of the run."""

    frame_id: str
    sites: Tuple[AppliedSiteRecord, ...]


@dataclass(frozen=True)
class ExecutionResult:
    """The uniform, plan-driven execution outcome over the whole run."""

    run_id: str
    frames: Tuple[FrameExecution, ...]


# ---------------------------------------------------------------------------
# Run-wide derivation (pure; the single source of the P-A decision)
# ---------------------------------------------------------------------------

_ABSTAIN_ACTIONS: Tuple[str, ...] = (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    ACTION_ABSTAIN_INSUFFICIENT,
)


def derive_run_wide(
    action_state: str, per_frame_donors: Tuple[str, ...]
) -> Tuple[str, Optional[str]]:
    """Map a LOT2 action + per-frame donor facts to a run-wide applicability.

    This is the *only* place LOT3 changes a LOT2 decision: an ``ELIGIBLE`` site
    stays ``ELIGIBLE`` only if every frame has valid same-CFA donors; otherwise
    it becomes ``ABSTAIN`` for the whole run. All other LOT2 action states pass
    through unchanged (they are already frame-independent).

    ``per_frame_donors`` must be non-empty (a run over zero frames has no
    run-wide decision): the empty tuple is a precondition violation, not the
    vacuous truth of ``all([])`` (rework S1).
    """
    if not per_frame_donors:
        raise EmptyFramesError(
            "cannot derive a run-wide applicability over zero frames"
        )
    if action_state == ACTION_ELIGIBLE:
        if all(token == DONORS_AVAILABLE for token in per_frame_donors):
            return RUN_WIDE_ELIGIBLE, None
        return RUN_WIDE_ABSTAIN, RC_DONORS_UNAVAILABLE_RUN_WIDE
    if action_state == ACTION_NO_ACTION:
        return RUN_WIDE_NO_ACTION, None
    if action_state == ACTION_REQUALIFY:
        return RUN_WIDE_REQUALIFY, None
    if action_state in _ABSTAIN_ACTIONS:
        return RUN_WIDE_ABSTAIN, action_state
    raise ValueError(f"unexpected LOT2 action state {action_state!r}")


# ---------------------------------------------------------------------------
# A read-only site table whose mutators raise the typed PlanFrozenError (S3)
# ---------------------------------------------------------------------------


class _FrozenSiteTable(dict):
    """A dict whose mutators raise :class:`PlanFrozenError` (never bare TypeError).

    ``freeze()`` wraps the mutable pre-freeze site table in this type, so that
    even direct internal mutation (``plan._sites[...] = ...``) is refused with
    the same typed error as every other post-freeze mutation.
    """

    _MESSAGE = "the site table of a frozen PreparationPlan is immutable"

    def __setitem__(self, key, value):
        raise PlanFrozenError(self._MESSAGE)

    def __delitem__(self, key):
        raise PlanFrozenError(self._MESSAGE)

    def clear(self):
        raise PlanFrozenError(self._MESSAGE)

    def pop(self, *args, **kwargs):
        raise PlanFrozenError(self._MESSAGE)

    def popitem(self):
        raise PlanFrozenError(self._MESSAGE)

    def setdefault(self, *args, **kwargs):
        raise PlanFrozenError(self._MESSAGE)

    def update(self, *args, **kwargs):
        raise PlanFrozenError(self._MESSAGE)

    def __ior__(self, other):
        raise PlanFrozenError(self._MESSAGE)


# ---------------------------------------------------------------------------
# The plan (mutable during preflight, frozen after ``freeze()``)
# ---------------------------------------------------------------------------


class PreparationPlan:
    """A run-wide preparation plan, built in preflight then frozen.

    Constructed unfrozen (``frozen=False``), accumulated with :meth:`add_site`,
    then sealed with :meth:`freeze`. After freezing, the plan is immutable
    (mutation raises :class:`PlanFrozenError`) and is the only object that can
    finalize outputs (:meth:`execute`).
    """

    def __init__(
        self,
        *,
        run_id: str,
        profile_revision: str,
        calibration_identity: str,
        geometry: GeometryBinding,
        frame_ids: Tuple[str, ...],
        frozen: bool = False,
    ) -> None:
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "profile_revision", profile_revision)
        object.__setattr__(self, "calibration_identity", calibration_identity)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "frame_ids", tuple(frame_ids))
        object.__setattr__(self, "_sites", {})
        object.__setattr__(self, "frozen", bool(frozen))

    def __setattr__(self, name: str, value) -> None:
        if getattr(self, "frozen", False):
            raise PlanFrozenError(
                f"PreparationPlan (run {self.run_id!r}) is frozen; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)

    def _require_not_frozen(self) -> None:
        if self.frozen:
            raise PlanFrozenError(
                f"PreparationPlan (run {self.run_id!r}) is frozen and refuses mutation"
            )

    def _require_frozen(self) -> None:
        if not self.frozen:
            raise PlanNotFrozenError(
                f"PreparationPlan (run {self.run_id!r}) is not frozen; "
                "outputs cannot be finalized from an unfrozen plan"
            )

    def add_site(self, entry: SitePlanEntry) -> None:
        """Add one site entry (pre-freeze only)."""
        self._require_not_frozen()
        if entry.site_id in self._sites:
            raise DuplicateSiteError(
                f"site {entry.site_id!r} declared more than once in run {self.run_id!r}"
            )
        if len(entry.per_frame_donor_availability) != len(self.frame_ids):
            raise ValueError(
                f"site {entry.site_id!r} has {len(entry.per_frame_donor_availability)} "
                f"per-frame donor facts but the run has {len(self.frame_ids)} frames"
            )
        self._sites[entry.site_id] = entry

    def sites(self) -> Tuple[SitePlanEntry, ...]:
        """Return the site entries in insertion order."""
        return tuple(self._sites.values())

    def site(self, site_id: str) -> SitePlanEntry:
        """Return one site entry (raises KeyError when absent)."""
        return self._sites[site_id]

    def freeze(self) -> "PreparationPlan":
        """Seal the plan: validate, freeze the site table, set ``frozen``."""
        if self.frozen:
            return self
        self._validate()
        object.__setattr__(self, "_sites", _FrozenSiteTable(self._sites))
        object.__setattr__(self, "frozen", True)
        return self

    def _validate(self) -> None:
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("run_id must be a non-empty string")
        if not self.profile_revision or not isinstance(self.profile_revision, str):
            raise ValueError("profile_revision must be a non-empty string")
        if not self.calibration_identity or not isinstance(self.calibration_identity, str):
            raise ValueError("calibration_identity must be a non-empty string")
        if not self.frame_ids:
            raise ValueError("a run must have at least one frame")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame ids must be unique within a run")

    def execute(self, observe: Callable[[str, str], str]) -> ExecutionResult:
        """Apply the frozen plan exactly over the run's frames (finalization).

        ``observe(site_id, frame_id) -> str`` returns the *current* donor
        availability for that site/frame (``AVAILABLE`` / ``UNAVAILABLE``). Each
        observed token is compared to the frozen plan's assumed fact; any
        mismatch is a contradiction -> :class:`PlanInvalidError` (abort), never a
        silent per-frame switch. ``applied`` is derived solely from the single
        run-wide value, so it is uniform across frames for each site.
        """
        self._require_frozen()
        frames_out: list = []
        for index, frame_id in enumerate(self.frame_ids):
            site_records: list = []
            for entry in self.sites():
                observed = observe(entry.site_id, frame_id)
                if observed not in DONOR_AVAILABILITY:
                    raise PlanInvalidError(
                        f"run {self.run_id!r}: uninterpretable donor fact "
                        f"{observed!r} for site {entry.site_id!r} frame {frame_id!r}"
                    )
                assumed = entry.per_frame_donor_availability[index]
                if observed != assumed:
                    raise PlanInvalidError(
                        f"run {self.run_id!r}: frozen plan assumed "
                        f"donor fact {assumed!r} for site {entry.site_id!r} on "
                        f"frame {frame_id!r} but observed {observed!r} — aborting "
                        "rather than improvising or switching silently"
                    )
                applied = entry.run_wide_applicability == RUN_WIDE_ELIGIBLE
                site_records.append(AppliedSiteRecord(entry.site_id, applied))
            frames_out.append(FrameExecution(frame_id, tuple(site_records)))
        return ExecutionResult(self.run_id, tuple(frames_out))


# ---------------------------------------------------------------------------
# Preflight (PASS 1): consume facts, build and freeze the plan, write nothing
# ---------------------------------------------------------------------------


def preflight(
    *,
    run_id: str,
    profile_revision: str,
    calibration_identity: str,
    geometry: GeometryBinding,
    frame_ids: Tuple[str, ...],
    site_qualifications: Tuple[SiteQualification, ...],
    donor_availability: Callable[[str, str], str],
) -> PreparationPlan:
    """Build and freeze a run preparation plan from facts (no final product).

    ``donor_availability(site_id, frame_id) -> str`` supplies the declared
    per-(site, frame) donor fact. The function is pure: it opens no file and
    writes nothing; it returns a frozen plan that retains only a few bytes per
    (frame, site).
    """
    plan = PreparationPlan(
        run_id=run_id,
        profile_revision=profile_revision,
        calibration_identity=calibration_identity,
        geometry=geometry,
        frame_ids=frame_ids,
        frozen=False,
    )
    for sq in site_qualifications:
        per_frame = tuple(
            donor_availability(sq.site_id, frame_id) for frame_id in frame_ids
        )
        entry = SitePlanEntry(
            site_id=sq.site_id,
            x=sq.x,
            y=sq.y,
            qualification_state=sq.decision.epistemic_state,
            action_state=sq.decision.action,
            reason_codes=sq.decision.reason_codes,
            per_frame_donor_availability=per_frame,
            operator_id=sq.operator_id,
            operator_version=sq.operator_version,
        )
        plan.add_site(entry)
    return plan.freeze()


__all__ = [
    "DONOR_AVAILABILITY",
    "DONORS_AVAILABLE",
    "DONORS_UNAVAILABLE",
    "RC_DONORS_UNAVAILABLE_RUN_WIDE",
    "RUN_WIDE_ABSTAIN",
    "RUN_WIDE_APPLICABILITY",
    "RUN_WIDE_ELIGIBLE",
    "RUN_WIDE_NO_ACTION",
    "RUN_WIDE_REQUALIFY",
    "AppliedSiteRecord",
    "DuplicateSiteError",
    "EmptyFramesError",
    "ExecutionResult",
    "FrameExecution",
    "GeometryBinding",
    "PlanFrozenError",
    "PlanInvalidError",
    "PlanNotFrozenError",
    "PreparationPlan",
    "PreparationPlanError",
    "SitePlanEntry",
    "SiteQualification",
    "derive_run_wide",
    "preflight",
]
