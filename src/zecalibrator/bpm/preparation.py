"""Run-wide preparation plan + applicator (P4-A2): preflight → freeze → execute.

This module links the Bad Pixel Database (P4-A) to the post-calibration images:
it builds a run-wide :class:`PreparationPlan` in a **preflight** pass, **freezes**
it, then **executes** it to produce per-frame :class:`PreparedCalibrationResult`
values with masks and provenance. It is internal and non-public (never imported
by ``zecalibrator.api.v1``), headless (no Qt), and imports no ``research/*``
(mission §85).

Stage separation (mission §37, invariant):

```text
preflight d'applicabilité  ->  GEL du PreparationPlan  ->  execute
```

A contradiction discovered *after* the freeze (an observed donor fact that
disagrees with the frozen plan) raises :class:`PlanInvalidError` and aborts the
prepared output — never ``apply / apply / skip / apply`` and never a silent
per-frame switch.

Atomicity (mission §36, P-A normative):

* a site applicable on **every** frame of the run may be prepared **run-wide**;
* **one** frame with no valid donor for that site ⇒ the site **abstains on all**
  frames (run-wide, never ``N-1 reconstructed + 1 original``);
* a blocked site never disables an independently eligible site in the same run
  (atomicity is site × run, never global).

Only ``ACTION ELIGIBLE`` sites may reach reconstruction (mission §29 / P4-A §29):
the plan is built exclusively from :func:`~zecalibrator.bpm.revision.action_eligible_sites`
of the selected revision, so a ``NO_ACTION_REQUIRED`` or ``ABSTAIN_*`` site in the
base stays **intact** (never reconstructed), even though it is known.

Confinement (mission §81):

```text
no compatible profile      -> CALIBRATION_ONLY   (ordinary calibration unchanged)
unqualified profile        -> CALIBRATION_ONLY
selected base corrupt      -> BASE_ERROR         (typed + explicit warning)
invalid plan at execution  -> ABANDON prepared output (PlanInvalidError)
```

There is no silent mixed preparation and no user scientific setting is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from zecalibrator.core.calibrate import CalibrationResult
from zecalibrator.core.geometry import is_bayer_phase

from .lookup import BpmResolution, ProvenanceNote
from .prepared_result import PreparedCalibrationResult
from .reconstruction import (
    DEFAULT_OPERATOR,
    NO_VALUE,
    ReconstructionOperator,
    donors_available,
    reconstruct_site_pixel,
)
from .revision import Revision, action_eligible_sites
from .vocabulary import (
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The preparation-plan schema version (distinct from BPM schema / operator schema).
PREPARATION_SCHEMA_VERSION = "zecalibrator.bpm.preparation.v1"

# Preparation outcomes.
OUTCOME_PREPARED = "PREPARED"

# Run-wide applicability (a single value per site, never per frame).
RUN_WIDE_ELIGIBLE = "ELIGIBLE"
RUN_WIDE_ABSTAIN = "ABSTAIN"

# Run-wide abstention reason for an ELIGIBLE site whose donors are not available
# on every frame of the run (the only case this lot adds to the BPM action state).
RC_DONORS_UNAVAILABLE_RUN_WIDE = "DONORS_UNAVAILABLE_RUN_WIDE"


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class PreparationError(Exception):
    """Base class for all run-preparation-plan errors."""


class PlanFrozenError(PreparationError):
    """A frozen plan was mutated (or a site added after freezing)."""


class PlanNotFrozenError(PreparationError):
    """An output was finalized (executed) from a plan that is not yet frozen."""


class PlanInvalidError(PreparationError):
    """An observed fact contradicts the frozen plan during execution -> abort."""


class DuplicateSiteError(PreparationError):
    """The same site position was declared more than once in one run."""


class EmptyFramesError(PreparationError):
    """A run-wide decision was requested over zero frames."""


class UnsupportedCfaError(PreparationError):
    """The run geometry has no Bayer CFA phase (same-CFA reconstruction requires one)."""


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibratedFrame:
    """A post-calibration CFA frame — the applicator's ONLY pixel input.

    Binds a frame id to the immutable post-calibration :class:`CalibrationResult`
    (float32 CFA + uint16 DQ). The applicator never sees the raw light or the dark
    master: it consumes only this post-calibration plane (SCIENCE §13.1, mission
    §35) — the order invariant is structural, not merely documented.
    """

    frame_id: str
    calibration: CalibrationResult

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")
        if not isinstance(self.calibration, CalibrationResult):
            raise ValueError("calibration must be a CalibrationResult")
        if self.calibration.data is None or self.calibration.mask is None:
            raise PreparationError(
                f"frame {self.frame_id!r} has no post-calibration data/mask "
                f"(status {self.calibration.status!r}); preparation requires a "
                "completed post-calibration CFA"
            )


@dataclass(frozen=True)
class SitePlanEntry:
    """One ELIGIBLE site's frozen, run-wide entry in a :class:`PreparationPlan`.

    ``run_wide_applicability`` and ``abstention_reason`` are **derived**
    (``init=False``) from the per-frame donor-availability facts: they cannot be
    supplied by the caller, so an entry whose run-wide decision contradicts the
    P-A policy is *not representable* (it cannot be constructed).
    """

    site_id: str
    position: Tuple[int, int]  # (y, x)
    action_state: str  # verbatim from the BPM SiteRecord
    knowledge_state: str  # verbatim from the BPM SiteRecord
    per_frame_donor_available: Tuple[bool, ...]  # assumed facts, per frame
    run_wide_applicability: str = field(init=False)
    abstention_reason: Optional[str] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.site_id, str) or not self.site_id:
            raise ValueError("site_id must be a non-empty string")
        if (
            not isinstance(self.position, (tuple, list))
            or len(self.position) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.position)
        ):
            raise ValueError(f"position must be a pair of non-negative ints, got {self.position!r}")
        object.__setattr__(self, "position", (int(self.position[0]), int(self.position[1])))
        facts = tuple(bool(v) for v in self.per_frame_donor_available)
        if not facts:
            raise EmptyFramesError(
                f"site {self.site_id!r} has no per-frame donor facts (empty run)"
            )
        object.__setattr__(self, "per_frame_donor_available", facts)
        run_wide, reason = derive_run_wide(facts)
        object.__setattr__(self, "run_wide_applicability", run_wide)
        object.__setattr__(self, "abstention_reason", reason)


def derive_run_wide(per_frame_donor_available: Tuple[bool, ...]) -> Tuple[str, Optional[str]]:
    """Map per-frame donor facts to a run-wide applicability (P-A atomicity).

    An ELIGIBLE site stays ``ELIGIBLE`` only when donors are available on **every**
    frame; otherwise it becomes ``ABSTAIN`` for the whole run. The empty tuple is a
    precondition violation (a run over zero frames has no run-wide decision).
    """
    if not per_frame_donor_available:
        raise EmptyFramesError("cannot derive a run-wide applicability over zero frames")
    if all(per_frame_donor_available):
        return RUN_WIDE_ELIGIBLE, None
    return RUN_WIDE_ABSTAIN, RC_DONORS_UNAVAILABLE_RUN_WIDE


class _FrozenSiteTable(dict):
    """A dict whose mutators raise :class:`PlanFrozenError` (never bare TypeError)."""

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

    Constructed unfrozen, accumulated with :meth:`add_site`, then sealed with
    :meth:`freeze`. After freezing the plan is immutable and is the only object
    that can finalize outputs (:meth:`execute`).
    """

    def __init__(
        self,
        *,
        run_id: str,
        revision_id: str,
        geometry: Tuple[int, int],
        cfa_phase: str,
        roi_origin: Tuple[int, int],
        frame_ids: Tuple[str, ...],
        frozen: bool = False,
    ) -> None:
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "revision_id", revision_id)
        object.__setattr__(self, "schema_version", PREPARATION_SCHEMA_VERSION)
        object.__setattr__(self, "geometry", tuple(geometry))
        object.__setattr__(self, "cfa_phase", cfa_phase)
        object.__setattr__(self, "roi_origin", tuple(roi_origin))
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
        """Add one ELIGIBLE site entry (pre-freeze only)."""
        self._require_not_frozen()
        if entry.position in self._sites:
            raise DuplicateSiteError(
                f"site at {entry.position!r} declared more than once in run {self.run_id!r}"
            )
        if len(entry.per_frame_donor_available) != len(self.frame_ids):
            raise ValueError(
                f"site {entry.site_id!r} has {len(entry.per_frame_donor_available)} "
                f"per-frame donor facts but the run has {len(self.frame_ids)} frames"
            )
        self._sites[entry.position] = entry

    def sites(self) -> Tuple[SitePlanEntry, ...]:
        """Return the site entries in insertion order."""
        return tuple(self._sites.values())

    def site(self, position: Tuple[int, int]) -> SitePlanEntry:
        """Return one site entry (raises KeyError when absent)."""
        return self._sites[tuple(position)]

    def eligible_sites(self) -> Tuple[SitePlanEntry, ...]:
        """Return the sites that are run-wide ELIGIBLE (the reconstruction set)."""
        return tuple(s for s in self.sites() if s.run_wide_applicability == RUN_WIDE_ELIGIBLE)

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
        if not self.revision_id or not isinstance(self.revision_id, str):
            raise ValueError("revision_id must be a non-empty string")
        if len(self.geometry) != 2 or any(v <= 0 for v in self.geometry):
            raise ValueError("geometry must be a positive (ny, nx)")
        if not is_bayer_phase(self.cfa_phase):
            raise ValueError(f"cfa_phase must be a Bayer phase, got {self.cfa_phase!r}")
        if len(self.roi_origin) != 2:
            raise ValueError("roi_origin must be a pair (oy, ox)")
        if not self.frame_ids:
            raise ValueError("a run must have at least one frame")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame ids must be unique within a run")


# ---------------------------------------------------------------------------
# Preflight (PASS 1): consume the selected revision + post-calibration facts,
# build and freeze the plan, write nothing.
# ---------------------------------------------------------------------------


def _frame_geometry(frame: CalibratedFrame) -> Tuple[int, int]:
    return tuple(int(s) for s in np.asarray(frame.calibration.data).shape)


def _revision_geometry(revision: Revision) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    geo = revision.sensor_identity.geometry
    cfa_phase = geo.cfa_phase
    if not is_bayer_phase(cfa_phase):
        raise UnsupportedCfaError(
            f"revision {revision.revision_id!r} binds cfa_phase {cfa_phase!r}; "
            "same-CFA reconstruction requires a Bayer CFA phase"
        )
    roi_origin = geo.roi_origin if geo.roi_origin is not None else (0, 0)
    return cfa_phase, tuple(int(s) for s in geo.shape), tuple(int(o) for o in roi_origin)


def preflight(
    resolution: BpmResolution,
    frames: Tuple[CalibratedFrame, ...],
    *,
    operator: ReconstructionOperator = DEFAULT_OPERATOR,
    run_id: Optional[str] = None,
) -> "PreparationOutcome":
    """Build and freeze a run preparation plan from the resolved base + facts.

    Returns a :class:`PreparationOutcome`. For ``CALIBRATION_ONLY`` /
    ``BASE_ERROR`` resolutions the outcome carries the confinement (ordinary
    calibration unchanged / typed failure); for ``SELECTED`` it builds and freezes
    the plan (``outcome == PREPARED``). Preflight writes nothing and is pure over
    the supplied post-calibration facts.

    Only ``action_eligible_sites(revision)`` enter the plan: a
    ``NO_ACTION_REQUIRED`` / ``ABSTAIN_*`` site in the base is preserved intact
    (never reconstructed) — mission §29.
    """
    if not isinstance(resolution, BpmResolution):
        raise ValueError("resolution must be a BpmResolution")
    if not frames:
        raise EmptyFramesError("cannot preflight a run over zero frames")

    if resolution.is_calibration_only:
        return PreparationOutcome(
            outcome=OUTCOME_CALIBRATION_ONLY,
            reason_code=resolution.reason_code,
            provenance=resolution.provenance,
        )
    if resolution.is_base_error:
        return PreparationOutcome(
            outcome=OUTCOME_BASE_ERROR,
            reason_code=resolution.reason_code,
            provenance=resolution.provenance,
        )
    if not resolution.is_selected or resolution.revision is None:
        raise PreparationError("resolution must be SELECTED, CALIBRATION_ONLY or BASE_ERROR")

    revision = resolution.revision
    cfa_phase, shape, roi_origin = _revision_geometry(revision)
    frame_ids = tuple(f.frame_id for f in frames)
    effective_run_id = run_id if run_id else revision.revision_id

    # Validate every frame's shape against the bound geometry before any decision.
    for frame in frames:
        if _frame_geometry(frame) != shape:
            raise PreparationError(
                f"frame {frame.frame_id!r} shape {_frame_geometry(frame)} does not match "
                f"the revision geometry {shape}"
            )

    plan = PreparationPlan(
        run_id=effective_run_id,
        revision_id=revision.revision_id,
        geometry=shape,
        cfa_phase=cfa_phase,
        roi_origin=roi_origin,
        frame_ids=frame_ids,
        frozen=False,
    )

    eligible = action_eligible_sites(revision)
    for site in eligible:
        position = site.position
        per_frame: list = []
        for frame in frames:
            data = np.asarray(frame.calibration.data, dtype=np.float32)
            usable = np.asarray(frame.calibration.mask, dtype=np.uint16) == 0
            per_frame.append(
                donors_available(
                    data, position[0], position[1],
                    cfa_phase=cfa_phase, roi_origin=roi_origin, usable=usable,
                    operator=operator,
                )
            )
        entry = SitePlanEntry(
            site_id=f"site-{position[0]}-{position[1]}",
            position=position,
            action_state=site.action_state,
            knowledge_state=site.knowledge_state,
            per_frame_donor_available=tuple(per_frame),
        )
        plan.add_site(entry)

    plan.freeze()
    return PreparationOutcome(
        outcome=OUTCOME_PREPARED,
        reason_code="",
        plan=plan,
        provenance=resolution.provenance,
    )


# ---------------------------------------------------------------------------
# Execution (PASS 2): apply the frozen plan exactly, aborting on contradiction.
# ---------------------------------------------------------------------------


def execute(
    plan: PreparationPlan,
    frames: Tuple[CalibratedFrame, ...],
    *,
    operator: ReconstructionOperator = DEFAULT_OPERATOR,
) -> Tuple[PreparedCalibrationResult, ...]:
    """Apply the frozen plan exactly over the run's frames.

    For each frame, every ELIGIBLE site is re-validated against the frozen donor
    fact; a contradiction (the observed donor availability disagrees with the
    frozen plan) raises :class:`PlanInvalidError` and aborts the prepared output —
    never a silent per-frame switch and never a local improvised fallback. A site
    whose reconstruction finds no donor (which the frozen plan said could not
    happen) also raises :class:`PlanInvalidError` (mission §40).
    """
    if not isinstance(plan, PreparationPlan):
        raise ValueError("plan must be a PreparationPlan")
    plan._require_frozen()
    if len(frames) != len(plan.frame_ids):
        raise PreparationError(
            f"plan expects {len(plan.frame_ids)} frames, got {len(frames)}"
        )

    results: list = []
    for index, frame in enumerate(frames):
        if frame.frame_id != plan.frame_ids[index]:
            raise PreparationError(
                f"frame order/id mismatch at index {index}: expected "
                f"{plan.frame_ids[index]!r}, got {frame.frame_id!r}"
            )
        data = np.asarray(frame.calibration.data, dtype=np.float32)
        mask = np.asarray(frame.calibration.mask, dtype=np.uint16)
        usable = mask == 0

        prepared = np.array(data, dtype=np.float32)
        reconstructed = np.zeros(data.shape, dtype=np.uint8)
        prepared_count = 0

        for entry in plan.eligible_sites():
            y, x = entry.position
            observed = donors_available(
                data, y, x, cfa_phase=plan.cfa_phase, roi_origin=plan.roi_origin,
                usable=usable, operator=operator,
            )
            assumed = entry.per_frame_donor_available[index]
            if observed != assumed:
                raise PlanInvalidError(
                    f"run {plan.run_id!r}: frozen plan assumed donor availability "
                    f"{assumed!r} for site {entry.site_id!r} on frame "
                    f"{frame.frame_id!r} but observed {observed!r} — aborting rather "
                    "than improvising or switching silently"
                )

            value = reconstruct_site_pixel(
                data, y, x, cfa_phase=plan.cfa_phase, roi_origin=plan.roi_origin,
                usable=usable, operator=operator,
            )
            if value is NO_VALUE:
                # A site the frozen plan declared reconstructable must never
                # discover it has no donor at execution (mission §40) — no local
                # improvised fallback.
                raise PlanInvalidError(
                    f"run {plan.run_id!r}: site {entry.site_id!r} discovered no valid "
                    f"donor on frame {frame.frame_id!r} — the preflight should have "
                    "rejected it; aborting the prepared output"
                )
            prepared[y, x] = np.float32(value)
            reconstructed[y, x] = 1
            prepared_count += 1

        # usable_mask: valid measurement AND not reconstructed (donor eligibility).
        usable_out = usable & ~(reconstructed.astype(bool))

        results.append(
            PreparedCalibrationResult(
                calibration=frame.calibration,
                prepared_data=prepared,
                measurement_dq=mask,
                reconstructed_mask=reconstructed,
                usable_mask=usable_out,
                operator_id=operator.operator_id,
                operator_version=operator.version,
                prepared_site_count=prepared_count,
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# Outcome + convenience facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparationOutcome:
    """The typed outcome of preparation (confinement or a frozen plan + results)."""

    outcome: str  # PREPARED | CALIBRATION_ONLY | BASE_ERROR
    reason_code: str
    plan: Optional[PreparationPlan] = None
    results: Optional[Tuple[PreparedCalibrationResult, ...]] = None
    provenance: Tuple[ProvenanceNote, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in (OUTCOME_PREPARED, OUTCOME_CALIBRATION_ONLY, OUTCOME_BASE_ERROR):
            raise ValueError(f"PreparationOutcome.outcome invalid: {self.outcome!r}")
        if self.outcome == OUTCOME_PREPARED and self.plan is None:
            raise ValueError("outcome PREPARED requires a plan")
        if self.outcome != OUTCOME_PREPARED and self.plan is not None:
            raise ValueError("non-PREPARED outcome must not carry a plan")
        if self.results is not None:
            object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "provenance", tuple(self.provenance))


def apply_preparation(
    resolution: BpmResolution,
    frames: Tuple[CalibratedFrame, ...],
    *,
    operator: ReconstructionOperator = DEFAULT_OPERATOR,
    run_id: Optional[str] = None,
) -> PreparationOutcome:
    """Preflight + freeze + execute in one bounded call (convenience facade).

    For ``CALIBRATION_ONLY`` / ``BASE_ERROR`` the outcome carries the confinement
    (ordinary calibration unchanged / typed failure). For ``SELECTED`` the frozen
    plan is built and immediately executed, and the outcome carries both the plan
    and the per-frame :class:`PreparedCalibrationResult` tuple.
    """
    outcome = preflight(resolution, frames, operator=operator, run_id=run_id)
    if outcome.outcome != OUTCOME_PREPARED:
        return outcome
    assert outcome.plan is not None
    results = execute(outcome.plan, frames, operator=operator)
    return PreparationOutcome(
        outcome=OUTCOME_PREPARED,
        reason_code="",
        plan=outcome.plan,
        results=results,
        provenance=outcome.provenance,
    )


__all__ = [
    "OUTCOME_BASE_ERROR",
    "OUTCOME_CALIBRATION_ONLY",
    "OUTCOME_PREPARED",
    "PREPARATION_SCHEMA_VERSION",
    "RC_DONORS_UNAVAILABLE_RUN_WIDE",
    "RUN_WIDE_ABSTAIN",
    "RUN_WIDE_ELIGIBLE",
    "CalibratedFrame",
    "DuplicateSiteError",
    "EmptyFramesError",
    "PlanFrozenError",
    "PlanInvalidError",
    "PlanNotFrozenError",
    "PreparationError",
    "PreparationOutcome",
    "PreparationPlan",
    "SitePlanEntry",
    "UnsupportedCfaError",
    "apply_preparation",
    "derive_run_wide",
    "execute",
    "preflight",
]
