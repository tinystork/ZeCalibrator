"""Batch-level run-wide BPM application (LOT 2): resolve → preflight → freeze → execute.

This module is the **application path** authorized by this lot. The run-wide
:class:`~zecalibrator.bpm.preparation.PreparationPlan` is frozen over the *whole
run*'s post-calibration frames, then executed per frame — the two-time structure
(mission §8):

```text
calibrate the run's frames (ordinary path, masters unchanged)
   -> preflight(BpmResolution, calibrated frames)   [engine zecalibrator.bpm.preparation]
   -> PLAN GELÉ (PreparationPlan)
   -> execute(plan, frames)  -> prepared CFA per frame
```

It is **headless** (no Qt), imports no ``research/*``, and never imports
``zecalibrator.api.v1``. It complements — never replaces — the P4.1 preview seam
(:mod:`zecalibrator.application.bpm_preview`), which stays inert/aperçu-only
(``BPM_AUTOMATIC_APPLICATION_ENABLED`` is **not** flipped here).

Atomicity (mission §36, P-A normative), enforced by the engine and preserved by
this orchestration:

* a site applicable on **every** admissible frame is reconstructed on **all** of
  them, never ``apply/apply/skip/apply``;
* **one** frame with no valid donor ⇒ the site **abstains on the whole run**;
* a blocked site never disables an independently eligible site;
* order **post-calibration → BPM → debayer** (the applicator consumes only the
  post-calibration CFA, never the raw light, never after debayer);
* operator **``SAME_CFA_MEDIAN``** unchanged; no new algorithm;
* fallback **``CALIBRATION_ONLY``** when no base / no compatible profile /
  unqualified profile; **``BASE_ERROR``** (typed) when the base is corrupt;
* **no pixel outside ``reconstructed_mask`` is modified**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from zecalibrator.bpm.identity import SensorIdentity
from zecalibrator.bpm.lookup import BpmResolution, ProvenanceNote
from zecalibrator.bpm.preparation import (
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_PREPARED,
    CalibratedFrame,
    PreparedCalibrationResult,
    PreparationError,
    PreparationOutcome,
    PreparationPlan,
    execute,
    preflight,
)
from zecalibrator.bpm.reconstruction import DEFAULT_OPERATOR, ReconstructionOperator
from zecalibrator.bpm.revision import action_eligible_sites
from zecalibrator.bpm.settings import BpmSettings, resolve_bad_pixel_database_root
from zecalibrator.bpm.store import resolve_bad_pixel_database
from zecalibrator.storage import StoragePaths


@dataclass(frozen=True)
class BpmApplicationOutcome:
    """The typed outcome of the run-wide BPM application (resolve + apply).

    ``outcome`` is ``PREPARED`` | ``CALIBRATION_ONLY`` | ``BASE_ERROR`` (the
    engine confinement vocabulary). ``resolution`` is the engine's resolved
    selection against the base. ``plan`` (frozen) and ``results`` are present
    only when ``PREPARED``. ``reconstructed_total`` is the run-wide sum of
    ``prepared_site_count`` across frames (the only number that may render the
    word ``applied``). ``bad_pixel_count`` is the number of ACTION-ELIGIBLE
    sites in the selected revision (0 when not selected).
    """

    outcome: str
    reason_code: str
    resolution: BpmResolution
    plan: Optional[PreparationPlan] = None
    results: Optional[Tuple[PreparedCalibrationResult, ...]] = None
    reconstructed_total: int = 0
    bad_pixel_count: int = 0
    provenance: Tuple[ProvenanceNote, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in (OUTCOME_PREPARED, OUTCOME_CALIBRATION_ONLY, OUTCOME_BASE_ERROR):
            raise ValueError(f"BpmApplicationOutcome.outcome invalid: {self.outcome!r}")
        if self.outcome == OUTCOME_PREPARED and (self.plan is None or self.results is None):
            raise ValueError("outcome PREPARED requires a plan and results")
        if self.outcome != OUTCOME_PREPARED and (self.plan is not None or self.results is not None):
            raise ValueError("non-PREPARED outcome must not carry a plan or results")
        if isinstance(self.reconstructed_total, bool) or not isinstance(self.reconstructed_total, int) or self.reconstructed_total < 0:
            raise ValueError("reconstructed_total must be a non-negative int")
        if isinstance(self.bad_pixel_count, bool) or not isinstance(self.bad_pixel_count, int) or self.bad_pixel_count < 0:
            raise ValueError("bad_pixel_count must be a non-negative int")
        object.__setattr__(self, "results", tuple(self.results) if self.results is not None else None)
        object.__setattr__(self, "provenance", tuple(self.provenance))

    @property
    def reconstructed_per_frame(self) -> Tuple[int, ...]:
        """Per-frame ``prepared_site_count`` (empty when nothing was prepared)."""
        if self.results is None:
            return ()
        return tuple(r.prepared_site_count for r in self.results)

    @property
    def reconstructed_sites(self) -> int:
        """The number of distinct sites reconstructed (per-frame, constant run-wide)."""
        per_frame = self.reconstructed_per_frame
        return per_frame[0] if per_frame else 0

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "revision_id": (
                self.resolution.revision.revision_id
                if self.resolution.is_selected and self.resolution.revision is not None
                else None
            ),
            "bad_pixel_count": self.bad_pixel_count,
            "reconstructed_sites": self.reconstructed_sites,
            "reconstructed_total": self.reconstructed_total,
            "reconstructed_per_frame": list(self.reconstructed_per_frame),
            "provenance": [{"code": p.code, "detail": p.detail} for p in self.provenance],
        }


def apply_resolution_run_wide(
    resolution: BpmResolution,
    frames: Tuple[CalibratedFrame, ...],
    *,
    operator: ReconstructionOperator = DEFAULT_OPERATOR,
    run_id: Optional[str] = None,
) -> BpmApplicationOutcome:
    """Apply an already-resolved selection over the run's calibrated frames.

    This is the **two-time** structure made explicit: ``preflight`` (which builds
    and *freezes* the run-wide plan) then ``execute`` (which applies the frozen
    plan per frame). A confinement resolution (``CALIBRATION_ONLY`` /
    ``BASE_ERROR``) carries the engine's confinement unchanged — ordinary
    calibration is never modified.
    """
    if not isinstance(resolution, BpmResolution):
        raise ValueError("resolution must be a BpmResolution")
    if not frames:
        raise ValueError("frames must be a non-empty tuple of CalibratedFrame")

    preflighted = preflight(resolution, frames, operator=operator, run_id=run_id)
    if preflighted.outcome != OUTCOME_PREPARED:
        return BpmApplicationOutcome(
            outcome=preflighted.outcome,
            reason_code=preflighted.reason_code,
            resolution=resolution,
            provenance=preflighted.provenance,
        )

    plan = preflighted.plan
    assert plan is not None and plan.frozen
    results = execute(plan, frames, operator=operator)
    reconstructed_total = sum(r.prepared_site_count for r in results)
    revision = resolution.revision
    bad_pixel_count = len(action_eligible_sites(revision)) if revision is not None else 0
    return BpmApplicationOutcome(
        outcome=OUTCOME_PREPARED,
        reason_code="",
        resolution=resolution,
        plan=plan,
        results=results,
        reconstructed_total=reconstructed_total,
        bad_pixel_count=bad_pixel_count,
        provenance=preflighted.provenance,
    )


def apply_bpm_run_wide(
    *,
    storage: StoragePaths,
    settings: BpmSettings,
    identity: SensorIdentity,
    frames: Tuple[CalibratedFrame, ...],
    operator: ReconstructionOperator = DEFAULT_OPERATOR,
    run_id: Optional[str] = None,
) -> BpmApplicationOutcome:
    """Resolve the base from storage/settings and apply the run-wide plan.

    Resolves the effective BPM root (configured value, else the portable default),
    resolves a compatible promoted revision against ``identity``, then applies the
    frozen run-wide plan (``apply_resolution_run_wide``). No user scientific
    setting is added; no master is fabricated.
    """
    root = resolve_bad_pixel_database_root(storage, settings)
    resolution = resolve_bad_pixel_database(root, identity)
    return apply_resolution_run_wide(
        resolution, frames, operator=operator, run_id=run_id,
    )


class BpmRunWideSeam:
    """Batch-level run-wide BPM application seam (LOT 3).

    The ordinary batch path calibrates every frame then hands the whole run to
    this seam, which accumulates the post-calibration CFA frames (:meth:`record`),
    resolves the base once, and applies the frozen run-wide plan
    (:meth:`finalize`). This is the **two-time** structure made explicit at the
    batch boundary: calibrate-all -> freeze -> execute -> per-frame corrected
    CFA.

    The seam is duck-typed and opaque to the batch generator: it is called with
    the *engine* ``CalibrationResult`` (``zecalibrator.core.calibrate``) and the
    run's ``LightConstraints`` (from which the BPM sensor identity is projected).
    When a compatible map is selected, :meth:`prepared` returns the per-frame
    :class:`PreparedCalibrationResult` (whose ``prepared_data`` replaces the
    ordinary calibration output for writing); otherwise the run is
    ``CALIBRATION_ONLY`` / ``BASE_ERROR`` and ``prepared`` returns ``None`` (the
    ordinary output is written unchanged).

    A contained failure (an invalid run geometry / non-Bayer CFA / a frozen-plan
    contradiction) never turns a successful calibration batch into a failure:
    :meth:`finalize` records the error and leaves ``prepared`` returning
    ``None``, so the ordinary outputs are preserved.
    """

    __slots__ = ("storage", "settings", "operator", "run_id", "_frames",
                 "_by_id", "_identity", "outcome", "error")

    def __init__(
        self,
        storage: StoragePaths,
        settings: BpmSettings,
        *,
        operator: ReconstructionOperator = DEFAULT_OPERATOR,
        run_id: Optional[str] = None,
    ) -> None:
        self.storage = storage
        self.settings = settings
        self.operator = operator
        self.run_id = run_id
        self._frames: list = []
        self._by_id: dict = {}
        self._identity: Optional[SensorIdentity] = None
        self.outcome: Optional[BpmApplicationOutcome] = None
        self.error: Optional[str] = None

    @property
    def has_frames(self) -> bool:
        return bool(self._frames)

    def record(self, frame_id: str, calibration, light_constraints) -> None:
        """Accumulate one calibrated frame + its BPM selection identity.

        ``calibration`` is the engine :class:`CalibrationResult` (float32 CFA +
        uint16 DQ); ``light_constraints`` is the run's
        :class:`~zecalibrator.core.descriptors.LightConstraints`, from which the
        sensor identity is projected (reusing the normative liaison — no
        parallel simplified identity). The identity is captured once from the
        first frame (one run = one camera).
        """
        from zecalibrator.bpm.identity import sensor_identity_from_light_constraints

        if self._identity is None:
            self._identity = sensor_identity_from_light_constraints(light_constraints)
        self._frames.append(CalibratedFrame(frame_id=frame_id, calibration=calibration))

    def finalize(self) -> Optional[BpmApplicationOutcome]:
        """Apply the frozen run-wide plan over the accumulated frames (once).

        Returns the typed :class:`BpmApplicationOutcome` (``PREPARED`` /
        ``CALIBRATION_ONLY`` / ``BASE_ERROR``). A contained preparation failure
        (invalid run geometry, non-Bayer CFA, plan contradiction) is recorded on
        ``self.error`` and ``self.outcome`` stays ``None`` — the caller keeps
        writing ordinary outputs (never degrades a successful calibration).
        """
        if self._identity is None or not self._frames:
            return None
        try:
            self.outcome = apply_bpm_run_wide(
                storage=self.storage,
                settings=self.settings,
                identity=self._identity,
                frames=tuple(self._frames),
                operator=self.operator,
                run_id=self.run_id,
            )
        except (PreparationError, ValueError) as exc:
            self.error = str(exc)
            self.outcome = None
            return None
        if self.outcome.results is not None:
            for frame, result in zip(self._frames, self.outcome.results):
                self._by_id[frame.frame_id] = result
        return self.outcome

    def prepared(self, frame_id: str) -> Optional[PreparedCalibrationResult]:
        """Return the corrected per-frame result for ``frame_id``, or ``None``.

        ``None`` means the run is calibration-only (or the frame was not
        prepared), so the ordinary calibration output is written unchanged.
        """
        return self._by_id.get(frame_id)


def synthesis_lines(
    outcome: BpmApplicationOutcome,
    *,
    map_origin: str = "selected",
) -> list[str]:
    """§11 run synthesis. The word ``applied`` is STRICTLY conditional on
    ``reconstructed_total > 0`` (P4.1 invariant preserved). No alarming message
    when the run is confined to calibration-only (or the application was
    contained, ``outcome is None``).
    """
    if outcome is None:
        return [
            f"Bad Pixel Map: {map_origin}",
            "Bad pixels: 0",
            "BPM correction: none",
            "Reconstructed sites: 0",
        ]
    correction = "applied" if outcome.reconstructed_total > 0 else "none"
    return [
        f"Bad Pixel Map: {map_origin}",
        f"Bad pixels: {outcome.bad_pixel_count}",
        f"BPM correction: {correction}",
        f"Reconstructed sites: {outcome.reconstructed_sites}",
    ]


__all__ = [
    "BpmApplicationOutcome",
    "BpmRunWideSeam",
    "apply_bpm_run_wide",
    "apply_resolution_run_wide",
    "synthesis_lines",
]
