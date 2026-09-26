"""BPM preview orchestrator (application layer, LOT 1) — safe preview only.

This module wires the Bad Pixel Database engine (:mod:`zecalibrator.bpm`) into the
ordinary calibration path as an **optional, safe, side-effect-free preview**. It
resolves the BPM database, resolves the sensor identity, selects a compatible
promoted profile, derives/reports applicability through the engine's ``preflight``
and records provenance — but it **never** reconstructs a pixel and **never** calls
``apply_preparation`` while the scientific gate is closed (P4 decision:
``automatic BPM promotion = DISABLED``, ``automatic BPM application = DISABLED``).

The internal scientific gate (:data:`BPM_AUTOMATIC_APPLICATION_ENABLED`) is a
module-level constant — not a user setting, not an environment variable, not a
CLI/GUI flag (§23). P5 may flip it to ``True`` without changing the UX or the
pipeline (§68). It is a *process-level constant*, not a cryptographic lock: an
in-process ``monkeypatch`` could rewrite it at runtime, so it is a deliberate
release gate, not an absolute impossibility. This module is headless (no Qt) and
imports no ``research/*``.

This orchestrator **calls** the engine; it never re-implements the logic already
owned by ``zecalibrator.bpm`` (store/lookup/identity/preparation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from zecalibrator.bpm.identity import SensorIdentity, sensor_identity_from_light_constraints
from zecalibrator.bpm.lookup import BpmResolution, ProvenanceNote
from zecalibrator.bpm.preparation import (
    OUTCOME_PREPARED,
    CalibratedFrame,
    PreparationError,
    preflight,
)
from zecalibrator.bpm.revision import action_eligible_sites
from zecalibrator.bpm.settings import BpmSettings, resolve_bad_pixel_database_root
from zecalibrator.bpm.store import resolve_bad_pixel_database
from zecalibrator.bpm.vocabulary import (
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_SELECTED,
    REASON_BASE_CORRUPTED,
    REASON_BASE_INCOMPATIBLE,
    REASON_BASE_INVALID,
    REASON_NO_BASE,
    REASON_NO_PROFILE,
    REASON_UNQUALIFIED_PROFILE,
)
from zecalibrator.core.calibrate import CalibrationResult
from zecalibrator.storage import StoragePaths

#: The internal scientific gate (§22/§68). Automatic BPM application is DISABLED
#: in this lot; P5 flips this single constant to ``True``. It is NOT a user
#: setting, NOT an environment variable, and NOT a CLI/GUI flag (§23). It is a
#: process-level constant (not a cryptographic lock: an in-process ``monkeypatch``
#: could still rewrite it), so it gates the *released* behaviour, not an
#: adversarial runtime.
BPM_AUTOMATIC_APPLICATION_ENABLED = False

# ---------------------------------------------------------------------------
# Typed preview statuses (§25–§28).
# ---------------------------------------------------------------------------
STATUS_NO_DATABASE = "NO_DATABASE"            # no BPM configured/present (benign)
STATUS_NO_PROFILE = "NO_PROFILE"              # valid base, no compatible profile
STATUS_UNQUALIFIED_PROFILE = "UNQUALIFIED_PROFILE"  # compatible but not promoted
STATUS_SELECTED_PREVIEW = "SELECTED_PREVIEW"  # compatible promoted profile selected
STATUS_PREVIEW_ERROR = "PREVIEW_ERROR"        # the preview itself failed (contained)
# Base/profile corruption is a TYPED status (+ warning), never NO_PROFILE (§28).
STATUS_BASE_CORRUPTED = REASON_BASE_CORRUPTED
STATUS_BASE_INVALID = REASON_BASE_INVALID
STATUS_BASE_INCOMPATIBLE = REASON_BASE_INCOMPATIBLE

_STATUSES: tuple[str, ...] = (
    STATUS_NO_DATABASE,
    STATUS_NO_PROFILE,
    STATUS_UNQUALIFIED_PROFILE,
    STATUS_SELECTED_PREVIEW,
    STATUS_PREVIEW_ERROR,
    STATUS_BASE_CORRUPTED,
    STATUS_BASE_INVALID,
    STATUS_BASE_INCOMPATIBLE,
)

#: calibration-only reasons (§32).
REASON_CALIBRATION_ONLY_GATE_CLOSED = "AUTOMATIC_APPLICATION_DISABLED"
REASON_CALIBRATION_ONLY_BASE_ERROR = "BASE_ERROR"
REASON_CALIBRATION_ONLY_PREVIEW_ERROR = "PREVIEW_ERROR"

#: The engine reason-code -> orchestrator typed status mapping (§25–§28).
_STATUS_BY_REASON = {
    REASON_NO_BASE: STATUS_NO_DATABASE,
    REASON_NO_PROFILE: STATUS_NO_PROFILE,
    REASON_UNQUALIFIED_PROFILE: STATUS_UNQUALIFIED_PROFILE,
    REASON_BASE_CORRUPTED: STATUS_BASE_CORRUPTED,
    REASON_BASE_INVALID: STATUS_BASE_INVALID,
    REASON_BASE_INCOMPATIBLE: STATUS_BASE_INCOMPATIBLE,
}

#: calibration-only reason carried alongside each engine confinement reason.
_CALIBRATION_ONLY_REASON_BY_REASON = {
    REASON_NO_BASE: REASON_NO_BASE,
    REASON_NO_PROFILE: REASON_NO_PROFILE,
    REASON_UNQUALIFIED_PROFILE: REASON_UNQUALIFIED_PROFILE,
}


@dataclass(frozen=True)
class BpmPreviewOutcome:
    """The typed preview outcome carried alongside the ordinary calibration result.

    ``status`` is the orchestrator's typed status (§25–§28). ``reason_code`` is
    the stable engine reason (empty for ``SELECTED_PREVIEW``). ``database_root``
    is the resolved BPM root (``None`` when none configured). ``lookup_outcome``
    is the engine's resolution outcome (``SELECTED`` | ``CALIBRATION_ONLY`` |
    ``BASE_ERROR``). ``revision_id`` is the selected promoted revision id
    (present only when a compatible promoted profile was selected).
    ``eligible_site_count`` is the number of ACTION-ELIGIBLE sites in the selected
    profile; ``applicable_site_count`` is the preflight-derived run-wide eligible
    count (``None`` when not derivable). ``reconstructed_site_count`` is always
    ``0`` in this lot (locked by the closed gate — it is the first field that will
    change when P5 flips the gate). ``application_enabled`` mirrors the gate.
    ``calibration_only`` is ``True`` whenever the ordinary calibration result is
    used unchanged, and ``calibration_only_reason`` records why (§32).

    ``STATUS_PREVIEW_ERROR`` (with ``lookup_outcome == ""``) is reserved for a
    preview that failed internally and was *contained*: the ordinary calibration
    result is unaffected and the failure is carried only as a warning (§80/§92).
    """

    status: str
    reason_code: str
    database_root: Optional[str]
    lookup_outcome: str
    revision_id: Optional[str]
    eligible_site_count: int
    applicable_site_count: Optional[int]
    reconstructed_site_count: int
    application_enabled: bool
    calibration_only: bool
    calibration_only_reason: str
    warnings: Tuple[str, ...] = ()
    provenance: Tuple[ProvenanceNote, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if self.status not in _STATUSES:
            raise ValueError(f"BpmPreviewOutcome.status invalid: {self.status!r}")
        if self.lookup_outcome not in (
            OUTCOME_SELECTED, OUTCOME_CALIBRATION_ONLY, OUTCOME_BASE_ERROR,
        ):
            # A contained preview failure carries no engine lookup outcome.
            if not (self.status == STATUS_PREVIEW_ERROR and self.lookup_outcome == ""):
                raise ValueError(
                    f"BpmPreviewOutcome.lookup_outcome invalid: {self.lookup_outcome!r}"
                )
        if not isinstance(self.eligible_site_count, int) or self.eligible_site_count < 0:
            raise ValueError("eligible_site_count must be a non-negative int")
        if not isinstance(self.reconstructed_site_count, int) or self.reconstructed_site_count < 0:
            raise ValueError("reconstructed_site_count must be a non-negative int")
        if self.applicable_site_count is not None and (
            not isinstance(self.applicable_site_count, int) or self.applicable_site_count < 0
        ):
            raise ValueError("applicable_site_count must be a non-negative int or None")

    @property
    def prepared_site_count(self) -> int:
        """§33 alias: the number of prepared/reconstructed sites (always 0 here)."""
        return self.reconstructed_site_count

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "database_root": self.database_root,
            "lookup_outcome": self.lookup_outcome,
            "revision_id": self.revision_id,
            "eligible_site_count": self.eligible_site_count,
            "applicable_site_count": self.applicable_site_count,
            "reconstructed_site_count": self.reconstructed_site_count,
            "application_enabled": self.application_enabled,
            "calibration_only": self.calibration_only,
            "calibration_only_reason": self.calibration_only_reason,
            "warnings": list(self.warnings),
            "provenance": [{"code": p.code, "detail": p.detail} for p in self.provenance],
        }


def resolve_identity_from_light(light) -> SensorIdentity:
    """Resolve the BPM selection identity from a decoded frame's metadata.

    Reuses the existing normative contracts (``light_constraints_from_sensor_metadata``
    then ``sensor_identity_from_light_constraints``) — no parallel simplified
    identity is invented (mission §1 "réutiliser le contrat existant").
    """
    from zecalibrator.application.library import light_constraints_from_sensor_metadata

    return sensor_identity_from_light_constraints(
        light_constraints_from_sensor_metadata(light.metadata)
    )


def _derive_applicability(
    resolution: BpmResolution,
    calibration: Optional[CalibrationResult],
    frame_id: str,
) -> Tuple[Optional[int], Tuple[str, ...]]:
    """Derive run-wide applicability via the engine's ``preflight`` (safe, no apply).

    Returns ``(applicable_site_count, warnings)``. Preflight builds and freezes a
    plan over the supplied post-calibration frame; it never writes a pixel and
    never calls ``apply_preparation``. ``None`` means applicability was not
    derivable (no usable calibration frame, or a preflight refusal) — still never
    fatal.
    """
    if calibration is None or calibration.data is None or calibration.mask is None:
        return None, ("no post-calibration CFA to derive applicability",)
    try:
        frame = CalibratedFrame(frame_id=frame_id, calibration=calibration)
        outcome = preflight(resolution, (frame,))
    except PreparationError as exc:
        return None, (f"applicability underivable: {exc}",)
    if outcome.outcome != OUTCOME_PREPARED or outcome.plan is None:
        return None, (f"preflight outcome {outcome.outcome!r}; no frozen plan",)
    return len(outcome.plan.eligible_sites()), ()


def _selected_preview_outcome(
    resolution: BpmResolution,
    root_str: str,
    calibration: Optional[CalibrationResult],
    frame_id: str,
) -> BpmPreviewOutcome:
    revision = resolution.revision
    assert revision is not None
    eligible = len(action_eligible_sites(revision))
    applicable, warnings = _derive_applicability(resolution, calibration, frame_id)
    application_enabled = BPM_AUTOMATIC_APPLICATION_ENABLED
    provenance = resolution.provenance + (
        ProvenanceNote(
            "automatic-application-gate",
            f"automatic BPM application is {'enabled' if application_enabled else 'disabled'}",
        ),
    )
    return BpmPreviewOutcome(
        status=STATUS_SELECTED_PREVIEW,
        reason_code="",
        database_root=root_str,
        lookup_outcome=OUTCOME_SELECTED,
        revision_id=revision.revision_id,
        eligible_site_count=eligible,
        applicable_site_count=applicable,
        reconstructed_site_count=0,
        application_enabled=application_enabled,
        calibration_only=True,
        calibration_only_reason=REASON_CALIBRATION_ONLY_GATE_CLOSED,
        warnings=warnings,
        provenance=provenance,
    )


def _calibration_only_outcome(resolution: BpmResolution, root_str: str) -> BpmPreviewOutcome:
    reason = resolution.reason_code
    status = _STATUS_BY_REASON.get(reason, STATUS_NO_PROFILE)
    return BpmPreviewOutcome(
        status=status,
        reason_code=reason,
        database_root=root_str,
        lookup_outcome=OUTCOME_CALIBRATION_ONLY,
        revision_id=None,
        eligible_site_count=0,
        applicable_site_count=None,
        reconstructed_site_count=0,
        application_enabled=BPM_AUTOMATIC_APPLICATION_ENABLED,
        calibration_only=True,
        calibration_only_reason=_CALIBRATION_ONLY_REASON_BY_REASON.get(reason, reason),
        warnings=(),
        provenance=resolution.provenance,
    )


def _base_error_outcome(resolution: BpmResolution, root_str: str) -> BpmPreviewOutcome:
    reason = resolution.reason_code
    status = _STATUS_BY_REASON.get(reason, STATUS_BASE_INVALID)
    return BpmPreviewOutcome(
        status=status,
        reason_code=reason,
        database_root=root_str,
        lookup_outcome=OUTCOME_BASE_ERROR,
        revision_id=None,
        eligible_site_count=0,
        applicable_site_count=None,
        reconstructed_site_count=0,
        application_enabled=BPM_AUTOMATIC_APPLICATION_ENABLED,
        calibration_only=True,
        calibration_only_reason=REASON_CALIBRATION_ONLY_BASE_ERROR,
        warnings=("Bad Pixel Database base error; ordinary calibration preserved",),
        provenance=resolution.provenance,
    )


def orchestrate_bpm_preview(
    *,
    storage: StoragePaths,
    settings: BpmSettings,
    identity: SensorIdentity,
    calibration: Optional[CalibrationResult] = None,
    frame_id: str = "frame",
) -> BpmPreviewOutcome:
    """Run the safe BPM preview orchestration for one calibrated frame.

    Resolves the database root (``bpm.settings`` + ``StoragePaths``), resolves a
    compatible promoted profile (``bpm.store``), derives/reports applicability via
    the engine's ``preflight``, and records provenance. While
    :data:`BPM_AUTOMATIC_APPLICATION_ENABLED` is ``False`` (this lot) the result is
    always calibration-only: no reconstruction, no ``apply_preparation`` (§3/§24).
    """
    root = resolve_bad_pixel_database_root(storage, settings)
    resolution = resolve_bad_pixel_database(root, identity)

    if resolution.is_calibration_only:
        return _calibration_only_outcome(resolution, str(root))
    if resolution.is_base_error:
        return _base_error_outcome(resolution, str(root))
    # SELECTED: a compatible promoted profile exists. Derive applicability (safe
    # preflight) but never apply — the scientific gate is closed (§22).
    return _selected_preview_outcome(resolution, str(root), calibration, frame_id)


class BpmPreviewSeam:
    """Duck-typed execution seam the CalibrationResult producer can call.

    Carries the storage/settings inputs and captures the orchestrator outcome.
    The executor treats this object as opaque and calls :meth:`record` with the
    completed calibration result and the decoded light after ordinary calibration
    produced its result (§5 — one chain, optional, no parallel pipeline).
    """

    __slots__ = ("storage", "settings", "frame_id", "outcome")

    def __init__(
        self,
        storage: StoragePaths,
        settings: BpmSettings,
        *,
        frame_id: str = "frame",
    ) -> None:
        self.storage = storage
        self.settings = settings
        self.frame_id = frame_id
        self.outcome: Optional[BpmPreviewOutcome] = None

    def record(self, result: CalibrationResult, light) -> None:
        """Run the preview orchestration and capture the typed outcome."""
        identity = resolve_identity_from_light(light)
        self.outcome = orchestrate_bpm_preview(
            storage=self.storage,
            settings=self.settings,
            identity=identity,
            calibration=result,
            frame_id=self.frame_id,
        )

    def record_failure(self, exc: BaseException) -> None:
        """Contain a preview failure: record it as a warning, never propagate.

        The executor calls this (duck-typed) when :meth:`record` raised, so a
        failed preview can never degrade an already-successful calibration
        (§80/§92). The failure is carried only as a warning on a
        ``STATUS_PREVIEW_ERROR`` outcome.
        """
        self.outcome = BpmPreviewOutcome(
            status=STATUS_PREVIEW_ERROR,
            reason_code=REASON_CALIBRATION_ONLY_PREVIEW_ERROR,
            database_root=None,
            lookup_outcome="",
            revision_id=None,
            eligible_site_count=0,
            applicable_site_count=None,
            reconstructed_site_count=0,
            application_enabled=BPM_AUTOMATIC_APPLICATION_ENABLED,
            calibration_only=True,
            calibration_only_reason=REASON_CALIBRATION_ONLY_PREVIEW_ERROR,
            warnings=(f"BPM preview failed (contained); ordinary calibration preserved: {exc}",),
        )


__all__ = [
    "BPM_AUTOMATIC_APPLICATION_ENABLED",
    "BpmPreviewOutcome",
    "BpmPreviewSeam",
    "REASON_CALIBRATION_ONLY_BASE_ERROR",
    "REASON_CALIBRATION_ONLY_GATE_CLOSED",
    "REASON_CALIBRATION_ONLY_PREVIEW_ERROR",
    "STATUS_BASE_CORRUPTED",
    "STATUS_BASE_INCOMPATIBLE",
    "STATUS_BASE_INVALID",
    "STATUS_NO_DATABASE",
    "STATUS_NO_PROFILE",
    "STATUS_PREVIEW_ERROR",
    "STATUS_SELECTED_PREVIEW",
    "STATUS_UNQUALIFIED_PROFILE",
    "orchestrate_bpm_preview",
    "resolve_identity_from_light",
]
