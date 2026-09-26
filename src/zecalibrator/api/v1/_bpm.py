"""Private Bad Pixel Database exposure facade (NOT a public capability).

This module is the thin, underscore-prefixed bridge the CLI and the GUI use to
expose the Bad Pixel Database preview in the normal flow *without* changing the
public ``zecalibrator.api.v1`` contract: the six capabilities, ``API_VERSION``,
``__all__`` and ``CalibrationResult`` v1 stay frozen. It mirrors the private
``_auto_route.py`` module (also not exported from ``api.v1``).

It wraps two existing layers, nothing else:

* the application-layer orchestrator (``zecalibrator.application.bpm_preview``)
  — the only application-layer entry for the BPM engine; it never reconstructs a
  pixel while the scientific gate is closed (§3/§35/§74);
* the single user setting (``zecalibrator.bpm.settings``) plus the injected
  ``zecalibrator.storage.StoragePaths`` (§16/§17: one existing persistence
  mechanism, no second configuration system).

The CLI is a thin facade over ``api.v1`` only, and the GUI must never import
``zecalibrator.bpm`` directly; importing this module (``zecalibrator.api.v1._bpm``)
satisfies both constraints. Because this module wraps the heavy BPM/application
layers, every heavy import is lazy (inside functions), so ``import
zecalibrator.cli`` stays NumPy-free.

User-facing terminology (§41) is "Bad Pixel Database" / "Bad Pixel Map" — never
``RTS`` / ``SensorProfile`` / ``PreparationPlan``. The scientific gate is never
toggled here: there is no "force BPM" flag and no scientific setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    "apply_bpm_run_wide",
    "auto_route_batch",
    "bpm_application_status_line",
    "bpm_application_synthesis",
    "bpm_creation_decision",
    "bpm_status_line",
    "bpm_settings",
    "calibrate_batch",
    "calibrate_frame",
    "create_bpm_map_from_binding",
    "default_bad_pixel_database_root",
    "default_bpm_settings",
    "ensure_bpm_root",
    "load_bpm_settings",
    "make_bpm_run_wide_seam",
    "make_bpm_seam",
    "resolve_bpm_root",
    "resolve_storage_paths",
    "save_bpm_settings",
    "storage_from_mapping",
    "synthesis_text",
    "validate_bpm_root",
]


# ---------------------------------------------------------------------------
# Settings + storage (wraps zecalibrator.bpm.settings + zecalibrator.storage)
# ---------------------------------------------------------------------------
def _bpm_settings_module():
    from zecalibrator.bpm.settings import (
        BpmSettings,
        default_bad_pixel_database_root,
        default_settings,
        load_settings,
        resolve_bad_pixel_database_root,
        save_settings,
    )
    return {
        "BpmSettings": BpmSettings,
        "default_bad_pixel_database_root": default_bad_pixel_database_root,
        "default_settings": default_settings,
        "load_settings": load_settings,
        "resolve_bad_pixel_database_root": resolve_bad_pixel_database_root,
        "save_settings": save_settings,
    }


def resolve_storage_paths(**kwargs):
    """Resolve :class:`zecalibrator.storage.StoragePaths` (no side effects)."""
    from zecalibrator.storage import resolve_paths

    return resolve_paths(**kwargs)


def load_bpm_settings(config_dir):
    """Load the single BPM setting file (``bad_pixel_database_root``)."""
    return _bpm_settings_module()["load_settings"](Path(config_dir))


def save_bpm_settings(config_dir, settings) -> None:
    """Atomically persist the single BPM setting (never a second config system)."""
    return _bpm_settings_module()["save_settings"](Path(config_dir), settings)


def default_bpm_settings():
    """The default BPM settings snapshot (no configured root)."""
    return _bpm_settings_module()["default_settings"]()


def bpm_settings(root: Optional[str] = None):
    """Build a BPM settings snapshot with an optional configured root."""
    BpmSettings = _bpm_settings_module()["BpmSettings"]
    if root is None or not str(root).strip():
        return default_bpm_settings()
    return BpmSettings(bad_pixel_database_root=str(root))


def default_bad_pixel_database_root(storage) -> Path:
    """The portable default BPM base root derived from ``StoragePaths``."""
    return _bpm_settings_module()["default_bad_pixel_database_root"](storage)


def resolve_bpm_root(storage, settings) -> Path:
    """The effective BPM base root: configured value, else the portable default."""
    return _bpm_settings_module()["resolve_bad_pixel_database_root"](storage, settings)


def storage_from_mapping(mapping):
    """Rebuild a :class:`StoragePaths` from ``StoragePaths.as_dict()`` output."""
    from zecalibrator.storage import StoragePaths

    return StoragePaths(
        user_config_path=Path(mapping["user_config_path"]),
        user_data_path=Path(mapping["user_data_path"]),
        user_cache_path=Path(mapping["user_cache_path"]),
        user_state_path=Path(mapping["user_state_path"]),
        user_log_path=Path(mapping["user_log_path"]),
    )


# ---------------------------------------------------------------------------
# Root validation / creation (§16: Browse → validate/create folder if absent)
# ---------------------------------------------------------------------------
def validate_bpm_root(root) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a candidate BPM root path.

    ``ok`` is True for an existing writable directory or a non-existent path
    that may be created; ``ok`` is False for an existing non-directory (a file)
    or an existing directory that is not writable.
    """
    import os

    p = Path(root)
    if p.exists():
        if not p.is_dir():
            return False, "the selected location is not a folder"
        if not os.access(p, os.W_OK):
            return False, "the selected folder is not writable"
        return True, ""
    return True, ""


def ensure_bpm_root(root, *, create: bool = True) -> Tuple[str, Optional[str]]:
    """Validate and (optionally) create the BPM root folder.

    Returns ``(normalized_path, error)`` where ``error`` is ``None`` on success.
    Never raises for an anticipated bad path: a file-as-path or an unwritable
    folder is reported as a structured error string, never a traceback.
    """
    import os

    p = Path(root)
    if p.exists():
        if not p.is_dir():
            return str(p), "the selected location is not a folder"
        if not os.access(p, os.W_OK):
            return str(p), "the selected folder is not writable"
        return str(p), None
    if not create:
        return str(p), None
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return str(p), f"could not create the folder: {exc}"
    return str(p), None


# ---------------------------------------------------------------------------
# Orchestrator (wraps zecalibrator.application.bpm_preview — the only engine entry)
# ---------------------------------------------------------------------------
def _orchestrator():
    from zecalibrator.application.bpm_preview import BpmPreviewSeam

    return BpmPreviewSeam


def make_bpm_seam(storage, settings, frame_id: str = "frame"):
    """Build the safe preview seam (the orchestrator's duck-typed front door)."""
    return _orchestrator()(storage, settings, frame_id=frame_id)


# ---------------------------------------------------------------------------
# Calibration entry points that thread the seam into the ordinary chain (§3).
# These mirror the public ``calibrate_batch`` / ``calibrate_frame`` /
# ``auto_route_batch`` exactly; the ONLY addition is the optional ``seam``.
# ---------------------------------------------------------------------------
def calibrate_frame(source, plan, options=None, *, cancel=None, progress=None, seam=None):
    """``calibrate_frame`` + an optional ``BpmPreviewSeam`` (additive only)."""
    from zecalibrator.api.v1 import _io
    from zecalibrator.api.v1.calibration import OPERATION_ID, _calibrate_frame_impl
    from zecalibrator.api.v1.errors import InvalidRequestError
    from zecalibrator.api.v1.models import (
        ArrayFrameSource,
        CalibrationPlan,
        ExecutionOptions,
        FitsFrameSource,
    )
    from zecalibrator.application.cancellation import CancellationToken

    if not isinstance(source, (FitsFrameSource, ArrayFrameSource)):
        raise InvalidRequestError("source must be a FitsFrameSource or ArrayFrameSource")
    if not isinstance(plan, CalibrationPlan):
        raise InvalidRequestError("plan must be a CalibrationPlan")
    options = options if options is not None else ExecutionOptions()
    if not isinstance(options, ExecutionOptions):
        raise InvalidRequestError("options must be ExecutionOptions")

    token = cancel or CancellationToken()
    obs = _io.normalize_progress(progress, OPERATION_ID)
    return _calibrate_frame_impl(source, plan, options, token=token, obs=obs, slot=None, bpm_preview=seam)


def calibrate_batch(frames, request, library, policy, options=None, *, cancel=None, progress=None, seam=None, run_wide=None):
    """``calibrate_batch`` + optional preview/run-wide BPM seams (additive only).

    ``seam`` is the inert P4.1 preview seam (observation only); ``run_wide`` is
    the LOT 3 batch-level run-wide application seam (the authorized path).
    """
    from zecalibrator.api.v1 import _io
    from zecalibrator.api.v1.batch import (
        BATCH_OPERATION_ID,
        _batch_generator,
        _validate_batch_args,
        ordered_frames,
    )
    from zecalibrator.api.v1.errors import InvalidRequestError
    from zecalibrator.api.v1.models import ArrayFrameSource, FitsFrameSource
    from zecalibrator.application.cancellation import CancellationToken

    options = _validate_batch_args(frames, request, library, policy, options)
    frame_list = ordered_frames(frames)
    for frame in frame_list:
        if not isinstance(frame, (FitsFrameSource, ArrayFrameSource)):
            raise InvalidRequestError("every frame must be a FitsFrameSource or ArrayFrameSource")

    token = cancel if cancel is not None else CancellationToken()
    obs = _io.normalize_progress(progress, BATCH_OPERATION_ID)
    return _batch_generator(
        frame_list, request, library, policy, options, token, obs,
        bpm_preview=seam, bpm_run_wide=run_wide,
    )


def auto_route_batch(frames, library, policy, options=None, *, cancel=None, progress=None, collision_decision=None, seam=None, run_wide=None):
    """``auto_route_batch`` + optional preview/run-wide BPM seams (additive only)."""
    from zecalibrator.api.v1._auto_route import auto_route_batch as _impl

    return _impl(
        frames, library, policy, options,
        cancel=cancel, progress=progress, collision_decision=collision_decision,
        bpm_preview=seam, bpm_run_wide=run_wide,
    )


# ---------------------------------------------------------------------------
# Status / log synthesis (§40/§50/§51) — user terminology, never "applied" while
# reconstructed == 0.
# ---------------------------------------------------------------------------
def _synthesis(outcome, root_fallback) -> dict:
    db = (getattr(outcome, "database_root", None) if outcome is not None else None) or root_fallback or "none"
    profile = getattr(outcome, "revision_id", None) if outcome is not None else None
    eligible = getattr(outcome, "eligible_site_count", 0) if outcome is not None else 0
    application_enabled = bool(getattr(outcome, "application_enabled", False)) if outcome is not None else False
    reconstructed = getattr(outcome, "reconstructed_site_count", 0) if outcome is not None else 0
    calibration_only = bool(getattr(outcome, "calibration_only", True)) if outcome is not None else True
    return {
        "database": db,
        "profile": profile if profile else "none",
        "eligible_sites": eligible,
        "application": "disabled" if not application_enabled else "enabled",
        "reconstructed": reconstructed,
        "mode": "calibration-only" if calibration_only else "reconstruction",
    }


def synthesis_text(outcome, root_fallback=None) -> list:
    """§50 log synthesis (six lines, user terminology)."""
    s = _synthesis(outcome, root_fallback)
    return [
        f"Bad Pixel Database: {s['database']}",
        f"Sensor profile: {s['profile']}",
        f"Eligible sites: {s['eligible_sites']}",
        "BPM scientific application: " + s["application"],
        "Reconstructed sites: " + str(s["reconstructed"]),
        "Result mode: " + s["mode"],
    ]


def bpm_status_line(outcome, root_fallback=None) -> str:
    """§40 discrete GUI status line (user terminology)."""
    s = _synthesis(outcome, root_fallback)
    db = "selected" if s["database"] not in ("none", None) else "not configured"
    profile = "selected" if s["profile"] != "none" else "none"
    correction = "applied" if s["reconstructed"] > 0 else "not required / preview disabled"
    return (
        f"Bad Pixel Database: {db} · Profile: {profile} · "
        f"BPM correction: {correction}"
    )


def apply_bpm_run_wide(
    *,
    storage,
    settings,
    identity,
    frames,
    operator=None,
    run_id=None,
):
    """Batch-level run-wide BPM application (LOT 2), reached through the facade.

    Thin bridge to :mod:`zecalibrator.application.bpm_application`: resolves the
    base from storage/settings, resolves a compatible promoted revision against
    ``identity``, then preflights (freezes) and executes the run-wide plan over
    the supplied post-calibration ``frames``. The P4.1 preview seam is untouched.
    """
    from zecalibrator.application.bpm_application import apply_bpm_run_wide as _impl
    from zecalibrator.bpm.reconstruction import DEFAULT_OPERATOR

    return _impl(
        storage=storage,
        settings=settings,
        identity=identity,
        frames=frames,
        operator=operator if operator is not None else DEFAULT_OPERATOR,
        run_id=run_id,
    )


def bpm_application_synthesis(outcome, *, map_origin="selected") -> list:
    """§11 run synthesis (application outcome); ``applied`` gated on reconstructed > 0."""
    from zecalibrator.application.bpm_application import synthesis_lines

    return synthesis_lines(outcome, map_origin=map_origin)


def make_bpm_run_wide_seam(storage, settings, run_id=None):
    """Build the batch-level run-wide BPM application seam (LOT 3).

    Thin bridge to :class:`~zecalibrator.application.bpm_application.BpmRunWideSeam`:
    the batch generator accumulates every calibrated frame through ``record``,
    ``finalize`` applies the frozen run-wide plan once, and ``prepared`` hands
    back the per-frame corrected CFA for output writing. When no compatible map
    exists the run stays CALIBRATION_ONLY / BASE_ERROR and ordinary outputs are
    written unchanged.
    """
    from zecalibrator.application.bpm_application import BpmRunWideSeam
    from zecalibrator.bpm.reconstruction import DEFAULT_OPERATOR

    return BpmRunWideSeam(storage, settings, operator=DEFAULT_OPERATOR, run_id=run_id)


def bpm_application_status_line(outcome, *, map_origin="selected") -> str:
    """§40 discrete GUI status line for the run-wide application outcome.

    User terminology ("Bad Pixel Map" / "Bad Pixel Database"); the word
    ``applied`` is STRICTLY conditional on ``reconstructed_total > 0``.
    """
    if outcome is None:
        return f"Bad Pixel Map: {map_origin} · BPM correction: none"
    correction = "applied" if outcome.reconstructed_total > 0 else "none"
    return (
        f"Bad Pixel Map: {map_origin} · Bad pixels: {outcome.bad_pixel_count} · "
        f"BPM correction: {correction} · Reconstructed sites: {outcome.reconstructed_sites}"
    )


def bpm_creation_decision(storage, settings, light_constraints, *, dark_available=False) -> dict:
    """Map-creation decision for one run's camera identity (§3/§9).

    Resolves the Bad Pixel Database against the run's sensor identity and
    combines it with whether a compatible Master Dark was selected for the run.
    Returns a small user-facing decision dict:

    * ``{"action": "use"}`` — a compatible map exists → use it automatically,
      never ask;
    * ``{"action": "propose"}`` — no compatible map and a Master Dark is
      available → offer to create one (at most once per run);
    * ``{"action": "unavailable"}`` — no compatible map and no Master Dark →
      never offer an impossible creation;
    * ``{"action": "error"}`` — the base is corrupt/invalid/incompatible → a
      typed, discreet note; never offer a creation into a corrupt base.
    """
    from zecalibrator.bpm.identity import sensor_identity_from_light_constraints
    from zecalibrator.bpm.store import resolve_bad_pixel_database

    identity = sensor_identity_from_light_constraints(light_constraints)
    root = resolve_bpm_root(storage, settings)
    resolution = resolve_bad_pixel_database(root, identity)
    if resolution.is_selected:
        return {"action": "use", "reason_code": ""}
    if resolution.is_base_error:
        return {"action": "error", "reason_code": resolution.reason_code}
    if dark_available:
        return {"action": "propose", "reason_code": resolution.reason_code}
    return {"action": "unavailable", "reason_code": resolution.reason_code}


def create_bpm_map_from_binding(storage, settings, binding, *, cancel=None):
    """Create a Bad Pixel Map from an already-selected dark master binding.

    Decodes the binding's dark FITS (same decode path the calibration uses) and
    builds the :class:`SelectedDarkMaster` from the descriptor identity, then
    stores new immutable candidate+promoted revisions into the configured base
    (``create_bad_pixel_map``). No master is fabricated: the dark is already
    selected by the matcher.
    """
    from zecalibrator.api.v1 import _io
    from zecalibrator.api.v1.calibration import _declaration_from_descriptor
    from zecalibrator.application.cancellation import CancellationToken
    from zecalibrator.bpm.identity import ReadoutContext, SensorIdentity
    from zecalibrator.bpm.map_creation import SelectedDarkMaster, create_bad_pixel_map

    token = cancel or CancellationToken()
    desc = binding.role_descriptor
    if desc.master_type != "dark":
        raise ValueError(f"binding is a {desc.master_type!r}, not a dark master")
    locator = binding.locators[0]
    data = _io.read_bytes(locator.path, cancel=token)
    declaration = _declaration_from_descriptor(desc)
    decoded = _io.decode_fits_from_bytes(
        data, locator.hdu, declaration, cancel=token, progress=None,
        admission="master", role="dark", flat_form=None,
    )
    identity = SensorIdentity(
        detector=desc.detector,
        geometry=desc.geometry,
        readout=ReadoutContext(
            gain=desc.acquisition.gain, offset=desc.acquisition.offset,
            readout_mode=desc.acquisition.readout_mode, adc_mode=desc.acquisition.adc_mode,
        ),
    )
    master = SelectedDarkMaster(
        data=decoded.data, identity=identity,
        metadata={"master_type": "dark", "source": "selected_master_dark"},
    )
    return create_bad_pixel_map(resolve_bpm_root(storage, settings), master)
