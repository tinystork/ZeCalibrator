"""P4-P1 LOT 1 orchestrator tests: typed statuses, the scientific gate, the
§34 mandatory guard (1 eligible site → 0 reconstructed, calibration-only), and
the §33 prepared_site_count invariant."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.application.bpm_preview import (
    BPM_AUTOMATIC_APPLICATION_ENABLED,
    BpmPreviewOutcome,
    REASON_CALIBRATION_ONLY_GATE_CLOSED,
    STATUS_BASE_CORRUPTED,
    STATUS_NO_DATABASE,
    STATUS_NO_PROFILE,
    STATUS_SELECTED_PREVIEW,
    STATUS_UNQUALIFIED_PROFILE,
    orchestrate_bpm_preview,
)
from zecalibrator.bpm.revision import make_revision
from zecalibrator.bpm.settings import BpmSettings
from zecalibrator.bpm.store import create_bad_pixel_database
from zecalibrator.bpm.vocabulary import (
    ACTION_STATE_NO_ACTION_REQUIRED,
    REVISION_STATE_CANDIDATE,
    REVISION_STATE_PROMOTED,
)
from zecalibrator.core.calibrate import CalibrationResult, FrameQuality
from zecalibrator.core.dq import CountSummary
from zecalibrator.storage import StoragePaths

from conftest import make_identity, make_rev, make_site

SHAPE = (12, 12)
CFA = "GRBG"


def _storage(tmp_path) -> StoragePaths:
    return StoragePaths(
        user_config_path=tmp_path / "config",
        user_data_path=tmp_path / "data",
        user_cache_path=tmp_path / "cache",
        user_state_path=tmp_path / "state",
        user_log_path=tmp_path / "log",
    )


def _settings(root=None) -> BpmSettings:
    return BpmSettings(bad_pixel_database_root=str(root) if root is not None else None)


def _calibration(data, mask=None):
    data = np.asarray(data, dtype=np.float32)
    if mask is None:
        mask = np.zeros(data.shape, dtype=np.uint16)
    mask = np.asarray(mask, dtype=np.uint16)
    return CalibrationResult(
        status="COMPLETED",
        data=np.ascontiguousarray(data),
        mask=np.ascontiguousarray(mask),
        counts=CountSummary.from_mask(mask),
        frame_quality=FrameQuality(saturation_evidence="unknown"),
        precision=None,
        scalars={},
    )


def _orchestrate(tmp_path, identity, root, calibration=None):
    return orchestrate_bpm_preview(
        storage=_storage(tmp_path),
        settings=_settings(root),
        identity=identity,
        calibration=calibration,
        frame_id="f0",
    )


# ---------------------------------------------------------------------------
# §2 — the scientific gate is an internal, readable, testable constant.
# ---------------------------------------------------------------------------

def test_scientific_gate_is_a_false_module_constant():
    # §22/§23: a module-level constant, False, not a user setting/env/flag.
    assert BPM_AUTOMATIC_APPLICATION_ENABLED is False


# ---------------------------------------------------------------------------
# §34 — MANDATORY GUARD: 1 ACTION_ELIGIBLE site -> 0 reconstructed, calibration-only.
# ---------------------------------------------------------------------------

def test_promoted_profile_with_one_eligible_site_is_calibration_only(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    site = make_site(5, 5)  # ACTION_ELIGIBLE
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[site], sequence=1))

    data = np.full(SHAPE, 100.0, dtype=np.float32)
    data[5, 5] = 9000.0  # the eligible site's anomalous value (would be replaced if applied)

    outcome = _orchestrate(tmp_path, ident, root, _calibration(data))

    # The profile WAS selected (revision carried), but the gate is closed.
    assert outcome.status == STATUS_SELECTED_PREVIEW
    assert outcome.lookup_outcome == "SELECTED"
    assert outcome.revision_id is not None
    assert outcome.eligible_site_count == 1
    # §34: 0 sites reconstructed, calibration-only, gate closed.
    assert outcome.reconstructed_site_count == 0
    assert outcome.application_enabled is False
    assert outcome.calibration_only is True
    assert outcome.calibration_only_reason == REASON_CALIBRATION_ONLY_GATE_CLOSED


def test_promoted_profile_one_eligible_site_never_calls_apply(tmp_path):
    # §3/§24: even with a compatible promoted profile + eligible sites, the
    # orchestrator never reconstructs a pixel while the gate is closed. The
    # returned calibration is therefore untouched.
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    site = make_site(5, 5)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[site], sequence=1))

    data = np.full(SHAPE, 100.0, dtype=np.float32)
    data[5, 5] = 9000.0
    outcome = _orchestrate(tmp_path, ident, root, _calibration(data))

    assert outcome.reconstructed_site_count == 0
    # The anomaly is untouched (calibration-only, never applied).
    assert _calibration(data).data[5, 5] == 9000.0


# ---------------------------------------------------------------------------
# §33 — when application_enabled is False, prepared_site_count stays 0 even if
#        the profile contains eligible sites.
# ---------------------------------------------------------------------------

def test_prepared_site_count_zero_when_application_disabled(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    sites = [make_site(5, 5), make_site(7, 7)]  # two ACTION_ELIGIBLE sites
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(ident, state=REVISION_STATE_PROMOTED, sites=sites, sequence=1))

    outcome = _orchestrate(tmp_path, ident, root, _calibration(np.full(SHAPE, 100.0)))

    assert outcome.application_enabled is False
    assert outcome.eligible_site_count == 2
    assert outcome.prepared_site_count == 0
    assert outcome.reconstructed_site_count == 0


# ---------------------------------------------------------------------------
# §60 (orchestrator part) — the full typed-status matrix.
# ---------------------------------------------------------------------------

def test_no_database_is_typed_no_database(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    outcome = _orchestrate(tmp_path, ident, tmp_path / "does-not-exist")
    assert outcome.status == STATUS_NO_DATABASE
    assert outcome.reason_code == "NO_BASE"
    assert outcome.calibration_only is True
    assert outcome.reconstructed_site_count == 0
    assert outcome.revision_id is None


def test_empty_database_is_no_profile_not_no_database(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    root = tmp_path / "base"
    create_bad_pixel_database(root)  # valid base, zero revisions
    outcome = _orchestrate(tmp_path, ident, root)
    # A valid but empty base is "no compatible profile", never NO_DATABASE.
    assert outcome.status == STATUS_NO_PROFILE
    assert outcome.reason_code == "NO_PROFILE"


def test_database_with_no_matching_profile_is_no_profile(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(
        make_identity(shape=SHAPE, cfa_phase=CFA, detector_instance_id="OTHER-DET"),
        state=REVISION_STATE_PROMOTED, sites=[make_site(5, 5)], sequence=1,
    ))
    outcome = _orchestrate(tmp_path, ident, root)
    assert outcome.status == STATUS_NO_PROFILE
    assert outcome.reason_code == "NO_PROFILE"


def test_wrong_sensor_is_no_profile(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(
        make_identity(shape=SHAPE, cfa_phase=CFA, detector_instance_id="DET-A"),
        state=REVISION_STATE_PROMOTED, sites=[make_site(5, 5)], sequence=1,
    ))
    outcome = _orchestrate(tmp_path, ident, root)
    assert outcome.status == STATUS_NO_PROFILE


def test_wrong_geometry_is_no_profile(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(
        make_identity(shape=(12, 13), cfa_phase=CFA),
        state=REVISION_STATE_PROMOTED, sites=[make_site(5, 5)], sequence=1,
    ))
    outcome = _orchestrate(tmp_path, ident, root)
    assert outcome.status == STATUS_NO_PROFILE
    assert outcome.reason_code == "NO_PROFILE"


def test_unqualified_profile_is_typed_unqualified_profile(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(
        ident, state=REVISION_STATE_CANDIDATE, sites=[make_site(5, 5)], sequence=1,
    ))
    outcome = _orchestrate(tmp_path, ident, root)
    assert outcome.status == STATUS_UNQUALIFIED_PROFILE
    assert outcome.reason_code == "UNQUALIFIED_PROFILE"
    assert outcome.calibration_only is True


def test_promoted_zero_eligible_is_selected_preview(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(
        ident, state=REVISION_STATE_PROMOTED,
        sites=[make_site(5, 5, action_state=ACTION_STATE_NO_ACTION_REQUIRED)], sequence=1,
    ))
    outcome = _orchestrate(tmp_path, ident, root, _calibration(np.full(SHAPE, 100.0)))
    assert outcome.status == STATUS_SELECTED_PREVIEW
    assert outcome.eligible_site_count == 0
    assert outcome.reconstructed_site_count == 0


def test_corrupt_database_is_typed_status_with_warning(tmp_path):
    import json

    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    rev = make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[make_site(5, 5)], sequence=1)
    db.add_revision(rev)
    rev_path = root / "revisions" / f"{rev.revision_id}.json"
    obj = json.loads(rev_path.read_text(encoding="utf-8"))
    obj["sites"][0]["position"] = [1, 1]
    rev_path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    outcome = _orchestrate(tmp_path, ident, root)
    # §28: corrupt base is a TYPED status + warning, NEVER presented as NO_PROFILE.
    assert outcome.status == STATUS_BASE_CORRUPTED
    assert outcome.reason_code == "BASE_CORRUPTED"
    assert outcome.warnings
    assert outcome.calibration_only is True


def test_outcome_to_dict_is_round_trip_safe(tmp_path):
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[make_site(5, 5)], sequence=1))
    outcome = _orchestrate(tmp_path, ident, root, _calibration(np.full(SHAPE, 100.0)))
    d = outcome.to_dict()
    assert d["status"] == STATUS_SELECTED_PREVIEW
    assert d["application_enabled"] is False
    assert d["reconstructed_site_count"] == 0
    assert d["revision_id"]
    assert isinstance(d["provenance"], list)
