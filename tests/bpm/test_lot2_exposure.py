"""LOT 2 exposure tests (headless): the ``zecalibrator.api.v1._bpm`` facade.

Covers the §59 config matrix (default/custom/persist/reload/create-missing/
invalid/read-only/return-to-default), the additive seam passthrough (§63 legacy:
``seam=None`` is byte-identical to the public path), the §50 log synthesis, the
§40 status line, and the §74 "never says applied while reconstructed == 0" rule.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import zecalibrator.api.v1 as v1
from zecalibrator.api.v1 import _bpm
from zecalibrator.bpm.settings import STATE_MISSING, STATE_OK
from zecalibrator.bpm.store import create_bad_pixel_database
from zecalibrator.bpm.vocabulary import REVISION_STATE_PROMOTED
from zecalibrator.storage import StoragePaths

from conftest import make_identity, make_rev, make_site

SHAPE = (4, 4)
CFA = "mono"


def _storage(tmp_path) -> StoragePaths:
    return StoragePaths(
        user_config_path=tmp_path / "config",
        user_data_path=tmp_path / "data",
        user_cache_path=tmp_path / "cache",
        user_state_path=tmp_path / "state",
        user_log_path=tmp_path / "log",
    )


def _paths_dict(storage) -> dict:
    return {k: str(v) for k, v in storage.as_dict().items()}


# ---------------------------------------------------------------------------
# §59 config: default / custom / persist+reload / create-missing / invalid /
# read-only / return-to-default.
# ---------------------------------------------------------------------------
def test_default_root_is_portable_and_under_data(tmp_path):
    storage = _storage(tmp_path)
    assert _bpm.default_bad_pixel_database_root(storage) == storage.user_data_path / "bad_pixel_database"


def test_no_settings_resolves_to_default(tmp_path):
    storage = _storage(tmp_path)
    assert _bpm.resolve_bpm_root(storage, _bpm.default_bpm_settings()) == \
        _bpm.default_bad_pixel_database_root(storage)


def test_custom_root_wins(tmp_path):
    storage = _storage(tmp_path)
    settings = _bpm.bpm_settings(str(tmp_path / "my-base"))
    assert _bpm.resolve_bpm_root(storage, settings) == tmp_path / "my-base"


def test_persist_and_reload_roundtrip(tmp_path):
    storage = _storage(tmp_path)
    assert _bpm.load_bpm_settings(storage.user_config_path).state == STATE_MISSING
    _bpm.save_bpm_settings(storage.user_config_path, _bpm.bpm_settings(str(tmp_path / "my-base")))
    loaded = _bpm.load_bpm_settings(storage.user_config_path)
    assert loaded.state == STATE_OK
    assert loaded.settings.bad_pixel_database_root == str(tmp_path / "my-base")


def test_create_missing_folder(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    path, error = _bpm.ensure_bpm_root(target, create=True)
    assert error is None
    assert Path(path).is_dir()


def test_validate_file_path_is_invalid(tmp_path):
    f = tmp_path / "a-file"
    f.write_text("x")
    ok, reason = _bpm.validate_bpm_root(f)
    assert ok is False
    assert "not a folder" in reason
    path, error = _bpm.ensure_bpm_root(f, create=True)
    assert error is not None


def test_read_only_folder_is_reported(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX permission model only")
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        ok, reason = _bpm.validate_bpm_root(ro)
        assert ok is False
        assert "not writable" in reason
    finally:
        os.chmod(ro, 0o700)


def test_return_to_default_when_root_cleared(tmp_path):
    storage = _storage(tmp_path)
    settings = _bpm.bpm_settings(None)  # cleared -> default
    assert _bpm.resolve_bpm_root(storage, settings) == _bpm.default_bad_pixel_database_root(storage)
    settings = _bpm.bpm_settings("")  # empty -> default
    assert _bpm.resolve_bpm_root(storage, settings) == _bpm.default_bad_pixel_database_root(storage)


# ---------------------------------------------------------------------------
# §63 legacy: the additive seam defaults to the public path byte-identically.
# ---------------------------------------------------------------------------
def _control_plan():
    decl = v1.ImportDeclaration(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-MONO", gain=100.0, offset=50.0,
        readout_mode="MODE_A", adc_mode="MODE_16", binning=(1, 1),
        sensor_dimensions=SHAPE, orientation="identity", cfa_phase="mono",
        roi_origin=(0, 0), exposure_s=10.0, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0, saturation_evidence="qualified",
    )
    md = v1.build_sensor_metadata(shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU")
    inspection = v1.FrameInspection(
        metadata=md, identity=v1.ArrayInputIdentity(caller_logical_id="light"),
        domain_finding="raw", hdu=None, shape=SHAPE,
    )
    return v1.resolve_calibration(
        inspection, v1.CalibrationRequest("control"), v1.LibraryHandle(
            v1.LibrarySnapshot(revision="r1", schema_version="zecalibrator.library.v1", candidates={})
        ), v1.default_match_policy()
    ).plan


def _array_source():
    decl = v1.ImportDeclaration(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-MONO", gain=100.0, offset=50.0,
        readout_mode="MODE_A", adc_mode="MODE_16", binning=(1, 1),
        sensor_dimensions=SHAPE, orientation="identity", cfa_phase="mono",
        roi_origin=(0, 0), exposure_s=10.0, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0, saturation_evidence="qualified",
    )
    md = v1.build_sensor_metadata(shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU")
    return v1.ArrayFrameSource(
        data=np.full(SHAPE, 100.0, dtype=np.float32),
        metadata=md,
        identity=v1.ArrayInputIdentity(caller_logical_id="light-1"),
    )


def test_facade_calibrate_frame_seam_none_is_identical_to_public(tmp_path):
    plan = _control_plan()
    source = _array_source()
    public = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    facaded = _bpm.calibrate_frame(source, plan, v1.ExecutionOptions(), seam=None)
    assert facaded.status == public.status
    assert np.array_equal(facaded.data, public.data)
    assert np.array_equal(facaded.mask, public.mask)


def test_facade_calibrate_frame_with_seam_records_outcome_gate_closed(tmp_path):
    # Configure a base with a compatible promoted profile + 1 eligible site.
    storage = _storage(tmp_path)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    ident = make_identity(
        shape=SHAPE, cfa_phase=CFA,
        detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-MONO",
        gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16",
    )
    db.add_revision(make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[make_site(1, 1)], sequence=1))

    settings = _bpm.bpm_settings(str(root))
    seam = _bpm.make_bpm_seam(storage, settings, frame_id="f0")

    plan = _control_plan()
    source = _array_source()
    result = _bpm.calibrate_frame(source, plan, v1.ExecutionOptions(), seam=seam)

    assert result.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert seam.outcome is not None
    # §34: gate closed — selected profile, 1 eligible site, 0 reconstructed.
    assert seam.outcome.eligible_site_count == 1
    assert seam.outcome.reconstructed_site_count == 0
    assert seam.outcome.application_enabled is False
    assert seam.outcome.calibration_only is True
    # §74: the synthesis/status must never say "applied" while reconstructed == 0.
    text = "\n".join(_bpm.synthesis_text(seam.outcome, str(root)))
    assert "applied" not in text
    assert "Reconstructed sites: 0" in text
    assert "Result mode: calibration-only" in text
    assert "BPM scientific application: disabled" in text
    status = _bpm.bpm_status_line(seam.outcome, str(root))
    assert "not required" in status
    assert "preview disabled" not in status


def test_synthesis_without_outcome_is_calibration_only_defaults(tmp_path):
    lines = _bpm.synthesis_text(None, "/x/base")
    assert lines == [
        "Bad Pixel Database: /x/base",
        "Sensor profile: none",
        "Eligible sites: 0",
        "BPM scientific application: disabled",
        "Reconstructed sites: 0",
        "Result mode: calibration-only",
    ]
    assert "applied" not in "\n".join(lines)


def test_status_line_never_applied_when_zero(tmp_path):
    # A fabricated outcome with reconstructed == 0 must never render "applied".
    class _Outcome:
        database_root = "/x/base"
        revision_id = "rev-1"
        eligible_site_count = 2
        application_enabled = False
        reconstructed_site_count = 0
        calibration_only = True

    line = _bpm.bpm_status_line(_Outcome(), "/x/base")
    assert "not required" in line
    assert "preview disabled" not in line
    assert "applied" not in line
    assert "Profile: selected" in line


def test_storage_from_mapping_roundtrip(tmp_path):
    storage = _storage(tmp_path)
    rebuilt = _bpm.storage_from_mapping(_paths_dict(storage))
    assert rebuilt == storage
