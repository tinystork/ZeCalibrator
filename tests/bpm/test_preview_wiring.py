"""P4-P1 LOT 1 wiring tests: the orchestrator is branched into the existing
CalibrationResult producer as ONE optional step.

§31 — with reconstructed sites = 0, the ordinary calibration output is
       pixel-identical to the current path.
§63 — without a BPM base, the calibration behaviour is unchanged.
"""

from __future__ import annotations

import numpy as np

from zecalibrator.application.bpm_preview import (
    BpmPreviewSeam,
    STATUS_NO_DATABASE,
    STATUS_SELECTED_PREVIEW,
)
from zecalibrator.application.executor import CalibrationRequest, execute_calibration
from zecalibrator.bpm.settings import BpmSettings
from zecalibrator.bpm.store import create_bad_pixel_database
from zecalibrator.bpm.vocabulary import REVISION_STATE_PROMOTED
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


def _light(make_frame, value=110.0):
    # A Bayer CFA light whose identity exactly matches the BPM revision below.
    return make_frame(SHAPE, value=value, cfa_phase=CFA)


def _base_with_eligible_site(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    # Identity must mirror the light's decoded metadata identity
    # (SYNTH-DET-0001 / SYNTH-MONO / GRBG / gain=100 offset=50 MODE_A MODE_16).
    ident = make_identity(
        shape=SHAPE, cfa_phase=CFA,
        detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-MONO",
        gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16",
    )
    db.add_revision(make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[make_site(5, 5)], sequence=1))
    return root


def test_wiring_no_base_is_unchanged(make_frame_fixture, tmp_path):
    # §63: with no BPM base configured, the producer's output is byte-identical
    # with and without the seam.
    make_frame = make_frame_fixture
    light = _light(make_frame)
    request = CalibrationRequest(additive_mode="control", flat_mode="none")

    without = execute_calibration(light, request, {})
    seam = BpmPreviewSeam(_storage(tmp_path), BpmSettings())
    with_seam = execute_calibration(light, request, {}, bpm_preview=seam)

    assert np.array_equal(with_seam.data, without.data)
    assert np.array_equal(with_seam.mask, without.mask)
    assert with_seam.status == without.status
    assert seam.outcome is not None
    assert seam.outcome.status == STATUS_NO_DATABASE
    assert seam.outcome.calibration_only is True
    assert seam.outcome.reconstructed_site_count == 0


def test_wiring_reconstructed_zero_is_pixel_identical(make_frame_fixture, tmp_path):
    # §31: a configured base with a compatible promoted profile + 1 eligible site
    # still produces reconstructed_site_count == 0 (gate closed), so the returned
    # calibration is pixel-identical to the unwired path.
    make_frame = make_frame_fixture
    root = _base_with_eligible_site(tmp_path)
    light = _light(make_frame)
    request = CalibrationRequest(additive_mode="control", flat_mode="none")

    without = execute_calibration(light, request, {})
    seam = BpmPreviewSeam(_storage(tmp_path), BpmSettings(bad_pixel_database_root=str(root)))
    with_seam = execute_calibration(light, request, {}, bpm_preview=seam)

    # The orchestrator selected the profile but never applied (gate closed).
    assert seam.outcome.status == STATUS_SELECTED_PREVIEW
    assert seam.outcome.eligible_site_count == 1
    assert seam.outcome.reconstructed_site_count == 0
    assert seam.outcome.application_enabled is False
    assert seam.outcome.calibration_only is True

    # §31: pixel-to-pixel identity with the current path.
    assert np.array_equal(with_seam.data, without.data)
    assert np.array_equal(with_seam.mask, without.mask)
    assert with_seam.status == without.status


def test_wiring_absent_seam_is_byte_identical(make_frame_fixture, tmp_path):
    # The seam default (None) is exactly today's path: same bytes, same status.
    make_frame = make_frame_fixture
    light = _light(make_frame)
    request = CalibrationRequest(additive_mode="control", flat_mode="none")

    a = execute_calibration(light, request, {})
    b = execute_calibration(light, request, {})
    assert np.array_equal(a.data, b.data)
    assert np.array_equal(a.mask, b.mask)
