"""PreparedCalibrationResult mask semantics (mission §33–§34) + §80 identity."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.bpm.prepared_result import PreparedCalibrationResult
from zecalibrator.core.calibrate import CalibrationResult, FrameQuality
from zecalibrator.core.dq import CountSummary


def make_calibration(data, mask=None):
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


def test_prepared_result_exposes_the_three_contract_masks():
    data = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[1, 1] = 0x0001
    rec = np.zeros((4, 4), dtype=np.uint8)
    rec[2, 2] = 1
    usable = (mask == 0) & ~(rec.astype(bool))

    result = PreparedCalibrationResult(
        calibration=make_calibration(data, mask),
        prepared_data=data,
        measurement_dq=mask,
        reconstructed_mask=rec,
        usable_mask=usable,
        operator_id="SAME_CFA_MEDIAN",
        operator_version="zecalibrator.bpm.reconstruction.v1",
        prepared_site_count=1,
    )
    assert np.array_equal(result.measurement_dq, mask)
    assert np.array_equal(result.reconstructed_mask, rec)
    assert np.array_equal(result.usable_mask, usable)
    assert result.prepared_site_count == 1
    assert result.nothing_prepared is False


def test_prepared_result_references_calibration_and_dq_untouched():
    data = np.full((4, 4), 7.0, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint16)
    cal = make_calibration(data, mask)
    result = PreparedCalibrationResult(
        calibration=cal,
        prepared_data=data,
        measurement_dq=mask,
        reconstructed_mask=np.zeros((4, 4), dtype=np.uint8),
        usable_mask=(mask == 0),
        operator_id="SAME_CFA_MEDIAN",
        operator_version="zecalibrator.bpm.reconstruction.v1",
        prepared_site_count=0,
    )
    assert result.calibration is cal  # references (does not copy) the v1 result
    assert result.nothing_prepared is True


def test_prepared_result_rejects_shape_mismatch():
    data = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        PreparedCalibrationResult(
            calibration=make_calibration(data),
            prepared_data=data,
            measurement_dq=np.zeros((3, 3), dtype=np.uint16),
            reconstructed_mask=np.zeros((4, 4), dtype=np.uint8),
            usable_mask=np.zeros((4, 4), dtype=bool),
            operator_id="SAME_CFA_MEDIAN",
            operator_version="zecalibrator.bpm.reconstruction.v1",
        )


def test_prepared_result_freezes_arrays_to_expected_dtypes():
    data = np.zeros((4, 4), dtype=np.float64)
    mask = np.zeros((4, 4), dtype=np.uint16)
    result = PreparedCalibrationResult(
        calibration=make_calibration(data.astype(np.float32), mask),
        prepared_data=data,
        measurement_dq=mask,
        reconstructed_mask=np.zeros((4, 4), dtype=np.uint8),
        usable_mask=np.zeros((4, 4), dtype=bool),
        operator_id="SAME_CFA_MEDIAN",
        operator_version="zecalibrator.bpm.reconstruction.v1",
    )
    assert result.prepared_data.dtype == np.float32
    assert result.measurement_dq.dtype == np.uint16
    assert result.reconstructed_mask.dtype == np.uint8
    assert result.usable_mask.dtype == bool
