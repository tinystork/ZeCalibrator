"""No-mutation and deep-immutability guarantees (SCIENCE §3.4, §10; M7, F)."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.application.cancellation import CancellationToken
from zecalibrator.application.executor import CalibrationRequest, MasterBinding, execute_calibration
from zecalibrator.core.calibrate import calibrate_light
from zecalibrator.core.dq import INPUT_INVALID
from zecalibrator.core.errors import GeometryMismatchError
from zecalibrator.core.metadata import SensorMetadata, Geometry


def _raw_md(shape=(2, 2)):
    geo = Geometry(shape, shape, (1, 1), (0, 0), None, "identity", "mono")
    return SensorMetadata((), {}, (), geo, "raw", units="ADU")


def test_calibrate_light_does_not_mutate_inputs():
    L = np.array([[110.0, 50.0], [210.0, -2.0]], dtype=np.float32)
    L_copy = L.copy()
    dark = np.full((2, 2), 10.0, dtype=np.float32)
    dark_copy = dark.copy()
    R = np.array([[1.0, 0.5], [2.0, 1.0]], dtype=np.float32)
    R_copy = R.copy()
    in_mask = np.zeros((2, 2), dtype=np.uint16)
    in_mask_copy = in_mask.copy()

    calibrate_light(
        L, additive_mode="dark_incl_bias", flat_mode="apply",
        dark_inc=dark, flat_response=R, flat_valid=np.ones((2, 2), bool),
        input_mask=in_mask, saturation_limit=60000.0,
    )
    assert np.array_equal(L, L_copy)
    assert np.array_equal(dark, dark_copy)
    assert np.array_equal(R, R_copy)
    assert np.array_equal(in_mask, in_mask_copy)


def test_calibrate_light_failure_does_not_mutate_inputs():
    L = np.full((2, 2), np.nan, dtype=np.float32)
    L_copy = L.copy()
    mask = np.full((2, 2), INPUT_INVALID, dtype=np.uint16)
    mask_copy = mask.copy()
    calibrate_light(L, additive_mode="control", input_mask=mask, saturation_limit=100.0)
    assert np.array_equal(L, L_copy, equal_nan=True)
    assert np.array_equal(mask, mask_copy)


def test_execute_calibration_does_not_mutate_masters_or_light(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=110.0)
    dark = make_frame((2, 2), value=10.0, exposure_s=10.0)
    light_data = light.data.copy()
    dark_data = dark.data.copy()

    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    execute_calibration(light, CalibrationRequest(additive_mode="dark_incl_bias"), masters)
    assert np.array_equal(light.data, light_data)
    assert np.array_equal(dark.data, dark_data)


def test_execute_calibration_cancelled_no_mutation(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=110.0)
    dark = make_frame((2, 2), value=10.0)
    light_data = light.data.copy()
    dark_data = dark.data.copy()

    token = CancellationToken()
    token.cancel()
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    result = execute_calibration(
        light, CalibrationRequest(additive_mode="dark_incl_bias"), masters, cancel=token
    )
    assert result.status == "CANCELLED"
    assert result.data is None and result.mask is None
    assert np.array_equal(light.data, light_data)
    assert np.array_equal(dark.data, dark_data)


def test_geometry_mismatch_rejects_without_crop(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=0.0)
    dark = make_frame((4, 4), value=0.0, roi_origin=(0, 1))
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError):
        execute_calibration(light, CalibrationRequest(additive_mode="dark_incl_bias"), masters)


def test_shape_mismatch_rejects(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2))
    dark = make_frame((3, 3))
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError):
        execute_calibration(light, CalibrationRequest(additive_mode="dark_incl_bias"), masters)


def test_sensor_metadata_deep_immutable():
    md = _raw_md()
    with pytest.raises(TypeError):
        md.normalized["mutated"] = True
    assert "mutated" not in md.normalized
    assert not hasattr(md.normalized, "clear")


def test_sensor_metadata_nested_list_is_frozen():
    md = SensorMetadata(
        original_cards=(), normalized={"cfa": ["GRBG", "mono"]},
        conflicts=(), geometry=Geometry((2, 2), (2, 2), (1, 1), (0, 0), None, "identity", "mono"),
        raw_domain_declaration="raw",
    )
    inner = md.normalized["cfa"]
    assert isinstance(inner, tuple)
    assert inner == ("GRBG", "mono")


def test_nested_numpy_evidence_is_frozen():
    # F: numpy evidence must be recursively frozen (tuple), never a writable copy.
    md = SensorMetadata(
        original_cards=(), normalized={"x": np.array([1, 2])},
        conflicts=(), geometry=Geometry((2, 2), (2, 2), (1, 1), (0, 0), None, "identity", "mono"),
        raw_domain_declaration="raw",
    )
    inner = md.normalized["x"]
    assert isinstance(inner, tuple)
    assert inner == (1, 2)


def test_calibration_result_scalars_immutable(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=110.0)
    dark = make_frame((2, 2), value=10.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    result = execute_calibration(light, CalibrationRequest(additive_mode="dark_incl_bias"), masters)
    with pytest.raises(TypeError):
        result.scalars["mono"] = 1.0


def test_unknown_roi_does_not_match_unknown(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), roi_origin=None)
    dark = make_frame((2, 2), roi_origin=None)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest(additive_mode="dark_incl_bias"), masters)
    assert "MISSING_REQUIRED_FIELD" in exc.value.reason_code or "ROI" in exc.value.reason_code
