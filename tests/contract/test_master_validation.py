"""Explicit supplied-master validation (not matching) — M2, B, L1, broadcasting."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.application.executor import CalibrationRequest, MasterBinding, execute_calibration
from zecalibrator.core.errors import GeometryMismatchError, InvalidRequestError


def test_incompatible_detector_exposure_temp_gain_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, detector_model="A", exposure_s=10.0,
                        temperature_c=0.0, gain=10.0)
    dark = make_frame((4, 4), value=10.0, detector_model="B", exposure_s=100.0,
                      temperature_c=50.0, gain=999.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError):
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)


def test_dark_bias_state_must_match_mode(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    dark = make_frame((4, 4), value=10.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="removed")}
    with pytest.raises(GeometryMismatchError):
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)


def test_role_mismatch_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    bias = make_frame((4, 4), value=10.0)
    masters = {"dark": MasterBinding(role="bias", frame=bias)}
    with pytest.raises(GeometryMismatchError):
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)


def test_thermal_zero_tolerance(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, temperature_c=0.0)
    dark = make_frame((4, 4), value=10.0, temperature_c=0.001)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert "TEMPERATURE_MISMATCH" in exc.value.reason_code


def test_thermal_parser_tolerance_accepted(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, temperature_c=20.0)
    dark = make_frame((4, 4), value=10.0, temperature_c=20.0 + 5e-7)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    r = execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert r.status == "COMPLETED"


def test_nan_temperature_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, temperature_c=float("nan"))
    dark = make_frame((4, 4), value=10.0, temperature_c=20.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert "MISSING_REQUIRED_FIELD" in exc.value.reason_code


def test_bias_does_not_require_temperature(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, temperature_c=None)
    bias = make_frame((4, 4), value=4.0, exposure_s=0.01, temperature_c=None)
    masters = {"bias": MasterBinding(role="bias", frame=bias)}
    r = execute_calibration(light, CalibrationRequest("bias_only"), masters)
    assert r.status == "COMPLETED"


def test_flat_dark_temperature_mismatch_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, temperature_c=20.0)
    finc = make_frame((4, 4), value=100.0, exposure_s=1.0, temperature_c=20.0)
    fdinc = make_frame((4, 4), value=10.0, exposure_s=1.0, temperature_c=20.0 + 0.001)
    masters = {
        "flat": MasterBinding(role="flat", frame=finc, flat_form="raw_response"),
        "flat_dark": MasterBinding(role="flat_dark", frame=fdinc, bias_state="included"),
    }
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("control", "apply"), masters,
                            flat_prep_mode="flat_dark_incl_bias")
    assert "TEMPERATURE_MISMATCH" in exc.value.reason_code


def test_missing_light_exposure_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, exposure_s=None)
    dark = make_frame((4, 4), value=10.0, exposure_s=123.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert "MISSING_REQUIRED_FIELD" in exc.value.reason_code


def test_qualified_short_bias_accepted(make_frame_fixture):
    # B: light 10s + bias 0.01s with qualified max 0.1s must pass.
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, exposure_s=10.0)
    bias = make_frame((4, 4), value=4.0, exposure_s=0.01)
    masters = {"bias": MasterBinding(role="bias", frame=bias)}
    r = execute_calibration(light, CalibrationRequest("bias_only"), masters)
    assert r.status == "COMPLETED"


def test_long_bias_rejected(make_frame_fixture):
    # B: bias 10s exceeds qualified max 0.1s -> reject.
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, exposure_s=10.0)
    bias = make_frame((4, 4), value=4.0, exposure_s=10.0)
    masters = {"bias": MasterBinding(role="bias", frame=bias)}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("bias_only"), masters)
    assert "EXPOSURE_MISMATCH" in exc.value.reason_code


def test_bias_without_qualified_range_rejected(make_frame_fixture):
    # Missing qualification cannot silently pass.
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, exposure_s=10.0, bias_exposure_max_s=None)
    bias = make_frame((4, 4), value=4.0, exposure_s=0.01)
    masters = {"bias": MasterBinding(role="bias", frame=bias)}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("bias_only"), masters)
    assert "MISSING_REQUIRED_FIELD" in exc.value.reason_code


def test_flat_filter_exact(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, filter="IRCUT")
    dark = make_frame((4, 4), value=10.0)
    finc = make_frame((4, 4), value=100.0, filter="LP", exposure_s=1.0)
    fdinc = make_frame((4, 4), value=10.0, exposure_s=1.0)
    masters = {
        "dark": MasterBinding(role="dark", frame=dark, bias_state="included"),
        "flat": MasterBinding(role="flat", frame=finc, flat_form="raw_response"),
        "flat_dark": MasterBinding(role="flat_dark", frame=fdinc, bias_state="included"),
    }
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias", "apply"), masters,
                            flat_prep_mode="flat_dark_incl_bias")
    assert "FILTER_MISMATCH" in exc.value.reason_code


def test_dark_is_filter_independent(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, filter="IRCUT")
    dark = make_frame((4, 4), value=10.0, filter="NONE")
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    r = execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert r.status == "COMPLETED"


def test_unknown_required_fact_does_not_pass(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, gain=None)
    dark = make_frame((4, 4), value=10.0, gain=None)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert "MISSING_REQUIRED_FIELD" in exc.value.reason_code


def test_wrong_flatdark_bias_state_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    finc = make_frame((4, 4), value=100.0, exposure_s=1.0)
    fdinc = make_frame((4, 4), value=10.0, exposure_s=1.0)
    masters = {
        "flat": MasterBinding(role="flat", frame=finc, flat_form="raw_response"),
        "flat_dark": MasterBinding(role="flat_dark", frame=fdinc, bias_state="removed"),
    }
    with pytest.raises(GeometryMismatchError):
        execute_calibration(light, CalibrationRequest("control", "apply"), masters,
                            flat_prep_mode="flat_dark_incl_bias")


def test_broadcast_shape_mismatch_rejected_at_primitive(make_frame_fixture):
    make_frame = make_frame_fixture
    from zecalibrator.io.raw_decoder import DecodedFrame

    light = make_frame((2, 2), value=100.0)
    dark = make_frame((2, 2), value=10.0)
    bad = DecodedFrame(
        data=np.zeros((3, 3), dtype=np.float32), mask=np.zeros((3, 3), dtype=np.uint16),
        metadata=dark.metadata, stored_dtype="float32", bscale=1.0, bzero=0.0,
        blank=None, hdu=0, precision=None,
    )
    masters = {"dark": MasterBinding(role="dark", frame=bad, bias_state="included")}
    with pytest.raises(InvalidRequestError):
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
