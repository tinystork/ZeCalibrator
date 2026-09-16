"""Per-acquisition saturation facts (D; SCIENCE §9.3)."""

from __future__ import annotations

import numpy as np

from zecalibrator.application.executor import CalibrationRequest, MasterBinding, execute_calibration
from zecalibrator.core.calibrate import calibrate_light


def test_light_saturation_honored_without_override(make_frame_fixture):
    # D: a light with qualified limit 60000, value 65000 must be masked (all-invalid
    # FAILED) without any redundant call override.
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=65000.0)  # qualified saturation_limit_adu=60000
    r = execute_calibration(light, CalibrationRequest("control"), {})
    assert r.status == "FAILED"
    assert r.frame_quality.saturation_evidence == "qualified"


def test_light_saturation_partial_warning(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    light.data[0, 0] = 65000.0  # one saturated pixel (limit 60000)
    r = execute_calibration(light, CalibrationRequest("control"), {})
    assert r.status == "COMPLETED_WITH_WARNINGS"
    assert r.counts.per_bit["SATURATED"] == 1
    assert np.isnan(r.data[0, 0])


def test_unknown_saturation_is_diagnostic_not_pixel_bit(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, saturation_evidence="unknown", saturation_limit_adu=None)
    r = execute_calibration(light, CalibrationRequest("control"), {})
    assert r.frame_quality.saturation_evidence == "unknown"
    assert r.counts.per_bit["SATURATED"] == 0
    assert r.status == "COMPLETED_WITH_WARNINGS"  # unknown quality evidence


def test_qualified_saturation_requires_positive_limit(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, saturation_evidence="qualified", saturation_limit_adu=-1.0)
    from zecalibrator.core.errors import InvalidRequestError

    with np.testing.assert_raises(InvalidRequestError):
        execute_calibration(light, CalibrationRequest("control"), {})
