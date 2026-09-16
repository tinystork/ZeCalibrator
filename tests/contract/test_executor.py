"""End-to-end executor orchestration (flat prep, saturation, already_normalized)."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.application.executor import (
    CalibrationRequest,
    MasterBinding,
    NormalizationProof,
    execute_calibration,
)
from zecalibrator.core.errors import InvalidRequestError


def _with_data(make_frame, data, **overrides):
    f = make_frame(data.shape, **overrides)
    f.data[:] = data.astype(np.float32)
    return f


def _proof(**kw):
    base = dict(
        algorithm="cfa-median-per-plane", version="1.0", population="mono-valid",
        scalars={"mono": 95.0}, plane_counts={"mono": 4}, quality_policy_state="qualified",
    )
    base.update(kw)
    return NormalizationProof(**base)


def test_flat_prep_end_to_end_matches_core(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=110.0)
    dark = make_frame((2, 2), value=10.0, exposure_s=10.0)
    finc = _with_data(make_frame, np.array([[100.0, 110.0], [120.0, 90.0]]), exposure_s=1.0)
    fdinc = make_frame((2, 2), value=10.0, exposure_s=1.0)

    masters = {
        "dark": MasterBinding(role="dark", frame=dark, bias_state="included"),
        "flat": MasterBinding(role="flat", frame=finc, flat_form="raw_response"),
        "flat_dark": MasterBinding(role="flat_dark", frame=fdinc, bias_state="included"),
    }
    result = execute_calibration(
        light,
        CalibrationRequest(additive_mode="dark_incl_bias", flat_mode="apply"),
        masters,
        flat_prep_mode="flat_dark_incl_bias",
    )
    Fcorr = np.array([[90.0, 100.0], [110.0, 80.0]])
    R = Fcorr / np.float32(95.0)
    expected = (np.full((2, 2), 100.0, dtype=np.float32)) / R.astype(np.float32)
    assert result.status == "COMPLETED"
    assert np.allclose(result.data, expected, rtol=1e-6)
    assert np.count_nonzero(result.mask) == 0
    assert result.scalars["mono"] == 95.0


def test_flat_quality_zero_valid_plane_refuses(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=1.0)
    dark = make_frame((2, 2), value=0.0)
    finc = make_frame((2, 2), value=0.0, exposure_s=1.0)
    fdinc = make_frame((2, 2), value=0.0, exposure_s=1.0)
    masters = {
        "dark": MasterBinding(role="dark", frame=dark, bias_state="included"),
        "flat": MasterBinding(role="flat", frame=finc, flat_form="raw_response"),
        "flat_dark": MasterBinding(role="flat_dark", frame=fdinc, bias_state="included"),
    }
    result = execute_calibration(
        light,
        CalibrationRequest(additive_mode="dark_incl_bias", flat_mode="apply"),
        masters,
        flat_prep_mode="flat_dark_incl_bias",
    )
    assert result.status == "FAILED"
    assert result.reason_code == "FLAT_UNUSABLE"


def test_saturation_evaluated_on_original_flat(make_frame_fixture):
    # M4: saturated Finc=100 minus pedestal=10 must remain excluded at limit=100.
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=110.0)
    dark = make_frame((2, 2), value=10.0, exposure_s=10.0)
    finc = make_frame((2, 2), value=100.0, exposure_s=1.0, saturation_limit_adu=100.0)
    fdinc = make_frame((2, 2), value=10.0, exposure_s=1.0)
    masters = {
        "dark": MasterBinding(role="dark", frame=dark, bias_state="included"),
        "flat": MasterBinding(role="flat", frame=finc, flat_form="raw_response"),
        "flat_dark": MasterBinding(role="flat_dark", frame=fdinc, bias_state="included"),
    }
    result = execute_calibration(
        light,
        CalibrationRequest(additive_mode="dark_incl_bias", flat_mode="apply"),
        masters,
        flat_prep_mode="flat_dark_incl_bias",
    )
    assert result.status == "FAILED"
    assert result.reason_code == "FLAT_UNUSABLE"


def test_already_normalized_branch(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=100.0)
    dark = make_frame((2, 2), value=0.0)
    R = _with_data(make_frame, np.full((2, 2), 2.0), exposure_s=1.0, units="dimensionless")
    masters = {
        "dark": MasterBinding(role="dark", frame=dark, bias_state="included"),
        "flat": MasterBinding(
            role="flat", frame=R, flat_form="normalized_response",
            normalization_proof=_proof(scalars={"mono": 1.0}),
        ),
    }
    result = execute_calibration(
        light,
        CalibrationRequest(additive_mode="dark_incl_bias", flat_mode="apply"),
        masters,
        flat_prep_mode="already_normalized",
    )
    assert result.status == "COMPLETED"
    assert np.allclose(result.data, np.full((2, 2), 50.0))  # 100 / 2


def test_already_normalized_requires_evidence(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=100.0)
    R = make_frame((2, 2), value=2.0, exposure_s=1.0, units="dimensionless")
    masters = {"flat": MasterBinding(role="flat", frame=R, flat_form="normalized_response")}
    with pytest.raises(InvalidRequestError):
        execute_calibration(
            light, CalibrationRequest(additive_mode="control", flat_mode="apply"),
            masters, flat_prep_mode="already_normalized",
        )


def test_normalized_flat_rejects_additive_subtraction(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=100.0)
    flat = make_frame((2, 2), value=100.0, exposure_s=1.0, units="dimensionless")
    fdinc = make_frame((2, 2), value=10.0, exposure_s=1.0)
    masters = {
        "flat": MasterBinding(role="flat", frame=flat, flat_form="normalized_response", normalization_proof=_proof()),
        "flat_dark": MasterBinding(role="flat_dark", frame=fdinc, bias_state="included"),
    }
    with pytest.raises(InvalidRequestError):
        execute_calibration(
            light, CalibrationRequest(additive_mode="control", flat_mode="apply"),
            masters, flat_prep_mode="flat_dark_incl_bias",
        )


def test_bias_only_flat_requires_profile(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=100.0)
    finc = make_frame((2, 2), value=100.0, exposure_s=1.0)
    bias_flat = make_frame((2, 2), value=4.0, exposure_s=0.01)
    masters = {
        "flat": MasterBinding(role="flat", frame=finc, flat_form="raw_response"),
        "bias_flat": MasterBinding(role="bias", frame=bias_flat),
    }
    with pytest.raises(InvalidRequestError):
        execute_calibration(
            light, CalibrationRequest(additive_mode="control", flat_mode="apply"),
            masters, flat_prep_mode="bias_only_flat",
        )
