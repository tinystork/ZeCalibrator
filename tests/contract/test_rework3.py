"""Rework-3 regression tests: R1 (flat bias range), M1 (exposure domain),
M2 (declaration tuple domain), S5 (normalization proof coherence)."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.application.executor import (
    CalibrationRequest,
    MasterBinding,
    NormalizationProof,
    execute_calibration,
)
from zecalibrator.core.errors import GeometryMismatchError, InvalidRequestError
from zecalibrator.core.metadata import ImportDeclaration


def _proof(population="mono-valid", **kw):
    base = dict(
        algorithm="median", version="1.0", population=population,
        scalars={"mono": 95.0}, plane_counts={"mono": 4}, quality_policy_state="qualified",
    )
    base.update(kw)
    return NormalizationProof(**base)


# --- R1: flat bias dependency must enforce the qualified bias range ----------

def _flat_bias_setup(make_frame, bias_exposure):
    light = make_frame((4, 4), value=100.0)
    finc = make_frame((4, 4), value=100.0, exposure_s=1.0)
    fdinc = make_frame((4, 4), value=6.0, exposure_s=1.0)
    bias_flat = make_frame((4, 4), value=4.0, exposure_s=bias_exposure)
    return light, finc, fdinc, bias_flat


def test_flat_bias_in_range_accepted(make_frame_fixture):
    make_frame = make_frame_fixture
    light, finc, fdinc, bias_flat = _flat_bias_setup(make_frame, 0.01)
    masters = {
        "flat": MasterBinding("flat", finc, flat_form="raw_response"),
        "flat_dark": MasterBinding("flat_dark", fdinc, bias_state="removed"),
        "bias_flat": MasterBinding("bias", bias_flat),
    }
    r = execute_calibration(
        light, CalibrationRequest("control", "apply"), masters,
        flat_prep_mode="flat_dark_bias_removed",
    )
    assert r.status == "COMPLETED"


def test_flat_bias_over_range_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light, finc, fdinc, bias_flat = _flat_bias_setup(make_frame, 10.0)
    masters = {
        "flat": MasterBinding("flat", finc, flat_form="raw_response"),
        "flat_dark": MasterBinding("flat_dark", fdinc, bias_state="removed"),
        "bias_flat": MasterBinding("bias", bias_flat),
    }
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(
            light, CalibrationRequest("control", "apply"), masters,
            flat_prep_mode="flat_dark_bias_removed",
        )
    assert "EXPOSURE_MISMATCH" in exc.value.reason_code


def test_bias_only_flat_bias_range(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    finc = make_frame((4, 4), value=100.0, exposure_s=1.0)
    bias_flat = make_frame((4, 4), value=4.0, exposure_s=10.0)
    masters = {
        "flat": MasterBinding("flat", finc, flat_form="raw_response", short_flat_profile=True),
        "bias_flat": MasterBinding("bias", bias_flat),
    }
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(
            light, CalibrationRequest("control", "apply"), masters,
            flat_prep_mode="bias_only_flat",
        )
    assert "EXPOSURE_MISMATCH" in exc.value.reason_code


def test_flat_bias_missing_range_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    # Flat profile with NO qualified bias range.
    finc = make_frame((4, 4), value=100.0, exposure_s=1.0, bias_exposure_max_s=None)
    fdinc = make_frame((4, 4), value=6.0, exposure_s=1.0)
    bias_flat = make_frame((4, 4), value=4.0, exposure_s=0.01)
    masters = {
        "flat": MasterBinding("flat", finc, flat_form="raw_response"),
        "flat_dark": MasterBinding("flat_dark", fdinc, bias_state="removed"),
        "bias_flat": MasterBinding("bias", bias_flat),
    }
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(
            light, CalibrationRequest("control", "apply"), masters,
            flat_prep_mode="flat_dark_bias_removed",
        )
    assert "MISSING_REQUIRED_FIELD" in exc.value.reason_code


# --- M1: exposure domain must be non-negative finite -------------------------

def test_negative_bias_exposure_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, exposure_s=10.0)
    bias = make_frame((4, 4), value=4.0, exposure_s=-1.0)
    masters = {"bias": MasterBinding("bias", bias)}
    with pytest.raises(GeometryMismatchError):
        execute_calibration(light, CalibrationRequest("bias_only"), masters)


def test_negative_light_exposure_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, exposure_s=-1.0)
    with pytest.raises(InvalidRequestError):
        execute_calibration(light, CalibrationRequest("control"), {})


def test_negative_dark_exposure_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, exposure_s=10.0)
    dark = make_frame((4, 4), value=10.0, exposure_s=-1.0)
    masters = {"dark": MasterBinding("dark", dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError):
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)


# --- M2: declaration numeric/tuple domain -----------------------------------

def test_degenerate_declaration_tuples_rejected():
    for kwargs in (
        {"sensor_dimensions": (0, 0)},
        {"sensor_dimensions": (100, -1)},
        {"binning": (0, 1)},
        {"binning": (1.5, 1)},
        {"roi_origin": (-1, 0)},
        {"orientation": "flip"},
        {"cfa_phase": "XYZW"},
    ):
        with pytest.raises(ValueError):
            ImportDeclaration("s", "i", "v", **kwargs)


# --- S5: normalization proof population vs CFA phase coherence ---------------

def test_mono_proof_on_cfa_flat_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, cfa_phase="GRBG")
    flat = make_frame((4, 4), value=2.0, exposure_s=1.0, cfa_phase="GRBG", units="dimensionless")
    masters = {
        "flat": MasterBinding(
            "flat", flat, flat_form="normalized_response",
            normalization_proof=_proof(population="mono-valid"),
        ),
    }
    with pytest.raises(InvalidRequestError):
        execute_calibration(
            light, CalibrationRequest("control", "apply"), masters,
            flat_prep_mode="already_normalized",
        )


def test_cfa_proof_on_mono_flat_rejected(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    flat = make_frame((4, 4), value=2.0, exposure_s=1.0, cfa_phase="mono", units="dimensionless")
    masters = {
        "flat": MasterBinding(
            "flat", flat, flat_form="normalized_response",
            normalization_proof=_proof(
                population="cfa-4-plane",
                scalars={"G1": 1.0, "R": 1.0, "B": 1.0, "G2": 1.0},
                plane_counts={"G1": 1, "R": 1, "B": 1, "G2": 1},
            ),
        ),
    }
    with pytest.raises(InvalidRequestError):
        execute_calibration(
            light, CalibrationRequest("control", "apply"), masters,
            flat_prep_mode="already_normalized",
        )
