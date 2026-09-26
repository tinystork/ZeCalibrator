"""Identity / liaison tests: strict, conservative sensor identity reuse."""

from __future__ import annotations

from zecalibrator.bpm.identity import (
    DETECTOR_MISMATCH,
    MISSING_REQUIRED_FIELD,
    READOUT_MISMATCH,
    identity_matches,
    identity_reasons,
    sensor_identity_from_light_constraints,
)
from zecalibrator.core.descriptors import (
    Acquisition,
    DetectorIdentity,
    LightConstraints,
    OpticalIdentity,
)
from zecalibrator.core.geometry import Geometry

from conftest import make_identity


def test_identical_identity_matches():
    a = make_identity()
    b = make_identity()
    assert identity_matches(a, b)
    assert identity_reasons(a, b) == ()


def test_identity_is_deterministic_digest():
    a = make_identity()
    b = make_identity()
    assert a.identity_digest() == b.identity_digest()


def test_wrong_detector_rejected():
    a = make_identity(detector_instance_id="DET-0001")
    b = make_identity(detector_instance_id="DET-0002")
    assert not identity_matches(a, b)
    assert DETECTOR_MISMATCH in identity_reasons(a, b)


def test_wrong_detector_model_rejected():
    a = make_identity(detector_model="MODEL-X")
    b = make_identity(detector_model="MODEL-Y")
    assert not identity_matches(a, b)
    assert DETECTOR_MISMATCH in identity_reasons(a, b)


def test_unknown_detector_never_matches():
    a = make_identity(detector_instance_id="DET-0001")
    b = make_identity(detector_instance_id="unknown")
    assert not identity_matches(a, b)
    assert MISSING_REQUIRED_FIELD in identity_reasons(a, b)


def test_unknown_geometry_never_matches():
    a = make_identity(cfa_phase="GRBG")
    b = make_identity(cfa_phase=None)
    assert not identity_matches(a, b)
    assert MISSING_REQUIRED_FIELD in identity_reasons(a, b)


def test_wrong_shape_rejected():
    a = make_identity(shape=(1080, 1920))
    b = make_identity(shape=(1080, 1930))
    assert not identity_matches(a, b)
    assert "GEOMETRY_MISMATCH" in identity_reasons(a, b)


def test_wrong_binning_rejected():
    a = make_identity(binning=(1, 1))
    b = make_identity(binning=(2, 2))
    assert not identity_matches(a, b)
    assert "BINNING_MISMATCH" in identity_reasons(a, b)


def test_wrong_cfa_phase_rejected():
    a = make_identity(cfa_phase="GRBG")
    b = make_identity(cfa_phase="RGGB")
    assert not identity_matches(a, b)
    assert "CFA_PHASE_MISMATCH" in identity_reasons(a, b)


def test_wrong_roi_origin_rejected():
    a = make_identity(roi_origin=(0, 0))
    b = make_identity(roi_origin=(1, 0))
    assert not identity_matches(a, b)
    assert "ROI_ORIGIN_MISMATCH" in identity_reasons(a, b)


def test_wrong_readout_rejected():
    a = make_identity(gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16")
    b = make_identity(gain=120, offset=50, readout_mode="MODE_A", adc_mode="MODE_16")
    assert not identity_matches(a, b)
    assert READOUT_MISMATCH in identity_reasons(a, b)


def test_wrong_readout_mode_rejected():
    a = make_identity(readout_mode="MODE_A")
    b = make_identity(readout_mode="MODE_B")
    assert not identity_matches(a, b)
    assert READOUT_MISMATCH in identity_reasons(a, b)


def test_unknown_readout_field_rejected():
    a = make_identity(adc_mode="MODE_16")
    b = make_identity(adc_mode=None)
    assert not identity_matches(a, b)
    assert MISSING_REQUIRED_FIELD in identity_reasons(a, b)


def test_sensor_identity_from_light_constraints_reuses_liaison():
    light = LightConstraints(
        geometry=Geometry(
            shape=(1080, 1920), sensor_dimensions=(1080, 1920), binning=(1, 1),
            roi_origin=(0, 0), roi_extent=(1080, 1920), orientation="identity",
            cfa_phase="GRBG",
        ),
        detector=DetectorIdentity(detector_instance_id="DET-0001", detector_model="MODEL-X"),
        acquisition=Acquisition(gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16"),
        optical=OpticalIdentity(filter=None, optical_train_id=None),
    )
    ident = sensor_identity_from_light_constraints(light)
    assert ident.detector.detector_instance_id == "DET-0001"
    assert ident.geometry.cfa_phase == "GRBG"
    assert ident.readout.gain == 100
    assert ident.readout.offset == 50
    assert ident.readout.readout_mode == "MODE_A"
    assert ident.readout.adc_mode == "MODE_16"
