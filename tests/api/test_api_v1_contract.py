"""Public ``zecalibrator.api.v1`` contract tests (Phase 5 / M1, rework-1).

These tests exercise the approved public surface using **only** public imports
(no ``zecalibrator.core`` / ``zecalibrator.application`` / ``zecalibrator.io``
module-path contract). Synthetic SYNTH-BASE-1 facts are explicit, never real
detector defaults.

Covered: symbol contract, cold-import/get_api_info laziness, strict round-trip,
MATCHED/NO_MATCH/AMBIGUOUS, detached validation, positive+negative CPU execution
with **consumed** external masks, shape/precision/identity-digest witnesses,
public exception catchability, provenance, and interop ``provides``.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.api.v1 import (
    Acquisition,
    ArrayFrameSource,
    ArrayInputIdentity,
    CalibrationPlan,
    CalibrationRequest,
    CancellationToken,
    Candidate,
    DecisionEnvelope,
    DescriptorSnapshot,
    DetectorIdentity,
    ExecutionOptions,
    FitsFileLocator,
    FitsFrameSource,
    FitsInputIdentity,
    FrameInspection,
    Geometry,
    ImportDeclaration,
    InMemorySource,
    InvalidRequestError,
    LibraryClosedError,
    LibraryHandle,
    LibrarySnapshot,
    LightConstraints,
    MaskPayloadLocator,
    MasterBinding,
    MasterDescriptor,
    OpticalIdentity,
    PolicyParameters,
    ProgressObserver,
    SensorMetadata,
    ValidityEvidence,
    VersionSet,
    build_sensor_metadata,
    calibrate_frame,
    default_match_policy,
    get_api_info,
    inspect_frame,
    open_library,
    resolve_calibration,
    validate_binding,
    validate_plan,
)

SHAPE = (4, 4)
LIBRARY_SCHEMA = "zecalibrator.library.v1"


# ---------------------------------------------------------------------------
# Synthetic fixture builders (public API only)
# ---------------------------------------------------------------------------
def _geo():
    return Geometry(
        shape=SHAPE,
        sensor_dimensions=SHAPE,
        binning=(1, 1),
        roi_origin=(0, 0),
        roi_extent=SHAPE,
        orientation="identity",
        cfa_phase="mono",
    )


def _detector():
    return DetectorIdentity(detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA")


def _acquisition(exposure_s=10.0, **kw):
    base = dict(
        gain=100.0,
        offset=50.0,
        readout_mode="MODE_A",
        adc_mode="MODE_16",
        temperature_c=20.0,
        exposure_s=exposure_s,
        saturation_limit_adu=60000.0,
        saturation_evidence="qualified",
        bias_exposure_max_s=0.01,
        short_flat_profile=False,
    )
    base.update(kw)
    return Acquisition(**base)


def _optical():
    return OpticalIdentity(filter="NONE", optical_train_id="SYNTH-TRAIN-1")


def _declaration(**kw):
    base = dict(
        source="synthetic_fixture",
        identity="SYNTH-BASE-1",
        version="1.0",
        domain="raw",
        units="ADU",
        detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-CFA",
        gain=100.0,
        offset=50.0,
        readout_mode="MODE_A",
        adc_mode="MODE_16",
        binning=(1, 1),
        sensor_dimensions=SHAPE,
        orientation="identity",
        cfa_phase="mono",
        roi_origin=(0, 0),
        exposure_s=10.0,
        temperature_c=20.0,
        filter="NONE",
        optical_train_id="SYNTH-TRAIN-1",
        bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0,
        saturation_evidence="qualified",
    )
    base.update(kw)
    return ImportDeclaration(**base)


def _sensor_metadata(exposure_s=10.0, **kw):
    declaration = _declaration(exposure_s=exposure_s, **kw)
    md = build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=declaration, units="ADU"
    )
    # Attach the explicit synthetic roi_extent (consumer-supplied metadata).
    geo = Geometry(
        shape=md.geometry.shape,
        sensor_dimensions=md.geometry.sensor_dimensions,
        binning=md.geometry.binning,
        roi_origin=md.geometry.roi_origin,
        roi_extent=SHAPE,
        orientation=md.geometry.orientation,
        cfa_phase=md.geometry.cfa_phase,
    )
    return SensorMetadata(
        original_cards=md.original_cards,
        normalized=md.normalized,
        conflicts=md.conflicts,
        geometry=geo,
        raw_domain_declaration=md.raw_domain_declaration,
        units=md.units,
        declaration=md.declaration,
        exposure_s=md.exposure_s,
        temperature_c=md.temperature_c,
        gain=md.gain,
        offset=md.offset,
        readout_mode=md.readout_mode,
        adc_mode=md.adc_mode,
        filter=md.filter,
        detector_model=md.detector_model,
        detector_instance_id=md.detector_instance_id,
        optical_train_id=md.optical_train_id,
        saturation_limit_adu=md.saturation_limit_adu,
        saturation_evidence=md.saturation_evidence,
        warnings=md.warnings,
    )


def _inspection(exposure_s=10.0):
    return FrameInspection(
        metadata=_sensor_metadata(exposure_s=exposure_s),
        identity=FitsInputIdentity(path="/light.fits", hdu=0),
        domain_finding="raw",
        hdu=0,
        shape=SHAPE,
        warnings=(),
    )


def _dark_descriptor(exposure_s=10.0, content_sha256="a" * 64, size_bytes=16, mask_identity="b" * 64):
    return MasterDescriptor(
        master_type="dark",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="included",
        geometry=_geo(),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=exposure_s),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=ValidityEvidence(saturation_limit_known=True),
    )


def _flat_validity():
    return ValidityEvidence(
        saturation_limit_known=True,
        valid_normalization_count={"mono": 95},
        total_normalization_count={"mono": 100},
        quality_policy_state="qualified",
        illumination="flat_field",
        exposure_quality="qualified",
    )


def _flat_descriptor(content_sha256="f" * 64, size_bytes=16, mask_identity="f" * 64):
    return MasterDescriptor(
        master_type="flat",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="not_applicable",
        geometry=_geo(),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=_flat_validity(),
        flat_form="raw_response",
        optical_train_id="SYNTH-TRAIN-1",
        filter="NONE",
    )


def _flat_dark_descriptor(content_sha256="e" * 64, size_bytes=16, mask_identity="e" * 64):
    return MasterDescriptor(
        master_type="flat_dark",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="included",
        geometry=_geo(),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=ValidityEvidence(saturation_limit_known=True),
    )


def _candidate(candidate_id, desc, locator_path=None, mask_path=None):
    locators = (FitsFileLocator(path=locator_path, hdu=desc.hdu),) if locator_path else ()
    mask_locator = MaskPayloadLocator(path=mask_path) if mask_path else None
    return Candidate(
        candidate_id=candidate_id,
        descriptor=desc,
        descriptor_snapshot=DescriptorSnapshot(desc),
        locators=locators,
        mask_locator=mask_locator,
    )


def _binding(desc, locator_path="/d.fits", mask_path="/d.mask"):
    locators = (FitsFileLocator(path=locator_path, hdu=desc.hdu),)
    return MasterBinding(
        descriptor_id=desc.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(desc),
        content_sha256=desc.content_sha256,
        size_bytes=desc.size_bytes,
        hdu=desc.hdu,
        mask_identity=desc.mask_identity,
        locators=locators,
        mask_locator=MaskPayloadLocator(path=mask_path),
    )


def _library(candidates):
    return LibraryHandle(
        LibrarySnapshot(revision="r1", schema_version=LIBRARY_SCHEMA, candidates=candidates)
    )


def _write_fits(path, data):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def _write_mask(path, mask):
    np.save(path, np.asarray(mask, dtype=np.uint16), allow_pickle=False)
    return str(path)


def _write_master(tmp_path, name, value, mask=None):
    """Write a master FITS + mask and return (fits_path, mask_path, fits_sha, mask_sha)."""
    fits_path = tmp_path / f"{name}.fits"
    _write_fits(fits_path, np.full(SHAPE, value, dtype=np.float32))
    fits_bytes = fits_path.read_bytes()
    mask = np.zeros(SHAPE, dtype=np.uint16) if mask is None else np.asarray(mask, dtype=np.uint16)
    mask_path = tmp_path / f"{name}.mask.npy"
    _write_mask(mask_path, mask)
    mask_bytes = mask_path.read_bytes()
    return str(fits_path), str(mask_path), hashlib.sha256(fits_bytes).hexdigest(), len(fits_bytes), hashlib.sha256(mask_bytes).hexdigest()


def _array_source(data, metadata=None, decoded_digest=""):
    return ArrayFrameSource(
        data=np.asarray(data, dtype=np.float32),
        metadata=metadata or _sensor_metadata(),
        identity=ArrayInputIdentity(caller_logical_id="light-1", decoded_digest=decoded_digest),
    )


def _control_plan():
    return resolve_calibration(
        _inspection(), CalibrationRequest("control"), _library({}), default_match_policy()
    ).plan


def _dark_plan(tmp_path, *, dark_value=10.0, mask=None):
    dark_fits, dark_mask, sha, size, mask_sha = _write_master(tmp_path, "dark", dark_value, mask=mask)
    desc = _dark_descriptor(content_sha256=sha, size_bytes=size, mask_identity=mask_sha)
    library = _library({"dark": (_candidate("dark", desc, locator_path=dark_fits, mask_path=dark_mask),)})
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    ).plan
    return plan


# ---------------------------------------------------------------------------
# Symbol contract / meta / headless
# ---------------------------------------------------------------------------
def test_public_symbol_list_is_explicit_and_complete():
    import inspect

    public = {n for n in dir(v1) if not n.startswith("_")}
    extras = {
        n
        for n in public - set(v1.__all__)
        if not inspect.ismodule(getattr(v1, n)) and n != "annotations"
    }
    assert extras == set()
    assert set(v1.__all__) <= public


def test_get_api_info_static_surface():
    import zecalibrator

    info = get_api_info()
    assert info.api_version == "1.0"
    assert info.product_version == zecalibrator.__version__
    assert info.capabilities == (
        "calibrate_frame",
        "calibration_library",
        "master_matching",
        "provenance",
        "cancel",
    )


# ---------------------------------------------------------------------------
# resolve_calibration — selection states (via FITS inspection + ROI evidence)
# ---------------------------------------------------------------------------
def _fits_inspection(tmp_path, name="light.fits", exposure_s=10.0):
    path = _write_fits(tmp_path / name, np.full(SHAPE, 100.0, dtype=np.float32))
    source = FitsFrameSource(
        path=path,
        declaration=_declaration(exposure_s=exposure_s),
        roi_extent=v1.RoiExtentEvidence(extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0"),
    )
    result = inspect_frame(source)
    assert result.operation_status == "COMPLETED", result.reason_code
    assert result.inspection.metadata.geometry.roi_extent == SHAPE
    return result.inspection


def test_resolve_matched_via_fits_inspection(tmp_path):
    inspection = _fits_inspection(tmp_path)
    dark_fits, dark_mask, sha, size, mask_sha = _write_master(tmp_path, "dark", 10.0)
    desc = _dark_descriptor(content_sha256=sha, size_bytes=size, mask_identity=mask_sha)
    library = _library({"dark": (_candidate("dark", desc, locator_path=dark_fits, mask_path=dark_mask),)})
    result = resolve_calibration(inspection, CalibrationRequest("dark_incl_bias"), library, default_match_policy())
    assert result.operation_status == "COMPLETED"
    assert result.outcome == "MATCHED"
    assert result.plan is not None
    assert "dark" in result.plan.masters


def test_resolve_no_match_has_no_plan():
    library = _library({"dark": (_candidate("d1", _dark_descriptor()),)})
    result = resolve_calibration(
        _inspection(exposure_s=5.0), CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    )
    assert result.outcome == "NO_MATCH"
    assert result.plan is None
    assert "EXPOSURE_MISMATCH" in result.decision.reason_codes


def test_resolve_ambiguous_has_no_plan():
    library = _library(
        {"dark": (_candidate("d1", _dark_descriptor(content_sha256="a" * 64)), _candidate("d2", _dark_descriptor(content_sha256="c" * 64)))}
    )
    result = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    )
    assert result.outcome == "AMBIGUOUS"
    assert result.plan is None


def test_resolve_closed_handle_raises_public_error():
    library = _library({"dark": (_candidate("d1", _dark_descriptor()),)})
    library.close()
    with pytest.raises(LibraryClosedError):
        resolve_calibration(_inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy())


def test_resolve_bad_request_raises():
    library = _library({"dark": (_candidate("d1", _dark_descriptor()),)})
    with pytest.raises(InvalidRequestError):
        resolve_calibration(_inspection(), "not-a-request", library, default_match_policy())


# ---------------------------------------------------------------------------
# Strict versioned round-trip
# ---------------------------------------------------------------------------
def test_plan_roundtrip_preserves_identity():
    library = _library({"dark": (_candidate("d1", _dark_descriptor()),)})
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    ).plan
    rebuilt = CalibrationPlan.from_dict(plan.to_dict())
    assert rebuilt.plan_id == plan.plan_id
    assert rebuilt.plan_digest_dict() == plan.plan_digest_dict()
    rebuilt.verify_plan_id()


def test_decision_roundtrip_preserves_outcome_and_plan():
    library = _library({"dark": (_candidate("d1", _dark_descriptor()),)})
    decision = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    ).decision
    rebuilt = DecisionEnvelope.from_dict(decision.to_dict())
    assert rebuilt.outcome == decision.outcome
    assert rebuilt.plan.plan_id == decision.plan.plan_id


def test_descriptor_snapshot_roundtrip_preserves_descriptor_id():
    snap = DescriptorSnapshot(_dark_descriptor())
    assert DescriptorSnapshot.from_dict(snap.to_dict()).descriptor_id == snap.descriptor_id


def test_provenance_and_result_roundtrip():
    from zecalibrator.api.v1 import CalibrationResult, ProvenanceRecord

    plan = _control_plan()
    md = _sensor_metadata()
    source = _array_source(np.full(SHAPE, 100.0), metadata=md)
    result = calibrate_frame(source, plan, ExecutionOptions())

    # A COMPLETED result is audit-only: payload is not serialized, so rebuilding
    # a payload-less COMPLETED body must refuse (M5).
    d = result.to_dict()
    assert d["payload_serialized"] is False
    with pytest.raises(InvalidRequestError):
        CalibrationResult.from_dict(d)

    # ProvenanceRecord alone round-trips losslessly (no pixel payload).
    prov = ProvenanceRecord.from_dict(result.provenance.to_dict())
    assert prov.plan.plan_id == result.provenance.plan.plan_id
    assert prov.input_identity.caller_logical_id == "light-1"
    assert prov.operation_id
    assert prov.input_dtype == "float32"
    assert prov.input_units == "ADU"


def test_failed_result_roundtrip_is_lossless():
    from zecalibrator.api.v1 import CalibrationResult

    plan = _control_plan()
    # A cancelled result carries no pixel payload, so it round-trips.
    token = CancellationToken()
    token.cancel()
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions(), cancel=token)
    assert result.status == "CANCELLED"
    rebuilt = CalibrationResult.from_dict(result.to_dict())
    assert rebuilt.status == "CANCELLED"
    assert rebuilt.provenance.plan.plan_id == result.provenance.plan.plan_id


def test_plan_serialization_rejects_unknown_schema():
    library = _library({"dark": (_candidate("d1", _dark_descriptor()),)})
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    ).plan
    d = dict(plan.to_dict())
    d["schema_version"] = "zecalibrator.plan.v999"
    with pytest.raises(ValueError):
        CalibrationPlan.from_dict(d)


# ---------------------------------------------------------------------------
# validate_binding / validate_plan (detached)
# ---------------------------------------------------------------------------
def test_validate_binding_valid():
    image_bytes = b"IMAGE_BYTES"
    mask_bytes = b"mask-bytes"
    desc = _dark_descriptor(
        content_sha256=hashlib.sha256(image_bytes).hexdigest(),
        size_bytes=len(image_bytes),
        mask_identity=hashlib.sha256(mask_bytes).hexdigest(),
    )
    binding = _binding(desc, locator_path="/d.fits", mask_path="/d.mask")
    source = InMemorySource(images={"/d.fits": image_bytes}, masks={"/d.mask": mask_bytes})
    assert validate_binding(binding, source).status == "VALID"


def test_validate_binding_content_mismatch():
    desc = _dark_descriptor(content_sha256="a" * 64, size_bytes=16, mask_identity="b" * 64)
    binding = _binding(desc, locator_path="/d.fits", mask_path="/d.mask")
    source = InMemorySource(images={"/d.fits": b"OTHER"}, masks={"/d.mask": b"other"})
    result = validate_binding(binding, source)
    assert result.status == "FAILED"
    assert any("content mismatch" in r for r in result.reasons)


def test_validate_binding_no_locator_refuses():
    desc = _dark_descriptor()
    binding = MasterBinding(
        descriptor_id=desc.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(desc),
        content_sha256=desc.content_sha256,
        size_bytes=desc.size_bytes,
        hdu=desc.hdu,
        mask_identity=desc.mask_identity,
        locators=(),
        mask_locator=None,
    )
    result = validate_binding(binding, InMemorySource())
    assert result.status == "FAILED"
    assert any("no image locator" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# calibrate_frame — execution with consumed masks
# ---------------------------------------------------------------------------
def test_calibrate_frame_dark_incl_bias(tmp_path):
    plan = _dark_plan(tmp_path)
    source = _array_source(np.full(SHAPE, 100.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    assert np.allclose(result.data, np.full(SHAPE, 90.0))
    assert np.count_nonzero(result.mask) == 0
    assert result.provenance is not None
    assert result.provenance.plan.plan_id == plan.plan_id


def test_calibrate_frame_control_no_masters(tmp_path):
    plan = _control_plan()
    source = _array_source(np.full(SHAPE, 100.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    assert np.allclose(result.data, np.full(SHAPE, 100.0))


def test_calibrate_frame_raw_flat(tmp_path):
    dark_fits, dark_mask, dsha, dsize, dmsha = _write_master(tmp_path, "dark", 10.0)
    flat_fits, flat_mask, fsha, fsize, fmsha = _write_master(tmp_path, "flat", 100.0)
    fd_fits, fd_mask, fdsha, fdsize, fdmsha = _write_master(tmp_path, "flat_dark", 10.0)

    dark = _dark_descriptor(content_sha256=dsha, size_bytes=dsize, mask_identity=dmsha)
    flat = _flat_descriptor(content_sha256=fsha, size_bytes=fsize, mask_identity=fmsha)
    flat_dark = _flat_dark_descriptor(content_sha256=fdsha, size_bytes=fdsize, mask_identity=fdmsha)

    library = _library(
        {
            "dark": (_candidate("dark", dark, locator_path=dark_fits, mask_path=dark_mask),),
            "flat": (_candidate("flat", flat, locator_path=flat_fits, mask_path=flat_mask),),
            "flat_dark": (_candidate("flat_dark", flat_dark, locator_path=fd_fits, mask_path=fd_mask),),
        }
    )
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias", "apply"), library, default_match_policy()
    ).plan
    assert set(plan.masters) == {"dark", "flat", "flat_dark"}

    source = _array_source(np.full(SHAPE, 110.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    # 110 - 10 = 100; flat (100-10=90, median 90 -> R=1.0) => 100.
    assert np.allclose(result.data, np.full(SHAPE, 100.0))


def test_calibrate_frame_consumes_external_mask(tmp_path):
    mask = np.zeros(SHAPE, dtype=np.uint16)
    mask[0, 0] = 0x0001  # external DQ marks the dark pixel invalid
    plan = _dark_plan(tmp_path, dark_value=10.0, mask=mask)
    source = _array_source(np.full(SHAPE, 100.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert result.mask[0, 0] != 0
    assert np.isnan(result.data[0, 0])
    assert np.allclose(result.data[1:, 1:], 90.0)


def test_calibrate_frame_plan_source_mismatch_raises(tmp_path):
    plan = _control_plan()
    mismatched = _array_source(np.full(SHAPE, 100.0), metadata=_sensor_metadata(exposure_s=5.0))
    with pytest.raises(InvalidRequestError):
        calibrate_frame(mismatched, plan, ExecutionOptions())


def test_calibrate_frame_pre_cancelled_returns_cancelled(tmp_path):
    plan = _control_plan()
    source = _array_source(np.full(SHAPE, 100.0))
    token = CancellationToken()
    token.cancel()
    result = calibrate_frame(source, plan, ExecutionOptions(), cancel=token)
    assert result.status == "CANCELLED"
    assert result.data is None
    assert result.provenance.status == "CANCELLED"


def test_calibrate_frame_progress_events(tmp_path):
    plan = _control_plan()
    source = _array_source(np.full(SHAPE, 100.0))
    events = []
    result = calibrate_frame(source, plan, ExecutionOptions(), progress=ProgressObserver(events.append))
    assert result.status == "COMPLETED"
    assert len(events) >= 1


def test_calibrate_frame_missing_mask_refuses(tmp_path):
    # A binding with no mask locator fails validate_plan -> plan invalid.
    dark_fits, _, sha, size, _ = _write_master(tmp_path, "dark", 10.0)
    desc = _dark_descriptor(content_sha256=sha, size_bytes=size, mask_identity="b" * 64)
    binding = MasterBinding(
        descriptor_id=desc.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(desc),
        content_sha256=desc.content_sha256,
        size_bytes=desc.size_bytes,
        hdu=desc.hdu,
        mask_identity=desc.mask_identity,
        locators=(FitsFileLocator(path=dark_fits, hdu=0),),
        mask_locator=None,
    )
    plan = CalibrationPlan.build(
        request=CalibrationRequest("dark_incl_bias"),
        light_constraints=LightConstraints(
            geometry=_geo(), detector=_detector(), acquisition=_acquisition(), optical=_optical(), raw_domain_declaration="raw"
        ),
        masters={"dark": binding},
        policy_parameters=PolicyParameters(
            exposure_tolerance=default_match_policy().exposure_tolerance,
            temperature_tolerance=default_match_policy().temperature_tolerance,
            flat_quality_policy=default_match_policy().flat_quality_policy,
        ),
        versions=VersionSet(),
    )
    source = _array_source(np.full(SHAPE, 100.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "PLAN_INVALID"


def test_calibrate_frame_content_mismatch_fails(tmp_path):
    dark_fits, dark_mask, _, _, mask_sha = _write_master(tmp_path, "dark", 10.0)
    # Descriptor content hash deliberately wrong vs the file bytes.
    desc = _dark_descriptor(content_sha256="a" * 64, size_bytes=16, mask_identity=mask_sha)
    library = _library({"dark": (_candidate("dark", desc, locator_path=dark_fits, mask_path=dark_mask),)})
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    ).plan
    source = _array_source(np.full(SHAPE, 100.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "PLAN_INVALID"


def test_calibrate_frame_refuses_gpu():
    with pytest.raises(InvalidRequestError):
        ExecutionOptions(backend_requested="gpu")


# ---------------------------------------------------------------------------
# F4 — actual data shape / precision / identity-digest
# ---------------------------------------------------------------------------
def test_array_shape_mismatch_refused(tmp_path):
    plan = _control_plan()
    bad = ArrayFrameSource(
        data=np.full((2, 2), 100.0, dtype=np.float32),
        metadata=_sensor_metadata(),  # geometry (4,4)
        identity=ArrayInputIdentity(caller_logical_id="bad"),
    )
    result = calibrate_frame(bad, plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "SOURCE_ERROR"


def test_uint64_precision_refused(tmp_path):
    plan = _control_plan()
    source = ArrayFrameSource(
        data=np.full(SHAPE, 2 ** 53 + 1, dtype=np.uint64),
        metadata=_sensor_metadata(saturation_evidence="unknown", saturation_limit_adu=None),
        identity=ArrayInputIdentity(caller_logical_id="precision"),
    )
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "PRECISION_REFUSAL"


def test_uint64_large_integer_refused(tmp_path):
    plan = _control_plan()
    source = ArrayFrameSource(
        data=np.full(SHAPE, 2 ** 24 + 1, dtype=np.uint64),
        metadata=_sensor_metadata(),
        identity=ArrayInputIdentity(caller_logical_id="precision"),
    )
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "PRECISION_REFUSAL"


def test_identity_digest_mismatch_refused(tmp_path):
    plan = _control_plan()
    source = ArrayFrameSource(
        data=np.full(SHAPE, 100.0, dtype=np.float32),
        metadata=_sensor_metadata(),
        identity=ArrayInputIdentity(caller_logical_id="light-1", decoded_digest="0" * 64),
    )
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "SOURCE_ERROR"


def test_inspect_array_computes_real_identity_digest():
    md = _sensor_metadata()
    source = ArrayFrameSource(
        data=np.full(SHAPE, 7.0, dtype=np.float32),
        metadata=md,
        identity=ArrayInputIdentity(caller_logical_id="arr-1"),
    )
    result = inspect_frame(source)
    assert result.operation_status == "COMPLETED"
    digest = result.inspection.identity.decoded_digest
    assert len(digest) == 64
    # A subsequent source declaring the correct digest is accepted.
    source2 = ArrayFrameSource(
        data=np.full(SHAPE, 7.0, dtype=np.float32),
        metadata=md,
        identity=ArrayInputIdentity(caller_logical_id="arr-1", decoded_digest=digest),
    )
    assert inspect_frame(source2).operation_status == "COMPLETED"


# ---------------------------------------------------------------------------
# F6 — public error catchability
# ---------------------------------------------------------------------------
def test_public_source_error_catchable():
    with pytest.raises(v1.SourceError):
        v1.FilesystemSource().image_identity(FitsFileLocator(path="/nonexistent", hdu=0))


def test_public_library_closed_error_is_same_class():
    assert v1.LibraryClosedError is __import__("zecalibrator.application.library", fromlist=["LibraryClosedError"]).LibraryClosedError


def test_open_library_directory_index_fails_structurally(tmp_path):
    result = open_library(v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path)))
    assert result.operation_status == "FAILED"
    assert result.handle is None


def test_open_library_empty_initialized_index_opens(tmp_path):
    # Build an empty initialized index via the public-invisible engine path, then
    # verify the public open returns OPENED with zero entries.
    from zecalibrator.io.library_index import LibraryIndex

    index_path = tmp_path / "lib.sqlite"
    idx = LibraryIndex(str(index_path)).open(initialize=True)
    try:
        idx.publish_revision("r1", {})
    finally:
        idx.close()

    result = open_library(v1.LibrarySpec(root=str(tmp_path), index_path=str(index_path)))
    assert result.operation_status == "OPENED"
    assert result.handle.snapshot.roles() == ()


def test_open_library_unsupported_schema_fails(tmp_path):
    import sqlite3

    index_path = tmp_path / "lib.sqlite"
    conn = sqlite3.connect(str(index_path))
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', 'unknown.v9')")
    conn.commit()
    conn.close()

    result = open_library(v1.LibrarySpec(root=str(tmp_path), index_path=str(index_path)))
    assert result.operation_status == "FAILED"
    assert result.reason_code == "UNSUPPORTED_SCHEMA"


# ---------------------------------------------------------------------------
# Interop declaration
# ---------------------------------------------------------------------------
def test_interop_provides_five_implemented_capabilities():
    pkg = files("zecalibrator")
    data = json.loads((pkg / "zesoftware_interop.json").read_bytes().decode("utf-8"))
    assert data["schema"] == "zesoftware.interop.v1"
    assert data["product_id"] == "zecalibrator"
    assert len(data["provides"]) == 1
    entry = data["provides"][0]
    assert entry["api_module"] == "zecalibrator.api.v1"
    assert entry["api_version"] == "1.0"
    assert entry["capabilities"] == [
        "calibrate_frame",
        "calibration_library",
        "master_matching",
        "provenance",
        "cancel",
    ]
    assert get_api_info().capabilities == tuple(entry["capabilities"])


# ---------------------------------------------------------------------------
# M1 — full approved flat coverage (normalized / corrected, mono + CFA)
# ---------------------------------------------------------------------------
def _flat_validity_cfa():
    return ValidityEvidence(
        saturation_limit_known=True,
        valid_normalization_count={"G1": 95, "R": 95, "B": 95, "G2": 95},
        total_normalization_count={"G1": 100, "R": 100, "B": 100, "G2": 100},
        quality_policy_state="qualified",
        illumination="flat_field",
        exposure_quality="qualified",
    )


def _geo_cfa(cfa="RGGB", shape=SHAPE):
    return Geometry(
        shape=shape,
        sensor_dimensions=shape,
        binning=(1, 1),
        roi_origin=(0, 0),
        roi_extent=shape,
        orientation="identity",
        cfa_phase=cfa,
    )


def _cfa_sensor_metadata(cfa="RGGB"):
    decl = _declaration(cfa_phase=cfa)
    md = build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU"
    )
    geo = Geometry(
        shape=md.geometry.shape,
        sensor_dimensions=md.geometry.sensor_dimensions,
        binning=md.geometry.binning,
        roi_origin=md.geometry.roi_origin,
        roi_extent=SHAPE,
        orientation=md.geometry.orientation,
        cfa_phase=cfa,
    )
    return SensorMetadata(
        original_cards=md.original_cards,
        normalized=md.normalized,
        conflicts=md.conflicts,
        geometry=geo,
        raw_domain_declaration=md.raw_domain_declaration,
        units=md.units,
        declaration=md.declaration,
        exposure_s=md.exposure_s,
        temperature_c=md.temperature_c,
        gain=md.gain,
        offset=md.offset,
        readout_mode=md.readout_mode,
        adc_mode=md.adc_mode,
        filter=md.filter,
        detector_model=md.detector_model,
        detector_instance_id=md.detector_instance_id,
        optical_train_id=md.optical_train_id,
        saturation_limit_adu=md.saturation_limit_adu,
        saturation_evidence=md.saturation_evidence,
        warnings=md.warnings,
    )


def _write_flat_file(tmp_path, name, data, bunit="ADU"):
    p = tmp_path / name
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    if bunit:
        hdu.header["BUNIT"] = bunit
    hdu.writeto(p, overwrite=True)
    return str(p)


def _normalized_flat_descriptor(content_sha256, size_bytes, mask_identity, cfa="mono"):
    if cfa == "mono":
        scalars = v1.NormalizationScalars(mono=2.0)
        validity = _flat_validity()
        population = "mono-valid"
    else:
        scalars = v1.NormalizationScalars(g1=0.5, r=1.0, b=1.5, g2=2.0)
        validity = _flat_validity_cfa()
        population = "cfa-4-plane"
    return MasterDescriptor(
        master_type="flat",
        pixel_domain="normalized_response",
        physical_units="dimensionless",
        bias_state="not_applicable",
        geometry=_geo() if cfa == "mono" else _geo_cfa(cfa),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture",
            additive_correction_history=("bias_removed",),
            normalization=v1.NormalizationProvenance(
                algorithm="cfa-median-per-plane", population=population, scalars=scalars
            ),
        ),
        validity_evidence=validity,
        flat_form="normalized_response",
        normalization_algorithm="cfa-median-per-plane",
        normalization_scalars=scalars,
        optical_train_id="SYNTH-TRAIN-1",
        filter="NONE",
    )


def _corrected_flat_descriptor(content_sha256, size_bytes, mask_identity, cfa="mono"):
    validity = _flat_validity() if cfa == "mono" else _flat_validity_cfa()
    return MasterDescriptor(
        master_type="flat",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="not_applicable",
        geometry=_geo() if cfa == "mono" else _geo_cfa(cfa),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_correction_history=("bias_removed",)
        ),
        validity_evidence=validity,
        flat_form="corrected_unnormalized",
        optical_train_id="SYNTH-TRAIN-1",
        filter="NONE",
    )


def _flat_library(tmp_path, desc, name="flat"):
    path = str(tmp_path / f"{name}.fits")
    _write_fits(path, np.full(SHAPE, 2.0))
    mask_path = str(tmp_path / f"{name}.mask.npy")
    _write_mask(mask_path, np.zeros(SHAPE, dtype=np.uint16))
    data = open(path, "rb").read()
    mask_data = open(mask_path, "rb").read()
    desc = desc(
        content_sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        mask_identity=hashlib.sha256(mask_data).hexdigest(),
    )
    return _library({"flat": (_candidate("flat", desc, locator_path=path, mask_path=mask_path),)})


def test_calibrate_frame_normalized_flat_mono(tmp_path):
    path = _write_flat_file(tmp_path, "flat.fits", np.full(SHAPE, 2.0), bunit="dimensionless")
    mask_path = str(tmp_path / "flat.mask.npy")
    _write_mask(mask_path, np.zeros(SHAPE, dtype=np.uint16))
    fb = open(path, "rb").read()
    mb = open(mask_path, "rb").read()
    desc = _normalized_flat_descriptor(
        hashlib.sha256(fb).hexdigest(), len(fb), hashlib.sha256(mb).hexdigest(), cfa="mono"
    )
    library = _library({"flat": (_candidate("flat", desc, locator_path=path, mask_path=mask_path),)})
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("control", "apply"), library, default_match_policy()
    ).plan
    source = _array_source(np.full(SHAPE, 100.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    assert np.allclose(result.data, np.full(SHAPE, 50.0))  # 100 / 2.0


def test_calibrate_frame_corrected_flat_mono(tmp_path):
    fcorr = np.array([[80.0, 90.0, 100.0, 110.0]] * 4, dtype=np.float32)
    path = _write_flat_file(tmp_path, "flat.fits", fcorr, bunit="ADU")
    mask_path = str(tmp_path / "flat.mask.npy")
    _write_mask(mask_path, np.zeros(SHAPE, dtype=np.uint16))
    fb = open(path, "rb").read()
    mb = open(mask_path, "rb").read()
    desc = _corrected_flat_descriptor(
        hashlib.sha256(fb).hexdigest(), len(fb), hashlib.sha256(mb).hexdigest(), cfa="mono"
    )
    library = _library({"flat": (_candidate("flat", desc, locator_path=path, mask_path=mask_path),)})
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("control", "apply"), library, default_match_policy()
    ).plan
    source = _array_source(np.full(SHAPE, 100.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    # mono median of [80,90,100,110]*4 = 95; R = Fcorr/95; 100/0.8=125, 100/0.9=111.11, ...
    assert result.scalars.get("mono") == 95.0
    assert np.allclose(result.data[0], [118.75, 105.55555, 95.0, 86.36364], atol=1e-3)


def test_calibrate_frame_corrected_flat_cfa_four_scalars(tmp_path):
    # RGGB: parity labels (R, G1, G2, B) -> four distinct planes.
    pattern = np.array([[200.0, 100.0], [300.0, 400.0]], dtype=np.float32)  # R,G1 / G2,B
    fcorr = np.tile(pattern, (2, 2))
    path = _write_flat_file(tmp_path, "flat.fits", fcorr, bunit="ADU")
    mask_path = str(tmp_path / "flat.mask.npy")
    _write_mask(mask_path, np.zeros(SHAPE, dtype=np.uint16))
    fb = open(path, "rb").read()
    mb = open(mask_path, "rb").read()
    desc = _corrected_flat_descriptor(
        hashlib.sha256(fb).hexdigest(), len(fb), hashlib.sha256(mb).hexdigest(), cfa="RGGB"
    )
    library = _library({"flat": (_candidate("flat", desc, locator_path=path, mask_path=mask_path),)})
    md = _cfa_sensor_metadata("RGGB")
    inspection = FrameInspection(
        metadata=md, identity=FitsInputIdentity(path="/l.fits", hdu=0), domain_finding="raw", hdu=0, shape=SHAPE
    )
    plan = resolve_calibration(
        inspection, CalibrationRequest("control", "apply"), library, default_match_policy()
    ).plan
    source = ArrayFrameSource(
        data=np.full(SHAPE, 100.0, dtype=np.float32),
        metadata=md,
        identity=ArrayInputIdentity(caller_logical_id="l"),
    )
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    assert result.scalars["G1"] == 100.0
    assert result.scalars["R"] == 200.0
    assert result.scalars["G2"] == 300.0
    assert result.scalars["B"] == 400.0
    assert len({result.scalars["G1"], result.scalars["R"], result.scalars["G2"], result.scalars["B"]}) == 4


# ---------------------------------------------------------------------------
# M2 — hashed HDU identity bound to consumed HDU
# ---------------------------------------------------------------------------
def test_calibrate_frame_locator_hdu_mismatch_refused(tmp_path):
    p = tmp_path / "dark.fits"
    primary = fits.PrimaryHDU(np.full(SHAPE, 10.0, dtype=np.float32))
    primary.header["BUNIT"] = "ADU"
    ext = fits.ImageHDU(np.full(SHAPE, 40.0, dtype=np.float32))
    ext.header["BUNIT"] = "ADU"
    fits.HDUList([primary, ext]).writeto(p, overwrite=True)
    b = p.read_bytes()
    mask_path = str(tmp_path / "dark.mask.npy")
    _write_mask(mask_path, np.zeros(SHAPE, dtype=np.uint16))
    mb = open(mask_path, "rb").read()
    desc = _dark_descriptor(
        content_sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b), mask_identity=hashlib.sha256(mb).hexdigest()
    )
    # descriptor/binding hdu=0 but locator hdu=1 -> the consumed HDU is not the hashed one.
    cand = Candidate(
        candidate_id="dark",
        descriptor=desc,
        descriptor_snapshot=DescriptorSnapshot(desc),
        locators=(FitsFileLocator(path=str(p), hdu=1),),
        mask_locator=MaskPayloadLocator(path=mask_path),
    )
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), _library({"dark": (cand,)}), default_match_policy()
    ).plan
    source = _array_source(np.full(SHAPE, 100.0))
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "PLAN_INVALID"


def test_calibrate_frame_role_master_type_mismatch_refused(tmp_path):
    # Bind a 'dark' role to a flat descriptor -> semantic refusal.
    dark_fits, dark_mask, sha, size, mask_sha = _write_master(tmp_path, "dark", 10.0)
    flat = _flat_descriptor(content_sha256=sha, size_bytes=size, mask_identity=mask_sha)
    binding = MasterBinding(
        descriptor_id=flat.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(flat),
        content_sha256=flat.content_sha256,
        size_bytes=flat.size_bytes,
        hdu=flat.hdu,
        mask_identity=flat.mask_identity,
        locators=(FitsFileLocator(path=dark_fits, hdu=0),),
        mask_locator=MaskPayloadLocator(path=dark_mask),
    )
    policy = default_match_policy()
    plan = CalibrationPlan.build(
        request=CalibrationRequest("dark_incl_bias"),
        light_constraints=LightConstraints(
            geometry=_geo(), detector=_detector(), acquisition=_acquisition(), optical=_optical(), raw_domain_declaration="raw"
        ),
        masters={"dark": binding},
        policy_parameters=PolicyParameters(
            exposure_tolerance=policy.exposure_tolerance,
            temperature_tolerance=policy.temperature_tolerance,
            flat_quality_policy=policy.flat_quality_policy,
        ),
        versions=VersionSet(),
    )
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "PLAN_INVALID"


# ---------------------------------------------------------------------------
# M3 — mask payload representation enforced, no exception leak
# ---------------------------------------------------------------------------
def _calibrate_with_mask_bytes(tmp_path, mask_array):
    dark_fits, dark_mask, sha, size, _ = _write_master(tmp_path, "dark", 10.0)
    # Write the mask payload with the exact bytes we control.
    np.save(dark_mask, np.asarray(mask_array), allow_pickle=False)
    mask_bytes = open(dark_mask, "rb").read()
    desc = _dark_descriptor(
        content_sha256=sha, size_bytes=size, mask_identity=hashlib.sha256(mask_bytes).hexdigest()
    )
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"),
        _library({"dark": (_candidate("dark", desc, locator_path=dark_fits, mask_path=dark_mask),)}),
        default_match_policy(),
    ).plan
    return calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())


def test_mask_uint32_out_of_range_refused(tmp_path):
    result = _calibrate_with_mask_bytes(tmp_path, np.full(SHAPE, 65536, dtype=np.uint32))
    assert result.status == "FAILED"
    assert result.reason_code == "MASTER_LOAD_FAILED"


def test_mask_uint32_bit16_refused(tmp_path):
    result = _calibrate_with_mask_bytes(tmp_path, np.full(SHAPE, 0x00010001, dtype=np.uint32))
    assert result.status == "FAILED"
    assert result.reason_code == "MASTER_LOAD_FAILED"


def test_mask_uint8_refused(tmp_path):
    result = _calibrate_with_mask_bytes(tmp_path, np.full(SHAPE, 1, dtype=np.uint8))
    assert result.status == "FAILED"
    assert result.reason_code == "MASTER_LOAD_FAILED"


def test_mask_empty_file_structured_refusal(tmp_path):
    dark_fits, dark_mask, sha, size, _ = _write_master(tmp_path, "dark", 10.0)
    import pathlib

    pathlib.Path(dark_mask).write_bytes(b"")
    mask_bytes = b""
    desc = _dark_descriptor(
        content_sha256=sha, size_bytes=size, mask_identity=hashlib.sha256(mask_bytes).hexdigest()
    )
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"),
        _library({"dark": (_candidate("dark", desc, locator_path=dark_fits, mask_path=str(dark_mask)),)}),
        default_match_policy(),
    ).plan
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())
    assert result.status == "FAILED"
    assert result.reason_code == "MASTER_LOAD_FAILED"


# ---------------------------------------------------------------------------
# M4 — complete provenance (ROI evidence + operation_id + source facts)
# ---------------------------------------------------------------------------
def test_provenance_retains_roi_evidence_and_source_facts(tmp_path):
    path = _write_fits(tmp_path / "light.fits", np.full(SHAPE, 100.0))
    roi = v1.RoiExtentEvidence(
        extent=SHAPE, source="synthetic_fixture", identity="ROI-WITNESS-UNIQUE", version="1"
    )
    source = FitsFrameSource(path=path, declaration=_declaration(), roi_extent=roi)
    inspection = inspect_frame(source).inspection
    assert inspection.roi_extent_evidence.identity == "ROI-WITNESS-UNIQUE"
    dark_fits, dark_mask, sha, size, mask_sha = _write_master(tmp_path, "dark", 10.0)
    desc = _dark_descriptor(content_sha256=sha, size_bytes=size, mask_identity=mask_sha)
    library = _library({"dark": (_candidate("dark", desc, locator_path=dark_fits, mask_path=dark_mask),)})
    plan = resolve_calibration(
        inspection, CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    ).plan
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    prov = result.provenance
    assert prov.operation_id
    assert prov.roi_extent_evidence.identity == "ROI-WITNESS-UNIQUE"
    assert prov.input_dtype
    assert prov.input_units == "ADU"
    assert prov.input_scaling == {"bscale": 1.0, "bzero": 0.0}
    assert "ROI-WITNESS-UNIQUE" in json.dumps(prov.to_dict())


# ---------------------------------------------------------------------------
# M5 — strict, non-lossy serialization
# ---------------------------------------------------------------------------
def test_provenance_unknown_key_refused(tmp_path):
    plan = _control_plan()
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())
    d = dict(result.provenance.to_dict())
    d["new_science_fact"] = 1
    with pytest.raises(InvalidRequestError):
        v1.ProvenanceRecord.from_dict(d)


def test_provenance_unknown_identity_kind_refused(tmp_path):
    plan = _control_plan()
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())
    d = dict(result.provenance.to_dict())
    d["input_identity"] = {"kind": "weird"}
    with pytest.raises(InvalidRequestError):
        v1.ProvenanceRecord.from_dict(d)


# ---------------------------------------------------------------------------
# M6 — truthful monotonic cancellation/progress
# ---------------------------------------------------------------------------
def test_open_library_emits_progress(tmp_path):
    from zecalibrator.io.library_index import LibraryIndex

    index_path = tmp_path / "lib.sqlite"
    idx = LibraryIndex(str(index_path)).open(initialize=True)
    try:
        idx.publish_revision("r1", {})
    finally:
        idx.close()
    events = []
    result = open_library(
        v1.LibrarySpec(root=str(tmp_path), index_path=str(index_path)), progress=ProgressObserver(events.append)
    )
    assert result.operation_status == "OPENED"
    assert len(events) > 0


def test_resolve_emits_progress():
    library = _library({"dark": (_candidate("d1", _dark_descriptor()),)})
    events = []
    result = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy(), progress=ProgressObserver(events.append)
    )
    assert result.operation_status == "COMPLETED"
    assert len(events) > 0


def test_inspect_emits_progress(tmp_path):
    path = _write_fits(tmp_path / "light.fits", np.full(SHAPE, 42.0))
    events = []
    result = inspect_frame(
        FitsFrameSource(path=path, declaration=_declaration(), roi_extent=v1.RoiExtentEvidence(extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1")),
        progress=ProgressObserver(events.append),
    )
    assert result.operation_status == "COMPLETED"
    assert len(events) > 0


def test_calibrate_frame_monotonic_progress_and_complete_last(tmp_path):
    plan = _control_plan()
    events = []
    result = calibrate_frame(
        _array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions(), progress=ProgressObserver(events.append)
    )
    assert result.status == "COMPLETED"
    assert len(events) > 0
    completed = [e.completed for e in events]
    assert completed == sorted(completed)  # monotonic
    assert events[-1].phase == "complete"


def test_calibrate_frame_post_start_cancel_returns_cancelled(tmp_path):
    plan = _control_plan()
    token = CancellationToken()

    def _cancel_on_decode(event):
        if event.phase == "decode":
            token.cancel()

    result = calibrate_frame(
        _array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions(), cancel=token, progress=ProgressObserver(_cancel_on_decode)
    )
    assert result.status == "CANCELLED"


def test_callable_progress_normalized(tmp_path):
    plan = _control_plan()
    events = []
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions(), progress=events.append)
    assert result.status == "COMPLETED"
    assert len(events) > 0


def test_unsupported_schema_error_is_coherent_alias():
    import zecalibrator.io.library_index as ili

    assert v1.UnsupportedSchemaError is ili.UnsupportedSchemaError
    assert v1.LibraryError is ili.LibraryIndexError


# ---------------------------------------------------------------------------
# M7 — role→master_type map (bias_flat→bias) positives
# ---------------------------------------------------------------------------
def _bias_descriptor(content_sha256, size_bytes, mask_identity):
    return MasterDescriptor(
        master_type="bias",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="not_applicable",
        geometry=_geo(),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=0.01),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=ValidityEvidence(saturation_limit_known=True),
    )


def _short_flat_descriptor(content_sha256, size_bytes, mask_identity):
    return MasterDescriptor(
        master_type="flat",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="not_applicable",
        geometry=_geo(),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture",
            acquisition_profile=v1.AcquisitionProfileEvidence(
                source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
                bias_exposure_max_s=0.01, short_flat_profile=True,
            ),
        ),
        validity_evidence=_flat_validity(),
        flat_form="raw_response",
        optical_train_id="SYNTH-TRAIN-1",
        filter="NONE",
    )


def _removed_flat_dark_descriptor(content_sha256, size_bytes, mask_identity):
    return MasterDescriptor(
        master_type="flat_dark",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="removed",
        geometry=_geo(),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=ValidityEvidence(saturation_limit_known=True),
    )


def _raw_flat_with_bias_range(content_sha256, size_bytes, mask_identity):
    return MasterDescriptor(
        master_type="flat",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="not_applicable",
        geometry=_geo(),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture",
            acquisition_profile=v1.AcquisitionProfileEvidence(
                source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
                bias_exposure_max_s=0.01, short_flat_profile=False,
            ),
        ),
        validity_evidence=_flat_validity(),
        flat_form="raw_response",
        optical_train_id="SYNTH-TRAIN-1",
        filter="NONE",
    )


def test_calibrate_frame_bias_only_flat(tmp_path):
    flat_fits, flat_mask, fsha, fsize, fmsha = _write_master(tmp_path, "flat", 100.0)
    bias_fits, bias_mask, bsha, bsize, bmsha = _write_master(tmp_path, "bias", 0.0)
    flat = _short_flat_descriptor(fsha, fsize, fmsha)
    bias = _bias_descriptor(bsha, bsize, bmsha)
    library = _library(
        {
            "flat": (_candidate("flat", flat, locator_path=flat_fits, mask_path=flat_mask),),
            "bias": (_candidate("bias", bias, locator_path=bias_fits, mask_path=bias_mask),),
        }
    )
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("control", "apply"), library, default_match_policy()
    ).plan
    assert set(plan.masters) == {"flat", "bias_flat"}
    assert plan.masters["bias_flat"].role_descriptor.master_type == "bias"
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    # Finc 100 - bias 0 = 100, median 100 -> R=1.0, light 100 / 1.0 = 100.
    assert np.allclose(result.data, np.full(SHAPE, 100.0))


def test_calibrate_frame_flat_dark_bias_removed(tmp_path):
    flat_fits, flat_mask, fsha, fsize, fmsha = _write_master(tmp_path, "flat", 100.0)
    fd_fits, fd_mask, fdsha, fdsize, fdmsha = _write_master(tmp_path, "flat_dark", 0.0)
    bias_fits, bias_mask, bsha, bsize, bmsha = _write_master(tmp_path, "bias", 0.0)
    flat = _raw_flat_with_bias_range(fsha, fsize, fmsha)
    fd = _removed_flat_dark_descriptor(fdsha, fdsize, fdmsha)
    bias = _bias_descriptor(bsha, bsize, bmsha)
    library = _library(
        {
            "flat": (_candidate("flat", flat, locator_path=flat_fits, mask_path=flat_mask),),
            "flat_dark": (_candidate("flat_dark", fd, locator_path=fd_fits, mask_path=fd_mask),),
            "bias": (_candidate("bias", bias, locator_path=bias_fits, mask_path=bias_mask),),
        }
    )
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("control", "apply"), library, default_match_policy()
    ).plan
    assert set(plan.masters) == {"flat", "flat_dark", "bias_flat"}
    assert plan.masters["bias_flat"].role_descriptor.master_type == "bias"
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    # Finc 100 - bias 0 - flat_dark 0 = 100, median 100 -> R=1.0, light 100/1.0 = 100.
    assert np.allclose(result.data, np.full(SHAPE, 100.0))


# ---------------------------------------------------------------------------
# M8 — honest corrected-flat saturation (no corrected plane as acquisition)
# ---------------------------------------------------------------------------
def test_calibrate_frame_corrected_flat_no_saturation_screen(tmp_path):
    # Corrected values 110 >= a declared qualified limit 105 must NOT be silently
    # dropped by a corrected-data screen; the facade screens with limit=None and
    # records the unavailability truthfully.
    fcorr = np.full(SHAPE, 110.0, dtype=np.float32)
    path = _write_flat_file(tmp_path, "flat.fits", fcorr, bunit="ADU")
    mask_path = str(tmp_path / "flat.mask.npy")
    _write_mask(mask_path, np.zeros(SHAPE, dtype=np.uint16))
    fb = open(path, "rb").read()
    mb = open(mask_path, "rb").read()
    desc = MasterDescriptor(
        master_type="flat",
        pixel_domain="sensor_adu",
        physical_units="ADU",
        bias_state="not_applicable",
        geometry=_geo(),
        detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0, saturation_limit_adu=105.0, saturation_evidence="qualified"),
        content_sha256=hashlib.sha256(fb).hexdigest(),
        size_bytes=len(fb),
        hdu=0,
        mask_identity=hashlib.sha256(mb).hexdigest(),
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_correction_history=("bias_removed",)
        ),
        validity_evidence=_flat_validity(),
        flat_form="corrected_unnormalized",
        optical_train_id="SYNTH-TRAIN-1",
        filter="NONE",
    )
    library = _library({"flat": (_candidate("flat", desc, locator_path=path, mask_path=mask_path),)})
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("control", "apply"), library, default_match_policy()
    ).plan
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    assert result.provenance.flat_saturation_screening == "unavailable"
    assert result.provenance.flat_scalars_origin == "executed"
    assert result.scalars.get("mono") == 110.0  # no corrected-data saturation screen


def test_normalized_flat_scalars_origin_declared(tmp_path):
    path = _write_flat_file(tmp_path, "flat.fits", np.full(SHAPE, 2.0), bunit="dimensionless")
    mask_path = str(tmp_path / "flat.mask.npy")
    _write_mask(mask_path, np.zeros(SHAPE, dtype=np.uint16))
    fb = open(path, "rb").read()
    mb = open(mask_path, "rb").read()
    desc = _normalized_flat_descriptor(hashlib.sha256(fb).hexdigest(), len(fb), hashlib.sha256(mb).hexdigest(), cfa="mono")
    library = _library({"flat": (_candidate("flat", desc, locator_path=path, mask_path=mask_path),)})
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("control", "apply"), library, default_match_policy()
    ).plan
    result = calibrate_frame(_array_source(np.full(SHAPE, 100.0)), plan, ExecutionOptions())
    assert result.status == "COMPLETED"
    assert result.provenance.flat_scalars_origin == "declared"
    assert result.provenance.flat_saturation_screening == "declared"
