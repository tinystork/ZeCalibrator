"""P8-A1 — prepared master forms: equivalence, direct-caller protection, parity.

These tests pin the plan-invariant prepared master forms introduced by P8-A1:

1. equivalence oracle (core + facade): the checked path
   (``calibrate_light(masters_prevalidated=False)``) and the prepared path
   (``masters_prevalidated=True`` on frozen float32/validated-mask forms)
   produce identical data/mask/counts/scalars/precision/status/warnings/
   reason_code AND identical ``provenance.to_dict()``;
2. direct-caller protection: arbitrary ``calibrate_light`` callers keep ALL
   existing checks — float64 input, integer input, non-2D input, a mask with
   reserved bits set, and an overflow-inducing input;
3. reserved-bit failure parity (a master mask with reserved bits fails
   identically in both paths) and overflow/ADDITIVE_INVALID bit parity;
4. no alias/mutation of caller arrays; prepared forms demonstrably read-only;
5. effective reuse + call-count witness: master conversion/validation once per
   plan, not per frame (instrumented counters on the real batch path).

No public symbol, no module-level state, no parallelism, and no change to the
precision instrumentation is introduced.
"""

from __future__ import annotations

import dataclasses
import hashlib
from types import MappingProxyType

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
import zecalibrator.api.v1.calibration as _cal
import zecalibrator.core.calibrate as _core
from zecalibrator.api.v1.calibration import (
    _apply_prepared_context,
    _build_prepared_master_forms,
    _calibrate_frame_impl,
    _prepare_calibration_context,
    _PreparedContextSlot,
    PreparedMasterForm,
)
from zecalibrator.application.executor import (
    CalibrationRequest as ExecutorCalibrationRequest,
    MasterBinding as ExecutorMasterBinding,
    execute_calibration,
)
from zecalibrator.core.calibrate import calibrate_light
from zecalibrator.core.dq import ADDITIVE_INVALID, INPUT_INVALID, validate_mask
from zecalibrator.core.errors import InvalidRequestError
from zecalibrator.io.raw_decoder import DecodedFrame

SHAPE = (4, 4)


# ---------------------------------------------------------------------------
# Synthetic fixture builders (public API)
# ---------------------------------------------------------------------------
def _geo():
    return v1.Geometry(
        shape=SHAPE, sensor_dimensions=SHAPE, binning=(1, 1),
        roi_origin=(0, 0), roi_extent=SHAPE, orientation="identity", cfa_phase="mono",
    )


def _detector():
    return v1.DetectorIdentity(detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA")


def _acquisition(exposure_s=10.0, **kw):
    base = dict(
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16",
        temperature_c=20.0, exposure_s=exposure_s, saturation_limit_adu=60000.0,
        saturation_evidence="qualified", bias_exposure_max_s=0.01, short_flat_profile=False,
    )
    base.update(kw)
    return v1.Acquisition(**base)


def _declaration(**kw):
    base = dict(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
        adc_mode="MODE_16", binning=(1, 1), sensor_dimensions=SHAPE,
        orientation="identity", cfa_phase="mono", roi_origin=(0, 0),
        exposure_s=10.0, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0, saturation_evidence="qualified",
    )
    base.update(kw)
    return v1.ImportDeclaration(**base)


def _sensor_metadata(exposure_s=10.0, cards=()):
    decl = _declaration(exposure_s=exposure_s)
    md = v1.build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=tuple(cards), declaration=decl, units="ADU"
    )
    geo = v1.Geometry(
        shape=md.geometry.shape, sensor_dimensions=md.geometry.sensor_dimensions,
        binning=md.geometry.binning, roi_origin=md.geometry.roi_origin,
        roi_extent=SHAPE, orientation=md.geometry.orientation, cfa_phase=md.geometry.cfa_phase,
    )
    return v1.SensorMetadata(
        original_cards=md.original_cards, normalized=md.normalized, conflicts=md.conflicts,
        geometry=geo, raw_domain_declaration=md.raw_domain_declaration, units=md.units,
        declaration=md.declaration, exposure_s=md.exposure_s, temperature_c=md.temperature_c,
        gain=md.gain, offset=md.offset, readout_mode=md.readout_mode, adc_mode=md.adc_mode,
        filter=md.filter, detector_model=md.detector_model,
        detector_instance_id=md.detector_instance_id, optical_train_id=md.optical_train_id,
        saturation_limit_adu=md.saturation_limit_adu,
        saturation_evidence=md.saturation_evidence, warnings=md.warnings,
    )


def _write_fits(path, data, bunit="ADU"):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    if bunit:
        hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)
    return str(path)


def _write_mask(path, mask):
    np.save(path, np.asarray(mask, dtype=np.uint16), allow_pickle=False)
    return str(path)


def _write_master(tmp_path, name, value, bunit="ADU", mask=None):
    fits_path = tmp_path / f"{name}.fits"
    _write_fits(fits_path, np.full(SHAPE, value, dtype=np.float32), bunit=bunit)
    fits_bytes = fits_path.read_bytes()
    mask = np.zeros(SHAPE, dtype=np.uint16) if mask is None else np.asarray(mask, dtype=np.uint16)
    mask_path = tmp_path / f"{name}.mask.npy"
    _write_mask(mask_path, mask)
    mask_bytes = mask_path.read_bytes()
    return (
        str(fits_path), str(mask_path),
        hashlib.sha256(fits_bytes).hexdigest(), len(fits_bytes),
        hashlib.sha256(mask_bytes).hexdigest(),
    )


def _flat_validity():
    return v1.ValidityEvidence(
        saturation_limit_known=True,
        valid_normalization_count={"mono": 95},
        total_normalization_count={"mono": 100},
        quality_policy_state="qualified", illumination="flat_field", exposure_quality="qualified",
    )


def _dark_descriptor(sha, size, mask_sha, bias_state="included", exposure_s=10.0):
    return v1.MasterDescriptor(
        master_type="dark", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state=bias_state, geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=exposure_s),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=v1.ValidityEvidence(saturation_limit_known=True),
    )


def _bias_descriptor(sha, size, mask_sha):
    return v1.MasterDescriptor(
        master_type="bias", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state="not_applicable", geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=0.01),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=v1.ValidityEvidence(saturation_limit_known=True),
    )


def _corrected_flat_descriptor(sha, size, mask_sha):
    return v1.MasterDescriptor(
        master_type="flat", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state="not_applicable", geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            additive_correction_history=("bias_removed",),
        ),
        validity_evidence=_flat_validity(),
        flat_form="corrected_unnormalized", optical_train_id="SYNTH-TRAIN-1", filter="NONE",
    )


def _normalized_flat_descriptor(sha, size, mask_sha):
    scalars = v1.NormalizationScalars(mono=2.0)
    return v1.MasterDescriptor(
        master_type="flat", pixel_domain="normalized_response", physical_units="dimensionless",
        bias_state="not_applicable", geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            additive_correction_history=("bias_removed",),
            normalization=v1.NormalizationProvenance(
                algorithm="median", population="mono-valid", scalars=scalars,
            ),
        ),
        validity_evidence=_flat_validity(),
        flat_form="normalized_response", normalization_algorithm="median",
        normalization_scalars=scalars, optical_train_id="SYNTH-TRAIN-1", filter="NONE",
    )


def _binding(desc, fits_path, mask_path):
    return v1.MasterBinding(
        descriptor_id=desc.descriptor_id,
        descriptor_snapshot=v1.DescriptorSnapshot(desc),
        content_sha256=desc.content_sha256,
        size_bytes=desc.size_bytes,
        hdu=desc.hdu,
        mask_identity=desc.mask_identity,
        locators=(v1.FitsFileLocator(path=fits_path, hdu=desc.hdu),),
        mask_locator=v1.MaskPayloadLocator(path=mask_path),
    )


def _build_plan(request, masters, metadata):
    policy = v1.default_match_policy()
    return v1.CalibrationPlan.build(
        request=request,
        light_constraints=v1.light_constraints_from_sensor_metadata(metadata),
        masters=masters,
        policy_parameters=v1.PolicyParameters(
            exposure_tolerance=policy.exposure_tolerance,
            temperature_tolerance=policy.temperature_tolerance,
            flat_quality_policy=policy.flat_quality_policy,
        ),
        versions=v1.VersionSet(),
    )


def _array_source(data, metadata=None):
    return v1.ArrayFrameSource(
        data=np.asarray(data, dtype=np.float32),
        metadata=metadata if metadata is not None else _sensor_metadata(),
        identity=v1.ArrayInputIdentity(caller_logical_id="light-1"),
    )


def _make_scenario(tmp_path, additive_mode, flat_mode="none", flat_form=None):
    """Build (plan, source) for one additive x flat combination."""
    metadata = _sensor_metadata()
    roles = []
    if additive_mode == "bias_only":
        roles.append(("bias", _bias_descriptor, 4.0, "ADU"))
    elif additive_mode == "dark_incl_bias":
        roles.append(("dark", lambda s, z, m: _dark_descriptor(s, z, m, bias_state="included"), 10.0, "ADU"))
    elif additive_mode == "dark_bias_removed":
        roles.append(("dark", lambda s, z, m: _dark_descriptor(s, z, m, bias_state="removed"), 10.0, "ADU"))
        roles.append(("bias", _bias_descriptor, 4.0, "ADU"))

    if flat_mode == "apply":
        if flat_form == "corrected_unnormalized":
            roles.append(("flat", _corrected_flat_descriptor, 100.0, "ADU"))
        elif flat_form == "normalized_response":
            roles.append(("flat", _normalized_flat_descriptor, 2.0, "dimensionless"))
        else:  # pragma: no cover - parametrization guard
            raise ValueError(flat_form)

    masters = {}
    for role, factory, value, bunit in roles:
        fits_path, mask_path, sha, size, mask_sha = _write_master(tmp_path, role, value, bunit=bunit)
        masters[role] = _binding(factory(sha, size, mask_sha), fits_path, mask_path)

    plan = _build_plan(v1.CalibrationRequest(additive_mode, flat_mode), masters, metadata)
    source = _array_source(np.full(SHAPE, 100.0, dtype=np.float32), metadata=metadata)
    return plan, source


# ---------------------------------------------------------------------------
# Oracle equality helper
# ---------------------------------------------------------------------------
def _assert_equivalent(a, b):
    assert a.status == b.status
    assert a.reason_code == b.reason_code
    assert a.warnings == b.warnings
    assert a.scalars == b.scalars
    if a.data is None:
        assert b.data is None
    else:
        assert np.array_equal(a.data, b.data, equal_nan=True)
        assert np.array_equal(a.mask, b.mask)
    if a.counts is None:
        assert b.counts is None
    else:
        assert a.counts.total == b.counts.total
        assert a.counts.valid_count == b.counts.valid_count
        assert a.counts.invalid_count == b.counts.invalid_count
        assert dict(a.counts.per_bit) == dict(b.counts.per_bit)
    if a.precision is None:
        assert b.precision is None
    else:
        assert a.precision == b.precision
    assert a.provenance.to_dict() == b.provenance.to_dict()


def _frozen_f32(value, shape=SHAPE):
    arr = np.full(shape, value, dtype=np.float32)
    arr.flags.writeable = False
    return arr


def _frozen_mask(mask, shape=SHAPE):
    arr = np.asarray(mask, dtype=np.uint16)
    arr.flags.writeable = False
    return arr


# ---------------------------------------------------------------------------
# 1. Equivalence oracle (core level)
# ---------------------------------------------------------------------------
ADDITIVE_MATRIX = ["control", "bias_only", "dark_incl_bias", "dark_bias_removed"]


@pytest.mark.parametrize("additive_mode", ADDITIVE_MATRIX)
@pytest.mark.parametrize("flat_mode,flat_form", [("none", None), ("apply", "corrected_unnormalized")])
def test_core_equivalence_old_vs_prepared(tmp_path, additive_mode, flat_mode, flat_form):
    """Old checked path vs prepared path produce identical results at core level."""
    plan, source = _make_scenario(tmp_path, additive_mode, flat_mode, flat_form)
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None

    # Build the checked-path inputs (float32 masters) and the prepared-path
    # inputs (frozen float32 masters + validated frozen masks) from the context.
    light = np.full(SHAPE, 100.0, dtype=np.float32)

    kw = {"additive_mode": additive_mode, "flat_mode": flat_mode, "input_mask": None}
    prep_kw = dict(kw)

    if "bias" in context.master_forms:
        form = context.master_forms["bias"]
        kw["bias"] = np.asarray(form.frame_f32)
        kw["bias_mask"] = np.asarray(form.mask_u16)
        prep_kw["bias"] = form.frame_f32
        prep_kw["bias_mask"] = form.mask_u16
    if "dark" in context.master_forms:
        form = context.master_forms["dark"]
        if additive_mode == "dark_incl_bias":
            kw["dark_inc"] = np.asarray(form.frame_f32)
            kw["dark_inc_mask"] = np.asarray(form.mask_u16)
            prep_kw["dark_inc"] = form.frame_f32
            prep_kw["dark_inc_mask"] = form.mask_u16
        elif additive_mode == "dark_bias_removed":
            kw["dark_removed"] = np.asarray(form.frame_f32)
            kw["dark_removed_mask"] = np.asarray(form.mask_u16)
            prep_kw["dark_removed"] = form.frame_f32
            prep_kw["dark_removed_mask"] = form.mask_u16

    if flat_mode == "apply":
        pf = context.prepared_flat
        assert pf is not None and pf.ok
        kw["flat_response"] = np.asarray(pf.flat_response)
        kw["flat_valid"] = np.asarray(pf.flat_valid)
        prep_kw["flat_response"] = pf.flat_response
        prep_kw["flat_valid"] = pf.flat_valid

    old = calibrate_light(light, **kw)
    new = calibrate_light(light, masters_prevalidated=True, **prep_kw)

    assert old.status == new.status
    assert old.reason_code == new.reason_code
    assert old.warnings == new.warnings
    assert old.scalars == new.scalars
    assert np.array_equal(old.data, new.data, equal_nan=True)
    assert np.array_equal(old.mask, new.mask)
    assert old.counts.total == new.counts.total
    assert old.counts.valid_count == new.counts.valid_count
    assert old.counts.invalid_count == new.counts.invalid_count
    assert dict(old.counts.per_bit) == dict(new.counts.per_bit)
    assert old.precision == new.precision


# ---------------------------------------------------------------------------
# 1b. Equivalence oracle (facade level, full provenance.to_dict())
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("additive_mode", ADDITIVE_MATRIX)
@pytest.mark.parametrize("flat_mode,flat_form", [("none", None), ("apply", "corrected_unnormalized")])
def test_facade_equivalence_old_vs_prepared(tmp_path, additive_mode, flat_mode, flat_form):
    """Prepared path (master_forms) == old checked path (master_forms stripped)."""
    plan, source = _make_scenario(tmp_path, additive_mode, flat_mode, flat_form)
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None

    stripped = dataclasses.replace(context, master_forms=None)
    old = _apply_prepared_context(source, stripped, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
    new = _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)

    _assert_equivalent(old, new)
    assert old.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")


# ---------------------------------------------------------------------------
# 2. Direct-caller protection (critical)
# ---------------------------------------------------------------------------
def test_direct_caller_float64_input_keeps_checks():
    # float64 light + masters: conversion/overflow detection still runs.
    L = np.full(SHAPE, 100.0, dtype=np.float64)
    dark = np.full(SHAPE, 10.0, dtype=np.float64)
    r = calibrate_light(L, additive_mode="dark_incl_bias", dark_inc=dark)
    assert r.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert r.data.dtype == np.float32
    assert np.allclose(r.data, np.full(SHAPE, 90.0, dtype=np.float32))


def test_direct_caller_integer_input_keeps_checks():
    L = np.full(SHAPE, 100, dtype=np.int32)
    dark = np.full(SHAPE, 10, dtype=np.int32)
    r = calibrate_light(L, additive_mode="dark_incl_bias", dark_inc=dark)
    assert r.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert r.data.dtype == np.float32
    assert np.allclose(r.data, np.full(SHAPE, 90.0, dtype=np.float32))


def test_direct_caller_non_2d_input_rejected():
    L = np.full((4,), 100.0, dtype=np.float32)
    with pytest.raises(InvalidRequestError):
        calibrate_light(L, additive_mode="control")


def test_direct_caller_non_2d_master_rejected():
    L = np.full(SHAPE, 100.0, dtype=np.float32)
    dark = np.full((4,), 10.0, dtype=np.float32)
    with pytest.raises(InvalidRequestError):
        calibrate_light(L, additive_mode="dark_incl_bias", dark_inc=dark)


def test_direct_caller_reserved_bit_mask_rejected():
    L = np.full(SHAPE, 100.0, dtype=np.float32)
    dark = np.full(SHAPE, 10.0, dtype=np.float32)
    reserved = np.zeros(SHAPE, dtype=np.uint16)
    reserved[0, 0] = 0x0020  # reserved bit 5
    with pytest.raises(ValueError, match="reserved bits"):
        calibrate_light(L, additive_mode="dark_incl_bias", dark_inc=dark, dark_inc_mask=reserved)


def test_direct_caller_overflow_inducing_input_semantics_unchanged():
    # A finite float64 master magnitude that overflows float32 must be flagged
    # ADDITIVE_INVALID (never silently returned as Inf).
    L = np.full(SHAPE, 1.0, dtype=np.float64)
    huge = float(np.finfo(np.float32).max) * 10.0  # finite in float64, overflows float32
    dark = np.full(SHAPE, 10.0, dtype=np.float64)
    dark[0, 0] = huge
    r = calibrate_light(L, additive_mode="dark_incl_bias", dark_inc=dark)
    assert r.status in ("COMPLETED_WITH_WARNINGS", "FAILED")
    assert r.mask is not None
    assert dict(r.counts.per_bit).get("ADDITIVE_INVALID", 0) == 1
    # The overflowed pixel is flagged ADDITIVE_INVALID, never returned as Inf.
    assert r.mask[0, 0] & ADDITIVE_INVALID
    assert np.isfinite(r.data).sum() == (SHAPE[0] * SHAPE[1] - 1)


# ---------------------------------------------------------------------------
# 3. Reserved-bit failure parity + overflow/ADDITIVE_INVALID bit parity
# ---------------------------------------------------------------------------
def test_builder_refuses_non_float32_master_data():
    """S1: a violated float32 precondition cannot silently enter the fast path.

    ``_build_prepared_master_forms`` must fail loudly (never silently convert)
    when a master's ``data`` is not already float32 — the invariant the whole
    prepared fast path depends on.
    """
    frame = DecodedFrame(
        data=np.full(SHAPE, 10.0, dtype=np.float64),  # non-float32 on purpose
        mask=np.zeros(SHAPE, dtype=np.uint16),
        metadata=None,
        stored_dtype="float64",
        bscale=1.0,
        bzero=0.0,
        blank=None,
        hdu=0,
        precision=None,
    )
    masters = {"dark": ExecutorMasterBinding(role="dark", frame=frame, bias_state="included")}
    with pytest.raises(ValueError, match="must already be float32"):
        _build_prepared_master_forms(masters)

    # And the invariant still holds: a genuine float32 master builds a form.
    ok_frame = DecodedFrame(
        data=np.full(SHAPE, 10.0, dtype=np.float32),
        mask=np.zeros(SHAPE, dtype=np.uint16),
        metadata=None,
        stored_dtype="float32",
        bscale=1.0,
        bzero=0.0,
        blank=None,
        hdu=0,
        precision=None,
    )
    ok = {"dark": ExecutorMasterBinding(role="dark", frame=ok_frame, bias_state="included")}
    forms = _build_prepared_master_forms(ok)
    assert "dark" in forms
    assert forms["dark"].frame_f32.dtype == np.float32


def test_reserved_bit_failure_parity():
    """A master mask with reserved bits fails identically in both paths.

    Direct caller: ``calibrate_light`` raises via ``_master_mask``. Prepared
    path: the form builder raises via ``validate_mask`` (proven once, never a
    silent relaxation). Same ValueError.
    """
    reserved = np.zeros(SHAPE, dtype=np.uint16)
    reserved[0, 0] = 0x0040  # reserved bit 6

    with pytest.raises(ValueError, match="reserved bits") as direct:
        calibrate_light(
            np.full(SHAPE, 100.0, dtype=np.float32),
            additive_mode="dark_incl_bias",
            dark_inc=np.full(SHAPE, 10.0, dtype=np.float32),
            dark_inc_mask=reserved,
        )

    frame = DecodedFrame(
        data=np.full(SHAPE, 10.0, dtype=np.float32),
        mask=reserved,
        metadata=None,
        stored_dtype="float32",
        bscale=1.0,
        bzero=0.0,
        blank=None,
        hdu=0,
        precision=None,
    )
    masters = {"dark": ExecutorMasterBinding(role="dark", frame=frame, bias_state="included")}
    with pytest.raises(ValueError, match="reserved bits") as prep:
        _build_prepared_master_forms(masters)

    assert str(direct.value) == str(prep.value)


def test_overflow_additive_invalid_bit_parity():
    """Prepared path and checked path agree on ADDITIVE_INVALID for a partial mask."""
    light = np.full(SHAPE, 100.0, dtype=np.float32)
    dark = _frozen_f32(10.0)
    mask = np.zeros(SHAPE, dtype=np.uint16)
    mask[0, 0] = INPUT_INVALID  # one invalid dark pixel -> ADDITIVE_INVALID on light
    frozen_mask = _frozen_mask(mask)

    old = calibrate_light(
        light, additive_mode="dark_incl_bias",
        dark_inc=np.asarray(dark), dark_inc_mask=np.asarray(frozen_mask),
    )
    new = calibrate_light(
        light, additive_mode="dark_incl_bias",
        dark_inc=dark, dark_inc_mask=frozen_mask, masters_prevalidated=True,
    )

    assert np.array_equal(old.mask, new.mask)
    assert dict(old.counts.per_bit) == dict(new.counts.per_bit)
    assert dict(new.counts.per_bit)["ADDITIVE_INVALID"] == 1
    assert np.array_equal(old.data, new.data, equal_nan=True)


# ---------------------------------------------------------------------------
# 4. No alias/mutation + read-only prepared forms
# ---------------------------------------------------------------------------
def test_caller_arrays_never_mutated():
    L = np.full(SHAPE, 100.0, dtype=np.float32)
    dark = np.full(SHAPE, 10.0, dtype=np.float32)
    mask = np.zeros(SHAPE, dtype=np.uint16)
    mask[0, 0] = INPUT_INVALID
    L_copy, dark_copy, mask_copy = L.copy(), dark.copy(), mask.copy()

    calibrate_light(
        L, additive_mode="dark_incl_bias", dark_inc=dark, dark_inc_mask=mask,
    )
    assert np.array_equal(L, L_copy)
    assert np.array_equal(dark, dark_copy)
    assert np.array_equal(mask, mask_copy)


def test_prepared_forms_are_read_only(tmp_path):
    plan, source = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None

    assert set(context.master_forms) == {"dark"}
    form = context.master_forms["dark"]
    assert isinstance(form, PreparedMasterForm)
    assert form.invariants_proven is True
    assert form.frame_f32.dtype == np.float32
    assert form.mask_u16.dtype == np.uint16
    assert form.frame_f32.flags.writeable is False
    assert form.mask_u16.flags.writeable is False
    with pytest.raises(ValueError):
        form.frame_f32[0, 0] = 1.0
    with pytest.raises(ValueError):
        form.mask_u16[0, 0] = 1

    # The context mapping itself is read-only.
    with pytest.raises(TypeError):
        context.master_forms["bogus"] = form


def test_prepared_path_does_not_mutate_forms(tmp_path):
    plan, source = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    form = context.master_forms["dark"]
    frame_before = np.asarray(form.frame_f32).copy()
    mask_before = np.asarray(form.mask_u16).copy()

    for _ in range(3):
        r = _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
        assert r.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")

    assert np.array_equal(form.frame_f32, frame_before)
    assert np.array_equal(form.mask_u16, mask_before)


# ---------------------------------------------------------------------------
# 5. Effective reuse + call-count witness (real batch path)
# ---------------------------------------------------------------------------
def _dark_flat_library(tmp_path):
    dark_fits, dark_mask, dsha, dsize, dmsha = _write_master(tmp_path, "dark", 10.0)
    flat_fits, flat_mask, fsha, fsize, fmsha = _write_master(tmp_path, "flat", 100.0)
    dark = _dark_descriptor(dsha, dsize, dmsha)
    flat = _corrected_flat_descriptor(fsha, fsize, fmsha)

    def _candidate(cid, desc, p, mp):
        return v1.Candidate(
            candidate_id=cid, descriptor=desc, descriptor_snapshot=v1.DescriptorSnapshot(desc),
            locators=(v1.FitsFileLocator(path=p, hdu=desc.hdu),),
            mask_locator=v1.MaskPayloadLocator(path=mp),
        )

    return v1.LibraryHandle(
        v1.LibrarySnapshot(
            revision="r1", schema_version="zecalibrator.library.v1",
            candidates={
                "dark": (_candidate("dark", dark, dark_fits, dark_mask),),
                "flat": (_candidate("flat", flat, flat_fits, flat_mask),),
            },
        )
    )


def test_master_conversion_validation_once_per_plan(tmp_path, monkeypatch):
    """Master-side conversion/validation runs once per plan, not per frame."""
    plan, source = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    assert context.master_forms is not None and "dark" in context.master_forms

    f32_calls = {"light": 0, "dark_inc": 0, "flat_response": 0}
    master_mask_calls = {"input": 0, "dark_inc": 0}
    build_form_calls = []

    real_f32 = _core._f32_checked
    real_master_mask = _core._master_mask
    real_reuse = _core._reuse_prevalidated_mask
    real_build = _cal._build_prepared_master_forms

    def counting_f32(arr, name):
        f32_calls[name] = f32_calls.get(name, 0) + 1
        return real_f32(arr, name)

    def counting_master_mask(mask, shape, name):
        master_mask_calls[name] = master_mask_calls.get(name, 0) + 1
        return real_master_mask(mask, shape, name)

    def counting_build(masters):
        build_form_calls.append(1)
        return real_build(masters)

    monkeypatch.setattr(_core, "_f32_checked", counting_f32)
    monkeypatch.setattr(_core, "_master_mask", counting_master_mask)
    monkeypatch.setattr(_cal, "_build_prepared_master_forms", counting_build)

    slot = _PreparedContextSlot()
    results = [
        _calibrate_frame_impl(source, plan, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
        for _ in range(10)
    ]
    assert all(r.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS") for r in results)

    # Light-side conversion/validation stays per frame.
    assert f32_calls.get("light", 0) == 10
    assert master_mask_calls.get("input", 0) == 10
    # Master-side conversion is NOT per frame (skipped in the prepared path).
    assert f32_calls.get("dark_inc", 0) == 0
    # The prepared flat response is also not re-converted per frame.
    assert f32_calls.get("flat_response", 0) == 0
    # The per-master reserved-bit validation (``_master_mask``) is NOT per frame.
    assert master_mask_calls.get("dark_inc", 0) == 0
    # The prepared master forms are built exactly once per plan.
    assert len(build_form_calls) == 1


def test_batch_master_form_reuse_across_frames(tmp_path, monkeypatch):
    """The real ``calibrate_batch`` path builds forms once and reuses them."""
    handle = _dark_flat_library(tmp_path)
    request = v1.CalibrationRequest("dark_incl_bias", "apply")
    metadata = _sensor_metadata()
    frames = [_array_source(np.full(SHAPE, 100.0, dtype=np.float32), metadata=metadata) for _ in range(10)]

    build_form_calls = []
    real_build = _cal._build_prepared_master_forms

    def counting_build(masters):
        build_form_calls.append(1)
        return real_build(masters)

    monkeypatch.setattr(_cal, "_build_prepared_master_forms", counting_build)

    items = list(v1.calibrate_batch(frames, request, handle, v1.default_match_policy()))
    assert len(items) == 10
    assert all(it.disposition in ("COMPLETED", "COMPLETED_WITH_WARNINGS") for it in items)
    # Prepared once (single plan_id), never rebuilt per frame.
    assert len(build_form_calls) == 1


def test_no_public_symbol_and_no_module_state():
    assert "PreparedMasterForm" not in v1.__all__
    assert "_build_prepared_master_forms" not in v1.__all__
    assert not hasattr(v1, "PreparedMasterForm")
    assert not hasattr(v1, "_build_prepared_master_forms")

    module_state = {k: val for k, val in vars(_cal).items() if not k.startswith("__")}
    for k, val in module_state.items():
        assert not isinstance(val, PreparedMasterForm), f"module-level prepared form leak: {k}"
