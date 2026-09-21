"""P8-A2 Stage 1 — prepared-context equivalence and semantics (contract).

These tests pin the *single-slot prepared calibration context* introduced by
P8-A2 Stage 1:

1. oracle equality of ``calibrate_frame`` vs ``prepare + apply`` across the full
   additive/flat matrix (corrected_unnormalized and normalized_response flats);
2. per-bound-master decode-count regression (dark+flat ⇒ 2 decodes over 10
   frames, ``validate_plan`` still 10×);
3. failure-not-cached semantics (transient retried next frame; persistent yields
   identical per-frame ``MASTER_LOAD_FAILED``);
4. decode-failure precedence (``("decode",)`` even with a valid context);
5. immutability (frozen arrays, identical repeated applies, in-place write
   refusal);
6. read-only reuse proof across threads (correctness only — no parallelism is
   introduced in production code);
7. array-source equivalence (in-memory light).

No public symbol, no module-level state, no disk persistence is introduced.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
import zecalibrator.api.v1._io as _io
import zecalibrator.application.library as _alib
from zecalibrator.api.v1.calibration import (
    _apply_prepared_context,
    _calibrate_frame_impl,
    _prepare_calibration_context,
    _PreparedContextSlot,
)

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


def _optical():
    return v1.OpticalIdentity(filter="NONE", optical_train_id="SYNTH-TRAIN-1")


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


def _make_scenario(tmp_path, additive_mode, flat_mode, flat_form=None):
    """Build (plan, source) for one additive×flat combination."""
    metadata = _sensor_metadata()
    roles = []
    if additive_mode == "bias_only":
        roles.append(("bias", _bias_descriptor, 0.0, "ADU"))
    elif additive_mode == "dark_incl_bias":
        roles.append(("dark", lambda s, z, m: _dark_descriptor(s, z, m, bias_state="included"), 10.0, "ADU"))
    elif additive_mode == "dark_bias_removed":
        roles.append(("dark", lambda s, z, m: _dark_descriptor(s, z, m, bias_state="removed"), 10.0, "ADU"))
        roles.append(("bias", _bias_descriptor, 0.0, "ADU"))
    # control: no additive masters

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


# ---------------------------------------------------------------------------
# 1. Equivalence oracle (full matrix) + array-source coverage
# ---------------------------------------------------------------------------
MATRIX = [
    ("control", "none", None),
    ("bias_only", "none", None),
    ("dark_incl_bias", "none", None),
    ("dark_bias_removed", "none", None),
    ("control", "apply", "corrected_unnormalized"),
    ("control", "apply", "normalized_response"),
    ("bias_only", "apply", "corrected_unnormalized"),
    ("bias_only", "apply", "normalized_response"),
    ("dark_incl_bias", "apply", "corrected_unnormalized"),
    ("dark_incl_bias", "apply", "normalized_response"),
    ("dark_bias_removed", "apply", "corrected_unnormalized"),
    ("dark_bias_removed", "apply", "normalized_response"),
]


@pytest.mark.parametrize("additive_mode,flat_mode,flat_form", MATRIX)
def test_calibrate_frame_equals_prepare_apply(tmp_path, additive_mode, flat_mode, flat_form):
    plan, source = _make_scenario(tmp_path, additive_mode, flat_mode, flat_form)

    expected = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    actual = _apply_prepared_context(
        source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None
    )
    _assert_equivalent(expected, actual)
    assert expected.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")


def test_calibrate_frame_equals_prepare_apply_fits_light(tmp_path):
    # The same oracle through the FITS light decode path.
    plan, _ = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    light_path = tmp_path / "light.fits"
    _write_fits(light_path, np.full(SHAPE, 100.0, dtype=np.float32))
    roi = v1.RoiExtentEvidence(
        extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
    )
    source = v1.FitsFrameSource(path=str(light_path), declaration=_declaration(), roi_extent=roi)

    expected = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    actual = _apply_prepared_context(
        source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None
    )
    _assert_equivalent(expected, actual)


# ---------------------------------------------------------------------------
# 2. Per-bound-master decode-count regression (dark+flat ⇒ 2, validate_plan 10×)
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

    handle = v1.LibraryHandle(
        v1.LibrarySnapshot(
            revision="r1", schema_version="zecalibrator.library.v1",
            candidates={
                "dark": (_candidate("dark", dark, dark_fits, dark_mask),),
                "flat": (_candidate("flat", flat, flat_fits, flat_mask),),
            },
        )
    )
    return handle


def test_master_decode_count_regression(tmp_path, monkeypatch):
    handle = _dark_flat_library(tmp_path)
    request = v1.CalibrationRequest("dark_incl_bias", "apply")
    metadata = _sensor_metadata()
    frames = [_array_source(np.full(SHAPE, 100.0, dtype=np.float32), metadata=metadata) for _ in range(10)]

    decode_calls = []
    validate_calls = []
    real_decode = _io.decode_fits_from_bytes
    real_validate = _alib.validate_plan

    def counting_decode(*args, **kwargs):
        decode_calls.append(1)
        return real_decode(*args, **kwargs)

    def counting_validate(*args, **kwargs):
        validate_calls.append(1)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(_io, "decode_fits_from_bytes", counting_decode)
    monkeypatch.setattr(_alib, "validate_plan", counting_validate)

    items = list(v1.calibrate_batch(frames, request, handle, v1.default_match_policy()))
    assert len(items) == 10
    assert all(it.disposition in ("COMPLETED", "COMPLETED_WITH_WARNINGS") for it in items)
    # dark ×1 + flat ×1 (corrected_unnormalized flat is decoded via decode_fits_from_bytes)
    assert len(decode_calls) == 2
    assert len(validate_calls) == 10


# ---------------------------------------------------------------------------
# 3. Failure-not-cached semantics
# ---------------------------------------------------------------------------
def test_transient_master_failure_retried_next_frame(tmp_path, monkeypatch):
    plan, source = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    slot = _PreparedContextSlot()

    real_decode = _io.decode_fits_from_bytes
    state = {"calls": 0}

    def flaky_decode(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise ValueError("transient master decode failure")
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(_io, "decode_fits_from_bytes", flaky_decode)

    r1 = _calibrate_frame_impl(source, plan, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
    assert r1.status == "FAILED"
    assert r1.reason_code == "MASTER_LOAD_FAILED"
    assert r1.provenance.executed_processing == ("decode", "validate_plan", "load_masters")

    # Frame 2 re-prepares (the failure was never cached) and is unaffected.
    r2 = _calibrate_frame_impl(source, plan, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
    assert r2.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert r2.provenance.executed_processing == ("decode", "validate_plan", "load_masters", "calibrate")


def test_persistent_master_failure_identical(tmp_path, monkeypatch):
    plan, source = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    slot = _PreparedContextSlot()

    def persistent_decode(*args, **kwargs):
        raise ValueError("persistent master decode failure")

    monkeypatch.setattr(_io, "decode_fits_from_bytes", persistent_decode)

    results = [
        _calibrate_frame_impl(source, plan, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
        for _ in range(2)
    ]
    for r in results:
        assert r.status == "FAILED"
        assert r.reason_code == "MASTER_LOAD_FAILED"
        assert r.provenance.executed_processing == ("decode", "validate_plan", "load_masters")
    assert results[0].reason_code == results[1].reason_code
    assert results[0].warnings == results[1].warnings
    assert results[0].provenance.to_dict() == results[1].provenance.to_dict()


# ---------------------------------------------------------------------------
# 4. Decode-failure precedence
# ---------------------------------------------------------------------------
def test_light_decode_failure_precedes_context_reuse(tmp_path):
    plan, source = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    slot = _PreparedContextSlot()

    # Frame 1 prepares a valid context.
    r1 = _calibrate_frame_impl(source, plan, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
    assert r1.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert not slot.is_empty

    # A light whose decode fails (shape mismatch vs declared geometry) must still
    # fail as a decode failure, even though a valid context is available.
    bad = v1.ArrayFrameSource(
        data=np.full((2, 2), 100.0, dtype=np.float32),
        metadata=_sensor_metadata(),  # geometry (4, 4)
        identity=v1.ArrayInputIdentity(caller_logical_id="bad"),
    )
    r2 = _calibrate_frame_impl(bad, plan, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
    assert r2.status == "FAILED"
    assert r2.reason_code == "SOURCE_ERROR"
    assert r2.provenance.executed_processing == ("decode",)


# ---------------------------------------------------------------------------
# 5. Immutability
# ---------------------------------------------------------------------------
def test_context_immutability(tmp_path):
    plan, source = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None

    for role, fm in context.bindings.items():
        assert fm.frame.data.flags.writeable is False, role
        assert fm.frame.mask.flags.writeable is False, role
        with pytest.raises(ValueError):
            fm.frame.data[0, 0] = 1.0
        with pytest.raises(ValueError):
            fm.frame.mask[0, 0] = 1

    # Two applies (same order) produce identical results; applying a second time
    # does not mutate the shared context.
    a = _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
    b = _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
    _assert_equivalent(a, b)


# ---------------------------------------------------------------------------
# 6. Read-only reuse proof (threads)
# ---------------------------------------------------------------------------
def test_shared_context_reuse_across_threads(tmp_path):
    plan, source = _make_scenario(tmp_path, "dark_incl_bias", "apply", "corrected_unnormalized")
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None

    n = 4
    serial = [
        _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
        for _ in range(n)
    ]

    threaded = [None] * n

    def run(i):
        threaded[i] = _apply_prepared_context(
            source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None
        )

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(n):
        _assert_equivalent(serial[i], threaded[i])


# ---------------------------------------------------------------------------
# 8. Per-frame light evidence is preserved on reuse (regression: context.plan
#    leakage). Two lights sharing one plan_id (evidence excluded from identity)
#    must each record their OWN original-cards in provenance.
# ---------------------------------------------------------------------------
def _card(keyword, value, index=0):
    return v1.CardRecord(
        keyword=keyword, value=value, comment="", index=index, source="primary", raw=str(value),
    )


def test_reuse_preserves_per_frame_light_evidence(tmp_path):
    base_plan, _ = _make_scenario(tmp_path, "dark_incl_bias", "none")
    masters = dict(base_plan.masters)
    request = v1.CalibrationRequest("dark_incl_bias", "none")

    card_a = _card("DATE-OBS", "2022-11-20T19:02:34.892")
    card_b = _card("DATE-OBS", "2022-11-20T19:00:02.406")
    metadata_a = _sensor_metadata(cards=[card_a])
    metadata_b = _sensor_metadata(cards=[card_b])

    plan_a = _build_plan(request, masters, metadata_a)
    plan_b = _build_plan(request, masters, metadata_b)
    # Evidence is audit-only: same identity, different light header evidence.
    assert plan_a.plan_id == plan_b.plan_id

    source_a = _array_source(np.full(SHAPE, 100.0, dtype=np.float32), metadata=metadata_a)
    source_b = _array_source(np.full(SHAPE, 100.0, dtype=np.float32), metadata=metadata_b)

    slot = _PreparedContextSlot()
    ra = _calibrate_frame_impl(source_a, plan_a, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
    rb = _calibrate_frame_impl(source_b, plan_b, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
    assert ra.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert rb.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")

    ev_a = ra.provenance.plan.light_constraints.evidence.original_cards
    ev_b = rb.provenance.plan.light_constraints.evidence.original_cards
    assert [c.value for c in ev_a] == ["2022-11-20T19:02:34.892"]
    assert [c.value for c in ev_b] == ["2022-11-20T19:00:02.406"]
