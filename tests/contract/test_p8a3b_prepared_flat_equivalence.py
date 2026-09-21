"""P8-A3B — prepared flat response: equivalence and effective-reuse contract.

These tests pin the light-independent flat preparation captured once per plan
inside the existing ``PreparedCalibrationContext`` (``prepared_flat``):

1. equivalence oracle per prepared route (facade): ``calibrate_frame`` ==
   prepare+apply(prepared outcome) == prepare+apply(fallback ``prepared_flat=None``)
   — identical data/mask/counts/scalars/precision/status/warnings/reason_code AND
   identical ``provenance.to_dict()``;
2. effective-reuse proof per eligible route (owner-required): a poisoned
   per-frame ``_prepare_flat`` that would fail if called, asserting every frame
   still succeeds and results stay identical;
3. CFA geometry discrimination (mono / RGGB/GRBG/BGGR/GBRG; roi_origin parity;
   non-square and odd shapes);
4. failure parity + precedence (executor seam + facade end-to-end);
5. immutability / reuse (prepared arrays read-only, repeated frames identical,
   shared-context read-only reuse across threads);
6. call-count proof (``normalize_flat_response`` once per plan).

No public symbol, no module-level state, no production parallelism is introduced.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
import zecalibrator.api.v1._io as _io
import zecalibrator.application.library as _alib
import zecalibrator.application.executor as _exec
import zecalibrator.core.equations as _eq
import zecalibrator.core.geometry as _geo_mod
from zecalibrator.api.v1.calibration import (
    PreparedFlatOutcome,
    _apply_prepared_context,
    _build_prepared_flat,
    _calibrate_frame_impl,
    _prepare_calibration_context,
    _PreparedContextSlot,
)
from zecalibrator.application.executor import (
    CalibrationRequest as ExecutorCalibrationRequest,
    MasterBinding as ExecutorMasterBinding,
    execute_calibration,
)
from zecalibrator.core.errors import GeometryMismatchError, InvalidRequestError

SHAPE = (4, 4)

# The five prepared routes (``none``/control prepares nothing by design).
ROUTES = (
    "normalize_only",
    "already_normalized",
    "flat_dark_incl_bias",
    "flat_dark_bias_removed",
    "bias_only_flat",
)


# ---------------------------------------------------------------------------
# Synthetic fixture builders (public API)
# ---------------------------------------------------------------------------
def _geo(shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0)):
    return v1.Geometry(
        shape=shape, sensor_dimensions=shape, binning=(1, 1),
        roi_origin=roi_origin, roi_extent=shape, orientation="identity",
        cfa_phase=cfa_phase,
    )


def _detector(instance="SYNTH-DET-0001", model="SYNTH-CFA"):
    return v1.DetectorIdentity(detector_instance_id=instance, detector_model=model)


def _acquisition(exposure_s=10.0, **kw):
    base = dict(
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16",
        temperature_c=20.0, exposure_s=exposure_s, saturation_limit_adu=60000.0,
        saturation_evidence="qualified", bias_exposure_max_s=0.01, short_flat_profile=False,
    )
    base.update(kw)
    return v1.Acquisition(**base)


def _optical(filter="NONE", train="SYNTH-TRAIN-1"):
    return v1.OpticalIdentity(filter=filter, optical_train_id=train)


def _declaration(shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0), **kw):
    base = dict(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
        adc_mode="MODE_16", binning=(1, 1), sensor_dimensions=shape,
        orientation="identity", cfa_phase=cfa_phase, roi_origin=roi_origin,
        exposure_s=10.0, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0, saturation_evidence="qualified",
    )
    base.update(kw)
    return v1.ImportDeclaration(**base)


def _sensor_metadata(shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0), cards=(), **decl_kw):
    decl = _declaration(shape=shape, cfa_phase=cfa_phase, roi_origin=roi_origin, **decl_kw)
    md = v1.build_sensor_metadata(
        shape=shape, normalized={}, conflicts=(), cards=tuple(cards), declaration=decl, units="ADU"
    )
    geo = v1.Geometry(
        shape=md.geometry.shape, sensor_dimensions=md.geometry.sensor_dimensions,
        binning=md.geometry.binning, roi_origin=md.geometry.roi_origin,
        roi_extent=shape, orientation=md.geometry.orientation, cfa_phase=md.geometry.cfa_phase,
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


def _write_master(tmp_path, name, data, bunit="ADU", mask=None):
    import os
    os.makedirs(tmp_path, exist_ok=True)
    fits_path = tmp_path / f"{name}.fits"
    _write_fits(fits_path, data, bunit=bunit)
    fits_bytes = fits_path.read_bytes()
    mask = np.zeros(data.shape, dtype=np.uint16) if mask is None else np.asarray(mask, dtype=np.uint16)
    mask_path = tmp_path / f"{name}.mask.npy"
    _write_mask(mask_path, mask)
    mask_bytes = mask_path.read_bytes()
    return (
        str(fits_path), str(mask_path),
        hashlib.sha256(fits_bytes).hexdigest(), len(fits_bytes),
        hashlib.sha256(mask_bytes).hexdigest(),
    )


def _flat_validity(phase="mono"):
    if phase == "mono":
        vc, tc = {"mono": 95}, {"mono": 100}
    else:
        vc = {"G1": 95, "R": 95, "B": 95, "G2": 95}
        tc = {"G1": 100, "R": 100, "B": 100, "G2": 100}
    return v1.ValidityEvidence(
        saturation_limit_known=True,
        valid_normalization_count=vc,
        total_normalization_count=tc,
        quality_policy_state="qualified", illumination="flat_field", exposure_quality="qualified",
    )


def _profile(bias_exposure_max_s=0.01, short_flat_profile=False):
    return v1.AcquisitionProfileEvidence(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        bias_exposure_max_s=bias_exposure_max_s, short_flat_profile=short_flat_profile,
    )


def _corrected_flat_descriptor(sha, size, mask_sha, shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0)):
    return v1.MasterDescriptor(
        master_type="flat", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state="not_applicable", geometry=_geo(shape, cfa_phase, roi_origin),
        detector=_detector(), acquisition=_acquisition(exposure_s=1.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            additive_correction_history=("bias_removed",),
        ),
        validity_evidence=_flat_validity(phase=cfa_phase),
        flat_form="corrected_unnormalized", optical_train_id="SYNTH-TRAIN-1", filter="NONE",
    )


def _normalized_flat_descriptor(sha, size, mask_sha, shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0)):
    if cfa_phase == "mono":
        scalars = v1.NormalizationScalars(mono=2.0)
        population = "mono-valid"
        algorithm = "median"
    else:
        scalars = v1.NormalizationScalars(g1=2.0, r=2.0, b=2.0, g2=2.0)
        population = "cfa-4-plane"
        algorithm = "cfa-median-per-plane"
    return v1.MasterDescriptor(
        master_type="flat", pixel_domain="normalized_response", physical_units="dimensionless",
        bias_state="not_applicable", geometry=_geo(shape, cfa_phase, roi_origin),
        detector=_detector(), acquisition=_acquisition(exposure_s=1.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            additive_correction_history=("bias_removed",),
            normalization=v1.NormalizationProvenance(
                algorithm=algorithm, population=population, scalars=scalars,
            ),
        ),
        validity_evidence=_flat_validity(phase=cfa_phase),
        flat_form="normalized_response", normalization_algorithm=algorithm,
        normalization_scalars=scalars, optical_train_id="SYNTH-TRAIN-1", filter="NONE",
    )


def _raw_flat_descriptor(sha, size, mask_sha, shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0),
                         bias_exposure_max_s=0.01, short_flat_profile=False,
                         detector_instance="SYNTH-DET-0001", exposure_s=1.0):
    return v1.MasterDescriptor(
        master_type="flat", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state="not_applicable", geometry=_geo(shape, cfa_phase, roi_origin),
        detector=_detector(instance=detector_instance), acquisition=_acquisition(exposure_s=exposure_s),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            acquisition_profile=_profile(
                bias_exposure_max_s=bias_exposure_max_s, short_flat_profile=short_flat_profile,
            ),
        ),
        validity_evidence=_flat_validity(phase=cfa_phase),
        flat_form="raw_response", optical_train_id="SYNTH-TRAIN-1", filter="NONE",
    )


def _flat_dark_descriptor(sha, size, mask_sha, shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0),
                          bias_state="included", detector_instance="SYNTH-DET-0001", exposure_s=1.0):
    return v1.MasterDescriptor(
        master_type="flat_dark", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state=bias_state, geometry=_geo(shape, cfa_phase, roi_origin),
        detector=_detector(instance=detector_instance), acquisition=_acquisition(exposure_s=exposure_s),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture", additive_history_state="known"),
        validity_evidence=v1.ValidityEvidence(saturation_limit_known=True),
    )


def _bias_flat_descriptor(sha, size, mask_sha, shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0), exposure_s=0.01):
    return v1.MasterDescriptor(
        master_type="bias", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state="not_applicable", geometry=_geo(shape, cfa_phase, roi_origin),
        detector=_detector(), acquisition=_acquisition(exposure_s=exposure_s),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture", additive_history_state="known"),
        validity_evidence=v1.ValidityEvidence(saturation_limit_known=True),
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


def _flat_pattern(shape, cfa_phase, roi_origin):
    """Per-plane distinct flat values for CFA; constant for mono."""
    if cfa_phase == "mono":
        return np.full(shape, 100.0, dtype=np.float32)
    yy, xx = np.indices(shape)
    idx = ((yy + roi_origin[0]) % 2) * 2 + ((xx + roi_origin[1]) % 2)
    return (100.0 + 20.0 * idx).astype(np.float32)


def _make_scenario(tmp_path, route, shape=SHAPE, cfa_phase="mono", roi_origin=(0, 0), flat_data=None):
    """Build (plan, source) for one prepared flat route (additive_mode=control)."""
    metadata = _sensor_metadata(shape=shape, cfa_phase=cfa_phase, roi_origin=roi_origin)
    masters = {}

    if route == "already_normalized":
        data = np.full(shape, 2.0, dtype=np.float32) if flat_data is None else flat_data
        fp, mp, sha, size, msha = _write_master(tmp_path, "flat", data, bunit=None)
        masters["flat"] = _binding(_normalized_flat_descriptor(sha, size, msha, shape, cfa_phase, roi_origin), fp, mp)
    elif route == "normalize_only":
        data = _flat_pattern(shape, cfa_phase, roi_origin) if flat_data is None else flat_data
        fp, mp, sha, size, msha = _write_master(tmp_path, "flat", data, bunit="ADU")
        masters["flat"] = _binding(_corrected_flat_descriptor(sha, size, msha, shape, cfa_phase, roi_origin), fp, mp)
    elif route == "flat_dark_incl_bias":
        fdata = np.full(shape, 100.0, dtype=np.float32) if flat_data is None else flat_data
        fp, mp, sha, size, msha = _write_master(tmp_path, "flat", fdata, bunit="ADU")
        masters["flat"] = _binding(_raw_flat_descriptor(sha, size, msha, shape, cfa_phase, roi_origin), fp, mp)
        fdp, fdm, fdsha, fdsize, fdmsha = _write_master(tmp_path, "flat_dark", np.full(shape, 10.0, dtype=np.float32), bunit="ADU")
        masters["flat_dark"] = _binding(_flat_dark_descriptor(fdsha, fdsize, fdmsha, shape, cfa_phase, roi_origin, bias_state="included"), fdp, fdm)
    elif route == "flat_dark_bias_removed":
        fdata = np.full(shape, 100.0, dtype=np.float32) if flat_data is None else flat_data
        fp, mp, sha, size, msha = _write_master(tmp_path, "flat", fdata, bunit="ADU")
        masters["flat"] = _binding(_raw_flat_descriptor(sha, size, msha, shape, cfa_phase, roi_origin), fp, mp)
        fdp, fdm, fdsha, fdsize, fdmsha = _write_master(tmp_path, "flat_dark", np.full(shape, 10.0, dtype=np.float32), bunit="ADU")
        masters["flat_dark"] = _binding(_flat_dark_descriptor(fdsha, fdsize, fdmsha, shape, cfa_phase, roi_origin, bias_state="removed"), fdp, fdm)
        bfp, bfm, bfsha, bfsize, bfmsha = _write_master(tmp_path, "bias_flat", np.full(shape, 0.0, dtype=np.float32), bunit="ADU")
        masters["bias_flat"] = _binding(_bias_flat_descriptor(bfsha, bfsize, bfmsha, shape, cfa_phase, roi_origin), bfp, bfm)
    elif route == "bias_only_flat":
        fdata = np.full(shape, 100.0, dtype=np.float32) if flat_data is None else flat_data
        fp, mp, sha, size, msha = _write_master(tmp_path, "flat", fdata, bunit="ADU")
        masters["flat"] = _binding(_raw_flat_descriptor(sha, size, msha, shape, cfa_phase, roi_origin, short_flat_profile=True), fp, mp)
        bfp, bfm, bfsha, bfsize, bfmsha = _write_master(tmp_path, "bias_flat", np.full(shape, 0.0, dtype=np.float32), bunit="ADU")
        masters["bias_flat"] = _binding(_bias_flat_descriptor(bfsha, bfsize, bfmsha, shape, cfa_phase, roi_origin), bfp, bfm)
    else:  # pragma: no cover - parametrization guard
        raise ValueError(route)

    plan = _build_plan(v1.CalibrationRequest("control", "apply"), masters, metadata)
    source = _array_source(np.full(shape, 100.0, dtype=np.float32), metadata=metadata)
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


def _prepared_context(tmp_path, route, **kw):
    plan, _ = _make_scenario(tmp_path, route, **kw)
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    return plan, context


# ---------------------------------------------------------------------------
# 1. Equivalence oracle per route (facade)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("route", ROUTES)
def test_equivalence_oracle_per_route(tmp_path, route):
    plan, source = _make_scenario(tmp_path, route)

    expected = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    assert context.prepared_flat is not None
    assert context.prepared_flat.ok
    assert context.prepared_flat.flat_prep_mode == context.flat_prep_mode

    # Old per-frame path: the same context with ``prepared_flat=None`` forces the
    # executor to re-run ``_prepare_flat`` per frame (today's semantics).
    fallback = dataclasses.replace(context, prepared_flat=None)
    old = _apply_prepared_context(source, fallback, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
    new = _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)

    _assert_equivalent(expected, old)
    _assert_equivalent(expected, new)
    assert expected.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")


# ---------------------------------------------------------------------------
# 2. Effective-reuse proof per eligible route (owner-required)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("route", ROUTES)
def test_effective_reuse_per_route(tmp_path, route, monkeypatch):
    plan, source = _make_scenario(tmp_path, route)
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    assert context.prepared_flat is not None and context.prepared_flat.ok

    # Reference results BEFORE poisoning (per-frame path and prepared path agree).
    expected = _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)

    calls = []
    real_prepare_flat = _exec._prepare_flat

    def poison(*args, **kwargs):
        calls.append(1)
        raise AssertionError("per-frame _prepare_flat must not be called when the prepared outcome is consumed")

    monkeypatch.setattr(_exec, "_prepare_flat", poison)

    for _ in range(3):
        result = _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
        assert result.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
        _assert_equivalent(expected, result)

    assert calls == []  # zero per-frame preparation calls
    assert real_prepare_flat is not None


# ---------------------------------------------------------------------------
# 3. CFA geometry discrimination
# ---------------------------------------------------------------------------
def _expected_plane_totals(shape, cfa_phase, roi_origin):
    labels = _plane_labels(shape, cfa_phase, roi_origin)
    planes = ("mono",) if cfa_phase == "mono" else ("G1", "R", "B", "G2")
    return {p: int(np.count_nonzero(labels == p)) for p in planes}


def _plane_labels(shape, cfa_phase, roi_origin):
    if cfa_phase == "mono":
        return np.full(shape, "mono", dtype=object)
    return _geo_mod.plane_label_array(shape, cfa_phase, roi_origin)


@pytest.mark.parametrize("cfa_phase", ["mono", "RGGB", "GRBG", "BGGR", "GBRG"])
def test_cfa_geometry_discrimination(tmp_path, cfa_phase):
    shape = (6, 5)  # non-square, odd dimensions
    plan, _ = _make_scenario(tmp_path, "normalize_only", shape=shape, cfa_phase=cfa_phase, roi_origin=(0, 0))
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    pf = context.prepared_flat
    assert pf is not None and pf.ok
    norm = pf.norm

    assert norm.cfa_phase == cfa_phase
    assert norm.roi_origin == (0, 0)
    assert norm.usable

    planes = ("mono",) if cfa_phase == "mono" else ("G1", "R", "B", "G2")
    assert tuple(norm.plane_counts.keys()) == planes
    assert tuple(norm.plane_totals.keys()) == planes
    assert norm.plane_totals == _expected_plane_totals(shape, cfa_phase, (0, 0))

    # Per-plane scalars equal the flat's per-plane median (100 + 20*idx for CFA).
    yy, xx = np.indices(shape)
    idx = (yy % 2) * 2 + (xx % 2)
    labels = _plane_labels(shape, cfa_phase, (0, 0))
    for plane in planes:
        plane_sel = labels == plane
        expected_median = float(np.median(_flat_pattern(shape, cfa_phase, (0, 0))[plane_sel].astype(np.float64)))
        assert norm.scalars[plane] == expected_median
        # The prepared response equals flat / scalar per plane (float32 division).
        r = pf.flat_response[plane_sel]
        assert np.allclose(r, _flat_pattern(shape, cfa_phase, (0, 0))[plane_sel] / np.float32(norm.scalars[plane]))


def test_roi_origin_parity_shift(tmp_path):
    shape = (5, 3)
    plan_a, _ = _make_scenario(tmp_path / "a", "normalize_only", shape=shape, cfa_phase="RGGB", roi_origin=(0, 0))
    plan_b, _ = _make_scenario(tmp_path / "b", "normalize_only", shape=shape, cfa_phase="RGGB", roi_origin=(1, 0))

    ctx_a, _ = _prepare_calibration_context(plan_a, token=v1.CancellationToken())
    ctx_b, _ = _prepare_calibration_context(plan_b, token=v1.CancellationToken())

    totals_a = ctx_a.prepared_flat.norm.plane_totals
    totals_b = ctx_b.prepared_flat.norm.plane_totals

    # Ground truth for the parity flip (roi (1,0) shifts the CFA phase by one row).
    assert totals_a == _expected_plane_totals(shape, "RGGB", (0, 0))
    assert totals_b == _expected_plane_totals(shape, "RGGB", (1, 0))
    assert totals_a != totals_b  # the two ROI origins label different planes
    assert set(totals_a.values()) == set(totals_b.values())  # same multiset, reassigned


# ---------------------------------------------------------------------------
# 4. Failure parity + precedence
# ---------------------------------------------------------------------------
def _raised(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


def _flat_frames(make_frame):
    light = make_frame((4, 4), value=100.0)
    raw_flat = make_frame((4, 4), value=100.0, exposure_s=1.0)
    norm_flat = make_frame((4, 4), value=2.0, exposure_s=1.0, units="dimensionless")
    fd_incl = make_frame((4, 4), value=10.0, exposure_s=1.0)
    fd_removed = make_frame((4, 4), value=10.0, exposure_s=1.0)
    fd_mismatch = make_frame((4, 4), value=10.0, exposure_s=1.0, detector_instance_id="OTHER-DET")
    return light, raw_flat, norm_flat, fd_incl, fd_removed, fd_mismatch


def _assert_seam_parity(light, request, masters, flat_prep_mode):
    old_exc = _raised(lambda: execute_calibration(
        light, request, masters, flat_prep_mode=flat_prep_mode, prepared_flat_outcome=None,
    ))
    assert old_exc is not None
    if isinstance(old_exc, GeometryMismatchError):
        failure_args = (old_exc.reason_code, old_exc.fields)
    else:
        failure_args = old_exc.args
    outcome = PreparedFlatOutcome(
        flat_prep_mode=flat_prep_mode, ok=False, flat_response=None, flat_valid=None,
        scalars={}, norm=None, failure_type=type(old_exc), failure_args=failure_args,
    )
    new_exc = _raised(lambda: execute_calibration(
        light, request, masters, flat_prep_mode=flat_prep_mode, prepared_flat_outcome=outcome,
    ))
    assert new_exc is not None
    assert type(new_exc) is type(old_exc)
    assert new_exc.args == old_exc.args
    if isinstance(old_exc, GeometryMismatchError):
        assert new_exc.reason_code == old_exc.reason_code
        assert new_exc.fields == old_exc.fields
    return old_exc


@pytest.mark.parametrize(
    "masters_factory,flat_prep_mode,exc_type",
    [
        # InvalidRequestError cases
        ("flat_form_mismatch_already", "already_normalized", InvalidRequestError),
        ("missing_proof", "already_normalized", InvalidRequestError),
        ("unsupported_mode", "bogus", InvalidRequestError),
        ("missing_flat_dark", "flat_dark_incl_bias", InvalidRequestError),
        # GeometryMismatchError cases
        ("role_unavailable", "flat_dark_incl_bias", GeometryMismatchError),
        ("flat_dark_binding_mismatch", "flat_dark_incl_bias", GeometryMismatchError),
    ],
)
def test_seam_failure_parity(make_frame_fixture, masters_factory, flat_prep_mode, exc_type):
    make_frame = make_frame_fixture
    light, raw_flat, norm_flat, fd_incl, fd_removed, fd_mismatch = _flat_frames(make_frame)
    flat = ExecutorMasterBinding("flat", raw_flat, flat_form="raw_response")
    flat_norm = ExecutorMasterBinding("flat", norm_flat, flat_form="normalized_response")
    fd_incl_b = ExecutorMasterBinding("flat_dark", fd_incl, bias_state="included")
    fd_removed_b = ExecutorMasterBinding("flat_dark", fd_removed, bias_state="removed")
    fd_mismatch_b = ExecutorMasterBinding("flat_dark", fd_mismatch, bias_state="included")

    cases = {
        "flat_form_mismatch_already": ({"flat": flat},),
        "missing_proof": ({"flat": flat_norm},),
        "unsupported_mode": ({"flat": flat},),
        "missing_flat_dark": ({"flat": flat},),
        "role_unavailable": ({"flat": flat, "flat_dark": fd_removed_b},),
        "flat_dark_binding_mismatch": ({"flat": flat, "flat_dark": fd_mismatch_b},),
    }
    masters = cases[masters_factory][0]

    exc = _assert_seam_parity(
        light, ExecutorCalibrationRequest("control", "apply"), masters, flat_prep_mode,
    )
    assert isinstance(exc, exc_type)


def test_facade_failure_parity_end_to_end(tmp_path):
    # raw flat + flat_dark with a mismatched detector -> GeometryMismatchError,
    # captured in the prepared context and re-emitted per frame identically.
    shape = SHAPE
    metadata = _sensor_metadata()
    fp, mp, sha, size, msha = _write_master(tmp_path, "flat", np.full(shape, 100.0, dtype=np.float32))
    fdp, fdm, fdsha, fdsize, fdmsha = _write_master(tmp_path, "flat_dark", np.full(shape, 10.0, dtype=np.float32))
    masters = {
        "flat": _binding(_raw_flat_descriptor(sha, size, msha), fp, mp),
        "flat_dark": _binding(
            _flat_dark_descriptor(fdsha, fdsize, fdmsha, bias_state="included", detector_instance="OTHER-DET"),
            fdp, fdm,
        ),
    }
    plan = _build_plan(v1.CalibrationRequest("control", "apply"), masters, metadata)
    source = _array_source(np.full(shape, 100.0, dtype=np.float32), metadata=metadata)

    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    assert context.prepared_flat is not None
    assert not context.prepared_flat.ok
    assert context.prepared_flat.failure_type is GeometryMismatchError

    via_calibrate = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    via_prepared = _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
    fallback = dataclasses.replace(context, prepared_flat=None)
    via_fallback = _apply_prepared_context(source, fallback, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)

    for r in (via_calibrate, via_prepared, via_fallback):
        assert r.status == "FAILED"
        assert r.reason_code == "DETECTOR_MISMATCH: detector.detector_instance_id"
    _assert_equivalent(via_calibrate, via_prepared)
    _assert_equivalent(via_calibrate, via_fallback)


def test_facade_invalid_request_failure_end_to_end(tmp_path):
    # normalized_response flat whose proof population contradicts the CFA phase
    # -> InvalidRequestError, captured and re-raised identically (never MASTER_LOAD_FAILED).
    shape = SHAPE
    metadata = _sensor_metadata()
    fp, mp, sha, size, msha = _write_master(tmp_path, "flat", np.full(shape, 2.0, dtype=np.float32), bunit=None)
    # mono descriptor but a cfa-4-plane normalization proof -> incoherent.
    scalars = v1.NormalizationScalars(g1=2.0, r=2.0, b=2.0, g2=2.0)
    desc = v1.MasterDescriptor(
        master_type="flat", pixel_domain="normalized_response", physical_units="dimensionless",
        bias_state="not_applicable", geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=msha,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            normalization=v1.NormalizationProvenance(algorithm="cfa-median-per-plane", population="cfa-4-plane", scalars=scalars),
        ),
        validity_evidence=_flat_validity(phase="mono"),
        flat_form="normalized_response", normalization_algorithm="cfa-median-per-plane",
        normalization_scalars=scalars, optical_train_id="SYNTH-TRAIN-1", filter="NONE",
    )
    masters = {"flat": _binding(desc, fp, mp)}
    plan = _build_plan(v1.CalibrationRequest("control", "apply"), masters, metadata)
    source = _array_source(np.full(shape, 100.0, dtype=np.float32), metadata=metadata)

    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    assert context.prepared_flat is not None
    assert not context.prepared_flat.ok
    assert context.prepared_flat.failure_type is InvalidRequestError

    with pytest.raises(InvalidRequestError) as ei:
        v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    assert "mono flat requires population='mono-valid'" in str(ei.value)

    with pytest.raises(InvalidRequestError) as ei2:
        _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
    assert str(ei2.value) == str(ei.value)


def test_flat_unusable_per_frame_and_precedence(make_frame_fixture):
    # Executor-level: an unusable prepared flat still reports FLAT_UNUSABLE per
    # frame, and a light failing the light<->flat binding still reports THAT
    # failure (never FLAT_UNUSABLE) even though a prepared outcome exists.
    make_frame = make_frame_fixture
    light_good = make_frame((4, 4), value=100.0)
    light_bad = make_frame((4, 4), value=100.0, filter="RED")
    flat = make_frame((4, 4), value=0.0, exposure_s=1.0)  # all-zero -> unusable
    masters = {"flat": ExecutorMasterBinding("flat", flat, flat_form="corrected_unnormalized")}

    flat_response, flat_valid, scalars, norm = _exec._prepare_flat(masters["flat"], masters, "normalize_only")
    assert norm is not None and not norm.usable
    outcome = PreparedFlatOutcome(
        flat_prep_mode="normalize_only", ok=True, flat_response=flat_response,
        flat_valid=flat_valid, scalars=scalars, norm=norm, failure_type=None, failure_args=None,
    )

    # Good light -> FLAT_UNUSABLE (per-frame result with the light's evidence).
    r = execute_calibration(
        light_good, ExecutorCalibrationRequest("control", "apply"), masters,
        flat_prep_mode="normalize_only", prepared_flat_outcome=outcome,
    )
    assert r.status == "FAILED"
    assert r.reason_code == "FLAT_UNUSABLE"
    assert r.frame_quality.saturation_evidence == light_good.metadata.saturation_evidence

    # Bad light -> binding failure wins, prepared unusable never reached.
    exc = _raised(lambda: execute_calibration(
        light_bad, ExecutorCalibrationRequest("control", "apply"), masters,
        flat_prep_mode="normalize_only", prepared_flat_outcome=outcome,
    ))
    assert isinstance(exc, GeometryMismatchError)
    assert exc.reason_code == "FILTER_MISMATCH: optical.filter"


def test_precedence_binding_beats_prepared_failure(make_frame_fixture):
    make_frame = make_frame_fixture
    light_bad = make_frame((4, 4), value=100.0, filter="RED")
    flat = make_frame((4, 4), value=100.0, exposure_s=1.0)
    masters = {"flat": ExecutorMasterBinding("flat", flat, flat_form="raw_response")}

    # A prepared FAILURE exists, but the light's own flat binding fails first.
    outcome = PreparedFlatOutcome(
        flat_prep_mode="flat_dark_incl_bias", ok=False, flat_response=None, flat_valid=None,
        scalars={}, norm=None, failure_type=GeometryMismatchError,
        failure_args=("ROLE_UNAVAILABLE", ("flat_dark requires bias_state='included', got 'removed'",)),
    )
    exc = _raised(lambda: execute_calibration(
        light_bad, ExecutorCalibrationRequest("control", "apply"), masters,
        flat_prep_mode="flat_dark_incl_bias", prepared_flat_outcome=outcome,
    ))
    assert isinstance(exc, GeometryMismatchError)
    assert exc.reason_code == "FILTER_MISMATCH: optical.filter"


# ---------------------------------------------------------------------------
# 5. Immutability / reuse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("route", ROUTES)
def test_prepared_arrays_immutable(tmp_path, route):
    plan, context = _prepared_context(tmp_path, route)
    pf = context.prepared_flat
    assert pf is not None and pf.ok
    assert pf.flat_response.flags.writeable is False
    assert pf.flat_valid.flags.writeable is False
    with pytest.raises(ValueError):
        pf.flat_response[0, 0] = 1.0
    with pytest.raises(ValueError):
        pf.flat_valid[0, 0] = True
    # norm.R / norm.valid are the same frozen arrays (route-dependent).
    if pf.norm is not None:
        assert pf.norm.R.flags.writeable is False
        assert pf.norm.valid.flags.writeable is False


def test_repeated_frames_identical(tmp_path):
    plan, source = _make_scenario(tmp_path, "normalize_only")
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    results = [
        _apply_prepared_context(source, context, v1.ExecutionOptions(), token=v1.CancellationToken(), progress=None)
        for _ in range(5)
    ]
    for r in results:
        _assert_equivalent(results[0], r)


def test_shared_context_reuse_across_threads(tmp_path):
    plan, source = _make_scenario(tmp_path, "normalize_only")
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
# 6. Call-count proof: normalize_flat_response once per plan
# ---------------------------------------------------------------------------
def test_normalize_flat_response_once_per_plan(tmp_path, monkeypatch):
    plan, source = _make_scenario(tmp_path, "normalize_only", shape=(6, 5), cfa_phase="RGGB")

    norm_calls = []
    label_calls = []
    index_calls = []
    real_norm = _exec.normalize_flat_response
    real_label = _eq.plane_label_array
    real_index = _geo_mod.plane_index_array

    def counting_norm(*a, **kw):
        norm_calls.append(1)
        return real_norm(*a, **kw)

    def counting_label(*a, **kw):
        label_calls.append(1)
        return real_label(*a, **kw)

    def counting_index(*a, **kw):
        index_calls.append(1)
        return real_index(*a, **kw)

    monkeypatch.setattr(_exec, "normalize_flat_response", counting_norm)
    monkeypatch.setattr(_eq, "plane_label_array", counting_label)
    monkeypatch.setattr(_geo_mod, "plane_index_array", counting_index)

    slot = _PreparedContextSlot()
    results = [
        _calibrate_frame_impl(source, plan, v1.ExecutionOptions(), token=v1.CancellationToken(), obs=None, slot=slot)
        for _ in range(10)
    ]
    assert all(r.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS") for r in results)

    # Prepared once (1 call at context build), reused 9x (0 per-frame calls).
    assert len(norm_calls) == 1
    assert len(label_calls) == 1
    assert len(index_calls) == 1


def test_control_prepares_nothing(tmp_path):
    # ``flat_mode == "none"`` prepares no flat outcome (None).
    metadata = _sensor_metadata()
    masters = {}
    plan = _build_plan(v1.CalibrationRequest("control", "none"), masters, metadata)
    context, failure = _prepare_calibration_context(plan, token=v1.CancellationToken())
    assert failure is None
    assert context.prepared_flat is None
    assert _build_prepared_flat(plan, context.flat_prep_mode, {}) is None
