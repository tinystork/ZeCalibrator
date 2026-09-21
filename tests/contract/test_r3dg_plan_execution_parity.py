"""R3D-G — plan/executor validation parity (P7-M3B).

The Standard planner (R3D-C/E) arbitrates gain/offset/temperature/orientation/
roi_origin unknowns as UNVERIFIED and exempts a prepared flat's gain/offset from
the Light<->Flat equality requirement. The executor must NOT re-arbitrate those
same unknowns under the old strict policy: execution applies the resolved
:class:`~zecalibrator.core.plans.CalibrationPlan` and rejects only *execution*
contradictions (known both-sides differences, byte/hash/shape/CFA identity
violations), never unknowns the planner already accepted.

These tests exercise the REAL production layers end-to-end (Siril-like FITS
light -> inspect -> managed library -> Standard auto-route -> CalibrationPlan ->
``calibrate_frame`` -> CalibrationResult) plus the independent execution-parity
boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.application.executor import (
    CalibrationRequest,
    MasterBinding,
    execute_calibration,
)
from zecalibrator.application.library import light_constraints_from_sensor_metadata
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.errors import GeometryMismatchError
from zecalibrator.core.routes import OUTCOME_NEEDS_ATTENTION, OUTCOME_READY

SHAPE = (4, 4)


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
    return v1.ImportDeclaration(**base)


def _write_fits(path, value):
    hdu = fits.PrimaryHDU(np.full(SHAPE, float(value), dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def _light_source(path, **decl_kw):
    roi = v1.RoiExtentEvidence(
        extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0"
    )
    return v1.FitsFrameSource(path=path, declaration=_declaration(**decl_kw), roi_extent=roi)


def _build_library(tmp_path, imports):
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "lib.sqlite"))
    idx = v1.index_library(spec, imports)
    assert idx.operation_status == "COMPLETED", idx.details
    return spec


# ---------------------------------------------------------------------------
# 1. Mandatory integration regression through the REAL production layers.
# ---------------------------------------------------------------------------
def test_siril_like_end_to_end_execution_parity(tmp_path):
    # Siril-like light: GAIN/OFFSET/CCD-TEMP known on the light; a Siril dark
    # master drops gain/offset/temperature; a prepared flat carries a *different*
    # gain (456 vs 120). The route must resolve dark_incl_bias + apply +
    # normalize_only, and execution must complete (never re-reject the planner's
    # UNVERIFIED unknowns or the prepared-flat gain mismatch).
    light_path = _write_fits(tmp_path / "light.fits", 100.0)
    source = _light_source(
        light_path, gain=120.0, offset=8.0, temperature_c=-9.9, exposure_s=10.0
    )

    inspection = v1.inspect_frame(source)
    assert inspection.operation_status == "COMPLETED", inspection.details
    light_md = inspection.inspection.metadata

    dark_path = _write_fits(tmp_path / "dark.fits", 10.0)
    flat_path = _write_fits(tmp_path / "flat.fits", 0.455)

    imports = [
        v1.MasterImportSpec(
            path=dark_path, master_type="dark", hdu=0,
            declaration=_declaration(gain=None, offset=None, temperature_c=None),
            bias_state="included", dq_state="no_source_dq",
        ),
        v1.MasterImportSpec(
            path=flat_path, master_type="flat", hdu=0,
            declaration=_declaration(
                gain=456.0, offset=166.0, temperature_c=17.9, exposure_s=1.0
            ),
            flat_form="corrected_unnormalized", dq_state="no_source_dq",
        ),
    ]
    spec = _build_library(tmp_path, imports)

    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        light = light_constraints_from_sensor_metadata(light_md)
        resolution = resolve_route(light, handle.snapshot, v1.default_match_policy())
    finally:
        handle.close()

    assert resolution.outcome == OUTCOME_READY
    assert resolution.plan is not None
    assert resolution.route.additive_mode == "dark_incl_bias"
    assert resolution.route.flat_mode == "apply"
    assert resolution.route.flat_prep_mode == "normalize_only"
    assert set(resolution.plan.masters) == {"dark", "flat"}

    result = v1.calibrate_frame(source, resolution.plan, v1.ExecutionOptions())
    assert result.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS"), (
        result.reason_code, result.warnings,
    )
    assert result.data.dtype == np.float32
    assert np.all(np.isfinite(result.data))


# ---------------------------------------------------------------------------
# 2. Independent execution-parity tests.
# ---------------------------------------------------------------------------
def test_missing_dark_gain_offset_temp_execution_succeeds(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, gain=120.0, offset=8.0, temperature_c=-9.9)
    dark = make_frame(
        (4, 4), value=10.0, gain=None, offset=None, temperature_c=None,
    )
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    result = execute_calibration(
        light, CalibrationRequest("dark_incl_bias"), masters
    )
    assert result.status == "COMPLETED"


def test_missing_bayer_orientation_roi_origin_execution_succeeds(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame(
        (4, 4), value=100.0, cfa_phase="GRBG",
        orientation="identity", roi_origin=(0, 0),
    )
    dark = make_frame(
        (4, 4), value=10.0, cfa_phase="GRBG",
        orientation=None, roi_origin=None,
    )
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    result = execute_calibration(
        light, CalibrationRequest("dark_incl_bias"), masters
    )
    assert result.status == "COMPLETED"


def test_prepared_flat_gain_mismatch_execution_succeeds(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, gain=120.0)
    dark = make_frame((4, 4), value=10.0, exposure_s=10.0, gain=120.0)
    finc = make_frame(
        (4, 4), value=0.455, exposure_s=1.0, gain=456.0, offset=166.0,
    )
    masters = {
        "dark": MasterBinding(role="dark", frame=dark, bias_state="included"),
        "flat": MasterBinding(role="flat", frame=finc, flat_form="corrected_unnormalized"),
    }
    result = execute_calibration(
        light,
        CalibrationRequest("dark_incl_bias", "apply"),
        masters,
        flat_prep_mode="normalize_only",
    )
    assert result.status == "COMPLETED"
    # R = flat / median(flat) = 0.455 / 0.455 = 1.0; (100 - 10) / 1.0 = 90.
    assert np.allclose(result.data, np.full((4, 4), 90.0, dtype=np.float32), atol=1e-4)


def test_known_dark_gain_mismatch_planner_blocks(tmp_path):
    # The planner (Standard auto-route) still refuses a known dark gain
    # mismatch; a valid Standard plan is never produced.
    light_path = _write_fits(tmp_path / "light.fits", 100.0)
    source = _light_source(light_path, gain=120.0)
    inspection = v1.inspect_frame(source)
    assert inspection.operation_status == "COMPLETED", inspection.details
    light_md = inspection.inspection.metadata

    dark_path = _write_fits(tmp_path / "dark.fits", 10.0)
    imports = [
        v1.MasterImportSpec(
            path=dark_path, master_type="dark", hdu=0,
            declaration=_declaration(gain=999.0),
            bias_state="included", dq_state="no_source_dq",
        ),
    ]
    spec = _build_library(tmp_path, imports)

    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        light = light_constraints_from_sensor_metadata(light_md)
        resolution = resolve_route(light, handle.snapshot, v1.default_match_policy())
    finally:
        handle.close()

    assert resolution.outcome == OUTCOME_NEEDS_ATTENTION
    assert resolution.plan is None
    assert any(r.code == "GAIN_MISMATCH" for r in resolution.reasons)


def test_master_byte_hash_mismatch_execution_fails(tmp_path):
    # Tampering with the master bytes after indexing must still fail execution.
    light_path = _write_fits(tmp_path / "light.fits", 100.0)
    source = _light_source(light_path)
    inspection = v1.inspect_frame(source)
    assert inspection.operation_status == "COMPLETED", inspection.details
    light_md = inspection.inspection.metadata

    dark_path = _write_fits(tmp_path / "dark.fits", 10.0)
    imports = [
        v1.MasterImportSpec(
            path=dark_path, master_type="dark", hdu=0,
            declaration=_declaration(),
            bias_state="included", dq_state="no_source_dq",
        ),
    ]
    spec = _build_library(tmp_path, imports)

    opened = v1.open_library(spec)
    handle = opened.handle
    try:
        light = light_constraints_from_sensor_metadata(light_md)
        resolution = resolve_route(light, handle.snapshot, v1.default_match_policy())
    finally:
        handle.close()
    assert resolution.outcome == OUTCOME_READY

    # Overwrite the dark master with different bytes (same shape, new content).
    _write_fits(dark_path, 99.0)

    result = v1.calibrate_frame(source, resolution.plan, v1.ExecutionOptions())
    assert result.status == "FAILED"
    # The tampered master content is rejected either at plan revalidation
    # (PLAN_INVALID) or at master load (MASTER_LOAD_FAILED) — execution must
    # never silently apply a master whose bytes no longer match the plan.
    assert result.reason_code in ("PLAN_INVALID", "MASTER_LOAD_FAILED")


def test_actual_shape_mismatch_execution_fails(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=100.0)
    dark = make_frame((3, 3), value=10.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert "GEOMETRY_MISMATCH" in exc.value.reason_code


def test_known_cfa_mismatch_execution_fails(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, cfa_phase="mono")
    dark = make_frame((4, 4), value=10.0, cfa_phase="RGGB")
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert "CFA_PHASE_MISMATCH" in exc.value.reason_code
