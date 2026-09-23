"""ZC-G2C-THERMAL-FLAT-SAFETY — rework-1 executor thermal parity (F2).

The matching/routes/digest work (r0) left the execution-time temperature gate in
``application/executor.py`` comparing the MEASURED CCD-TEMP. Rework-1 brings the
executor's dark/flat_dark temperature clause into the SAME setpoint-precedence
semantics as matching:

* both setpoints finite  -> compare the SETPOINTS (numeric equality under
  TEMP_PARSER_TOLERANCE_C); CCD-TEMP must NOT produce a blocking mismatch;
* otherwise              -> preserve the pre-existing measured-CCD-TEMP clause
  byte-for-byte.

These tests mirror T1/T2/T3 at the execution layer, plus a full
resolve -> execute end-to-end witness.
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
from zecalibrator.core.routes import OUTCOME_READY

SHAPE = (4, 4)


# ---------------------------------------------------------------------------
# Executor thermal parity (synthetic SensorMetadata / master-frame equivalents)
# ---------------------------------------------------------------------------
def test_executor_t1_setpoints_equal_ccd_equal_accepted(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, temperature_c=-10.0, temperature_setpoint_c=-10.0)
    dark = make_frame((4, 4), value=10.0, temperature_c=-10.0, temperature_setpoint_c=-10.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    result = execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert result.status == "COMPLETED"


def test_executor_t2_setpoints_equal_ccd_different_accepted_witness(make_frame_fixture):
    make_frame = make_frame_fixture
    # The witness divergence: equal SET-TEMP=-10, measured CCD-TEMP -10.5 vs -10.0.
    light = make_frame((4, 4), value=100.0, temperature_c=-10.5, temperature_setpoint_c=-10.0)
    dark = make_frame((4, 4), value=10.0, temperature_c=-10.0, temperature_setpoint_c=-10.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    result = execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert result.status == "COMPLETED"


def test_executor_t3_different_setpoints_mismatch(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0, temperature_c=-10.0, temperature_setpoint_c=-10.0)
    dark = make_frame((4, 4), value=10.0, temperature_c=-10.0, temperature_setpoint_c=-20.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert "TEMPERATURE_MISMATCH" in exc.value.reason_code


def test_executor_setpoint_missing_one_side_preserves_ccd_temp_clause(make_frame_fixture):
    make_frame = make_frame_fixture
    # Setpoint missing on the LIGHT side -> not both known -> pre-existing
    # CCD-TEMP clause applies (measured -10.5 vs -10.0 -> mismatch).
    light = make_frame((4, 4), value=100.0, temperature_c=-10.5, temperature_setpoint_c=None)
    dark = make_frame((4, 4), value=10.0, temperature_c=-10.0, temperature_setpoint_c=-10.0)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    with pytest.raises(GeometryMismatchError) as exc:
        execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert "TEMPERATURE_MISMATCH" in exc.value.reason_code


def test_executor_setpoint_missing_both_sides_preserves_ccd_temp_clause(make_frame_fixture):
    make_frame = make_frame_fixture
    # Both setpoints unknown -> measured CCD-TEMP clause (equal -> accepted).
    light = make_frame((4, 4), value=100.0, temperature_c=-10.0, temperature_setpoint_c=None)
    dark = make_frame((4, 4), value=10.0, temperature_c=-10.0, temperature_setpoint_c=None)
    masters = {"dark": MasterBinding(role="dark", frame=dark, bias_state="included")}
    result = execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters)
    assert result.status == "COMPLETED"


def test_skipped_flat_raw_passthrough_pixel_identical(make_frame_fixture):
    # H2 numeric passthrough parity: a RAW light with NO additive correction
    # (the skipped-flat route) must execute to a PIXEL-IDENTICAL output plane.
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    result = execute_calibration(light, CalibrationRequest("control", "none"), {})
    assert result.status == "COMPLETED"
    assert np.array_equal(result.data, light.data)


# ---------------------------------------------------------------------------
# End-to-end execution witness (synthetic FITS -> resolve -> calibrate_frame)
# ---------------------------------------------------------------------------
def _declaration(**kw):
    base = dict(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU",
        detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA",
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16",
        binning=(1, 1), sensor_dimensions=SHAPE, orientation="identity",
        cfa_phase="mono", roi_origin=(0, 0), exposure_s=10.0,
        temperature_c=20.0, filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        bias_exposure_max_s=0.01, saturation_limit_adu=60000.0,
        saturation_evidence="qualified",
    )
    base.update(kw)
    return v1.ImportDeclaration(**base)


def _write_fits(path, value, header_cards=()):
    hdu = fits.PrimaryHDU(np.full(SHAPE, float(value), dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    for kw, val in header_cards:
        hdu.header[kw] = val
    hdu.writeto(path, overwrite=True)
    return str(path)


def _light_source(path, **decl_kw):
    roi = v1.RoiExtentEvidence(
        extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0"
    )
    return v1.FitsFrameSource(path=path, declaration=_declaration(**decl_kw), roi_extent=roi)


def _resolve_and_execute(tmp_path, light_ccd_temp):
    """Resolve a light (SET-TEMP=-10, given CCD-TEMP) against a dark+flat and
    execute; return the calibration result status + reason_code."""
    light_path = _write_fits(
        tmp_path / "light.fits", 100.0,
        header_cards=[("SET-TEMP", -10.0), ("CCD-TEMP", light_ccd_temp)],
    )
    source = _light_source(
        light_path, temperature_c=light_ccd_temp, temperature_setpoint_c=-10.0,
    )
    inspection = v1.inspect_frame(source)
    assert inspection.operation_status == "COMPLETED", inspection.details
    light_md = inspection.inspection.metadata
    # The setpoint must reach the light metadata.
    assert light_md.temperature_setpoint_c == -10.0

    # Dark master: SET-TEMP read from FITS via the admission path (_fits_setpoint).
    dark_path = _write_fits(
        tmp_path / "dark.fits", 10.0, header_cards=[("SET-TEMP", -10.0)],
    )
    flat_path = _write_fits(tmp_path / "flat.fits", 0.455)

    imports = [
        v1.MasterImportSpec(
            path=dark_path, master_type="dark", hdu=0,
            declaration=_declaration(temperature_c=-10.0),
            bias_state="included", dq_state="no_source_dq",
        ),
        v1.MasterImportSpec(
            path=flat_path, master_type="flat", hdu=0,
            declaration=_declaration(exposure_s=1.0, temperature_c=-10.0),
            flat_form="corrected_unnormalized", dq_state="no_source_dq",
        ),
    ]
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / f"lib_{light_ccd_temp}.sqlite"))
    idx = v1.index_library(spec, imports)
    assert idx.operation_status == "COMPLETED", idx.diagnostics

    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        light = light_constraints_from_sensor_metadata(light_md)
        resolution = resolve_route(light, handle.snapshot, v1.default_match_policy())
    finally:
        handle.close()

    assert resolution.outcome == OUTCOME_READY, resolution.reasons
    assert resolution.plan is not None
    assert set(resolution.plan.masters) == {"dark", "flat"}

    result = v1.calibrate_frame(source, resolution.plan, v1.ExecutionOptions())
    return result.status, result.reason_code


def test_thermal_setpoint_end_to_end_execution_completes_no_divergence(tmp_path):
    # The witness (CCD-TEMP -10.5) and its counterpart (-10.0) must both resolve
    # to dark+flat AND execute COMPLETED — no divergence from measured CCD-TEMP.
    status_primary, code_primary = _resolve_and_execute(tmp_path, -10.5)
    assert status_primary == "COMPLETED", (status_primary, code_primary)

    status_control, code_control = _resolve_and_execute(tmp_path, -10.0)
    assert status_control == "COMPLETED", (status_control, code_control)


# ---------------------------------------------------------------------------
# G3 — library-index setpoint persistence (rework-2)
# ---------------------------------------------------------------------------
def test_library_index_roundtrip_preserves_setpoint(tmp_path):
    dark_path = _write_fits(
        tmp_path / "dark.fits", 10.0, header_cards=[("SET-TEMP", -10.0)]
    )
    imports = [
        v1.MasterImportSpec(
            path=dark_path, master_type="dark", hdu=0,
            declaration=_declaration(temperature_c=-10.0),
            bias_state="included", dq_state="no_source_dq",
        ),
    ]
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "lib.sqlite"))
    idx = v1.index_library(spec, imports)
    assert idx.operation_status == "COMPLETED", idx.diagnostics

    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        dark_cands = handle.snapshot.candidates["dark"]
        assert len(dark_cands) == 1
        # The setpoint survives the persisted index round-trip.
        assert dark_cands[0].descriptor.acquisition.temperature_setpoint_c == -10.0
    finally:
        handle.close()


def test_real_path_negative_dark_settemp_mismatch_rejected(tmp_path):
    # Light SET-TEMP=-10 / CCD-TEMP=-10.0 vs dark admitted with SET-TEMP=-20.
    light_path = _write_fits(
        tmp_path / "light.fits", 100.0,
        header_cards=[("SET-TEMP", -10.0), ("CCD-TEMP", -10.0)],
    )
    source = _light_source(
        light_path, temperature_c=-10.0, temperature_setpoint_c=-10.0,
    )
    inspection = v1.inspect_frame(source)
    assert inspection.operation_status == "COMPLETED", inspection.details
    light_md = inspection.inspection.metadata
    assert light_md.temperature_setpoint_c == -10.0

    dark_path = _write_fits(
        tmp_path / "dark.fits", 10.0, header_cards=[("SET-TEMP", -20.0)]
    )
    imports = [
        v1.MasterImportSpec(
            path=dark_path, master_type="dark", hdu=0,
            declaration=_declaration(temperature_c=-10.0),
            bias_state="included", dq_state="no_source_dq",
        ),
    ]
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "lib.sqlite"))
    idx = v1.index_library(spec, imports)
    assert idx.operation_status == "COMPLETED", idx.diagnostics

    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        dark_cands = handle.snapshot.candidates["dark"]
        # The admitted dark descriptor must carry the REAL setpoint -20.0.
        assert dark_cands[0].descriptor.acquisition.temperature_setpoint_c == -20.0
        light = light_constraints_from_sensor_metadata(light_md)
        resolution = resolve_route(light, handle.snapshot, v1.default_match_policy())
    finally:
        handle.close()

    # The dark must be REJECTED with TEMPERATURE_MISMATCH (never a silent dark).
    assert any(r.code == "TEMPERATURE_MISMATCH" for r in resolution.reasons)
    assert resolution.plan is not None
    assert "dark" not in resolution.plan.masters
    assert "dark" not in resolution.plan.composition.applied_roles
