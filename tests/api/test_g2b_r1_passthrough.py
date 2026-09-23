"""G2B R1 — end-to-end passthrough + provenance + output-header truthfulness.

Exercises the REAL public layers: a raw FITS light -> inspect -> resolve
(control+none -> passthrough plan) -> calibrate, asserting the passthrough output
is pixel-identical to the decoded input (no artificial transformation), the
provenance truthfully records level NONE (never "calibrated"), and the
standalone output header carries the ZECALLEVEL/ZECALCOMP cards.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1

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


def _write_light(path, value=100.0):
    hdu = fits.PrimaryHDU(np.full(SHAPE, float(value), dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def _source(path):
    roi = v1.RoiExtentEvidence(
        extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0"
    )
    return v1.FitsFrameSource(path=path, declaration=_declaration(), roi_extent=roi)


def _passthrough_plan(tmp_path):
    path = _write_light(tmp_path / "light.fits", 100.0)
    source = _source(path)
    inspection = v1.inspect_frame(source)
    assert inspection.operation_status == "COMPLETED", inspection.details
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "lib.sqlite"))
    # Empty library (no masters): initialize the index with zero imports.
    idx = v1.index_library(spec, [])
    assert idx.operation_status == "COMPLETED", idx.details
    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        result = v1.resolve_calibration(
            inspection.inspection, v1.CalibrationRequest("control", "none"), handle,
            v1.default_match_policy(),
        )
    finally:
        handle.close()
    assert result.outcome == "MATCHED"
    plan = result.plan
    assert plan.composition.level == "NONE"
    return source, plan, inspection


def test_passthrough_pixel_identical(tmp_path):
    source, plan, _ = _passthrough_plan(tmp_path)
    out = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    assert out.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    # No correction was applied: the output plane is pixel-identical to the
    # decoded input (100.0 everywhere, no artificial transformation).
    assert np.array_equal(out.data, np.full(SHAPE, 100.0, dtype=np.float32), equal_nan=True)
    # Composition is NONE — never declared "calibrated".
    assert out.composition.level == "NONE"
    assert out.composition.applied_roles == ()


def test_passthrough_provenance_truthful(tmp_path):
    source, plan, _ = _passthrough_plan(tmp_path)
    out = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    prov = out.provenance.to_dict()
    comp = prov["composition"]
    assert comp["level"] == "NONE"
    assert comp["applied_roles"] == []
    assert prov["selection"] == []
    # The plan itself carries no master bindings.
    assert plan.masters == {}


def test_passthrough_output_header_level_none(tmp_path):
    source, plan, _ = _passthrough_plan(tmp_path)
    out = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    from zecalibrator.io.output_header import build_output_header_fields

    fields = build_output_header_fields(
        light_constraints=plan.light_constraints,
        science_shape=out.data.shape,
        status=out.status,
        plan_id=plan.plan_id,
        provenance_schema=plan.versions.provenance_schema,
        composition=plan.composition,
    )
    assert fields["HIERARCH ZECALLEVEL"] == "NONE"
    assert fields["HIERARCH ZECALCOMP"] == "none"
    # No false scaling/checksum cards are ever emitted.
    for forbidden in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
        assert forbidden not in fields
