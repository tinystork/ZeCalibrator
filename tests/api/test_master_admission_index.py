"""Master-admission index/execution witnesses (P7-M3B rework §2/§7).

Public-API-only tests: a NAXIS=3/RGB master is rejected at index (no bogus 2-D
shape), while a stacked raw 2-D master is admitted by the role-aware master
execution path (the additive ``admission="master"`` decode) even though the light
decoder would refuse the same HISTORY marker.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

import zecalibrator.api.v1 as v1

SHAPE = (4, 4)


def _declaration(**kw):
    base = dict(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU",
        detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA",
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16",
        binning=(1, 1), sensor_dimensions=SHAPE, orientation="identity",
        cfa_phase="mono", roi_origin=(0, 0), exposure_s=10.0, temperature_c=20.0,
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        bias_exposure_max_s=0.01, saturation_limit_adu=60000.0,
        saturation_evidence="qualified",
    )
    base.update(kw)
    return v1.ImportDeclaration(**base)


def _write_rgb(path):
    cube = np.zeros((3, SHAPE[0], SHAPE[1]), dtype=np.int16)
    hdu = fits.PrimaryHDU(cube)
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def _write_stacked_dark(path):
    hdu = fits.PrimaryHDU(np.full(SHAPE, 10.0, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.header["HISTORY"] = "master stacked with sigma rejection"
    hdu.writeto(path, overwrite=True)
    return str(path)


def _write_light(path, value=100.0):
    hdu = fits.PrimaryHDU(np.full(SHAPE, value, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def test_rgb_naaxis3_rejected_at_index(tmp_path):
    rgb = _write_rgb(tmp_path / "rgb.fits")
    spec = v1.MasterImportSpec(
        path=rgb, master_type="dark", declaration=_declaration(),
        bias_state="included", dq_state="no_source_dq", mask_path=None,
    )
    index_path = str(tmp_path / "lib.sqlite")
    result = v1.index_library(v1.LibrarySpec(root=str(tmp_path), index_path=index_path), [spec])
    assert result.operation_status == "COMPLETED"
    assert result.candidate_count == 0
    # Explicit incompatibility, never a bogus 2-D shape entry.
    reasons = [d.to_dict().get("reason", "") for d in result.diagnostics]
    assert any("incompatible" in r and "RGB / debayered" in r for r in reasons), reasons


def test_stacked_master_admitted_through_execution(tmp_path):
    dark = _write_stacked_dark(tmp_path / "dark.fits")
    light = _write_light(tmp_path / "light.fits")
    spec = v1.MasterImportSpec(
        path=dark, master_type="dark", declaration=_declaration(),
        bias_state="included", dq_state="no_source_dq", mask_path=None,
    )
    index_path = str(tmp_path / "lib.sqlite")
    assert v1.index_library(
        v1.LibrarySpec(root=str(tmp_path), index_path=index_path), [spec]
    ).operation_status == "COMPLETED"

    source = v1.FitsFrameSource(
        path=light, declaration=_declaration(),
        roi_extent=v1.RoiExtentEvidence(
            extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0"
        ),
    )
    inspection = v1.inspect_frame(source)
    assert inspection.operation_status == "COMPLETED", inspection.details

    handle = v1.open_library(v1.LibrarySpec(root=str(tmp_path), index_path=index_path)).handle
    try:
        resolved = v1.resolve_calibration(
            inspection.inspection, v1.CalibrationRequest("dark_incl_bias"),
            handle, v1.default_match_policy(),
        )
    finally:
        handle.close()
    assert resolved.operation_status == "COMPLETED"
    assert resolved.outcome == "MATCHED"

    # The execution path decodes the stacked dark with admission="master" and
    # admits it (the light decoder would refuse the same HISTORY "stacked").
    result = v1.calibrate_frame(source, resolved.plan, v1.ExecutionOptions())
    assert result.status == "COMPLETED", (result.reason_code, result.warnings)
