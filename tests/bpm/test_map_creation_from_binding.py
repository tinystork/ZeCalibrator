"""LOT 3 map creation from a selected dark binding (headless glue test).

The GUI's "Create Bad Pixel Map" action decodes the already-selected dark
binding and stores new immutable revisions. This test exercises that glue end
to end: build a dark master (with a hot pixel), index + resolve a plan binding
the dark, then create the map from the binding and verify a promoted revision
with detected sites exists.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.api.v1 import _bpm
from zecalibrator.storage import StoragePaths

SHAPE = (8, 8)

DECL = dict(
    source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
    domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
    detector_model="SYNTH-MONO", gain=100.0, offset=50.0, readout_mode="MODE_A",
    adc_mode="MODE_16", binning=[1, 1], sensor_dimensions=[8, 8],
    orientation="identity", cfa_phase="mono", roi_origin=[0, 0], exposure_s=10.0,
    temperature_c=20.0, filter="NONE", optical_train_id="SYNTH-TRAIN-1",
    bias_exposure_max_s=0.01, saturation_limit_adu=60000.0, saturation_evidence="qualified",
)
ROI = dict(extent=[8, 8], source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0")


def _storage(tmp_path) -> StoragePaths:
    return StoragePaths(
        user_config_path=tmp_path / "config",
        user_data_path=tmp_path / "data",
        user_cache_path=tmp_path / "cache",
        user_state_path=tmp_path / "state",
        user_log_path=tmp_path / "log",
    )


def _write_fits(path, arr):
    hdu = fits.PrimaryHDU(np.asarray(arr, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)


def _npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.uint16), allow_pickle=False)
    return buf.getvalue()


def test_create_bpm_map_from_selected_dark_binding(tmp_path):
    dark = tmp_path / "dark.fits"
    dark_arr = np.full(SHAPE, 10.0, dtype=np.float32)
    dark_arr[3, 3] = 1_000_000.0  # a clear hot pixel the recipe must flag
    _write_fits(dark, dark_arr)
    (tmp_path / "dark.mask.npy").write_bytes(_npy_bytes(np.zeros(SHAPE, dtype=np.uint16)))
    light = tmp_path / "light.fits"
    _write_fits(light, np.full(SHAPE, 100.0, dtype=np.float32))

    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "zecalibrator.library.sqlite"))
    imports = [
        v1.MasterImportSpec(
            path="dark.fits", master_type="dark", hdu=0, mask_path="dark.mask.npy",
            bias_state="included", declaration=v1.ImportDeclaration(**DECL),
        )
    ]
    idx = v1.index_library(spec, imports)
    assert idx.operation_status == "COMPLETED", idx.details

    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED", opened.details
    handle = opened.handle
    try:
        source = v1.FitsFrameSource(
            path=str(light), hdu=0, declaration=v1.ImportDeclaration(**DECL),
            roi_extent=v1.RoiExtentEvidence(**ROI),
        )
        inspection = v1.inspect_frame(source)
        assert inspection.operation_status == "COMPLETED", inspection.details
        resolved = v1.resolve_calibration(
            inspection.inspection, v1.CalibrationRequest("dark_incl_bias"),
            handle, v1.default_match_policy(),
        )
        assert resolved.operation_status == "COMPLETED", resolved.details
        plan = resolved.plan
        assert plan is not None and "dark" in plan.masters
    finally:
        handle.close()

    storage = _storage(tmp_path)
    settings = _bpm.bpm_settings(str(tmp_path / "base"))
    result = _bpm.create_bpm_map_from_binding(storage, settings, plan.masters["dark"])

    assert result.site_count >= 1
    assert result.candidate.state == "candidate"
    assert result.promoted.state == "promoted"

    # The created map is immediately selectable by the store lookup.
    from zecalibrator.bpm.store import load_bad_pixel_database

    res = load_bad_pixel_database(tmp_path / "base").database.resolve(
        result.promoted.sensor_identity
    )
    assert res.outcome == "SELECTED"
    assert res.revision.revision_id == result.promoted.revision_id
