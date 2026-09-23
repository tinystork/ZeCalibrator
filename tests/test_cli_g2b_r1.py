"""G2B R1 — CLI passthrough/partial presentation (successful, not an error)."""

from __future__ import annotations

import json

import numpy as np
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.cli import main

SHAPE = (4, 4)


def _declaration():
    return {
        "source": "synthetic_fixture", "identity": "SYNTH-BASE-1", "version": "1.0",
        "domain": "raw", "units": "ADU", "detector_instance_id": "SYNTH-DET-0001",
        "detector_model": "SYNTH-CFA", "gain": 100.0, "offset": 50.0,
        "readout_mode": "MODE_A", "adc_mode": "MODE_16", "binning": (1, 1),
        "sensor_dimensions": list(SHAPE), "orientation": "identity",
        "cfa_phase": "mono", "roi_origin": (0, 0), "exposure_s": 10.0,
        "temperature_c": 20.0, "filter": "NONE", "optical_train_id": "SYNTH-TRAIN-1",
        "bias_exposure_max_s": 0.01, "saturation_limit_adu": 60000.0,
        "saturation_evidence": "qualified",
    }


def _write_light(path):
    hdu = fits.PrimaryHDU(np.full(SHAPE, 100.0, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def test_cli_match_passthrough_is_successful(tmp_path, capsys):
    light_path = _write_light(tmp_path / "light.fits")
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "lib.sqlite"))
    idx = v1.index_library(spec, [])
    assert idx.operation_status == "COMPLETED", idx.details

    code = main([
        "match", light_path,
        "--library", str(tmp_path / "lib.sqlite"),
        "--additive-mode", "control",
        "--flat-mode", "none",
        "--declaration", json.dumps(_declaration()),
    ])
    out = capsys.readouterr().out
    assert code == 0, out
    payload = json.loads(out)
    assert payload["status"] == "COMPLETED"
    assert payload["outcome"] == "MATCHED"  # passthrough, never an error
