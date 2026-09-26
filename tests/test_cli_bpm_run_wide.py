"""CLI end-to-end tests for the LOT 3 run-wide BPM path (subprocess).

``zecalibrator calibrate`` must use the run-wide application path: with a
compatible map the output is BPM-corrected and the log reports ``applied``;
without a base the run stays CALIBRATION_ONLY. These tests use a Bayer CFA
fixture so the same-CFA reconstruction operator has donors.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.cli as cli

SHAPE = (8, 8)
CFA = "GRBG"
_REPO = Path(__file__).resolve().parents[1]
_SRC = str(_REPO / "src")

BAYER_DECL = dict(
    source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
    domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
    detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
    adc_mode="MODE_16", binning=[1, 1], sensor_dimensions=[8, 8],
    orientation="identity", cfa_phase=CFA, roi_origin=[0, 0], exposure_s=10.0,
    temperature_c=20.0, filter="NONE", optical_train_id="SYNTH-TRAIN-1",
    bias_exposure_max_s=0.01, saturation_limit_adu=60000.0, saturation_evidence="qualified",
)
ROI = dict(extent=[8, 8], source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0")


def _env(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "xdg-config")
    env["XDG_DATA_HOME"] = str(tmp_path / "xdg-data")
    return env


def run_cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-m", "zecalibrator", *args],
        capture_output=True, text=True, env=_env(tmp_path),
    )


def _write_array_fits(path, arr):
    hdu = fits.PrimaryHDU(np.asarray(arr, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)


def _setup(tmp_path):
    light = tmp_path / "light.fits"
    data = np.full(SHAPE, 100.0, dtype=np.float32)
    data[3, 3] = 9000.0  # the hot pixel the BPM will reconstruct
    _write_array_fits(light, data)
    decl_path = tmp_path / "light.decl.json"
    decl_path.write_text(json.dumps(BAYER_DECL))
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(ROI))
    # Empty library (control mode needs no master).
    idx = run_cli(tmp_path, "index", str(tmp_path), "--declarations", "[]")
    assert idx.returncode == 0, idx.stderr
    return str(light), str(decl_path), str(roi_path), str(tmp_path / "zecalibrator.library.sqlite")


def _make_base(tmp_path, y, x):
    from zecalibrator.bpm.identity import ReadoutContext, SensorIdentity
    from zecalibrator.bpm.revision import SiteRecord, make_revision
    from zecalibrator.bpm.store import create_bad_pixel_database
    from zecalibrator.bpm.vocabulary import (
        ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION,
        REVISION_STATE_PROMOTED,
    )
    from zecalibrator.core.descriptors import DetectorIdentity
    from zecalibrator.core.geometry import Geometry

    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    identity = SensorIdentity(
        detector=DetectorIdentity(
            detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA", serial=None,
        ),
        geometry=Geometry(
            shape=SHAPE, sensor_dimensions=SHAPE, binning=(1, 1),
            roi_origin=(0, 0), roi_extent=SHAPE, orientation="identity", cfa_phase=CFA,
        ),
        readout=ReadoutContext(gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16"),
    )
    site = SiteRecord(
        position=(y, x),
        action_state=ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION,
        knowledge_state="QUALIFIED",
    )
    db.add_revision(make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=identity, sites=(site,), sequence=1))
    return root


def _calibrate(tmp_path, light, decl, roi, index, outdir, *extra):
    return run_cli(
        tmp_path, "calibrate", light, "--library", index, "--additive-mode",
        "control", "--declaration", "@" + decl, "--roi-extent", "@" + roi,
        "--destination", str(outdir), "--batch-id", "b1", *extra,
    )


def _output_fits(outdir):
    fits_files = [p for p in outdir.glob("*.fits")]
    assert len(fits_files) == 1, fits_files
    with fits.open(fits_files[0]) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


def test_calibrate_without_base_is_calibration_only(tmp_path):
    light, decl, roi, index = _setup(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    proc = _calibrate(tmp_path, light, decl, roi, index, outdir)
    assert proc.returncode == 0, proc.stderr
    assert "BPM correction: none" in proc.stderr
    assert "Reconstructed sites: 0" in proc.stderr
    # The hot pixel is NOT reconstructed (ordinary passthrough).
    assert _output_fits(outdir)[3, 3] == 9000.0


def test_calibrate_with_compatible_map_applies_run_wide(tmp_path):
    light, decl, roi, index = _setup(tmp_path)
    base = _make_base(tmp_path, 3, 3)
    outdir = tmp_path / "out"
    outdir.mkdir()
    proc = _calibrate(tmp_path, light, decl, roi, index, outdir, "--bad-pixel-db", str(base))
    assert proc.returncode == 0, proc.stderr
    assert "BPM correction: applied" in proc.stderr
    assert "Reconstructed sites: 1" in proc.stderr
    assert "Bad pixels: 1" in proc.stderr
    # The hot pixel is reconstructed from same-CFA donors (~100.0).
    assert _output_fits(outdir)[3, 3] == pytest.approx(100.0)


def test_calibrate_with_compatible_map_has_no_force_flag():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    names = {name.lstrip("-") for action in sub.choices["calibrate"]._actions for name in action.option_strings}
    for bad in ("force-bpm", "enable-bpm", "apply-bpm", "bpm-force"):
        assert bad not in names, f"forbidden BPM flag {bad!r} present"
