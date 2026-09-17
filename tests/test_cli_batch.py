"""Phase 6 CLI tests: exit contract (0/2/3/130), thin facade, scriptable JSON.

The CLI is a thin facade over ``zecalibrator.api.v1``; it must not import
``zecalibrator.core`` / ``zecalibrator.application`` / ``zecalibrator.io``
directly. These tests run the CLI in subprocesses (for 0/2/3) and in-process
(for 130 via an injected cancellation token).
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

import zecalibrator.api.v1 as v1
import zecalibrator.cli as cli

SHAPE = (4, 4)
_REPO = Path(__file__).resolve().parents[1]
_SRC = str(_REPO / "src")

DECL = dict(
    source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
    domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
    detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
    adc_mode="MODE_16", binning=[1, 1], sensor_dimensions=[4, 4],
    orientation="identity", cfa_phase="mono", roi_origin=[0, 0], exposure_s=10.0,
    temperature_c=20.0, filter="NONE", optical_train_id="SYNTH-TRAIN-1",
    bias_exposure_max_s=0.01, saturation_limit_adu=60000.0, saturation_evidence="qualified",
)
ROI = dict(extent=[4, 4], source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0")


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "zecalibrator", *args],
        capture_output=True, text=True, env=_env(),
    )


def _write_fits(path, value, bunit="ADU"):
    hdu = fits.PrimaryHDU(np.full(SHAPE, value, dtype=np.float32))
    hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)


def _setup_library(tmp_path):
    """Write a dark master + light, index the library, return paths."""
    dark = tmp_path / "dark.fits"
    _write_fits(dark, 10.0)
    (tmp_path / "dark.mask.npy").write_bytes(
        _npy_bytes(np.zeros(SHAPE, dtype=np.uint16))
    )
    light = tmp_path / "light.fits"
    _write_fits(light, 100.0)

    decl_path = tmp_path / "light.decl.json"
    decl_path.write_text(json.dumps(DECL))
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(ROI))
    imports_path = tmp_path / "imports.json"
    imports_path.write_text(
        json.dumps(
            [dict(path="dark.fits", master_type="dark", hdu=0,
                  mask_path="dark.mask.npy", bias_state="included", declaration=DECL)]
        )
    )

    idx = run_cli("index", str(tmp_path), "--declarations", "@" + str(imports_path))
    assert idx.returncode == 0, idx.stderr
    index_path = str(tmp_path / "zecalibrator.library.sqlite")
    return str(light), str(decl_path), str(roi_path), index_path


def _npy_bytes(arr):
    import io

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Exit contract
# ---------------------------------------------------------------------------
def test_cli_inspect_exit_0(tmp_path):
    light, decl, roi, _ = _setup_library(tmp_path)
    proc = run_cli("inspect", light, "--declaration", "@" + decl)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data[0]["status"] == "COMPLETED"
    assert data[0]["domain_finding"] == "raw"


def test_cli_index_and_match_and_calibrate_exit_0(tmp_path):
    light, decl, roi, index = _setup_library(tmp_path)
    proc = run_cli(
        "match", light, "--library", index, "--additive-mode", "dark_incl_bias",
        "--declaration", "@" + decl, "--roi-extent", "@" + roi,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["outcome"] == "MATCHED"

    outdir = tmp_path / "out"
    outdir.mkdir()
    proc = run_cli(
        "calibrate", light, "--library", index, "--additive-mode", "dark_incl_bias",
        "--declaration", "@" + decl, "--roi-extent", "@" + roi,
        "--destination", str(outdir), "--batch-id", "b1",
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "COMPLETED"
    assert data["items"][0]["disposition"] == "COMPLETED"
    assert os.path.exists(data["manifest"])


def test_cli_exit_2_on_validation_refusal(tmp_path):
    # argparse choice refusal -> 2
    proc = run_cli(
        "calibrate", "x.fits", "--library", "y.sqlite",
        "--additive-mode", "not-a-mode",
    )
    assert proc.returncode == 2

    # malformed declaration JSON -> 2
    proc = run_cli("inspect", "x.fits", "--declaration", "{not-json")
    assert proc.returncode == 2


def test_cli_exit_2_on_bad_destination_no_traceback(tmp_path):
    light, decl, roi, index = _setup_library(tmp_path)
    # destination is an existing file -> 2, no traceback
    f = tmp_path / "dest-file"
    f.write_text("x")
    proc = run_cli(
        "calibrate", light, "--library", index, "--additive-mode", "dark_incl_bias",
        "--declaration", "@" + decl, "--roi-extent", "@" + roi,
        "--destination", str(f),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr

    # destination is a missing directory -> 2, no traceback
    proc = run_cli(
        "calibrate", light, "--library", index, "--additive-mode", "dark_incl_bias",
        "--declaration", "@" + decl, "--roi-extent", "@" + roi,
        "--destination", str(tmp_path / "missing"),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_emit_refuses_non_json_value():
    with pytest.raises(cli._CliValidationError):
        cli._emit({"bad": object()})


def test_cli_exit_3_on_partial_failure(tmp_path):
    light, decl, roi, index = _setup_library(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    # A request with no matching master (flat apply) -> per-item FAILED -> 3.
    proc = run_cli(
        "calibrate", light, "--library", index, "--additive-mode", "control",
        "--flat-mode", "apply", "--declaration", "@" + decl,
        "--roi-extent", "@" + roi, "--destination", str(outdir), "--batch-id", "b2",
    )
    assert proc.returncode == 3, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "PARTIAL"
    assert data["items"][0]["disposition"] == "FAILED"


def test_cli_exit_130_on_cancellation(tmp_path):
    token = v1.CancellationToken()
    token.cancel()
    rc = cli.main(["inspect", str(tmp_path / "x.fits")], cancel=token)
    assert rc == 130


def test_cli_inprocess_validation_refusal_returns_2(tmp_path):
    rc = cli.main(["inspect", "x.fits", "--declaration", "{bad"])
    assert rc == 2


# ---------------------------------------------------------------------------
# Thin facade
# ---------------------------------------------------------------------------
def test_cli_source_has_no_internal_imports():
    src = (_REPO / "src" / "zecalibrator" / "cli.py").read_text(encoding="utf-8")
    for mod in ("zecalibrator.core", "zecalibrator.application", "zecalibrator.io"):
        assert f"import {mod}" not in src, mod
        assert f"from {mod}" not in src, mod


def test_cli_import_does_not_pull_internal_modules():
    code = (
        "import sys\n"
        "import zecalibrator.cli\n"
        "for mod in ('zecalibrator.core', 'zecalibrator.application', 'zecalibrator.io'):\n"
        "    assert mod not in sys.modules, mod\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=_env()
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_cli_help_has_no_capability_keywords(tmp_path):
    proc = run_cli("--help")
    out = " ".join((proc.stdout + proc.stderr).lower().split())
    for keyword in ("calibrate_frame", "calibrate_batch", "gpu", "cuda"):
        assert keyword not in out
    assert "bootstrap skeleton" not in out
