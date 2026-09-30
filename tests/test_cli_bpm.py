"""CLI Bad Pixel Database exposure tests (LOT 2).

The CLI keeps stdout as deterministic, scriptable JSON; the §50 BPM log
synthesis goes to stderr. These tests cover the automatic persistent config,
the optional ``--bad-pixel-db`` override, and the calm §51 wording (no
"ERROR: BPM failed" when the gate is closed).
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


def _isolated_storage(tmp_path, *, platform=None):
    """Isolate ZeCalibrator's platformdirs roots for CLI subprocesses.

    Returns ``(env_overrides, config_dir)``:

    * ``env_overrides`` — the platform-specific environment variables that
      redirect the non-roaming ``PlatformDirs(appname="ZeCalibrator",
      appauthor="ZeSoftware", version=None, roaming=False)`` roots into
      ``tmp_path`` so a subprocess can never read/write a live user's config
      or data.
    * ``config_dir`` — the exact ``user_config_path`` the CLI resolves under
      those overrides, where ``bpm_settings.json`` must be written.

    ``platform`` is injectable for pure tests of the Windows branch on Linux;
    it defaults to ``sys.platform``.
    """
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        # Non-roaming (roaming=False) config/data roots resolve under
        # CSIDL_LOCAL_APPDATA. platformdirs honours WIN_PD_OVERRIDE_LOCAL_APPDATA
        # as a portable override for that root (checked before ctypes/registry),
        # so no runner user path is hardcoded.
        local = tmp_path / "localappdata"
        return {"WIN_PD_OVERRIDE_LOCAL_APPDATA": str(local)}, local / "ZeSoftware" / "ZeCalibrator"
    # POSIX (Linux/macOS): XDG config/data roots; appauthor is not used.
    return {
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_DATA_HOME": str(tmp_path / "xdg-data"),
    }, tmp_path / "xdg-config" / "ZeCalibrator"


def _env(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Redirect ZeCalibrator's platformdirs config/data roots into the tmp dir
    # so the CLI reads the persisted BPM setting from an isolated location.
    overrides, _config_dir = _isolated_storage(tmp_path)
    env.update(overrides)
    return env


def run_cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-m", "zecalibrator", *args],
        capture_output=True, text=True, env=_env(tmp_path),
    )


def _write_fits(path, value, bunit="ADU"):
    hdu = fits.PrimaryHDU(np.full(SHAPE, value, dtype=np.float32))
    hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)


def _npy_bytes(arr):
    import io

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _setup_library(tmp_path):
    dark = tmp_path / "dark.fits"
    _write_fits(dark, 10.0)
    (tmp_path / "dark.mask.npy").write_bytes(_npy_bytes(np.zeros(SHAPE, dtype=np.uint16)))
    light = tmp_path / "light.fits"
    _write_fits(light, 100.0)
    decl_path = tmp_path / "light.decl.json"
    decl_path.write_text(json.dumps(DECL))
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(ROI))
    imports_path = tmp_path / "imports.json"
    imports_path.write_text(json.dumps(
        [dict(path="dark.fits", master_type="dark", hdu=0,
              mask_path="dark.mask.npy", bias_state="included", declaration=DECL)]
    ))
    idx = run_cli(tmp_path, "index", str(tmp_path), "--declarations", "@" + str(imports_path))
    assert idx.returncode == 0, idx.stderr
    return str(light), str(decl_path), str(roi_path), str(tmp_path / "zecalibrator.library.sqlite")


def _calibrate(tmp_path, light, decl, roi, index, outdir, *extra):
    return run_cli(
        tmp_path, "calibrate", light, "--library", index, "--additive-mode",
        "dark_incl_bias", "--declaration", "@" + decl, "--roi-extent", "@" + roi,
        "--destination", str(outdir), "--batch-id", "b1", *extra,
    )


def test_calibrate_emits_bpm_synthesis_to_stderr(tmp_path):
    light, decl, roi, index = _setup_library(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    proc = _calibrate(tmp_path, light, decl, roi, index, outdir)
    assert proc.returncode == 0, proc.stderr
    # stdout stays scriptable JSON.
    data = json.loads(proc.stdout)
    assert data["status"] == "COMPLETED"
    # LOT 3 §11 run synthesis on stderr (never corrupts stdout JSON).
    lines = proc.stderr.splitlines()
    assert "Bad Pixel Database:" in proc.stderr
    assert "Bad Pixel Map: selected" in proc.stderr
    assert "Bad pixels: 0" in proc.stderr
    assert "BPM correction: none" in proc.stderr
    assert "Reconstructed sites: 0" in proc.stderr
    # §51: no alarming "ERROR: BPM failed" when everything is fine.
    assert "ERROR: BPM failed" not in proc.stderr


def test_calibrate_bad_pixel_db_override_is_accepted_and_reported(tmp_path):
    light, decl, roi, index = _setup_library(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    root = tmp_path / "my-base"
    root.mkdir()
    proc = _calibrate(tmp_path, light, decl, roi, index, outdir, "--bad-pixel-db", str(root))
    assert proc.returncode == 0, proc.stderr
    assert f"Bad Pixel Database: {root}" in proc.stderr


def test_calibrate_persisted_config_is_automatic(tmp_path):
    light, decl, roi, index = _setup_library(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    # Persist the setting where the CLI resolves it (platformdirs config root).
    _, config_dir = _isolated_storage(tmp_path)
    config_dir.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "persisted-base"
    base.mkdir()
    config_dir.joinpath("bpm_settings.json").write_text(
        json.dumps({"schema_version": 1, "bad_pixel_database_root": str(base)}),
        encoding="utf-8",
    )
    proc = _calibrate(tmp_path, light, decl, roi, index, outdir)
    assert proc.returncode == 0, proc.stderr
    assert f"Bad Pixel Database: {base}" in proc.stderr


def test_isolated_storage_windows_branch_uses_localappdata_override(tmp_path):
    overrides, config_dir = _isolated_storage(tmp_path, platform="win32")
    assert overrides == {"WIN_PD_OVERRIDE_LOCAL_APPDATA": str(tmp_path / "localappdata")}
    # Non-roaming PlatformDirs(appname, appauthor, roaming=False) config root:
    # <LOCALAPPDATA>\ZeSoftware\ZeCalibrator — never a live Windows user path.
    assert config_dir == tmp_path / "localappdata" / "ZeSoftware" / "ZeCalibrator"
    assert "Users" not in str(config_dir)


def test_isolated_storage_posix_branch_uses_xdg(tmp_path):
    overrides, config_dir = _isolated_storage(tmp_path, platform="linux")
    assert overrides == {
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_DATA_HOME": str(tmp_path / "xdg-data"),
    }
    assert config_dir == tmp_path / "xdg-config" / "ZeCalibrator"


def test_calibrate_without_destination_still_synthesizes(tmp_path):
    light, decl, roi, index = _setup_library(tmp_path)
    proc = run_cli(
        tmp_path, "calibrate", light, "--library", index, "--additive-mode",
        "dark_incl_bias", "--declaration", "@" + decl, "--roi-extent", "@" + roi,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "COMPLETED"
    assert "Bad Pixel Database:" in proc.stderr


def test_cli_has_no_force_bpm_flag():
    parser = cli.build_parser()
    # The calibrate subparser must expose no force/enable BPM flag (§74).
    sub = next(a for a in parser._actions if a.dest == "command")
    names = {name.lstrip("-") for action in sub.choices["calibrate"]._actions for name in action.option_strings}
    for bad in ("force-bpm", "enable-bpm", "apply-bpm", "bpm-force"):
        assert bad not in names, f"forbidden BPM flag {bad!r} present"
