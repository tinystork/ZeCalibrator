"""Bootstrap tests: CLI --help / --version and no advertised capability."""

from __future__ import annotations

import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "zecalibrator", *args],
        capture_output=True,
        text=True,
    )


def test_version():
    proc = run_cli("--version")
    assert proc.returncode == 0
    assert "zecalibrator 0.1.0" in proc.stdout


def test_help():
    proc = run_cli("--help")
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()


def test_no_args_prints_help():
    proc = run_cli()
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()


def test_help_reflects_current_state_not_bootstrap():
    proc = run_cli("--help")
    out = " ".join((proc.stdout + proc.stderr).lower().split())
    # The help no longer describes the product as a bootstrap skeleton; the
    # accepted calibration engine is described. Capability identifier keywords
    # are not exposed.
    assert "bootstrap skeleton" not in out
    assert "calibration" in out
    for keyword in ("calibrate_frame", "calibrate_batch", "gpu", "cuda"):
        assert keyword not in out
