"""Independent validation that the hand-written FITS writer is standard-conformant.

Runs in a SUBPROCESS which imports the FITS library and reads back the bytes
produced by :func:`tests.gui.conftest.fits_bytes`. The FITS library never runs on
the GUI main thread; the subprocess is the independent oracle (mirroring the
byte-level + FITS read-back checks). It asserts BITPIX, NAXIS, NAXIS1/2/3,
dtype, shape, BUNIT and exact pixel values for a representative set: int16 2x2,
float32 4x4, int16 3x2x2, plus IMAGETYP/EXPTIME/GAIN/HISTORY header cards.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_REPO = Path(__file__).resolve().parents[2]
_SRC = str(_REPO / "src")
_TESTS = str(_REPO / "tests")

_ROUNDTRIP_SCRIPT = r'''
import io
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, sys.argv[1])  # src
sys.path.insert(0, sys.argv[2])  # repo root (for tests.gui.conftest)

from tests.gui.conftest import fits_bytes


def read_back(raw):
    return fits.open(io.BytesIO(raw), memmap=False)


# --- int16 2x2 with header cards --------------------------------------
a = np.array([[1, 2], [3, 4]], dtype=np.int16)
raw = fits_bytes(a, header_cards=[
    ("IMAGETYP", "DARK"), ("EXPTIME", 300.0), ("GAIN", 100.0),
    ("HISTORY", "median stack of frames"),
], bunit="ADU")
with read_back(raw) as hdul:
    h = hdul[0].header
    assert h.get("BITPIX") == 16, h.get("BITPIX")
    assert h.get("NAXIS") == 2, h.get("NAXIS")
    assert h.get("NAXIS1") == 2 and h.get("NAXIS2") == 2
    assert hdul[0].data.dtype == np.dtype(">i2"), hdul[0].data.dtype
    assert hdul[0].data.shape == (2, 2)
    assert h.get("BUNIT") == "ADU"
    assert h.get("IMAGETYP") == "DARK"
    assert float(h.get("EXPTIME")) == 300.0
    assert float(h.get("GAIN")) == 100.0
    assert h.get("HISTORY") == "median stack of frames"
    assert np.array_equal(hdul[0].data, a)

# --- float32 4x4 ------------------------------------------------------
b = np.full((4, 4), 100.0, dtype=np.float32)
raw = fits_bytes(b, bunit="ADU")
with read_back(raw) as hdul:
    h = hdul[0].header
    assert h.get("BITPIX") == -32, h.get("BITPIX")
    assert h.get("NAXIS") == 2
    assert h.get("NAXIS1") == 4 and h.get("NAXIS2") == 4
    assert hdul[0].data.dtype == np.dtype(">f4"), hdul[0].data.dtype
    assert hdul[0].data.shape == (4, 4)
    assert np.array_equal(hdul[0].data, b)

# --- int16 3x2x2 (NAXIS=3) -------------------------------------------
c = np.arange(12, dtype=np.int16).reshape(3, 2, 2)
raw = fits_bytes(c, bunit="ADU")
with read_back(raw) as hdul:
    h = hdul[0].header
    assert h.get("BITPIX") == 16
    assert h.get("NAXIS") == 3
    assert h.get("NAXIS1") == 2 and h.get("NAXIS2") == 2 and h.get("NAXIS3") == 3
    assert hdul[0].data.shape == (3, 2, 2)
    assert np.array_equal(hdul[0].data, c)

print("ROUNDTRIP_OK")
sys.exit(0)
'''


def _env():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def test_fits_writer_roundtrip_subprocess():
    proc = subprocess.run(
        [sys.executable, "-c", _ROUNDTRIP_SCRIPT, _SRC, _REPO],
        capture_output=True, text=True, env=_env(), timeout=120,
    )
    assert proc.returncode == 0, (
        f"roundtrip subprocess exited rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "ROUNDTRIP_OK" in proc.stdout
