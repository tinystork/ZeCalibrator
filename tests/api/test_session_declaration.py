"""C15 — session-level orientation declaration (fallback-only, traced).

Option 3: an explicit user-provided ``SessionDeclaration`` on
``open_session_library`` supplies ``orientation="identity"`` **only** when the
master's own header carries no exploitable orientation evidence.  The retained
source is reported per admission (``header`` vs ``declared``) and the declaration
is echoed on the result so a resume can never silently change the hypothesis.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1


def _bayer_dark(path, *, orientation=None):
    rng = np.random.default_rng(20260927)
    data = np.clip(rng.normal(200, 4, size=(40, 32)), 0, 65535).astype(np.int16)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["IMAGETYP"] = "Dark"
    h["EXPTIME"] = 60.0
    h["INSTRUME"] = "ZWO ASI294MC Pro"
    h["GAIN"] = 120.0
    h["OFFSET"] = 30.0
    h["CCD-TEMP"] = -10.0
    h["SET-TEMP"] = -10.0
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["XORGSUBF"] = 0
    h["YORGSUBF"] = 0
    h["BAYERPAT"] = "RGGB"
    h["BUNIT"] = "ADU"
    if orientation is not None:
        h["ORIENTATION"] = orientation
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def _bayer_flat(path):
    rng = np.random.default_rng(7)
    data = np.clip(rng.normal(20000, 400, size=(40, 32)), 0, 65535).astype(np.int16)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["IMAGETYP"] = "Flat"
    h["EXPTIME"] = 5.0
    h["INSTRUME"] = "ZWO ASI294MC Pro"
    h["GAIN"] = 120.0
    h["OFFSET"] = 30.0
    h["CCD-TEMP"] = -10.0
    h["SET-TEMP"] = -10.0
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["XORGSUBF"] = 0
    h["YORGSUBF"] = 0
    h["BAYERPAT"] = "RGGB"
    h["FILTER"] = "L"
    h["BUNIT"] = "ADU"
    hdu.writeto(str(path), overwrite=True)
    return str(path)


# ---------------------------------------------------------------------------
# 1. header evidence wins -> source "header", declaration ignored
# ---------------------------------------------------------------------------
def test_header_orientation_wins_source_header(tmp_path):
    _bayer_dark(tmp_path / "dark.fits", orientation="identity")
    res = v1.open_session_library(
        str(tmp_path), declaration=v1.SessionDeclaration(orientation="identity")
    )
    assert len(res.admissions) == 1
    adm = res.admissions[0]
    assert adm.orientation_source == "header"
    assert adm.declaration.orientation == "identity"


# ---------------------------------------------------------------------------
# 2. no header evidence + declaration -> admitted, source "declared"
# ---------------------------------------------------------------------------
def test_declaration_fallback_source_declared(tmp_path):
    _bayer_dark(tmp_path / "dark.fits", orientation=None)
    res = v1.open_session_library(
        str(tmp_path), declaration=v1.SessionDeclaration(orientation="identity")
    )
    assert len(res.admissions) == 1
    adm = res.admissions[0]
    assert adm.orientation_source == "declared"
    assert adm.declaration.orientation == "identity"
    assert res.declaration is not None
    assert res.declaration.orientation == "identity"


# ---------------------------------------------------------------------------
# 3. no header evidence + no declaration -> refused (unchanged behaviour)
# ---------------------------------------------------------------------------
def test_no_declaration_refused(tmp_path):
    _bayer_dark(tmp_path / "dark.fits", orientation=None)
    res = v1.open_session_library(str(tmp_path))
    assert len(res.admissions) == 0
    assert res.rejected[0].reason_code == "MISSING_REQUIRED_FIELDS"
    assert "orientation" in res.rejected[0].detail


# ---------------------------------------------------------------------------
# 4. declaration cannot override a present + different orientation
# ---------------------------------------------------------------------------
def test_declaration_cannot_override_different_orientation(tmp_path):
    _bayer_dark(tmp_path / "dark.fits", orientation="flipped")
    res = v1.open_session_library(
        str(tmp_path), declaration=v1.SessionDeclaration(orientation="identity")
    )
    assert len(res.admissions) == 0
    assert res.rejected[0].reason_code == "INCOMPATIBLE"
    assert "flipped" in res.rejected[0].detail


# ---------------------------------------------------------------------------
# 5. C24: flat quality (R4) is informational, not an admission filter
# ---------------------------------------------------------------------------
def test_r4_flat_admitted_with_needs_attention(tmp_path):
    _bayer_flat(tmp_path / "flat.fits")
    res = v1.open_session_library(
        str(tmp_path), declaration=v1.SessionDeclaration(orientation="identity")
    )
    # C24: flat is ADMITTED (indexed); R4 quality evidence is informational
    # needs_attention, resolved as traced UNVERIFIED by the Standard matcher at
    # route time (never silently converted to qualified).
    assert len(res.admissions) == 1
    adm = res.admissions[0]
    assert adm.role == "flat"
    assert any("flat quality evidence" in r for r in adm.needs_attention)


# ---------------------------------------------------------------------------
# surface: invalid declaration rejected
# ---------------------------------------------------------------------------
def test_invalid_declaration_orientation_rejected():
    with pytest.raises(ValueError):
        v1.SessionDeclaration(orientation="flipped")
