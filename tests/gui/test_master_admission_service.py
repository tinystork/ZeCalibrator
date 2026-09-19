"""Service-level master-admission witnesses (headless, no PySide6).

Covers the frozen IMAGETYP role map, the master-admissibility detector (reports
*why* a master is incompatible), role-conflict detection, and the unchanged
evidence-candidate detection. Mixed-folder scan-once behavior is exercised via
:func:`scan_folder_inputs` + :func:`scan_master_header` (each file gets one role,
never triple-indexed).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from zecalibrator.gui import service


def _write(path, *, imagetyp=None, history=None, naxis=2, extra=()):
    if naxis == 3:
        data = np.zeros((3, 2, 2), dtype=np.int16)
    else:
        data = np.zeros((2, 2), dtype=np.int16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BUNIT"] = "ADU"
    hdu.header["EXPTIME"] = 300.0
    hdu.header["CCD-TEMP"] = 20.0
    hdu.header["GAIN"] = 100.0
    hdu.header["OFFSET"] = 50.0
    hdu.header["INSTRUME"] = "SYNTH-CFA"
    hdu.header["FILTER"] = "L"
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    if imagetyp is not None:
        hdu.header["IMAGETYP"] = imagetyp
    if history is not None:
        hdu.header["HISTORY"] = history
    for key, val in extra:
        hdu.header[key] = val
    hdu.writeto(path, overwrite=True)
    return str(path)


def _cards(path):
    src = __import__("zecalibrator.api.v1", fromlist=["FilesystemSource"]).FilesystemSource()
    return [(c.keyword, c.value) for c in src.read_header(path, hdu=0)]


# ---------------------------------------------------------------------------
# Evidence cards (unchanged) + IMAGETYP role candidate
# ---------------------------------------------------------------------------
def test_evidence_cards_produce_expected_candidates(tmp_path):
    p = _write(tmp_path / "dark.fits", imagetyp="DARK")
    cards = _cards(p)
    candidates, conflicts = service.detect_header_candidates(cards)
    assert conflicts == {}
    assert candidates["exposure_s"].value == 300.0
    assert candidates["temperature_c"].value == 20.0
    assert candidates["gain"].value == 100.0
    assert candidates["offset"].value == 50.0
    assert candidates["detector_model"].value == "SYNTH-CFA"
    assert candidates["filter"].value == "L"
    assert candidates["binning"].value == (1, 1)
    # IMAGETYP is a role candidate, not an evidence field.
    assert "imagetyp" not in candidates
    assert service.detect_imagetyp_role(cards) == "dark"


@pytest.mark.parametrize("imagetyp,role", [
    ("DARK", "dark"),
    ("BIAS", "bias"),
    ("FLAT", "flat"),
    ("DARKFLAT", "flat_dark"),
])
def test_imagetyp_role_map(tmp_path, imagetyp, role):
    p = _write(tmp_path / f"{role}.fits", imagetyp=imagetyp)
    assert service.detect_imagetyp_role(_cards(p)) == role


def test_imagetyp_absent_or_unknown_yields_no_role(tmp_path):
    p = _write(tmp_path / "no_role.fits")
    assert service.detect_imagetyp_role(_cards(p)) is None
    p2 = _write(tmp_path / "unknown_role.fits", imagetyp="LIGHT")
    assert service.detect_imagetyp_role(_cards(p2)) is None


# ---------------------------------------------------------------------------
# Master admissibility detector (reports *why* incompatible)
# ---------------------------------------------------------------------------
def test_rgb_naaxis3_reports_incompatible_and_retains_metadata(tmp_path):
    p = _write(tmp_path / "rgb.fits", naxis=3)
    entry = service.scan_master_header(p)
    # Detected metadata is retained; incompatibility is explicit.
    assert entry["status"] == "COMPLETED"
    assert entry["candidates"]["exposure_s"]["value"] == 300.0
    assert entry["admissible"] is False
    assert any("RGB / debayered" in r for r in entry["incompatible"])


def test_debayered_master_incompatible(tmp_path):
    p = _write(tmp_path / "debayer.fits", imagetyp="DARK", history="debayer RGB output")
    entry = service.scan_master_header(p)
    assert entry["detected_role"] == "dark"
    assert entry["admissible"] is False
    assert any("debayer" in r for r in entry["incompatible"])


def test_stacked_master_admissible(tmp_path):
    p = _write(tmp_path / "stacked.fits", imagetyp="DARK", history="median stack of frames")
    entry = service.scan_master_header(p)
    assert entry["detected_role"] == "dark"
    assert entry["admissible"] is True
    assert entry["incompatible"] == []


def test_dark_calibrated_incompatible(tmp_path):
    p = _write(tmp_path / "dark_cal.fits", imagetyp="DARK", history="calibrated dark")
    entry = service.scan_master_header(p)
    assert entry["admissible"] is False
    assert any("calibrated" in r for r in entry["incompatible"])


def test_flat_calibrated_form_aware(tmp_path):
    p = _write(tmp_path / "flat_cal.fits", imagetyp="FLAT", history="calibrated flat")
    cards = _cards(p)
    # Flat with matching flat_form is admissible.
    assert service.master_incompatibility(cards, role="flat", flat_form="corrected_unnormalized") == ()
    # Flat with raw_response form is not.
    reasons = service.master_incompatibility(cards, role="flat", flat_form="raw_response")
    assert reasons and any("flat_form" in r for r in reasons)


# ---------------------------------------------------------------------------
# Role conflict: user-selected vs detected (never auto-resolved)
# ---------------------------------------------------------------------------
def test_role_conflict_user_bias_vs_header_darkflat(tmp_path):
    p = _write(tmp_path / "conflict.fits", imagetyp="DARKFLAT")
    entry = service.scan_master_header(p, selected_role="bias")
    conflict = entry["conflict"]
    assert conflict is not None
    assert conflict["selected_role"] == "bias"
    assert conflict["detected_role"] == "flat_dark"
    assert conflict["source"] == "FITS IMAGETYP=DARKFLAT"
    assert conflict["needs_confirmation"] is True


def test_role_conflict_absent_when_agreeing_or_unknown(tmp_path):
    p = _write(tmp_path / "dark.fits", imagetyp="DARK")
    assert service.detect_role_conflict("dark", "dark") is None
    assert service.detect_role_conflict("dark", None) is None
    assert service.detect_role_conflict(None, "dark") is None


# ---------------------------------------------------------------------------
# Mixed folder: scanned once, each file gets one role (never triple-indexed)
# ---------------------------------------------------------------------------
def test_mixed_folder_scanned_once_single_role_per_file(tmp_path):
    for name, imagetyp in (
        ("dark.fits", "DARK"),
        ("flat_dark.fits", "DARKFLAT"),
        ("flat.fits", "FLAT"),
    ):
        _write(tmp_path / name, imagetyp=imagetyp)
    paths, _ = service.scan_folder_inputs(str(tmp_path))
    assert len(paths) == 3  # scanned once
    roles = []
    for path in paths:
        entry = service.scan_master_header(path)
        assert entry["detected_role"] is not None
        roles.append(entry["detected_role"])
    # Each file yields exactly one role; no file is triple-indexed.
    assert sorted(roles) == ["dark", "flat", "flat_dark"]


# ---------------------------------------------------------------------------
# F2: explicit normalization/calibration negation semantics (Siril HISTORY)
# ---------------------------------------------------------------------------
def test_unnormalized_input_and_output_are_admissible(tmp_path):
    for hist in ("unnormalized input", "unnormalized output"):
        p = _write(tmp_path / "u.fits", imagetyp="DARK", history=hist)
        cards = _cards(p)
        assert service.master_incompatibility(cards, role="dark") == ()


def test_multiplicative_normalized_input_unnormalized_output_admissible(tmp_path):
    # Real Siril flat HISTORY: not a normalized output -> admissible.
    p = _write(
        tmp_path / "siril.fits", imagetyp="FLAT",
        history="multiplicative normalized input, unnormalized output",
    )
    cards = _cards(p)
    assert service.master_incompatibility(cards, role="flat", flat_form="raw_response") == ()


def test_uncalibrated_is_admissible(tmp_path):
    p = _write(tmp_path / "uc.fits", imagetyp="DARK", history="uncalibrated dark")
    cards = _cards(p)
    assert service.master_incompatibility(cards, role="dark") == ()


def test_normalized_output_still_signals_normalized(tmp_path):
    p = _write(tmp_path / "n.fits", imagetyp="DARK", history="normalized output")
    cards = _cards(p)
    reasons = service.master_incompatibility(cards, role="dark")
    assert reasons and any("normalized" in r for r in reasons)


# ---------------------------------------------------------------------------
# Three real-Siril-equivalent 2-D CFA headers + role detection
# ---------------------------------------------------------------------------
def test_three_siril_equivalent_cfa_headers_and_roles(tmp_path):
    def _mk(name, **header):
        hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.int16))
        hdu.header["BUNIT"] = "ADU"
        for k, v in header.items():
            hdu.header[k] = v
        p = tmp_path / name
        hdu.writeto(p, overwrite=True)
        return str(p)

    # dark: NAXIS=2 / IMAGETYP=DARK / EXPTIME / BAYERPAT=RGGB.
    dark = _mk("dark.fits", IMAGETYP="DARK", EXPTIME=300.0, BAYERPAT="RGGB")
    # bias-as-DARKFLAT: NAXIS=2 / IMAGETYP=DARKFLAT / CCD-TEMP / GAIN / OFFSET / BAYERPAT.
    fd = _mk(
        "fd.fits", IMAGETYP="DARKFLAT",
        **{"CCD-TEMP": 20.0, "GAIN": 100.0, "OFFSET": 50.0, "BAYERPAT": "RGGB"},
    )
    # flat: NAXIS=2 / IMAGETYP=FLAT / FILTER / BAYERPAT.
    flat = _mk("flat.fits", IMAGETYP="FLAT", FILTER="L", BAYERPAT="RGGB")

    # DARK/DARKFLAT/FLAT role detection (exactly one role each).
    assert service.detect_imagetyp_role(_cards(dark)) == "dark"
    assert service.detect_imagetyp_role(_cards(fd)) == "flat_dark"
    assert service.detect_imagetyp_role(_cards(flat)) == "flat"

    # CFA evidence is detected from BAYERPAT (never a role candidate).
    c, _ = service.detect_header_candidates(_cards(dark))
    assert c["cfa_phase"].value == "RGGB"
    assert c["exposure_s"].value == 300.0

    # All three are admissible raw 2-D CFA masters (no processed markers).
    assert service.master_incompatibility(_cards(dark), role="dark") == ()
    assert service.master_incompatibility(_cards(fd), role="flat_dark") == ()
    assert service.master_incompatibility(_cards(flat), role="flat") == ()
