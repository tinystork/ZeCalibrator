"""Strict raw FITS decoder witnesses (SCIENCE §2, ASTRA §4.1)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from astropy.io import fits

from zecalibrator.core.dq import INPUT_INVALID
from zecalibrator.core.errors import DecodeError, PrecisionRefusalError
from zecalibrator.core.metadata import ImportDeclaration
from zecalibrator.io.raw_decoder import decode_fits


def _decl(**overrides) -> ImportDeclaration:
    base = dict(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU",
        detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-MONO",
        gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16",
        binning=(1, 1), orientation="identity", cfa_phase="mono", roi_origin=(0, 0),
        exposure_s=10.0, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.1,
        saturation_limit_adu=60000, saturation_evidence="qualified",
    )
    base.update(overrides)
    return ImportDeclaration(**base)


def _write(path, data, header_cards=None, with_bunit=True):
    hdu = fits.PrimaryHDU(data)
    if with_bunit:
        hdu.header["BUNIT"] = "ADU"
    for key, val in (header_cards or []):
        hdu.header[key] = val
    hdu.writeto(path, overwrite=True)
    return path


def _write_header(path, data, header):
    fits.writeto(path, data, header, overwrite=True)
    return path


def test_unsigned_bzero(tmp_path):
    u16 = np.array([[0, 100], [200, 65535]], dtype=np.uint16)
    p = _write(tmp_path / "u16.fits", u16)
    f = decode_fits(p, declaration=_decl())
    assert f.data.dtype == np.float32
    assert np.array_equal(f.data, u16.astype(np.float32))
    assert f.bzero == 32768.0
    assert np.count_nonzero(f.mask) == 0
    assert f.stored_dtype == ">i2"


def test_nontrivial_integer_scaling(tmp_path):
    stored = np.array([[1, 2], [3, 4]], dtype=np.int16)
    p = _write(tmp_path / "iscale.fits", stored, [("BSCALE", 2), ("BZERO", 10)])
    f = decode_fits(p, declaration=_decl())
    assert np.array_equal(f.data, (2 * stored + 10).astype(np.float32))


def test_nontrivial_float_scaling(tmp_path):
    stored = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    p = _write(tmp_path / "fscale.fits", stored, [("BSCALE", 2.0), ("BZERO", 10.0)])
    f = decode_fits(p, declaration=_decl())
    assert np.allclose(f.data, [[11.0, 13.0], [15.0, 17.0]])


def test_integer_blank_stored_space(tmp_path):
    stored = np.array([[-32768, -32668], [-32568, 32767]], dtype=np.int16)
    p = _write(tmp_path / "blank.fits", stored, [("BZERO", 32768), ("BLANK", -32768)])
    f = decode_fits(p, declaration=_decl())
    assert f.mask[0, 0] & INPUT_INVALID
    assert np.isnan(f.data[0, 0])
    assert f.data[0, 1] == 100.0
    assert f.data[1, 1] == 65535.0


def test_fractional_blank_rejected(tmp_path):
    stored = np.array([[-32768, -32668], [-32568, 32767]], dtype=np.int16)
    p = _write(tmp_path / "blankfrac.fits", stored, [("BZERO", 32768), ("BLANK", 1.5)])
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "MALFORMED_CARD"


def test_nan_inf_invalid(tmp_path):
    stored = np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    p = _write(tmp_path / "naninf.fits", stored)
    f = decode_fits(p, declaration=_decl())
    assert f.mask[0, 1] & INPUT_INVALID
    assert f.mask[1, 0] & INPUT_INVALID
    assert f.mask[1, 1] & INPUT_INVALID
    assert np.isnan(f.data[0, 1])
    assert np.isnan(f.data[1, 0])
    assert f.data[0, 0] == 1.0


def test_big_endian_byte_order(tmp_path):
    data = np.array([[10, 20], [30, 40]], dtype=">i2")
    p = _write(tmp_path / "be.fits", data, [("BZERO", 32768)])
    f = decode_fits(p, declaration=_decl())
    assert f.data.dtype == np.float32
    assert np.array_equal(f.data, (data.astype(np.int32) + 32768).astype(np.float32))


def test_malformed_numeric_card(tmp_path):
    p = _write(tmp_path / "malformed.fits", np.zeros((2, 2), dtype=np.int16))
    with fits.open(p, mode="update") as hdul:
        hdul[0].header["EXPTIME"] = "abc"
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "MALFORMED_CARD"


def test_duplicate_conflicting_aliases(tmp_path):
    p = _write(tmp_path / "conflict.fits", np.zeros((2, 2), dtype=np.int16))
    with fits.open(p, mode="update") as hdul:
        hdul[0].header["EXPTIME"] = 20.0
        hdul[0].header.append(fits.Card("EXPOSURE", 10.0))
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "ALIAS_CONFLICT"


def test_duplicate_conflicting_gain(tmp_path):
    hdr = fits.Header()
    hdr["BUNIT"] = "ADU"
    hdr["GAIN"] = 80
    hdr.append(("GAIN", 200))
    p = _write_header(tmp_path / "gain.fits", np.zeros((2, 2), dtype=np.int16), hdr)
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "ALIAS_CONFLICT"


def test_duplicate_conflicting_bzero_at_card_level():
    from zecalibrator.core.metadata import CardRecord, resolve_aliases

    cards = (
        CardRecord("BUNIT", "ADU", "", 0, "primary"),
        CardRecord("BZERO", 0, "", 1, "primary"),
        CardRecord("BZERO", 100, "", 2, "primary"),
    )
    _, conflicts, _ = resolve_aliases(cards)
    assert any(c.field == "bzero" for c in conflicts)


def test_numeric_equivalent_aliases_agree(tmp_path):
    p = _write(tmp_path / "agree.fits", np.zeros((2, 2), dtype=np.int16))
    with fits.open(p, mode="update") as hdul:
        hdul[0].header["EXPTIME"] = 1
        hdul[0].header.append(fits.Card("EXPOSURE", 1.0))
    f = decode_fits(p, declaration=_decl(exposure_s=1.0))
    assert f.metadata.exposure_s == 1.0


def test_duplicate_agreeing_aliases_preserved(tmp_path):
    p = _write(tmp_path / "agree2.fits", np.zeros((2, 2), dtype=np.int16))
    with fits.open(p, mode="update") as hdul:
        hdul[0].header["EXPTIME"] = 20.0
        hdul[0].header.append(fits.Card("EXPOSURE", 20.0))
    f = decode_fits(p, declaration=_decl(exposure_s=20.0))
    assert f.metadata.exposure_s == 20.0
    cards = [c for c in f.metadata.original_cards if c.keyword in ("EXPTIME", "EXPOSURE")]
    assert len(cards) == 2


def test_hdu_ambiguity_requires_explicit_hdu(tmp_path):
    p = tmp_path / "multi.fits"
    primary = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.int16))
    primary.header["BUNIT"] = "ADU"
    second = fits.ImageHDU(np.ones((2, 2), dtype=np.int16))
    second.header["BUNIT"] = "ADU"
    fits.HDUList([primary, second]).writeto(p, overwrite=True)
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "HDU_AMBIGUOUS"
    f = decode_fits(p, hdu=1, declaration=_decl())
    assert np.array_equal(f.data, np.ones((2, 2), dtype=np.float32))
    assert f.hdu == 1
    assert all(c.source == "hdu:1" for c in f.metadata.original_cards)


def test_rgb_rejection(tmp_path):
    cube = np.zeros((3, 2, 2), dtype=np.int16)
    p = _write(tmp_path / "rgb.fits", cube)
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "RGB_UNSUPPORTED"


def test_processed_history_rejection(tmp_path):
    p = _write(tmp_path / "processed.fits", np.zeros((2, 2), dtype=np.int16))
    with fits.open(p, mode="update") as hdul:
        hdul[0].header["HISTORY"] = "frame was calibrated and normalized"
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "PROCESSED_HISTORY"


def test_non_raw_bunit_rejection(tmp_path):
    p = _write(tmp_path / "normalized.fits", np.zeros((2, 2), dtype=np.int16), with_bunit=False)
    with fits.open(p, mode="update") as hdul:
        hdul[0].header["BUNIT"] = "normalized"
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "UNITS_UNSUPPORTED"


def test_jy_units_rejected(tmp_path):
    p = _write(tmp_path / "jy.fits", np.zeros((2, 2), dtype=np.int16), with_bunit=False)
    with fits.open(p, mode="update") as hdul:
        hdul[0].header["BUNIT"] = "Jy"
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "UNITS_UNSUPPORTED"


def test_plain_2d_without_units_rejected(tmp_path):
    p = _write(tmp_path / "plain.fits", np.zeros((2, 2), dtype=np.int16), with_bunit=False)
    with pytest.raises(DecodeError) as exc:
        decode_fits(p)
    assert exc.value.reason_code in ("UNKNOWN_UNITS", "UNKNOWN_DOMAIN")


def test_adu_without_declaration_rejected(tmp_path):
    # BUNIT=ADU alone is not raw-domain evidence (SCIENCE §2.7).
    p = _write(tmp_path / "aduonly.fits", np.zeros((2, 2), dtype=np.int16))
    with pytest.raises(DecodeError) as exc:
        decode_fits(p)
    assert exc.value.reason_code == "UNKNOWN_DOMAIN"


def test_units_supplied_by_declaration(tmp_path):
    p = _write(tmp_path / "decl.fits", np.zeros((2, 2), dtype=np.int16), with_bunit=False)
    f = decode_fits(p, declaration=_decl())
    assert f.metadata.units == "ADU"
    assert f.metadata.raw_domain_declaration == "raw"
    assert f.metadata.gain == 100
    assert f.metadata.detector_instance_id == "SYNTH-DET-0001"


def test_declaration_cannot_override_contrary_fits(tmp_path):
    p = _write(tmp_path / "contrary.fits", np.zeros((2, 2), dtype=np.int16), with_bunit=False)
    with fits.open(p, mode="update") as hdul:
        hdul[0].header["BUNIT"] = "Jy"
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "UNITS_UNSUPPORTED"


def test_contradictory_gain_declaration_rejected(tmp_path):
    p = _write(tmp_path / "gainconflict.fits", np.zeros((2, 2), dtype=np.int16), [("GAIN", 80)])
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl(gain=200))
    assert exc.value.reason_code == "DECLARATION_CONFLICT"


def test_contradictory_binning_declaration_rejected(tmp_path):
    p = _write(tmp_path / "binconflict.fits", np.zeros((2, 2), dtype=np.int16), [("XBINNING", 1), ("YBINNING", 1)])
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl(binning=(2, 2)))
    assert exc.value.reason_code == "DECLARATION_CONFLICT"


def test_processed_domain_declaration_rejected():
    with pytest.raises(ValueError):
        ImportDeclaration("s", "i", "v", domain="processed")


def test_electron_units_declaration_rejected():
    with pytest.raises(ValueError):
        ImportDeclaration("s", "i", "v", units="electron")


def test_empty_identity_declaration_rejected():
    with pytest.raises(ValueError):
        ImportDeclaration("", "", "")


def test_large_integer_refusal(tmp_path):
    stored = np.array([[2**24 + 1, 0]], dtype=np.int64)
    p = _write(tmp_path / "big.fits", stored)
    with pytest.raises(PrecisionRefusalError):
        decode_fits(p, declaration=_decl())


def test_source_file_not_modified(tmp_path):
    u16 = np.array([[0, 100], [200, 65535]], dtype=np.uint16)
    p = _write(tmp_path / "immutable.fits", u16)
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    decode_fits(p, declaration=_decl())
    after = hashlib.sha256(p.read_bytes()).hexdigest()
    assert before == after


def test_decoder_emits_progress(tmp_path):
    from zecalibrator.application.cancellation import ProgressObserver

    u16 = np.array([[0, 100], [200, 65535]], dtype=np.uint16)
    p = _write(tmp_path / "prog.fits", u16)
    events = []
    obs = ProgressObserver(events.append)
    decode_fits(p, declaration=_decl(), progress=obs)
    assert len(events) >= 2
    assert events[0].phase == "read"
    assert events[-1].phase == "complete"


def test_decoder_bare_observer_isolated(tmp_path):
    u16 = np.array([[0, 100], [200, 65535]], dtype=np.uint16)
    p = _write(tmp_path / "bare.fits", u16)

    def _boom(event):
        raise RuntimeError("observer bug")

    # A bare raising callback must not fail the decode (G).
    f = decode_fits(p, declaration=_decl(), progress=_boom)
    assert f.metadata.units == "ADU"
