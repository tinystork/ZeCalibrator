"""Master-admission decoder witnesses (P7-M3B rework §7-§9).

The additive ``admission="master"`` context applies a role/form/provenance-aware
processed-history policy WITHOUT weakening the light decoder (``"light"`` default
is byte-for-byte unchanged). A master is by definition a raw 2-D sensor-domain
stack/combination, so stacked/median/mean/rejection/combination markers are
admitted; debayer/demosaic/RGB/stretch/white-balance/resample/display markers are
always rejected; calibrated/normalized are role/form aware.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from zecalibrator.core.errors import DecodeError
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


def _write_with_history(tmp_path, name, history_entries, *, naxis=2, bunit="ADU"):
    """Write a synthetic master FITS with the given HISTORY entries."""
    if naxis == 3:
        data = np.zeros((3, 2, 2), dtype=np.int16)
    else:
        data = np.zeros((2, 2), dtype=np.int16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BUNIT"] = bunit
    for entry in history_entries:
        hdu.header["HISTORY"] = entry
    path = tmp_path / name
    hdu.writeto(path, overwrite=True)
    return path


def _decode(path, *, admission="master", role="dark", flat_form=None):
    return decode_fits(path, declaration=_decl(), admission=admission, role=role, flat_form=flat_form)


# ---------------------------------------------------------------------------
# Light decoder strictness unchanged (byte-for-byte default)
# ---------------------------------------------------------------------------
def test_light_default_still_rejects_stacked(tmp_path):
    p = _write_with_history(tmp_path, "light_stacked.fits", ["frame was stacked"])
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "PROCESSED_HISTORY"


def test_light_admission_explicit_is_identical_to_default(tmp_path):
    p = _write_with_history(tmp_path, "light_median.fits", ["median stack"])
    # "median stack" has no light token match -> admitted under light default.
    f_default = decode_fits(p, declaration=_decl())
    f_explicit = decode_fits(p, declaration=_decl(), admission="light")
    assert np.array_equal(f_default.data, f_explicit.data)
    assert f_default.metadata.units == "ADU"


def test_light_still_rejects_calibrated(tmp_path):
    p = _write_with_history(tmp_path, "light_cal.fits", ["frame was calibrated"])
    with pytest.raises(DecodeError) as exc:
        decode_fits(p, declaration=_decl())
    assert exc.value.reason_code == "PROCESSED_HISTORY"


# ---------------------------------------------------------------------------
# Master admission: stacked/median/mean/rejection/combination ADMITTED
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("history", [
    "master stacked with sigma rejection",
    "median stack of 32 frames",
    "mean combination",
    "sigma-clipped rejection combination",
])
def test_master_stacked_combination_admitted(tmp_path, history):
    p = _write_with_history(tmp_path, "stacked.fits", [history])
    f = _decode(p, role="dark")
    assert f.metadata.units == "ADU"
    assert f.data.shape == (2, 2)


def test_master_flat_stacked_admitted(tmp_path):
    p = _write_with_history(tmp_path, "flat_stacked.fits", ["stacked flat"])
    f = _decode(p, role="flat", flat_form="raw_response")
    assert f.metadata.units == "ADU"


# ---------------------------------------------------------------------------
# Master admission: debayer/demosaic/stretch/white-balance/resample/display reject
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("history", [
    "debayer",
    "demosaiced RGB output",
    "stretched for display",
    "white balance applied",
    "resampled to 2x",
    "screen-transfer function display stretch",
])
def test_master_always_incompatible_markers_rejected(tmp_path, history):
    p = _write_with_history(tmp_path, "bad.fits", [history])
    with pytest.raises(DecodeError) as exc:
        _decode(p, role="dark")
    assert exc.value.reason_code == "MASTER_INCOMPATIBLE"


def test_master_rgb_naaxis3_rejected(tmp_path):
    p = _write_with_history(tmp_path, "rgb.fits", [], naxis=3)
    with pytest.raises(DecodeError) as exc:
        _decode(p, role="dark")
    assert exc.value.reason_code == "RGB_UNSUPPORTED"


# ---------------------------------------------------------------------------
# Master admission: calibrated/normalized role/form aware
# ---------------------------------------------------------------------------
def test_dark_calibrated_rejected(tmp_path):
    p = _write_with_history(tmp_path, "dark_cal.fits", ["calibrated master"])
    with pytest.raises(DecodeError) as exc:
        _decode(p, role="dark")
    assert exc.value.reason_code == "MASTER_PROCESSED"


def test_bias_normalized_rejected(tmp_path):
    p = _write_with_history(tmp_path, "bias_norm.fits", ["normalized bias"])
    with pytest.raises(DecodeError) as exc:
        _decode(p, role="bias")
    assert exc.value.reason_code == "MASTER_PROCESSED"


def test_flat_dark_calibrated_rejected(tmp_path):
    p = _write_with_history(tmp_path, "fd_cal.fits", ["calibrated flat dark"])
    with pytest.raises(DecodeError) as exc:
        _decode(p, role="flat_dark")
    assert exc.value.reason_code == "MASTER_PROCESSED"


def test_flat_calibrated_admitted_with_corrected_form(tmp_path):
    p = _write_with_history(tmp_path, "flat_cal.fits", ["calibrated flat"])
    f = _decode(p, role="flat", flat_form="corrected_unnormalized")
    assert f.metadata.units == "ADU"


def test_flat_calibrated_rejected_with_raw_form(tmp_path):
    p = _write_with_history(tmp_path, "flat_cal_raw.fits", ["calibrated flat"])
    with pytest.raises(DecodeError) as exc:
        _decode(p, role="flat", flat_form="raw_response")
    assert exc.value.reason_code == "MASTER_PROCESSED"


def test_flat_normalized_admitted_with_normalized_form(tmp_path):
    p = _write_with_history(tmp_path, "flat_norm.fits", ["normalized flat response"])
    f = _decode(p, role="flat", flat_form="normalized_response")
    assert f.metadata.units == "ADU"


def test_flat_normalized_rejected_with_corrected_form(tmp_path):
    p = _write_with_history(tmp_path, "flat_norm_corr.fits", ["normalized flat response"])
    with pytest.raises(DecodeError) as exc:
        _decode(p, role="flat", flat_form="corrected_unnormalized")
    assert exc.value.reason_code == "MASTER_PROCESSED"


# ---------------------------------------------------------------------------
# F2: explicit normalization/calibration negation semantics (Siril HISTORY)
# ---------------------------------------------------------------------------
def test_unnormalized_input_is_admissible(tmp_path):
    p = _write_with_history(tmp_path, "un_norm_in.fits", ["unnormalized input"])
    f = _decode(p, role="dark")
    assert f.metadata.units == "ADU"


def test_unnormalized_output_is_admissible(tmp_path):
    p = _write_with_history(tmp_path, "un_norm_out.fits", ["unnormalized output"])
    f = _decode(p, role="dark")
    assert f.metadata.units == "ADU"


def test_multiplicative_normalized_input_unnormalized_output_admissible(tmp_path):
    # Real Siril flat HISTORY: stacking input normalization + unnormalized output
    # is neither a normalized output nor a calibrated master -> admissible.
    p = _write_with_history(
        tmp_path, "siril_flat.fits",
        ["multiplicative normalized input, unnormalized output"],
    )
    f = _decode(p, role="flat", flat_form="raw_response")
    assert f.metadata.units == "ADU"


def test_uncalibrated_is_admissible(tmp_path):
    p = _write_with_history(tmp_path, "uncal.fits", ["uncalibrated master"])
    f = _decode(p, role="dark")
    assert f.metadata.units == "ADU"


def test_normalized_output_still_signals_normalized(tmp_path):
    p = _write_with_history(tmp_path, "norm_out.fits", ["normalized output"])
    with pytest.raises(DecodeError) as exc:
        _decode(p, role="dark")
    assert exc.value.reason_code == "MASTER_PROCESSED"
