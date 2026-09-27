"""C20 — RAW/master discrimination via stacking-count evidence (STACKCNT).

Bounded, evidence-based rule (never a filename/folder heuristic): a producer
that stamps its masters with a stacking-count card (>= 2) refuses an
identifiable-role file of the same producer that carries no such proof
(``NOT_A_MASTER``).  A library with no stacking proof anywhere is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.application.masters import detect_stacking_proof

SHAPE = (40, 32)


def _write(path, imagetyp, exptime, *, stackcnt=None, creator="ZWO ASIAIR Plus",
           filter_=None, mean=100.0):
    rng = np.random.default_rng(20260927)
    data = np.clip(rng.normal(mean, mean * 0.02, size=SHAPE), 0, 65535).astype(np.int16)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["IMAGETYP"] = imagetyp
    h["EXPTIME"] = float(exptime)
    h["INSTRUME"] = "Seestar S50"
    h["GAIN"] = 120.0
    h["OFFSET"] = 50.0
    h["CCD-TEMP"] = -10.0
    h["SET-TEMP"] = -10.0
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["XORGSUBF"] = 0
    h["YORGSUBF"] = 0
    h["BAYERPAT"] = "MONO"
    h["BUNIT"] = "ADU"
    h["BSCALE"] = 1.0
    h["BZERO"] = 0.0
    if creator is not None:
        h["CREATOR"] = creator
    if stackcnt is not None:
        h["STACKCNT"] = int(stackcnt)
    if filter_ is not None:
        h["FILTER"] = filter_
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def _cards(**kw):
    # detect_stacking_proof accepts card objects or (kw, val) pairs.
    return [(k.upper(), v) for k, v in kw.items()]


# ---------------------------------------------------------------------------
# detect_stacking_proof
# ---------------------------------------------------------------------------
def test_stacking_proof_ge_2():
    assert detect_stacking_proof(_cards(STACKCNT=20)) is True
    assert detect_stacking_proof(_cards(STACKCNT=2)) is True
    assert detect_stacking_proof(_cards(NCOMBINE=50)) is True
    assert detect_stacking_proof(_cards(NFRAMES=10)) is True


def test_stacking_proof_below_2_or_absent():
    assert detect_stacking_proof(_cards(STACKCNT=1)) is False
    assert detect_stacking_proof(_cards(STACKCNT=0)) is False
    assert detect_stacking_proof(_cards(EXPTIME=60.0)) is False


# ---------------------------------------------------------------------------
# admission: master admitted, raw of same producer refused NOT_A_MASTER
# ---------------------------------------------------------------------------
def test_master_with_proof_admitted_and_raw_refused_not_a_master(tmp_path):
    # A stamped dark master + a raw dark frame from the SAME producer.
    _write(tmp_path / "dark_master.fits", "Dark", 60.0, stackcnt=20, mean=200.0)
    _write(tmp_path / "dark_raw_0001.fits", "Dark", 60.0, stackcnt=None, mean=200.0)
    res = v1.open_session_library(str(tmp_path))
    assert res.operation_status == "COMPLETED"
    # The master is admitted.
    assert any("dark_master" in a.path for a in res.admissions)
    # The raw frame is refused as non-master, with a structured diagnostic.
    raw = [r for r in res.rejected if "dark_raw" in r.path]
    assert len(raw) == 1
    assert raw[0].reason_code == "NOT_A_MASTER"
    assert "stacking-count" in raw[0].detail


def test_bias_master_admitted_raw_bias_refused_not_a_master(tmp_path):
    _write(tmp_path / "bias_master.fits", "Bias", 0.0, stackcnt=50, mean=100.0)
    _write(tmp_path / "bias_raw.fits", "Bias", 0.0, stackcnt=None, mean=100.0)
    res = v1.open_session_library(str(tmp_path))
    assert any("bias_master" in a.path for a in res.admissions)
    raw = [r for r in res.rejected if "bias_raw" in r.path]
    assert len(raw) == 1
    assert raw[0].reason_code == "NOT_A_MASTER"


def test_library_without_any_proof_nothing_refused(tmp_path):
    # No stacking card anywhere: nothing is refused (non-regression).
    _write(tmp_path / "dark_60s.fits", "Dark", 60.0, stackcnt=None, mean=200.0)
    _write(tmp_path / "dark_180s.fits", "Dark", 180.0, stackcnt=None, mean=600.0)
    _write(tmp_path / "bias.fits", "Bias", 0.0, stackcnt=None, mean=100.0)
    res = v1.open_session_library(str(tmp_path))
    assert res.operation_status == "COMPLETED"
    assert not any(r.reason_code == "NOT_A_MASTER" for r in res.rejected)
    assert dict(res.counts_by_role) == {"dark": 2, "bias": 1}


def test_generic_producer_not_refused_when_another_stamps(tmp_path):
    # A different (unknown) producer's raw frame is NOT refused: the refusal is
    # per-producer, only for the producer that demonstrably stamps its masters.
    _write(tmp_path / "dark_master.fits", "Dark", 60.0, stackcnt=20, mean=200.0)
    _write(tmp_path / "dark_raw.fits", "Dark", 60.0, stackcnt=None,
           creator="SomeOtherProducer", mean=200.0)
    res = v1.open_session_library(str(tmp_path))
    # The unknown producer's raw frame is admitted (not NOT_A_MASTER).
    assert any("dark_raw" in a.path for a in res.admissions)
    assert not any(r.reason_code == "NOT_A_MASTER" for r in res.rejected)
