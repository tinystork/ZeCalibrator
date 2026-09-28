"""C20b — RAW/master refinements: proof family + conflict + library-local.

Verifies (a) the proof is a family of stacking-count cards (no card mandatory,
any one >= 2 satisfies), (b) two contradicting cards (one >= 2, another < 2)
declare a conflict (conservative refusal, never a guess), and (c) the
stamping observation is library-local (no global learning across sessions).
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.application.masters import (
    detect_stacking_conflict,
    detect_stacking_proof,
)

SHAPE = (40, 32)


def _write(path, imagetyp, exptime, *, stackcnt=None, ncombine=None, nframes=None,
           creator="ZWO ASIAIR Plus", mean=100.0):
    rng = np.random.default_rng(20260928)
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
    if ncombine is not None:
        h["NCOMBINE"] = int(ncombine)
    if nframes is not None:
        h["NFRAMES"] = int(nframes)
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def _cards(**kw):
    return [(k.upper(), v) for k, v in kw.items()]


# ---------------------------------------------------------------------------
# proof family + conflict
# ---------------------------------------------------------------------------
def test_proof_is_a_family_no_card_mandatory():
    assert detect_stacking_proof(_cards(STACKCNT=20)) is True
    assert detect_stacking_proof(_cards(NCOMBINE=50)) is True
    assert detect_stacking_proof(_cards(NFRAMES=10)) is True
    # several cards, one >= 2 => satisfied
    assert detect_stacking_proof(_cards(STACKCNT=1, NCOMBINE=20)) is True
    # none >= 2 => no proof
    assert detect_stacking_proof(_cards(STACKCNT=1)) is False


def test_conflict_detected_when_cards_contradict():
    assert detect_stacking_conflict(_cards(STACKCNT=20, NCOMBINE=1)) is True
    assert detect_stacking_conflict(_cards(STACKCNT=1, NFRAMES=10)) is True


def test_no_conflict_when_cards_agree_or_single():
    # both >= 2 (agree "stacked"), even with different counts
    assert detect_stacking_conflict(_cards(STACKCNT=20, NCOMBINE=20)) is False
    assert detect_stacking_conflict(_cards(STACKCNT=20, NCOMBINE=5)) is False
    # single card
    assert detect_stacking_conflict(_cards(STACKCNT=20)) is False
    assert detect_stacking_conflict(_cards(STACKCNT=1)) is False
    # no card
    assert detect_stacking_conflict(_cards(EXPTIME=60.0)) is False


def test_conflicting_stack_count_refused_structurally(tmp_path):
    _write(tmp_path / "dark_master.fits", "Dark", 60.0, stackcnt=20, mean=200.0)
    _write(tmp_path / "dark_conflict.fits", "Dark", 60.0, stackcnt=20, ncombine=1,
           mean=200.0)
    res = v1.open_session_library(str(tmp_path))
    assert res.operation_status == "COMPLETED"
    # the conflicting file is refused with a structured diagnostic
    conf = [r for r in res.rejected if "dark_conflict" in r.path]
    assert len(conf) == 1
    assert conf[0].reason_code == "CONFLICTING_STACK_COUNT"
    # the well-formed master is still admitted
    assert any("dark_master" in a.path for a in res.admissions)


def test_library_local_no_global_learning(tmp_path):
    # Library A: producer stamps (a master with STACKCNT=20) -> a raw frame of
    # the same producer (no STACKCNT) is refused NOT_A_MASTER.
    folder_a = tmp_path / "a"
    folder_a.mkdir()
    _write(folder_a / "dark_master.fits", "Dark", 60.0, stackcnt=20, mean=200.0)
    _write(folder_a / "dark_raw.fits", "Dark", 60.0, stackcnt=None, mean=200.0)
    res_a = v1.open_session_library(str(folder_a))
    assert any(r.reason_code == "NOT_A_MASTER" for r in res_a.rejected)

    # Library B: the SAME producer, but NO stacking proof anywhere -> nothing is
    # refused (no global memory of library A's observation).
    folder_b = tmp_path / "b"
    folder_b.mkdir()
    _write(folder_b / "dark_60s.fits", "Dark", 60.0, stackcnt=None, mean=200.0)
    _write(folder_b / "bias.fits", "Bias", 0.0, stackcnt=None, mean=100.0)
    res_b = v1.open_session_library(str(folder_b))
    assert res_b.operation_status == "COMPLETED"
    assert not any(r.reason_code == "NOT_A_MASTER" for r in res_b.rejected)
    assert dict(res_b.counts_by_role) == {"dark": 1, "bias": 1}
