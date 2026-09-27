"""C13 — GAIN/EGAIN admission fix: distinct fields, no false conflict.

The canonical model (metadata.py) treats ``gain`` (producer gain units, GAIN) and
``gain_e_per_adu`` (electrons/ADU, EGAIN) as DISTINCT fields.  The admission
layer's evidence map must not map EGAIN onto ``gain`` (which produced a false
``CONFLICTING_HEADER`` for every real ZWO master carrying both cards).
"""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.application import masters


def test_gain_and_egain_distinct_no_conflict():
    cards = [
        ("GAIN", 120), ("EGAIN", 1.00224268436432),
        ("EXPTIME", 60.0), ("OFFSET", 30), ("XBINNING", 1), ("YBINNING", 1),
        ("BAYERPAT", "RGGB"), ("INSTRUME", "ZWO ASI294MC Pro"),
        ("CCD-TEMP", -10.0), ("SET-TEMP", -10), ("IMAGETYP", "Dark"),
    ]
    candidates, conflicts = masters.detect_header_candidates(cards)
    assert conflicts == {}
    assert candidates["gain"].value == 120.0
    assert abs(candidates["gain_e_per_adu"].value - 1.00224268436432) < 1e-9


def test_gain_only_gain_set_egain_absent():
    cards = [("GAIN", 120), ("EXPTIME", 60.0), ("OFFSET", 30)]
    candidates, conflicts = masters.detect_header_candidates(cards)
    assert conflicts == {}
    assert candidates["gain"].value == 120.0
    assert "gain_e_per_adu" not in candidates


def test_egain_only_egain_set_gain_absent():
    cards = [("EGAIN", 1.00224268436432), ("EXPTIME", 60.0)]
    candidates, conflicts = masters.detect_header_candidates(cards)
    assert conflicts == {}
    assert "gain" not in candidates
    assert abs(candidates["gain_e_per_adu"].value - 1.00224268436432) < 1e-9


def test_two_diverging_gain_still_conflicts():
    cards = [("GAIN", 120), ("GAIN", 121)]
    candidates, conflicts = masters.detect_header_candidates(cards)
    assert "gain" not in candidates
    assert "gain" in conflicts
    assert len(conflicts["gain"]) == 2


def test_declaration_carries_both_gain_fields():
    cards = [("GAIN", 120), ("EGAIN", 1.00224268436432)]
    candidates, _ = masters.detect_header_candidates(cards)
    decl = masters.build_declaration("src", "id", "1", candidates)
    assert decl.gain == 120.0
    assert abs(decl.gain_e_per_adu - 1.00224268436432) < 1e-9
