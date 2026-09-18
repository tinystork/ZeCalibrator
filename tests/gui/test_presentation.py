"""Headless tests for pure presentation formatting helpers (no PySide6)."""

from __future__ import annotations

from zecalibrator.gui import presentation


def test_status_labels_cover_frozen_vocabulary():
    for status in ("OPENED", "COMPLETED", "COMPLETED_WITH_WARNINGS", "SKIPPED",
                   "FAILED", "CANCELLED", "PARTIAL", "MATCHED", "NO_MATCH", "AMBIGUOUS"):
        assert presentation.status_label(status)


def test_is_partial_mode_is_explicit():
    assert presentation.is_partial_mode("control", "none")
    assert presentation.is_partial_mode("control", "apply")
    assert presentation.is_partial_mode("dark_incl_bias", "none")
    assert not presentation.is_partial_mode("dark_incl_bias", "apply")


def test_format_rejection_table_renders_field_level_reasons():
    text = presentation.format_rejection_table([{
        "candidate_id": "d1", "role": "dark",
        "reasons": [
            {"code": "EXPOSURE_MISMATCH", "field": "acquisition.exposure_s",
             "expected": 10.0, "observed": 5.0},
        ],
    }])
    assert "EXPOSURE_MISMATCH" in text
    assert "acquisition.exposure_s" in text
    assert "d1" in text


def test_format_coherent_sets_renders_sets():
    text = presentation.format_coherent_sets([{"dark": "d1"}, {"dark": "d2"}])
    assert "set 1" in text
    assert "set 2" in text
    assert "dark=d1" in text


def test_format_metadata_renders_geometry():
    text = presentation.format_metadata({
        "geometry": {"shape": [4, 4], "cfa_phase": "mono"},
        "units": "ADU",
        "warnings": ["w1"],
    })
    assert "shape" in text
    assert "mono" in text
    assert "w1" in text


def test_reason_codes_text():
    assert presentation.reason_codes_text(("A", "B")) == "A, B"
    assert presentation.reason_codes_text(()) == "(none)"
