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


def test_standard_mode_labels_map_exact_values():
    # Standard labels are presentation-only; each maps 1:1 to the frozen value.
    assert presentation.standard_additive_mode_label("control") == "None"
    assert presentation.standard_additive_mode_label("bias_only") == "Bias only"
    assert presentation.standard_additive_mode_label("dark_incl_bias") == "Standard dark"
    assert presentation.standard_additive_mode_label("dark_bias_removed") == "Dark already bias-corrected"
    assert presentation.standard_flat_mode_label("none") == "None"
    assert presentation.standard_flat_mode_label("apply") == "Use flat"


def test_standard_mode_labels_roundtrip_technical_values():
    # Every runtime value must still have a technical label in Advanced.
    for mode in presentation.additive_modes():
        assert presentation.additive_mode_label(mode)
        assert presentation.standard_additive_mode_label(mode)
    for mode in presentation.flat_modes():
        assert presentation.flat_mode_label(mode)
        assert presentation.standard_flat_mode_label(mode)


def test_human_outcome_labels_exact():
    assert presentation.human_outcome_label("MATCHED") == "Ready"
    assert presentation.human_outcome_label("NO_MATCH") == "Needs attention"
    assert presentation.human_outcome_label("AMBIGUOUS") == "Ambiguous calibration set"
    assert presentation.human_outcome_label(None) == "Needs attention"


def test_human_reason_text_never_guesses_a_candidate():
    assert presentation.human_reason_text({"outcome": "MATCHED"}) == "A compatible calibration set was found."
    assert presentation.human_reason_text({"outcome": "AMBIGUOUS"}) == \
        "More than one compatible calibration set was found."
    assert presentation.human_reason_text({"outcome": "NO_MATCH", "reasons": []}) == \
        "No compatible calibration set was found."
    assert presentation.human_reason_text({"outcome": None}) == \
        "This image could not be inspected."


def test_human_reason_text_names_only_unavailable_role():
    # The fallback names a *missing* required role, never a chosen candidate.
    assert presentation.human_reason_text({
        "outcome": "NO_MATCH",
        "reasons": [{"code": "ROLE_UNAVAILABLE", "role": "flat", "field": None}],
    }) == "No compatible flat found."
    assert presentation.human_reason_text({
        "outcome": "NO_MATCH",
        "reasons": [{"code": "ROLE_UNAVAILABLE", "role": "dark", "field": None}],
    }) == "No compatible dark found."
    assert presentation.human_reason_text({
        "outcome": "NO_MATCH",
        "reasons": [{"code": "ROLE_UNAVAILABLE", "role": "flat_dark", "field": None}],
    }) == "No compatible flat dark found."


def test_human_reason_text_ignores_non_unavailable_codes():
    assert presentation.human_reason_text({
        "outcome": "NO_MATCH",
        "reasons": [{"code": "EXPOSURE_MISMATCH", "role": "dark", "field": "acquisition.exposure_s"}],
    }) == "No compatible calibration set was found."


def test_human_reason_text_inspect_failure_source_unreadable():
    assert presentation.human_reason_text(
        {"outcome": None, "reason_code": "SOURCE_ERROR"}
    ) == "Could not read this image."


def test_human_reason_text_inspect_failure_needs_evidence():
    for code in ("UNKNOWN_DOMAIN", "UNKNOWN_UNITS", "UNITS_UNSUPPORTED"):
        assert presentation.human_reason_text(
            {"outcome": None, "reason_code": code}
        ) == "This image needs valid raw-sensor import evidence."


def test_human_reason_text_inspect_failure_decode_failed():
    for code in ("PRECISION_REFUSAL", "MALFORMED_CARD", "METADATA_ADAPTER",
                 "INVALID_SOURCE", "ALIAS_CONFLICT", "DECLARATION_CONFLICT"):
        assert presentation.human_reason_text(
            {"outcome": None, "reason_code": code}
        ) == "This file could not be decoded as a supported raw FITS image."


def test_human_reason_text_inspect_failure_not_raw():
    for code in ("PROCESSED_HISTORY", "RGB_UNSUPPORTED"):
        assert presentation.human_reason_text(
            {"outcome": None, "reason_code": code}
        ) == "This image is not raw 2-D sensor data (it looks processed or colour)."


def test_human_reason_text_inspect_failure_hdu():
    for code in ("HDU_NOT_FOUND", "HDU_NOT_2D_IMAGE", "HDU_AMBIGUOUS",
                 "NO_2D_IMAGE", "NOT_2D_PLANE"):
        assert presentation.human_reason_text(
            {"outcome": None, "reason_code": code}
        ) == "The selected HDU has no usable 2-D image — pick another HDU in Advanced."


def test_human_reason_text_inspect_failure_generic_fallback():
    assert presentation.human_reason_text(
        {"outcome": None, "reason_code": "SOME_UNKNOWN_CODE"}
    ) == "This image could not be inspected."
    assert presentation.human_reason_text(
        {"outcome": None, "reason_code": None}
    ) == "This image could not be inspected."


def test_human_reason_text_never_leaks_raw_details():
    # The human reason is a fixed sentence; raw filesystem details must never appear.
    text = presentation.human_reason_text(
        {"outcome": None, "reason_code": "SOURCE_ERROR",
         "details": "[Errno 2] No such file or directory: '/tmp/missing.fits'"}
    )
    assert text == "Could not read this image."
    assert "Errno" not in text
    assert "missing.fits" not in text


def test_format_folder_add_feedback_truthful_plural():
    assert presentation.format_folder_add_feedback(1, 0) == "1 image added"
    assert presentation.format_folder_add_feedback(2, 0) == "2 images added"
    assert presentation.format_folder_add_feedback(3, 1) == "3 images added (1 unsupported file ignored)"
    assert presentation.format_folder_add_feedback(0, 2) == "0 images added (2 unsupported files ignored)"
    assert presentation.format_folder_add_feedback(0, 0) == "0 images added"


def test_summarize_and_format_outcomes_truthful():
    summaries = [
        {"outcome": "MATCHED"}, {"outcome": "MATCHED"}, {"outcome": "MATCHED"},
        {"outcome": "NO_MATCH"}, {"outcome": None}, {"outcome": "AMBIGUOUS"},
    ]
    ready, attention, ambiguous = presentation.summarize_outcomes(summaries)
    assert (ready, attention, ambiguous) == (3, 2, 1)
    text = presentation.format_outcome_summary(ready, attention, ambiguous)
    assert "3 images ready" in text
    assert "2 need attention" in text
    assert "1 ambiguous calibration set" in text
    assert presentation.format_outcome_summary(0, 0, 0) == "No images verified."
    assert presentation.format_outcome_summary(1, 0, 0) == "1 image ready"
