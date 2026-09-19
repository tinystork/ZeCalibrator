"""R3C Standard UX: necessary/disambiguator tiers + human-readable confirmation.

Asserts the three R3C invariants at the service boundary (no Qt) and at the
window level (offscreen Qt):

* disambiguators are never prompted;
* genuinely-necessary-and-missing CFA geometry facts prompt a human-readable
  confirmation mapped explicitly to internal values (never guessed);
* header-derived facts never prompt.
"""

from __future__ import annotations

import pytest

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service

SHAPE = (4, 4)


# ---------------------------------------------------------------------------
# Service (no Qt): the field tiers and prompt plan
# ---------------------------------------------------------------------------
def test_necessary_tier_excludes_all_disambiguators():
    for master_type in ("dark", "bias", "flat", "flat_dark"):
        necessary = service.necessary_fields(master_type, "RGGB")
        for field in ("detector_instance_id", "readout_mode", "adc_mode",
                      "sensor_dimensions", "optical_train_id"):
            assert field not in necessary, (master_type, field)


def test_disambiguators_are_never_prompted_for_a_bayer_dark():
    # A Bayer dark header supplies every necessary non-CFA fact; only the CFA
    # geometry facts remain. Disambiguators must never appear as missing.
    candidates = {
        "detector_model": "SYNTH-CFA",
        "gain": 100.0,
        "offset": 50.0,
        "binning": (1, 1),
        "cfa_phase": "RGGB",
        "exposure_s": 300.0,
        "temperature_c": 20.0,
    }
    missing = service.missing_fields_for_master("dark", candidates)
    assert set(missing) == {"orientation", "roi_origin"}
    for field in ("detector_instance_id", "readout_mode", "adc_mode",
                  "sensor_dimensions", "optical_train_id"):
        assert field not in missing


def test_header_derived_facts_never_prompt_mono_dark():
    # Every necessary field is header-derived and mono, so nothing is prompted.
    candidates = {
        "detector_model": "SYNTH-CFA",
        "gain": 100.0,
        "offset": 50.0,
        "binning": (1, 1),
        "cfa_phase": "mono",
        "exposure_s": 300.0,
        "temperature_c": 20.0,
    }
    assert service.missing_fields_for_master("dark", candidates) == ()


def test_cfa_geometry_confirmation_maps_explicitly():
    assert service.cfa_geometry_confirmation("roi_origin") == (
        "full_frame", "Full frame (no crop)", (0, 0),
    )
    assert service.cfa_geometry_confirmation("orientation") == (
        "standard_orientation", "Standard orientation (not flipped)", "identity",
    )
    assert service.cfa_geometry_confirmation("gain") is None


def test_missing_required_fields_uses_necessary_tier_only():
    decl = service.build_declaration(
        "user", "managed-import", "1",
        {
            "exposure_s": v1.EvidenceFact("exposure_s", 300.0, "fits_header", "EXPTIME", "user", "1"),
            "temperature_c": v1.EvidenceFact("temperature_c", 20.0, "fits_header", "CCD-TEMP", "user", "1"),
            "gain": v1.EvidenceFact("gain", 100.0, "fits_header", "GAIN", "user", "1"),
            "offset": v1.EvidenceFact("offset", 50.0, "fits_header", "OFFSET", "user", "1"),
            "detector_model": v1.EvidenceFact("detector_model", "SYNTH-CFA", "fits_header", "INSTRUME", "user", "1"),
            "cfa_phase": v1.EvidenceFact("cfa_phase", "mono", "fits_header", "BAYERPAT", "user", "1"),
            "binning": v1.EvidenceFact("binning", (1, 1), "fits_header", "BINNING", "user", "1"),
        },
    )
    # Mono dark: only the necessary tier is checked; disambiguators stay absent
    # and never block "ready".
    assert service.missing_required_fields("dark", decl) == ()
    assert service.master_evidence_status("dark", decl) == ("ready", ())


def test_bayer_dark_needs_cfa_geometry_to_be_ready():
    decl = service.build_declaration(
        "user", "managed-import", "1",
        {
            "exposure_s": v1.EvidenceFact("exposure_s", 300.0, "fits_header", "EXPTIME", "user", "1"),
            "temperature_c": v1.EvidenceFact("temperature_c", 20.0, "fits_header", "CCD-TEMP", "user", "1"),
            "gain": v1.EvidenceFact("gain", 100.0, "fits_header", "GAIN", "user", "1"),
            "offset": v1.EvidenceFact("offset", 50.0, "fits_header", "OFFSET", "user", "1"),
            "detector_model": v1.EvidenceFact("detector_model", "SYNTH-CFA", "fits_header", "INSTRUME", "user", "1"),
            "cfa_phase": v1.EvidenceFact("cfa_phase", "RGGB", "fits_header", "BAYERPAT", "user", "1"),
            "binning": v1.EvidenceFact("binning", (1, 1), "fits_header", "BINNING", "user", "1"),
        },
    )
    # Bayer sensor: orientation/roi_origin are genuinely necessary.
    assert set(service.missing_required_fields("dark", decl)) == {"orientation", "roi_origin"}
    status, _ = service.master_evidence_status("dark", decl)
    assert status == "needs_attention"


# ---------------------------------------------------------------------------
# Window (offscreen Qt): no internal field name is ever shown
# ---------------------------------------------------------------------------
pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402


def test_collect_user_facts_shows_human_labels_not_internal_names(qapp, tmp_path, monkeypatch):
    from zecalibrator.gui.window import MainWindow
    from zecalibrator.storage import resolve_paths

    w = MainWindow(resolve_paths(base=str(tmp_path)))
    try:
        shown_labels: list[str] = []

        real_add_row = QtWidgets.QFormLayout.addRow

        def fake_add_row(self, *args):
            if args and isinstance(args[0], str) and args[0]:
                shown_labels.append(args[0])
            return real_add_row(self, *args)

        monkeypatch.setattr(QtWidgets.QFormLayout, "addRow", fake_add_row)
        monkeypatch.setattr(
            QtWidgets.QDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Accepted
        )

        # A Bayer dark with a header that has every necessary non-CFA fact:
        # the only prompt is the CFA geometry confirmation.
        candidates = {
            "detector_model": "SYNTH-CFA",
            "gain": 100.0,
            "offset": 50.0,
            "binning": (1, 1),
            "cfa_phase": "RGGB",
            "exposure_s": 300.0,
            "temperature_c": 20.0,
        }
        w._collect_user_facts("dark", candidates)

        internal_names = {
            "detector_instance_id", "detector_model", "readout_mode", "adc_mode",
            "sensor_dimensions", "orientation", "roi_origin", "roi_extent",
            "optical_train_id", "cfa_phase", "binning", "gain", "offset",
            "exposure_s", "temperature_c", "filter",
        }
        for label in shown_labels:
            assert label not in internal_names
            assert not any(name in label for name in internal_names)
    finally:
        w._controller.shutdown()
        QtWidgets.QApplication.processEvents()


def test_collect_user_facts_cfa_confirmation_maps_values(qapp, tmp_path, monkeypatch):
    """Checked CFA confirmations map to their internal values (provenance = user)."""
    from zecalibrator.gui.window import MainWindow
    from zecalibrator.storage import resolve_paths

    w = MainWindow(resolve_paths(base=str(tmp_path)))
    try:
        checked_flags: list[bool] = []

        class _FakeCheck(QtWidgets.QCheckBox):
            def __init__(self, label, parent=None):
                super().__init__(label, parent)
                self.setChecked(True)

            def isChecked(self):
                checked_flags.append(True)
                return True

        monkeypatch.setattr(QtWidgets, "QCheckBox", _FakeCheck)
        monkeypatch.setattr(
            QtWidgets.QDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Accepted
        )

        candidates = {
            "detector_model": "SYNTH-CFA",
            "gain": 100.0,
            "offset": 50.0,
            "binning": (1, 1),
            "cfa_phase": "RGGB",
            "exposure_s": 300.0,
            "temperature_c": 20.0,
        }
        extra = w._collect_user_facts("dark", candidates)
        assert extra.get("roi_origin") == (0, 0)
        assert extra.get("orientation") == "identity"
        assert checked_flags, "CFA confirmations were not rendered as checkboxes"
    finally:
        w._controller.shutdown()
        QtWidgets.QApplication.processEvents()
