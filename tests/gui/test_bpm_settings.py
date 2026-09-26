"""GUI Bad Pixel Database settings surface tests (LOT 2).

Exercises the new Settings-tab "Bad Pixel Database location" surface: presence,
Browse → validate/create → persist, the discrete status label, and the §62 rule
(no scientific setting is exposed; the single user setting is the location).
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from zecalibrator.gui.window import MainWindow
from zecalibrator.storage import resolve_paths


@pytest.fixture
def paths(tmp_path):
    return resolve_paths(base=str(tmp_path))


def _pump(cond, timeout_ms=15000):
    from PySide6 import QtTest

    start = time.monotonic()
    while time.monotonic() - start < timeout_ms / 1000.0:
        QtWidgets.QApplication.processEvents()
        if cond():
            return True
        QtTest.QTest.qWait(5)
    return False


def _close(w):
    w.close()
    _pump(lambda: w._controller.is_finished)


def test_settings_tab_exposes_bpm_location_surface(qapp, paths):
    w = MainWindow(paths)
    try:
        assert w.bpm_root_edit is not None
        assert w.bpm_browse_btn.text() == "Browse…"
        assert w.bpm_status_label is not None
        # §41 user terminology: "Bad Pixel Database" (never RTS/SensorProfile/PreparationPlan).
        group_titles = [
            g.title() for g in w.main_tabs.findChildren(QtWidgets.QGroupBox)
        ]
        assert any("Bad Pixel Database" in t for t in group_titles)
        assert not any("SensorProfile" in t or "PreparationPlan" in t or "RTS" == t for t in group_titles)
    finally:
        _close(w)


def test_bpm_setting_loads_and_browse_persists(qapp, tmp_path):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    # Persist an initial root before constructing the window.
    initial = tmp_path / "initial-base"
    initial.mkdir()
    _bpm.save_bpm_settings(paths.user_config_path, _bpm.bpm_settings(str(initial)))

    w = MainWindow(paths)
    try:
        assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
        assert w.bpm_root_edit.text() == str(initial)
        assert w.bpm_browse_btn.isEnabled()

        # Browse to a new (missing) folder: validate + create + persist.
        new_base = tmp_path / "new" / "base"
        QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(new_base))
        w._on_browse_bpm_root()
        assert _pump(lambda: not w._controller.is_active)
        assert new_base.is_dir()
        assert w.bpm_root_edit.text() == str(new_base)

        loaded = _bpm.load_bpm_settings(paths.user_config_path)
        assert loaded.settings.bad_pixel_database_root == str(new_base)
    finally:
        _close(w)


def test_bpm_status_label_reflects_configuration(qapp, paths):
    w = MainWindow(paths)
    try:
        assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
        assert w.bpm_status_label.text() == "No Bad Pixel Database configured"
        w._bpm_root = "/some/base"
        w._refresh_bpm_status_label()
        # §7: product wording — exactly the path, no "preview" / "not configured"
        # relics, no misleading "applied automatically".
        assert w.bpm_status_label.text() == "Bad Pixel Database: /some/base"
        assert "preview" not in w.bpm_status_label.text()
    finally:
        _close(w)


def test_bpm_surface_never_exposes_scientific_terms(qapp, paths):
    """§62: the BPM settings group exposes the location only — never scientific
    terms (sigma/threshold/radius/epoch/conflict/net-benefit)."""
    import zecalibrator.gui.window as win_mod

    w = MainWindow(paths)
    try:
        # Only the BPM group's widgets are under scrutiny here: the single user
        # setting is the location; no scientific input is present in that group.
        bpm_group = None
        for g in w.main_tabs.findChildren(QtWidgets.QGroupBox):
            if "Bad Pixel Database" in g.title():
                bpm_group = g
                break
        assert bpm_group is not None
        text = " ".join(
            child.text() for child in bpm_group.findChildren(QtWidgets.QLabel)
        )
        for token in ("sigma", "threshold", "radius", "epoch", "conflict", "net-benefit", "net benefit"):
            assert token not in text.lower()
        assert "Bad Pixel Database location" in text
    finally:
        _close(w)
