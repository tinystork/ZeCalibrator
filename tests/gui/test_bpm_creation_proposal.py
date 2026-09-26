"""LOT 3 GUI map-creation proposal tests (§9 E: one offer per run).

Verifies the window's proposal gate: a "propose" decision opens the offer dialog
exactly once per run, declining proceeds to normal calibration with no error and
no per-light repetition, a compatible map asks no question, and "unavailable"
only logs a discreet note.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from zecalibrator.gui.window import MainWindow
from zecalibrator.storage import resolve_paths


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


class _FakePlan:
    def __init__(self, *, has_dark=True):
        self.light_constraints = "dummy-light"
        self.masters = {"dark": object()} if has_dark else {}


def _prep(qapp, monkeypatch, tmp_path, decision):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
    w._bpm_root = str(tmp_path / "base")
    monkeypatch.setattr(_bpm, "bpm_creation_decision", decision)
    return w


def test_propose_offer_is_emitted_once_per_run(qapp, monkeypatch, tmp_path):
    w = _prep(qapp, monkeypatch, tmp_path,
              lambda *a, dark_available=False, **k: {"action": "propose", "reason_code": "NO_PROFILE"})
    try:
        w._plans = {"r1": _FakePlan(has_dark=True)}
        offers = []
        exports = []
        w._offer_bpm_creation = lambda request, destination: offers.append((request, destination))
        w._start_export = lambda request, destination: exports.append((request, destination))

        w._check_bpm_proposal_then_export(None, "/dest")
        assert len(offers) == 1
        assert w._bpm_proposed is True
        # Second light / second export in the SAME run: never re-offered.
        w._check_bpm_proposal_then_export(None, "/dest")
        assert len(offers) == 1
    finally:
        _close(w)


def test_declined_proposal_proceeds_without_error_and_no_repeat(qapp, monkeypatch, tmp_path):
    w = _prep(qapp, monkeypatch, tmp_path,
              lambda *a, dark_available=False, **k: {"action": "propose", "reason_code": "NO_PROFILE"})
    try:
        w._plans = {"r1": _FakePlan(has_dark=True)}
        offers = []
        exports = []

        def _offer(request, destination):
            offers.append((request, destination))
            # The human declines -> normal calibration, no error.
            exports.append(("declined", request, destination))

        w._offer_bpm_creation = _offer
        w._start_export = lambda request, destination: exports.append(("export", request, destination))

        w._check_bpm_proposal_then_export(None, "/dest")
        assert len(offers) == 1
        assert w._bpm_proposed is True
        # No export was started by the offer itself; the decline just records.
        # A second run-triggered check never re-offers.
        w._check_bpm_proposal_then_export(None, "/dest")
        assert len(offers) == 1
    finally:
        _close(w)


def test_compatible_map_asks_no_question(qapp, monkeypatch, tmp_path):
    w = _prep(qapp, monkeypatch, tmp_path,
              lambda *a, dark_available=False, **k: {"action": "use", "reason_code": ""})
    try:
        w._plans = {"r1": _FakePlan(has_dark=True)}
        offers = []
        exports = []
        w._offer_bpm_creation = lambda request, destination: offers.append((request, destination))
        w._start_export = lambda request, destination: exports.append((request, destination))

        w._check_bpm_proposal_then_export(None, "/dest")
        assert offers == []  # case D: no question
        assert len(exports) == 1  # export proceeds directly
    finally:
        _close(w)


def test_unavailable_logs_discreet_note_and_no_offer(qapp, monkeypatch, tmp_path):
    w = _prep(qapp, monkeypatch, tmp_path,
              lambda *a, dark_available=False, **k: {"action": "unavailable", "reason_code": "NO_PROFILE"})
    try:
        w._plans = {"r1": _FakePlan(has_dark=False)}
        offers = []
        exports = []
        logs = []
        w._offer_bpm_creation = lambda request, destination: offers.append((request, destination))
        w._start_export = lambda request, destination: exports.append((request, destination))
        w._log = lambda text: logs.append(text)

        w._check_bpm_proposal_then_export(None, "/dest")
        assert offers == []  # no impossible creation offered
        assert len(exports) == 1
        assert any("No compatible Bad Pixel Map is available" in t for t in logs)
    finally:
        _close(w)
