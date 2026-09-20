"""Window-level managed auto-flow + lifecycle witnesses (P7-M3B-R2 F1/F3/F4/F5/F6).

Exercises the *actual* window managed-master lifecycle: a single scan op for the
complete master set (no GUI-side chaining), a complete multi-master confirmation
lifecycle drained on ``worker_ended`` (never re-entrant while active), the
empty-library-never-Ready rule (F4), exact managed ``LibrarySpec`` wiring into
Verify (F5), and the bounded Standard auto-flow (F6). All dialogs are mocked;
no real user interaction.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui.window import MainWindow
from zecalibrator.storage import resolve_paths

from .conftest import write_fits_array

SHAPE = (4, 4)


@pytest.fixture
def paths(tmp_path):
    return resolve_paths(base=str(tmp_path))


def _pump(cond, timeout_ms=20000):
    from PySide6 import QtTest

    start = time.monotonic()
    while time.monotonic() - start < timeout_ms / 1000.0:
        QtWidgets.QApplication.processEvents()
        if cond():
            return True
        QtTest.QTest.qWait(5)
    return False


def _write_dark(path, *, imagetyp="DARK", value=10.0, bayerpat="RGGB"):
    data = np.full(SHAPE, value, dtype=np.float32)
    cards = [
        ("IMAGETYP", imagetyp),
        ("EXPTIME", 300.0),
        ("CCD-TEMP", 20.0),
        ("GAIN", 100.0),
        ("OFFSET", 50.0),
        ("INSTRUME", "SYNTH-CFA"),
        ("BAYERPAT", bayerpat),
        ("XBINNING", 1),
        ("YBINNING", 1),
    ]
    return write_fits_array(path, data, header_cards=cards, bunit="ADU")


_FULL_EXTRA = {
    "detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA",
    "gain": 100.0, "offset": 50.0, "readout_mode": "MODE_A", "adc_mode": "MODE_16",
    "sensor_dimensions": SHAPE, "orientation": "identity", "roi_origin": (0, 0),
    "temperature_c": 20.0, "binning": (1, 1), "cfa_phase": "mono",
}


def _shutdown(w):
    w._controller.shutdown()
    _pump(lambda: w._controller.is_finished)


# ---------------------------------------------------------------------------
# F1: a single scan op detects the COMPLETE master set (no first-item-only bug)
# ---------------------------------------------------------------------------
def test_single_scan_detects_all_masters(qapp, paths, tmp_path):
    from .conftest import wait_idle

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        m1 = _write_dark(tmp_path / "dark1.fits")
        m2 = _write_dark(tmp_path / "dark2.fits")
        w._master_files = [m1, m2]
        w._refresh_masters_list()

        w._on_scan_masters()
        assert _pump(lambda: not w._controller.is_active)
        assert len(w._managed_scan) == 2, "one scan op must scan the whole set"
        assert {m["detected_role"] for m in w._managed_scan} == {"dark"}
        assert all(m["admissible"] for m in w._managed_scan)
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# F1: complete multi-master confirmation lifecycle drains on worker_ended
# ---------------------------------------------------------------------------
def test_complete_multi_master_confirmation_lifecycle(qapp, paths, tmp_path, monkeypatch):
    from .conftest import wait_idle
    from zecalibrator.gui import service

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        m1 = _write_dark(tmp_path / "dark1.fits", value=10.0)
        m2 = _write_dark(tmp_path / "dark2.fits", value=11.0)
        # Pre-populate the scan result as if detection already ran.
        w._managed_scan = [service.scan_master_header(m1), service.scan_master_header(m2)]
        assert all(e["admissible"] for e in w._managed_scan)

        monkeypatch.setattr(w, "_collect_user_facts", lambda role, candidates: dict(_FULL_EXTRA))
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None)
        )

        w._on_confirm_masters()
        assert _pump(lambda: not w._controller.is_active and not w._confirm_batch_active)

        # Both masters were confirmed (complete lifecycle, not just the first).
        assert len(w._session_selection) == 2
        ledger = v1.load_managed_ledger(v1.managed_ledger_path(str(paths.user_data_path)))
        assert ledger.state == "ok"
        assert len(ledger.records) == 2
        assert {r.role for r in ledger.records} == {"dark"}
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# F4: an empty managed library is never Ready (and refuses Verify/export)
# ---------------------------------------------------------------------------
def test_empty_managed_library_never_ready(qapp, paths, monkeypatch):
    from .conftest import wait_idle

    w = MainWindow(paths)
    assert wait_idle(w)
    warned = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(True))
    )
    try:
        spec = v1.LibrarySpec(
            root=str(paths.user_cache_path),
            index_path=str(paths.user_cache_path / "managed.sqlite"),
        )
        w._managed_spec = spec
        w._set_active_source("managed")
        w._handle_build_managed({
            "kind": "build_managed_library", "status": "COMPLETED",
            "revision": "fp", "candidate_count": 0, "needs_attention": [],
            "reason_code": None, "details": "",
        })
        assert "needs attention" in w.library_human_status.text()
        assert w._library_spec is None
        assert w._ensure_library() is False
        assert warned, "Verify/export readiness must be refused for an empty library"
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# F5: a successful managed build wires the exact LibrarySpec into Verify
# ---------------------------------------------------------------------------
def test_managed_build_wires_exact_library_spec(qapp, paths):
    from .conftest import wait_idle

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        spec = v1.LibrarySpec(
            root=str(paths.user_cache_path),
            index_path=str(paths.user_cache_path / "managed.sqlite"),
        )
        w._managed_spec = spec
        w._set_active_source("managed")
        w._handle_build_managed({
            "kind": "build_managed_library", "status": "COMPLETED",
            "revision": "fp", "candidate_count": 1, "needs_attention": [],
            "reason_code": None, "details": "",
        })
        assert w._library_spec is spec
        assert w._library_spec.index_path == spec.index_path
        assert "Library ready" in w.library_human_status.text()
        assert w._ensure_library() is True, "ready managed library must satisfy _ensure_library()"
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# F6: bounded Standard auto-flow — add masters -> detect -> confirm -> prepare
# (R3D-C: no modal dialogs; role auto-detected from IMAGETYP; facts auto-derived).
# ---------------------------------------------------------------------------
def test_add_masters_auto_flow_detects_confirms_builds(qapp, paths, tmp_path, monkeypatch):
    from .conftest import wait_idle

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        # A mono dark carries every necessary fact in the header, so the
        # auto-flow confirms it without any user questionnaire.
        dark = _write_dark(tmp_path / "dark.fits", bayerpat="mono")
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileNames",
            staticmethod(lambda *a, **k: ([dark], "")),
        )
        prompts = []
        real_information = QtWidgets.QMessageBox.information
        real_warning = QtWidgets.QMessageBox.warning
        real_exec = QtWidgets.QDialog.exec
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "information",
            staticmethod(lambda *a, **k: prompts.append("information") or real_information(*a, **k)),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            staticmethod(lambda *a, **k: prompts.append("warning") or real_warning(*a, **k)),
        )
        monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: prompts.append("dialog"))

        w._on_add_masters_file()
        # auto-flow: detect -> confirm -> auto-prepare; wait until fully settled.
        assert _pump(
            lambda: not w._controller.is_active
            and not w._confirm_batch_active
            and w._auto_flow_active is False
        )
        assert w._library_spec is not None, "auto-flow must wire the managed LibrarySpec"
        assert "Library ready" in w.library_human_status.text()
        assert w._ensure_library() is True
        # R3D-C: the Standard auto-flow must never show a modal dialog.
        assert prompts == [], f"unexpected modal prompts in auto-flow: {prompts}"
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# F6: auto-prepare is skipped when the confirmed set has incomplete evidence
# ---------------------------------------------------------------------------
def test_auto_flow_skips_prepare_when_incomplete_evidence(qapp, paths, tmp_path, monkeypatch):
    from .conftest import wait_idle

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        # A Bayer dark missing the CFA geometry facts (orientation/roi_origin)
        # stays incomplete under the auto-flow (no questionnaire to fill them).
        dark = _write_dark(tmp_path / "dark.fits")
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileNames",
            staticmethod(lambda *a, **k: ([dark], "")),
        )

        w._on_add_masters_file()
        assert _pump(
            lambda: not w._controller.is_active
            and not w._confirm_batch_active
            and w._auto_flow_active is False
        )
        # Not fully admissible -> no managed library is prepared/wired.
        assert w._library_spec is None
        assert "needs attention" in w.library_human_status.text()
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# R3D-C: the Standard auto-flow reports role problems as NON-MODAL per-master
# needs-attention notes (never a dialog).
# ---------------------------------------------------------------------------
def test_auto_flow_no_modal_for_undetermined_role(qapp, paths, tmp_path, monkeypatch):
    from .conftest import wait_idle
    from zecalibrator.gui import service

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        m = tmp_path / "norole.fits"
        # A mono dark header with NO IMAGETYP -> role undeterminable.
        data = np.full(SHAPE, 10.0, dtype=np.float32)
        write_fits_array(
            m, data, bunit="ADU",
            header_cards=[
                ("EXPTIME", 300.0), ("CCD-TEMP", 20.0), ("GAIN", 100.0),
                ("OFFSET", 50.0), ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "mono"),
                ("XBINNING", 1), ("YBINNING", 1),
            ],
        )
        w._managed_scan = [service.scan_master_header(str(m))]
        assert w._managed_scan[0].get("role") is None

        prompts = []
        monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: prompts.append("dialog"))
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getItem",
            staticmethod(lambda *a, **k: prompts.append("getItem") or ("dark", True)),
        )

        w._auto_flow_active = True
        w._collect_and_dispatch_confirms()
        # No master was confirmed (role could not be determined) and no modal.
        assert w._managed_pending == []
        assert prompts == [], f"unexpected modal prompts for undetermined role: {prompts}"
    finally:
        _shutdown(w)


def test_auto_flow_no_modal_for_role_conflict(qapp, paths, tmp_path, monkeypatch):
    from .conftest import wait_idle
    from zecalibrator.gui import service

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        m = tmp_path / "conflict.fits"
        _write_dark(m, bayerpat="mono")
        # Synthesize a selected-vs-detected role conflict (as if the user had
        # pre-selected a conflicting role); the auto-flow must not prompt.
        entry = service.scan_master_header(str(m), selected_role="bias")
        assert entry.get("conflict") is not None
        w._managed_scan = [entry]

        prompts = []
        monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: prompts.append("dialog"))
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            staticmethod(lambda *a, **k: prompts.append("warning")),
        )

        w._auto_flow_active = True
        w._collect_and_dispatch_confirms()
        assert w._managed_pending == []
        assert prompts == [], f"unexpected modal prompts for role conflict: {prompts}"
    finally:
        _shutdown(w)
