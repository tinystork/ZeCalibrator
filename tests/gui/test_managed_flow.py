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
from astropy.io import fits

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui.window import MainWindow
from zecalibrator.storage import resolve_paths

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


def _write_dark(path, *, imagetyp="DARK", value=10.0):
    hdu = fits.PrimaryHDU(np.full(SHAPE, value, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.header["IMAGETYP"] = imagetyp
    hdu.header["EXPTIME"] = 300.0
    hdu.header["CCD-TEMP"] = 20.0
    hdu.header["GAIN"] = 100.0
    hdu.header["OFFSET"] = 50.0
    hdu.header["INSTRUME"] = "SYNTH-CFA"
    hdu.header["BAYERPAT"] = "RGGB"
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    hdu.writeto(path, overwrite=True)
    return str(path)


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
# ---------------------------------------------------------------------------
def test_add_masters_auto_flow_detects_confirms_builds(qapp, paths, tmp_path, monkeypatch):
    from .conftest import wait_idle

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        dark = _write_dark(tmp_path / "dark.fits")
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileNames",
            staticmethod(lambda *a, **k: ([dark], "")),
        )
        monkeypatch.setattr(w, "_collect_user_facts", lambda role, candidates: dict(_FULL_EXTRA))

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
        dark = _write_dark(tmp_path / "dark.fits")
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileNames",
            staticmethod(lambda *a, **k: ([dark], "")),
        )
        # No user facts supplied -> required fields stay missing -> not ready.
        monkeypatch.setattr(w, "_collect_user_facts", lambda role, candidates: {})

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
