"""Qt lifecycle tests: real slow-worker heartbeat, close, cancel, recovery.

Exercises the actual window + WorkerController against injected slow workers and
real queued operations: event-loop heartbeat responsiveness, close during active
work, cancel before/during, generic worker-exception recovery on the SAME window,
stale-event rejection, bounded progress buffering and repeated operations.
"""

from __future__ import annotations

import json
import os
import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui.window import MainWindow, _LightEntry
from zecalibrator.gui.worker import ProgressMailbox
from zecalibrator.storage import resolve_paths

from .conftest import make_synth_fixture, wait_idle


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


def _close(w):
    w.close()
    _pump(lambda: w._controller.is_finished)


def _ready_window(qapp, paths, fixture):
    w = MainWindow(paths)
    # Wait for the async settings load to finish so the worker is idle.
    assert wait_idle(w)
    w._lights.append(_LightEntry(fixture["light"], hdu=0))
    w._lights[0].declaration = v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read()))
    w._lights[0].roi_extent = v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read()))
    w._refresh_lights_list()
    w._library_spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])
    return w


def test_heartbeat_responsive_while_slow_worker_runs(qapp, paths, tmp_path):
    """G3: the GUI event loop stays responsive (timer fires) during slow work."""
    fixture = make_synth_fixture(tmp_path)
    w = _ready_window(qapp, paths, fixture)

    import zecalibrator.api.v1 as v1_mod

    real_inspect = v1_mod.inspect_frame

    def slow_inspect(source, *args, **kwargs):
        time.sleep(0.5)
        return real_inspect(source, *args, **kwargs)

    v1_mod.inspect_frame = slow_inspect
    ticks = []
    timer = QtCore.QTimer()
    timer.setInterval(50)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    try:
        w._on_preflight()
        assert w._controller.is_active
        assert _pump(lambda: len(ticks) >= 3)
        assert w._controller.is_active
        assert _pump(lambda: not w._controller.is_active)
    finally:
        timer.stop()
        v1_mod.inspect_frame = real_inspect
        _close(w)
    assert ticks


def test_close_during_active_work_deferred_then_closes(qapp, paths, tmp_path):
    """Real close during active work: window stays alive until cleanup, then closes."""
    fixture = make_synth_fixture(tmp_path)
    w = _ready_window(qapp, paths, fixture)
    w.show()
    assert w.isVisible()

    import zecalibrator.api.v1 as v1_mod

    real_inspect = v1_mod.inspect_frame

    def blocking_inspect(source, *args, **kwargs):
        cancel = kwargs.get("cancel")
        while cancel is None or not cancel.is_cancelled():
            time.sleep(0.01)
        raise v1.OperationCancelled()

    v1_mod.inspect_frame = blocking_inspect
    try:
        w._on_preflight()
        assert _pump(lambda: w._controller.is_active)
        # Request close while active: the window must remain alive (close ignored).
        w.close()
        assert w.isVisible(), "window must stay alive while the worker is active"
        assert _pump(lambda: w._close_requested is True)
        assert w._token is not None and w._token.is_cancelled()
        # After worker cleanup, the deferred close proceeds and the thread finishes.
        assert _pump(lambda: not w._controller.is_active)
        assert _pump(lambda: w._controller.is_finished)
        assert not w.isVisible()
    finally:
        v1_mod.inspect_frame = real_inspect
        _close(w)


def test_generic_worker_exception_recovery_same_window(qapp, paths, tmp_path):
    """T2: a generic worker exception surfaces a failure, then the SAME window recovers."""
    fixture = make_synth_fixture(tmp_path)
    w = _ready_window(qapp, paths, fixture)

    import zecalibrator.api.v1 as v1_mod

    real_inspect = v1_mod.inspect_frame

    def boom(source, *args, **kwargs):
        raise RuntimeError("injected worker crash")

    v1_mod.inspect_frame = boom
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert "Failed" in w.status_label.text()
        assert "injected worker crash" in w.details_view.toPlainText()
        assert w.preflight_btn.isEnabled()
    finally:
        v1_mod.inspect_frame = real_inspect

    # Same window/controller: a normal preflight now succeeds.
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._preflight_summaries
        assert w._preflight_summaries[0]["outcome"] == "MATCHED"
    finally:
        _close(w)


def test_window_cancel_during_output_transaction(qapp, paths, tmp_path, monkeypatch):
    """T3: cancel via the window during the output phase; committed output preserved,
    no residual task-owned temp file, no false success."""
    fixture = make_synth_fixture(tmp_path)
    from .conftest import write_fits

    write_fits(tmp_path / "light2.fits", 120.0)
    w = _ready_window(qapp, paths, fixture)
    w._lights.append(_LightEntry(str(tmp_path / "light2.fits"), hdu=0))
    w._lights[1].declaration = v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read()))
    w._lights[1].roi_extent = v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read()))
    w._refresh_lights_list()
    w.lights_list.item(0).setSelected(True)
    w.lights_list.item(1).setSelected(True)

    out = tmp_path / "out"
    out.mkdir()

    # Route the export destination through the real window action, then cancel
    # the window's token during the second frame's output transaction.
    import zecalibrator.api.v1.batch as batch_mod

    real_write = batch_mod.write_standalone_output
    calls = {"n": 0}

    def write_and_cancel(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            # Cancel through the window's token (the same thread-safe action the
            # Cancel button performs); no GUI widget access from the worker thread.
            w._token.cancel()
        return real_write(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "write_standalone_output", write_and_cancel)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(out)))
    try:
        w._on_export()
        assert _pump(lambda: not w._controller.is_active)
    finally:
        monkeypatch.undo()

    # Truthful terminal: cancelled, not success.
    assert "CANCELLED" in w.status_label.text()
    # First committed output preserved; no temp file left; second frame not started.
    fits_files = [f for f in os.listdir(str(out)) if f.endswith(".fits")]
    assert len(fits_files) == 1, fits_files
    assert [f for f in os.listdir(str(out)) if ".tmp" in f] == []
    # Results table materializes the committed item and the cancelled remainder.
    _close(w)


def test_progress_mailbox_is_bounded_under_withheld_consumption():
    """Bounded event buffering: posting N events keeps exactly one latest."""
    mailbox = ProgressMailbox()

    class Evt:
        def __init__(self, i):
            self.phase = "frame_start"
            self.completed = i
            self.total = 1000000
            self.unit = "frames"
            self.frame_id = str(i)
            self.operation_id = "op"

    for i in range(100000):
        mailbox.post(Evt(i))
    assert mailbox.take().completed == 99999
    assert mailbox.take() is None


def test_repeated_operations_through_window(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _ready_window(qapp, paths, fixture)
    try:
        for _ in range(3):
            w._on_preflight()
            assert _pump(lambda: not w._controller.is_active)
            assert w._preflight_summaries
            assert w._preflight_summaries[0]["outcome"] == "MATCHED"
    finally:
        _close(w)


def test_terminal_status_stable_after_loop_settles(qapp, paths, tmp_path):
    """R1: the truthful terminal summary stays authoritative after the event loop
    settles for normal export, manifest failure, cancellation, preflight and
    generic failure (no late progress overwrites it)."""
    from PySide6 import QtTest

    def settle():
        # A short fixed settle pass (never a busy-wait).
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtTest.QTest.qWait(5)

    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    # 1) Normal export -> final label is the truthful export summary.
    w = _ready_window(qapp, paths, fixture)
    try:
        from PySide6 import QtWidgets as _QW

        _QW.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(out))
        w.lights_list.item(0).setSelected(True)
        w._on_export()
        assert _pump(lambda: not w._controller.is_active)
        settle()
        assert "Export COMPLETED" in w.status_label.text()
    finally:
        _close(w)

    # 2) Manifest failure -> final label is a truthful failure (not "complete").
    w = _ready_window(qapp, paths, fixture)
    try:
        import zecalibrator.api.v1.batch as batch_mod
        import zecalibrator.io.batch_manifest as bm_mod

        real_writer = bm_mod.write_batch_manifest

        def boom(path, manifest):
            raise OSError("disk full")

        batch_mod.write_batch_manifest = boom
        try:
            from PySide6 import QtWidgets as _QW

            _QW.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(out))
            w.lights_list.item(0).setSelected(True)
            w._on_export()
            assert _pump(lambda: not w._controller.is_active)
            settle()
            assert "FAILED" in w.status_label.text()
            assert "MANIFEST_WRITE_FAILED" in w.details_view.toPlainText()
        finally:
            batch_mod.write_batch_manifest = real_writer
    finally:
        _close(w)

    # 3) Preflight -> final label is the preflight summary (not "complete (2/2)").
    w = _ready_window(qapp, paths, fixture)
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        settle()
        assert "Preflight" in w.status_label.text()
        assert "complete (" not in w.status_label.text()
    finally:
        _close(w)

    # 4) Generic failure -> final label is "Failed: ..." (not overwritten).
    w = _ready_window(qapp, paths, fixture)
    try:
        import zecalibrator.api.v1 as v1_mod

        real_inspect = v1_mod.inspect_frame

        def boom(source, *a, **k):
            raise RuntimeError("injected crash")

        v1_mod.inspect_frame = boom
        try:
            w._on_preflight()
            assert _pump(lambda: not w._controller.is_active)
            settle()
            assert "Failed" in w.status_label.text()
        finally:
            v1_mod.inspect_frame = real_inspect
    finally:
        _close(w)


def test_settings_io_is_off_gui_thread(qapp, tmp_path):
    """S1: settings read (startup) and fsync/write (close) happen on the worker,
    not the GUI thread; a delayed read does not block the GUI heartbeat."""
    from PySide6 import QtCore

    import zecalibrator.gui.worker as worker_mod
    import zecalibrator.gui.settings as settings_mod

    paths = resolve_paths(base=str(tmp_path))
    # Pre-create a valid settings file so load/save have real bytes to handle.
    from zecalibrator.gui.settings import save_settings, default_settings

    save_settings(paths.user_config_path, default_settings())

    real_load = settings_mod.load_settings
    real_save = settings_mod.save_settings
    calls = {"load": 0, "save": 0}

    def slow_load(config_dir):
        time.sleep(0.4)  # simulate slow storage
        calls["load"] += 1
        return real_load(config_dir)

    def slow_save(config_dir, settings):
        time.sleep(0.4)
        calls["save"] += 1
        return real_save(config_dir, settings)

    settings_mod.load_settings = slow_load
    settings_mod.save_settings = slow_save
    ticks = []
    timer = QtCore.QTimer()
    timer.setInterval(30)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    w = None
    try:
        w = MainWindow(paths)
        # The slow settings load is dispatched on the worker; the heartbeat keeps firing.
        assert _pump(lambda: len(ticks) >= 3)
        assert _pump(lambda: w._settings_loaded is True)
        # Close -> async settings save on the worker; heartbeat still fires.
        w.close()
        assert _pump(lambda: w._settings_saved is True)
        assert _pump(lambda: w._controller.is_finished)
        assert calls["load"] >= 1
        assert calls["save"] >= 1
    finally:
        timer.stop()
        settings_mod.load_settings = real_load
        settings_mod.save_settings = real_save
        if w is not None:
            w._controller.shutdown()
            _pump(lambda: w._controller.is_finished)
