"""Deterministic Qt thread-affinity invariants.

Proves the thread-affinity contract of the worker path without hardcoding any
thread id:

* ``_OperationWorker._run`` executes on the dedicated worker QThread (NOT the
  QApplication main thread).
* ``WorkerController._on_worker_ended`` and ``_on_thread_finished`` execute on
  the QApplication main thread.
* The relayed ``operation_started`` / ``operation_finished`` / ``worker_ended``
  signals are delivered (and their receivers run) on the QApplication main
  thread.

These are the invariants that the Qt ownership fix (Finding C) must preserve:
thread affinity was already correct; the defects are ownership/lifetime. This
test pins the affinity so any regression is caught deterministically.

Implementation note: the handler methods are ``@QtCore.Slot``-decorated in
production, so this test does NOT subclass the production QObjects to record
their thread (PySide6 resolves inherited/overridden ``@Slot`` connections by C++
slot index, which would change the observed affinity). Instead it attaches
*DirectConnection* observers to the signals each handler emits, so the observer
runs synchronously on the thread that emitted the signal — which is exactly the
thread the handler/worker runs on.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service
from zecalibrator.gui.worker import WorkerController

from .conftest import DECL, ROI, make_synth_fixture, run_operation


def _library_spec(fixture):
    return v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])


def _light(fixture):
    return service.LightInput(
        path=fixture["light"], hdu=0,
        declaration=service.parse_declaration(DECL),
        roi_extent=service.parse_roi(ROI),
        display_name="light.fits",
    )


def _preflight_snapshot(fixture):
    return service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="preflight",
        library_spec=_library_spec(fixture),
        request=v1.CalibrationRequest("dark_incl_bias", "none"),
        policy=v1.default_match_policy(),
        lights=(_light(fixture),),
    )


def _pump(cond, timeout_ms=20000):
    start = time.monotonic()
    while time.monotonic() - start < timeout_ms / 1000.0:
        QtWidgets.QApplication.processEvents()
        if cond():
            return True
        QtCore.QThread.msleep(5)
    return False


class _ThreadObserver:
    """Records the OS thread on which each observed signal is EMITTED.

    Connected with ``DirectConnection`` so each handler runs synchronously on the
    emitting thread (the thread the worker/handler actually executes on). Records
    both ``threading.get_ident()`` and ``QtCore.QThread.currentThread()``.
    """

    def __init__(self):
        self.threads = {}

    def _record(self, name):
        self.threads[name] = (
            threading.get_ident(), QtCore.QThread.currentThread(),
        )

    # _OperationWorker._run (worker thread) — ``started``/``ended`` are emitted
    # from within ``_run``.
    def on_worker_started(self, op_id):
        self._record("_run_start")

    def on_worker_ended_signal(self):
        self._record("_run_end")

    # WorkerController._on_worker_ended emits ``worker_ended`` as its last act.
    def on_controller_worker_ended(self):
        self._record("_on_worker_ended")

    # WorkerController._on_thread_finished emits ``shutdown_finished``.
    def on_shutdown_finished(self):
        self._record("_on_thread_finished")

    # Relayed signals (emitted on the controller/main thread).
    def on_operation_started(self, op_id):
        self._record("operation_started")

    def on_operation_finished(self, op_id, summary):
        self._record("operation_finished")


def test_worker_and_handler_thread_affinity(qapp, tmp_path):
    main_thread_id = threading.get_ident()
    main_qthread = qapp.thread()

    fixture = make_synth_fixture(tmp_path)
    observer = _ThreadObserver()

    ctl = WorkerController()
    try:
        worker_qthread = ctl._thread

        # DirectConnection: run the observer on the EMITTING thread.
        dc = QtCore.Qt.ConnectionType.DirectConnection
        ctl._worker.started.connect(observer.on_worker_started, dc)
        ctl._worker.ended.connect(observer.on_worker_ended_signal, dc)
        ctl.worker_ended.connect(observer.on_controller_worker_ended, dc)
        ctl.shutdown_finished.connect(observer.on_shutdown_finished, dc)
        ctl.operation_started.connect(observer.on_operation_started, dc)
        ctl.operation_finished.connect(observer.on_operation_finished, dc)

        res = run_operation(ctl, _preflight_snapshot(fixture), v1.CancellationToken())
        assert not res.timed_out
        assert res.finished

        # --- _run executes on the worker thread (NOT the main thread) --------
        assert "_run_start" in observer.threads, "_run did not emit started"
        run_start_id, run_start_qthread = observer.threads["_run_start"]
        assert run_start_id != main_thread_id, "_run ran on the main Python thread"
        assert run_start_qthread is not main_qthread, "_run ran on the main QThread"
        assert run_start_qthread is worker_qthread, "_run ran on an unexpected QThread"

        run_end_id, _ = observer.threads.get("_run_end", (None, None))
        assert run_end_id is not None and run_end_id != main_thread_id

        # --- _on_worker_ended runs on the main thread ------------------------
        assert "_on_worker_ended" in observer.threads, "worker_ended was not emitted"
        ended_id, ended_qthread = observer.threads["_on_worker_ended"]
        assert ended_id == main_thread_id, "_on_worker_ended ran off the main thread"
        assert ended_qthread is main_qthread

        # --- relayed signals are delivered on the main thread ----------------
        for name in ("operation_started", "operation_finished"):
            assert name in observer.threads, f"{name} was not emitted"
            rid, rqthread = observer.threads[name]
            assert rid == main_thread_id, f"{name} emitted off the main thread"
            assert rqthread is main_qthread, f"{name} emitted on a non-main QThread"

        # --- _on_thread_finished runs on the main thread (after shutdown) ----
        ctl.shutdown()
        assert _pump(lambda: "_on_thread_finished" in observer.threads)
        finished_id, finished_qthread = observer.threads["_on_thread_finished"]
        assert finished_id == main_thread_id, "_on_thread_finished ran off the main thread"
        assert finished_qthread is main_qthread
    finally:
        ctl.shutdown()
        _pump(lambda: ctl.is_finished)
