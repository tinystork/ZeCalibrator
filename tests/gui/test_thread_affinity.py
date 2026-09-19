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

Why this observer is deterministic (previous witness was flaky on Windows CI):

* The observer is a real :class:`~PySide6.QtCore.QObject` (NOT a plain Python
  class). A plain Python object connected with ``DirectConnection`` to a
  cross-thread signal is an unreliable oracle: PySide6 must wrap the callable in
  a functor, and on some platforms the functor delivery of the worker's
  terminal ``ended`` (emitted as the *last* act of ``_run``, immediately before
  the method returns) could be deferred or dropped, so ``_run_end`` was never
  recorded.
* The worker thread is witnessed by attaching the QObject observer's slots to
  ``_OperationWorker.started`` / ``ended`` with ``DirectConnection``. A
  ``DirectConnection`` to a QObject slot is the canonical Qt synchronous
  mechanism: the slot runs on the EMITTING thread *before* ``emit()`` returns,
  so it cannot be skipped, reordered, or dropped. Both witnesses are therefore
  recorded before ``run_operation`` observes the terminal event.
* Handler/main-thread affinity is witnessed on the CONTROLLER's already-delivered
  signals (``operation_started`` / ``operation_finished`` / ``worker_ended`` /
  ``shutdown_finished``) with the default (auto) connection. Those signals are
  emitted from the main thread, so the observer runs on the main thread. This
  never depends on cross-thread signal delivery from the worker.
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
        time.sleep(0.0005)
    return False


class _ThreadObserver(QtCore.QObject):
    """QObject observer recording the OS thread on which each signal is handled.

    Worker-thread witnesses (``_run_start`` / ``_run_end``) are connected with
    ``DirectConnection`` so they run synchronously on the worker thread; all
    other witnesses run on the main thread via the controller's own signals.
    """

    def __init__(self):
        super().__init__()
        self.threads = {}

    def _record(self, name):
        self.threads[name] = (
            threading.get_ident(), QtCore.QThread.currentThread(),
        )

    # _OperationWorker._run (worker thread) — DirectConnection observers, so
    # these run synchronously on the emitting worker thread and cannot be skipped.
    def on_worker_started(self, op_id):
        self._record("_run_start")

    def on_worker_ended_signal(self):
        self._record("_run_end")

    # WorkerController handlers / relayed signals (main thread).
    def on_controller_worker_ended(self):
        self._record("_on_worker_ended")

    def on_shutdown_finished(self):
        self._record("_on_thread_finished")

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

        dc = QtCore.Qt.ConnectionType.DirectConnection
        # Worker-thread witnesses: DirectConnection to a QObject slot -> runs
        # synchronously on the emitting (worker) thread, cannot be skipped.
        ctl._worker.started.connect(observer.on_worker_started, dc)
        ctl._worker.ended.connect(observer.on_worker_ended_signal, dc)
        # Main-thread witnesses: default (auto) connection on controller signals
        # that the controller emits from the main thread.
        ctl.worker_ended.connect(observer.on_controller_worker_ended)
        ctl.shutdown_finished.connect(observer.on_shutdown_finished)
        ctl.operation_started.connect(observer.on_operation_started)
        ctl.operation_finished.connect(observer.on_operation_finished)

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
        ctl.finalize()
