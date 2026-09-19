"""Deterministic GC-boundary diagnostic test (ZC-P7-M3B-R2F).

Proves the owner's "GC only while idle" boundary is exact and reversible:

* before an operation, the ambient cyclic-GC state is recorded (never assumed
  to be enabled);
* ``start()`` synchronously disables cyclic GC (process-global) while the worker
  is active — the only restore paths (``_on_worker_ended`` / ``finalize``) are
  main-thread code, so GC stays disabled until the main thread processes the
  terminal event;
* after ``worker_ended`` (operation fully idle) the exact prior GC state is
  restored and the recorded ``_gc_was_enabled`` flag is cleared.

Regression coverage (REWORK r1):

* F1: a late ``_on_worker_ended`` after ``finalize`` already drained the flag
  (``_gc_was_enabled is None``) must NOT call ``gc.disable()`` (the tri-state
  guard) — cyclic GC is not left disabled process-wide.
* "already disabled → restored to disabled": if GC was disabled before the
  operation, it stays disabled after (exact prior state, not an assumed True).

This is a real ``WorkerController`` running one real ``load_settings`` operation
(no QThread spin, no QEventLoop, no subprocess): the manual pump loop mirrors
the production harness (see ``tests/gui/conftest.py``).
"""

from __future__ import annotations

import gc
import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service


def _load_settings_snapshot(config_dir: str) -> service.OperationSnapshot:
    return service.OperationSnapshot(
        op_id=service.new_operation_id(),
        kind="load_settings",
        library_spec=None,
        request=None,
        policy=None,
        lights=(),
        config_dir=config_dir,
    )


def _pump_until_ended(controller, ended: dict) -> None:
    deadline = time.monotonic() + 20.0
    while not ended["value"] and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.0005)
    assert ended["value"], "worker_ended was not observed within the deadline"


def test_gc_boundary_around_operation(controller, tmp_path):
    # Record the ambient state; never assume it is True.
    ambient = gc.isenabled()
    assert controller._gc_was_enabled is None

    ended = {"value": False}

    def on_worker_ended():
        ended["value"] = True

    handle = controller.worker_ended.connect(on_worker_ended)
    try:
        controller.start(_load_settings_snapshot(str(tmp_path)), v1.CancellationToken())

        # ``start()`` disables cyclic GC synchronously on the main thread before
        # the request is dispatched; the only re-enable path is a queued
        # main-thread slot, so GC is still disabled while the worker is active.
        assert gc.isenabled() is False
        assert controller._gc_was_enabled == ambient
        assert controller.is_active is True

        _pump_until_ended(controller, ended)

        # Operation fully idle: GC restored to the exact prior (ambient) state
        # and the saved flag cleared.
        assert gc.isenabled() is ambient
        assert controller._gc_was_enabled is None
        assert controller.is_active is False
    finally:
        try:
            controller.worker_ended.disconnect(handle)
        except (RuntimeError, TypeError):
            pass


def test_gc_already_disabled_restored_to_disabled(controller, tmp_path):
    ambient = gc.isenabled()
    try:
        # Force a disabled start state; the exact prior state is "disabled".
        gc.disable()
        assert gc.isenabled() is False

        ended = {"value": False}

        def on_worker_ended():
            ended["value"] = True

        handle = controller.worker_ended.connect(on_worker_ended)
        try:
            controller.start(
                _load_settings_snapshot(str(tmp_path)), v1.CancellationToken()
            )
            assert gc.isenabled() is False
            assert controller._gc_was_enabled is False

            _pump_until_ended(controller, ended)

            # Restored to the recorded prior state (disabled), NOT hard-enabled.
            assert gc.isenabled() is False
            assert controller._gc_was_enabled is None
        finally:
            try:
                controller.worker_ended.disconnect(handle)
            except (RuntimeError, TypeError):
                pass
    finally:
        # Restore the ambient GC state so this test never pollutes others.
        if ambient:
            gc.enable()
        else:
            gc.disable()


def test_gc_not_left_disabled_on_late_ended_after_finalize(controller, tmp_path):
    """F1 regression: a late ``_on_worker_ended`` must not re-disable GC.

    ``finalize()``'s safety net restores GC and clears ``_gc_was_enabled`` to
    None while the operation is still recorded as active. A late terminal
    delivery that then reaches ``_on_worker_ended`` sees ``_gc_was_enabled is
    None``; the tri-state guard must leave GC untouched (never call
    ``gc.disable()`` on an unrecorded state), or cyclic GC stays disabled
    process-wide with no remaining restore path.
    """
    ambient = gc.isenabled()
    # Deterministic baseline: ensure GC is enabled before the operation.
    if not ambient:
        gc.enable()
    try:
        controller.start(
            _load_settings_snapshot(str(tmp_path)), v1.CancellationToken()
        )
        assert gc.isenabled() is False
        assert controller._gc_was_enabled is True

        # finalize() mid-operation: the safety net restores GC, clears the flag,
        # and stops/joins the worker thread.
        controller.finalize()

        assert gc.isenabled() is True
        assert controller._gc_was_enabled is None

        # A late `ended` delivery now arrives. The actual queued signal may be
        # dropped when the worker QObject is released by finalize(), so invoke
        # the main-thread slot directly to deterministically exercise the guard.
        controller._on_worker_ended()

        # GC must STILL be enabled (the pre-operation state); the None guard
        # prevented the old `else: gc.disable()` path.
        assert gc.isenabled() is True
        assert controller._gc_was_enabled is None
    finally:
        if not ambient:
            gc.disable()
