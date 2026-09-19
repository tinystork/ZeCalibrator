"""Weakref lifetime witness (subprocess-isolated).

Diagnostic witness for the Qt ownership/lifetime fix. It answers, empirically,
the question at the heart of the flaky native SIGSEGV:

* When ``WorkerController.finalize()`` sets ``_worker = None`` / ``_thread =
  None``, do the Python wrappers for ``_OperationWorker`` and ``QThread`` die
  immediately, or only at the next ``gc.collect()``?
* If they survive until GC, what keeps ``_OperationWorker`` alive — in
  particular, does the worker's self-referential ``start_op -> self._run``
  connection (``self.start_op.connect(self._run)`` in ``_OperationWorker.__init__``)
  keep the wrapper alive?

The witness runs in a SUBPROCESS so a native crash (SIGSEGV / abort) surfaces as
a non-zero child exit, reported as a test failure instead of killing pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_REPO = Path(__file__).resolve().parents[2]
_SRC = str(_REPO / "src")
_TESTS = str(_REPO / "tests")

_WITNESS_SCRIPT = r'''
import gc
import json
import sys
import time
import weakref
from pathlib import Path

sys.path.insert(0, sys.argv[1])  # src
sys.path.insert(0, sys.argv[2])  # repo root (for tests.gui.conftest)

from PySide6 import QtWidgets

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service
from zecalibrator.gui.worker import WorkerController
from tests.gui.conftest import DECL, ROI, make_synth_fixture

tmp = Path(sys.argv[3])


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


class Collector:
    """Plain-Python collector mirroring conftest.run_operation's collector."""

    def __init__(self, result):
        self.result = result

    def on_finished(self, op_id, summary):
        self.result.finished.append((op_id, summary))

    def on_failed(self, op_id, reason_code, details):
        self.result.failed.append((op_id, reason_code, details))

    def on_progress(self, op_id, event):
        self.result.progress.append((op_id, event))

    def on_preflight(self, op_id, summary, plan):
        self.result.preflight.append((op_id, summary, plan))

    def on_batch_item(self, op_id, item):
        self.result.batch_items.append((op_id, item))

    def on_worker_ended(self):
        self.result.ended = True


class Result:
    def __init__(self):
        self.finished = []
        self.failed = []
        self.progress = []
        self.preflight = []
        self.batch_items = []
        self.ended = False


def run_op(ctl, snapshot, token):
    result = Result()
    collector = Collector(result)
    connections = [
        (ctl.operation_finished, ctl.operation_finished.connect(collector.on_finished)),
        (ctl.operation_failed, ctl.operation_failed.connect(collector.on_failed)),
        (ctl.progress, ctl.progress.connect(collector.on_progress)),
        (ctl.preflight_light, ctl.preflight_light.connect(collector.on_preflight)),
        (ctl.batch_item, ctl.batch_item.connect(collector.on_batch_item)),
        (ctl.worker_ended, ctl.worker_ended.connect(collector.on_worker_ended)),
    ]
    deadline = time.monotonic() + 20.0
    ctl.start(snapshot, token)
    try:
        while not result.ended and time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.0005)
    finally:
        for sig, handle in connections:
            try:
                sig.disconnect(handle)
            except (RuntimeError, TypeError):
                pass
    return result, collector, connections


fixture = make_synth_fixture(tmp)
ctl = WorkerController()

worker_ref = weakref.ref(ctl._worker)
thread_ref = weakref.ref(ctl._thread)
ctl_ref = weakref.ref(ctl)

result, collector, connections = run_op(ctl, _preflight_snapshot(fixture), v1.CancellationToken())
assert result.finished and result.ended, "preflight did not complete"
collector_ref = weakref.ref(collector)

# Drop our strong references to the per-operation collector + connection handles.
del collector, connections

before = {
    "worker": worker_ref() is not None,
    "thread": thread_ref() is not None,
    "controller": ctl_ref() is not None,
    "collector": collector_ref() is not None,
}

# Finished-driven shutdown, then finalize (releases _worker/_thread references).
ctl.shutdown()
deadline = time.monotonic() + 20.0
while not ctl.is_finished and time.monotonic() < deadline:
    QtWidgets.QApplication.processEvents()
    time.sleep(0.0005)
ctl.finalize()

after_finalize = {
    "worker_attr_is_none": ctl._worker is None,
    "thread_attr_is_none": ctl._thread is None,
    "worker_alive": worker_ref() is not None,
    "thread_alive": thread_ref() is not None,
    "collector_alive": collector_ref() is not None,
}

# Inspect what still references the worker wrapper after finalize (before GC).
referrers = []
w = worker_ref()
if w is not None:
    for r in gc.get_referrers(w):
        desc = {
            "type": type(r).__module__ + "." + type(r).__name__,
            "repr": repr(r)[:180],
        }
        # Detect the self-referential start_op -> self._run connection: a bound
        # method named _run whose __self__ is the worker itself.
        if callable(r):
            fn = getattr(r, "__func__", None)
            self_ = getattr(r, "__self__", None)
            if fn is not None and getattr(fn, "__name__", None) == "_run":
                desc["is_start_op_self_run"] = self_ is w
        referrers.append(desc)

# gc.collect() BEFORE the next operation (the diagnostic's key step).
collected = gc.collect()

after_gc = {
    "worker_alive": worker_ref() is not None,
    "thread_alive": thread_ref() is not None,
    "controller_alive": ctl_ref() is not None,
    "collector_alive": collector_ref() is not None,
}

# Verdict: do the worker/thread wrappers die immediately at finalize()
# (refcount -> 0) or only at the next gc.collect()? A wrapper is
# "immediate" if it is already dead after finalize but BEFORE gc.collect().
verdict = {
    "worker": (
        "dies_immediately" if not after_finalize["worker_alive"]
        else ("dies_at_gc" if not after_gc["worker_alive"] else "LEAKS")
    ),
    "thread": (
        "dies_immediately" if not after_finalize["thread_alive"]
        else ("dies_at_gc" if not after_gc["thread_alive"] else "LEAKS")
    ),
    "collector": (
        "dies_immediately" if not before["collector"]
        else ("dies_at_gc" if not after_gc["collector_alive"] else "LEAKS")
    ),
}

print("WITNESS_JSON_BEGIN")
print(json.dumps({
    "before_gc": before,
    "after_finalize": after_finalize,
    "after_gc": after_gc,
    "gc_collected_count": collected,
    "worker_referrers": referrers,
    "verdict": verdict,
}))
print("WITNESS_JSON_END")
sys.exit(0)
'''


def _env():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [_SRC, _TESTS, env.get("PYTHONPATH", "")]))
    return env


def _run_witness(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", _WITNESS_SCRIPT, _SRC, _REPO, str(tmp_path)],
        capture_output=True, text=True, env=_env(), timeout=180,
    )
    return proc


def test_worker_lifetime_witness_subprocess(tmp_path):
    """Lifetime witness: worker/thread wrappers die deterministically after teardown.

    The diagnostic report is printed (observable in ``-s`` output); the assertions
    pin the conclusions that matter for the ownership fix: a clean child exit (no
    native crash) and, after ``gc.collect()``, no tracked Qt/Python wrapper leaks.
    """
    proc = _run_witness(tmp_path)
    assert proc.returncode == 0, (
        f"lifetime witness subprocess exited rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "Destroyed while thread is still running" not in proc.stderr
    assert "QObject: Cannot create children" not in proc.stderr

    # Parse the JSON report block.
    out = proc.stdout
    start = out.index("WITNESS_JSON_BEGIN") + len("WITNESS_JSON_BEGIN")
    end = out.index("WITNESS_JSON_END")
    report = json.loads(out[start:end])

    # Diagnostic conclusions surfaced to the pytest summary.
    print(f"\n[lifetime-witness] before_gc = {report['before_gc']}")
    print(f"[lifetime-witness] after_finalize = {report['after_finalize']}")
    print(f"[lifetime-witness] after_gc = {report['after_gc']}")
    print(f"[lifetime-witness] gc_collected_count = {report['gc_collected_count']}")
    print(f"[lifetime-witness] verdict = {report['verdict']}")

    # The ownership fix requires: after teardown + gc.collect(), the Python
    # wrappers for the worker, thread and per-op collector are all actually dead
    # (no reference leak that could be finalized later while a worker runs).
    assert report["after_gc"]["worker_alive"] is False, \
        "worker wrapper leaked after gc.collect()"
    assert report["after_gc"]["thread_alive"] is False, \
        "thread wrapper leaked after gc.collect()"
    assert report["after_gc"]["collector_alive"] is False, \
        "per-op collector leaked after gc.collect()"

    # finalize() must have dropped the controller's _worker/_thread references.
    assert report["after_finalize"]["worker_attr_is_none"] is True
    assert report["after_finalize"]["thread_attr_is_none"] is True

    # The self-referential start_op -> self._run connection is the expected
    # reason the worker wrapper survives from finalize() until gc.collect();
    # report whether the witness observed that exact referrer.
    any_self_run = any(r.get("is_start_op_self_run") for r in report["worker_referrers"])
    print(f"[lifetime-witness] worker held by start_op->self._run: {any_self_run}")
    print(f"[lifetime-witness] worker referrers: {report['worker_referrers']}")
