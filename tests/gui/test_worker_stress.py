"""Worker lifecycle stress witness (subprocess-isolated).

Repeatedly runs ``preflight`` -> ``finished`` cycles on ONE ``WorkerController``
against the synthetic SYNTH fixture, then shuts the controller down. The loop is
isolated in a SUBPROCESS so a native SIGSEGV in the Qt worker path surfaces as a
non-zero subprocess exit code (reported as a test failure) instead of killing
the pytest process. This is the flaky-path witness for the Qt ownership fix
(Finding C: canonical ``deleteLater`` worker destruction; Finding D: QObject
result collector).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_REPO = Path(__file__).resolve().parents[2]
_SRC = str(_REPO / "src")
_TESTS = str(_REPO / "tests")

_ITERATIONS = 40  # bounded "few dozen" cycles; fast but exercises the flaky path

_STRESS_SCRIPT = r'''
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])  # src
sys.path.insert(0, sys.argv[4])  # repo root (for tests.gui.conftest)

from PySide6 import QtWidgets

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service
from zecalibrator.gui.worker import WorkerController

from tests.gui.conftest import DECL, ROI, make_synth_fixture, run_operation

tmp = Path(sys.argv[1])
iterations = int(sys.argv[2])


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


fixture = make_synth_fixture(tmp)
ctl = WorkerController()
try:
    for i in range(iterations):
        res = run_operation(ctl, _preflight_snapshot(fixture), v1.CancellationToken())
        assert not res.timed_out, "timed out at iteration %d" % i
        summary = res.finished_summary()
        assert summary["status"] == "COMPLETED", summary
finally:
    ctl.shutdown()
    # Finished-driven, non-blocking shutdown: pump the event loop until the
    # worker thread has actually finished (no GUI-thread wait, no terminate).
    import time
    from PySide6 import QtCore, QtTest

    start = time.monotonic()
    while time.monotonic() - start < 20.0:
        QtWidgets.QApplication.processEvents()
        if ctl.is_finished:
            break
        QtTest.QTest.qWait(5)
    assert ctl.is_finished, "worker thread did not finish after shutdown"

print("EXIT_CLEAN")
sys.exit(0)
'''


def _env():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Keep src + repo root importable in the child (the script also inserts them
    # explicitly, but this is belt-and-suspenders for PYTHONPATH cleanliness).
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [_SRC, _TESTS, env.get("PYTHONPATH", "")]))
    return env


def test_worker_lifecycle_stress_subprocess(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, "-c", _STRESS_SCRIPT,
            str(tmp_path), str(_ITERATIONS), _SRC, _REPO,
        ],
        capture_output=True, text=True, env=_env(), timeout=180,
    )
    assert proc.returncode == 0, (
        f"stress subprocess exited rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "EXIT_CLEAN" in proc.stdout
    # A native SIGSEGV must never appear in the child (it would be rc != 0 above,
    # but assert the absence of the Qt destroyed-while-running warning too).
    assert "Destroyed while thread is still running" not in proc.stderr
    assert "QObject: Cannot create children" not in proc.stderr


_WEDGED_FINALIZE_SCRIPT = r'''
import sys
import threading
import time

sys.path.insert(0, sys.argv[1])  # src

from PySide6 import QtWidgets

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service
from zecalibrator.gui.worker import WorkerController

# Wedge the worker thread: block the first v1.open_library call forever.
blocked = threading.Event()
blocker = threading.Event()


def blocking_open(*args, **kwargs):
    blocked.set()
    blocker.wait()
    raise RuntimeError("unwedge")


v1.open_library = blocking_open

ctl = WorkerController()
spec = v1.LibrarySpec(root="/tmp", index_path="/tmp/zecalibrator_nonexistent.sqlite")
snapshot = service.OperationSnapshot(
    op_id=service.new_operation_id(), kind="open_library",
    library_spec=spec, request=None, policy=None, lights=(),
)
ctl.start(snapshot, v1.CancellationToken())
assert blocked.wait(10.0), "worker never reached the blocked call"

# finalize() with a small timeout: the worker is wedged, so wait() times out.
ctl.finalize(200)

# The references must be RETAINED because the thread did not stop.
assert ctl._thread is not None, "thread reference was dropped while still running"
assert ctl._worker is not None, "worker reference was dropped while still running"
assert ctl._thread.isRunning(), "thread should still be running after the wait timed out"

# Unwedge and drain so the child exits cleanly (rc 0, no native abort).
blocker.set()
start = time.monotonic()
while time.monotonic() - start < 10.0:
    QtWidgets.QApplication.processEvents()
    if ctl._thread is not None and not ctl._thread.isRunning():
        break
    time.sleep(0.01)
ctl.finalize(2000)
assert ctl._thread is None, "references were not released after the thread stopped"
assert ctl._worker is None, "worker reference was not released after the thread stopped"

print("EXIT_CLEAN")
sys.exit(0)
'''


def test_wedged_finalize_retains_running_thread(tmp_path):
    """REWORK-2: finalize() on a wedged thread must NOT destroy a running QThread.

    Wedges the worker thread (a blocking queued slot that never returns), calls
    ``finalize(small_timeout_ms)``, and asserts the child exits 0 with no
    "QThread: Destroyed while thread is still running" abort, and that the
    worker/thread references were retained (not dropped) because ``wait()`` timed
    out. Isolated in a subprocess so a native abort surfaces as a non-zero exit.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _WEDGED_FINALIZE_SCRIPT, _SRC],
        capture_output=True, text=True, env=_env(), timeout=60,
    )
    assert proc.returncode == 0, (
        f"wedged-finalize subprocess exited rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "EXIT_CLEAN" in proc.stdout
    assert "Destroyed while thread is still running" not in proc.stderr
