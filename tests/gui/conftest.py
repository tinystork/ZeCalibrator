"""Shared fixtures for the GUI test suite.

Provides the synthetic SYNTH-BASE-1 fixture (light + dark master + mask +
imports, indexed via the public ``zecalibrator.api.v1`` only) plus a small
blocking harness that runs one worker operation on the shared Qt application and
collects its queued events. Synthetic facts are explicit, never real detector
defaults.
"""

from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import pytest
from astropy.io import fits

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SHAPE = (4, 4)

DECL = dict(
    source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
    domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
    detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
    adc_mode="MODE_16", binning=[1, 1], sensor_dimensions=[4, 4],
    orientation="identity", cfa_phase="mono", roi_origin=[0, 0], exposure_s=10.0,
    temperature_c=20.0, filter="NONE", optical_train_id="SYNTH-TRAIN-1",
    bias_exposure_max_s=0.01, saturation_limit_adu=60000.0, saturation_evidence="qualified",
)
ROI = dict(extent=[4, 4], source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0")


def write_fits(path, value, bunit="ADU"):
    hdu = fits.PrimaryHDU(np.full(SHAPE, value, dtype=np.float32))
    hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)
    return str(path)


def npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.uint16), allow_pickle=False)
    return buf.getvalue()


def make_dark_imports(tmp_path):
    """Return the master-imports list (dark master) for ``index_library``."""
    return [dict(
        path="dark.fits", master_type="dark", hdu=0, mask_path="dark.mask.npy",
        bias_state="included", declaration=DECL,
    )]


def make_synth_fixture(tmp_path, *, light_value=100.0, dark_value=10.0, dark_mask=None):
    """Write a dark master + light, index the library, return a path dict.

    ``dark_mask`` (optional uint16 array) supplies a non-zero DQ master mask; when
    omitted, an all-zero mask is used.
    """
    import zecalibrator.api.v1 as v1
    from zecalibrator.gui import service

    dark = tmp_path / "dark.fits"
    write_fits(dark, dark_value)
    mask = np.zeros(SHAPE, dtype=np.uint16) if dark_mask is None else np.asarray(dark_mask, dtype=np.uint16)
    (tmp_path / "dark.mask.npy").write_bytes(npy_bytes(mask))
    light = tmp_path / "light.fits"
    write_fits(light, light_value)

    decl_path = tmp_path / "light.decl.json"
    decl_path.write_text(json.dumps(DECL))
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(ROI))
    imports_path = tmp_path / "imports.json"
    imports_path.write_text(json.dumps(make_dark_imports(tmp_path)))

    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "zecalibrator.library.sqlite"))
    result = v1.index_library(spec, service.parse_imports(make_dark_imports(tmp_path)))
    assert result.operation_status == "COMPLETED", result.details

    return {
        "light": str(light),
        "dark": str(dark),
        "mask": str(tmp_path / "dark.mask.npy"),
        "imports": str(imports_path),
        "decl": str(decl_path),
        "roi": str(roi_path),
        "index": str(tmp_path / "zecalibrator.library.sqlite"),
        "root": str(tmp_path),
    }


def wait_idle(window, timeout_ms=20000) -> bool:
    """Pump Qt events until the window's worker is idle (no active operation)."""
    from PySide6 import QtTest, QtWidgets

    start = time.monotonic()
    while time.monotonic() - start < timeout_ms / 1000.0:
        QtWidgets.QApplication.processEvents()
        if not window._controller.is_active:
            return True
        QtTest.QTest.qWait(5)
    return False


class RunResult:
    def __init__(self):
        self.finished = []  # list[(op_id, summary)]
        self.failed = []  # list[(op_id, reason_code, details)]
        self.progress = []  # list[(op_id, event)]
        self.preflight = []  # list[(op_id, summary, plan)]
        self.batch_items = []  # list[(op_id, item)]
        self.ended = False
        self.timed_out = False

    def finished_summary(self):
        assert self.finished, f"no finished summary; failed={self.failed}"
        return self.finished[-1][1]


@pytest.fixture
def controller(qapp):
    """A fresh WorkerController per test (finished-driven shutdown at teardown)."""
    from PySide6 import QtCore

    from zecalibrator.gui.worker import WorkerController

    ctl = WorkerController()
    yield ctl
    ctl.shutdown()
    # Pump the event loop until the thread has actually finished (finished-driven
    # cleanup; never block the GUI thread with a synchronous wait).
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(20000)
    ctl.shutdown_finished.connect(loop.quit)
    if not ctl.is_finished:
        loop.exec()
    # Bounded post-loop join: once the thread has stopped, finalize() releases the
    # worker/thread references deterministically (instead of relying on controller
    # collection while the thread may still be winding down).
    ctl.finalize()


def run_operation(controller, snapshot, token, timeout_ms=20000) -> RunResult:
    """Start one operation and block (nested event loop) until it ends.

    Collects events through a small QObject ``RunCollector`` whose handlers are
    plain bound methods — no transient lambda/free-function receivers (a
    documented PySide6 GC-segfault pattern). The collector owns the
    ``QEventLoop`` and the timeout ``QTimer`` for the whole operation; the timer
    is stopped and exactly the connections made here are disconnected before the
    collector is released.

    The handlers are deliberately NOT decorated with ``@QtCore.Slot``: in
    PySide6 6.11.2, ``@Slot`` handlers that take ``object`` arguments (e.g.
    ``@Slot(str, object, object)``) corrupt the refcounts of those objects when
    invoked against signals whose values cross the worker→GUI thread boundary —
    a native double-free that surfaces as "Fatal Python error: Aborted" during
    the next GC pass. Plain bound methods on the collector take the same
    arguments with the same whole-operation lifetime but without that
    marshalling path.
    """
    from PySide6 import QtCore

    class RunCollector(QtCore.QObject):
        """QObject result collector with an explicit, whole-operation lifetime."""

        def __init__(self, result, loop, timer):
            super().__init__()
            self.result = result
            self.loop = loop
            self.timer = timer

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
            self.loop.quit()

    result = RunResult()
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    collector = RunCollector(result, loop, timer)

    connections = [
        (controller.operation_finished,
         controller.operation_finished.connect(collector.on_finished)),
        (controller.operation_failed,
         controller.operation_failed.connect(collector.on_failed)),
        (controller.progress,
         controller.progress.connect(collector.on_progress)),
        (controller.preflight_light,
         controller.preflight_light.connect(collector.on_preflight)),
        (controller.batch_item,
         controller.batch_item.connect(collector.on_batch_item)),
        (controller.worker_ended,
         controller.worker_ended.connect(collector.on_worker_ended)),
    ]

    timer.start(timeout_ms)
    controller.start(snapshot, token)
    loop.exec()

    result.timed_out = not result.ended
    timer.stop()
    for sig, handle in connections:
        sig.disconnect(handle)
    return result
