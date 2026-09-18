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


def run_operation(controller, snapshot, token, timeout_ms=20000) -> RunResult:
    """Start one operation and block (nested event loop) until it ends."""
    from PySide6 import QtCore

    result = RunResult()
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)

    def on_ended():
        result.ended = True
        loop.quit()

    connections = [
        (controller.operation_finished,
         controller.operation_finished.connect(lambda op, s: result.finished.append((op, s)))),
        (controller.operation_failed,
         controller.operation_failed.connect(lambda op, rc, d: result.failed.append((op, rc, d)))),
        (controller.progress,
         controller.progress.connect(lambda op, e: result.progress.append((op, e)))),
        (controller.preflight_light,
         controller.preflight_light.connect(lambda op, s, p: result.preflight.append((op, s, p)))),
        (controller.batch_item,
         controller.batch_item.connect(lambda op, it: result.batch_items.append((op, it)))),
        (controller.worker_ended, controller.worker_ended.connect(on_ended)),
    ]

    controller.start(snapshot, token)
    loop.exec()

    result.timed_out = not result.ended
    for sig, handle in connections:
        sig.disconnect(handle)
    return result
