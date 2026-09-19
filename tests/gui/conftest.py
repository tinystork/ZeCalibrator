"""Shared fixtures for the GUI test suite.

Provides the synthetic SYNTH-BASE-1 fixture (light + dark master + mask +
imports, indexed via the public ``zecalibrator.api.v1`` only) plus a small
blocking harness that runs one worker operation on the shared Qt application and
collects its queued events. Synthetic facts are explicit, never real detector
defaults.

Harness-architecture note (diagnostic isolation — the whole point of this file):
the GUI main Qt process must reproduce the PRODUCTION architecture, where the
FITS I/O library and all FITS object construction live ONLY on the worker thread
(``zecalibrator.io.raw_decoder`` owns that import; the GUI controller never
does). The harness therefore never imports or constructs FITS library
objects on the main thread:

* ``write_fits`` emits a minimal, valid FITS primary HDU directly from NumPy
  (header cards + big-endian float32 data) — no FITS library is imported or
  constructed in-process, and no subprocess is involved.
* ``run_operation`` and the ``controller`` teardown use a manual pump loop
  (``QApplication.processEvents()`` against a ``time.monotonic()`` deadline)
  with NO ``QEventLoop``/``QTimer`` and NO QObject result collector: the result
  collector is a plain Python object whose bound methods are connected to the
  controller signals for the WHOLE operation and disconnected before returning.

Both measures remove the transient Qt/Python object graphs (FITS HDU objects,
ephemeral event loops/timers, QObject collectors) that could otherwise be
finalized LATER by cyclic GC while the worker thread is active — exactly the
"gc.disable() suppresses the crash" signature that motivated this rewrite.
"""

from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import pytest

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


_FITS_BLOCK = 2880
_FITS_VALUE_FIELD = 20


def _fits_card(keyword, value) -> str:
    """Format one 80-byte FITS header card.

    Booleans and integers/floats are right-justified in the 20-char value field;
    strings are single-quoted and left-justified (FITS standard). ``HISTORY`` /
    ``COMMENT`` cards are commentary cards (keyword + free text, no ``=``
    separator). ``value is None`` emits an ``END`` card.
    """
    kw = str(keyword).upper()[:8].ljust(8)
    if value is None:  # END card
        return kw.ljust(80)
    if kw.strip() in ("HISTORY", "COMMENT"):
        # Commentary card: text begins at column 9 (after the 8-char keyword).
        return (kw + str(value)).ljust(80)[:80]
    if isinstance(value, bool):
        field = ("T" if value else "F").rjust(_FITS_VALUE_FIELD)
    elif isinstance(value, str):
        field = ("'" + value + "'").ljust(_FITS_VALUE_FIELD)
    else:
        field = str(value).rjust(_FITS_VALUE_FIELD)
    return (kw + "= " + field).ljust(80)[:80]


def _pad_block(raw: bytes, fill: bytes) -> bytes:
    """Pad ``raw`` to the next 2880-byte boundary with ``fill`` bytes."""
    rem = len(raw) % _FITS_BLOCK
    if rem:
        raw += fill * (_FITS_BLOCK - rem)
    return raw


def fits_bytes(data, header_cards=(), *, bunit=None) -> bytes:
    """Return a valid FITS primary-HDU byte stream for ``data`` + header cards.

    A minimal, standard-conformant FITS encoder built entirely with NumPy (no
    FITS library imported or constructed in-process):

    * ``data`` — a 2-D or 3-D NumPy array. int16 is encoded as ``BITPIX=16``,
      float32 as ``BITPIX=-32``; both are stored big-endian (FITS standard).
    * ``NAXIS`` is derived from ``data.ndim``; ``NAXIS1``/``NAXIS2``/``NAXIS3``
      are derived from the (reversed) shape (``NAXIS1`` = last axis).
    * ``header_cards`` — an iterable of ``(keyword, value)`` tuples written in
      order after the structural cards; ``HISTORY``/``COMMENT`` become
      commentary cards, everything else uses the ``=`` value syntax.
    * ``bunit`` — optional BUNIT card (placed before ``header_cards``).

    Header and data sections are each padded to 2880-byte blocks.
    """
    arr = np.ascontiguousarray(data)
    if arr.dtype.kind == "i" and arr.dtype.itemsize == 2:
        bitpix = 16
        big = arr.astype(">i2")
    elif arr.dtype.kind == "f" and arr.dtype.itemsize == 4:
        bitpix = -32
        big = arr.astype(">f4")
    else:
        raise ValueError(
            f"unsupported FITS dtype {arr.dtype!r}; expected int16 or float32"
        )
    if arr.ndim not in (2, 3):
        raise ValueError(f"unsupported FITS ndim {arr.ndim}; expected 2 or 3")

    cards = [
        _fits_card("SIMPLE", True),
        _fits_card("BITPIX", bitpix),
        _fits_card("NAXIS", arr.ndim),
    ]
    # FITS axis 1 is the fastest-varying (last) NumPy axis.
    for i in range(arr.ndim):
        cards.append(_fits_card(f"NAXIS{i + 1}", int(arr.shape[arr.ndim - 1 - i])))
    if bunit is not None:
        cards.append(_fits_card("BUNIT", bunit))
    for item in header_cards:
        cards.append(_fits_card(item[0], item[1]))
    cards.append(_fits_card("END", None))

    header = _pad_block("".join(cards).encode("ascii"), b" ")
    data_bytes = _pad_block(big.tobytes(), b"\x00")
    return header + data_bytes


def write_fits(path, value, bunit="ADU"):
    """Write a synthetic constant float32 FITS file WITHOUT a FITS library.

    The FITS bytes are produced directly from NumPy; the main Qt process never
    imports/constructs FITS library objects. Signature/behavior are unchanged so
    existing tests keep working.
    """
    data = np.full(SHAPE, float(value), dtype=np.float32)
    with open(path, "wb") as fh:
        fh.write(fits_bytes(data, bunit=bunit))
    return str(path)


def write_fits_array(path, data, header_cards=(), *, bunit=None):
    """Write a synthetic FITS primary HDU from a NumPy array + header cards.

    General helper for GUI tests that need int16/float32, 2-D/3-D or extra
    header cards (IMAGETYP/EXPTIME/HISTORY/...) without a FITS library in the
    main process.
    """
    with open(path, "wb") as fh:
        fh.write(fits_bytes(data, header_cards=header_cards, bunit=bunit))
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


def _pump_until(cond, timeout_ms=20000) -> bool:
    """Pump Qt events (no QEventLoop/QTimer) until ``cond()`` or deadline."""
    from PySide6 import QtWidgets

    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.0005)
    return False


def wait_idle(window, timeout_ms=20000) -> bool:
    """Pump Qt events until the window's worker is idle (no active operation).

    Manual pump loop only — no ``QEventLoop``/``QTimer`` and no connections that
    could leak into a later GC pass.
    """
    return _pump_until(lambda: not window._controller.is_active, timeout_ms)


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


class _RunCollector:
    """Plain-Python result collector (NOT a QObject).

    The handlers are plain bound methods connected directly to the controller's
    (main-thread) signals. No ``@QtCore.Slot`` decoration: in PySide6 6.11.2,
    ``@Slot`` handlers taking ``object`` arguments corrupt the refcounts of
    cross-thread relayed values (a native double-free at the next GC pass). A
    plain object with whole-operation lifetime and an explicit disconnect before
    return avoids both that marshalling path and any lingering Qt object graph.
    """

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


@pytest.fixture
def controller(qapp):
    """A fresh WorkerController per test (pump-loop shutdown at teardown).

    Teardown makes NO ``shutdown_finished`` connection and creates no
    ``QEventLoop``/``QTimer``; it pumps ``processEvents()`` until ``is_finished``
    (with a monotonic deadline), then calls ``finalize()`` to release the
    worker/thread references after the thread has actually stopped.
    """
    from zecalibrator.gui.worker import WorkerController

    ctl = WorkerController()
    yield ctl
    ctl.shutdown()
    _pump_until(lambda: ctl.is_finished, 20000)
    ctl.finalize()


def run_operation(controller, snapshot, token, timeout_ms=20000) -> RunResult:
    """Start one operation and block (manual pump loop) until it ends.

    Collects events through a plain Python ``_RunCollector`` whose bound methods
    are connected to the controller signals, kept alive in a local variable for
    the WHOLE operation, and disconnected exactly before returning. There is no
    ``QEventLoop``, no ``QTimer`` and no QObject collector: the terminal
    ``worker_ended`` sets ``result.ended`` and the pump loop observes it against
    a ``time.monotonic()`` deadline. This mirrors production (no transient Qt
    object graph to be finalized later by cyclic GC while the worker is active).
    """
    from PySide6 import QtWidgets

    result = RunResult()
    collector = _RunCollector(result)

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

    deadline = time.monotonic() + timeout_ms / 1000.0
    try:
        controller.start(snapshot, token)
        while not result.ended and time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.0005)
    finally:
        for sig, handle in connections:
            try:
                sig.disconnect(handle)
            except (RuntimeError, TypeError):
                pass

    result.timed_out = not result.ended
    return result
