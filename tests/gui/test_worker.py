"""Qt worker tests: one active operation, queued events, progress, cancellation.

Exercises the real :class:`WorkerController` against the synthetic SYNTH-BASE-1
fixture using only ``zecalibrator.api.v1``. All matching states (MATCHED /
NO_MATCH / AMBIGUOUS), structured rejection display, single in-memory and
standalone one/many export, progress delivery, cancellation phases and manifest
failure are covered.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service
from zecalibrator.gui.worker import ProgressMailbox

from .conftest import SHAPE, make_dark_imports, make_synth_fixture, run_operation


def _library_spec(fixture):
    return v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])


def _declaration():
    from .conftest import DECL

    return service.parse_declaration(DECL)


def _roi():
    from .conftest import ROI

    return service.parse_roi(ROI)


def _light(fixture):
    return service.LightInput(
        path=fixture["light"], hdu=0, declaration=_declaration(),
        roi_extent=_roi(), display_name="light.fits",
    )


def _preflight_snapshot(fixture, *, request=None, lights=None):
    return service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="preflight",
        library_spec=_library_spec(fixture),
        request=request or v1.CalibrationRequest("dark_incl_bias", "none"),
        policy=v1.default_match_policy(),
        lights=lights if lights is not None else (_light(fixture),),
    )


def _export_snapshot(fixture, destination, *, lights=None, request=None):
    return service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="export",
        library_spec=_library_spec(fixture),
        request=request or v1.CalibrationRequest("dark_incl_bias", "none"),
        policy=v1.default_match_policy(),
        lights=lights if lights is not None else (_light(fixture),),
        destination=destination, batch_id=service.new_batch_id(),
    )


# ---------------------------------------------------------------------------
# open / index
# ---------------------------------------------------------------------------
def test_open_library_operation(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    res = run_operation(controller, _preflight_snapshot(fixture), v1.CancellationToken())
    assert not res.timed_out
    assert res.finished


def test_index_library_operation(controller, tmp_path):
    from .conftest import write_fits, npy_bytes

    write_fits(tmp_path / "dark.fits", 10.0)
    (tmp_path / "dark.mask.npy").write_bytes(npy_bytes(np.zeros(SHAPE, dtype=np.uint16)))
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "new.sqlite"))
    snapshot = service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="index_library",
        library_spec=spec, request=None, policy=None, lights=(),
        index_imports=service.parse_imports(make_dark_imports(tmp_path)),
    )
    res = run_operation(controller, snapshot, v1.CancellationToken())
    assert not res.timed_out
    summary = res.finished_summary()
    assert summary["kind"] == "index_library"
    assert summary["status"] == "COMPLETED"
    assert summary["candidate_count"] == 1


# ---------------------------------------------------------------------------
# preflight matching states
# ---------------------------------------------------------------------------
def test_preflight_matched(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    res = run_operation(controller, _preflight_snapshot(fixture), v1.CancellationToken())
    assert not res.timed_out
    summary = res.finished_summary()
    assert summary["status"] == "COMPLETED"
    assert len(res.preflight) == 1
    op_id, light_summary, plan = res.preflight[0]
    assert light_summary["outcome"] == "MATCHED"
    assert light_summary["plan_id"]
    assert plan is not None
    assert "dark" in plan.masters


def test_preflight_no_match(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    snap = _preflight_snapshot(fixture, request=v1.CalibrationRequest("dark_incl_bias", "apply"))
    res = run_operation(controller, snap, v1.CancellationToken())
    assert not res.timed_out
    light_summary = res.preflight[0][1]
    assert light_summary["outcome"] == "NO_MATCH"
    assert light_summary["plan_id"] is None
    assert res.preflight[0][2] is None


def test_preflight_ambiguous(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    from .conftest import write_fits, npy_bytes, DECL

    write_fits(tmp_path / "dark2.fits", 12.0)
    (tmp_path / "dark2.mask.npy").write_bytes(npy_bytes(np.zeros(SHAPE, dtype=np.uint16)))
    imports = make_dark_imports(tmp_path) + [dict(
        path="dark2.fits", master_type="dark", hdu=0, mask_path="dark2.mask.npy",
        bias_state="included", declaration=DECL,
    )]
    spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])
    v1.index_library(spec, service.parse_imports(imports))
    res = run_operation(controller, _preflight_snapshot(fixture), v1.CancellationToken())
    assert not res.timed_out
    light_summary = res.preflight[0][1]
    assert light_summary["outcome"] == "AMBIGUOUS"
    assert len(light_summary["coherent_sets"]) == 2
    assert light_summary["plan_id"] is None


def test_preflight_rejection_reasons_structured(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    snap = _preflight_snapshot(fixture, request=v1.CalibrationRequest("dark_incl_bias", "apply"))
    res = run_operation(controller, snap, v1.CancellationToken())
    light_summary = res.preflight[0][1]
    assert light_summary["outcome"] == "NO_MATCH"
    assert "ROLE_UNAVAILABLE" in light_summary["reason_codes"]


# ---------------------------------------------------------------------------
# in-memory single calibration
# ---------------------------------------------------------------------------
def test_calibrate_in_memory_dark(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    pre = run_operation(controller, _preflight_snapshot(fixture), v1.CancellationToken())
    plan = pre.preflight[0][2]
    snap = service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="calibrate_in_memory",
        library_spec=None, request=None, policy=None,
        lights=(_light(fixture),), plan=plan,
    )
    res = run_operation(controller, snap, v1.CancellationToken())
    assert not res.timed_out
    summary = res.finished_summary()
    assert summary["kind"] == "calibrate_in_memory"
    assert summary["status"] == "COMPLETED"
    assert summary["plan_id"] == plan.plan_id
    assert summary["provenance"]["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# standalone export (single + batch) + manifest
# ---------------------------------------------------------------------------
def test_export_single_standalone(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    res = run_operation(controller, _export_snapshot(fixture, str(out)), v1.CancellationToken())
    assert not res.timed_out
    summary = res.finished_summary()
    assert summary["status"] == "COMPLETED"
    assert len(summary["items"]) == 1
    assert summary["items"][0]["disposition"] == "COMPLETED"
    assert summary["items"][0]["output"] is not None
    assert os.path.exists(summary["manifest"])
    assert len([f for f in os.listdir(str(out)) if f.endswith(".fits")]) == 1


def test_export_batch_ordered(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    from .conftest import write_fits

    write_fits(tmp_path / "light2.fits", 120.0)
    light2 = service.LightInput(
        path=str(tmp_path / "light2.fits"), hdu=0, declaration=_declaration(),
        roi_extent=_roi(), display_name="light2.fits",
    )
    out = tmp_path / "out"
    out.mkdir()
    snap = _export_snapshot(fixture, str(out), lights=(_light(fixture), light2))
    res = run_operation(controller, snap, v1.CancellationToken())
    assert not res.timed_out
    summary = res.finished_summary()
    assert summary["status"] == "COMPLETED"
    assert [i["index"] for i in summary["items"]] == [0, 1]
    assert all(i["disposition"] == "COMPLETED" for i in summary["items"])
    assert len([f for f in os.listdir(str(out)) if f.endswith(".fits")]) == 2


def test_export_partial_when_no_match(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    snap = _export_snapshot(
        fixture, str(out), request=v1.CalibrationRequest("dark_incl_bias", "apply")
    )
    res = run_operation(controller, snap, v1.CancellationToken())
    summary = res.finished_summary()
    assert summary["status"] == "PARTIAL"
    assert summary["items"][0]["disposition"] == "FAILED"


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------
def test_progress_events_engine_sourced(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    res = run_operation(controller, _preflight_snapshot(fixture), v1.CancellationToken())
    assert not res.timed_out
    assert res.progress
    # every event carries an engine operation id and phase; events are ProgressEvent-like
    for _, event in res.progress:
        assert getattr(event, "operation_id", None)
        assert getattr(event, "phase", None)


def test_progress_mailbox_bounds_to_latest_event():
    mailbox = ProgressMailbox()

    class Evt:
        def __init__(self, phase, completed):
            self.phase = phase
            self.completed = completed
            self.total = 3
            self.unit = "steps"
            self.frame_id = None
            self.operation_id = "op"

    mailbox.post(Evt("batch_start", 0))
    mailbox.post(Evt("frame_start", 0))
    mailbox.post(Evt("frame_complete", 1))
    # Only the latest survives (bounded single slot); take drains it.
    latest = mailbox.take()
    assert latest.phase == "frame_complete"
    assert mailbox.take() is None
    mailbox.post(Evt("complete", 3))
    assert mailbox.take().phase == "complete"


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------
def test_cancel_before_start_export(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    token = v1.CancellationToken()
    token.cancel()
    res = run_operation(controller, _export_snapshot(fixture, str(out)), token)
    summary = res.finished_summary()
    assert summary["status"] == "CANCELLED"
    assert [f for f in os.listdir(str(out)) if f.endswith(".fits")] == []


def test_cancel_between_batch_items_preserves_committed(controller, tmp_path, monkeypatch):
    fixture = make_synth_fixture(tmp_path)
    from .conftest import write_fits

    write_fits(tmp_path / "light2.fits", 120.0)
    light2 = service.LightInput(
        path=str(tmp_path / "light2.fits"), hdu=0, declaration=_declaration(),
        roi_extent=_roi(), display_name="light2.fits",
    )
    out = tmp_path / "out"
    out.mkdir()
    token = v1.CancellationToken()

    # Deterministic: cancel the shared token synchronously (in the worker
    # thread) at the frame boundary, immediately after frame 0 commits and
    # before frame 1 begins. This mirrors the "cancel between batch items"
    # checkpoint without racing the main thread.
    import zecalibrator.api.v1.batch as batch_mod

    real_emit = batch_mod.emit_batch_progress

    def emit_and_cancel(obs, phase, completed, total, frame_id=None):
        real_emit(obs, phase, completed, total, frame_id=frame_id)
        if phase == "frame_complete" and completed == 1:
            token.cancel()

    monkeypatch.setattr(batch_mod, "emit_batch_progress", emit_and_cancel)
    snap = _export_snapshot(fixture, str(out), lights=(_light(fixture), light2))
    res = run_operation(controller, snap, token)
    summary = res.finished_summary()
    assert summary["status"] == "CANCELLED"
    assert len([f for f in os.listdir(str(out)) if f.endswith(".fits")]) == 1


def test_cancel_during_single_in_memory(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    pre = run_operation(controller, _preflight_snapshot(fixture), v1.CancellationToken())
    plan = pre.preflight[0][2]

    class AutoCancel(v1.CancellationToken):
        def __init__(self):
            super().__init__()
            self._n = 0

        def raise_if_cancelled(self):
            self._n += 1
            if self._n >= 2:
                self.cancel()
            super().raise_if_cancelled()

    snap = service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="calibrate_in_memory",
        library_spec=None, request=None, policy=None,
        lights=(_light(fixture),), plan=plan,
    )
    res = run_operation(controller, snap, AutoCancel())
    summary = res.finished_summary()
    assert summary["status"] == "CANCELLED"


# ---------------------------------------------------------------------------
# manifest failure
# ---------------------------------------------------------------------------
def test_manifest_write_failure_is_reported(controller, tmp_path, monkeypatch):
    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    import zecalibrator.api.v1.batch as batch_mod

    def boom(path, manifest):
        raise OSError("disk full")

    monkeypatch.setattr(batch_mod, "write_batch_manifest", boom)
    res = run_operation(controller, _export_snapshot(fixture, str(out)), v1.CancellationToken())
    summary = res.finished_summary()
    assert summary["status"] == "FAILED"
    assert summary["reason_code"] == "MANIFEST_WRITE_FAILED"
    assert "disk full" in summary["details"]
    # items already delivered before the manifest failure are retained
    assert len(summary["items"]) == 1
    assert len(res.batch_items) == 1


# ---------------------------------------------------------------------------
# worker boundary: repeated operation, no overlap, no live-thread destruction
# ---------------------------------------------------------------------------
def test_repeated_operations_on_same_controller(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    for _ in range(3):
        res = run_operation(controller, _preflight_snapshot(fixture), v1.CancellationToken())
        assert not res.timed_out
        assert res.finished_summary()["status"] == "COMPLETED"


def test_second_start_while_active_is_refused(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    controller.start(_preflight_snapshot(fixture), v1.CancellationToken())
    with pytest.raises(RuntimeError):
        controller.start(_preflight_snapshot(fixture), v1.CancellationToken())
    # drain to completion
    from PySide6 import QtCore

    loop = QtCore.QEventLoop()
    controller.worker_ended.connect(loop.quit)
    QtCore.QTimer.singleShot(20000, loop.quit)
    loop.exec()
