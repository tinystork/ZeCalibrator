"""Standard Output-folder + collision-dialog UX tests (GUI, offscreen Qt).

Covers the P8 owner UX without touching science:

* the Standard tab exposes a visible, read-only Output folder field + Browse;
* the choice is persisted through the existing ``last_output_dir`` setting;
* Standard export refuses an empty / missing / unwritable destination BEFORE
  calibration (human messages; the writer stays the final authority);
* the collision decision channel reaches the worker exactly once per batch and
  maps Overwrite / Skip / Cancel to the right policy (skip/overwrite/cancel),
  never blocking the GUI thread synchronously;
* no collision -> no dialog at all;
* the modal dialog wording is exactly the owner-specified wording.
"""

from __future__ import annotations

import dataclasses
import os
import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service
from zecalibrator.gui.worker import CollisionDecisionChannel

from .conftest import (
    DECL,
    ROI,
    make_synth_fixture,
    run_operation,
    wait_idle,
)


def _library_spec(fixture):
    return v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])


def _light(fixture):
    return service.LightInput(
        path=fixture["light"], hdu=0, declaration=service.parse_declaration(DECL),
        roi_extent=service.parse_roi(ROI), display_name="light.fits",
    )


def _standard_export_snapshot(fixture, destination, *, lights=None):
    return service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="export",
        library_spec=_library_spec(fixture),
        request=None,  # None => the Standard auto-route path
        policy=v1.default_match_policy(),
        lights=lights if lights is not None else (_light(fixture),),
        destination=destination, batch_id=service.new_batch_id(),
    )


# ---------------------------------------------------------------------------
# Worker-level collision harness (blocking pump + collision decision injection)
# ---------------------------------------------------------------------------
class _CollisionRun:
    def __init__(self):
        self.finished = []
        self.failed = []
        self.collisions = []
        self.ended = False
        self.timed_out = False


def _run_export_with_collision(controller, snapshot, token, decide, timeout_ms=20000):
    """Run one export, auto-resolving any collision query with ``decide``."""
    result = _CollisionRun()

    def on_finished(op_id, summary):
        result.finished.append(summary)

    def on_failed(op_id, reason_code, details):
        result.failed.append((reason_code, details))

    def on_collision(op_id, payload):
        result.collisions.append(payload)
        controller.resolve_collision(decide(payload))

    def on_ended():
        result.ended = True

    connections = [
        (controller.operation_finished, controller.operation_finished.connect(on_finished)),
        (controller.operation_failed, controller.operation_failed.connect(on_failed)),
        (controller.collision_query, controller.collision_query.connect(on_collision)),
        (controller.worker_ended, controller.worker_ended.connect(on_ended)),
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


def _seed_collision(controller, fixture, out):
    """Commit the light once so a second export collides on the same destination."""
    snap = _standard_export_snapshot(fixture, str(out))
    r = run_operation(controller, snap, v1.CancellationToken())
    assert not r.timed_out
    assert r.finished_summary()["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# 1. Standard output-folder field (visible, read-only, persisted)
# ---------------------------------------------------------------------------
def test_standard_output_folder_field_present_and_readonly(qapp, tmp_path):
    from zecalibrator.gui.window import MainWindow
    from zecalibrator.storage import resolve_paths

    w = MainWindow(resolve_paths(base=str(tmp_path)))
    assert wait_idle(w)
    try:
        assert w.output_dir_edit is not None
        assert w.output_browse_btn is not None
        assert w.output_dir_edit.isReadOnly()
        assert w.output_browse_btn.text() == "Browse…"
        # The control lives on the Standard page (tab 0).
        standard_page = w.main_tabs.widget(0)
        assert w.output_dir_edit in set(standard_page.findChildren(QtWidgets.QLineEdit))
        assert w.output_browse_btn in set(standard_page.findChildren(QtWidgets.QPushButton))
    finally:
        w._controller.shutdown()
        QtWidgets.QApplication.processEvents()


def test_standard_output_folder_persists_through_settings(qapp, tmp_path, monkeypatch):
    from zecalibrator.gui.window import MainWindow
    from zecalibrator.storage import resolve_paths

    folder = str(tmp_path / "chosen_out")
    os.makedirs(folder, exist_ok=True)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: folder)
    )
    w = MainWindow(resolve_paths(base=str(tmp_path)))
    assert wait_idle(w)
    try:
        w._on_browse_output()
        assert w.output_dir_edit.text() == folder
        assert w._settings.last_output_dir == folder
    finally:
        w._controller.shutdown()
        QtWidgets.QApplication.processEvents()


def test_standard_output_folder_seeded_from_loaded_settings(qapp, tmp_path, monkeypatch):
    from zecalibrator.gui.window import MainWindow
    from zecalibrator.storage import resolve_paths

    paths = resolve_paths(base=str(tmp_path))
    folder = str(tmp_path / "persisted_out")
    os.makedirs(folder, exist_ok=True)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: folder)
    )

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        w._on_browse_output()
        w._request_settings_save()
        assert wait_idle(w)

        w2 = MainWindow(paths)
        assert wait_idle(w2)
        try:
            assert w2.output_dir_edit.text() == folder
        finally:
            w2._controller.shutdown()
            QtWidgets.QApplication.processEvents()
    finally:
        w._controller.shutdown()
        QtWidgets.QApplication.processEvents()


# ---------------------------------------------------------------------------
# 2. Standard export refuses empty / missing / unwritable destination
# ---------------------------------------------------------------------------
def test_standard_output_destination_validation(qapp, tmp_path, monkeypatch):
    from zecalibrator.gui.window import MainWindow
    from zecalibrator.storage import resolve_paths

    w = MainWindow(resolve_paths(base=str(tmp_path)))
    assert wait_idle(w)
    try:
        # empty
        w.output_dir_edit.setText("")
        dest, err = w._standard_output_destination()
        assert dest is None
        assert err == "Please select an output folder before starting calibration."

        # not a directory
        w.output_dir_edit.setText(str(tmp_path / "missing_dir"))
        dest, err = w._standard_output_destination()
        assert dest is None
        assert err == (
            "The selected output folder no longer exists. "
            "Please choose a valid destination."
        )

        # not writable
        real = str(tmp_path / "readonly")
        os.makedirs(real, exist_ok=True)
        monkeypatch.setattr(os, "access", lambda path, mode: False)
        w.output_dir_edit.setText(real)
        dest, err = w._standard_output_destination()
        assert dest is None
        assert "not writable" in err

        # valid
        monkeypatch.setattr(os, "access", lambda path, mode: True)
        w.output_dir_edit.setText(real)
        dest, err = w._standard_output_destination()
        assert dest == real
        assert err is None
    finally:
        w._controller.shutdown()
        QtWidgets.QApplication.processEvents()


# ---------------------------------------------------------------------------
# 3. Collision -> one decision per batch, mapped to the right policy
# ---------------------------------------------------------------------------
def test_collision_skip_policy(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _seed_collision(controller, fixture, out)

    snap = _standard_export_snapshot(fixture, str(out))
    r = _run_export_with_collision(controller, snap, v1.CancellationToken(), decide=lambda p: "skip")
    assert not r.timed_out
    assert len(r.collisions) == 1
    assert r.collisions[0]["basename"].startswith("zecalibrator_")
    assert r.collisions[0]["folder"] == str(out)
    assert r.finished, f"no finished summary; failed={r.failed}"
    summary = r.finished[-1]
    assert summary["status"] == "COMPLETED"
    assert len(summary["items"]) == 1
    assert summary["items"][0]["disposition"] == "SKIPPED"


def test_collision_overwrite_policy(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _seed_collision(controller, fixture, out)

    snap = _standard_export_snapshot(fixture, str(out))
    r = _run_export_with_collision(controller, snap, v1.CancellationToken(), decide=lambda p: "overwrite")
    assert not r.timed_out
    assert len(r.collisions) == 1
    assert r.finished
    summary = r.finished[-1]
    assert summary["status"] == "COMPLETED"
    assert summary["items"][0]["disposition"] == "COMPLETED"
    assert summary["items"][0]["output"] is not None


def test_collision_cancel_policy(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _seed_collision(controller, fixture, out)

    before = sorted(f for f in os.listdir(str(out)) if f.endswith(".fits"))
    snap = _standard_export_snapshot(fixture, str(out))
    r = _run_export_with_collision(controller, snap, v1.CancellationToken(), decide=lambda p: "cancel")
    assert not r.timed_out
    assert len(r.collisions) == 1
    assert r.finished
    summary = r.finished[-1]
    assert summary["status"] == "CANCELLED"
    # Nothing new was written for the cancelled frame.
    after = sorted(f for f in os.listdir(str(out)) if f.endswith(".fits"))
    assert after == before


def test_no_collision_no_dialog(controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    snap = _standard_export_snapshot(fixture, str(out))
    r = _run_export_with_collision(controller, snap, v1.CancellationToken(), decide=lambda p: "cancel")
    assert not r.timed_out
    assert r.collisions == []  # no collision -> no dialog at all
    assert r.finished
    assert r.finished[-1]["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# 4. CollisionDecisionChannel transport (never Qt Signal(object))
# ---------------------------------------------------------------------------
def test_collision_decision_channel_fails_safe_to_cancel():
    ch = CollisionDecisionChannel()
    assert ch.result == "cancel"  # missing decision fails safe
    ch.set("garbage")
    assert ch.result == "cancel"  # garbled decision fails safe
    ch2 = CollisionDecisionChannel()
    ch2.set("skip")
    assert ch2.result == "skip"
    ch3 = CollisionDecisionChannel()
    ch3.set("overwrite")
    assert ch3.result == "overwrite"


# ---------------------------------------------------------------------------
# 5. Modal dialog wording + action mapping (window handler)
# ---------------------------------------------------------------------------
def test_collision_dialog_wording_and_action_mapping(qapp, tmp_path, monkeypatch):
    from zecalibrator.gui.window import MainWindow
    from zecalibrator.storage import resolve_paths

    w = MainWindow(resolve_paths(base=str(tmp_path)))
    assert wait_idle(w)
    try:
        # Make the window consider an op current so the handler shows the dialog.
        w._current_op_id = "op-test"
        w._active_generation = w._generation

        boxes = []

        class _FakeButton:
            pass

        class _FakeBox:
            chosen = "Skip existing"

            class Icon:
                Warning = object()
                Information = object()

            class ButtonRole:
                AcceptRole = object()
                DestructiveRole = object()
                RejectRole = object()

            def __init__(self, parent=None):
                self.text = None
                self.buttons = {}
                self.clicked = None
                boxes.append(self)

            def setWindowTitle(self, t):
                pass

            def setIcon(self, i):
                pass

            def setText(self, t):
                self.text = t

            def addButton(self, label, role):
                b = _FakeButton()
                b.label = label
                b.role = role
                self.buttons[label] = b
                return b

            def exec(self):
                self.clicked = self.buttons.get(_FakeBox.chosen)

            def clickedButton(self):
                return self.clicked

        decisions = []
        monkeypatch.setattr(QtWidgets, "QMessageBox", _FakeBox)
        monkeypatch.setattr(
            w._controller, "resolve_collision", lambda d: decisions.append(d)
        )

        payload = {"folder": "/out", "path": "/out/x.fits", "basename": "x.fits"}

        # The exact owner wording, asserted on the last shown box.
        _FakeBox.chosen = "Skip existing"
        w._on_collision_query("op-test", payload)
        assert boxes[-1].text == (
            "An output file already exists in the selected folder.\n"
            "Choose how existing output files should be handled for this batch."
        )
        assert set(boxes[-1].buttons) == {"Overwrite existing", "Skip existing", "Cancel"}
        assert decisions[-1] == "skip"

        _FakeBox.chosen = "Overwrite existing"
        w._on_collision_query("op-test", payload)
        assert decisions[-1] == "overwrite"

        _FakeBox.chosen = "Cancel"
        w._on_collision_query("op-test", payload)
        assert decisions[-1] == "cancel"

        # A stale/foreign op id never shows a dialog and releases with "cancel".
        w._on_collision_query("op-stale", payload)
        assert decisions[-1] == "cancel"
    finally:
        w._controller.shutdown()
        QtWidgets.QApplication.processEvents()


# ---------------------------------------------------------------------------
# 6. S1 — abnormal teardown bound (no listener answer + token never cancelled)
# ---------------------------------------------------------------------------
def test_collision_wait_bounded_when_listener_gone_and_token_never_cancelled(
    controller, tmp_path, monkeypatch
):
    """S1: an orphaned collision wait (listener gone, token never cancelled) must
    terminate within the bound, fail safe to cancel, and never publish
    destructively — the pre-existing destination stays byte-identical."""
    import zecalibrator.gui.worker as worker_mod

    fixture = make_synth_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _seed_collision(controller, fixture, out)

    before = sorted(f for f in os.listdir(str(out)) if f.endswith(".fits"))
    assert len(before) == 1
    original = open(os.path.join(str(out), before[0]), "rb").read()

    # Shrink the bound so the test itself is fast and bounded (the production
    # default is a generous multi-minute value).
    monkeypatch.setattr(worker_mod, "_COLLISION_DECISION_TIMEOUT_S", 0.5)

    snap = _standard_export_snapshot(fixture, str(out))
    result = _CollisionRun()

    def on_finished(op_id, summary):
        result.finished.append(summary)

    def on_failed(op_id, reason_code, details):
        result.failed.append((reason_code, details))

    def on_collision(op_id, payload):
        result.collisions.append(payload)
        # Deliberately do NOT resolve: simulate a listener that disappeared.

    def on_ended():
        result.ended = True

    connections = [
        (controller.operation_finished, controller.operation_finished.connect(on_finished)),
        (controller.operation_failed, controller.operation_failed.connect(on_failed)),
        (controller.collision_query, controller.collision_query.connect(on_collision)),
        (controller.worker_ended, controller.worker_ended.connect(on_ended)),
    ]
    deadline = time.monotonic() + 20.0
    try:
        controller.start(snap, v1.CancellationToken())
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

    # The operation terminates — it does NOT hang and the harness does NOT time out.
    assert not result.timed_out
    assert result.ended
    # Exactly one collision query was relayed and never answered.
    assert len(result.collisions) == 1
    # Safe cancellation (never overwrite/skip).
    assert result.finished, f"no finished summary; failed={result.failed}"
    assert result.finished[-1]["status"] == "CANCELLED"
    # No destructive publication: the pre-existing destination is byte-unchanged.
    after = sorted(f for f in os.listdir(str(out)) if f.endswith(".fits"))
    assert after == before
    assert open(os.path.join(str(out), before[0]), "rb").read() == original
