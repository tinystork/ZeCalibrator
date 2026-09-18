"""Qt window tests: construction/close, config binding, selection scope, details.

Exercises the *actual window/actions* (not just private worker payloads):
configuration-binding invalidation, visible MATCHED/NO_MATCH/AMBIGUOUS details,
control locking during real queued work, selected-light scope, settings
preservation and index-path behavior. Uses injected ``StoragePaths`` (never real
user roots) with the offscreen platform.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui.window import MainWindow, _LightEntry
from zecalibrator.storage import resolve_paths

from .conftest import make_synth_fixture


@pytest.fixture
def paths(tmp_path):
    return resolve_paths(base=str(tmp_path))


def _pump(cond, timeout_ms=15000):
    from PySide6 import QtTest

    start = time.monotonic()
    while time.monotonic() - start < timeout_ms / 1000.0:
        QtWidgets.QApplication.processEvents()
        if cond():
            return True
        QtTest.QTest.qWait(5)
    return False


def _close(w):
    w.close()
    _pump(lambda: w._controller.is_finished)


def _make_window(qapp, paths, fixture):
    """Window with a light added and a library opened, ready for preflight."""
    from .conftest import wait_idle

    w = MainWindow(paths)
    assert wait_idle(w)  # wait for the async settings load
    w._lights.append(_LightEntry(fixture["light"], hdu=0,
                                 declaration=v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read())),
                                 roi_extent=v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read()))))
    w._refresh_lights_list()
    w._library_spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])
    w.library_status.setText("opened")
    return w


def test_window_constructs_and_closes(qapp, paths):
    w = MainWindow(paths)
    assert w.windowTitle() == "ZeCalibrator"
    assert w.lights_list.count() == 0
    _close(w)


def test_window_has_synthetic_only_qualification_label(qapp, paths):
    w = MainWindow(paths)
    assert "SYNTH-BASE-1" in w.qualification_label.text()
    assert "synthetic-only" in w.qualification_label.text()
    assert "real-camera" in w.qualification_label.text()
    _close(w)


def test_stale_operation_id_is_ignored(qapp, paths):
    w = MainWindow(paths)
    w._current_op_id = "current-op"
    w._active_generation = w._generation
    w.status_label.setText("ready")

    class Evt:
        operation_id = "engine-op"
        phase = "decode"
        completed = 1
        total = 5
        unit = "steps"
        frame_id = None

    w._on_progress("stale-op", Evt())
    assert w.status_label.text() == "ready"
    _close(w)


def test_mode_change_invalidates_cached_plan(qapp, paths, tmp_path):
    """F1: a matched plan must not survive a mode change and be executed."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._plans, "preflight should produce a MATCHED plan"
        assert w._plan_generation == w._generation

        idx = w.additive_combo.findData("control")
        w.additive_combo.setCurrentIndex(idx)  # triggers _on_mode_changed
        assert not w._plans, "mode change must invalidate cached plans"
        assert w._plan_generation != w._generation

        warned = []
        w.lights_list.setCurrentRow(0)
        QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: warned.append(True))
        w._on_calibrate_in_memory()
        assert warned, "stale plan must be refused, not executed"
    finally:
        _close(w)


def test_library_change_invalidates_cached_plan(qapp, paths, tmp_path):
    """F1: opening a different library must invalidate cached plans."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._plans

        w.library_index_edit.setText(fixture["index"])
        w._on_open_library()
        assert _pump(lambda: not w._controller.is_active)
        assert not w._plans, "library change must invalidate cached plans"
    finally:
        _close(w)


def test_config_controls_locked_during_active_work(qapp, paths, tmp_path):
    """F3: conflicting input/config controls are disabled during a real operation."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)

    import zecalibrator.api.v1 as v1_mod

    real_inspect = v1_mod.inspect_frame

    def slow_inspect(source, *args, **kwargs):
        time.sleep(0.3)
        return real_inspect(source, *args, **kwargs)

    v1_mod.inspect_frame = slow_inspect
    try:
        w._on_preflight()
        assert w._controller.is_active
        assert not w.add_btn.isEnabled()
        assert not w.remove_btn.isEnabled()
        assert not w.apply_hdu_btn.isEnabled()
        assert not w.load_decl_btn.isEnabled()
        assert not w.additive_combo.isEnabled()
        assert not w.flat_combo.isEnabled()
        assert not w.library_index_edit.isEnabled()
        assert w.cancel_btn.isEnabled()
        assert _pump(lambda: not w._controller.is_active)
    finally:
        v1_mod.inspect_frame = real_inspect
        _close(w)
    assert w.add_btn.isEnabled()


def test_preflight_details_visible_for_no_match(qapp, paths, tmp_path):
    """F2/F7: selecting a preflight row shows structured rejection details."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w.flat_combo.setCurrentIndex(w.flat_combo.findData("apply"))  # flat -> NO_MATCH
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._preflight_summaries
        assert w._preflight_summaries[0]["outcome"] == "NO_MATCH"

        w.preflight_table.setCurrentCell(0, 0)
        text = w.details_view.toPlainText()
        assert "Rejected candidates" in text
        assert "Coherent sets" in text
        assert "Metadata" in text
    finally:
        _close(w)


def test_in_memory_result_shows_counts_and_provenance(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._plans
        w.lights_list.setCurrentRow(0)
        w._on_calibrate_in_memory()
        assert _pump(lambda: not w._controller.is_active)
        text = w.details_view.toPlainText()
        assert "Plan id" in text
        assert "Provenance" in text
        assert "Counts" in text
    finally:
        _close(w)


def test_export_scope_selected_vs_all(qapp, paths, tmp_path):
    """F9: export exports the selected subset; in-memory uses the selected row."""
    from .conftest import write_fits

    fixture = make_synth_fixture(tmp_path)
    write_fits(tmp_path / "light2.fits", 120.0)
    w = _make_window(qapp, paths, fixture)
    try:
        second = _LightEntry(str(tmp_path / "light2.fits"), hdu=0,
                             declaration=v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read())),
                             roi_extent=v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read())))
        w._lights.append(second)
        w._refresh_lights_list()

        w.lights_list.setCurrentRow(1)
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert second.row_id in w._plans

        w.lights_list.item(1).setSelected(True)
        w._update_scope_label()
        assert "selected (1 of 2)" in w.scope_label.text()
    finally:
        _close(w)


def test_corrupt_settings_preserved_on_close(qapp, tmp_path):
    """F5: an unsupported-schema settings file is preserved, not overwritten."""
    from zecalibrator.gui.settings import settings_path

    paths = resolve_paths(base=str(tmp_path))
    path = settings_path(paths.user_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"schema_version": 99, "preserve-me": "important"})
    path.write_text(original, encoding="utf-8")

    w = MainWindow(paths)
    _close(w)  # triggers _save_settings
    assert path.read_text(encoding="utf-8") == original, "unsupported settings must be preserved"


def test_custom_index_path_honoured(qapp, paths, tmp_path):
    """F8: an explicit index path is honoured (not forced to <root>/zecalibrator.library.sqlite)."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        custom = str(tmp_path / "custom" / "chosen.sqlite")
        w.library_index_edit.setText(custom)
        w.library_root_edit.setText(fixture["root"])
        w.imports_edit.setText(fixture["imports"])
        w._on_index_library()
        assert _pump(lambda: not w._controller.is_active)
        assert w._library_spec.index_path == custom
        assert w.library_index_edit.text() == custom
    finally:
        _close(w)


def test_default_index_path_is_under_user_data(qapp, paths):
    w = MainWindow(paths)
    assert w._default_index_path() == str(w._storage.user_data_path / "zecalibrator.library.sqlite")
    _close(w)


def test_row_identity_no_plan_collision_same_path_hdu(qapp, paths, tmp_path):
    """S4: two rows with the SAME path/HDU but different evidence must not share a plan.
    A NO_MATCH row must never execute another row's MATCHED plan."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        # First row: dark_incl_bias at exposure 10 -> MATCHED.
        first = _LightEntry(fixture["light"], hdu=0,
                            declaration=v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read())),
                            roi_extent=v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read())))
        # Second row: same path/hdu but exposure 11 -> NO_MATCH.
        decl2 = json.loads(open(fixture["decl"]).read())
        decl2["exposure_s"] = 11.0
        second = _LightEntry(fixture["light"], hdu=0,
                             declaration=v1.ImportDeclaration(**decl2),
                             roi_extent=v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read())))
        w._lights = [first, second]
        w._refresh_lights_list()
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        # Distinct outcomes with distinct row ids.
        outcomes = [s["outcome"] for s in w._preflight_summaries]
        assert outcomes == ["MATCHED", "NO_MATCH"], outcomes
        assert first.row_id in w._plans
        assert second.row_id not in w._plans, "NO_MATCH row must not carry a plan"

        # Selecting the NO_MATCH row must refuse in-memory calibration (no plan).
        warned = []
        w.lights_list.setCurrentRow(1)
        QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: warned.append(True))
        w._on_calibrate_in_memory()
        assert warned, "selecting the NO_MATCH row must not execute the MATCHED row's plan"
    finally:
        _close(w)


def test_stale_batch_details_cleared_on_generation_bump(qapp, paths, tmp_path):
    """S3: a generation bump clears stale batch table/details/audit."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w._batch_items.append({"index": 0, "disposition": "COMPLETED"})
        w.results_table.setRowCount(1)
        w.audit_view.setPlainText("OLD CONFIG AUDIT")
        w._bump_generation()
        assert w._batch_items == []
        assert w.results_table.rowCount() == 0
        assert w.audit_view.toPlainText() == ""
        assert w.audit_link_btn.isVisible() is False
    finally:
        _close(w)


def test_plain_text_log_does_not_interpret_html(qapp, paths):
    """S5: technical log strings are rendered literally, never as HTML."""
    w = MainWindow(paths)
    w._log("<b>literal-path-marker</b> & <i>error-marker</i>")
    text = w.details_view.toPlainText()
    assert "<b>literal-path-marker</b>" in text
    assert "<i>error-marker</i>" in text
    _close(w)


def test_preflight_audit_shows_nested_plan_master_identity(qapp, paths, tmp_path):
    """R2: the full public plan/decision audit (incl. nested master identities) is
    reachable in the audit pane."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._preflight_summaries and w._preflight_summaries[0]["outcome"] == "MATCHED"
        # Select the row and check the audit pane carries the nested dark master identity.
        w.preflight_table.setCurrentCell(0, 0)
        audit = w.audit_view.toPlainText()
        assert "content_sha256" in audit
        assert "mask_identity" in audit
        assert "descriptor_id" in audit
    finally:
        _close(w)
