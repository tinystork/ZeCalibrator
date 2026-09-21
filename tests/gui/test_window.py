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
    """F2/F7: selecting a preflight row shows structured rejection details.

    Auto-route (Standard) resolves the route automatically; a light with an
    incompatible exposure yields NEEDS_ATTENTION (NO_MATCH) and its details are
    visible in the Advanced pane.
    """
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        # Incompatible exposure -> no compatible dark -> needs attention.
        decl = json.loads(open(fixture["decl"]).read())
        decl["exposure_s"] = 11.0
        w._lights[0].declaration = v1.ImportDeclaration(**decl)
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


def test_export_summary_counts_warnings_as_committed(qapp, paths):
    """R3D-H: a COMPLETED_WITH_WARNINGS item is both committed AND a warning.

    A successfully committed output that carries warnings must never be
    reported as committed=0 (the pre-R3D-H accounting dropped it from the
    committed count)."""
    w = MainWindow(paths)
    try:
        summary = {
            "status": "COMPLETED_WITH_WARNINGS",
            "total_inputs": 10,
            "input_displays": [f"light{i}" for i in range(10)],
            "items": [
                {"index": i, "disposition": "COMPLETED_WITH_WARNINGS",
                 "plan_id": f"plan-{i}", "reason_code": None}
                for i in range(10)
            ],
        }
        w._handle_export_finished(summary)
        text = w.status_label.text()
        assert "committed=10" in text
        assert "warnings=10" in text
        assert "skipped=0" in text
        assert "failed=0" in text
        assert "not-started=0" in text
    finally:
        _close(w)


def test_export_summary_mixed_dispositions_truthful(qapp, paths):
    """R3D-H: mixed dispositions count committed=COMPLETED+CWW, warnings=CWW."""
    w = MainWindow(paths)
    try:
        items = [
            {"index": 0, "disposition": "COMPLETED", "plan_id": "p0", "reason_code": None},
            {"index": 1, "disposition": "COMPLETED", "plan_id": "p1", "reason_code": None},
            {"index": 2, "disposition": "COMPLETED_WITH_WARNINGS", "plan_id": "p2", "reason_code": None},
            {"index": 3, "disposition": "COMPLETED_WITH_WARNINGS", "plan_id": "p3", "reason_code": None},
            {"index": 4, "disposition": "COMPLETED_WITH_WARNINGS", "plan_id": "p4", "reason_code": None},
            {"index": 5, "disposition": "FAILED", "plan_id": None, "reason_code": "X"},
            {"index": 6, "disposition": "SKIPPED", "plan_id": None, "reason_code": None},
        ]
        summary = {
            "status": "COMPLETED_WITH_WARNINGS",
            "total_inputs": 7,
            "input_displays": [f"light{i}" for i in range(7)],
            "items": items,
        }
        w._handle_export_finished(summary)
        text = w.status_label.text()
        assert "committed=5" in text
        assert "warnings=3" in text
        assert "skipped=1" in text
        assert "failed=1" in text
        assert "not-started=0" in text
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


# ---------------------------------------------------------------------------
# P7-M2 GUI/UX simplification: top-level tabs, human Standard workflow,
# shared Standard/Advanced state, and System/Light/Dark theme.
# ---------------------------------------------------------------------------
def test_top_level_tabs_exact_order_and_standard_default(qapp, paths):
    w = MainWindow(paths)
    assert [w.main_tabs.tabText(i) for i in range(w.main_tabs.count())] == \
        ["Standard", "Advanced", "Settings"]
    assert w.main_tabs.currentIndex() == 0  # Standard is the default view
    _close(w)


def test_standard_exposes_human_workflow(qapp, paths):
    w = MainWindow(paths)
    assert w.add_btn.text() == "Add files…"
    assert w.add_folder_btn.text() == "Add folder…"
    assert w.remove_btn.text() == "Remove selected"
    assert w.clear_btn.text() == "Clear"
    # F2: the explicit SQLite chooser is NOT in Standard; Standard exposes the
    # managed Calibration masters flow only.
    assert not hasattr(w, "choose_library_btn")
    # R3D-D: Standard is a ONE-STEP workflow — add masters (user intent) + the
    # single scientific action (Calibrate / Export). No technical buttons.
    assert w.add_masters_folder_btn.text() == "Add masters folder…"
    assert w.add_masters_file_btn.text() == "Add masters file…"
    assert w.remove_masters_btn.text() == "Remove selected"
    assert w.clear_masters_btn.text() == "Clear"
    assert w.export_btn.text() == "Calibrate / Export…"
    assert w.managed_status_label is not None
    assert not hasattr(w, "preflight_btn")
    assert w.lights_count_label.text() == "0 images selected"
    # Lifecycle footer is globally reachable (Cancel/progress/status exist in the
    # footer below the top-level tabs, outside any single tab page).
    assert w.cancel_btn is not None
    assert w.progress_bar is not None
    assert w.status_label is not None
    _close(w)


def test_standard_has_no_dark_flat_choice(qapp, paths):
    """Standard has no Dark/Flat combos and no technical route/source labels
    (the route is resolved automatically)."""
    w = MainWindow(paths)
    assert not hasattr(w, "standard_additive_combo")
    assert not hasattr(w, "standard_flat_combo")
    assert not hasattr(w, "standard_route_label")
    assert not hasattr(w, "active_source_label")
    assert not hasattr(w, "library_human_status")
    assert not hasattr(w, "preflight_btn")
    _close(w)


def test_advanced_retains_technical_controls(qapp, paths):
    from zecalibrator.gui import presentation

    w = MainWindow(paths)
    assert w.hdu_edit is not None and w.apply_hdu_btn is not None
    assert w.load_decl_btn.text() == "Load declaration JSON…"
    assert w.load_roi_btn.text() == "Load ROI JSON…"
    assert w.open_library_btn.text() == "Open…"
    assert w.index_btn.text() == "Index library…"
    assert w.calibrate_btn.text() == "Calibrate selected in memory"
    assert "SYNTH-BASE-1" in w.qualification_label.text()
    assert "synthetic-only" in w.qualification_label.text()
    assert [w.additive_combo.itemData(i) for i in range(w.additive_combo.count())] == \
        list(presentation.additive_modes())
    assert [w.flat_combo.itemData(i) for i in range(w.flat_combo.count())] == \
        list(presentation.flat_modes())
    _close(w)


def test_tab_switch_does_not_change_state(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        gen = w._generation
        plans = dict(w._plans)
        summaries = list(w._preflight_summaries)
        spec = w._library_spec
        lights = list(w._lights)
        request = w._request()

        for idx in (1, 2, 0):
            w.main_tabs.setCurrentIndex(idx)

        assert w._generation == gen
        assert w._plans == plans
        assert w._preflight_summaries == summaries
        assert w._library_spec == spec
        assert w._lights == lights
        assert w._request() == request
    finally:
        _close(w)


def test_advanced_mode_change_updates_request(qapp, paths):
    """Advanced combos remain independent; no Standard mirror exists anymore."""
    w = MainWindow(paths)
    w.additive_combo.setCurrentIndex(w.additive_combo.findData("dark_bias_removed"))
    assert w._request().additive_mode == "dark_bias_removed"
    w.flat_combo.setCurrentIndex(w.flat_combo.findData("none"))
    assert w._request().flat_mode == "none"
    assert not hasattr(w, "standard_additive_combo")
    _close(w)


def test_single_generation_bump_per_mode_change(qapp, paths):
    w = MainWindow(paths)
    before = w._generation
    w.additive_combo.setCurrentIndex(w.additive_combo.findData("bias_only"))
    assert w._generation == before + 1

    before = w._generation
    w.additive_combo.setCurrentIndex(w.additive_combo.findData("dark_incl_bias"))
    assert w._generation == before + 1
    _close(w)


def test_standard_summary_counts_truthful_human_outcomes(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        # Incompatible exposure -> auto-route needs attention.
        decl = json.loads(open(fixture["decl"]).read())
        decl["exposure_s"] = 11.0
        w._lights[0].declaration = v1.ImportDeclaration(**decl)
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._preflight_summaries[0]["outcome"] == "NO_MATCH"
        assert "needs attention" in w.standard_summary_label.text()
        # Human outcome table shows the human label, not the technical token.
        assert w.standard_results_table.item(0, 1).text() == "Needs attention"
        assert "partial correction" in w.standard_results_table.item(0, 2).text()
    finally:
        _close(w)


def test_standard_summary_matched_shows_ready(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._preflight_summaries[0]["outcome"] == "MATCHED"
        assert "image ready" in w.standard_summary_label.text()
        assert w.standard_results_table.item(0, 1).text() == "Ready"
        assert w.standard_results_table.item(0, 2).text() == \
            "A compatible calibration set was found."
    finally:
        _close(w)


def test_theme_options_exact_and_default_system(qapp, paths):
    w = MainWindow(paths)
    assert [w.theme_combo.itemData(i) for i in range(w.theme_combo.count())] == \
        ["system", "light", "dark"]
    assert w.theme_combo.currentData() == "system"
    _close(w)


def test_theme_change_does_not_bump_scientific_generation(qapp, paths):
    from zecalibrator.gui import theme as theme_mod

    w = MainWindow(paths)
    assert _pump(lambda: w._settings_loaded)
    before = w._generation
    w.theme_combo.setCurrentIndex(w.theme_combo.findData("dark"))
    assert w._generation == before
    assert w._settings.appearance_theme == "dark"
    w.theme_combo.setCurrentIndex(w.theme_combo.findData("light"))
    assert w._generation == before
    assert w._settings.appearance_theme == "light"
    # Restore System for the shared QApplication.
    theme_mod.apply_theme(
        QtWidgets.QApplication.instance(), theme_mod.THEME_SYSTEM,
        theme_mod.get_system_palette(QtWidgets.QApplication.instance()),
    )
    _close(w)


def test_theme_preference_persisted_through_settings(qapp, tmp_path):
    from zecalibrator.gui.settings import save_settings, default_settings, settings_path

    paths = resolve_paths(base=str(tmp_path))
    save_settings(paths.user_config_path, default_settings())
    w = MainWindow(paths)
    assert _pump(lambda: w._settings_loaded)
    w.theme_combo.setCurrentIndex(w.theme_combo.findData("dark"))
    assert w._settings.appearance_theme == "dark"
    _close(w)  # triggers async settings save
    from zecalibrator.gui.settings import load_settings

    loaded = load_settings(paths.user_config_path)
    assert loaded.settings.appearance_theme == "dark"


# ---------------------------------------------------------------------------
# REWORK-1: F1 (truthful idle/progress), F2 (inspect-failure explanation),
# F3 (theme-load race) + acceptance-alignment (summary/scope/token label).
# ---------------------------------------------------------------------------
def test_fresh_window_idle_state_is_truthful(qapp, paths):
    """F1: after the startup settings load reaches idle, the window shows Ready.
    and no visible fake progress indicator (never Working…/0% at idle)."""
    w = MainWindow(paths)
    assert _pump(lambda: w._settings_loaded and not w._controller.is_active)
    assert w.status_label.text() == "Ready."
    assert w.progress_bar.isHidden()
    assert not w.cancel_btn.isEnabled()  # no active operation at idle
    _close(w)


def test_progress_visible_only_during_active_operation(qapp, paths, tmp_path):
    """F1: the progress indicator is shown only while an operation is active."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)

    import zecalibrator.api.v1 as v1_mod

    real_inspect = v1_mod.inspect_frame

    def slow_inspect(source, *args, **kwargs):
        time.sleep(0.3)
        return real_inspect(source, *args, **kwargs)

    v1_mod.inspect_frame = slow_inspect
    try:
        assert w.progress_bar.isHidden()
        w._on_preflight()
        assert w._controller.is_active
        assert not w.progress_bar.isHidden()
        assert _pump(lambda: not w._controller.is_active)
        assert w.progress_bar.isHidden()
    finally:
        v1_mod.inspect_frame = real_inspect
        _close(w)


def test_standard_explains_missing_source(qapp, paths, tmp_path):
    """F2: a missing/unreadable image is explained in plain language in Standard
    (never `Not inspected.` and never raw filesystem exception prose)."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        missing_path = str(tmp_path / "missing.fits")
        w._lights.append(_LightEntry(missing_path, hdu=0))
        w._refresh_lights_list()
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)

        row = None
        for i, s in enumerate(w._preflight_summaries):
            if s.get("path") == missing_path:
                row = i
                break
        assert row is not None, "missing-source summary not present"
        assert w.standard_results_table.item(row, 1).text() == "Needs attention"
        assert w.standard_results_table.item(row, 2).text() == "Could not read this image."
        # Raw filesystem exception prose must not leak into Standard.
        assert "Errno" not in w.standard_results_table.item(row, 2).text()
    finally:
        _close(w)


def test_theme_control_disabled_during_load_and_applies_persisted(qapp, tmp_path):
    """F3: Theme is disabled during the async settings load, enabled after, and the
    persisted preference is applied without silently discarding a user choice."""
    import zecalibrator.gui.settings as settings_mod
    from zecalibrator.gui.settings import GuiSettings, save_settings

    paths = resolve_paths(base=str(tmp_path))
    save_settings(paths.user_config_path, GuiSettings(appearance_theme="dark"))

    real_load = settings_mod.load_settings
    release = {"go": False}

    def slow_load(config_dir):
        while not release["go"]:
            time.sleep(0.01)
        return real_load(config_dir)

    settings_mod.load_settings = slow_load
    w = None
    try:
        w = MainWindow(paths)
        assert not w._settings_loaded
        assert not w.theme_combo.isEnabled()
        release["go"] = True
        assert _pump(lambda: w._settings_loaded)
        assert w.theme_combo.isEnabled()
        assert w.theme_combo.currentData() == "dark"
        assert w._settings.appearance_theme == "dark"
    finally:
        settings_mod.load_settings = real_load
        if w is not None:
            w._controller.shutdown()
            _pump(lambda: w._controller.is_finished)


def test_standard_summary_initialized_and_reset_no_images_verified(qapp, paths):
    w = MainWindow(paths)
    assert w.standard_summary_label.text() == "No images verified."
    w._bump_generation()
    assert w.standard_summary_label.text() == "No images verified."
    _close(w)


def test_scope_all_wording_explicit_images(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w._update_scope_label()
        assert w.scope_label.text() == "Export scope: all (1 image)"
        from .conftest import write_fits

        write_fits(tmp_path / "light2.fits", 120.0)
        w._lights.append(_LightEntry(str(tmp_path / "light2.fits"), hdu=0))
        w._refresh_lights_list()
        w._update_scope_label()
        assert w.scope_label.text() == "Export scope: all (2 images)"
    finally:
        _close(w)


def test_advanced_token_label_shows_exact_values(qapp, paths):
    w = MainWindow(paths)
    assert w.mode_token_label.text() == "Exact values: additive=dark_incl_bias  flat=none"
    w.additive_combo.setCurrentIndex(w.additive_combo.findData("bias_only"))
    assert w.mode_token_label.text() == "Exact values: additive=bias_only  flat=none"
    w.flat_combo.setCurrentIndex(w.flat_combo.findData("apply"))
    assert w.mode_token_label.text() == "Exact values: additive=bias_only  flat=apply"
    _close(w)


# ---------------------------------------------------------------------------
# REWORK-2: F2 (inspect-failure buckets) + F4 (no false sub-phase completion).
# ---------------------------------------------------------------------------
def test_standard_explains_hdu_failure(qapp, paths, tmp_path):
    """F2 (non-SOURCE bucket): a wrong HDU is explained in plain language via the
    public GUI seam, never leaking a raw reason-code string in Standard."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        bad = _LightEntry(
            fixture["light"], hdu=1,
            declaration=v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read())),
            roi_extent=v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read())),
        )
        w._lights.append(bad)
        w._refresh_lights_list()
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)

        row = None
        for i, s in enumerate(w._preflight_summaries):
            if s.get("row_id") == bad.row_id:
                row = i
                break
        assert row is not None, "HDU-failure summary not present"
        assert w.standard_results_table.item(row, 1).text() == "Needs attention"
        assert w.standard_results_table.item(row, 2).text() == \
            "The selected HDU has no usable 2-D image — pick another HDU in Advanced."
        assert "HDU_NOT_FOUND" not in w.standard_results_table.item(row, 2).text()
    finally:
        _close(w)


def test_nested_complete_progress_cannot_show_terminal_while_active(qapp, paths):
    """F4: a phase-local `complete (2/2)` event while active must not produce a
    full/terminal-looking bar or status."""
    w = MainWindow(paths)
    w._current_op_id = "op-1"
    w._active_generation = w._generation
    w._active_kind = "preflight"
    w._terminal_seen = False
    w._cancel_requested = False
    w.progress_bar.setRange(0, 0)
    w.status_label.setText("Checking calibration…")

    class Evt:
        phase = "complete"
        completed = 2
        total = 2
        unit = "steps"
        frame_id = None

    w._on_progress("op-1", Evt())
    assert w.progress_bar.minimum() == 0 and w.progress_bar.maximum() == 0
    assert w.status_label.text() == "Checking calibration…"
    assert "complete" not in w.status_label.text()
    _close(w)


def test_slow_preflight_status_is_kind_label_not_complete(qapp, paths, tmp_path):
    """F4: during a slow preflight the status stays on the kind label and the bar
    stays indeterminate (never `complete`/100%) while the controller is active."""
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)

    import zecalibrator.api.v1 as v1_mod

    real_inspect = v1_mod.inspect_frame

    def slow_inspect(source, *args, **kwargs):
        time.sleep(0.4)
        return real_inspect(source, *args, **kwargs)

    v1_mod.inspect_frame = slow_inspect
    try:
        w._on_preflight()
        assert _pump(lambda: w.status_label.text() == "Checking calibration…")
        assert w._controller.is_active
        assert w.status_label.text() == "Checking calibration…"
        assert "complete" not in w.status_label.text()
        assert w.progress_bar.minimum() == 0 and w.progress_bar.maximum() == 0
        assert w.cancel_btn.isEnabled()
        assert _pump(lambda: not w._controller.is_active)
        assert "Preflight COMPLETED" in w.status_label.text()
        assert w.progress_bar.isHidden()
    finally:
        v1_mod.inspect_frame = real_inspect
        _close(w)


def test_cancel_requested_status_stable_against_later_progress(qapp, paths):
    """F4: progress arriving after Cancel is requested must not overwrite
    `Cancellation requested…` before the terminal outcome."""
    w = MainWindow(paths)
    w._current_op_id = "op-1"
    w._active_generation = w._generation
    w._active_kind = "export"
    w._terminal_seen = False
    w._cancel_requested = False
    w._on_cancel()
    assert w.status_label.text() == "Cancellation requested…"

    class Evt:
        phase = "frame_complete"
        completed = 1
        total = 1
        unit = "frames"
        frame_id = "0"

    w._on_progress("op-1", Evt())
    assert w.status_label.text() == "Cancellation requested…"
    _close(w)


# ---------------------------------------------------------------------------
# P7-M3A LIGHTS: Add files / Add folder / Clear (Standard tab).
# ---------------------------------------------------------------------------
def _mock_get_open_file_names(paths, filt="FITS files (*.fits *.fit *.fts);;All files (*)"):
    from PySide6 import QtWidgets as qw

    qw.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (list(paths), filt))


def _mock_get_existing_directory(folder):
    from PySide6 import QtWidgets as qw

    qw.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(folder))


def test_add_files_keeps_multi_file_behavior(qapp, paths, tmp_path):
    w = MainWindow(paths)
    f1 = tmp_path / "one.fits"
    f2 = tmp_path / "two.fit"
    f1.write_bytes(b"")
    f2.write_bytes(b"")
    _mock_get_open_file_names([str(f1), str(f2)])
    w.hdu_edit.setText("1")
    w._on_add_lights()
    assert [e.path for e in w._lights] == [str(f1), str(f2)]
    assert all(e.hdu == 1 for e in w._lights)
    assert w.lights_count_label.text() == "2 images selected"
    _close(w)


def test_add_folder_adds_supported_top_level_and_ignores_unsupported(qapp, paths, tmp_path):
    w = MainWindow(paths)
    folder = tmp_path / "lights"
    folder.mkdir()
    (folder / "a.FITS").write_bytes(b"")
    (folder / "b.fit").write_bytes(b"")
    (folder / "notes.txt").write_text("x")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "nested.fits").write_bytes(b"")  # must NOT be added (non-recursive)
    _mock_get_existing_directory(folder)
    w.hdu_edit.setText("0")
    w._on_add_folder()
    assert [e.path for e in w._lights] == [str(folder / "a.FITS"), str(folder / "b.fit")]
    assert w.lights_count_label.text() == "2 images selected"
    assert w.status_label.text() == "2 images added (1 unsupported file ignored)"
    _close(w)


def test_add_folder_is_additive_and_dedups_existing(qapp, paths, tmp_path):
    w = MainWindow(paths)
    folder = tmp_path / "lights"
    folder.mkdir()
    existing = folder / "one.fits"
    existing.write_bytes(b"")
    newfit = folder / "two.fts"
    newfit.write_bytes(b"")
    # First add one file through the multi-file dialog.
    _mock_get_open_file_names([str(existing)])
    w._on_add_lights()
    assert len(w._lights) == 1
    # Then add the whole folder: the existing path must not be duplicated.
    _mock_get_existing_directory(folder)
    w._on_add_folder()
    assert [e.path for e in w._lights] == [str(existing), str(newfit)]
    assert w.lights_count_label.text() == "2 images selected"
    assert w.status_label.text() == "1 image added"
    _close(w)


def test_add_folder_empty_folder_is_clean(qapp, paths, tmp_path):
    w = MainWindow(paths)
    folder = tmp_path / "empty"
    folder.mkdir()
    _mock_get_existing_directory(folder)
    w._on_add_folder()
    assert w._lights == []
    assert w.lights_count_label.text() == "0 images selected"
    assert w.status_label.text() == "0 images added"
    _close(w)


def test_remove_selected_and_clear_keep_truthful_count(qapp, paths, tmp_path):
    w = MainWindow(paths)
    folder = tmp_path / "lights"
    folder.mkdir()
    (folder / "a.fits").write_bytes(b"")
    (folder / "b.fits").write_bytes(b"")
    (folder / "c.fits").write_bytes(b"")
    _mock_get_existing_directory(folder)
    w._on_add_folder()
    assert len(w._lights) == 3
    assert w.lights_count_label.text() == "3 images selected"

    w.lights_list.item(0).setSelected(True)
    w.lights_list.item(2).setSelected(True)
    w._on_remove_lights()
    assert [e.path for e in w._lights] == [str(folder / "b.fits")]
    assert w.lights_count_label.text() == "1 image selected"

    w._on_clear_lights()
    assert w._lights == []
    assert w.lights_count_label.text() == "0 images selected"
    _close(w)


def test_folder_add_bumps_generation_once_and_clears_cached_plans(qapp, paths, tmp_path):
    """Folder add is an input change: it must invalidate cached plans exactly like
    the multi-file add (no shared-state divergence)."""
    w = MainWindow(paths)
    folder = tmp_path / "lights"
    folder.mkdir()
    (folder / "a.fits").write_bytes(b"")
    w._plans["stale"] = object()
    before = w._generation
    _mock_get_existing_directory(folder)
    w._on_add_folder()
    assert w._generation == before + 1
    assert w._plans == {}
    _close(w)
