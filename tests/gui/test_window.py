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
    assert w.add_btn.text() == "Add images…"
    assert w.remove_btn.text() == "Remove selected"
    assert w.choose_library_btn.text() == "Choose library…"
    assert w.preflight_btn.text() == "Verify calibration"
    assert w.export_btn.text() == "Calibrate / Export…"
    assert w.lights_count_label.text() == "0 images selected"
    # Lifecycle footer is globally reachable (Cancel/progress/status exist in the
    # footer below the top-level tabs, outside any single tab page).
    assert w.cancel_btn is not None
    assert w.progress_bar is not None
    assert w.status_label is not None
    _close(w)


def test_standard_mode_labels_are_presentation_only(qapp, paths):
    from zecalibrator.gui import presentation

    w = MainWindow(paths)
    labels = [w.standard_additive_combo.itemText(i) for i in range(w.standard_additive_combo.count())]
    values = [w.standard_additive_combo.itemData(i) for i in range(w.standard_additive_combo.count())]
    assert labels == ["None", "Bias only", "Standard dark", "Dark already bias-corrected"]
    assert values == list(presentation.additive_modes())
    flat_labels = [w.standard_flat_combo.itemText(i) for i in range(w.standard_flat_combo.count())]
    flat_values = [w.standard_flat_combo.itemData(i) for i in range(w.standard_flat_combo.count())]
    assert flat_labels == ["None", "Use flat"]
    assert flat_values == list(presentation.flat_modes())
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


def test_standard_mode_change_maps_to_advanced_and_request(qapp, paths):
    w = MainWindow(paths)
    w.standard_additive_combo.setCurrentIndex(w.standard_additive_combo.findData("bias_only"))
    assert w.additive_combo.currentData() == "bias_only"
    assert w._request().additive_mode == "bias_only"
    w.standard_flat_combo.setCurrentIndex(w.standard_flat_combo.findData("apply"))
    assert w.flat_combo.currentData() == "apply"
    assert w._request().flat_mode == "apply"
    _close(w)


def test_advanced_mode_change_maps_to_standard_and_request(qapp, paths):
    w = MainWindow(paths)
    w.additive_combo.setCurrentIndex(w.additive_combo.findData("dark_bias_removed"))
    assert w.standard_additive_combo.currentData() == "dark_bias_removed"
    assert w._request().additive_mode == "dark_bias_removed"
    w.flat_combo.setCurrentIndex(w.flat_combo.findData("none"))
    assert w.standard_flat_combo.currentData() == "none"
    assert w._request().flat_mode == "none"
    _close(w)


def test_single_generation_bump_per_mode_change(qapp, paths):
    w = MainWindow(paths)
    before = w._generation
    w.standard_additive_combo.setCurrentIndex(w.standard_additive_combo.findData("bias_only"))
    assert w._generation == before + 1
    # Sync reflection must not cause a second invalidation.
    assert w.additive_combo.currentData() == "bias_only"

    before = w._generation
    w.additive_combo.setCurrentIndex(w.additive_combo.findData("dark_incl_bias"))
    assert w._generation == before + 1
    assert w.standard_additive_combo.currentData() == "dark_incl_bias"
    _close(w)


def test_standard_summary_counts_truthful_human_outcomes(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _make_window(qapp, paths, fixture)
    try:
        w.flat_combo.setCurrentIndex(w.flat_combo.findData("apply"))  # NO_MATCH
        w._on_preflight()
        assert _pump(lambda: not w._controller.is_active)
        assert w._preflight_summaries[0]["outcome"] == "NO_MATCH"
        assert "needs attention" in w.standard_summary_label.text()
        # Human outcome table shows the human label, not the technical token.
        assert w.standard_results_table.item(0, 1).text() == "Needs attention"
        assert "No compatible" in w.standard_results_table.item(0, 2).text()
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
