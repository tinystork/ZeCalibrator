"""PySide6 main window for the ZeCalibrator minimum GUI client.

The window assembles *presentation requests only*: it builds immutable
:class:`~zecalibrator.gui.service.OperationSnapshot` values and hands them to a
:class:`~zecalibrator.gui.worker.WorkerController`. No calibration equations,
master selection, FITS parsing or provenance assembly happen in any widget or
slot; all such work (plus settings read/fsync/write and audit serialization)
runs on the worker thread through ``zecalibrator.api.v1``.

Configuration binding (F1/F3): a generation counter is bumped on every
mode/library/input/evidence change; cached plans/results/audit are cleared and
every cached plan is bound to (generation, stable row identity). All conflicting
input/config controls are frozen while an operation is active, and stale
row/config-generation events are rejected in addition to stale operation ids.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui import identity, presentation, service, theme
from zecalibrator.gui.settings import (
    STATE_MALFORMED,
    STATE_UNSUPPORTED,
    GuiSettings,
    default_settings,
)
from zecalibrator.gui.worker import WorkerController
from zecalibrator.storage import StoragePaths

_SYNTH_ONLY_LABEL = "Qualification: SYNTH-BASE-1 synthetic-only. No real-camera claim."

# Truthful human active-status labels per operation kind (progress counters are
# phase-local and never a global percentage; these labels stay until terminal).
_ACTIVE_LABELS = {
    "load_settings": "Loading settings…",
    "save_settings": "Saving settings…",
    "open_library": "Opening calibration library…",
    "index_library": "Updating calibration library…",
    "preflight": "Checking calibration…",
    "calibrate_in_memory": "Calibrating selected image…",
    "export": "Calibrating / exporting…",
    "load_declaration": "Loading input details…",
    "load_roi": "Loading input details…",
    "scan_masters": "Detecting calibration masters…",
    "load_ledger": "Loading managed masters…",
    "confirm_evidence": "Confirming master evidence…",
    "build_managed_library": "Building managed calibration library…",
}


class _LightEntry:
    """Mutable GUI-side light row (worker sees only the frozen snapshot).

    ``row_id`` is a stable, per-row unique identity so a cached plan can never be
    bound to another row sharing the same path/HDU but different evidence.
    """

    def __init__(self, path: str, hdu=0, declaration=None, roi_extent=None, row_id=None):
        self.path = path
        self.hdu = hdu
        self.declaration = declaration
        self.roi_extent = roi_extent
        self.row_id = row_id or uuid.uuid4().hex

    def display(self) -> str:
        return f"{Path(self.path).name}  [HDU={self.hdu}]"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, storage_paths: StoragePaths, parent=None):
        super().__init__(parent)
        self._storage = storage_paths
        # Settings start as defaults; the actual read happens off-thread at
        # startup (see _start_settings_load) and is applied on completion.
        self._settings = default_settings()
        self._settings_state = None
        self._settings_loaded = False
        self._settings_saved = False

        self._lights: list[_LightEntry] = []
        self._library_spec: v1.LibrarySpec | None = None
        self._plans: dict[str, object] = {}  # row_id -> plan (current generation)
        self._plan_generation = 0
        self._generation = 0
        self._preflight_summaries: list[dict] = []
        self._batch_items: list[dict] = []
        self._batch_displays: list[str] = []
        self._batch_total_inputs = 0
        self._in_memory_result: dict | None = None

        self._controller = WorkerController(self)
        self._token: v1.CancellationToken | None = None
        self._current_op_id: str | None = None
        self._active_generation = 0
        self._close_requested = False
        self._terminal_seen = False
        self._syncing_modes = False
        self._syncing_theme = False
        self._active_kind = None
        self._cancel_requested = False

        # Managed master ingestion state (P7-M3B).
        self._managed_scan: list[dict] = []
        self._managed_pending: list[dict] = []
        self._active_source: str | None = None  # "managed" | "explicit" | None
        self._scan_targets: list = []
        self._scan_index = 0
        self._scan_results: list[dict] = []
        self._confirm_index = 0

        self._build_ui()
        self._wire_controller()
        self._apply_icon()
        self.resize(self._settings.window_width, self._settings.window_height)
        self._set_active(False)

        # Capture the native Qt palette once before any theme override (System).
        app = QtWidgets.QApplication.instance()
        if app is not None:
            theme.get_system_palette(app)

        # Production exit safety: request non-blocking worker quit at app exit.
        if app is not None:
            app.aboutToQuit.connect(self._controller.shutdown)

        # Asynchronous settings load (off the GUI thread).
        self._start_settings_load()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.setWindowTitle("ZeCalibrator")

        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)

        # Top-level progressive-disclosure navigation (exact owner order).
        self.main_tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.main_tabs, 1)

        self._build_standard_tab()
        self._build_advanced_tab()
        self._build_settings_tab()
        self._build_footer(root_layout)

        self.setCentralWidget(central)
        self.main_tabs.setCurrentIndex(0)  # Standard is the default view

        self._connect_signals()

        self._config_widgets = [
            self.add_btn, self.add_folder_btn, self.remove_btn, self.clear_btn,
            self.apply_hdu_btn, self.hdu_edit,
            self.load_decl_btn, self.load_roi_btn,
            self.library_index_edit, self.open_library_btn,
            self.library_root_edit, self.imports_edit, self.imports_browse_btn,
            self.root_browse_btn, self.index_btn,
            self.additive_combo, self.flat_combo,
            self.standard_additive_combo, self.standard_flat_combo,
            self.darks_folder_edit, self.darks_browse_btn,
            self.bias_folder_edit, self.bias_browse_btn,
            self.flats_folder_edit, self.flats_browse_btn,
            self.scan_masters_btn, self.confirm_masters_btn, self.build_managed_btn,
        ]
        self._launch_widgets = [self.preflight_btn, self.calibrate_btn, self.export_btn]
        self._update_scope_label()

    def _build_standard_tab(self) -> None:
        """Standard (nominal) workflow: human-first, progressive disclosure."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        inputs_box = QtWidgets.QGroupBox("Images to calibrate")
        inputs_layout = QtWidgets.QVBoxLayout(inputs_box)
        self.lights_list = QtWidgets.QListWidget()
        self.lights_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        inputs_layout.addWidget(self.lights_list)
        light_btn_row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add files…")
        self.add_folder_btn = QtWidgets.QPushButton("Add folder…")
        self.remove_btn = QtWidgets.QPushButton("Remove selected")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        for b in (self.add_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
            light_btn_row.addWidget(b)
        light_btn_row.addStretch(1)
        inputs_layout.addLayout(light_btn_row)
        self.lights_count_label = QtWidgets.QLabel("0 images selected")
        inputs_layout.addWidget(self.lights_count_label)
        layout.addWidget(inputs_box)

        self.library_human_status = QtWidgets.QLabel("Library unavailable")
        layout.addWidget(self.library_human_status)

        masters_box = QtWidgets.QGroupBox("Calibration masters (managed)")
        masters_layout = QtWidgets.QVBoxLayout(masters_box)
        dark_row = QtWidgets.QHBoxLayout()
        dark_row.addWidget(QtWidgets.QLabel("Darks folder:"))
        self.darks_folder_edit = QtWidgets.QLineEdit()
        dark_row.addWidget(self.darks_folder_edit, 1)
        self.darks_browse_btn = QtWidgets.QPushButton("Browse…")
        dark_row.addWidget(self.darks_browse_btn)
        masters_layout.addLayout(dark_row)
        bias_row = QtWidgets.QHBoxLayout()
        bias_row.addWidget(QtWidgets.QLabel("Bias folder:"))
        self.bias_folder_edit = QtWidgets.QLineEdit()
        bias_row.addWidget(self.bias_folder_edit, 1)
        self.bias_browse_btn = QtWidgets.QPushButton("Browse…")
        bias_row.addWidget(self.bias_browse_btn)
        masters_layout.addLayout(bias_row)
        flats_row = QtWidgets.QHBoxLayout()
        flats_row.addWidget(QtWidgets.QLabel("Flats folder/file:"))
        self.flats_folder_edit = QtWidgets.QLineEdit()
        flats_row.addWidget(self.flats_folder_edit, 1)
        self.flats_browse_btn = QtWidgets.QPushButton("Browse…")
        flats_row.addWidget(self.flats_browse_btn)
        masters_layout.addLayout(flats_row)
        masters_btn_row = QtWidgets.QHBoxLayout()
        self.scan_masters_btn = QtWidgets.QPushButton("Detect masters…")
        self.confirm_masters_btn = QtWidgets.QPushButton("Confirm detected facts")
        self.build_managed_btn = QtWidgets.QPushButton("Build managed library")
        for b in (self.scan_masters_btn, self.confirm_masters_btn, self.build_managed_btn):
            masters_btn_row.addWidget(b)
        masters_btn_row.addStretch(1)
        masters_layout.addLayout(masters_btn_row)
        self.managed_status_label = QtWidgets.QLabel("No managed masters detected.")
        masters_layout.addWidget(self.managed_status_label)
        layout.addWidget(masters_box)

        self.active_source_label = QtWidgets.QLabel("Active calibration source: none")
        layout.addWidget(self.active_source_label)

        modes_box = QtWidgets.QGroupBox("Calibration")
        modes_layout = QtWidgets.QGridLayout(modes_box)
        modes_layout.addWidget(QtWidgets.QLabel("Dark:"), 0, 0)
        self.standard_additive_combo = QtWidgets.QComboBox()
        for mode in presentation.additive_modes():
            self.standard_additive_combo.addItem(
                presentation.standard_additive_mode_label(mode), mode
            )
        self.standard_additive_combo.setCurrentIndex(2)  # dark_incl_bias
        modes_layout.addWidget(self.standard_additive_combo, 0, 1)
        modes_layout.addWidget(QtWidgets.QLabel("Flat:"), 1, 0)
        self.standard_flat_combo = QtWidgets.QComboBox()
        for mode in presentation.flat_modes():
            self.standard_flat_combo.addItem(presentation.standard_flat_mode_label(mode), mode)
        modes_layout.addWidget(self.standard_flat_combo, 1, 1)
        layout.addWidget(modes_box)

        actions_row = QtWidgets.QHBoxLayout()
        self.preflight_btn = QtWidgets.QPushButton("Verify calibration")
        self.export_btn = QtWidgets.QPushButton("Calibrate / Export…")
        actions_row.addWidget(self.preflight_btn)
        actions_row.addWidget(self.export_btn)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.standard_summary_label = QtWidgets.QLabel(presentation.format_outcome_summary(0, 0, 0))
        layout.addWidget(self.standard_summary_label)
        self.standard_results_table = QtWidgets.QTableWidget(0, 3)
        self.standard_results_table.setHorizontalHeaderLabels(["Image", "Outcome", "Summary"])
        self.standard_results_table.horizontalHeader().setStretchLastSection(True)
        self.standard_results_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.standard_results_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        layout.addWidget(self.standard_results_table)

        self.main_tabs.addTab(page, "Standard")

    def _build_advanced_tab(self) -> None:
        """Advanced: full technical controls/outcomes, simple grouped layout."""
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        input_box = QtWidgets.QGroupBox("Input details")
        input_layout = QtWidgets.QVBoxLayout(input_box)
        hdu_row = QtWidgets.QHBoxLayout()
        hdu_row.addWidget(QtWidgets.QLabel("HDU for selected:"))
        self.hdu_edit = QtWidgets.QLineEdit("0")
        self.hdu_edit.setFixedWidth(80)
        hdu_row.addWidget(self.hdu_edit)
        self.apply_hdu_btn = QtWidgets.QPushButton("Apply HDU")
        hdu_row.addWidget(self.apply_hdu_btn)
        hdu_row.addStretch(1)
        input_layout.addLayout(hdu_row)
        ev_row = QtWidgets.QHBoxLayout()
        self.load_decl_btn = QtWidgets.QPushButton("Load declaration JSON…")
        self.load_roi_btn = QtWidgets.QPushButton("Load ROI JSON…")
        ev_row.addWidget(self.load_decl_btn)
        ev_row.addWidget(self.load_roi_btn)
        ev_row.addStretch(1)
        input_layout.addLayout(ev_row)
        self.evidence_label = QtWidgets.QLabel("Evidence applies to the currently selected light(s).")
        input_layout.addWidget(self.evidence_label)
        layout.addWidget(input_box)

        library_box = QtWidgets.QGroupBox("Library management")
        library_layout = QtWidgets.QVBoxLayout(library_box)
        open_row = QtWidgets.QHBoxLayout()
        open_row.addWidget(QtWidgets.QLabel("Index path:"))
        self.library_index_edit = QtWidgets.QLineEdit()
        self.library_index_edit.setPlaceholderText(
            f"{self._storage.user_data_path / 'zecalibrator.library.sqlite'}"
        )
        open_row.addWidget(self.library_index_edit)
        self.open_library_btn = QtWidgets.QPushButton("Open…")
        open_row.addWidget(self.open_library_btn)
        library_layout.addLayout(open_row)

        index_row = QtWidgets.QHBoxLayout()
        index_row.addWidget(QtWidgets.QLabel("Root:"))
        self.library_root_edit = QtWidgets.QLineEdit()
        index_row.addWidget(self.library_root_edit)
        self.root_browse_btn = QtWidgets.QPushButton("Browse…")
        index_row.addWidget(self.root_browse_btn)
        index_row.addWidget(QtWidgets.QLabel("Imports JSON:"))
        self.imports_edit = QtWidgets.QLineEdit()
        index_row.addWidget(self.imports_edit)
        self.imports_browse_btn = QtWidgets.QPushButton("Browse…")
        index_row.addWidget(self.imports_browse_btn)
        self.index_btn = QtWidgets.QPushButton("Index library…")
        index_row.addWidget(self.index_btn)
        library_layout.addLayout(index_row)
        self.library_status = QtWidgets.QLabel("No library opened.")
        library_layout.addWidget(self.library_status)
        layout.addWidget(library_box)

        modes_box = QtWidgets.QGroupBox("Calibration details")
        modes_layout = QtWidgets.QGridLayout(modes_box)
        modes_layout.addWidget(QtWidgets.QLabel("Additive:"), 0, 0)
        self.additive_combo = QtWidgets.QComboBox()
        for mode in presentation.additive_modes():
            self.additive_combo.addItem(presentation.additive_mode_label(mode), mode)
        self.additive_combo.setCurrentIndex(2)  # dark_incl_bias
        modes_layout.addWidget(self.additive_combo, 0, 1)
        modes_layout.addWidget(QtWidgets.QLabel("Flat:"), 1, 0)
        self.flat_combo = QtWidgets.QComboBox()
        for mode in presentation.flat_modes():
            self.flat_combo.addItem(presentation.flat_mode_label(mode), mode)
        modes_layout.addWidget(self.flat_combo, 1, 1)
        self.mode_note = QtWidgets.QLabel("")
        modes_layout.addWidget(self.mode_note, 2, 0, 1, 2)
        self.mode_token_label = QtWidgets.QLabel("")
        self.mode_token_label.setStyleSheet("color: gray;")
        modes_layout.addWidget(self.mode_token_label, 4, 0, 1, 2)
        in_memory_row = QtWidgets.QHBoxLayout()
        self.calibrate_btn = QtWidgets.QPushButton("Calibrate selected in memory")
        in_memory_row.addWidget(self.calibrate_btn)
        in_memory_row.addStretch(1)
        modes_layout.addLayout(in_memory_row, 3, 0, 1, 2)
        layout.addWidget(modes_box)
        self._update_mode_note()

        self.qualification_label = QtWidgets.QLabel(_SYNTH_ONLY_LABEL)
        self.qualification_label.setStyleSheet("color: gray;")
        layout.addWidget(self.qualification_label)

        # Existing result/details sub-tab widget (kept as the `tabs` attribute).
        self.tabs = QtWidgets.QTabWidget()

        self.preflight_table = QtWidgets.QTableWidget(0, 4)
        self.preflight_table.setHorizontalHeaderLabels(["File", "Outcome", "Plan", "Reason codes"])
        self.preflight_table.horizontalHeader().setStretchLastSection(True)
        self.preflight_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.preflight_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tabs.addTab(self.preflight_table, "Preflight")

        self.results_table = QtWidgets.QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["#", "File", "Disposition", "Plan", "Reason"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tabs.addTab(self.results_table, "Results / audit")

        self.details_view = QtWidgets.QPlainTextEdit()
        self.details_view.setReadOnly(True)
        self.tabs.addTab(self.details_view, "Details")

        audit_widget = QtWidgets.QWidget()
        audit_layout = QtWidgets.QVBoxLayout(audit_widget)
        self.audit_view = QtWidgets.QPlainTextEdit()
        self.audit_view.setReadOnly(True)
        font = QtGui.QFont("monospace")
        font.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
        self.audit_view.setFont(font)
        audit_layout.addWidget(self.audit_view)
        self.audit_link_btn = QtWidgets.QPushButton("Open manifest…")
        self.audit_link_btn.setVisible(False)
        audit_layout.addWidget(self.audit_link_btn)
        self.tabs.addTab(audit_widget, "Audit")

        layout.addWidget(self.tabs)

        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.main_tabs.addTab(page, "Advanced")

    def _build_settings_tab(self) -> None:
        """Settings: application preferences only (Appearance / Theme)."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        appearance_box = QtWidgets.QGroupBox("Appearance")
        appearance_layout = QtWidgets.QVBoxLayout(appearance_box)
        theme_row = QtWidgets.QHBoxLayout()
        theme_row.addWidget(QtWidgets.QLabel("Theme:"))
        self.theme_combo = QtWidgets.QComboBox()
        for name in theme.THEMES:
            self.theme_combo.addItem(theme.theme_label(name), name)
        self.theme_combo.setEnabled(False)  # disabled until async settings load applies
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch(1)
        appearance_layout.addLayout(theme_row)
        layout.addWidget(appearance_box)
        layout.addStretch(1)

        self.main_tabs.addTab(page, "Settings")

    def _build_footer(self, root_layout) -> None:
        """Global lifecycle footer below the top-level tabs (Cancel always reachable)."""
        self.scope_label = QtWidgets.QLabel("Export scope: all")
        root_layout.addWidget(self.scope_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        root_layout.addWidget(self.progress_bar)

        status_row = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.status_label = QtWidgets.QLabel("Ready.")
        status_row.addWidget(self.cancel_btn)
        status_row.addWidget(self.status_label, 1)
        root_layout.addLayout(status_row)

    def _connect_signals(self) -> None:
        self.add_btn.clicked.connect(self._on_add_lights)
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        self.remove_btn.clicked.connect(self._on_remove_lights)
        self.clear_btn.clicked.connect(self._on_clear_lights)
        self.apply_hdu_btn.clicked.connect(self._on_apply_hdu)
        self.load_decl_btn.clicked.connect(self._on_load_declaration)
        self.load_roi_btn.clicked.connect(self._on_load_roi)
        self.open_library_btn.clicked.connect(self._on_open_library)
        self.root_browse_btn.clicked.connect(self._on_browse_root)
        self.imports_browse_btn.clicked.connect(self._on_browse_imports)
        self.index_btn.clicked.connect(self._on_index_library)
        self.preflight_btn.clicked.connect(self._on_preflight)
        self.calibrate_btn.clicked.connect(self._on_calibrate_in_memory)
        self.export_btn.clicked.connect(self._on_export)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.audit_link_btn.clicked.connect(self._on_open_manifest)
        self.darks_browse_btn.clicked.connect(self._on_browse_darks)
        self.bias_browse_btn.clicked.connect(self._on_browse_bias)
        self.flats_browse_btn.clicked.connect(self._on_browse_flats)
        self.scan_masters_btn.clicked.connect(self._on_scan_masters)
        self.confirm_masters_btn.clicked.connect(self._on_confirm_masters)
        self.build_managed_btn.clicked.connect(self._on_build_managed)

        self.additive_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.flat_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.standard_additive_combo.currentIndexChanged.connect(self._on_standard_mode_changed)
        self.standard_flat_combo.currentIndexChanged.connect(self._on_standard_mode_changed)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.lights_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.preflight_table.currentCellChanged.connect(self._on_preflight_selection_changed)
        self.results_table.currentCellChanged.connect(self._on_results_selection_changed)

    def _wire_controller(self) -> None:
        self._controller.operation_started.connect(self._on_operation_started)
        self._controller.progress.connect(self._on_progress)
        self._controller.preflight_light.connect(self._on_preflight_light)
        self._controller.batch_item.connect(self._on_batch_item)
        self._controller.operation_finished.connect(self._on_operation_finished)
        self._controller.operation_failed.connect(self._on_operation_failed)
        self._controller.worker_ended.connect(self._on_worker_ended)

    def _apply_icon(self) -> None:
        icon = identity.load_app_icon()
        if icon is not None and not icon.isNull():
            self.setWindowIcon(icon)

    # -------------------------------------------------------------- helpers
    def _bump_generation(self) -> None:
        """Invalidate cached plans/results/audit on any config/input change."""
        self._generation += 1
        self._plans.clear()
        self._preflight_summaries.clear()
        self.preflight_table.setRowCount(0)
        self._in_memory_result = None
        self._clear_batch_presentation()
        self._reset_standard_summary()

    def _clear_batch_presentation(self) -> None:
        self._batch_items.clear()
        self._batch_displays = []
        self._batch_total_inputs = 0
        self.results_table.setRowCount(0)
        self._audit_clear()

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.lights_list.selectedIndexes()})

    def _current_row(self) -> int | None:
        row = self.lights_list.currentRow()
        return row if row >= 0 else None

    def _refresh_lights_list(self) -> None:
        self.lights_list.clear()
        for entry in self._lights:
            item = QtWidgets.QListWidgetItem(entry.display())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry.path)
            self.lights_list.addItem(item)
        self._update_lights_count()

    def _update_lights_count(self) -> None:
        n = len(self._lights)
        self.lights_count_label.setText(
            f"{n} {'image' if n == 1 else 'images'} selected"
        )

    def _request(self) -> v1.CalibrationRequest:
        return v1.CalibrationRequest(self.additive_combo.currentData(), self.flat_combo.currentData())

    def _policy(self) -> v1.MatchPolicy:
        return v1.default_match_policy()

    def _update_mode_note(self) -> None:
        additive = self.additive_combo.currentData()
        flat = self.flat_combo.currentData()
        self.mode_token_label.setText(f"Exact values: additive={additive}  flat={flat}")
        if presentation.is_partial_mode(additive, flat):
            self.mode_note.setText(
                "Note: this selection is an explicit partial/uncalibrated mode, "
                "never advertised as full calibration."
            )
        else:
            self.mode_note.setText("")

    def _update_scope_label(self) -> None:
        selected = self._selected_rows()
        total = len(self._lights)
        if selected:
            self.scope_label.setText(f"Export scope: selected ({len(selected)} of {total})")
        else:
            noun = "image" if total == 1 else "images"
            self.scope_label.setText(f"Export scope: all ({total} {noun})")

    def _set_active(self, active: bool) -> None:
        self._active = active
        for w in self._config_widgets:
            w.setEnabled(not active)
        for w in self._launch_widgets:
            w.setEnabled(not active)
        self.cancel_btn.setEnabled(active)
        if active:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        else:
            # No operation active: hide the busy indicator (never a fake 0%).
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.hide()

    def _start_operation(self, snapshot: service.OperationSnapshot) -> None:
        if self._controller.is_active:
            self._log("an operation is already running")
            return
        self._token = v1.CancellationToken()
        self._current_op_id = snapshot.op_id
        self._active_generation = self._generation
        self._active_kind = snapshot.kind
        self._terminal_seen = False
        self._cancel_requested = False
        self._set_active(True)
        self._controller.start(snapshot, self._token)

    def _is_current(self, op_id: str) -> bool:
        return op_id == self._current_op_id and self._generation == self._active_generation

    def _active_label(self) -> str:
        return _ACTIVE_LABELS.get(self._active_kind, "Working…")

    def _log(self, text: str) -> None:
        # Literal plain-text logging: never interpret technical strings as HTML.
        self.details_view.appendPlainText(text)

    def _ensure_library(self) -> bool:
        if self._library_spec is None:
            QtWidgets.QMessageBox.warning(
                self, "No library",
                "Open or index a library before preflight/calibration/export.",
            )
            return False
        return True

    def _selected_light(self) -> _LightEntry | None:
        row = self._current_row()
        if row is None or row >= len(self._lights):
            return None
        return self._lights[row]

    def _lights_snapshot(self, entries: list[_LightEntry]) -> tuple[service.LightInput, ...]:
        return tuple(
            service.LightInput(
                path=e.path, hdu=e.hdu, declaration=e.declaration,
                roi_extent=e.roi_extent, display_name=Path(e.path).name,
                row_id=e.row_id,
            )
            for e in entries
        )

    def _default_index_path(self) -> str:
        return str(self._storage.user_data_path / "zecalibrator.library.sqlite")

    # -- audit pane --------------------------------------------------------
    def _audit_clear(self) -> None:
        self.audit_view.setPlainText("")
        self.audit_link_btn.setVisible(False)
        self._manifest_path = None

    def _audit_show(self, obj) -> None:
        if obj is None:
            self._audit_clear()
            return
        self.audit_view.setPlainText(json.dumps(obj, indent=2, ensure_ascii=False, default=str))

    def _show_manifest_link(self, path: str | None) -> None:
        self._manifest_path = path
        self.audit_link_btn.setVisible(bool(path))

    def _on_open_manifest(self) -> None:
        if getattr(self, "_manifest_path", None):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._manifest_path))

    # -- settings (async, off GUI thread) ----------------------------------
    def _start_settings_load(self) -> None:
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="load_settings",
            library_spec=None, request=None, policy=None, lights=(),
            config_dir=str(self._storage.user_config_path),
        )
        self._start_operation(snapshot)

    def _handle_settings_loaded(self, summary: dict) -> None:
        self._settings_state = summary.get("state")
        self._settings = GuiSettings.from_dict(summary.get("settings", {}))
        self._settings_loaded = True
        self.resize(self._settings.window_width, self._settings.window_height)
        self._apply_theme(self._settings.appearance_theme)
        self._sync_theme_combo()
        self.theme_combo.setEnabled(True)
        # Truthful idle state: startup settings load is complete; no fake progress.
        self.status_label.setText("Ready.")
        if self._settings_state in (STATE_MALFORMED, STATE_UNSUPPORTED):
            self._log(
                f"[settings] existing settings file is {self._settings_state}; "
                "it will be preserved (not overwritten) on close."
            )

    def _request_settings_save(self) -> None:
        if self._settings_state in (STATE_MALFORMED, STATE_UNSUPPORTED):
            # Preserve corrupt/unsupported bytes: nothing to write.
            self._settings_saved = True
            return
        self._close_requested = True
        self._settings = dataclasses.replace(
            self._settings, window_width=self.width(), window_height=self.height(),
        )
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="save_settings",
            library_spec=None, request=None, policy=None, lights=(),
            config_dir=str(self._storage.user_config_path),
            settings_payload=self._settings.to_dict(),
        )
        self._start_operation(snapshot)

    def _handle_settings_saved(self, summary: dict) -> None:
        self._settings_saved = True
        if summary.get("status") == "PRESERVED":
            self._log("[settings] existing settings preserved (not overwritten).")

    # -- theme / appearance ------------------------------------------------
    def _apply_theme(self, theme_name) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        theme.apply_theme(app, theme_name, theme.get_system_palette(app))

    def _sync_theme_combo(self) -> None:
        self._syncing_theme = True
        try:
            idx = self.theme_combo.findData(self._settings.appearance_theme)
            self.theme_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._syncing_theme = False

    def _on_theme_changed(self, *_args) -> None:
        if self._syncing_theme:
            return
        theme_name = self.theme_combo.currentData()
        self._apply_theme(theme_name)
        # Theme is a presentation preference only; never bumps scientific generation.
        self._settings = dataclasses.replace(self._settings, appearance_theme=theme_name)

    # -- library human status ----------------------------------------------
    _LIBRARY_HUMAN = {
        "ready": "Library ready",
        "unavailable": "Library unavailable",
        "attention": "Library needs attention",
    }

    def _set_library_human_status(self, state: str, detail: str = "") -> None:
        base = self._LIBRARY_HUMAN.get(state, "Library unavailable")
        self.library_human_status.setText(f"{base}{' — ' + detail if detail else ''}")

    # -- exactly-one-active calibration source (managed vs explicit) ----------
    def _set_active_source(self, source: str | None) -> None:
        """Set exactly one active calibration source and clear the other.

        Selecting the managed source clears the explicit library spec (and vice
        versa) so the two never mix silently; the active source is visible in the
        Standard tab.
        """
        if source == "managed":
            self._active_source = "managed"
            self._library_spec = None
            self.library_index_edit.clear()
            self.library_root_edit.clear()
            self.active_source_label.setText("Active calibration source: managed masters")
        elif source == "explicit":
            self._active_source = "explicit"
            self._managed_scan.clear()
            self._managed_pending.clear()
            self.managed_status_label.setText("No managed masters detected.")
            self.active_source_label.setText("Active calibration source: explicit library")
        else:
            self._active_source = None
            self.active_source_label.setText("Active calibration source: none")

    # -- Standard summary (human outcomes) ----------------------------------
    def _reset_standard_summary(self) -> None:
        if hasattr(self, "standard_summary_label"):
            self.standard_summary_label.setText(presentation.format_outcome_summary(0, 0, 0))
        if hasattr(self, "standard_results_table"):
            self.standard_results_table.setRowCount(0)

    def _update_standard_summary(self) -> None:
        ready, attention, ambiguous = presentation.summarize_outcomes(self._preflight_summaries)
        self.standard_summary_label.setText(
            presentation.format_outcome_summary(ready, attention, ambiguous)
        )
        self.standard_results_table.setRowCount(0)
        for summary in self._preflight_summaries:
            row = self.standard_results_table.rowCount()
            self.standard_results_table.insertRow(row)
            self.standard_results_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(summary.get("display", ""))
            )
            self.standard_results_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(
                    presentation.human_outcome_label(summary.get("outcome"))
                )
            )
            self.standard_results_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(presentation.human_reason_text(summary))
            )

    # -------------------------------------------------------------- slots
    def _on_add_lights(self) -> None:
        start_dir = self._settings.last_input_dir or ""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select FITS light frames", start_dir,
            service.input_dialog_filter(),
        )
        if not paths:
            return
        hdu = service.parse_hdu(self.hdu_edit.text())
        for path in paths:
            self._lights.append(_LightEntry(path, hdu=hdu))
        self._settings = dataclasses.replace(
            self._settings,
            last_input_dir=str(Path(paths[0]).parent),
            window_width=self.width(), window_height=self.height(),
        )
        self._refresh_lights_list()
        self._bump_generation()
        self._update_scope_label()

    def _on_add_folder(self) -> None:
        start_dir = self._settings.last_input_dir or ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a folder of light frames", start_dir
        )
        if not folder:
            return
        try:
            paths, unsupported = service.scan_folder_inputs(folder)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid folder", str(exc))
            return
        hdu = service.parse_hdu(self.hdu_edit.text())
        # Additive append with deterministic dedup (never replace existing rows).
        new_paths = service.dedup_input_paths(paths, [e.path for e in self._lights])
        for path in new_paths:
            self._lights.append(_LightEntry(path, hdu=hdu))
        self._settings = dataclasses.replace(
            self._settings,
            last_input_dir=folder,
            window_width=self.width(), window_height=self.height(),
        )
        self._refresh_lights_list()
        self._bump_generation()
        self._update_scope_label()
        self.status_label.setText(
            presentation.format_folder_add_feedback(len(new_paths), unsupported)
        )

    def _on_remove_lights(self) -> None:
        for index in sorted(self._selected_rows(), reverse=True):
            if index < len(self._lights):
                del self._lights[index]
        self._refresh_lights_list()
        self._bump_generation()
        self._update_scope_label()

    def _on_clear_lights(self) -> None:
        if not self._lights:
            return
        self._lights.clear()
        self._refresh_lights_list()
        self._bump_generation()
        self._update_scope_label()

    def _on_apply_hdu(self) -> None:
        hdu = service.parse_hdu(self.hdu_edit.text())
        for index in self._selected_rows():
            self._lights[index].hdu = hdu
        self._refresh_lights_list()
        self._bump_generation()

    def _on_mode_changed(self, *_args) -> None:
        # Canonical (Advanced) combo changed: sync Standard labels, then invalidate once.
        self._sync_standard_modes()
        self._update_mode_note()
        self._bump_generation()

    def _on_standard_mode_changed(self, *_args) -> None:
        """Standard combo changed: forward to the canonical combo (single source of truth).

        Forwarding triggers ``_on_mode_changed`` exactly once (guarded), which is the
        only place that bumps the generation. This handler never bumps directly.
        """
        if self._syncing_modes:
            return
        self._syncing_modes = True
        try:
            self.additive_combo.setCurrentIndex(
                self.additive_combo.findData(self.standard_additive_combo.currentData())
            )
            self.flat_combo.setCurrentIndex(
                self.flat_combo.findData(self.standard_flat_combo.currentData())
            )
        finally:
            self._syncing_modes = False

    def _sync_standard_modes(self) -> None:
        """Reflect the canonical combo values into the Standard presentation combos."""
        if self._syncing_modes:
            return
        self._syncing_modes = True
        try:
            self.standard_additive_combo.setCurrentIndex(
                self.standard_additive_combo.findData(self.additive_combo.currentData())
            )
            self.standard_flat_combo.setCurrentIndex(
                self.standard_flat_combo.findData(self.flat_combo.currentData())
            )
        finally:
            self._syncing_modes = False

    def _on_selection_changed(self) -> None:
        self._update_scope_label()

    def _on_load_declaration(self) -> None:
        indices = self._selected_rows()
        if not indices:
            QtWidgets.QMessageBox.information(
                self, "No selection",
                "Select the light(s) the declaration applies to first.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select import declaration JSON", "", "JSON files (*.json)"
        )
        if not path:
            return
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="load_declaration",
            library_spec=None, request=None, policy=None, lights=(),
            evidence_path=path, target_indices=tuple(indices),
        )
        self._start_operation(snapshot)

    def _on_load_roi(self) -> None:
        indices = self._selected_rows()
        if not indices:
            QtWidgets.QMessageBox.information(
                self, "No selection",
                "Select the light(s) the ROI evidence applies to first.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ROI-extent evidence JSON", "", "JSON files (*.json)"
        )
        if not path:
            return
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="load_roi",
            library_spec=None, request=None, policy=None, lights=(),
            evidence_path=path, target_indices=tuple(indices),
        )
        self._start_operation(snapshot)

    def _on_open_library(self) -> None:
        index_path = self.library_index_edit.text().strip()
        if not index_path:
            index_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select library index", "",
                "SQLite index (*.sqlite);;All files (*)",
            )
            if not index_path:
                return
            self.library_index_edit.setText(index_path)
        root = str(Path(index_path).parent)
        self._library_spec = v1.LibrarySpec(root=root, index_path=index_path)
        self._set_active_source("explicit")
        self._bump_generation()
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="open_library",
            library_spec=self._library_spec, request=None, policy=None, lights=(),
        )
        self._start_operation(snapshot)

    def _on_browse_root(self) -> None:
        start = self._settings.last_library_dir or ""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select library root", start)
        if path:
            self.library_root_edit.setText(path)

    def _on_browse_imports(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select master imports JSON", "", "JSON files (*.json)"
        )
        if path:
            self.imports_edit.setText(path)

    def _on_index_library(self) -> None:
        root = self.library_root_edit.text().strip()
        imports_path = self.imports_edit.text().strip()
        if not root or not imports_path:
            QtWidgets.QMessageBox.warning(
                self, "Missing fields",
                "Provide both a library root directory and a master imports JSON file.",
            )
            return
        index_path = self.library_index_edit.text().strip() or self._default_index_path()
        self._library_spec = v1.LibrarySpec(root=root, index_path=index_path)
        self.library_index_edit.setText(index_path)
        self._set_active_source("explicit")
        self._bump_generation()
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="index_library",
            library_spec=self._library_spec, request=None, policy=None, lights=(),
            imports_path=imports_path,
        )
        self._start_operation(snapshot)

    # -- managed master ingestion (P7-M3B) -----------------------------------
    def _on_browse_darks(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select darks folder")
        if folder:
            self.darks_folder_edit.setText(folder)

    def _on_browse_bias(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select bias folder")
        if folder:
            self.bias_folder_edit.setText(folder)

    def _on_browse_flats(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select flats folder")
        if folder:
            self.flats_folder_edit.setText(folder)

    def _managed_master_targets(self) -> list:
        """Collect the selected master files (darks folder, bias folder, flats
        folder OR single flat file) as ``(path, master_type)`` targets."""
        targets: list = []

        def add_folder(edit, role):
            folder = edit.text().strip()
            if not folder:
                return
            try:
                paths, _ = service.scan_folder_inputs(folder)
            except ValueError:
                return
            for p in paths:
                targets.append((p, role))

        add_folder(self.darks_folder_edit, "dark")
        add_folder(self.bias_folder_edit, "bias")
        flats = self.flats_folder_edit.text().strip()
        if flats:
            if os.path.isdir(flats):
                add_folder(self.flats_folder_edit, "flat")
            elif service.is_supported_input_file(flats):
                targets.append((flats, "flat"))
        return targets

    def _on_scan_masters(self) -> None:
        targets = self._managed_master_targets()
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "No masters",
                "Select a darks folder, bias folder, or flats folder/file first.",
            )
            return
        self._managed_scan.clear()
        self._managed_pending.clear()
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="scan_masters",
            library_spec=None, request=None, policy=None, lights=(),
            master_paths=tuple(p for p, _ in targets),
            master_type=None,  # per-file type resolved from the target list
        )
        # Scan each role group independently so master_type is known per file.
        self._scan_targets = targets
        self._scan_index = 0
        self._scan_results: list[dict] = []
        self._start_scan_next()

    def _start_scan_next(self) -> None:
        if self._scan_index >= len(self._scan_targets):
            self._managed_scan = list(self._scan_results)
            self._present_managed_scan()
            return
        path, role = self._scan_targets[self._scan_index]
        self._scan_index += 1
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="scan_masters",
            library_spec=None, request=None, policy=None, lights=(),
            master_paths=(path,), master_type=role,
        )
        self._start_operation(snapshot)

    def _present_managed_scan(self) -> None:
        detected = sum(1 for m in self._managed_scan if m.get("candidates"))
        conflicted = sum(1 for m in self._managed_scan if m.get("conflicts"))
        self.managed_status_label.setText(
            f"Detected {len(self._managed_scan)} master(s): {detected} with facts, {conflicted} with conflicts."
        )
        lines = []
        for m in self._managed_scan:
            lines.append(f"[{m.get('master_type')}] {m.get('path')}")
            for field, fact in m.get("candidates", {}).items():
                lines.append("    " + presentation.format_evidence_fact(fact))
            for field, facts in m.get("conflicts", {}).items():
                lines.append("    " + presentation.format_conflict(field, facts))
        self.details_view.setPlainText("\n".join(lines) if lines else "(no candidates detected)")

    def _collect_user_facts(self, master_type: str, candidates: dict) -> dict:
        """Prompt for the required facts not already header-detected.

        Returns a dict of ``field -> parsed value`` for the facts the user
        supplied (empty entries are skipped, so absent stays absent). Flat
        quality evidence is never offered (R4); it is not in the required-field
        minimum.
        """
        missing = service.missing_fields_for_master(master_type, candidates)
        if not missing:
            return {}
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Supply required facts — {master_type}")
        form = QtWidgets.QFormLayout(dialog)
        edits: dict = {}
        for field in missing:
            edit = QtWidgets.QLineEdit()
            edits[field] = edit
            form.addRow(field, edit)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return {}
        extra: dict = {}
        for field, edit in edits.items():
            text = edit.text().strip()
            if not text:
                continue
            try:
                extra[field] = service.parse_user_fact(field, text)
            except ValueError as exc:
                self._log(f"[managed] invalid value for {field}: {exc}")
        return extra

    def _on_confirm_masters(self) -> None:
        if not self._managed_scan:
            QtWidgets.QMessageBox.information(self, "Nothing to confirm", "Detect masters first.")
            return
        self._managed_pending.clear()
        for m in self._managed_scan:
            if m.get("status") != "COMPLETED":
                continue
            if m.get("conflicts"):
                self._log(f"conflict not auto-resolved: {m.get('path')}")
                continue
            candidates = m.get("candidates", {})
            if not candidates:
                self._log(f"no detected facts for: {m.get('path')}")
                continue
            master_type = m.get("master_type")
            extra = self._collect_user_facts(master_type, candidates)
            evidence = {
                field: {
                    "value": fact.get("value"),
                    "origin_type": fact.get("origin_type", "fits_header"),
                    "origin_field": fact.get("origin_field"),
                }
                for field, fact in candidates.items()
            }
            self._managed_pending.append({
                "role": master_type,
                "path": m.get("path"),
                "hdu": 0,
                "evidence": evidence,
                "extra": extra,
                "dq_state": "no_source_dq",
                "mask_path": None,
                "declaration_source": "user",
                "declaration_identity": "managed-session",
                "declaration_version": "1",
            })
        if not self._managed_pending:
            QtWidgets.QMessageBox.information(
                self, "Nothing to confirm",
                "No unambiguous detected facts to confirm (conflicts are not auto-resolved).",
            )
            return
        self._confirm_index = 0
        self._start_confirm_next()

    def _start_confirm_next(self) -> None:
        if self._confirm_index >= len(self._managed_pending):
            self.managed_status_label.setText(
                f"Confirmed {len(self._managed_pending)} master(s)."
            )
            return
        payload = self._managed_pending[self._confirm_index]
        self._confirm_index += 1
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="confirm_evidence",
            library_spec=None, request=None, policy=None, lights=(),
            ledger_dir=str(self._storage.user_data_path),
            confirm_payload=payload,
        )
        self._start_operation(snapshot)

    def _on_build_managed(self) -> None:
        index_path = str(self._storage.user_cache_path / "zecalibrator.managed.sqlite")
        spec = v1.LibrarySpec(root=str(self._storage.user_cache_path), index_path=index_path)
        self._set_active_source("managed")
        self._bump_generation()
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="build_managed_library",
            library_spec=None, request=None, policy=None, lights=(),
            ledger_dir=str(self._storage.user_data_path),
            managed_spec=spec,
        )
        self._start_operation(snapshot)

    def _on_preflight(self) -> None:
        if not self._lights:
            QtWidgets.QMessageBox.information(self, "No lights", "Add at least one light frame.")
            return
        if not self._ensure_library():
            return
        self._bump_generation()
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="preflight",
            library_spec=self._library_spec,
            request=self._request(), policy=self._policy(),
            lights=self._lights_snapshot(self._lights),
        )
        self._start_operation(snapshot)

    def _on_calibrate_in_memory(self) -> None:
        light = self._selected_light()
        if light is None:
            QtWidgets.QMessageBox.information(self, "Select one light", "Select the light to calibrate.")
            return
        plan = self._plans.get(light.row_id)
        if plan is None or self._plan_generation != self._generation:
            QtWidgets.QMessageBox.information(
                self, "No matched plan",
                "Run preflight first; in-memory calibration requires a MATCHED plan "
                "for the selected light under the current configuration.",
            )
            return
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="calibrate_in_memory",
            library_spec=None, request=None, policy=None,
            lights=self._lights_snapshot([light]), plan=plan,
        )
        self._start_operation(snapshot)

    def _on_export(self) -> None:
        if not self._lights:
            QtWidgets.QMessageBox.information(self, "No lights", "Add at least one light frame.")
            return
        if not self._ensure_library():
            return
        selected = self._selected_rows()
        entries = [self._lights[i] for i in selected] if selected else list(self._lights)
        start_dir = self._settings.last_output_dir or ""
        destination = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output destination", start_dir
        )
        if not destination:
            return
        self._settings = dataclasses.replace(
            self._settings,
            last_output_dir=destination,
            window_width=self.width(), window_height=self.height(),
        )
        self._batch_items.clear()
        self.results_table.setRowCount(0)
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="export",
            library_spec=self._library_spec,
            request=self._request(), policy=self._policy(),
            lights=self._lights_snapshot(entries),
            destination=destination, batch_id=service.new_batch_id(),
        )
        self._start_operation(snapshot)

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        if self._token is not None:
            self._token.cancel()
        self.status_label.setText("Cancellation requested…")

    # ------------------------------------------------- worker signal handlers
    def _on_operation_started(self, op_id: str) -> None:
        if not self._is_current(op_id):
            return
        self.status_label.setText(self._active_label())

    def _on_progress(self, op_id: str, event) -> None:
        if not self._is_current(op_id):
            return
        if self._terminal_seen:
            return  # terminal result is authoritative; never overwrite it
        if self._cancel_requested:
            return  # never overwrite "Cancellation requested…" before terminal
        # Engine progress counters are phase-local (nested sub-phases), never a
        # reliable global operation denominator. Keep the busy indicator
        # indeterminate and leave the truthful operation-kind label in place;
        # never render a sub-phase fraction/percentage or `complete` as global
        # completion while the operation is still active.
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)

    def _on_preflight_light(self, op_id: str, summary: dict, plan) -> None:
        if not self._is_current(op_id):
            return
        self._preflight_summaries.append(summary)
        if plan is not None:
            self._plans[summary["row_id"]] = plan
            self._plan_generation = self._generation
        row = self.preflight_table.rowCount()
        self.preflight_table.insertRow(row)
        self.preflight_table.setItem(row, 0, QtWidgets.QTableWidgetItem(summary.get("display", "")))
        self.preflight_table.setItem(row, 1, QtWidgets.QTableWidgetItem(
            summary.get("outcome") or summary.get("inspect_status", "")))
        self.preflight_table.setItem(row, 2, QtWidgets.QTableWidgetItem(summary.get("plan_id") or ""))
        self.preflight_table.setItem(row, 3, QtWidgets.QTableWidgetItem(
            presentation.reason_codes_text(summary.get("reason_codes", ()))))
        self._update_standard_summary()

    def _on_batch_item(self, op_id: str, item: dict) -> None:
        if not self._is_current(op_id):
            return
        self._batch_items.append(item)
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self._set_results_row(row, item)

    def _set_results_row(self, row: int, item: dict) -> None:
        self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(item.get("index", ""))))
        self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(
            self._batch_displays[item.get("index", 0)] if item.get("index", 0) < len(self._batch_displays) else ""))
        self.results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(item.get("disposition", "")))
        self.results_table.setItem(row, 3, QtWidgets.QTableWidgetItem(item.get("plan_id") or ""))
        self.results_table.setItem(row, 4, QtWidgets.QTableWidgetItem(
            item.get("reason_code") or item.get("reason_details", "")))

    def _on_operation_finished(self, op_id: str, summary: dict) -> None:
        if not self._is_current(op_id):
            return
        self._terminal_seen = True
        kind = summary.get("kind")
        status = summary.get("status")
        if kind == "open_library":
            self._handle_open_finished(summary)
        elif kind == "index_library":
            self._handle_index_finished(summary)
        elif kind == "preflight":
            self._handle_preflight_finished(summary)
        elif kind == "calibrate_in_memory":
            self._handle_in_memory_finished(summary)
        elif kind == "export":
            self._handle_export_finished(summary)
        elif kind == "load_declaration":
            self._handle_declaration_loaded(summary)
        elif kind == "load_roi":
            self._handle_roi_loaded(summary)
        elif kind == "load_settings":
            self._handle_settings_loaded(summary)
        elif kind == "save_settings":
            self._handle_settings_saved(summary)
        elif kind == "scan_masters":
            self._handle_scan_masters(summary)
        elif kind == "confirm_evidence":
            self._handle_confirm_evidence(summary)
        elif kind == "build_managed_library":
            self._handle_build_managed(summary)
        self.progress_bar.setRange(0, 1)
        if status in ("COMPLETED", "OPENED"):
            self.progress_bar.setValue(1)
        else:
            self.progress_bar.setValue(0)

    def _handle_open_finished(self, summary: dict) -> None:
        if summary.get("status") == "OPENED":
            self.library_status.setText(
                f"Library opened: revision={summary.get('revision')!r} "
                f"roles={summary.get('roles')} counts={summary.get('candidate_counts')}"
            )
            detail = Path(self._library_spec.index_path).name if self._library_spec else ""
            self._set_library_human_status("ready", detail)
        else:
            self.library_status.setText(
                f"Library open failed: {summary.get('reason_code')} {summary.get('details')}"
            )
            self._set_library_human_status("unavailable", summary.get("reason_code") or "open failed")
            self._log(f"library open FAILED [{summary.get('reason_code')}]: {summary.get('details')}")

    def _handle_index_finished(self, summary: dict) -> None:
        if summary.get("status") == "COMPLETED":
            diagnostics = summary.get("diagnostics", ())
            self.library_status.setText(
                f"Indexed: revision={summary.get('revision')!r} "
                f"candidates={summary.get('candidate_count')} diagnostics={len(diagnostics)}"
            )
            for diag in diagnostics:
                self._log(f"index diagnostic: {diag}")
            if diagnostics:
                self._set_library_human_status("attention", f"{len(diagnostics)} diagnostic(s)")
            else:
                self._set_library_human_status("ready")
        else:
            self.library_status.setText(
                f"Indexing {summary.get('status')}: {summary.get('reason_code')} {summary.get('details')}"
            )
            self._set_library_human_status("unavailable", summary.get("reason_code") or "index failed")
            self._log(f"indexing FAILED [{summary.get('reason_code')}]: {summary.get('details')}")

    def _handle_preflight_finished(self, summary: dict) -> None:
        status = summary.get("status")
        if status == "FAILED":
            self.status_label.setText(
                f"Preflight failed: {summary.get('reason_code')} {summary.get('details')}"
            )
            self._log(f"preflight FAILED [{summary.get('reason_code')}]: {summary.get('details')}")
        else:
            self.status_label.setText(f"Preflight {status} ({summary.get('count', 0)} light(s)).")

    def _handle_in_memory_finished(self, summary: dict) -> None:
        self._in_memory_result = summary
        status = summary.get("status")
        if status in ("FAILED", "CANCELLED"):
            self.status_label.setText(f"In-memory calibration: {status} — {summary.get('reason_code', '')}")
            self._log(f"in-memory {status} [{summary.get('reason_code')}]: {summary.get('details', '')}")
        else:
            self.status_label.setText(f"In-memory calibration: {status}")
        self._render_in_memory_details(summary)
        self._audit_show(summary.get("provenance_audit"))

    def _handle_export_finished(self, summary: dict) -> None:
        self._batch_displays = list(summary.get("input_displays", []))
        self._batch_total_inputs = int(summary.get("total_inputs", 0))
        status = summary.get("status")
        # Truthful counters (F6).
        items = summary.get("items", [])
        committed = sum(1 for i in items if i["disposition"] == "COMPLETED")
        warnings = sum(1 for i in items if i["disposition"] == "COMPLETED_WITH_WARNINGS")
        skipped = sum(1 for i in items if i["disposition"] == "SKIPPED")
        failed = sum(1 for i in items if i["disposition"] == "FAILED")
        ran_indices = {i["index"] for i in items}
        not_started = [k for k in range(self._batch_total_inputs) if k not in ran_indices]

        parts = [
            f"committed={committed}", f"warnings={warnings}", f"skipped={skipped}",
            f"failed={failed}", f"not-started={len(not_started)}",
        ]
        self.status_label.setText(f"Export {status}: " + ", ".join(parts))
        if summary.get("reason_code"):
            self._log(f"export {status} [{summary['reason_code']}]: {summary.get('details')}")
        if status == "CANCELLED":
            self._log("cancellation requested — committed outputs preserved; not-started items remain explicit.")

        # Materialize not-started remainder rows (F6).
        self.results_table.setRowCount(0)
        for item in items:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self._set_results_row(row, item)
        for idx in not_started:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            disp = "CANCELLED" if status == "CANCELLED" else "NOT_STARTED"
            placeholder = {
                "index": idx,
                "disposition": disp,
                "plan_id": None,
                "reason_code": None,
                "reason_details": "not started" if status != "CANCELLED" else "cancelled before start",
            }
            self._set_results_row(row, placeholder)

        # Full batch result provenance audit + reachable manifest link.
        self._audit_show(summary.get("manifest_audit"))
        if summary.get("manifest_exists"):
            self._show_manifest_link(summary.get("manifest"))
            self._log(f"batch manifest: {summary.get('manifest')}")

    def _handle_declaration_loaded(self, summary: dict) -> None:
        declaration = summary.get("declaration")
        for index in summary.get("target_indices", []):
            if index < len(self._lights):
                self._lights[index].declaration = declaration
        self._refresh_lights_list()
        self._bump_generation()
        self.evidence_label.setText(f"Declaration loaded from {summary.get('display')} (selected lights).")
        self.status_label.setText("Declaration loaded.")

    def _handle_roi_loaded(self, summary: dict) -> None:
        roi = summary.get("roi")
        for index in summary.get("target_indices", []):
            if index < len(self._lights):
                self._lights[index].roi_extent = roi
        self._refresh_lights_list()
        self._bump_generation()
        self.evidence_label.setText(f"ROI evidence loaded from {summary.get('display')} (selected lights).")
        self.status_label.setText("ROI evidence loaded.")

    def _handle_scan_masters(self, summary: dict) -> None:
        for m in summary.get("masters", []):
            self._scan_results.append(m)
        if summary.get("status") == "FAILED":
            self.status_label.setText("Master detection failed.")
            return
        self._start_scan_next()

    def _handle_confirm_evidence(self, summary: dict) -> None:
        if summary.get("status") == "PRESERVED":
            self._log(f"[managed] ledger preserved ({summary.get('state')}); not overwritten.")
            self.status_label.setText("Managed ledger preserved (not overwritten).")
            return
        ev_status = summary.get("evidence_status")
        missing = summary.get("missing", [])
        if ev_status == "needs_attention":
            self._log(
                f"[managed] confirmed with insufficient evidence — missing/insufficient: "
                f"{', '.join(missing) or '(none)'}"
            )
            self.status_label.setText("Master confirmed (needs attention — insufficient evidence).")
        else:
            self.status_label.setText("Master evidence confirmed.")
        self._start_confirm_next()

    def _handle_build_managed(self, summary: dict) -> None:
        if summary.get("status") == "PRESERVED":
            self.status_label.setText("Managed ledger preserved (not overwritten).")
            self._log(f"[managed] ledger preserved ({summary.get('state')}).")
            return
        attention = summary.get("needs_attention", [])
        if summary.get("status") in ("REUSED", "COMPLETED"):
            self.managed_status_label.setText(
                f"Managed library {summary.get('status')}: revision={summary.get('revision')!r} "
                f"candidates={summary.get('candidate_count')}"
            )
            if attention:
                self._set_library_human_status("attention", f"{len(attention)} master(s) insufficient evidence")
                for a in attention:
                    self._log(
                        f"[managed] needs attention — {a.get('role')} {a.get('path')}: "
                        f"insufficient evidence ({', '.join(a.get('missing', ())) or '(none)'})"
                    )
            else:
                self._set_library_human_status("ready", "managed")
        else:
            self.managed_status_label.setText(
                f"Managed library {summary.get('status')}: {summary.get('reason_code')} {summary.get('details')}"
            )
            self._set_library_human_status("unavailable", summary.get('reason_code') or "managed build failed")

    def _on_operation_failed(self, op_id: str, reason_code: str, details: str) -> None:
        if not self._is_current(op_id):
            return
        self._terminal_seen = True
        self.status_label.setText(f"Failed: {reason_code}")
        self._log(f"operation {op_id[:12]} FAILED [{reason_code}]: {details}")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

    def _on_worker_ended(self) -> None:
        self._set_active(False)
        if self._close_requested:
            self.close()

    # ------------------------------------------------- detail rendering
    def _on_preflight_selection_changed(self, current_row: int, *_args) -> None:
        if current_row < 0 or current_row >= len(self._preflight_summaries):
            return
        summary = self._preflight_summaries[current_row]
        self._render_preflight_details(summary)
        audit = summary.get("decision_audit")
        if audit is not None:
            plan_audit = summary.get("plan_audit")
            if plan_audit is not None:
                audit = {"decision": audit, "plan": plan_audit}
        self._audit_show(audit)

    def _on_results_selection_changed(self, current_row: int, *_args) -> None:
        if current_row < 0:
            return
        if current_row < len(self._batch_items):
            self._render_batch_item_details(self._batch_items[current_row])
        else:
            self.details_view.setPlainText("(not started — no engine result to display)")

    def _render_preflight_details(self, summary: dict) -> None:
        lines = [
            f"File: {summary.get('display', '')}",
            f"Path: {summary.get('path', '')}",
            f"HDU: {summary.get('hdu', '')}",
            f"Inspect status: {summary.get('inspect_status', '')}",
            f"Domain: {summary.get('domain_finding', '')}",
            f"Outcome: {summary.get('outcome') or '(inspection failed)'}",
            f"Plan id: {summary.get('plan_id') or '(none)'}",
            f"Reason codes: {presentation.reason_codes_text(summary.get('reason_codes', ()))}",
            "",
            "Rejected candidates:",
            presentation.format_rejection_table(summary.get("rejected_candidates", ())),
            "",
            "Coherent sets:",
            presentation.format_coherent_sets(summary.get("coherent_sets", ())),
        ]
        structural = summary.get("reasons", ())
        if structural:
            lines += ["", "Structural reasons:"]
            for r in structural:
                lines.append(
                    f"    - {r.get('code')} {r.get('field') or ''}: "
                    f"expected={r.get('expected')} observed={r.get('observed')}"
                )
        lines += ["", "Metadata:"]
        lines.append(presentation.format_metadata(summary.get("metadata", {})))
        if summary.get("inspect_status") == "FAILED":
            lines += ["", f"Reason: {summary.get('reason_code')} {summary.get('details', '')}"]
        self.details_view.setPlainText("\n".join(lines))

    def _render_batch_item_details(self, item: dict) -> None:
        out = item.get("output")
        lines = [
            f"Index: {item.get('index')}",
            f"Disposition: {item.get('disposition')}",
            f"Plan id: {item.get('plan_id') or '(none)'}",
            f"Reason code: {item.get('reason_code') or '(none)'}",
            f"Reason details: {item.get('reason_details') or '(none)'}",
            f"Warnings: {', '.join(item.get('warnings', ())) or '(none)'}",
        ]
        if out:
            lines += [
                "",
                "Committed output:",
                f"    path: {out.get('path')}",
                f"    logical_id: {out.get('logical_id')}",
                f"    science_digest: {out.get('science_digest')}",
                f"    whole_file_sha256: {out.get('whole_file_sha256')}",
                f"    size_bytes: {out.get('size_bytes')}",
                f"    committed: {out.get('committed')}",
            ]
        self.details_view.setPlainText("\n".join(lines))

    def _render_in_memory_details(self, summary: dict) -> None:
        lines = [
            f"In-memory calibration: {summary.get('status')}",
            f"File: {summary.get('display')}",
            f"Plan id: {summary.get('plan_id')}",
            f"Saturation evidence: {summary.get('saturation_evidence')}",
            f"Warnings: {', '.join(summary.get('warnings', ())) or '(none)'}",
            f"Reason code: {summary.get('reason_code') or '(none)'}",
            f"Scalars: {summary.get('scalars')}",
            f"Counts: {summary.get('counts')}",
        ]
        prov = summary.get("provenance")
        if prov:
            lines += [
                "",
                "Provenance:",
                f"    operation_id: {prov.get('operation_id')}",
                f"    status: {prov.get('status')}",
                f"    science_contract: {prov.get('science_contract')}",
                f"    matching_policy: {prov.get('matching_policy')}",
                f"    provenance_schema: {prov.get('provenance_schema')}",
            ]
        self.details_view.setPlainText("\n".join(lines))

    # ------------------------------------------------------------- close
    def _request_close(self) -> None:
        if not self._close_requested:
            self._close_requested = True
            self.status_label.setText("Close requested — cancelling and waiting for cleanup…")
            if self._token is not None:
                self._token.cancel()

    def closeEvent(self, event) -> None:
        if self._controller.is_active:
            event.ignore()
            self._request_close()
            return
        if not self._settings_saved:
            event.ignore()
            self._request_settings_save()
            if self._settings_saved:
                # Preservation path: nothing to write; close now.
                self._controller.shutdown()
                super().closeEvent(event)
            return
        self._controller.shutdown()  # finished-driven quit (non-blocking)
        super().closeEvent(event)

    def _save_settings(self) -> None:
        # Kept only for compatibility with external tests that may call it; the
        # real path is the async _request_settings_save / save_settings worker op.
        self._request_settings_save()


def create_qapplication(argv=None):
    """Create/return the singleton QApplication with identity + icon applied."""
    identity.apply_process_identity()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(list(argv or []))
    app.setApplicationName("ZeCalibrator")
    app.setApplicationDisplayName("ZeCalibrator")
    app.setOrganizationName("ZeSoftware")
    identity.apply_linux_desktop_file(app)
    icon = identity.load_app_icon()
    if icon is not None and not icon.isNull():
        app.setWindowIcon(icon)
    return app


def create_main_window(storage_paths: StoragePaths | None = None, parent=None) -> MainWindow:
    """Create a main window with injected storage paths (defaults to platform roots)."""
    from zecalibrator.storage import resolve_paths

    paths = storage_paths or resolve_paths()
    return MainWindow(paths, parent=parent)


def run_application(argv=None, storage_paths: StoragePaths | None = None) -> int:
    """GUI entry point: build the app, show the window, run the event loop.

    After ``app.exec()`` returns (the event loop has stopped), perform a bounded
    post-loop join of the worker thread — this is safe teardown, not a GUI-thread
    block, and never terminates the thread.
    """
    app = create_qapplication(argv)
    window = create_main_window(storage_paths, parent=None)
    window.show()
    rc = app.exec()
    window._controller.finalize()
    return rc


__all__ = ["MainWindow", "create_main_window", "create_qapplication", "run_application"]
