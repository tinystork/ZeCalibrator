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
        self._bpm_root: str | None = None
        self._bpm_settings_state = None
        self._bpm_loaded = False
        self._bpm_saved = False
        self._bpm_detector_k = 30.0

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
        self._syncing_theme = False
        self._active_kind = None
        self._cancel_requested = False

        # Managed master ingestion state (P7-M3B).
        self._master_files: list[str] = []
        self._managed_scan: list[dict] = []
        self._managed_pending: list[dict] = []
        self._active_source: str | None = None  # "managed" | "explicit" | None
        self._managed_spec: v1.LibrarySpec | None = None
        self._confirm_queue: list[dict] = []
        self._confirm_batch_active: bool = False
        self._session_selection: list[tuple[str, str]] = []
        self._confirmed_ready: list[bool] = []  # diagnostic only (never a gate)
        self._auto_flow_active: bool = False
        self._master_attention_notes: list[str] = []
        self._export_after_preflight: bool = False
        self._preflight_is_standard: bool = False
        self._standard_route_generation: int = -1
        # LOT 3: map-creation proposal state (one offer per run, §9 E).
        self._bpm_proposed = False
        self._pending_bpm_export = None
        # P4.2.1: one rebuild decision per run (§3) + one no-database prompt (§4).
        # The mismatch decision is an explicit per-run cache (never just "prompt
        # seen"): "rebuild" | "use_existing" | "cancel" | None (not yet asked).
        self._bpm_mismatch_decision = None
        self._bpm_no_db_prompted = False
        # Serialized §4 onboarding: continue the BPM decision workflow only after
        # the settings-save worker has ended (never while it is still active).
        self._pending_bpm_workflow = None  # (request, destination) or None
        self._pending_bpm_workflow_ok = False

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
            self.add_masters_folder_btn, self.add_masters_file_btn,
            self.remove_masters_btn, self.clear_masters_btn,
            self.include_subfolders_check,
            self.scan_masters_btn, self.confirm_masters_btn, self.build_managed_btn,
            self.output_browse_btn,
            self.bpm_k_edit, self.bpm_k_reset_btn,
        ]
        self._launch_widgets = [self.calibrate_btn, self.export_btn,
                                self.advanced_preflight_btn, self.advanced_export_btn]
        self._update_scope_label()

    def _build_standard_tab(self) -> None:
        """Standard (nominal) one-step workflow: human-first, auto-route, no technical controls."""
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

        masters_box = QtWidgets.QGroupBox("Calibration masters")
        masters_layout = QtWidgets.QVBoxLayout(masters_box)
        self.masters_files_list = QtWidgets.QListWidget()
        self.masters_files_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        masters_layout.addWidget(self.masters_files_list)
        masters_add_row = QtWidgets.QHBoxLayout()
        self.add_masters_folder_btn = QtWidgets.QPushButton("Add masters folder…")
        self.add_masters_file_btn = QtWidgets.QPushButton("Add masters file…")
        self.remove_masters_btn = QtWidgets.QPushButton("Remove selected")
        self.clear_masters_btn = QtWidgets.QPushButton("Clear")
        for b in (
            self.add_masters_folder_btn, self.add_masters_file_btn,
            self.remove_masters_btn, self.clear_masters_btn,
        ):
            masters_add_row.addWidget(b)
        self.include_subfolders_check = QtWidgets.QCheckBox("Include subfolders")
        self.include_subfolders_check.setChecked(False)
        masters_add_row.addWidget(self.include_subfolders_check)
        masters_add_row.addStretch(1)
        masters_layout.addLayout(masters_add_row)
        self.managed_status_label = QtWidgets.QLabel("Add calibration masters to begin.")
        masters_layout.addWidget(self.managed_status_label)
        layout.addWidget(masters_box)

        output_box = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QHBoxLayout(output_box)
        output_layout.addWidget(QtWidgets.QLabel("Output folder:"))
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        output_layout.addWidget(self.output_dir_edit, 1)
        self.output_browse_btn = QtWidgets.QPushButton("Browse…")
        output_layout.addWidget(self.output_browse_btn)
        layout.addWidget(output_box)

        actions_row = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Calibrate / Export…")
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

        managed_box = QtWidgets.QGroupBox("Managed masters")
        managed_layout = QtWidgets.QVBoxLayout(managed_box)
        managed_row = QtWidgets.QHBoxLayout()
        self.scan_masters_btn = QtWidgets.QPushButton("Detect masters…")
        self.confirm_masters_btn = QtWidgets.QPushButton("Confirm detected facts")
        self.build_managed_btn = QtWidgets.QPushButton("Build managed library")
        for b in (self.scan_masters_btn, self.confirm_masters_btn, self.build_managed_btn):
            managed_row.addWidget(b)
        managed_row.addStretch(1)
        managed_layout.addLayout(managed_row)
        layout.addWidget(managed_box)

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
        # Advanced explicit-route actions: dispatch the Advanced additive/flat
        # combos explicitly (``request=self._request()``), never the auto-route.
        explicit_row = QtWidgets.QHBoxLayout()
        self.advanced_preflight_btn = QtWidgets.QPushButton("Verify with selected modes")
        self.advanced_export_btn = QtWidgets.QPushButton("Export with selected modes")
        explicit_row.addWidget(self.advanced_preflight_btn)
        explicit_row.addWidget(self.advanced_export_btn)
        explicit_row.addStretch(1)
        modes_layout.addLayout(explicit_row, 5, 0, 1, 2)
        layout.addWidget(modes_box)
        self._update_mode_note()

        # P4.2.1: the single scientific BPM setting — detector threshold K.
        bpm_box = QtWidgets.QGroupBox("Bad Pixel Map")
        bpm_layout = QtWidgets.QGridLayout(bpm_box)
        bpm_layout.addWidget(QtWidgets.QLabel("Bad pixel detection threshold:"), 0, 0)
        self.bpm_k_edit = QtWidgets.QDoubleSpinBox()
        self.bpm_k_edit.setRange(0.01, 1000.0)
        self.bpm_k_edit.setDecimals(1)
        self.bpm_k_edit.setSingleStep(1.0)
        self.bpm_k_edit.setValue(30.0)
        self.bpm_k_edit.setToolTip("Lower values detect more candidate bad pixels.")
        bpm_layout.addWidget(self.bpm_k_edit, 0, 1)
        self.bpm_k_reset_btn = QtWidgets.QPushButton("Reset default")
        self.bpm_k_reset_btn.setToolTip("Restore the default threshold (30.0).")
        bpm_layout.addWidget(self.bpm_k_reset_btn, 0, 2)
        self.bpm_k_note = QtWidgets.QLabel("")
        self.bpm_k_note.setStyleSheet("color: gray;")
        bpm_layout.addWidget(self.bpm_k_note, 1, 0, 1, 3)
        layout.addWidget(bpm_box)

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
        """Settings: application preferences (Appearance / Theme + BPM location)."""
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

        # §16/§17: the single Bad Pixel Database user setting (location only —
        # never a scientific setting). Persisted through the existing bpm.settings
        # + injected StoragePaths, never a second configuration system.
        bpm_box = QtWidgets.QGroupBox("Bad Pixel Database")
        bpm_layout = QtWidgets.QVBoxLayout(bpm_box)
        bpm_row = QtWidgets.QHBoxLayout()
        bpm_row.addWidget(QtWidgets.QLabel("Bad Pixel Database location:"))
        self.bpm_root_edit = QtWidgets.QLineEdit()
        self.bpm_root_edit.setPlaceholderText("No Bad Pixel Database configured")
        self.bpm_root_edit.setReadOnly(True)
        bpm_row.addWidget(self.bpm_root_edit, 1)
        self.bpm_browse_btn = QtWidgets.QPushButton("Browse…")
        self.bpm_browse_btn.setEnabled(False)  # enabled after async BPM settings load
        bpm_row.addWidget(self.bpm_browse_btn)
        bpm_layout.addLayout(bpm_row)
        self.bpm_status_label = QtWidgets.QLabel("")
        self.bpm_status_label.setStyleSheet("color: gray;")
        bpm_layout.addWidget(self.bpm_status_label)
        layout.addWidget(bpm_box)
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
        self.calibrate_btn.clicked.connect(self._on_calibrate_in_memory)
        self.export_btn.clicked.connect(self._on_export)
        self.advanced_preflight_btn.clicked.connect(self._on_advanced_preflight)
        self.advanced_export_btn.clicked.connect(self._on_advanced_export)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.audit_link_btn.clicked.connect(self._on_open_manifest)
        self.output_browse_btn.clicked.connect(self._on_browse_output)
        self.add_masters_folder_btn.clicked.connect(self._on_add_masters_folder)
        self.add_masters_file_btn.clicked.connect(self._on_add_masters_file)
        self.remove_masters_btn.clicked.connect(self._on_remove_masters)
        self.clear_masters_btn.clicked.connect(self._on_clear_masters)
        self.scan_masters_btn.clicked.connect(self._on_scan_masters)
        self.confirm_masters_btn.clicked.connect(self._on_confirm_masters)
        self.build_managed_btn.clicked.connect(self._on_build_managed)
        self.bpm_browse_btn.clicked.connect(self._on_browse_bpm_root)
        self.bpm_k_edit.valueChanged.connect(self._on_bpm_k_changed)
        self.bpm_k_reset_btn.clicked.connect(self._on_bpm_k_reset)

        self.additive_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.flat_combo.currentIndexChanged.connect(self._on_mode_changed)
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
        self._controller.collision_query.connect(self._on_collision_query)
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
        # A new run (config/input change) allows one fresh map-creation offer.
        self._bpm_proposed = False
        self._bpm_mismatch_decision = None
        self._bpm_no_db_prompted = False

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
        # BPM settings load folded into the same startup operation (single async
        # load, no second queued operation).
        self._bpm_settings_state = summary.get("bpm_state")
        self._bpm_root = summary.get("bpm_root")
        self._bpm_loaded = True
        self.bpm_root_edit.setText(self._bpm_root or "")
        self.bpm_browse_btn.setEnabled(True)
        # P4.2.1: restore the single scientific BPM setting (detector K).
        self._bpm_detector_k = float(summary.get("bpm_detector_k", 30.0))
        self._sync_bpm_k_edit(self._bpm_detector_k)
        self._refresh_bpm_status_label()
        self.resize(self._settings.window_width, self._settings.window_height)
        self._apply_theme(self._settings.appearance_theme)
        self._sync_theme_combo()
        self.theme_combo.setEnabled(True)
        if self._settings.last_output_dir:
            self.output_dir_edit.setText(self._settings.last_output_dir)
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

    # -- Bad Pixel Database setting (location + the single detector K) -------
    def _request_bpm_settings_save(self, root: str, detector_k: float | None = None) -> None:
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="save_bpm_settings",
            library_spec=None, request=None, policy=None, lights=(),
            config_dir=str(self._storage.user_config_path),
            bpm_root=root,
            bpm_detector_k=detector_k if detector_k is not None else self._bpm_detector_k,
        )
        self._start_operation(snapshot)

    def _handle_bpm_settings_saved(self, summary: dict) -> None:
        self._bpm_saved = True
        # Record whether persistence actually succeeded so a serialized §4
        # onboarding continuation only proceeds after a real, successful save.
        self._pending_bpm_workflow_ok = summary.get("status") == "COMPLETED"
        if summary.get("status") == "PRESERVED":
            self._log("[bpm] existing Bad Pixel Database settings preserved (not overwritten).")
        else:
            self._bpm_root = summary.get("root") if "root" in summary else self.bpm_root_edit.text().strip()
            self.bpm_root_edit.setText(self._bpm_root or "")
            if "detector_k" in summary and summary.get("detector_k") is not None:
                self._bpm_detector_k = float(summary["detector_k"])
        self._refresh_bpm_status_label()

    def _refresh_bpm_status_label(self) -> None:
        if not self._bpm_loaded:
            self.bpm_status_label.setText("")
            return
        if self._bpm_root:
            self.bpm_status_label.setText(f"Bad Pixel Database: {self._bpm_root}")
        else:
            self.bpm_status_label.setText("No Bad Pixel Database configured")

    def _sync_bpm_k_edit(self, value: float) -> None:
        """Set the threshold spin box without triggering a persistence write."""
        from PySide6 import QtCore

        self.bpm_k_edit.blockSignals(True)
        try:
            self.bpm_k_edit.setValue(float(value))
        finally:
            self.bpm_k_edit.blockSignals(False)

    def _on_bpm_k_changed(self, value: float) -> None:
        """Persist a changed detector threshold calmly (reject invalid values)."""
        from zecalibrator.api.v1 import _bpm

        try:
            k = _bpm.validate_detector_k(value)
        except ValueError as exc:
            self.bpm_k_note.setText(str(exc))
            self.status_label.setText(f"Bad pixel detection threshold unchanged: {exc}")
            self._sync_bpm_k_edit(self._bpm_detector_k)
            return
        self._bpm_detector_k = k
        self.bpm_k_note.setText("")
        # Persist via the existing BPM settings mechanism (root + K).
        self._request_bpm_settings_save(self._bpm_root or "", detector_k=k)

    def _on_bpm_k_reset(self) -> None:
        """Reset the detector threshold to exactly the default (30.0)."""
        self._bpm_detector_k = 30.0
        self._sync_bpm_k_edit(30.0)
        self.bpm_k_note.setText("")
        self._request_bpm_settings_save(self._bpm_root or "", detector_k=30.0)

    def _on_browse_bpm_root(self) -> None:
        """Browse → validate/create the selected folder → persist the location."""
        start = self._bpm_root or str(self._storage.user_data_path)
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Bad Pixel Database location", start
        )
        if not folder:
            return
        from zecalibrator.api.v1 import _bpm

        path, error = _bpm.ensure_bpm_root(folder, create=True)
        if error is not None:
            QtWidgets.QMessageBox.warning(self, "Bad Pixel Database location", error)
            return
        self._request_bpm_settings_save(path)

    def _storage_paths_dict(self) -> dict:
        return {name: str(p) for name, p in self._storage.as_dict().items()}

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

    # -- Standard human status (single human-readable state) --------------
    _ROLE_HUMAN_CAPITAL = {
        "dark": "Dark", "bias": "Bias", "flat": "Flat", "flat_dark": "Dark flat",
    }

    def _human_role_list(self, roles) -> str:
        return " · ".join(self._ROLE_HUMAN_CAPITAL.get(r, r) for r in roles)

    def _bound_plan_roles(self) -> list[str]:
        """The sorted set of master roles actually bound by the resolved plans.

        R3D-G: only the roles the resolved :class:`CalibrationPlan` consumes are
        shown — never a supplied session master the plan does not use (e.g. a
        DARKFLAT that is not consumed by a prepared flat).
        """
        roles: set[str] = set()
        for plan in self._plans.values():
            if plan is not None:
                roles.update(plan.masters.keys())
        return sorted(roles)

    def _set_standard_human_status(self, state: str, detail: str = "") -> None:
        if state == "ready":
            text = "Ready to calibrate" + (f" — {detail}" if detail else "")
        elif state in ("attention", "unavailable"):
            text = "Needs attention" + (f" — {detail}" if detail else "")
        else:
            text = detail or "Add calibration masters to begin."
        self.managed_status_label.setText(text)

    # -- exactly-one-active calibration source (managed vs explicit) ----------
    def _set_active_source(self, source: str | None) -> None:
        """Set exactly one active calibration source and clear the other.

        Selecting the managed source clears the explicit library spec (and vice
        versa) so the two never mix silently.
        """
        if source == "managed":
            self._active_source = "managed"
            self._library_spec = None
            self.library_index_edit.clear()
            self.library_root_edit.clear()
        elif source == "explicit":
            self._active_source = "explicit"
            self._managed_scan.clear()
            self._managed_pending.clear()
        else:
            self._active_source = None

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
        # Advanced combo changed: invalidate cached plans/results once.
        self._update_mode_note()
        self._bump_generation()

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

    def _on_browse_output(self) -> None:
        start = self.output_dir_edit.text().strip() or self._settings.last_output_dir or ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", start
        )
        if not folder:
            return
        self.output_dir_edit.setText(folder)
        self._settings = dataclasses.replace(
            self._settings,
            last_output_dir=folder,
            window_width=self.width(), window_height=self.height(),
        )

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
    def _on_add_masters_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select masters folder")
        if not folder:
            return
        try:
            paths, unsupported = service.scan_folder_inputs(
                folder, recursive=self.include_subfolders_check.isChecked()
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid folder", str(exc))
            return
        # One folder is scanned once; each file later gets a single role from
        # IMAGETYP evidence or explicit user confirmation (never triple-indexed).
        new_paths = service.dedup_input_paths(paths, self._master_files)
        self._master_files.extend(new_paths)
        self._refresh_masters_list()
        self.status_label.setText(
            presentation.format_folder_add_feedback(len(new_paths), unsupported)
        )
        if new_paths:
            self._invalidate_managed_source()
            self._auto_detect_masters()

    def _on_add_masters_file(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select master FITS files", "", service.input_dialog_filter(),
        )
        if not paths:
            return
        new_paths = service.dedup_input_paths(paths, self._master_files)
        self._master_files.extend(new_paths)
        self._refresh_masters_list()
        self.status_label.setText(
            presentation.format_folder_add_feedback(len(new_paths), 0)
        )
        if new_paths:
            self._invalidate_managed_source()
            self._auto_detect_masters()

    def _on_remove_masters(self) -> None:
        rows = sorted({i.row() for i in self.masters_files_list.selectedIndexes()}, reverse=True)
        for index in rows:
            if index < len(self._master_files):
                del self._master_files[index]
        self._refresh_masters_list()
        self._invalidate_managed_source()

    def _on_clear_masters(self) -> None:
        if not self._master_files:
            return
        self._master_files.clear()
        self._managed_scan.clear()
        self._managed_pending.clear()
        self._refresh_masters_list()
        self._invalidate_managed_source()

    def _refresh_masters_list(self) -> None:
        self.masters_files_list.clear()
        for path in self._master_files:
            item = QtWidgets.QListWidgetItem(Path(path).name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.masters_files_list.addItem(item)

    def _invalidate_managed_source(self) -> None:
        """Reset managed-source state whenever the master set changes."""
        if self._active_source == "managed":
            self._active_source = None
            self._library_spec = None
        self._managed_spec = None
        self._session_selection.clear()
        self._master_attention_notes.clear()
        self._confirm_queue.clear()
        self._confirm_batch_active = False
        self._auto_flow_active = False
        self._set_standard_human_status(None, "")

    def _auto_detect_masters(self) -> None:
        """F6: adding masters auto-detects (one scan) then auto-confirms/builds."""
        self._auto_flow_active = True
        self._start_scan_all()

    def _auto_confirm(self) -> None:
        self._collect_and_dispatch_confirms()

    def _on_scan_masters(self) -> None:
        if not self._master_files:
            QtWidgets.QMessageBox.information(
                self, "No masters",
                "Add a masters folder or master files first.",
            )
            return
        self._auto_flow_active = False
        self._start_scan_all()

    def _start_scan_all(self) -> None:
        """Detect the complete master set in ONE worker op (no GUI-side chaining)."""
        self._managed_scan.clear()
        self._managed_pending.clear()
        self._master_attention_notes.clear()
        self._confirm_queue.clear()
        self._confirm_batch_active = False
        self._session_selection.clear()
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="scan_masters",
            library_spec=None, request=None, policy=None, lights=(),
            master_targets=tuple((p, None) for p in self._master_files),
        )
        self._start_operation(snapshot)

    def _present_managed_scan(self) -> None:
        detected = sum(1 for m in self._managed_scan if m.get("candidates"))
        incompatible = sum(1 for m in self._managed_scan if not m.get("admissible", True))
        conflicted = sum(1 for m in self._managed_scan if m.get("conflict"))
        # Scan facts are rendered into the Advanced details pane; the Standard
        # human status label is left for the (auto) flow's terminal human state.
        lines = [
            f"Detected {len(self._managed_scan)} master(s): {detected} with facts, "
            f"{incompatible} incompatible, {conflicted} role conflict(s).",
        ]
        for m in self._managed_scan:
            role = m.get("role") or m.get("detected_role")
            lines.append(f"[{role or 'no role'}] {m.get('path')}")
            if m.get("incompatible"):
                for reason in m["incompatible"]:
                    lines.append("    " + presentation.format_incompatibility(reason))
            if m.get("conflict"):
                lines.append("    " + presentation.format_role_conflict(m["conflict"]))
            for field, fact in m.get("candidates", {}).items():
                lines.append("    " + presentation.format_evidence_fact(fact))
            for field, facts in m.get("conflicts", {}).items():
                lines.append("    " + presentation.format_conflict(field, facts))
        self.details_view.setPlainText("\n".join(lines) if lines else "(no candidates detected)")

    def _collect_user_facts(self, master_type: str, candidates: dict) -> dict:
        """Confirm only genuinely-necessary-and-missing facts, in human terms.

        Never shows internal field names. Header-derived facts and disambiguators
        are never prompted. CFA-only geometry facts (orientation/roi_origin) are
        presented as ONE clear checkbox confirmation mapped explicitly to their
        internal values (provenance = user confirmation, never guessed silently).
        Returns a dict of ``field -> parsed value`` for the facts the user
        supplied (absent stays absent). Flat quality evidence is never offered
        (R4).
        """
        missing = service.missing_fields_for_master(master_type, candidates)
        if not missing:
            return {}
        text_fields = [f for f in missing if f not in service.CFA_ONLY_FIELDS]
        cfa_fields = [f for f in missing if f in service.CFA_ONLY_FIELDS]

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Confirm required facts — {master_type}")
        form = QtWidgets.QFormLayout(dialog)

        edits: dict = {}
        for field in text_fields:
            label = service.human_fact_label(field) or field
            edit = QtWidgets.QLineEdit()
            edits[field] = edit
            form.addRow(label, edit)

        cfa_checks: dict = {}
        for field in cfa_fields:
            confirmation = service.cfa_geometry_confirmation(field)
            if confirmation is None:
                continue
            _cid, label, value = confirmation
            check = QtWidgets.QCheckBox(label)
            cfa_checks[field] = (check, value)
            form.addRow("", check)

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
        for field, (check, value) in cfa_checks.items():
            if check.isChecked():
                extra[field] = value
        return extra

    def _resolve_role_conflict(self, conflict: dict) -> str | None:
        """Ask the human to resolve a selected-vs-detected role conflict.

        Never auto-resolves. Returns the chosen role, or ``None`` to skip.
        """
        selected = conflict.get("selected_role")
        detected = conflict.get("detected_role")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Role conflict — needs confirmation")
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setText(
            f"Selected role: {selected}\nDetected role: {detected}\n"
            f"Source: {conflict.get('source', 'FITS IMAGETYP')}\nNeeds confirmation"
        )
        selected_btn = box.addButton(f"Use {selected}", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        detected_btn = box.addButton(f"Use {detected}", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is selected_btn:
            return selected
        if clicked is detected_btn:
            return detected
        return None

    def _collect_role(self, path: str) -> str | None:
        """Ask the human for a master role when IMAGETYP is absent."""
        roles = ("dark", "bias", "flat", "flat_dark")
        chosen, ok = QtWidgets.QInputDialog.getItem(
            self, "Select master role", f"Role for {Path(path).name}:", roles, 0, False,
        )
        return chosen if ok else None

    def _on_confirm_masters(self) -> None:
        self._auto_flow_active = False
        self._collect_and_dispatch_confirms()

    def _collect_and_dispatch_confirms(self) -> None:
        if not self._managed_scan:
            if not self._auto_flow_active:
                QtWidgets.QMessageBox.information(self, "Nothing to confirm", "Detect masters first.")
            return
        self._managed_pending.clear()
        self._confirmed_ready.clear()
        self._master_attention_notes.clear()
        for m in self._managed_scan:
            if m.get("status") != "COMPLETED":
                continue
            if not m.get("admissible", True):
                reasons = "; ".join(m.get("incompatible", ())) or "incompatible"
                self._master_attention_notes.append(
                    f"{Path(m.get('path')).name} is incompatible: {reasons}"
                )
                self._log(
                    f"[managed] incompatible master not indexed: {m.get('path')} — "
                    f"{reasons}"
                )
                continue
            role = m.get("role")
            conflict = m.get("conflict")
            candidates = m.get("candidates", {})
            if self._auto_flow_active:
                # Standard auto-flow (R3D-C): no modal dialogs. The role is
                # auto-detected from IMAGETYP; a role conflict or an
                # undeterminable role is reported as a non-modal per-master
                # needs-attention note (never a questionnaire).
                if conflict:
                    self._master_attention_notes.append(
                        f"declared and detected master roles disagree: {Path(m.get('path')).name}"
                    )
                    self._log(
                        f"[managed] needs attention — declared and detected "
                        f"master roles disagree: {m.get('path')}"
                    )
                    continue
                if not role:
                    self._master_attention_notes.append(
                        f"master role could not be determined: {Path(m.get('path')).name}"
                    )
                    self._log(
                        f"[managed] needs attention — master role could not be "
                        f"determined: {m.get('path')}"
                    )
                    continue
                extra = {}
            else:
                if conflict:
                    role = self._resolve_role_conflict(conflict)
                    if role is None:
                        self._log(f"[managed] role conflict skipped (not auto-resolved): {m.get('path')}")
                        continue
                if not role:
                    role = self._collect_role(m.get("path"))
                    if role is None:
                        continue
                extra = self._collect_user_facts(role, candidates)
            evidence = {
                field: {
                    "value": fact.get("value"),
                    "origin_type": fact.get("origin_type", "fits_header"),
                    "origin_field": fact.get("origin_field"),
                }
                for field, fact in candidates.items()
            }
            self._managed_pending.append({
                "role": role,
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
            if self._auto_flow_active:
                self._set_standard_human_status("attention", self._first_attention_note())
            else:
                QtWidgets.QMessageBox.information(
                    self, "Nothing to confirm",
                    "No admissible detected masters to confirm (incompatible/conflicts are not auto-resolved).",
                )
            return
        self._confirm_queue = list(self._managed_pending)
        self._confirm_batch_active = True
        self._dispatch_next_confirm()

    def _first_attention_note(self) -> str:
        """Return the first concrete human attention note (or a safe fallback)."""
        if self._master_attention_notes:
            return self._master_attention_notes[0]
        return "add compatible calibration masters."

    def _dispatch_next_confirm(self) -> None:
        """Dispatch the next queued confirm; the queue is drained on ``worker_ended``
        (never re-entrantly from ``operation_finished`` while ``is_active`` is true)."""
        if self._confirm_queue:
            if self._controller.is_active:
                return  # drained from _on_worker_ended
            payload = self._confirm_queue.pop(0)
            snapshot = service.OperationSnapshot(
                op_id=service.new_operation_id(), kind="confirm_evidence",
                library_spec=None, request=None, policy=None, lights=(),
                ledger_dir=str(self._storage.user_data_path),
                confirm_payload=payload,
            )
            self._start_operation(snapshot)
            return
        if self._confirm_batch_active:
            self._confirm_batch_active = False
            self._finish_confirm_flow()

    def _finish_confirm_flow(self) -> None:
        """After the confirm queue drains, ALWAYS prepare the managed library when
        at least one admitted master is pending (R3D-D). Missing/incomplete
        evidence is indexed (UNKNOWN/UNVERIFIED) and resolved by the R3D-C
        matcher, never dropped here. ``_confirmed_ready`` is diagnostic only."""
        if self._auto_flow_active and self._managed_pending:
            self._on_build_managed()
        self._auto_flow_active = False

    def _on_build_managed(self) -> None:
        index_path = str(self._storage.user_cache_path / "zecalibrator.managed.sqlite")
        spec = v1.LibrarySpec(root=str(self._storage.user_cache_path), index_path=index_path)
        self._set_active_source("managed")
        self._managed_spec = spec
        self._bump_generation()
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="build_managed_library",
            library_spec=None, request=None, policy=None, lights=(),
            ledger_dir=str(self._storage.user_data_path),
            managed_spec=spec,
            session_selection=tuple(self._session_selection),
        )
        self._start_operation(snapshot)

    def _on_preflight(self) -> None:
        # Standard: auto-route (request=None).
        self._dispatch_preflight(None)

    def _on_advanced_preflight(self) -> None:
        # Advanced: explicit request from the additive/flat combos.
        self._dispatch_preflight(self._request())

    def _dispatch_preflight(self, request) -> None:
        if not self._lights:
            QtWidgets.QMessageBox.information(self, "No lights", "Add at least one light frame.")
            return
        if not self._ensure_library():
            return
        self._bump_generation()
        self._preflight_is_standard = request is None
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="preflight",
            library_spec=self._library_spec,
            # request=None selects the auto-route resolver in the worker; a
            # non-None request selects the explicit Advanced path.
            request=request, policy=self._policy(),
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
            storage_paths=self._storage_paths_dict(),
            bpm_root=self._bpm_root,
        )
        self._start_operation(snapshot)

    def _on_export(self) -> None:
        # Standard one-step: the single scientific action. No manual Verify/
        # preflight step — the auto-route is resolved internally when stale.
        if not self._lights:
            QtWidgets.QMessageBox.information(self, "No images", "Add at least one image to calibrate.")
            return
        if self._library_spec is None:
            reason = self._first_attention_note() if self._master_attention_notes else "add calibration masters first."
            self._set_standard_human_status("attention", reason)
            self.status_label.setText("Calibration not ready.")
            QtWidgets.QMessageBox.warning(self, "Calibration not ready", reason)
            return
        if self._standard_route_generation != self._generation:
            # Auto-route not yet resolved (or stale): run it internally first,
            # then continue automatically if at least one light is READY.
            self._export_after_preflight = True
            self._dispatch_preflight(None)
            return
        # Auto-route already resolved for the current configuration.
        if not self._standard_route_ready():
            self._set_standard_human_status("attention", self._first_blocking_human_reason())
            self.status_label.setText("Calibration not ready.")
            return
        self._start_standard_export()

    def _on_advanced_export(self) -> None:
        # Advanced: explicit request from the additive/flat combos.
        if not self._lights:
            QtWidgets.QMessageBox.information(self, "No lights", "Add at least one light frame.")
            return
        if not self._ensure_library():
            return
        self._prompt_destination_and_export(self._request())

    def _standard_route_ready(self) -> bool:
        """True when the current auto-route resolved at least one READY light."""
        return any(
            s.get("outcome") == "MATCHED" and s.get("auto_route")
            for s in self._preflight_summaries
        )

    def _first_blocking_human_reason(self) -> str:
        for summary in self._preflight_summaries:
            if summary.get("outcome") != "MATCHED":
                return presentation.human_reason_text(summary)
        return "no compatible calibration set was found."

    def _continue_export_after_preflight(self) -> None:
        if self._standard_route_ready():
            self._start_standard_export()
        else:
            self._set_standard_human_status("attention", self._first_blocking_human_reason())
            self.status_label.setText("Calibration not ready.")

    def _standard_output_destination(self):
        """Validate the persisted Standard output folder; return (destination, error).

        UX preflight only: existence + writability are checked so the user gets a
        human message BEFORE calibration starts. Writability is NOT a durable
        guarantee and no reservation/locking machinery is built — the writer
        remains the final authority for real write failures. No directory is ever
        auto-created here.
        """
        value = self.output_dir_edit.text().strip()
        if not value:
            return None, "Please select an output folder before starting calibration."
        if not os.path.isdir(value):
            return None, (
                "The selected output folder no longer exists. "
                "Please choose a valid destination."
            )
        if not os.access(value, os.W_OK):
            return None, (
                "The selected output folder is not writable. "
                "Please choose a valid destination."
            )
        return str(value), None

    def _start_standard_export(self) -> None:
        destination, error = self._standard_output_destination()
        if error is not None:
            QtWidgets.QMessageBox.warning(self, "Output folder", error)
            return
        self._check_bpm_proposal_then_export(None, destination)

    def _prompt_destination_and_export(self, request) -> None:
        # Advanced path only (kept unchanged): the modal destination chooser.
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
        self._check_bpm_proposal_then_export(request, destination)

    def _run_light_constraints(self):
        """The run's light constraints from the first resolved plan (or None)."""
        for plan in self._plans.values():
            if plan is not None:
                return plan.light_constraints
        return None

    def _run_dark_available(self) -> bool:
        """True when any resolved plan binds a (compatible) dark master."""
        return any(
            plan is not None and "dark" in plan.masters
            for plan in self._plans.values()
        )

    def _run_dark_plan(self):
        """The first resolved plan that binds a dark master (or None)."""
        for plan in self._plans.values():
            if plan is not None and "dark" in plan.masters:
                return plan
        return None

    def _check_bpm_proposal_then_export(self, request, destination) -> None:
        """§3/§4/§9: BPM decision gate at Calibrate/Export, then export.

        Uses the headless decisions (``_bpm.bpm_no_database_decision`` +
        ``_bpm.bpm_creation_decision``):

        * no explicit database configured → offer Create/Select/Not now once per
          run (§4);
        * a compatible map whose K differs from the current setting → offer
          Rebuild / Use existing / Cancel once per run (§3);
        * a compatible map with matching K → no question, used automatically
          (§6);
        * no compatible map + a selected Master Dark → one creation offer (§5);
        * no compatible map + no dark → a discreet note;
        * a corrupt base → a typed discreet note.

        Declining never raises an error and never repeats per light.
        """
        from zecalibrator.api.v1 import _bpm

        light = self._run_light_constraints()
        if light is None:
            self._start_export(request, destination)
            return
        storage = _bpm.storage_from_mapping(self._storage_paths_dict())
        settings = _bpm.bpm_settings(self._bpm_root, detector_k=self._bpm_detector_k)

        # §4: no explicit database configured → one prompt per run.
        if not _bpm.bpm_no_database_decision(settings)["configured"]:
            if not self._bpm_no_db_prompted:
                self._bpm_no_db_prompted = True
                self._offer_bpm_database_setup(request, destination)
                return
            self._start_export(request, destination)
            return

        try:
            decision = _bpm.bpm_creation_decision(
                storage, settings, light, dark_available=self._run_dark_available()
            )
        except Exception as exc:  # noqa: BLE001 - a failed check never blocks calibration
            self._log(f"[bpm] map check unavailable: {exc}")
            self._start_export(request, destination)
            return
        action = decision["action"]
        if action == "mismatch":
            # §3: cache an explicit per-run decision, not just "prompt seen".
            cached = self._bpm_mismatch_decision
            if cached is None:
                # First time this run: ask once, then remember the outcome.
                self._offer_bpm_mismatch(request, destination, decision)
                return
            if cached == "cancel":
                # Cancel is sticky for this unchanged run: never launch and never
                # silently fall through to "Use existing".
                self.status_label.setText("Calibration cancelled.")
                return
            if cached == "use_existing":
                # Explicit use-existing: later exports reuse it without re-prompting.
                self.status_label.setText("Using the existing Bad Pixel Map.")
                self._start_export(request, destination)
                return
            # cached == "rebuild": the rebuild already happened; the new revision
            # now matches (or creation failed and was reported). Proceed rather
            # than re-prompting for this run.
            self._start_export(request, destination)
            return
        if action == "propose" and not self._bpm_proposed:
            self._bpm_proposed = True
            self._offer_bpm_creation(request, destination)
            return
        if action == "unavailable":
            self._log("No compatible Bad Pixel Map is available for this camera.")
        elif action == "error":
            self._log(
                f"Bad Pixel Database is unavailable ({decision['reason_code']})."
            )
        self._start_export(request, destination)

    def _offer_bpm_database_setup(self, request, destination) -> None:
        """§4: no Bad Pixel Database configured — Create / Select / Not now."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Bad Pixel Database")
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setText("No Bad Pixel Database is configured for this camera.")
        create_btn = box.addButton(
            "Create new database", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        select_btn = box.addButton(
            "Select existing database", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton("Not now", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is create_btn:
            self._choose_bpm_database(request, destination, create=True)
        elif clicked is select_btn:
            self._choose_bpm_database(request, destination, create=False)
        else:
            # Not now: normal calibration without BPM; never re-asked this run.
            self.status_label.setText("Calibration continues without a Bad Pixel Database.")
            self._start_export(request, destination)

    def _choose_bpm_database(self, request, destination, *, create: bool) -> None:
        """§4 folder chooser: create/select + validate + persist, then continue.

        Cancelling the picker is calm and never launches a partly-configured BPM
        path: it falls back to normal calibration without BPM.
        """
        from zecalibrator.api.v1 import _bpm

        start = self._bpm_root or str(self._storage.user_data_path)
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Create Bad Pixel Database location" if create else "Select Bad Pixel Database",
            start,
        )
        if not folder:
            self.status_label.setText("Calibration continues without a Bad Pixel Database.")
            self._start_export(request, destination)
            return
        path, error = _bpm.ensure_bpm_root(folder, create=create)
        if error is not None:
            QtWidgets.QMessageBox.warning(self, "Bad Pixel Database location", error)
            return
        self._bpm_root = path
        self.bpm_root_edit.setText(path)
        self._refresh_bpm_status_label()
        # Persist first; continue the SAME normal BPM workflow only AFTER the
        # settings-save worker ends (serialized), never while it is still active.
        # This prevents map creation from being offered/accepted mid-save (which
        # would otherwise drop the create op on "an operation is already running").
        self._pending_bpm_workflow = (request, destination)
        self._pending_bpm_workflow_ok = False
        self._request_bpm_settings_save(path, detector_k=self._bpm_detector_k)

    def _offer_bpm_mismatch(self, request, destination, decision) -> None:
        """§3: compatible map K differs from the current setting — one decision."""
        map_k = decision.get("map_k")
        setting_k = decision.get("setting_k")

        def fmt(v):
            return f"{float(v):g}"

        has_dark = self._run_dark_available()
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Bad Pixel Map")
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setText(
            f"The compatible Bad Pixel Map was created with threshold {fmt(map_k)}.\n"
            f"The current threshold is {fmt(setting_k)}.\n"
            "Rebuild the Bad Pixel Map with the current threshold?"
        )
        rebuild_btn = None
        if has_dark:
            rebuild_btn = box.addButton(
                "Rebuild", QtWidgets.QMessageBox.ButtonRole.AcceptRole
            )
        use_btn = box.addButton(
            "Use existing map", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is rebuild_btn and has_dark:
            # Rebuild: create a NEW immutable revision at the current K, promote
            # it; the old revision stays intact/addressable. Then export.
            self._bpm_mismatch_decision = "rebuild"
            self._request_bpm_map_creation(request, destination)
            return
        if clicked is use_btn:
            # Use existing: keep that exact revision and its recorded K; never
            # relabel as the current K. Export proceeds with the existing map,
            # and later exports in this run reuse it without re-prompting.
            self._bpm_mismatch_decision = "use_existing"
            self.status_label.setText("Using the existing Bad Pixel Map.")
            self._start_export(request, destination)
            return
        # Cancel (or no rebuild available, or dialog closed): do NOT launch the
        # run, write no outputs/revision. The decision is cached as "cancel" so
        # later exports in this unchanged run never silently use the old map.
        self._bpm_mismatch_decision = "cancel"
        self.status_label.setText("Calibration cancelled.")

    def _offer_bpm_creation(self, request, destination) -> None:
        """The single "Create Bad Pixel Map" dialog (case C)."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Bad Pixel Map")
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setText(
            "No compatible Bad Pixel Map was found for this camera.\n"
            "Create one from the selected Master Dark?"
        )
        create_btn = box.addButton(
            "Create Bad Pixel Map", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton("Not now", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is create_btn:
            self._request_bpm_map_creation(request, destination)
        else:
            # Declined: normal calibration, no error, never re-asked this run.
            self.status_label.setText("Calibration continues without a Bad Pixel Map.")
            self._start_export(request, destination)

    def _request_bpm_map_creation(self, request, destination) -> None:
        # Never set the pending-export continuation for a start that would be
        # dropped: ``_pending_bpm_export`` must only mean the map-creation
        # operation was ACTUALLY admitted.
        if self._controller.is_active:
            self._log("an operation is already running")
            self.status_label.setText(
                "Calibration not started (an operation is still running)."
            )
            return
        plan = self._run_dark_plan()
        if plan is None:
            self._start_export(request, destination)
            return
        self._pending_bpm_export = (request, destination)
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="create_bpm_map",
            library_spec=None, request=None, policy=None, lights=(),
            plan=plan, storage_paths=self._storage_paths_dict(),
            bpm_root=self._bpm_root,
            bpm_detector_k=self._bpm_detector_k,
        )
        self._start_operation(snapshot)

    def _handle_bpm_map_created(self, summary: dict) -> None:
        if summary.get("status") == "COMPLETED":
            self._log(
                f"Bad Pixel Map created ({summary.get('site_count', 0)} sites; "
                f"revision {summary.get('revision_id')})."
            )
            self.status_label.setText("Bad Pixel Map created.")
        else:
            self._log(
                f"Bad Pixel Map creation failed: {summary.get('reason_code')} "
                f"{summary.get('details')}"
            )
            self.status_label.setText("Bad Pixel Map creation failed.")
            # A failed create/rebuild must never be followed by an export, and the
            # per-run mismatch decision is cleared so a later export re-asks.
            self._pending_bpm_export = None
            self._bpm_mismatch_decision = None

    def _start_export(self, request, destination) -> None:
        selected = self._selected_rows()
        entries = [self._lights[i] for i in selected] if selected else list(self._lights)
        self._batch_items.clear()
        self.results_table.setRowCount(0)
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="export",
            library_spec=self._library_spec,
            # request=None selects the auto-route resolver in the worker; a
            # non-None request selects the explicit Advanced path.
            request=request, policy=self._policy(),
            lights=self._lights_snapshot(entries),
            destination=destination, batch_id=service.new_batch_id(),
            storage_paths=self._storage_paths_dict(),
            bpm_root=self._bpm_root,
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

    def _on_collision_query(self, op_id: str, payload: dict) -> None:
        """Show the ONE batch-level collision dialog and release the worker.

        Runs on the GUI thread; the worker is blocked on its own thread waiting
        on the channel. At most one such dialog appears per batch (the facade
        caches the policy after the first collision). The safe default is
        ``"cancel"`` (overwrite/skip are never selected implicitly).
        """
        if not self._is_current(op_id):
            # Stale/foreign collision query: never prompt for a dead batch;
            # release the worker with the safe default (cancel).
            self._controller.resolve_collision("cancel")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Output file already exists")
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setText(
            "An output file already exists in the selected folder.\n"
            "Choose how existing output files should be handled for this batch."
        )
        overwrite_btn = box.addButton(
            "Overwrite existing", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        skip_btn = box.addButton(
            "Skip existing", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn = box.addButton(
            "Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        box.exec()
        clicked = box.clickedButton()
        if clicked is overwrite_btn:
            decision = "overwrite"
        elif clicked is skip_btn:
            decision = "skip"
        else:
            decision = "cancel"
        self._controller.resolve_collision(decision)

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
        elif kind == "save_bpm_settings":
            self._handle_bpm_settings_saved(summary)
        elif kind == "scan_masters":
            self._handle_scan_masters(summary)
        elif kind == "confirm_evidence":
            self._handle_confirm_evidence(summary)
        elif kind == "build_managed_library":
            self._handle_build_managed(summary)
        elif kind == "create_bpm_map":
            self._handle_bpm_map_created(summary)
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
        else:
            self.library_status.setText(
                f"Library open failed: {summary.get('reason_code')} {summary.get('details')}"
            )
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
        else:
            self.library_status.setText(
                f"Indexing {summary.get('status')}: {summary.get('reason_code')} {summary.get('details')}"
            )
            self._log(f"indexing FAILED [{summary.get('reason_code')}]: {summary.get('details')}")

    def _handle_preflight_finished(self, summary: dict) -> None:
        if self._preflight_is_standard:
            # Record the generation for which the Standard auto-route was resolved
            # (used by the one-step export to decide stale-vs-valid).
            self._standard_route_generation = self._generation
            # R3D-G: after route resolution, the Standard status reflects the
            # bound plan roles only — never an unused supplied master.
            if self._standard_route_ready():
                roles = self._bound_plan_roles()
                detail = self._human_role_list(roles) if roles else "masters"
                self._set_standard_human_status("ready", detail)
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
        bpm_status = summary.get("bpm_status")
        if bpm_status:
            self._log(f"Bad Pixel Database status: {bpm_status}")
        self._render_in_memory_details(summary)
        self._audit_show(summary.get("provenance_audit"))

    def _handle_export_finished(self, summary: dict) -> None:
        self._batch_displays = list(summary.get("input_displays", []))
        self._batch_total_inputs = int(summary.get("total_inputs", 0))
        status = summary.get("status")
        # Truthful counters (F6).
        items = summary.get("items", [])
        committed = sum(1 for i in items if i["disposition"] in ("COMPLETED", "COMPLETED_WITH_WARNINGS"))
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
        bpm_status = summary.get("bpm_status")
        if bpm_status:
            self._log(f"Bad Pixel Database status: {bpm_status}")
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
        if summary.get("status") == "FAILED":
            self.status_label.setText("Master detection failed.")
            self._auto_flow_active = False
            return
        self._managed_scan = list(summary.get("masters", []))
        self._present_managed_scan()
        if self._auto_flow_active:
            self._auto_confirm()

    def _handle_confirm_evidence(self, summary: dict) -> None:
        if summary.get("status") == "PRESERVED":
            self._log(f"[managed] ledger preserved ({summary.get('state')}); not overwritten.")
            self.status_label.setText("Managed ledger preserved (not overwritten).")
            self._confirm_queue.clear()
            self._confirm_batch_active = False
            return
        record = summary.get("record") or {}
        sha = record.get("content_sha256")
        role = record.get("role")
        if sha and role:
            key = (sha, role)
            if key not in self._session_selection:
                self._session_selection.append(key)
        ev_status = summary.get("evidence_status")
        if summary.get("status") == "REUSED":
            self._log(f"[managed] known master reused (unchanged): {summary.get('path')}")
            self.status_label.setText("Master reused (unchanged).")
            self._confirmed_ready.append(ev_status == "ready")
            return
        missing = summary.get("missing", [])
        if ev_status == "needs_attention":
            self._log(
                f"[managed] confirmed with insufficient evidence — missing/insufficient: "
                f"{', '.join(missing) or '(none)'}"
            )
            self.status_label.setText("Master confirmed (needs attention — insufficient evidence).")
            self._confirmed_ready.append(False)
        else:
            self.status_label.setText("Master evidence confirmed.")
            self._confirmed_ready.append(True)
        # No dispatch here: the pending confirm queue is drained in _on_worker_ended
        # (after the controller clears is_active), never re-entrantly while active.

    def _handle_build_managed(self, summary: dict) -> None:
        if summary.get("status") == "PRESERVED":
            self.status_label.setText("Managed ledger preserved (not overwritten).")
            self._log(f"[managed] ledger preserved ({summary.get('state')}).")
            return
        attention = summary.get("needs_attention", [])
        candidate_count = summary.get("candidate_count", 0)
        status = summary.get("status")
        if status in ("REUSED", "COMPLETED"):
            for a in attention:
                self._log(
                    f"[managed] informational — {a.get('role')} {a.get('path')}: "
                    f"insufficient evidence ({', '.join(a.get('missing', ())) or '(none)'})"
                )
            if candidate_count == 0:
                # Nothing was indexable: never Ready; refuse Verify/export with a
                # concrete human reason (never "no usable masters"/"candidate_count=0").
                self._library_spec = None
                self._set_standard_human_status("attention", self._first_attention_note())
                self.status_label.setText("Calibration not ready.")
            else:
                # Wire the exact managed LibrarySpec so _ensure_library()/export use
                # it; a ready managed state never coexists with an empty library.
                self._library_spec = self._managed_spec
                # R3D-G: the bound plan roles are only known after route
                # resolution. Never list the full supplied session-role set here
                # (it may include a master the resolved plan does not consume,
                # e.g. an unused DARKFLAT).
                self._set_standard_human_status("ready")
                self.status_label.setText("Masters ready.")
        else:
            self._library_spec = None
            self._set_standard_human_status("attention", "calibration masters could not be prepared")
            self.status_label.setText(
                f"Managed library {status}: {summary.get('reason_code')} {summary.get('details')}"
            )
            self._log(f"[managed] build FAILED [{summary.get('reason_code')}]: {summary.get('details')}")

    def _on_operation_failed(self, op_id: str, reason_code: str, details: str) -> None:
        if not self._is_current(op_id):
            return
        self._terminal_seen = True
        self.status_label.setText(f"Failed: {reason_code}")
        self._log(f"operation {op_id[:12]} FAILED [{reason_code}]: {details}")
        # A failed operation must never leave a pending BPM continuation that
        # later fires as if it succeeded.
        self._pending_bpm_export = None
        self._pending_bpm_workflow = None
        self._pending_bpm_workflow_ok = False
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

    def _on_worker_ended(self) -> None:
        self._set_active(False)
        if self._close_requested:
            self.close()
            return
        if self._confirm_batch_active:
            self._dispatch_next_confirm()
        elif self._export_after_preflight:
            self._export_after_preflight = False
            self._continue_export_after_preflight()
        elif self._pending_bpm_workflow is not None:
            # The settings save just ended: continue the SAME BPM decision
            # workflow only if persistence succeeded. A failed/preserved save
            # must never enter a partly-configured BPM workflow or silently
            # export as if persistence had succeeded.
            request, destination = self._pending_bpm_workflow
            self._pending_bpm_workflow = None
            if self._pending_bpm_workflow_ok:
                self._pending_bpm_workflow_ok = False
                self._check_bpm_proposal_then_export(request, destination)
            else:
                self.status_label.setText("Could not save Bad Pixel Database settings.")
                self._log("[bpm] Bad Pixel Database settings save failed; calibration not started.")
        elif self._pending_bpm_export is not None:
            # After the map-creation op ended, resume the pending export (the
            # freshly created map is now found automatically).
            request, destination = self._pending_bpm_export
            self._pending_bpm_export = None
            self._start_export(request, destination)

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
