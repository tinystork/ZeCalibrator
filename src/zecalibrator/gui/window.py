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

import json
import uuid
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.gui import identity, presentation, service
from zecalibrator.gui.settings import (
    STATE_MALFORMED,
    STATE_UNSUPPORTED,
    GuiSettings,
    default_settings,
)
from zecalibrator.gui.worker import WorkerController
from zecalibrator.storage import StoragePaths

_SYNTH_ONLY_LABEL = "Qualification: SYNTH-BASE-1 synthetic-only. No real-camera claim."


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

        self._build_ui()
        self._wire_controller()
        self._apply_icon()
        self.resize(self._settings.window_width, self._settings.window_height)
        self._set_active(False)

        # Production exit safety: request non-blocking worker quit at app exit.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._controller.shutdown)

        # Asynchronous settings load (off the GUI thread).
        self._start_settings_load()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.setWindowTitle("ZeCalibrator")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        # -- Inputs -------------------------------------------------------
        inputs_box = QtWidgets.QGroupBox("Lights")
        inputs_layout = QtWidgets.QVBoxLayout(inputs_box)
        self.lights_list = QtWidgets.QListWidget()
        self.lights_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        inputs_layout.addWidget(self.lights_list)
        light_btn_row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add FITS…")
        self.remove_btn = QtWidgets.QPushButton("Remove selected")
        light_btn_row.addWidget(self.add_btn)
        light_btn_row.addWidget(self.remove_btn)
        light_btn_row.addStretch(1)
        inputs_layout.addLayout(light_btn_row)

        hdu_row = QtWidgets.QHBoxLayout()
        hdu_row.addWidget(QtWidgets.QLabel("HDU for selected:"))
        self.hdu_edit = QtWidgets.QLineEdit("0")
        self.hdu_edit.setFixedWidth(80)
        hdu_row.addWidget(self.hdu_edit)
        self.apply_hdu_btn = QtWidgets.QPushButton("Apply HDU")
        hdu_row.addWidget(self.apply_hdu_btn)
        hdu_row.addStretch(1)
        inputs_layout.addLayout(hdu_row)

        ev_row = QtWidgets.QHBoxLayout()
        self.load_decl_btn = QtWidgets.QPushButton("Load declaration JSON…")
        self.load_roi_btn = QtWidgets.QPushButton("Load ROI JSON…")
        ev_row.addWidget(self.load_decl_btn)
        ev_row.addWidget(self.load_roi_btn)
        ev_row.addStretch(1)
        inputs_layout.addLayout(ev_row)
        self.evidence_label = QtWidgets.QLabel("Evidence applies to the currently selected light(s).")
        inputs_layout.addWidget(self.evidence_label)
        layout.addWidget(inputs_box)

        # -- Library ------------------------------------------------------
        library_box = QtWidgets.QGroupBox("Library")
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

        # -- Modes --------------------------------------------------------
        modes_box = QtWidgets.QGroupBox("Calibration modes")
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
        layout.addWidget(modes_box)
        self._update_mode_note()

        # -- Qualification label (G4) -------------------------------------
        self.qualification_label = QtWidgets.QLabel(_SYNTH_ONLY_LABEL)
        self.qualification_label.setStyleSheet("color: gray;")
        layout.addWidget(self.qualification_label)

        # -- Actions / progress -------------------------------------------
        actions_row = QtWidgets.QHBoxLayout()
        self.preflight_btn = QtWidgets.QPushButton("Preflight (inspect & match)")
        self.calibrate_btn = QtWidgets.QPushButton("Calibrate selected in memory")
        self.export_btn = QtWidgets.QPushButton("Export…")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        actions_row.addWidget(self.preflight_btn)
        actions_row.addWidget(self.calibrate_btn)
        actions_row.addWidget(self.export_btn)
        actions_row.addWidget(self.cancel_btn)
        layout.addLayout(actions_row)
        self.scope_label = QtWidgets.QLabel("Export scope: all")
        layout.addWidget(self.scope_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel("Ready.")
        layout.addWidget(self.status_label)

        # -- Results tabs --------------------------------------------------
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

        # Details (concise human summary) — plain text, never HTML-interpreted.
        self.details_view = QtWidgets.QPlainTextEdit()
        self.details_view.setReadOnly(True)
        self.tabs.addTab(self.details_view, "Details")

        # Full read-only raw audit pane (serialized public plan/decision/provenance).
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
        self.audit_link_btn.clicked.connect(self._on_open_manifest)
        audit_layout.addWidget(self.audit_link_btn)
        self.tabs.addTab(audit_widget, "Audit")

        layout.addWidget(self.tabs)

        self.setCentralWidget(central)

        # -- Connect control slots -----------------------------------------
        self.add_btn.clicked.connect(self._on_add_lights)
        self.remove_btn.clicked.connect(self._on_remove_lights)
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

        self.additive_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.flat_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.lights_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.preflight_table.currentCellChanged.connect(self._on_preflight_selection_changed)
        self.results_table.currentCellChanged.connect(self._on_results_selection_changed)

        self._config_widgets = [
            self.add_btn, self.remove_btn, self.apply_hdu_btn, self.hdu_edit,
            self.load_decl_btn, self.load_roi_btn,
            self.library_index_edit, self.open_library_btn,
            self.library_root_edit, self.imports_edit, self.imports_browse_btn,
            self.root_browse_btn, self.index_btn,
            self.additive_combo, self.flat_combo,
        ]
        self._launch_widgets = [self.preflight_btn, self.calibrate_btn, self.export_btn]
        self._update_scope_label()

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

    def _request(self) -> v1.CalibrationRequest:
        return v1.CalibrationRequest(self.additive_combo.currentData(), self.flat_combo.currentData())

    def _policy(self) -> v1.MatchPolicy:
        return v1.default_match_policy()

    def _update_mode_note(self) -> None:
        additive = self.additive_combo.currentData()
        flat = self.flat_combo.currentData()
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
            self.scope_label.setText(f"Export scope: all ({total})")

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
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def _start_operation(self, snapshot: service.OperationSnapshot) -> None:
        if self._controller.is_active:
            self._log("an operation is already running")
            return
        self._token = v1.CancellationToken()
        self._current_op_id = snapshot.op_id
        self._active_generation = self._generation
        self._terminal_seen = False
        self._set_active(True)
        self._controller.start(snapshot, self._token)

    def _is_current(self, op_id: str) -> bool:
        return op_id == self._current_op_id and self._generation == self._active_generation

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
        self._settings = GuiSettings(
            last_input_dir=self._settings.last_input_dir,
            last_library_dir=self._settings.last_library_dir,
            last_output_dir=self._settings.last_output_dir,
            window_width=self.width(), window_height=self.height(),
            extra=self._settings.extra,
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

    # -------------------------------------------------------------- slots
    def _on_add_lights(self) -> None:
        start_dir = self._settings.last_input_dir or ""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select FITS light frames", start_dir,
            "FITS files (*.fits *.fit *.fts);;All files (*)",
        )
        if not paths:
            return
        hdu = service.parse_hdu(self.hdu_edit.text())
        for path in paths:
            self._lights.append(_LightEntry(path, hdu=hdu))
        self._settings = GuiSettings(
            last_input_dir=str(Path(paths[0]).parent),
            last_library_dir=self._settings.last_library_dir,
            last_output_dir=self._settings.last_output_dir,
            window_width=self.width(), window_height=self.height(),
            extra=self._settings.extra,
        )
        self._refresh_lights_list()
        self._bump_generation()
        self._update_scope_label()

    def _on_remove_lights(self) -> None:
        for index in sorted(self._selected_rows(), reverse=True):
            if index < len(self._lights):
                del self._lights[index]
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
        self._bump_generation()
        snapshot = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="index_library",
            library_spec=self._library_spec, request=None, policy=None, lights=(),
            imports_path=imports_path,
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
        self._settings = GuiSettings(
            last_input_dir=self._settings.last_input_dir,
            last_library_dir=self._settings.last_library_dir,
            last_output_dir=destination,
            window_width=self.width(), window_height=self.height(),
            extra=self._settings.extra,
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
        if self._token is not None:
            self._token.cancel()
        self.status_label.setText("Cancellation requested…")

    # ------------------------------------------------- worker signal handlers
    def _on_operation_started(self, op_id: str) -> None:
        if not self._is_current(op_id):
            return
        self.status_label.setText("Working…")

    def _on_progress(self, op_id: str, event) -> None:
        if not self._is_current(op_id):
            return
        if self._terminal_seen:
            return  # terminal result is authoritative; never overwrite it
        total = event.total
        if total:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(event.completed, total))
        else:
            self.progress_bar.setRange(0, 0)
        frame = f" — {event.frame_id}" if getattr(event, "frame_id", None) else ""
        self.status_label.setText(f"{event.phase} ({event.completed}/{event.total}){frame}")

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
            self.library_status.setText(
                f"Indexed: revision={summary.get('revision')!r} "
                f"candidates={summary.get('candidate_count')} diagnostics={len(summary.get('diagnostics', ()))}"
            )
            for diag in summary.get("diagnostics", ()):
                self._log(f"index diagnostic: {diag}")
        else:
            self.library_status.setText(
                f"Indexing {summary.get('status')}: {summary.get('reason_code')} {summary.get('details')}"
            )
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
