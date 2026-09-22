"""R3D-D Standard one-step workflow — owner-mandated acceptance tests.

Twelve witnesses for "ADMISSION != COMPATIBILITY" and the Standard one-step
workflow:

1.  Siril-like dark missing gain/offset/temp is INDEXED in a Standard session.
2.  Siril-like flat without R4 evidence is INDEXED; R4 absence reaches the
    matcher as UNVERIFIED.
3.  Bayer master missing orientation/roi_origin is INDEXED and reaches the
    Standard matcher.
4.  A fresh session with 3 admissible masters -> candidate_count == 3.
5.  A REUSED ledger with the same 3 masters -> candidate_count == 3.
6.  Standard add-folder automatically: scan -> record -> build/reuse -> route
    resolution.
7.  No Standard buttons: Detect masters / Confirm detected facts / Build managed
    library / Verify calibration.
8.  No technical modal questionnaire.
9.  Calibrate / Export does not require manual preflight.
10. A known incompatibility still blocks at route level with its real reason.
11. The Advanced workflow is unchanged.
12. queue.Queue / GC / Qt lifetime / cancellation invariants are preserved.
"""

from __future__ import annotations

import gc
import queue as _queue
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

import zecalibrator.api.v1 as v1
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.descriptors import (
    Acquisition,
    DetectorIdentity,
    LightConstraints,
    OpticalIdentity,
)
from zecalibrator.core.geometry import Geometry
from zecalibrator.core.routes import OUTCOME_NEEDS_ATTENTION, OUTCOME_READY
from zecalibrator.gui.window import MainWindow
from zecalibrator.storage import resolve_paths

from .conftest import SHAPE, make_synth_fixture, wait_idle, write_fits_array


@pytest.fixture
def paths(tmp_path):
    return resolve_paths(base=str(tmp_path))


def _pump(cond, timeout_ms=20000):
    from PySide6 import QtTest

    start = time.monotonic()
    while time.monotonic() - start < timeout_ms / 1000.0:
        QtWidgets.QApplication.processEvents()
        if cond():
            return True
        QtTest.QTest.qWait(5)
    return False


def _shutdown(w):
    w._controller.shutdown()
    _pump(lambda: w._controller.is_finished)


def _settled(w):
    return (
        not w._controller.is_active
        and not w._confirm_batch_active
        and w._auto_flow_active is False
    )


def _write_master(path, *, imagetyp, cards=()):
    data = np.full(SHAPE, 1.0, dtype=np.float32)
    return write_fits_array(
        path, data, header_cards=[("IMAGETYP", imagetyp), *cards], bunit="ADU",
    )


def _mock_get_open_file_names(paths):
    QtWidgets.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (list(paths), ""))


def _managed_light(cfa="mono", exposure_s=10.0, gain=120.0, offset=50.0, temperature_c=20.0):
    return LightConstraints(
        geometry=Geometry(
            shape=SHAPE, sensor_dimensions=None, binning=(1, 1),
            roi_origin=(0, 0), roi_extent=SHAPE, orientation="identity", cfa_phase=cfa,
        ),
        detector=DetectorIdentity(detector_instance_id="unknown", detector_model="SYNTH-CFA"),
        acquisition=Acquisition(
            gain=gain, offset=offset, temperature_c=temperature_c, exposure_s=exposure_s,
            saturation_limit_adu=60000.0, saturation_evidence="qualified",
        ),
        optical=OpticalIdentity(filter="NONE", optical_train_id=None),
    )


def _open_candidates(spec):
    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        return handle.snapshot.candidates
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# 1. Siril-like dark missing gain/offset/temp is indexed (never dropped).
# ---------------------------------------------------------------------------
def test_siril_dark_missing_acquisition_indexed(qapp, paths, tmp_path, monkeypatch):
    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        dark = _write_master(
            tmp_path / "siril_dark.fits", imagetyp="DARK",
            cards=[
                ("EXPTIME", 10.0), ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "mono"),
                ("XBINNING", 1), ("YBINNING", 1),
            ],
        )
        _mock_get_open_file_names([dark])
        w._on_add_masters_file()
        assert _pump(lambda: _settled(w))
        assert w._library_spec is not None, "incomplete-but-admissible dark must build"

        candidates = _open_candidates(w._library_spec)
        assert len(candidates.get("dark", ())) == 1
        acq = candidates["dark"][0].descriptor.acquisition
        # Unknown metadata is preserved (UNKNOWN), never dropped or invented.
        assert acq.gain is None
        assert acq.offset is None
        assert acq.temperature_c is None
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# 2. Siril-like flat without R4 evidence is indexed; R4 absence -> UNVERIFIED.
# ---------------------------------------------------------------------------
def test_siril_flat_without_r4_evidence_indexed_and_unverified(qapp, paths, tmp_path, monkeypatch):
    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        flat = _write_master(
            tmp_path / "siril_flat.fits", imagetyp="FLAT",
            cards=[
                ("EXPTIME", 1.0), ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "mono"),
                ("GAIN", 120.0), ("OFFSET", 50.0), ("FILTER", "NONE"),
                ("XBINNING", 1), ("YBINNING", 1),
            ],
        )
        _mock_get_open_file_names([flat])
        w._on_add_masters_file()
        assert _pump(lambda: _settled(w))
        assert w._library_spec is not None

        candidates = _open_candidates(w._library_spec)
        flat_cands = candidates.get("flat", ())
        assert len(flat_cands) == 1
        desc = flat_cands[0].descriptor
        # R4 quality evidence is absent -> saturation limit unknown (never invented).
        assert desc.validity_evidence.saturation_limit_known is False

        resolution = resolve_route(
            _managed_light(), _open_candidates_snapshot(w._library_spec), v1.default_match_policy()
        )
        # The flat reaches the matcher and its absent R4 facts are UNVERIFIED.
        fields = {r.field for r in resolution.unverified}
        assert "validity_evidence.saturation_limit_known" in fields
        assert "validity_evidence.illumination" in fields
        assert "validity_evidence.exposure_quality" in fields
        assert resolution.routes and resolution.routes[0].masters.get("flat") is not None
    finally:
        _shutdown(w)


def _open_candidates_snapshot(spec):
    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    return opened.handle.snapshot


# ---------------------------------------------------------------------------
# 3. Bayer master missing orientation/roi_origin is indexed and reaches matcher.
# ---------------------------------------------------------------------------
def test_bayer_dark_missing_geometry_indexed_and_reaches_matcher(qapp, paths, tmp_path, monkeypatch):
    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        dark = _write_master(
            tmp_path / "bayer_dark.fits", imagetyp="DARK",
            cards=[
                ("EXPTIME", 10.0), ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "RGGB"),
                ("GAIN", 120.0), ("OFFSET", 50.0), ("CCD-TEMP", 20.0),
                ("XBINNING", 1), ("YBINNING", 1),
            ],
        )
        _mock_get_open_file_names([dark])
        w._on_add_masters_file()
        assert _pump(lambda: _settled(w))
        assert w._library_spec is not None

        candidates = _open_candidates(w._library_spec)
        dark_cands = candidates.get("dark", ())
        assert len(dark_cands) == 1
        desc = dark_cands[0].descriptor
        assert desc.geometry.orientation is None
        assert desc.geometry.roi_origin is None

        resolution = resolve_route(
            _managed_light(cfa="RGGB"), _open_candidates_snapshot(w._library_spec),
            v1.default_match_policy(),
        )
        assert resolution.outcome == OUTCOME_READY
        fields = {r.field for r in resolution.unverified}
        assert "geometry.orientation" in fields
        assert "geometry.roi_origin" in fields
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# 4 & 5. 3 admissible masters -> candidate_count == 3 (fresh and reused).
# ---------------------------------------------------------------------------
def _snapshot(kind, **kw):
    from zecalibrator.gui import service

    return service.OperationSnapshot(
        op_id=service.new_operation_id(), kind=kind,
        library_spec=None, request=None, policy=None, lights=(), **kw,
    )


def _confirm_dark(controller, ledger_dir, path):
    from .conftest import run_operation

    payload = {
        "role": "dark", "path": path, "hdu": 0,
        "evidence": {
            "exposure_s": {"value": 300.0, "origin_type": "fits_header", "origin_field": "EXPTIME"},
            "detector_model": {"value": "SYNTH-CFA", "origin_type": "fits_header", "origin_field": "INSTRUME"},
            "cfa_phase": {"value": "mono", "origin_type": "fits_header", "origin_field": "BAYERPAT"},
            "binning": {"value": (1, 1), "origin_type": "fits_header", "origin_field": "XBINNING"},
        },
        "extra": {},
        "dq_state": "no_source_dq", "mask_path": None, "bias_state": "included",
        "declaration_source": "user", "declaration_identity": "managed-session",
        "declaration_version": "1",
    }
    snap = _snapshot("confirm_evidence", ledger_dir=ledger_dir, confirm_payload=payload)
    r = run_operation(controller, snap, v1.CancellationToken())
    assert r.finished_summary()["status"] in ("COMPLETED", "REUSED")


def _three_incomplete_darks(tmp_path):
    from .conftest import write_fits_array

    paths = []
    for i in (10.0, 11.0, 12.0):
        p = tmp_path / f"dark_{int(i)}.fits"
        write_fits_array(
            p, np.full(SHAPE, i, dtype=np.float32),
            header_cards=[
                ("IMAGETYP", "DARK"), ("EXPTIME", 300.0), ("INSTRUME", "SYNTH-CFA"),
                ("BAYERPAT", "mono"), ("XBINNING", 1), ("YBINNING", 1),
            ],
            bunit="ADU",
        )
        paths.append(str(p))
    return paths


def _build_and_count(controller, ledger_dir, cache_dir, selection, expect_reused=False):
    from .conftest import run_operation

    spec = v1.LibrarySpec(root=str(cache_dir), index_path=str(cache_dir / "managed.sqlite"))
    snap = _snapshot(
        "build_managed_library", ledger_dir=ledger_dir, managed_spec=spec,
        session_selection=tuple(selection),
    )
    r = run_operation(controller, snap, v1.CancellationToken())
    s = r.finished_summary()
    assert s["status"] in ("COMPLETED", "REUSED")
    if expect_reused:
        assert s["status"] == "REUSED"
    assert s["candidate_count"] == 3
    return s


def test_fresh_session_three_masters_candidate_count_three(controller, tmp_path):
    darks = _three_incomplete_darks(tmp_path)
    ledger_dir = str(tmp_path / "data")
    cache_dir = tmp_path / "cache"
    for d in darks:
        _confirm_dark(controller, ledger_dir, d)
    selection = [(v1.content_identity(d, hdu=0)[0], "dark") for d in darks]
    _build_and_count(controller, ledger_dir, cache_dir, selection)


def test_reused_ledger_three_masters_candidate_count_three(controller, tmp_path):
    darks = _three_incomplete_darks(tmp_path)
    ledger_dir = str(tmp_path / "data")
    cache_dir = tmp_path / "cache"
    for d in darks:
        _confirm_dark(controller, ledger_dir, d)
    selection = [(v1.content_identity(d, hdu=0)[0], "dark") for d in darks]
    _build_and_count(controller, ledger_dir, cache_dir, selection)
    # Same session selection + unchanged ledger -> REUSED, same count.
    _build_and_count(controller, ledger_dir, cache_dir, selection, expect_reused=True)


# ---------------------------------------------------------------------------
# 6. Standard add-folder: scan -> record -> build/reuse -> route resolution.
# ---------------------------------------------------------------------------
def test_standard_add_folder_auto_flow_to_route_resolution(qapp, paths, tmp_path, monkeypatch):
    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        folder = tmp_path / "masters"
        folder.mkdir()
        _write_master(folder / "dark.fits", imagetyp="DARK", cards=[
            ("EXPTIME", 10.0), ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "mono"),
            ("GAIN", 120.0), ("OFFSET", 50.0), ("CCD-TEMP", 20.0),
            ("XBINNING", 1), ("YBINNING", 1),
        ])
        _write_master(folder / "flat.fits", imagetyp="FLAT", cards=[
            ("EXPTIME", 1.0), ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "mono"),
            ("GAIN", 120.0), ("OFFSET", 50.0), ("FILTER", "NONE"),
            ("XBINNING", 1), ("YBINNING", 1),
        ])
        _write_master(folder / "flat_dark.fits", imagetyp="DARKFLAT", cards=[
            ("EXPTIME", 1.0), ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "mono"),
            ("GAIN", 120.0), ("OFFSET", 50.0), ("CCD-TEMP", 20.0),
            ("XBINNING", 1), ("YBINNING", 1),
        ])

        QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(folder))
        w._on_add_masters_folder()
        assert _pump(lambda: _settled(w))

        # Ledger has the three admitted records (scan -> record).
        ledger = v1.load_managed_ledger(v1.managed_ledger_path(str(paths.user_data_path)))
        assert ledger.state == "ok"
        assert len(ledger.records) == 3
        assert {r.role for r in ledger.records} == {"dark", "flat", "flat_dark"}

        # Derived session index is built (build/reuse) and routes resolve.
        assert w._library_spec is not None
        candidates = _open_candidates(w._library_spec)
        assert sum(len(cs) for cs in candidates.values()) == 3

        resolution = resolve_route(
            _managed_light(), _open_candidates_snapshot(w._library_spec), v1.default_match_policy(),
        )
        assert resolution.outcome == OUTCOME_READY
        assert "dark" in resolution.plan.masters
        assert "flat" in resolution.plan.masters
    finally:
        _shutdown(w)


def test_masters_folder_subfolders_checkbox_wiring(qapp, paths, tmp_path, monkeypatch):
    """B4 — the "Include subfolders" checkbox is unchecked by default, carries
    the exact label, and drives ``scan_folder_inputs(recursive=...)`` for the
    masters folder action only (OFF -> False, ON -> True)."""
    from zecalibrator.gui import service as svc

    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        assert w.include_subfolders_check is not None
        assert w.include_subfolders_check.text() == "Include subfolders"
        assert w.include_subfolders_check.isChecked() is False

        folder = tmp_path / "masters"
        folder.mkdir()

        recursive_calls = []
        real_scan = svc.scan_folder_inputs

        def fake_scan(f, *, recursive=False):
            recursive_calls.append(recursive)
            return ([], 0)

        monkeypatch.setattr(svc, "scan_folder_inputs", fake_scan)
        QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(folder))

        # Default (unchecked) -> non-recursive.
        w._on_add_masters_folder()
        assert recursive_calls == [False]

        # Checked -> recursive.
        w.include_subfolders_check.setChecked(True)
        w._on_add_masters_folder()
        assert recursive_calls == [False, True]
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# 7. No Standard buttons for the removed technical steps.
# ---------------------------------------------------------------------------
def test_standard_has_no_technical_buttons(qapp, paths):
    w = MainWindow(paths)
    try:
        # Verify calibration is gone entirely; Detect/Confirm/Build live in Advanced.
        assert not hasattr(w, "preflight_btn")
        assert w.scan_masters_btn is not None
        assert w.confirm_masters_btn is not None
        assert w.build_managed_btn is not None
        # None of the technical buttons is a child of the Standard page (tab 0).
        standard_page = w.main_tabs.widget(0)
        standard_children = set(standard_page.findChildren(QtWidgets.QPushButton))
        for attr in ("scan_masters_btn", "confirm_masters_btn", "build_managed_btn"):
            assert getattr(w, attr) not in standard_children, attr
        # The Standard page exposes only the single scientific action.
        assert w.export_btn.text() == "Calibrate / Export…"
        assert w.export_btn in standard_children
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# 8. No technical modal questionnaire during the one-step auto-flow.
# ---------------------------------------------------------------------------
def test_auto_flow_never_shows_technical_questionnaire(qapp, paths, tmp_path, monkeypatch):
    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        # A Bayer dark missing CFA geometry (orientation/roi_origin) is the classic
        # "would-have-prompted" case; the one-step flow must index it silently.
        dark = _write_master(
            tmp_path / "bayer_dark.fits", imagetyp="DARK",
            cards=[
                ("EXPTIME", 10.0), ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "RGGB"),
                ("GAIN", 120.0), ("OFFSET", 50.0), ("CCD-TEMP", 20.0),
                ("XBINNING", 1), ("YBINNING", 1),
            ],
        )
        _mock_get_open_file_names([dark])

        prompts = []
        real_exec = QtWidgets.QDialog.exec
        monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: prompts.append("dialog"))
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getItem",
            staticmethod(lambda *a, **k: prompts.append("getItem") or ("dark", True)),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            staticmethod(lambda *a, **k: prompts.append("warning")),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "information",
            staticmethod(lambda *a, **k: prompts.append("information")),
        )

        w._on_add_masters_file()
        assert _pump(lambda: _settled(w))
        assert w._library_spec is not None
        assert prompts == [], f"unexpected modal prompts during one-step auto-flow: {prompts}"
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# 9. Calibrate / Export does not require manual preflight.
# ---------------------------------------------------------------------------
def test_export_does_not_require_manual_preflight(qapp, paths, tmp_path, monkeypatch):
    fixture = make_synth_fixture(tmp_path)
    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        from zecalibrator.gui.window import _LightEntry

        decl = v1.ImportDeclaration(**__import__("json").loads(open(fixture["decl"]).read()))
        roi = v1.RoiExtentEvidence(**__import__("json").loads(open(fixture["roi"]).read()))
        w._lights.append(_LightEntry(fixture["light"], hdu=0, declaration=decl, roi_extent=roi))
        w._refresh_lights_list()
        w._library_spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])

        out = tmp_path / "out"
        out.mkdir()
        # Standard export reads the visible Output-folder field (no modal chooser).
        w.output_dir_edit.setText(str(out))

        # No _on_preflight call: the one-step export resolves the route internally.
        w._on_export()
        assert _pump(lambda: not w._controller.is_active)
        assert "Export COMPLETED" in w.status_label.text()
        fits_files = [f for f in __import__("os").listdir(str(out)) if f.endswith(".fits")]
        assert len(fits_files) == 1
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# 10. A known incompatibility still blocks at route level with its real reason.
# ---------------------------------------------------------------------------
_SIRIL_BASE = dict(
    source="observed", identity="SIRIL-SESSION", version="1.0",
    domain="raw", units="ADU", detector_instance_id="SIRIL-DET-1",
    detector_model="SIRIL-CAM", binning=[1, 1], sensor_dimensions=[4, 4],
    orientation="identity", cfa_phase="mono", roi_origin=[0, 0],
    filter="NONE", optical_train_id="SIRIL-TRAIN-1",
    saturation_limit_adu=60000.0, saturation_evidence="qualified",
)


def _siril_light():
    return LightConstraints(
        geometry=Geometry(
            shape=SHAPE, sensor_dimensions=SHAPE, binning=(1, 1),
            roi_origin=(0, 0), roi_extent=SHAPE, orientation="identity", cfa_phase="mono",
        ),
        detector=DetectorIdentity(detector_instance_id="SIRIL-DET-1", detector_model="SIRIL-CAM"),
        acquisition=Acquisition(
            gain=120.0, offset=50.0, temperature_c=20.0, exposure_s=10.0,
            saturation_limit_adu=60000.0, saturation_evidence="qualified",
        ),
        optical=OpticalIdentity(filter="NONE", optical_train_id="SIRIL-TRAIN-1"),
    )


def test_known_incompatibility_blocks_at_route_level(tmp_path):
    data = np.full(SHAPE, 1.0, dtype=np.float32)
    write_fits_array(
        tmp_path / "dark.fits", data,
        header_cards=[("GAIN", 120), ("OFFSET", 50), ("CCD-TEMP", 20.0)],
    )
    write_fits_array(
        tmp_path / "flat.fits", data,
        header_cards=[("GAIN", 456), ("OFFSET", 50), ("CCD-TEMP", 20.0)],
    )

    imports = [
        v1.MasterImportSpec(
            path="dark.fits", master_type="dark", hdu=0,
            declaration=v1.ImportDeclaration(**_SIRIL_BASE, exposure_s=10.0),
            dq_state="no_source_dq",
        ),
        v1.MasterImportSpec(
            path="flat.fits", master_type="flat", hdu=0,
            declaration=v1.ImportDeclaration(
                **{
                    **_SIRIL_BASE, "gain": 456.0, "offset": 50.0,
                    "exposure_s": 10.0,
                    # R3D-E F4: a prepared flat's gain/offset no longer block;
                    # a CFA-phase contradiction is a genuine structural
                    # incompatibility that STILL blocks at route level.
                    "cfa_phase": "RGGB",
                }
            ),
            flat_form="corrected_unnormalized", dq_state="no_source_dq",
        ),
    ]
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "lib.sqlite"))
    idx = v1.index_library(spec, imports)
    assert idx.operation_status == "COMPLETED", idx.details

    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        resolution = resolve_route(_siril_light(), handle.snapshot, v1.default_match_policy())
    finally:
        handle.close()

    assert resolution.outcome == OUTCOME_NEEDS_ATTENTION
    assert resolution.plan is None  # never a silent route
    codes = {r.code for r in resolution.reasons}
    assert "CFA_PHASE_MISMATCH" in codes
    # F4: the flat's gain mismatch (456 vs 120) must NOT be the blocking reason
    # for a prepared flat.
    assert "GAIN_MISMATCH" not in codes


# ---------------------------------------------------------------------------
# 11. Advanced workflow unchanged.
# ---------------------------------------------------------------------------
def test_advanced_workflow_unchanged(qapp, paths):
    from zecalibrator.gui import presentation

    w = MainWindow(paths)
    try:
        # Evidence controls.
        assert w.hdu_edit is not None and w.apply_hdu_btn is not None
        assert w.load_decl_btn.text() == "Load declaration JSON…"
        assert w.load_roi_btn.text() == "Load ROI JSON…"
        # Library controls.
        assert w.open_library_btn.text() == "Open…"
        assert w.index_btn.text() == "Index library…"
        # Explicit-mode + manual-Verify + audit.
        assert [w.additive_combo.itemData(i) for i in range(w.additive_combo.count())] == \
            list(presentation.additive_modes())
        assert [w.flat_combo.itemData(i) for i in range(w.flat_combo.count())] == \
            list(presentation.flat_modes())
        assert w.advanced_preflight_btn.text() == "Verify with selected modes"
        assert w.advanced_export_btn.text() == "Export with selected modes"
        assert w.calibrate_btn.text() == "Calibrate selected in memory"
        # Managed technical controls retained in Advanced.
        assert w.scan_masters_btn.text() == "Detect masters…"
        assert w.confirm_masters_btn.text() == "Confirm detected facts"
        assert w.build_managed_btn.text() == "Build managed library"
        # Audit tabs.
        assert w.tabs is not None
        assert w.preflight_table is not None
        assert w.results_table is not None
        assert w.audit_view is not None
    finally:
        _shutdown(w)


# ---------------------------------------------------------------------------
# 12. queue.Queue / GC / Qt lifetime / cancellation invariants preserved.
# ---------------------------------------------------------------------------
def test_worker_transport_and_lifetime_invariants_preserved(qapp, tmp_path):
    from zecalibrator.gui import service
    from zecalibrator.gui.worker import WorkerController
    from .conftest import run_operation

    before = gc.isenabled()
    ctl = WorkerController()
    try:
        # Transport invariant: shared plain-Python queue.Queue (never Signal(object)).
        assert isinstance(ctl._req_queue, _queue.Queue)
        assert isinstance(ctl._res_queue, _queue.Queue)
        # Run one real operation to exercise the GC idle boundary, then assert the
        # exact prior GC state is restored (never left disabled process-wide).
        snap = service.OperationSnapshot(
            op_id=service.new_operation_id(), kind="load_settings",
            library_spec=None, request=None, policy=None, lights=(),
            config_dir=str(tmp_path),
        )
        result = run_operation(ctl, snap, v1.CancellationToken())
        assert result.ended
        assert gc.isenabled() == before
    finally:
        ctl.shutdown()
        _pump(lambda: ctl.is_finished)
        ctl.finalize()
    assert ctl.is_finished
    assert gc.isenabled() == before


def test_cancellation_restores_idle_and_terminal(qapp, paths, tmp_path):
    from .conftest import write_fits

    fixture = make_synth_fixture(tmp_path)
    w = MainWindow(paths)
    assert wait_idle(w)
    try:
        from zecalibrator.gui.window import _LightEntry

        decl = v1.ImportDeclaration(**__import__("json").loads(open(fixture["decl"]).read()))
        roi = v1.RoiExtentEvidence(**__import__("json").loads(open(fixture["roi"]).read()))
        w._lights.append(_LightEntry(fixture["light"], hdu=0, declaration=decl, roi_extent=roi))
        w._refresh_lights_list()
        w._library_spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])

        import zecalibrator.api.v1 as v1_mod

        real_inspect = v1_mod.inspect_frame

        def slow_inspect(source, *args, **kwargs):
            cancel = kwargs.get("cancel")
            while cancel is None or not cancel.is_cancelled():
                time.sleep(0.01)
            raise v1.OperationCancelled()

        v1_mod.inspect_frame = slow_inspect
        try:
            w._on_preflight()
            assert _pump(lambda: w._controller.is_active)
            w._on_cancel()
            assert _pump(lambda: not w._controller.is_active)
            # Terminal is truthful (CANCELLED) and the controller returns to idle.
            assert "CANCELLED" in w.status_label.text()
            assert not w._controller.is_active
        finally:
            v1_mod.inspect_frame = real_inspect
    finally:
        _shutdown(w)
