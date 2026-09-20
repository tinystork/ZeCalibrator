"""Standard auto-route (R3D-B) GUI tests + Siril-witness.

Standard expresses ONLY user intent ("use the compatible masters I supplied");
ZeCalibrator resolves the scientific route automatically. These tests assert:

* Standard no longer contains Dark/Flat combos (no mode choice);
* Advanced retains its technical controls;
* a nominal Standard smoke never prompts metadata/orientation/full-frame;
* the Siril witness (dark GAIN/OFFSET/CCD-TEMP absent; flat/darkflat GAIN=456 vs
  light GAIN=120) yields NEEDS_ATTENTION — never a silent route — modeled in
  synthetic FITS via ``conftest.fits_bytes`` (no dependency on the real files).
"""

from __future__ import annotations

import json

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
from zecalibrator.core.routes import (
    BIAS_STATE_UNKNOWN,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_READY,
)
from zecalibrator.gui import presentation
from zecalibrator.gui.window import MainWindow
from zecalibrator.storage import resolve_paths

from .conftest import SHAPE, fits_bytes, make_synth_fixture, wait_idle, write_fits_array


@pytest.fixture
def paths(tmp_path):
    return resolve_paths(base=str(tmp_path))


def _close(w):
    from PySide6 import QtTest

    w.close()
    start = __import__("time").monotonic()
    while __import__("time").monotonic() - start < 20:
        QtWidgets.QApplication.processEvents()
        if w._controller.is_finished:
            return
        QtTest.QTest.qWait(5)


# ---------------------------------------------------------------------------
# 1. Standard no longer contains Dark/Flat combos.
# ---------------------------------------------------------------------------
def test_standard_has_no_dark_flat_combos(qapp, paths):
    w = MainWindow(paths)
    try:
        assert not hasattr(w, "standard_additive_combo")
        assert not hasattr(w, "standard_flat_combo")
        # The Standard tab shows lights, masters, active source, actions, summary.
        assert w.lights_list is not None
        assert w.masters_files_list is not None
        assert w.active_source_label is not None
        assert w.preflight_btn.text() == "Verify calibration"
        assert w.export_btn.text() == "Calibrate / Export…"
        assert w.standard_summary_label is not None
        assert "auto" in w.standard_route_label.text().lower()
    finally:
        _close(w)


# ---------------------------------------------------------------------------
# 2. Advanced retains technical controls.
# ---------------------------------------------------------------------------
def test_advanced_retains_technical_controls(qapp, paths):
    w = MainWindow(paths)
    try:
        assert [w.additive_combo.itemData(i) for i in range(w.additive_combo.count())] == \
            list(presentation.additive_modes())
        assert [w.flat_combo.itemData(i) for i in range(w.flat_combo.count())] == \
            list(presentation.flat_modes())
        assert w._request().additive_mode == "dark_incl_bias"
        assert w._request().flat_mode == "none"
        assert w.calibrate_btn.text() == "Calibrate selected in memory"
    finally:
        _close(w)


# ---------------------------------------------------------------------------
# 16. nominal Standard smoke does NOT prompt metadata/orientation/full-frame.
# ---------------------------------------------------------------------------
def test_nominal_standard_smoke_prompts_no_metadata_dialogs(qapp, paths, tmp_path, monkeypatch):
    fixture = make_synth_fixture(tmp_path)
    w = MainWindow(paths)
    assert wait_idle(w)
    prompts = []

    real_information = QtWidgets.QMessageBox.information
    real_warning = QtWidgets.QMessageBox.warning
    real_exec = QtWidgets.QDialog.exec

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information",
        staticmethod(lambda *a, **k: prompts.append("information") or real_information(*a, **k)),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda *a, **k: prompts.append("warning") or real_warning(*a, **k)),
    )
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: prompts.append("dialog"))

    try:
        decl = v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read()))
        roi = v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read()))
        from zecalibrator.gui.window import _LightEntry

        w._lights.append(_LightEntry(fixture["light"], hdu=0, declaration=decl, roi_extent=roi))
        w._refresh_lights_list()
        w._library_spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])
        w._on_preflight()
        assert wait_idle(w)

        # Auto-route resolved a compatible dark -> READY/MATCHED, with no
        # metadata/orientation/full-frame questionnaire shown.
        assert w._preflight_summaries
        assert w._preflight_summaries[0]["outcome"] == "MATCHED"
        assert prompts == [], f"unexpected prompts during Standard smoke: {prompts}"
    finally:
        _close(w)


def _window_with_light(qapp, paths, fixture):
    """A window with one light + opened library, ready for preflight."""
    from zecalibrator.gui.window import _LightEntry

    w = MainWindow(paths)
    assert wait_idle(w)
    decl = v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read()))
    roi = v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read()))
    w._lights.append(_LightEntry(fixture["light"], hdu=0, declaration=decl, roi_extent=roi))
    w._refresh_lights_list()
    w._library_spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])
    return w


# ---------------------------------------------------------------------------
# rework-1: Advanced explicit-route path is reachable (not dead) and Standard
# still auto-routes.
# ---------------------------------------------------------------------------
def test_advanced_explicit_route_reflects_selected_modes(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _window_with_light(qapp, paths, fixture)
    try:
        # Advanced explicit route: control (no additive) + none (no flat).
        w.additive_combo.setCurrentIndex(w.additive_combo.findData("control"))
        w.flat_combo.setCurrentIndex(w.flat_combo.findData("none"))
        w._on_advanced_preflight()
        assert wait_idle(w)
        assert w._preflight_summaries
        assert w._preflight_summaries[0]["outcome"] == "MATCHED"
        plan = w._plans[w._lights[0].row_id]
        # The plan reflects the explicitly selected modes (NOT the auto-route).
        assert plan.request.additive_mode == "control"
        assert plan.request.flat_mode == "none"
    finally:
        _close(w)


def test_standard_still_auto_routes(qapp, paths, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    w = _window_with_light(qapp, paths, fixture)
    try:
        w._on_preflight()
        assert wait_idle(w)
        assert w._preflight_summaries
        assert w._preflight_summaries[0]["outcome"] == "MATCHED"
        plan = w._plans[w._lights[0].row_id]
        # Standard auto-route resolves the compatible dark automatically.
        assert plan.request.additive_mode == "dark_incl_bias"
        assert plan.request.flat_mode == "none"
    finally:
        _close(w)


# ---------------------------------------------------------------------------
# Siril witness: NEEDS_ATTENTION, never a silent route.
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


def test_siril_witness_needs_attention_not_silent_route(tmp_path):
    data = np.full(SHAPE, 1.0, dtype=np.float32)

    # Dark: GAIN/OFFSET/CCD-TEMP absent (bias state not established).
    write_fits_array(tmp_path / "siril_dark.fits", data, header_cards=[])
    # Flat / darkflat: GAIN=456 (vs light 120) — gain incompatibility.
    write_fits_array(
        tmp_path / "siril_flat.fits", data,
        header_cards=[("GAIN", 456), ("OFFSET", 50), ("CCD-TEMP", 20.0)],
    )
    write_fits_array(
        tmp_path / "siril_darkflat.fits", data,
        header_cards=[("GAIN", 456), ("OFFSET", 50), ("CCD-TEMP", 20.0)],
    )

    imports = [
        v1.MasterImportSpec(
            path="siril_dark.fits", master_type="dark", hdu=0,
            # no gain/offset/temperature; exposure present so the dark can route.
            declaration=v1.ImportDeclaration(**_SIRIL_BASE, exposure_s=10.0),
            dq_state="no_source_dq",
        ),
        v1.MasterImportSpec(
            path="siril_flat.fits", master_type="flat", hdu=0,
            declaration=v1.ImportDeclaration(
                **_SIRIL_BASE, gain=456.0, offset=50.0, exposure_s=10.0,
            ),
            flat_form="raw_response", dq_state="no_source_dq",
        ),
        v1.MasterImportSpec(
            path="siril_darkflat.fits", master_type="flat_dark", hdu=0,
            declaration=v1.ImportDeclaration(
                **_SIRIL_BASE, gain=456.0, offset=50.0, exposure_s=10.0,
            ),
            dq_state="no_source_dq",
        ),
    ]

    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "siril.sqlite"))
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
    # The dark routes as bias-included (bias_state defaults to "included" under
    # the Standard master contract); it is NOT rejected for missing
    # gain/temperature, and never reported as an indeterminate bias state.
    assert BIAS_STATE_UNKNOWN not in codes
    missing_fields = {
        r.field for r in resolution.reasons if r.code == "MISSING_REQUIRED_FIELD"
    }
    assert "acquisition.gain" not in missing_fields
    assert "acquisition.offset" not in missing_fields
    assert "acquisition.temperature_c" not in missing_fields
    # The flat's known GAIN_MISMATCH (456 vs 120) stays blocking.
    assert "GAIN_MISMATCH" in codes
    # The dark's missing acquisition facts are recorded as non-blocking
    # UNVERIFIED notes (never a blocking rejection).
    unverified_fields = {r.field for r in resolution.unverified}
    assert "acquisition.gain" in unverified_fields
