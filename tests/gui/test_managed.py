"""Managed master ingestion: service evidence confirmation + worker ops + active source.

Headless service tests (no Qt) for the frozen R3 evidence-source mapping and the
detect/present/persist flow, plus Qt worker-op tests (scan/load-ledger/confirm/
build-managed-library) and the exactly-one-active-source window state.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service

from .conftest import write_fits_array

SHAPE = (4, 4)


# ---------------------------------------------------------------------------
# Service: frozen R3 evidence-source mapping + detect/present/persist
# ---------------------------------------------------------------------------
def test_detect_header_candidates_exact_approved_table():
    cards = [
        ("EXPTIME", 300.0), ("CCD-TEMP", 20.0), ("GAIN", 100.0), ("EGA", 100.0),
        ("OFFSET", 50.0), ("READOUTM", "MODE_A"), ("ADCMODE", "MODE_16"),
        ("INSTRUME", "SYNTH-CFA"), ("BAYERPAT", "RGGB"), ("FILTER", "L"),
    ]
    candidates, conflicts = service.detect_header_candidates(cards)
    assert conflicts == {}
    assert candidates["exposure_s"].value == 300.0
    assert candidates["exposure_s"].origin_field == "EXPTIME"
    assert candidates["temperature_c"].value == 20.0
    assert candidates["gain"].value == 100.0
    assert candidates["offset"].value == 50.0
    assert candidates["readout_mode"].value == "MODE_A"
    assert candidates["adc_mode"].value == "MODE_16"
    assert candidates["detector_model"].value == "SYNTH-CFA"
    assert candidates["cfa_phase"].value == "RGGB"
    assert candidates["filter"].value == "L"
    # confirmed_by is None until the user confirms.
    assert all(f.confirmed_by is None for f in candidates.values())


def test_unapproved_alias_produces_no_candidate():
    # SET-TEMP is now an approved source for ``temperature_setpoint_c`` (G2C);
    # genuinely unapproved aliases still produce no candidate.
    cards = [("TEMP_ALIAS", 20.0), ("READMODE_ALIAS", "X"), ("FOO", 1.0)]
    candidates, conflicts = service.detect_header_candidates(cards)
    assert candidates == {}
    assert conflicts == {}


def test_conflict_not_auto_arbitrated():
    cards = [("EXPTIME", 300.0), ("EXPOSURE", 600.0)]
    candidates, conflicts = service.detect_header_candidates(cards)
    assert "exposure_s" not in candidates
    assert len(conflicts["exposure_s"]) == 2


def test_heuristic_filename_never_proposed():
    cards = [("FILENAME", "Dark_300s.fits"), ("OBJNAME", "master")]
    candidates, conflicts = service.detect_header_candidates(cards)
    assert candidates == {}
    assert conflicts == {}


def test_binning_scalar_and_pair():
    c_scalar, _ = service.detect_header_candidates([("BINNING", 2)])
    assert c_scalar["binning"].value == (2, 2)
    c_pair, _ = service.detect_header_candidates([("XBINNING", 2), ("YBINNING", 3)])
    assert c_pair["binning"].value == (3, 2)


def test_evidence_for_fields_persists_confirmed_only():
    candidates, _ = service.detect_header_candidates([("EXPTIME", 300.0), ("CCD-TEMP", 20.0)])
    evidence = service.evidence_for_fields(candidates, {"exposure_s": True})
    assert "exposure_s" in evidence
    assert "temperature_c" not in evidence  # unconfirmed stays absent
    assert evidence["exposure_s"].confirmed_by == "user"
    assert evidence["exposure_s"].value == 300.0


def test_build_declaration_from_confirmed_evidence_and_user_facts():
    candidates, _ = service.detect_header_candidates([("EXPTIME", 300.0)])
    evidence = service.evidence_for_fields(candidates, {"exposure_s": True})
    decl = service.build_declaration(
        "user", "managed-import", "1", evidence,
        {"detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA"},
    )
    assert decl.exposure_s == 300.0
    assert decl.detector_instance_id == "SYNTH-DET-0001"
    assert decl.domain == "raw" and decl.units == "ADU"


# ---------------------------------------------------------------------------
# F1: per-master-type required-field gate + honest R4 reporting
# ---------------------------------------------------------------------------
def test_parse_user_fact_types():
    assert service.parse_user_fact("sensor_dimensions", "4 4") == (4, 4)
    assert service.parse_user_fact("roi_origin", "0,0") == (0, 0)
    assert service.parse_user_fact("gain", "100.5") == 100.5
    assert service.parse_user_fact("detector_instance_id", "SYNTH-DET-0001") == "SYNTH-DET-0001"
    assert service.parse_user_fact("gain", "") is None
    with pytest.raises(ValueError):
        service.parse_user_fact("sensor_dimensions", "4")
    with pytest.raises(ValueError):
        service.parse_user_fact("gain", "not-a-number")


def _full_dark_decl():
    return service.build_declaration(
        "user", "managed-import", "1",
        {
            "exposure_s": v1.EvidenceFact("exposure_s", 300.0, "fits_header", "EXPTIME", "user", "1"),
            "temperature_c": v1.EvidenceFact("temperature_c", 20.0, "fits_header", "CCD-TEMP", "user", "1"),
            "gain": v1.EvidenceFact("gain", 100.0, "fits_header", "GAIN", "user", "1"),
            "offset": v1.EvidenceFact("offset", 50.0, "fits_header", "OFFSET", "user", "1"),
            "readout_mode": v1.EvidenceFact("readout_mode", "MODE_A", "fits_header", "READOUTM", "user", "1"),
            "adc_mode": v1.EvidenceFact("adc_mode", "MODE_16", "fits_header", "ADCMODE", "user", "1"),
            "detector_model": v1.EvidenceFact("detector_model", "SYNTH-CFA", "fits_header", "INSTRUME", "user", "1"),
            "cfa_phase": v1.EvidenceFact("cfa_phase", "mono", "fits_header", "BAYERPAT", "user", "1"),
            "binning": v1.EvidenceFact("binning", (1, 1), "fits_header", "BINNING", "user", "1"),
        },
        {
            "detector_instance_id": "SYNTH-DET-0001",
            "sensor_dimensions": (4, 4), "orientation": "identity", "roi_origin": (0, 0),
        },
    )


def test_header_only_master_is_insufficient_evidence():
    # A header-only dark leaves the NECESSARY (matching-blocking) fields unset;
    # disambiguators are never reported missing (R3C) and never block "ready".
    decl = service.build_declaration(
        "user", "managed-import", "1",
        {"exposure_s": v1.EvidenceFact("exposure_s", 300.0, "fits_header", "EXPTIME", "user", "1")},
    )
    missing = service.missing_required_fields("dark", decl)
    # Disambiguators are never part of the necessary tier.
    assert "detector_instance_id" not in missing
    assert "sensor_dimensions" not in missing
    assert "readout_mode" not in missing
    assert "adc_mode" not in missing
    # cfa_phase is unknown (None), so CFA-only geometry is not required.
    assert "orientation" not in missing
    assert "roi_origin" not in missing
    # The genuinely necessary facts are reported.
    assert "detector_model" in missing
    assert "gain" in missing
    assert "offset" in missing
    assert "binning" in missing
    assert "cfa_phase" in missing
    assert "temperature_c" in missing
    status, reasons = service.master_evidence_status("dark", decl)
    assert status == "needs_attention"


def test_complete_dark_is_ready():
    decl = _full_dark_decl()
    assert service.missing_required_fields("dark", decl) == ()
    assert service.master_evidence_status("dark", decl) == ("ready", ())


def test_managed_flat_without_quality_provenance_is_insufficient():
    # Even with complete matching facts, a managed flat lacks machine-readable
    # quality evidence (R4) and must be reported insufficient, never silent.
    decl = service.build_declaration(
        "user", "managed-import", "1",
        {
            "exposure_s": v1.EvidenceFact("exposure_s", 1.0, "fits_header", "EXPTIME", "user", "1"),
            "filter": v1.EvidenceFact("filter", "L", "fits_header", "FILTER", "user", "1"),
        },
        {
            "detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA",
            "gain": 100.0, "offset": 50.0, "readout_mode": "MODE_A", "adc_mode": "MODE_16",
            "temperature_c": 20.0, "sensor_dimensions": (4, 4), "binning": (1, 1),
            "orientation": "identity", "roi_origin": (0, 0), "cfa_phase": "mono",
            "optical_train_id": "SYNTH-TRAIN-1",
        },
    )
    assert service.missing_required_fields("flat", decl) == ()
    status, reasons = service.master_evidence_status("flat", decl)
    assert status == "needs_attention"
    assert any("quality" in r for r in reasons)


def test_user_only_facts_supplied_via_extra_carried_in_declaration():
    decl = _full_dark_decl()
    assert decl.detector_instance_id == "SYNTH-DET-0001"
    assert decl.sensor_dimensions == (4, 4)
    assert decl.orientation == "identity"
    assert decl.roi_origin == (0, 0)


# ---------------------------------------------------------------------------
# Worker ops: scan / load-ledger / confirm / build-managed-library
# ---------------------------------------------------------------------------
pytest.importorskip("PySide6")


def _write_dark_with_cards(path):
    data = np.full(SHAPE, 10.0, dtype=np.float32)
    cards = [
        ("EXPTIME", 300.0),
        ("CCD-TEMP", 20.0),
        ("GAIN", 100.0),
    ]
    return write_fits_array(path, data, header_cards=cards, bunit="ADU")


def _snapshot(kind, **kw):
    return service.OperationSnapshot(
        op_id=service.new_operation_id(), kind=kind,
        library_spec=None, request=None, policy=None, lights=(), **kw,
    )


def test_worker_scan_masters(controller, tmp_path):
    from .conftest import run_operation

    dark = _write_dark_with_cards(tmp_path / "dark.fits")
    snap = _snapshot("scan_masters", master_paths=(dark,), master_type="dark")
    result = run_operation(controller, snap, v1.CancellationToken())
    summary = result.finished_summary()
    assert summary["kind"] == "scan_masters"
    assert summary["status"] == "COMPLETED"
    master = summary["masters"][0]
    assert master["status"] == "COMPLETED"
    assert master["candidates"]["exposure_s"]["value"] == 300.0
    assert master["candidates"]["temperature_c"]["value"] == 20.0


def test_worker_confirm_load_and_build_managed(controller, tmp_path):
    from .conftest import run_operation

    dark = _write_dark_with_cards(tmp_path / "dark.fits")
    ledger_dir = str(tmp_path / "data")
    payload = {
        "role": "dark", "path": dark, "hdu": 0,
        "evidence": {
            "exposure_s": {"value": 300.0, "origin_type": "fits_header", "origin_field": "EXPTIME"},
        },
        "extra": {
            "detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA",
            "gain": 100.0, "offset": 50.0, "readout_mode": "MODE_A", "adc_mode": "MODE_16",
            "binning": (1, 1), "sensor_dimensions": SHAPE, "orientation": "identity",
            "cfa_phase": "mono", "roi_origin": (0, 0), "temperature_c": 20.0,
            "filter": "NONE", "optical_train_id": "SYNTH-TRAIN-1",
        },
        "dq_state": "no_source_dq", "mask_path": None, "bias_state": "included",
        "declaration_source": "user", "declaration_identity": "managed-import", "declaration_version": "1",
    }
    confirm = _snapshot("confirm_evidence", ledger_dir=ledger_dir, confirm_payload=payload)
    r1 = run_operation(controller, confirm, v1.CancellationToken())
    assert r1.finished_summary()["status"] == "COMPLETED"

    load = _snapshot("load_ledger", ledger_dir=ledger_dir)
    r2 = run_operation(controller, load, v1.CancellationToken())
    s2 = r2.finished_summary()
    assert s2["state"] == "ok"
    assert s2["count"] == 1
    assert s2["records"][0]["role"] == "dark"
    assert s2["records"][0]["dq_state"] == "no_source_dq"

    spec = v1.LibrarySpec(root=str(tmp_path / "cache"), index_path=str(tmp_path / "cache" / "managed.sqlite"))
    build = _snapshot("build_managed_library", ledger_dir=ledger_dir, managed_spec=spec)
    r3 = run_operation(controller, build, v1.CancellationToken())
    s3 = r3.finished_summary()
    assert s3["status"] in ("COMPLETED", "REUSED")
    assert s3["candidate_count"] == 1


def test_worker_build_indexes_insufficient_evidence_master(controller, tmp_path):
    """R3D-D: ADMISSION != COMPATIBILITY — a header-only dark (missing necessary
    matching fields) is still INDEXED; ``needs_attention`` is informational only,
    never an ingestion filter."""
    from .conftest import run_operation

    dark = _write_dark_with_cards(tmp_path / "dark.fits")
    ledger_dir = str(tmp_path / "data")
    payload = {
        "role": "dark", "path": dark, "hdu": 0,
        "evidence": {
            "exposure_s": {"value": 300.0, "origin_type": "fits_header", "origin_field": "EXPTIME"},
        },
        "extra": {},  # header-only: required user-only facts absent
        "dq_state": "no_source_dq", "mask_path": None, "bias_state": "included",
        "declaration_source": "user", "declaration_identity": "managed-import", "declaration_version": "1",
    }
    confirm = _snapshot("confirm_evidence", ledger_dir=ledger_dir, confirm_payload=payload)
    r1 = run_operation(controller, confirm, v1.CancellationToken())
    s1 = r1.finished_summary()
    assert s1["status"] == "COMPLETED"
    assert s1["evidence_status"] == "needs_attention"
    # Disambiguators never block readiness (R3C); only the necessary tier does.
    assert "detector_model" in s1["missing"]
    assert "detector_instance_id" not in s1["missing"]

    spec = v1.LibrarySpec(root=str(tmp_path / "cache"), index_path=str(tmp_path / "cache" / "managed.sqlite"))
    build = _snapshot("build_managed_library", ledger_dir=ledger_dir, managed_spec=spec)
    r2 = run_operation(controller, build, v1.CancellationToken())
    s2 = r2.finished_summary()
    # The insufficient-evidence master is now INDEXED (admission != compatibility);
    # its incomplete evidence is reported as an informational diagnostic, never a
    # drop.
    assert s2["candidate_count"] == 1
    assert len(s2["needs_attention"]) == 1
    assert "detector_model" in s2["needs_attention"][0]["missing"]
    assert "detector_instance_id" not in s2["needs_attention"][0]["missing"]


def test_worker_build_filters_by_session_selection(controller, tmp_path):
    """F3: the derived index is built ONLY from the current session selection,
    reusing evidence from the persistent ledger by content identity."""
    from .conftest import run_operation

    dark = _write_dark_with_cards(tmp_path / "dark.fits")
    bias = _write_dark_with_cards(tmp_path / "bias.fits")
    ledger_dir = str(tmp_path / "data")

    common = {
        "detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA",
        "gain": 100.0, "offset": 50.0, "readout_mode": "MODE_A", "adc_mode": "MODE_16",
        "binning": (1, 1), "sensor_dimensions": SHAPE, "orientation": "identity",
        "cfa_phase": "mono", "roi_origin": (0, 0), "temperature_c": 20.0,
    }
    dark_extra = dict(common, exposure_s=300.0)
    bias_extra = dict(common, bias_exposure_max_s=0.01)

    def confirm(path, role, extra):
        payload = {
            "role": role, "path": path, "hdu": 0,
            "evidence": {
                "exposure_s": {"value": 300.0, "origin_type": "fits_header", "origin_field": "EXPTIME"},
            },
            "extra": extra, "dq_state": "no_source_dq", "mask_path": None,
            "bias_state": "included",
            "declaration_source": "user", "declaration_identity": "managed-import",
            "declaration_version": "1",
        }
        snap = _snapshot("confirm_evidence", ledger_dir=ledger_dir, confirm_payload=payload)
        r = run_operation(controller, snap, v1.CancellationToken())
        assert r.finished_summary()["status"] == "COMPLETED"

    confirm(dark, "dark", dark_extra)
    confirm(bias, "bias", bias_extra)

    sha_dark, _ = v1.content_identity(dark, hdu=0)
    spec = v1.LibrarySpec(
        root=str(tmp_path / "cache"), index_path=str(tmp_path / "cache" / "managed.sqlite")
    )

    # No session filter -> both roles built from the full ledger.
    r1 = run_operation(
        controller, _snapshot("build_managed_library", ledger_dir=ledger_dir, managed_spec=spec),
        v1.CancellationToken(),
    )
    assert r1.finished_summary()["candidate_count"] == 2

    # Session filter (only the dark) -> only the dark is in the derived index.
    r2 = run_operation(
        controller,
        _snapshot(
            "build_managed_library", ledger_dir=ledger_dir, managed_spec=spec,
            session_selection=((sha_dark, "dark"),),
        ),
        v1.CancellationToken(),
    )
    s2 = r2.finished_summary()
    assert s2["candidate_count"] == 1

    opened = v1.open_library(spec)
    handle = opened.handle
    try:
        roles = handle.snapshot.roles()
        assert "dark" in roles
        assert "bias" not in roles
    finally:
        handle.close()


def test_window_exactly_one_active_source(qapp, tmp_path):
    from zecalibrator.gui.window import MainWindow
    from zecalibrator.storage import resolve_paths

    paths = resolve_paths(base=str(tmp_path))
    w = MainWindow(paths)
    assert w._active_source is None

    # Managed source clears the explicit library spec.
    w._library_spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "e.sqlite"))
    w._set_active_source("managed")
    assert w._active_source == "managed"
    assert w._library_spec is None

    # Explicit source clears the managed scan/pending state.
    w._managed_scan = [{"path": "x"}]
    w._set_active_source("explicit")
    assert w._active_source == "explicit"
    assert w._managed_scan == []

    w._set_active_source(None)
    assert w._active_source is None
    w._controller.shutdown()
    from PySide6 import QtWidgets
    QtWidgets.QApplication.processEvents()
