"""Worker-level master-admission witnesses (Qt, offscreen).

Covers the scan worker producing IMAGETYP-detected roles + admissibility, and the
confirm worker reusing an unchanged known master (same content identity + role +
evidence + declaration) without re-asking or duplicating the ledger record.
"""

from __future__ import annotations

import numpy as np
import pytest

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service

from .conftest import write_fits_array

pytest.importorskip("PySide6")

SHAPE = (4, 4)


def _write(path, *, imagetyp=None, history=None):
    data = np.full(SHAPE, 10.0, dtype=np.float32)
    cards = [
        ("EXPTIME", 300.0),
        ("CCD-TEMP", 20.0),
        ("GAIN", 100.0),
    ]
    if imagetyp is not None:
        cards.append(("IMAGETYP", imagetyp))
    if history is not None:
        cards.append(("HISTORY", history))
    return write_fits_array(path, data, header_cards=cards, bunit="ADU")


def _snapshot(kind, **kw):
    return service.OperationSnapshot(
        op_id=service.new_operation_id(), kind=kind,
        library_spec=None, request=None, policy=None, lights=(), **kw,
    )


def _confirm_payload(path, role, extra=None):
    return {
        "role": role, "path": path, "hdu": 0,
        "evidence": {
            "exposure_s": {"value": 300.0, "origin_type": "fits_header", "origin_field": "EXPTIME"},
        },
        "extra": extra or {
            "detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA",
            "gain": 100.0, "offset": 50.0, "readout_mode": "MODE_A", "adc_mode": "MODE_16",
            "binning": (1, 1), "sensor_dimensions": SHAPE, "orientation": "identity",
            "cfa_phase": "mono", "roi_origin": (0, 0), "temperature_c": 20.0,
            "filter": "NONE", "optical_train_id": "SYNTH-TRAIN-1",
        },
        "dq_state": "no_source_dq", "mask_path": None, "bias_state": "included",
        "declaration_source": "user", "declaration_identity": "managed-import",
        "declaration_version": "1",
    }


def test_worker_scan_detects_imagetyp_role_and_admissibility(controller, tmp_path):
    from .conftest import run_operation

    dark = _write(tmp_path / "dark.fits", imagetyp="DARK")
    rgb = _write(tmp_path / "rgb.fits", imagetyp="DARK", history="debayer RGB output")

    # Debayer (2-D but processed) is incompatible, not RGB.
    snap = _snapshot("scan_masters", master_paths=(dark, rgb), master_type=None)
    result = run_operation(controller, snap, v1.CancellationToken())
    summary = result.finished_summary()
    assert summary["status"] == "COMPLETED"
    masters = {m["path"]: m for m in summary["masters"]}
    assert masters[dark]["detected_role"] == "dark"
    assert masters[dark]["admissible"] is True
    assert masters[rgb]["admissible"] is False
    assert any("debayer" in r for r in masters[rgb]["incompatible"])


def test_worker_confirm_reuses_unchanged_master(controller, tmp_path):
    from .conftest import run_operation

    dark = _write(tmp_path / "dark.fits", imagetyp="DARK")
    ledger_dir = str(tmp_path / "data")
    payload = _confirm_payload(dark, "dark")

    first = _snapshot("confirm_evidence", ledger_dir=ledger_dir, confirm_payload=payload)
    r1 = run_operation(controller, first, v1.CancellationToken())
    assert r1.finished_summary()["status"] == "COMPLETED"

    # Re-confirm the identical unchanged master: reused, not duplicated.
    second = _snapshot("confirm_evidence", ledger_dir=ledger_dir, confirm_payload=payload)
    r2 = run_operation(controller, second, v1.CancellationToken())
    s2 = r2.finished_summary()
    assert s2["status"] == "REUSED"
    assert s2["count"] == 1

    # Ledger still holds exactly one record (no duplicate).
    loaded = v1.load_managed_ledger(v1.managed_ledger_path(ledger_dir))
    assert loaded.state == "ok"
    assert len(loaded.records) == 1
