"""Parity: identical fixture through CLI, API and GUI worker.

ASTRA G7 requires "same plan/result/provenance via CLI/API/GUI on identical
fixture". This test compares the plan id, calibrated science payload (float32
including negatives) and provenance semantic content across three paths over the
same synthetic SYNTH-BASE-1 dark fixture. Only explicitly nondeterministic
operation/batch/path identity fields are normalized; scientific content is never
normalized.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service

from .conftest import SHAPE, make_synth_fixture, run_operation

_REPO = Path(__file__).resolve().parents[2]
_SRC = str(_REPO / "src")


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


# Reads back a committed multi-extension GUI export FITS in a SUBPROCESS (the
# FITS library runs in the child, never on the GUI main thread). The child
# returns the primary float32 science plane, the DQ uint16 extension, or the
# CALPROV uint8 extension bytes, base64-encoded to preserve exact bits.
_OUTPUT_READER_SCRIPT = r'''
import sys, json, base64
import numpy as np
from astropy.io import fits

path = sys.argv[1]
mode = sys.argv[2]  # "science" or "calprov"

with fits.open(path, memmap=False) as hdul:
    primary = np.ascontiguousarray(hdul[0].data, dtype=np.float32)
    if mode == "science":
        mask = None
        for hdu in hdul[1:]:
            if hdu.header.get("EXTNAME") == "DQ":
                mask = np.ascontiguousarray(hdu.data, dtype=np.uint16)
        out = {
            "data_b64": base64.b64encode(primary.tobytes()).decode("ascii"),
            "data_shape": list(primary.shape),
            "mask_b64": base64.b64encode(mask.tobytes()).decode("ascii") if mask is not None else None,
            "mask_shape": list(mask.shape) if mask is not None else None,
        }
    else:
        calprov = None
        for hdu in hdul[1:]:
            if hdu.header.get("EXTNAME") == "CALPROV":
                calprov = np.ascontiguousarray(hdu.data, dtype=np.uint8).tobytes()
        out = {
            "calprov_b64": base64.b64encode(calprov).decode("ascii") if calprov is not None else None,
        }
print(json.dumps(out))
'''


def _read_committed_output(path, mode):
    """Run the FITS read-back in a subprocess and return the decoded fields."""
    import base64

    proc = subprocess.run(
        [sys.executable, "-c", _OUTPUT_READER_SCRIPT, str(path), mode],
        capture_output=True, text=True, env=_env(),
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    if mode == "science":
        data = np.frombuffer(
            base64.b64decode(out["data_b64"]), dtype=np.float32
        ).reshape(out["data_shape"])
        if out["mask_b64"] is None:
            return data, None
        mask = np.frombuffer(
            base64.b64decode(out["mask_b64"]), dtype=np.uint16
        ).reshape(out["mask_shape"])
        return data, mask
    calprov_b64 = out["calprov_b64"]
    return base64.b64decode(calprov_b64) if calprov_b64 is not None else None


def _light_source(fixture):
    from .conftest import DECL, ROI

    return v1.FitsFrameSource(
        path=fixture["light"], hdu=0,
        declaration=service.parse_declaration(DECL),
        roi_extent=service.parse_roi(ROI),
    )


def _library(fixture):
    return v1.open_library(v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"]))


def _api_reference(fixture):
    """Direct public-API path: inspect -> resolve -> calibrate_frame."""
    inspection = v1.inspect_frame(_light_source(fixture)).inspection
    opened = _library(fixture)
    handle = opened.handle
    try:
        resolve = v1.resolve_calibration(
            inspection, v1.CalibrationRequest("dark_incl_bias", "none"),
            handle, v1.default_match_policy(),
        )
        assert resolve.outcome == "MATCHED"
        plan = resolve.plan
    finally:
        handle.close()
    result = v1.calibrate_frame(_light_source(fixture), plan, v1.ExecutionOptions())
    assert result.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    return plan, result


def _gui_plan_and_result(qapp, controller, fixture):
    """GUI worker path: preflight -> in-memory calibrate."""
    from zecalibrator.gui.service import LightInput

    from .conftest import DECL, ROI

    light = LightInput(
        path=fixture["light"], hdu=0, declaration=service.parse_declaration(DECL),
        roi_extent=service.parse_roi(ROI), display_name="light.fits",
    )
    spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])
    snap = service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="preflight",
        library_spec=spec, request=v1.CalibrationRequest("dark_incl_bias", "none"),
        policy=v1.default_match_policy(), lights=(light,),
    )
    res = run_operation(controller, snap, v1.CancellationToken())
    assert not res.timed_out
    plan = res.preflight[0][2]
    assert plan is not None
    snap2 = service.OperationSnapshot(
        op_id=service.new_operation_id(), kind="calibrate_in_memory",
        library_spec=None, request=None, policy=None, lights=(light,), plan=plan,
    )
    res2 = run_operation(controller, snap2, v1.CancellationToken())
    summary = res2.finished_summary()
    assert summary["status"] == "COMPLETED"
    return plan, summary


def _cli_reference(fixture):
    """CLI path (subprocess) for plan id + result status + manifest science."""
    decl = fixture["decl"]
    roi = fixture["roi"]
    index = fixture["index"]
    light = fixture["light"]
    out = Path(fixture["root"]) / "cli_out"
    out.mkdir()

    proc = subprocess.run(
        [sys.executable, "-m", "zecalibrator", "match", light,
         "--library", index, "--additive-mode", "dark_incl_bias",
         "--declaration", "@" + decl, "--roi-extent", "@" + roi],
        capture_output=True, text=True, env=_env(),
    )
    assert proc.returncode == 0, proc.stderr
    match = json.loads(proc.stdout)
    assert match["outcome"] == "MATCHED"

    proc2 = subprocess.run(
        [sys.executable, "-m", "zecalibrator", "calibrate", light,
         "--library", index, "--additive-mode", "dark_incl_bias",
         "--declaration", "@" + decl, "--roi-extent", "@" + roi,
         "--destination", str(out), "--batch-id", "parity"],
        capture_output=True, text=True, env=_env(),
    )
    assert proc2.returncode == 0, proc2.stderr
    cal = json.loads(proc2.stdout)
    assert cal["status"] == "COMPLETED"
    return match["plan_id"], cal


def test_plan_id_matches_across_api_and_gui(qapp, controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    api_plan, _ = _api_reference(fixture)
    gui_plan, _ = _gui_plan_and_result(qapp, controller, fixture)
    assert gui_plan.plan_id == api_plan.plan_id


def test_plan_id_matches_across_api_and_cli(tmp_path):
    fixture = make_synth_fixture(tmp_path)
    api_plan, _ = _api_reference(fixture)
    cli_plan_id, _ = _cli_reference(fixture)
    assert cli_plan_id == api_plan.plan_id


def test_science_payload_and_provenance_match_gui_vs_api(qapp, controller, tmp_path):
    fixture = make_synth_fixture(tmp_path)
    api_plan, api_result = _api_reference(fixture)
    gui_plan, gui_summary = _gui_plan_and_result(qapp, controller, fixture)

    # Provenance plan identity and status must agree.
    gui_prov = gui_summary["provenance"]
    api_prov = api_result.provenance.to_dict()
    assert gui_prov["plan"]["plan_id"] == api_prov["plan"]["plan_id"]
    assert gui_prov["status"] == api_prov["status"]
    assert gui_prov["science_contract"] == api_prov["science_contract"]
    assert gui_prov["matching_policy"] == api_prov["matching_policy"]

    # The GUI summary carries scalars (public to_dict); direct API carries data.
    assert gui_summary["scalars"] == api_result.to_dict()["scalars"]


def test_cli_manifest_science_digest_matches_api(tmp_path):
    """Standalone CLI output science digest equals the direct API science digest."""
    from zecalibrator.core.digests import science_digest

    fixture = make_synth_fixture(tmp_path)
    _, api_result = _api_reference(fixture)
    _, cal = _cli_reference(fixture)
    cli_science_digest = cal["items"][0]["output"]["science_digest"]

    # Direct API science digest of the same calibrated plane.
    assert science_digest(api_result.data, api_result.mask) == cli_science_digest


def test_gui_export_payload_matches_api_and_cli(qapp, controller, tmp_path):
    """G1/T4: the GUI's standalone export science/DQ (with negatives and nonzero DQ)
    matches API and CLI on the same fixture, including full semantic provenance."""
    from zecalibrator.core.digests import science_digest

    # dark=100, light=90 -> calibrated output is negative (-10); dark mask has
    # one ADDITIVE_INVALID pixel so the output DQ is non-zero.
    mask = np.zeros(SHAPE, dtype=np.uint16)
    mask[0, 0] = 0x0002  # ADDITIVE_INVALID
    fixture = make_synth_fixture(tmp_path, light_value=90.0, dark_value=100.0, dark_mask=mask)
    _, api_result = _api_reference(fixture)

    # Fixture must actually contain a finite negative and a nonzero DQ bit.
    finite = api_result.data[np.isfinite(api_result.data)]
    assert (finite < 0).any(), "fixture must produce a finite negative value"
    assert (api_result.mask != 0).any(), "fixture must produce nonzero DQ"

    # GUI worker export path (standalone).
    from zecalibrator.gui import service as svc

    out = tmp_path / "gui_out"
    out.mkdir()
    light = svc.LightInput(
        path=fixture["light"], hdu=0,
        declaration=svc.parse_declaration(__import__("json").loads(open(fixture["decl"]).read())),
        roi_extent=svc.parse_roi(__import__("json").loads(open(fixture["roi"]).read())),
        display_name="light.fits",
    )
    snap = svc.OperationSnapshot(
        op_id=svc.new_operation_id(), kind="export",
        library_spec=v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"]),
        request=v1.CalibrationRequest("dark_incl_bias", "none"),
        policy=v1.default_match_policy(),
        lights=(light,),
        destination=str(out), batch_id=svc.new_batch_id(),
    )
    res = run_operation(controller, snap, v1.CancellationToken())
    assert not res.timed_out
    summary = res.finished_summary()
    assert summary["status"] == "COMPLETED"
    gui_output = summary["items"][0]["output"]
    gui_fits = gui_output["path"]

    # Read the committed GUI FITS (primary float32 science + DQ extension) in a
    # SUBPROCESS — no FITS library on the GUI main thread.
    gui_data, gui_mask = _read_committed_output(gui_fits, "science")
    assert gui_mask is not None

    # Full scientific parity: float32 data (incl. negatives) and uint16 DQ.
    assert np.array_equal(gui_data, api_result.data, equal_nan=True)
    assert np.array_equal(gui_mask, api_result.mask)

    # Science digest parity vs the independent API digest (real reference, not a tautology).
    assert gui_output["science_digest"] == science_digest(api_result.data, api_result.mask)

    # And against the CLI manifest science digest.
    _, cal = _cli_reference(fixture)
    assert gui_output["science_digest"] == cal["items"][0]["output"]["science_digest"]


def test_gui_export_calprov_provenance_matches_api(tmp_path, qapp, controller):
    """T4: the GUI's exported CALPROV carries the full semantic public provenance,
    equal to the direct API provenance on the identical fixture (normalizing only
    output-logical-id/path nondeterminism)."""
    from zecalibrator.io.output_writer import parse_calprov

    mask = np.zeros(SHAPE, dtype=np.uint16)
    mask[0, 0] = 0x0002
    fixture = make_synth_fixture(tmp_path, light_value=90.0, dark_value=100.0, dark_mask=mask)
    _, api_result = _api_reference(fixture)

    from zecalibrator.gui import service as svc

    out = tmp_path / "gui_out"
    out.mkdir()
    light = svc.LightInput(
        path=fixture["light"], hdu=0,
        declaration=svc.parse_declaration(json.loads(open(fixture["decl"]).read())),
        roi_extent=svc.parse_roi(json.loads(open(fixture["roi"]).read())),
        display_name="light.fits",
    )
    snap = svc.OperationSnapshot(
        op_id=svc.new_operation_id(), kind="export",
        library_spec=v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"]),
        request=v1.CalibrationRequest("dark_incl_bias", "none"),
        policy=v1.default_match_policy(), lights=(light,),
        destination=str(out), batch_id=svc.new_batch_id(),
    )
    res = run_operation(controller, snap, v1.CancellationToken())
    assert not res.timed_out
    gui_fits = res.finished_summary()["items"][0]["output"]["path"]

    # Read the committed CALPROV extension in a SUBPROCESS (no FITS library on
    # the GUI main thread), then parse the returned bytes.
    calprov_bytes = _read_committed_output(gui_fits, "calprov")
    assert calprov_bytes is not None
    calprov = dict(parse_calprov(calprov_bytes))

    api_prov = api_result.provenance.to_dict()
    # Semantic provenance parity (plan, policy, scalars, status, versions).
    assert calprov["plan"]["plan_id"] == api_prov["plan"]["plan_id"]
    assert calprov["status"] == api_prov["status"]
    assert calprov["science_contract"] == api_prov["science_contract"]
    assert calprov["matching_policy"] == api_prov["matching_policy"]
    assert calprov["provenance_schema"] == api_prov["provenance_schema"]
    assert calprov["backend"] == api_prov["backend"]
    assert dict(calprov["scalars"]) == dict(api_prov["scalars"])
    assert calprov["executed_processing"] == api_prov["executed_processing"]
    # Nested master identities (role -> content_sha256/mask_identity) preserved.
    assert set(calprov["plan"]["masters"]) == set(api_prov["plan"]["masters"])
    for role in api_prov["plan"]["masters"]:
        a = api_prov["plan"]["masters"][role]
        c = calprov["plan"]["masters"][role]
        assert c["content_sha256"] == a["content_sha256"]
        assert c["mask_identity"] == a["mask_identity"]
        assert c["descriptor_id"] == a["descriptor_id"]
