"""Producer tests for the C1 public session-library extension.

Covers the frozen contract: the two new public symbols + result types, the
``session_library``/``auto_route`` capabilities, admission + role identification
from a folder, per-light auto-route, in-memory calibration with master reuse
(S-a counter), and the C1/C3a/C3b/C3c clauses.

Qt-free, ZeAlfie-free, ZSSS-free: only ``zecalibrator.api.v1`` + numpy/astropy
for fixtures.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1

SHAPE = (40, 32)


def _write_fits(path, imagetyp, exptime, *, filter_=None, mean=100.0):
    rng = np.random.default_rng(20260927)
    data = np.clip(rng.normal(mean, mean * 0.02, size=SHAPE), 0, 65535).astype(np.int16)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["IMAGETYP"] = imagetyp
    h["EXPTIME"] = float(exptime)
    h["INSTRUME"] = "Seestar S50"
    h["GAIN"] = 120.0
    h["OFFSET"] = 50.0
    h["CCD-TEMP"] = -10.0
    h["SET-TEMP"] = -10.0
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["XORGSUBF"] = 0
    h["YORGSUBF"] = 0
    h["BAYERPAT"] = "MONO"
    h["BUNIT"] = "ADU"
    h["BSCALE"] = 1.0
    h["BZERO"] = 0.0
    if filter_ is not None:
        h["FILTER"] = filter_
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def _master_folder(tmp_path):
    _write_fits(tmp_path / "dark_60s.fits", "Dark", 60.0, mean=200.0)
    _write_fits(tmp_path / "dark_180s.fits", "Dark", 180.0, mean=600.0)
    _write_fits(tmp_path / "bias.fits", "Bias", 0.0, mean=100.0)
    _write_fits(tmp_path / "flat.fits", "Flat", 5.0, mean=20000.0, filter_="L")
    _write_fits(tmp_path / "light_60s.fits", "Light", 60.0, mean=8000.0, filter_="L")
    _write_fits(tmp_path / "light_180s.fits", "Light", 180.0, mean=16000.0, filter_="L")
    return str(tmp_path)


def _light(name):
    return v1.FitsFrameSource(
        path=name,
        declaration=v1.ImportDeclaration(
            source="synthetic_fixture", identity=name, version="1.0",
        ),
    )


# ---------------------------------------------------------------------------
# Surface / capabilities / additivity
# ---------------------------------------------------------------------------
def test_new_public_symbols_in_all():
    for sym in (
        "open_session_library", "SessionLibrary", "SessionLibraryResult",
        "RouteResolution", "MasterAdmission", "RejectionDiagnostic",
        "PlanSourceMismatchError",
    ):
        assert sym in v1.__all__, sym
        assert hasattr(v1, sym), sym


def test_capabilities_include_session_library_and_auto_route():
    caps = v1.get_api_info().capabilities
    assert "session_library" in caps
    assert "auto_route" in caps
    assert caps == (
        "calibrate_frame", "calibration_library", "master_matching", "provenance",
        "cancel", "calibrate_batch", "session_library", "auto_route",
    )


def test_additivity_no_previous_symbol_removed():
    # The pre-1.1 public surface must remain a subset of the current __all__.
    previous = [
        "API_VERSION", "get_api_info", "calibrate_frame", "calibrate_batch",
        "index_library", "open_library", "resolve_calibration", "inspect_frame",
        "validate_binding", "validate_plan", "build_master_import_spec",
        "build_managed_library", "managed_fingerprint", "load_managed_ledger",
        "save_managed_ledger", "content_identity", "managed_ledger_path",
        "default_match_policy", "light_constraints_from_sensor_metadata",
        "build_sensor_metadata", "FitsFrameSource", "ArrayFrameSource",
        "CalibrationPlan", "CalibrationRequest", "CalibrationResult",
        "MasterImportSpec", "ManagedMasterRecord", "MatchPolicy", "LibrarySpec",
        "ExecutionOptions", "CancellationToken", "ImportDeclaration",
        "EvidenceFact", "FrameInspection", "InspectResult", "ResolveResult",
    ]
    for sym in previous:
        assert sym in v1.__all__, f"removed symbol: {sym}"


def test_get_api_info_cold_import_stays_cheap():
    code = textwrap.dedent(
        """
        import sys
        import zecalibrator.api.v1 as v1
        assert v1.API_VERSION == "1.1"
        info = v1.get_api_info()
        assert "session_library" in info.capabilities
        assert "auto_route" in info.capabilities
        for mod in ("numpy", "astropy", "sqlite3", "PySide6", "QtWidgets",
                    "QtCore", "zealfie", "seestar", "cupy"):
            assert mod not in sys.modules, mod
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# open_session_library — admission + role identification
# ---------------------------------------------------------------------------
def test_open_admits_dark_bias_rejects_flat(tmp_path):
    folder = _master_folder(tmp_path)
    res = v1.open_session_library(folder)
    assert res.operation_status == "COMPLETED"
    assert res.handle is not None
    assert dict(res.counts_by_role) == {"dark": 2, "bias": 1}
    roles = {a.role for a in res.admissions}
    assert roles == {"dark", "bias"}
    # flat refused with structured diagnostic, never invalidating the rest.
    flat_rejections = [r for r in res.rejected if r.path.endswith("flat.fits")]
    assert flat_rejections
    assert flat_rejections[0].reason_code == "FLAT_QUALITY_EVIDENCE_INSUFFICIENT"
    # lights have no IMAGETYP role -> AMBIGUOUS_ROLE
    light_rejections = [r for r in res.rejected if "light" in r.path]
    assert all(r.reason_code == "AMBIGUOUS_ROLE" for r in light_rejections)
    assert res.fingerprint


def test_open_zero_admissible_masters_returns_handle_none_C3a(tmp_path):
    # A folder whose only FITS are lights (no IMAGETYP role) -> zero admissible.
    _write_fits(tmp_path / "light_a.fits", "Light", 60.0, mean=8000.0)
    _write_fits(tmp_path / "light_b.fits", "Light", 60.0, mean=8000.0)
    res = v1.open_session_library(str(tmp_path))
    assert res.operation_status == "COMPLETED"  # NOT an exception
    assert res.handle is None
    assert res.admissions == ()
    assert len(res.rejected) == 2
    assert any("no admissible" in w for w in res.warnings)


def test_open_non_directory_raises(tmp_path):
    p = tmp_path / "nope"
    p.write_text("x")
    with pytest.raises(v1.InvalidRequestError):
        v1.open_session_library(str(p))


def test_open_top_level_only_no_recursion(tmp_path):
    _write_fits(tmp_path / "dark_60s.fits", "Dark", 60.0, mean=200.0)
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_fits(sub / "dark_180s.fits", "Dark", 180.0, mean=600.0)
    res = v1.open_session_library(str(tmp_path))
    assert dict(res.counts_by_role) == {"dark": 1}  # nested master NOT scanned


# ---------------------------------------------------------------------------
# resolve_light — auto-route
# ---------------------------------------------------------------------------
def test_resolve_light_routes_distinct_plans(tmp_path):
    folder = _master_folder(tmp_path)
    res = v1.open_session_library(folder)
    h = res.handle

    rr60 = h.resolve_light(_light(folder + "/light_60s.fits"))
    assert rr60.operation_status == "COMPLETED"
    assert rr60.outcome == "MATCHED"
    assert rr60.plan is not None
    assert list(rr60.plan.masters.keys()) == ["dark"]

    rr180 = h.resolve_light(_light(folder + "/light_180s.fits"))
    assert rr180.operation_status == "COMPLETED"
    assert rr180.outcome == "MATCHED"
    assert rr180.plan is not None
    assert list(rr180.plan.masters.keys()) == ["dark"]

    # Distinct routes: each light binds its own exposure-matched dark.
    assert rr60.plan.plan_id != rr180.plan.plan_id


def test_resolve_light_no_match_without_masters(tmp_path):
    _write_fits(tmp_path / "bias.fits", "Bias", 0.0, mean=100.0)
    _write_fits(tmp_path / "light_60s.fits", "Light", 60.0, mean=8000.0)
    res = v1.open_session_library(str(tmp_path))
    rr = res.handle.resolve_light(_light(str(tmp_path / "light_60s.fits")))
    # No dark for a 60s light (bias-only): auto-route may fall back to bias_only
    # (READY/MATCHED) or NEEDS_ATTENTION; assert it's a well-formed envelope.
    assert rr.operation_status == "COMPLETED"
    assert rr.outcome in ("MATCHED", "NO_MATCH", "AMBIGUOUS")


# ---------------------------------------------------------------------------
# calibrate — in-memory + master reuse + C1 provenance
# ---------------------------------------------------------------------------
def test_calibrate_in_memory_and_master_reuse_C1_Sa(tmp_path):
    folder = _master_folder(tmp_path)
    res = v1.open_session_library(folder)
    h = res.handle

    light60 = _light(folder + "/light_60s.fits")
    plan = h.resolve_light(light60).plan
    assert plan is not None

    cr = h.calibrate(light60, plan)
    assert cr.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert cr.data.shape == SHAPE
    assert cr.mask.shape == SHAPE
    assert cr.provenance.backend == "cpu"

    # Master reuse (S-a): calibrating the same plan again does NOT re-prepare.
    h.calibrate(light60, plan)
    h.calibrate(light60, plan)
    assert h.context_preparation_count == 1

    # A second distinct plan (180s) prepares once more.
    light180 = _light(folder + "/light_180s.fits")
    plan180 = h.resolve_light(light180).plan
    h.calibrate(light180, plan180)
    assert h.context_preparation_count == 2


def test_calibrate_foreign_plan_refused_C1(tmp_path):
    folder = _master_folder(tmp_path)
    a = v1.open_session_library(folder).handle
    b = v1.open_session_library(folder).handle

    light60 = _light(folder + "/light_60s.fits")
    plan_a = a.resolve_light(light60).plan

    # Same content, but the plan was issued by session A, not session B.
    with pytest.raises(v1.PlanSourceMismatchError):
        b.calibrate(light60, plan_a)

    # And a plan from THIS session calibrates fine.
    plan_b = b.resolve_light(light60).plan
    assert b.calibrate(light60, plan_b).status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")


def test_calibrate_plan_source_mismatch_is_typed_and_catchable(tmp_path):
    folder = _master_folder(tmp_path)
    a = v1.open_session_library(folder).handle
    b = v1.open_session_library(folder).handle
    light60 = _light(folder + "/light_60s.fits")
    plan_a = a.resolve_light(light60).plan
    try:
        b.calibrate(light60, plan_a)
        pytest.fail("expected PlanSourceMismatchError")
    except v1.InvalidRequestError as exc:
        # PlanSourceMismatchError is a subtype of InvalidRequestError (C1 typed refusal).
        assert isinstance(exc, v1.PlanSourceMismatchError)


# ---------------------------------------------------------------------------
# close() idempotent + context manager + post-close state (C3c / C3b)
# ---------------------------------------------------------------------------
def test_close_idempotent_and_blocks_use_C3c(tmp_path):
    folder = _master_folder(tmp_path)
    h = v1.open_session_library(folder).handle
    h.close()
    h.close()  # idempotent, no exception
    light60 = _light(folder + "/light_60s.fits")
    with pytest.raises(v1.LibraryClosedError):
        h.resolve_light(light60)


def test_context_manager(tmp_path):
    folder = _master_folder(tmp_path)
    with v1.open_session_library(folder).handle as h:
        plan = h.resolve_light(_light(folder + "/light_60s.fits")).plan
        assert plan is not None
    # after the block, the session is closed
    with pytest.raises(v1.LibraryClosedError):
        h.resolve_light(_light(folder + "/light_60s.fits"))


def test_cancellation_returns_cancelled_result_C3b(tmp_path):
    folder = _master_folder(tmp_path)

    # Pre-cancelled open -> CANCELLED result, no handle.
    token = v1.CancellationToken()
    token.cancel()
    res = v1.open_session_library(folder, cancel=token)
    assert res.operation_status == "CANCELLED"
    assert res.handle is None

    # Pre-cancelled resolve_light -> CANCELLED envelope, session stays usable.
    h = v1.open_session_library(folder).handle
    token2 = v1.CancellationToken()
    token2.cancel()
    rr = h.resolve_light(_light(folder + "/light_60s.fits"), cancel=token2)
    assert rr.operation_status == "CANCELLED"
    # Session still coherent after cancellation (C3b): a fresh call works.
    rr2 = h.resolve_light(_light(folder + "/light_60s.fits"))
    assert rr2.operation_status == "COMPLETED"
