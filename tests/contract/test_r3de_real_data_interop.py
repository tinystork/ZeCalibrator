"""R3D-E real-data end-to-end interop (owner-mandated acceptance tests).

Covers the nine owner-mandated witnesses for the float32 ADU-equivalent canonical
domain end-to-end:

1.  raw uint16 light + normalized_real dark master reach the SAME canonical
    scale before subtraction (dark -> ~8.94 ADU); result = light - dark, finite
    signed float32.
2.  a corrected_unnormalized flat (median ~0.455, no normalization proof) is
    accepted by the router AND executor, normalized-only (no flat_dark
    re-subtraction), applied as (light-dark)/R.
3.  a normalized_response flat (with proof) still routes already_normalized.
4.  a prepared flat's gain 456 vs light 120 does NOT block; geometry/CFA/filter/
    optical contradictions still block.
5.  a raw_response flat still requires its additive flat_dark/bias dependency.
6.  a dark with missing gain/temp is UNVERIFIED non-blocking; a known gain
    mismatch stays blocking.
7.  Standard light contract: a raw uint16 CFA light with no processing history
    decodes with standard_light_contract; a debayer/stretch history still blocks.
8.  float32 end-to-end: no integer round-trip/clipping in the calibrated output.
9.  (git-level) no change to equations/dq/plans/raw_decoder.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from zecalibrator.api.v1 import _io
from zecalibrator.application.executor import (
    CalibrationRequest,
    MasterBinding,
    execute_calibration,
)
from zecalibrator.api.v1.frames import inspect_frame
from zecalibrator.api.v1.models import FitsFrameSource
from zecalibrator.core.calibrate import calibrate_light
from zecalibrator.core.metadata import (
    NORMALIZED_REAL_REFERENCE,
    ImportDeclaration,
    is_normalized_real_master,
)
from zecalibrator.core.routes import OUTCOME_NEEDS_ATTENTION, OUTCOME_READY
from zecalibrator.io.raw_decoder import decode_fits


SHAPE = (8, 8)


def _write_fits(path, data, header_cards=(), *, bunit=None):
    hdu = fits.PrimaryHDU(np.ascontiguousarray(data))
    if bunit is not None:
        hdu.header["BUNIT"] = bunit
    for kw, val in header_cards:
        if kw == "HISTORY":
            hdu.header.add_history(val)
        else:
            hdu.header[kw] = val
    hdu.writeto(path, overwrite=True)
    return str(path)


def _master_domain(path, *, declaration, role="dark", flat_form=None):
    """Decode a master and run the F2 conversion gate; return (converted, record)."""
    decoded = decode_fits(
        path, declaration=declaration, admission="master", role=role, flat_form=flat_form,
    )
    return _io.convert_normalized_real_master(decoded, role=role)


def _siril_dark_decl(**overrides):
    """A producer-proven Siril dark master: no gain/offset/temperature facts."""
    base = dict(
        source="observed", identity="SIRIL-DARK", version="1.0",
        domain="raw", units="ADU",
        detector_model="SYNTH-CFA",
        binning=(1, 1), cfa_phase="mono",
        exposure_s=10.0, filter="NONE",
    )
    base.update(overrides)
    return ImportDeclaration(**base)


# ---------------------------------------------------------------------------
# 1. normalized_real dark reaches the same canonical scale before subtraction.
# ---------------------------------------------------------------------------
def test_normalized_real_dark_reaches_adu_scale_before_subtraction(tmp_path):
    # Light: raw uint16, median ~534 ADU.
    light_path = _write_fits(
        tmp_path / "light.fits",
        np.full(SHAPE, 534, dtype=np.uint16),
        bunit="ADU",
    )
    light = decode_fits(light_path, declaration=_io.standard_light_contract())

    # Dark master: float32 [0,1] normalized_real, producer-proven Siril.
    dark_path = _write_fits(
        tmp_path / "dark.fits",
        np.full(SHAPE, 0.000137, dtype=np.float32),
        header_cards=[
            ("PROGRAM", "Siril v1.2.6"),
            ("HISTORY", "median stacking without rejection, unnormalized input, unnormalized output"),
        ],
    )
    dark = decode_fits(dark_path, declaration=_siril_dark_decl(), admission="master", role="dark")

    # The decoder leaves the master [0,1] but labels it ADU; the producer
    # provenance proves normalized_real [0,1].
    assert is_normalized_real_master(dark.metadata.original_cards) == "siril"
    assert np.isclose(float(np.median(dark.data)), 0.000137, atol=1e-6)

    converted, record = _io.convert_normalized_real_master(dark, role="dark")
    assert record is not None
    assert record["storage_domain"] == "normalized_real"
    assert record["producer"] == "siril"
    assert record["producer_evidence"] == "Siril v1.2.6"
    assert record["from_domain"] == "normalized_real"
    assert record["to_domain"] == "adu_equivalent"
    assert record["scale"] == float(NORMALIZED_REAL_REFERENCE)
    assert record["scale_source"] == "producer_normalized_real_16bit_reference"
    # dark 0.000137 * 65535 ~= 8.94 ADU: same canonical scale as the light.
    assert np.isclose(float(np.median(converted.data)), 0.000137 * 65535.0, atol=1e-2)

    result = calibrate_light(
        light.data,
        additive_mode="dark_incl_bias",
        input_mask=light.mask,
        dark_inc=converted.data,
        dark_inc_mask=converted.mask,
    )
    assert result.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert result.data.dtype == np.float32
    assert np.all(np.isfinite(result.data))
    # light - dark in ADU, finite signed float32.
    expected = (534.0 - 0.000137 * 65535.0)
    assert np.allclose(result.data, expected, atol=1e-2)


# ---------------------------------------------------------------------------
# 2. corrected_unnormalized flat -> normalize_only, no proof, no flat_dark.
# ---------------------------------------------------------------------------
def test_corrected_unnormalized_flat_normalize_only(tmp_path, make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=100.0)
    dark = make_frame((4, 4), value=10.0, exposure_s=10.0)
    # corrected-but-unnormalized flat, median ~0.455 (NOT normalized to 1.0).
    finc = make_frame((4, 4), value=0.455, exposure_s=1.0)
    masters = {
        "dark": MasterBinding(role="dark", frame=dark, bias_state="included"),
        "flat": MasterBinding(role="flat", frame=finc, flat_form="corrected_unnormalized"),
    }
    # No flat_dark / bias_flat supplied: normalize_only must not consume them.
    result = execute_calibration(
        light,
        CalibrationRequest(additive_mode="dark_incl_bias", flat_mode="apply"),
        masters,
        flat_prep_mode="normalize_only",
    )
    assert result.status == "COMPLETED"
    # R = flat / median(flat) = 0.455 / 0.455 = 1.0; result = (100-10)/1.0 = 90.
    assert np.allclose(result.data, np.full((4, 4), 90.0, dtype=np.float32), atol=1e-4)
    assert result.scalars.get("mono") == pytest.approx(0.455, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. normalized_response flat (with proof) still routes already_normalized.
# ---------------------------------------------------------------------------
def test_normalized_response_flat_still_already_normalized():
    from _phase4_fixtures import (
        acquisition,
        candidate,
        descriptor,
        geo,
        light,
        policy,
    )
    from zecalibrator.application.library import LibrarySnapshot
    from zecalibrator.application.routes import resolve_route
    from zecalibrator.core.descriptors import NormalizationProvenance, NormalizationScalars, ProcessingProvenance

    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    sc = NormalizationScalars(mono=1.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
        normalization=NormalizationProvenance(algorithm="median", population="mono-valid", scalars=sc),
    )
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0,
        flat_form="normalized_response", normalization_algorithm="median",
        normalization_scalars=sc, pixel_domain="normalized_response",
        physical_units="dimensionless", filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", processing=pp,
        geometry=geo(cfa_phase="mono"),
    )
    snap = LibrarySnapshot(
        revision="r", schema_version="zecalibrator.library.v1",
        candidates={
            "dark": (candidate("d1", dk),),
            "flat": (candidate("f1", flat),),
        },
    )
    res = resolve_route(lt, snap, policy())
    assert res.outcome == OUTCOME_READY
    assert res.route.flat_mode == "apply"
    assert res.route.flat_prep_mode == "already_normalized"
    assert "flat" in res.plan.masters


def _policy():
    from _phase4_fixtures import policy as _policy_fixture

    return _policy_fixture()


# ---------------------------------------------------------------------------
# 4. prepared flat gain 456 vs light 120 does NOT block.
# ---------------------------------------------------------------------------
def test_prepared_flat_gain_mismatch_does_not_block():
    from _phase4_fixtures import acquisition, candidate, descriptor, light, policy
    from zecalibrator.application.library import LibrarySnapshot
    from zecalibrator.application.routes import resolve_route
    from zecalibrator.core.descriptors import ProcessingProvenance

    lt = light()  # gain=100
    dk = descriptor("dark", "included", exposure_s=10.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    flat = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, exposure_s=1.0,
        acquisition_obj=acquisition(gain=456.0, offset=50.0, exposure_s=1.0),
    )
    snap = LibrarySnapshot(
        revision="r", schema_version="zecalibrator.library.v1",
        candidates={
            "dark": (candidate("d1", dk),),
            "flat": (candidate("f1", flat),),
        },
    )
    res = resolve_route(lt, snap, policy())

    # The flat's gain 456 vs light 120 is NOT a blocking reason.
    assert res.outcome == OUTCOME_READY
    assert "flat" in res.plan.masters
    assert all(r.code != "GAIN_MISMATCH" for r in res.reasons)


def test_prepared_flat_cfa_phase_mismatch_flat_passthrough():
    from _phase4_fixtures import acquisition, candidate, descriptor, geo, light, policy
    from zecalibrator.application.library import LibrarySnapshot
    from zecalibrator.application.routes import resolve_route
    from zecalibrator.core.descriptors import ProcessingProvenance

    lt = light()  # cfa_phase="mono"
    dk = descriptor("dark", "included", exposure_s=10.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    flat = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, exposure_s=1.0,
        acquisition_obj=acquisition(gain=456.0, offset=50.0, exposure_s=1.0),
        geometry=geo(cfa_phase="RGGB"),
    )
    snap = LibrarySnapshot(
        revision="r", schema_version="zecalibrator.library.v1",
        candidates={
            "dark": (candidate("d1", dk),),
            "flat": (candidate("f1", flat),),
        },
    )
    res = resolve_route(lt, snap, policy())

    # G2B R1: a CFA-phase-mismatched flat is scientifically inapplicable -> the
    # flat is not applied (passthrough) and the dark correction still applies;
    # the CFA_PHASE_MISMATCH reason is preserved (never silent).
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.route.flat_mode == "none"
    assert "dark" in res.plan.masters
    assert "flat" not in res.plan.masters
    assert any(r.code == "CFA_PHASE_MISMATCH" for r in res.reasons)


# ---------------------------------------------------------------------------
# 5. raw_response flat still requires its additive flat_dark/bias dependency.
# ---------------------------------------------------------------------------
def test_raw_response_flat_still_requires_flat_dark():
    from _phase4_fixtures import candidate, descriptor, light, policy
    from zecalibrator.application.library import LibrarySnapshot
    from zecalibrator.application.routes import resolve_route
    from zecalibrator.core.descriptors import ProcessingProvenance

    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    pp = ProcessingProvenance(source="synthetic_fixture", additive_history_state="known")
    flat = descriptor(
        "flat", "not_applicable", flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, exposure_s=1.0,
    )
    fd = descriptor("flat_dark", "included", exposure_s=1.0)
    snap = LibrarySnapshot(
        revision="r", schema_version="zecalibrator.library.v1",
        candidates={
            "dark": (candidate("d1", dk),),
            "flat": (candidate("f1", flat),),
            "flat_dark": (candidate("fd1", fd),),
        },
    )
    res = resolve_route(lt, snap, policy())

    # Standard never auto-constructs a flat_dark from a raw flat: no silent
    # bypass; the flat is unsupported without its additive dependency.
    assert res.outcome == OUTCOME_NEEDS_ATTENTION
    assert res.plan is None
    assert any(
        r.code in ("FLAT_UNSUPPORTED_RAW", "FLAT_UNUSABLE", "FLAT_ADDITIVE_DEPENDENCY_MISSING")
        for r in res.reasons
    )


# ---------------------------------------------------------------------------
# 6. dark missing gain/temp -> UNVERIFIED non-blocking; known mismatch blocks.
# ---------------------------------------------------------------------------
def test_dark_missing_gain_temp_unverified_non_blocking():
    from _phase4_fixtures import acquisition, candidate, descriptor, geo, light, policy
    from zecalibrator.application.library import LibrarySnapshot
    from zecalibrator.application.routes import resolve_route
    from zecalibrator.core.descriptors import ProcessingProvenance

    lt = light()
    dk = descriptor(
        "dark", "included",
        processing=ProcessingProvenance(source="synthetic_fixture"),
        acquisition_obj=acquisition(gain=None, offset=None, temperature_c=None),
        geometry=geo(orientation=None, roi_origin=None),
    )
    snap = LibrarySnapshot(
        revision="r", schema_version="zecalibrator.library.v1",
        candidates={"dark": (candidate("d1", dk),)},
    )
    res = resolve_route(lt, snap, policy())

    assert res.outcome == OUTCOME_READY
    fields = {r.field for r in res.unverified}
    assert "acquisition.gain" in fields
    assert "acquisition.offset" in fields
    assert "acquisition.temperature_setpoint_c" in fields
    assert all(not r.blocking for r in res.unverified)


def test_dark_known_gain_mismatch_passthrough():
    from _phase4_fixtures import acquisition, candidate, descriptor, light, policy
    from zecalibrator.application.library import LibrarySnapshot
    from zecalibrator.application.routes import resolve_route
    from zecalibrator.core.descriptors import ProcessingProvenance

    lt = light()
    dk = descriptor(
        "dark", "included",
        processing=ProcessingProvenance(source="synthetic_fixture"),
        acquisition_obj=acquisition(gain=999.0),
    )
    snap = LibrarySnapshot(
        revision="r", schema_version="zecalibrator.library.v1",
        candidates={"dark": (candidate("d1", dk),)},
    )
    res = resolve_route(lt, snap, policy())

    # G2B R1: a known dark gain mismatch makes the dark inapplicable -> READY
    # passthrough (never a hard failure); GAIN_MISMATCH stays blocking
    # (candidate-level rejection) and is preserved.
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.composition.level == "NONE"
    assert any(r.code == "GAIN_MISMATCH" and r.blocking for r in res.reasons)


# ---------------------------------------------------------------------------
# 7. Standard light contract.
# ---------------------------------------------------------------------------
def test_standard_light_contract_decodes_raw_cfa_light(tmp_path):
    # A raw uint16 CFA (Bayer) light with no processing history decodes with the
    # standard_light_contract (no UNKNOWN_DOMAIN).
    light_path = _write_fits(
        tmp_path / "cfa_light.fits",
        np.full(SHAPE, 534, dtype=np.uint16),
        header_cards=[("BAYERPAT", "RGGB")],
        bunit="ADU",
    )
    result = inspect_frame(FitsFrameSource(path=light_path, declaration=None))
    assert result.operation_status == "COMPLETED", result.reason_code
    md = result.inspection.metadata
    assert md.declaration is not None
    assert md.declaration.source == _io.STANDARD_LIGHT_CONTRACT_SOURCE
    assert md.raw_domain_declaration == "raw"
    assert md.units == "ADU"
    assert md.cfa_phase == "RGGB"


def test_standard_light_contract_debayered_still_blocks(tmp_path):
    light_path = _write_fits(
        tmp_path / "debayer_light.fits",
        np.full(SHAPE, 534, dtype=np.uint16),
        header_cards=[("HISTORY", "frame was debayered")],
        bunit="ADU",
    )
    result = inspect_frame(FitsFrameSource(path=light_path, declaration=None))
    assert result.operation_status == "FAILED"
    assert result.reason_code == "PROCESSED_HISTORY"


def test_standard_light_contract_stretch_still_blocks(tmp_path):
    light_path = _write_fits(
        tmp_path / "stretch_light.fits",
        np.full(SHAPE, 534, dtype=np.uint16),
        header_cards=[("HISTORY", "stretched for display")],
        bunit="ADU",
    )
    result = inspect_frame(FitsFrameSource(path=light_path, declaration=None))
    assert result.operation_status == "FAILED"
    assert result.reason_code == "PROCESSED_HISTORY"


# ---------------------------------------------------------------------------
# 8. float32 end-to-end: no integer round-trip/clipping.
# ---------------------------------------------------------------------------
def test_float32_end_to_end_no_integer_roundtrip(tmp_path):
    light_path = _write_fits(
        tmp_path / "light.fits",
        np.full(SHAPE, 534, dtype=np.uint16),
        bunit="ADU",
    )
    light = decode_fits(light_path, declaration=_io.standard_light_contract())

    dark_path = _write_fits(
        tmp_path / "dark.fits",
        np.full(SHAPE, 0.000137, dtype=np.float32),
        header_cards=[
            ("PROGRAM", "Siril v1.2.6"),
            ("HISTORY", "median stacking, unnormalized output"),
        ],
    )
    dark = decode_fits(dark_path, declaration=_siril_dark_decl(), admission="master", role="dark")
    converted, _ = _io.convert_normalized_real_master(dark, role="dark")

    result = calibrate_light(
        light.data,
        additive_mode="dark_incl_bias",
        input_mask=light.mask,
        dark_inc=converted.data,
        dark_inc_mask=converted.mask,
    )
    assert result.data.dtype == np.float32
    # The result is NOT integer-quantized: it carries the fractional ADU
    # remainder 534 - 8.979... (not clipped/rounded to an integer).
    assert not np.allclose(result.data, np.round(result.data))
    assert np.all(np.isfinite(result.data))
    # Signed float32: the subtraction may produce a value far from the uint16
    # light range but still finite (no clipping to the input integer domain).
    assert result.data.dtype == np.float32


# ===========================================================================
# Rework-1: owner-corrected domain rule (producer-qualified storage semantics).
# ===========================================================================

_SIRIL_FLOAT_ID = [("PROGRAM", "Siril v1.2.6")]
_SIRIL_FLOAT_OUTPUTRANGE = [
    ("PROGRAM", "Siril v1.2.6"),
    ("HISTORY", "median stacking, normalized input, unnormalized output"),
]
_PIXINSIGHT_FLOAT_ID = [("PROGRAM", "PixInsight 1.9.3")]
_PIXINSIGHT_FLOAT_OUTPUTRANGE = [
    ("PROGRAM", "PixInsight 1.9.3"),
    ("HISTORY", "ImageIntegration.outputRangeHigh: 1.00000000e+00"),
]


def test_d1_siril_float_identity_absent_scale_is_normalized_real(tmp_path):
    p = _write_fits(tmp_path / "m.fits", np.full(SHAPE, 0.5, dtype=np.float32), header_cards=_SIRIL_FLOAT_ID)
    conv, rec = _master_domain(p, declaration=_siril_dark_decl())
    assert rec is not None
    assert rec["storage_domain"] == "normalized_real"
    assert rec["producer"] == "siril"
    assert np.isclose(float(np.median(conv.data)), 0.5 * 65535.0, atol=1.0)


def test_d2_siril_integer_not_normalized_real(tmp_path):
    p = _write_fits(tmp_path / "m.fits", np.full(SHAPE, 100, dtype=np.int16), header_cards=_SIRIL_FLOAT_ID)
    conv, rec = _master_domain(p, declaration=_siril_dark_decl())
    assert rec is None
    assert conv is not None  # decoded frame returned unchanged


def test_d3_siril_float_non_identity_scale_not_normalized_real(tmp_path):
    p = _write_fits(
        tmp_path / "m.fits", np.full(SHAPE, 0.5, dtype=np.float32),
        header_cards=[("PROGRAM", "Siril v1.2.6"), ("BSCALE", 65535.0)],
    )
    decoded = decode_fits(p, declaration=_siril_dark_decl(), admission="master", role="dark")
    conv, rec = _master_domain(p, declaration=_siril_dark_decl())
    assert rec is None
    # The frame is returned unchanged (no additional ×65535 beyond the
    # decoder's own bscale application).
    assert conv is not None
    assert np.array_equal(conv.data, decoded.data)


def test_d4_pixinsight_float_identity_without_outputrange_is_normalized_real(tmp_path):
    p = _write_fits(tmp_path / "m.fits", np.full(SHAPE, 0.25, dtype=np.float32), header_cards=_PIXINSIGHT_FLOAT_ID)
    conv, rec = _master_domain(p, declaration=_siril_dark_decl())
    assert rec is not None
    assert rec["producer"] == "pixinsight"
    assert np.isclose(float(np.median(conv.data)), 0.25 * 65535.0, atol=1.0)


def test_d5_pixinsight_float_identity_with_outputrange_is_normalized_real(tmp_path):
    p = _write_fits(tmp_path / "m.fits", np.full(SHAPE, 0.25, dtype=np.float32), header_cards=_PIXINSIGHT_FLOAT_OUTPUTRANGE)
    conv, rec = _master_domain(p, declaration=_siril_dark_decl())
    assert rec is not None
    assert rec["producer"] == "pixinsight"
    assert np.isclose(float(np.median(conv.data)), 0.25 * 65535.0, atol=1.0)


def test_d6_generic_writer_float_identity_not_normalized_real(tmp_path):
    p = _write_fits(tmp_path / "m.fits", np.full(SHAPE, 0.5, dtype=np.float32), header_cards=[])
    conv, rec = _master_domain(p, declaration=_siril_dark_decl())
    assert rec is None


def test_d7_history_stacking_strings_only_not_normalized_real(tmp_path):
    # Siril stacking HISTORY strings (normalized/unnormalized input/output) are
    # NOT consulted: with a generic writer (no recognized producer) they must
    # never trigger conversion, even though the old naive substring matcher
    # would have matched "normalized input".
    for history in ("normalized input", "unnormalized input", "denormalized input"):
        p = _write_fits(
            tmp_path / f"m_{history.replace(' ', '_')}.fits",
            np.full(SHAPE, 0.5, dtype=np.float32),
            header_cards=[("HISTORY", history)],
        )
        _, rec = _master_domain(p, declaration=_siril_dark_decl())
        assert rec is None, history


def test_d8_prepared_flat_gain_exemption_both_tiers():
    from _phase4_fixtures import acquisition, candidate, descriptor, light, policy, pool, request
    from zecalibrator.core.descriptors import ProcessingProvenance
    from zecalibrator.core.matching import match_calibration

    lt = light()  # gain=100
    dk = descriptor("dark", "included", exposure_s=10.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    flat = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, exposure_s=1.0,
        acquisition_obj=acquisition(gain=456.0, offset=50.0, exposure_s=1.0),
    )
    pools = pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)])

    # Strict tier (standard_contract=False): gain 456 vs 120 must NOT block.
    strict = match_calibration(lt, request("dark_incl_bias", "apply"), pools, policy())
    assert strict.outcome == "MATCHED"
    assert "GAIN_MISMATCH" not in strict.reason_codes

    # Standard tier (standard_contract=True): gain 456 vs 120 must NOT block.
    standard = match_calibration(
        lt, request("dark_incl_bias", "apply"), pools, policy(), standard_contract=True,
    )
    assert standard.outcome == "MATCHED"
    assert "GAIN_MISMATCH" not in standard.reason_codes


def test_d9_no_gain_exemption_leak_to_dependencies_or_dark():
    from _phase4_fixtures import acquisition, candidate, descriptor, light, policy, pool, request
    from zecalibrator.core.descriptors import ProcessingProvenance
    from zecalibrator.core.matching import match_calibration

    lt = light()  # gain=100

    # (a) raw_response flat <-> flat_dark gain mismatch must still block.
    flat_proc = ProcessingProvenance(source="synthetic_fixture", additive_history_state="known")
    raw_flat = descriptor(
        "flat", "not_applicable", flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=flat_proc, exposure_s=1.0,
        acquisition_obj=acquisition(gain=100.0, offset=50.0, exposure_s=1.0),
    )
    fd = descriptor(
        "flat_dark", "included", exposure_s=1.0,
        acquisition_obj=acquisition(gain=999.0, offset=50.0, exposure_s=1.0),
    )
    dk = descriptor("dark", "included", exposure_s=10.0)
    r_flat = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", raw_flat)], flat_dark=[candidate("fd1", fd)]),
        policy(),
    )
    assert r_flat.outcome == "NO_MATCH"
    assert "GAIN_MISMATCH" in r_flat.reason_codes

    # (b) Light <-> dark known gain mismatch must still block.
    dark_mismatch = descriptor(
        "dark", "included", exposure_s=10.0,
        acquisition_obj=acquisition(gain=999.0),
    )
    r_dark = match_calibration(
        lt, request("dark_incl_bias"), pool(dark=[candidate("d1", dark_mismatch)]), policy(),
    )
    assert r_dark.outcome == "NO_MATCH"
    assert "GAIN_MISMATCH" in r_dark.reason_codes
