"""ZC-G2C-THERMAL-FLAT-SAFETY bounded corrective gate tests.

Two bounded defects, each with its own matrix plus identity/science regression:

A. THERMAL — the cooling SETPOINT (SET-TEMP) is a distinct matching criterion
   from the measured CCD-TEMP. When both sides know the setpoint, it governs
   COMPLETELY; when it is not known on both sides the tier degrades honestly.
B. RAW FLAT-ONLY — a RAW sensor-domain light with NO qualified additive
   correction must never receive a flat alone; the flat is skipped and the
   route is a pixel-identical passthrough.

Also: digest byte-stability (the setpoint is a matching criterion, never part of
the frozen descriptor/plan identity) and a numerical science regression.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    light,
    policy,
    pool,
    request,
)
from zecalibrator.application.library import LibrarySnapshot
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.descriptors import ProcessingProvenance
from zecalibrator.core.equations import (
    additive_numerator,
    final_plane,
    normalize_flat_response,
)
from zecalibrator.core.matching import (
    OUTCOME_MATCHED,
    OUTCOME_NO_MATCH,
    match_calibration,
)
from zecalibrator.core.plans import (
    CalibrationPlan,
    CalibrationRequest,
    Candidate,
    PolicyParameters,
    VersionSet,
)
from zecalibrator.core.routes import ADDITIVE_PREREQUISITE_MISSING, OUTCOME_READY


def snapshot(**roles) -> LibrarySnapshot:
    return LibrarySnapshot(
        revision="r",
        schema_version="zecalibrator.library.v1",
        candidates={k: tuple(v) for k, v in roles.items()},
    )


def _corrected_flat(cid, content_sha="a" * 64):
    pp = ProcessingProvenance(
        source="synthetic_fixture", additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    f = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp,
        content_sha256=content_sha, mask_identity="b" * 64,
    )
    return candidate(cid, f)


# ===========================================================================
# A. THERMAL MATRIX (Standard contract)
# ===========================================================================
def _standard_dark_match(light_acq, dark_acq, additive_mode="dark_incl_bias"):
    lt = light(acquisition=light_acq)
    dk = descriptor("dark", "included", acquisition_obj=dark_acq)
    return match_calibration(
        lt, request(additive_mode), pool(dark=[candidate("d1", dk)]), policy(),
        standard_contract=True,
    )


def test_t1_setpoints_equal_ccd_equal_compatible():
    r = _standard_dark_match(
        acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
        acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
    )
    assert r.outcome == OUTCOME_MATCHED


def test_t2_setpoints_equal_ccd_different_compatible_witness():
    # The witness: both frames requested SET-TEMP=-10; one measured CCD-TEMP is
    # -10.5 and the other -10.0. Equal known setpoints govern — compatible.
    r = _standard_dark_match(
        acquisition(temperature_c=-10.5, temperature_setpoint_c=-10.0),
        acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
    )
    assert r.outcome == OUTCOME_MATCHED
    # CCD-TEMP must NOT be emitted as a blocking reason here.
    assert "TEMPERATURE_MISMATCH" not in r.reason_codes


def test_t3_different_setpoints_mismatch():
    r = _standard_dark_match(
        acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
        acquisition(temperature_c=-10.0, temperature_setpoint_c=-20.0),
    )
    assert r.outcome == OUTCOME_NO_MATCH
    assert "TEMPERATURE_MISMATCH" in r.reason_codes


def test_t4_light_setpoint_missing_no_invented_setpoint():
    r = _standard_dark_match(
        acquisition(temperature_c=-10.5, temperature_setpoint_c=None),
        acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
    )
    # Missing setpoint degrades honestly (UNVERIFIED non-blocking), never a hard
    # mismatch caused only by a CCD-TEMP difference.
    assert r.outcome == OUTCOME_MATCHED
    assert "TEMPERATURE_MISMATCH" not in r.reason_codes
    assert any(x.field == "acquisition.temperature_setpoint_c" for x in r.unverified)


def test_t5_dark_setpoint_missing_no_invented_setpoint():
    r = _standard_dark_match(
        acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
        acquisition(temperature_c=-10.5, temperature_setpoint_c=None),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert "TEMPERATURE_MISMATCH" not in r.reason_codes
    assert any(x.field == "acquisition.temperature_setpoint_c" for x in r.unverified)


def test_t6_both_setpoints_missing_standard_honest_non_fragile():
    # Both setpoints unknown: Standard stays honest and non-fragile — a small
    # CCD-TEMP difference must NOT produce a hard mismatch.
    r = _standard_dark_match(
        acquisition(temperature_c=-10.5, temperature_setpoint_c=None),
        acquisition(temperature_c=-10.0, temperature_setpoint_c=None),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert "TEMPERATURE_MISMATCH" not in r.reason_codes
    assert any(x.field == "acquisition.temperature_setpoint_c" for x in r.unverified)


def test_t7_malformed_ccd_temp_does_not_poison_setpoint_matching():
    # CCD-TEMP malformed (temperature_c=None) but both setpoints valid and equal:
    # the setpoint comparison must not depend on a parseable CCD-TEMP.
    r = _standard_dark_match(
        acquisition(temperature_c=None, temperature_setpoint_c=-10.0),
        acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert "TEMPERATURE_MISMATCH" not in r.reason_codes


def test_strict_tier_preserves_ccd_temp_clause_when_setpoint_unknown():
    # Strict contract (standard_contract=False): when the setpoint is not known
    # on both sides, the pre-existing CCD-TEMP clause is preserved byte-for-byte.
    lt = light(acquisition=acquisition(temperature_c=20.0, temperature_setpoint_c=None))
    dk = descriptor(
        "dark", "included",
        acquisition_obj=acquisition(temperature_c=20.0 + 1e-5, temperature_setpoint_c=None),
    )
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "TEMPERATURE_MISMATCH" in r.reason_codes


def test_strict_tier_setpoint_governs_when_both_known():
    # Strict contract: both setpoints known and equal -> compatible even though
    # the measured CCD-TEMP differs.
    lt = light(acquisition=acquisition(temperature_c=-10.5, temperature_setpoint_c=-10.0))
    dk = descriptor(
        "dark", "included",
        acquisition_obj=acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
    )
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_MATCHED


# ===========================================================================
# B. ROUTING MATRIX — RAW flat-only guard
# ===========================================================================
def test_r1_raw_dark_flat_gets_dark_and_flat():
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="d" * 64, mask_identity="e" * 64))
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(dark=[dark], flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert set(res.plan.composition.applied_roles) == {"dark", "flat"}
    assert res.plan.composition.flat_applied is True


def test_r2_raw_incompatible_dark_flat_is_passthrough():
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", acquisition_obj=acquisition(gain=999)))
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(dark=[dark], flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.composition.flat_applied is False
    assert "flat" not in res.plan.composition.applied_roles
    assert any(r.code == ADDITIVE_PREREQUISITE_MISSING for r in res.reasons)


def test_r3_raw_no_dark_flat_not_applied_passthrough():
    lt = light()
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.composition.flat_applied is False
    assert res.plan.composition.additive_state == "none"
    assert res.plan.composition.applied_roles == ()
    assert any(r.code == ADDITIVE_PREREQUISITE_MISSING for r in res.reasons)


def test_r4_raw_dark_only_no_flat():
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="d" * 64, mask_identity="e" * 64))
    res = resolve_route(lt, snapshot(dark=[dark]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.composition.applied_roles == ("dark",)
    assert res.plan.composition.flat_applied is False


def test_r5_raw_no_applicable_master_pixel_identical_passthrough():
    lt = light()
    res = resolve_route(lt, snapshot(), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.composition.level == "NONE"
    assert res.plan.composition.applied_roles == ()
    assert res.plan.masters == {}


def test_r6_raw_valid_bias_plus_flat_is_valid():
    # A real additive correction (bias_only) prepares the light, so the flat is
    # applied — the ACTUAL existing science contract, tested here.
    lt = light()
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001, content_sha256="c" * 64, mask_identity="d" * 64))
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(bias=[bias], flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert set(res.plan.composition.applied_roles) == {"bias", "flat"}
    assert res.plan.composition.flat_applied is True
    assert res.plan.composition.additive_state == "bias_only"


def test_h1_flat_skip_composition_reason_is_additive_prereq():
    # The durable composition block must record the precise skip reason, not the
    # generic NOT_REQUIRED.
    lt = light()
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    skipped = {s.role: s.reason_code for s in res.plan.composition.skipped_roles}
    assert skipped.get("flat") == "ADDITIVE_PREREQUISITE_MISSING"


def test_h2_no_additive_prereq_reason_on_dark_flat():
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="d" * 64, mask_identity="e" * 64))
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(dark=[dark], flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert set(res.plan.composition.applied_roles) == {"dark", "flat"}
    assert all(r.code != ADDITIVE_PREREQUISITE_MISSING for r in res.reasons)


def test_h2_no_additive_prereq_reason_on_bias_flat():
    lt = light()
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001, content_sha256="c" * 64, mask_identity="d" * 64))
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(bias=[bias], flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert set(res.plan.composition.applied_roles) == {"bias", "flat"}
    assert all(r.code != ADDITIVE_PREREQUISITE_MISSING for r in res.reasons)


# ===========================================================================
# C. NUMERICAL SCIENCE REGRESSION
# ===========================================================================
def test_naive_flat_division_modulates_pedestal_and_additive_flat_recovers():
    shape = (64, 64)
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    # A genuinely spatially varying response (radial-ish falloff).
    cx, cy = shape[1] / 2.0, shape[0] / 2.0
    radial_response = (1.0 + 0.6 * ((x - cx) ** 2 + (y - cy) ** 2) / (cx * cy)).astype(np.float32)
    true_signal = np.float32(1000.0)
    pedestal = np.float32(120.0)

    raw = (true_signal * radial_response + pedestal).astype(np.float32)

    # Naive flat-only: raw / radial_response = true_signal + pedestal / response,
    # which is spatially modulated (not a constant true_signal).
    naive = (raw / radial_response).astype(np.float32)
    assert float(np.ptp(naive)) > 1.0  # the pedestal term is spatially modulated

    # Supported valid additive+flat sequence: (raw - dark) / R, R = flat/median.
    dark = np.full(shape, pedestal, dtype=np.float32)
    A = additive_numerator(raw, "dark_incl_bias", dark_inc=dark)
    norm = normalize_flat_response(radial_response, cfa_phase="mono", roi_origin=(0, 0))
    assert norm.usable
    C = final_plane(A, norm.R)

    # C = true_signal * median(radial_response) within float32 tolerance.
    expected = true_signal * np.float32(norm.scalars["mono"])
    assert np.allclose(C, expected, rtol=5e-5, atol=1e-3)


# ===========================================================================
# D. DIGEST STABILITY — the setpoint is NOT part of identity
# ===========================================================================
def _plan(masters, lt=None):
    lt = lt or light()
    return CalibrationPlan.build(
        request=request(),
        light_constraints=lt,
        masters=masters,
        policy_parameters=PolicyParameters(
            exposure_tolerance=policy().exposure_tolerance,
            temperature_tolerance=policy().temperature_tolerance,
            flat_quality_policy=policy().flat_quality_policy,
        ),
        versions=VersionSet(),
    )


def _binding(desc):
    from zecalibrator.core.descriptors import DescriptorSnapshot
    from zecalibrator.core.plans import FitsFileLocator, MaskPayloadLocator, MasterBinding

    return MasterBinding(
        descriptor_id=desc.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(desc),
        content_sha256=desc.content_sha256,
        size_bytes=desc.size_bytes,
        hdu=desc.hdu,
        mask_identity=desc.mask_identity,
        locators=(FitsFileLocator(path="/lib/d.fits", hdu=desc.hdu),),
        mask_locator=MaskPayloadLocator(path="/lib/d.mask"),
    )


def test_descriptor_id_stable_under_setpoint():
    d_none = descriptor(
        "dark", "included",
        acquisition_obj=acquisition(temperature_c=-10.0, temperature_setpoint_c=None),
    )
    d_sp = descriptor(
        "dark", "included",
        acquisition_obj=acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0),
    )
    assert d_none.descriptor_id == d_sp.descriptor_id


def test_plan_id_stable_under_setpoint():
    lt_none = light(acquisition=acquisition(temperature_c=-10.0, temperature_setpoint_c=None))
    lt_sp = light(acquisition=acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0))
    d_none = descriptor("dark", "included", acquisition_obj=acquisition(temperature_c=-10.0, temperature_setpoint_c=None))
    d_sp = descriptor("dark", "included", acquisition_obj=acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0))
    p_none = _plan({"dark": _binding(d_none)}, lt=lt_none)
    p_sp = _plan({"dark": _binding(d_sp)}, lt=lt_sp)
    assert p_none.plan_id == p_sp.plan_id


def test_setpoint_serialized_but_excluded_from_plan_digest():
    # The setpoint is fully SERIALIZED (survives the library index round-trip),
    # but the plan digest projects explicit acquisition subpaths that EXCLUDE it.
    acq = acquisition(temperature_c=-10.0, temperature_setpoint_c=-10.0)
    assert acq.to_dict()["temperature_setpoint_c"] == -10.0
    from zecalibrator.core.digests import PLAN_PROJECTION

    acq_paths = [p for p in PLAN_PROJECTION if p.startswith("light_constraints.acquisition")]
    # No whole-subtree entry anymore; exactly the 10 explicit subpaths, no setpoint.
    assert "light_constraints.acquisition" not in PLAN_PROJECTION
    assert set(acq_paths) == {
        "light_constraints.acquisition.gain",
        "light_constraints.acquisition.offset",
        "light_constraints.acquisition.readout_mode",
        "light_constraints.acquisition.adc_mode",
        "light_constraints.acquisition.temperature_c",
        "light_constraints.acquisition.exposure_s",
        "light_constraints.acquisition.saturation_limit_adu",
        "light_constraints.acquisition.saturation_evidence",
        "light_constraints.acquisition.bias_exposure_max_s",
        "light_constraints.acquisition.short_flat_profile",
    }


# ===========================================================================
# E. REAL M74 WITNESS (do NOT fake — real files, else NOT_RUN)
# ===========================================================================
M74_ROOT = "/media/tristan/X10 Pro/M74"


def _read_kw(path, kw, default=None):
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        return hdul[0].header.get(kw, default)


def _m74_available():
    primary = os.path.join(
        M74_ROOT, "data", "Light_M 74_60.0s_Bin1_irct_20260915-021456_83deg_0041.fit"
    )
    control = os.path.join(
        M74_ROOT, "data", "Light_M 74_60.0s_Bin1_irct_20260915-022328_83deg_0047.fit"
    )
    dark = os.path.join(
        M74_ROOT, "masters", "dark", "IRCUT GAIN 120 TEMP -10C", "dark",
        "MasterDark_Stack10_60.0s_Bin1_20230919-204844.fit",
    )
    flat = os.path.join(
        M74_ROOT, "masters", "Flat", "MasterFlat_Stack20_10.0ms_Bin1_irct_20260915-084028.fit",
    )
    paths = (primary, control, dark, flat)
    return all(os.path.exists(p) for p in paths), paths


@pytest.mark.skipif(
    not _m74_available()[0],
    reason="M74 real corpus not available on this host",
)
def test_real_m74_both_lights_resolve_to_dark_and_flat(tmp_path):
    import zecalibrator.api.v1 as v1
    from zecalibrator.application.library import light_constraints_from_sensor_metadata
    from zecalibrator.api.v1._io import standard_light_contract
    from zecalibrator.io.raw_decoder import decode_fits

    _, (primary, control, dark, flat) = _m74_available()

    # -- masters via the real admission path (reads SET-TEMP from FITS) -------
    def master_decl(path, *, exposure_s, temperature_c, filter_):
        return v1.ImportDeclaration(
            source="user_import", identity="m74-witness", version="1.0",
            domain="raw", units="ADU",
            detector_instance_id=None,
            detector_model=_read_kw(path, "INSTRUME"),
            gain=float(_read_kw(path, "GAIN")),
            offset=float(_read_kw(path, "OFFSET")),
            binning=(1, 1),
            # sensor_dimensions left None so the light's standard_light_contract
            # (which also leaves it None) matches as a disambiguator (both-unknown
            # -> non-blocking UNVERIFIED) rather than one-known-one-unknown MISSING.
            sensor_dimensions=None,
            orientation="identity",
            cfa_phase="RGGB",
            roi_origin=(0, 0),
            exposure_s=exposure_s,
            temperature_c=temperature_c,
            filter=filter_,
        )

    imports = [
        v1.MasterImportSpec(
            path=dark, master_type="dark", hdu=0,
            declaration=master_decl(dark, exposure_s=60.0, temperature_c=-10.0, filter_=None),
            dq_state="no_source_dq",
        ),
        v1.MasterImportSpec(
            path=flat, master_type="flat", hdu=0,
            declaration=master_decl(flat, exposure_s=0.01, temperature_c=-10.0, filter_="irct"),
            dq_state="no_source_dq",
        ),
    ]
    spec = v1.LibrarySpec(root=M74_ROOT, index_path=str(tmp_path / "m74.sqlite"))
    idx = v1.index_library(spec, imports)
    assert idx.operation_status == "COMPLETED", idx.diagnostics
    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle

    try:
        # G3: the admitted dark/flat descriptors must carry the REAL setpoint
        # (surviving the persisted index round-trip) BEFORE any composition claim.
        dark_cands = handle.snapshot.candidates["dark"]
        flat_cands = handle.snapshot.candidates["flat"]
        assert len(dark_cands) == 1
        assert len(flat_cands) == 1
        assert dark_cands[0].descriptor.acquisition.temperature_setpoint_c == -10.0
        assert flat_cands[0].descriptor.acquisition.temperature_setpoint_c == -10.0

        for light_path in (primary, control):
            decoded = decode_fits(light_path, declaration=standard_light_contract())
            lc = light_constraints_from_sensor_metadata(decoded.metadata)
            # SET-TEMP must have reached the light acquisition constraints.
            assert lc.acquisition.temperature_setpoint_c == -10.0
            res = resolve_route(lc, handle.snapshot, v1.default_match_policy())
            assert res.outcome == OUTCOME_READY, res.reasons
            assert res.plan is not None
            assert set(res.plan.composition.applied_roles) == {"dark", "flat"}
    finally:
        handle.close()
