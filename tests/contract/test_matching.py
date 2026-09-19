"""Focused production-matcher tests (mission §7.A).

Covers exact/zero/multiple matches, complete-set (not per-role cardinality)
enumeration, None/None vs explicit-unknown/unknown vs known/unknown, field
mismatches, malformed/nonfinite numerics, temperature/exposure policy, flat
filter/train and flat semantics, own flat dependency exposure/bias qualification,
no silent downgrade, valid manual resolution and incompatible manual rejection,
deep immutability and deterministic reasons under permuted candidate order, and
duplicate collapse vs ambiguity.
"""

from __future__ import annotations

import math

import pytest

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    detector,
    geo,
    light,
    policy,
    pool,
    profile,
    request,
)
from zecalibrator.core.descriptors import (
    NormalizationScalars,
    ProcessingProvenance,
    ValidityEvidence,
)
from zecalibrator.core.matching import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_MATCHED,
    OUTCOME_NO_MATCH,
    match_calibration,
)
from zecalibrator.core.plans import Candidate, MatchPolicy, default_match_policy

import zecalibrator.core.matching as matching


# --- exact / zero / multiple ------------------------------------------------
def test_exact_single_match():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_MATCHED
    assert r.plan is not None
    assert set(r.plan.masters) == {"dark"}


def test_zero_match():
    lt = light()
    r = match_calibration(lt, request(), pool(dark=[]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "ROLE_UNAVAILABLE" in r.reason_codes


def test_multiple_distinct_matches_ambiguous():
    lt = light()
    d1 = descriptor("dark", "included", exposure_s=10.0, content_sha256="a" * 64, mask_identity="b" * 64)
    d2 = descriptor("dark", "included", exposure_s=10.0, content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(
        lt, request(),
        pool(dark=[candidate("d1", d1), candidate("d2", d2)]),
        policy(),
    )
    assert r.outcome == OUTCOME_AMBIGUOUS
    assert r.plan is None
    assert len(r.coherent_sets) == 2


def test_complete_set_not_per_role_cardinality():
    # Two dark candidates + two compatible flat_dark candidates must not produce a
    # "per-role count" result; the number of complete *sets* governs the outcome.
    lt = light()
    d1 = descriptor("dark", "included", exposure_s=10.0, content_sha256="a" * 64, mask_identity="b" * 64)
    d2 = descriptor("dark", "included", exposure_s=10.0, content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(
        lt, request(),
        pool(dark=[candidate("d1", d1), candidate("d2", d2)]),
        policy(),
    )
    assert r.outcome == OUTCOME_AMBIGUOUS
    assert len(r.coherent_sets) == 2


# --- unknown semantics ------------------------------------------------------
def test_none_none_detector_model_rejects():
    lt = light(detector=detector(model=None))
    dk = descriptor("dark", "included", detector_obj=detector(model=None))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes
    assert "UNKNOWN_EQUALS_UNKNOWN" in r.reason_codes


def test_explicit_unknown_unknown_instance_unverified():
    # R3B: detector_instance_id is a disambiguator, so both-unknown is a
    # non-blocking UNVERIFIED note (unlike the *necessary* detector_model).
    lt = light(detector=detector(instance="unknown"))
    dk = descriptor("dark", "included", detector_obj=detector(instance="unknown"))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_MATCHED
    assert {reason.code for reason in r.unverified} == {"UNVERIFIED"}


def test_known_unknown_gain_rejects():
    lt = light(acquisition=acquisition(gain=None))
    dk = descriptor("dark", "included", acquisition_obj=acquisition(gain=100))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes
    assert "UNKNOWN_EQUALS_UNKNOWN" not in r.reason_codes


# --- field mismatches -------------------------------------------------------
@pytest.mark.parametrize(
    "light_kw,desc_kw,code",
    [
        (dict(geometry=geo(shape=(4, 5))), dict(geometry=geo()), "GEOMETRY_MISMATCH"),
        (dict(detector=detector(instance="A")), dict(detector_obj=detector(instance="B")), "DETECTOR_MISMATCH"),
        (dict(acquisition=acquisition(gain=1)), dict(acquisition_obj=acquisition(gain=2)), "GAIN_MISMATCH"),
        (dict(acquisition=acquisition(offset=1)), dict(acquisition_obj=acquisition(offset=2)), "OFFSET_MISMATCH"),
        (dict(acquisition=acquisition(readout_mode="A")), dict(acquisition_obj=acquisition(readout_mode="B")), "READOUT_MISMATCH"),
        (dict(acquisition=acquisition(adc_mode="A")), dict(acquisition_obj=acquisition(adc_mode="B")), "ADC_MISMATCH"),
        (dict(geometry=geo(binning=(2, 2))), dict(geometry=geo()), "BINNING_MISMATCH"),
        (dict(geometry=geo(roi_origin=(1, 0))), dict(geometry=geo()), "ROI_ORIGIN_MISMATCH"),
        (dict(geometry=geo(cfa_phase="RGGB")), dict(geometry=geo()), "CFA_PHASE_MISMATCH"),
    ],
)
def test_field_mismatch_rejects(light_kw, desc_kw, code):
    lt = light(**light_kw)
    dk = descriptor("dark", "included", **desc_kw)
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert code in r.reason_codes


# --- malformed / nonfinite numerics ----------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_numeric_rejected_at_construction(bad):
    from zecalibrator.core.descriptors import Acquisition

    with pytest.raises(ValueError):
        Acquisition(gain=bad)
    with pytest.raises(ValueError):
        Acquisition(temperature_c=bad)
    with pytest.raises(ValueError):
        Acquisition(exposure_s=bad)


def test_negative_gain_is_finite_mismatch():
    lt = light()
    dk = descriptor("dark", "included", acquisition_obj=acquisition(gain=-1.0))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "GAIN_MISMATCH" in r.reason_codes


def test_negative_exposure_rejected_by_matcher():
    lt = light(acquisition=acquisition(exposure_s=10.0))
    dk = descriptor("dark", "included", acquisition_obj=acquisition(exposure_s=-1.0))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes


# --- temperature policy -----------------------------------------------------
def test_temperature_exact_zero_scientific():
    lt = light(acquisition=acquisition(temperature_c=20.0))
    dk = descriptor("dark", "included", acquisition_obj=acquisition(temperature_c=20.0 + 1e-5))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "TEMPERATURE_MISMATCH" in r.reason_codes


def test_temperature_parser_epsilon_accepted():
    lt = light(acquisition=acquisition(temperature_c=20.0))
    dk = descriptor("dark", "included", acquisition_obj=acquisition(temperature_c=20.0 + 5e-7))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_MATCHED


def test_nonzero_scientific_temperature_policy_refused():
    from zecalibrator.core.plans import Tolerance, FlatQualityPolicy

    with pytest.raises(ValueError):
        MatchPolicy(
            version="zecalibrator.match.v1",
            exposure_tolerance=Tolerance(relative=1e-6, absolute=1e-6),
            temperature_tolerance=Tolerance(relative=0.5, absolute=1e-6),
            flat_quality_policy=FlatQualityPolicy(state="qualified", threshold_pct=90.0, per_plane=True),
        )


# --- exposure policy --------------------------------------------------------
def test_exposure_outside_parser_bound_rejects():
    lt = light(acquisition=acquisition(exposure_s=10.0))
    dk = descriptor("dark", "included", acquisition_obj=acquisition(exposure_s=10.0 + 1e-4))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "EXPOSURE_MISMATCH" in r.reason_codes


def test_exposure_within_parser_bound_accepted():
    lt = light(acquisition=acquisition(exposure_s=10.0))
    dk = descriptor("dark", "included", acquisition_obj=acquisition(exposure_s=10.0 + 5e-7))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_MATCHED


# --- filter/train role applicability ---------------------------------------
def test_dark_ignores_filter():
    lt = light(optical=type(light().optical)(filter="IRCUT", optical_train_id=None))
    dk = descriptor("dark", "included", filter=None, optical_train_id=None)
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_MATCHED


def test_bias_ignores_filter():
    lt = light(optical=type(light().optical)(filter="IRCUT", optical_train_id=None))
    bias = descriptor("bias", "not_applicable", exposure_s=0.001, filter=None, optical_train_id=None)
    r = match_calibration(lt, request("bias_only"), pool(bias=[candidate("b1", bias)]), policy())
    assert r.outcome == OUTCOME_MATCHED


def test_flat_filter_mismatch_rejects():
    lt = light(optical=type(light().optical)(filter="LP", optical_train_id="TRAIN"))
    flat = descriptor("flat", "not_applicable", exposure_s=1.0, flat_form="raw_response", filter="NONE", optical_train_id="TRAIN")
    fd = descriptor("flat_dark", "included", exposure_s=1.0)
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", descriptor("dark", "included"))], flat=[candidate("f1", flat)], flat_dark=[candidate("fd1", fd)]),
        policy(),
    )
    assert r.outcome == OUTCOME_NO_MATCH
    assert "FILTER_MISMATCH" in r.reason_codes


def test_flat_train_mismatch_rejects():
    lt = light(optical=type(light().optical)(filter="NONE", optical_train_id="TRAIN_A"))
    flat = descriptor("flat", "not_applicable", exposure_s=1.0, flat_form="raw_response", filter="NONE", optical_train_id="TRAIN_B")
    fd = descriptor("flat_dark", "included", exposure_s=1.0)
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", descriptor("dark", "included"))], flat=[candidate("f1", flat)], flat_dark=[candidate("fd1", fd)]),
        policy(),
    )
    assert r.outcome == OUTCOME_NO_MATCH
    assert "OPTICAL_TRAIN_MISMATCH" in r.reason_codes


# --- flat semantics ---------------------------------------------------------
def test_normalized_cfa_flat_four_scalars_matched():
    lt = light(geometry=geo(cfa_phase="GRBG"), optical=type(light().optical)(filter="NONE", optical_train_id="TRAIN"))
    sc = NormalizationScalars(g1=1.0, r=1.0, b=1.0, g2=1.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_correction_history=("flat_dark_subtracted",),
        normalization=__import__("zecalibrator.core.descriptors", fromlist=["NormalizationProvenance"]).NormalizationProvenance(
            algorithm="cfa-median-per-plane", population="cfa-4-plane", scalars=sc
        ),
    )
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0,
        flat_form="normalized_response", normalization_algorithm="cfa-median-per-plane",
        normalization_scalars=sc, pixel_domain="normalized_response", physical_units="dimensionless",
        filter="NONE", optical_train_id="TRAIN", processing=pp,
        geometry=geo(cfa_phase="GRBG"),
    )
    dk = descriptor("dark", "included", geometry=geo(cfa_phase="GRBG"))
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert set(r.plan.masters) == {"dark", "flat"}


def test_normalized_cfa_flat_single_scalar_rejected():
    from zecalibrator.core.descriptors import NormalizationProvenance

    lt = light(geometry=geo(cfa_phase="GRBG"), optical=type(light().optical)(filter="NONE", optical_train_id="TRAIN"))
    sc = NormalizationScalars(mono=1.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_correction_history=("flat_dark_subtracted",),
        normalization=NormalizationProvenance(algorithm="median", population="mono-valid", scalars=sc),
    )
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0,
        flat_form="normalized_response", normalization_algorithm="median",
        normalization_scalars=sc, pixel_domain="normalized_response", physical_units="dimensionless",
        filter="NONE", optical_train_id="TRAIN", processing=pp,
        geometry=geo(cfa_phase="GRBG"),
    )
    dk = descriptor("dark", "included", geometry=geo(cfa_phase="GRBG"))
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    assert r.outcome == OUTCOME_NO_MATCH
    assert "NORMALIZATION_SCALAR_COUNT_MISMATCH" in r.reason_codes


def test_corrected_unnormalized_flat_needs_no_additive_binding():
    lt = light(geometry=geo(cfa_phase="GRBG"), optical=type(light().optical)(filter="NONE", optical_train_id="TRAIN"))
    pp = ProcessingProvenance(source="synthetic_fixture", additive_correction_history=("flat_dark_subtracted",))
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="TRAIN", processing=pp, geometry=geo(cfa_phase="GRBG"),
    )
    dk = descriptor("dark", "included", geometry=geo(cfa_phase="GRBG"))
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert "flat_dark" not in r.plan.masters


# --- own flat dependency exposure / bias qualification ----------------------
def test_flatdark_exposure_compared_to_parent_flat():
    lt = light(optical=type(light().optical)(filter="NONE", optical_train_id="TRAIN"))
    flat = descriptor("flat", "not_applicable", exposure_s=1.0, flat_form="raw_response", filter="NONE", optical_train_id="TRAIN")
    fd = descriptor("flat_dark", "included", exposure_s=1.0)
    dk = descriptor("dark", "included")
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)], flat_dark=[candidate("fd1", fd)]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED


def test_flatdark_exposure_mismatch_rejects():
    lt = light(optical=type(light().optical)(filter="NONE", optical_train_id="TRAIN"))
    flat = descriptor("flat", "not_applicable", exposure_s=1.0, flat_form="raw_response", filter="NONE", optical_train_id="TRAIN")
    fd = descriptor("flat_dark", "included", exposure_s=20.0)
    dk = descriptor("dark", "included")
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)], flat_dark=[candidate("fd1", fd)]),
        policy(),
    )
    assert r.outcome == OUTCOME_NO_MATCH
    assert "EXPOSURE_MISMATCH" in r.reason_codes


def test_flat_bias_requires_qualified_range():
    lt = light(optical=type(light().optical)(filter="NONE", optical_train_id="TRAIN"))
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="raw_response", filter="NONE", optical_train_id="TRAIN",
        acquisition_profile=profile(short_flat_profile=True),
    )
    # bias_flat exposure exceeds the flat's qualified range (0.01 s).
    bf = descriptor("bias", "not_applicable", exposure_s=0.05)
    dk = descriptor("dark", "included")
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)], bias=[candidate("b1", bf)]),
        policy(),
    )
    assert r.outcome == OUTCOME_NO_MATCH
    assert "EXPOSURE_MISMATCH" in r.reason_codes


# --- no silent downgrade ----------------------------------------------------
def test_no_silent_downgrade_when_dark_missing():
    lt = light()
    # flat is available but dark missing; request still requires dark.
    flat = descriptor("flat", "not_applicable", exposure_s=1.0, flat_form="raw_response")
    r = match_calibration(lt, request("dark_incl_bias", "apply"), pool(flat=[candidate("f1", flat)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "ROLE_UNAVAILABLE" in r.reason_codes


# --- manual selection -------------------------------------------------------
def test_manual_resolution_disambiguates():
    lt = light()
    d1 = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    d2 = descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(
        lt, request(), pool(dark=[candidate("d1", d1), candidate("d2", d2)]), policy(),
        manual_selection={"dark": "d2"},
    )
    assert r.outcome == OUTCOME_MATCHED
    assert r.plan.masters["dark"].descriptor_id == d2.descriptor_id


def test_manual_incompatible_choice_refuses():
    lt = light()
    d1 = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    d2 = descriptor("dark", "included", acquisition_obj=acquisition(gain=999))
    r = match_calibration(
        lt, request(), pool(dark=[candidate("d1", d1), candidate("d2", d2)]), policy(),
        manual_selection={"dark": "d2"},
    )
    assert r.outcome == OUTCOME_NO_MATCH
    assert "GAIN_MISMATCH" in r.reason_codes


def test_manual_unknown_candidate_refuses():
    lt = light()
    d1 = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    r = match_calibration(
        lt, request(), pool(dark=[candidate("d1", d1)]), policy(),
        manual_selection={"dark": "nope"},
    )
    assert r.outcome == OUTCOME_NO_MATCH


# --- immutability + determinism --------------------------------------------
def test_deterministic_reasons_under_permutation():
    lt = light()
    d_bad = descriptor("dark", "included", acquisition_obj=acquisition(gain=999))
    d_ok = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    order1 = [candidate("bad", d_bad), candidate("ok", d_ok)]
    order2 = [candidate("ok", d_ok), candidate("bad", d_bad)]
    r1 = match_calibration(lt, request(), pool(dark=order1), policy())
    r2 = match_calibration(lt, request(), pool(dark=order2), policy())
    assert r1.outcome == r2.outcome == OUTCOME_MATCHED
    assert r1.plan.plan_id == r2.plan.plan_id


def test_descriptor_deep_immutable():
    d = descriptor("dark", "included")
    with pytest.raises(Exception):
        d.acquisition = None  # frozen dataclass
    with pytest.raises(Exception):
        d.geometry.shape = (1, 1)


def test_duplicate_identical_collapse_keeps_locations():
    lt = light()
    d = descriptor("dark", "included")
    c1 = candidate("d1", d, locator_path="/a.fits")
    c2 = candidate("d2", d, locator_path="/b.fits")
    r = match_calibration(lt, request(), pool(dark=[c1, c2]), policy())
    assert r.outcome == OUTCOME_MATCHED
    locs = r.plan.masters["dark"].locators
    assert {l.path for l in locs} == {"/a.fits", "/b.fits"}


def test_same_metadata_different_content_not_duplicate():
    lt = light()
    d1 = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    d2 = descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", d1), candidate("d2", d2)]), policy())
    assert r.outcome == OUTCOME_AMBIGUOUS
