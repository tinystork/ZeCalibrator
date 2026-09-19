"""R3B relation-matcher tier contract (necessary / disambiguator / conditional).

Verifies the selection-semantics change: an unknown *disambiguator* (and, for a
mono sensor, the CFA-conditional orientation/ROI fields) no longer hard-rejects a
match the *necessary* facts already support, while genuinely necessary scientific
invariants remain blocking. UNVERIFIED is a non-blocking note, recorded on
``MatchResult.unverified`` and never present in ``reason_codes``.
"""

from __future__ import annotations

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
    request,
)
from zecalibrator.core.descriptors import OpticalIdentity, ProcessingProvenance
from zecalibrator.application.library import LibrarySnapshot, resolve_calibration
from zecalibrator.io.library_index import LIBRARY_INDEX_SCHEMA
from zecalibrator.core.matching import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_MATCHED,
    OUTCOME_NO_MATCH,
    match_calibration,
)


def _unverified_codes(r) -> set[str]:
    return {reason.code for reason in r.unverified}


# --- Necessary tier: unknown/mismatch still blocks (unchanged) --------------
def test_necessary_detector_model_unknown_blocks():
    lt = light(detector=detector(model=None))
    dk = descriptor("dark", "included", detector_obj=detector(model=None))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes
    assert "UNKNOWN_EQUALS_UNKNOWN" in r.reason_codes


def test_necessary_gain_unknown_blocks():
    lt = light(acquisition=acquisition(gain=None))
    dk = descriptor("dark", "included", acquisition_obj=acquisition(gain=100))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes
    assert not r.unverified


def test_necessary_binning_unknown_blocks():
    lt = light(geometry=geo(binning=None))
    dk = descriptor("dark", "included", geometry=geo(binning=None))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes


def test_necessary_cfa_phase_unknown_blocks():
    lt = light(geometry=geo(cfa_phase=None))
    dk = descriptor("dark", "included", geometry=geo(cfa_phase=None))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes
    assert not r.unverified


def test_necessary_cfa_phase_mismatch_blocks():
    lt = light(geometry=geo(cfa_phase="GRBG"))
    dk = descriptor("dark", "included", geometry=geo(cfa_phase="mono"))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "CFA_PHASE_MISMATCH" in r.reason_codes


# --- Disambiguator tier: both unknown -> non-blocking UNVERIFIED -------------
@pytest.mark.parametrize(
    "light_kw,desc_kw,field",
    [
        (dict(detector=detector(instance="unknown")), dict(detector_obj=detector(instance="unknown")), "detector.detector_instance_id"),
        (dict(acquisition=acquisition(readout_mode=None)), dict(acquisition_obj=acquisition(readout_mode=None)), "acquisition.readout_mode"),
        (dict(acquisition=acquisition(adc_mode=None)), dict(acquisition_obj=acquisition(adc_mode=None)), "acquisition.adc_mode"),
        (dict(geometry=geo(sensor_dimensions=None)), dict(geometry=geo(sensor_dimensions=None)), "geometry.sensor_dimensions"),
    ],
)
def test_disambiguator_both_unknown_unverified(light_kw, desc_kw, field):
    lt = light(**light_kw)
    dk = descriptor("dark", "included", **desc_kw)
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_MATCHED
    assert _unverified_codes(r) == {"UNVERIFIED"}
    assert "UNVERIFIED" not in r.reason_codes
    # The note is non-blocking and records the field under question.
    notes = [reason for reason in r.unverified if reason.field == field]
    assert notes
    assert all(not reason.blocking for reason in r.unverified)


# --- Disambiguator tier: both known + different -> blocking mismatch ---------
@pytest.mark.parametrize(
    "light_kw,desc_kw,code",
    [
        (dict(detector=detector(instance="A")), dict(detector_obj=detector(instance="B")), "DETECTOR_MISMATCH"),
        (dict(acquisition=acquisition(readout_mode="A")), dict(acquisition_obj=acquisition(readout_mode="B")), "READOUT_MISMATCH"),
        (dict(acquisition=acquisition(adc_mode="A")), dict(acquisition_obj=acquisition(adc_mode="B")), "ADC_MISMATCH"),
        (dict(geometry=geo(sensor_dimensions=(4, 4))), dict(geometry=geo(sensor_dimensions=(8, 8))), "GEOMETRY_MISMATCH"),
    ],
)
def test_disambiguator_mismatch_blocks(light_kw, desc_kw, code):
    lt = light(**light_kw)
    dk = descriptor("dark", "included", **desc_kw)
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert code in r.reason_codes
    assert not r.unverified


# --- Disambiguator tier: one known + one unknown -> conservative MISSING -----
@pytest.mark.parametrize(
    "light_kw,desc_kw",
    [
        (dict(detector=detector(instance="SYNTH-DET-0001")), dict(detector_obj=detector(instance="unknown"))),
        (dict(acquisition=acquisition(readout_mode="MODE_A")), dict(acquisition_obj=acquisition(readout_mode=None))),
        (dict(acquisition=acquisition(adc_mode="MODE_16")), dict(acquisition_obj=acquisition(adc_mode=None))),
        (dict(geometry=geo(sensor_dimensions=(4, 4))), dict(geometry=geo(sensor_dimensions=None))),
    ],
)
def test_disambiguator_one_known_one_unknown_blocks(light_kw, desc_kw):
    lt = light(**light_kw)
    dk = descriptor("dark", "included", **desc_kw)
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes
    assert not r.unverified


# --- CFA-conditional tier: necessary for CFA, disambiguator for mono ---------
@pytest.mark.parametrize("field", ["orientation", "roi_origin", "roi_extent"])
def test_cfa_conditional_unknown_blocks_for_bayer(field):
    lt = light(geometry=geo(cfa_phase="GRBG", **{field: None}))
    dk = descriptor("dark", "included", geometry=geo(cfa_phase="GRBG", **{field: None}))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes
    assert not r.unverified


@pytest.mark.parametrize("field", ["orientation", "roi_origin", "roi_extent"])
def test_cfa_conditional_unknown_unverified_for_mono(field):
    lt = light(geometry=geo(cfa_phase="mono", **{field: None}))
    dk = descriptor("dark", "included", geometry=geo(cfa_phase="mono", **{field: None}))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_MATCHED
    assert "UNVERIFIED" in _unverified_codes(r)
    assert "UNVERIFIED" not in r.reason_codes


# --- Flat-only disambiguator: optical_train_id ------------------------------
def _mono_flat(optical_train_id, filter_="NONE"):
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_correction_history=("flat_dark_subtracted",),
    )
    return descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
        filter=filter_, optical_train_id=optical_train_id, processing=pp,
        geometry=geo(cfa_phase="mono"),
    )


def test_flat_optical_train_both_unknown_unverified():
    lt = light(optical=OpticalIdentity(filter="NONE", optical_train_id=None))
    dk = descriptor("dark", "included")
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", _mono_flat(None))]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert "UNVERIFIED" in _unverified_codes(r)
    assert "UNVERIFIED" not in r.reason_codes


def test_flat_optical_train_mismatch_blocks():
    lt = light(optical=OpticalIdentity(filter="NONE", optical_train_id="TRAIN_A"))
    dk = descriptor("dark", "included")
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", _mono_flat("TRAIN_B"))]),
        policy(),
    )
    assert r.outcome == OUTCOME_NO_MATCH
    assert "OPTICAL_TRAIN_MISMATCH" in r.reason_codes


# --- Disambiguator cannot distinguish -> still AMBIGUOUS via coherent sets ---
def test_unknown_disambiguator_still_ambiguous_when_indistinguishable():
    lt = light(detector=detector(instance="unknown"))
    d1 = descriptor(
        "dark", "included", detector_obj=detector(instance="unknown"),
        content_sha256="a" * 64, mask_identity="b" * 64,
    )
    d2 = descriptor(
        "dark", "included", detector_obj=detector(instance="unknown"),
        content_sha256="c" * 64, mask_identity="d" * 64,
    )
    r = match_calibration(
        lt, request(), pool(dark=[candidate("d1", d1), candidate("d2", d2)]), policy()
    )
    assert r.outcome == OUTCOME_AMBIGUOUS
    assert len(r.coherent_sets) == 2
    assert "UNVERIFIED" in _unverified_codes(r)


# --- UNVERIFIED never leaks into the manifest reason codes ------------------
def test_unverified_excluded_from_reason_codes_even_with_blocking():
    # A blocking necessary fact plus a non-blocking disambiguator: the result is
    # NO_MATCH driven only by the blocking reason; UNVERIFIED stays out.
    lt = light(acquisition=acquisition(gain=None, readout_mode=None))
    dk = descriptor("dark", "included", acquisition_obj=acquisition(gain=100, readout_mode=None))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == OUTCOME_NO_MATCH
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes
    assert "UNVERIFIED" not in r.reason_codes


# --- UNVERIFIED survives the public decision boundary ----------------------
def test_envelope_carries_unverified_note_on_matched():
    # R3B REWORK-1: resolve_calibration (public decision boundary) must carry the
    # UNVERIFIED note on a MATCHED outcome with no reason codes.
    lt = light(detector=detector(instance="unknown"))
    dk = descriptor("dark", "included", detector_obj=detector(instance="unknown"))
    snap = LibrarySnapshot(
        revision="r1", schema_version=LIBRARY_INDEX_SCHEMA,
        candidates={"dark": (candidate("d1", dk),)},
    )
    env = resolve_calibration(lt, request(), snap, policy())
    assert env.outcome == OUTCOME_MATCHED
    assert {r.code for r in env.unverified} == {"UNVERIFIED"}
    assert env.reason_codes == ()


def test_envelope_unverified_roundtrips():
    # The UNVERIFIED note survives DecisionEnvelope JSON serialization (the code
    # string is the round-trip-safe signal).
    import json

    lt = light(detector=detector(instance="unknown"))
    dk = descriptor("dark", "included", detector_obj=detector(instance="unknown"))
    snap = LibrarySnapshot(
        revision="r1", schema_version=LIBRARY_INDEX_SCHEMA,
        candidates={"dark": (candidate("d1", dk),)},
    )
    env = resolve_calibration(lt, request(), snap, policy())
    restored = type(env).from_dict(json.loads(json.dumps(dict(env.to_dict()), allow_nan=False)))
    assert {r.code for r in restored.unverified} == {"UNVERIFIED"}
