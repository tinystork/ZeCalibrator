"""Rework-2 regression tests (contract N-A, N-B, R1).

N-A: the light's matching-relevant acquisition-profile facts (qualified bias
range) participate in the plan identity via the already-projected
``light_constraints.acquisition`` whole-subtree path — no projection allowlist
widening.
N-B + R1: every NO_MATCH carries non-empty deterministic reason_codes (unroutable
flat_dark bias_state, manual wrong-role / role-not-required).
"""

from __future__ import annotations

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    light,
    policy,
    pool,
    request,
)
from zecalibrator.core.matching import match_calibration


# --- N-A: light bias range is in the plan identity --------------------------
def test_light_bias_range_changes_plan_id():
    lt1 = light(acquisition=acquisition(bias_exposure_max_s=0.01))
    lt2 = light(acquisition=acquisition(bias_exposure_max_s=0.5))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001))
    r1 = match_calibration(lt1, request("bias_only"), pool(bias=[bias]), policy())
    r2 = match_calibration(lt2, request("bias_only"), pool(bias=[bias]), policy())
    # Bias 0.001 s is within both ranges, so both MATCHED.
    assert r1.outcome == r2.outcome == "MATCHED"
    # But the plan identity must reflect the different light bias ranges.
    assert r1.plan.plan_id != r2.plan.plan_id


def test_light_bias_range_changes_bias_outcome_conservatively():
    lt_tight = light(acquisition=acquisition(bias_exposure_max_s=0.01))
    lt_wide = light(acquisition=acquisition(bias_exposure_max_s=0.5))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.05))
    r_tight = match_calibration(lt_tight, request("bias_only"), pool(bias=[bias]), policy())
    r_wide = match_calibration(lt_wide, request("bias_only"), pool(bias=[bias]), policy())
    # 0.05 s exceeds the tight range but fits the wide range; the value must change
    # the outcome, never be silently ignored.
    assert r_tight.outcome == "NO_MATCH"
    assert r_wide.outcome == "MATCHED"


def test_audit_only_relocation_does_not_change_plan_id():
    lt = light()
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001))
    r = match_calibration(lt, request("bias_only"), pool(bias=[bias]), policy())
    base = r.plan.plan_digest_dict()
    # path/imported_at/locator are retrieval-only; injecting them must not change
    # the plan digest (they are outside PLAN_PROJECTION).
    import copy

    alt = copy.deepcopy(dict(base))
    alt["masters"]["bias"]["locator"] = {"path": "/elsewhere.fits", "hdu": 0}
    alt["imported_at"] = "2026-09-17T00:00:00Z"
    alt["source_path"] = "/x/y.fits"
    from zecalibrator.core.digests import plan_digest

    assert plan_digest(base) == plan_digest(alt)


# --- N-B: flat_dark unroutable bias_state is never a silent skip -------------
def test_flatdark_unknown_bias_state_explains_refusal():
    lt = light()
    dk = descriptor("dark", "included")
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
    )
    fd = descriptor("flat_dark", "unknown", exposure_s=1.0)
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)], flat_dark=[candidate("fd1", fd)]),
        policy(),
    )
    assert r.outcome == "NO_MATCH"
    assert "FLAT_ADDITIVE_DEPENDENCY_MISSING" in r.reason_codes


def test_flatdark_wrong_master_type_explains_refusal():
    lt = light()
    dk = descriptor("dark", "included")
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
    )
    # A dark placed in the flat_dark pool has the wrong semantic master_type.
    wrong = descriptor("dark", "included", exposure_s=1.0)
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)], flat_dark=[candidate("fd1", wrong)]),
        policy(),
    )
    assert r.outcome == "NO_MATCH"
    assert r.reason_codes  # never empty
    assert "FLAT_ADDITIVE_DEPENDENCY_MISSING" in r.reason_codes


# --- R1: manual selection non-empty reason_codes ----------------------------
def test_manual_wrong_role_explains_refusal():
    lt = light()
    d1 = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001))
    r = match_calibration(
        lt, request("dark_incl_bias"), pool(dark=[d1], bias=[bias]), policy(),
        manual_selection={"dark": "b1"},
    )
    assert r.outcome == "NO_MATCH"
    assert "MANUAL_SELECTION_NO_MATCH" in r.reason_codes


def test_manual_role_not_required_explains_refusal():
    lt = light()
    d1 = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001))
    r = match_calibration(
        lt, request("dark_incl_bias"), pool(dark=[d1], bias=[bias]), policy(),
        manual_selection={"bias": "b1"},
    )
    assert r.outcome == "NO_MATCH"
    assert "MANUAL_SELECTION_NO_MATCH" in r.reason_codes


def test_manual_known_incompatible_keeps_compatibility_reason():
    lt = light()
    d1 = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64))
    d2 = candidate("d2", descriptor("dark", "included", acquisition_obj=acquisition(gain=999)))
    r = match_calibration(
        lt, request("dark_incl_bias"), pool(dark=[d1, d2]), policy(),
        manual_selection={"dark": "d2"},
    )
    assert r.outcome == "NO_MATCH"
    assert "GAIN_MISMATCH" in r.reason_codes
    # The compatibility reason is the primary explanation; no extra empty-refusal
    # noise is required for a known-but-incompatible id.
    assert "MANUAL_SELECTION_NO_MATCH" not in r.reason_codes
