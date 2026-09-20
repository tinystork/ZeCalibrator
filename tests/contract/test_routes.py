"""Auto-route (R3D-B) contract tests: pure route enumeration + application resolver.

Covers the owner-mandated route decision table for the Standard UX. Standard
expresses ONLY user intent ("use the compatible masters I supplied"); ZeCalibrator
resolves the scientific route automatically and never invents a hidden default or
picks a winner arbitrarily.
"""

from __future__ import annotations

import pytest

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    light,
    policy,
    profile,
    pool,
)
from zecalibrator.application.library import LibrarySnapshot
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.descriptors import ProcessingProvenance
from zecalibrator.core.routes import (
    BIAS_STATE_UNKNOWN,
    FLAT_UNSUPPORTED_RAW,
    FLAT_UNUSABLE,
    OUTCOME_AMBIGUOUS,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_READY,
    STANDARD_MASTER_CONTRACT,
    enumerate_routes,
)


def snapshot(**roles) -> LibrarySnapshot:
    return LibrarySnapshot(
        revision="r",
        schema_version="zecalibrator.library.v1",
        candidates={k: tuple(v) for k, v in roles.items()},
    )


def _raw_flat_proc(*, short_flat=False):
    return ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        acquisition_profile=profile(short_flat_profile=short_flat) if short_flat else None,
    )


# ---------------------------------------------------------------------------
# 3. Standard + compatible standard dark -> dark applied automatically.
# ---------------------------------------------------------------------------
def test_standard_dark_applied_automatically():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    res = resolve_route(lt, snapshot(dark=[candidate("d1", dk)]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_incl_bias"
    assert res.plan is not None
    assert set(res.plan.masters) == {"dark"}


# ---------------------------------------------------------------------------
# 4. Standard dark + extra bias -> NO double-bias subtraction.
# ---------------------------------------------------------------------------
def test_dark_with_extra_bias_no_double_subtraction():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    bias = descriptor("bias", "not_applicable", exposure_s=0.001)
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], bias=[candidate("b1", bias)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_incl_bias"
    # A present bias is never consumed by dark_incl_bias (no second subtraction).
    assert "bias" not in res.route.masters
    assert set(res.plan.masters) == {"dark"}


# ---------------------------------------------------------------------------
# 5. bias-removed dark + required bias -> correct additive route.
# ---------------------------------------------------------------------------
def test_bias_removed_dark_requires_bias():
    lt = light()
    dk = descriptor("dark", "removed", exposure_s=10.0)
    bias = descriptor("bias", "not_applicable", exposure_s=0.001)
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], bias=[candidate("b1", bias)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_bias_removed"
    assert set(res.route.masters) == {"dark", "bias"}
    assert set(res.plan.masters) == {"dark", "bias"}


# ---------------------------------------------------------------------------
# 6. dark bias_state unknown -> needs-attention, never a hidden default.
# ---------------------------------------------------------------------------
def test_dark_bias_state_unknown_needs_attention():
    lt = light()
    dk = descriptor("dark", "unknown", exposure_s=10.0)
    res = resolve_route(lt, snapshot(dark=[candidate("d1", dk)]), policy())
    assert res.outcome == OUTCOME_NEEDS_ATTENTION
    assert res.plan is None
    assert any(r.code == BIAS_STATE_UNKNOWN for r in res.reasons)
    assert not res.routes  # never silently pick dark_incl_bias or bias_only


# ---------------------------------------------------------------------------
# 7. compatible flat -> automatically requested.
# 8. flat absent -> no flat correction.
# ---------------------------------------------------------------------------
def test_compatible_flat_automatically_requested():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    flat = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, exposure_s=1.0,
    )
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.flat_mode == "apply"
    assert res.route.flat_prep_mode == "normalize_only"
    assert "flat" in res.plan.masters


def test_flat_absent_no_flat_correction():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    res = resolve_route(lt, snapshot(dark=[candidate("d1", dk)]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.route.flat_mode == "none"
    assert "flat" not in res.plan.masters


# ---------------------------------------------------------------------------
# 9. flat supplied but unusable -> needs-attention, not silently ignored.
# ---------------------------------------------------------------------------
def test_flat_supplied_but_unusable_needs_attention():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    flat = descriptor(
        "flat", "not_applicable", flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        processing=_raw_flat_proc(), exposure_s=1.0,
    )
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    assert res.outcome == OUTCOME_NEEDS_ATTENTION
    assert res.plan is None
    assert any(r.code in (FLAT_UNUSABLE, "FLAT_ADDITIVE_DEPENDENCY_MISSING") for r in res.reasons)


# ---------------------------------------------------------------------------
# 10. raw flat is unsupported in Standard -> NEEDS_ATTENTION, never auto flat_dark.
# ---------------------------------------------------------------------------
def test_raw_flat_with_flat_dark_prep():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    flat = descriptor(
        "flat", "not_applicable", flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        processing=_raw_flat_proc(), exposure_s=1.0,
    )
    fd = descriptor("flat_dark", "included", exposure_s=1.0)
    res = resolve_route(
        lt,
        snapshot(
            dark=[candidate("d1", dk)],
            flat=[candidate("f1", flat)],
            flat_dark=[candidate("fd1", fd)],
        ),
        policy(),
    )
    # Standard never auto-constructs a flat_dark from a raw flat.
    assert res.outcome == OUTCOME_NEEDS_ATTENTION
    assert res.plan is None
    assert any(r.code == FLAT_UNSUPPORTED_RAW for r in res.reasons)


# ---------------------------------------------------------------------------
# 11. raw flat + qualified bias is still unsupported in Standard.
# ---------------------------------------------------------------------------
def test_raw_flat_with_qualified_bias_prep():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    flat = descriptor(
        "flat", "not_applicable", flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        processing=_raw_flat_proc(short_flat=True), exposure_s=1.0,
    )
    bias = descriptor("bias", "not_applicable", exposure_s=0.001)
    res = resolve_route(
        lt,
        snapshot(
            dark=[candidate("d1", dk)],
            flat=[candidate("f1", flat)],
            bias=[candidate("b1", bias)],
        ),
        policy(),
    )
    assert res.outcome == OUTCOME_NEEDS_ATTENTION
    assert res.plan is None
    assert any(r.code == FLAT_UNSUPPORTED_RAW for r in res.reasons)


# ---------------------------------------------------------------------------
# 12. flat_dark never becomes a direct light correction.
# ---------------------------------------------------------------------------
def test_flat_dark_never_a_direct_light_correction():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    fd = descriptor("flat_dark", "included", exposure_s=1.0)
    # A flat_dark placed in the dark pool is rejected by role semantics; only
    # the genuine dark is ever a direct light correction. In Standard, flat_dark
    # is a flat-preparation dependency only (raw flats are unsupported).
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk), candidate("fd1", fd)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_incl_bias"
    assert res.route.masters["dark"].descriptor.master_type == "dark"
    assert res.route.masters["dark"].descriptor.descriptor_id == dk.descriptor_id


# ---------------------------------------------------------------------------
# 13. multiple complete routes -> AMBIGUOUS.
# 14. single complete route -> deterministic READY/MATCHED.
# ---------------------------------------------------------------------------
def test_multiple_complete_routes_ambiguous():
    lt = light()
    d1 = descriptor("dark", "included", exposure_s=10.0, content_sha256="a" * 64, mask_identity="b" * 64)
    d2 = descriptor("dark", "included", exposure_s=10.0, content_sha256="c" * 64, mask_identity="d" * 64)
    res = resolve_route(
        lt, snapshot(dark=[candidate("d1", d1), candidate("d2", d2)]), policy()
    )
    assert res.outcome == OUTCOME_AMBIGUOUS
    assert len(res.routes) == 2
    assert res.plan is None


def test_single_complete_route_ready_matched():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    res = resolve_route(lt, snapshot(dark=[candidate("d1", dk)]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.request.additive_mode == "dark_incl_bias"
    assert res.plan.request.flat_mode == "none"


# ---------------------------------------------------------------------------
# 15. final CalibrationPlan records exact effective modes + chosen masters.
# ---------------------------------------------------------------------------
def test_plan_records_exact_effective_modes_and_dependency():
    lt = light()
    dk = descriptor("dark", "included", exposure_s=10.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    flat = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        processing=pp, exposure_s=1.0,
    )
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    plan = res.plan
    assert plan.request.additive_mode == "dark_incl_bias"
    assert plan.request.flat_mode == "apply"
    assert set(plan.masters) == {"dark", "flat"}
    assert plan.masters["dark"].descriptor_id == dk.descriptor_id
    assert plan.masters["flat"].descriptor_id == flat.descriptor_id


# ---------------------------------------------------------------------------
# Mixed bias states -> AMBIGUOUS (distinct routes), never a hidden default.
# ---------------------------------------------------------------------------
def test_mixed_bias_states_ambiguous():
    lt = light()
    d_inc = descriptor("dark", "included", exposure_s=10.0, content_sha256="a" * 64, mask_identity="b" * 64)
    d_rem = descriptor("dark", "removed", exposure_s=10.0, content_sha256="c" * 64, mask_identity="d" * 64)
    bias = descriptor("bias", "not_applicable", exposure_s=0.001)
    res = resolve_route(
        lt,
        snapshot(
            dark=[candidate("d1", d_inc), candidate("d2", d_rem)],
            bias=[candidate("b1", bias)],
        ),
        policy(),
    )
    assert res.outcome == OUTCOME_AMBIGUOUS
    modes = {r.additive_mode for r in res.routes}
    assert modes == {"dark_incl_bias", "dark_bias_removed"}


# ---------------------------------------------------------------------------
# Partial routes (control / bias_only) are never reported as READY/full.
# ---------------------------------------------------------------------------
def test_control_and_bias_only_are_partial():
    lt = light()
    assert enumerate_routes(lt, {}, policy()).outcome == OUTCOME_NEEDS_ATTENTION
    bias = descriptor("bias", "not_applicable", exposure_s=0.001)
    res = resolve_route(lt, snapshot(bias=[candidate("b1", bias)]), policy())
    assert res.outcome == OUTCOME_NEEDS_ATTENTION
    assert res.plan is None
    assert res.routes[0].partial is True
    assert res.routes[0].additive_mode == "bias_only"


# ---------------------------------------------------------------------------
# F1 reproduction: a READY route via a compatible included dark even when an
# INCOMPATIBLE dark has an indeterminate bias_state. The audit reason is kept
# (durable channel) but the route decision is still READY.
# ---------------------------------------------------------------------------
def test_ready_with_incompatible_unknown_dark_keeps_audit_reason():
    lt = light()
    dk_unknown = descriptor("dark", "unknown", exposure_s=10.0, acquisition_obj=acquisition(gain=999))
    dk_incl = descriptor("dark", "included", exposure_s=10.0)
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk_unknown), candidate("d2", dk_incl)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_incl_bias"
    assert res.plan is not None
    # The indeterminate-bias note is retained in the durable audit reasons, but
    # it never changes the READY route decision.
    assert any(r.code == BIAS_STATE_UNKNOWN for r in res.reasons)
