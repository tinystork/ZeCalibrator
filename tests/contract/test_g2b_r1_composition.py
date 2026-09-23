"""G2B R1 — passthrough/partial route classification + composition contract.

Asserts the derived Standard route produces a READY plan for passthrough and
partial outcomes, with an honest availability-relative composition (level +
applied/skipped/no-candidate roles + additive_state + flat_applied), and that
all-incompatible masters fall back to passthrough with the rejection reasons
preserved (never a hard failure, never silent).
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
from zecalibrator.application.library import LibrarySnapshot
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.descriptors import OpticalIdentity, ProcessingProvenance
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.routes import NO_APPLICABLE_MASTER, OUTCOME_READY


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


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------
def test_no_master_is_ready_passthrough_none():
    lt = light()
    res = resolve_route(lt, snapshot(), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.route.additive_mode == "control"
    assert res.route.flat_mode == "none"
    assert set(res.plan.masters) == set()
    assert res.plan.composition.level == "NONE"
    assert res.plan.composition.additive_state == "none"
    assert res.plan.composition.flat_applied is False
    assert res.plan.composition.applied_roles == ()
    # The "nothing applicable" fact is a non-blocking audit entry.
    assert any(r.code == NO_APPLICABLE_MASTER and not r.blocking for r in res.reasons)


def test_all_incompatible_masters_passthrough_with_reasons():
    lt = light()
    bad = candidate("d1", descriptor("dark", "included", acquisition_obj=acquisition(gain=999)))
    res = resolve_route(lt, snapshot(dark=[bad]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.composition.level == "NONE"
    assert "dark" not in res.plan.masters
    # The rejection reason is preserved and visible (never silent).
    assert any(r.code == "GAIN_MISMATCH" for r in res.reasons)


# ---------------------------------------------------------------------------
# Partial routes
# ---------------------------------------------------------------------------
def test_bias_only_is_ready_partial():
    lt = light()
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001))
    res = resolve_route(lt, snapshot(bias=[bias]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.composition.additive_state == "bias_only"
    assert res.plan.composition.applied_roles == ("bias",)
    assert res.plan.composition.flat_applied is False


def test_flat_only_is_ready():
    lt = light()
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    assert res.plan.composition.additive_state == "none"
    assert res.plan.composition.flat_applied is True
    assert res.plan.composition.applied_roles == ("flat",)


def test_dark_only_is_ready():
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    res = resolve_route(lt, snapshot(dark=[dark]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan.composition.additive_state == "dark_incl_bias"
    assert res.plan.composition.applied_roles == ("dark",)
    assert res.plan.composition.flat_applied is False


# ---------------------------------------------------------------------------
# Full routes
# ---------------------------------------------------------------------------
def test_dark_bias_flat_complete():
    lt = light()
    dark = candidate("d1", descriptor("dark", "removed", content_sha256="a" * 64, mask_identity="b" * 64))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001, content_sha256="c" * 64, mask_identity="d" * 64))
    flat = _corrected_flat("f1")
    res = resolve_route(lt, snapshot(dark=[dark], bias=[bias], flat=[flat]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan.composition.level == "COMPLETE"
    assert res.plan.composition.additive_state == "dark_bias_removed"
    assert res.plan.composition.flat_applied is True
    assert set(res.plan.composition.applied_roles) == {"dark", "bias", "flat"}


# ---------------------------------------------------------------------------
# Strict public matching stays NO_MATCH for an unsatisfiable explicit request.
# ---------------------------------------------------------------------------
def test_strict_explicit_request_unsatisfiable_stays_no_match():
    lt = light()
    r = match_calibration(lt, request("dark_incl_bias"), pool(dark=[]), policy())
    assert r.outcome == "NO_MATCH"
