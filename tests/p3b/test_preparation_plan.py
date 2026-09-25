"""LOT3 tests: run preparation plan (site x run atomicity, P0, freeze guards).

Scoped and fast: no corpus generation, no FITS, no numpy in the module under
test. These tests exercise the *semantic* atomicity of a run — the file-output
transaction (G6) is deliberately out of scope and is not imported here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from research.p3b.preparation_plan import (
    DONORS_AVAILABLE,
    DONORS_UNAVAILABLE,
    RC_DONORS_UNAVAILABLE_RUN_WIDE,
    RUN_WIDE_ABSTAIN,
    RUN_WIDE_ELIGIBLE,
    RUN_WIDE_NO_ACTION,
    RUN_WIDE_REQUALIFY,
    DuplicateSiteError,
    ExecutionResult,
    GeometryBinding,
    PlanFrozenError,
    PlanInvalidError,
    PlanNotFrozenError,
    PreparationPlan,
    SitePlanEntry,
    SiteQualification,
    derive_run_wide,
    preflight,
)
from research.p3b.qualification_policy import (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
    ACTION_NO_ACTION,
    ACTION_REQUALIFY,
    EPISTEMIC_CHARACTERISED_INTERMITTENT,
    EPISTEMIC_CENSORED,
    RC_ELIGIBLE,
    QualificationDecision,
)

_PLAN_PATH = (
    Path(__file__).resolve().parents[2] / "research" / "p3b" / "preparation_plan.py"
)

GEOMETRY = GeometryBinding(shape=(96, 128), cfa_pattern="GRBG")
FRAMES = ("f0", "f1", "f2", "f3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_decision(action=ACTION_ELIGIBLE, epistemic=EPISTEMIC_CHARACTERISED_INTERMITTENT,
                  reason_codes=(RC_ELIGIBLE,)):
    return QualificationDecision(
        epistemic_state=epistemic,
        representativeness="REPRESENTATIVE",
        action=action,
        reason_codes=tuple(reason_codes),
        evidence_summary=(),
    )


def make_qual(site_id, x=0, y=0, action=ACTION_ELIGIBLE,
              epistemic=EPISTEMIC_CHARACTERISED_INTERMITTENT):
    return SiteQualification(
        site_id=site_id, x=x, y=y,
        decision=make_decision(action=action, epistemic=epistemic),
    )


def make_plan(*, frame_ids=FRAMES, sites=(), per_frame=None):
    """Preflight a run; ``per_frame`` is a tuple of donor tokens indexed by frame."""
    if per_frame is None:
        per_frame = (DONORS_AVAILABLE,) * len(frame_ids)
    by_frame = dict(zip(frame_ids, per_frame))
    return preflight(
        run_id="run-1",
        profile_revision="profile-v3",
        calibration_identity="calib-abc123",
        geometry=GEOMETRY,
        frame_ids=frame_ids,
        site_qualifications=tuple(sites),
        donor_availability=lambda sid, fid: by_frame[fid],
    )


def observed_as_assumed(plan: PreparationPlan):
    """An observer that reports the plan's own frozen facts (world unchanged)."""
    def observe(sid, fid):
        return plan.site(sid).per_frame_donor_availability[plan.frame_ids.index(fid)]
    return observe


def per_site_applied(result: ExecutionResult, site_id: str):
    return [
        rec.applied
        for frame in result.frames
        for rec in frame.sites
        if rec.site_id == site_id
    ]


# ---------------------------------------------------------------------------
# Property 1 — site x run atomicity (ABSTAIN for the whole run)
# ---------------------------------------------------------------------------

def test_site_missing_donors_on_last_frame_abstains_whole_run():
    # Site A is ELIGIBLE (LOT2) and has valid donors on frames 0..2 but NO valid
    # same-CFA donor on frame 3.
    plan = make_plan(
        sites=(make_qual("A"),),
        per_frame=(DONORS_AVAILABLE, DONORS_AVAILABLE, DONORS_AVAILABLE, DONORS_UNAVAILABLE),
    )
    entry = plan.site("A")
    assert entry.action_state == ACTION_ELIGIBLE  # LOT2 says eligible
    # P-A: one frame without donors => ABSTAIN for the whole run, never
    # "3 reconstructed + 1 original".
    assert entry.run_wide_applicability == RUN_WIDE_ABSTAIN
    assert entry.abstention_reason == RC_DONORS_UNAVAILABLE_RUN_WIDE

    result = plan.execute(observed_as_assumed(plan))
    assert per_site_applied(result, "A") == [False, False, False, False]


def test_site_with_all_donors_eligible_is_applied_uniformly():
    plan = make_plan(sites=(make_qual("A"),))
    entry = plan.site("A")
    assert entry.run_wide_applicability == RUN_WIDE_ELIGIBLE

    result = plan.execute(observed_as_assumed(plan))
    assert per_site_applied(result, "A") == [True, True, True, True]


# ---------------------------------------------------------------------------
# Property 2 — independence of sites (never global)
# ---------------------------------------------------------------------------

def test_blocked_site_does_not_disable_independent_site_in_same_run():
    # Site A: ELIGIBLE but missing donors on frame 3 -> ABSTAIN.
    # Site B: ELIGIBLE with donors on every frame -> stays ELIGIBLE in the SAME run.
    def provider(site_id, frame_id):
        if site_id == "A":
            return DONORS_UNAVAILABLE if frame_id == "f3" else DONORS_AVAILABLE
        return DONORS_AVAILABLE

    plan = preflight(
        run_id="run-2",
        profile_revision="profile-v3",
        calibration_identity="calib-abc123",
        geometry=GEOMETRY,
        frame_ids=FRAMES,
        site_qualifications=(make_qual("A", 0, 0), make_qual("B", 4, 4)),
        donor_availability=provider,
    )
    assert plan.site("A").run_wide_applicability == RUN_WIDE_ABSTAIN
    assert plan.site("B").run_wide_applicability == RUN_WIDE_ELIGIBLE

    result = plan.execute(provider)
    assert per_site_applied(result, "A") == [False, False, False, False]
    assert per_site_applied(result, "B") == [True, True, True, True]


# ---------------------------------------------------------------------------
# Property 3 — P0: no inter-frame switching (gate PERMANENT)
# ---------------------------------------------------------------------------

def test_p0_naive_per_frame_switching_is_mixed_and_is_rejected_by_the_plan():
    # A naive implementation decides per frame from donor availability and would
    # produce apply/apply/skip/apply for one site inside one run.
    per_frame = [DONORS_AVAILABLE, DONORS_AVAILABLE, DONORS_UNAVAILABLE, DONORS_AVAILABLE]

    naive = ["APPLY" if tok == DONORS_AVAILABLE else "SKIP" for tok in per_frame]
    # The forbidden inter-frame switch, exactly as a naive impl would emit:
    assert naive == ["APPLY", "APPLY", "SKIP", "APPLY"]
    assert len(set(naive)) > 1, "naive implementation switched between frames"

    # The frozen plan can only hold ONE run-wide value for the site; the mixed
    # sequence above is unrepresentable and is therefore collapsed to a uniform
    # ABSTAIN.
    plan = make_plan(sites=(make_qual("A"),), per_frame=tuple(per_frame))
    entry = plan.site("A")
    assert entry.run_wide_applicability == RUN_WIDE_ABSTAIN

    result = plan.execute(observed_as_assumed(plan))
    applied = per_site_applied(result, "A")
    assert len(set(applied)) == 1, "execution switched between frames"
    assert applied == [False, False, False, False]


def test_p0_execution_is_uniform_for_every_site():
    # Mixed-donor and clean-donor sites in the same run: each site is applied
    # either on every frame or on none — never a per-frame mix.
    def provider(site_id, frame_id):
        if site_id == "A":
            return DONORS_UNAVAILABLE if frame_id == "f2" else DONORS_AVAILABLE
        return DONORS_AVAILABLE

    plan = preflight(
        run_id="run-3", profile_revision="p", calibration_identity="c",
        geometry=GEOMETRY, frame_ids=FRAMES,
        site_qualifications=(make_qual("A", 0, 0), make_qual("B", 4, 4)),
        donor_availability=provider,
    )
    result = plan.execute(provider)
    for site_id in ("A", "B"):
        assert len(set(per_site_applied(result, site_id))) == 1


# ---------------------------------------------------------------------------
# Property 4 — finalization guard (no output from an unfrozen plan)
# ---------------------------------------------------------------------------

def test_execute_requires_frozen_plan():
    plan = PreparationPlan(
        run_id="run-4", profile_revision="p", calibration_identity="c",
        geometry=GEOMETRY, frame_ids=FRAMES, frozen=False,
    )
    plan.add_site(
        SitePlanEntry(
            site_id="A", x=0, y=0,
            qualification_state=EPISTEMIC_CHARACTERISED_INTERMITTENT,
            action_state=ACTION_ELIGIBLE,
            run_wide_applicability=RUN_WIDE_ELIGIBLE,
            abstention_reason=None,
            reason_codes=(RC_ELIGIBLE,),
            per_frame_donor_availability=(DONORS_AVAILABLE,) * 4,
        )
    )
    with pytest.raises(PlanNotFrozenError):
        plan.execute(lambda sid, fid: DONORS_AVAILABLE)


def test_preflight_returns_a_frozen_plan_and_writes_nothing():
    plan = make_plan(sites=(make_qual("A"),))
    assert plan.frozen is True


def test_plan_module_has_no_file_io():
    # Preflight/plan must be pure: no open() and no Path write/read helpers, so
    # it cannot finalize (write) any product.
    source = _PLAN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open", "plan module must not call open()"
        if isinstance(node, ast.Attribute) and node.attr in (
            "read_text", "read_bytes", "write_text", "write_bytes",
        ):
            raise AssertionError(f"plan module must not access files via {node.attr}()")


# ---------------------------------------------------------------------------
# Property 5 — contradiction after freeze => PLAN_INVALID => ABORT
# ---------------------------------------------------------------------------

def test_contradiction_after_freeze_aborts_plan_invalid():
    # The plan froze assuming donors available on every frame for site A.
    plan = make_plan(sites=(make_qual("A"),))
    assert plan.site("A").run_wide_applicability == RUN_WIDE_ELIGIBLE

    # At execution, frame 2 is observed WITHOUT donors: a contradiction. It must
    # abort — never silently skip frame 2 and keep applying elsewhere.
    def observing_world(sid, fid):
        return DONORS_UNAVAILABLE if fid == "f2" else DONORS_AVAILABLE

    with pytest.raises(PlanInvalidError):
        plan.execute(observing_world)


def test_contradiction_on_abstained_site_also_aborts():
    # Plan froze site A as ABSTAIN (no donors on frame 3). If execution then
    # observes donors everywhere (the world changed), that is still a
    # contradiction and must abort rather than silently flip to ELIGIBLE.
    plan = make_plan(
        sites=(make_qual("A"),),
        per_frame=(DONORS_AVAILABLE, DONORS_AVAILABLE, DONORS_AVAILABLE, DONORS_UNAVAILABLE),
    )
    assert plan.site("A").run_wide_applicability == RUN_WIDE_ABSTAIN

    with pytest.raises(PlanInvalidError):
        plan.execute(lambda sid, fid: DONORS_AVAILABLE)


# ---------------------------------------------------------------------------
# Property 6 — consumes LOT2, never recomputes, introduces no threshold
# ---------------------------------------------------------------------------

def test_plan_consumes_lot2_decision_verbatim():
    decision = QualificationDecision(
        epistemic_state=EPISTEMIC_CENSORED,
        representativeness="NOT_REPRESENTATIVE",
        action=ACTION_ABSTAIN_CENSORED,
        reason_codes=("EVIDENCE_CENSORED",),
        evidence_summary=(),
    )
    plan = preflight(
        run_id="run-5", profile_revision="p", calibration_identity="c",
        geometry=GEOMETRY, frame_ids=FRAMES,
        site_qualifications=(SiteQualification("A", 1, 2, decision),),
        donor_availability=lambda sid, fid: DONORS_AVAILABLE,
    )
    entry = plan.site("A")
    assert entry.qualification_state == EPISTEMIC_CENSORED
    assert entry.action_state == ACTION_ABSTAIN_CENSORED
    assert entry.reason_codes == ("EVIDENCE_CENSORED",)
    assert entry.run_wide_applicability == RUN_WIDE_ABSTAIN
    assert entry.abstention_reason == ACTION_ABSTAIN_CENSORED


def test_derive_run_wide_maps_non_eligible_lot2_states():
    assert derive_run_wide(ACTION_NO_ACTION, (DONORS_AVAILABLE,)) == (RUN_WIDE_NO_ACTION, None)
    assert derive_run_wide(ACTION_REQUALIFY, (DONORS_UNAVAILABLE,)) == (RUN_WIDE_REQUALIFY, None)
    assert derive_run_wide(ACTION_ABSTAIN_INSUFFICIENT, (DONORS_AVAILABLE,)) == (
        RUN_WIDE_ABSTAIN, ACTION_ABSTAIN_INSUFFICIENT,
    )
    # ELIGIBLE is the only state gated on donors.
    assert derive_run_wide(ACTION_ELIGIBLE, (DONORS_AVAILABLE,)) == (RUN_WIDE_ELIGIBLE, None)
    assert derive_run_wide(ACTION_ELIGIBLE, (DONORS_UNAVAILABLE,)) == (
        RUN_WIDE_ABSTAIN, RC_DONORS_UNAVAILABLE_RUN_WIDE,
    )


def test_plan_module_introduces_no_numeric_threshold():
    # LOT3 introduces no detector, no admission threshold, no amplitude budget.
    # The module may define string constants only; any module-level numeric
    # constant would be a potential threshold and is forbidden.
    source = _PLAN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    value = node.value.value
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        raise AssertionError(
                            f"module-level numeric constant {target.id!r} is a potential threshold"
                        )


# ---------------------------------------------------------------------------
# Property 7 — frozen plan refuses mutation with a typed error
# ---------------------------------------------------------------------------

def test_frozen_plan_refuses_attribute_mutation():
    plan = make_plan(sites=(make_qual("A"),))
    assert plan.frozen is True
    with pytest.raises(PlanFrozenError):
        plan.run_id = "tampered"
    with pytest.raises(PlanFrozenError):
        plan.frozen = False


def test_frozen_plan_refuses_add_site():
    plan = make_plan(sites=(make_qual("A"),))
    entry = SitePlanEntry(
        site_id="B", x=0, y=0,
        qualification_state=EPISTEMIC_CHARACTERISED_INTERMITTENT,
        action_state=ACTION_ELIGIBLE,
        run_wide_applicability=RUN_WIDE_ELIGIBLE,
        abstention_reason=None,
        reason_codes=(RC_ELIGIBLE,),
        per_frame_donor_availability=(DONORS_AVAILABLE,) * 4,
    )
    with pytest.raises(PlanFrozenError):
        plan.add_site(entry)


def test_freeze_is_idempotent():
    plan = make_plan(sites=(make_qual("A"),))
    frozen_once = plan.freeze()
    assert frozen_once is plan
    assert plan.frozen is True


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------

def test_duplicate_site_id_raises_typed_error():
    with pytest.raises(DuplicateSiteError):
        preflight(
            run_id="run-6", profile_revision="p", calibration_identity="c",
            geometry=GEOMETRY, frame_ids=FRAMES,
            site_qualifications=(make_qual("A", 0, 0), make_qual("A", 1, 1)),
            donor_availability=lambda sid, fid: DONORS_AVAILABLE,
        )


def test_empty_run_raises():
    with pytest.raises(ValueError):
        preflight(
            run_id="run-7", profile_revision="p", calibration_identity="c",
            geometry=GEOMETRY, frame_ids=(),
            site_qualifications=(), donor_availability=lambda sid, fid: DONORS_AVAILABLE,
        )
