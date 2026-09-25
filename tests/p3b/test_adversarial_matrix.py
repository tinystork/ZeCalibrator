"""LOT8: the adversarial test matrix — cases a naive implementation would get wrong.

This is the lot that proves the bench *bites*: it records, as a permanent gate,
the exact refusals the policy/harness must produce on adversarial inputs, so a
future implementation that quietly re-introduces the naive behaviour fails.

Three parts, mirroring the mission:

* **A — the 10 adversarial rows** (one test each, with the expected refusal
  stated explicitly in the assertion, never implied);
* **B — the P0 regression** (a *voluntary* naive per-frame executor is provided
  in the test; we show its sequence is mixed and that the frozen
  :class:`research.p3b.preparation_plan.PreparationPlan` makes that mixed
  sequence unrepresentable — uniform execution per site, never per frame);
* **C — perimeter regressions** (scope invariants asserted without touching
  ``src/``: import boundary, no capability, no DQ-v1 redefinition, native
  unbinned base-0 coordinates, typed refusals for unsupported geometry, and the
  explicit statement that ``measurement_dq``/``usable`` are **not** modelled
  here).

Everything is scoped and fast: qualification rows run on declared facts alone
(no FITS, no numpy in the decision path); only the coordinate/geometry rows
materialise a tiny corpus.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from research.p3b.declared_facts import build_declared_facts, evidence_packet
from research.p3b.metrics import build_outcomes, is_promoted, is_qualified, metrics_for_scenario
from research.p3b.model import (
    AggregateSpec,
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SiteSpec,
    compute_bookkeeping,
)
from research.p3b.preparation_plan import (
    DONORS_AVAILABLE,
    DONORS_UNAVAILABLE,
    RC_DONORS_UNAVAILABLE_RUN_WIDE,
    RUN_WIDE_ABSTAIN,
    RUN_WIDE_ELIGIBLE,
    RUN_WIDE_REQUALIFY,
    GeometryBinding,
    SiteQualification,
    preflight,
)
from research.p3b.qualification_policy import (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
    ACTION_NO_ACTION,
    ACTION_REQUALIFY,
    EPISTEMIC_CHARACTERISED_STABLE,
    EPISTEMIC_CENSORED,
    EPISTEMIC_OBSERVED_CANDIDATE,
    EPISTEMIC_TRANSIENT_OR_UNRESOLVED,
    RC_CALIBRATION_NOT_REPRESENTATIVE,
    RC_EVIDENCE_CENSORED,
    RC_INSUFFICIENT_EPOCHS,
    RC_INSUFFICIENT_INDEPENDENT_GROUPS,
    RC_NO_ACTION_NEEDED_AFTER_CALIBRATION,
    RC_NOT_PERSISTENT_IN_SENSOR_COORDINATES,
    RC_TRANSIENT_ONLY,
    CensoredInferenceError,
    QualificationDecision,
    evaluate,
)
from tests.p3b.conftest import make_scenario, make_sensor

_REPO_ROOT = Path(__file__).resolve().parents[2]
_P3B_DIR = _REPO_ROOT / "research" / "p3b"
_API_V1_DIR = _REPO_ROOT / "src" / "zecalibrator" / "api" / "v1"

FRAMES = ("f0", "f1", "f2", "f3")


# ---------------------------------------------------------------------------
# Scenario builders (declared facts only — no FITS generation needed)
# ---------------------------------------------------------------------------


def _scenario(sites=(), *, epochs=2, groups_per_epoch=2, frames_per_group=2,
              n_darks=1, seed=0) -> ScenarioSpec:
    """A small synthetic scenario with representative dark frames present.

    Two epochs x two groups x two light frames (plus one dark) is enough for a
    *candidate* to reach the independence requirement, so any refusal we assert
    is caused by the adversarial fact itself, not by a missing dark or missing
    independence.
    """
    sensor = make_sensor()
    epoch_specs = []
    groups0 = []
    if n_darks:
        darks = tuple(
            FrameSpec(frame_id=f"dark{i}", frame_type="dark", epoch_id="e0",
                      group_id="gd", ordinal=i)
            for i in range(n_darks)
        )
        groups0.append(GroupSpec("gd", darks))
    for g in range(groups_per_epoch):
        frames = tuple(
            FrameSpec(frame_id=f"e0g{g}f{i}", frame_type="light", epoch_id="e0",
                      group_id=f"g{g}", ordinal=i)
            for i in range(frames_per_group)
        )
        groups0.append(GroupSpec(f"g{g}", frames))
    epoch_specs.append(EpochSpec("e0", tuple(groups0)))
    for e in range(1, epochs):
        groups = []
        for g in range(groups_per_epoch):
            frames = tuple(
                FrameSpec(frame_id=f"e{e}g{g}f{i}", frame_type="light",
                          epoch_id=f"e{e}", group_id=f"g{g}", ordinal=i)
                for i in range(frames_per_group)
            )
            groups.append(GroupSpec(f"g{g}", frames))
        epoch_specs.append(EpochSpec(f"e{e}", tuple(groups)))
    return make_scenario(name="adv", seed=seed, sensor=sensor,
                         epochs=tuple(epoch_specs), sites=tuple(sites))


def _decision(scenario: ScenarioSpec, site_id: str) -> QualificationDecision:
    declared = build_declared_facts(scenario)
    sf = next(s for s in declared.sites if s.site_id == site_id)
    return evaluate(evidence_packet(sf, declared))


def _declared_site(scenario: ScenarioSpec, site_id: str):
    declared = build_declared_facts(scenario)
    return next(s for s in declared.sites if s.site_id == site_id)


def _eligible_decision() -> QualificationDecision:
    """A fully-qualified ELIGIBLE decision (as LOT2 would emit for a good site)."""
    return QualificationDecision(
        epistemic_state=EPISTEMIC_CHARACTERISED_STABLE,
        representativeness="REPRESENTATIVE",
        action=ACTION_ELIGIBLE,
        reason_codes=("ELIGIBLE",),
        evidence_summary=(),
    )


def _preflight_apply(site_quals, donor_fn):
    plan = preflight(
        run_id="run-adv",
        profile_revision="rev",
        calibration_identity="cal",
        geometry=GeometryBinding(shape=(96, 128), cfa_pattern="GRBG"),
        frame_ids=FRAMES,
        site_qualifications=tuple(site_quals),
        donor_availability=donor_fn,
    )
    result = plan.execute(donor_fn)
    applied = {
        site_id: [rec.applied for frame in result.frames for rec in frame.sites
                  if rec.site_id == site_id]
        for site_id in {q.site_id for q in site_quals}
    }
    return plan, applied


# ---------------------------------------------------------------------------
# A — the 10 adversarial rows
# ---------------------------------------------------------------------------


def test_adversarial_01_old_inadequate_dark_requires_requalify_and_never_reconstructs():
    # "vieux dark inadéquat": a *stable* anomaly whose supplied dark is not
    # representative. The site looks permanently bad, but the science answer is
    # to requalify the additive calibration — never to reconstruct the pixel.
    site = SiteSpec(site_id="a", x=30, y=30,
                    cfa_class="STABLE_ANOMALY_WITH_MISMATCHED_DARK",
                    calibration_representativeness="mismatched")
    d = _decision(_scenario((site,)), "a")

    assert d.action == ACTION_REQUALIFY
    assert d.reason_codes == (RC_CALIBRATION_NOT_REPRESENTATIVE,)

    # At the plan level it stays REQUALIFY even with donors everywhere, so it is
    # never applied on any frame — never reconstructed.
    plan, applied = _preflight_apply(
        (SiteQualification("a", 30, 30, d),),
        lambda sid, fid: DONORS_AVAILABLE,
    )
    assert plan.site("a").run_wide_applicability == RUN_WIDE_REQUALIFY
    assert applied["a"] == [False, False, False, False]


def test_adversarial_02_single_huge_transient_excursion_yields_no_persistent_qualification():
    # One enormous transient excursion must not produce a persistent
    # qualification: the harness abstains (transient-only), never characterizes.
    site = SiteSpec(site_id="b", x=30, y=30, cfa_class="SINGLE_TRANSIENT")
    d = _decision(_scenario((site,)), "b")

    assert d.action == ACTION_ABSTAIN_INCONSISTENT
    assert RC_TRANSIENT_ONLY in d.reason_codes
    assert d.epistemic_state == EPISTEMIC_TRANSIENT_OR_UNRESOLVED
    assert d.action != ACTION_ELIGIBLE


def test_adversarial_03_many_frames_of_one_session_keep_epoch_count_at_one():
    # 20 frames from a single session are one independent group, one epoch —
    # not 20. Adding frames must never inflate independence.
    sensor = make_sensor()
    frames = tuple(
        FrameSpec(frame_id=f"l{i}", frame_type="light", epoch_id="e0",
                  group_id="g0", ordinal=i)
        for i in range(20)
    )
    scenario = make_scenario(
        name="many-frames", seed=0, sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
    )
    bk = compute_bookkeeping(scenario)
    assert bk.observation_count == 20
    assert bk.epoch_count == 1
    assert bk.independent_group_count == 1

    # The independence requirement is therefore NOT met: a residual candidate in
    # this single-session scenario abstains for insufficient independence, and
    # is never characterized as persistent.
    dark = FrameSpec(frame_id="dk0", frame_type="dark", epoch_id="e0",
                     group_id="gd", ordinal=0)
    site = SiteSpec(site_id="c", x=30, y=30, cfa_class="INTERMITTENT_TWO_STATE")
    single = make_scenario(
        name="many-frames-site", seed=0, sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("gd", (dark,)), GroupSpec("g0", frames))),),
        sites=(site,),
    )
    d = _decision(single, "c")
    assert d.action == ACTION_ABSTAIN_INSUFFICIENT
    assert RC_INSUFFICIENT_EPOCHS in d.reason_codes
    assert RC_INSUFFICIENT_INDEPENDENT_GROUPS in d.reason_codes
    assert d.epistemic_state == EPISTEMIC_OBSERVED_CANDIDATE


def test_adversarial_04_aggregate_plus_constituents_feign_no_independence():
    # A derived aggregate (master) of existing frames is a frame, but it is
    # neither a new observation nor a new independent group nor a new epoch.
    sensor = make_sensor()
    frames = tuple(
        FrameSpec(frame_id=f"l{i}", frame_type="light", epoch_id="e0",
                  group_id="g0", ordinal=i)
        for i in range(3)
    )
    base = make_scenario(
        name="base", seed=0, sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
    )
    with_aggregate = make_scenario(
        name="agg", seed=0, sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
        aggregates=(AggregateSpec("master", "median", ("l0", "l1", "l2")),),
    )
    bk_base = compute_bookkeeping(base)
    bk_agg = compute_bookkeeping(with_aggregate)

    # The only thing that changes is frame_count (the materialised master).
    assert bk_agg.frame_count == bk_base.frame_count + 1
    assert bk_agg.observation_count == bk_base.observation_count == 3
    assert bk_agg.independent_group_count == bk_base.independent_group_count == 1
    assert bk_agg.epoch_count == bk_base.epoch_count == 1


def test_adversarial_05_censored_site_abstains_and_never_yields_quantitative_inference():
    # A site clipped by the acquisition hard limit is censored: ABSTAIN_CENSORED,
    # and no quantitative inference (characterised residual / net benefit) may
    # be claimed from it.
    site = SiteSpec(site_id="d", x=30, y=30, cfa_class="CENSORED_ANOMALY",
                    censored=True, hard_limit_adu=60000.0)
    scenario = _scenario((site,))
    d = _decision(scenario, "d")

    assert d.action == ACTION_ABSTAIN_CENSORED
    assert d.reason_codes == (RC_EVIDENCE_CENSORED,)
    assert d.epistemic_state == EPISTEMIC_CENSORED

    # A censored measurement cannot support a characterised residual ...
    from research.p3b.qualification_policy import (
        NET_BENEFIT_ESTABLISHED, NO, RESIDUAL_VARIABLE, YES,
    )
    from research.p3b.qualification_policy import EvidencePacket
    with pytest.raises(CensoredInferenceError):
        evaluate(EvidencePacket(
            sensor_identity_resolved=YES, geometry_compatible=YES,
            calibration_present=YES, calibration_representativeness="REPRESENTATIVE",
            persisted_at_same_sensor_coord=YES, independent_group_count=2, epoch_count=2,
            censored_measurement_present=YES, site_residual_behaviour=RESIDUAL_VARIABLE,
            neighbourhood_residual_stable=YES, transient_only=NO,
            conflicting_evidence=NO, net_benefit_established=NET_BENEFIT_ESTABLISHED,
        ))

    # ... and the harness never promotes or qualifies the censored site, even
    # when a net benefit is injected as a candidate operating point.
    outcomes = build_outcomes(scenario, net_benefit_established=NET_BENEFIT_ESTABLISHED)
    assert is_promoted(outcomes[0]) is False
    assert is_qualified(outcomes[0]) is False
    assert metrics_for_scenario(scenario).censored_mis_inference_count == 0


def test_adversarial_06_star_crossing_same_coordinate_is_never_promoted_as_a_sensor_site():
    # A star whose PSF crosses the same sensor coordinate is a *sky* structure,
    # not a sensor defect: persistence at the sensor coordinate is NO, so the
    # harness must never promote it to a sensor reconstruction candidate.
    site = SiteSpec(site_id="e", x=30, y=30, cfa_class="STAR_CROSSING_SITE")
    scenario = _scenario((site,))
    sf = _declared_site(scenario, "e")
    d = _decision(scenario, "e")

    assert sf.persisted_at_same_sensor_coord == "NO"  # structure ciel, pas capteur
    assert d.action != ACTION_ELIGIBLE
    assert d.epistemic_state == EPISTEMIC_TRANSIENT_OR_UNRESOLVED


def test_adversarial_07_flat_structure_is_never_promoted_to_hot_or_rts():
    # A structure that lives only in the flat is not a persistent sensor-site
    # anomaly: it must never be promoted to a hot pixel / RTS reconstruction.
    site = SiteSpec(site_id="f", x=30, y=30, cfa_class="FLAT_STRUCTURE")
    scenario = _scenario((site,))
    sf = _declared_site(scenario, "f")
    d = _decision(scenario, "f")

    assert sf.persisted_at_same_sensor_coord == "NO"
    assert d.action == ACTION_ABSTAIN_INCONSISTENT
    assert RC_NOT_PERSISTENT_IN_SENSOR_COORDINATES in d.reason_codes
    assert d.action != ACTION_ELIGIBLE


def test_adversarial_08_historically_known_site_corrected_by_representative_dark_is_no_action():
    # A known, persistent site that a representative dark already corrects is
    # NO_ACTION_REQUIRED — and the site *remains in knowledge* (never removed).
    site = SiteSpec(site_id="g", x=30, y=30,
                    cfa_class="STABLE_ANOMALY_CORRECTED_BY_DARK")
    d = _decision(_scenario((site,)), "g")

    assert d.action == ACTION_NO_ACTION
    assert d.reason_codes == (RC_NO_ACTION_NEEDED_AFTER_CALIBRATION,)
    # knowledge is not action: the site stays known (persistent), not erased.
    assert d.epistemic_state == EPISTEMIC_CHARACTERISED_STABLE


def test_adversarial_09_qualified_site_with_one_donorless_frame_abstains_for_the_whole_run():
    # P-A atomicity: an ELIGIBLE site whose donors are unavailable on a *single*
    # frame is reconstructed on *no* frame of the run — never N-1 + 1 original.
    d = _eligible_decision()
    plan, applied = _preflight_apply(
        (SiteQualification("A", 4, 4, d),),
        lambda sid, fid: DONORS_UNAVAILABLE if fid == "f2" else DONORS_AVAILABLE,
    )
    entry = plan.site("A")
    assert entry.action_state == ACTION_ELIGIBLE  # LOT2 still says eligible
    assert entry.run_wide_applicability == RUN_WIDE_ABSTAIN
    assert entry.abstention_reason == RC_DONORS_UNAVAILABLE_RUN_WIDE
    assert applied["A"] == [False, False, False, False]


def test_adversarial_10_another_valid_site_in_the_same_run_keeps_its_eligibility():
    # Atomicity is site x run, never global: a blocked site does not disable an
    # independently qualified site in the same run.
    good = _eligible_decision()
    def donor(sid, fid):
        if sid == "A":
            return DONORS_UNAVAILABLE if fid == "f2" else DONORS_AVAILABLE
        return DONORS_AVAILABLE

    plan, applied = _preflight_apply(
        (SiteQualification("A", 4, 4, good), SiteQualification("B", 20, 20, good)),
        donor,
    )
    assert plan.site("A").run_wide_applicability == RUN_WIDE_ABSTAIN
    assert plan.site("B").run_wide_applicability == RUN_WIDE_ELIGIBLE
    assert applied["A"] == [False, False, False, False]
    assert applied["B"] == [True, True, True, True]


# ---------------------------------------------------------------------------
# B — the P0 regression (permanent gate)
# ---------------------------------------------------------------------------


class NaivePerFrameExecutor:
    """A *voluntary* naive implementation: it decides APPLY/SKIP per frame from
    that frame's donor fact alone.

    This is exactly the implementation the P-A policy forbids
    (``apply / apply / skip / apply`` for one site inside a run). It is provided
    here on purpose so the test can (i) demonstrate the mixed sequence it would
    emit, and (ii) prove the frozen plan makes that sequence impossible.
    """

    def decide(self, per_frame_donors):
        return tuple(
            "APPLY" if token == DONORS_AVAILABLE else "SKIP"
            for token in per_frame_donors
        )


def test_p0_naive_per_frame_switching_is_mixed_but_frozen_plan_makes_it_uniform():
    naive = NaivePerFrameExecutor()

    # (i) The naive executor, deciding per frame, emits the forbidden mixed
    # sequence apply/apply/skip/apply for one site inside one run.
    donor_facts = [DONORS_AVAILABLE, DONORS_AVAILABLE, DONORS_UNAVAILABLE, DONORS_AVAILABLE]
    naive_sequence = naive.decide(donor_facts)
    assert naive_sequence == ("APPLY", "APPLY", "SKIP", "APPLY")
    assert len(set(naive_sequence)) > 1, "naive executor switched between frames"

    # (ii) The frozen PreparationPlan holds a single run-wide value per site and
    # executes uniformly: the mixed sequence above is collapsed to ABSTAIN, and
    # the per-site `applied` flag is identical on every frame. A future
    # implementation that re-introduced per-frame decisions would fail the
    # uniformity assertion below.
    d = _eligible_decision()
    plan, applied = _preflight_apply(
        (SiteQualification("A", 4, 4, d),),
        lambda sid, fid: donor_facts[FRAMES.index(fid)],
    )
    entry = plan.site("A")
    assert entry.run_wide_applicability == RUN_WIDE_ABSTAIN  # never a per-frame mix

    result = plan.execute(lambda sid, fid: donor_facts[FRAMES.index(fid)])
    for frame in result.frames:
        for rec in frame.sites:
            if rec.site_id == "A":
                assert rec.applied is False
    assert applied["A"] == [False, False, False, False]
    # The permanent gate: execution is uniform per site, never per frame.
    assert len(set(applied["A"])) == 1, "execution switched between frames"

    # The mixed naive sequence is *not* what the plan produced.
    plan_flags = ["APPLY" if a else "SKIP" for a in applied["A"]]
    assert plan_flags != list(naive_sequence)


def test_p0_execution_is_uniform_for_every_site_in_a_mixed_run():
    # Two sites in one run — one with a donorless frame, one clean — must each
    # be applied uniformly (all frames or none), never a per-frame mix.
    good = _eligible_decision()
    def donor(sid, fid):
        if sid == "A":
            return DONORS_UNAVAILABLE if fid == "f2" else DONORS_AVAILABLE
        return DONORS_AVAILABLE

    plan, applied = _preflight_apply(
        (SiteQualification("A", 4, 4, good), SiteQualification("B", 20, 20, good)),
        donor,
    )
    for site_id in ("A", "B"):
        assert len(set(applied[site_id])) == 1, f"site {site_id!r} switched per frame"


# ---------------------------------------------------------------------------
# C — perimeter regressions (§40), asserted without modifying src/
# ---------------------------------------------------------------------------


def _py_files(directory: Path):
    return sorted(directory.rglob("*.py"))


def _imported_modules(path: Path):
    """All module names imported (``import`` / ``from``) in one .py file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_scope_api_v1_never_imports_research_p3b():
    # api/v1 must not import research/p3b — no import statement, not merely no
    # documented reference. We scan every .py under src/zecalibrator/api/v1.
    offenders = []
    for path in _py_files(_API_V1_DIR):
        for module in _imported_modules(path):
            if module == "research" or module.startswith("research."):
                offenders.append((str(path), module))
    assert offenders == [], f"api/v1 imports research: {offenders}"


def test_scope_no_capability_added():
    # No capability was added: CAPABILITIES stays the frozen six, and research/p3b
    # declares no capability / provides entry.
    from zecalibrator.api.v1._meta import CAPABILITIES

    assert CAPABILITIES == (
        "calibrate_frame",
        "calibration_library",
        "master_matching",
        "provenance",
        "cancel",
        "calibrate_batch",
    )

    # research/p3b contributes nothing to the capability surface: no module
    # defines a ``CAPABILITIES`` / ``provides`` symbol (checked at the AST level
    # so prose/docstrings mentioning the words are not false positives).
    for path in _py_files(_P3B_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)
            elif isinstance(node, (ast.AnnAssign,)) and isinstance(node.target, ast.Name):
                names.append(node.target.id)
            for name in names:
                assert name.upper() != "CAPABILITIES", (
                    f"research/p3b declares a capability in {path}")
                assert name.lower() != "provides", (
                    f"research/p3b declares a provides entry in {path}")


def test_scope_lot8_does_not_redefine_or_import_dq_v1_or_calibration_result_v1():
    # LOT8 (and the whole research/p3b harness) neither imports zecalibrator's
    # DQ-v1 / CalibrationResult-v1 semantics nor redefines them. Its prepared
    # result is a *different* type (ARCHITECTURE §18.4); DQ v1 is untouched.
    offenders = []
    for path in _py_files(_P3B_DIR):
        for module in _imported_modules(path):
            if module == "zecalibrator" or module.startswith("zecalibrator."):
                offenders.append((str(path), module))
    assert offenders == [], f"research/p3b imports zecalibrator: {offenders}"

    # The prepared-result / harness modules define none of the DQ-v1 symbols.
    for path in _py_files(_P3B_DIR):
        src = path.read_text(encoding="utf-8")
        for symbol in ("measurement_dq", "reconstructed_mask", "usable_mask",
                       "CalibrationResult", "CountSummary"):
            assert symbol not in src, f"{path} redefines DQ-v1 symbol {symbol!r}"


def test_scope_synthetic_frames_use_native_unbinned_base0_coordinates(tmp_path):
    # Synthetic frames are native, unbinned, base-0: binning (1,1), roi_origin
    # (0,0), written verbatim into the FITS header and the manifest.
    from astropy.io import fits

    from research.p3b.generator import generate_corpus

    scenario = _scenario(sites=(SiteSpec(site_id="s", x=30, y=30,
                                         cfa_class="NORMAL"),), n_darks=1)
    result = generate_corpus(scenario, tmp_path / "corpus")

    sensor = result.manifest["scenario"]["sensor"]
    assert sensor["binning"] == [1, 1]
    assert sensor["roi_origin"] == [0, 0]

    with fits.open(result.frame_paths["e0g0f0"]) as hdul:
        hdr = hdul[0].header
        assert hdr["XBINNING"] == 1
        assert hdr["YBINNING"] == 1
        assert hdr["XORGSUBF"] == 0
        assert hdr["YORGSUBF"] == 0


def test_scope_unsupported_geometry_produces_a_typed_refusal_not_silence():
    # Unsupported geometry is refused with a typed outcome, never a silent
    # continue: donor availability returns False, the frozen operator raises a
    # typed ReconstructionError, and an invalid GeometryBinding is refused.
    import numpy as np

    from research.p3b.reconstruction import (
        ReconstructionError,
        donor_offsets_available,
        reconstruct_site_pixel,
    )

    # A site at the corner has donors out of bounds -> typed False (not silent).
    assert donor_offsets_available((96, 128), 0, 0, "GRBG") is False
    # A centre site is fine.
    assert donor_offsets_available((96, 128), 48, 64, "GRBG") is True

    # A degenerate frame with no same-CFA donor -> typed error, not a value.
    with pytest.raises(ReconstructionError):
        reconstruct_site_pixel(np.zeros((2, 2)), 0, 0, "GRBG")

    # An unknown reconstruction operator is a typed refusal, not silence.
    with pytest.raises(ValueError):
        reconstruct_site_pixel(np.zeros((96, 128)), 48, 64, "GRBG", "NOT_AN_OPERATOR")

    # Invalid geometry binding is refused at construction.
    with pytest.raises(ValueError):
        GeometryBinding(shape=(0, 0), cfa_pattern="GRBG")


def test_scope_measurement_dq_and_usable_are_not_modelled_here():
    # The harness exposes no ``measurement_dq`` / ``usable`` concept, so the
    # implication ``measurement_dq == 0  =>  usable`` is *not modelled* here —
    # stated explicitly rather than asserted vacuously. The DQ-v1 notion of
    # "usable" lives only in the v1 engine (src/zecalibrator/core/equations.py),
    # which research/p3b does not import (see the import-boundary test above).
    from research.p3b import preparation_plan as pp
    from research.p3b import qualification_policy as qp

    # The internal decision/plan value objects have no DQ/usable field.
    for obj in (pp.SitePlanEntry, qp.QualificationDecision):
        field_names = {f for f in dir(obj) if not f.startswith("_")}
        assert "measurement_dq" not in field_names
        assert "usable" not in field_names
        assert "usable_mask" not in field_names

    # The decision vocabulary carries no DQ/usable token.
    assert not any("DQ" in code or "USABLE" in code.upper()
                   for code in qp.REASON_CODES)
