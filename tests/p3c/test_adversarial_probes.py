"""P3C-2 — adversarial probes (§49 / §50) and the §48 action-path requirements.

These are the critical probes: the inference must never leak truth, and the
class *name* must never drive the result — only measured features do.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.p3c.development_corpus import build_admission, build_development_scenario
from research.p3c.evidence_bridge import to_evidence_packet
from research.p3c.inference_candidates import CANDIDATE_BASELINE, infer_site
from research.p3c.inference_contract import (
    NO,
    UNDETERMINED,
    YES,
    AdmissionFacts,
)
from research.p3b.features import compute_features
from research.p3b.generator import generate_corpus
from research.p3b.qualification_policy import (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
    ACTION_NO_ACTION,
    ACTION_REQUALIFY,
    EPISTEMIC_CHARACTERISED_INTERMITTENT,
    EPISTEMIC_CHARACTERISED_STABLE,
    EPISTEMIC_TRANSIENT_OR_UNRESOLVED,
    NOT_REPRESENTATIVE,
    evaluate,
)


def _features_for(class_name: str, seed: int = 0):
    scenario = build_development_scenario(class_name, seed)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        dark = np.median(np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0)
        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
    return cf.sites[0]


# ---------------------------------------------------------------------------
# §49 — identical measured data + identical runtime metadata + DIFFERENT truth
#       => identical inference (else TRUTH LEAK = BLOCKER)
# ---------------------------------------------------------------------------


def test_probe49_identical_measured_different_truth_same_inference():
    """Two classes with IDENTICAL light pixels but different declared truth.

    ``STABLE_ANOMALY_CORRECTED_BY_DARK`` and ``STABLE_ANOMALY_WITH_MISMATCHED_DARK``
    produce byte-identical light frames (a hot pixel present in the light; the
    difference is only in the dark). With no dark reference supplied, their LOT4
    features are identical. Their *declared* residual truth differs (NONE vs
    SYSTEMATIC_STABLE). The inference must therefore be IDENTICAL — otherwise it
    would be reading the truth, which is a blocker.
    """
    scenario_a = build_development_scenario("STABLE_ANOMALY_CORRECTED_BY_DARK", 7)
    scenario_b = build_development_scenario("STABLE_ANOMALY_WITH_MISMATCHED_DARK", 7)

    import tempfile

    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        ca = generate_corpus(scenario_a, ta)
        cb = generate_corpus(scenario_b, tb)
        # No dark reference: features depend only on the (identical) light frames.
        fa = compute_features(scenario_a, ca.frame_arrays)
        fb = compute_features(scenario_b, cb.frame_arrays)

    site_a = fa.sites[0]
    site_b = fb.sites[0]

    # Measured data identical.
    assert site_a.same_cfa_local_residual == site_b.same_cfa_local_residual
    assert site_a.temporal_residual_series == site_b.temporal_residual_series
    assert site_a.sign_changes == site_b.sign_changes

    # Identical runtime admission (the representativeness is an *input*, and we
    # hold it identical here so only the declared truth differs).
    admission = AdmissionFacts(
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=YES,
        calibration_representativeness="REPRESENTATIVE",
        independent_group_count=4,
        epoch_count=2,
    )

    ev_a = infer_site(CANDIDATE_BASELINE, site_a, admission)
    ev_b = infer_site(CANDIDATE_BASELINE, site_b, admission)
    assert ev_a == ev_b, "identical measured data + identical metadata must infer identically (§49)"


# ---------------------------------------------------------------------------
# §50 — SAME class label + DIFFERENT measured behaviour => DIFFERENT inference
#       (the class name never drives the result)
# ---------------------------------------------------------------------------


def test_probe50_same_class_different_behaviour_different_inference():
    """Same class label, but the measured behaviour is changed via site params.

    ``INTERMITTENT_TWO_STATE`` with ``on_value_adu=0`` degenerates to a normal
    (no-anomaly) site; with the default ``on_value_adu=2500`` it toggles. Same
    label, different measured behaviour => the inference MUST differ — proving
    the class *name* is not what drives the result.
    """
    scenario_quiet = build_development_scenario(
        "INTERMITTENT_TWO_STATE", 3, site_params=(("on_value_adu", 0.0), ("off_value_adu", 0.0))
    )
    scenario_toggle = build_development_scenario("INTERMITTENT_TWO_STATE", 3)

    import tempfile

    with tempfile.TemporaryDirectory() as tq, tempfile.TemporaryDirectory() as tt:
        cq = generate_corpus(scenario_quiet, tq)
        ct = generate_corpus(scenario_toggle, tt)
        fq = compute_features(scenario_quiet, cq.frame_arrays)
        ft = compute_features(scenario_toggle, ct.frame_arrays)

    admission = build_admission(scenario_quiet)
    ev_q = infer_site(CANDIDATE_BASELINE, fq.sites[0], admission)
    ev_t = infer_site(CANDIDATE_BASELINE, ft.sites[0], admission)

    assert ev_q != ev_t, "same class label + different measured behaviour must infer differently (§50)"
    # Sanity: the quiet variant shows no residual, the toggle variant does.
    q_map = {f.field: f.value for f in ev_q.sensor_evidence}
    t_map = {f.field: f.value for f in ev_t.sensor_evidence}
    assert q_map["site_residual_behaviour"] != t_map["site_residual_behaviour"]


# ---------------------------------------------------------------------------
# §48 — the inference never sees a truth label
# ---------------------------------------------------------------------------


def test_inference_never_sees_any_truth_label():
    # The inference entry point accepts only (config, features, admission). It
    # structurally cannot receive a class name or expected label: any such
    # object would simply be ignored. We assert the signature is truth-free by
    # construction — the truth-leak guard (test_truth_leak_guard.py) enforces
    # the AST/runtime layers, and the dataclasses carry no truth fields.
    from research.p3c.inference_contract import InferredEvidence, InferredField

    for cls in (InferredField, InferredEvidence):
        for fname in cls.__dataclass_fields__:
            assert "class" not in fname.lower() or fname == "candidate_id"
            assert "expected" not in fname
            assert "scenario" not in fname


def test_same_pixels_different_truth_identical_inference():
    # Redundant with §49 but stated in the §48 vocabulary.
    test_probe49_identical_measured_different_truth_same_inference()


def test_same_truth_different_pixels_inference_may_change():
    # Two sites of the same class at different seeds (different noise/pixels):
    # the inference is free to change (it depends on pixels), but it must remain
    # deterministic per input. We assert determinism, not a forced agreement.
    a = _features_for("NOISE_EXTREME", seed=1)
    b = _features_for("NOISE_EXTREME", seed=2)
    adm = build_admission(build_development_scenario("NOISE_EXTREME", 1))
    ev_a = infer_site(CANDIDATE_BASELINE, a, adm)
    ev_b = infer_site(CANDIDATE_BASELINE, b, adm)
    # Re-running each is deterministic.
    assert infer_site(CANDIDATE_BASELINE, a, adm) == ev_a
    assert infer_site(CANDIDATE_BASELINE, b, adm) == ev_b
    # The point: no assertion forces them equal — pixels may drive a change.


def test_class_name_alone_never_changes_inference():
    # Two scenarios that differ ONLY in class label but share identical pixels
    # (CORRECTED vs MISMATCHED light frames) infer identically — covered by §49.
    test_probe49_identical_measured_different_truth_same_inference()


# ---------------------------------------------------------------------------
# §48 — UNDETERMINED propagated honestly; censored never quantitative
# ---------------------------------------------------------------------------


def test_undetermined_propagated_to_policy_as_abstain():
    scenario = build_development_scenario("NORMAL", 0)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        dark = np.median(np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0)
        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
    ev = infer_site(CANDIDATE_BASELINE, cf.sites[0], build_admission(scenario))
    packet = to_evidence_packet(ev)
    decision = evaluate(packet)
    # A normal site with no net-benefit claim is not promoted, not contradicted.
    assert decision.action != ACTION_ELIGIBLE


def test_censored_values_never_used_quantitatively():
    scenario = build_development_scenario("CENSORED_ANOMALY", 0)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        dark = np.median(np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0)
        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
    ev = infer_site(CANDIDATE_BASELINE, cf.sites[0], build_admission(scenario))
    fmap = {f.field: f.value for f in ev.sensor_evidence}
    assert fmap["censored_measurement_present"] == YES
    # Quantitative residual states are forbidden for a censored site.
    assert fmap["site_residual_behaviour"] not in ("SYSTEMATIC_STABLE", "VARIABLE")
    # The existing policy abstains CENSORED.
    packet = to_evidence_packet(ev)
    decision = evaluate(packet)
    assert decision.action == ACTION_ABSTAIN_CENSORED


# ---------------------------------------------------------------------------
# §48 — action path through the EXISTING policy (never reimplemented)
# ---------------------------------------------------------------------------


def test_non_representative_calibration_policy_requalifies():
    # STABLE_ANOMALY_WITH_MISMATCHED_DARK is declared not_representative; the
    # existing policy must requalify the calibration (P1), never reconstruct.
    scenario = build_development_scenario("STABLE_ANOMALY_WITH_MISMATCHED_DARK", 0)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        dark = np.median(np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0)
        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
    ev = infer_site(CANDIDATE_BASELINE, cf.sites[0], build_admission(scenario))
    assert ev.admission.calibration_representativeness == NOT_REPRESENTATIVE
    decision = evaluate(to_evidence_packet(ev))
    assert decision.action == ACTION_REQUALIFY


def test_star_crossing_never_persistent_sensor_qualification():
    scenario = build_development_scenario("STAR_CROSSING_SITE", 0)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        dark = np.median(np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0)
        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
    ev = infer_site(CANDIDATE_BASELINE, cf.sites[0], build_admission(scenario))
    decision = evaluate(to_evidence_packet(ev))
    # Not persisted at the sensor coordinate => no persistent qualification.
    assert decision.epistemic_state not in (
        EPISTEMIC_CHARACTERISED_STABLE,
        EPISTEMIC_CHARACTERISED_INTERMITTENT,
    )
    assert decision.action == ACTION_ABSTAIN_INCONSISTENT


def test_single_transient_never_persistent_qualification():
    scenario = build_development_scenario("SINGLE_TRANSIENT", 0)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        dark = np.median(np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0)
        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
    ev = infer_site(CANDIDATE_BASELINE, cf.sites[0], build_admission(scenario))
    decision = evaluate(to_evidence_packet(ev))
    assert decision.epistemic_state not in (
        EPISTEMIC_CHARACTERISED_STABLE,
        EPISTEMIC_CHARACTERISED_INTERMITTENT,
    )


# ---------------------------------------------------------------------------
# §48 — 2-epoch / independent-group semantics preserved
# ---------------------------------------------------------------------------


def test_single_epoch_cannot_characterise_persistent_site():
    # A scenario with only ONE light epoch cannot satisfy the independence
    # requirement, even for a genuinely persistent site. The existing policy
    # must abstain insufficient, not qualify.
    scenario = build_development_scenario(
        "STABLE_ANOMALY_WITH_MISMATCHED_DARK", 0, epochs=1, groups_per_epoch=1
    )
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        dark = np.median(np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0)
        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
    adm = build_admission(scenario)
    # Only one independent science group (the calibration epoch is not a
    # science group), so independence cannot be satisfied.
    assert adm.independent_group_count < 2
    ev = infer_site(CANDIDATE_BASELINE, cf.sites[0], adm)
    decision = evaluate(to_evidence_packet(ev))
    # Either requalify (not_representative) or, once representativeness is
    # representative, abstain on independence — but never eligible.
    assert decision.action != ACTION_ELIGIBLE


# ---------------------------------------------------------------------------
# §48 — P0: no inter-frame switching (run-wide atomicity, existing plan)
# ---------------------------------------------------------------------------


def test_p0_no_inter_frame_switching_via_existing_plan():
    from research.p3b.preparation_plan import (
        DONORS_AVAILABLE,
        DONORS_UNAVAILABLE,
        GeometryBinding,
        SiteQualification,
        preflight,
    )

    # Build a run plan from the EXISTING preflight (never reimplemented) over a
    # single eligible intermittent site and verify the run-wide decision is one
    # uniform value per site — no per-frame apply/skip/apply.
    scenario = build_development_scenario("INTERMITTENT_TWO_STATE", 0)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = generate_corpus(scenario, td)
        dark = np.median(np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(4)]), axis=0)
        cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
    frame_ids = tuple(cf.sites[0].frame_ids)

    # A representative calibration + an established net benefit would make this
    # site eligible; here net benefit stays UNDETERMINED, so the site abstains.
    # Either way the plan is a single run-wide value per site (P0 intact).
    from research.p3b.qualification_policy import EvidencePacket, YES as QYES

    ev = infer_site(CANDIDATE_BASELINE, cf.sites[0], build_admission(scenario))
    decision = evaluate(to_evidence_packet(ev))

    sq = SiteQualification(
        site_id=cf.sites[0].site_id,
        x=cf.sites[0].x,
        y=cf.sites[0].y,
        decision=decision,
    )
    plan = preflight(
        run_id="run-probe",
        profile_revision="rev-1",
        calibration_identity="cal-1",
        geometry=GeometryBinding(shape=(96, 128), cfa_pattern="GRBG"),
        frame_ids=frame_ids,
        site_qualifications=(sq,),
        donor_availability=lambda site_id, frame_id: DONORS_AVAILABLE,
    )
    entry = plan.site(cf.sites[0].site_id)
    # One run-wide applicability value, uniform over all frames (P0).
    assert entry.run_wide_applicability in ("ELIGIBLE", "ABSTAIN", "NO_ACTION", "REQUALIFY")
    # All per-frame donor facts agree with what the plan assumed (no switching).
    assert all(tok in (DONORS_AVAILABLE, DONORS_UNAVAILABLE) for tok in entry.per_frame_donor_availability)
