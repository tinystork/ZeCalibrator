"""LOT6 tests: net-benefit harness (paired measurement, no verdict).

Covered requirements (mission ZC-SENSOR-P3B-LOT6-NET-BENEFIT-HARNESS):

1. measured reduction — on a scenario with a known injected anomalous site,
   branch B reduces the site residual significantly vs A (quantitative);
2. control non-degradation — a control (non-reconstructed) site has branch B
   identical to A;
3. untouched-domain invariance = 0 outside the site neighbourhood;
4. localization — the B − A difference at the site exceeds the same measure at
   shifted positions (spatial witness);
5. determinism — same seed ⇒ same measures;
6. no threshold / no selection — AST scan (no argmax / sorted / best) and no
   acceptance verdict is produced;
7. atomicity — a site forbidden by the LOT3 plan (donors unavailable on one
   frame) is reconstructed on NO frame of the run;
8. censoring — no quantitative inference on a censored measurement.
"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from research.p3b.generator import generate_corpus
from research.p3b.model import EpochSpec, FrameSpec, GroupSpec, ScenarioSpec, SiteSpec
from research.p3b.net_benefit import (
    NetBenefitHarnessResult,
    SiteNetBenefit,
    representative_dark,
    run_net_benefit_harness,
)
from research.p3b.parameters import all_params
from research.p3b.preparation_plan import (
    DONORS_UNAVAILABLE,
    RUN_WIDE_ABSTAIN,
    RUN_WIDE_ELIGIBLE,
)
from research.p3b.reconstruction import (
    DONOR_OFFSETS,
    RESEARCH_WITNESS_ONLY,
    donor_offsets_available,
    reconstruct_site_pixel,
)
from tests.p3b.conftest import make_sensor

REPO_ROOT = Path(__file__).resolve().parents[2]
NET_BENEFIT_PATH = REPO_ROOT / "research" / "p3b" / "net_benefit.py"
RECONSTRUCTION_PATH = REPO_ROOT / "research" / "p3b" / "reconstruction.py"

# The candidate site's injected amplitude (INTERMITTENT_TWO_STATE, always on).
CANDIDATE_AMP = 2500.0


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _nb_scenario(seed=0, shape=(64, 64), candidate_params=(), extra_sites=()):
    """2 epochs × 2 groups × 2 light frames + 4 darks.

    Sites: a reconstruction candidate (INTERMITTENT_TWO_STATE, always on) and a
    control site (STABLE_ANOMALY_CORRECTED_BY_DARK) whose residual a
    representative dark already corrects.
    """
    sensor = make_sensor(shape=shape)
    dark_frames = tuple(
        FrameSpec(frame_id=f"d{i}", frame_type="dark", epoch_id="cal", group_id="calg", ordinal=i)
        for i in range(4)
    )
    light = []
    for e in range(2):
        for g in range(2):
            for i in range(2):
                light.append(
                    FrameSpec(
                        frame_id=f"l{e}{g}{i}",
                        frame_type="light",
                        epoch_id=f"e{e}",
                        group_id=f"g{g}",
                        ordinal=i,
                    )
                )
    epochs = (
        EpochSpec("cal", (GroupSpec("calg", dark_frames),)),
        EpochSpec("e0", (GroupSpec("g0", tuple(light[:2])), GroupSpec("g1", tuple(light[2:4])))),
        EpochSpec("e1", (GroupSpec("g0", tuple(light[4:6])), GroupSpec("g1", tuple(light[6:8])))),
    )
    candidate = SiteSpec(
        site_id="cand",
        x=32,
        y=24,
        cfa_class="INTERMITTENT_TWO_STATE",
        params=(("on_value_adu", CANDIDATE_AMP), ("off_value_adu", 0.0), ("period_frames", 1)),
    )
    control = SiteSpec(site_id="ctrl", x=24, y=40, cfa_class="STABLE_ANOMALY_CORRECTED_BY_DARK")
    sites = (candidate, control) + tuple(extra_sites)
    return ScenarioSpec(name="nb", seed=seed, sensor=sensor, epochs=epochs, sites=sites)


def _corpus_and_dark(scenario, tmp_path):
    corpus = generate_corpus(scenario, tmp_path / "corpus")
    dark = representative_dark(scenario, corpus.frame_arrays)
    return corpus, dark


def _site_by_id(result, site_id):
    for s in result.sites:
        if s.site_id == site_id:
            return s
    raise KeyError(site_id)


# ---------------------------------------------------------------------------
# Requirement 1 — measured reduction
# ---------------------------------------------------------------------------


def test_residual_reduction_is_significant_and_hand_computable(tmp_path):
    scenario = _nb_scenario(seed=0)
    corpus, dark = _corpus_and_dark(scenario, tmp_path)
    result = run_net_benefit_harness(scenario, corpus.frame_arrays, dark_reference=dark)

    cand = _site_by_id(result, "cand")
    assert cand.reconstructed is True
    assert cand.run_wide_applicability == RUN_WIDE_ELIGIBLE

    # The site is always on (period 1): the baseline residual at the site is the
    # injected amplitude, and reconstruction (median of same-CFA donors) removes
    # essentially all of it.
    assert cand.baseline_residual_adu == pytest.approx(CANDIDATE_AMP, abs=120)
    assert cand.residual_after_adu == pytest.approx(0.0, abs=120)
    assert cand.pair.residual_reduction_at_site == pytest.approx(CANDIDATE_AMP, abs=120)
    # Fraction of the baseline residual removed.
    assert cand.pair.residual_reduction_fraction == pytest.approx(1.0, abs=0.08)
    # The reduction is in ADU and the baseline is non-zero: a real measurement.
    assert cand.baseline_residual_adu > 0


def test_profile_is_reported_and_center_dominated_in_A(tmp_path):
    scenario = _nb_scenario(seed=0)
    corpus, dark = _corpus_and_dark(scenario, tmp_path)
    result = run_net_benefit_harness(scenario, corpus.frame_arrays, dark_reference=dark)

    cand = _site_by_id(result, "cand")
    # Never a bare number: the mean centred stamp is present on both branches.
    assert cand.mean_profile_a != ()
    assert cand.mean_profile_b != ()
    side = 2 * cand.profile_radius + 1
    assert len(cand.mean_profile_a) == side
    assert len(cand.mean_profile_a[0]) == side
    # In branch A the centre pixel carries most of the profile residual…
    assert cand.center_fraction_a == pytest.approx(1.0, abs=0.25)
    # …and after reconstruction the centre no longer dominates.
    assert cand.center_fraction_b < cand.center_fraction_a


# ---------------------------------------------------------------------------
# Requirement 2 — control non-degradation (branch B == branch A off the site)
# ---------------------------------------------------------------------------


def test_control_site_is_untouched(tmp_path):
    scenario = _nb_scenario(seed=0)
    corpus, dark = _corpus_and_dark(scenario, tmp_path)
    result = run_net_benefit_harness(scenario, corpus.frame_arrays, dark_reference=dark)

    ctrl = _site_by_id(result, "ctrl")
    assert ctrl.reconstructed is False
    # The representative dark already corrects the control site: its baseline
    # residual is ~0, and branch B changes nothing at it.
    assert ctrl.pair.residual_reduction_at_site == pytest.approx(0.0, abs=1e-9)
    # The control site's own collateral is all-zero: no reconstruction means no
    # effect anywhere, so untouched-domain invariance and photometric impact are 0.
    ctrl_coll = ctrl.pair.collateral_degradation
    assert ctrl_coll.untouched_domain_invariance == 0.0
    assert ctrl_coll.photometric_impact == pytest.approx(0.0, abs=1e-9)
    assert ctrl_coll.new_artifact_created is False
    assert ctrl_coll.cfa_structure_preserved is True


# ---------------------------------------------------------------------------
# Requirement 3 — untouched-domain invariance = 0 outside the neighbourhood
# ---------------------------------------------------------------------------


def test_untouched_domain_invariance_is_zero(tmp_path):
    scenario = _nb_scenario(seed=0)
    corpus, dark = _corpus_and_dark(scenario, tmp_path)
    result = run_net_benefit_harness(scenario, corpus.frame_arrays, dark_reference=dark)

    cand = _site_by_id(result, "cand")
    assert cand.pair.collateral_degradation.untouched_domain_invariance == 0.0
    assert cand.pair.collateral_degradation.new_artifact_created is False
    assert cand.pair.collateral_degradation.cfa_structure_preserved is True
    # Local background is not biased and not noisier (reconstruction is a single
    # pixel; the surrounding same-CFA annulus is untouched).
    assert cand.pair.collateral_degradation.local_background_bias == pytest.approx(0.0, abs=1e-9)
    assert cand.pair.collateral_degradation.local_noise == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Requirement 4 — localization (spatial witness)
# ---------------------------------------------------------------------------


def test_difference_is_localized_at_the_site(tmp_path):
    scenario = _nb_scenario(seed=0)
    corpus, dark = _corpus_and_dark(scenario, tmp_path)
    result = run_net_benefit_harness(scenario, corpus.frame_arrays, dark_reference=dark)

    cand = _site_by_id(result, "cand")
    loc = cand.localization
    # The effect at the site is non-zero…
    assert loc.site_magnitude > 0
    # …and exactly zero at every shifted witness position.
    assert all(m == 0.0 for m in loc.shifted_magnitudes)
    # Fully concentrated at the site.
    assert loc.concentration == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Requirement 5 — determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_measures(tmp_path):
    scenario = _nb_scenario(seed=7)
    corpus, dark = _corpus_and_dark(scenario, tmp_path)
    r1 = run_net_benefit_harness(scenario, corpus.frame_arrays, dark_reference=dark)
    r2 = run_net_benefit_harness(scenario, corpus.frame_arrays, dark_reference=dark)
    assert r1 == r2


def test_different_seed_different_frames_same_structure(tmp_path):
    s1 = _nb_scenario(seed=1)
    s2 = _nb_scenario(seed=2)
    c1, d1 = _corpus_and_dark(s1, tmp_path / "a")
    c2, d2 = _corpus_and_dark(s2, tmp_path / "b")
    r1 = run_net_benefit_harness(s1, c1.frame_arrays, dark_reference=d1)
    r2 = run_net_benefit_harness(s2, c2.frame_arrays, dark_reference=d2)
    # Same structure (sites, applicability), different numeric noise.
    assert [s.site_id for s in r1.sites] == [s.site_id for s in r2.sites]
    assert [s.run_wide_applicability for s in r1.sites] == [
        s.run_wide_applicability for s in r2.sites
    ]


# ---------------------------------------------------------------------------
# Requirement 6 — no threshold / no selection, no acceptance verdict
# ---------------------------------------------------------------------------


def test_harness_modules_select_no_optimum_ast():
    for path in (NET_BENEFIT_PATH, RECONSTRUCTION_PATH):
        tree = ast.parse(path.read_text())
        forbidden_calls = {"argmax", "argmin"}
        forbidden_names = {"max", "min", "sorted", "argmax", "argmin"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                    raise AssertionError(f"{path.name} selects via .{node.func.attr}")
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                    raise AssertionError(f"{path.name} selects via {node.func.id}(...)")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_calls:
                raise AssertionError(f"{path.name} references .{node.attr}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for token in ("best", "optimal", "optimum", "rank", "select"):
                    if token in node.value.lower():
                        raise AssertionError(f"{path.name} mentions {token!r} in a string")


def test_no_acceptance_verdict_is_produced():
    # The harness result carries measurements only; no field decides "accept",
    # "approve", "net benefit established", or any operating point.
    for cls in (NetBenefitHarnessResult, SiteNetBenefit):
        names = {f.name.lower() for f in fields(cls)}
        for token in ("accept", "approve", "verdict", "threshold", "decision", "established"):
            assert token not in names, f"{cls.__name__} exposes a decision field {token!r}"


def test_harness_constants_are_marked_and_not_thresholds():
    for name, (_value, kind) in all_params().items():
        if name.startswith("NET_BENEFIT_") or name.startswith("RESEARCH_WITNESS_"):
            assert "THRESHOLD" not in name.upper()


# ---------------------------------------------------------------------------
# Requirement 7 — run-wide atomicity (LOT3)
# ---------------------------------------------------------------------------


def test_atomicity_no_reconstruction_when_donors_unavailable_on_one_frame(tmp_path):
    scenario = _nb_scenario(seed=0)
    corpus, dark = _corpus_and_dark(scenario, tmp_path)

    # Force the candidate's donors unavailable on exactly one light frame.
    def donors(site_id, frame_id):
        if site_id == "cand" and frame_id == "l000":
            return DONORS_UNAVAILABLE
        return "AVAILABLE"

    result = run_net_benefit_harness(
        scenario, corpus.frame_arrays, dark_reference=dark, donor_availability=donors
    )
    cand = _site_by_id(result, "cand")
    assert cand.run_wide_applicability == RUN_WIDE_ABSTAIN
    assert cand.reconstructed is False
    # No reconstruction on any frame of the run: the effect is exactly zero.
    assert cand.pair.residual_reduction_at_site == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Requirement 8 — censoring (no quantitative inference)
# ---------------------------------------------------------------------------


def test_censored_site_yields_no_quantitative_inference(tmp_path):
    sensor = make_sensor(shape=(64, 64))
    dark_frames = tuple(
        FrameSpec(frame_id=f"d{i}", frame_type="dark", epoch_id="cal", group_id="calg", ordinal=i)
        for i in range(4)
    )
    light = [
        FrameSpec(frame_id=f"l{i}", frame_type="light", epoch_id="e0", group_id="g0", ordinal=i)
        for i in range(2)
    ]
    scenario = ScenarioSpec(
        name="cen",
        seed=0,
        sensor=sensor,
        epochs=(
            EpochSpec("cal", (GroupSpec("calg", dark_frames),)),
            EpochSpec("e0", (GroupSpec("g0", tuple(light)),)),
        ),
        sites=(SiteSpec(site_id="c", x=32, y=32, cfa_class="CENSORED_ANOMALY", censored=True),),
    )
    corpus = generate_corpus(scenario, tmp_path / "corpus")
    dark = representative_dark(scenario, corpus.frame_arrays)

    result = run_net_benefit_harness(scenario, corpus.frame_arrays, dark_reference=dark)
    site = _site_by_id(result, "c")
    assert site.censored is True
    assert site.reconstructed is False
    # Censored → no quantitative residual reduction is derivable.
    assert site.pair.residual_reduction_at_site is None
    assert site.baseline_residual_adu is None
    assert site.n_frames == 0


def test_nearby_star_integrity_with_published_distance(tmp_path):
    # A nearby star (a static, non-reconstructed sky structure) whose integrity
    # the collateral controls must verify — with the published peak<->site
    # distance (lesson P2I/S2).
    star = SiteSpec(site_id="star", x=44, y=24, cfa_class="UNDERSAMPLED_STAR_CORE")
    scenario = _nb_scenario(seed=0, extra_sites=(star,))
    corpus, dark = _corpus_and_dark(scenario, tmp_path)
    result = run_net_benefit_harness(
        scenario, corpus.frame_arrays, dark_reference=dark, nearby_star=(24, 44)
    )

    cand = _site_by_id(result, "cand")
    coll = cand.pair.collateral_degradation
    # The star is 12 px from the site along x — the distance is published.
    assert coll.nearby_star_distance_px == pytest.approx(12.0)
    # Reconstruction at the site leaves the star exactly untouched.
    assert coll.nearby_star_peak_delta == pytest.approx(0.0, abs=1e-9)
    assert coll.nearby_star_core_delta == pytest.approx(0.0, abs=1e-9)
    # And the star site itself is not reconstructed.
    star_site = _site_by_id(result, "star")
    assert star_site.reconstructed is False


# ---------------------------------------------------------------------------
# Operator unit tests (frozen research witness only)
# ---------------------------------------------------------------------------


def test_operator_has_exactly_36_donors_and_only_research_witness():
    assert len(DONOR_OFFSETS) == 36
    assert RESEARCH_WITNESS_ONLY == "RESEARCH_WITNESS_ONLY"


def test_operator_rejects_unknown_operator_id():
    import numpy as np

    frame = np.zeros((32, 32), dtype=np.float64)
    with pytest.raises(ValueError):
        reconstruct_site_pixel(frame, 16, 16, "GRBG", operator_id="SOME_OTHER_OPERATOR")


def test_operator_reconstructs_to_local_median():
    import numpy as np

    frame = np.full((32, 32), 100.0, dtype=np.float64)
    frame[16, 16] = 9000.0  # an outlier site
    out = reconstruct_site_pixel(frame, 16, 16, "GRBG")
    assert out == pytest.approx(100.0)


def test_donor_offsets_available_respects_bounds():
    # Interior site: available. Edge site (within ±8 of the border): not.
    assert donor_offsets_available((64, 64), 32, 32, "GRBG") is True
    assert donor_offsets_available((64, 64), 2, 32, "GRBG") is False
    assert donor_offsets_available((64, 64), 32, 2, "GRBG") is False
