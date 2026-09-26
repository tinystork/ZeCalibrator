"""P4-A2 application-lot tests: plan-then-execute, site×run atomicity, §29 gate,
§80 numerical identity, order invariant, confinement, contradiction → abort."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.bpm.lookup import BpmResolution, select_revision
from zecalibrator.bpm.preparation import (
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_PREPARED,
    CalibratedFrame,
    PlanFrozenError,
    PlanInvalidError,
    PlanNotFrozenError,
    PreparationOutcome,
    PreparationPlan,
    SitePlanEntry,
    apply_preparation,
    execute,
    preflight,
)
from zecalibrator.bpm.reconstruction import reconstruct_site_pixel
from zecalibrator.bpm.revision import make_revision
from zecalibrator.bpm.vocabulary import (
    ACTION_STATE_ABSTAIN_CENSORED,
    ACTION_STATE_NO_ACTION_REQUIRED,
    REVISION_STATE_PROMOTED,
)
from zecalibrator.core.calibrate import CalibrationResult, FrameQuality
from zecalibrator.core.dq import CountSummary

from conftest import make_identity, make_site

SHAPE = (12, 12)
CFA = "GRBG"
ROI = (0, 0)


def make_calibration(data, mask=None):
    data = np.asarray(data, dtype=np.float32)
    if mask is None:
        mask = np.zeros(data.shape, dtype=np.uint16)
    mask = np.asarray(mask, dtype=np.uint16)
    return CalibrationResult(
        status="COMPLETED",
        data=np.ascontiguousarray(data),
        mask=np.ascontiguousarray(mask),
        counts=CountSummary.from_mask(mask),
        frame_quality=FrameQuality(saturation_evidence="unknown"),
        precision=None,
        scalars={},
    )


def make_frame(frame_id, data, mask=None):
    return CalibratedFrame(frame_id=frame_id, calibration=make_calibration(data, mask))


def selected(identity, sites):
    rev = make_revision(
        state=REVISION_STATE_PROMOTED, sensor_identity=identity, sites=sites, sequence=1,
    )
    return select_revision(identity, [rev])


def uniform(shape, value):
    return np.full(shape, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Confinement (mission §81)
# ---------------------------------------------------------------------------

def test_no_profile_is_calibration_only():
    res = BpmResolution(outcome=OUTCOME_CALIBRATION_ONLY, reason_code="NO_PROFILE")
    frames = (make_frame("f0", uniform(SHAPE, 100.0)),)
    outcome = apply_preparation(res, frames)
    assert outcome.outcome == OUTCOME_CALIBRATION_ONLY
    assert outcome.plan is None and outcome.results is None


def test_unqualified_profile_is_calibration_only():
    res = BpmResolution(outcome=OUTCOME_CALIBRATION_ONLY, reason_code="UNQUALIFIED_PROFILE")
    outcome = apply_preparation(res, (make_frame("f0", uniform(SHAPE, 100.0)),))
    assert outcome.outcome == OUTCOME_CALIBRATION_ONLY


def test_corrupt_base_is_typed_base_error(tmp_path):
    from zecalibrator.bpm.store import (
        create_bad_pixel_database,
        resolve_bad_pixel_database,
    )

    ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    rev = make_revision(
        state=REVISION_STATE_PROMOTED, sensor_identity=ident,
        sites=[make_site(5, 5)], sequence=1,
    )
    db.add_revision(rev)
    # Tamper the persisted revision (site position changed).
    import json

    rev_path = root / "revisions" / f"{rev.revision_id}.json"
    obj = json.loads(rev_path.read_text(encoding="utf-8"))
    obj["sites"][0]["position"] = [1, 1]
    rev_path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    res = resolve_bad_pixel_database(root, ident)
    assert res.outcome == OUTCOME_BASE_ERROR

    outcome = apply_preparation(res, (make_frame("f0", uniform(SHAPE, 100.0)),))
    assert outcome.outcome == OUTCOME_BASE_ERROR
    assert outcome.provenance  # explicit provenance, never a silent fallback


# ---------------------------------------------------------------------------
# §80 — numerical identity when nothing is prepared
# ---------------------------------------------------------------------------

def test_nothing_prepared_is_bit_identical():
    # An empty revision (no sites) prepares nothing: the output CFA must be
    # numerically identical to ordinary calibration, with provenance-only masks.
    ident = make_identity(shape=SHAPE)
    data = uniform(SHAPE, 100.0)
    data[5, 5] = 123.0
    mask = np.zeros(SHAPE, dtype=np.uint16)
    mask[3, 3] = 0x0001  # one invalid measurement, still present in measurement_dq
    frame = make_frame("f0", data, mask)

    outcome = apply_preparation(selected(ident, ()), (frame,))
    assert outcome.outcome == OUTCOME_PREPARED
    result = outcome.results[0]
    assert np.array_equal(result.prepared_data, data)  # bit-for-bit identical
    assert result.reconstructed_mask.sum() == 0
    assert np.array_equal(result.measurement_dq, mask)  # DQ v1 untouched
    assert np.array_equal(result.usable_mask, mask == 0)


def test_abstained_site_preserves_identity_on_every_frame():
    # An ELIGIBLE site whose donors are unavailable on one frame abstains on ALL
    # frames -> nothing reconstructed -> numerically identical everywhere.
    ident = make_identity(shape=SHAPE)
    site = make_site(5, 5)

    data0 = uniform(SHAPE, 100.0)
    data1 = uniform(SHAPE, 100.0)
    # Frame f1: invalidate all 8 donors of (5,5) -> no donor on that frame.
    mask1 = np.zeros(SHAPE, dtype=np.uint16)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if (dy, dx) == (0, 0):
                continue
            mask1[5 + dy, 5 + dx] = 0x0001

    frames = (make_frame("f0", data0), make_frame("f1", data1, mask1))
    outcome = apply_preparation(selected(ident, [site]), frames)
    assert outcome.outcome == OUTCOME_PREPARED
    for result, expected in zip(outcome.results, (data0, data1)):
        assert np.array_equal(result.prepared_data, expected)
        assert result.reconstructed_mask.sum() == 0


# ---------------------------------------------------------------------------
# §29 gate — only ACTION ELIGIBLE reaches reconstruction
# ---------------------------------------------------------------------------

def test_non_action_eligible_sites_stay_intact():
    ident = make_identity(shape=SHAPE)
    eligible = make_site(5, 5)  # ELIGIBLE
    no_action = make_site(3, 3, action_state=ACTION_STATE_NO_ACTION_REQUIRED)
    abstain = make_site(7, 7, action_state=ACTION_STATE_ABSTAIN_CENSORED)

    data = uniform(SHAPE, 100.0)
    data[5, 5] = 9000.0  # the eligible site's anomalous value
    data[3, 3] = 8000.0  # a "known" NO_ACTION site
    data[7, 7] = 7000.0  # a "known" ABSTAIN site

    frame = make_frame("f0", data)
    outcome = apply_preparation(selected(ident, [eligible, no_action, abstain]), (frame,))
    result = outcome.results[0]

    # Only the ELIGIBLE site was reconstructed.
    assert result.reconstructed_mask[5, 5] == 1
    assert result.reconstructed_mask[3, 3] == 0
    assert result.reconstructed_mask[7, 7] == 0
    # The known-but-not-eligible sites are intact (numerically unchanged).
    assert result.prepared_data[3, 3] == 8000.0
    assert result.prepared_data[7, 7] == 7000.0
    # The eligible site was reconstructed from its ~100 donors.
    assert result.prepared_data[5, 5] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Atomicity site × run (mission §36, P-A normative)
# ---------------------------------------------------------------------------

def test_one_invalid_frame_abstains_site_on_all_frames():
    ident = make_identity(shape=SHAPE)
    site = make_site(5, 5)

    data = [uniform(SHAPE, 100.0) for _ in range(4)]
    for d in data:
        d[5, 5] = 9000.0
    mask3 = np.zeros(SHAPE, dtype=np.uint16)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if (dy, dx) == (0, 0):
                continue
            mask3[5 + dy, 5 + dx] = 0x0001  # frame 3 has no donors for the site

    frames = (
        make_frame("f0", data[0]),
        make_frame("f1", data[1]),
        make_frame("f2", data[2]),
        make_frame("f3", data[3], mask3),
    )
    outcome = apply_preparation(selected(ident, [site]), frames)
    # The site abstains run-wide: reconstructed on NO frame (never 3+1 mix).
    for result in outcome.results:
        assert result.reconstructed_mask.sum() == 0
        assert result.prepared_data[5, 5] == 9000.0


def test_blocked_site_does_not_disable_independent_site():
    ident = make_identity(shape=SHAPE)
    site_a = make_site(3, 3)  # blocked on frame f1
    site_b = make_site(9, 9)  # fine everywhere

    data0 = uniform(SHAPE, 100.0)
    data1 = uniform(SHAPE, 100.0)
    data0[3, 3] = data0[9, 9] = 9000.0
    data1[3, 3] = data1[9, 9] = 9000.0

    mask1 = np.zeros(SHAPE, dtype=np.uint16)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if (dy, dx) == (0, 0):
                continue
            mask1[3 + dy, 3 + dx] = 0x0001  # block A on frame 1 only

    frames = (make_frame("f0", data0), make_frame("f1", data1, mask1))
    outcome = apply_preparation(selected(ident, [site_a, site_b]), frames)

    for result in outcome.results:
        # A is never reconstructed; B is reconstructed on every frame.
        assert result.reconstructed_mask[3, 3] == 0
        assert result.reconstructed_mask[9, 9] == 1
        assert result.prepared_data[9, 9] == pytest.approx(100.0)
        assert result.prepared_data[3, 3] == 9000.0


# ---------------------------------------------------------------------------
# Order invariant (mission §35) — post-calibration CFA only
# ---------------------------------------------------------------------------

def test_order_operator_consumes_post_calibration_cfa():
    # Raw light L and dark D; the dark itself carries the site's anomaly.
    L = uniform(SHAPE, 100.0)
    D = np.zeros(SHAPE, dtype=np.float32)
    y, x = 5, 5
    D[y, x] = 5000.0
    L[y, x] = 6000.0  # 5000 (dark site) + 1000 residual anomaly

    C = (L - D).astype(np.float32)
    assert C[y, x] == 1000.0  # residual survives ordinary calibration
    usable = np.ones(SHAPE, dtype=bool)

    # Correct: reconstruct from the post-calibration CFA C (~100 donors).
    value_post = reconstruct_site_pixel(C, y, x, cfa_phase=CFA, roi_origin=ROI, usable=usable)
    assert value_post == pytest.approx(100.0)

    # Forbidden: reconstruct the raw light first, then subtract the dark.
    value_raw = reconstruct_site_pixel(L, y, x, cfa_phase=CFA, roi_origin=ROI, usable=usable)
    forbidden = value_raw - D[y, x]
    assert forbidden == pytest.approx(-4900.0)  # artefact of order D[site]

    # The two orders differ: the product consumes the post-calibration CFA.
    assert value_post != forbidden


def test_applicator_reconstructs_from_post_calibration_only():
    ident = make_identity(shape=SHAPE)
    site = make_site(5, 5)
    L = uniform(SHAPE, 100.0)
    D = np.zeros(SHAPE, dtype=np.float32)
    D[5, 5] = 5000.0
    L[5, 5] = 6000.0
    C = (L - D).astype(np.float32)

    frame = make_frame("f0", C)
    outcome = apply_preparation(selected(ident, [site]), (frame,))
    result = outcome.results[0]
    # The applicator saw only the post-calibration CFA (C); its reconstruction
    # replaces the 1000 residual with the ~100 median of C's donors — never the
    # raw light's donor pedestal.
    assert result.prepared_data[5, 5] == pytest.approx(100.0)
    assert result.reconstructed_mask[5, 5] == 1


# ---------------------------------------------------------------------------
# Plan first, then execute; contradiction → abort (mission §37)
# ---------------------------------------------------------------------------

def test_preflight_returns_frozen_plan_then_execute():
    ident = make_identity(shape=SHAPE)
    site = make_site(5, 5)
    data = uniform(SHAPE, 100.0)
    data[5, 5] = 9000.0
    frames = (make_frame("f0", data),)

    outcome = preflight(selected(ident, [site]), frames)
    assert outcome.outcome == OUTCOME_PREPARED
    assert outcome.plan.frozen is True

    results = execute(outcome.plan, frames)
    assert results[0].reconstructed_mask[5, 5] == 1
    assert results[0].prepared_data[5, 5] == pytest.approx(100.0)


def test_contradiction_after_freeze_aborts_plan_invalid():
    ident = make_identity(shape=SHAPE)
    site = make_site(5, 5)
    data = uniform(SHAPE, 100.0)
    data[5, 5] = 9000.0
    frame = make_frame("f0", data)

    outcome = preflight(selected(ident, [site]), (frame,))
    assert outcome.plan.site((5, 5)).run_wide_applicability == "ELIGIBLE"

    # After the freeze, invalidate every donor of the site (the world changed).
    mask = np.asarray(frame.calibration.mask)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if (dy, dx) == (0, 0):
                continue
            mask[5 + dy, 5 + dx] = 0x0001

    with pytest.raises(PlanInvalidError):
        execute(outcome.plan, (frame,))


def test_frozen_plan_refuses_mutation():
    ident = make_identity(shape=SHAPE)
    site = make_site(5, 5)
    data = uniform(SHAPE, 100.0)
    data[5, 5] = 9000.0
    outcome = preflight(selected(ident, [site]), (make_frame("f0", data),))
    plan = outcome.plan
    with pytest.raises(PlanFrozenError):
        plan.run_id = "tampered"
    with pytest.raises(PlanFrozenError):
        plan.add_site(
            SitePlanEntry(
                site_id="x", position=(1, 1), action_state="ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
                knowledge_state="KNOWN", per_frame_donor_available=(True,),
            )
        )


def test_execute_requires_frozen_plan():
    plan = PreparationPlan(
        run_id="r", revision_id="rev", geometry=SHAPE, cfa_phase=CFA,
        roi_origin=ROI, frame_ids=("f0",), frozen=False,
    )
    with pytest.raises(PlanNotFrozenError):
        execute(plan, (make_frame("f0", uniform(SHAPE, 100.0)),))


def test_outcome_type_validation():
    with pytest.raises(ValueError):
        PreparationOutcome(outcome="BOGUS", reason_code="")
    with pytest.raises(ValueError):
        PreparationOutcome(outcome=OUTCOME_PREPARED, reason_code="", plan=None)
