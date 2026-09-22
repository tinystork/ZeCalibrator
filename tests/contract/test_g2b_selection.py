"""G2B master selection/ranking contract tests (core level).

Covers the exact deterministic ranking rules (additive most-recent; flat
same-civil-day), the DATE-OBS parser, light acquisition-date derivation, manual
precedence, and the "date never bypasses compatibility" invariant. No
time-dependence: every date is an explicit literal.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    light,
    policy,
    pool,
    request,
)
from zecalibrator.core.descriptors import (
    LightEvidence,
    OpticalIdentity,
    ProcessingProvenance,
)
from zecalibrator.core.matching import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_MATCHED,
    match_calibration,
)
from zecalibrator.core.selection import (
    RANKED_BELOW_WINNER,
    RULE_ADDITIVE,
    RULE_FLAT,
    RULE_MANUAL,
    RULE_SINGLE_CANDIDATE,
    SELECTION_POLICY_VERSION,
    light_acquisition_date,
    parse_date_obs,
    select_role_candidates,
)


# ---------------------------------------------------------------------------
# DATE-OBS parser
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-09-14T23:41:23.350475", datetime(2026, 9, 14, 23, 41, 23, 350475, tzinfo=timezone.utc)),
        ("2023-09-19T18:47:42.009615", datetime(2023, 9, 19, 18, 47, 42, 9615, tzinfo=timezone.utc)),
        ("2026-03-18T06:40:47.541308", datetime(2026, 3, 18, 6, 40, 47, 541308, tzinfo=timezone.utc)),
        ("'2024-01-02T03:04:05'", datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)),
        ("2024-01-02T03:04:05Z", datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)),
        ("2024-01-02", datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)),
        ("2024-01-02T03:04:05+02:00", datetime(2024, 1, 2, 1, 4, 5, tzinfo=timezone.utc)),
        ("", None),
        (None, None),
        ("not a date", None),
    ],
)
def test_parse_date_obs(value, expected):
    assert parse_date_obs(value) == expected


def test_parse_date_obs_never_raises():
    for bad in (object(), 123, True, [], {}, "   ", "''"):
        assert parse_date_obs(bad) is None


def _light_with_date(date_str):
    cards = ({"keyword": "DATE-OBS", "value": date_str},)
    return light(evidence=LightEvidence(original_cards=cards))


def test_light_acquisition_date_single_value():
    lt = _light_with_date("2024-09-21T03:04:05")
    assert light_acquisition_date(lt) == datetime(2024, 9, 21, 3, 4, 5, tzinfo=timezone.utc)


def test_light_acquisition_date_hierarch_insensitive():
    cards = ({"keyword": "HIERARCH DATE-OBS", "value": "2024-09-21T03:04:05"},)
    lt = light(evidence=LightEvidence(original_cards=cards))
    assert light_acquisition_date(lt) is not None


def test_light_acquisition_date_disagreeing_none():
    cards = (
        {"keyword": "DATE-OBS", "value": "2024-09-21T03:04:05"},
        {"keyword": "DATE-OBS", "value": "2024-09-22T03:04:05"},
    )
    lt = light(evidence=LightEvidence(original_cards=cards))
    assert light_acquisition_date(lt) is None


def test_light_acquisition_date_absent_none():
    assert light_acquisition_date(light()) is None


# ---------------------------------------------------------------------------
# Additive (dark / bias) ranking — rule A
# ---------------------------------------------------------------------------
def test_single_compatible_no_ranking():
    lt = light()
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", descriptor("dark", "included"))]), policy())
    assert r.outcome == OUTCOME_MATCHED
    assert r.selection[0].rule == RULE_SINGLE_CANDIDATE
    assert r.ranked_out == ()


def test_two_compatible_different_dates_most_recent():
    lt = light()
    older = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    newer = descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(
        lt, request(),
        pool(dark=[
            candidate("older", older, acquired_at="2024-01-01T00:00:00"),
            candidate("newer", newer, acquired_at="2024-06-01T00:00:00"),
        ]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert r.plan.masters["dark"].descriptor_id == newer.descriptor_id
    assert [rec.candidate_id for rec in r.ranked_out] == ["older"]
    assert all(rec.reason_code == RANKED_BELOW_WINNER for rec in r.ranked_out)


def test_known_date_before_unknown_date():
    lt = light()
    known = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    unknown = descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(
        lt, request(),
        pool(dark=[
            candidate("unknown", unknown, acquired_at=None),
            candidate("known", known, acquired_at="2020-01-01T00:00:00"),
        ]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert r.plan.masters["dark"].descriptor_id == known.descriptor_id
    assert [rec.candidate_id for rec in r.ranked_out] == ["unknown"]


def test_equal_dates_candidate_id_tiebreak():
    lt = light()
    a = candidate("a", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64), acquired_at="2024-01-01")
    b = candidate("b", descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64), acquired_at="2024-01-01")
    r = match_calibration(lt, request(), pool(dark=[a, b]), policy())
    assert r.outcome == OUTCOME_MATCHED
    assert r.selection[0].chosen_candidate_id == "a"
    assert [rec.candidate_id for rec in r.ranked_out] == ["b"]


def test_identical_full_key_ambiguous_tie():
    lt = light()
    a = candidate("same", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64), acquired_at="2024-01-01")
    b = candidate("same", descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64), acquired_at="2024-01-01")
    r = match_calibration(lt, request(), pool(dark=[a, b]), policy())
    assert r.outcome == OUTCOME_AMBIGUOUS


def test_newer_incompatible_older_compatible_wins():
    # Rule D: the date NEVER bypasses compatibility. A newer-but-incompatible
    # master is a compatibility rejection, not a ranking loss.
    lt = light()
    older = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    newer_bad = descriptor(
        "dark", "included", acquisition_obj=acquisition(gain=999),
        content_sha256="c" * 64, mask_identity="d" * 64,
    )
    r = match_calibration(
        lt, request(),
        pool(dark=[
            candidate("older", older, acquired_at="2024-01-01T00:00:00"),
            candidate("newer", newer_bad, acquired_at="2024-06-01T00:00:00"),
        ]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert r.plan.masters["dark"].descriptor_id == older.descriptor_id
    assert r.ranked_out == ()  # the newer was rejected, never ranked
    rejected = [rec for rec in r.rejected_candidates if rec.candidate_id == "newer"]
    assert rejected and any("GAIN_MISMATCH" in rec.reason_codes for rec in rejected)


def test_bias_ranked_by_additive_rule():
    lt = light()
    older = descriptor("bias", "not_applicable", exposure_s=0.001, content_sha256="a" * 64, mask_identity="b" * 64)
    newer = descriptor("bias", "not_applicable", exposure_s=0.001, content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(
        lt, request("bias_only"),
        pool(bias=[
            candidate("older", older, acquired_at="2024-01-01T00:00:00"),
            candidate("newer", newer, acquired_at="2024-06-01T00:00:00"),
        ]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert r.plan.masters["bias"].descriptor_id == newer.descriptor_id
    assert [rec.candidate_id for rec in r.ranked_out] == ["older"]


# ---------------------------------------------------------------------------
# Flat ranking — rule B
# ---------------------------------------------------------------------------
def _corrected_flat(cid, acquired_at, content_sha, mask_identity):
    pp = ProcessingProvenance(
        source="synthetic_fixture", additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    f = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp,
        content_sha256=content_sha, mask_identity=mask_identity,
    )
    return candidate(cid, f, acquired_at=acquired_at)


def _flat_match(light_date, flats):
    lt = _light_with_date(light_date)
    r = match_calibration(
        lt, request("control", "apply"),
        pool(flat=flats),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    return r


def test_flat_same_day_beats_recency():
    # owner scenario 1: light 21/09, flats 21/09 + 22/09 -> 21/09
    f1 = _corrected_flat("f1", "2024-09-21T10:00:00", "a" * 64, "b" * 64)
    f2 = _corrected_flat("f2", "2024-09-22T10:00:00", "c" * 64, "d" * 64)
    r = _flat_match("2024-09-21", [f1, f2])
    assert r.selection[0].chosen_candidate_id == "f1"
    assert r.selection[0].rule == RULE_FLAT


def test_flat_light_22_same_day_wins():
    # owner scenario 2: light 22/09, flats 21/09 + 22/09 -> 22/09
    f1 = _corrected_flat("f1", "2024-09-21T10:00:00", "a" * 64, "b" * 64)
    f2 = _corrected_flat("f2", "2024-09-22T10:00:00", "c" * 64, "d" * 64)
    r = _flat_match("2024-09-22", [f1, f2])
    assert r.selection[0].chosen_candidate_id == "f2"


def test_flat_no_same_day_not_a_failure():
    # owner scenario 3: light 22/09, flat 21/09 only -> 21/09 (still MATCHED)
    f1 = _corrected_flat("f1", "2024-09-21T10:00:00", "a" * 64, "b" * 64)
    r = _flat_match("2024-09-22", [f1])
    assert r.selection[0].chosen_candidate_id == "f1"


def test_flat_nearest_day_not_newest():
    # owner scenario 4: light 20/09, flats 21/09 + 22/09 -> 21/09
    f1 = _corrected_flat("f1", "2024-09-21T10:00:00", "a" * 64, "b" * 64)
    f2 = _corrected_flat("f2", "2024-09-22T10:00:00", "c" * 64, "d" * 64)
    r = _flat_match("2024-09-20", [f1, f2])
    assert r.selection[0].chosen_candidate_id == "f1"


def test_flat_same_day_later_timestamp_wins():
    # owner scenario 5a: light 21/09, two same-day flats -> later timestamp
    f1 = _corrected_flat("f1", "2024-09-21T10:00:00", "a" * 64, "b" * 64)
    f2 = _corrected_flat("f2", "2024-09-21T11:00:00", "c" * 64, "d" * 64)
    r = _flat_match("2024-09-21", [f1, f2])
    assert r.selection[0].chosen_candidate_id == "f2"


def test_flat_same_day_identical_timestamp_candidate_id_asc():
    # owner scenario 5b: identical timestamps -> candidate_id asc
    f1 = _corrected_flat("fa", "2024-09-21T10:00:00", "a" * 64, "b" * 64)
    f2 = _corrected_flat("fb", "2024-09-21T10:00:00", "c" * 64, "d" * 64)
    r = _flat_match("2024-09-21", [f1, f2])
    assert r.selection[0].chosen_candidate_id == "fa"


def test_flat_light_date_unknown_most_recent():
    # light date unknown + dated flats -> most recent flat (class 1, distance 0)
    f1 = _corrected_flat("f1", "2024-09-21T10:00:00", "a" * 64, "b" * 64)
    f2 = _corrected_flat("f2", "2024-09-22T10:00:00", "c" * 64, "d" * 64)
    lt = light()
    r = match_calibration(lt, request("control", "apply"), pool(flat=[f1, f2]), policy())
    assert r.outcome == OUTCOME_MATCHED
    assert r.selection[0].chosen_candidate_id == "f2"


def test_flat_date_unknown_after_all_dated():
    # flat date unknown + known light date -> class 2 (after every dated flat)
    f_dated = _corrected_flat("f1", "2024-09-20T10:00:00", "a" * 64, "b" * 64)
    f_unknown = _corrected_flat("f2", None, "c" * 64, "d" * 64)
    r = _flat_match("2024-09-21", [f_dated, f_unknown])
    assert r.selection[0].chosen_candidate_id == "f1"
    assert [rec.candidate_id for rec in r.ranked_out] == ["f2"]


def test_flat_equidistant_later_date_wins():
    # equidistant flats -> the later date (documented tie-break)
    f_early = _corrected_flat("f_early", "2024-09-21T12:00:00", "a" * 64, "b" * 64)
    f_late = _corrected_flat("f_late", "2024-09-23T12:00:00", "c" * 64, "d" * 64)
    r = _flat_match("2024-09-22", [f_early, f_late])
    assert r.selection[0].chosen_candidate_id == "f_late"


def test_flat_same_day_incompatible_other_date_wins():
    # A same-day flat that is INCOMPATIBLE must not win; a compatible flat from
    # another date wins (compatibility is checked before ranking).
    lt = _light_with_date("2024-09-21")
    bad = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
        filter="IRCUT", optical_train_id="SYNTH-TRAIN-1",  # filter mismatch
        processing=ProcessingProvenance(source="synthetic_fixture", additive_history_state="known", additive_correction_history=("flat_dark_subtracted",)),
        content_sha256="a" * 64, mask_identity="b" * 64,
    )
    good = _corrected_flat("good", "2024-09-22T10:00:00", "c" * 64, "d" * 64)
    r = match_calibration(
        lt, request("control", "apply"),
        pool(flat=[candidate("bad", bad, acquired_at="2024-09-21T10:00:00"), good]),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert r.selection[0].chosen_candidate_id == "good"
    assert r.ranked_out == ()


# ---------------------------------------------------------------------------
# flat_dark — additive rule (never the light's civil day)
# ---------------------------------------------------------------------------
def test_flat_dark_ranked_by_additive_rule():
    lt = light(optical=OpticalIdentity(filter="NONE", optical_train_id="SYNTH-TRAIN-1"))
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
    )
    fd_older = descriptor("flat_dark", "included", exposure_s=1.0, content_sha256="a" * 64, mask_identity="b" * 64)
    fd_newer = descriptor("flat_dark", "included", exposure_s=1.0, content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(
        lt, request("control", "apply"),
        pool(
            flat=[candidate("f1", flat)],
            flat_dark=[
                candidate("fd1", fd_older, acquired_at="2024-01-01T00:00:00"),
                candidate("fd2", fd_newer, acquired_at="2024-06-01T00:00:00"),
            ],
        ),
        policy(),
    )
    assert r.outcome == OUTCOME_MATCHED
    assert r.plan.masters["flat_dark"].descriptor_id == fd_newer.descriptor_id
    fd_sel = [s for s in r.selection if s.role == "flat_dark"]
    assert fd_sel and fd_sel[0].rule == RULE_ADDITIVE


# ---------------------------------------------------------------------------
# Manual precedence
# ---------------------------------------------------------------------------
def test_manual_pick_of_compatible_non_winner_still_works():
    lt = light()
    older = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    newer = descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64)
    r = match_calibration(
        lt, request(),
        pool(dark=[
            candidate("older", older, acquired_at="2024-01-01T00:00:00"),
            candidate("newer", newer, acquired_at="2024-06-01T00:00:00"),
        ]),
        policy(),
        manual_selection={"dark": "older"},
    )
    assert r.outcome == OUTCOME_MATCHED
    assert r.plan.masters["dark"].descriptor_id == older.descriptor_id
    assert r.selection[0].rule == RULE_MANUAL
    assert [rec.candidate_id for rec in r.ranked_out] == ["newer"]


def test_manual_incompatible_refused():
    lt = light()
    d1 = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    d2 = descriptor("dark", "included", acquisition_obj=acquisition(gain=999))
    r = match_calibration(
        lt, request(), pool(dark=[candidate("d1", d1), candidate("d2", d2)]), policy(),
        manual_selection={"dark": "d2"},
    )
    assert r.outcome == "NO_MATCH"
    assert "GAIN_MISMATCH" in r.reason_codes


def test_duplicate_candidate_id_manual_preserved_as_ambiguity():
    lt = light()
    a = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    b = descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64)
    cs = pool(dark=[candidate("same", a), candidate("same", b)])
    r = match_calibration(lt, request(), cs, policy(), manual_selection={"dark": "same"})
    assert r.outcome == OUTCOME_AMBIGUOUS


def test_select_role_candidates_policy_version():
    lt = light()
    sel = select_role_candidates(
        "dark", lt,
        [candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))],
    )
    assert sel.winner.policy_version == SELECTION_POLICY_VERSION
