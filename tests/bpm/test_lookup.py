"""LOOKUP matrix (mission §77): deterministic selection + benign fallbacks."""

from __future__ import annotations

import pytest

from zecalibrator.bpm.lookup import select_revision
from zecalibrator.bpm.revision import make_revision
from zecalibrator.bpm.store import (
    STATE_CORRUPTED,
    create_bad_pixel_database,
    resolve_bad_pixel_database,
)
from zecalibrator.bpm.vocabulary import (
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_SELECTED,
    REASON_BASE_CORRUPTED,
    REASON_NO_PROFILE,
    REASON_UNQUALIFIED_PROFILE,
    REVISION_STATE_CANDIDATE,
    REVISION_STATE_PROMOTED,
)

from conftest import make_identity, make_site


def test_no_profile_is_not_fatal():
    res = select_revision(make_identity(), ())
    assert res.outcome == OUTCOME_CALIBRATION_ONLY
    assert res.reason_code == REASON_NO_PROFILE
    assert res.revision is None


def test_one_exact_profile_selected():
    ident = make_identity()
    rev = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident, sites=[make_site(1, 1)], sequence=1)
    res = select_revision(ident, [rev])
    assert res.outcome == OUTCOME_SELECTED
    assert res.revision.revision_id == rev.revision_id


def test_multiple_revisions_select_latest_promoted():
    ident = make_identity()
    older = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident, sites=[make_site(1, 1)], sequence=1)
    newer = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident, sites=[make_site(2, 2)], sequence=2)
    res = select_revision(ident, [older, newer])
    assert res.outcome == OUTCOME_SELECTED
    assert res.revision.revision_id == newer.revision_id


def test_deterministic_selection_promoted_most_recent():
    ident = make_identity()
    a = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident, sites=[make_site(1, 1)], sequence=3)
    b = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident, sites=[make_site(2, 2)], sequence=1)
    c = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident, sites=[make_site(3, 3)], sequence=2)
    # input order shuffled; winner is the highest sequence
    res = select_revision(ident, [b, c, a])
    assert res.revision.revision_id == a.revision_id


def test_wrong_sensor_no_profile():
    ident_a = make_identity(detector_instance_id="DET-A")
    rev = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident_a, sites=[make_site(1, 1)], sequence=1)
    res = select_revision(make_identity(detector_instance_id="DET-B"), [rev])
    assert res.outcome == OUTCOME_CALIBRATION_ONLY
    assert res.reason_code == REASON_NO_PROFILE


def test_wrong_geometry_no_profile():
    ident_a = make_identity(shape=(1080, 1920))
    rev = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident_a, sites=[make_site(1, 1)], sequence=1)
    res = select_revision(make_identity(shape=(1080, 1930)), [rev])
    assert res.outcome == OUTCOME_CALIBRATION_ONLY


def test_wrong_binning_or_readout_no_profile():
    ident_a = make_identity(binning=(1, 1), readout_mode="MODE_A")
    rev = make_revision(state=REVISION_STATE_PROMOTED, sensor_identity=ident_a, sites=[make_site(1, 1)], sequence=1)
    assert select_revision(make_identity(binning=(2, 2), readout_mode="MODE_A"), [rev]).outcome == OUTCOME_CALIBRATION_ONLY
    assert select_revision(make_identity(binning=(1, 1), readout_mode="MODE_B"), [rev]).outcome == OUTCOME_CALIBRATION_ONLY


def test_unqualified_profile_is_calibration_only():
    ident = make_identity()
    candidate = make_revision(state=REVISION_STATE_CANDIDATE, sensor_identity=ident, sites=[make_site(1, 1)], sequence=1)
    res = select_revision(ident, [candidate])
    assert res.outcome == OUTCOME_CALIBRATION_ONLY
    assert res.reason_code == REASON_UNQUALIFIED_PROFILE


def test_corrupt_candidate_revision_refused_explicitly(tmp_path):
    import json

    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    ident = make_identity()
    # a CANDIDATE (not promoted) revision
    cand = make_revision(state=REVISION_STATE_CANDIDATE, sensor_identity=ident, sites=[make_site(1, 1)], sequence=0)
    db.add_revision(cand)
    # tamper it
    rev_path = root / "revisions" / f"{cand.revision_id}.json"
    obj = json.loads(rev_path.read_text(encoding="utf-8"))
    obj["sites"][0]["position"] = [123, 456]
    rev_path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    # Explicit refusal (typed + provenance), never silently skipped as "no profile".
    res = resolve_bad_pixel_database(root, ident)
    assert res.outcome == OUTCOME_BASE_ERROR
    assert res.reason_code == REASON_BASE_CORRUPTED
    assert res.provenance


def test_resolve_facade_missing_base_is_calibration_only(tmp_path):
    res = resolve_bad_pixel_database(tmp_path / "absent", make_identity())
    assert res.outcome == OUTCOME_CALIBRATION_ONLY
    assert res.reason_code == "NO_BASE"
