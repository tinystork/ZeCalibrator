"""BASE matrix (mission §76): store create/open/read-only/corruption/immutability/
determinism, and the identity-based fallback behaviour."""

from __future__ import annotations

import json
import os

import pytest

from zecalibrator.bpm.errors import BpmBaseCorrupted, BpmWriteError
from zecalibrator.bpm.store import (
    STATE_CORRUPTED,
    STATE_OPENED,
    create_bad_pixel_database,
    load_bad_pixel_database,
)
from zecalibrator.bpm.vocabulary import (
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_SELECTED,
    REASON_NO_BASE,
    REASON_NO_PROFILE,
    REVISION_STATE_PROMOTED,
)

from conftest import make_identity, make_rev, make_site


def _write_revision_bytes(root, revision_id, obj):
    rev_dir = root / "revisions"
    rev_dir.mkdir(parents=True, exist_ok=True)
    (rev_dir / f"{revision_id}.json").write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_create_empty_base(tmp_path):
    db = create_bad_pixel_database(tmp_path / "base")
    assert load_bad_pixel_database(db.root).state == STATE_OPENED
    assert db.revisions() == ()


def test_open_existing_base(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    ident = make_identity()
    db.add_revision(make_rev(ident, sites=[make_site(1, 1)]))
    # reload from disk (fresh object) -> one revision, same content
    load = load_bad_pixel_database(root)
    assert load.state == STATE_OPENED
    revs = load.database.revisions()
    assert len(revs) == 1
    assert revs[0].sensor_identity == ident


def test_missing_root_is_missing_not_error(tmp_path):
    root = tmp_path / "does-not-exist"
    load = load_bad_pixel_database(root)
    assert load.state == "MISSING"
    assert load.reason_code == REASON_NO_BASE
    from zecalibrator.bpm.store import resolve_bad_pixel_database

    res = resolve_bad_pixel_database(root, make_identity())
    assert res.outcome == OUTCOME_CALIBRATION_ONLY
    assert res.reason_code == REASON_NO_BASE


def test_existing_dir_without_manifest_is_missing(tmp_path):
    root = tmp_path / "empty-dir"
    root.mkdir()
    load = load_bad_pixel_database(root)
    assert load.state == "MISSING"


def test_open_does_not_write(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(make_identity(), sites=[make_site(1, 1)]))
    manifest = root / "manifest.json"
    before = manifest.read_bytes()
    # open + read only
    load = load_bad_pixel_database(root)
    load.database.revisions()
    assert manifest.read_bytes() == before


def test_read_only_base_write_refused(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:  # pragma: no cover - CI as root
        pytest.skip("root bypasses directory write permissions")
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    rev = make_rev(make_identity(), sites=[make_site(1, 1)])
    os.chmod(root, 0o555)
    os.chmod(root / "revisions", 0o555)
    try:
        # read works
        assert load_bad_pixel_database(root).state == STATE_OPENED
        # write is refused as a typed error, never silently ignored
        with pytest.raises(BpmWriteError):
            db.add_revision(make_rev(make_identity(), sites=[make_site(2, 2)]))
    finally:
        os.chmod(root / "revisions", 0o755)
        os.chmod(root, 0o755)


def test_corrupted_revision_is_typed_refusal(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    rev = make_rev(make_identity(), sites=[make_site(1, 1)])
    db.add_revision(rev)
    # tamper the revision content (change a site position)
    obj = json.loads((root / "revisions" / f"{rev.revision_id}.json").read_text(encoding="utf-8"))
    obj["sites"][0]["position"] = [999, 999]
    _write_revision_bytes(root, rev.revision_id, obj)
    load = load_bad_pixel_database(root)
    assert load.state == STATE_CORRUPTED
    assert load.provenance  # typed + provenance


def test_incompatible_sensor_is_no_profile_not_error(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(make_identity(detector_instance_id="DET-A"), sites=[make_site(1, 1)]))
    res = db.resolve(make_identity(detector_instance_id="DET-B"))
    assert res.outcome == OUTCOME_CALIBRATION_ONLY
    assert res.reason_code == REASON_NO_PROFILE


def test_incompatible_geometry_is_no_profile(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(make_identity(shape=(1080, 1920)), sites=[make_site(1, 1)]))
    res = db.resolve(make_identity(shape=(1080, 1930)))
    assert res.outcome == OUTCOME_CALIBRATION_ONLY
    assert res.reason_code == REASON_NO_PROFILE


def test_revision_is_immutable_in_store(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    rev = make_rev(make_identity(), sites=[make_site(1, 1)])
    db.add_revision(rev)
    # re-adding the same content-addressed revision is refused (never overwritten)
    with pytest.raises(BpmWriteError):
        db.add_revision(rev)
    # the on-disk bytes are unchanged
    rev_path = root / "revisions" / f"{rev.revision_id}.json"
    obj = json.loads(rev_path.read_text(encoding="utf-8"))
    assert obj["sites"][0]["position"] == [1, 1]


def test_same_evidence_deterministic_profile(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    ident = make_identity()
    r1 = make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[make_site(1, 1)], sequence=1)
    r2 = make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[make_site(1, 1)], sequence=1)
    assert r1.revision_id == r2.revision_id
    db.add_revision(r1)
    res = db.resolve(ident)
    assert res.outcome == OUTCOME_SELECTED
    assert res.revision.revision_id == r1.revision_id


def test_new_evidence_new_revision_old_remains(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    ident = make_identity()
    r1 = make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[make_site(1, 1)], sequence=1)
    r2 = make_rev(ident, state=REVISION_STATE_PROMOTED, sites=[make_site(1, 1), make_site(2, 2)], sequence=2)
    db.add_revision(r1)
    db.add_revision(r2)
    revs = db.revisions()
    assert len(revs) == 2
    assert {r.revision_id for r in revs} == {r1.revision_id, r2.revision_id}
