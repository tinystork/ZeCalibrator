"""G2B acquisition-date plumbing + digest-isolation + backward-read tests.

Covers: ``acquired_at`` never enters any digest (plan_id / descriptor_id
isolation); library-index ``acquired_at`` round-trip and old-revision backward
read; plan serialization backward read (missing new keys); ``MasterImportSpec`` /
``ManagedMasterRecord`` ``acquired_at`` validation and round-trip; managed
fingerprint sensitivity to date; the DATE-OBS source-boundary reader.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from _phase4_fixtures import candidate, descriptor, light, policy, pool, request
from zecalibrator.core.descriptors import DescriptorSnapshot, ImportDeclaration
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.plans import (
    CalibrationPlan,
    MasterBinding,
    PolicyParameters,
    VersionSet,
)
from zecalibrator.api.v1.models import (
    ManagedMasterRecord,
    MasterImportSpec,
)
from zecalibrator.api.v1.managed import (
    build_master_import_spec,
    managed_fingerprint,
)
from zecalibrator.io.library_index import LIBRARY_INDEX_SCHEMA, LibraryIndex
from zecalibrator.io.master_source import read_date_obs


def _decl():
    return ImportDeclaration(source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0")


# ---------------------------------------------------------------------------
# Digest isolation
# ---------------------------------------------------------------------------
def test_plan_id_unchanged_by_selection_block():
    lt = light()
    dk = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    r = match_calibration(lt, request(), pool(dark=[dk]), policy())
    plan_with = r.plan
    plan_without = CalibrationPlan.build(
        request=plan_with.request,
        light_constraints=plan_with.light_constraints,
        masters=plan_with.masters,
        policy_parameters=plan_with.policy_parameters,
        versions=plan_with.versions,
    )
    assert plan_with.plan_id == plan_without.plan_id
    assert plan_with.selection != ()
    assert plan_without.selection == ()


def test_acquired_at_never_in_plan_digest():
    lt = light()
    d = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    c1 = candidate("d1", d, acquired_at="2024-01-01T00:00:00")
    c2 = candidate("d1", d, acquired_at="2024-06-01T00:00:00")
    r1 = match_calibration(lt, request(), pool(dark=[c1]), policy())
    r2 = match_calibration(lt, request(), pool(dark=[c2]), policy())
    assert r1.plan.plan_id == r2.plan.plan_id
    assert "acquired_at" not in r1.plan.plan_digest_dict()["masters"]["dark"]


def test_descriptor_id_unchanged_by_acquired_at():
    # acquired_at lives on Candidate/Binding, never the descriptor projection.
    d1 = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    c = candidate("d1", d1, acquired_at="2024-01-01T00:00:00")
    assert c.descriptor.descriptor_id == d1.descriptor_id
    assert "acquired_at" not in d1.projection_dict()


def test_binding_identity_dict_excludes_acquired_at():
    d = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    b = MasterBinding(
        descriptor_id=d.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(d),
        content_sha256=d.content_sha256,
        size_bytes=d.size_bytes,
        hdu=d.hdu,
        mask_identity=d.mask_identity,
        acquired_at="2024-01-01T00:00:00",
    )
    assert "acquired_at" not in b.identity_dict()
    assert b.acquired_at == "2024-01-01T00:00:00"


# ---------------------------------------------------------------------------
# Plan serialization round-trip + backward read
# ---------------------------------------------------------------------------
def test_plan_roundtrip_carries_selection_and_composition():
    lt = light()
    dk = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    r = match_calibration(lt, request(), pool(dark=[dk]), policy())
    plan = r.plan
    restored = CalibrationPlan.from_dict(json.loads(json.dumps(dict(plan.to_dict()), allow_nan=False)))
    assert restored.plan_id == plan.plan_id
    assert [s.chosen_candidate_id for s in restored.selection] == [s.chosen_candidate_id for s in plan.selection]
    assert restored.selection_policy_version == plan.selection_policy_version
    assert restored.composition.level == plan.composition.level


def test_old_plan_dict_loads_without_new_keys():
    lt = light()
    dk = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    r = match_calibration(lt, request(), pool(dark=[dk]), policy())
    d = dict(r.plan.to_dict())
    del d["selection"]
    del d["selection_policy_version"]
    del d["composition"]
    restored = CalibrationPlan.from_dict(d)
    assert restored.selection == ()
    assert restored.selection_policy_version == ""
    assert restored.composition is None
    assert restored.plan_id == r.plan.plan_id


def test_binding_roundtrip_acquired_at_and_backward_read():
    d = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    b = MasterBinding(
        descriptor_id=d.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(d),
        content_sha256=d.content_sha256,
        size_bytes=d.size_bytes,
        hdu=d.hdu,
        mask_identity=d.mask_identity,
        acquired_at="2024-01-01T00:00:00",
    )
    restored = MasterBinding.from_dict(dict(b.to_dict()))
    assert restored.acquired_at == "2024-01-01T00:00:00"
    # backward read: no acquired_at key -> None
    dd = dict(b.to_dict())
    del dd["acquired_at"]
    assert MasterBinding.from_dict(dd).acquired_at is None


# ---------------------------------------------------------------------------
# Library index acquired_at round-trip + backward read
# ---------------------------------------------------------------------------
def test_library_index_acquired_at_roundtrip(tmp_path):
    dk = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    c = candidate("d1", dk, acquired_at="2024-01-02T03:04:05")
    idx = LibraryIndex(str(tmp_path / "lib.sqlite")).open(initialize=True)
    try:
        idx.publish_revision("r1", {"dark": (c,)})
        loaded = idx.load_snapshot()
    finally:
        idx.close()
    assert loaded.candidates["dark"][0].acquired_at == "2024-01-02T03:04:05"


def test_library_index_old_revision_without_acquired_at_loads(tmp_path):
    # A revision written by the pre-G2B schema (no acquired_at column) still
    # loads with acquired_at=None.
    p = str(tmp_path / "old.sqlite")
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (LIBRARY_INDEX_SCHEMA,))
    conn.execute("CREATE TABLE revisions (revision TEXT PRIMARY KEY, ordinal INTEGER NOT NULL)")
    conn.execute(
        "CREATE TABLE entries (revision TEXT NOT NULL, role TEXT NOT NULL, candidate_id TEXT NOT NULL, "
        "snapshot_json TEXT NOT NULL, locators_json TEXT NOT NULL, mask_locator_json TEXT, "
        "PRIMARY KEY (revision, role, candidate_id))"
    )
    conn.execute("INSERT INTO revisions VALUES ('r1', 1)")
    dk = descriptor("dark", "included")
    snap = json.dumps(dict(DescriptorSnapshot(dk).to_dict()))
    loc = json.dumps([{"path": "/d.fits", "hdu": 0}])
    conn.execute("INSERT INTO entries VALUES ('r1', 'dark', 'd1', ?, ?, NULL)", (snap, loc))
    conn.commit()
    conn.close()

    idx = LibraryIndex(p).open()
    try:
        snap = idx.load_snapshot()
    finally:
        idx.close()
    c = snap.candidates["dark"][0]
    assert c.acquired_at is None
    assert c.descriptor.descriptor_id == dk.descriptor_id


# ---------------------------------------------------------------------------
# MasterImportSpec / ManagedMasterRecord acquired_at
# ---------------------------------------------------------------------------
def test_master_import_spec_acquired_at_validation():
    decl = _decl()
    # valid None and valid string
    MasterImportSpec(path="/f.fits", master_type="dark", declaration=decl, dq_state="no_source_dq", acquired_at=None)
    MasterImportSpec(path="/f.fits", master_type="dark", declaration=decl, dq_state="no_source_dq", acquired_at="2024-01-02")
    with pytest.raises(Exception):
        MasterImportSpec(path="/f.fits", master_type="dark", declaration=decl, dq_state="no_source_dq", acquired_at="")
    with pytest.raises(Exception):
        MasterImportSpec(path="/f.fits", master_type="dark", declaration=decl, dq_state="no_source_dq", acquired_at="   ")


def _managed_record(acquired_at, last_seen_path="/f.fits"):
    return ManagedMasterRecord(
        role="dark",
        content_sha256="a" * 64,
        size_bytes=16,
        declaration=_decl(),
        dq_state="no_source_dq",
        mask_path=None,
        last_seen_path=last_seen_path,
        acquired_at=acquired_at,
    )


def test_managed_record_acquired_at_roundtrip():
    rec = _managed_record("2024-01-02T03:04:05")
    d = rec.to_dict()
    assert d["acquired_at"] == "2024-01-02T03:04:05"
    assert ManagedMasterRecord.from_dict(d).acquired_at == "2024-01-02T03:04:05"


def test_managed_fingerprint_changes_on_date():
    rec1 = _managed_record("2024-01-01T00:00:00")
    rec2 = _managed_record("2024-06-01T00:00:00")
    assert managed_fingerprint([rec1]) != managed_fingerprint([rec2])


def test_build_master_import_spec_forwards_acquired_at():
    rec = _managed_record("2024-01-02T03:04:05")
    spec = build_master_import_spec(rec)
    assert spec.acquired_at == "2024-01-02T03:04:05"


# ---------------------------------------------------------------------------
# DATE-OBS source-boundary reader
# ---------------------------------------------------------------------------
def test_read_date_obs_single_value():
    assert read_date_obs([("DATE-OBS", "2024-01-02T03:04:05")]) == "2024-01-02T03:04:05"
    assert read_date_obs([("HIERARCH DATE-OBS", "2024-01-02")]) == "2024-01-02"
    assert read_date_obs([("DATE-OBS", "2024-01-02"), ("DATE-OBS", "2024-01-02")]) == "2024-01-02"
    assert read_date_obs([("DATE-OBS", "2024-01-02"), ("DATE-OBS", "2024-01-03")]) is None
    assert read_date_obs([("EXPTIME", 1.0)]) is None
    assert read_date_obs([]) is None


# ---------------------------------------------------------------------------
# Rework-2 D-1: additive library-index migration (backward-compatible write)
# ---------------------------------------------------------------------------
def _write_old_schema_index(path):
    """Create a pre-G2B index file (no ``acquired_at`` column)."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE revisions (revision TEXT PRIMARY KEY, ordinal INTEGER NOT NULL)")
    conn.execute(
        "CREATE TABLE entries (revision TEXT NOT NULL, role TEXT NOT NULL, candidate_id TEXT NOT NULL, "
        "snapshot_json TEXT NOT NULL, locators_json TEXT NOT NULL, mask_locator_json TEXT, "
        "PRIMARY KEY (revision, role, candidate_id))"
    )
    conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (LIBRARY_INDEX_SCHEMA,))
    conn.commit()
    conn.close()


def _entry_columns(path):
    conn = sqlite3.connect(path)
    try:
        return [r[1] for r in conn.execute("PRAGMA table_info(entries)").fetchall()]
    finally:
        conn.close()


def test_publish_into_pre_g2b_index_persists_acquired_at(tmp_path):
    p = str(tmp_path / "old.sqlite")
    _write_old_schema_index(p)
    assert "acquired_at" not in _entry_columns(p)

    dk = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    c = candidate("d1", dk, acquired_at="2024-01-02T03:04:05")
    idx = LibraryIndex(p).open(initialize=True)
    try:
        idx.publish_revision("r1", {"dark": (c,)})
        snap = idx.load_snapshot()
    finally:
        idx.close()
    assert snap.candidates["dark"][0].acquired_at == "2024-01-02T03:04:05"


def test_migration_is_additive_only(tmp_path):
    # Prove the pre-R0 reader invariant: the ALTER is additive-only — every old
    # column is preserved and only ``acquired_at`` is appended, so a pre-R0 reader
    # (which neither SELECTs nor INSERTs ``acquired_at``) still works.
    p = str(tmp_path / "old.sqlite")
    _write_old_schema_index(p)
    before = _entry_columns(p)
    idx = LibraryIndex(p).open(initialize=True)
    idx.close()
    after = _entry_columns(p)
    assert after == before + ["acquired_at"]


def test_initialize_false_never_migrates(tmp_path):
    p = str(tmp_path / "old.sqlite")
    _write_old_schema_index(p)
    idx = LibraryIndex(p).open(initialize=False)
    idx.close()
    assert "acquired_at" not in _entry_columns(p)


def test_fresh_index_unchanged(tmp_path):
    p = str(tmp_path / "fresh.sqlite")
    idx = LibraryIndex(p).open(initialize=True)
    idx.close()
    assert "acquired_at" in _entry_columns(p)  # fresh schema already has the column


def test_no_schema_version_change(tmp_path):
    p = str(tmp_path / "old.sqlite")
    _write_old_schema_index(p)
    idx = LibraryIndex(p).open(initialize=True)
    idx.close()
    conn = sqlite3.connect(p)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    finally:
        conn.close()
    assert row[0] == LIBRARY_INDEX_SCHEMA
    assert LIBRARY_INDEX_SCHEMA == "zecalibrator.library.v1"
