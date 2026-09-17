"""Library/index/source integration tests (mission §7.D).

Memory vs local SQLite equivalence, no master pixel load for matching, reopen,
unsupported-schema safety, transactional rollback, concurrent readers, changed
image/mask detection, in-memory source handles, and TemporaryDirectory isolation
(no global tmp/cache cleanup, no large residue).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time

import pytest

from _phase4_fixtures import candidate, descriptor, light, make_binding, policy, pool, request
from zecalibrator.application.library import (
    ArrayInputIdentity,
    FitsInputIdentity,
    LibraryHandle,
    LibrarySnapshot,
    light_constraints_from_sensor_metadata,
    resolve_calibration,
)
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.metadata import ImportDeclaration, build_sensor_metadata
from zecalibrator.core.plans import FitsFileLocator, MaskPayloadLocator
from zecalibrator.io.library_index import (
    LIBRARY_INDEX_SCHEMA,
    LibraryIndex,
    UnsupportedSchemaError,
)
from zecalibrator.io.master_source import (
    FilesystemSource,
    InMemorySource,
    SourceError,
)


# --- memory vs SQLite same decisions ---------------------------------------
def test_memory_and_sqlite_same_decision(tmp_path):
    dk = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    cands = {"dark": (candidate("d1", dk, locator_path="/d.fits"),)}
    mem_snap = LibrarySnapshot(revision="r1", schema_version=LIBRARY_INDEX_SCHEMA, candidates=cands)

    idx_path = str(tmp_path / "lib.sqlite")
    idx = LibraryIndex(idx_path).open(initialize=True)
    try:
        idx.publish_revision("r1", cands)
        loaded = idx.load_snapshot()
    finally:
        idx.close()

    assert loaded.revision == "r1"
    lt = light()
    mem = match_calibration(lt, request(), mem_snap.candidates, policy())
    disk = match_calibration(lt, request(), loaded.candidates, policy())
    assert mem.outcome == disk.outcome == "MATCHED"
    assert mem.plan.plan_id == disk.plan.plan_id
    assert mem.plan.masters["dark"].descriptor_id == disk.plan.masters["dark"].descriptor_id


def test_matching_uses_metadata_only_no_pixels():
    # Descriptors carry no pixel arrays; matching is pure metadata.
    dk = descriptor("dark", "included")
    snap = LibrarySnapshot(revision="r1", schema_version=LIBRARY_INDEX_SCHEMA, candidates={"dark": (candidate("d1", dk),)})
    r = resolve_calibration(light(), request(), snap, policy())
    assert r.outcome == "MATCHED"


# --- reopen / unsupported schema -------------------------------------------
def test_reopen_snapshot_roundtrip(tmp_path):
    dk = descriptor("dark", "included")
    idx_path = str(tmp_path / "lib.sqlite")
    idx = LibraryIndex(idx_path).open(initialize=True)
    try:
        idx.publish_revision("r1", {"dark": (candidate("d1", dk),)})
    finally:
        idx.close()

    idx2 = LibraryIndex(idx_path).open()
    try:
        loaded = idx2.load_snapshot()
    finally:
        idx2.close()
    assert loaded.revision == "r1"
    assert loaded.candidates["dark"][0].descriptor.descriptor_id == dk.descriptor_id


def test_unsupported_schema_refused(tmp_path):
    p = str(tmp_path / "bad.sqlite")
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', 'unknown.v9')")
    conn.commit()
    conn.close()
    idx = LibraryIndex(p)
    with pytest.raises(UnsupportedSchemaError):
        idx.open()


def test_no_destructive_recreate_on_existing_index(tmp_path):
    p = str(tmp_path / "keep.sqlite")
    idx = LibraryIndex(p).open(initialize=True)
    try:
        idx.publish_revision("r1", {"dark": ()})
    finally:
        idx.close()
    # Reopen (not initialize) must preserve the existing revision.
    idx2 = LibraryIndex(p).open()
    try:
        assert idx2.load_snapshot().revision == "r1"
    finally:
        idx2.close()


# --- transactional rollback ------------------------------------------------
def test_publish_failure_preserves_previous_revision(tmp_path):
    dk = descriptor("dark", "included")
    p = str(tmp_path / "lib.sqlite")
    idx = LibraryIndex(p).open(initialize=True)
    try:
        idx.publish_revision("r1", {"dark": (candidate("d1", dk),)})
        # Force a failure by passing a non-serializable candidate id (object).
        with pytest.raises(Exception):
            idx.publish_revision("r2", {"dark": (candidate(object(), dk),)})
        assert idx.list_revisions() == ("r1",)
        assert idx.load_snapshot().revision == "r1"
    finally:
        idx.close()


# --- concurrent readers -----------------------------------------------------
def test_concurrent_readers_isolated_from_writer(tmp_path):
    dk = descriptor("dark", "included")
    p = str(tmp_path / "lib.sqlite")
    idx = LibraryIndex(p).open(initialize=True)
    idx.publish_revision("r1", {"dark": (candidate("d1", dk),)})
    errors = []

    def reader():
        try:
            r = LibraryIndex(p).open()
            try:
                snap = r.load_snapshot()
                assert snap.revision in ("r1", "r2")
            finally:
                r.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    idx.publish_revision("r2", {"dark": (candidate("d1", dk),)})
    for t in threads:
        t.join()
    idx.close()
    assert not errors


# --- changed image / mask and during-read race ------------------------------
def test_changed_image_detected_on_revalidate():
    import hashlib

    image_bytes = b"ORIGINAL_BYTES"
    mask_bytes = b"mask-bytes"
    dk = descriptor(
        "dark", "included",
        content_sha256=hashlib.sha256(b"DIFFERENT_BYTES").hexdigest(),
        size_bytes=len(b"DIFFERENT_BYTES"),
        mask_identity=hashlib.sha256(b"other-mask").hexdigest(),
    )
    src = InMemorySource(images={"/d.fits": image_bytes}, masks={"/d.mask": mask_bytes})
    binding = make_binding(dk)
    snap = LibrarySnapshot(revision="r1", schema_version=LIBRARY_INDEX_SCHEMA, candidates={"dark": (candidate("d1", dk),)})
    handle = LibraryHandle(snap)
    result = handle.revalidate_binding(binding, src)
    assert result.status == "FAILED"
    assert any("content mismatch" in r for r in result.reasons)
    assert any("size mismatch" in r for r in result.reasons)
    assert any("mask mismatch" in r for r in result.reasons)


def test_revalidate_matches_actual_bytes():
    import hashlib

    data = b"ORIGINAL_BYTES"
    mask = b"mask-bytes"
    sha = hashlib.sha256(data).hexdigest()
    mask_sha = hashlib.sha256(mask).hexdigest()
    dk = descriptor("dark", "included", content_sha256=sha, size_bytes=len(data), mask_identity=mask_sha)
    src = InMemorySource(images={"/d.fits": data}, masks={"/d.mask": mask})
    binding = make_binding(dk)
    snap = LibrarySnapshot(revision="r1", schema_version=LIBRARY_INDEX_SCHEMA, candidates={"dark": (candidate("d1", dk),)})
    handle = LibraryHandle(snap)
    result = handle.revalidate_binding(binding, src)
    assert result.status == "VALID", result.reasons


def test_missing_image_source_diagnosed():
    src = InMemorySource(images={})
    with pytest.raises(SourceError):
        src.image_identity(FitsFileLocator(path="/nope.fits", hdu=0))


# --- filesystem source diagnostics -----------------------------------------
def test_filesystem_missing_file_diagnosed(tmp_path):
    fs = FilesystemSource()
    with pytest.raises(SourceError):
        fs.image_identity(FitsFileLocator(path=str(tmp_path / "missing.fits"), hdu=0))


def test_filesystem_identity_stable(tmp_path):
    p = tmp_path / "m.fits"
    p.write_bytes(b"0123456789")
    fs = FilesystemSource()
    ident = fs.image_identity(FitsFileLocator(path=str(p), hdu=0))
    assert ident.size_bytes == 10
    import hashlib

    assert ident.content_sha256 == hashlib.sha256(b"0123456789").hexdigest()


# --- input identity (array vs FITS) ----------------------------------------
def test_input_identity_array_has_no_fits_fields():
    arr = ArrayInputIdentity(caller_logical_id="arr-1", decoded_digest="d" * 64)
    assert arr.caller_logical_id == "arr-1"
    assert not hasattr(arr, "path")
    fits_id = FitsInputIdentity(path="/f.fits", hdu=0, whole_fits_sha256="a" * 64, decoded_digest="b" * 64)
    assert fits_id.path == "/f.fits"
    assert fits_id.hdu == 0


# --- sensor metadata adaptation --------------------------------------------
def test_sensor_metadata_adapts_to_light_constraints():
    decl = ImportDeclaration(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA",
        gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16",
        binning=(1, 1), sensor_dimensions=(4, 4), orientation="identity",
        cfa_phase="mono", roi_origin=(0, 0), exposure_s=10.0, temperature_c=20.0,
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
        saturation_limit_adu=60000, saturation_evidence="qualified",
    )
    md = build_sensor_metadata(shape=(4, 4), normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU")
    lc = light_constraints_from_sensor_metadata(md)
    assert lc.detector.detector_instance_id == "SYNTH-DET-0001"
    assert lc.acquisition.gain == 100
    assert lc.acquisition.bias_exposure_max_s == 0.01
    assert lc.optical.filter == "NONE"
    assert lc.raw_domain_declaration == "raw"


# --- no large residue on success -------------------------------------------
def test_no_large_residue_on_success(tmp_path):
    idx_path = str(tmp_path / "lib.sqlite")
    dk = descriptor("dark", "included")
    idx = LibraryIndex(idx_path).open(initialize=True)
    try:
        idx.publish_revision("r1", {"dark": (candidate("d1", dk),)})
        idx.load_snapshot()
    finally:
        idx.close()
    # The index file exists; no unexpected large artifacts beyond sqlite's own files.
    files = sorted(os.listdir(str(tmp_path)))
    assert "lib.sqlite" in files
    for f in files:
        assert f.startswith("lib.sqlite"), files
