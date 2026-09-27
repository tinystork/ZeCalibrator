"""Regression: managed cache A→B→A republish (ZC-MANAGED-CACHE-A-B-A-HOTFIX).

Real SQLite index + one persistent ``LibrarySpec``. Verifies the managed
fingerprint lifecycle fix: publishing A, then B, then requesting A again must
**replace/republish** the historical A revision (no ``revisions`` primary-key
collision), promote A to latest/current, keep exactly one A row, and carry a
meaningful non-None structured reason on a publication failure — all without
changing science, matching, schema, or the public API.

Public-API-only tests (no ``zecalibrator.core`` / ``zecalibrator.application``
module-path contract); the only ``io`` import is the ``LibraryIndex`` seam needed
to assert revision bookkeeping and ``PRAGMA integrity_check``.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.api.v1 import (
    ImportDeclaration,
    LibrarySpec,
    ManagedMasterRecord,
    open_library,
)

SHAPE = (4, 4)


def _declaration(**kw):
    base = dict(
        source="synthetic_fixture",
        identity="SYNTH-BASE-1",
        version="1.0",
        domain="raw",
        units="ADU",
        detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-CFA",
        gain=100.0,
        offset=50.0,
        readout_mode="MODE_A",
        adc_mode="MODE_16",
        binning=(1, 1),
        sensor_dimensions=SHAPE,
        orientation="identity",
        cfa_phase="mono",
        roi_origin=(0, 0),
        exposure_s=10.0,
        temperature_c=20.0,
        filter="NONE",
        optical_train_id="SYNTH-TRAIN-1",
        bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0,
        saturation_evidence="qualified",
    )
    base.update(kw)
    return ImportDeclaration(**base)


def _write_fits(path, value):
    hdu = fits.PrimaryHDU(np.full(SHAPE, value, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def _record(tmp_path, name="dark.fits", role="dark", value=10.0, decl=None):
    path = _write_fits(tmp_path / name, value)
    sha, size = v1.content_identity(path)
    return ManagedMasterRecord(
        role=role, content_sha256=sha, size_bytes=size,
        declaration=decl or _declaration(), hdu=0, bias_state="included",
        evidence={}, dq_state="no_source_dq", mask_path=None, last_seen_path=path,
    )


def _spec(tmp_path) -> LibrarySpec:
    return LibrarySpec(root=str(tmp_path / "cache"), index_path=str(tmp_path / "cache" / "managed.sqlite"))


def _opened_dark(spec):
    opened = open_library(spec)
    assert opened.operation_status == "OPENED", opened.details
    return opened.handle


def test_managed_cache_aba_republish(tmp_path):
    spec = _spec(tmp_path)
    rec_a = _record(tmp_path, "a.fits", value=10.0)
    rec_b = _record(tmp_path, "b.fits", value=20.0)
    fp_a = v1.managed_fingerprint([rec_a])
    fp_b = v1.managed_fingerprint([rec_b])
    assert fp_a != fp_b

    # Publish A.
    r = v1.build_managed_library(spec, [rec_a])
    assert r.status == "COMPLETED", r.details
    assert r.revision == fp_a
    with _opened_dark(spec) as h:
        assert h.snapshot.revision == fp_a
        assert list(h.snapshot.candidates.keys()) == ["dark"]
        assert h.snapshot.candidates["dark"][0].descriptor.content_sha256 == rec_a.content_sha256

    # Publish B (different content -> different fingerprint -> latest).
    r = v1.build_managed_library(spec, [rec_b])
    assert r.status == "COMPLETED", r.details
    assert r.revision == fp_b
    with _opened_dark(spec) as h:
        assert h.snapshot.revision == fp_b
        assert h.snapshot.candidates["dark"][0].descriptor.content_sha256 == rec_b.content_sha256

    # Request A again: must republish, not collide, and become latest again.
    r = v1.build_managed_library(spec, [rec_a])
    assert r.status == "COMPLETED", r.details
    assert r.revision == fp_a
    with _opened_dark(spec) as h:
        assert h.snapshot.revision == fp_a
        cand = h.snapshot.candidates["dark"][0]
        assert cand.descriptor.content_sha256 == rec_a.content_sha256
        assert cand.descriptor.content_sha256 != rec_b.content_sha256

    # Bookkeeping: exactly one A revision, A is latest, B still present, index intact.
    from zecalibrator.io.library_index import LibraryIndex

    idx = LibraryIndex(spec.resolved_index_path).open()
    try:
        revs = idx.list_revisions()
        assert revs.count(fp_a) == 1
        assert revs.count(fp_b) == 1
        assert revs[0] == fp_a
        assert idx._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        idx.close()


def test_managed_cache_aba_refresh_locator_on_new_path(tmp_path):
    # The fingerprint intentionally omits ``last_seen_path``; a republish must
    # rebuild from the CURRENT record and refresh the current locator path.
    spec = _spec(tmp_path)
    rec_a = _record(tmp_path, "a.fits", value=10.0)
    rec_b = _record(tmp_path, "b.fits", value=20.0)
    fp_a = v1.managed_fingerprint([rec_a])

    assert v1.build_managed_library(spec, [rec_a]).status == "COMPLETED"
    assert v1.build_managed_library(spec, [rec_b]).status == "COMPLETED"

    # Same content identity, now presented from a different current path.
    moved = _write_fits(tmp_path / "a_moved.fits", 10.0)
    rec_a2 = ManagedMasterRecord(
        role=rec_a.role, content_sha256=rec_a.content_sha256, size_bytes=rec_a.size_bytes,
        declaration=rec_a.declaration, hdu=rec_a.hdu, bias_state=rec_a.bias_state,
        evidence={}, dq_state=rec_a.dq_state, mask_path=rec_a.mask_path,
        last_seen_path=moved,
    )
    assert v1.managed_fingerprint([rec_a2]) == fp_a  # fingerprint omits last_seen_path

    r = v1.build_managed_library(spec, [rec_a2])
    assert r.status == "COMPLETED", r.details
    with _opened_dark(spec) as h:
        assert h.snapshot.revision == fp_a
        cand = h.snapshot.candidates["dark"][0]
        assert cand.locators[0].path == moved  # refreshed to the current path
        assert cand.descriptor.content_sha256 == rec_a.content_sha256


def test_managed_cache_aba_reuse_fast_path(tmp_path):
    # A→A (unchanged latest) stays a fast REUSED — no rebuild, no republish.
    spec = _spec(tmp_path)
    rec_a = _record(tmp_path, "a.fits", value=10.0)
    first = v1.build_managed_library(spec, [rec_a])
    assert first.status == "COMPLETED"
    second = v1.build_managed_library(spec, [rec_a])
    assert second.status == "REUSED"
    assert second.revision == first.revision
    assert second.candidate_count == 1


def test_managed_cache_forced_republish_failure_rolls_back(tmp_path, monkeypatch):
    # A republish failure must roll back AFTER transaction mutation (delete +
    # reinsert of the A revision) and leave the previous latest (B) usable.
    from zecalibrator.io.library_index import LibraryIndex

    spec = _spec(tmp_path)
    rec_a = _record(tmp_path, "a.fits", value=10.0)
    rec_b = _record(tmp_path, "b.fits", value=20.0)
    fp_a = v1.managed_fingerprint([rec_a])
    fp_b = v1.managed_fingerprint([rec_b])

    assert v1.build_managed_library(spec, [rec_a]).status == "COMPLETED"
    assert v1.build_managed_library(spec, [rec_b]).status == "COMPLETED"

    def _boom(self, conn, revision, entries):
        raise RuntimeError("forced republish failure")

    monkeypatch.setattr(LibraryIndex, "_insert_entries", _boom)

    r = v1.build_managed_library(spec, [rec_a])
    assert r.status == "FAILED"
    assert r.reason_code is not None
    assert "forced republish failure" in r.details

    # Previous latest (B) still opens and is usable.
    with _opened_dark(spec) as h:
        assert h.snapshot.revision == fp_b
        assert h.snapshot.candidates["dark"][0].descriptor.content_sha256 == rec_b.content_sha256

    # A is still present exactly once; the index is intact.
    idx = LibraryIndex(spec.resolved_index_path).open()
    try:
        revs = idx.list_revisions()
        assert revs.count(fp_a) == 1
        assert revs.count(fp_b) == 1
        assert revs[0] == fp_b
        assert idx._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        idx.close()
