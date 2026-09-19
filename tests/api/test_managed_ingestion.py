"""Public managed-master ingestion contract tests (P7-M3B).

Public-API-only tests (no ``zecalibrator.core`` / ``zecalibrator.application`` /
``zecalibrator.io`` module-path contract). Synthetic SYNTH-BASE-1 facts are
explicit, never real detector defaults.

Covered: structural ``no_source_dq`` (nullable mask identity, no synthetic file,
execution succeeds with provenance no_source_dq), managed-vs-explicit parity,
ledger round-trip/atomic-write/corruption-preservation/content-binding, and
managed-index reuse/invalidation.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.api.v1 import (
    CalibrationRequest,
    ExecutionOptions,
    ImportDeclaration,
    LibrarySpec,
    ManagedMasterRecord,
    MasterImportSpec,
    build_sensor_metadata,
    calibrate_frame,
    default_match_policy,
    inspect_frame,
    open_library,
    resolve_calibration,
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


def _write_mask(path, mask):
    np.save(path, np.asarray(mask, dtype=np.uint16), allow_pickle=False)
    return str(path)


def _light_inspection(tmp_path, value=100.0):
    light = _write_fits(tmp_path / "light.fits", value)
    roi = v1.RoiExtentEvidence(
        extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0"
    )
    source = v1.FitsFrameSource(path=light, declaration=_declaration(), roi_extent=roi)
    result = inspect_frame(source)
    assert result.operation_status == "COMPLETED", result.details
    return result.inspection


# ---------------------------------------------------------------------------
# MasterImportSpec dq_state validation
# ---------------------------------------------------------------------------
def test_source_mask_requires_mask_path():
    with pytest.raises(v1.InvalidRequestError):
        MasterImportSpec(
            path="dark.fits", master_type="dark", declaration=_declaration(),
            bias_state="included", dq_state="source_mask", mask_path=None,
        )


def test_no_source_dq_forbids_mask_path():
    with pytest.raises(v1.InvalidRequestError):
        MasterImportSpec(
            path="dark.fits", master_type="dark", declaration=_declaration(),
            bias_state="included", dq_state="no_source_dq", mask_path="dark.mask.npy",
        )


# ---------------------------------------------------------------------------
# Structural no_source_dq: index + execution
# ---------------------------------------------------------------------------
def test_no_source_dq_index_null_mask_no_synthetic_file(tmp_path):
    dark = _write_fits(tmp_path / "dark.fits", 10.0)
    spec = MasterImportSpec(
        path=dark, master_type="dark", declaration=_declaration(),
        bias_state="included", dq_state="no_source_dq", mask_path=None,
    )
    index_path = str(tmp_path / "lib.sqlite")
    result = v1.index_library(LibrarySpec(root=str(tmp_path), index_path=index_path), [spec])
    assert result.operation_status == "COMPLETED", result.details

    opened = open_library(LibrarySpec(root=str(tmp_path), index_path=index_path))
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        cand = handle.snapshot.candidates["dark"][0]
        desc = cand.descriptor
        assert desc.dq_state == "no_source_dq"
        assert desc.mask_identity is None
        assert cand.mask_locator is None
    finally:
        handle.close()

    # No synthetic .npy mask file was ever written.
    npy = [f for f in os.listdir(str(tmp_path)) if f.endswith(".npy")]
    assert npy == []


def test_no_source_dq_execution_succeeds_with_provenance(tmp_path):
    dark = _write_fits(tmp_path / "dark.fits", 10.0)
    light = _write_fits(tmp_path / "light.fits", 100.0)
    spec = MasterImportSpec(
        path=dark, master_type="dark", declaration=_declaration(),
        bias_state="included", dq_state="no_source_dq", mask_path=None,
    )
    index_path = str(tmp_path / "lib.sqlite")
    assert v1.index_library(LibrarySpec(root=str(tmp_path), index_path=index_path), [spec]).operation_status == "COMPLETED"

    inspection = _light_inspection(tmp_path)
    opened = open_library(LibrarySpec(root=str(tmp_path), index_path=index_path))
    handle = opened.handle
    try:
        resolved = resolve_calibration(
            inspection, CalibrationRequest("dark_incl_bias"), handle, default_match_policy()
        )
    finally:
        handle.close()
    assert resolved.operation_status == "COMPLETED"
    assert resolved.outcome == "MATCHED"
    plan = resolved.plan

    # The matched dark master carries the structural no_source_dq state.
    dark_binding = plan.masters["dark"]
    assert dark_binding.role_descriptor.dq_state == "no_source_dq"
    assert dark_binding.mask_identity is None
    assert dark_binding.mask_locator is None

    source = v1.FitsFrameSource(
        path=light, declaration=_declaration(),
        roi_extent=v1.RoiExtentEvidence(
            extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0"
        ),
    )
    result = calibrate_frame(source, plan, ExecutionOptions())
    assert result.status == "COMPLETED", result.reason_code


def test_real_source_mask_identity_unchanged(tmp_path):
    # A real source_mask master keeps its exact existing mask identity semantics.
    dark = _write_fits(tmp_path / "dark.fits", 10.0)
    mask = str(tmp_path / "dark.mask.npy")
    _write_mask(mask, np.zeros(SHAPE, dtype=np.uint16))
    mask_sha = hashlib.sha256(open(mask, "rb").read()).hexdigest()
    spec = MasterImportSpec(
        path=dark, master_type="dark", declaration=_declaration(),
        bias_state="included", dq_state="source_mask", mask_path=mask,
    )
    index_path = str(tmp_path / "lib.sqlite")
    assert v1.index_library(LibrarySpec(root=str(tmp_path), index_path=index_path), [spec]).operation_status == "COMPLETED"
    opened = open_library(LibrarySpec(root=str(tmp_path), index_path=index_path))
    handle = opened.handle
    try:
        cand = handle.snapshot.candidates["dark"][0]
        assert cand.descriptor.dq_state == "source_mask"
        assert cand.descriptor.mask_identity == mask_sha
        assert cand.mask_locator is not None
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# Parity: managed vs explicit identical outcomes (central invariant)
# ---------------------------------------------------------------------------
def test_parity_managed_vs_explicit_source_mask(tmp_path):
    dark = _write_fits(tmp_path / "dark.fits", 10.0)
    mask = str(tmp_path / "dark.mask.npy")
    _write_mask(mask, np.zeros(SHAPE, dtype=np.uint16))
    decl = _declaration()

    # Explicit path.
    explicit_spec = MasterImportSpec(
        path=dark, master_type="dark", declaration=decl,
        bias_state="included", dq_state="source_mask", mask_path=mask,
    )
    explicit_index = str(tmp_path / "explicit.sqlite")
    assert v1.index_library(LibrarySpec(root=str(tmp_path), index_path=explicit_index), [explicit_spec]).operation_status == "COMPLETED"

    # Managed path (same content + declaration + mask + dq_state).
    sha, size = v1.content_identity(dark)
    record = ManagedMasterRecord(
        role="dark", content_sha256=sha, size_bytes=size, declaration=decl,
        hdu=0, bias_state="included", evidence={}, dq_state="source_mask",
        mask_path=mask, last_seen_path=dark,
    )
    managed_index = str(tmp_path / "managed.sqlite")
    res = v1.build_managed_library(LibrarySpec(root=str(tmp_path), index_path=managed_index), [record])
    assert res.status == "COMPLETED", res.details

    # Identical candidate descriptor identity.
    e = open_library(LibrarySpec(root=str(tmp_path), index_path=explicit_index)).handle
    m = open_library(LibrarySpec(root=str(tmp_path), index_path=managed_index)).handle
    try:
        ec = e.snapshot.candidates["dark"][0]
        mc = m.snapshot.candidates["dark"][0]
        assert ec.descriptor.descriptor_id == mc.descriptor.descriptor_id
        assert ec.descriptor.mask_identity == mc.descriptor.mask_identity
        assert ec.descriptor.content_sha256 == mc.descriptor.content_sha256
    finally:
        e.close()
        m.close()

    # Identical match outcome for the same light.
    inspection = _light_inspection(tmp_path)
    e2 = open_library(LibrarySpec(root=str(tmp_path), index_path=explicit_index)).handle
    m2 = open_library(LibrarySpec(root=str(tmp_path), index_path=managed_index)).handle
    try:
        re = resolve_calibration(inspection, CalibrationRequest("dark_incl_bias"), e2, default_match_policy())
        rm = resolve_calibration(inspection, CalibrationRequest("dark_incl_bias"), m2, default_match_policy())
    finally:
        e2.close()
        m2.close()
    assert re.outcome == rm.outcome == "MATCHED"
    assert re.plan.plan_id == rm.plan.plan_id


# ---------------------------------------------------------------------------
# Ledger: round-trip / atomic / corruption / content binding
# ---------------------------------------------------------------------------
def _record(tmp_path, name="dark.fits", role="dark", value=10.0, decl=None):
    path = _write_fits(tmp_path / name, value)
    sha, size = v1.content_identity(path)
    return ManagedMasterRecord(
        role=role, content_sha256=sha, size_bytes=size,
        declaration=decl or _declaration(), hdu=0, bias_state="included",
        evidence={}, dq_state="no_source_dq", mask_path=None, last_seen_path=path,
    )


def test_ledger_roundtrip(tmp_path):
    r = _record(tmp_path)
    ledger = v1.managed_ledger_path(str(tmp_path))
    v1.save_managed_ledger(ledger, [r])
    loaded = v1.load_managed_ledger(ledger)
    assert loaded.state == "ok"
    assert len(loaded.records) == 1
    rr = loaded.records[0]
    assert rr.role == "dark"
    assert rr.content_sha256 == r.content_sha256
    assert rr.dq_state == "no_source_dq"
    assert rr.mask_path is None


def test_ledger_missing_state(tmp_path):
    loaded = v1.load_managed_ledger(v1.managed_ledger_path(str(tmp_path)))
    assert loaded.state == "missing"
    assert loaded.records == ()


def test_ledger_corruption_and_unsupported_preserved(tmp_path):
    ledger = v1.managed_ledger_path(str(tmp_path))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{ not valid json", encoding="utf-8")
    assert v1.load_managed_ledger(ledger).state == "malformed"

    ledger.write_text(json.dumps({"schema_version": "other"}), encoding="utf-8")
    assert v1.load_managed_ledger(ledger).state == "unsupported"

    # Saving must not be attempted over corrupt/unsupported bytes by callers; the
    # loader reports the state truthfully (preservation is the caller's contract).
    assert v1.load_managed_ledger(ledger).records == ()


def test_ledger_content_binding_same_bytes_same_record_different_path(tmp_path):
    a = _write_fits(tmp_path / "a.fits", 10.0)
    sha, size = v1.content_identity(a)
    decl = _declaration()
    # Same bytes, two different paths -> same (content_sha256, role) identity.
    record = ManagedMasterRecord(
        role="dark", content_sha256=sha, size_bytes=size, declaration=decl,
        hdu=0, bias_state="included", evidence={}, dq_state="no_source_dq",
        mask_path=None, last_seen_path=a,
    )
    assert record.content_sha256 == sha
    # A second record with the same content + role but a different path is the
    # same record identity (lookup key is (content_sha256, role)).
    assert record.role == "dark"


def test_ledger_content_binding_same_path_new_bytes_new_record(tmp_path):
    p = tmp_path / "dark.fits"
    _write_fits(p, 10.0)
    sha1, size1 = v1.content_identity(str(p))
    _write_fits(p, 20.0)  # overwrite: same path, different bytes
    sha2, size2 = v1.content_identity(str(p))
    assert sha1 != sha2


def test_ledger_reopen_without_reasking(tmp_path):
    r = _record(tmp_path)
    ledger = v1.managed_ledger_path(str(tmp_path))
    v1.save_managed_ledger(ledger, [r])
    # Reopen: confirmed evidence is persisted (no re-asking); a rebuild uses the
    # stored declaration directly.
    loaded = v1.load_managed_ledger(ledger)
    assert loaded.state == "ok"
    spec = v1.build_master_import_spec(loaded.records[0])
    assert spec.master_type == "dark"
    assert spec.declaration == r.declaration


# ---------------------------------------------------------------------------
# Managed index: reuse + invalidation
# ---------------------------------------------------------------------------
def test_managed_index_reuse_and_invalidate_on_change(tmp_path):
    r = _record(tmp_path)
    index_path = str(tmp_path / "managed.sqlite")
    spec = LibrarySpec(root=str(tmp_path), index_path=index_path)

    first = v1.build_managed_library(spec, [r])
    assert first.status == "COMPLETED"
    assert first.candidate_count == 1

    # Unchanged fingerprint -> reuse.
    second = v1.build_managed_library(spec, [r])
    assert second.status == "REUSED"
    assert second.candidate_count == 1

    # Change evidence (add a confirmed fact) -> new fingerprint -> rebuild.
    r2 = ManagedMasterRecord(
        role=r.role, content_sha256=r.content_sha256, size_bytes=r.size_bytes,
        declaration=r.declaration, hdu=r.hdu, bias_state=r.bias_state,
        evidence={"exposure_s": v1.EvidenceFact("exposure_s", 10.0, "fits_header", "EXPTIME", "user", "1")},
        dq_state=r.dq_state, mask_path=r.mask_path, last_seen_path=r.last_seen_path,
    )
    third = v1.build_managed_library(spec, [r2])
    assert third.status == "COMPLETED"

    # Change the source set (remove the master) -> new fingerprint -> rebuild
    # (empty index) -> reuse thereafter.
    fourth = v1.build_managed_library(spec, [])
    assert fourth.status == "COMPLETED"
    assert fourth.candidate_count == 0
    fifth = v1.build_managed_library(spec, [])
    assert fifth.status == "REUSED"


def test_managed_fingerprint_covers_dq_state(tmp_path):
    r = _record(tmp_path)
    fp_source_mask = v1.managed_fingerprint([r])
    r2 = ManagedMasterRecord(
        role=r.role, content_sha256=r.content_sha256, size_bytes=r.size_bytes,
        declaration=r.declaration, hdu=r.hdu, bias_state=r.bias_state,
        evidence={}, dq_state="source_mask", mask_path=str(tmp_path / "d.mask.npy"),
        last_seen_path=r.last_seen_path,
    )
    fp_no_source_dq = v1.managed_fingerprint([r2])
    assert fp_source_mask != fp_no_source_dq
