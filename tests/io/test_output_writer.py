"""Transactional standalone FITS output writer tests (Phase 6).

Covered: DQ + CALPROV + science payload layout, strict CALPROV round-trip,
deterministic collision-safe logical id, no-clobber publication (collision
refusal without overwrite), exclusive temporary-file cleanup on success and on
failure, atomic promotion (final file present, temp gone), and whole-file
SHA-256 after closure (never self-referenced).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
from astropy.io import fits

from zecalibrator.io.output_writer import (
    CALPROV_EXTNAME,
    DQ_EXTNAME,
    NoClobberViolation,
    build_calprov_record,
    output_filename,
    output_logical_id,
    parse_calprov,
    publish_no_clobber,
    serialize_calprov,
    write_standalone_output,
)

SHAPE = (4, 4)
INPUT_IDENTITY = {"kind": "fits", "path": "/light.fits", "hdu": 0}
PLAN_ID = "a" * 64


def _provenance(operation_id="op-1"):
    return {"schema_version": "zecalibrator.provenance.v1", "operation_id": operation_id}


def _write(tmp_path, **kwargs):
    data = np.full(SHAPE, 90.0, dtype=np.float32)
    mask = np.zeros(SHAPE, dtype=np.uint16)
    kwargs.setdefault("input_identity", INPUT_IDENTITY)
    kwargs.setdefault("plan_id", PLAN_ID)
    kwargs.setdefault("destination", str(tmp_path))
    kwargs.setdefault("status", "COMPLETED")
    return write_standalone_output(data, mask, _provenance(), **kwargs)


def _tmp_files(path):
    return [n for n in os.listdir(path) if n.endswith(".tmp")]


def test_write_standalone_output_layout(tmp_path):
    out = _write(tmp_path)
    assert out.committed is True
    assert os.path.exists(out.path)
    with fits.open(out.path, memmap=False) as hdul:
        names = [h.name for h in hdul]
        assert names[0] == "PRIMARY"
        assert names[1] == DQ_EXTNAME
        assert names[2] == CALPROV_EXTNAME
        assert hdul[0].data.dtype.itemsize == 4 and np.issubdtype(hdul[0].data.dtype, np.floating)
        assert hdul[0].header["BUNIT"] == "ADU"
        assert hdul[1].data.dtype.itemsize == 2 and np.issubdtype(hdul[1].data.dtype, np.unsignedinteger)
        assert hdul[2].data.dtype.itemsize == 1 and np.issubdtype(hdul[2].data.dtype, np.unsignedinteger)
        # no stale scaling/checksum cards on the science payload
        for key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
            assert key not in hdul[0].header


def test_calprov_roundtrip_is_strict(tmp_path):
    out = _write(tmp_path)
    with fits.open(out.path, memmap=False) as hdul:
        calprov = hdul[CALPROV_EXTNAME]
        payload = np.ascontiguousarray(calprov.data, dtype=np.uint8).tobytes()
    record = parse_calprov(payload)
    assert record["schema_version"] == "zecalibrator.provenance.v1"
    assert record["operation_id"] == "op-1"
    assert record["output"]["logical_id"] == out.logical_id
    assert record["output"]["science_digest"] == out.science_digest
    assert "whole_file_sha256" not in record["output"]  # no hash self-reference


def test_output_logical_id_deterministic_and_collision_safe():
    a = output_logical_id(INPUT_IDENTITY, PLAN_ID)
    b = output_logical_id(INPUT_IDENTITY, PLAN_ID)
    assert a == b
    assert len(a) == 64
    # different plan id -> different logical id
    assert a != output_logical_id(INPUT_IDENTITY, "b" * 64)
    # different input identity -> different logical id
    other = dict(INPUT_IDENTITY, path="/other.fits")
    assert a != output_logical_id(other, PLAN_ID)


def test_output_filename_is_deterministic():
    lid = output_logical_id(INPUT_IDENTITY, PLAN_ID)
    assert output_filename(lid) == f"zecalibrator_{lid}.fits"


def test_no_clobber_refuses_existing_output_without_overwrite(tmp_path):
    lid = output_logical_id(INPUT_IDENTITY, PLAN_ID)
    final_path = os.path.join(str(tmp_path), output_filename(lid))
    original = b"ORIGINAL-CONTENT"
    with open(final_path, "wb") as f:
        f.write(original)

    with pytest.raises(NoClobberViolation):
        _write(tmp_path)

    # the existing file is untouched (no silent overwrite)
    assert open(final_path, "rb").read() == original
    # no stray temp file left behind
    assert _tmp_files(str(tmp_path)) == []


def test_no_temp_files_left_after_success(tmp_path):
    _write(tmp_path)
    assert _tmp_files(str(tmp_path)) == []


def test_atomic_promotion_leaves_only_final_file(tmp_path):
    out = _write(tmp_path)
    files = sorted(os.listdir(str(tmp_path)))
    assert files == [os.path.basename(out.path)]
    # no temp prefix survives
    assert not any(".tmp" in f for f in files)


def test_whole_file_sha256_matches_actual_file(tmp_path):
    out = _write(tmp_path)
    actual = hashlib.sha256(open(out.path, "rb").read()).hexdigest()
    assert out.whole_file_sha256 == actual
    assert out.size_bytes == os.path.getsize(out.path)


def test_publish_no_clobber_removes_temp(tmp_path):
    src = tmp_path / "src.tmp"
    src.write_bytes(b"payload")
    dst = tmp_path / "final.fits"
    publish_no_clobber(str(src), str(dst))
    assert dst.read_bytes() == b"payload"
    assert not src.exists()


def test_serialize_calprov_rejects_nonfinite():
    with pytest.raises(ValueError):
        serialize_calprov({"bad": float("nan")})


def test_build_calprov_record_carries_output_identity_not_whole_hash():
    record = build_calprov_record(
        {"schema_version": "zecalibrator.provenance.v1"},
        logical_id="lid",
        science_digest="d" * 64,
        commit_status="COMPLETED",
    )
    assert record["output"]["logical_id"] == "lid"
    assert record["output"]["science_digest"] == "d" * 64
    assert "whole_file_sha256" not in record["output"]
