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


def _open_deleted_tmp_fds():
    fd_dir = "/proc/self/fd"
    if not os.path.isdir(fd_dir):
        return []
    leaked = []
    for name in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue
        if ".fits.tmp" in target:
            leaked.append(target)
    return leaked


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="requires /proc/self/fd (Linux)")
def test_write_standalone_output_closes_temp_fd(tmp_path):
    # Catches the Windows-style fd leak on Linux too: a leaked mkstemp descriptor
    # keeps the unlinked `.fits.tmp` inode alive, visible via /proc/self/fd.
    _write(tmp_path)
    assert _open_deleted_tmp_fds() == []


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


# ---------------------------------------------------------------------------
# Windows ERROR_INVALID_FUNCTION (winerror==1 -> errno.EINVAL) publication
# ---------------------------------------------------------------------------
def test_windows_link_einval_falls_back_to_exclusive_copy(tmp_path, monkeypatch):
    """A1 — the real Windows witness: ``os.link`` raising ``OSError(EINVAL)``
    with ``winerror == 1`` ("Fonction incorrecte", filesystem without hard-link
    support) must use the exclusive-copy fallback, never fail the frame.

    Asserts *observable* publication behaviour (fallback invoked, final file
    present with the exact science payload, temp gone), not a private constant.
    """
    import errno

    import zecalibrator.io.output_writer as ow

    data = np.full(SHAPE, 42.0, dtype=np.float32)
    mask = np.zeros(SHAPE, dtype=np.uint16)

    def fake_link(path, link, **kwargs):
        exc = OSError(errno.EINVAL, "Fonction incorrecte")
        exc.winerror = 1
        raise exc

    monkeypatch.setattr(ow.os, "link", fake_link)
    fallback_calls = []
    real_copy = ow._copy_exclusive
    monkeypatch.setattr(
        ow, "_copy_exclusive",
        lambda tmp, final: fallback_calls.append((tmp, final)) or real_copy(tmp, final),
    )

    out = write_standalone_output(
        data, mask, _provenance(),
        input_identity=INPUT_IDENTITY, plan_id=PLAN_ID,
        destination=str(tmp_path), status="COMPLETED",
    )

    assert fallback_calls, "exclusive-copy fallback was not used for winerror==1"
    assert out.committed is True
    assert os.path.exists(out.path)
    with fits.open(out.path, memmap=False) as hdul:
        written = np.ascontiguousarray(hdul[0].data, dtype=np.float32)
        assert np.array_equal(written, data, equal_nan=True)
    assert _tmp_files(str(tmp_path)) == []


def test_publication_succeeds_when_hard_link_available(tmp_path):
    """A2 — normal publication (os.link available) still succeeds; final file
    is correct and the temp is gone."""
    out = _write(tmp_path)
    assert out.committed is True
    assert os.path.exists(out.path)
    with fits.open(out.path, memmap=False) as hdul:
        names = [h.name for h in hdul]
        assert names == ["PRIMARY", DQ_EXTNAME, CALPROV_EXTNAME]
    assert _tmp_files(str(tmp_path)) == []


def test_publish_no_clobber_never_overwrites_existing(tmp_path):
    """A3 — a pre-existing destination is never overwritten (no-clobber)."""
    src = tmp_path / "src.tmp"
    src.write_bytes(b"NEW")
    dst = tmp_path / "final.fits"
    dst.write_bytes(b"OLD")
    with pytest.raises(NoClobberViolation):
        publish_no_clobber(str(src), str(dst))
    assert dst.read_bytes() == b"OLD"
    assert not src.exists()


def test_publication_real_error_not_silently_fallback(tmp_path, monkeypatch):
    """A4 — a genuine non-classified link error (ENOSPC) must still raise and
    must NOT silently take the exclusive-copy fallback."""
    import errno

    import zecalibrator.io.output_writer as ow

    def fake_link(path, link, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(ow.os, "link", fake_link)
    fallback_calls = []
    real_copy = ow._copy_exclusive
    monkeypatch.setattr(
        ow, "_copy_exclusive",
        lambda tmp, final: fallback_calls.append(1) or real_copy(tmp, final),
    )

    src = tmp_path / "src.tmp"
    src.write_bytes(b"PAYLOAD")
    dst = tmp_path / "final.fits"
    with pytest.raises(OSError):
        publish_no_clobber(str(src), str(dst))
    assert fallback_calls == []  # no silent fallback
    assert not os.path.exists(str(dst))
    assert not src.exists()  # task-owned temp still cleaned


def test_no_orphaned_temp_or_fd_on_validation_failure(tmp_path, monkeypatch):
    """A5 — no orphaned ``.tmp`` and no leaked descriptor on failure."""
    import zecalibrator.io.output_writer as ow
    from zecalibrator.io.output_writer import OutputValidationError

    def boom(*args, **kwargs):
        raise OutputValidationError("synthetic validation failure")

    monkeypatch.setattr(ow, "_validate_output", boom)
    with pytest.raises(OutputValidationError):
        _write(tmp_path)
    assert _tmp_files(str(tmp_path)) == []
    assert _open_deleted_tmp_fds() == []
