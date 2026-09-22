"""Transactional-overwrite gates for the standalone FITS output writer.

The overwrite path is the provenance-critical gate for the "Overwrite existing"
collision decision: a previous destination must remain byte-identical until a
fully validated replacement atomically replaces it, and the task-owned temp must
be cleaned in every case. These tests prove:

* a successful validated replacement replaces the destination atomically;
* a validation failure leaves the previous destination STRICTLY byte-identical;
* a cancellation before publication leaves the previous destination byte-identical;
* the task-owned temp is cleaned in all cases;
* ``overwrite_existing=False`` (the default) keeps today's no-clobber behaviour.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zecalibrator.application.cancellation import OperationCancelled
from zecalibrator.io.output_writer import (
    NoClobberViolation,
    OutputValidationError,
    output_filename,
    output_logical_id,
    write_standalone_output,
)

SHAPE = (4, 4)
INPUT_IDENTITY = {"kind": "fits", "path": "/light.fits", "hdu": 0}
PLAN_ID = "a" * 64


def _provenance(operation_id="op-1"):
    return {"schema_version": "zecalibrator.provenance.v1", "operation_id": operation_id}


def _write(tmp_path, value=90.0, **kwargs):
    data = np.full(SHAPE, value, dtype=np.float32)
    mask = np.zeros(SHAPE, dtype=np.uint16)
    kwargs.setdefault("input_identity", INPUT_IDENTITY)
    kwargs.setdefault("plan_id", PLAN_ID)
    kwargs.setdefault("destination", str(tmp_path))
    kwargs.setdefault("status", "COMPLETED")
    return write_standalone_output(data, mask, _provenance(), **kwargs)


def _final_path(tmp_path):
    lid = output_logical_id(INPUT_IDENTITY, PLAN_ID)
    return os.path.join(str(tmp_path), output_filename(lid))


def _tmp_files(path):
    return [n for n in os.listdir(path) if n.endswith(".tmp")]


def test_overwrite_success_replaces_destination_atomically(tmp_path):
    first = _write(tmp_path, value=90.0)
    original = open(first.path, "rb").read()
    assert original != b""  # original committed

    second = _write(tmp_path, value=123.0, overwrite_existing=True)
    assert second.path == first.path
    assert os.path.exists(second.path)
    # The replacement is the new payload, not the old one.
    assert open(second.path, "rb").read() != original
    # No temp survives.
    assert _tmp_files(str(tmp_path)) == []
    # Exactly one final file remains.
    assert sorted(os.listdir(str(tmp_path))) == [os.path.basename(second.path)]


def test_overwrite_validation_failure_preserves_previous_destination(tmp_path, monkeypatch):
    first = _write(tmp_path, value=90.0)
    original = open(first.path, "rb").read()

    import zecalibrator.io.output_writer as ow

    def boom(*args, **kwargs):
        raise OutputValidationError("synthetic validation failure")

    monkeypatch.setattr(ow, "_validate_output", boom)
    with pytest.raises(OutputValidationError):
        _write(tmp_path, value=123.0, overwrite_existing=True)

    # Previous destination is STRICTLY byte-identical; temp is cleaned.
    assert open(first.path, "rb").read() == original
    assert _tmp_files(str(tmp_path)) == []


def test_overwrite_cancellation_before_publication_preserves_previous_destination(tmp_path):
    first = _write(tmp_path, value=90.0)
    original = open(first.path, "rb").read()

    calls = []

    def check_cancelled():
        calls.append(1)
        # Cancellation fires after validation, before publication (3rd call).
        if len(calls) >= 3:
            raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        _write(tmp_path, value=123.0, overwrite_existing=True, check_cancelled=check_cancelled)

    assert open(first.path, "rb").read() == original
    assert _tmp_files(str(tmp_path)) == []


def test_overwrite_failure_on_directory_destination_is_truthful(tmp_path):
    # A destination that exists as a DIRECTORY cannot be atomically replaced:
    # overwrite must fail loudly (never silently), and clean its temp.
    _write(tmp_path, value=90.0)
    final = _final_path(tmp_path)
    os.unlink(final)
    os.mkdir(final)  # destination now a directory

    with pytest.raises(OSError):
        _write(tmp_path, value=123.0, overwrite_existing=True)
    assert _tmp_files(str(tmp_path)) == []


def test_overwrite_existing_false_keeps_no_clobber(tmp_path):
    first = _write(tmp_path, value=90.0)
    original = open(first.path, "rb").read()
    # Default path refuses to overwrite (regression guard).
    with pytest.raises(NoClobberViolation):
        _write(tmp_path, value=123.0)
    assert open(first.path, "rb").read() == original


def test_publish_replace_existing_primitive_replaces_and_cleans(tmp_path):
    from zecalibrator.io.output_writer import _publish_replace_existing

    src = tmp_path / "src.tmp"
    src.write_bytes(b"NEW")
    dst = tmp_path / "final.fits"
    dst.write_bytes(b"OLD")
    _publish_replace_existing(str(src), str(dst))
    assert dst.read_bytes() == b"NEW"
    assert not src.exists()


def test_publish_replace_existing_removes_temp_on_failure(tmp_path):
    from zecalibrator.io.output_writer import _publish_replace_existing

    src = tmp_path / "src.tmp"
    src.write_bytes(b"NEW")
    dst = tmp_path / "final.fits"
    dst.mkdir()  # a directory target: os.replace fails
    with pytest.raises(OSError):
        _publish_replace_existing(str(src), str(dst))
    assert not src.exists()  # task-owned temp removed on failure
    assert dst.is_dir()  # previous (directory) destination intact
