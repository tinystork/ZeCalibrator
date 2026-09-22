"""Transactional standalone FITS output writer (PROVENANCE §3/§5).

Writes a calibrated science payload (float32 ``PrimaryHDU``, ``BITPIX=-32``,
``BUNIT=ADU``), a ``DQ`` uint16 image extension, and a ``CALPROV`` 1D-uint8
image extension carrying the UTF-8 bytes of the strict-JSON provenance record.
The whole file is written into an **exclusive temporary file in the destination
filesystem**, flushed/closed, reopened and validated, then published through a
platform-appropriate **no-clobber** publication primitive (``os.link`` on
POSIX/NTFS, with an exclusive-create copy fallback where hard links are
unavailable). The only authorized overwrite is the private
``_publish_replace_existing`` primitive (an explicitly authorised internal
destructive publication operation, used solely for a user-approved
"Overwrite existing" collision decision); ``os.replace`` is never used for a
data output through any consumer-facing path.

* No BSCALE/BZERO/BLANK/CHECKSUM/DATASUM are written.
* The final whole-file SHA-256 is computed **after** closure and is never part of
  the embedded record (no hash self-reference).
* Only task-owned incomplete temporary files are removed on failure/cancel;
  committed outputs and user masters are never touched.

This module imports ``astropy.io.fits`` only for writing/validation; it never
imports ``zecalibrator.application`` (dependency direction: application -> io).
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

OUTPUT_SUFFIX = ".fits"
DQ_EXTNAME = "DQ"
CALPROV_EXTNAME = "CALPROV"
OUTPUT_UNITS = "ADU"
OUTPUT_DTYPE = "float32"
MASK_DTYPE = "uint16"

# errno values for which ``os.link`` is not a viable no-clobber publication
# primitive (filesystem without hard-link support, permission, cross-device or
# link-count exhaustion). In those cases we fall back to an exclusive-create
# copy with fsync (documented fallback, never a silent overwrite).
_LINK_UNAVAILABLE = (
    errno.EPERM,
    errno.EACCES,
    errno.EXDEV,
    errno.EMLINK,
    errno.ENOSYS,
    getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    getattr(errno, "ENOTSUP", errno.ENOTSUP),
)


class NoClobberViolation(Exception):
    """The destination output already exists; no-clobber publication refused."""


class OutputValidationError(Exception):
    """The written output could not be re-read/validated as committed."""


@dataclass(frozen=True)
class StandaloneOutput:
    """A committed standalone FITS output locator + whole-file identity."""

    path: str
    logical_id: str
    science_digest: str
    whole_file_sha256: str
    size_bytes: int
    committed: bool = True


def output_logical_id(input_identity: Mapping[str, object], plan_id: str) -> str:
    """Collision-safe output logical id from input identity + plan id (§2.7).

    The id is derived from the *input* identity and plan id only, never from the
    output's own final bytes, so it is deterministic and collision-safe.
    """
    from zecalibrator.core.digests import canonical_json, sha256_hex

    payload = canonical_json({"input_identity": input_identity, "plan_id": plan_id})
    return sha256_hex(payload.encode("utf-8"))


def output_filename(logical_id: str) -> str:
    """Deterministic collision-safe output filename for a logical id."""
    return f"zecalibrator_{logical_id}{OUTPUT_SUFFIX}"


def serialize_calprov(record: Mapping[str, object]) -> bytes:
    """Serialize a CALPROV provenance record to strict UTF-8 JSON bytes (§3)."""
    text = json.dumps(
        dict(record), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return text.encode("utf-8")


def parse_calprov(data) -> Mapping[str, object]:
    """Decode a CALPROV payload with strict parsing (no bare NaN/Inf)."""

    def _reject_constant(name: str) -> None:
        raise ValueError(f"non-finite JSON constant {name!r} in CALPROV payload")

    if isinstance(data, bytes):
        text = data.decode("utf-8")
    elif isinstance(data, str):
        text = data
    else:
        raise ValueError("CALPROV payload must be bytes or str")
    value = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("CALPROV payload must decode to a JSON object")
    return value


def build_calprov_record(
    provenance: Mapping[str, object],
    *,
    logical_id: str,
    science_digest: str,
    commit_status: str,
) -> dict:
    """Build the full standalone provenance record (§2.7, §3).

    The record carries the output logical id and science digest (never the final
    whole-file SHA-256 and never a digest of the record itself).
    """
    record = dict(provenance)
    record["output"] = {
        "logical_id": logical_id,
        "science_digest": science_digest,
        "dtype": OUTPUT_DTYPE,
        "units": OUTPUT_UNITS,
        "commit_status": commit_status,
    }
    return record


def _remove_temp(path: Optional[str]) -> None:
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _write_fits_bytes(
    path: str, data: np.ndarray, mask: np.ndarray, calprov_bytes: bytes, header_fields: Mapping[str, str]
) -> None:
    from astropy.io import fits

    primary = fits.PrimaryHDU(np.ascontiguousarray(data, dtype=np.float32))
    primary.header["BUNIT"] = OUTPUT_UNITS
    for key, value in header_fields.items():
        primary.header[key] = value

    dq = fits.ImageHDU(np.ascontiguousarray(mask, dtype=np.uint16), name=DQ_EXTNAME)
    calprov_arr = np.frombuffer(calprov_bytes, dtype=np.uint8).copy()
    calprov = fits.ImageHDU(calprov_arr, name=CALPROV_EXTNAME)

    hdul = fits.HDUList([primary, dq, calprov])
    # checksum=False (default) means no CHECKSUM/DATASUM is written (§3).
    hdul.writeto(path, overwrite=True, checksum=False)


def _validate_output(path: str, data: np.ndarray, mask: np.ndarray, calprov_bytes: bytes) -> None:
    """Reopen the written file and verify science payload, DQ and CALPROV."""
    from astropy.io import fits

    try:
        with fits.open(path, memmap=False) as hdul:
            primary = hdul[0]
            if primary.data is None:
                raise OutputValidationError("output primary HDU has no data")
            written_data = np.ascontiguousarray(primary.data, dtype=np.float32)
            if written_data.shape != data.shape or not np.array_equal(
                written_data, data, equal_nan=True
            ):
                raise OutputValidationError("output science payload mismatch on re-read")

            dq_hdu = None
            calprov_hdu = None
            for hdu in hdul[1:]:
                name = hdu.header.get("EXTNAME")
                if name == DQ_EXTNAME:
                    dq_hdu = hdu
                elif name == CALPROV_EXTNAME:
                    calprov_hdu = hdu
            if dq_hdu is None or dq_hdu.data is None:
                raise OutputValidationError("output missing DQ extension")
            written_mask = np.ascontiguousarray(dq_hdu.data, dtype=np.uint16)
            if written_mask.shape != mask.shape or not np.array_equal(written_mask, mask):
                raise OutputValidationError("output DQ mask mismatch on re-read")
            if calprov_hdu is None or calprov_hdu.data is None:
                raise OutputValidationError("output missing CALPROV extension")
            written_calprov = np.ascontiguousarray(calprov_hdu.data, dtype=np.uint8).tobytes()
            if written_calprov != calprov_bytes:
                raise OutputValidationError("output CALPROV payload mismatch on re-read")
    except OutputValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrap any reopen failure as a validation failure
        raise OutputValidationError(f"cannot re-read output for validation: {exc}") from exc


def _copy_exclusive(tmp_path: str, final_path: str) -> None:
    """Exclusive-create copy fallback for filesystems without hard links."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(final_path, flags)
    except FileExistsError as exc:
        raise NoClobberViolation(f"output already exists: {final_path}") from exc
    try:
        with os.fdopen(fd, "wb") as dst, open(tmp_path, "rb") as src:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
    except Exception:
        _remove_temp(final_path)
        raise


def publish_no_clobber(tmp_path: str, final_path: str) -> None:
    """Publish ``tmp_path`` to ``final_path`` without ever overwriting an existing
    file. ``os.link`` gives atomic no-clobber semantics on POSIX and NTFS; the
    exclusive-create copy fallback is a documented no-overwrite alternative for
    filesystems that cannot hard link. The temporary file is always removed.
    """
    try:
        try:
            os.link(tmp_path, final_path)
        except FileExistsError as exc:
            raise NoClobberViolation(f"output already exists: {final_path}") from exc
        except OSError as exc:
            if exc.errno in _LINK_UNAVAILABLE:
                _copy_exclusive(tmp_path, final_path)
            else:
                raise
    finally:
        _remove_temp(tmp_path)


def _publish_replace_existing(tmp_path: str, final_path: str) -> None:
    """Atomically replace ``final_path`` with an ALREADY-VALIDATED temp file.

    PRIVATE destructive publication primitive — an explicitly authorised
    internal operation, NOT a general consumer-facing output policy. The caller
    must have already flushed/closed and re-validated the temp (science + DQ +
    CALPROV payloads) before calling this; it performs no validation itself.

    ``os.replace`` atomically swaps the temp over ``final_path``. The temp is
    created in the destination directory, so it is on the same filesystem (no
    cross-device rename). On failure the task-owned temp is removed and
    ``final_path`` is left intact; there is no missing/partial publication
    window.
    """
    try:
        os.replace(tmp_path, final_path)
    except Exception:
        _remove_temp(tmp_path)
        raise


def write_standalone_output(
    data: np.ndarray,
    mask: np.ndarray,
    provenance: Mapping[str, object],
    *,
    input_identity: Mapping[str, object],
    plan_id: str,
    destination: str,
    status: str,
    header_fields: Optional[Mapping[str, str]] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
    overwrite_existing: bool = False,
) -> StandaloneOutput:
    """Write one standalone FITS output transactionally (§5).

    Exclusive temp file -> flush/close -> validate -> publish. Publication is
    no-clobber by default (``overwrite_existing=False``); when
    ``overwrite_existing=True`` the already-validated temp atomically replaces
    the destination via the private ``_publish_replace_existing`` primitive.
    The final whole-file SHA-256 is computed after closure and returned, never
    embedded. On failure/cancellation only the task-owned temp file is removed
    and any previous destination remains intact until publication succeeds.
    """
    if check_cancelled is not None:
        check_cancelled()

    destination = os.fspath(destination)
    if not os.path.isdir(destination):
        raise ValueError(f"destination is not a directory: {destination}")

    from zecalibrator.core.digests import science_digest, sha256_hex

    data32 = np.ascontiguousarray(data, dtype=np.float32)
    mask16 = np.ascontiguousarray(mask, dtype=np.uint16)
    science = science_digest(data32, mask16)

    logical_id = output_logical_id(input_identity, plan_id)
    filename = output_filename(logical_id)
    final_path = os.path.join(destination, filename)

    record = build_calprov_record(
        provenance, logical_id=logical_id, science_digest=science, commit_status=status
    )
    calprov_bytes = serialize_calprov(record)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".zecalibrator-{logical_id[:16]}-", suffix=".fits.tmp", dir=destination
    )
    # Close the mkstemp descriptor immediately: the empty file is rewritten by
    # path (astropy opens its own handle), and the descriptor must not remain
    # open across publication/removal — Windows locks open files, so a leaked fd
    # would keep the `.tmp` alive (frozen PROVENANCE §5 "close FITS handles
    # before rename on Windows").
    os.close(fd)
    try:
        # mkstemp created the empty file; astropy rewrites it in place.
        _write_fits_bytes(tmp_path, data32, mask16, calprov_bytes, header_fields or {})

        if check_cancelled is not None:
            check_cancelled()

        _validate_output(tmp_path, data32, mask16, calprov_bytes)

        if check_cancelled is not None:
            check_cancelled()

        if overwrite_existing:
            _publish_replace_existing(tmp_path, final_path)
        else:
            publish_no_clobber(tmp_path, final_path)
    except Exception:
        _remove_temp(tmp_path)
        raise

    with open(final_path, "rb") as f:
        whole_file_sha256 = sha256_hex(f.read())
    size_bytes = os.path.getsize(final_path)

    return StandaloneOutput(
        path=final_path,
        logical_id=logical_id,
        science_digest=science,
        whole_file_sha256=whole_file_sha256,
        size_bytes=size_bytes,
        committed=True,
    )


__all__ = [
    "CALPROV_EXTNAME",
    "DQ_EXTNAME",
    "MASK_DTYPE",
    "NoClobberViolation",
    "OUTPUT_DTYPE",
    "OUTPUT_SUFFIX",
    "OUTPUT_UNITS",
    "OutputValidationError",
    "StandaloneOutput",
    "build_calprov_record",
    "output_filename",
    "output_logical_id",
    "parse_calprov",
    "publish_no_clobber",
    "serialize_calprov",
    "write_standalone_output",
]
