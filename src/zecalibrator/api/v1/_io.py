"""Private facade helpers: verified byte reads, mask payload, array validation.

Not part of the public ``__all__`` contract. These bind hashed bytes to decoded
bytes (single read + decode-from-bytes), consume external DQ mask payloads,
apply the frozen G3 precision envelope to the array-input path, and read flat
master domains (dimensionless normalized / corrected ADU).
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile

import numpy as np

from zecalibrator.core.errors import InvalidRequestError, PrecisionRefusalError
from zecalibrator.core.precision import (
    integer_exceeds_float32_exact_range,
    measure_float32_roundoff,
)


def sha256_bytes(data: bytes, *, cancel=None) -> str:
    if cancel is not None:
        cancel.raise_if_cancelled()
    return hashlib.sha256(data).hexdigest()


def read_bytes(path, *, cancel=None) -> bytes:
    if cancel is not None:
        cancel.raise_if_cancelled()
    with open(path, "rb") as f:
        data = f.read()
    if cancel is not None:
        cancel.raise_if_cancelled()
    return data


def decode_fits_from_bytes(data: bytes, hdu, declaration, *, cancel=None, progress=None, admission="light", role=None, flat_form=None):
    """Decode FITS from the exact ``data`` bytes supplied (single-read binding).

    The verified bytes are written to a temporary file and decoded from it, so
    the decoded result is always the bytes that were hashed; the temporary file
    is removed in ``finally`` (bounded, no persistent cache). ``admission`` /
    ``role`` / ``flat_form`` forward the additive master admission context to
    :func:`zecalibrator.io.raw_decoder.decode_fits` (light default unchanged).
    """
    from zecalibrator.io.raw_decoder import decode_fits

    if cancel is not None:
        cancel.raise_if_cancelled()
    fd, tmp = tempfile.mkstemp(suffix=".fits")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return decode_fits(
            tmp, hdu=hdu, declaration=declaration, cancel=cancel, progress=progress,
            admission=admission, role=role, flat_form=flat_form,
        )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def load_mask_payload(data: bytes, shape, *, expected_sha: str = None):
    """Decode a documented uint16 DQ mask payload (``.npy``, no pickled objects).

    Enforces the documented dtype/range **before** any narrowing: only a 2D
    ``uint16`` array with no reserved bits (5..15) and the exact declared shape
    is accepted; wider/narrower dtypes and malformed payloads (``EOFError`` /
    ``ValueError`` / ``OSError``) become a structured ``ValueError`` refusal.
    """
    if expected_sha is not None and sha256_bytes(data) != expected_sha:
        raise ValueError("mask content mismatch")
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    except (EOFError, ValueError, OSError) as exc:
        raise ValueError(f"malformed mask payload: {exc}") from exc
    if getattr(arr, "dtype", None) != np.uint16:
        raise ValueError(f"mask payload dtype must be uint16, got {getattr(arr, 'dtype', None)!r}")
    if arr.ndim != 2 or arr.shape != tuple(shape):
        raise ValueError(f"mask shape {tuple(arr.shape)} vs {tuple(shape)}")
    mask = np.ascontiguousarray(arr.astype(np.uint16, copy=True))
    from zecalibrator.core.dq import validate_mask  # reserved bits (5..15)

    validate_mask(mask)
    return mask


def read_flat_array_bytes(data: bytes, hdu, *, cancel=None):
    """Read a flat master's data plane (BSCALE/BZERO applied) as float32 + mask.

    This is the flat-domain reader for ``normalized_response`` (dimensionless)
    and ``corrected_unnormalized`` (ADU) masters; it does not impose the
    raw-light ADU/declaration domain checks that ``decode_fits`` does.
    """
    from zecalibrator.core.dq import INPUT_INVALID

    if cancel is not None:
        cancel.raise_if_cancelled()
    fd, tmp = tempfile.mkstemp(suffix=".fits")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        from astropy.io import fits

        with fits.open(tmp, do_not_scale_image_data=True, memmap=False) as hdul:
            hdu_obj = hdul[hdu]
            arr = np.asarray(hdu_obj.data)
            if arr.ndim != 2:
                raise ValueError("flat master must be a 2D plane")
            bscale = float(hdu_obj.header.get("BSCALE", 1.0))
            bzero = float(hdu_obj.header.get("BZERO", 0.0))
            f64 = bscale * arr.astype(np.float64) + bzero
            nonfinite = ~np.isfinite(f64)
            mask = np.zeros(arr.shape, dtype=np.uint16)
            mask[nonfinite] |= INPUT_INVALID
            data32 = np.ascontiguousarray(f64.astype(np.float32))
            data32[nonfinite] = np.nan
            if cancel is not None:
                cancel.raise_if_cancelled()
            return data32, mask
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def validate_array_precision(arr) -> None:
    """Apply the frozen G3 precision envelope to caller array input (F4).

    Rejects 64-bit integers beyond float64's exact range, magnitudes beyond
    float32 range and integer ADU beyond the float32-exact bound 2**24, before
    any destructive float64/float32 cast. Never relaxed.
    """
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.integer) and a.dtype.itemsize == 8:
        if np.issubdtype(a.dtype, np.signedinteger):
            mag = np.abs(a.astype(np.int64))
        else:
            mag = a.astype(np.uint64)
        if np.any(mag > np.uint64(2 ** 53)):
            raise PrecisionRefusalError(
                "stored 64-bit integer exceeds float64 exact-integer bound 2**53"
            )

    f64 = a.astype(np.float64)
    finite = np.isfinite(f64)
    if finite.any():
        vals = f64[finite]
        if np.any(np.abs(vals) > float(np.finfo(np.float32).max)):
            raise PrecisionRefusalError("decoded magnitude exceeds float32 range (extreme scale)")
        if np.any(
            np.fromiter(
                (integer_exceeds_float32_exact_range(v) for v in vals),
                dtype=bool,
                count=int(vals.size),
            )
        ):
            raise PrecisionRefusalError(
                "decoded integer ADU exceeds float32-exact bound 2**24"
            )


def verify_identity_digest(identity, data, mask) -> str:
    """Compute the canonical decoded-data digest and verify a declared digest.

    A non-empty declared ``decoded_digest`` must equal the actual digest of the
    decoded bytes; mismatch refuses. Returns the actual digest.
    """
    from zecalibrator.core.digests import science_digest

    actual = science_digest(data, mask)
    declared = getattr(identity, "decoded_digest", "")
    if declared and declared != actual:
        raise ValueError("identity decoded_digest does not match actual decoded bytes")
    return actual


def apply_roi_extent(metadata, evidence, actual_shape):
    """Apply explicit evidence-backed ROI extent to decoded metadata.

    Validates the declared extent against the actual decoded shape and any known
    metadata extent; returns a new ``SensorMetadata`` with the extent set. No
    blanket ``None -> shape`` derivation and no override of contrary facts.
    """
    if evidence is None:
        return metadata
    from zecalibrator.core.geometry import Geometry
    from zecalibrator.core.metadata import SensorMetadata

    if tuple(evidence.extent) != tuple(actual_shape):
        raise ValueError("ROI extent evidence does not match actual decoded shape")
    if (
        metadata.geometry.roi_extent is not None
        and tuple(metadata.geometry.roi_extent) != tuple(evidence.extent)
    ):
        raise ValueError("ROI extent evidence conflicts with known metadata extent")

    geo = Geometry(
        shape=metadata.geometry.shape,
        sensor_dimensions=metadata.geometry.sensor_dimensions,
        binning=metadata.geometry.binning,
        roi_origin=metadata.geometry.roi_origin,
        roi_extent=tuple(evidence.extent),
        orientation=metadata.geometry.orientation,
        cfa_phase=metadata.geometry.cfa_phase,
    )
    return SensorMetadata(
        original_cards=metadata.original_cards,
        normalized=metadata.normalized,
        conflicts=metadata.conflicts,
        geometry=geo,
        raw_domain_declaration=metadata.raw_domain_declaration,
        units=metadata.units,
        declaration=metadata.declaration,
        exposure_s=metadata.exposure_s,
        temperature_c=metadata.temperature_c,
        gain=metadata.gain,
        offset=metadata.offset,
        readout_mode=metadata.readout_mode,
        adc_mode=metadata.adc_mode,
        filter=metadata.filter,
        detector_model=metadata.detector_model,
        detector_instance_id=metadata.detector_instance_id,
        optical_train_id=metadata.optical_train_id,
        saturation_limit_adu=metadata.saturation_limit_adu,
        saturation_evidence=metadata.saturation_evidence,
        warnings=metadata.warnings,
    )


def normalize_progress(progress, operation_id):
    """Return a ``ProgressObserver`` for the supplied progress (observer/callable).

    A bare callable is wrapped; an invalid type raises ``InvalidRequestError``.
    """
    from zecalibrator.application.cancellation import ProgressObserver

    if progress is None:
        return ProgressObserver(operation_id=operation_id)
    if isinstance(progress, ProgressObserver):
        return progress
    if callable(progress):
        return ProgressObserver(progress, operation_id=operation_id)
    raise InvalidRequestError("progress must be a ProgressObserver or callable")


def emit_progress(progress, operation_id, phase, completed, total=None):
    """Emit one bounded progress event through a ProgressObserver or callable.

    Observer failures are isolated and never change the operation.
    """
    if progress is None:
        return
    from zecalibrator.application.cancellation import ProgressEvent

    report = getattr(progress, "report", None)
    if report is None and callable(progress):
        report = progress
    if report is None:
        return
    try:
        report(
            ProgressEvent(
                operation_id=operation_id,
                phase=phase,
                completed=completed,
                total=total,
                unit="steps",
            )
        )
    except Exception:
        return


def array_source_to_decoded(source, *, cancel=None):
    """Validate and convert an ``ArrayFrameSource`` into an owned ``DecodedFrame``.

    Validates shape vs declared geometry and precision, marks non-finite values
    invalid (never silently dropped), and never mutates the caller's arrays.
    """
    from zecalibrator.core.dq import INPUT_INVALID, validate_mask
    from zecalibrator.io.raw_decoder import DecodedFrame

    if cancel is not None:
        cancel.raise_if_cancelled()
    data = np.asarray(source.data)
    if data.ndim != 2:
        raise ValueError("ArrayFrameSource.data must be a 2D plane")
    metadata = source.metadata
    if tuple(metadata.geometry.shape) != tuple(data.shape):
        raise ValueError(
            f"ArrayFrameSource data shape {tuple(data.shape)} does not match metadata geometry {tuple(metadata.geometry.shape)}"
        )

    validate_array_precision(data)

    if source.invalid_mask is None:
        mask = np.zeros(data.shape, dtype=np.uint16)
    else:
        mask = validate_mask(source.invalid_mask)
        if mask.shape != data.shape:
            raise ValueError(
                f"ArrayFrameSource invalid_mask shape {mask.shape} vs data {data.shape}"
            )
        mask = mask.astype(np.uint16).copy()

    if cancel is not None:
        cancel.raise_if_cancelled()
    f64 = data.astype(np.float64)
    nonfinite = ~np.isfinite(f64)
    mask[nonfinite] |= INPUT_INVALID
    data32 = np.ascontiguousarray(f64.astype(np.float32))
    data32[nonfinite] = np.nan

    return DecodedFrame(
        data=data32,
        mask=np.ascontiguousarray(mask),
        metadata=metadata,
        stored_dtype=str(data.dtype),
        bscale=1.0,
        bzero=0.0,
        blank=None,
        hdu=None,
        precision=measure_float32_roundoff(f64),
    )
