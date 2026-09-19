"""Strict raw FITS decoder (SCIENCE §2, ASTRA §4.1).

Rules enforced:

* Preserve the original header/card sequence (including duplicates) **before**
  any scaling, with the actual selected HDU recorded as the card source.
* Decode ``physical = BSCALE * stored + BZERO`` **exactly once**, including
  scaled floating FITS. Defaults follow the FITS standard (1.0 / 0.0).
* Detect integer ``BLANK`` in *stored* space before scaling (integrality
  enforced: ``BLANK=1.5`` rejects); NaN/±Inf are invalid samples.
* Byte order is storage, not camera identity: convert to native-endian
  contiguous float32 only after float64 physical decoding.
* Require a raw-domain declaration backed by header ADU units or an explicit
  evidence-backed import declaration; reject Jy/electrons/rates/normalized/
  processed inputs.
* Multi-HDU FITS requires a unique supported 2D sensor plane or an explicit
  HDU identifier; RGB cubes and processed-history markers reject.
* Alias/duplicate singleton/scaling contradictions and malformed numeric cards
  reject, never first-value preference.
* Precision refusal: integer ADU beyond 2**24, 64-bit integers beyond float64's
  exact range, or magnitudes overflowing float32.

This module imports ``astropy.io.fits`` only; it never imports ZSSS/ZeAlfie.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike
from typing import Optional, Union

import numpy as np
from astropy.io import fits

from zecalibrator.core.dq import INPUT_INVALID
from zecalibrator.core.errors import DecodeError, PrecisionRefusalError
from zecalibrator.core.metadata import (
    ImportDeclaration,
    build_sensor_metadata,
    collect_cards,
    resolve_aliases_with_provenance,
)
from zecalibrator.core.precision import (
    PrecisionInfo,
    integer_exceeds_float32_exact_range,
    measure_float32_roundoff,
)

StrPath = Union[str, PathLike]

# Recognized processed-history markers that cause rejection (SCIENCE §2.7).
_PROCESSED_KEYWORDS = (
    "ZECALIBR",
    "ZECAL",
    "CALIBRAT",
    "PROCESSED",
    "DEBAYER",
    "DEBAYERED",
    "NORMALIZE",
    "NORMALIZED",
    "WHITEBAL",
    "STRETCH",
    "REDUCED",
    "STACKED",
)

_PROCESSED_HISTORY_TOKENS = (
    "calibrat",
    "normaliz",
    "debayer",
    "stretch",
    "white balance",
    "white-balance",
    "processed",
    "reduced",
    "stacked",
)

_ADU_UNITS = ("adu", "adus", "dn", "count", "counts")

FLOAT64_EXACT_INT_BOUND = 2 ** 53


@dataclass(frozen=True)
class DecodedFrame:
    """A strictly decoded raw FITS sensor plane.

    ``data`` is a signed native contiguous float32 physical-ADU plane; ``mask``
    is a uint16 DQ mask with ``INPUT_INVALID`` set for stored-space BLANK /
    NaN / ±Inf. Both are freshly allocated; the source file and header are never
    modified.
    """

    data: np.ndarray
    mask: np.ndarray
    metadata: object  # SensorMetadata (immutable)
    stored_dtype: str
    bscale: float
    bzero: float
    blank: Optional[int]
    hdu: Union[int, str]
    precision: PrecisionInfo


def _iter_image_hdus(hdulist) -> list:
    found = []
    for i, hdu in enumerate(hdulist):
        if hdu.data is None:
            continue
        if not hasattr(hdu, "header") or hdu.header.get("NAXIS", 0) < 2:
            continue
        found.append((i, hdu))
    return found


def _check_rgb(hdu) -> None:
    naxis = hdu.header.get("NAXIS", 0)
    if naxis >= 3:
        raise DecodeError("RGB_UNSUPPORTED", f"NAXIS={naxis} cube/multispectral plane")


def _hdu_source(hdu_key: Union[int, str]) -> str:
    if isinstance(hdu_key, int):
        return "primary" if hdu_key == 0 else f"hdu:{hdu_key}"
    return f"hdu:{hdu_key}"


def _select_hdu(hdulist, hdu: Optional[Union[int, str]]) -> tuple[Union[int, str], object]:
    if hdu is not None:
        try:
            target = hdulist[hdu]
        except (KeyError, IndexError) as exc:
            raise DecodeError("HDU_NOT_FOUND", str(hdu)) from exc
        if target.data is None:
            raise DecodeError("HDU_NOT_2D_IMAGE", str(hdu))
        _check_rgb(target)
        if target.header.get("NAXIS", 0) != 2:
            raise DecodeError("HDU_NOT_2D_IMAGE", str(hdu))
        return hdu, target
    image_hdus = _iter_image_hdus(hdulist)
    if len(image_hdus) == 0:
        raise DecodeError("NO_2D_IMAGE", "no supported 2D sensor plane found")
    if len(image_hdus) > 1:
        raise DecodeError(
            "HDU_AMBIGUOUS",
            "multiple 2D image HDUs; select an explicit HDU",
        )
    idx, target = image_hdus[0]
    _check_rgb(target)
    if target.header.get("NAXIS", 0) != 2:
        raise DecodeError("NO_2D_IMAGE", "the only image HDU is a 3D cube")
    return idx, target


def _check_processed(cards: tuple) -> None:
    for c in cards:
        kw = c.keyword.upper()
        val = c.value
        if kw in _PROCESSED_KEYWORDS:
            raise DecodeError("PROCESSED_HISTORY", f"keyword {kw}")
        if kw in ("HISTORY", "COMMENT"):
            s = str(val).lower()
            for token in _PROCESSED_HISTORY_TOKENS:
                if token in s:
                    raise DecodeError("PROCESSED_HISTORY", f"{kw} token {token!r}")


# ---------------------------------------------------------------------------
# Master admission (additive; the light path above is byte-for-byte unchanged).
#
# A master is by definition a raw 2-D sensor-domain combination (stacked / median
# / mean / rejection / combination), so those markers are ADMITTED. A master must
# never be a debayered/colour/geometry-changed/display-processed product, so those
# markers are ALWAYS rejected. "calibrated"/"normalized" are role/form aware:
# dark/bias/flat_dark reject them; a flat admits them only with the matching
# flat_form evidence (corrected_unnormalized / normalized_response).
# ---------------------------------------------------------------------------
_MASTER_INCOMPATIBLE_KEYWORDS = (
    "DEBAYER",
    "DEBAYERED",
    "DEMOSAIC",
    "STRETCH",
    "STRETCHED",
    "WHITEBAL",
    "RESAMPLE",
    "RESAMPLED",
    "DISPLAY",
)

_MASTER_INCOMPATIBLE_TOKENS = (
    "debayer",
    "demosaic",
    "stretch",
    "white balance",
    "white-balance",
    "resample",
    "display",
)

_MASTER_CALIBRATED_KEYWORDS = ("CALIBRAT",)
_MASTER_NORMALIZED_KEYWORDS = ("NORMALIZE", "NORMALIZED")

# HISTORY/COMMENT marker vocabulary. Only an un-negated marker signals processed
# history: "uncalibrated"/"unnormalized" are admissible (the "un-" prefix negates
# the marker), and "normalized input" (stacking input normalization) is admissible
# because it is not a normalized *output*.
_MARKER_RE = re.compile(
    r"(?P<neg>un)?(?P<kind>calibrat|normaliz)[A-Za-z]*", re.IGNORECASE
)


def _next_word_after(text: str, pos: int) -> str:
    m = re.match(r"[^A-Za-z]*([A-Za-z]+)", text[pos:])
    return m.group(1).lower() if m else ""


def _master_normalization_signals(text: str) -> tuple:
    """Return ``(calibrated, normalized_output)`` signals for a HISTORY/COMMENT.

    * ``uncalibrated`` / ``unnormalized`` are admissible (negated by ``un-``).
    * ``normalized input`` (stacking input normalization) is admissible, not a
      normalized output.
    * ``calibrated`` signals a calibrated history; any other un-negated
      ``normalized …`` (``normalized output`` / ``normalized response`` / bare
      ``normalized``) signals a normalized output.
    """
    calibrated = False
    normalized_output = False
    for m in _MARKER_RE.finditer(text):
        if m.group("neg") is not None:
            continue
        kind = m.group("kind").lower()
        if kind == "calibrat":
            calibrated = True
        else:
            if _next_word_after(text, m.end()) == "input":
                continue
            normalized_output = True
    return calibrated, normalized_output


def _reject_master_calibrated(role: Optional[str], flat_form: Optional[str], origin: str) -> None:
    """Apply the calibrated-history role/form rule (§8/§9)."""
    if role == "flat":
        if flat_form in ("corrected_unnormalized", "normalized_response"):
            return
        raise DecodeError(
            "MASTER_PROCESSED",
            f"calibrated flat master requires flat_form corrected/normalized "
            f"(got {flat_form!r}); {origin}",
        )
    raise DecodeError(
        "MASTER_PROCESSED",
        f"{role or 'unknown'} master with calibrated history is inadmissible; {origin}",
    )


def _reject_master_normalized(role: Optional[str], flat_form: Optional[str], origin: str) -> None:
    """Apply the normalized-history role/form rule (§8/§9)."""
    if role == "flat":
        if flat_form == "normalized_response":
            return
        raise DecodeError(
            "MASTER_PROCESSED",
            f"normalized flat master requires flat_form normalized_response "
            f"(got {flat_form!r}); {origin}",
        )
    raise DecodeError(
        "MASTER_PROCESSED",
        f"{role or 'unknown'} master with normalized history is inadmissible; {origin}",
    )


def _check_processed_master(cards: tuple, *, role: Optional[str], flat_form: Optional[str]) -> None:
    """Master processed-history admission policy (prepared report §8/§9).

    Strictly additive to the light decoder: never weakens the light path. A
    debayer/demosaic/RGB/stretch/white-balance/resample/display marker is always
    rejected; stacked/median/mean/rejection/combination are admitted (provenance
    preserved); calibrated/normalized are role/form aware.
    """
    for c in cards:
        kw = c.keyword.upper()
        val = c.value
        if kw in _MASTER_INCOMPATIBLE_KEYWORDS:
            raise DecodeError("MASTER_INCOMPATIBLE", f"master processed marker keyword {kw!r}")
        if kw in _MASTER_CALIBRATED_KEYWORDS:
            _reject_master_calibrated(role, flat_form, f"keyword {kw!r}")
        if kw in _MASTER_NORMALIZED_KEYWORDS:
            _reject_master_normalized(role, flat_form, f"keyword {kw!r}")
        if kw in ("HISTORY", "COMMENT"):
            s = str(val).lower()
            for token in _MASTER_INCOMPATIBLE_TOKENS:
                if token in s:
                    raise DecodeError("MASTER_INCOMPATIBLE", f"{kw} token {token!r}")
            calibrated, normalized_output = _master_normalization_signals(s)
            if calibrated:
                _reject_master_calibrated(role, flat_form, f"{kw} token 'calibrat'")
            if normalized_output:
                _reject_master_normalized(role, flat_form, f"{kw} token 'normaliz'")


def _classify_bunit(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    s = s.lower()
    return s or None


def _resolve_units(bunit_raw, declaration) -> tuple[Optional[str], Optional[str]]:
    """Return (units, error_reason). units is "ADU" or None."""
    b = _classify_bunit(bunit_raw)
    if b is None:
        du = None
        if declaration is not None:
            du = _classify_bunit(declaration.units)
        if du is None:
            return None, "UNKNOWN_UNITS"
        if du in _ADU_UNITS:
            return "ADU", None
        return None, "UNITS_UNSUPPORTED"
    if b in _ADU_UNITS:
        return "ADU", None
    return None, "UNITS_UNSUPPORTED"


def decode_fits(
    path: StrPath,
    *,
    hdu: Optional[Union[int, str]] = None,
    declaration: Optional[ImportDeclaration] = None,
    cancel=None,
    progress=None,
    admission: str = "light",
    role: Optional[str] = None,
    flat_form: Optional[str] = None,
) -> DecodedFrame:
    """Decode a raw FITS sensor plane into native contiguous float32 + metadata.

    ``declaration`` is an optional evidence-backed :class:`ImportDeclaration`
    supplying unknown facts (units/domain/acquisition); it never overrides a
    contrary measured FITS card. ``cancel``/``progress`` are optional Qt-free
    cooperative tokens (checked before read and after decode); they may be None.

    ``admission`` selects the processed-history policy. The default ``"light"``
    is byte-for-byte the historical strict raw-light policy (unchanged).
    ``"master"`` applies the additive role/form/provenance-aware master admission
    policy (``role`` is the master role; ``flat_form`` the flat form evidence).
    """
    if cancel is not None and cancel.is_cancelled():
        from zecalibrator.application.cancellation import OperationCancelled

        raise OperationCancelled()

    def emit_progress(phase: str, completed: int, total: int) -> None:
        if progress is None:
            return
        report = getattr(progress, "report", None)
        if report is None and callable(progress):
            report = progress
        if report is None:
            return
        from zecalibrator.application.cancellation import ProgressEvent

        try:
            report(ProgressEvent(operation_id="zecalibrator-decode", phase=phase, completed=completed, total=total, unit="frames"))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Observer failure is isolated and never changes decode science.
            return
        if cancel is not None and cancel.is_cancelled():
            from zecalibrator.application.cancellation import OperationCancelled

            raise OperationCancelled()

    emit_progress("read", 0, 1)

    with fits.open(path, do_not_scale_image_data=True, memmap=False) as hdulist:
        hdu_key, hdu = _select_hdu(hdulist, hdu)
        header = hdu.header
        cards = collect_cards(header, source=_hdu_source(hdu_key))

        if admission == "master":
            _check_processed_master(cards, role=role, flat_form=flat_form)
        else:
            _check_processed(cards)

        normalized, conflicts, malformed, provenance = resolve_aliases_with_provenance(cards)
        if conflicts:
            first = conflicts[0]
            raise DecodeError(
                "ALIAS_CONFLICT",
                f"{first.field}: {first.keywords} disagree ({first.values})",
            )
        if malformed:
            raise DecodeError("MALFORMED_CARD", f"{malformed[0]} is not numeric")

        bscale = float(normalized.get("bscale", 1.0))
        bzero = float(normalized.get("bzero", 0.0))
        blank = normalized.get("blank")
        if blank is not None:
            bf = float(blank)
            if not bf.is_integer():
                raise DecodeError("MALFORMED_CARD", f"BLANK={blank!r} is not an integer")
            blank = int(bf)

        stored = np.asarray(hdu.data)
        if stored.ndim != 2:
            raise DecodeError("NOT_2D_PLANE", f"NAXIS={stored.ndim}")

        # Precision guard: 64-bit integer storage may exceed float64's exact range.
        if stored.dtype.itemsize == 8 and np.issubdtype(stored.dtype, np.integer):
            if np.issubdtype(stored.dtype, np.signedinteger):
                mag = np.abs(stored.astype(np.int64))
            else:
                mag = stored.astype(np.uint64)
            if np.any(mag > np.uint64(FLOAT64_EXACT_INT_BOUND)):
                raise PrecisionRefusalError(
                    "stored 64-bit integer exceeds float64 exact-integer bound 2**53"
                )

        invalid = np.zeros(stored.shape, dtype=bool)
        if blank is not None and np.issubdtype(stored.dtype, np.integer):
            invalid |= stored == blank

        stored_f64 = stored.astype(np.float64)
        if np.issubdtype(stored.dtype, np.floating):
            invalid |= ~np.isfinite(stored_f64)

        physical = bscale * stored_f64 + bzero

        finite = np.isfinite(physical)
        if finite.any():
            vals = physical[finite]
            if np.any(np.abs(vals) > float(np.finfo(np.float32).max)):
                raise PrecisionRefusalError(
                    "decoded magnitude exceeds float32 range (extreme scale)"
                )
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

        data32 = np.ascontiguousarray(physical.astype(np.float32))
        mask = np.zeros(stored.shape, dtype=np.uint16)
        mask[invalid] |= INPUT_INVALID
        mask[~np.isfinite(data32)] |= INPUT_INVALID
        data32[~np.isfinite(data32)] = np.nan
        data32[invalid] = np.nan

        precision = measure_float32_roundoff(physical)

        # Units come from header BUNIT (must be ADU) or the declaration's units.
        units, units_error = _resolve_units(normalized.get("bunit"), declaration)
        if units_error is not None:
            raise DecodeError(units_error, f"BUNIT={normalized.get('bunit')!r}")

        # Raw-domain evidence is required independent of units (SCIENCE §2.7): an
        # evidence-backed declaration (nonempty source/identity/version) is the raw
        # proof; BUNIT=ADU alone is not evidence the frame is unprocessed raw.
        if declaration is None:
            raise DecodeError("UNKNOWN_DOMAIN", "no raw-domain declaration/import evidence")

        metadata = build_sensor_metadata(
            shape=stored.shape,
            normalized=normalized,
            conflicts=conflicts,
            cards=cards,
            declaration=declaration,
            units=units,
            provenance=provenance,
        )
        if metadata.conflicts:
            first = metadata.conflicts[0]
            raise DecodeError(
                "DECLARATION_CONFLICT",
                f"{first.field}: {first.keywords} disagree ({first.values})",
            )

    if cancel is not None and cancel.is_cancelled():
        from zecalibrator.application.cancellation import OperationCancelled

        raise OperationCancelled()

    emit_progress("complete", 1, 1)

    return DecodedFrame(
        data=data32,
        mask=mask,
        metadata=metadata,
        stored_dtype=str(stored.dtype),
        bscale=bscale,
        bzero=bzero,
        blank=blank,
        hdu=hdu_key,
        precision=precision,
    )


__all__ = ["DecodedFrame", "decode_fits"]
