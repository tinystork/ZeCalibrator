"""Policy-driven standalone FITS output-header builder (CFA-metadata preservation).

A single pure function that projects the accepted plan's canonical
``light_constraints`` plus a *closed whitelist* of audited evidence cards onto
the interoperable standalone-header surface. It performs **no header copy**:
every emitted card is either (a) an explicitly named canonical field or (b) an
explicitly whitelisted evidence card whose value is written verbatim only when
it is present with a single unambiguous value.

This module is deliberately free of I/O, ``astropy`` and mutation: it is a
deterministic dict builder, unit-testable against synthetic
``LightConstraints`` / ``CardRecord`` inputs.

The structural/scaling/checksum cards (``SIMPLE BITPIX NAXIS NAXIS1 NAXIS2
EXTEND BSCALE BZERO BLANK CHECKSUM DATASUM``), ``EGAIN``, ``HISTORY``/``COMMENT``
and any unknown/vendor card are **never** emitted here (and are never copied).
``BUNIT=ADU`` remains owned by the writer, exactly where it is set today.
"""

from __future__ import annotations

from typing import Optional

# Explicit Bayer CFA phases for which ``BAYERPAT`` is meaningful. ``"mono"`` is
# an explicit monochrome sensor (no Bayer pattern) and ``None`` is an unknown
# CFA (never silently assumed mono or Bayer).
_BAYER_PHASES = frozenset({"GRBG", "RGGB", "BGGR", "GBRG"})

# Closed evidence whitelist (category B). Each is written verbatim only when
# present with a single unambiguous value. ``OFFSET`` is handled separately
# because its canonical value takes precedence when known.
_EVIDENCE_WHITELIST = ("IMAGETYP", "DATE-OBS", "OBJECT", "FOCALLEN", "ROTATOR")

_OFFSET_KEYWORD = "OFFSET"
_HIERARCH_PREFIX = "HIERARCH "


def _card_keyword(keyword: object) -> str:
    """Normalize a card keyword for comparison (HIERARCH-prefix + case).

    Mirrors the decoder's ``_strip_hierarch``/uppercase convention so a
    ``HIERARCH``-encoded standard keyword matches its plain counterpart.
    """
    k = str(keyword)
    if k.startswith(_HIERARCH_PREFIX):
        k = k[len(_HIERARCH_PREFIX):]
    return k.upper()


def _is_pair(value: object) -> bool:
    """Return ``True`` when ``value`` is a 2-element tuple/list."""
    return isinstance(value, (tuple, list)) and len(value) == 2


def _values_equal(a: object, b: object) -> bool:
    """Numeric-aware equality for "single unambiguous value" evidence checks."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def _evidence_value(cards, keyword: str) -> Optional[object]:
    """Return the single unambiguous value for ``keyword`` among ``cards``.

    ``None`` when the keyword is absent, or when it is present multiple times
    with disagreeing values (ambiguous). Never bulk-copies: only the one named
    keyword is inspected.
    """
    target = _card_keyword(keyword)
    matches = [c for c in cards if _card_keyword(c.keyword) == target]
    if not matches:
        return None
    first = matches[0].value
    for c in matches[1:]:
        if not _values_equal(first, c.value):
            return None
    return first


def build_output_header_fields(
    *,
    light_constraints,
    science_shape,
    status,
    plan_id,
    provenance_schema,
) -> dict[str, object]:
    """Build the standalone-output header cards for one calibrated light.

    Emits the ZeCalibrator identity/status cards (byte-identical to the
    previous hardcoded batch cards), then the canonical ``light_constraints``
    acquisition/geometry cards, then the closed evidence whitelist. No header
    copy of any kind; structural/scaling/checksum cards are left to the writer.
    """
    fields: dict[str, object] = {}

    # ------------------------------------------------------------------
    # ZeCalibrator identity + status (category D) — byte-identical values.
    # ------------------------------------------------------------------
    fields["ZECALCAL"] = "zecalibrator"
    fields["HIERARCH ZECALSCHEMA"] = provenance_schema
    fields["HIERARCH ZECALPLAN"] = plan_id[:16]
    fields["HIERARCH ZECALSTAT"] = status

    geometry = light_constraints.geometry
    detector = light_constraints.detector
    acquisition = light_constraints.acquisition
    optical = light_constraints.optical

    # ------------------------------------------------------------------
    # Canonical-valued cards (category A) from the accepted plan.
    # ------------------------------------------------------------------
    # BAYERPAT: only for an explicit Bayer phase, and only when the calibrated
    # science shape still equals the canonical geometry shape (fail-closed guard;
    # the current architecture is shape-preserving, so this is a net, not a
    # transformation).
    cfa_phase = geometry.cfa_phase
    if (
        cfa_phase in _BAYER_PHASES
        and science_shape is not None
        and geometry.shape is not None
        and tuple(science_shape) == tuple(geometry.shape)
    ):
        fields["BAYERPAT"] = cfa_phase

    if detector.detector_model is not None:
        fields["INSTRUME"] = detector.detector_model

    if optical.filter is not None:
        fields["FILTER"] = optical.filter

    if acquisition.exposure_s is not None:
        fields["EXPTIME"] = acquisition.exposure_s

    # GAIN is the gain *setting*, never EGAIN (electrons/ADU).
    if acquisition.gain is not None:
        fields["GAIN"] = acquisition.gain

    # Binning: geometry.binning is stored as (bin_y, bin_x) (see
    # metadata.build_sensor_metadata), so the X/Y cards map to the x/y
    # components respectively — a faithful round-trip of the decoder.
    binning = geometry.binning
    if _is_pair(binning):
        fields["XBINNING"] = binning[1]
        fields["YBINNING"] = binning[0]

    if acquisition.temperature_c is not None:
        fields["CCD-TEMP"] = acquisition.temperature_c

    # ROI origin: geometry.roi_origin is (y, x); XORGSUBF is the x component,
    # YORGSUBF is the y component.
    roi_origin = geometry.roi_origin
    if _is_pair(roi_origin):
        fields["XORGSUBF"] = roi_origin[1]
        fields["YORGSUBF"] = roi_origin[0]

    # Canonical OFFSET wins when known (usually unknown for a Standard light).
    if acquisition.offset is not None:
        fields["OFFSET"] = acquisition.offset

    # ------------------------------------------------------------------
    # Evidence-card cards (category B) from the audited light's own cards.
    # ------------------------------------------------------------------
    evidence = light_constraints.evidence
    original_cards = evidence.original_cards if evidence is not None else ()

    # OFFSET only when the canonical acquisition.offset is unknown.
    if acquisition.offset is None:
        offset_value = _evidence_value(original_cards, _OFFSET_KEYWORD)
        if offset_value is not None:
            fields["OFFSET"] = offset_value

    for keyword in _EVIDENCE_WHITELIST:
        value = _evidence_value(original_cards, keyword)
        if value is not None:
            fields[keyword] = value

    return fields


__all__ = ["build_output_header_fields"]
