"""Application-layer master-admission + role-identification logic (P7-M3B).

Relocated out of the ``api/v1`` facade (formerly a private ``api/v1`` helper)
into the ``application`` layer so that **no importer depends on a private ``_``
module inside the public API package**: the GUI (``zecalibrator.gui.service``)
and the public session-library facade (``zecalibrator.api.v1.session``) both
import this logic from here, keeping the layering ``api.v1 -> application ->
core/io``.

``zecalibrator.gui.service`` re-exports these names unchanged for backward
compatibility (its public surface is preserved). This module is Qt-free and
imports only the standard library, ``zecalibrator.core`` value objects and the
filesystem source adapter — never ``api.v1``, never Qt/ZeAlfie/ZSSS.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Optional, Tuple, Union

from zecalibrator.io.master_source import FilesystemSource

from zecalibrator.core.managed import EvidenceFact, ManagedMasterRecord
from zecalibrator.core.metadata import ImportDeclaration


EVIDENCE_SOURCE_MAP: Mapping[str, Tuple[str, ...]] = {
    "exposure_s": ("EXPTIME", "EXPOSURE"),
    "temperature_c": ("CCD-TEMP", "TEMPCCD"),
    "temperature_setpoint_c": ("SET-TEMP",),
    "gain": ("GAIN",),
    "gain_e_per_adu": ("EGAIN",),
    "offset": ("OFFSET", "PEDESTAL"),
    "readout_mode": ("READOUTM", "READMODE"),
    "adc_mode": ("ADCMODE",),
    "detector_model": ("INSTRUME", "CAMERA", "DETECTOR"),
    "cfa_phase": ("BAYERPAT", "CFA"),
    "filter": ("FILTER",),
    # binning handled specially (scalar BINNING or XBINNING+YBINNING pair).
}


_BINNING_SCALAR_KEYWORDS = ("BINNING",)


_BINNING_PAIR_KEYWORDS = ("XBINNING", "YBINNING")


_NUMERIC_EVIDENCE_FIELDS = frozenset({"exposure_s", "temperature_c", "temperature_setpoint_c", "gain", "gain_e_per_adu", "offset"})


def _coerce_evidence_value(field: str, raw):
    """Coerce a raw header value into the ImportDeclaration field's value.

    Numeric fields become finite floats; all others become stripped strings.
    ``cfa_phase`` is normalized (``MONO``/``mono`` -> ``mono``; Bayer patterns
    uppercase). ``None``/unparseable numeric values yield ``None`` (no candidate).
    """
    if raw is None:
        return None
    if field in _NUMERIC_EVIDENCE_FIELDS:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        import math

        if not math.isfinite(v):
            return None
        return v
    s = str(raw).strip()
    if not s:
        return None
    if field == "cfa_phase":
        return "mono" if s.upper() == "MONO" else s.upper()
    return s


def _normalize_cards(cards):
    """Return ``{keyword: [values in order]}`` from card objects or ``(kw, val)`` pairs."""
    by_keyword: dict = {}
    for c in cards:
        if isinstance(c, (tuple, list)) and len(c) == 2:
            kw, val = c[0], c[1]
        else:
            kw = getattr(c, "keyword", None)
            val = getattr(c, "value", None)
        if kw is None:
            continue
        by_keyword.setdefault(kw, []).append(val)
    return by_keyword


def detect_header_candidates(cards):
    """Detect candidate ``EvidenceFact`` values from approved header cards only.

    Returns ``(candidates, conflicts)``:

    * ``candidates`` — ``{field: EvidenceFact}`` for fields with exactly one
      distinct value (``confirmed_by=None`` until the user confirms).
    * ``conflicts`` — ``{field: tuple[EvidenceFact, ...]}`` for fields with more
      than one distinct value; never auto-arbitrated.

    Only cards in :data:`EVIDENCE_SOURCE_MAP` produce a candidate; an
    unapproved alias produces no candidate. Heuristic inference never runs.
    """
    by_keyword = _normalize_cards(cards)
    candidates: dict = {}
    conflicts: dict = {}

    for field, keywords in EVIDENCE_SOURCE_MAP.items():
        values: list = []
        origins: list = []
        for kw in keywords:
            for raw in by_keyword.get(kw, ()):
                v = _coerce_evidence_value(field, raw)
                if v is None:
                    continue
                values.append(v)
                origins.append(kw)
        if not values:
            continue
        distinct: list = []
        for v in values:
            if v not in distinct:
                distinct.append(v)
        if len(distinct) > 1:
            conflicts[field] = tuple(
                EvidenceFact(field=field, value=value, origin_type="fits_header",
                                origin_field=origin, confirmed_by=None, version="1")
                for value, origin in zip(values, origins)
            )
        else:
            candidates[field] = EvidenceFact(
                field=field, value=distinct[0], origin_type="fits_header",
                origin_field=origins[0], confirmed_by=None, version="1",
            )

    # binning: scalar BINNING -> (v, v); XBINNING+YBINNING pair -> (y, x).
    bin_values: list = []
    bin_origins: list = []
    scalar_b = by_keyword.get("BINNING")
    x_vals = by_keyword.get("XBINNING")
    y_vals = by_keyword.get("YBINNING")
    if scalar_b:
        for raw in scalar_b:
            try:
                n = int(float(raw))
            except (TypeError, ValueError):
                continue
            if n >= 1:
                bin_values.append((n, n))
                bin_origins.append("BINNING")
    elif x_vals and y_vals:
        for xraw, yraw in zip(x_vals, y_vals):
            try:
                x = int(float(xraw))
                y = int(float(yraw))
            except (TypeError, ValueError):
                continue
            if x >= 1 and y >= 1:
                bin_values.append((y, x))
                bin_origins.append("XBINNING+YBINNING")
    if bin_values:
        distinct = [v for i, v in enumerate(bin_values) if v not in bin_values[:i]]
        if len(distinct) > 1:
            conflicts["binning"] = tuple(
                EvidenceFact(field="binning", value=value, origin_type="fits_header",
                                origin_field=origin, confirmed_by=None, version="1")
                for value, origin in zip(bin_values, bin_origins)
            )
        else:
            candidates["binning"] = EvidenceFact(
                field="binning", value=distinct[0], origin_type="fits_header",
                origin_field=bin_origins[0], confirmed_by=None, version="1",
            )

    return candidates, conflicts


def read_header_candidates(path: str, hdu=0):
    """Read FITS header cards (public source adapter) and detect candidates.

    Returns ``(candidates, conflicts)`` from :func:`detect_header_candidates`.
    """
    cards = FilesystemSource().read_header(path, hdu=hdu)
    return detect_header_candidates([(c.keyword, c.value) for c in cards])


IMAGETYP_ROLE_MAP: Mapping[str, str] = {
    "DARK": "dark",
    "BIAS": "bias",
    "FLAT": "flat",
    "DARKFLAT": "flat_dark",
}


_ROLE_TO_IMAGETYP: Mapping[str, str] = {v: k for k, v in IMAGETYP_ROLE_MAP.items()}


_RGB_DEBAYER_INCOMPATIBLE = (
    "RGB / debayered 3-channel master; ZeCalibrator requires raw 2-D sensor-domain master"
)


_MASTER_INCOMPATIBLE_KEYWORDS = (
    "DEBAYER", "DEBAYERED", "DEMOSAIC", "STRETCH", "STRETCHED",
    "WHITEBAL", "RESAMPLE", "RESAMPLED", "DISPLAY",
)


_MASTER_INCOMPATIBLE_TOKENS = (
    "debayer", "demosaic", "stretch", "white balance", "white-balance",
    "resample", "display",
)


_MASTER_CALIBRATED_KEYWORDS = ("CALIBRAT",)


_MASTER_NORMALIZED_KEYWORDS = ("NORMALIZE", "NORMALIZED")


_MARKER_RE = re.compile(
    r"(?P<neg>un)?(?P<kind>calibrat|normaliz)[A-Za-z]*", re.IGNORECASE
)


def _next_word_after(text: str, pos: int) -> str:
    m = re.match(r"[^A-Za-z]*([A-Za-z]+)", text[pos:])
    return m.group(1).lower() if m else ""


def _master_normalization_signals(text: str) -> Tuple[bool, bool]:
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


def _iter_keyword_values(cards):
    """Yield ``(UPPERCASED keyword, value)`` from card objects or ``(kw, val)`` pairs."""
    for c in cards:
        if isinstance(c, (tuple, list)) and len(c) == 2:
            kw, val = c[0], c[1]
        else:
            kw = getattr(c, "keyword", None)
            val = getattr(c, "value", None)
        if kw is None:
            continue
        yield str(kw).upper(), val


def detect_imagetyp_role(cards) -> Optional[str]:
    """Return the detected master role from a FITS ``IMAGETYP`` card (or ``None``).

    Frozen map: DARK→dark, BIAS→bias, FLAT→flat, DARKFLAT→flat_dark. Multiple
    *distinct* IMAGETYP values yield ``None`` (never auto-arbitrated). Filename/
    folder/directory layout never produce a role.
    """
    roles: list = []
    for kw, val in _iter_keyword_values(cards):
        if kw == "IMAGETYP":
            role = IMAGETYP_ROLE_MAP.get(str(val).strip().upper())
            if role is not None and role not in roles:
                roles.append(role)
    return roles[0] if len(roles) == 1 else None


def _master_calibrated_reason(role, flat_form, origin) -> Optional[str]:
    if role == "flat":
        if flat_form in ("corrected_unnormalized", "normalized_response"):
            return None
        return (
            f"calibrated flat master requires flat_form corrected/normalized "
            f"(got {flat_form!r}); {origin}"
        )
    return f"{role or 'unknown'} master with calibrated history is inadmissible; {origin}"


def _master_normalized_reason(role, flat_form, origin) -> Optional[str]:
    if role == "flat":
        if flat_form == "normalized_response":
            return None
        return (
            f"normalized flat master requires flat_form normalized_response "
            f"(got {flat_form!r}); {origin}"
        )
    return f"{role or 'unknown'} master with normalized history is inadmissible; {origin}"


def master_incompatibility(cards, *, role=None, flat_form=None) -> Tuple[str, ...]:
    """Return the reasons a master is inadmissible (empty tuple = admissible).

    Mirrors the decoder's additive master admission (§8/§9): NAXIS>=3 / debayer /
    demosaic / stretch / white-balance / resample / display are always
    incompatible; calibrated/normalized are role/form aware. This reports *why* a
    master is incompatible without loading pixel arrays.
    """
    reasons: list = []
    for kw, val in _iter_keyword_values(cards):
        if kw == "NAXIS":
            try:
                if int(float(val)) >= 3:
                    reasons.append(_RGB_DEBAYER_INCOMPATIBLE)
            except (TypeError, ValueError):
                pass
            continue
        if kw in _MASTER_INCOMPATIBLE_KEYWORDS:
            reasons.append(f"processed marker {kw!r}")
            continue
        if kw in _MASTER_CALIBRATED_KEYWORDS:
            reason = _master_calibrated_reason(role, flat_form, f"keyword {kw!r}")
            if reason:
                reasons.append(reason)
            continue
        if kw in _MASTER_NORMALIZED_KEYWORDS:
            reason = _master_normalized_reason(role, flat_form, f"keyword {kw!r}")
            if reason:
                reasons.append(reason)
            continue
        if kw in ("HISTORY", "COMMENT"):
            s = str(val).lower()
            for token in _MASTER_INCOMPATIBLE_TOKENS:
                if token in s:
                    reasons.append(f"processed marker {token!r}")
                    break
            calibrated, normalized_output = _master_normalization_signals(s)
            if calibrated:
                reason = _master_calibrated_reason(role, flat_form, "HISTORY 'calibrat'")
                if reason:
                    reasons.append(reason)
            if normalized_output:
                reason = _master_normalized_reason(role, flat_form, "HISTORY 'normaliz'")
                if reason:
                    reasons.append(reason)
    return tuple(dict.fromkeys(reasons))


USER_ONLY_FIELDS = (
    "detector_instance_id", "sensor_dimensions", "orientation", "roi_origin",
    "optical_train_id", "bias_exposure_max_s", "saturation_limit_adu",
    "saturation_evidence",
)


_BAYER_CFA_PHASES = frozenset({"GRBG", "RGGB", "BGGR", "GBRG"})


NECESSARY_FIELDS_BY_MASTER_TYPE: Mapping[str, Tuple[str, ...]] = {
    "dark": (
        "detector_model", "gain", "offset", "binning", "cfa_phase",
        "exposure_s", "temperature_c",
    ),
    "bias": (
        "detector_model", "gain", "offset", "binning", "cfa_phase",
    ),
    "flat": (
        "detector_model", "gain", "offset", "binning", "cfa_phase",
        "filter",
    ),
    "flat_dark": (
        "detector_model", "gain", "offset", "binning", "cfa_phase",
        "exposure_s", "temperature_c",
    ),
}


CFA_ONLY_FIELDS: Tuple[str, ...] = ("orientation", "roi_origin")


DISAMBIGUATOR_FIELDS_BY_MASTER_TYPE: Mapping[str, Tuple[str, ...]] = {
    "dark": (
        "detector_instance_id", "readout_mode", "adc_mode", "sensor_dimensions",
    ),
    "bias": (
        "detector_instance_id", "readout_mode", "adc_mode", "sensor_dimensions",
    ),
    "flat": (
        "detector_instance_id", "readout_mode", "adc_mode", "sensor_dimensions",
        "optical_train_id",
    ),
    "flat_dark": (
        "detector_instance_id", "readout_mode", "adc_mode", "sensor_dimensions",
    ),
}


def is_bayer_phase(cfa_phase) -> bool:
    """Return ``True`` when ``cfa_phase`` denotes a Bayer CFA sensor."""
    return cfa_phase in _BAYER_CFA_PHASES


def necessary_fields(master_type: str, cfa_phase=None) -> Tuple[str, ...]:
    """Return the necessary (matching-blocking) fields for a master type.

    CFA-only geometry facts (``orientation``/``roi_origin``) are included only
    when ``cfa_phase`` is a Bayer phase.
    """
    base = NECESSARY_FIELDS_BY_MASTER_TYPE.get(master_type, ())
    if is_bayer_phase(cfa_phase):
        return base + CFA_ONLY_FIELDS
    return base


REQUIRED_FIELDS_BY_MASTER_TYPE = NECESSARY_FIELDS_BY_MASTER_TYPE


def required_fields_for_master_type(master_type: str) -> Tuple[str, ...]:
    """Return the necessary (matching-blocking) field minimum for a master type."""
    return NECESSARY_FIELDS_BY_MASTER_TYPE.get(master_type, ())


_FLAT_QUALITY_REASON = "flat quality evidence (normalization/validity/saturation) insufficient"


def missing_required_fields(master_type: str, declaration) -> Tuple[str, ...]:
    """Return the necessary-field names whose declaration value is ``None``.

    Uses only the NECESSARY tier (R3C): a missing disambiguator never appears
    here and never blocks "ready". CFA-only geometry facts are required only
    for a Bayer sensor. Absent facts are never invented.
    """
    required = necessary_fields(master_type, getattr(declaration, "cfa_phase", None))
    return tuple(
        f for f in required if getattr(declaration, f, None) is None
    )


def master_evidence_status(master_type: str, declaration) -> Tuple[str, Tuple[str, ...]]:
    """Return ``(status, reasons)`` for a master's evidence completeness.

    ``status`` is ``"ready"`` when every NECESSARY (matching-blocking) field is
    present and (for flats) machine-readable quality evidence exists; otherwise
    ``"needs_attention"`` with the concrete missing/insufficient reasons. A
    missing disambiguator never reports "needs attention" (R3C).

    For flats, the R4 quality evidence (normalization/validity/saturation/CFA
    quality) can never be a user assertion; without machine-readable provenance
    it is reported as insufficient rather than silently accepted.
    """
    reasons = list(missing_required_fields(master_type, declaration))
    if master_type == "flat":
        reasons.append(_FLAT_QUALITY_REASON)
    if reasons:
        return ("needs_attention", tuple(reasons))
    return ("ready", ())


def build_declaration(source, identity, version, evidence: Mapping, extra: Optional[Mapping] = None) -> "ImportDeclaration":
    """Build a validated :class:`ImportDeclaration` from confirmed evidence facts
    plus explicit user/native facts (``extra``).

    ``evidence`` values are the confirmed per-field facts (canonical field names);
    ``extra`` supplies user/native-only facts (never header-derived). Absent
    facts stay absent (``None``) and are never invented.
    """
    kwargs: dict = {
        "source": source, "identity": identity, "version": version,
        "domain": "raw", "units": "ADU",
    }
    for field, fact in evidence.items():
        kwargs[field] = fact.value
    if extra:
        for k, v in extra.items():
            kwargs[k] = v
    return ImportDeclaration(**kwargs)


def make_managed_record(
    *,
    role: str,
    content_sha256: str,
    size_bytes: int,
    declaration,
    hdu=0,
    bias_state: Optional[str] = None,
    flat_form: Optional[str] = None,
    evidence: Optional[Mapping] = None,
    dq_state: str = "no_source_dq",
    mask_path: Optional[str] = None,
    last_seen_path: Optional[str] = None,
) -> "ManagedMasterRecord":
    """Build an immutable :class:`ManagedMasterRecord` (public value object)."""
    return ManagedMasterRecord(
        role=role,
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        declaration=declaration,
        hdu=hdu,
        bias_state=bias_state,
        flat_form=flat_form,
        evidence=dict(evidence or {}),
        dq_state=dq_state,
        mask_path=mask_path,
        last_seen_path=last_seen_path,
    )


__all__ = [
    "EVIDENCE_SOURCE_MAP",
    "_BINNING_SCALAR_KEYWORDS",
    "_BINNING_PAIR_KEYWORDS",
    "_NUMERIC_EVIDENCE_FIELDS",
    "_coerce_evidence_value",
    "_normalize_cards",
    "detect_header_candidates",
    "read_header_candidates",
    "IMAGETYP_ROLE_MAP",
    "_ROLE_TO_IMAGETYP",
    "_RGB_DEBAYER_INCOMPATIBLE",
    "_MASTER_INCOMPATIBLE_KEYWORDS",
    "_MASTER_INCOMPATIBLE_TOKENS",
    "_MASTER_CALIBRATED_KEYWORDS",
    "_MASTER_NORMALIZED_KEYWORDS",
    "_MARKER_RE",
    "_next_word_after",
    "_master_normalization_signals",
    "_iter_keyword_values",
    "detect_imagetyp_role",
    "_master_calibrated_reason",
    "_master_normalized_reason",
    "master_incompatibility",
    "USER_ONLY_FIELDS",
    "_BAYER_CFA_PHASES",
    "NECESSARY_FIELDS_BY_MASTER_TYPE",
    "CFA_ONLY_FIELDS",
    "DISAMBIGUATOR_FIELDS_BY_MASTER_TYPE",
    "is_bayer_phase",
    "necessary_fields",
    "REQUIRED_FIELDS_BY_MASTER_TYPE",
    "required_fields_for_master_type",
    "_FLAT_QUALITY_REASON",
    "missing_required_fields",
    "master_evidence_status",
    "build_declaration",
    "make_managed_record",
]
