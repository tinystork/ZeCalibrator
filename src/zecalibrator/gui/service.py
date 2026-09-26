"""GUI API client — pure, Qt-free request construction and value-object parsing.

This module is the *only* place under ``zecalibrator.gui`` that constructs the
public ``zecalibrator.api.v1`` request/value objects and parses user-supplied
JSON (import declarations, ROI-extent evidence, master import specs) into those
public value objects. It imports **only** ``zecalibrator.api.v1`` (the exported
stable contract) plus the standard library — never ``zecalibrator.core`` /
``zecalibrator.application`` / ``zecalibrator.io`` / ``zecalibrator.cli``.

The worker owns the actual expensive API calls; this module is deliberately
import-safe on a headless install (``zecalibrator.api.v1`` cold import is cheap
and Qt/NumPy-free until first lazy use).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import zecalibrator.api.v1 as v1

# ---------------------------------------------------------------------------
# Identity helpers (GUI operation ids are distinct from the fixed engine ids)
# ---------------------------------------------------------------------------
def new_operation_id() -> str:
    """Return a fresh GUI operation id (128-bit hex, independent of engine ids)."""
    return uuid.uuid4().hex


def new_batch_id() -> str:
    """Return a fresh standalone batch id (128-bit hex)."""
    return uuid.uuid4().hex


def parse_hdu(value: str) -> Union[int, str]:
    """Parse an HDU selector exactly like the CLI (integer when numeric, else name)."""
    s = str(value).strip()
    try:
        return int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Supported input file extensions + folder scan (single shared definition)
# ---------------------------------------------------------------------------
# The one authoritative set of raw-FITS input suffixes. Both the Add-files
# dialog filter and the Add-folder scan derive from this tuple so the two can
# never diverge. No new formats are invented here.
SUPPORTED_INPUT_EXTENSIONS: Tuple[str, ...] = (".fits", ".fit", ".fts")


def input_dialog_filter() -> str:
    """Return the shared FITS input file dialog filter string.

    Built from :data:`SUPPORTED_INPUT_EXTENSIONS` (never a second hard-coded
    list). ``All files (*)`` remains available so the user can override the
    extension hint; the folder scan still only ever admits supported input
    files.
    """
    patterns = " ".join(f"*{ext}" for ext in SUPPORTED_INPUT_EXTENSIONS)
    return f"FITS files ({patterns});;All files (*)"


def is_supported_input_file(name: str) -> bool:
    """True when ``name`` has a supported input-file suffix (case-insensitive).

    Suffix matching is deterministic and identical for the dialog filter and the
    folder scan: a path/name is accepted iff its lower-cased suffix is in
    :data:`SUPPORTED_INPUT_EXTENSIONS`.
    """
    return Path(name).suffix.lower() in SUPPORTED_INPUT_EXTENSIONS


def path_identity(path: str) -> str:
    """Deterministic dedup identity for a light path (normcase + abspath)."""
    return os.path.normcase(os.path.abspath(path))


def dedup_input_paths(paths, existing_paths=()) -> list:
    """Return ``paths`` with duplicates removed by :func:`path_identity`.

    Order is preserved (first occurrence wins); any path already present in
    ``existing_paths`` (compared by the same identity) is skipped.
    """
    seen = {path_identity(p) for p in existing_paths}
    result: list = []
    for p in paths:
        ident = path_identity(p)
        if ident not in seen:
            seen.add(ident)
            result.append(p)
    return result


def scan_folder_inputs(folder: str, *, recursive: bool = False) -> Tuple[list, int]:
    """Scan ``folder`` for supported input FITS files and return ``(input_paths, unsupported)``.

    - ``input_paths`` are the files whose suffix matches the shared supported
      input extension set (case-insensitive), in deterministic sorted order.
    - ``unsupported`` counts *files* whose suffix does not match the supported
      set (a genuine candidate FITS is never silently dropped); directories are
      never counted as unsupported.

    ``recursive=False`` (default) scans the top level only: subdirectories are
    never recursed into, and are not counted as unsupported (they are not files).

    ``recursive=True`` discovers the same admissible extensions at every depth.
    Traversal uses ``os.walk(root, followlinks=False)`` with ``dirnames`` sorted
    in place and ``filenames`` iterated in sorted order:

    * Only entries for which ``os.path.islink()`` is False are descended into
      (``followlinks=False`` semantics). On Windows, CPython reports directory
      junctions as symlinks (``os.lstat`` -> ``S_IFLNK``), so junctions are not
      followed either; this relies on that CPython behaviour rather than on an
      explicit ``os.walk`` junction guarantee. The scan root itself is never
      checked: if the selected folder is itself a symlink or junction, its
      contents ARE scanned; only nested links are skipped.
      (Linux/POSIX behaviour is defined by ``os.path.islink`` directly.)
    * Ordering is deterministic: depth-first, directories lexicographically by
      name, files lexicographically by name within each directory.
    * No duplicate paths (each real path is yielded exactly once by ``os.walk``).
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"not a directory: {folder!r}")
    inputs: list = []
    unsupported = 0
    if recursive:
        for root, dirnames, filenames in os.walk(folder_path, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                entry_path = Path(root) / name
                if not entry_path.is_file():
                    continue
                if is_supported_input_file(name):
                    inputs.append(str(entry_path))
                else:
                    unsupported += 1
        return inputs, unsupported
    for entry in sorted(folder_path.iterdir()):
        if entry.is_file():
            if is_supported_input_file(entry.name):
                inputs.append(str(entry))
            else:
                unsupported += 1
    return inputs, unsupported


# ---------------------------------------------------------------------------
# Managed master ingestion: frozen evidence-source mapping (R3) + candidates
# ---------------------------------------------------------------------------
# Frozen R3 table: the ONLY approved header card(s) that may produce a candidate
# ``EvidenceFact`` for each ``ImportDeclaration`` field. Do not extend; an
# unapproved alias produces no candidate. Heuristic (filename/token/layout)
# inference never produces a candidate.
EVIDENCE_SOURCE_MAP: Mapping[str, Tuple[str, ...]] = {
    "exposure_s": ("EXPTIME", "EXPOSURE"),
    "temperature_c": ("CCD-TEMP", "TEMPCCD"),
    "temperature_setpoint_c": ("SET-TEMP",),
    "gain": ("GAIN", "EGAIN"),
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

# Fields whose candidate value is a float, and those kept as a string.
_NUMERIC_EVIDENCE_FIELDS = frozenset({"exposure_s", "temperature_c", "temperature_setpoint_c", "gain", "offset"})


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
                v1.EvidenceFact(field=field, value=value, origin_type="fits_header",
                                origin_field=origin, confirmed_by=None, version="1")
                for value, origin in zip(values, origins)
            )
        else:
            candidates[field] = v1.EvidenceFact(
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
                v1.EvidenceFact(field="binning", value=value, origin_type="fits_header",
                                origin_field=origin, confirmed_by=None, version="1")
                for value, origin in zip(bin_values, bin_origins)
            )
        else:
            candidates["binning"] = v1.EvidenceFact(
                field="binning", value=distinct[0], origin_type="fits_header",
                origin_field=bin_origins[0], confirmed_by=None, version="1",
            )

    return candidates, conflicts


def read_header_candidates(path: str, hdu=0):
    """Read FITS header cards (public source adapter) and detect candidates.

    Returns ``(candidates, conflicts)`` from :func:`detect_header_candidates`.
    """
    cards = v1.FilesystemSource().read_header(path, hdu=hdu)
    return detect_header_candidates([(c.keyword, c.value) for c in cards])


# ---------------------------------------------------------------------------
# IMAGETYP role candidate evidence + master admissibility (P7-M3B rework)
# ---------------------------------------------------------------------------
# Frozen IMAGETYP -> role map (objective 4). Only the explicit FITS IMAGETYP card
# produces a role candidate; filename/folder/directory layout never do.
IMAGETYP_ROLE_MAP: Mapping[str, str] = {
    "DARK": "dark",
    "BIAS": "bias",
    "FLAT": "flat",
    "DARKFLAT": "flat_dark",
}

_ROLE_TO_IMAGETYP: Mapping[str, str] = {v: k for k, v in IMAGETYP_ROLE_MAP.items()}

# Master-admissibility marker vocabulary (mirrors the additive master admission in
# ``zecalibrator.io.raw_decoder``; this module is stdlib-only so the vocabulary is
# duplicated here and must stay in sync).
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


def detect_role_conflict(selected_role, detected_role) -> Optional[Mapping]:
    """Return a structured role conflict, or ``None`` when none.

    A user-selected role that contradicts the detected FITS IMAGETYP role is a
    human conflict (never auto-resolved): selected role / detected role / source
    / needs confirmation.
    """
    if not selected_role or not detected_role:
        return None
    if selected_role == detected_role:
        return None
    imagetyp = _ROLE_TO_IMAGETYP.get(detected_role, str(detected_role).upper())
    return {
        "selected_role": selected_role,
        "detected_role": detected_role,
        "source": f"FITS IMAGETYP={imagetyp}",
        "needs_confirmation": True,
    }


def scan_master_header(path: str, selected_role: Optional[str] = None) -> Mapping:
    """Read one master header and produce a full scan entry.

    Combines evidence detection (unchanged), IMAGETYP role detection,
    role-conflict detection and the master-admissibility detector into one pure
    (no pixel load) summary. ``selected_role`` is an optional user-selected role;
    when absent the detected IMAGETYP role (if any) is used.
    """
    cards = v1.FilesystemSource().read_header(path, hdu=0)
    card_pairs = [(c.keyword, c.value) for c in cards]
    candidates, conflicts = detect_header_candidates(card_pairs)
    detected_role = detect_imagetyp_role(card_pairs)
    role = selected_role or detected_role
    conflict = detect_role_conflict(selected_role, detected_role)
    incompatible = master_incompatibility(card_pairs, role=role)
    return {
        "path": path,
        "selected_role": selected_role,
        "detected_role": detected_role,
        "role": role,
        "conflict": conflict,
        "incompatible": list(incompatible),
        "admissible": not incompatible,
        "status": "COMPLETED",
        "candidates": {f: fact.to_dict() for f, fact in candidates.items()},
        "conflicts": {f: [x.to_dict() for x in facts] for f, facts in conflicts.items()},
        # G2B additive ranking evidence (NOT an ImportDeclaration field; never
        # enters evidence_for_fields / USER_ONLY_FIELDS / build_declaration).
        "ranking": {"acquired_at": _scan_date_obs(card_pairs), "source": "fits_header"},
    }


def _scan_date_obs(card_pairs):
    """Return the DATE-OBS raw string for a master header scan (else None).

    HIERARCH-insensitive, single unambiguous value (same semantics as the
    source-boundary DATE-OBS reader).
    """
    values = []
    for kw, val in _iter_keyword_values(card_pairs):
        if kw == "DATE-OBS":
            if val is not None:
                values.append(val)
    distinct = []
    for v in values:
        if v not in distinct:
            distinct.append(v)
    return distinct[0] if len(distinct) == 1 else None


def evidence_for_fields(candidates: Mapping, confirmed_fields: Mapping[str, bool]) -> Mapping[str, "v1.EvidenceFact"]:
    """Return confirmed evidence facts from detected candidates.

    ``confirmed_fields`` maps field name -> True (confirmed by the user). Only
    confirmed candidate fields become persisted evidence (``confirmed_by="user"``);
    unconfirmed candidates stay absent (never persisted, never invented).
    """
    out: dict = {}
    for field, fact in candidates.items():
        if confirmed_fields.get(field):
            out[field] = v1.EvidenceFact(
                field=fact.field, value=fact.value, origin_type=fact.origin_type,
                origin_field=fact.origin_field, confirmed_by="user", version="1",
            )
    return out


# Fields that must come from user/native/measured provenance, never a header card
# (R3 "no candidate"). These are the fields the confirmation flow may still
# accept from explicit user/native sources only.
USER_ONLY_FIELDS = (
    "detector_instance_id", "sensor_dimensions", "orientation", "roi_origin",
    "optical_train_id", "bias_exposure_max_s", "saturation_limit_adu",
    "saturation_evidence",
)

# ---------------------------------------------------------------------------
# R3C field tiers: necessary (matching-blocking) vs disambiguator (UNVERIFIED).
# ---------------------------------------------------------------------------
# A missing *necessary* field blocks "ready" and reports "needs attention"; a
# missing *disambiguator* never does (R3B: non-blocking UNVERIFIED) and is never
# prompted to the user. ``orientation``/``roi_origin`` are necessary ONLY for a
# Bayer sensor (cfa_phase in GRBG/RGGB/BGGR/GBRG); for mono/unknown CFA they
# degrade to disambiguators. ``roi_extent`` is not a declaration field — on the
# managed path it is always the full array shape — so "full frame (no crop)"
# only ever sets ``roi_origin=(0,0)``.
_BAYER_CFA_PHASES = frozenset({"GRBG", "RGGB", "BGGR", "GBRG"})

# Shape-driven geometry (binning) + detector_model/gain/offset + per-type
# acquisition/optical facts + cfa_phase. CFA-only geometry (orientation/roi_origin)
# is added conditionally by :func:`necessary_fields`. ``filter`` is flat-only;
# ``exposure_s``/``temperature_c`` are dark/flat_dark-only (a bias has neither).
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

# CFA-only necessary geometry facts (required only for a Bayer sensor).
CFA_ONLY_FIELDS: Tuple[str, ...] = ("orientation", "roi_origin")

# Matching-relevant but non-blocking (UNVERIFIED) fields: never necessary, never
# prompted. ``optical_train_id`` is flat-only.
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


# Backward-compatible alias: "required" now means the necessary (blocking) tier.
REQUIRED_FIELDS_BY_MASTER_TYPE = NECESSARY_FIELDS_BY_MASTER_TYPE


def required_fields_for_master_type(master_type: str) -> Tuple[str, ...]:
    """Return the necessary (matching-blocking) field minimum for a master type."""
    return NECESSARY_FIELDS_BY_MASTER_TYPE.get(master_type, ())


# Flat quality evidence is never user-assertable (R4): normalization/validity/
# saturation/CFA-quality must come from machine-readable provenance or an
# authorized measurement. The v1 managed path has no such source, so a managed
# flat is always reported as needing quality evidence.
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


# Field value parsing for the bounded confirmation input.
_TUPLE_FIELDS = frozenset({"sensor_dimensions", "binning", "roi_origin"})
_FLOAT_FIELDS = frozenset({
    "gain", "offset", "temperature_c", "exposure_s", "bias_exposure_max_s",
    "saturation_limit_adu",
})


def parse_user_fact(field: str, text: str):
    """Parse a user-supplied fact string into the ImportDeclaration field's type.

    Empty text yields ``None`` (absent). Tuple fields accept two integers
    (``"4 4"`` or ``"4,4"`` → ``(4, 4)``); numeric fields accept finite floats;
    everything else is a stripped string. Raises ``ValueError`` on malformed
    input (never coerces silently).
    """
    text = (text or "").strip()
    if not text:
        return None
    if field in _TUPLE_FIELDS:
        parts = [p for p in text.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError(f"{field} requires two integers (y, x)")
        try:
            a, b = int(float(parts[0])), int(float(parts[1]))
        except ValueError as exc:
            raise ValueError(f"{field} requires two integers (y, x)") from exc
        return (a, b)
    if field in _FLOAT_FIELDS:
        v = float(text)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"{field} must be a finite number")
        return v
    return text


def _candidate_value(entry):
    """Return the value of a candidate entry (fact dict / fact object / raw)."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("value")
    return getattr(entry, "value", entry)


def missing_fields_for_master(master_type: str, candidates: Mapping) -> Tuple[str, ...]:
    """Return the NECESSARY fields not already detected from the header.

    Disambiguators are never returned (→ never prompted). CFA-only geometry
    facts (``orientation``/``roi_origin``) are returned only when the detected
    ``cfa_phase`` is a Bayer phase. ``candidates`` maps field -> value (or a
    fact dict / :class:`~zecalibrator.api.v1.EvidenceFact` carrying ``value``).
    """
    cfa_phase = _candidate_value(candidates.get("cfa_phase"))
    required = necessary_fields(master_type, cfa_phase)
    return tuple(f for f in required if f not in candidates)


# ---------------------------------------------------------------------------
# Human-readable confirmation (R3C Standard UX): never expose internal names.
# ---------------------------------------------------------------------------
# Human labels for the NECESSARY facts a normal user may still need to supply.
# Internal field names are never shown. Header-derived facts (INSTRUME/GAIN/
# OFFSET/CCD-TEMP/EXPTIME/XBINNING/YBINNING/BAYERPAT/FILTER) are auto-detected
# and therefore never reach this mapping for prompting.
HUMAN_FACT_LABELS: Mapping[str, str] = {
    "detector_model": "Camera / detector model",
    "gain": "Gain (e⁻/ADU)",
    "offset": "Offset / pedestal (ADU)",
    "exposure_s": "Exposure time (seconds)",
    "temperature_c": "Sensor temperature (°C)",
    "binning": "Binning (height × width)",
    "cfa_phase": "CFA / Bayer pattern",
    "filter": "Filter",
}

# CFA-only geometry confirmation: a bounded, explicit mapping from a clear
# user-facing choice to the internal geometry fact. Provenance = user
# confirmation (never guessed silently). ``roi_extent`` is not a declaration
# field — on the managed path it is always the full array shape — so "full
# frame (no crop)" maps to ``roi_origin=(0,0)``.
CFA_GEOMETRY_CONFIRMATIONS: Tuple[Tuple[str, str, str, object], ...] = (
    ("full_frame", "Full frame (no crop)", "roi_origin", (0, 0)),
    ("standard_orientation", "Standard orientation (not flipped)", "orientation", "identity"),
)


def human_fact_label(field: str) -> Optional[str]:
    """Return the human label for a necessary fact (or ``None``)."""
    return HUMAN_FACT_LABELS.get(field)


def cfa_geometry_confirmation(field: str) -> Optional[Tuple[str, str, object]]:
    """Return ``(id, human_label, value)`` for a CFA geometry field, or ``None``."""
    for cid, label, fld, value in CFA_GEOMETRY_CONFIRMATIONS:
        if fld == field:
            return (cid, label, value)
    return None


def build_declaration(source, identity, version, evidence: Mapping, extra: Optional[Mapping] = None) -> "v1.ImportDeclaration":
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
    return v1.ImportDeclaration(**kwargs)


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
) -> "v1.ManagedMasterRecord":
    """Build an immutable :class:`ManagedMasterRecord` (public value object)."""
    return v1.ManagedMasterRecord(
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


# ---------------------------------------------------------------------------
# Immutable request snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LightInput:
    """One selected light frame, frozen for a worker operation.

    ``declaration`` / ``roi_extent`` are public immutable value objects (or
    ``None`` when the user did not supply evidence — never synthesized).
    ``row_id`` is the GUI's stable per-row identity so a cached plan can never
    be bound to another row with the same path/HDU but different evidence.
    """

    path: str
    hdu: object  # int | str
    declaration: object  # ImportDeclaration | None
    roi_extent: object  # RoiExtentEvidence | None
    display_name: str
    row_id: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("LightInput.path must be non-empty")
        if not self.display_name:
            object.__setattr__(self, "display_name", Path(self.path).name)

    def to_source(self) -> "v1.FitsFrameSource":
        return v1.FitsFrameSource(
            path=self.path, hdu=self.hdu,
            declaration=self.declaration, roi_extent=self.roi_extent,
        )


@dataclass(frozen=True)
class OperationSnapshot:
    """An immutable snapshot of everything one worker operation needs.

    All scientific value objects carried here are frozen public API value
    objects; the snapshot itself is frozen, so the worker cannot observe a
    half-constructed request and the GUI cannot mutate an in-flight request.
    """

    op_id: str
    kind: str  # open_library | index_library | preflight | calibrate_in_memory | export | load_declaration | load_roi | scan_masters | load_ledger | confirm_evidence | build_managed_library | create_bpm_map | load_settings | save_settings | save_bpm_settings
    library_spec: Optional["v1.LibrarySpec"]
    request: Optional["v1.CalibrationRequest"]
    policy: Optional["v1.MatchPolicy"]
    lights: Tuple[LightInput, ...]
    plan: Optional["v1.CalibrationPlan"] = None
    destination: Optional[str] = None
    batch_id: Optional[str] = None
    index_imports: Tuple["v1.MasterImportSpec", ...] = ()
    imports_path: Optional[str] = None
    index_revision: Optional[str] = None
    evidence_path: Optional[str] = None  # declaration/ROI JSON path
    target_indices: Tuple[int, ...] = ()  # light rows a declaration/ROI applies to
    config_dir: Optional[str] = None  # settings root (load_settings/save_settings)
    settings_payload: Optional[dict] = None  # GuiSettings.to_dict() for save
    storage_paths: Optional[dict] = None  # 5 StoragePaths (str) for the BPM seam
    bpm_root: Optional[str] = None  # Bad Pixel Database root override (save_bpm_settings)
    bpm_detector_k: Optional[float] = None  # detector threshold K (create/save BPM)
    # managed master ingestion (P7-M3B)
    master_paths: Tuple[str, ...] = ()
    master_type: Optional[str] = None
    # Per-file scan targets ``(path, role)``; role is None to auto-detect from
    # IMAGETYP. When empty, the worker falls back to ``master_paths``/``master_type``.
    master_targets: Tuple[Tuple[str, Optional[str]], ...] = ()
    # Session selection ``(content_sha256, role)`` pairs for the derived index.
    # None = no session filter (legacy full-ledger build); () = explicit empty
    # selection (build nothing).
    session_selection: Optional[Tuple[Tuple[str, str], ...]] = None
    ledger_dir: Optional[str] = None  # data root for the managed ledger
    managed_spec: Optional["v1.LibrarySpec"] = None  # managed library index spec
    confirm_payload: Optional[dict] = None  # confirm_evidence inputs

    def __post_init__(self) -> None:
        object.__setattr__(self, "lights", tuple(self.lights))
        object.__setattr__(self, "index_imports", tuple(self.index_imports))
        object.__setattr__(self, "target_indices", tuple(self.target_indices))
        object.__setattr__(self, "master_paths", tuple(self.master_paths))
        object.__setattr__(self, "master_targets", tuple(self.master_targets))
        if self.session_selection is not None:
            object.__setattr__(self, "session_selection", tuple(self.session_selection))


# ---------------------------------------------------------------------------
# JSON -> public value objects (fail-closed; no facts invented)
# ---------------------------------------------------------------------------
def _require_object(obj, what: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{what} must be a JSON object")
    return obj


def parse_declaration(value) -> "v1.ImportDeclaration":
    """Build an :class:`ImportDeclaration` from a decoded JSON object.

    The public model validates source/identity/version/domain/units/geometry and
    raises a ``ValueError`` on any invalid/contrary fact (never a silent default).
    """
    obj = _require_object(value, "declaration")
    try:
        return v1.ImportDeclaration(**dict(obj))
    except TypeError as exc:
        raise ValueError(f"invalid declaration: {exc}") from exc


def parse_roi(value) -> "v1.RoiExtentEvidence":
    """Build a :class:`RoiExtentEvidence` from a decoded JSON object."""
    obj = _require_object(value, "roi-extent")
    try:
        return v1.RoiExtentEvidence(**dict(obj))
    except TypeError as exc:
        raise ValueError(f"invalid roi-extent: {exc}") from exc


def _build_scalars(value) -> "v1.NormalizationScalars":
    if not isinstance(value, dict):
        raise ValueError("normalization_scalars must be a JSON object")
    if set(value.keys()) == {"mono"}:
        return v1.NormalizationScalars(mono=float(value["mono"]))
    if set(value.keys()) == {"g1", "r", "b", "g2"}:
        return v1.NormalizationScalars(
            g1=float(value["g1"]), r=float(value["r"]),
            b=float(value["b"]), g2=float(value["g2"]),
        )
    raise ValueError("normalization_scalars must be {mono} or {g1,r,b,g2}")


def _build_norm_provenance(value) -> "v1.NormalizationProvenance":
    obj = _require_object(value, "normalization_provenance")
    return v1.NormalizationProvenance(
        algorithm=obj["algorithm"],
        population=obj["population"],
        scalars=_build_scalars(obj["scalars"]),
    )


def _build_validity(value) -> "v1.ValidityEvidence":
    obj = _require_object(value, "validity_evidence")
    return v1.ValidityEvidence(
        saturation_limit_known=obj.get("saturation_limit_known", True),
        valid_normalization_count=obj.get("valid_normalization_count"),
        total_normalization_count=obj.get("total_normalization_count"),
        quality_policy_state=obj.get("quality_policy_state", "qualified"),
        illumination=obj.get("illumination"),
        exposure_quality=obj.get("exposure_quality"),
    )


def _build_processing(value) -> "v1.ProcessingProvenance":
    obj = _require_object(value, "processing_provenance")
    norm = obj.get("normalization")
    return v1.ProcessingProvenance(
        source=obj["source"],
        additive_history_state=obj.get("additive_history_state", "unknown"),
        additive_correction_history=tuple(obj.get("additive_correction_history", ())),
        normalization=_build_norm_provenance(norm) if norm is not None else None,
        acquisition_profile=None,
    )


def _build_import_spec(value) -> "v1.MasterImportSpec":
    obj = _require_object(value, "import spec")
    if "declaration" not in obj:
        raise ValueError("import spec requires a 'declaration' object")
    declaration = v1.ImportDeclaration(**dict(obj["declaration"]))
    if "path" not in obj or "master_type" not in obj:
        raise ValueError("import spec requires 'path' and 'master_type'")
    kwargs: dict = {
        "path": obj["path"],
        "master_type": obj["master_type"],
        "declaration": declaration,
        "hdu": obj.get("hdu", 0),
        "mask_path": obj.get("mask_path"),
        "bias_state": obj.get("bias_state"),
        "flat_form": obj.get("flat_form"),
    }
    if obj.get("normalization_algorithm") is not None:
        kwargs["normalization_algorithm"] = obj["normalization_algorithm"]
    if obj.get("normalization_scalars") is not None:
        kwargs["normalization_scalars"] = _build_scalars(obj["normalization_scalars"])
    if obj.get("normalization_provenance") is not None:
        kwargs["normalization_provenance"] = _build_norm_provenance(obj["normalization_provenance"])
    if obj.get("validity_evidence") is not None:
        kwargs["validity_evidence"] = _build_validity(obj["validity_evidence"])
    if obj.get("processing_provenance") is not None:
        kwargs["processing_provenance"] = _build_processing(obj["processing_provenance"])
    return v1.MasterImportSpec(**kwargs)


def parse_imports(value) -> Tuple["v1.MasterImportSpec", ...]:
    """Build an ordered tuple of :class:`MasterImportSpec` from a decoded JSON array."""
    if not isinstance(value, list):
        raise ValueError("declarations must be a JSON array")
    return tuple(_build_import_spec(d) for d in value)


def load_json_object(path: str) -> dict:
    """Read and decode a small JSON file into a ``dict`` (used for evidence/imports)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path!r}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path!r} must decode to a JSON object")
    return obj


def load_json_array(path: str) -> list:
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path!r}: {exc}") from exc
    if not isinstance(obj, list):
        raise ValueError(f"{path!r} must decode to a JSON array")
    return obj


# ---------------------------------------------------------------------------
# JSON-safe rendering of public value objects (display only, never science)
# ---------------------------------------------------------------------------
def to_jsonable(value):
    """Render a public value object / container into strict-JSON-safe Python.

    Uses each object's public ``to_dict`` / ``to_mapping`` / dataclass fields;
    never assembles replacement scientific provenance.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "to_mapping"):
        return to_jsonable(value.to_mapping())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return str(value)


__all__ = [
    "CFA_GEOMETRY_CONFIRMATIONS",
    "CFA_ONLY_FIELDS",
    "DISAMBIGUATOR_FIELDS_BY_MASTER_TYPE",
    "EVIDENCE_SOURCE_MAP",
    "HUMAN_FACT_LABELS",
    "IMAGETYP_ROLE_MAP",
    "NECESSARY_FIELDS_BY_MASTER_TYPE",
    "REQUIRED_FIELDS_BY_MASTER_TYPE",
    "SUPPORTED_INPUT_EXTENSIONS",
    "USER_ONLY_FIELDS",
    "LightInput",
    "OperationSnapshot",
    "build_declaration",
    "cfa_geometry_confirmation",
    "dedup_input_paths",
    "detect_header_candidates",
    "detect_imagetyp_role",
    "detect_role_conflict",
    "evidence_for_fields",
    "human_fact_label",
    "input_dialog_filter",
    "is_bayer_phase",
    "is_supported_input_file",
    "load_json_array",
    "load_json_object",
    "make_managed_record",
    "master_evidence_status",
    "master_incompatibility",
    "missing_fields_for_master",
    "missing_required_fields",
    "necessary_fields",
    "new_batch_id",
    "new_operation_id",
    "parse_declaration",
    "parse_hdu",
    "parse_imports",
    "parse_roi",
    "parse_user_fact",
    "path_identity",
    "read_header_candidates",
    "required_fields_for_master_type",
    "scan_folder_inputs",
    "scan_master_header",
    "to_jsonable",
]
