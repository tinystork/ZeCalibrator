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


def scan_folder_inputs(folder: str) -> Tuple[list, int]:
    """Scan ``folder`` top-level only and return ``(input_paths, unsupported)``.

    - **Top-level only**: subdirectories are never recursed into, and are not
      counted as unsupported (they are not files).
    - ``input_paths`` are the top-level files whose suffix matches the shared
      supported input extension set (case-insensitive), returned in sorted
      order for determinism.
    - ``unsupported`` counts top-level *files* whose suffix does not match the
      supported set (a genuine candidate FITS is never silently dropped).
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"not a directory: {folder!r}")
    inputs: list = []
    unsupported = 0
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
_NUMERIC_EVIDENCE_FIELDS = frozenset({"exposure_s", "temperature_c", "gain", "offset"})


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

# Matching-relevant field minimum per master type (prepared §5). ``filter`` and
# ``optical_train_id`` are flat-only; ``exposure_s``/``temperature_c`` are not
# needed for bias; ``bias_exposure_max_s`` is bias/flat-dependency-relevant.
REQUIRED_FIELDS_BY_MASTER_TYPE: Mapping[str, Tuple[str, ...]] = {
    "dark": (
        "detector_instance_id", "detector_model", "gain", "offset", "readout_mode",
        "adc_mode", "temperature_c", "exposure_s", "sensor_dimensions", "binning",
        "orientation", "roi_origin", "cfa_phase",
    ),
    "bias": (
        "detector_instance_id", "detector_model", "gain", "offset", "readout_mode",
        "adc_mode", "sensor_dimensions", "binning", "orientation", "roi_origin",
        "cfa_phase", "bias_exposure_max_s",
    ),
    "flat": (
        "detector_instance_id", "detector_model", "gain", "offset", "readout_mode",
        "adc_mode", "temperature_c", "exposure_s", "sensor_dimensions", "binning",
        "orientation", "roi_origin", "cfa_phase", "filter", "optical_train_id",
    ),
    "flat_dark": (
        "detector_instance_id", "detector_model", "gain", "offset", "readout_mode",
        "adc_mode", "temperature_c", "exposure_s", "sensor_dimensions", "binning",
        "orientation", "roi_origin", "cfa_phase",
    ),
}


def required_fields_for_master_type(master_type: str) -> Tuple[str, ...]:
    """Return the matching-relevant field minimum for a master type (or empty)."""
    return REQUIRED_FIELDS_BY_MASTER_TYPE.get(master_type, ())


# Flat quality evidence is never user-assertable (R4): normalization/validity/
# saturation/CFA-quality must come from machine-readable provenance or an
# authorized measurement. The v1 managed path has no such source, so a managed
# flat is always reported as needing quality evidence.
_FLAT_QUALITY_REASON = "flat quality evidence (normalization/validity/saturation) insufficient"


def missing_required_fields(master_type: str, declaration) -> Tuple[str, ...]:
    """Return the required-field names whose declaration value is ``None``.

    Uses the frozen per-master-type minimum; absent facts are never invented.
    """
    required = REQUIRED_FIELDS_BY_MASTER_TYPE.get(master_type, ())
    return tuple(
        f for f in required if getattr(declaration, f, None) is None
    )


def master_evidence_status(master_type: str, declaration) -> Tuple[str, Tuple[str, ...]]:
    """Return ``(status, reasons)`` for a master's evidence completeness.

    ``status`` is ``"ready"`` when every matching-relevant required field is
    present and (for flats) machine-readable quality evidence exists; otherwise
    ``"needs_attention"`` with the concrete missing/insufficient reasons.

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


def missing_fields_for_master(master_type: str, candidates: Mapping) -> Tuple[str, ...]:
    """Return the required fields not already detected from the header.

    Used by the confirmation UI to know which facts must still be supplied by the
    user (``candidates`` holds the header-detected fields).
    """
    required = REQUIRED_FIELDS_BY_MASTER_TYPE.get(master_type, ())
    return tuple(f for f in required if f not in candidates)


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
    kind: str  # open_library | index_library | preflight | calibrate_in_memory | export | load_declaration | load_roi | scan_masters | load_ledger | confirm_evidence | build_managed_library
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
    # managed master ingestion (P7-M3B)
    master_paths: Tuple[str, ...] = ()
    master_type: Optional[str] = None
    ledger_dir: Optional[str] = None  # data root for the managed ledger
    managed_spec: Optional["v1.LibrarySpec"] = None  # managed library index spec
    confirm_payload: Optional[dict] = None  # confirm_evidence inputs

    def __post_init__(self) -> None:
        object.__setattr__(self, "lights", tuple(self.lights))
        object.__setattr__(self, "index_imports", tuple(self.index_imports))
        object.__setattr__(self, "target_indices", tuple(self.target_indices))
        object.__setattr__(self, "master_paths", tuple(self.master_paths))


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
    "EVIDENCE_SOURCE_MAP",
    "REQUIRED_FIELDS_BY_MASTER_TYPE",
    "SUPPORTED_INPUT_EXTENSIONS",
    "USER_ONLY_FIELDS",
    "LightInput",
    "OperationSnapshot",
    "build_declaration",
    "dedup_input_paths",
    "detect_header_candidates",
    "evidence_for_fields",
    "input_dialog_filter",
    "is_supported_input_file",
    "load_json_array",
    "load_json_object",
    "make_managed_record",
    "master_evidence_status",
    "missing_fields_for_master",
    "missing_required_fields",
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
    "to_jsonable",
]
