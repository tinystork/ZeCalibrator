"""Immutable metadata value objects, FITS card evidence and import declarations.

The decoder preserves the *original ordered card sequence* — including
duplicate candidate cards — before any scaling/normalization. Metadata is deep
immutable: a frozen dataclass that holds an immutable ordered tuple of
:class:`CardRecord`, read-only mappings, and recursively frozen nested values
(numpy arrays become nested tuples, never writable copies).

An :class:`ImportDeclaration` supplies unknown facts with a visible
identity/source/version; it never overrides a *contrary* measured FITS card and
its own domain/units must be consistent with the frozen v1 raw-ADU light domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import numpy as np

from zecalibrator.core.geometry import CFA_PHASES, Geometry

# Confirmed v1 keyword -> normalized field map. This single map drives both
# alias-group resolution (EXPTIME/EXPOSURE -> exposure_seconds) and duplicate
# singleton/scaling conflict detection (BSCALE/BZERO/BLANK/GAIN/CFA/units, …).
#
# R3A: the mapping is now owned by a named :class:`FitsStandardAdapter` behind a
# small adapter registry (TASK-3). ``FIELD_BY_KEYWORD`` is retained as a
# read-only alias so existing consumers are unchanged. Later Siril/ZWO/NINA/
# SharpCap/PixInsight/Seestar/INDI/ASCOM adapters extend the *registry* without
# touching alias resolution or matching.
FIELD_BY_KEYWORD: Mapping[str, str] = MappingProxyType(
    {
        "EXPTIME": "exposure_seconds",
        "EXPOSURE": "exposure_seconds",
        "XBINNING": "bin_x",
        "CCDXBIN": "bin_x",
        "YBINNING": "bin_y",
        "CCDYBIN": "bin_y",
        "XORGSUBF": "roi_origin_x",
        "YORGSUBF": "roi_origin_y",
        "INSTRUME": "detector_model",
        "BAYERPAT": "cfa",
        "CCD-TEMP": "temperature_c",
        "GAIN": "gain",
        "FILTER": "filter",
        "BSCALE": "bscale",
        "BZERO": "bzero",
        "BLANK": "blank",
        "BUNIT": "bunit",
    }
)

# The historical numeric-keyword set (shared by the standard adapter and the
# retained NUMERIC_KEYWORDS alias).
_STANDARD_NUMERIC_KEYWORDS: tuple[str, ...] = (
    "BSCALE",
    "BZERO",
    "BLANK",
    "EXPTIME",
    "EXPOSURE",
    "XBINNING",
    "CCDXBIN",
    "YBINNING",
    "CCDYBIN",
    "XORGSUBF",
    "YORGSUBF",
    "GAIN",
    "CCD-TEMP",
)


@dataclass(frozen=True)
class FitsStandardAdapter:
    """The current (standard FITS) keyword→canonical adapter (TASK-3).

    A named adapter holding the existing keyword→canonical mappings plus the
    numeric-keyword set. It sits behind the module-level ``_ADAPTER_REGISTRY``;
    ``resolve_aliases`` consults the registry rather than ``FIELD_BY_KEYWORD``
    directly, so future vendor adapters (Siril/ZWO/NINA/SharpCap/PixInsight/
    Seestar/INDI/ASCOM) can be registered without touching matching.
    """

    name: str = "fits_standard"
    field_by_keyword: Mapping[str, str] = FIELD_BY_KEYWORD
    numeric_keywords: tuple[str, ...] = _STANDARD_NUMERIC_KEYWORDS

    def canonical(self, keyword: str) -> Optional[str]:
        """Return the canonical field for a keyword, or ``None`` (unmapped)."""
        return self.field_by_keyword.get(keyword)

    def is_numeric(self, keyword: str) -> bool:
        """Return whether ``keyword`` is a numeric card subject to malformed checks."""
        return keyword in self.numeric_keywords


_FITS_STANDARD_ADAPTER = FitsStandardAdapter()

# R3A extensible adapter registry (TASK-3). Only the standard adapter ships now.
_ADAPTER_REGISTRY: tuple[FitsStandardAdapter, ...] = (_FITS_STANDARD_ADAPTER,)

_HIERARCH_PREFIX = "HIERARCH "


def _strip_hierarch(keyword: str) -> str:
    """Return ``keyword`` with a leading ``HIERARCH `` prefix removed.

    HIERARCH-encoded standard keywords must normalize EXACTLY like their plain
    counterparts (REWORK-1 F1): the audit keeps the prefix, the adapter lookup
    ignores it.
    """
    if keyword.startswith(_HIERARCH_PREFIX):
        return keyword[len(_HIERARCH_PREFIX):]
    return keyword


def resolve_keyword(keyword: str) -> Optional[str]:
    """Consult the adapter registry for ``keyword``'s canonical field.

    Later adapters are consulted in registration order; the first mapping wins.
    A leading ``HIERARCH `` prefix is ignored for the lookup so HIERARCH-encoded
    standard keywords map like their plain counterparts (REWORK-1).
    """
    stripped = _strip_hierarch(keyword)
    for adapter in _ADAPTER_REGISTRY:
        field = adapter.canonical(stripped)
        if field is not None:
            return field
    return None


def numeric_keyword(keyword: str) -> bool:
    """Return whether ``keyword`` is a numeric card per the adapter registry.

    A leading ``HIERARCH `` prefix is ignored for the lookup (REWORK-1).
    """
    stripped = _strip_hierarch(keyword)
    return any(adapter.is_numeric(stripped) for adapter in _ADAPTER_REGISTRY)


NUMERIC_KEYWORDS: tuple[str, ...] = _STANDARD_NUMERIC_KEYWORDS


def _validate_int_tuple(value, name: str, min_value: int) -> None:
    """Validate an optional 2-tuple of integers with a lower bound (reason-coded)."""
    if value is None:
        return
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-tuple of integers, got {value!r}")
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must contain integers, got {value!r}")
        f = float(v)
        if not math.isfinite(f) or f != int(f) or int(f) < min_value:
            raise ValueError(f"{name} must be integers >= {min_value}, got {value!r}")


def _freeze(value):
    """Recursively coerce mutable containers into immutable structures.

    numpy arrays become nested tuples (never writable copies); mappings become
    read-only mappings; lists/tuples become tuples; numpy scalars become Python
    scalars.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class CardRecord:
    """One original header card, in its original position.

    ``value`` is the parsed value (``None`` when absent/empty); ``raw`` is the
    raw textual value as written. ``source`` records the actual selected HDU.
    """

    keyword: str
    value: object
    comment: str
    index: int
    source: str
    raw: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze(self.value))
        object.__setattr__(self, "comment", _freeze(self.comment))
        object.__setattr__(self, "source", _freeze(self.source))
        object.__setattr__(self, "raw", _freeze(self.raw))


@dataclass(frozen=True)
class ConflictDiagnostic:
    """A disagreement between duplicate/alias candidate cards (never resolved)."""

    field: str
    keywords: tuple[str, ...]
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _freeze(self.field))
        object.__setattr__(self, "keywords", _freeze(self.keywords))
        object.__setattr__(self, "values", _freeze(self.values))


@dataclass(frozen=True)
class FactProvenance:
    """Provenance of one canonical fact: ``(value, source, confidence)`` (TASK-1).

    ``source`` is the originating FITS card keyword(s) or ``"structural"`` for
    NAXIS-derived facts; ``confidence`` is ``"explicit"`` for a recognized card
    mapping or ``"structural"`` for NAXIS-derived plane shape.
    """

    value: object
    source: object  # str (single keyword) or tuple[str, ...] (alias group)
    confidence: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze(self.value))
        object.__setattr__(self, "source", _freeze(self.source))
        if self.confidence not in ("explicit", "structural"):
            raise ValueError(
                f"FactProvenance.confidence must be 'explicit' or 'structural', "
                f"got {self.confidence!r}"
            )


@dataclass(frozen=True)
class ImportDeclaration:
    """Evidence-backed metadata/profile/import declaration (immutable).

    Supplies *unknown* facts with a visible identity/source/version. It never
    overrides a contrary measured FITS card. ``domain``/``units`` must be
    consistent with the frozen v1 raw-ADU light domain.
    """

    source: str
    identity: str
    version: str
    domain: str = "raw"
    units: str = "ADU"
    detector_instance_id: Optional[str] = None
    detector_model: Optional[str] = None
    gain: Optional[float] = None
    offset: Optional[float] = None
    readout_mode: Optional[str] = None
    adc_mode: Optional[str] = None
    binning: Optional[tuple[int, int]] = None
    sensor_dimensions: Optional[tuple[int, int]] = None
    orientation: Optional[str] = None
    cfa_phase: Optional[str] = None
    roi_origin: Optional[tuple[int, int]] = None
    exposure_s: Optional[float] = None
    temperature_c: Optional[float] = None
    filter: Optional[str] = None
    optical_train_id: Optional[str] = None
    bias_exposure_max_s: Optional[float] = None
    short_flat_profile: bool = False
    saturation_limit_adu: Optional[float] = None
    saturation_evidence: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("ImportDeclaration.source must be a non-empty string")
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("ImportDeclaration.identity must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("ImportDeclaration.version must be a non-empty string")
        if self.domain != "raw":
            raise ValueError(
                f"ImportDeclaration.domain must be 'raw' (v1), got {self.domain!r}"
            )
        if self.units != "ADU":
            raise ValueError(
                f"ImportDeclaration.units must be 'ADU' for a raw light, got {self.units!r}"
            )
        # Numeric/tuple domain validation (rework-2 M2): reject degenerate-but-finite
        # values with reason-coded errors; never coerce or invent.
        _validate_int_tuple(self.sensor_dimensions, "sensor_dimensions", min_value=1)
        _validate_int_tuple(self.binning, "binning", min_value=1)
        _validate_int_tuple(self.roi_origin, "roi_origin", min_value=0)
        if self.orientation is not None and self.orientation != "identity":
            raise ValueError(f"ImportDeclaration.orientation must be 'identity' (v1), got {self.orientation!r}")
        if self.cfa_phase is not None and self.cfa_phase not in CFA_PHASES:
            raise ValueError(
                f"ImportDeclaration.cfa_phase must be one of {CFA_PHASES}, got {self.cfa_phase!r}"
            )
        for name in ("binning", "sensor_dimensions", "roi_origin"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "detector_instance_id", _freeze(self.detector_instance_id))
        object.__setattr__(self, "detector_model", _freeze(self.detector_model))
        object.__setattr__(self, "readout_mode", _freeze(self.readout_mode))
        object.__setattr__(self, "adc_mode", _freeze(self.adc_mode))
        object.__setattr__(self, "orientation", _freeze(self.orientation))
        object.__setattr__(self, "cfa_phase", _freeze(self.cfa_phase))
        object.__setattr__(self, "filter", _freeze(self.filter))
        object.__setattr__(self, "optical_train_id", _freeze(self.optical_train_id))


@dataclass(frozen=True)
class SensorMetadata:
    """Immutable decoded sensor metadata (G3 subset)."""

    original_cards: tuple[CardRecord, ...]
    normalized: Mapping[str, object]
    conflicts: tuple[ConflictDiagnostic, ...]
    geometry: Geometry
    raw_domain_declaration: str  # "raw" | "processed" | "unknown"
    units: Optional[str] = None  # "ADU" | "dimensionless" | None
    declaration: Optional[ImportDeclaration] = None
    exposure_s: Optional[float] = None
    temperature_c: Optional[float] = None
    gain: Optional[float] = None
    offset: Optional[float] = None
    readout_mode: Optional[str] = None
    adc_mode: Optional[str] = None
    filter: Optional[str] = None
    detector_model: Optional[str] = None
    detector_instance_id: Optional[str] = None
    optical_train_id: Optional[str] = None
    saturation_limit_adu: Optional[float] = None
    saturation_evidence: str = "unknown"
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, FactProvenance] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_cards", _freeze(self.original_cards))
        object.__setattr__(self, "normalized", _freeze(self.normalized))
        object.__setattr__(self, "conflicts", _freeze(self.conflicts))
        object.__setattr__(self, "warnings", _freeze(self.warnings))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def cfa_phase(self) -> Optional[str]:
        return self.geometry.cfa_phase


def _canonical(value):
    """Return a hashable canonical form for numeric-aware equality."""
    if value is None:
        return None
    if isinstance(value, bool):
        return ("num", float(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ("num", float(value))
    if isinstance(value, str):
        s = value.strip()
        try:
            return ("num", float(s))
        except ValueError:
            if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                s = s[1:-1].strip()
            return ("str", s)
    return ("other", str(value))


def _parse_number(raw):
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return raw
    return raw


def collect_cards(header, source: str = "primary") -> tuple[CardRecord, ...]:
    """Extract an ordered, duplicate-preserving ``CardRecord`` sequence."""
    records: list[CardRecord] = []
    for index, card in enumerate(header.cards):
        raw = getattr(card, "rawvalue", None)
        # Preserve the full HIERARCH prefix so the collected audit keeps
        # unknown/vendor/HIERARCH evidence intact (R3A TASK-4a). astropy's
        # ``card.keyword`` strips the ``HIERARCH `` prefix; the card image
        # (or the ``_hierarch`` flag) is the only carrier of that prefix.
        keyword = card.keyword
        if getattr(card, "_hierarch", False):
            keyword = f"HIERARCH {keyword}"
        records.append(
            CardRecord(
                keyword=keyword,
                value=card.value,
                comment=getattr(card, "comment", "") or "",
                index=index,
                source=source,
                raw=None if raw is None else str(raw),
            )
        )
    return tuple(records)


def resolve_aliases_with_provenance(
    cards: tuple[CardRecord, ...],
) -> tuple[
    Mapping[str, object],
    tuple[ConflictDiagnostic, ...],
    tuple[str, ...],
    Mapping[str, FactProvenance],
]:
    """Resolve keyword/alias groups and carry per-fact provenance (TASK-1).

    Returns ``(normalized, conflicts, malformed, provenance)``. ``normalized``
    is byte-for-byte the historical flat mapping (unchanged values);
    ``provenance`` is the additive parallel structure mapping each *agreed*
    canonical field to a :class:`FactProvenance` ``(value, source, confidence)``.
    Conflicted fields are absent from both ``normalized`` and ``provenance``.
    """
    by_field: dict[str, list[tuple[str, object]]] = {}
    for c in cards:
        f = resolve_keyword(c.keyword)
        if f is None:
            continue
        by_field.setdefault(f, []).append((c.keyword, c.value))

    normalized: dict[str, object] = {}
    provenance: dict[str, FactProvenance] = {}
    conflicts: list[ConflictDiagnostic] = []
    for fld, present in by_field.items():
        canon = [_canonical(v) for _, v in present]
        first = canon[0]
        if any(ca != first for ca in canon):
            conflicts.append(
                ConflictDiagnostic(
                    field=fld,
                    keywords=tuple(kw for kw, _ in present),
                    values=tuple(v for _, v in present),
                )
            )
        else:
            value = _parse_number(present[0][1])
            normalized[fld] = value
            keywords = tuple(kw for kw, _ in present)
            provenance[fld] = FactProvenance(
                value=value,
                source=keywords[0] if len(keywords) == 1 else keywords,
                confidence="explicit",
            )

    malformed: list[str] = []
    for c in cards:
        kw = _strip_hierarch(c.keyword)
        if numeric_keyword(kw) and isinstance(c.value, str) and c.value.strip():
            if isinstance(_parse_number(c.value), str):
                malformed.append(kw)

    return MappingProxyType(normalized), tuple(conflicts), tuple(malformed), MappingProxyType(provenance)


def resolve_aliases(
    cards: tuple[CardRecord, ...],
) -> tuple[Mapping[str, object], tuple[ConflictDiagnostic, ...], tuple[str, ...]]:
    """Resolve all critical keyword/alias groups, detecting conflicts.

    Backward-compatible wrapper over :func:`resolve_aliases_with_provenance`;
    provenance is discarded to preserve the historical 3-tuple contract.
    """
    normalized, conflicts, malformed, _ = resolve_aliases_with_provenance(cards)
    return normalized, conflicts, malformed


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    return None


def _str_or_none(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _int_tuple(value) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (ValueError, TypeError):
            return None
    return None


def _cfa_norm(value) -> Optional[str]:
    s = _str_or_none(value)
    if s is None:
        return None
    u = s.upper()
    if u == "MONO":
        return "mono"
    return u


def _resolve_fact(
    normalized: Mapping[str, object],
    fits_field: str,
    declaration: Optional[ImportDeclaration],
    decl_field: str,
    conflicts: list,
    coerce,
) -> object:
    fits_v = normalized.get(fits_field)
    decl_v = getattr(declaration, decl_field, None) if declaration is not None else None
    if fits_v is not None and decl_v is not None:
        if _canonical(fits_v) != _canonical(decl_v):
            conflicts.append(
                ConflictDiagnostic(
                    field=decl_field,
                    keywords=(fits_field, f"declaration.{decl_field}"),
                    values=(fits_v, decl_v),
                )
            )
            return coerce(fits_v)
    if fits_v is not None:
        return coerce(fits_v)
    return coerce(decl_v)


def build_sensor_metadata(
    *,
    shape: tuple[int, int],
    normalized: Mapping[str, object],
    conflicts: tuple[ConflictDiagnostic, ...],
    cards: tuple[CardRecord, ...],
    declaration: Optional[ImportDeclaration] = None,
    units: str = "ADU",
    provenance: Optional[Mapping[str, FactProvenance]] = None,
) -> SensorMetadata:
    """Merge FITS-derived facts and an import declaration into immutable metadata.

    FITS-vs-declaration contradictions (exposure/temperature/gain/filter/
    detector/CFA/binning/ROI) append to ``conflicts`` — never silently resolved.
    Geometry is resolved with no invented defaults (unknown stays ``None``).

    ``provenance`` is the additive parallel provenance mapping (TASK-1); the
    decoded plane is always recorded as ``actual_plane_shape`` with
    ``source="structural"`` / ``confidence="structural"`` (TASK-2), distinct
    from ``sensor_dimensions`` which remains declaration-only.
    """
    extra: list[ConflictDiagnostic] = list(conflicts)

    merged_provenance: dict[str, FactProvenance] = dict(provenance or {})
    merged_provenance.setdefault(
        "actual_plane_shape",
        FactProvenance(
            value=tuple(int(v) for v in shape),
            source="structural",
            confidence="structural",
        ),
    )

    def resolve(fits_field, decl_field, coerce=_float_or_none):
        return _resolve_fact(normalized, fits_field, declaration, decl_field, extra, coerce)

    exposure_s = resolve("exposure_seconds", "exposure_s")
    temperature_c = resolve("temperature_c", "temperature_c")
    gain = resolve("gain", "gain")
    filter_ = resolve("filter", "filter", _str_or_none)
    detector_model = resolve("detector_model", "detector_model", _str_or_none)

    # CFA: FITS BAYERPAT vs declaration.cfa_phase.
    fits_cfa = _cfa_norm(normalized.get("cfa"))
    decl_cfa = _cfa_norm(getattr(declaration, "cfa_phase", None)) if declaration else None
    cfa_phase = fits_cfa
    if fits_cfa is not None and decl_cfa is not None and fits_cfa != decl_cfa:
        extra.append(
            ConflictDiagnostic("cfa_phase", ("BAYERPAT", "declaration.cfa_phase"), (fits_cfa, decl_cfa))
        )
    if cfa_phase is None:
        cfa_phase = decl_cfa

    # Binning: FITS bin_x/bin_y vs declaration.binning.
    bin_x = _float_or_none(normalized.get("bin_x"))
    bin_y = _float_or_none(normalized.get("bin_y"))
    fits_binning = None
    if bin_x is not None and bin_y is not None:
        fits_binning = (int(bin_y), int(bin_x))
    decl_binning = _int_tuple(getattr(declaration, "binning", None)) if declaration else None
    binning = fits_binning
    if fits_binning is not None and decl_binning is not None and fits_binning != decl_binning:
        extra.append(
            ConflictDiagnostic("binning", ("XBINNING/YBINNING", "declaration.binning"), (fits_binning, decl_binning))
        )
    if binning is None:
        binning = decl_binning

    # ROI origin: FITS XORGSUBF/YORGSUBF vs declaration.roi_origin.
    ox = _float_or_none(normalized.get("roi_origin_x"))
    oy = _float_or_none(normalized.get("roi_origin_y"))
    fits_roi = None
    if ox is not None and oy is not None:
        fits_roi = (int(oy), int(ox))
    decl_roi = _int_tuple(getattr(declaration, "roi_origin", None)) if declaration else None
    roi_origin = fits_roi
    if fits_roi is not None and decl_roi is not None and fits_roi != decl_roi:
        extra.append(
            ConflictDiagnostic("roi_origin", ("XORGSUBF/YORGSUBF", "declaration.roi_origin"), (fits_roi, decl_roi))
        )
    if roi_origin is None:
        roi_origin = decl_roi

    offset = _float_or_none(getattr(declaration, "offset", None)) if declaration else None
    readout_mode = _str_or_none(getattr(declaration, "readout_mode", None)) if declaration else None
    adc_mode = _str_or_none(getattr(declaration, "adc_mode", None)) if declaration else None
    detector_instance_id = _str_or_none(getattr(declaration, "detector_instance_id", None)) if declaration else None
    optical_train_id = _str_or_none(getattr(declaration, "optical_train_id", None)) if declaration else None
    saturation_limit_adu = _float_or_none(getattr(declaration, "saturation_limit_adu", None)) if declaration else None
    saturation_evidence = (declaration.saturation_evidence if declaration else "unknown")
    sensor_dimensions = _int_tuple(getattr(declaration, "sensor_dimensions", None)) if declaration else None
    orientation = _str_or_none(getattr(declaration, "orientation", None)) if declaration else None

    geometry = Geometry(
        shape=shape,
        sensor_dimensions=sensor_dimensions,
        binning=binning,
        roi_origin=roi_origin,
        roi_extent=None,
        orientation=orientation,
        cfa_phase=cfa_phase,
    )

    return SensorMetadata(
        original_cards=cards,
        normalized=normalized,
        conflicts=tuple(extra),
        geometry=geometry,
        raw_domain_declaration="raw",
        units=units,
        declaration=declaration,
        exposure_s=exposure_s,
        temperature_c=temperature_c,
        gain=gain,
        offset=offset,
        readout_mode=readout_mode,
        adc_mode=adc_mode,
        filter=filter_,
        detector_model=detector_model,
        detector_instance_id=detector_instance_id,
        optical_train_id=optical_train_id,
        saturation_limit_adu=saturation_limit_adu,
        saturation_evidence=saturation_evidence,
        provenance=merged_provenance,
    )


__all__ = [
    "FIELD_BY_KEYWORD",
    "FactProvenance",
    "FitsStandardAdapter",
    "CardRecord",
    "ConflictDiagnostic",
    "ImportDeclaration",
    "NUMERIC_KEYWORDS",
    "SensorMetadata",
    "build_sensor_metadata",
    "collect_cards",
    "resolve_aliases",
    "resolve_aliases_with_provenance",
]
