"""Deep-immutable scientific master descriptors and metadata/profile snapshots.

This module defines the matching-relevant identity value objects for the G4
library/matching layer (ARCHITECTURE §3.3, PROVENANCE §2.3). No paths, pixels,
timestamps or retrieval locators are part of the *identity*; those live on the
retrieval-facing :class:`~zecalibrator.core.plans.MasterBinding` and library
candidate records.

``descriptor_id`` is **derived** (``SHA-256(CANONICAL_JSON(projection))`` over
the frozen descriptor projection) and never trusted from a caller-supplied hash.

Matching-relevant *profile* facts (the qualified bias acquisition range and the
short-flat declaration) are **not** free-floating ``Acquisition`` extras: they are
retained inside :class:`AcquisitionProfileEvidence` which is carried by
:class:`ProcessingProvenance` (already part of the frozen descriptor projection),
so tampering with them changes the verified ``descriptor_id``. The G1 projection
allowlist is therefore unchanged; no new digest definition is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields as _dc_fields
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import numpy as np

from zecalibrator.core.digests import descriptor_digest
from zecalibrator.core.geometry import CFA_PHASES, Geometry
from zecalibrator.core.metadata import ImportDeclaration

MASTER_TYPES: tuple[str, ...] = ("bias", "dark", "flat", "flat_dark")
PIXEL_DOMAINS: tuple[str, ...] = ("sensor_adu", "normalized_response")
PHYSICAL_UNITS: tuple[str, ...] = ("ADU", "dimensionless")
BIAS_STATES: tuple[str, ...] = ("included", "removed", "not_applicable", "unknown")
FLAT_FORMS: tuple[str, ...] = ("raw_response", "corrected_unnormalized", "normalized_response")
PROCESSING_SOURCES: tuple[str, ...] = ("synthetic_fixture", "user_import", "observed")
# Discriminated additive-processing-history state (P7-M3B R3D-A D1d).
# ``unknown`` means "no information about whether additive corrections were
# applied"; ``known`` means the history is authoritative (possibly empty).
ADDITIVE_HISTORY_STATES: tuple[str, ...] = ("unknown", "known")
# Discriminated DQ state (P7-M3B, R1). ``no_source_dq`` is a structural state:
# ``mask_identity=None`` and no source-mask payload; never a fabricated hex64.
DQ_STATES: tuple[str, ...] = ("source_mask", "no_source_dq")
_BAYER_PHASES: tuple[str, ...] = ("GRBG", "RGGB", "BGGR", "GBRG")
_CFA_SCALAR_KEYS: tuple[str, ...] = ("g1", "r", "b", "g2")

ACQUISITION_PROFILE_SCHEMA = "zecalibrator.acquisition_profile.v1"


class DescriptorIntegrityError(ValueError):
    """A recomputed descriptor digest does not match the recorded ``descriptor_id``."""


class DescriptorSchemaError(ValueError):
    """An unknown/unsupported key was present in a serialized descriptor snapshot."""


def _freeze(value):
    """Recursively coerce mutable containers into immutable structures."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _require_hex64(value, name: str) -> None:
    if not _is_hex64(value):
        raise ValueError(f"{name} must be a 64-character lowercase hex string, got {value!r}")


def _reject_unknown_keys(d: Mapping, allowed: frozenset, path: str) -> None:
    unknown = set(d.keys()) - allowed
    if unknown:
        raise DescriptorSchemaError(f"unknown key(s) at {path}: {sorted(unknown)}")


# ---------------------------------------------------------------------------
# Structured evidence value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalizationScalars:
    """Discriminated normalization-scalar value (ARCHITECTURE §3.3)."""

    mono: Optional[float] = None
    g1: Optional[float] = None
    r: Optional[float] = None
    b: Optional[float] = None
    g2: Optional[float] = None

    def __post_init__(self) -> None:
        mono_set = self.mono is not None
        cfa = (self.g1, self.r, self.b, self.g2)
        cfa_set = all(v is not None for v in cfa)
        cfa_any = any(v is not None for v in cfa)
        if mono_set and cfa_any:
            raise ValueError("NormalizationScalars must be mono OR CFA, not both")
        if not mono_set and cfa_any and not cfa_set:
            raise ValueError("CFA NormalizationScalars requires all four labelled scalars (g1/r/b/g2)")
        if not mono_set and not cfa_set:
            raise ValueError("NormalizationScalars requires mono or the four CFA scalars")
        for name, v in (("mono", self.mono), ("g1", self.g1), ("r", self.r), ("b", self.b), ("g2", self.g2)):
            if v is None:
                continue
            f = float(v)
            if not math.isfinite(f):
                raise ValueError(f"normalization scalar {name!r} must be finite")
            object.__setattr__(self, name, f)

    @property
    def kind(self) -> str:
        return "mono" if self.mono is not None else "cfa"

    def to_mapping(self) -> Mapping[str, float]:
        if self.kind == "mono":
            return {"mono": self.mono}
        return {"g1": self.g1, "r": self.r, "b": self.b, "g2": self.g2}


@dataclass(frozen=True)
class NormalizationProvenance:
    """Normalization source/population/scalars evidence (ARCHITECTURE §3.3)."""

    algorithm: str
    population: str
    scalars: NormalizationScalars

    def __post_init__(self) -> None:
        if not self.algorithm or not self.population:
            raise ValueError("NormalizationProvenance requires non-empty algorithm/population")
        object.__setattr__(self, "algorithm", _freeze(self.algorithm))
        object.__setattr__(self, "population", _freeze(self.population))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "algorithm": self.algorithm,
            "population": self.population,
            "scalars": dict(self.scalars.to_mapping()),
        }
@dataclass(frozen=True)
class AcquisitionProfileEvidence:
    """Explicit evidence-backed acquisition profile facts (G3-inherited).

    Carries the qualified near-zero bias acquisition range and the short-flat
    declaration with explicit schema/source/identity/version. Retained inside
    :class:`ProcessingProvenance` so it is part of the hashed descriptor identity.
    """

    schema_version: str = ACQUISITION_PROFILE_SCHEMA
    source: str = "synthetic_fixture"
    identity: str = ""
    version: str = ""
    bias_exposure_max_s: Optional[float] = None
    short_flat_profile: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != ACQUISITION_PROFILE_SCHEMA:
            raise ValueError(f"unsupported acquisition profile schema {self.schema_version!r}")
        for name in ("source", "identity", "version"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"AcquisitionProfileEvidence.{name} must be a non-empty string")
            object.__setattr__(self, name, _freeze(v))
        if self.bias_exposure_max_s is not None:
            f = float(self.bias_exposure_max_s)
            if not math.isfinite(f) or f < 0:
                raise ValueError(f"bias_exposure_max_s must be finite and non-negative, got {self.bias_exposure_max_s!r}")
            object.__setattr__(self, "bias_exposure_max_s", f)
        object.__setattr__(self, "short_flat_profile", bool(self.short_flat_profile))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "identity": self.identity,
            "version": self.version,
            "bias_exposure_max_s": self.bias_exposure_max_s,
            "short_flat_profile": self.short_flat_profile,
        }
@dataclass(frozen=True)
class ProcessingProvenance:
    """Structured processing evidence describing what was done before import.

    ``additive_history_state`` is the explicit UNKNOWN vs KNOWN discriminator
    (P7-M3B R3D-A D1d): ``"unknown"`` means no information about whether
    additive corrections were applied (history MUST be empty); ``"known"``
    means the history is authoritative (empty or a validated tuple of
    non-empty strings). The two are never conflated.
    """

    source: str
    additive_history_state: str = "unknown"
    additive_correction_history: tuple[str, ...] = ()
    normalization: Optional[NormalizationProvenance] = None
    acquisition_profile: Optional[AcquisitionProfileEvidence] = None

    def __post_init__(self) -> None:
        if self.source not in PROCESSING_SOURCES:
            raise ValueError(f"processing source must be one of {PROCESSING_SOURCES}, got {self.source!r}")
        if self.additive_history_state not in ADDITIVE_HISTORY_STATES:
            raise ValueError(
                f"additive_history_state must be one of {ADDITIVE_HISTORY_STATES}, got {self.additive_history_state!r}"
            )
        hist = _freeze(self.additive_correction_history)
        # HARD INVARIANT (never normalized silently): ``unknown`` requires an
        # empty history; a non-empty history with ``unknown`` is rejected at
        # construction AND deserialization.
        if self.additive_history_state == "unknown":
            if hist:
                raise ValueError(
                    "additive_history_state='unknown' requires an empty "
                    f"additive_correction_history, got {hist!r}"
                )
        else:  # known
            for entry in hist:
                if not isinstance(entry, str) or not entry.strip():
                    raise ValueError(
                        "additive_correction_history entries must be non-empty strings, "
                        f"got {entry!r}"
                    )
        object.__setattr__(self, "source", _freeze(self.source))
        object.__setattr__(self, "additive_history_state", _freeze(self.additive_history_state))
        object.__setattr__(self, "additive_correction_history", hist)

    def to_dict(self) -> Mapping[str, object]:
        norm = self.normalization.to_dict() if self.normalization is not None else None
        profile = self.acquisition_profile.to_dict() if self.acquisition_profile is not None else None
        return {
            "source": self.source,
            "additive_history_state": self.additive_history_state,
            "additive_correction_history": list(self.additive_correction_history),
            "normalization": dict(norm) if norm is not None else None,
            "acquisition_profile": dict(profile) if profile is not None else None,
        }
@dataclass(frozen=True)
class ValidityEvidence:
    """Flat validity evidence (ARCHITECTURE §3.3 + per-plane population counts).

    ``illumination`` and ``exposure_quality`` are explicit applicable optical/
    exposure declarations for flats (SCIENCE §6.2: illumination mode + exposure
    quality are flat-vs-light criteria); ``None`` means missing, which matching
    refuses. No all-time validity or universal exposure range is invented.
    """

    saturation_limit_known: bool
    valid_normalization_count: Optional[Mapping[str, int]] = None
    total_normalization_count: Optional[Mapping[str, int]] = None
    quality_policy_state: str = "qualified"
    illumination: Optional[str] = None
    exposure_quality: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quality_policy_state not in ("pending", "qualified"):
            raise ValueError(f"quality_policy_state must be 'pending' or 'qualified', got {self.quality_policy_state!r}")
        object.__setattr__(self, "saturation_limit_known", bool(self.saturation_limit_known))
        object.__setattr__(self, "valid_normalization_count", _freeze(self.valid_normalization_count))
        object.__setattr__(self, "total_normalization_count", _freeze(self.total_normalization_count))
        object.__setattr__(self, "quality_policy_state", _freeze(self.quality_policy_state))
        object.__setattr__(self, "illumination", _freeze(self.illumination))
        object.__setattr__(self, "exposure_quality", _freeze(self.exposure_quality))

    def to_dict(self) -> Mapping[str, object]:
        vc = self.valid_normalization_count
        tc = self.total_normalization_count
        return {
            "saturation_limit_known": self.saturation_limit_known,
            "valid_normalization_count": dict(vc) if vc is not None else None,
            "total_normalization_count": dict(tc) if tc is not None else None,
            "quality_policy_state": self.quality_policy_state,
            "illumination": self.illumination,
            "exposure_quality": self.exposure_quality,
        }
@dataclass(frozen=True)
class DetectorIdentity:
    """Detector identity (ARCHITECTURE §3.3)."""

    detector_instance_id: str
    detector_model: Optional[str] = None
    serial: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.detector_instance_id, str) or self.detector_instance_id == "":
            raise ValueError("detector_instance_id must be a non-empty string")
        object.__setattr__(self, "detector_instance_id", _freeze(self.detector_instance_id))
        object.__setattr__(self, "detector_model", _freeze(self.detector_model))
        object.__setattr__(self, "serial", _freeze(self.serial))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "detector_instance_id": self.detector_instance_id,
            "detector_model": self.detector_model,
            "serial": self.serial,
        }
@dataclass(frozen=True)
class Acquisition:
    """Acquisition facts (ARCHITECTURE §3.3).

    ``bias_exposure_max_s`` (qualified near-zero bias range) and
    ``short_flat_profile`` are matching-relevant *light* acquisition facts folded
    into this projected object so they participate in the plan identity (the
    ``light_constraints.acquisition`` whole-subtree projection path). For MASTER
    descriptors these stay ``None``/``False``; the master-side equivalents live
    inside the hashed ``processing_provenance.acquisition_profile`` subtree.
    """

    gain: Optional[float] = None
    offset: Optional[float] = None
    readout_mode: Optional[str] = None
    adc_mode: Optional[str] = None
    temperature_c: Optional[float] = None
    exposure_s: Optional[float] = None
    saturation_limit_adu: Optional[float] = None
    saturation_evidence: str = "unknown"
    bias_exposure_max_s: Optional[float] = None
    short_flat_profile: bool = False

    def __post_init__(self) -> None:
        if self.saturation_evidence not in ("qualified", "unknown"):
            raise ValueError(f"saturation_evidence must be 'qualified' or 'unknown', got {self.saturation_evidence!r}")
        for name in ("gain", "offset", "temperature_c", "exposure_s", "saturation_limit_adu", "bias_exposure_max_s"):
            v = getattr(self, name)
            if v is not None:
                f = float(v)
                if not math.isfinite(f):
                    raise ValueError(f"Acquisition.{name} must be finite, got {v!r}")
                object.__setattr__(self, name, f)
        for name in ("readout_mode", "adc_mode"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "saturation_evidence", _freeze(self.saturation_evidence))
        object.__setattr__(self, "short_flat_profile", bool(self.short_flat_profile))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "gain": self.gain,
            "offset": self.offset,
            "readout_mode": self.readout_mode,
            "adc_mode": self.adc_mode,
            "temperature_c": self.temperature_c,
            "exposure_s": self.exposure_s,
            "saturation_limit_adu": self.saturation_limit_adu,
            "saturation_evidence": self.saturation_evidence,
            "bias_exposure_max_s": self.bias_exposure_max_s,
            "short_flat_profile": self.short_flat_profile,
        }
@dataclass(frozen=True)
class OpticalIdentity:
    """Optical identity: filter + optical train (each known or unknown)."""

    filter: Optional[str] = None
    optical_train_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "filter", _freeze(self.filter))
        object.__setattr__(self, "optical_train_id", _freeze(self.optical_train_id))

    def to_dict(self) -> Mapping[str, object]:
        return {"filter": self.filter, "optical_train_id": self.optical_train_id}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _geometry_to_dict(geo: Geometry) -> Mapping[str, object]:
    def tup(v):
        return None if v is None else list(v)

    return {
        "shape": list(geo.shape),
        "sensor_dimensions": tup(geo.sensor_dimensions),
        "binning": tup(geo.binning),
        "roi_origin": tup(geo.roi_origin),
        "roi_extent": tup(geo.roi_extent),
        "orientation": geo.orientation,
        "cfa_phase": geo.cfa_phase,
    }
def _geometry_from_dict(d: Mapping[str, object]) -> Geometry:
    def tup(v):
        return None if v is None else (int(v[0]), int(v[1]))

    return Geometry(
        shape=(int(d["shape"][0]), int(d["shape"][1])),
        sensor_dimensions=tup(d.get("sensor_dimensions")),
        binning=tup(d.get("binning")),
        roi_origin=tup(d.get("roi_origin")),
        roi_extent=tup(d.get("roi_extent")),
        orientation=d.get("orientation"),
        cfa_phase=d.get("cfa_phase"),
    )


# ---------------------------------------------------------------------------
# Light constraints + evidence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LightEvidence:
    """Preserved original input evidence (audit, not part of plan identity)."""

    original_cards: tuple = ()
    declaration: Optional[ImportDeclaration] = None
    units: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_cards", _freeze(self.original_cards))
        object.__setattr__(self, "units", _freeze(self.units))

    def to_dict(self) -> Mapping[str, object]:
        decl = _declaration_to_dict(self.declaration) if self.declaration is not None else None
        return {
            "original_cards": [dict(c.__dict__) for c in self.original_cards] if self.original_cards else [],
            "declaration": decl,
            "units": self.units,
        }
@dataclass(frozen=True)
class LightConstraints:
    """Frozen, validated light-side matching constraints (ARCHITECTURE §3.3).

    Matching-relevant light facts (including the qualified bias range and
    short-flat profile) live on :class:`Acquisition`, which is projected as a
    whole subtree in the plan digest. ``evidence`` is audit-only (original cards/
    declaration/units) and is excluded from plan identity by design.
    """

    geometry: Geometry
    detector: DetectorIdentity
    acquisition: Acquisition
    optical: OpticalIdentity
    raw_domain_declaration: str = "raw"
    evidence: Optional[LightEvidence] = None

    def __post_init__(self) -> None:
        if self.raw_domain_declaration != "raw":
            raise ValueError(f"raw_domain_declaration must be 'raw' (v1), got {self.raw_domain_declaration!r}")

    def to_plan_dict(self) -> Mapping[str, object]:
        return {
            "geometry": dict(_geometry_to_dict(self.geometry)),
            "detector": dict(self.detector.to_dict()),
            "acquisition": dict(self.acquisition.to_dict()),
            "optical": dict(self.optical.to_dict()),
            "raw_domain_declaration": self.raw_domain_declaration,
        }
    def to_full_dict(self) -> Mapping[str, object]:
        d = dict(self.to_plan_dict())
        d["evidence"] = dict(self.evidence.to_dict()) if self.evidence is not None else None
        return d


# ---------------------------------------------------------------------------
# Master descriptor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MasterDescriptor:
    """Immutable scientific master descriptor (ARCHITECTURE §3.3)."""

    master_type: str
    pixel_domain: str
    physical_units: str
    bias_state: str
    geometry: Geometry
    detector: DetectorIdentity
    acquisition: Acquisition
    content_sha256: str
    size_bytes: int
    hdu: object  # int | str
    mask_identity: Optional[str]
    processing_provenance: ProcessingProvenance
    validity_evidence: ValidityEvidence
    flat_form: Optional[str] = None
    normalization_algorithm: Optional[str] = None
    normalization_scalars: Optional[NormalizationScalars] = None
    optical_train_id: Optional[str] = None
    filter: Optional[str] = None
    dq_state: str = "source_mask"
    descriptor_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dq_state not in DQ_STATES:
            raise ValueError(f"dq_state must be one of {DQ_STATES}, got {self.dq_state!r}")
        if self.master_type not in MASTER_TYPES:
            raise ValueError(f"master_type must be one of {MASTER_TYPES}, got {self.master_type!r}")
        if self.pixel_domain not in PIXEL_DOMAINS:
            raise ValueError(f"pixel_domain must be one of {PIXEL_DOMAINS}, got {self.pixel_domain!r}")
        if self.physical_units not in PHYSICAL_UNITS:
            raise ValueError(f"physical_units must be one of {PHYSICAL_UNITS}, got {self.physical_units!r}")
        if self.bias_state not in BIAS_STATES:
            raise ValueError(f"bias_state must be one of {BIAS_STATES}, got {self.bias_state!r}")

        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError(f"size_bytes must be a non-negative integer, got {self.size_bytes!r}")
        if isinstance(self.hdu, bool) or not isinstance(self.hdu, (int, str)) or (isinstance(self.hdu, int) and self.hdu < 0):
            raise ValueError(f"hdu must be a non-negative int or str, got {self.hdu!r}")
        _require_hex64(self.content_sha256, "content_sha256")
        # R1: mask_identity is nullable; None iff dq_state == "no_source_dq" (structural,
        # never a fabricated hex64 that could be mistaken for a real content hash).
        if self.dq_state == "no_source_dq":
            if self.mask_identity is not None:
                raise ValueError("no_source_dq master requires mask_identity=None")
        else:
            _require_hex64(self.mask_identity, "mask_identity")

        if self.master_type == "flat":
            if self.flat_form not in FLAT_FORMS:
                raise ValueError(f"flat master requires flat_form in {FLAT_FORMS}, got {self.flat_form!r}")
            if self.flat_form == "normalized_response":
                if self.pixel_domain != "normalized_response" or self.physical_units != "dimensionless":
                    raise ValueError("normalized_response flat requires normalized_response pixel_domain and dimensionless units")
                if not self.normalization_algorithm:
                    raise ValueError("normalized_response flat requires a normalization_algorithm")
                if self.normalization_scalars is None:
                    raise ValueError("normalized_response flat requires normalization_scalars")
            else:
                if self.pixel_domain != "sensor_adu" or self.physical_units != "ADU":
                    raise ValueError(f"{self.flat_form} flat requires sensor_adu pixel_domain and ADU units")
        else:
            if self.flat_form is not None:
                raise ValueError(f"flat_form must be None for master_type {self.master_type!r}")
            if self.normalization_algorithm is not None or self.normalization_scalars is not None:
                raise ValueError(f"normalization_algorithm/scalars must be None for master_type {self.master_type!r}")

        object.__setattr__(self, "content_sha256", _freeze(self.content_sha256))
        object.__setattr__(self, "mask_identity", _freeze(self.mask_identity))
        object.__setattr__(self, "dq_state", _freeze(self.dq_state))
        object.__setattr__(self, "flat_form", _freeze(self.flat_form))
        object.__setattr__(self, "normalization_algorithm", _freeze(self.normalization_algorithm))
        object.__setattr__(self, "optical_train_id", _freeze(self.optical_train_id))
        object.__setattr__(self, "filter", _freeze(self.filter))

        object.__setattr__(self, "descriptor_id", descriptor_digest(self.projection_dict()))

    @property
    def optical(self) -> OpticalIdentity:
        return OpticalIdentity(filter=self.filter, optical_train_id=self.optical_train_id)

    @property
    def cfa_phase(self) -> Optional[str]:
        return self.geometry.cfa_phase

    @property
    def bias_exposure_max_s(self) -> Optional[float]:
        p = self.processing_provenance.acquisition_profile
        return p.bias_exposure_max_s if p is not None else None

    @property
    def short_flat_profile(self) -> bool:
        p = self.processing_provenance.acquisition_profile
        return p.short_flat_profile if p is not None else False

    def projection_dict(self) -> Mapping[str, object]:
        """The explicit descriptor projection (PROVENANCE §2.3 item 2)."""
        norm = self.normalization_scalars.to_mapping() if self.normalization_scalars is not None else None
        return {
            "master_type": self.master_type,
            "bias_state": self.bias_state,
            "pixel_domain": self.pixel_domain,
            "physical_units": self.physical_units,
            "flat_form": self.flat_form,
            "normalization_algorithm": self.normalization_algorithm,
            "normalization_scalars": dict(norm) if norm is not None else None,
            "geometry": dict(_geometry_to_dict(self.geometry)),
            "detector": dict(self.detector.to_dict()),
            "acquisition": dict(self.acquisition.to_dict()),
            "optical_train_id": self.optical_train_id,
            "filter": self.filter,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "hdu": self.hdu,
            "mask_identity": self.mask_identity,
            "dq_state": self.dq_state,
            "processing_provenance": dict(self.processing_provenance.to_dict()),
            "validity_evidence": dict(self.validity_evidence.to_dict()),
        }
    def recompute_descriptor_id(self) -> str:
        return descriptor_digest(self.projection_dict())

    def verify_descriptor_id(self) -> None:
        recomputed = self.recompute_descriptor_id()
        if recomputed != self.descriptor_id:
            raise DescriptorIntegrityError(
                f"descriptor digest mismatch: recorded {self.descriptor_id!r}, recomputed {recomputed!r}"
            )

    def to_dict(self) -> Mapping[str, object]:
        """Full acyclic descriptor snapshot (for serialization/replay)."""
        return dict(self.projection_dict()) | {"descriptor_id": self.descriptor_id}


# ---------------------------------------------------------------------------
# ImportDeclaration (G3 type) serialization helpers — not modifying G3
# ---------------------------------------------------------------------------
def _declaration_to_dict(decl: ImportDeclaration) -> Mapping[str, object]:
    return {
        f.name: (list(getattr(decl, f.name)) if isinstance(getattr(decl, f.name), tuple) else getattr(decl, f.name))
        for f in _dc_fields(ImportDeclaration)
    }


def _declaration_from_dict(d: Mapping[str, object]) -> ImportDeclaration:
    kwargs = {}
    for f in _dc_fields(ImportDeclaration):
        v = d.get(f.name)
        if isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    return ImportDeclaration(**kwargs)


def _scalars_from_value(value) -> Optional[NormalizationScalars]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        m = {k: float(v) for k, v in value.items()}
        if set(m.keys()) == {"mono"}:
            return NormalizationScalars(mono=m["mono"])
        if set(m.keys()) == set(_CFA_SCALAR_KEYS):
            return NormalizationScalars(g1=m["g1"], r=m["r"], b=m["b"], g2=m["g2"])
        if len(m) == 1:
            return NormalizationScalars(mono=float(next(iter(m.values()))))
        raise ValueError(f"unsupported normalization_scalars mapping: {value!r}")
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return NormalizationScalars(mono=float(value[0]))
        if len(value) == 4:
            return NormalizationScalars(g1=float(value[0]), r=float(value[1]), b=float(value[2]), g2=float(value[3]))
        raise ValueError(f"normalization_scalars list must have 1 or 4 elements, got {len(value)}")
    raise ValueError(f"unsupported normalization_scalars value: {value!r}")


def _norm_from_dict(value) -> Optional[NormalizationProvenance]:
    if value is None:
        return None
    _reject_unknown_keys(value, frozenset({"algorithm", "population", "scalars"}), "normalization")
    return NormalizationProvenance(
        algorithm=value["algorithm"],
        population=value["population"],
        scalars=_scalars_from_value(value["scalars"]),
    )


def _profile_from_dict(value) -> Optional[AcquisitionProfileEvidence]:
    if value is None:
        return None
    _reject_unknown_keys(value, frozenset({"schema_version", "source", "identity", "version", "bias_exposure_max_s", "short_flat_profile"}), "acquisition_profile")
    return AcquisitionProfileEvidence(
        schema_version=value.get("schema_version", ACQUISITION_PROFILE_SCHEMA),
        source=value.get("source", "synthetic_fixture"),
        identity=value.get("identity", ""),
        version=value.get("version", ""),
        bias_exposure_max_s=value.get("bias_exposure_max_s"),
        short_flat_profile=value.get("short_flat_profile", False),
    )


def _processing_from_dict(value) -> ProcessingProvenance:
    _reject_unknown_keys(value, frozenset({"source", "additive_history_state", "additive_correction_history", "normalization", "acquisition_profile"}), "processing_provenance")
    # Legacy records without ``additive_history_state`` read back as "unknown"
    # (never a fabricated known-empty); a legacy non-empty history with no state
    # therefore fails the invariant at reconstruction (NO compatibility bypass).
    return ProcessingProvenance(
        source=value["source"],
        additive_history_state=value.get("additive_history_state", "unknown"),
        additive_correction_history=tuple(value.get("additive_correction_history", ())),
        normalization=_norm_from_dict(value.get("normalization")),
        acquisition_profile=_profile_from_dict(value.get("acquisition_profile")),
    )


def _validity_from_dict(value) -> ValidityEvidence:
    _reject_unknown_keys(value, frozenset({"saturation_limit_known", "valid_normalization_count", "total_normalization_count", "quality_policy_state", "illumination", "exposure_quality"}), "validity_evidence")
    return ValidityEvidence(
        saturation_limit_known=value["saturation_limit_known"],
        valid_normalization_count=value.get("valid_normalization_count"),
        total_normalization_count=value.get("total_normalization_count"),
        quality_policy_state=value.get("quality_policy_state", "qualified"),
        illumination=value.get("illumination"),
        exposure_quality=value.get("exposure_quality"),
    )


def _acquisition_from_dict(value) -> Acquisition:
    _reject_unknown_keys(value, frozenset({"gain", "offset", "readout_mode", "adc_mode", "temperature_c", "exposure_s", "saturation_limit_adu", "saturation_evidence", "bias_exposure_max_s", "short_flat_profile"}), "acquisition")
    return Acquisition(
        gain=value.get("gain"),
        offset=value.get("offset"),
        readout_mode=value.get("readout_mode"),
        adc_mode=value.get("adc_mode"),
        temperature_c=value.get("temperature_c"),
        exposure_s=value.get("exposure_s"),
        saturation_limit_adu=value.get("saturation_limit_adu"),
        saturation_evidence=value.get("saturation_evidence", "unknown"),
        bias_exposure_max_s=value.get("bias_exposure_max_s"),
        short_flat_profile=value.get("short_flat_profile", False),
    )


def _detector_from_dict(value) -> DetectorIdentity:
    _reject_unknown_keys(value, frozenset({"detector_instance_id", "detector_model", "serial"}), "detector")
    return DetectorIdentity(
        detector_instance_id=value["detector_instance_id"],
        detector_model=value.get("detector_model"),
        serial=value.get("serial"),
    )


def _geometry_keys() -> frozenset:
    return frozenset({"shape", "sensor_dimensions", "binning", "roi_origin", "roi_extent", "orientation", "cfa_phase"})


_DESCRIPTOR_TOP_KEYS = frozenset({
    "master_type", "bias_state", "pixel_domain", "physical_units", "flat_form",
    "normalization_algorithm", "normalization_scalars", "geometry", "detector",
    "acquisition", "optical_train_id", "filter", "content_sha256", "size_bytes",
    "hdu", "mask_identity", "dq_state", "processing_provenance", "validity_evidence", "descriptor_id",
})


def master_descriptor_from_dict(d: Mapping[str, object]) -> MasterDescriptor:
    """Reconstruct a ``MasterDescriptor`` from a serialized snapshot.

    Fail-closed: unknown top-level or nested keys raise
    :class:`DescriptorSchemaError`; a recorded ``descriptor_id`` that does not
    match the recomputed digest raises :class:`DescriptorIntegrityError`.
    """
    _reject_unknown_keys(d, _DESCRIPTOR_TOP_KEYS, "descriptor")
    _reject_unknown_keys(d["geometry"], _geometry_keys(), "geometry")

    desc = MasterDescriptor(
        master_type=d["master_type"],
        bias_state=d["bias_state"],
        pixel_domain=d["pixel_domain"],
        physical_units=d["physical_units"],
        flat_form=d.get("flat_form"),
        normalization_algorithm=d.get("normalization_algorithm"),
        normalization_scalars=_scalars_from_value(d.get("normalization_scalars")),
        geometry=_geometry_from_dict(d["geometry"]),
        detector=_detector_from_dict(d["detector"]),
        acquisition=_acquisition_from_dict(d["acquisition"]),
        optical_train_id=d.get("optical_train_id"),
        filter=d.get("filter"),
        content_sha256=d["content_sha256"],
        size_bytes=d["size_bytes"],
        hdu=d["hdu"],
        mask_identity=d["mask_identity"],
        dq_state=d.get("dq_state", "source_mask"),
        processing_provenance=_processing_from_dict(d["processing_provenance"]),
        validity_evidence=_validity_from_dict(d["validity_evidence"]),
    )
    if d.get("descriptor_id") is not None:
        if d["descriptor_id"] != desc.descriptor_id:
            raise DescriptorIntegrityError(
                f"descriptor digest mismatch: recorded {d['descriptor_id']!r}, recomputed {desc.descriptor_id!r}"
            )
    return desc


@dataclass(frozen=True)
class DescriptorSnapshot:
    """Immutable complete descriptor record + import declarations (F2)."""

    descriptor: MasterDescriptor
    declarations: tuple[ImportDeclaration, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "declarations", _freeze(self.declarations))

    @property
    def descriptor_id(self) -> str:
        return self.descriptor.descriptor_id

    def verify(self) -> None:
        """Reject a tampered snapshot (recompute the descriptor digest)."""
        self.descriptor.verify_descriptor_id()

    def to_dict(self) -> Mapping[str, object]:
        return {
            "descriptor": dict(self.descriptor.to_dict()),
            "declarations": [_declaration_to_dict(decl) for decl in self.declarations],
        }
    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "DescriptorSnapshot":
        _reject_unknown_keys(d, frozenset({"descriptor", "declarations"}), "descriptor_snapshot")
        desc = master_descriptor_from_dict(d["descriptor"])
        decls = tuple(_declaration_from_dict(x) for x in d.get("declarations", ()))
        return cls(descriptor=desc, declarations=decls)


__all__ = [
    "ACQUISITION_PROFILE_SCHEMA",
    "ADDITIVE_HISTORY_STATES",
    "Acquisition",
    "AcquisitionProfileEvidence",
    "BIAS_STATES",
    "DescriptorIntegrityError",
    "DescriptorSchemaError",
    "DescriptorSnapshot",
    "DetectorIdentity",
    "DQ_STATES",
    "FLAT_FORMS",
    "LightConstraints",
    "LightEvidence",
    "MASTER_TYPES",
    "MasterDescriptor",
    "NormalizationProvenance",
    "NormalizationScalars",
    "OpticalIdentity",
    "PHYSICAL_UNITS",
    "PIXEL_DOMAINS",
    "ProcessingProvenance",
    "ValidityEvidence",
    "master_descriptor_from_dict",
]
