"""Immutable match policy, request/plan/bindings and decision provenance.

This module defines the *plan* layer (ARCHITECTURE §3.3): an immutable
``MatchPolicy`` with actual policy parameters (fixed v1 values, never widened),
the explicit ``CalibrationRequest`` (no optional-role flag), and the
``CalibrationPlan`` whose ``plan_id`` is derived over the frozen plan projection
(PROVENANCE §2.3) and which supports strict full serialization/deserialization
(with snapshots, declarations and locators) replayable after library closure.

Bindings here are **descriptor snapshots**, not the G3
``application.executor.MasterBinding`` (which owns decoded pixels). Retrieval
locators are excluded from the plan digest; the three hashes
(``descriptor_id``, ``content_sha256``, ``mask_identity``) are distinct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from zecalibrator.core.descriptors import (
    DescriptorSnapshot,
    LightConstraints,
    MasterDescriptor,
    master_descriptor_from_dict,
)
from zecalibrator.core.digests import plan_digest

MATCH_POLICY_VERSION = "zecalibrator.match.v1"
# R3D-A D1d: ProcessingProvenance gained the explicit additive-history-state
# discriminator, a breaking change to the provenance projection (and therefore
# to ``descriptor_id``/``plan_id``). Bumped v1 -> v2; old data is NOT silently
# reinterpreted (see PROVENANCE.md §9).
PROVENANCE_SCHEMA_VERSION = "zecalibrator.provenance.v2"
LIBRARY_SCHEMA_VERSION = "zecalibrator.library.v1"
DIGEST_SCHEMA_VERSION = "zecalibrator.digest.v1"
SCIENCE_CONTRACT_VERSION = "1.0"
PLAN_SERIALIZATION_SCHEMA = "zecalibrator.plan.v1"

ADDITIVE_MODES: tuple[str, ...] = ("control", "bias_only", "dark_incl_bias", "dark_bias_removed")
FLAT_MODES: tuple[str, ...] = ("none", "apply")

# v1 fixed policy: zero scientific temperature tolerance + parser 1e-6 exactly;
# exposure engineering bound absolute AND relative exactly 1e-6. Any wider value
# is a scientific window and is refused (a versioned qualified profile is out of
# scope).
_TEMP_SCIENTIFIC_C = 0.0
_TEMP_PARSER_C = 1e-6
_EXPOSURE_ABS = 1e-6
_EXPOSURE_REL = 1e-6
FLAT_QUALITY_THRESHOLD = 90.0


class PolicyError(ValueError):
    """A widened/non-finite/custom policy was supplied."""


@dataclass(frozen=True)
class Tolerance:
    """Engineering tolerance bound (SCIENCE §6.3)."""

    relative: float
    absolute: float

    def __post_init__(self) -> None:
        for name in ("relative", "absolute"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"Tolerance.{name} must be finite and non-negative, got {v!r}")
            object.__setattr__(self, name, v)

    def to_dict(self) -> Mapping[str, float]:
        return {"relative": self.relative, "absolute": self.absolute}


@dataclass(frozen=True)
class FlatQualityPolicy:
    """Discriminated flat-quality policy (never an implicit ``None`` default).

    Only the adopted 90%-per-plane qualified policy is representable; a
    ``pending`` policy is representable as a refusal state; any other qualified
    threshold/per_plane is refused (versioned profile qualification out of scope).
    """

    state: str
    threshold_pct: Optional[float] = None
    per_plane: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.state == "qualified":
            if self.threshold_pct != FLAT_QUALITY_THRESHOLD or self.per_plane is not True:
                raise PolicyError(
                    f"only the adopted {FLAT_QUALITY_THRESHOLD}% per-plane qualified policy is allowed "
                    "(versioned profile qualification is out of scope)"
                )
            object.__setattr__(self, "threshold_pct", float(self.threshold_pct))
            object.__setattr__(self, "per_plane", True)
        elif self.state == "pending":
            object.__setattr__(self, "threshold_pct", None)
            object.__setattr__(self, "per_plane", None)
        else:
            raise ValueError(f"flat quality state must be 'qualified' or 'pending', got {self.state!r}")

    def to_dict(self) -> Mapping[str, object]:
        if self.state == "qualified":
            return {"state": "qualified", "threshold_pct": self.threshold_pct, "per_plane": self.per_plane}
        return {"state": "pending"}


@dataclass(frozen=True)
class MatchPolicy:
    """Immutable matching policy with actual parameter values.

    The v1 policy is fixed here: temperature relative 0 + absolute 1e-6 exactly,
    exposure absolute AND relative 1e-6 exactly. Wider/non-finite values are
    refused; a pending flat policy is representable and refused at matching.
    """

    version: str
    exposure_tolerance: Tolerance
    temperature_tolerance: Tolerance
    flat_quality_policy: FlatQualityPolicy

    def __post_init__(self) -> None:
        if self.version != MATCH_POLICY_VERSION:
            raise PolicyError(f"MatchPolicy.version must be {MATCH_POLICY_VERSION!r}, got {self.version!r}")
        if self.temperature_tolerance.relative != _TEMP_SCIENTIFIC_C or self.temperature_tolerance.absolute != _TEMP_PARSER_C:
            raise PolicyError(
                "v1 temperature policy requires relative=0 (zero scientific) and absolute=1e-6 (parser); "
                f"got relative={self.temperature_tolerance.relative!r}, absolute={self.temperature_tolerance.absolute!r}"
            )
        if self.exposure_tolerance.absolute != _EXPOSURE_ABS or self.exposure_tolerance.relative != _EXPOSURE_REL:
            raise PolicyError(
                "v1 exposure policy requires absolute=1e-6 and relative=1e-6; "
                f"got absolute={self.exposure_tolerance.absolute!r}, relative={self.exposure_tolerance.relative!r}"
            )

    def to_dict(self) -> Mapping[str, object]:
        return {
            "version": self.version,
            "exposure_tolerance": dict(self.exposure_tolerance.to_dict()),
            "temperature_tolerance": dict(self.temperature_tolerance.to_dict()),
            "flat_quality_policy": dict(self.flat_quality_policy.to_dict()),
        }
def default_match_policy() -> MatchPolicy:
    """The adopted v1 default policy (owner decision 14.3/14.4)."""
    return MatchPolicy(
        version=MATCH_POLICY_VERSION,
        exposure_tolerance=Tolerance(relative=_EXPOSURE_REL, absolute=_EXPOSURE_ABS),
        temperature_tolerance=Tolerance(relative=_TEMP_SCIENTIFIC_C, absolute=_TEMP_PARSER_C),
        flat_quality_policy=FlatQualityPolicy(state="qualified", threshold_pct=FLAT_QUALITY_THRESHOLD, per_plane=True),
    )


@dataclass(frozen=True)
class CalibrationRequest:
    """Explicit requested roles; ``required_roles`` is derived from the modes."""

    additive_mode: str
    flat_mode: str = "none"

    def __post_init__(self) -> None:
        if self.additive_mode not in ADDITIVE_MODES:
            raise ValueError(f"unsupported additive_mode: {self.additive_mode!r}")
        if self.flat_mode not in FLAT_MODES:
            raise ValueError(f"unsupported flat_mode: {self.flat_mode!r}")

    @property
    def required_roles(self) -> tuple[str, ...]:
        roles: list[str] = []
        if self.additive_mode == "bias_only":
            roles.append("bias")
        elif self.additive_mode == "dark_incl_bias":
            roles.append("dark")
        elif self.additive_mode == "dark_bias_removed":
            roles.extend(["dark", "bias"])
        if self.flat_mode == "apply":
            roles.append("flat")
        return tuple(roles)

    def to_dict(self) -> Mapping[str, str]:
        return {"additive_mode": self.additive_mode, "flat_mode": self.flat_mode}


@dataclass(frozen=True)
class PolicyParameters:
    """Actual policy parameter values frozen into a plan."""

    exposure_tolerance: Tolerance
    temperature_tolerance: Tolerance
    flat_quality_policy: FlatQualityPolicy

    def to_dict(self) -> Mapping[str, object]:
        return {
            "exposure_tolerance": dict(self.exposure_tolerance.to_dict()),
            "temperature_tolerance": dict(self.temperature_tolerance.to_dict()),
            "flat_quality_policy": dict(self.flat_quality_policy.to_dict()),
        }
@dataclass(frozen=True)
class VersionSet:
    """Distinct version strings governing a plan (ARCHITECTURE §2)."""

    science_contract: str = SCIENCE_CONTRACT_VERSION
    decoder: str = "1.0"
    provenance_schema: str = PROVENANCE_SCHEMA_VERSION
    matching_policy: str = MATCH_POLICY_VERSION

    def to_dict(self) -> Mapping[str, str]:
        return {
            "science_contract": self.science_contract,
            "decoder": self.decoder,
            "provenance_schema": self.provenance_schema,
            "matching_policy": self.matching_policy,
        }
@dataclass(frozen=True)
class FitsFileLocator:
    """Narrow immutable retrieval locator for local FITS bytes (not identity)."""

    path: str
    hdu: object  # int | str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("FitsFileLocator.path must be a non-empty string")
        if isinstance(self.hdu, bool) or not isinstance(self.hdu, (int, str)):
            raise ValueError("FitsFileLocator.hdu must be int or str")
        object.__setattr__(self, "path", self.path)
        object.__setattr__(self, "hdu", self.hdu)

    def to_dict(self) -> Mapping[str, object]:
        return {"path": self.path, "hdu": self.hdu}


@dataclass(frozen=True)
class MaskPayloadLocator:
    """Narrow immutable retrieval locator for an external DQ/mask payload."""

    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("MaskPayloadLocator.path must be a non-empty string")
        object.__setattr__(self, "path", self.path)

    def to_dict(self) -> Mapping[str, object]:
        return {"path": self.path}


@dataclass(frozen=True)
class MasterBinding:
    """A plan binding: hashed science identity + retrieval locators (ARCHITECTURE §3.3).

    Construction validates that the descriptor snapshot is exactly coherent with
    the recorded identity fields (``descriptor_id``/``content_sha256``/
    ``size_bytes``/``hdu``/``mask_identity``); a mismatched snapshot raises.

    ``acquired_at`` is a retrieval-facing DATE-OBS string (ranking evidence only);
    it is deliberately NOT part of ``identity_dict`` and therefore never enters
    the plan digest.
    """

    descriptor_id: str
    descriptor_snapshot: DescriptorSnapshot
    content_sha256: str
    size_bytes: int
    hdu: object
    mask_identity: Optional[str]
    locators: tuple[FitsFileLocator, ...] = ()
    mask_locator: Optional[MaskPayloadLocator] = None
    acquired_at: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "locators", tuple(self.locators))
        d = self.descriptor_snapshot.descriptor
        if d.descriptor_id != self.descriptor_id:
            raise ValueError("MasterBinding.descriptor_snapshot descriptor_id does not match descriptor_id")
        if d.content_sha256 != self.content_sha256:
            raise ValueError("MasterBinding.descriptor_snapshot content_sha256 does not match content_sha256")
        if d.size_bytes != self.size_bytes:
            raise ValueError("MasterBinding.descriptor_snapshot size_bytes does not match size_bytes")
        if d.hdu != self.hdu:
            raise ValueError("MasterBinding.descriptor_snapshot hdu does not match hdu")
        if d.mask_identity != self.mask_identity:
            raise ValueError("MasterBinding.descriptor_snapshot mask_identity does not match mask_identity")

    @property
    def role_descriptor(self) -> MasterDescriptor:
        return self.descriptor_snapshot.descriptor

    def identity_dict(self) -> Mapping[str, object]:
        return {
            "descriptor_id": self.descriptor_id,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "hdu": self.hdu,
            "mask_identity": self.mask_identity,
        }
    def verify_snapshot(self) -> None:
        """Reject a tampered descriptor snapshot."""
        self.descriptor_snapshot.verify()

    def to_dict(self) -> Mapping[str, object]:
        return {
            "descriptor_id": self.descriptor_id,
            "descriptor_snapshot": dict(self.descriptor_snapshot.to_dict()),
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "hdu": self.hdu,
            "mask_identity": self.mask_identity,
            "locators": [l.to_dict() for l in self.locators],
            "mask_locator": self.mask_locator.to_dict() if self.mask_locator is not None else None,
            "acquired_at": self.acquired_at,
        }
    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "MasterBinding":
        return cls(
            descriptor_id=d["descriptor_id"],
            descriptor_snapshot=DescriptorSnapshot.from_dict(d["descriptor_snapshot"]),
            content_sha256=d["content_sha256"],
            size_bytes=d["size_bytes"],
            hdu=d["hdu"],
            mask_identity=d["mask_identity"],
            locators=tuple(FitsFileLocator(path=l["path"], hdu=l["hdu"]) for l in d.get("locators", ())),
            mask_locator=MaskPayloadLocator(path=d["mask_locator"]["path"]) if d.get("mask_locator") else None,
            acquired_at=d.get("acquired_at"),
        )


@dataclass(frozen=True)
class SkippedRole:
    """A role that had a compatible candidate but was not applied."""

    role: str
    reason_code: str
    detail: str = ""

    def to_dict(self) -> Mapping[str, object]:
        return {"role": self.role, "reason_code": self.reason_code, "detail": self.detail}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "SkippedRole":
        return cls(role=d["role"], reason_code=d["reason_code"], detail=d.get("detail", ""))


@dataclass(frozen=True)
class CalibrationComposition:
    """Honest, availability-relative record of what a plan actually applied.

    Level semantics (availability-relative):

    * ``NONE``    = no master at all is bound in the plan;
    * ``PARTIAL`` = at least one master bound, and at least one role that HAD a
      compatible candidate after ranking was not applied (or a scientifically
      dependent role could not be satisfied);
    * ``COMPLETE`` = at least one master bound and every role that had a
      compatible candidate was applied.

    Roles with no compatible candidate are recorded in ``no_candidate_roles``;
    roles that had a candidate but were skipped are recorded in
    ``skipped_roles`` with their reason codes. Nothing is silently hidden.
    """

    applied_roles: tuple[str, ...]
    skipped_roles: tuple[SkippedRole, ...]
    level: str  # NONE | PARTIAL | COMPLETE
    additive_state: str  # none | bias_only | dark_incl_bias | dark_bias_removed
    flat_applied: bool
    no_candidate_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.level not in ("NONE", "PARTIAL", "COMPLETE"):
            raise ValueError(f"CalibrationComposition.level invalid: {self.level!r}")
        if self.additive_state not in ("none", "bias_only", "dark_incl_bias", "dark_bias_removed"):
            raise ValueError(f"CalibrationComposition.additive_state invalid: {self.additive_state!r}")
        object.__setattr__(self, "applied_roles", tuple(self.applied_roles))
        object.__setattr__(self, "skipped_roles", tuple(self.skipped_roles))
        object.__setattr__(self, "no_candidate_roles", tuple(self.no_candidate_roles))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "applied_roles": list(self.applied_roles),
            "skipped_roles": [s.to_dict() for s in self.skipped_roles],
            "level": self.level,
            "additive_state": self.additive_state,
            "flat_applied": self.flat_applied,
            "no_candidate_roles": list(self.no_candidate_roles),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "CalibrationComposition":
        return cls(
            applied_roles=tuple(d.get("applied_roles", ())),
            skipped_roles=tuple(SkippedRole.from_dict(s) for s in d.get("skipped_roles", ())),
            level=d["level"],
            additive_state=d.get("additive_state", "none"),
            flat_applied=bool(d.get("flat_applied", False)),
            no_candidate_roles=tuple(d.get("no_candidate_roles", ())),
        )


@dataclass(frozen=True)
class CalibrationPlan:
    """Executable-intent plan binding validated light constraints, exact master
    identities and actual policy parameters. ``plan_id`` is derived and excludes
    retrieval locators and execution fields.

    ``selection`` / ``selection_policy_version`` / ``composition`` are NON-DIGEST
    audit blocks: they are carried by ``to_dict``/``from_dict`` but never enter
    ``plan_digest_dict`` (so two plans identical except for the selection block
    share the same ``plan_id``).
    """

    plan_id: str
    request: CalibrationRequest
    light_constraints: LightConstraints
    masters: Mapping[str, MasterBinding]
    policy_parameters: PolicyParameters
    versions: VersionSet
    selection: tuple = ()
    selection_policy_version: str = ""
    composition: Optional["CalibrationComposition"] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "masters", MappingProxyType(dict(self.masters)))
        object.__setattr__(self, "selection", tuple(self.selection))

    @classmethod
    def build(
        cls,
        request: CalibrationRequest,
        light_constraints: LightConstraints,
        masters: Mapping[str, MasterBinding],
        policy_parameters: PolicyParameters,
        versions: VersionSet,
        selection: tuple = (),
        selection_policy_version: str = "",
        composition: Optional["CalibrationComposition"] = None,
    ) -> "CalibrationPlan":
        plan = cls(
            plan_id="",
            request=request,
            light_constraints=light_constraints,
            masters=masters,
            policy_parameters=policy_parameters,
            versions=versions,
            selection=tuple(selection),
            selection_policy_version=selection_policy_version,
            composition=composition,
        )
        object.__setattr__(plan, "plan_id", plan.recompute_plan_id())
        return plan

    def plan_digest_dict(self) -> Mapping[str, object]:
        return {
            "request": dict(self.request.to_dict()),
            "light_constraints": dict(self.light_constraints.to_plan_dict()),
            "policy_parameters": dict(self.policy_parameters.to_dict()),
            "masters": {role: dict(b.identity_dict()) for role, b in self.masters.items()},
            "versions": dict(self.versions.to_dict()),
        }
    def recompute_plan_id(self) -> str:
        return plan_digest(self.plan_digest_dict())

    def verify_plan_id(self) -> None:
        recomputed = self.recompute_plan_id()
        if recomputed != self.plan_id:
            raise ValueError(f"plan digest mismatch: recorded {self.plan_id!r}, recomputed {recomputed!r}")

    def to_dict(self) -> Mapping[str, object]:
        """Full strict versioned serialization (snapshots/locators/declarations)."""
        return {
            "schema_version": PLAN_SERIALIZATION_SCHEMA,
            "plan_id": self.plan_id,
            "request": dict(self.request.to_dict()),
            "light_constraints": dict(self.light_constraints.to_full_dict()),
            "masters": {role: dict(b.to_dict()) for role, b in self.masters.items()},
            "policy_parameters": dict(self.policy_parameters.to_dict()),
            "versions": dict(self.versions.to_dict()),
            "selection": [s.to_dict() for s in self.selection],
            "selection_policy_version": self.selection_policy_version,
            "composition": dict(self.composition.to_dict()) if self.composition is not None else None,
        }
    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "CalibrationPlan":
        """Deserialize a plan; verify schema + recomputed plan_id (tamper rejection)."""
        if d.get("schema_version") != PLAN_SERIALIZATION_SCHEMA:
            raise ValueError(f"unsupported plan schema {d.get('schema_version')!r}")
        light = cls._light_from_dict(d["light_constraints"])
        masters = {role: MasterBinding.from_dict(b) for role, b in d["masters"].items()}
        plan = cls.build(
            request=CalibrationRequest(d["request"]["additive_mode"], d["request"]["flat_mode"]),
            light_constraints=light,
            masters=masters,
            policy_parameters=cls._policy_from_dict(d["policy_parameters"]),
            versions=VersionSet(**d["versions"]),
            selection=tuple(cls._selection_from_dict(s) for s in d.get("selection", ())),
            selection_policy_version=d.get("selection_policy_version", ""),
            composition=CalibrationComposition.from_dict(d["composition"]) if d.get("composition") is not None else None,
        )
        if d.get("plan_id") is not None and d["plan_id"] != plan.plan_id:
            raise ValueError(f"plan digest mismatch: recorded {d['plan_id']!r}, recomputed {plan.plan_id!r}")
        return plan

    @staticmethod
    def _selection_from_dict(d: Mapping[str, object]):
        from zecalibrator.core.selection import MasterSelectionRecord

        return MasterSelectionRecord.from_dict(d)

    @staticmethod
    def _light_from_dict(d: Mapping[str, object]) -> LightConstraints:
        from zecalibrator.core.descriptors import (
            AcquisitionProfileEvidence,
            DetectorIdentity,
            LightEvidence,
            OpticalIdentity,
        )
        from zecalibrator.core.geometry import Geometry

        def tup(v):
            return None if v is None else (int(v[0]), int(v[1]))

        geo = d["geometry"]
        acq = d["acquisition"]
        opt = d["optical"]
        det = d["detector"]
        evidence = d.get("evidence")
        return LightConstraints(
            geometry=Geometry(
                shape=(int(geo["shape"][0]), int(geo["shape"][1])),
                sensor_dimensions=tup(geo.get("sensor_dimensions")),
                binning=tup(geo.get("binning")),
                roi_origin=tup(geo.get("roi_origin")),
                roi_extent=tup(geo.get("roi_extent")),
                orientation=geo.get("orientation"),
                cfa_phase=geo.get("cfa_phase"),
            ),
            detector=DetectorIdentity(
                detector_instance_id=det["detector_instance_id"],
                detector_model=det.get("detector_model"),
                serial=det.get("serial"),
            ),
            acquisition=CalibrationPlan._acq_from_dict(acq),
            optical=OpticalIdentity(filter=opt.get("filter"), optical_train_id=opt.get("optical_train_id")),
            raw_domain_declaration=d["raw_domain_declaration"],
            evidence=(LightEvidence(original_cards=tuple(evidence.get("original_cards", ())), declaration=None, units=evidence.get("units")) if evidence is not None else None),
        )

    @staticmethod
    def _acq_from_dict(d: Mapping[str, object]):
        from zecalibrator.core.descriptors import Acquisition

        return Acquisition(
            gain=d.get("gain"), offset=d.get("offset"), readout_mode=d.get("readout_mode"),
            adc_mode=d.get("adc_mode"), temperature_c=d.get("temperature_c"), exposure_s=d.get("exposure_s"),
            saturation_limit_adu=d.get("saturation_limit_adu"), saturation_evidence=d.get("saturation_evidence", "unknown"),
            bias_exposure_max_s=d.get("bias_exposure_max_s"),
            short_flat_profile=d.get("short_flat_profile", False),
        )

    @staticmethod
    def _policy_from_dict(d: Mapping[str, object]) -> PolicyParameters:
        return PolicyParameters(
            exposure_tolerance=Tolerance(**d["exposure_tolerance"]),
            temperature_tolerance=Tolerance(**d["temperature_tolerance"]),
            flat_quality_policy=FlatQualityPolicy(**d["flat_quality_policy"]),
        )


@dataclass(frozen=True)
class Candidate:
    """A library candidate: one descriptor identity + its retrieval locations.

    Construction validates that the descriptor snapshot is exactly coherent with
    the descriptor identity (never a mismatched pair).

    ``acquired_at`` is a retrieval-facing DATE-OBS string (ranking evidence only);
    it is not part of the descriptor identity and never enters any digest.
    """

    candidate_id: str
    descriptor: MasterDescriptor
    descriptor_snapshot: DescriptorSnapshot
    locators: tuple[FitsFileLocator, ...] = ()
    mask_locator: Optional[MaskPayloadLocator] = None
    acquired_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("Candidate.candidate_id must be a non-empty string")
        object.__setattr__(self, "locators", tuple(self.locators))
        if self.acquired_at is not None and not isinstance(self.acquired_at, str):
            raise ValueError("Candidate.acquired_at must be a string or None")
        snap = self.descriptor_snapshot.descriptor
        if snap.descriptor_id != self.descriptor.descriptor_id:
            raise ValueError("Candidate descriptor_snapshot descriptor_id does not match descriptor")
        if snap.content_sha256 != self.descriptor.content_sha256:
            raise ValueError("Candidate descriptor_snapshot content_sha256 does not match descriptor")
        if snap.mask_identity != self.descriptor.mask_identity:
            raise ValueError("Candidate descriptor_snapshot mask_identity does not match descriptor")
        if snap.size_bytes != self.descriptor.size_bytes:
            raise ValueError("Candidate descriptor_snapshot size_bytes does not match descriptor")
        if snap.hdu != self.descriptor.hdu:
            raise ValueError("Candidate descriptor_snapshot hdu does not match descriptor")

    @property
    def role(self) -> str:
        return self.descriptor.master_type

    @property
    def identity_key(self) -> tuple[str, str]:
        """Collapse key: byte-identical AND descriptor-identical (SCIENCE §6.5)."""
        return (self.descriptor.content_sha256, self.descriptor.descriptor_id)


__all__ = [
    "ADDITIVE_MODES",
    "CalibrationComposition",
    "CalibrationPlan",
    "CalibrationRequest",
    "Candidate",
    "DIGEST_SCHEMA_VERSION",
    "FLAT_MODES",
    "FLAT_QUALITY_THRESHOLD",
    "FitsFileLocator",
    "FlatQualityPolicy",
    "LIBRARY_SCHEMA_VERSION",
    "MATCH_POLICY_VERSION",
    "MasterBinding",
    "MaskPayloadLocator",
    "MatchPolicy",
    "PLAN_SERIALIZATION_SCHEMA",
    "PROVENANCE_SCHEMA_VERSION",
    "PolicyError",
    "PolicyParameters",
    "SCIENCE_CONTRACT_VERSION",
    "SkippedRole",
    "Tolerance",
    "VersionSet",
    "default_match_policy",
]
