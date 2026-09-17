"""Public value objects and enums for ``zecalibrator.api.v1``.

This is a re-export layer over the frozen G3/G4 scientific value objects plus
the public facade value objects defined here. Consumers import these names from
``zecalibrator.api.v1`` only; they never import the ``zecalibrator.core`` /
``zecalibrator.application`` / ``zecalibrator.io`` module paths.

All value objects are immutable (frozen dataclasses). ``None`` means
unknown/not-present, never a silent default. Serialization (``to_dict`` /
``from_dict`` where provided) is strict and versioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from zecalibrator.application.cancellation import (
    CancellationToken,
    OperationCancelled,
    ProgressEvent,
    ProgressObserver,
)
from zecalibrator.application.library import (
    ArrayInputIdentity,
    DecisionEnvelope,
    FitsInputIdentity,
    LibraryHandle,
    LibrarySnapshot,
    ValidationResult,
    light_constraints_from_sensor_metadata,
)
from zecalibrator.core.calibrate import (
    CalibrationResult as _EngineCalibrationResult,
    FrameQuality,
)
from zecalibrator.core.descriptors import (
    Acquisition,
    AcquisitionProfileEvidence,
    DescriptorSnapshot,
    DetectorIdentity,
    LightConstraints,
    LightEvidence,
    MasterDescriptor,
    NormalizationProvenance,
    NormalizationScalars,
    OpticalIdentity,
    ProcessingProvenance,
    ValidityEvidence,
)
from zecalibrator.core.dq import CountSummary
from zecalibrator.core.geometry import Geometry
from zecalibrator.core.matching import MatchResult, Reason, RejectionRecord
from zecalibrator.core.metadata import (
    CardRecord,
    ConflictDiagnostic,
    ImportDeclaration,
    SensorMetadata,
    build_sensor_metadata,
)
from zecalibrator.core.plans import (
    CalibrationPlan,
    CalibrationRequest,
    Candidate,
    FitsFileLocator,
    FlatQualityPolicy,
    MaskPayloadLocator,
    MasterBinding,
    MatchPolicy,
    PolicyParameters,
    Tolerance,
    VersionSet,
    default_match_policy,
)
from zecalibrator.core.precision import PrecisionInfo
from zecalibrator.io.master_source import FilesystemSource, InMemorySource

from ._meta import ApiInfo, CAPABILITIES, PROVENANCE_SCHEMA
from .errors import InvalidRequestError

# Discriminated input identity (field-identical to ARCHITECTURE §3.3's
# ``FitsIdentity``/``ArrayIdentity``; these are the frozen G4 names).
InputIdentity = Union[FitsInputIdentity, ArrayInputIdentity]


# ---------------------------------------------------------------------------
# Explicit ROI-extent evidence (Junior R3 decision)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoiExtentEvidence:
    """Explicit versioned evidence-backed ROI extent (``(ny, nx)``).

    Supplied at the public source boundary so a FITS-decoded light can satisfy
    the G4 matching requirement for a known ROI extent without a blanket
    ``None -> shape`` derivation. Validated against the actual decoded shape.
    """

    extent: Tuple[int, int]
    source: str
    identity: str
    version: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.extent, (tuple, list))
            or len(self.extent) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in self.extent)
        ):
            raise InvalidRequestError("RoiExtentEvidence.extent must be a pair of positive ints")
        object.__setattr__(self, "extent", (int(self.extent[0]), int(self.extent[1])))
        for name in ("source", "identity", "version"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise InvalidRequestError(f"RoiExtentEvidence.{name} must be a non-empty string")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "extent": list(self.extent),
            "source": self.source,
            "identity": self.identity,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# FrameSource — tagged union (ARCHITECTURE §3.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FitsFrameSource:
    """A raw FITS frame the product decodes (``path`` + optional ``hdu``).

    ``declaration`` supplies the evidence-backed raw-domain/import facts
    required for strict decode. ``roi_extent`` optionally supplies the explicit
    evidence-backed ROI extent the strict decoder does not infer.
    """

    path: Union[str, PathLike]
    hdu: Union[int, str] = 0
    declaration: Optional[ImportDeclaration] = None
    roi_extent: Optional[RoiExtentEvidence] = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, PathLike)) or str(self.path) == "":
            raise InvalidRequestError("FitsFrameSource.path must be a non-empty path")
        if isinstance(self.hdu, bool) or not isinstance(self.hdu, (int, str)):
            raise InvalidRequestError("FitsFrameSource.hdu must be int or str")
        if isinstance(self.hdu, int) and self.hdu < 0:
            raise InvalidRequestError("FitsFrameSource.hdu must be a non-negative int or str")
        if self.roi_extent is not None and not isinstance(self.roi_extent, RoiExtentEvidence):
            raise InvalidRequestError("FitsFrameSource.roi_extent must be a RoiExtentEvidence")


@dataclass(frozen=True)
class ArrayFrameSource:
    """A caller-supplied decoded physical-ADU array (read-only to the callee)."""

    data: object  # ArrayLike
    metadata: SensorMetadata
    identity: ArrayInputIdentity
    invalid_mask: object = None
    domain: str = "sensor_adu"
    scaling_applied: bool = True

    def __post_init__(self) -> None:
        if self.domain != "sensor_adu":
            raise InvalidRequestError(
                f"ArrayFrameSource.domain must be 'sensor_adu', got {self.domain!r}"
            )
        if not self.scaling_applied:
            raise InvalidRequestError(
                "ArrayFrameSource requires scaling_applied=True (physical ADU already applied)"
            )
        if not isinstance(self.metadata, SensorMetadata):
            raise InvalidRequestError("ArrayFrameSource.metadata must be a SensorMetadata")
        if self.metadata.raw_domain_declaration != "raw":
            raise InvalidRequestError("ArrayFrameSource.metadata.raw_domain_declaration must be 'raw'")
        if self.metadata.units not in (None, "ADU"):
            raise InvalidRequestError("ArrayFrameSource.metadata.units must be 'ADU'")


FrameSource = Union[FitsFrameSource, ArrayFrameSource]


@dataclass(frozen=True)
class FrameInspection:
    """Immutable inspection of a decoded frame (ARCHITECTURE §3.3)."""

    metadata: SensorMetadata
    identity: InputIdentity
    domain_finding: str  # "raw" | "processed" | "unknown" | "unsupported"
    hdu: Union[int, str, None]  # None for array sources (no invented FITS identity)
    shape: Tuple[int, int]
    warnings: tuple[str, ...] = ()
    roi_extent_evidence: Optional[RoiExtentEvidence] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True)
class InspectResult:
    """Structured ``inspect_frame`` outcome envelope (§3.2)."""

    operation_status: str  # "COMPLETED" | "CANCELLED" | "FAILED"
    inspection: Optional[FrameInspection] = None
    reason_code: Optional[str] = None
    details: str = ""


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LibrarySpec:
    """User library root + optional index override (no ZeAlfie discovery)."""

    root: Union[str, PathLike]
    index_path: Optional[Union[str, PathLike]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.root, (str, PathLike)) or str(self.root) == "":
            raise InvalidRequestError("LibrarySpec.root must be a non-empty path")
        if self.index_path is not None and (
            not isinstance(self.index_path, (str, PathLike)) or str(self.index_path) == ""
        ):
            raise InvalidRequestError("LibrarySpec.index_path must be a non-empty path when set")

    @property
    def resolved_index_path(self) -> str:
        if self.index_path is not None:
            return str(self.index_path)
        import os

        return os.path.join(str(self.root), "zecalibrator.library.sqlite")


@dataclass(frozen=True)
class OpenLibraryResult:
    """Structured ``open_library`` outcome envelope (§3.2)."""

    operation_status: str  # "OPENED" | "CANCELLED" | "FAILED"
    handle: Optional[LibraryHandle] = None
    reason_code: Optional[str] = None
    details: str = ""


@dataclass(frozen=True)
class ResolveResult:
    """Structured ``resolve_calibration`` outcome envelope (§3.2)."""

    operation_status: str  # "COMPLETED" | "CANCELLED" | "FAILED"
    decision: Optional[DecisionEnvelope] = None
    reason_code: Optional[str] = None
    details: str = ""

    @property
    def outcome(self) -> Optional[str]:
        return self.decision.outcome if self.decision is not None else None

    @property
    def plan(self) -> Optional[CalibrationPlan]:
        return self.decision.plan if self.decision is not None else None


@dataclass(frozen=True)
class ExecutionOptions:
    """Execution options for ``calibrate_frame``.

    Only the CPU backend is implemented/advertised in this release. Unimplemented
    options (GPU, standalone FITS output, memory/tile hints) are refused rather
    than silently claimed.
    """

    backend_requested: str = "cpu"
    memory_budget: object = None
    tile_shape: Optional[Tuple[int, int]] = None
    output_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.backend_requested != "cpu":
            raise InvalidRequestError(
                f"only backend_requested='cpu' is supported, got {self.backend_requested!r}"
            )
        if self.output_path is not None:
            raise InvalidRequestError("standalone FITS output is not implemented in this release")
        if self.memory_budget is not None or self.tile_shape is not None:
            raise InvalidRequestError("memory_budget/tile_shape are not implemented in this release")


# ---------------------------------------------------------------------------
# Public execution provenance (ARCHITECTURE §3.3/§3.8)
# ---------------------------------------------------------------------------
def _reject_unknown(d: Mapping, allowed: frozenset, path: str) -> None:
    unknown = set(d.keys()) - allowed
    if unknown:
        raise InvalidRequestError(f"unknown key(s) at {path}: {sorted(unknown)}")


def _identity_to_dict(identity) -> Mapping[str, object]:
    if identity is None:
        return {"kind": None}
    if isinstance(identity, FitsInputIdentity):
        return {
            "kind": "fits",
            "path": identity.path,
            "hdu": identity.hdu,
            "whole_fits_sha256": identity.whole_fits_sha256,
            "decoded_digest": identity.decoded_digest,
        }
    return {
        "kind": "array",
        "caller_logical_id": identity.caller_logical_id,
        "decoded_digest": identity.decoded_digest,
    }


def _identity_from_dict(d) -> Optional[InputIdentity]:
    kind = d.get("kind")
    if kind == "fits":
        return FitsInputIdentity(
            path=d["path"],
            hdu=d["hdu"],
            whole_fits_sha256=d.get("whole_fits_sha256"),
            decoded_digest=d.get("decoded_digest", ""),
        )
    if kind == "array":
        return ArrayInputIdentity(
            caller_logical_id=d["caller_logical_id"],
            decoded_digest=d.get("decoded_digest", ""),
        )
    if kind is None:
        return None
    raise InvalidRequestError(f"unsupported input_identity kind {kind!r}")


def _roi_evidence_to_dict(ev) -> Optional[Mapping[str, object]]:
    return None if ev is None else ev.to_dict()


def _roi_evidence_from_dict(d) -> Optional[RoiExtentEvidence]:
    if d is None:
        return None
    return RoiExtentEvidence(
        extent=tuple(d["extent"]), source=d["source"], identity=d["identity"], version=d["version"]
    )


_PROVENANCE_TOP_KEYS = frozenset({
    "schema_version", "operation_id", "input_identity", "input_dtype", "input_units",
    "input_scaling", "roi_extent_evidence", "plan", "policy", "api_version",
    "product_version", "decoder_version", "provenance_schema", "matching_policy",
    "science_contract", "backend", "request", "executed_processing", "scalars",
    "status", "reason_code", "warnings", "flat_saturation_screening",
    "flat_scalars_origin",
})


@dataclass(frozen=True)
class ProvenanceRecord:
    """Structured in-memory execution provenance (ARCHITECTURE §3.3/§3.8).

    Retains the operation id, actual input identity/evidence (dtype/units/scaling
    and ROI evidence source/identity/version), the accepted plan and master
    snapshots/digests, policy and version facts, the executed backend/processing
    and operation status/scalars/warnings. No output-file hash self-reference and
    no fabricated FITS identity for array inputs. Genuinely unavailable facts are
    explicit ``None``, never invented.
    """

    operation_id: str
    input_identity: Optional[InputIdentity]
    plan: Optional[CalibrationPlan]
    policy: Optional[MatchPolicy]
    api_version: str
    product_version: str
    decoder_version: str
    provenance_schema: str
    matching_policy: str
    science_contract: str
    backend: str
    request: Optional[CalibrationRequest]
    executed_processing: tuple[str, ...]
    scalars: Mapping[str, Optional[float]]
    status: str
    reason_code: Optional[str]
    warnings: tuple[str, ...] = ()
    schema_version: str = PROVENANCE_SCHEMA
    input_dtype: Optional[str] = None
    input_units: Optional[str] = None
    input_scaling: Optional[Mapping[str, object]] = None
    roi_extent_evidence: Optional[RoiExtentEvidence] = None
    flat_saturation_screening: Optional[str] = None
    flat_scalars_origin: Optional[str] = None

    def __post_init__(self) -> None:
        if self.schema_version != PROVENANCE_SCHEMA:
            raise InvalidRequestError(f"unsupported provenance schema {self.schema_version!r}")
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise InvalidRequestError("ProvenanceRecord.operation_id must be a non-empty string")
        object.__setattr__(self, "executed_processing", tuple(self.executed_processing))
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.input_scaling is not None:
            object.__setattr__(self, "input_scaling", MappingProxyType(dict(self.input_scaling)))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "input_identity": dict(_identity_to_dict(self.input_identity)),
            "input_dtype": self.input_dtype,
            "input_units": self.input_units,
            "input_scaling": dict(self.input_scaling) if self.input_scaling is not None else None,
            "roi_extent_evidence": _roi_evidence_to_dict(self.roi_extent_evidence),
            "flat_saturation_screening": self.flat_saturation_screening,
            "flat_scalars_origin": self.flat_scalars_origin,
            "plan": dict(self.plan.to_dict()) if self.plan is not None else None,
            "policy": dict(self.policy.to_dict()) if self.policy is not None else None,
            "api_version": self.api_version,
            "product_version": self.product_version,
            "decoder_version": self.decoder_version,
            "provenance_schema": self.provenance_schema,
            "matching_policy": self.matching_policy,
            "science_contract": self.science_contract,
            "backend": self.backend,
            "request": dict(self.request.to_dict()) if self.request is not None else None,
            "executed_processing": list(self.executed_processing),
            "scalars": dict(self.scalars),
            "status": self.status,
            "reason_code": self.reason_code,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "ProvenanceRecord":
        unknown = set(d.keys()) - _PROVENANCE_TOP_KEYS
        if unknown:
            raise InvalidRequestError(f"unknown provenance key(s): {sorted(unknown)}")
        if d.get("schema_version") != PROVENANCE_SCHEMA:
            raise InvalidRequestError(f"unsupported provenance schema {d.get('schema_version')!r}")
        plan = CalibrationPlan.from_dict(d["plan"]) if d.get("plan") is not None else None
        policy = d.get("policy")
        if policy is not None:
            _reject_unknown(policy, frozenset({"version", "exposure_tolerance", "temperature_tolerance", "flat_quality_policy"}), "policy")
            _reject_unknown(policy["exposure_tolerance"], frozenset({"relative", "absolute"}), "policy.exposure_tolerance")
            _reject_unknown(policy["temperature_tolerance"], frozenset({"relative", "absolute"}), "policy.temperature_tolerance")
            policy = MatchPolicy(
                version=policy["version"],
                exposure_tolerance=Tolerance(**policy["exposure_tolerance"]),
                temperature_tolerance=Tolerance(**policy["temperature_tolerance"]),
                flat_quality_policy=FlatQualityPolicy(**policy["flat_quality_policy"]),
            )
        request = d.get("request")
        if request is not None:
            _reject_unknown(request, frozenset({"additive_mode", "flat_mode"}), "request")
            request = CalibrationRequest(request["additive_mode"], request["flat_mode"])
        return cls(
            schema_version=d["schema_version"],
            operation_id=d["operation_id"],
            input_identity=_identity_from_dict(d.get("input_identity", {})),
            input_dtype=d.get("input_dtype"),
            input_units=d.get("input_units"),
            input_scaling=dict(d["input_scaling"]) if d.get("input_scaling") is not None else None,
            roi_extent_evidence=_roi_evidence_from_dict(d.get("roi_extent_evidence")),
            flat_saturation_screening=d.get("flat_saturation_screening"),
            flat_scalars_origin=d.get("flat_scalars_origin"),
            plan=plan,
            policy=policy,
            api_version=d["api_version"],
            product_version=d["product_version"],
            decoder_version=d["decoder_version"],
            provenance_schema=d["provenance_schema"],
            matching_policy=d["matching_policy"],
            science_contract=d["science_contract"],
            backend=d["backend"],
            request=request,
            executed_processing=tuple(d.get("executed_processing", ())),
            scalars=dict(d.get("scalars", {})),
            status=d["status"],
            reason_code=d.get("reason_code"),
            warnings=tuple(d.get("warnings", ())),
        )


@dataclass(frozen=True)
class CalibrationResult:
    """Public single-frame calibration result (composition over the G3 engine).

    Preserves the G3 result fields (status/data/mask/counts/frame-quality/
    precision/scalars/warnings/reason_code) and attaches the complete structured
    in-memory :class:`ProvenanceRecord`. The underlying engine result is private;
    no duplicate arithmetic is performed.
    """

    _engine: _EngineCalibrationResult
    provenance: ProvenanceRecord

    @property
    def status(self) -> str:
        return self._engine.status

    @property
    def data(self):
        return self._engine.data

    @property
    def mask(self):
        return self._engine.mask

    @property
    def counts(self) -> Optional[CountSummary]:
        return self._engine.counts

    @property
    def frame_quality(self) -> FrameQuality:
        return self._engine.frame_quality

    @property
    def precision(self) -> Optional[PrecisionInfo]:
        return self._engine.precision

    @property
    def scalars(self) -> Mapping[str, Optional[float]]:
        return self._engine.scalars

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._engine.warnings

    @property
    def reason_code(self) -> Optional[str]:
        return self._engine.reason_code

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": PROVENANCE_SCHEMA,
            "payload_serialized": False,  # data/mask/precision are in-memory only
            "status": self.status,
            "warnings": list(self.warnings),
            "reason_code": self.reason_code,
            "scalars": dict(self.scalars),
            "counts": {
                "total": self.counts.total,
                "valid_count": self.counts.valid_count,
                "invalid_count": self.counts.invalid_count,
                "per_bit": dict(self.counts.per_bit),
            }
            if self.counts is not None
            else None,
            "frame_quality": {
                "saturation_evidence": self.frame_quality.saturation_evidence,
            },
            "provenance": dict(self.provenance.to_dict()),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "CalibrationResult":
        from zecalibrator.core.dq import CountSummary

        if d.get("status") in ("COMPLETED", "COMPLETED_WITH_WARNINGS") and not d.get("payload_serialized"):
            raise InvalidRequestError(
                "cannot reconstruct a COMPLETED result from audit-only serialization (payload not serialized)"
            )
        counts = d.get("counts")
        if counts is not None:
            counts = CountSummary(
                total=counts["total"],
                valid_count=counts["valid_count"],
                invalid_count=counts["invalid_count"],
                per_bit=counts["per_bit"],
            )
        fq = d["frame_quality"]
        engine = _EngineCalibrationResult(
            status=d["status"],
            data=None,
            mask=None,
            counts=counts,
            frame_quality=FrameQuality(saturation_evidence=fq["saturation_evidence"]),
            precision=None,
            scalars=dict(d.get("scalars", {})),
            warnings=tuple(d.get("warnings", ())),
            reason_code=d.get("reason_code"),
        )
        return cls(_engine=engine, provenance=ProvenanceRecord.from_dict(d["provenance"]))


__all__ = [
    "CAPABILITIES",
    "PROVENANCE_SCHEMA",
    "Acquisition",
    "AcquisitionProfileEvidence",
    "ApiInfo",
    "ArrayFrameSource",
    "ArrayInputIdentity",
    "CalibrationPlan",
    "CalibrationRequest",
    "CalibrationResult",
    "CancellationToken",
    "Candidate",
    "CardRecord",
    "ConflictDiagnostic",
    "CountSummary",
    "DecisionEnvelope",
    "DescriptorSnapshot",
    "DetectorIdentity",
    "ExecutionOptions",
    "FilesystemSource",
    "FitsFileLocator",
    "FitsFrameSource",
    "FitsInputIdentity",
    "FlatQualityPolicy",
    "FrameInspection",
    "FrameQuality",
    "FrameSource",
    "Geometry",
    "ImportDeclaration",
    "InMemorySource",
    "InputIdentity",
    "InspectResult",
    "LibraryHandle",
    "LibrarySnapshot",
    "LibrarySpec",
    "LightConstraints",
    "LightEvidence",
    "MaskPayloadLocator",
    "MasterBinding",
    "MasterDescriptor",
    "MatchPolicy",
    "MatchResult",
    "NormalizationProvenance",
    "NormalizationScalars",
    "OpenLibraryResult",
    "OperationCancelled",
    "OpticalIdentity",
    "PolicyParameters",
    "PrecisionInfo",
    "ProcessingProvenance",
    "ProgressEvent",
    "ProgressObserver",
    "ProvenanceRecord",
    "Reason",
    "RejectionRecord",
    "ResolveResult",
    "RoiExtentEvidence",
    "SensorMetadata",
    "Tolerance",
    "ValidationResult",
    "ValidityEvidence",
    "VersionSet",
    "build_sensor_metadata",
    "default_match_policy",
    "light_constraints_from_sensor_metadata",
]
