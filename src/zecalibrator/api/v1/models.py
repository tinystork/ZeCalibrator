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

import os
import dataclasses
from dataclasses import dataclass, field
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
    "flat_scalars_origin", "master_domain_transforms",
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
    master_domain_transforms: tuple = ()

    def __post_init__(self) -> None:
        if self.schema_version != PROVENANCE_SCHEMA:
            raise InvalidRequestError(f"unsupported provenance schema {self.schema_version!r}")
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise InvalidRequestError("ProvenanceRecord.operation_id must be a non-empty string")
        object.__setattr__(self, "executed_processing", tuple(self.executed_processing))
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "master_domain_transforms", tuple(self.master_domain_transforms))
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
            "master_domain_transforms": [dict(t) for t in self.master_domain_transforms],
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
            master_domain_transforms=tuple(d.get("master_domain_transforms", ())),
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


# ---------------------------------------------------------------------------
# Batch (Phase 6): BatchOptions / BatchOutputRecord / BatchItem / BatchManifest
# ---------------------------------------------------------------------------
BATCH_MANIFEST_SCHEMA = "zecalibrator.batch_manifest.v1"

_BATCH_DISPOSITIONS = (
    "COMPLETED", "COMPLETED_WITH_WARNINGS", "SKIPPED", "FAILED", "CANCELLED",
)
_BATCH_COMMIT_STATES = ("COMMITTED", "CANCELLED")
_BATCH_STATUSES = ("COMPLETED", "PARTIAL", "CANCELLED")


@dataclass(frozen=True)
class BatchOptions:
    """Batch orchestration options (destination, overwrite policy, batch id).

    ``destination=None`` selects the bounded in-memory path (per-item
    ``CalibrationResult``). A non-``None`` ``destination`` selects the standalone
    transactional path (per-item committed FITS output); it must already exist
    as a directory (never auto-created). Only the no-clobber overwrite policy is
    supported in v1.

    This is the M1 batch-options surface: ``calibrate_batch`` takes
    ``BatchOptions`` (per the prepared Phase-6 contract), not the earlier
    ARCHITECTURE §3.2 ``ExecutionOptions`` sketch.
    """

    destination: Optional[Union[str, PathLike]] = None
    overwrite_policy: str = "no_clobber"
    batch_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.destination is not None and (
            not isinstance(self.destination, (str, PathLike)) or str(self.destination) == ""
        ):
            raise InvalidRequestError("BatchOptions.destination must be a non-empty path when set")
        if self.destination is not None and not os.path.isdir(os.fspath(self.destination)):
            raise InvalidRequestError(
                f"BatchOptions.destination must be an existing directory, got {self.destination!r}"
            )
        if self.overwrite_policy != "no_clobber":
            raise InvalidRequestError(
                f"only overwrite_policy='no_clobber' is supported, got {self.overwrite_policy!r}"
            )
        if self.batch_id is not None and (
            not isinstance(self.batch_id, str) or not self.batch_id.strip()
        ):
            raise InvalidRequestError("BatchOptions.batch_id must be a non-empty string")

    @property
    def standalone(self) -> bool:
        return self.destination is not None


@dataclass(frozen=True)
class BatchOutputRecord:
    """A committed standalone output locator + whole-file identity.

    ``science_digest`` is the canonical science digest (§2.3);
    ``whole_file_sha256`` is the final whole-file SHA-256 computed after output
    closure (manifest-only, never self-referenced).
    """

    path: str
    logical_id: str
    science_digest: str
    whole_file_sha256: str
    size_bytes: int
    committed: bool = True

    def __post_init__(self) -> None:
        for name in ("path", "logical_id", "science_digest", "whole_file_sha256"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v:
                raise InvalidRequestError(f"BatchOutputRecord.{name} must be a non-empty string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise InvalidRequestError("BatchOutputRecord.size_bytes must be a non-negative int")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "path": self.path,
            "logical_id": self.logical_id,
            "science_digest": self.science_digest,
            "whole_file_sha256": self.whole_file_sha256,
            "size_bytes": self.size_bytes,
            "committed": self.committed,
        }


@dataclass(frozen=True)
class BatchItem:
    """One bounded per-frame batch result (ARCHITECTURE §3.5).

    In-memory path: ``result`` carries the complete ``CalibrationResult``
    (data + mask + provenance). Standalone path: ``result`` is ``None`` and
    ``output`` carries the committed locator + commit state. ``disposition`` is
    one of the five frozen statuses; ``reason_code``/``reason_details`` carry a
    structured error/cancel reason.
    """

    index: int
    disposition: str
    input_identity: Optional[InputIdentity]
    plan_id: Optional[str]
    result: Optional[CalibrationResult] = None
    output: Optional[BatchOutputRecord] = None
    reason_code: Optional[str] = None
    reason_details: str = ""
    warnings: tuple = ()

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise InvalidRequestError("BatchItem.index must be a non-negative int")
        if self.disposition not in _BATCH_DISPOSITIONS:
            raise InvalidRequestError(f"BatchItem.disposition invalid: {self.disposition!r}")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.output is not None and not isinstance(self.output, BatchOutputRecord):
            raise InvalidRequestError("BatchItem.output must be a BatchOutputRecord")
        if self.result is not None and not isinstance(self.result, CalibrationResult):
            raise InvalidRequestError("BatchItem.result must be a CalibrationResult")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "index": self.index,
            "disposition": self.disposition,
            "input_identity": dict(_identity_to_dict(self.input_identity)),
            "plan_id": self.plan_id,
            "reason_code": self.reason_code,
            "reason_details": self.reason_details,
            "warnings": list(self.warnings),
            "output": dict(self.output.to_dict()) if self.output is not None else None,
        }


@dataclass(frozen=True)
class BatchManifest:
    """The external batch manifest value object (strict versioned serialization).

    ``inputs``/``items`` are ordered tuples of already-validated structured
    records (JSON-safe). ``from_dict`` performs strict schema validation via the
    io manifest parser.
    """

    batch_id: str
    operation_id: str
    commit_state: str
    batch_status: str
    destination: Optional[str]
    inputs: Tuple[Mapping[str, object], ...]
    items: Tuple[Mapping[str, object], ...]
    api_version: str = "1.0"
    product_version: str = ""
    science_contract: str = "1.0"
    matching_policy: str = "zecalibrator.match.v1"
    provenance_schema: str = "zecalibrator.provenance.v1"
    decoder_version: str = "1.0"
    schema_version: str = BATCH_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.commit_state not in _BATCH_COMMIT_STATES:
            raise InvalidRequestError(f"BatchManifest.commit_state invalid: {self.commit_state!r}")
        if self.batch_status not in _BATCH_STATUSES:
            raise InvalidRequestError(f"BatchManifest.batch_status invalid: {self.batch_status!r}")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "items", tuple(self.items))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "operation_id": self.operation_id,
            "api_version": self.api_version,
            "product_version": self.product_version,
            "science_contract": self.science_contract,
            "matching_policy": self.matching_policy,
            "provenance_schema": self.provenance_schema,
            "decoder_version": self.decoder_version,
            "commit_state": self.commit_state,
            "batch_status": self.batch_status,
            "destination": self.destination,
            "inputs": [dict(i) for i in self.inputs],
            "items": [dict(i) for i in self.items],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "BatchManifest":
        from zecalibrator.io.batch_manifest import parse_batch_manifest

        parsed = parse_batch_manifest(d)
        return cls(
            batch_id=parsed["batch_id"],
            operation_id=parsed["operation_id"],
            commit_state=parsed["commit_state"],
            batch_status=parsed["batch_status"],
            destination=parsed.get("destination"),
            inputs=tuple(dict(i) for i in parsed.get("inputs", ())),
            items=tuple(dict(i) for i in parsed.get("items", ())),
            api_version=parsed["api_version"],
            product_version=parsed["product_version"],
            science_contract=parsed["science_contract"],
            matching_policy=parsed["matching_policy"],
            provenance_schema=parsed["provenance_schema"],
            decoder_version=parsed["decoder_version"],
            schema_version=parsed["schema_version"],
        )


# ---------------------------------------------------------------------------
# Library indexing (Phase 6): MasterImportSpec / IndexLibraryResult
# ---------------------------------------------------------------------------
# Discriminated DQ-state vocabulary (P7-M3B). ``no_source_dq`` is structural.
DQ_STATES: tuple[str, ...] = ("source_mask", "no_source_dq")


class DqState:
    """Discriminated DQ-state constants (additive-only vocabulary)."""

    SOURCE_MASK = "source_mask"
    NO_SOURCE_DQ = "no_source_dq"
    VALUES: tuple[str, ...] = DQ_STATES


@dataclass(frozen=True)
class MasterImportSpec:
    """An explicit evidence-backed master import for the public library-indexing
    path. Descriptors are built from FITS headers + this declaration; it is never
    a header->descriptor inference. ``mask_path`` supplies the external DQ mask
    payload identity; ``path`` may be absolute or relative to the library root.

    ``dq_state`` is a discriminated DQ state: ``"source_mask"`` (default;
    ``mask_path`` required, existing semantics) or ``"no_source_dq"`` (structural
    no-source-DQ state; ``mask_path`` must be absent, never a synthetic file).
    """

    path: str
    master_type: str
    declaration: ImportDeclaration
    hdu: Union[int, str] = 0
    mask_path: Optional[str] = None
    bias_state: Optional[str] = None
    flat_form: Optional[str] = None
    normalization_algorithm: Optional[str] = None
    normalization_scalars: Optional[NormalizationScalars] = None
    normalization_provenance: Optional[NormalizationProvenance] = None
    validity_evidence: Optional[ValidityEvidence] = None
    processing_provenance: Optional[ProcessingProvenance] = None
    dq_state: str = "source_mask"
    acquired_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise InvalidRequestError("MasterImportSpec.path must be a non-empty string")
        if self.master_type not in ("bias", "dark", "flat", "flat_dark"):
            raise InvalidRequestError(f"MasterImportSpec.master_type invalid: {self.master_type!r}")
        if not isinstance(self.declaration, ImportDeclaration):
            raise InvalidRequestError("MasterImportSpec.declaration must be an ImportDeclaration")
        if isinstance(self.hdu, bool) or not isinstance(self.hdu, (int, str)):
            raise InvalidRequestError("MasterImportSpec.hdu must be int or str")
        if isinstance(self.hdu, int) and self.hdu < 0:
            raise InvalidRequestError("MasterImportSpec.hdu must be a non-negative int or str")
        if self.acquired_at is not None and (not isinstance(self.acquired_at, str) or not self.acquired_at.strip()):
            raise InvalidRequestError("MasterImportSpec.acquired_at must be a non-empty string or None")
        if self.dq_state not in DQ_STATES:
            raise InvalidRequestError(f"MasterImportSpec.dq_state invalid: {self.dq_state!r}")
        if self.dq_state == "no_source_dq":
            if self.mask_path is not None:
                raise InvalidRequestError(
                    "MasterImportSpec with dq_state='no_source_dq' must have mask_path=None"
                )
        else:
            if self.mask_path is None or not isinstance(self.mask_path, str) or not self.mask_path.strip():
                raise InvalidRequestError(
                    "MasterImportSpec with dq_state='source_mask' requires a non-empty mask_path"
                )


# ---------------------------------------------------------------------------
# Managed master ingestion (P7-M3B): EvidenceFact / ManagedMasterRecord
# ---------------------------------------------------------------------------
MANAGED_LEDGER_SCHEMA = "zecalibrator.managed_ledger.v1"

# Evidence origin types (frozen vocabulary).
EVIDENCE_ORIGIN_TYPES: tuple[str, ...] = ("fits_header", "user", "sidecar", "native")


def _freeze_evidence_value(value):
    """Recursively coerce a value into an immutable JSON-safe structure."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_evidence_value(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return _freeze_evidence_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_evidence_value(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class EvidenceFact:
    """Per-field provenance/confirmation annotation (R2: annotative only).

    Records *how* a scientific fact was obtained (``origin_type``/``origin_field``/
    ``confirmed_by``/``version``). It is **never** an independent competing copy of
    the scientific facts: the validated :class:`ImportDeclaration` remains the
    canonical scientific-fact representation.
    """

    field: str
    value: object
    origin_type: str
    origin_field: Optional[str] = None
    confirmed_by: Optional[str] = None
    version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise InvalidRequestError("EvidenceFact.field must be a non-empty string")
        if self.origin_type not in EVIDENCE_ORIGIN_TYPES:
            raise InvalidRequestError(
                f"EvidenceFact.origin_type invalid: {self.origin_type!r}"
            )
        if self.origin_field is not None and not isinstance(self.origin_field, str):
            raise InvalidRequestError("EvidenceFact.origin_field must be a string or None")
        if self.confirmed_by is not None and not isinstance(self.confirmed_by, str):
            raise InvalidRequestError("EvidenceFact.confirmed_by must be a string or None")
        if not isinstance(self.version, str) or not self.version.strip():
            raise InvalidRequestError("EvidenceFact.version must be a non-empty string")
        object.__setattr__(self, "field", self.field)
        object.__setattr__(self, "value", _freeze_evidence_value(self.value))
        object.__setattr__(self, "origin_type", self.origin_type)
        object.__setattr__(self, "origin_field", self.origin_field)
        object.__setattr__(self, "confirmed_by", self.confirmed_by)
        object.__setattr__(self, "version", self.version)

    def to_dict(self) -> Mapping[str, object]:
        value = self.value
        if isinstance(value, tuple):
            value = list(value)
        return {
            "field": self.field,
            "value": value,
            "origin_type": self.origin_type,
            "origin_field": self.origin_field,
            "confirmed_by": self.confirmed_by,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "EvidenceFact":
        value = d.get("value")
        if isinstance(value, list):
            value = tuple(value)
        return cls(
            field=d["field"],
            value=value,
            origin_type=d["origin_type"],
            origin_field=d.get("origin_field"),
            confirmed_by=d.get("confirmed_by"),
            version=d.get("version", "1"),
        )


@dataclass(frozen=True)
class ManagedMasterRecord:
    """Immutable managed-master record (P7-M3B, R2).

    ``content_sha256``/``size_bytes`` are the content identity of the master
    FITS bytes; ``role``/``master_type`` the semantic role; ``declaration`` the
    validated canonical :class:`ImportDeclaration` (the single source of
    scientific truth); ``evidence`` the per-field provenance annotations;
    ``dq_state`` the structural DQ state; ``mask_path``/``last_seen_path`` are
    location/provenance only, never scientific identity.
    """

    role: str
    content_sha256: str
    size_bytes: int
    declaration: ImportDeclaration
    hdu: Union[int, str] = 0
    bias_state: Optional[str] = None
    flat_form: Optional[str] = None
    evidence: Mapping[str, EvidenceFact] = field(default_factory=dict)
    dq_state: str = "source_mask"
    mask_path: Optional[str] = None
    last_seen_path: Optional[str] = None
    schema_version: str = MANAGED_LEDGER_SCHEMA
    acquired_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ("bias", "dark", "flat", "flat_dark"):
            raise InvalidRequestError(f"ManagedMasterRecord.role invalid: {self.role!r}")
        if self.acquired_at is not None and (not isinstance(self.acquired_at, str) or not self.acquired_at.strip()):
            raise InvalidRequestError("ManagedMasterRecord.acquired_at must be a non-empty string or None")
        if not _is_hex64(self.content_sha256):
            raise InvalidRequestError(
                "ManagedMasterRecord.content_sha256 must be a 64-char lowercase hex string"
            )
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise InvalidRequestError("ManagedMasterRecord.size_bytes must be a non-negative int")
        if not isinstance(self.declaration, ImportDeclaration):
            raise InvalidRequestError("ManagedMasterRecord.declaration must be an ImportDeclaration")
        if isinstance(self.hdu, bool) or not isinstance(self.hdu, (int, str)):
            raise InvalidRequestError("ManagedMasterRecord.hdu must be int or str")
        if isinstance(self.hdu, int) and self.hdu < 0:
            raise InvalidRequestError("ManagedMasterRecord.hdu must be a non-negative int or str")
        if self.dq_state not in DQ_STATES:
            raise InvalidRequestError(f"ManagedMasterRecord.dq_state invalid: {self.dq_state!r}")
        if self.dq_state == "no_source_dq":
            if self.mask_path is not None:
                raise InvalidRequestError(
                    "ManagedMasterRecord with dq_state='no_source_dq' must have mask_path=None"
                )
        else:
            if self.mask_path is None or not isinstance(self.mask_path, str) or not self.mask_path.strip():
                raise InvalidRequestError(
                    "ManagedMasterRecord with dq_state='source_mask' requires a non-empty mask_path"
                )
        if self.schema_version != MANAGED_LEDGER_SCHEMA:
            raise InvalidRequestError(
                f"ManagedMasterRecord.schema_version unsupported: {self.schema_version!r}"
            )
        for fld, fact in self.evidence.items():
            if not isinstance(fld, str) or not fld:
                raise InvalidRequestError("ManagedMasterRecord.evidence keys must be non-empty strings")
            if not isinstance(fact, EvidenceFact):
                raise InvalidRequestError("ManagedMasterRecord.evidence values must be EvidenceFact")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(self, "role", self.role)
        object.__setattr__(self, "content_sha256", self.content_sha256)
        object.__setattr__(self, "bias_state", self.bias_state)
        object.__setattr__(self, "flat_form", self.flat_form)
        object.__setattr__(self, "dq_state", self.dq_state)
        object.__setattr__(self, "mask_path", self.mask_path)
        object.__setattr__(self, "last_seen_path", self.last_seen_path)

    def to_dict(self) -> Mapping[str, object]:
        decl = self.declaration
        decl_dict = {f.name: (list(getattr(decl, f.name)) if isinstance(getattr(decl, f.name), tuple) else getattr(decl, f.name)) for f in dataclasses.fields(ImportDeclaration)}
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "hdu": self.hdu,
            "bias_state": self.bias_state,
            "flat_form": self.flat_form,
            "declaration": decl_dict,
            "evidence": {fld: dict(fact.to_dict()) for fld, fact in self.evidence.items()},
            "dq_state": self.dq_state,
            "mask_path": self.mask_path,
            "last_seen_path": self.last_seen_path,
            "acquired_at": self.acquired_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "ManagedMasterRecord":
        decl = ImportDeclaration(**dict(d["declaration"]))
        evidence = {
            fld: EvidenceFact.from_dict(fact) for fld, fact in d.get("evidence", {}).items()
        }
        return cls(
            role=d["role"],
            content_sha256=d["content_sha256"],
            size_bytes=d["size_bytes"],
            declaration=decl,
            hdu=d.get("hdu", 0),
            bias_state=d.get("bias_state"),
            flat_form=d.get("flat_form"),
            evidence=evidence,
            dq_state=d.get("dq_state", "source_mask"),
            mask_path=d.get("mask_path"),
            last_seen_path=d.get("last_seen_path"),
            schema_version=d.get("schema_version", MANAGED_LEDGER_SCHEMA),
            acquired_at=d.get("acquired_at"),
        )


@dataclass(frozen=True)
class IndexLibraryResult:
    """Structured ``index_library`` outcome envelope."""

    operation_status: str  # "COMPLETED" | "CANCELLED" | "FAILED"
    revision: Optional[str] = None
    candidate_count: int = 0
    diagnostics: tuple = ()
    reason_code: Optional[str] = None
    details: str = ""

    def __post_init__(self) -> None:
        if self.operation_status not in ("COMPLETED", "CANCELLED", "FAILED"):
            raise InvalidRequestError(
                f"IndexLibraryResult.operation_status invalid: {self.operation_status!r}"
            )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "operation_status": self.operation_status,
            "revision": self.revision,
            "candidate_count": self.candidate_count,
            "diagnostics": [
                dict(d.to_dict()) if hasattr(d, "to_dict") else d for d in self.diagnostics
            ],
            "reason_code": self.reason_code,
            "details": self.details,
        }


__all__ = [
    "BATCH_MANIFEST_SCHEMA",
    "CAPABILITIES",
    "DQ_STATES",
    "DqState",
    "EVIDENCE_ORIGIN_TYPES",
    "EvidenceFact",
    "MANAGED_LEDGER_SCHEMA",
    "ManagedMasterRecord",
    "PROVENANCE_SCHEMA",
    "Acquisition",
    "AcquisitionProfileEvidence",
    "ApiInfo",
    "ArrayFrameSource",
    "ArrayInputIdentity",
    "BatchItem",
    "BatchManifest",
    "BatchOptions",
    "BatchOutputRecord",
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
    "IndexLibraryResult",
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
    "MasterImportSpec",
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
