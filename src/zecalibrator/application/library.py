"""Library snapshot/handle protocol, resolve service and revalidation orchestration.

This is the application-layer boundary that adapts the G3 immutable
:class:`~zecalibrator.core.metadata.SensorMetadata` into the pure
:class:`~zecalibrator.core.descriptors.LightConstraints` the matcher consumes,
and wraps the pure matcher into a structured decision envelope that retains
input identity, request/policy, library revision/snapshot membership, candidate
accept/reject/ambiguity reasons and explicit selection decisions.

It does **not** import the G3 ``application.executor`` (which owns decoded
pixels) nor the ``io`` package; filesystem identity/revalidation is orchestrated
against an injected source adapter (e.g. :mod:`zecalibrator.io.master_source`).

Two resolution paths are explicit:

* ``resolve_calibration`` — **metadata-only** (no filesystem, no hashing): it
  never claims verified file-backed planning.
* ``plan_calibration_file_backed`` — metadata resolution **plus** verified
  image/mask identity at planning through an injected source.

``validate_binding`` / ``validate_plan`` are standalone boundaries usable after a
library handle has been closed. A plan is never a claim that any calibration has
executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple, Union

from zecalibrator.core.descriptors import (
    Acquisition,
    AcquisitionProfileEvidence,
    DescriptorSnapshot,
    DetectorIdentity,
    LightConstraints,
    LightEvidence,
    MasterDescriptor,
    OpticalIdentity,
)
from zecalibrator.core.matching import MatchResult, Reason, RejectionRecord, match_calibration
from zecalibrator.core.metadata import SensorMetadata
from zecalibrator.core.plans import (
    CalibrationPlan,
    CalibrationRequest,
    Candidate,
    FitsFileLocator,
    FlatQualityPolicy,
    MaskPayloadLocator,
    MatchPolicy,
    Tolerance,
)

LIBRARY_SCHEMA = "zecalibrator.library.v1"


class MetadataAdapterError(ValueError):
    """The G3 SensorMetadata cannot be adapted (unresolved conflicts/units)."""


class LibraryClosedError(Exception):
    """A resolve attempt on a closed library handle."""


@dataclass(frozen=True)
class FitsInputIdentity:
    """Exact FITS input identity (no fake path/HDU; whole-FITS hash only when computable)."""

    path: str
    hdu: object  # int | str
    whole_fits_sha256: Optional[str] = None
    decoded_digest: str = ""


@dataclass(frozen=True)
class ArrayInputIdentity:
    """Array input identity: caller logical id + decoded-data digest (no FITS identity)."""

    caller_logical_id: str
    decoded_digest: str = ""


InputIdentity = Union[FitsInputIdentity, ArrayInputIdentity]


def _input_identity_to_dict(ident) -> Mapping[str, object]:
    if ident is None:
        return {"kind": None}
    if isinstance(ident, FitsInputIdentity):
        return {"kind": "fits", "path": ident.path, "hdu": ident.hdu, "whole_fits_sha256": ident.whole_fits_sha256, "decoded_digest": ident.decoded_digest}
    return {"kind": "array", "caller_logical_id": ident.caller_logical_id, "decoded_digest": ident.decoded_digest}


def _input_identity_from_dict(d) -> Optional[InputIdentity]:
    kind = d.get("kind")
    if kind == "fits":
        return FitsInputIdentity(path=d["path"], hdu=d["hdu"], whole_fits_sha256=d.get("whole_fits_sha256"), decoded_digest=d.get("decoded_digest", ""))
    if kind == "array":
        return ArrayInputIdentity(caller_logical_id=d["caller_logical_id"], decoded_digest=d.get("decoded_digest", ""))
    return None


def _candidate_to_dict(c: Candidate) -> Mapping[str, object]:
    return {
        "candidate_id": c.candidate_id,
        "descriptor_snapshot": dict(c.descriptor_snapshot.to_dict()),
        "locators": [l.to_dict() for l in c.locators],
        "mask_locator": c.mask_locator.to_dict() if c.mask_locator is not None else None,
    }


def _candidate_from_dict(d: Mapping[str, object]) -> Candidate:
    snap = DescriptorSnapshot.from_dict(d["descriptor_snapshot"])
    return Candidate(
        candidate_id=d["candidate_id"],
        descriptor=snap.descriptor,
        descriptor_snapshot=snap,
        locators=tuple(FitsFileLocator(path=l["path"], hdu=l["hdu"]) for l in d.get("locators", ())),
        mask_locator=MaskPayloadLocator(path=d["mask_locator"]["path"]) if d.get("mask_locator") else None,
    )


def _reason_from_dict(d: Mapping[str, object]) -> Reason:
    return Reason(code=d["code"], field=d.get("field"), role=d.get("role"), parent=d.get("parent"), expected=d.get("expected"), observed=d.get("observed"))


def _rejection_from_dict(d: Mapping[str, object]) -> RejectionRecord:
    return RejectionRecord(
        candidate_id=d["candidate_id"],
        role=d["role"],
        reasons=tuple(_reason_from_dict(r) for r in d.get("reasons", ())),
        descriptor_snapshot=DescriptorSnapshot.from_dict(d["descriptor_snapshot"]) if d.get("descriptor_snapshot") else None,
    )


def _policy_from_dict(d: Mapping[str, object]) -> MatchPolicy:
    return MatchPolicy(
        version=d["version"],
        exposure_tolerance=Tolerance(**d["exposure_tolerance"]),
        temperature_tolerance=Tolerance(**d["temperature_tolerance"]),
        flat_quality_policy=FlatQualityPolicy(**d["flat_quality_policy"]),
    )


@dataclass(frozen=True)
class LibrarySnapshot:
    """An immutable, frozen library snapshot: metadata only, never loaded pixels.

    ``revision`` may be ``""`` for an empty/uninitialized library (a well-defined
    empty state). ``candidates`` maps semantic role to an ordered tuple of
    :class:`Candidate`.
    """

    revision: str
    schema_version: str
    candidates: Mapping[str, Tuple[Candidate, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str):
            raise ValueError("LibrarySnapshot.revision must be a string")
        if self.schema_version != LIBRARY_SCHEMA:
            raise ValueError(f"unsupported library schema {self.schema_version!r}")
        frozen = {k: tuple(v) for k, v in self.candidates.items()}
        object.__setattr__(self, "candidates", MappingProxyType(frozen))

    def roles(self) -> Tuple[str, ...]:
        return tuple(sorted(self.candidates.keys()))


@dataclass(frozen=True)
class ValidationResult:
    """Structured standalone binding/plan validation result."""

    status: str  # "VALID" | "FAILED"
    reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True)
class DecisionEnvelope:
    """Full operation decision audit (never only winners)."""

    outcome: str
    library_revision: str
    request: CalibrationRequest
    policy: MatchPolicy
    plan: Optional[CalibrationPlan] = None
    rejected_candidates: Tuple[RejectionRecord, ...] = ()
    reason_codes: Tuple[str, ...] = ()
    reasons: Tuple[Reason, ...] = ()
    coherent_sets: Tuple[Mapping[str, Candidate], ...] = ()
    input_identity: Optional[InputIdentity] = None
    light_constraints: Optional[LightConstraints] = None
    selection_decisions: Tuple[Tuple[str, str], ...] = ()
    verification: Optional[ValidationResult] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected_candidates", tuple(self.rejected_candidates))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "coherent_sets", tuple(MappingProxyType(dict(s)) for s in self.coherent_sets))
        object.__setattr__(self, "selection_decisions", tuple((k, v) for k, v in self.selection_decisions))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": LIBRARY_SCHEMA,
            "outcome": self.outcome,
            "library_revision": self.library_revision,
            "request": dict(self.request.to_dict()),
            "policy": dict(self.policy.to_dict()),
            "plan": dict(self.plan.to_dict()) if self.plan is not None else None,
            "rejected_candidates": [r.to_dict() for r in self.rejected_candidates],
            "reason_codes": list(self.reason_codes),
            "reasons": [r.to_dict() for r in self.reasons],
            "coherent_sets": [
                {role: _candidate_to_dict(c) for role, c in s.items()}
                for s in self.coherent_sets
            ],
            "input_identity": dict(_input_identity_to_dict(self.input_identity)),
            "light_constraints": dict(self.light_constraints.to_full_dict()) if self.light_constraints is not None else None,
            "selection_decisions": [list(kv) for kv in self.selection_decisions],
            "verification": {"status": self.verification.status, "reasons": list(self.verification.reasons)} if self.verification is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "DecisionEnvelope":
        """Deserialize a decision envelope (strict schema + request/policy replay)."""
        if d.get("schema_version") != LIBRARY_SCHEMA:
            raise ValueError(f"unsupported decision envelope schema {d.get('schema_version')!r}")
        plan = CalibrationPlan.from_dict(d["plan"]) if d.get("plan") is not None else None
        return cls(
            outcome=d["outcome"],
            library_revision=d["library_revision"],
            request=CalibrationRequest(d["request"]["additive_mode"], d["request"]["flat_mode"]),
            policy=_policy_from_dict(d["policy"]),
            plan=plan,
            rejected_candidates=tuple(_rejection_from_dict(r) for r in d.get("rejected_candidates", ())),
            reason_codes=tuple(d.get("reason_codes", ())),
            reasons=tuple(_reason_from_dict(r) for r in d.get("reasons", ())),
            coherent_sets=tuple(
                {role: _candidate_from_dict(c) for role, c in s.items()}
                for s in d.get("coherent_sets", ())
            ),
            input_identity=_input_identity_from_dict(d.get("input_identity", {})),
            light_constraints=CalibrationPlan._light_from_dict(d["light_constraints"]) if d.get("light_constraints") is not None else None,
            selection_decisions=tuple(tuple(kv) for kv in d.get("selection_decisions", ())),
            verification=ValidationResult(status=d["verification"]["status"], reasons=tuple(d["verification"].get("reasons", ()))) if d.get("verification") is not None else None,
        )
def light_constraints_from_sensor_metadata(md: SensorMetadata) -> LightConstraints:
    """Adapt G3 ``SensorMetadata`` into pure matching constraints (no reverse import).

    Refuses unresolved conflicts and incompatible units (never erases them into a
    plausible raw constraint); preserves original cards/declaration provenance via
    :class:`LightEvidence`.
    """
    if md.conflicts:
        first = md.conflicts[0]
        raise MetadataAdapterError(f"unresolved metadata conflict: {first.field} {first.keywords} disagree")
    if md.units is not None and md.units != "ADU":
        raise MetadataAdapterError(f"incompatible light units {md.units!r} (raw light requires ADU)")

    decl = md.declaration
    evidence = LightEvidence(
        original_cards=tuple(md.original_cards),
        declaration=decl,
        units=md.units,
    )
    return LightConstraints(
        geometry=md.geometry,
        detector=DetectorIdentity(
            detector_instance_id=md.detector_instance_id or "unknown",
            detector_model=md.detector_model,
            serial=None,
        ),
        acquisition=Acquisition(
            gain=md.gain,
            offset=md.offset,
            readout_mode=md.readout_mode,
            adc_mode=md.adc_mode,
            temperature_c=md.temperature_c,
            exposure_s=md.exposure_s,
            saturation_limit_adu=md.saturation_limit_adu,
            saturation_evidence=md.saturation_evidence,
            bias_exposure_max_s=decl.bias_exposure_max_s if decl is not None else None,
            short_flat_profile=decl.short_flat_profile if decl is not None else False,
        ),
        optical=OpticalIdentity(filter=md.filter, optical_train_id=md.optical_train_id),
        raw_domain_declaration=md.raw_domain_declaration,
        evidence=evidence,
    )


def resolve_calibration(
    light: LightConstraints,
    request: CalibrationRequest,
    snapshot: LibrarySnapshot,
    policy: MatchPolicy,
    *,
    manual_selection: Optional[Mapping[str, str]] = None,
    input_identity: Optional[InputIdentity] = None,
) -> DecisionEnvelope:
    """Run the pure matcher against a frozen library snapshot (metadata only).

    This performs **no** filesystem hashing and must not be reported as verified
    file-backed planning.
    """
    result = match_calibration(light, request, snapshot.candidates, policy, manual_selection=manual_selection)
    selection = tuple(manual_selection.items()) if manual_selection else ()
    return DecisionEnvelope(
        outcome=result.outcome,
        library_revision=snapshot.revision,
        request=request,
        policy=policy,
        plan=result.plan,
        rejected_candidates=result.rejected_candidates,
        reason_codes=result.reason_codes,
        reasons=result.reasons,
        coherent_sets=result.coherent_sets,
        input_identity=input_identity,
        light_constraints=light,
        selection_decisions=selection,
    )


def validate_binding(binding, source) -> ValidationResult:
    """Standalone binding validation against an injected source (usable after
    library closure). Refuses ``VALID`` with no locators/bytes checked."""
    reasons: list[str] = []
    try:
        binding.verify_snapshot()
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"descriptor_snapshot: {exc}")

    if not binding.locators:
        reasons.append("no image locator to verify (zero bytes checked)")
    else:
        for loc in binding.locators:
            try:
                ident = source.image_identity(loc)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"image_identity({loc.path}): {exc}")
                continue
            if ident.content_sha256 != binding.content_sha256:
                reasons.append(f"content mismatch at {loc.path}")
            if ident.size_bytes != binding.size_bytes:
                reasons.append(f"size mismatch at {loc.path}")

    if binding.mask_locator is None:
        reasons.append("no mask locator to verify mask identity")
    else:
        try:
            mask_id = source.mask_identity(binding.mask_locator)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"mask_identity({binding.mask_locator.path}): {exc}")
        else:
            if mask_id != binding.mask_identity:
                reasons.append(f"mask mismatch at {binding.mask_locator.path}")

    return ValidationResult(status="FAILED" if reasons else "VALID", reasons=tuple(reasons))


def validate_plan(plan: CalibrationPlan, source) -> ValidationResult:
    """Validate every binding in a plan against the injected source."""
    reasons: list[str] = []
    for role, binding in sorted(plan.masters.items()):
        r = validate_binding(binding, source)
        if r.status == "FAILED":
            reasons.append(f"role {role}: " + "; ".join(r.reasons))
    return ValidationResult(status="FAILED" if reasons else "VALID", reasons=tuple(reasons))


def plan_calibration_file_backed(
    light: LightConstraints,
    request: CalibrationRequest,
    snapshot: LibrarySnapshot,
    policy: MatchPolicy,
    *,
    source,
    manual_selection: Optional[Mapping[str, str]] = None,
    input_identity: Optional[InputIdentity] = None,
    cancel=None,
    progress=None,
) -> DecisionEnvelope:
    """Metadata resolution **plus** verified image/mask identity at planning."""
    if cancel is not None and cancel.is_cancelled():
        from zecalibrator.application.cancellation import OperationCancelled

        raise OperationCancelled()

    envelope = resolve_calibration(
        light, request, snapshot, policy, manual_selection=manual_selection, input_identity=input_identity
    )
    if envelope.plan is None:
        return envelope

    verification = validate_plan(envelope.plan, source)
    if cancel is not None and cancel.is_cancelled():
        from zecalibrator.application.cancellation import OperationCancelled

        raise OperationCancelled()

    return DecisionEnvelope(
        outcome=envelope.outcome,
        library_revision=envelope.library_revision,
        request=envelope.request,
        policy=envelope.policy,
        plan=envelope.plan if verification.status == "VALID" else None,
        rejected_candidates=envelope.rejected_candidates,
        reason_codes=envelope.reason_codes if verification.status == "VALID" else envelope.reason_codes,
        reasons=envelope.reasons,
        coherent_sets=envelope.coherent_sets,
        input_identity=envelope.input_identity,
        light_constraints=envelope.light_constraints,
        selection_decisions=envelope.selection_decisions,
        verification=verification,
    )


class LibraryHandle:
    """Context-managed read-only library handle. ``close()`` is idempotent."""

    def __init__(self, snapshot: LibrarySnapshot) -> None:
        self._snapshot = snapshot
        self._closed = False

    @property
    def snapshot(self) -> LibrarySnapshot:
        return self._snapshot

    def resolve(
        self,
        light: LightConstraints,
        request: CalibrationRequest,
        policy: MatchPolicy,
        *,
        manual_selection: Optional[Mapping[str, str]] = None,
        input_identity: Optional[InputIdentity] = None,
    ) -> DecisionEnvelope:
        self._ensure_open()
        return resolve_calibration(
            light, request, self._snapshot, policy,
            manual_selection=manual_selection, input_identity=input_identity,
        )

    def revalidate_binding(self, binding, source) -> ValidationResult:
        self._ensure_open()
        return validate_binding(binding, source)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "LibraryHandle":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None

    def _ensure_open(self) -> None:
        if self._closed:
            raise LibraryClosedError("library handle is closed")


__all__ = [
    "ArrayInputIdentity",
    "DecisionEnvelope",
    "FitsInputIdentity",
    "InputIdentity",
    "LibraryClosedError",
    "LibraryHandle",
    "LibrarySnapshot",
    "MetadataAdapterError",
    "ValidationResult",
    "light_constraints_from_sensor_metadata",
    "plan_calibration_file_backed",
    "resolve_calibration",
    "validate_binding",
    "validate_plan",
]
