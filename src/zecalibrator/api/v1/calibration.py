"""Public execution: ``calibrate_frame``.

``calibrate_frame`` validates the plan (semantic + byte identity) exactly as the
validation boundaries require, decodes the light source, verifies/consumes every
master binding's image bytes and external DQ mask, and runs the CPU calibration
pipeline (G3 executor, whose pixel-level types remain private). It returns the
public :class:`CalibrationResult` wrapper with complete in-memory provenance.

All flat forms (raw_response / corrected_unnormalized / normalized_response) are
supported: the facade pre-normalizes a corrected flat and builds a truthful
``NormalizationProof`` for a normalized flat, then reuses the frozen G3
``already_normalized`` execution path (no G3 edit).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from zecalibrator.application import library as _alib
from zecalibrator.application.cancellation import (
    CancellationToken,
    OperationCancelled,
    ProgressObserver,
)
from zecalibrator.core.calibrate import (
    CalibrationResult as _EngineCalibrationResult,
    FrameQuality,
)
from zecalibrator.core.errors import (
    DecodeError,
    GeometryMismatchError,
    InvalidRequestError,
    PrecisionRefusalError,
)
from zecalibrator.core.metadata import ImportDeclaration, SensorMetadata
from zecalibrator.core.plans import MasterBinding, MatchPolicy
from zecalibrator.io.master_source import FilesystemSource

from . import _io
from .errors import SourceError
from .models import (
    ArrayFrameSource,
    CalibrationPlan,
    CalibrationRequest,
    CalibrationResult,
    ExecutionOptions,
    FitsFrameSource,
    FrameSource,
    MasterDescriptor,
    ProvenanceRecord,
)

OPERATION_ID = "zecalibrator-calibrate-frame"


def _engine_cancelled() -> _EngineCalibrationResult:
    return _EngineCalibrationResult(
        status="CANCELLED",
        data=None,
        mask=None,
        counts=None,
        frame_quality=FrameQuality(saturation_evidence="unknown"),
        precision=None,
        scalars={},
        warnings=("cancelled",),
        reason_code="CANCELLED",
    )


def _engine_failed(reason_code: str, detail: str = "") -> _EngineCalibrationResult:
    return _EngineCalibrationResult(
        status="FAILED",
        data=None,
        mask=None,
        counts=None,
        frame_quality=FrameQuality(saturation_evidence="unknown"),
        precision=None,
        scalars={},
        warnings=(detail,) if detail else (),
        reason_code=reason_code,
    )


def _reconstruct_policy(plan: CalibrationPlan) -> MatchPolicy:
    return MatchPolicy(
        version=plan.versions.matching_policy,
        exposure_tolerance=plan.policy_parameters.exposure_tolerance,
        temperature_tolerance=plan.policy_parameters.temperature_tolerance,
        flat_quality_policy=plan.policy_parameters.flat_quality_policy,
    )


def _wrap(
    engine: _EngineCalibrationResult,
    *,
    plan: CalibrationPlan,
    policy: MatchPolicy,
    request: CalibrationRequest,
    input_identity,
    facts: dict,
    executed: tuple[str, ...],
    flat_notes: dict = None,
) -> CalibrationResult:
    from zecalibrator import _version

    flat_notes = flat_notes or {}
    provenance = ProvenanceRecord(
        operation_id=OPERATION_ID,
        input_identity=input_identity,
        input_dtype=facts.get("input_dtype"),
        input_units=facts.get("input_units"),
        input_scaling=facts.get("input_scaling"),
        roi_extent_evidence=facts.get("roi_extent_evidence"),
        flat_saturation_screening=flat_notes.get("flat_saturation_screening"),
        flat_scalars_origin=flat_notes.get("flat_scalars_origin"),
        master_domain_transforms=flat_notes.get("master_domain_transforms", ()),
        plan=plan,
        policy=policy,
        api_version="1.0",
        product_version=_version.__version__,
        decoder_version="1.0",
        provenance_schema=plan.versions.provenance_schema,
        matching_policy=plan.versions.matching_policy,
        science_contract=plan.versions.science_contract,
        backend="cpu",
        request=request,
        executed_processing=executed,
        scalars=engine.scalars,
        status=engine.status,
        reason_code=engine.reason_code,
        warnings=engine.warnings,
    )
    return CalibrationResult(engine, provenance)


def _decode_light(source: FrameSource, *, token, obs):
    """Decode the light; returns ``(DecodedFrame, identity, facts)``.

    A single acquisition via the private P8-A4 carrier (``obs`` is retained for
    signature compatibility; progress reporting is owned by the caller).
    """
    carrier = _io._decode_light_once(source, token=token)
    return carrier.frame, carrier.identity, dict(carrier.facts)


def _light_matches_plan(light, plan: CalibrationPlan) -> None:
    try:
        decoded_lc = _alib.light_constraints_from_sensor_metadata(light.metadata)
    except _alib.MetadataAdapterError as exc:
        raise InvalidRequestError(f"plan/source mismatch: {exc}") from exc
    if decoded_lc.to_plan_dict() != plan.light_constraints.to_plan_dict():
        raise InvalidRequestError(
            "plan/source mismatch: decoded light metadata does not match plan light_constraints"
        )


def _declaration_from_descriptor(desc: MasterDescriptor) -> ImportDeclaration:
    return ImportDeclaration(
        source=desc.processing_provenance.source,
        identity=desc.descriptor_id,
        version="1.0",
        domain="raw",
        units=desc.physical_units,
        detector_instance_id=desc.detector.detector_instance_id,
        detector_model=desc.detector.detector_model,
        gain=desc.acquisition.gain,
        offset=desc.acquisition.offset,
        readout_mode=desc.acquisition.readout_mode,
        adc_mode=desc.acquisition.adc_mode,
        binning=desc.geometry.binning,
        sensor_dimensions=desc.geometry.sensor_dimensions,
        orientation=desc.geometry.orientation,
        cfa_phase=desc.geometry.cfa_phase,
        roi_origin=desc.geometry.roi_origin,
        exposure_s=desc.acquisition.exposure_s,
        temperature_c=desc.acquisition.temperature_c,
        filter=desc.filter,
        optical_train_id=desc.optical_train_id,
        bias_exposure_max_s=desc.bias_exposure_max_s,
        short_flat_profile=desc.short_flat_profile,
        saturation_limit_adu=desc.acquisition.saturation_limit_adu,
        saturation_evidence=desc.acquisition.saturation_evidence,
    )


def _flat_prep_mode(plan: CalibrationPlan) -> str:
    if plan.request.flat_mode != "apply":
        return "flat_dark_incl_bias"

    flat_binding = plan.masters.get("flat")
    if flat_binding is None:
        raise InvalidRequestError("flat_mode 'apply' requires a 'flat' master binding")
    ff = flat_binding.role_descriptor.flat_form

    if ff == "raw_response":
        if "flat_dark" in plan.masters:
            if plan.masters["flat_dark"].role_descriptor.bias_state == "removed":
                return "flat_dark_bias_removed"
            return "flat_dark_incl_bias"
        if "bias_flat" in plan.masters:
            return "bias_only_flat"
        raise InvalidRequestError("raw_response flat requires flat_dark or bias_flat dependency")
    if ff == "normalized_response":
        return "already_normalized"
    if ff == "corrected_unnormalized":
        return "normalize_only"
    raise InvalidRequestError(f"unsupported flat_form: {ff!r}")


def _flat_metadata_from_descriptor(desc: MasterDescriptor, units: str) -> SensorMetadata:
    return SensorMetadata(
        original_cards=(),
        normalized={},
        conflicts=(),
        geometry=desc.geometry,
        raw_domain_declaration="processed" if units == "dimensionless" else "raw",
        units=units,
        declaration=None,
        exposure_s=desc.acquisition.exposure_s,
        temperature_c=desc.acquisition.temperature_c,
        gain=desc.acquisition.gain,
        offset=desc.acquisition.offset,
        readout_mode=desc.acquisition.readout_mode,
        adc_mode=desc.acquisition.adc_mode,
        filter=desc.filter,
        detector_model=desc.detector.detector_model,
        detector_instance_id=desc.detector.detector_instance_id,
        optical_train_id=desc.optical_train_id,
        saturation_limit_adu=desc.acquisition.saturation_limit_adu,
        saturation_evidence=desc.acquisition.saturation_evidence,
    )


def _normalization_proof_from_descriptor(desc: MasterDescriptor):
    from zecalibrator.application.executor import NormalizationProof

    norm = desc.processing_provenance.normalization
    scalars = desc.normalization_scalars
    if norm is None or scalars is None:
        raise InvalidRequestError("normalized_response flat requires normalization provenance")
    if scalars.kind == "mono":
        scalar_map = {"mono": scalars.mono}
    else:
        scalar_map = {"G1": scalars.g1, "R": scalars.r, "B": scalars.b, "G2": scalars.g2}
    plane_counts = dict(desc.validity_evidence.valid_normalization_count or {})
    return NormalizationProof(
        algorithm=desc.normalization_algorithm,
        version="declared-1.0",
        population=norm.population,
        scalars=scalar_map,
        plane_counts=plane_counts,
        quality_policy_state=desc.validity_evidence.quality_policy_state,
    )


def _normalization_proof_from_norm(norm):
    from zecalibrator.application.executor import NormalizationProof

    population = "mono-valid" if norm.cfa_phase == "mono" else "cfa-4-plane"
    algorithm = "median" if norm.cfa_phase == "mono" else "cfa-median-per-plane"
    scalars = {k: v for k, v in norm.scalars.items() if v is not None}
    return NormalizationProof(
        algorithm=algorithm,
        version="computed-1.0",
        population=population,
        scalars=scalars,
        plane_counts=dict(norm.plane_counts),
        quality_policy_state="qualified",
    )


def _normalize_corrected_flat(desc: MasterDescriptor, decoded):
    from zecalibrator.core.equations import normalize_flat_response

    finc = decoded.data.astype(np.float32)
    finc_invalid = decoded.mask != 0
    cfa_phase = desc.geometry.cfa_phase
    if cfa_phase is None:
        raise InvalidRequestError("flat geometry requires a known CFA phase")
    # A corrected master carries no original acquisition samples; saturation
    # screening against the corrected plane would silently admit genuinely
    # saturated pixels (corrected values are lower than the originals). The
    # facade therefore performs NO saturation screen here and records the
    # unavailability truthfully (M8).
    norm = normalize_flat_response(
        finc,
        cfa_phase=cfa_phase,
        roi_origin=desc.geometry.roi_origin or (0, 0),
        saturation_samples=None,
        saturation_limit=None,
        fcorr_invalid=finc_invalid,
    )
    if not norm.usable:
        raise ValueError(f"corrected flat unusable: planes {tuple(norm.unusable_planes)}")
    return norm


def _new_frame(data, mask, metadata, hdu=0):
    from zecalibrator.io.raw_decoder import DecodedFrame

    return DecodedFrame(
        data=data,
        mask=mask,
        metadata=metadata,
        stored_dtype="float32",
        bscale=1.0,
        bzero=0.0,
        blank=None,
        hdu=hdu,
        precision=None,
    )


# Frozen role -> descriptor master_type map (G3/G4 contract). ``bias_flat`` is a
# bias master used as the flat's additive bias dependency.
_ROLE_MASTER_TYPE = {
    "bias": "bias",
    "dark": "dark",
    "flat": "flat",
    "flat_dark": "flat_dark",
    "bias_flat": "bias",
}


def _semantic_plan_errors(plan: CalibrationPlan) -> list:
    reasons = []
    for role, binding in plan.masters.items():
        desc = binding.role_descriptor
        expected = _ROLE_MASTER_TYPE.get(role)
        if expected is None:
            reasons.append(f"unknown role {role!r}")
        elif desc.master_type != expected:
            reasons.append(
                f"role {role!r}: descriptor master_type {desc.master_type!r} != expected {expected!r}"
            )
        if binding.hdu != desc.hdu:
            reasons.append(f"role {role}: binding hdu {binding.hdu!r} != descriptor hdu {desc.hdu!r}")
        for loc in binding.locators:
            if loc.hdu != binding.hdu:
                reasons.append(
                    f"role {role}: locator hdu {loc.hdu!r} != binding hdu {binding.hdu!r}"
                )
    return reasons


def _load_masters(plan: CalibrationPlan, *, token, obs):
    from zecalibrator.application.executor import MasterBinding as ExecutorMasterBinding

    masters: dict = {}
    flat_notes: dict = {}
    domain_transforms: list = []
    for role, binding in plan.masters.items():
        token.raise_if_cancelled()
        try:
            binding.verify_snapshot()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"binding {role}: descriptor snapshot invalid ({exc})") from exc

        desc = binding.role_descriptor
        expected = _ROLE_MASTER_TYPE.get(role)
        if expected is None:
            raise ValueError(f"unknown role {role!r}")
        if desc.master_type != expected:
            raise ValueError(f"binding role {role!r}: descriptor master_type {desc.master_type!r} != expected {expected!r}")
        if not binding.locators:
            raise InvalidRequestError(f"binding {role} has no image locator")
        if binding.mask_locator is None and desc.dq_state != "no_source_dq":
            raise InvalidRequestError(f"binding {role} has no mask locator")
        for loc in binding.locators:
            if loc.hdu != binding.hdu or loc.hdu != desc.hdu:
                raise ValueError(
                    f"binding {role}: locator hdu {loc.hdu!r} does not match binding/descriptor hdu {binding.hdu!r}"
                )

        locator = binding.locators[0]

        image_data = _io.read_bytes(locator.path, cancel=token)
        image_sha = _io.sha256_bytes(image_data, cancel=token)
        if image_sha != binding.content_sha256 or len(image_data) != binding.size_bytes:
            raise ValueError(f"binding {role}: content/size mismatch")

        if binding.mask_locator is None:
            # R1 structural ``no_source_dq``: no source DQ mask payload; the
            # neutral all-valid mask is an execution-only detail (never
            # persisted/represented as source DQ; provenance stays no_source_dq).
            external_mask = np.zeros(desc.geometry.shape, dtype=np.uint16)
        else:
            mask_data = _io.read_bytes(binding.mask_locator.path, cancel=token)
            external_mask = _io.load_mask_payload(
                mask_data, desc.geometry.shape, expected_sha=binding.mask_identity
            )

        if desc.master_type == "flat" and desc.flat_form == "normalized_response":
            data32, dq = _io.read_flat_array_bytes(image_data, locator.hdu, cancel=token)
            if tuple(data32.shape) != tuple(desc.geometry.shape):
                raise ValueError(
                    f"binding {role}: decoded shape {tuple(data32.shape)} vs descriptor {tuple(desc.geometry.shape)}"
                )
            if external_mask.shape != data32.shape:
                raise ValueError(f"binding {role}: mask shape mismatch")
            dq.__ior__(external_mask)
            frame = _new_frame(data32, dq, _flat_metadata_from_descriptor(desc, "dimensionless"))
            proof = _normalization_proof_from_descriptor(desc)
            flat_form = "normalized_response"
            flat_notes["flat_saturation_screening"] = "declared"
            flat_notes["flat_scalars_origin"] = "declared"
        elif desc.master_type == "flat" and desc.flat_form == "corrected_unnormalized":
            decoded = _io.decode_fits_from_bytes(
                image_data, locator.hdu, _declaration_from_descriptor(desc), cancel=token, progress=None,
                admission="master", role="flat", flat_form="corrected_unnormalized",
            )
            if tuple(decoded.data.shape) != tuple(desc.geometry.shape):
                raise ValueError(
                    f"binding {role}: decoded shape {tuple(decoded.data.shape)} vs descriptor {tuple(desc.geometry.shape)}"
                )
            if external_mask.shape != decoded.data.shape:
                raise ValueError(f"binding {role}: mask shape mismatch")
            decoded.mask.__ior__(external_mask)
            # R3D-E F2: a producer-proven normalized_real [0,1] flat is lifted
            # ×65535 at input to the canonical ADU-equivalent domain.
            decoded, transform = _io.convert_normalized_real_master(decoded, role=desc.master_type)
            if transform is not None:
                domain_transforms.append(transform)
            # R3D-E F3: the additive flat correction has ALREADY occurred. Keep
            # flat_form="corrected_unnormalized" (no proof) and let the
            # executor's ``normalize_only`` branch normalize the response
            # directly — never re-subtract flat_dark/bias_flat here.
            frame = decoded
            proof = None
            flat_form = "corrected_unnormalized"
            flat_notes["flat_saturation_screening"] = "unavailable"
            flat_notes["flat_scalars_origin"] = "executed"
        else:
            decoded = _io.decode_fits_from_bytes(
                image_data, locator.hdu, _declaration_from_descriptor(desc), cancel=token, progress=None,
                admission="master", role=desc.master_type, flat_form=desc.flat_form,
            )
            if tuple(decoded.data.shape) != tuple(desc.geometry.shape):
                raise ValueError(
                    f"binding {role}: decoded shape {tuple(decoded.data.shape)} vs descriptor {tuple(desc.geometry.shape)}"
                )
            if external_mask.shape != decoded.data.shape:
                raise ValueError(f"binding {role}: mask shape mismatch")
            decoded.mask.__ior__(external_mask)
            # R3D-E F2: producer-proven normalized_real [0,1] additive masters
            # (dark/bias/flat_dark) are lifted ×65535 at input so the dark
            # reaches the same canonical scale as the light before subtraction.
            decoded, transform = _io.convert_normalized_real_master(decoded, role=desc.master_type)
            if transform is not None:
                domain_transforms.append(transform)
            frame = decoded
            proof = None
            flat_form = desc.flat_form
            if desc.master_type == "flat":
                flat_notes["flat_saturation_screening"] = desc.acquisition.saturation_evidence
                flat_notes["flat_scalars_origin"] = "executed"

        masters[role] = ExecutorMasterBinding(
            role=desc.master_type,
            frame=frame,
            bias_state=desc.bias_state,
            flat_form=flat_form,
            normalization_proof=proof,
            short_flat_profile=desc.short_flat_profile,
        )
    flat_notes["master_domain_transforms"] = tuple(domain_transforms)
    return masters, flat_notes


# ---------------------------------------------------------------------------
# Prepared calibration context (private, execution-only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrozenMaster:
    """Immutable decoded master bound to its verified plan identity."""

    role: str
    frame: "DecodedFrame"
    bias_state: str
    flat_form: Optional[str]
    normalization_proof: Optional[object]
    short_flat_profile: bool
    content_sha256: str
    size_bytes: int
    hdu: object
    mask_identity: Optional[str]


@dataclass(frozen=True)
class PreparedFlatOutcome:
    """Plan-invariant prepared-flat result captured once per plan (P8-A3B).

    Carries either the frozen ``_prepare_flat`` return tuple (``ok=True``) or the
    immutable failure description (``ok=False``). The response/validity arrays are
    frozen read-only and ``scalars`` is a read-only mapping *before* the outcome
    enters the context. A captured failure stores only the exception **type** and
    **args** — never an exception instance, traceback, context or cause; the
    executor reconstructs a fresh instance per frame at the same execution point.
    """

    flat_prep_mode: str
    ok: bool
    flat_response: Optional[np.ndarray]
    flat_valid: Optional[np.ndarray]
    scalars: Mapping[str, Optional[float]]
    norm: Optional[object]
    failure_type: Optional[type]
    failure_args: Optional[tuple]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        if self.failure_args is not None:
            object.__setattr__(self, "failure_args", tuple(self.failure_args))


@dataclass(frozen=True)
class PreparedCalibrationContext:
    """Plan-invariant master-side work, prepared once and applied per frame.

    Immutable: ``bindings``/``flat_notes`` are read-only mappings and every
    bound frame's ``data``/``mask`` arrays are frozen (``writeable=False``). The
    context never hands out a mutable reference. ``prepared_flat`` carries the
    light-independent flat response (or its deterministic failure) once per plan.
    """

    plan: CalibrationPlan
    flat_prep_mode: str
    bindings: Mapping[str, FrozenMaster]
    flat_notes: Mapping[str, object]
    context_id: str
    prepared_flat: Optional[PreparedFlatOutcome] = None


class _PreparedContextSlot:
    """Private single-slot holder, keyed by ``plan_id`` (batch-local).

    A successfully prepared context is reused only while its matching plan
    occupies the slot; a miss (different ``plan_id``, an evicted/replaced
    context, or no context) re-prepares. No module-level state, no disk, no
    settings.
    """

    __slots__ = ("_plan_id", "_context")

    def __init__(self) -> None:
        self._plan_id: Optional[str] = None
        self._context: Optional[PreparedCalibrationContext] = None

    def get(self, plan: CalibrationPlan) -> Optional[PreparedCalibrationContext]:
        context = self._context
        if context is None or self._plan_id != plan.plan_id:
            return None
        if not _context_matches_plan(context, plan):
            return None
        return context

    def put(self, plan: CalibrationPlan, context: PreparedCalibrationContext) -> None:
        self._plan_id = plan.plan_id
        self._context = context

    def clear(self) -> None:
        self._plan_id = None
        self._context = None

    @property
    def is_empty(self) -> bool:
        return self._context is None


def _freeze_master(master, binding: MasterBinding) -> FrozenMaster:
    """Freeze a decoded master *after* the load-time mask mutation.

    The load path already performed ``decoded.mask.__ior__(external_mask)``;
    freezing here (as the final preparation step) makes the bound frame arrays
    immutable so a reused context can never be mutated in place.
    """
    frame = master.frame
    frame.data.flags.writeable = False
    frame.mask.flags.writeable = False
    return FrozenMaster(
        role=master.role,
        frame=frame,
        bias_state=master.bias_state,
        flat_form=master.flat_form,
        normalization_proof=master.normalization_proof,
        short_flat_profile=master.short_flat_profile,
        content_sha256=binding.content_sha256,
        size_bytes=binding.size_bytes,
        hdu=binding.hdu,
        mask_identity=binding.mask_identity,
    )


def _context_id(plan: CalibrationPlan, flat_prep_mode: str, bindings) -> str:
    from zecalibrator.core.digests import canonical_json, sha256_hex

    payload = {
        "plan_id": plan.plan_id,
        "flat_prep_mode": flat_prep_mode,
        "bindings": {role: fm.content_sha256 for role, fm in sorted(bindings.items())},
    }
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def _context_matches_plan(
    context: PreparedCalibrationContext, plan: CalibrationPlan
) -> bool:
    """O(1) binding-identity guard: never silently reuse a mismatched context."""
    if context.plan.plan_id != plan.plan_id:
        return False
    if set(context.bindings) != set(plan.masters):
        return False
    for role, fm in context.bindings.items():
        binding = plan.masters[role]
        if (
            fm.content_sha256 != binding.content_sha256
            or fm.size_bytes != binding.size_bytes
            or fm.hdu != binding.hdu
            or fm.mask_identity != binding.mask_identity
        ):
            return False
    return True


def _build_prepared_flat(plan, flat_prep_mode, masters):
    """Eagerly prepare the light-independent flat response once per plan.

    Only for ``flat_mode == "apply"`` with a ``flat`` master; otherwise returns
    ``None`` (the executor's per-frame ``_prepare_flat`` path applies). A
    deterministic preparation failure is captured as the outcome rather than
    allowed to escape at context-build time: ``InvalidRequestError`` and
    ``GeometryMismatchError`` are the only types ``_prepare_flat`` can raise, and
    both are re-emitted per frame at the identical execution point by the
    executor seam (preserving today's precedence and per-frame FAILED mapping).
    """
    if plan.request.flat_mode != "apply" or "flat" not in masters:
        return None
    from zecalibrator.application.executor import _prepare_flat

    flat_m = masters["flat"]
    try:
        flat_response, flat_valid, scalars, norm = _prepare_flat(flat_m, masters, flat_prep_mode)
    except (InvalidRequestError, GeometryMismatchError) as exc:
        # Store only immutable failure facts. ``GeometryMismatchError``
        # reconstructs from (reason_code, fields) — NOT from ``exc.args`` (which
        # is the flattened message) — while ``InvalidRequestError`` reconstructs
        # from ``exc.args`` (its message). Both reconstruct byte-identically via
        # ``raise failure_type(*failure_args)`` at the executor seam.
        if isinstance(exc, GeometryMismatchError):
            failure_args = (exc.reason_code, exc.fields)
        else:
            failure_args = exc.args
        return PreparedFlatOutcome(
            flat_prep_mode=flat_prep_mode,
            ok=False,
            flat_response=None,
            flat_valid=None,
            scalars=MappingProxyType({}),
            norm=None,
            failure_type=type(exc),
            failure_args=failure_args,
        )
    flat_response.flags.writeable = False
    flat_valid.flags.writeable = False
    return PreparedFlatOutcome(
        flat_prep_mode=flat_prep_mode,
        ok=True,
        flat_response=flat_response,
        flat_valid=flat_valid,
        scalars=MappingProxyType(dict(scalars)),
        norm=norm,
        failure_type=None,
        failure_args=None,
    )


def _prepare_calibration_context(plan: CalibrationPlan, *, token):
    """Prepare the plan-invariant master-side work exactly once.

    Returns ``(context, failure)``: ``context`` is a
    :class:`PreparedCalibrationContext` on success (``None`` on failure) and
    ``failure`` is the typed exception on failure (``None`` on success). Only
    success is ever cached; a failure returns ``(None, exc)`` so the caller's
    miss path re-runs this exact function (today's ``_load_masters`` semantics,
    including transient-failure retry). ``InvalidRequestError`` from
    ``_flat_prep_mode`` propagates unchanged, exactly as today.
    """
    # Computed outside the master-load try/except so its InvalidRequestError
    # propagates (matching the pre-refactor call order).
    flat_prep_mode = _flat_prep_mode(plan)
    try:
        masters, flat_notes = _load_masters(plan, token=token, obs=None)
    except (
        OperationCancelled,
        DecodeError,
        PrecisionRefusalError,
        SourceError,
        OSError,
        ValueError,
    ) as exc:
        return None, exc

    prepared_flat = _build_prepared_flat(plan, flat_prep_mode, masters)
    bindings = MappingProxyType(
        {role: _freeze_master(master, plan.masters[role]) for role, master in masters.items()}
    )
    context = PreparedCalibrationContext(
        plan=plan,
        flat_prep_mode=flat_prep_mode,
        bindings=bindings,
        flat_notes=MappingProxyType(dict(flat_notes)),
        context_id=_context_id(plan, flat_prep_mode, bindings),
        prepared_flat=prepared_flat,
    )
    return context, None


def _prepare_or_reuse(plan: CalibrationPlan, slot, *, token):
    if slot is not None:
        context = slot.get(plan)
        if context is not None:
            return context, None
    context, failure = _prepare_calibration_context(plan, token=token)
    if failure is not None:
        return None, failure
    if slot is not None:
        slot.put(plan, context)
    return context, None


def _to_executor_masters(bindings):
    from zecalibrator.application.executor import MasterBinding as ExecutorMasterBinding

    return {
        role: ExecutorMasterBinding(
            role=fm.role,
            frame=fm.frame,
            bias_state=fm.bias_state,
            flat_form=fm.flat_form,
            normalization_proof=fm.normalization_proof,
            short_flat_profile=fm.short_flat_profile,
        )
        for role, fm in bindings.items()
    }


def _wrap_prepare_failure(
    failure, *, plan, policy, request, input_identity, facts
):
    executed = ("decode", "validate_plan", "load_masters")
    if isinstance(failure, OperationCancelled):
        return _wrap(
            _engine_cancelled(), plan=plan, policy=policy, request=request,
            input_identity=input_identity, facts=facts, executed=executed,
        )
    if isinstance(failure, (DecodeError, PrecisionRefusalError)):
        reason = getattr(failure, "reason_code", None) or "DECODE_ERROR"
        return _wrap(
            _engine_failed(reason, str(failure)), plan=plan, policy=policy,
            request=request, input_identity=input_identity, facts=facts,
            executed=executed,
        )
    return _wrap(
        _engine_failed("MASTER_LOAD_FAILED", str(failure)), plan=plan, policy=policy,
        request=request, input_identity=input_identity, facts=facts,
        executed=executed,
    )


def _decode_light_for_plan(source, plan, *, policy, request, token, obs, decoded_light=None):
    """Decode the light (or reuse a P8-A4 carrier) and match it to the plan.

    Returns ``(failure, light, input_identity, facts)`` where ``failure`` is a
    wrapped :class:`CalibrationResult` on decode failure (or ``None`` on
    success); ``light``/``input_identity``/``facts`` are ``None`` on failure.
    ``decoded_light`` (private) carries the single acquisition already performed
    by the batch/auto-route inspection stage; when absent the light is decoded
    once here (standalone ``calibrate_frame`` behaviour, unchanged).
    """
    _io.emit_progress(obs, OPERATION_ID, "decode", 1, 5)
    if decoded_light is not None:
        light = decoded_light.frame
        input_identity = decoded_light.identity
        facts = dict(decoded_light.facts)
    else:
        try:
            light, input_identity, facts = _decode_light(source, token=token, obs=obs)
        except OperationCancelled:
            return _wrap(
                _engine_cancelled(), plan=plan, policy=policy, request=request,
                input_identity=None, facts={}, executed=("decode",),
            ), None, None, None
        except DecodeError as exc:
            return _wrap(
                _engine_failed(exc.reason_code, str(exc)), plan=plan, policy=policy,
                request=request, input_identity=None, facts={}, executed=("decode",),
            ), None, None, None
        except PrecisionRefusalError as exc:
            return _wrap(
                _engine_failed("PRECISION_REFUSAL", str(exc)), plan=plan, policy=policy,
                request=request, input_identity=None, facts={}, executed=("decode",),
            ), None, None, None
        except (SourceError, OSError, ValueError) as exc:
            return _wrap(
                _engine_failed("SOURCE_ERROR", str(exc)), plan=plan, policy=policy,
                request=request, input_identity=None, facts={}, executed=("decode",),
            ), None, None, None

    _light_matches_plan(light, plan)
    return None, light, input_identity, facts


def _execute_prepared(light, context, plan, input_identity, facts, *, token, obs):
    request = plan.request
    policy = _reconstruct_policy(plan)
    masters = _to_executor_masters(context.bindings)

    try:
        from zecalibrator.application.executor import (
            CalibrationRequest as ExecutorRequest,
            execute_calibration,
        )

        engine = execute_calibration(
            light,
            ExecutorRequest(additive_mode=request.additive_mode, flat_mode=request.flat_mode),
            masters,
            flat_prep_mode=context.flat_prep_mode,
            cancel=token,
            progress=None,  # facade owns the monotonic coordinator
            operation_id=OPERATION_ID,
            prepared_flat_outcome=context.prepared_flat,
        )
    except OperationCancelled:
        return _wrap(
            _engine_cancelled(), plan=plan, policy=policy, request=request,
            input_identity=input_identity, facts=facts,
            executed=("decode", "validate_plan", "load_masters", "calibrate"),
        )
    except GeometryMismatchError as exc:
        return _wrap(
            _engine_failed(exc.reason_code, str(exc)), plan=plan, policy=policy,
            request=request, input_identity=input_identity, facts=facts,
            executed=("decode", "validate_plan", "load_masters", "calibrate"),
        )

    token.raise_if_cancelled()
    result = _wrap(
        engine,
        plan=plan,
        policy=policy,
        request=request,
        input_identity=input_identity,
        facts=facts,
        executed=("decode", "validate_plan", "load_masters", "calibrate"),
        flat_notes=dict(context.flat_notes),
    )
    _io.emit_progress(obs, OPERATION_ID, "complete", 5, 5)
    return result


def _apply_prepared_context(source, context, options, *, token, progress):
    """Apply a prepared context to one light source (decode + calibrate + wrap).

    Private seam: ``prepare + apply`` reproduces ``calibrate_frame`` exactly
    (same science, statuses, reason codes, warnings, scalars, provenance).
    """
    obs = _io.normalize_progress(progress, OPERATION_ID)
    plan = context.plan
    request = plan.request
    policy = _reconstruct_policy(plan)

    failure, light, input_identity, facts = _decode_light_for_plan(
        source, plan, policy=policy, request=request, token=token, obs=obs
    )
    if failure is not None:
        return failure
    return _execute_prepared(light, context, context.plan, input_identity, facts, token=token, obs=obs)


def _calibrate_frame_impl(source, plan, options, *, token, obs, slot, decoded_light=None):
    """Single per-frame execution path shared by ``calibrate_frame`` and batch.

    Preserves today's exact order and failure precedence: pre-cancel →
    verify_plan_id → validate_plan → decode → light_matches_plan → master stage
    (prepare once / reuse) → execute. ``slot`` is an optional batch-local
    single-slot holder; ``calibrate_frame`` passes ``None`` (prepare every time).
    ``decoded_light`` (private) is the P8-A4 single-acquisition carrier from the
    batch/auto-route inspection stage; when ``None`` this path decodes once
    itself (standalone behaviour, unchanged).
    """
    request = plan.request
    policy = _reconstruct_policy(plan)

    if token.is_cancelled():
        return _wrap(
            _engine_cancelled(), plan=plan, policy=policy, request=request,
            input_identity=None, facts={}, executed=(),
        )

    try:
        plan.verify_plan_id()
    except ValueError as exc:
        raise InvalidRequestError(f"plan digest mismatch: {exc}") from exc

    semantic = _semantic_plan_errors(plan)
    if semantic:
        return _wrap(
            _engine_failed("PLAN_INVALID", "; ".join(semantic)),
            plan=plan, policy=policy, request=request, input_identity=None, facts={},
            executed=("validate_plan",),
        )
    validation = _alib.validate_plan(plan, FilesystemSource())
    if validation.status == "FAILED":
        return _wrap(
            _engine_failed("PLAN_INVALID", "; ".join(validation.reasons)),
            plan=plan, policy=policy, request=request, input_identity=None, facts={},
            executed=("validate_plan",),
        )

    failure, light, input_identity, facts = _decode_light_for_plan(
        source, plan, policy=policy, request=request, token=token, obs=obs,
        decoded_light=decoded_light,
    )
    if failure is not None:
        return failure

    _io.emit_progress(obs, OPERATION_ID, "load_masters", 3, 5)
    context, failure = _prepare_or_reuse(plan, slot, token=token)
    if failure is not None:
        return _wrap_prepare_failure(
            failure, plan=plan, policy=policy, request=request,
            input_identity=input_identity, facts=facts,
        )

    return _execute_prepared(light, context, plan, input_identity, facts, token=token, obs=obs)


def calibrate_frame(
    source: FrameSource,
    plan: CalibrationPlan,
    options: ExecutionOptions = None,
    *,
    cancel=None,
    progress=None,
) -> CalibrationResult:
    """Decode ``source``, revalidate the plan, and run CPU calibration.

    Parameter failures (bad source/plan/options, plan digest tamper, plan/source
    mismatch, unsupported flat form) raise :class:`InvalidRequestError`.
    Anticipated operational failures return a structured public
    :class:`CalibrationResult` with complete provenance.
    """
    if not isinstance(source, (FitsFrameSource, ArrayFrameSource)):
        raise InvalidRequestError("source must be a FitsFrameSource or ArrayFrameSource")
    if not isinstance(plan, CalibrationPlan):
        raise InvalidRequestError("plan must be a CalibrationPlan")
    options = options if options is not None else ExecutionOptions()
    if not isinstance(options, ExecutionOptions):
        raise InvalidRequestError("options must be ExecutionOptions")

    token = cancel or CancellationToken()
    obs = _io.normalize_progress(progress, OPERATION_ID)
    return _calibrate_frame_impl(source, plan, options, token=token, obs=obs, slot=None)


__all__ = ["calibrate_frame"]
