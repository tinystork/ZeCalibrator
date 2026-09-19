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

from dataclasses import replace

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
from zecalibrator.core.plans import MatchPolicy
from zecalibrator.io.master_source import FilesystemSource

from . import _io
from .errors import SourceError
from .models import (
    ArrayFrameSource,
    ArrayInputIdentity,
    CalibrationPlan,
    CalibrationRequest,
    CalibrationResult,
    ExecutionOptions,
    FitsFrameSource,
    FitsInputIdentity,
    FrameSource,
    MasterDescriptor,
    PROVENANCE_SCHEMA,
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
        plan=plan,
        policy=policy,
        api_version="1.0",
        product_version=_version.__version__,
        decoder_version="1.0",
        provenance_schema=PROVENANCE_SCHEMA,
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
    """Decode the light; returns ``(DecodedFrame, identity, facts)``."""
    if isinstance(source, ArrayFrameSource):
        decoded = _io.array_source_to_decoded(source, cancel=token)
        actual_digest = _io.verify_identity_digest(source.identity, decoded.data, decoded.mask)
        identity = ArrayInputIdentity(
            caller_logical_id=source.identity.caller_logical_id,
            decoded_digest=actual_digest,
        )
        facts = {
            "input_dtype": str(np.asarray(source.data).dtype),
            "input_units": source.metadata.units,
            "input_scaling": {"scaling_applied": True, "domain": source.domain},
            "roi_extent_evidence": None,
        }
        return decoded, identity, facts

    data = _io.read_bytes(source.path, cancel=token)
    whole_fits_sha256 = _io.sha256_bytes(data)
    decoded = _io.decode_fits_from_bytes(
        data, source.hdu, source.declaration, cancel=token, progress=None
    )
    metadata = _io.apply_roi_extent(decoded.metadata, source.roi_extent, decoded.data.shape)
    decoded = replace(decoded, metadata=metadata)
    from zecalibrator.core.digests import science_digest

    identity = FitsInputIdentity(
        path=str(source.path),
        hdu=decoded.hdu,
        whole_fits_sha256=whole_fits_sha256,
        decoded_digest=science_digest(decoded.data, decoded.mask),
    )
    facts = {
        "input_dtype": decoded.stored_dtype,
        "input_units": decoded.metadata.units,
        "input_scaling": {"bscale": decoded.bscale, "bzero": decoded.bzero},
        "roi_extent_evidence": source.roi_extent,
    }
    return decoded, identity, facts


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
    if ff in ("normalized_response", "corrected_unnormalized"):
        return "already_normalized"
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
                image_data, locator.hdu, _declaration_from_descriptor(desc), cancel=token, progress=None
            )
            if tuple(decoded.data.shape) != tuple(desc.geometry.shape):
                raise ValueError(
                    f"binding {role}: decoded shape {tuple(decoded.data.shape)} vs descriptor {tuple(desc.geometry.shape)}"
                )
            if external_mask.shape != decoded.data.shape:
                raise ValueError(f"binding {role}: mask shape mismatch")
            decoded.mask.__ior__(external_mask)
            norm = _normalize_corrected_flat(desc, decoded)
            frame = _new_frame(norm.R, decoded.mask, _flat_metadata_from_descriptor(desc, "dimensionless"))
            proof = _normalization_proof_from_norm(norm)
            flat_form = "normalized_response"
            flat_notes["flat_saturation_screening"] = "unavailable"
            flat_notes["flat_scalars_origin"] = "executed"
        else:
            decoded = _io.decode_fits_from_bytes(
                image_data, locator.hdu, _declaration_from_descriptor(desc), cancel=token, progress=None
            )
            if tuple(decoded.data.shape) != tuple(desc.geometry.shape):
                raise ValueError(
                    f"binding {role}: decoded shape {tuple(decoded.data.shape)} vs descriptor {tuple(desc.geometry.shape)}"
                )
            if external_mask.shape != decoded.data.shape:
                raise ValueError(f"binding {role}: mask shape mismatch")
            decoded.mask.__ior__(external_mask)
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
    return masters, flat_notes


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
    request = plan.request
    policy = _reconstruct_policy(plan)

    if token.is_cancelled():
        return _wrap(_engine_cancelled(), plan=plan, policy=policy, request=request, input_identity=None, facts={}, executed=())

    try:
        plan.verify_plan_id()
    except ValueError as exc:
        raise InvalidRequestError(f"plan digest mismatch: {exc}") from exc

    # Validation parity: semantic (role/master_type/hdu) + byte identity.
    semantic = _semantic_plan_errors(plan)
    if semantic:
        return _wrap(
            _engine_failed("PLAN_INVALID", "; ".join(semantic)),
            plan=plan, policy=policy, request=request, input_identity=None, facts={}, executed=("validate_plan",),
        )
    validation = _alib.validate_plan(plan, FilesystemSource())
    if validation.status == "FAILED":
        return _wrap(
            _engine_failed("PLAN_INVALID", "; ".join(validation.reasons)),
            plan=plan, policy=policy, request=request, input_identity=None, facts={}, executed=("validate_plan",),
        )

    _io.emit_progress(obs, OPERATION_ID, "decode", 1, 5)

    try:
        light, input_identity, facts = _decode_light(source, token=token, obs=obs)
    except OperationCancelled:
        return _wrap(_engine_cancelled(), plan=plan, policy=policy, request=request, input_identity=None, facts={}, executed=("decode",))
    except DecodeError as exc:
        return _wrap(_engine_failed(exc.reason_code, str(exc)), plan=plan, policy=policy, request=request, input_identity=None, facts={}, executed=("decode",))
    except PrecisionRefusalError as exc:
        return _wrap(_engine_failed("PRECISION_REFUSAL", str(exc)), plan=plan, policy=policy, request=request, input_identity=None, facts={}, executed=("decode",))
    except (SourceError, OSError, ValueError) as exc:
        return _wrap(_engine_failed("SOURCE_ERROR", str(exc)), plan=plan, policy=policy, request=request, input_identity=None, facts={}, executed=("decode",))

    _light_matches_plan(light, plan)
    flat_prep_mode = _flat_prep_mode(plan)

    _io.emit_progress(obs, OPERATION_ID, "load_masters", 3, 5)
    try:
        masters, flat_notes = _load_masters(plan, token=token, obs=obs)
    except OperationCancelled:
        return _wrap(_engine_cancelled(), plan=plan, policy=policy, request=request, input_identity=input_identity, facts=facts, executed=("decode", "validate_plan", "load_masters"))
    except (DecodeError, PrecisionRefusalError) as exc:
        reason = getattr(exc, "reason_code", None) or "DECODE_ERROR"
        return _wrap(_engine_failed(reason, str(exc)), plan=plan, policy=policy, request=request, input_identity=input_identity, facts=facts, executed=("decode", "validate_plan", "load_masters"))
    except (SourceError, OSError, ValueError) as exc:
        return _wrap(_engine_failed("MASTER_LOAD_FAILED", str(exc)), plan=plan, policy=policy, request=request, input_identity=input_identity, facts=facts, executed=("decode", "validate_plan", "load_masters"))

    try:
        from zecalibrator.application.executor import (
            CalibrationRequest as ExecutorRequest,
            execute_calibration,
        )

        engine = execute_calibration(
            light,
            ExecutorRequest(additive_mode=request.additive_mode, flat_mode=request.flat_mode),
            masters,
            flat_prep_mode=flat_prep_mode,
            cancel=token,
            progress=None,  # facade owns the monotonic coordinator
            operation_id=OPERATION_ID,
        )
    except OperationCancelled:
        return _wrap(_engine_cancelled(), plan=plan, policy=policy, request=request, input_identity=input_identity, facts=facts, executed=("decode", "validate_plan", "load_masters", "calibrate"))
    except GeometryMismatchError as exc:
        return _wrap(_engine_failed(exc.reason_code, str(exc)), plan=plan, policy=policy, request=request, input_identity=input_identity, facts=facts, executed=("decode", "validate_plan", "load_masters", "calibrate"))

    token.raise_if_cancelled()
    result = _wrap(
        engine,
        plan=plan,
        policy=policy,
        request=request,
        input_identity=input_identity,
        facts=facts,
        executed=("decode", "validate_plan", "load_masters", "calibrate"),
        flat_notes=flat_notes,
    )
    _io.emit_progress(obs, OPERATION_ID, "complete", 5, 5)
    return result


__all__ = ["calibrate_frame"]
