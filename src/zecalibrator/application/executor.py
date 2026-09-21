"""Single-frame calibration orchestration (Phase 3 / G3 scope).

Bounded orchestration over the pure ``core`` primitives: validates explicit
master semantic role / bias_state / flat_form / normalization proof and exact
detector/geometry/acquisition/thermal/exposure/units compatibility (validation,
*not* automatic matching), prepares the flat response with per-frame saturation
evidence, and runs the additive/division pipeline with cooperative cancellation
and no partial mutation. No producer queues, no scanner, no plan resolver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from zecalibrator.application.cancellation import (
    CancellationToken,
    OperationCancelled,
    ProgressEvent,
    ProgressObserver,
)
from zecalibrator.core.calibrate import (
    STATUS_CANCELLED,
    CalibrationResult,
    FrameQuality,
    calibrate_light,
)
from zecalibrator.core.dq import validate_mask
from zecalibrator.core.equations import flat_correction, normalize_flat_response
from zecalibrator.core.errors import GeometryMismatchError, InvalidRequestError
from zecalibrator.core.geometry import is_bayer_phase
from zecalibrator.core.metadata import SensorMetadata
from zecalibrator.io.raw_decoder import DecodedFrame

TEMP_PARSER_TOLERANCE_C = 1e-6
EXPOSURE_TOLERANCE_REL = 1e-6
EXPOSURE_TOLERANCE_ABS = 1e-6
RESPONSE_FLOOR = 1e-6


@dataclass(frozen=True)
class NormalizationProof:
    """Retained normalization evidence for an already-normalized flat."""

    algorithm: str
    version: str
    population: str  # "mono-valid" | "cfa-4-plane"
    scalars: Mapping[str, float]  # non-empty, finite, positive
    plane_counts: Mapping[str, int]  # non-empty valid-population counts
    quality_policy_state: str = "qualified"

    def __post_init__(self) -> None:
        scalars = dict(self.scalars)
        counts = dict(self.plane_counts)
        if not scalars:
            raise ValueError("normalization proof requires non-empty scalars")
        for name, v in scalars.items():
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
                raise ValueError(f"normalization scalar {name!r} must be finite and positive")
        if not counts:
            raise ValueError("normalization proof requires non-empty plane_counts")
        for name, c in counts.items():
            if not isinstance(c, int) or c < 0:
                raise ValueError(f"plane_count {name!r} must be a non-negative integer")
        if self.quality_policy_state != "qualified":
            raise ValueError("normalization proof quality_policy_state must be 'qualified'")
        if not self.algorithm or not self.version or not self.population:
            raise ValueError("normalization proof requires algorithm/version/population")
        object.__setattr__(self, "scalars", MappingProxyType(scalars))
        object.__setattr__(self, "plane_counts", MappingProxyType(counts))


@dataclass(frozen=True)
class CalibrationRequest:
    """Explicit requested roles; no downgrade and no optional-role flag."""

    additive_mode: str
    flat_mode: str = "none"


@dataclass(frozen=True)
class MasterBinding:
    """An explicitly supplied, already-decoded master frame.

    ``role`` is semantic (bias/dark/flat/flat_dark), never a filename.
    ``bias_state`` is required for darks ("included"/"removed"); "unknown" never
    silently means bias-inclusive. ``flat_form`` is required for flats.
    """

    role: str
    frame: DecodedFrame
    bias_state: str = "unknown"
    flat_form: Optional[str] = None
    normalization_proof: Optional[NormalizationProof] = None
    short_flat_profile: bool = False


def _reasons(*codes: Optional[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(c for c in codes if c is not None))


def _is_finite(v) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _field_reason(light_val, master_val, mismatch_code: str, field: str) -> Optional[str]:
    if light_val is None or master_val is None:
        return "MISSING_REQUIRED_FIELD"
    if light_val != master_val:
        return f"{mismatch_code}: {field}"
    return None


def _unknown_val(v) -> bool:
    return v is None


def _unknown_instance_val(v) -> bool:
    if v is None:
        return True
    return isinstance(v, str) and v.strip() == "unknown"


def _disambiguator_field_reason(light_val, master_val, mismatch_code: str, field: str, *, unknown=_unknown_val) -> Optional[str]:
    """Disambiguator-tier reason (R3B mirror of the matcher).

    Both-unknown is a non-blocking UNVERIFIED note (returns ``None`` — no
    rejection); one-known/one-unknown is a conservative ``MISSING_REQUIRED_FIELD``;
    both-known-different is a blocking ``mismatch_code``.
    """
    lu = unknown(light_val)
    mu = unknown(master_val)
    if lu and mu:
        return None
    if lu or mu:
        return "MISSING_REQUIRED_FIELD"
    if light_val != master_val:
        return f"{mismatch_code}: {field}"
    return None


def _conditional_field_reason(light_val, master_val, mismatch_code: str, field: str, *, necessary: bool, unknown=_unknown_val) -> Optional[str]:
    """CFA-conditional tier: Standard-optional for a Bayer sensor (R3D-G
    executor parity), disambiguator otherwise.

    For a Bayer sensor, ``orientation``/``roi_origin`` are no longer fatal when
    missing (the Standard matcher already arbitrated that unknown as UNVERIFIED);
    only a known both-sides contradiction stays fatal. For a non-Bayer sensor
    they remain disambiguators.
    """
    if necessary:
        if unknown(light_val) or unknown(master_val):
            return None
        if light_val != master_val:
            return f"{mismatch_code}: {field}"
        return None
    return _disambiguator_field_reason(light_val, master_val, mismatch_code, field, unknown=unknown)


def _tiered_geometry_reasons(lg, mg) -> tuple[str, ...]:
    """Tiered geometry reasons: necessary / disambiguator / Standard-optional."""
    reasons: list[str] = []
    if lg.shape != mg.shape:
        reasons.append("GEOMETRY_MISMATCH: geometry.shape")

    # binning — necessary.
    if lg.binning is None or mg.binning is None:
        reasons.append("MISSING_REQUIRED_FIELD: geometry.binning")
    elif lg.binning != mg.binning:
        reasons.append("BINNING_MISMATCH: geometry.binning")
        reasons.append("GEOMETRY_MISMATCH: geometry.binning")

    # sensor_dimensions — disambiguator.
    reasons.append(_disambiguator_field_reason(lg.sensor_dimensions, mg.sensor_dimensions, "GEOMETRY_MISMATCH", "geometry.sensor_dimensions"))

    cfa_applies = is_bayer_phase(lg.cfa_phase) or is_bayer_phase(mg.cfa_phase)

    # orientation — Standard-optional for Bayer (missing no longer fatal),
    # disambiguator otherwise.
    reasons.append(_conditional_field_reason(lg.orientation, mg.orientation, "GEOMETRY_MISMATCH", "geometry.orientation", necessary=cfa_applies))

    # roi_origin — Standard-optional for Bayer (missing no longer fatal; a
    # known both-sides difference flips the CFA parity phase), disambiguator
    # otherwise.
    if cfa_applies:
        if lg.roi_origin is not None and mg.roi_origin is not None and lg.roi_origin != mg.roi_origin:
            reasons.append("ROI_ORIGIN_MISMATCH: geometry.roi_origin")
            reasons.append("CFA_PHASE_MISMATCH: geometry.roi_origin")
    else:
        reasons.append(_disambiguator_field_reason(lg.roi_origin, mg.roi_origin, "ROI_ORIGIN_MISMATCH", "geometry.roi_origin"))

    # roi_extent — optional (redundant with shape): checked only when both
    # known. The executor's master metadata never carries roi_extent (the
    # descriptor->declaration roundtrip drops it), so it must NOT be treated as
    # a disambiguator/necessary tier here (that would reject a binding the
    # matcher already accepted).
    if lg.roi_extent is not None and mg.roi_extent is not None:
        if lg.roi_extent != mg.roi_extent:
            reasons.append("GEOMETRY_MISMATCH: geometry.roi_extent")

    # cfa_phase — necessary.
    if lg.cfa_phase is None or mg.cfa_phase is None:
        reasons.append("MISSING_REQUIRED_FIELD: geometry.cfa_phase")
    elif lg.cfa_phase != mg.cfa_phase:
        reasons.append("CFA_PHASE_MISMATCH: geometry.cfa_phase")

    return _reasons(*reasons)


def _numeric_field_reason(light_val, master_val, mismatch_code: str, field: str) -> Optional[str]:
    """Standard-optional numeric tier (R3D-G): gain/offset missing on either
    side is UNVERIFIED (no reason); only a known both-sides difference is fatal."""
    if light_val is None or master_val is None:
        return None
    if not _is_finite(light_val) or not _is_finite(master_val):
        return None
    if float(light_val) != float(master_val):
        return f"{mismatch_code}: {field}"
    return None


def _temp_reason(lt: Optional[float], mt: Optional[float]) -> Optional[str]:
    """Standard-optional temperature tier (R3D-G): missing on either side is
    UNVERIFIED (no reason); only a known both-sides out-of-tolerance is fatal."""
    if lt is None or mt is None:
        return None
    if not _is_finite(lt) or not _is_finite(mt):
        return None
    if abs(float(lt) - float(mt)) > TEMP_PARSER_TOLERANCE_C:
        return "TEMPERATURE_MISMATCH: acquisition.temperature_c"
    return None


def _exposure_reason(reference: Optional[float], master: Optional[float]) -> Optional[str]:
    if reference is None or master is None:
        return "MISSING_REQUIRED_FIELD: acquisition.exposure_s"
    if not _is_finite(reference) or not _is_finite(master):
        return "MISSING_REQUIRED_FIELD: acquisition.exposure_s"
    t1, t2 = float(reference), float(master)
    if t1 < 0 or t2 < 0:
        return "MISSING_REQUIRED_FIELD: acquisition.exposure_s"  # negative exposure domain
    tol = max(EXPOSURE_TOLERANCE_ABS, EXPOSURE_TOLERANCE_REL * max(abs(t1), abs(t2)))
    if abs(t1 - t2) > tol:
        return "EXPOSURE_MISMATCH: acquisition.exposure_s"
    return None


def _bias_exposure_reason(bias_exposure: Optional[float], max_s: Optional[float]) -> Optional[str]:
    if bias_exposure is None or max_s is None:
        return "MISSING_REQUIRED_FIELD"
    if not _is_finite(bias_exposure) or not _is_finite(max_s) or float(max_s) < 0:
        return "MISSING_REQUIRED_FIELD"
    if float(bias_exposure) < 0:
        return "MISSING_REQUIRED_FIELD"  # negative exposure domain
    if float(bias_exposure) > float(max_s):
        return "EXPOSURE_MISMATCH"
    return None


def _bias_range_for(md: SensorMetadata) -> Optional[float]:
    """Return the declared qualified bias acquisition range for a frame's own
    acquisition profile, or ``None`` when absent (no invented range). The light's
    declaration governs the light's bias; the flat's declaration governs the flat's
    bias dependency (bias_only_flat / flat_dark_bias_removed).
    """
    if md.declaration is None:
        return None
    return md.declaration.bias_exposure_max_s


def _validate_light_source(light: SensorMetadata) -> None:
    if light.raw_domain_declaration != "raw":
        raise InvalidRequestError("light raw_domain_declaration must be 'raw'")
    if light.units != "ADU":
        raise InvalidRequestError("light units must be 'ADU'")
    if light.conflicts:
        raise InvalidRequestError("light metadata contains unresolved conflicts")
    if light.exposure_s is not None and (not _is_finite(light.exposure_s) or float(light.exposure_s) < 0):
        raise InvalidRequestError("light exposure must be non-negative finite")


def _validate_compatibility(
    light: SensorMetadata,
    master: MasterBinding,
    *,
    exposure_reference: Optional[float] = None,
    check_filter: bool = False,
    check_optical: bool = False,
    expected_units: str = "ADU",
    prepared_flat: bool = False,
) -> tuple[str, ...]:
    m = master.frame.metadata
    reasons: list[str] = []

    reasons += _tiered_geometry_reasons(light.geometry, m.geometry)
    reasons.append(_disambiguator_field_reason(light.detector_instance_id, m.detector_instance_id, "DETECTOR_MISMATCH", "detector.detector_instance_id", unknown=_unknown_instance_val))
    reasons.append(_field_reason(light.detector_model, m.detector_model, "DETECTOR_MISMATCH", "detector.detector_model"))

    # R3D-E F4 / R3D-G: a prepared flat (corrected_unnormalized /
    # normalized_response) is already the multiplicative response; its original
    # gain/offset are irrelevant to the light and must not emit
    # GAIN_MISMATCH/OFFSET_MISMATCH. A raw_response flat (and every other role)
    # keeps the gain/offset comparison (Standard-optional, both-known-different
    # stays fatal).
    if not prepared_flat:
        reasons.append(_numeric_field_reason(light.gain, m.gain, "GAIN_MISMATCH", "acquisition.gain"))
        reasons.append(_numeric_field_reason(light.offset, m.offset, "OFFSET_MISMATCH", "acquisition.offset"))

    reasons.append(_disambiguator_field_reason(light.readout_mode, m.readout_mode, "READOUT_MISMATCH", "acquisition.readout_mode"))
    reasons.append(_disambiguator_field_reason(light.adc_mode, m.adc_mode, "ADC_MISMATCH", "acquisition.adc_mode"))

    # Units: the light is always sensor-ADU (validated separately); the master
    # must match the role-specific expectation (ADU for sensor masters,
    # dimensionless for a normalized_response flat).
    if m.units != expected_units:
        reasons.append("MISSING_REQUIRED_FIELD: physical_units")

    # Temperature is relation-scoped like exposure (dark/flat_dark only): a bias
    # is temperature-stable and a flat is normalized, so neither requires it.
    # A prepared flat never passes an exposure_reference, so its temperature is
    # already skipped; guard it anyway for parity clarity.
    if exposure_reference is not None:
        reasons.append(_exposure_reason(exposure_reference, m.exposure_s))
        if not prepared_flat:
            reasons.append(_temp_reason(light.temperature_c, m.temperature_c))

    if check_filter:
        reasons.append(_field_reason(light.filter, m.filter, "FILTER_MISMATCH", "optical.filter"))
    if check_optical:
        reasons.append(_disambiguator_field_reason(light.optical_train_id, m.optical_train_id, "OPTICAL_TRAIN_MISMATCH", "optical.optical_train_id"))

    return _reasons(*reasons)


def _validate_binding(
    light: SensorMetadata,
    master: MasterBinding,
    expected_role: str,
    *,
    exposure_reference: Optional[float] = None,
    check_filter: bool = False,
    check_optical: bool = False,
    expected_units: str = "ADU",
    prepared_flat: bool = False,
) -> None:
    if master.role != expected_role:
        raise GeometryMismatchError(
            "ROLE_UNAVAILABLE",
            (f"expected role {expected_role!r}, got {master.role!r}",),
        )
    reasons = _validate_compatibility(
        light, master,
        exposure_reference=exposure_reference,
        check_filter=check_filter,
        check_optical=check_optical,
        expected_units=expected_units,
        prepared_flat=prepared_flat,
    )
    if reasons:
        raise GeometryMismatchError(reasons[0], reasons)


def _cancelled_result() -> CalibrationResult:
    return CalibrationResult(
        status=STATUS_CANCELLED, data=None, mask=None, counts=None,
        frame_quality=FrameQuality(saturation_evidence="unknown"),
        precision=None, scalars={}, warnings=("cancelled",), reason_code="CANCELLED",
    )


def _flat_unusable_result(norm, saturation_evidence: str) -> CalibrationResult:
    reasons = norm.unusable_planes if norm.unusable_planes else norm.empty_planes
    return CalibrationResult(
        status="FAILED", data=None, mask=None, counts=None,
        frame_quality=FrameQuality(saturation_evidence=saturation_evidence),
        precision=None, scalars=dict(norm.scalars),
        warnings=(f"flat unusable: planes {tuple(reasons)}",),
        reason_code="FLAT_UNUSABLE",
    )


def _qualified_saturation(md: SensorMetadata) -> Optional[float]:
    if md.saturation_evidence != "qualified":
        return None
    limit = md.saturation_limit_adu
    if limit is None or not _is_finite(limit) or float(limit) <= 0:
        raise InvalidRequestError("qualified saturation requires a finite positive limit")
    return float(limit)


def execute_calibration(
    light: DecodedFrame,
    request: CalibrationRequest,
    masters: Mapping[str, MasterBinding],
    *,
    flat_prep_mode: str = "flat_dark_incl_bias",
    cancel: Optional[CancellationToken] = None,
    progress: Optional[ProgressObserver] = None,
    operation_id: str = "zecalibrator-calibration",
    prepared_flat_outcome: object = None,
) -> CalibrationResult:
    """Execute one bounded frame calibration with no partial mutation.

    Saturation evidence comes from each frame's own qualified metadata, never a
    redundant call override. Returns ``CalibrationResult(status=CANCELLED)`` on
    cooperative cancellation.

    ``prepared_flat_outcome`` is an additive private seam (P8-A3B): when the
    caller supplies a plan-invariant prepared-flat outcome, ``_execute`` consumes
    it instead of re-running ``_prepare_flat`` per frame. It is opaque to this
    module (duck-typed); ``None`` (the default) preserves today's per-frame
    ``_prepare_flat`` path exactly.
    """
    if request.additive_mode not in ("control", "bias_only", "dark_incl_bias", "dark_bias_removed"):
        raise InvalidRequestError(f"unsupported additive_mode: {request.additive_mode!r}")
    if request.flat_mode not in ("none", "apply"):
        raise InvalidRequestError(f"unsupported flat_mode: {request.flat_mode!r}")

    token = cancel or CancellationToken()
    obs = progress or ProgressObserver(operation_id=operation_id)

    def emit(phase: str, completed: int, total: Optional[int]) -> None:
        obs.report(
            ProgressEvent(operation_id=operation_id, phase=phase, completed=completed, total=total, unit="pixels")
        )

    try:
        return _execute(
            light, request, masters, flat_prep_mode=flat_prep_mode, token=token, emit=emit,
            prepared_flat_outcome=prepared_flat_outcome,
        )
    except OperationCancelled:
        return _cancelled_result()


def _resolve_light_masters(light, request, masters) -> dict[str, MasterBinding]:
    light_md = light.metadata
    _validate_light_source(light_md)
    bound: dict[str, MasterBinding] = {}
    mode = request.additive_mode
    bias_max = (
        light_md.declaration.bias_exposure_max_s if light_md.declaration is not None else None
    )

    if mode == "bias_only":
        m = masters.get("bias")
        if m is None:
            raise InvalidRequestError("required master role missing: 'bias'")
        _validate_binding(light_md, m, "bias")
        r = _bias_exposure_reason(m.frame.metadata.exposure_s, bias_max)
        if r:
            raise GeometryMismatchError(r, ("bias exposure vs qualified range",))
        bound["bias"] = m
    elif mode == "dark_incl_bias":
        m = masters.get("dark")
        if m is None:
            raise InvalidRequestError("required master role missing: 'dark'")
        if not _is_finite(light_md.exposure_s):
            raise GeometryMismatchError("MISSING_REQUIRED_FIELD", ("light exposure required for dark",))
        _validate_binding(light_md, m, "dark", exposure_reference=light_md.exposure_s)
        if m.bias_state != "included":
            raise GeometryMismatchError("ROLE_UNAVAILABLE", (f"dark requires bias_state='included', got {m.bias_state!r}",))
        bound["dark"] = m
    elif mode == "dark_bias_removed":
        m = masters.get("dark")
        b = masters.get("bias")
        if m is None or b is None:
            raise InvalidRequestError("dark_bias_removed requires 'dark' and 'bias'")
        if not _is_finite(light_md.exposure_s):
            raise GeometryMismatchError("MISSING_REQUIRED_FIELD", ("light exposure required for dark",))
        _validate_binding(light_md, m, "dark", exposure_reference=light_md.exposure_s)
        if m.bias_state != "removed":
            raise GeometryMismatchError("ROLE_UNAVAILABLE", (f"dark requires bias_state='removed', got {m.bias_state!r}",))
        _validate_binding(light_md, b, "bias")
        r = _bias_exposure_reason(b.frame.metadata.exposure_s, bias_max)
        if r:
            raise GeometryMismatchError(r, ("bias exposure vs qualified range",))
        bound["dark"] = m
        bound["bias"] = b
    # control: no additive masters; source already validated.
    return bound


def _validate_proof_coherence(proof: NormalizationProof, cfa_phase: str) -> None:
    """The normalization proof's population/scalars/plane_counts must match the
    flat's declared CFA phase (mono-valid for mono; cfa-4-plane for Bayer)."""
    if cfa_phase == "mono":
        if proof.population != "mono-valid":
            raise InvalidRequestError("mono flat requires population='mono-valid'")
        if set(proof.scalars.keys()) != {"mono"} or set(proof.plane_counts.keys()) != {"mono"}:
            raise InvalidRequestError("mono proof requires 'mono' scalars and plane_counts")
    elif cfa_phase in ("GRBG", "RGGB", "BGGR", "GBRG"):
        if proof.population != "cfa-4-plane":
            raise InvalidRequestError("CFA flat requires population='cfa-4-plane'")
        planes = {"G1", "R", "B", "G2"}
        if set(proof.scalars.keys()) != planes or set(proof.plane_counts.keys()) != planes:
            raise InvalidRequestError("CFA proof requires four labelled scalars/plane_counts (G1/R/B/G2)")
    else:
        raise InvalidRequestError(f"unsupported cfa_phase: {cfa_phase!r}")


def _prepare_flat(
    flat_m: MasterBinding,
    masters: Mapping[str, MasterBinding],
    flat_prep_mode: str,
):
    """Return (flat_response, flat_valid, scalars, norm)."""
    finc = flat_m.frame.data.astype(np.float32)
    finc_invalid = flat_m.frame.mask != 0
    geo = flat_m.frame.metadata.geometry
    cfa_phase = geo.cfa_phase
    if cfa_phase is None:
        raise InvalidRequestError("flat geometry requires a known CFA phase")

    if flat_prep_mode == "already_normalized":
        if flat_m.flat_form != "normalized_response":
            raise InvalidRequestError(
                "flat_prep_mode 'already_normalized' requires flat_form='normalized_response'"
            )
        proof = flat_m.normalization_proof
        if proof is None:
            raise InvalidRequestError("normalized_response flat requires normalization proof")
        _validate_proof_coherence(proof, cfa_phase)
        R = finc
        flat_valid = np.isfinite(R) & (R > 0) & (R > RESPONSE_FLOOR) & ~finc_invalid
        return R, flat_valid, dict(proof.scalars), None

    if flat_prep_mode == "normalize_only":
        # R3D-E F3: the additive flat correction has ALREADY occurred (the flat
        # is the corrected-but-unnormalized multiplicative response). Do NOT call
        # ``flat_correction`` and do NOT consume ``flat_dark``/``bias_flat``;
        # normalize the response directly, with no normalization proof required.
        if flat_m.flat_form != "corrected_unnormalized":
            raise InvalidRequestError(
                "flat_prep_mode 'normalize_only' requires flat_form='corrected_unnormalized'"
            )
        norm = normalize_flat_response(
            finc,
            cfa_phase=cfa_phase,
            roi_origin=geo.roi_origin or (0, 0),
            saturation_samples=None,
            saturation_limit=None,
            fcorr_invalid=finc_invalid,
        )
        return norm.R, norm.valid, dict(norm.scalars), norm

    if flat_prep_mode == "flat_dark_incl_bias":
        if flat_m.flat_form not in ("raw_response", None):
            raise InvalidRequestError(
                "additive flat correction requires a raw_response flat, not a normalized response"
            )
        fd = masters.get("flat_dark")
        if fd is None:
            raise InvalidRequestError("required master role missing: 'flat_dark'")
        _validate_binding(flat_m.frame.metadata, fd, "flat_dark", exposure_reference=flat_m.frame.metadata.exposure_s)
        if fd.bias_state != "included":
            raise GeometryMismatchError("ROLE_UNAVAILABLE", (f"flat_dark requires bias_state='included', got {fd.bias_state!r}",))
        fd_inc = fd.frame.data.astype(np.float32)
        fcorr = flat_correction(finc, "flat_dark_incl_bias", flat_dark_inc=fd_inc)
        fcorr_invalid = finc_invalid | (fd.frame.mask != 0)
    elif flat_prep_mode == "flat_dark_bias_removed":
        if flat_m.flat_form not in ("raw_response", None):
            raise InvalidRequestError("additive flat correction requires a raw_response flat")
        bf = masters.get("bias_flat")
        fd = masters.get("flat_dark")
        if bf is None or fd is None:
            raise InvalidRequestError("flat_dark_bias_removed requires 'bias_flat' and 'flat_dark'")
        _validate_binding(flat_m.frame.metadata, bf, "bias")
        r = _bias_exposure_reason(bf.frame.metadata.exposure_s, _bias_range_for(flat_m.frame.metadata))
        if r:
            raise GeometryMismatchError(r, ("flat bias exposure vs qualified range",))
        _validate_binding(flat_m.frame.metadata, fd, "flat_dark", exposure_reference=flat_m.frame.metadata.exposure_s)
        if fd.bias_state != "removed":
            raise GeometryMismatchError("ROLE_UNAVAILABLE", (f"flat_dark requires bias_state='removed', got {fd.bias_state!r}",))
        bias_flat = bf.frame.data.astype(np.float32)
        fd_inc = fd.frame.data.astype(np.float32)
        fcorr = flat_correction(finc, "flat_dark_bias_removed", flat_dark_removed=fd_inc, bias_flat=bias_flat)
        fcorr_invalid = finc_invalid | (bf.frame.mask != 0) | (fd.frame.mask != 0)
    elif flat_prep_mode == "bias_only_flat":
        if flat_m.flat_form not in ("raw_response", None):
            raise InvalidRequestError("additive flat correction requires a raw_response flat")
        if not flat_m.short_flat_profile:
            raise InvalidRequestError("bias_only_flat requires an explicit qualified short-flat profile")
        bf = masters.get("bias_flat")
        if bf is None:
            raise InvalidRequestError("bias_only_flat requires 'bias_flat'")
        _validate_binding(flat_m.frame.metadata, bf, "bias")
        r = _bias_exposure_reason(bf.frame.metadata.exposure_s, _bias_range_for(flat_m.frame.metadata))
        if r:
            raise GeometryMismatchError(r, ("flat bias exposure vs qualified range",))
        bias_flat = bf.frame.data.astype(np.float32)
        fcorr = flat_correction(finc, "bias_only_flat", bias_flat=bias_flat)
        fcorr_invalid = finc_invalid | (bf.frame.mask != 0)
    else:
        raise InvalidRequestError(f"unsupported flat_prep_mode: {flat_prep_mode!r}")

    norm = normalize_flat_response(
        fcorr,
        cfa_phase=cfa_phase,
        roi_origin=geo.roi_origin or (0, 0),
        saturation_samples=finc,
        saturation_limit=_qualified_saturation(flat_m.frame.metadata),
        fcorr_invalid=fcorr_invalid,
    )
    return norm.R, norm.valid, dict(norm.scalars), norm


def _execute(
    light, request, masters, *, flat_prep_mode, token, emit, prepared_flat_outcome=None
) -> CalibrationResult:
    token.raise_if_cancelled()
    emit("geometry_validation", 0, 3)

    light_md = light.metadata
    _validate_light_source(light_md)
    bound = _resolve_light_masters(light, request, masters)

    flat_response: Optional[np.ndarray] = None
    flat_valid: Optional[np.ndarray] = None
    scalars: dict[str, Optional[float]] = {}

    if request.flat_mode == "apply":
        flat_m = masters.get("flat")
        if flat_m is None:
            raise InvalidRequestError("required master role missing: 'flat'")
        expected_units = "dimensionless" if flat_m.flat_form == "normalized_response" else "ADU"
        prepared_flat = flat_m.flat_form in ("corrected_unnormalized", "normalized_response")
        _validate_binding(light_md, flat_m, "flat", check_filter=True, check_optical=True, expected_units=expected_units, prepared_flat=prepared_flat)

        token.raise_if_cancelled()
        emit("flat_preparation", 1, 3)
        # P8-A3B: consume the plan-invariant prepared-flat outcome when it
        # matches this route; otherwise fall back to today's per-frame
        # ``_prepare_flat``. A prepared *failure* is re-emitted as a fresh
        # exception instance (never the cached instance/traceback) at this exact
        # point, preserving today's precedence and per-frame FAILED mapping.
        pf = prepared_flat_outcome
        if pf is None or pf.flat_prep_mode != flat_prep_mode:
            flat_response, flat_valid, scalars, norm = _prepare_flat(flat_m, masters, flat_prep_mode)
        elif not pf.ok:
            raise pf.failure_type(*pf.failure_args)
        else:
            flat_response, flat_valid, scalars, norm = pf.flat_response, pf.flat_valid, pf.scalars, pf.norm
        if norm is not None and not norm.usable:
            return _flat_unusable_result(norm, light_md.saturation_evidence)

    token.raise_if_cancelled()
    emit("additive_division", 2, 3)

    additive_kwargs: dict = {}
    if "bias" in bound:
        additive_kwargs["bias"] = bound["bias"].frame.data.astype(np.float32)
        additive_kwargs["bias_mask"] = bound["bias"].frame.mask
    if "dark" in bound:
        if request.additive_mode == "dark_incl_bias":
            additive_kwargs["dark_inc"] = bound["dark"].frame.data.astype(np.float32)
            additive_kwargs["dark_inc_mask"] = bound["dark"].frame.mask
        elif request.additive_mode == "dark_bias_removed":
            additive_kwargs["dark_removed"] = bound["dark"].frame.data.astype(np.float32)
            additive_kwargs["dark_removed_mask"] = bound["dark"].frame.mask

    result = calibrate_light(
        light.data.astype(np.float32),
        additive_mode=request.additive_mode,
        flat_mode=request.flat_mode,
        input_mask=light.mask,
        flat_response=flat_response,
        flat_valid=flat_valid,
        saturation_limit=_qualified_saturation(light_md),
        **additive_kwargs,
    )

    if scalars:
        result = CalibrationResult(
            status=result.status, data=result.data, mask=result.mask,
            counts=result.counts, frame_quality=result.frame_quality,
            precision=result.precision, scalars=dict(scalars),
            warnings=result.warnings, reason_code=result.reason_code,
        )

    token.raise_if_cancelled()
    if result.status not in (STATUS_CANCELLED, "FAILED"):
        emit("complete", 3, 3)
    return result


__all__ = [
    "CalibrationRequest",
    "MasterBinding",
    "NormalizationProof",
    "execute_calibration",
    "EXPOSURE_TOLERANCE_ABS",
    "EXPOSURE_TOLERANCE_REL",
    "TEMP_PARSER_TOLERANCE_C",
]
