"""Pure compatibility and complete-set enumeration (SCIENCE §6, ARCHITECTURE §3).

The matcher is deterministic, conservative and filesystem-independent. It
validates each candidate's semantic ``master_type`` (never trusts a container
key), makes role-applicable required facts explicit (missing evidence always
rejects), enumerates **complete coherent sets** (including each flat's own
additive dependency), enforces real flat qualification evidence (per-plane
population, saturation semantics, normalization coherence), collapses
byte-identical *and* descriptor-identical duplicates (keeping every location),
and never ranks or picks a winner: 0 sets ⇒ ``NO_MATCH``, 1 ⇒ ``MATCHED``,
>1 ⇒ ``AMBIGUOUS``.

Two audit channels are kept distinct:

* ``rejected_candidates`` — the full audit: every candidate examined that did not
  become part of a coherent set, with structured reasons (wrong role, wrong
  bias-state, incompatible).
* ``reason_codes`` — the *manifest* reasons that determine the outcome (why the
  required roles cannot be satisfied). ``NO_MATCH`` is never unexplained.

Manual selection **filters** the coherent sets and then recounts; an empty map
changes nothing, a partial valid selection leaves residual ambiguity, and an
invalid/unknown selection refuses. Sorting stabilizes reporting only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from zecalibrator.core.descriptors import (
    Acquisition,
    DescriptorSnapshot,
    DetectorIdentity,
    LightConstraints,
    MasterDescriptor,
    OpticalIdentity,
)
from zecalibrator.core.geometry import Geometry
from zecalibrator.core.plans import (
    CalibrationPlan,
    CalibrationRequest,
    Candidate,
    MasterBinding,
    MatchPolicy,
    PolicyParameters,
    VersionSet,
)

OUTCOME_MATCHED = "MATCHED"
OUTCOME_NO_MATCH = "NO_MATCH"
OUTCOME_AMBIGUOUS = "AMBIGUOUS"

_MISSING = "MISSING_REQUIRED_FIELD"
_UNKNOWN_EQ = "UNKNOWN_EQUALS_UNKNOWN"
_BAYER_PHASES = ("GRBG", "RGGB", "BGGR", "GBRG")
_CFA_PLANES = ("G1", "R", "B", "G2")
_MONO_PLANES = ("mono",)

_CHECK_NONE = "none"
_CHECK_DARK_EXPOSURE = "dark_exposure"
_CHECK_BIAS_RANGE = "bias_range"
_CHECK_FLATDARK_EXPOSURE = "flatdark_exposure"
_CHECK_FLATBIAS_RANGE = "flatbias_range"


def _freeze_value(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    return value


@dataclass(frozen=True)
class Reason:
    """A structured rejection reason (code + field/role/parent + evidence)."""

    code: str
    field: Optional[str] = None
    role: Optional[str] = None
    parent: Optional[str] = None
    expected: object = None
    observed: object = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", _freeze_value(self.expected))
        object.__setattr__(self, "observed", _freeze_value(self.observed))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "code": self.code,
            "field": self.field,
            "role": self.role,
            "parent": self.parent,
            "expected": self.expected,
            "observed": self.observed,
        }
@dataclass(frozen=True)
class RejectionRecord:
    """A rejected candidate with structured reasons."""

    candidate_id: str
    role: str
    reasons: tuple[Reason, ...] = ()
    descriptor_snapshot: Optional[DescriptorSnapshot] = None

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.code for r in self.reasons))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "reasons": [r.to_dict() for r in self.reasons],
            "descriptor_snapshot": dict(self.descriptor_snapshot.to_dict()) if self.descriptor_snapshot is not None else None,
        }
@dataclass(frozen=True)
class MatchResult:
    """Pure selection result (ARCHITECTURE §3.3 ``MatchResult``)."""

    outcome: str
    plan: Optional[CalibrationPlan] = None
    rejected_candidates: tuple[RejectionRecord, ...] = ()
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[Reason, ...] = ()
    coherent_sets: tuple[Mapping[str, Candidate], ...] = ()
    structural_reasons: tuple[Reason, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected_candidates", tuple(self.rejected_candidates))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "coherent_sets", tuple(MappingProxyType(dict(s)) for s in self.coherent_sets))
        object.__setattr__(self, "structural_reasons", tuple(self.structural_reasons))


# ---------------------------------------------------------------------------
# Unknown / numeric helpers
# ---------------------------------------------------------------------------
def _unknown(v) -> bool:
    return v is None


def _unknown_instance(v) -> bool:
    if v is None:
        return True
    return isinstance(v, str) and v.strip() == "unknown"


def _finite(v) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _dedup(codes: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(codes))


# ---------------------------------------------------------------------------
# Per-field compatibility reasons
# ---------------------------------------------------------------------------
def _geometry_reasons(light: Geometry, master: Geometry) -> list[Reason]:
    reasons: list[Reason] = []

    def req(code, field, lv, mv):
        reasons.append(Reason(code=code, field=field, expected=lv, observed=mv))

    if light.shape != master.shape:
        req("GEOMETRY_MISMATCH", "geometry.shape", light.shape, master.shape)

    for fld, code in (("sensor_dimensions", "GEOMETRY_MISMATCH"), ("binning", "BINNING_MISMATCH"), ("orientation", "GEOMETRY_MISMATCH")):
        lv = getattr(light, fld)
        mv = getattr(master, fld)
        if lv is None or mv is None:
            req(_MISSING, f"geometry.{fld}", lv, mv)
        elif lv != mv:
            req(code, f"geometry.{fld}", lv, mv)
            if fld == "binning":
                req("GEOMETRY_MISMATCH", "geometry.binning", lv, mv)

    lre = light.roi_extent
    mre = master.roi_extent
    if lre is None or mre is None:
        req(_MISSING, "geometry.roi_extent", lre, mre)
    elif lre != mre:
        req("GEOMETRY_MISMATCH", "geometry.roi_extent", lre, mre)

    lroi = light.roi_origin
    mroi = master.roi_origin
    if lroi is None or mroi is None:
        req(_MISSING, "geometry.roi_origin", lroi, mroi)
    elif lroi != mroi:
        req("ROI_ORIGIN_MISMATCH", "geometry.roi_origin", lroi, mroi)
        req("CFA_PHASE_MISMATCH", "geometry.roi_origin", lroi, mroi)

    lcfa = light.cfa_phase
    mcfa = master.cfa_phase
    if lcfa is None or mcfa is None:
        req(_MISSING, "geometry.cfa_phase", lcfa, mcfa)
    elif lcfa != mcfa:
        req("CFA_PHASE_MISMATCH", "geometry.cfa_phase", lcfa, mcfa)

    return reasons


def _detector_reasons(light: DetectorIdentity, master: DetectorIdentity) -> list[Reason]:
    reasons: list[Reason] = []
    li = None if _unknown_instance(light.detector_instance_id) else light.detector_instance_id
    mi = None if _unknown_instance(master.detector_instance_id) else master.detector_instance_id
    if li is None or mi is None:
        reasons.append(Reason(_MISSING, "detector.detector_instance_id", expected=light.detector_instance_id, observed=master.detector_instance_id))
        if li is None and mi is None:
            reasons.append(Reason(_UNKNOWN_EQ, "detector.detector_instance_id", expected=light.detector_instance_id, observed=master.detector_instance_id))
    elif li != mi:
        reasons.append(Reason("DETECTOR_MISMATCH", "detector.detector_instance_id", expected=light.detector_instance_id, observed=master.detector_instance_id))

    lm = light.detector_model
    mm = master.detector_model
    if lm is None or mm is None:
        reasons.append(Reason(_MISSING, "detector.detector_model", expected=lm, observed=mm))
        if lm is None and mm is None:
            reasons.append(Reason(_UNKNOWN_EQ, "detector.detector_model", expected=lm, observed=mm))
    elif lm != mm:
        reasons.append(Reason("DETECTOR_MISMATCH", "detector.detector_model", expected=lm, observed=mm))

    return reasons


def _numeric_reason(light, master, code: str, field: str) -> list[Reason]:
    if _unknown(light) or _unknown(master):
        reasons = [Reason(_MISSING, field, expected=light, observed=master)]
        if _unknown(light) and _unknown(master):
            reasons.append(Reason(_UNKNOWN_EQ, field, expected=light, observed=master))
        return reasons
    if not _finite(light) or not _finite(master):
        return [Reason(_MISSING, field, expected=light, observed=master)]
    if float(light) != float(master):
        return [Reason(code, field, expected=light, observed=master)]
    return []


def _string_reason(light, master, code: str, field: str) -> list[Reason]:
    if _unknown(light) or _unknown(master):
        reasons = [Reason(_MISSING, field, expected=light, observed=master)]
        if _unknown(light) and _unknown(master):
            reasons.append(Reason(_UNKNOWN_EQ, field, expected=light, observed=master))
        return reasons
    if light != master:
        return [Reason(code, field, expected=light, observed=master)]
    return []


def _temperature_reason(light, master, policy: MatchPolicy) -> list[Reason]:
    if _unknown(light) or _unknown(master):
        reasons = [Reason(_MISSING, "acquisition.temperature_c", expected=light, observed=master)]
        if _unknown(light) and _unknown(master):
            reasons.append(Reason(_UNKNOWN_EQ, "acquisition.temperature_c", expected=light, observed=master))
        return reasons
    if not _finite(light) or not _finite(master):
        return [Reason(_MISSING, "acquisition.temperature_c", expected=light, observed=master)]
    lt, mt = float(light), float(master)
    tol = policy.temperature_tolerance
    bound = max(tol.absolute, tol.relative * max(abs(lt), abs(mt)))
    if abs(lt - mt) > bound:
        return [Reason("TEMPERATURE_MISMATCH", "acquisition.temperature_c", expected=light, observed=master)]
    return []


def _exposure_reason(reference, master, policy: MatchPolicy) -> list[Reason]:
    if _unknown(reference) or _unknown(master):
        reasons = [Reason(_MISSING, "acquisition.exposure_s", expected=reference, observed=master)]
        if _unknown(reference) and _unknown(master):
            reasons.append(Reason(_UNKNOWN_EQ, "acquisition.exposure_s", expected=reference, observed=master))
        return reasons
    if not _finite(reference) or not _finite(master) or float(reference) < 0 or float(master) < 0:
        return [Reason(_MISSING, "acquisition.exposure_s", expected=reference, observed=master)]
    t1, t2 = float(reference), float(master)
    tol = policy.exposure_tolerance
    bound = max(tol.absolute, tol.relative * max(abs(t1), abs(t2)))
    if abs(t1 - t2) > bound:
        return [Reason("EXPOSURE_MISMATCH", "acquisition.exposure_s", expected=reference, observed=master)]
    return []


def _bias_exposure_reason(bias_exposure, max_s) -> list[Reason]:
    if _unknown(bias_exposure) or _unknown(max_s):
        reasons = [Reason(_MISSING, "acquisition.exposure_s", expected=bias_exposure, observed=max_s)]
        if _unknown(bias_exposure) and _unknown(max_s):
            reasons.append(Reason(_UNKNOWN_EQ, "acquisition.exposure_s", expected=bias_exposure, observed=max_s))
        return reasons
    if not _finite(bias_exposure) or not _finite(max_s) or float(max_s) < 0:
        return [Reason(_MISSING, "acquisition.exposure_s", expected=bias_exposure, observed=max_s)]
    if float(bias_exposure) < 0:
        return [Reason(_MISSING, "acquisition.exposure_s", expected=bias_exposure, observed=max_s)]
    if float(bias_exposure) > float(max_s):
        return [Reason("EXPOSURE_MISMATCH", "acquisition.exposure_s", expected=f"<= {max_s}", observed=bias_exposure)]
    return []


def _units_reason(desc: MasterDescriptor, expected_units: str) -> list[Reason]:
    if desc.physical_units != expected_units:
        return [Reason(_MISSING, "physical_units", expected=expected_units, observed=desc.physical_units)]
    return []


def _acquisition_reasons(light: Acquisition, master: Acquisition, policy: MatchPolicy) -> list[Reason]:
    reasons: list[Reason] = []
    reasons += _numeric_reason(light.gain, master.gain, "GAIN_MISMATCH", "acquisition.gain")
    reasons += _numeric_reason(light.offset, master.offset, "OFFSET_MISMATCH", "acquisition.offset")
    reasons += _string_reason(light.readout_mode, master.readout_mode, "READOUT_MISMATCH", "acquisition.readout_mode")
    reasons += _string_reason(light.adc_mode, master.adc_mode, "ADC_MISMATCH", "acquisition.adc_mode")
    reasons += _temperature_reason(light.temperature_c, master.temperature_c, policy)
    return reasons


# ---------------------------------------------------------------------------
# Flat evidence
# ---------------------------------------------------------------------------
def _required_planes(desc: MasterDescriptor) -> tuple[str, ...]:
    if desc.cfa_phase == "mono":
        return _MONO_PLANES
    return _CFA_PLANES


def _scalars_match_phase(desc: MasterDescriptor) -> bool:
    sc = desc.normalization_scalars
    if sc is None:
        return False
    phase = desc.cfa_phase
    if phase == "mono":
        return sc.kind == "mono"
    if phase in _BAYER_PHASES:
        return sc.kind == "cfa"
    return False


def _normalization_coherence_reasons(desc: MasterDescriptor) -> list[Reason]:
    reasons: list[Reason] = []
    ff = desc.flat_form
    hist = desc.processing_provenance.additive_correction_history
    norm = desc.processing_provenance.normalization
    scalars = desc.normalization_scalars

    if ff == "normalized_response":
        scalars_valid = scalars is not None and _scalars_match_phase(desc)
        if not scalars_valid:
            reasons.append(Reason("NORMALIZATION_SCALAR_COUNT_MISMATCH", "normalization_scalars", role="flat"))
        if norm is None:
            reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance.normalization", role="flat"))
        else:
            if scalars_valid:
                phase = desc.cfa_phase
                expected_pop = "mono-valid" if phase == "mono" else "cfa-4-plane"
                if norm.population != expected_pop:
                    reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance.normalization.population", role="flat", expected=expected_pop, observed=norm.population))
                if scalars is not None and dict(norm.scalars.to_mapping()) != dict(scalars.to_mapping()):
                    reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance.normalization.scalars", role="flat", expected=dict(scalars.to_mapping()), observed=dict(norm.scalars.to_mapping())))
        if not hist:
            reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance.additive_correction_history", role="flat"))
    elif ff == "corrected_unnormalized":
        if scalars is not None or norm is not None:
            reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance", role="flat"))
        if not hist:
            reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance.additive_correction_history", role="flat"))
    elif ff == "raw_response":
        if scalars is not None or norm is not None or hist:
            reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance", role="flat"))

    return reasons


def _flat_evidence_reasons(desc: MasterDescriptor, policy: MatchPolicy) -> list[Reason]:
    reasons: list[Reason] = []
    ve = desc.validity_evidence

    if ve.quality_policy_state != "qualified":
        reasons.append(Reason("QUALITY_POLICY_PENDING", "validity_evidence.quality_policy_state", role="flat"))
    if not ve.saturation_limit_known:
        reasons.append(Reason(_MISSING, "validity_evidence.saturation_limit_known", role="flat", expected=True, observed=ve.saturation_limit_known))
    if desc.acquisition.saturation_evidence != "qualified":
        reasons.append(Reason(_MISSING, "acquisition.saturation_evidence", role="flat", expected="qualified", observed=desc.acquisition.saturation_evidence))
    if not ve.illumination:
        reasons.append(Reason(_MISSING, "validity_evidence.illumination", role="flat"))
    if not ve.exposure_quality:
        reasons.append(Reason(_MISSING, "validity_evidence.exposure_quality", role="flat"))

    valid = ve.valid_normalization_count
    total = ve.total_normalization_count
    if valid is None or total is None:
        reasons.append(Reason(_MISSING, "validity_evidence.valid_normalization_count", role="flat"))
    else:
        for p in _required_planes(desc):
            v = valid.get(p)
            t = total.get(p)
            if v is None or t is None:
                reasons.append(Reason(_MISSING, f"validity_evidence.valid_normalization_count.{p}", role="flat"))
            elif t <= 0 or v <= 0:
                reasons.append(Reason("FLAT_UNUSABLE", f"validity_evidence.{p}", role="flat", expected=">0 valid", observed=(v, t)))
            elif 100.0 * v / t < policy.flat_quality_policy.threshold_pct:
                reasons.append(Reason("SATURATED_FLAT", f"validity_evidence.{p}", role="flat", expected=f">= {policy.flat_quality_policy.threshold_pct}%", observed=f"{100.0 * v / t:.2f}%"))

    reasons += _normalization_coherence_reasons(desc)
    return reasons


# ---------------------------------------------------------------------------
# Candidate compatibility
# ---------------------------------------------------------------------------
def _candidate_compatibility(
    reference,
    desc: MasterDescriptor,
    *,
    policy: MatchPolicy,
    check: str,
    reference_exposure: Optional[float],
    reference_bias_range: Optional[float],
    check_filter: bool,
    check_optical: bool,
    expected_units: str,
    flat_extra: bool,
) -> list[Reason]:
    reasons: list[Reason] = []
    reasons += _geometry_reasons(reference.geometry, desc.geometry)
    reasons += _detector_reasons(reference.detector, desc.detector)
    reasons += _acquisition_reasons(reference.acquisition, desc.acquisition, policy)

    if check in (_CHECK_DARK_EXPOSURE, _CHECK_FLATDARK_EXPOSURE):
        reasons += _exposure_reason(reference_exposure, desc.acquisition.exposure_s, policy)
    elif check in (_CHECK_BIAS_RANGE, _CHECK_FLATBIAS_RANGE):
        reasons += _bias_exposure_reason(desc.acquisition.exposure_s, reference_bias_range)

    reasons += _units_reason(desc, expected_units)

    if check_filter:
        reasons += _string_reason(reference.optical.filter, desc.filter, "FILTER_MISMATCH", "optical.filter")
    if check_optical:
        reasons += _string_reason(reference.optical.optical_train_id, desc.optical_train_id, "OPTICAL_TRAIN_MISMATCH", "optical.optical_train_id")

    if flat_extra:
        reasons += _flat_evidence_reasons(desc, policy)

    seen = set()
    out = []
    for r in reasons:
        key = (r.code, r.field)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _collapse_duplicates(candidates: Sequence[Candidate]) -> list[Candidate]:
    by_key: dict[tuple[str, str], Candidate] = {}
    order: list[tuple[str, str]] = []
    for c in candidates:
        key = c.identity_key
        if key in by_key:
            merged = by_key[key]
            locs = tuple(sorted({(l.path, l.hdu): l for l in merged.locators + c.locators}.values(), key=lambda l: (l.path, str(l.hdu))))
            by_key[key] = Candidate(
                candidate_id=merged.candidate_id,
                descriptor=merged.descriptor,
                descriptor_snapshot=merged.descriptor_snapshot,
                locators=locs,
                mask_locator=merged.mask_locator or c.mask_locator,
            )
        else:
            by_key[key] = c
            order.append(key)
    return [by_key[k] for k in order]


def _sorted_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.candidate_id)


def _role_of(desc: MasterDescriptor) -> str:
    return desc.master_type


# ---------------------------------------------------------------------------
# Flat dependency options
# ---------------------------------------------------------------------------
def _flat_dependency_options(
    flat: MasterDescriptor,
    flat_candidate: Candidate,
    candidates: Mapping[str, Sequence[Candidate]],
    policy: MatchPolicy,
):
    """Return ``(options, manifest_records, structural, audit_records)``.

    ``options`` are complete dependency bindings; ``manifest_records`` are
    applicable-but-incompatible dependency rejections (exposure/bias-range);
    ``structural`` is ``FLAT_ADDITIVE_DEPENDENCY_MISSING`` when the flat's
    additive dependency is entirely absent; ``audit_records`` are wrong-role
    candidates in the dependency pools (audit only, not manifest).
    """
    ff = flat.flat_form
    if ff in ("normalized_response", "corrected_unnormalized"):
        return [{}], [], [], []

    if ff == "raw_response":
        options = []
        manifest_records: list[RejectionRecord] = []
        structural: list[Reason] = []
        audit_records: list[RejectionRecord] = []

        all_fd = list(candidates.get("flat_dark", ()))
        all_bias = list(candidates.get("bias", ()))
        fd_pool = [c for c in all_fd if _role_of(c.descriptor) == "flat_dark"]
        bias_pool = [c for c in all_bias if _role_of(c.descriptor) == "bias"]
        for c in all_fd:
            if _role_of(c.descriptor) != "flat_dark":
                audit_records.append(RejectionRecord(c.candidate_id, "flat_dark", (Reason("ROLE_UNAVAILABLE", "master_type", role="flat_dark", expected="flat_dark", observed=c.descriptor.master_type),), c.descriptor_snapshot))
            elif c.descriptor.bias_state not in ("included", "removed"):
                audit_records.append(RejectionRecord(c.candidate_id, "flat_dark", (Reason("ROLE_UNAVAILABLE", "bias_state", role="flat_dark", parent="flat", expected="included|removed", observed=c.descriptor.bias_state),), c.descriptor_snapshot))
        for c in all_bias:
            if _role_of(c.descriptor) != "bias":
                audit_records.append(RejectionRecord(c.candidate_id, "bias", (Reason("ROLE_UNAVAILABLE", "master_type", role="bias", expected="bias", observed=c.descriptor.master_type),), c.descriptor_snapshot))

        flat_bias_range = flat.bias_exposure_max_s
        flat_short = flat.short_flat_profile

        # Route 1: flat_dark_incl_bias.
        for fd in _sorted_candidates(fd_pool):
            if fd.descriptor.bias_state != "included":
                continue
            reasons = _candidate_compatibility(
                flat, fd.descriptor, policy=policy,
                check=_CHECK_FLATDARK_EXPOSURE, reference_exposure=flat.acquisition.exposure_s, reference_bias_range=None,
                check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
            )
            if not reasons:
                options.append({"flat_dark": fd})
            else:
                manifest_records.append(RejectionRecord(fd.candidate_id, "flat_dark", tuple(reasons), fd.descriptor_snapshot))

        # Route 2: flat_dark_bias_removed.
        for fd in _sorted_candidates(fd_pool):
            if fd.descriptor.bias_state != "removed":
                continue
            fd_reasons = _candidate_compatibility(
                flat, fd.descriptor, policy=policy,
                check=_CHECK_FLATDARK_EXPOSURE, reference_exposure=flat.acquisition.exposure_s, reference_bias_range=None,
                check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
            )
            if fd_reasons:
                manifest_records.append(RejectionRecord(fd.candidate_id, "flat_dark", tuple(fd_reasons), fd.descriptor_snapshot))
                continue
            for bf in _sorted_candidates(bias_pool):
                bf_reasons = _candidate_compatibility(
                    flat, bf.descriptor, policy=policy,
                    check=_CHECK_FLATBIAS_RANGE, reference_exposure=None, reference_bias_range=flat_bias_range,
                    check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
                )
                if not bf_reasons:
                    options.append({"flat_dark": fd, "bias_flat": bf})
                else:
                    manifest_records.append(RejectionRecord(bf.candidate_id, "bias_flat", tuple(bf_reasons), bf.descriptor_snapshot))

        # Route 3: bias_only_flat.
        if flat_short:
            for bf in _sorted_candidates(bias_pool):
                bf_reasons = _candidate_compatibility(
                    flat, bf.descriptor, policy=policy,
                    check=_CHECK_FLATBIAS_RANGE, reference_exposure=None, reference_bias_range=flat_bias_range,
                    check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
                )
                if not bf_reasons:
                    options.append({"bias_flat": bf})
                else:
                    manifest_records.append(RejectionRecord(bf.candidate_id, "bias_flat", tuple(bf_reasons), bf.descriptor_snapshot))

        routable_fd = [c for c in fd_pool if c.descriptor.bias_state in ("included", "removed")]
        if not options and len(routable_fd) == 0:
            structural.append(Reason("FLAT_ADDITIVE_DEPENDENCY_MISSING", "flat_dark", role="flat", parent="flat"))

        return options, manifest_records, structural, audit_records
    return [], [], [Reason("UNDOCUMENTED_PROCESSING", "flat_form", role="flat")], []


# ---------------------------------------------------------------------------
# Top-level match
# ---------------------------------------------------------------------------
def match_calibration(
    light: LightConstraints,
    request: CalibrationRequest,
    candidates: Mapping[str, Sequence[Candidate]],
    policy: MatchPolicy,
    *,
    manual_selection: Optional[Mapping[str, str]] = None,
) -> MatchResult:
    roles = request.required_roles

    rejected: list[RejectionRecord] = []
    manifest_rejected: list[RejectionRecord] = []
    structural: list[Reason] = []
    per_role_compatible: dict[str, list[Candidate]] = {}

    light_bias_range = light.acquisition.bias_exposure_max_s
    light_exposure = light.acquisition.exposure_s

    for role in roles:
        if role == "flat":
            continue
        need = None
        if role == "dark":
            need = "included" if request.additive_mode == "dark_incl_bias" else "removed"

        applicable = []
        for c in candidates.get(role, ()):
            if _role_of(c.descriptor) != role:
                rejected.append(RejectionRecord(c.candidate_id, role, (Reason("ROLE_UNAVAILABLE", "master_type", role=role, expected=role, observed=c.descriptor.master_type),), c.descriptor_snapshot))
            elif need is not None and c.descriptor.bias_state != need:
                rejected.append(RejectionRecord(c.candidate_id, role, (Reason("ROLE_UNAVAILABLE", "bias_state", role=role, expected=need, observed=c.descriptor.bias_state),), c.descriptor_snapshot))
            else:
                applicable.append(c)

        if not applicable:
            structural.append(Reason("ROLE_UNAVAILABLE", role=role))
            per_role_compatible[role] = []
            continue

        compatible = []
        for c in _sorted_candidates(_collapse_duplicates(applicable)):
            if role == "dark":
                check = _CHECK_DARK_EXPOSURE
                ref_exposure = light_exposure
                ref_bias_range = None
            elif role == "bias":
                check = _CHECK_BIAS_RANGE
                ref_exposure = None
                ref_bias_range = light_bias_range
            else:
                check = _CHECK_NONE
                ref_exposure = None
                ref_bias_range = None
            reasons = _candidate_compatibility(
                light, c.descriptor, policy=policy,
                check=check, reference_exposure=ref_exposure, reference_bias_range=ref_bias_range,
                check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
            )
            if reasons:
                manifest_rejected.append(RejectionRecord(c.candidate_id, role, tuple(reasons), c.descriptor_snapshot))
                rejected.append(RejectionRecord(c.candidate_id, role, tuple(reasons), c.descriptor_snapshot))
            else:
                compatible.append(c)
        per_role_compatible[role] = compatible

    flat_options: list[tuple[Candidate, dict[str, Candidate]]] = []
    if request.flat_mode == "apply":
        all_flat = list(candidates.get("flat", ()))
        applicable_flat = []
        for c in all_flat:
            if _role_of(c.descriptor) != "flat":
                rejected.append(RejectionRecord(c.candidate_id, "flat", (Reason("ROLE_UNAVAILABLE", "master_type", role="flat", expected="flat", observed=c.descriptor.master_type),), c.descriptor_snapshot))
            else:
                applicable_flat.append(c)
        if not applicable_flat:
            structural.append(Reason("ROLE_UNAVAILABLE", role="flat"))
        for c in _sorted_candidates(_collapse_duplicates(applicable_flat)):
            expected_units = "dimensionless" if c.descriptor.flat_form == "normalized_response" else "ADU"
            reasons = _candidate_compatibility(
                light, c.descriptor, policy=policy,
                check=_CHECK_NONE, reference_exposure=None, reference_bias_range=None,
                check_filter=True, check_optical=True, expected_units=expected_units, flat_extra=True,
            )
            if reasons:
                manifest_rejected.append(RejectionRecord(c.candidate_id, "flat", tuple(reasons), c.descriptor_snapshot))
                rejected.append(RejectionRecord(c.candidate_id, "flat", tuple(reasons), c.descriptor_snapshot))
                continue
            opts, recs, struct, audit = _flat_dependency_options(c.descriptor, c, candidates, policy)
            rejected.extend(audit)
            rejected.extend(recs)
            manifest_rejected.extend(recs)
            structural.extend(struct)
            for deps in opts:
                flat_options.append((c, deps))

    additive_role_order = [r for r in roles if r != "flat"]
    role_lists = [per_role_compatible[r] for r in additive_role_order]
    if request.flat_mode == "apply":
        role_lists.append(flat_options)

    sets: list[dict[str, Candidate]] = []
    for combo in _cartesian(role_lists):
        chosen: dict[str, Candidate] = {}
        for r, item in zip(additive_role_order, combo[:len(additive_role_order)]):
            chosen[r] = item
        if request.flat_mode == "apply":
            flat_c, deps = combo[len(additive_role_order)]
            chosen["flat"] = flat_c
            for dep_role, dep_c in deps.items():
                chosen[dep_role] = dep_c
        sets.append(chosen)

    unique_sets: list[dict[str, Candidate]] = []
    seen_sets: set[tuple] = set()
    for s in sets:
        key = tuple(sorted((role, c.identity_key) for role, c in s.items()))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_sets.append(s)

    if manual_selection is not None:
        unique_sets = _filter_by_manual(manual_selection, unique_sets)
        if not unique_sets and manual_selection:
            known_ids = {c.candidate_id for role_cands in candidates.values() for c in role_cands}
            rejected_ids = {rec.candidate_id for rec in manifest_rejected}
            explained = False
            for role, cid in manual_selection.items():
                if cid not in known_ids:
                    structural.append(Reason("ROLE_UNAVAILABLE", "manual_selection", role=role, expected=cid, observed=None))
                    explained = True
                    break
            if not explained:
                # Known id that did not resolve to a coherent set. If it was a
                # rejected candidate, the compatibility reason is already present;
                # otherwise (wrong role / role-not-required) emit a deterministic
                # manual-selection refusal reason.
                if not any(cid in rejected_ids for cid in manual_selection.values()):
                    for role, cid in sorted(manual_selection.items()):
                        structural.append(Reason("MANUAL_SELECTION_NO_MATCH", "manual_selection", role=role, expected=cid, observed=None))

    if not unique_sets:
        reason_codes = _dedup([r.code for r in structural] + [code for rec in manifest_rejected for code in rec.reason_codes])
        reasons = tuple(r for rec in rejected for r in rec.reasons) + tuple(structural)
        return MatchResult(
            outcome=OUTCOME_NO_MATCH,
            rejected_candidates=tuple(rejected),
            reason_codes=tuple(sorted(reason_codes)),
            reasons=reasons,
            structural_reasons=tuple(structural),
        )

    if len(unique_sets) > 1:
        return MatchResult(
            outcome=OUTCOME_AMBIGUOUS,
            rejected_candidates=tuple(rejected),
            reason_codes=(),
            coherent_sets=tuple(unique_sets),
        )

    chosen = unique_sets[0]
    plan = _build_plan(light, request, chosen, policy)
    return MatchResult(
        outcome=OUTCOME_MATCHED,
        plan=plan,
        rejected_candidates=tuple(rejected),
        reason_codes=(),
        coherent_sets=(chosen,),
    )


def _cartesian(lists):
    if not lists:
        return [()]
    result = [()]
    for lst in lists:
        result = [prefix + (item,) for prefix in result for item in lst]
    return result


def _filter_by_manual(manual_selection, sets) -> list[dict[str, Candidate]]:
    sel = dict(manual_selection)
    result = []
    for s in sets:
        if all(role in s and s[role].candidate_id == cid for role, cid in sel.items()):
            result.append(s)
    return result


def _build_plan(light, request, chosen: Mapping[str, Candidate], policy: MatchPolicy) -> CalibrationPlan:
    bindings: dict[str, MasterBinding] = {}
    for role, c in sorted(chosen.items()):
        bindings[role] = MasterBinding(
            descriptor_id=c.descriptor.descriptor_id,
            descriptor_snapshot=c.descriptor_snapshot,
            content_sha256=c.descriptor.content_sha256,
            size_bytes=c.descriptor.size_bytes,
            hdu=c.descriptor.hdu,
            mask_identity=c.descriptor.mask_identity,
            locators=c.locators,
            mask_locator=c.mask_locator,
        )
    return CalibrationPlan.build(
        request=request,
        light_constraints=light,
        masters=bindings,
        policy_parameters=PolicyParameters(
            exposure_tolerance=policy.exposure_tolerance,
            temperature_tolerance=policy.temperature_tolerance,
            flat_quality_policy=policy.flat_quality_policy,
        ),
        versions=VersionSet(matching_policy=policy.version),
    )


__all__ = [
    "MatchResult",
    "OUTCOME_AMBIGUOUS",
    "OUTCOME_MATCHED",
    "OUTCOME_NO_MATCH",
    "Reason",
    "RejectionRecord",
    "match_calibration",
]
