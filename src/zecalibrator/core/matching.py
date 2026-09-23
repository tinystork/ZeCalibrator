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
from zecalibrator.core.geometry import Geometry, is_bayer_phase
from zecalibrator.core.plans import (
    CalibrationComposition,
    CalibrationPlan,
    CalibrationRequest,
    Candidate,
    MasterBinding,
    MatchPolicy,
    PolicyParameters,
    SkippedRole,
    VersionSet,
)
from zecalibrator.core.selection import (
    ROUTE_UNSATISFIABLE,
    SELECTION_POLICY_VERSION,
    STATUS_AMBIGUOUS_TIE,
    RankedOutRecord,
    select_role_candidates,
    selection_key,
)

OUTCOME_MATCHED = "MATCHED"
OUTCOME_NO_MATCH = "NO_MATCH"
OUTCOME_AMBIGUOUS = "AMBIGUOUS"

_MISSING = "MISSING_REQUIRED_FIELD"
_UNKNOWN_EQ = "UNKNOWN_EQUALS_UNKNOWN"
_UNVERIFIED = "UNVERIFIED"
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
    """A structured rejection reason (code + field/role/parent + evidence).

    ``blocking`` (default ``True``) marks a reason that rejects a candidate.
    A non-blocking ``UNVERIFIED`` note (``blocking=False``) records an unknown
    *disambiguator* (e.g. both sides unknown) without rejecting: a candidate is
    compatible iff it has **zero blocking** reasons (R3B). ``blocking`` is an
    in-memory decision flag; the serialized ``code`` (``"UNVERIFIED"``) is the
    round-trip-safe signal of a non-blocking note.
    """

    code: str
    field: Optional[str] = None
    role: Optional[str] = None
    parent: Optional[str] = None
    expected: object = None
    observed: object = None
    blocking: bool = True

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
    unverified: tuple[Reason, ...] = ()
    selection: tuple = ()
    ranked_out: tuple = ()
    composition: Optional["CalibrationComposition"] = None
    selection_policy_version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected_candidates", tuple(self.rejected_candidates))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "coherent_sets", tuple(MappingProxyType(dict(s)) for s in self.coherent_sets))
        object.__setattr__(self, "structural_reasons", tuple(self.structural_reasons))
        object.__setattr__(self, "unverified", tuple(self.unverified))
        object.__setattr__(self, "selection", tuple(self.selection))
        object.__setattr__(self, "ranked_out", tuple(self.ranked_out))


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


def _disambiguator_reasons(light, master, code: str, field: str, *, unknown=_unknown) -> list[Reason]:
    """Disambiguator-tier reasons (R3B).

    Semantics:

    * both known + equal      -> ok (no reason)
    * both known + different  -> mismatch (blocking)
    * both unknown            -> UNVERIFIED (non-blocking, recorded)
    * one known + one unknown -> MISSING (blocking, conservative)
    """
    lu = unknown(light)
    mu = unknown(master)
    if lu and mu:
        return [Reason(_UNVERIFIED, field, expected=light, observed=master, blocking=False)]
    if lu or mu:
        return [Reason(_MISSING, field, expected=light, observed=master, blocking=True)]
    if light != master:
        return [Reason(code, field, expected=light, observed=master, blocking=True)]
    return []


def _standard_optional_reasons(light, master, code: str, field: str, *, numeric: bool = False) -> list[Reason]:
    """Standard auto-route tier (R3D-C): a session-supplied master fact that is
    missing on either side is UNVERIFIED (non-blocking, recorded in the audit);
    both known + different is a blocking mismatch.

    This is the relaxed tier for ``gain``/``offset``/``orientation``/
    ``roi_origin`` under the Standard master contract: the act of supplying a
    master carries contract semantics, so a missing acquisition fact no longer
    hard-rejects; a genuine comparable contradiction still does.
    """
    lu = _unknown(light)
    mu = _unknown(master)
    if lu or mu:
        return [Reason(_UNVERIFIED, field, expected=light, observed=master, blocking=False)]
    if numeric:
        if not _finite(light) or not _finite(master):
            return [Reason(_UNVERIFIED, field, expected=light, observed=master, blocking=False)]
        if float(light) != float(master):
            return [Reason(code, field, expected=light, observed=master, blocking=True)]
        return []
    if light != master:
        return [Reason(code, field, expected=light, observed=master, blocking=True)]
    return []


def _standard_temperature_reasons(light, master, policy: MatchPolicy) -> list[Reason]:
    """Standard auto-route temperature tier (R3D-C): missing -> UNVERIFIED
    (non-blocking); both known + out-of-tolerance -> blocking TEMPERATURE_MISMATCH."""
    field = "acquisition.temperature_c"
    if _unknown(light) or _unknown(master):
        return [Reason(_UNVERIFIED, field, expected=light, observed=master, blocking=False)]
    if not _finite(light) or not _finite(master):
        return [Reason(_UNVERIFIED, field, expected=light, observed=master, blocking=False)]
    lt, mt = float(light), float(master)
    tol = policy.temperature_tolerance
    bound = max(tol.absolute, tol.relative * max(abs(lt), abs(mt)))
    if abs(lt - mt) > bound:
        return [Reason("TEMPERATURE_MISMATCH", field, expected=light, observed=master)]
    return []


def _split_reasons(reasons: Iterable[Reason]) -> tuple[list[Reason], list[Reason]]:
    """Partition reasons into ``(blocking, unverified)``."""
    blocking: list[Reason] = []
    unverified: list[Reason] = []
    for r in reasons:
        if r.blocking:
            blocking.append(r)
        else:
            unverified.append(r)
    return blocking, unverified


def _hashable(value):
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _dedup_unverified(reasons: Iterable[Reason]) -> list[Reason]:
    """Dedup UNVERIFIED notes by ``(code, field, expected, observed)`` (R3B)."""
    seen: set[tuple] = set()
    out: list[Reason] = []
    for r in reasons:
        key = (r.code, r.field, _hashable(r.expected), _hashable(r.observed))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Per-field compatibility reasons
# ---------------------------------------------------------------------------
def _geometry_reasons(light: Geometry, master: Geometry, *, standard_contract: bool = False) -> list[Reason]:
    reasons: list[Reason] = []

    def req(code, field, lv, mv, blocking=True):
        reasons.append(Reason(code=code, field=field, expected=lv, observed=mv, blocking=blocking))

    if light.shape != master.shape:
        req("GEOMETRY_MISMATCH", "geometry.shape", light.shape, master.shape)

    # binning — necessary (unchanged).
    lb = light.binning
    mb = master.binning
    if lb is None or mb is None:
        req(_MISSING, "geometry.binning", lb, mb)
    elif lb != mb:
        req("BINNING_MISMATCH", "geometry.binning", lb, mb)
        req("GEOMETRY_MISMATCH", "geometry.binning", lb, mb)

    # sensor_dimensions — disambiguator.
    reasons += _disambiguator_reasons(
        light.sensor_dimensions, master.sensor_dimensions, "GEOMETRY_MISMATCH", "geometry.sensor_dimensions"
    )

    # CFA-conditional tier: orientation/roi_origin/roi_extent are *necessary*
    # for a Bayer sensor, *disambiguators* otherwise (mono / CFA not applicable).
    cfa_applies = is_bayer_phase(light.cfa_phase) or is_bayer_phase(master.cfa_phase)

    lo = light.orientation
    mo = master.orientation
    if standard_contract:
        # R3D-C: missing orientation is UNVERIFIED (non-blocking) in Standard;
        # a known contradiction stays blocking.
        reasons += _standard_optional_reasons(lo, mo, "GEOMETRY_MISMATCH", "geometry.orientation")
    elif cfa_applies:
        if lo is None or mo is None:
            req(_MISSING, "geometry.orientation", lo, mo)
        elif lo != mo:
            req("GEOMETRY_MISMATCH", "geometry.orientation", lo, mo)
    else:
        reasons += _disambiguator_reasons(lo, mo, "GEOMETRY_MISMATCH", "geometry.orientation")

    lroi = light.roi_origin
    mroi = master.roi_origin
    if standard_contract:
        # R3D-C: missing roi_origin is UNVERIFIED (non-blocking) in Standard.
        reasons += _standard_optional_reasons(lroi, mroi, "ROI_ORIGIN_MISMATCH", "geometry.roi_origin")
    elif cfa_applies:
        if lroi is None or mroi is None:
            req(_MISSING, "geometry.roi_origin", lroi, mroi)
        elif lroi != mroi:
            req("ROI_ORIGIN_MISMATCH", "geometry.roi_origin", lroi, mroi)
            req("CFA_PHASE_MISMATCH", "geometry.roi_origin", lroi, mroi)
    else:
        reasons += _disambiguator_reasons(lroi, mroi, "ROI_ORIGIN_MISMATCH", "geometry.roi_origin")

    lre = light.roi_extent
    mre = master.roi_extent
    if cfa_applies:
        if lre is None or mre is None:
            req(_MISSING, "geometry.roi_extent", lre, mre)
        elif lre != mre:
            req("GEOMETRY_MISMATCH", "geometry.roi_extent", lre, mre)
    else:
        reasons += _disambiguator_reasons(lre, mre, "GEOMETRY_MISMATCH", "geometry.roi_extent")

    # cfa_phase — necessary (unchanged).
    lcfa = light.cfa_phase
    mcfa = master.cfa_phase
    if lcfa is None or mcfa is None:
        req(_MISSING, "geometry.cfa_phase", lcfa, mcfa)
    elif lcfa != mcfa:
        req("CFA_PHASE_MISMATCH", "geometry.cfa_phase", lcfa, mcfa)

    return reasons


def _detector_reasons(light: DetectorIdentity, master: DetectorIdentity) -> list[Reason]:
    reasons: list[Reason] = []

    # detector_instance_id — disambiguator ("unknown" string or None = unknown).
    reasons += _disambiguator_reasons(
        light.detector_instance_id, master.detector_instance_id,
        "DETECTOR_MISMATCH", "detector.detector_instance_id", unknown=_unknown_instance,
    )

    # detector_model — necessary (unchanged).
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


def _acquisition_reasons(light: Acquisition, master: Acquisition, policy: MatchPolicy, *, standard_contract: bool = False, prepared_flat: bool = False) -> list[Reason]:
    reasons: list[Reason] = []
    if not prepared_flat:
        # R3D-E F4: a prepared flat (corrected_unnormalized / normalized_response)
        # is already the multiplicative response; its original gain/offset are
        # irrelevant to the light and must not emit GAIN_MISMATCH/OFFSET_MISMATCH
        # hard reasons. A raw_response flat keeps its additive-dependency
        # compatibility unchanged (gain/offset still checked).
        if standard_contract:
            # R3D-C: missing gain/offset on a session-supplied master is UNVERIFIED
            # (non-blocking); a known comparable mismatch stays blocking.
            reasons += _standard_optional_reasons(light.gain, master.gain, "GAIN_MISMATCH", "acquisition.gain", numeric=True)
            reasons += _standard_optional_reasons(light.offset, master.offset, "OFFSET_MISMATCH", "acquisition.offset", numeric=True)
        else:
            reasons += _numeric_reason(light.gain, master.gain, "GAIN_MISMATCH", "acquisition.gain")  # necessary
            reasons += _numeric_reason(light.offset, master.offset, "OFFSET_MISMATCH", "acquisition.offset")  # necessary
    reasons += _disambiguator_reasons(light.readout_mode, master.readout_mode, "READOUT_MISMATCH", "acquisition.readout_mode")
    reasons += _disambiguator_reasons(light.adc_mode, master.adc_mode, "ADC_MISMATCH", "acquisition.adc_mode")
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


def _normalization_coherence_reasons(desc: MasterDescriptor, *, standard_contract: bool = False) -> list[Reason]:
    reasons: list[Reason] = []
    ff = desc.flat_form
    hist = desc.processing_provenance.additive_correction_history
    hist_state = desc.processing_provenance.additive_history_state
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
            # R3D-C Standard master contract: a ``corrected_unnormalized`` flat
            # with ``additive_history_state == "unknown"`` is the ready-to-use
            # contract master flat (already the multiplicative response,
            # normalized at execution). The strict explicit path still requires
            # a documented additive history.
            if not (standard_contract and hist_state == "unknown"):
                reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance.additive_correction_history", role="flat"))
    elif ff == "raw_response":
        # R3D-A D1d: a raw flat must declare a KNOWN empty additive history.
        # ``not hist`` alone is insufficient — ``additive_history_state ==
        # "unknown"`` means "no information", which cannot certify a raw flat.
        if scalars is not None or norm is not None or hist or hist_state != "known":
            reasons.append(Reason("UNDOCUMENTED_PROCESSING", "processing_provenance", role="flat"))

    return reasons


def _flat_evidence_reasons(desc: MasterDescriptor, policy: MatchPolicy, *, standard_contract: bool = False) -> list[Reason]:
    reasons: list[Reason] = []
    ve = desc.validity_evidence

    # Known failure (explicit failed policy) is ALWAYS blocking.
    if ve.quality_policy_state != "qualified":
        reasons.append(Reason("QUALITY_POLICY_PENDING", "validity_evidence.quality_policy_state", role="flat"))

    def missing(code: str, field: str, *, expected=None, observed=None) -> None:
        """Missing-evidence branch: blocking in strict mode, UNVERIFIED (non-
        blocking, recorded) under the Standard master contract (F1)."""
        if standard_contract:
            reasons.append(Reason(_UNVERIFIED, field, role="flat", expected=expected, observed=observed, blocking=False))
        else:
            reasons.append(Reason(code, field, role="flat", expected=expected, observed=observed))

    if not ve.saturation_limit_known:
        missing(_MISSING, "validity_evidence.saturation_limit_known", expected=True, observed=ve.saturation_limit_known)
    if desc.acquisition.saturation_evidence != "qualified":
        missing(_MISSING, "acquisition.saturation_evidence", expected="qualified", observed=desc.acquisition.saturation_evidence)
    if not ve.illumination:
        missing(_MISSING, "validity_evidence.illumination")
    if not ve.exposure_quality:
        missing(_MISSING, "validity_evidence.exposure_quality")

    valid = ve.valid_normalization_count
    total = ve.total_normalization_count
    if valid is None or total is None:
        missing(_MISSING, "validity_evidence.valid_normalization_count")
    else:
        for p in _required_planes(desc):
            v = valid.get(p)
            t = total.get(p)
            if v is None or t is None:
                missing(_MISSING, f"validity_evidence.valid_normalization_count.{p}")
            elif t <= 0 or v <= 0:
                # Known failure (zero/negative valid population) is ALWAYS blocking.
                reasons.append(Reason("FLAT_UNUSABLE", f"validity_evidence.{p}", role="flat", expected=">0 valid", observed=(v, t)))
            elif 100.0 * v / t < policy.flat_quality_policy.threshold_pct:
                # Known failure (below-threshold valid fraction) is ALWAYS blocking.
                reasons.append(Reason("SATURATED_FLAT", f"validity_evidence.{p}", role="flat", expected=f">= {policy.flat_quality_policy.threshold_pct}%", observed=f"{100.0 * v / t:.2f}%"))

    reasons += _normalization_coherence_reasons(desc, standard_contract=standard_contract)
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
    standard_contract: bool = False,
) -> list[Reason]:
    reasons: list[Reason] = []
    reasons += _geometry_reasons(reference.geometry, desc.geometry, standard_contract=standard_contract)
    reasons += _detector_reasons(reference.detector, desc.detector)
    prepared_flat = (
        desc.master_type == "flat"
        and desc.flat_form in ("corrected_unnormalized", "normalized_response")
    )
    reasons += _acquisition_reasons(
        reference.acquisition, desc.acquisition, policy,
        standard_contract=standard_contract, prepared_flat=prepared_flat,
    )

    if check in (_CHECK_DARK_EXPOSURE, _CHECK_FLATDARK_EXPOSURE):
        reasons += _exposure_reason(reference_exposure, desc.acquisition.exposure_s, policy)
        if standard_contract:
            reasons += _standard_temperature_reasons(reference.acquisition.temperature_c, desc.acquisition.temperature_c, policy)
        else:
            reasons += _temperature_reason(reference.acquisition.temperature_c, desc.acquisition.temperature_c, policy)
    elif check in (_CHECK_BIAS_RANGE, _CHECK_FLATBIAS_RANGE):
        reasons += _bias_exposure_reason(desc.acquisition.exposure_s, reference_bias_range)

    reasons += _units_reason(desc, expected_units)

    if check_filter:
        reasons += _string_reason(reference.optical.filter, desc.filter, "FILTER_MISMATCH", "optical.filter")
    if check_optical:
        reasons += _disambiguator_reasons(reference.optical.optical_train_id, desc.optical_train_id, "OPTICAL_TRAIN_MISMATCH", "optical.optical_train_id")

    if flat_extra:
        reasons += _flat_evidence_reasons(desc, policy, standard_contract=standard_contract)

    seen = set()
    out = []
    for r in reasons:
        key = (r.code, r.field)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _min_date(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _collapse_duplicates(candidates: Sequence[Candidate]) -> list[Candidate]:
    by_key: dict[tuple[str, str], Candidate] = {}
    order: list[tuple[str, str]] = []
    # Deterministic representative: sort by candidate_id (stable) so the merged
    # candidate is independent of pool order; the merged acquired_at is also picked
    # deterministically (earliest) so two identical-content masters with different
    # DATE-OBS never make the winner pool-order dependent (N-5).
    for c in sorted(candidates, key=lambda c: (c.candidate_id, c.descriptor.content_sha256)):
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
                acquired_at=_min_date(merged.acquired_at, c.acquired_at),
            )
        else:
            by_key[key] = c
            order.append(key)
    return [by_key[k] for k in order]


def _sorted_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.candidate_id)


def _role_of(desc: MasterDescriptor) -> str:
    return desc.master_type


def _find_candidate(cands: Sequence[Candidate], candidate_id: str) -> Optional[Candidate]:
    for c in cands:
        if c.candidate_id == candidate_id:
            return c
    return None


# ---------------------------------------------------------------------------
# Flat dependency options
# ---------------------------------------------------------------------------
def _flat_dependency_options(
    flat: MasterDescriptor,
    flat_candidate: Candidate,
    candidates: Mapping[str, Sequence[Candidate]],
    policy: MatchPolicy,
    *,
    light=None,
    manual_selection: Optional[Mapping[str, str]] = None,
    probe: bool = False,
):
    """Return ``(options, manifest_records, structural, audit_records, unverified,
    selection_records, ranked_out_records, dep_no_candidate, dep_tie)``.

    ``options`` are complete dependency bindings, each built from the ranked
    winner of its dependency role (flat_dark is ranked by the additive rule, never
    the light's civil day); ``manifest_records`` are applicable-but-incompatible
    dependency rejections (exposure/bias-range); ``structural`` is
    ``FLAT_ADDITIVE_DEPENDENCY_MISSING`` when the flat's additive dependency is
    entirely absent; ``audit_records`` are wrong-role candidates in the dependency
    pools (audit only, not manifest); ``unverified`` are non-blocking UNVERIFIED
    notes from accepted dependency candidates (R3B). ``selection_records`` /
    ``ranked_out_records`` carry the ranking audit; ``dep_no_candidate`` lists
    dependency roles with no compatible candidate; ``dep_tie`` is True when a
    dependency role reported AMBIGUOUS_TIE.

    ``probe=True`` returns only the satisfiability signal (``options`` is ``[{}]``
    when at least one executable dependency route exists, else ``[]``) without
    ranking dependency roles or producing any audit records — it is used by the
    applicability filter to decide route-satisfiability before ranking the flat
    role.
    """
    manual = dict(manual_selection or {})
    ff = flat.flat_form
    if ff in ("normalized_response", "corrected_unnormalized"):
        return [{}], [], [], [], [], (), (), (), False

    if ff != "raw_response":
        return [], [], [Reason("UNDOCUMENTED_PROCESSING", "flat_form", role="flat")], [], [], (), (), (), False

    options = []
    manifest_records: list[RejectionRecord] = []
    structural: list[Reason] = []
    audit_records: list[RejectionRecord] = []
    unverified: list[Reason] = []
    selection_records: list = []
    ranked_out_records: list = []
    dep_no_candidate: list[str] = []

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

    fd_incl_compat: list[Candidate] = []
    fd_removed_compat: list[Candidate] = []
    bias_flat_compat: list[Candidate] = []

    for fd in _sorted_candidates(_collapse_duplicates(fd_pool)):
        reasons = _candidate_compatibility(
            flat, fd.descriptor, policy=policy,
            check=_CHECK_FLATDARK_EXPOSURE, reference_exposure=flat.acquisition.exposure_s, reference_bias_range=None,
            check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
        )
        blocking, unv = _split_reasons(reasons)
        if blocking:
            manifest_records.append(RejectionRecord(fd.candidate_id, "flat_dark", tuple(blocking), fd.descriptor_snapshot))
            continue
        unverified.extend(unv)
        if fd.descriptor.bias_state == "included":
            fd_incl_compat.append(fd)
        elif fd.descriptor.bias_state == "removed":
            fd_removed_compat.append(fd)

    for bf in _sorted_candidates(_collapse_duplicates(bias_pool)):
        reasons = _candidate_compatibility(
            flat, bf.descriptor, policy=policy,
            check=_CHECK_FLATBIAS_RANGE, reference_exposure=None, reference_bias_range=flat_bias_range,
            check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
        )
        blocking, unv = _split_reasons(reasons)
        if blocking:
            manifest_records.append(RejectionRecord(bf.candidate_id, "bias_flat", tuple(blocking), bf.descriptor_snapshot))
            continue
        unverified.extend(unv)
        bias_flat_compat.append(bf)

    if probe:
        # Applicability probe: satisfiable iff at least one executable dependency
        # route exists (flat_dark incl, flat_dark removed + bias, or short-flat bias).
        satisfiable = bool(
            fd_incl_compat
            or (fd_removed_compat and bias_flat_compat)
            or (flat_short and bias_flat_compat)
        )
        structural = []
        if not satisfiable:
            routable_fd = [c for c in fd_pool if c.descriptor.bias_state in ("included", "removed")]
            if len(routable_fd) == 0:
                structural.append(Reason("FLAT_ADDITIVE_DEPENDENCY_MISSING", "flat_dark", role="flat", parent="flat"))
        return ([{}] if satisfiable else []), manifest_records, structural, audit_records, unverified, (), (), (), False

    fd_incl_sel = select_role_candidates("flat_dark", light, fd_incl_compat, manual_choice=manual.get("flat_dark"))
    fd_removed_sel = select_role_candidates("flat_dark", light, fd_removed_compat, manual_choice=manual.get("flat_dark"))
    bias_flat_sel = select_role_candidates("bias_flat", light, bias_flat_compat, manual_choice=manual.get("bias_flat"))

    dep_tie = (
        fd_incl_sel.status == STATUS_AMBIGUOUS_TIE
        or fd_removed_sel.status == STATUS_AMBIGUOUS_TIE
        or bias_flat_sel.status == STATUS_AMBIGUOUS_TIE
    )

    for sel in (fd_incl_sel, fd_removed_sel, bias_flat_sel):
        if sel.winner is not None:
            selection_records.append(sel.winner)
        ranked_out_records.extend(sel.ranked_out)

    fd_incl_cand = _find_candidate(fd_incl_compat, fd_incl_sel.winner.chosen_candidate_id) if fd_incl_sel.winner is not None else None
    fd_removed_cand = _find_candidate(fd_removed_compat, fd_removed_sel.winner.chosen_candidate_id) if fd_removed_sel.winner is not None else None
    bias_cand = _find_candidate(bias_flat_compat, bias_flat_sel.winner.chosen_candidate_id) if bias_flat_sel.winner is not None else None

    # Route 1: flat_dark_incl_bias.
    if fd_incl_cand is not None:
        options.append({"flat_dark": fd_incl_cand})
    # Route 2: flat_dark_bias_removed.
    if fd_removed_cand is not None and bias_cand is not None:
        options.append({"flat_dark": fd_removed_cand, "bias_flat": bias_cand})
    # Route 3: bias_only_flat.
    if flat_short and bias_cand is not None:
        options.append({"bias_flat": bias_cand})

    if not fd_incl_compat and not fd_removed_compat:
        dep_no_candidate.append("flat_dark")
    if (flat_short or fd_removed_compat) and not bias_flat_compat:
        dep_no_candidate.append("bias_flat")

    routable_fd = [c for c in fd_pool if c.descriptor.bias_state in ("included", "removed")]
    if not options and len(routable_fd) == 0:
        structural.append(Reason("FLAT_ADDITIVE_DEPENDENCY_MISSING", "flat_dark", role="flat", parent="flat"))

    return (
        options, manifest_records, structural, audit_records, unverified,
        tuple(selection_records), tuple(ranked_out_records), tuple(dep_no_candidate), dep_tie,
    )


def _flat_route_satisfiable(flat: MasterDescriptor, candidates, policy: MatchPolicy):
    """Return ``(satisfiable, cause)`` for ``flat``'s preparation route.

    Route-satisfiability is an **applicability** filter applied BEFORE ranking,
    never a compatibility criterion. ``corrected_unnormalized`` /
    ``normalized_response`` are always satisfiable (no dependency); a
    ``raw_response`` flat is satisfiable iff ``_flat_dependency_options`` (probe
    mode) yields at least one executable option — the dependency logic is reused
    verbatim, never reimplemented. When unsatisfiable, ``cause`` is the
    dependency-level reason (e.g. ``FLAT_ADDITIVE_DEPENDENCY_MISSING`` or an
    exposure/bias mismatch code) for the audit.
    """
    if flat.flat_form in ("corrected_unnormalized", "normalized_response"):
        return True, ""
    if flat.flat_form != "raw_response":
        return False, "UNDOCUMENTED_PROCESSING"
    opts, manifest_records, structural, _audit, _unv, _sel, _rank, _nocand, _tie = _flat_dependency_options(
        flat, None, candidates, policy, probe=True,
    )
    if opts:
        return True, ""
    if structural:
        return False, structural[0].code
    if manifest_records:
        codes = sorted({code for rec in manifest_records for code in rec.reason_codes})
        return False, codes[0] if codes else "DEPENDENCY_INCOMPATIBLE"
    return False, "ROUTE_UNSATISFIABLE"


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
    standard_contract: bool = False,
) -> MatchResult:
    manual = dict(manual_selection or {})
    roles = request.required_roles

    rejected: list[RejectionRecord] = []
    manifest_rejected: list[RejectionRecord] = []
    structural: list[Reason] = []
    unverified: list[Reason] = []
    per_role_compatible: dict[str, list[Candidate]] = {}

    light_bias_range = light.acquisition.bias_exposure_max_s
    light_exposure = light.acquisition.exposure_s

    # -- additive compatibility (unchanged) ---------------------------------
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
                standard_contract=standard_contract,
            )
            blocking, unv = _split_reasons(reasons)
            if blocking:
                manifest_rejected.append(RejectionRecord(c.candidate_id, role, tuple(blocking), c.descriptor_snapshot))
                rejected.append(RejectionRecord(c.candidate_id, role, tuple(blocking), c.descriptor_snapshot))
            else:
                compatible.append(c)
                unverified.extend(unv)
        per_role_compatible[role] = compatible

    # -- flat compatibility (unchanged) -------------------------------------
    compatible_flats: list[Candidate] = []
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
                standard_contract=standard_contract,
            )
            blocking, unv = _split_reasons(reasons)
            if blocking:
                manifest_rejected.append(RejectionRecord(c.candidate_id, "flat", tuple(blocking), c.descriptor_snapshot))
                rejected.append(RejectionRecord(c.candidate_id, "flat", tuple(blocking), c.descriptor_snapshot))
                continue
            unverified.extend(unv)
            compatible_flats.append(c)

    unverified = _dedup_unverified(unverified)

    # -- ranking -------------------------------------------------------------
    selection_records: list = []
    ranked_out_records: list = []
    compatible_ids_by_role: dict[str, set] = {}
    additive_role_order = [r for r in roles if r != "flat"]

    role_tie = False
    additive_winners: dict[str, Candidate] = {}
    tie_candidates_by_role: dict[str, list[Candidate]] = {}
    ambiguous_reasons: list[Reason] = []
    for role in additive_role_order:
        compat = per_role_compatible[role]
        compatible_ids_by_role[role] = {c.candidate_id for c in compat}
        sel = select_role_candidates(role, light, compat, manual_choice=manual.get(role))
        if sel.status == STATUS_AMBIGUOUS_TIE:
            role_tie = True
            tie_candidates_by_role[role] = [c for c in compat if c.candidate_id in set(sel.tie_candidates)]
            ambiguous_reasons.append(Reason(
                "AMBIGUOUS_TIE", role=role, field=role, expected=sel.tie_candidates,
                observed=None, blocking=False,
            ))
            continue
        if sel.winner is not None:
            additive_winners[role] = _find_candidate(compat, sel.winner.chosen_candidate_id)
            selection_records.append(sel.winner)
            ranked_out_records.extend(sel.ranked_out)

    # -- flat ranking --------------------------------------------------------
    flat_winner_cand: Optional[Candidate] = None
    flat_route_options: list[dict[str, Candidate]] = []
    flat_dep_no_candidate: tuple[str, ...] = ()
    flat_dep_missing = False
    mixed_flat_form = False
    satisfiable_flat_ids: set = set()
    form_winners_list: list[tuple[str, Candidate]] = []

    if request.flat_mode == "apply":
        compatible_ids_by_role["flat"] = {c.candidate_id for c in compatible_flats}

        # Route-satisfiability is an APPLICABILITY filter (not a compatibility
        # criterion), applied BEFORE ranking so a route-unsatisfiable flat can
        # never shadow a satisfiable peer. See _flat_route_satisfiable.
        satisfiable_flats: list[Candidate] = []
        unsatisfiable_flats: list[tuple[Candidate, str]] = []
        for f in compatible_flats:
            sat, cause = _flat_route_satisfiable(f.descriptor, candidates, policy)
            if sat:
                satisfiable_flats.append(f)
            else:
                unsatisfiable_flats.append((f, cause))
        satisfiable_flat_ids = {c.candidate_id for c in satisfiable_flats}

        # Audit: compatible-but-route-unsatisfiable peers are recorded with a
        # DISTINCT reason code (never RANKED_BELOW_WINNER, never a compat code)
        # and the dependency-level cause. When NO satisfiable flat exists, also run
        # the full dependency options helper so the flat's own dependency rejection
        # reasons (exposure mismatch / dependency-missing) are recorded — never a
        # silent fallback.
        for f, cause in unsatisfiable_flats:
            ranked_out_records.append(RankedOutRecord(
                role="flat",
                candidate_id=f.candidate_id,
                content_sha256=f.descriptor.content_sha256,
                acquired_at=f.acquired_at,
                reason_code=ROUTE_UNSATISFIABLE,
                key=selection_key("flat", light, f),
                detail=cause,
            ))
            if not satisfiable_flats:
                _opts, recs, struct, audit, dep_unv, _dsel, _drank, dep_no_cand, _dtie = _flat_dependency_options(
                    f.descriptor, f, candidates, policy,
                    light=light, manual_selection=manual,
                )
                rejected.extend(audit)
                rejected.extend(recs)
                manifest_rejected.extend(recs)
                structural.extend(struct)
                unverified.extend(dep_unv)
                if dep_no_cand:
                    flat_dep_no_candidate = tuple(dict.fromkeys(flat_dep_no_candidate + dep_no_cand))

        if not satisfiable_flats and unsatisfiable_flats:
            flat_dep_missing = True

        by_form: dict[str, list[Candidate]] = {}
        for f in satisfiable_flats:
            by_form.setdefault(f.descriptor.flat_form, []).append(f)
        for form in sorted(by_form.keys()):
            fl = by_form[form]
            sel = select_role_candidates("flat", light, fl, manual_choice=manual.get("flat"))
            if sel.status == STATUS_AMBIGUOUS_TIE:
                role_tie = True
                tie_candidates_by_role["flat"] = [c for c in fl if c.candidate_id in set(sel.tie_candidates)]
                ambiguous_reasons.append(Reason(
                    "AMBIGUOUS_TIE", role="flat", field="flat", expected=sel.tie_candidates,
                    observed=None, blocking=False,
                ))
                continue
            if sel.winner is not None:
                form_winners_list.append((form, _find_candidate(fl, sel.winner.chosen_candidate_id)))
                selection_records.append(sel.winner)
                ranked_out_records.extend(sel.ranked_out)
        if len(form_winners_list) > 1:
            mixed_flat_form = True
            ambiguous_reasons.append(Reason(
                "MIXED_FLAT_FORM", role="flat", field="flat_form",
                expected=tuple(sorted(f for f, _ in form_winners_list)),
                observed=None, blocking=False,
            ))
        elif form_winners_list:
            flat_winner_cand = form_winners_list[0][1]

        if flat_winner_cand is not None:
            opts, recs, struct, audit, dep_unv, dep_sel, dep_ranked, dep_no_cand, dep_tie = _flat_dependency_options(
                flat_winner_cand.descriptor, flat_winner_cand, candidates, policy,
                light=light, manual_selection=manual,
            )
            rejected.extend(audit)
            rejected.extend(recs)
            manifest_rejected.extend(recs)
            structural.extend(struct)
            unverified.extend(dep_unv)
            selection_records.extend(dep_sel)
            ranked_out_records.extend(dep_ranked)
            flat_dep_no_candidate = dep_no_cand
            if dep_tie:
                role_tie = True
            flat_route_options = [dict(o) for o in opts]
            if not flat_route_options:
                flat_dep_missing = True
        else:
            # flat_mode apply but no satisfiable flat: no route (a route-
            # unsatisfiable flat is recorded in ranked_out as ROUTE_UNSATISFIABLE).
            flat_route_options = []

    # -- assemble chosen routes ---------------------------------------------
    routes: list[dict[str, Candidate]] = []
    if request.flat_mode == "apply":
        if flat_winner_cand is not None:
            for deps in flat_route_options:
                chosen: dict[str, Candidate] = dict(additive_winners)
                chosen["flat"] = flat_winner_cand
                chosen.update(deps)
                routes.append(chosen)
    else:
        routes.append(dict(additive_winners))

    # -- manual refusal (mirrors the pre-G2B filtering semantics) ------------
    manual_refusal = False
    if manual:
        known_ids = {c.candidate_id for role_cands in candidates.values() for c in role_cands}
        rejected_ids = {rec.candidate_id for rec in manifest_rejected}
        # A manual choice must name a USABLE candidate for its role: for the flat
        # role, a route-unsatisfiable flat is not usable (applicability filter).
        usable_ids_by_role = dict(compatible_ids_by_role)
        if request.flat_mode == "apply":
            usable_ids_by_role["flat"] = satisfiable_flat_ids
        unsatisfied = any(
            cid not in known_ids or cid not in usable_ids_by_role.get(role, ())
            for role, cid in manual.items()
        )
        if unsatisfied:
            manual_refusal = True
            explained = False
            for role, cid in manual.items():
                if cid not in known_ids:
                    structural.append(Reason("ROLE_UNAVAILABLE", "manual_selection", role=role, expected=cid, observed=None))
                    explained = True
                    break
            if not explained:
                if not any(cid in rejected_ids for cid in manual.values()):
                    for role, cid in sorted(manual.items()):
                        structural.append(Reason("MANUAL_SELECTION_NO_MATCH", "manual_selection", role=role, expected=cid, observed=None))

    unverified = _dedup_unverified(unverified)

    def _no_match():
        reason_codes = _dedup([r.code for r in structural] + [code for rec in manifest_rejected for code in rec.reason_codes])
        reasons = tuple(r for rec in rejected for r in rec.reasons) + tuple(structural)
        composition = _compute_composition(
            request, None, compatible_ids_by_role, flat_dep_no_candidate, flat_dep_missing,
        )
        return MatchResult(
            outcome=OUTCOME_NO_MATCH,
            rejected_candidates=tuple(rejected),
            reason_codes=tuple(sorted(reason_codes)),
            reasons=reasons,
            structural_reasons=tuple(structural),
            unverified=tuple(unverified),
            selection=tuple(selection_records),
            ranked_out=tuple(ranked_out_records),
            composition=composition,
            selection_policy_version=SELECTION_POLICY_VERSION,
        )

    def _ambiguous():
        # Evidence payload: AMBIGUOUS must carry the real competing alternatives
        # (never an empty mapping) plus a non-blocking diagnostic reason.
        if role_tie or mixed_flat_form:
            coherent = _build_ambiguous_coherent_sets(
                additive_role_order, additive_winners, tie_candidates_by_role,
                form_winners_list, flat_winner_cand, request,
            )
        else:
            coherent = tuple(routes)
        return MatchResult(
            outcome=OUTCOME_AMBIGUOUS,
            rejected_candidates=tuple(rejected),
            reason_codes=(),
            reasons=tuple(ambiguous_reasons),
            coherent_sets=coherent,
            unverified=tuple(unverified),
            selection=tuple(selection_records),
            ranked_out=tuple(ranked_out_records),
            composition=None,
            selection_policy_version=SELECTION_POLICY_VERSION,
        )

    if manual_refusal:
        return _no_match()
    if role_tie or mixed_flat_form:
        return _ambiguous()
    if any(r not in additive_winners for r in additive_role_order):
        return _no_match()
    if not routes:
        return _no_match()
    if len(routes) > 1:
        return _ambiguous()

    chosen = routes[0]
    composition = _compute_composition(
        request, chosen, compatible_ids_by_role, flat_dep_no_candidate, flat_dep_missing,
    )
    plan = _build_plan(
        light, request, chosen, policy,
        selection=tuple(selection_records),
        selection_policy_version=SELECTION_POLICY_VERSION,
        composition=composition,
        ranked_out=tuple(ranked_out_records),
    )
    return MatchResult(
        outcome=OUTCOME_MATCHED,
        plan=plan,
        rejected_candidates=tuple(rejected),
        reason_codes=(),
        coherent_sets=(chosen,),
        unverified=tuple(unverified),
        selection=tuple(selection_records),
        ranked_out=tuple(ranked_out_records),
        composition=composition,
        selection_policy_version=SELECTION_POLICY_VERSION,
    )


def _additive_state(applied: tuple[str, ...]) -> str:
    if "dark" in applied:
        return "dark_bias_removed" if "bias" in applied else "dark_incl_bias"
    if "bias" in applied:
        return "bias_only"
    return "none"


def _build_ambiguous_coherent_sets(
    additive_role_order,
    additive_winners: Mapping[str, Candidate],
    tie_candidates_by_role: Mapping[str, Sequence[Candidate]],
    form_winners_list: Sequence[Tuple[str, Candidate]],
    flat_winner_cand: Optional[Candidate],
    request: CalibrationRequest,
) -> tuple[Mapping[str, Candidate], ...]:
    """Build the competing coherent sets for a ranking-path AMBIGUOUS outcome.

    One alternative per tied candidate (exact-key tie) or per flat-form winner
    (mixed ``flat_form``). Deterministic order: candidates are sorted by
    ``(candidate_id, content_sha256)`` within a tie, and form winners are kept in
    their already-sorted form order. Never emits an empty mapping.
    """
    alternatives: dict[str, list[Candidate]] = {}
    for role in additive_role_order:
        if role in tie_candidates_by_role:
            tied = list(tie_candidates_by_role[role])
            tied.sort(key=lambda c: (c.candidate_id, c.descriptor.content_sha256))
            alternatives[role] = tied
        elif role in additive_winners:
            alternatives[role] = [additive_winners[role]]
    if request.flat_mode == "apply":
        if "flat" in tie_candidates_by_role:
            tied = list(tie_candidates_by_role["flat"])
            tied.sort(key=lambda c: (c.candidate_id, c.descriptor.content_sha256))
            alternatives["flat"] = tied
        elif form_winners_list:
            alternatives["flat"] = [c for _, c in form_winners_list]
        elif flat_winner_cand is not None:
            alternatives["flat"] = [flat_winner_cand]

    sets: list[dict[str, Candidate]] = [{}]
    for role in alternatives:
        next_sets: list[dict[str, Candidate]] = []
        for s in sets:
            for c in alternatives[role]:
                s2 = dict(s)
                s2[role] = c
                next_sets.append(s2)
        sets = next_sets
    return tuple(sets)


def _compute_composition(
    request: CalibrationRequest,
    chosen: Optional[Mapping[str, Candidate]],
    compatible_ids_by_role: Mapping[str, set],
    flat_dep_no_candidate: tuple[str, ...],
    flat_dep_missing: bool,
) -> CalibrationComposition:
    applied = tuple(sorted(chosen.keys())) if chosen is not None else ()
    required = list(request.required_roles)
    no_candidate: list[str] = []
    skipped: list[SkippedRole] = []
    for role in required:
        compat_ids = compatible_ids_by_role.get(role, set())
        if not compat_ids:
            no_candidate.append(role)
        elif chosen is None or role not in chosen:
            reason = "FLAT_ADDITIVE_DEPENDENCY_MISSING" if (role == "flat" and flat_dep_missing) else "NOT_APPLIED"
            skipped.append(SkippedRole(role, reason, ""))
    for dr in flat_dep_no_candidate:
        if dr not in no_candidate:
            no_candidate.append(dr)

    if not applied:
        level = "NONE"
    elif skipped or no_candidate:
        level = "PARTIAL"
    else:
        level = "COMPLETE"

    return CalibrationComposition(
        applied_roles=applied,
        skipped_roles=tuple(skipped),
        level=level,
        additive_state=_additive_state(applied),
        flat_applied="flat" in applied,
        no_candidate_roles=tuple(sorted(set(no_candidate))),
    )


def _build_plan(
    light,
    request,
    chosen: Mapping[str, Candidate],
    policy: MatchPolicy,
    *,
    selection: tuple = (),
    selection_policy_version: str = "",
    composition: Optional[CalibrationComposition] = None,
    ranked_out: tuple = (),
) -> CalibrationPlan:
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
            acquired_at=c.acquired_at,
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
        selection=tuple(selection),
        selection_policy_version=selection_policy_version,
        composition=composition,
        ranked_out=tuple(ranked_out),
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
