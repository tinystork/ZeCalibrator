"""Pure scientific route enumeration (auto-route) for the Standard UX.

Standard expresses ONLY user intent ("use the compatible masters I supplied");
ZeCalibrator resolves the scientific calibration route automatically from each
candidate's declared compatibility and scientific facts. This module is pure
(no filesystem I/O, no Qt): it **reuses** the existing compatibility helpers in
:mod:`zecalibrator.core.matching` (``_candidate_compatibility`` and
``_flat_dependency_options``) and never reimplements compatibility.

A :class:`Route` is a fully-resolved scientific choice:

* ``additive_mode`` — one of the frozen ``CalibrationRequest`` additive modes
  (``control`` / ``bias_only`` / ``dark_incl_bias`` / ``dark_bias_removed``).
* ``flat_mode`` — ``none`` or ``apply``.
* ``flat_prep_mode`` — how a flat is prepared (``already_normalized`` /
  ``normalize_only`` / ``flat_dark_incl_bias`` / ``flat_dark_bias_removed`` /
  ``bias_only_flat``).
* ``masters`` — the exact chosen :class:`~zecalibrator.core.plans.Candidate`
  per role, including flat dependency roles ``flat_dark`` / ``bias_flat``.

Additive classification is strict and never invents a hidden default:

* no compatible dark, no compatible bias  -> ``control`` (passthrough).
* no compatible dark, compatible bias     -> ``bias_only`` (partial).
* compatible darks all ``included``       -> ``dark_incl_bias``.
* compatible darks all ``removed``        -> ``dark_bias_removed`` (requires a
  compatible bias; none => no route => ``NEEDS_ATTENTION``).
* mixed ``included``+``removed``          -> ``AMBIGUOUS``.
* any compatible dark ``unknown``         -> ``NEEDS_ATTENTION`` (never silently
  pick ``dark_incl_bias`` or fall back to ``bias_only``).

A present bias never causes a second subtraction when a dark route is chosen:
``dark_incl_bias`` does **not** consume the bias role. ``flat_dark`` is ONLY ever
a flat dependency, never a direct light correction.

Flat classification:

* no compatible flat                    -> flat ``none``.
* compatible flat(s) determine preparation from ``flat_form``:
  ``normalized_response`` -> ``already_normalized`` (flat quality evidence
  required via ``_flat_evidence_reasons``);
  ``corrected_unnormalized`` -> ``normalize_only`` (additive correction already
  occurred; executor normalizes directly, no flat_dark/bias_flat, no proof);
  ``raw_response`` -> unsupported in Standard (``FLAT_UNSUPPORTED_RAW``; a raw
  flat can never auto-construct a flat_dark dependency here).
* mixed ``flat_form`` values            -> ``AMBIGUOUS``.

Combine + count (Cartesian additive x flat): 0 routes => ``NEEDS_ATTENTION``;
exactly 1 => ``READY`` (including ``control``/``bias_only`` passthrough/partial
routes); more than 1 => ``AMBIGUOUS``. The resolver never ranks or picks a winner
arbitrarily. A flat that was supplied but rejected for a *scientific*
incompatibility (geometry/detector/filter/train/units) falls back to flat ``none``
(passthrough) with the rejection reasons preserved; a flat rejected for
*quality/usage* reasons (``_flat_evidence_reasons``) stays ``NEEDS_ATTENTION``
(never silently ignored).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from zecalibrator.core.descriptors import LightConstraints, MasterDescriptor
from zecalibrator.core.matching import (
    Reason,
    _CHECK_BIAS_RANGE,
    _CHECK_DARK_EXPOSURE,
    _CHECK_NONE,
    _candidate_compatibility,
    _collapse_duplicates,
    _dedup_unverified,
    _role_of,
    _sorted_candidates,
    _split_reasons,
)
from zecalibrator.core.plans import Candidate, MatchPolicy, RejectedMasterRecord
from zecalibrator.core.selection import STATUS_AMBIGUOUS_TIE, select_role_candidates

OUTCOME_READY = "READY"
OUTCOME_NEEDS_ATTENTION = "NEEDS_ATTENTION"
OUTCOME_AMBIGUOUS = "AMBIGUOUS"

# Structured reason codes emitted by the enumerator (in addition to the reused
# compatibility reason codes like GAIN_MISMATCH / MISSING_REQUIRED_FIELD).
BIAS_STATE_UNKNOWN = "BIAS_STATE_UNKNOWN"
BIAS_REQUIRED = "BIAS_REQUIRED"
FLAT_UNUSABLE = "FLAT_UNUSABLE"
FLAT_UNSUPPORTED_RAW = "FLAT_UNSUPPORTED_RAW"
NO_APPLICABLE_MASTER = "NO_APPLICABLE_MASTER"
# G2C: a RAW sensor-domain light with NO qualified additive correction applied
# must NOT receive a flat alone (unsafe multiplicative flat on uncorrected RAW).
# Non-blocking audit: the flat is skipped; the route is passthrough RAW.
ADDITIVE_PREREQUISITE_MISSING = "ADDITIVE_PREREQUISITE_MISSING"

# Semantic source for a Standard-contract-default assignment (R3D-C). A supplied
# master whose bias_state/flat_form was assigned by the contract (derivable from
# ``additive_history_state == "unknown"``) is recorded honestly with this source
# — never claimed as FITS evidence.
STANDARD_MASTER_CONTRACT = "standard_master_contract"

_INDETERMINATE_BIAS_STATES = ("unknown", "not_applicable")


@dataclass(frozen=True)
class Route:
    """One fully-resolved scientific route (additive + flat + chosen masters)."""

    additive_mode: str
    flat_mode: str
    flat_prep_mode: Optional[str]
    masters: Mapping[str, Candidate]

    def __post_init__(self) -> None:
        object.__setattr__(self, "masters", MappingProxyType(dict(self.masters)))


@dataclass(frozen=True)
class RouteEnumeration:
    """The complete route enumeration and its deterministic outcome."""

    outcome: str
    routes: Tuple[Route, ...]
    reasons: Tuple[Reason, ...]
    unverified: Tuple[Reason, ...]
    contract_defaults: Tuple[Mapping[str, object], ...] = ()
    rejected_masters: Tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "unverified", tuple(self.unverified))
        object.__setattr__(self, "contract_defaults", tuple(self.contract_defaults))
        object.__setattr__(self, "rejected_masters", tuple(self.rejected_masters))


def _hashable(value):
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _dedup_reasons(reasons) -> list[Reason]:
    """Dedup blocking reasons by ``(code, field, role, parent, expected, observed)``."""
    seen = set()
    out = []
    for r in reasons:
        key = (r.code, r.field, r.role, r.parent, _hashable(r.expected), _hashable(r.observed))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _find_candidate(cands, candidate_id: str) -> Optional[Candidate]:
    for c in cands:
        if c.candidate_id == candidate_id:
            return c
    return None


def _rank_one(role: str, light: LightConstraints, cands):
    """Return ``(winner, tied)`` for ``role``.

    Reuses :func:`zecalibrator.core.selection.select_role_candidates` (the one
    ranking implementation) — never a second ranking. ``winner`` is the single
    ranked :class:`Candidate` (or ``None`` when absent); ``tied`` is the tuple of
    tied candidates when the selector reported ``AMBIGUOUS_TIE`` (else ``()``).
    """
    sel = select_role_candidates(role, light, cands)
    if sel.status == STATUS_AMBIGUOUS_TIE:
        tied = [c for c in cands if c.candidate_id in set(sel.tie_candidates)]
        return None, tuple(tied)
    if sel.winner is None:
        return None, ()
    return _find_candidate(cands, sel.winner.chosen_candidate_id), ()


def _compatible_candidates(
    light: LightConstraints,
    candidates: Mapping[str, Sequence[Candidate]],
    role: str,
    policy: MatchPolicy,
    *,
    check: str,
    reference_exposure,
    reference_bias_range,
    check_filter: bool,
    check_optical: bool,
    expected_units: str,
    flat_extra: bool,
    standard_contract: bool = False,
) -> Tuple[list, list, list]:
    """Return ``(compatible, rejected, unverified)`` for one role.

    Reuses the existing ``_candidate_compatibility`` helper verbatim (no
    reimplementation of compatibility). Candidates whose semantic
    ``master_type`` differs from ``role`` are skipped (never trusted by key).
    ``rejected`` is a list of ``(candidate, blocking_reasons)`` so the
    considered-but-rejected masters can be recorded in the audit.
    """
    compatible = []
    rejected = []
    unverified = []
    for c in _sorted_candidates(_collapse_duplicates(candidates.get(role, ()))):
        if _role_of(c.descriptor) != role:
            continue
        reasons = _candidate_compatibility(
            light,
            c.descriptor,
            policy=policy,
            check=check,
            reference_exposure=reference_exposure,
            reference_bias_range=reference_bias_range,
            check_filter=check_filter,
            check_optical=check_optical,
            expected_units=expected_units,
            flat_extra=flat_extra,
            standard_contract=standard_contract,
        )
        blocking, unv = _split_reasons(reasons)
        if blocking:
            rejected.append((c, blocking))
        else:
            compatible.append(c)
            unverified.extend(unv)
    return compatible, rejected, unverified


def _contract_default_records(desc, role: str) -> list[dict]:
    """Honest audit records for a Standard-contract-default semantic assignment.

    A session-supplied master whose ``additive_history_state == "unknown"``
    carries the Standard master contract semantics: the recorded
    ``bias_state``/``flat_form`` is a *contract default*, never FITS evidence.
    The audit records the semantic source (``standard_master_contract``) and the
    precise note that processing provenance from FITS is unavailable.
    """
    if desc.processing_provenance.additive_history_state != "unknown":
        return []
    if role in ("dark", "flat_dark") and desc.bias_state == "included":
        return [{
            "role": role,
            "field": "bias_state",
            "value": "included",
            "source": STANDARD_MASTER_CONTRACT,
            "provenance_note": "processing provenance from FITS = unavailable",
        }]
    if role == "flat" and desc.flat_form == "corrected_unnormalized":
        return [{
            "role": role,
            "field": "flat_form",
            "value": "corrected_unnormalized",
            "source": STANDARD_MASTER_CONTRACT,
            "provenance_note": "processing provenance from FITS = unavailable",
        }]
    return []


def _flat_filter_unrelated(light: LightConstraints, desc) -> bool:
    """S1: a flat is unrelated to the light when its filter is KNOWN and differs.

    Standard enumerates a conventional third-party master flat; a flat for a
    different filter is not a candidate for this light and must never force
    NEEDS_ATTENTION (nor pollute the audit). The rule is deliberately minimal:
    only a KNOWN-and-different filter is excluded; an unknown filter on either
    side stays candidate-relevant (conservative).
    """
    lf = light.optical.filter
    mf = desc.filter
    return mf is not None and lf is not None and mf != lf


def _flat_quality_rejection(flat_reject) -> bool:
    """True when a supplied flat was rejected for quality/usage (never silent).

    ``_flat_evidence_reasons`` / ``_normalization_coherence_reasons`` set
    ``role="flat"`` on their reasons; the general compatibility reasons (geometry/
    detector/filter/train/units/gain/offset) do not. A flat rejected only for the
    latter is scientifically inapplicable (passthrough), while a flat rejected for
    the former is quality-refused and must stay NEEDS_ATTENTION.
    """
    return any(r.role == "flat" for r in flat_reject)


def enumerate_routes(
    light: LightConstraints,
    candidates: Mapping[str, Sequence[Candidate]],
    policy: MatchPolicy,
) -> RouteEnumeration:
    """Enumerate every coherent scientific route and classify the outcome.

    Pure and deterministic; never picks a winner and never invents a default.
    """
    reasons: list[Reason] = []
    unverified: list[Reason] = []
    contract_defaults: list[dict] = []

    light_exposure = light.acquisition.exposure_s
    light_bias_range = light.acquisition.bias_exposure_max_s

    # ------------------------------------------------------------------ additive
    compatible_darks, dark_rejected, dark_unv = _compatible_candidates(
        light, candidates, "dark", policy,
        check=_CHECK_DARK_EXPOSURE, reference_exposure=light_exposure, reference_bias_range=None,
        check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
        standard_contract=True,
    )
    compatible_biases, bias_rejected, bias_unv = _compatible_candidates(
        light, candidates, "bias", policy,
        check=_CHECK_BIAS_RANGE, reference_exposure=None, reference_bias_range=light_bias_range,
        check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
        standard_contract=True,
    )
    dark_reject = [r for _, rs in dark_rejected for r in rs]
    bias_reject = [r for _, rs in bias_rejected for r in rs]
    reasons.extend(dark_reject)
    reasons.extend(bias_reject)
    unverified.extend(dark_unv)
    unverified.extend(bias_unv)

    # Standard-contract-default semantic source audit (R3D-C): a supplied
    # dark/flat_dark/flat whose bias_state/flat_form was assigned by the
    # contract (unknown additive history) is recorded honestly — never claimed
    # as FITS evidence.
    for role in ("dark", "flat_dark", "flat"):
        for c in _collapse_duplicates(candidates.get(role, ())):
            if _role_of(c.descriptor) != role:
                continue
            contract_defaults.extend(_contract_default_records(c.descriptor, role))

    # A dark with an indeterminate bias state is surfaced truthfully regardless of
    # its scientific compatibility (its processing history cannot establish
    # whether bias has already been removed).
    for c in _collapse_duplicates(candidates.get("dark", ())):
        if _role_of(c.descriptor) != "dark":
            continue
        if c.descriptor.bias_state in _INDETERMINATE_BIAS_STATES:
            reasons.append(
                Reason(
                    BIAS_STATE_UNKNOWN, "bias_state", role="dark", parent="dark",
                    expected="included|removed", observed=c.descriptor.bias_state,
                )
            )

    additive_options: list[tuple[str, dict]] = []
    additive_ambiguous = False

    def _tie_reason(role, tied):
        return Reason(
            "AMBIGUOUS_TIE", role=role, field=role,
            expected=tuple(sorted({c.candidate_id for c in tied})),
            observed=None, blocking=False,
        )

    def _emit_ranked(role, light, cands, amode, *, base=None):
        """Rank ``cands`` for ``role``; append one option per winner (or, on a tie,
        one option per tied candidate). Returns the tied tuple (or ``()``)."""
        winner, tied = _rank_one(role, light, cands)
        if tied:
            for c in tied:
                m = dict(base or {})
                m[role] = c
                additive_options.append((amode, m))
            return tied
        if winner is not None:
            m = dict(base or {})
            m[role] = winner
            additive_options.append((amode, m))
        return ()

    def _emit_pair(role1, light, cands1, role2, cands2, amode):
        """Rank two roles and combine winners into ``dark_bias_removed`` options
        (cartesian product on any tie). Returns ``(tie1, tie2)``."""
        w1, t1 = _rank_one(role1, light, cands1)
        w2, t2 = _rank_one(role2, light, cands2)
        c1 = list(t1) if t1 else ([w1] if w1 is not None else [])
        c2 = list(t2) if t2 else ([w2] if w2 is not None else [])
        if not c1 or not c2:
            return t1, t2
        for a in c1:
            for b in c2:
                additive_options.append((amode, {role1: a, role2: b}))
        return t1, t2

    if not compatible_darks:
        if not compatible_biases:
            additive_options.append(("control", {}))
        else:
            # G2B: same-role bias peers are ranked (most-recent-first) instead of
            # each producing a separate route.
            tied = _emit_ranked("bias", light, compatible_biases, "bias_only")
            if tied:
                additive_ambiguous = True
                reasons.append(_tie_reason("bias", tied))
    else:
        included = [d for d in compatible_darks if d.descriptor.bias_state == "included"]
        removed = [d for d in compatible_darks if d.descriptor.bias_state == "removed"]
        indeterminate = [d for d in compatible_darks if d.descriptor.bias_state in _INDETERMINATE_BIAS_STATES]
        if indeterminate:
            # Indeterminate bias state: never silently pick dark_incl_bias or fall
            # back to bias_only. No additive route is produced.
            additive_options = []
        elif included and removed:
            additive_ambiguous = True
            # G2B: rank each bias-state group to a single winner; the
            # included-vs-removed SCIENTIFIC route ambiguity is unchanged.
            d_incl_tie = _emit_ranked("dark", light, included, "dark_incl_bias")
            if d_incl_tie:
                reasons.append(_tie_reason("dark", d_incl_tie))
            if compatible_biases:
                d_rem_tie, b_tie = _emit_pair("dark", light, removed, "bias", compatible_biases, "dark_bias_removed")
                if d_rem_tie:
                    reasons.append(_tie_reason("dark", d_rem_tie))
                if b_tie:
                    reasons.append(_tie_reason("bias", b_tie))
            else:
                reasons.append(
                    Reason(BIAS_REQUIRED, role="bias", parent="dark",
                           expected="compatible bias", observed=None)
                )
        elif included:
            d_incl_tie = _emit_ranked("dark", light, included, "dark_incl_bias")
            if d_incl_tie:
                additive_ambiguous = True
                reasons.append(_tie_reason("dark", d_incl_tie))
        else:  # removed only
            if not compatible_biases:
                reasons.append(
                    Reason(BIAS_REQUIRED, role="bias", parent="dark",
                           expected="compatible bias", observed=None)
                )
                additive_options = []
            else:
                d_rem_tie, b_tie = _emit_pair("dark", light, removed, "bias", compatible_biases, "dark_bias_removed")
                if d_rem_tie:
                    additive_ambiguous = True
                    reasons.append(_tie_reason("dark", d_rem_tie))
                if b_tie:
                    additive_ambiguous = True
                    reasons.append(_tie_reason("bias", b_tie))

    # ---------------------------------------------------------------------- flat
    compatible_flats = []
    flat_reject = []
    flat_rejected = []
    flat_unv = []
    flat_supplied = False
    for c in _sorted_candidates(_collapse_duplicates(candidates.get("flat", ()))):
        if _role_of(c.descriptor) != "flat":
            continue
        if _flat_filter_unrelated(light, c.descriptor):
            # S1: an unrelated flat (known-and-different filter) is not a
            # candidate for this light; never counted as "supplied", never
            # forces NEEDS_ATTENTION, never pollutes the audit.
            continue
        flat_supplied = True
        expected_units = "dimensionless" if c.descriptor.flat_form == "normalized_response" else "ADU"
        rs = _candidate_compatibility(
            light, c.descriptor, policy=policy,
            check=_CHECK_NONE, reference_exposure=None, reference_bias_range=None,
            check_filter=True, check_optical=True, expected_units=expected_units, flat_extra=True,
            standard_contract=True,
        )
        blocking, unv = _split_reasons(rs)
        if blocking:
            flat_reject.extend(blocking)
            flat_rejected.append((c, blocking))
        else:
            compatible_flats.append(c)
            flat_unv.extend(unv)
    reasons.extend(flat_reject)
    unverified.extend(flat_unv)

    flat_options: list[tuple[str, Optional[str], dict]] = []
    flat_ambiguous = False
    if not compatible_flats:
        if flat_supplied:
            # A flat was supplied but every candidate was rejected. Distinguish
            # scientific incompatibility (geometry/detector/filter/train/units —
            # no ``role``) from quality/usage refusal (``_flat_evidence_reasons``
            # sets ``role="flat"``): an incompatible flat falls back to flat
            # ``none`` (passthrough) with its reasons preserved, while a
            # quality-refused flat stays NEEDS_ATTENTION (never silently ignored).
            if _flat_quality_rejection(flat_reject):
                flat_options = []
            else:
                flat_options.append(("none", None, {}))
        else:
            flat_options.append(("none", None, {}))
    else:
        flat_forms = {f.descriptor.flat_form for f in compatible_flats}
        if len(flat_forms) > 1:
            flat_ambiguous = True
        # G2B: applicability-before-ranking — rank only route-satisfiable flats.
        # In Standard, corrected_unnormalized / normalized_response are always
        # satisfiable (no dependency); a raw_response flat is never applicable
        # (FLAT_UNSUPPORTED_RAW), so it is never ranked (a raw flat can never
        # shadow a satisfiable flat).
        by_form: dict[str, list] = {}
        for f in compatible_flats:
            by_form.setdefault(f.descriptor.flat_form, []).append(f)
        for form in sorted(by_form.keys()):
            ff = form
            if ff == "raw_response":
                # R3D-C Standard: a raw flat is unsupported (flat-master
                # construction from raw stacks is future/Advanced). Never
                # auto-construct a flat_dark dependency here, and never rank it.
                reasons.append(Reason(FLAT_UNSUPPORTED_RAW, "flat_form", role="flat", observed=ff))
                continue
            if ff not in ("normalized_response", "corrected_unnormalized"):
                reasons.append(Reason("UNDOCUMENTED_PROCESSING", "flat_form", role="flat", observed=ff))
                continue
            f, tied = _rank_one("flat", light, by_form[form])
            if tied:
                flat_ambiguous = True
                reasons.append(Reason(
                    "AMBIGUOUS_TIE", role="flat", field="flat",
                    expected=tuple(sorted({c.candidate_id for c in tied})),
                    observed=None, blocking=False,
                ))
                for c in tied:
                    if ff == "normalized_response":
                        flat_options.append(("apply", "already_normalized", {"flat": c}))
                    elif ff == "corrected_unnormalized":
                        flat_options.append(("apply", "normalize_only", {"flat": c}))
                continue
            if f is None:
                continue
            if ff == "normalized_response":
                flat_options.append(("apply", "already_normalized", {"flat": f}))
            elif ff == "corrected_unnormalized":
                # R3D-E F3: the additive flat correction has ALREADY occurred;
                # the executor normalizes the corrected-but-unnormalized flat
                # directly (no flat_dark/bias_flat, no normalization proof).
                flat_options.append(("apply", "normalize_only", {"flat": f}))

    flat_unusable = bool(compatible_flats) and not flat_options
    if flat_unusable:
        reasons.append(
            Reason(FLAT_UNUSABLE, "flat_dark", role="flat", parent="flat",
                   expected="flat_dark|bias_flat dependency", observed=None)
        )

    # ------------------------------------------------------------------- combine
    routes: list[Route] = []
    flat_prereq_missing_emitted = False
    for amode, am in additive_options:
        additive_applied = bool(am)
        effective_flat_options = flat_options
        # G2C RAW flat-only guard: a RAW sensor-domain light with NO additive
        # master applied by this additive option MUST NOT receive a flat alone
        # (a multiplicative flat on an uncorrected RAW light spatially modulates
        # the pedestal). The flat is SKIPPED (not applied); the additive option
        # still resolves to its own passthrough route. The prerequisite is about
        # what the resolved plan ACTUALLY applied — never "a dark exists in the
        # library". bias_only / dark_incl_bias / dark_bias_removed all apply a
        # real additive correction and keep their flat pairing.
        if light.raw_domain_declaration == "raw" and not additive_applied:
            if any(o[0] == "apply" for o in flat_options):
                if not flat_prereq_missing_emitted:
                    reasons.append(Reason(
                        ADDITIVE_PREREQUISITE_MISSING, "flat", role="flat",
                        parent="additive",
                        expected="additive correction applied",
                        observed=amode, blocking=False,
                    ))
                    flat_prereq_missing_emitted = True
                effective_flat_options = [("none", None, {})]
        for fmode, fprep, fm in effective_flat_options:
            masters = dict(am)
            masters.update(fm)
            routes.append(Route(amode, fmode, fprep, masters))

    forced_ambiguous = additive_ambiguous or flat_ambiguous
    if not routes:
        outcome = OUTCOME_NEEDS_ATTENTION
    elif forced_ambiguous:
        outcome = OUTCOME_AMBIGUOUS
    elif len(routes) == 1:
        # G2B R1: control / bias_only / passthrough are legitimate READY routes;
        # the level is carried by the plan composition, not by a route flag.
        outcome = OUTCOME_READY
    else:
        outcome = OUTCOME_AMBIGUOUS

    # A READY passthrough (no master bound) is surfaced truthfully as a
    # non-blocking audit entry (never silent, never an error).
    if outcome == OUTCOME_READY and routes and not routes[0].masters:
        reasons.append(Reason(NO_APPLICABLE_MASTER, blocking=False))

    reasons = _dedup_reasons(reasons)
    unverified = _dedup_unverified(unverified)

    # Considered-but-rejected masters (never silently invisible): a supplied
    # master evaluated and rejected by compatibility is recorded with its role
    # and reason codes, so a passthrough plan still records what was supplied.
    rejected_masters: list[RejectedMasterRecord] = []
    seen: set = set()
    for role, rejected in (("dark", dark_rejected), ("bias", bias_rejected), ("flat", flat_rejected)):
        for c, blocking in rejected:
            content = c.descriptor.content_sha256
            key = (role, c.candidate_id, content)
            if key in seen:
                continue
            seen.add(key)
            rejected_masters.append(RejectedMasterRecord(
                role=role,
                candidate_id=c.candidate_id,
                content_sha256=content,
                reason_codes=tuple(dict.fromkeys(r.code for r in blocking)),
            ))

    return RouteEnumeration(
        outcome, tuple(routes), tuple(reasons), tuple(unverified), tuple(contract_defaults),
        tuple(rejected_masters),
    )


__all__ = [
    "ADDITIVE_PREREQUISITE_MISSING",
    "BIAS_REQUIRED",
    "BIAS_STATE_UNKNOWN",
    "FLAT_UNSUPPORTED_RAW",
    "FLAT_UNUSABLE",
    "NO_APPLICABLE_MASTER",
    "OUTCOME_AMBIGUOUS",
    "OUTCOME_NEEDS_ATTENTION",
    "OUTCOME_READY",
    "STANDARD_MASTER_CONTRACT",
    "Route",
    "RouteEnumeration",
    "enumerate_routes",
]
