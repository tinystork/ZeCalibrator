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
  ``flat_dark_incl_bias`` / ``flat_dark_bias_removed`` / ``bias_only_flat``).
* ``masters`` — the exact chosen :class:`~zecalibrator.core.plans.Candidate`
  per role, including flat dependency roles ``flat_dark`` / ``bias_flat``.

Additive classification is strict and never invents a hidden default:

* no compatible dark, no compatible bias  -> ``control`` (partial).
* no compatible dark, compatible bias     -> ``bias_only`` (partial, never full).
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
  ``normalized_response`` / ``corrected_unnormalized`` -> ``already_normalized``
  (flat quality evidence required via ``_flat_evidence_reasons``);
  ``raw_response`` -> reuse ``_flat_dependency_options`` for flat_dark/bias
  dependencies (``flat_dark_incl_bias`` / ``flat_dark_bias_removed`` /
  ``bias_only_flat``); zero dependencies => ``NEEDS_ATTENTION`` (flat supplied
  but unusable, never silently ``flat none``).
* mixed ``flat_form`` values            -> ``AMBIGUOUS``.

Combine + count (Cartesian additive x flat): 0 complete (full) routes =>
``NEEDS_ATTENTION``; exactly 1 => ``READY``; more than 1 => ``AMBIGUOUS``. A
``control``/``bias_only`` route is *partial* and is therefore reported as
``NEEDS_ATTENTION`` rather than ``READY``. The resolver never ranks or picks a
winner arbitrarily.
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
    _flat_dependency_options,
    _role_of,
    _sorted_candidates,
    _split_reasons,
)
from zecalibrator.core.plans import Candidate, MatchPolicy

OUTCOME_READY = "READY"
OUTCOME_NEEDS_ATTENTION = "NEEDS_ATTENTION"
OUTCOME_AMBIGUOUS = "AMBIGUOUS"

# Structured reason codes emitted by the enumerator (in addition to the reused
# compatibility reason codes like GAIN_MISMATCH / MISSING_REQUIRED_FIELD).
BIAS_STATE_UNKNOWN = "BIAS_STATE_UNKNOWN"
BIAS_REQUIRED = "BIAS_REQUIRED"
FLAT_UNUSABLE = "FLAT_UNUSABLE"
PARTIAL_ADDITIVE = "PARTIAL_ADDITIVE"

_INDETERMINATE_BIAS_STATES = ("unknown", "not_applicable")


@dataclass(frozen=True)
class Route:
    """One fully-resolved scientific route (additive + flat + chosen masters)."""

    additive_mode: str
    flat_mode: str
    flat_prep_mode: Optional[str]
    masters: Mapping[str, Candidate]
    partial: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "masters", MappingProxyType(dict(self.masters)))


@dataclass(frozen=True)
class RouteEnumeration:
    """The complete route enumeration and its deterministic outcome."""

    outcome: str
    routes: Tuple[Route, ...]
    reasons: Tuple[Reason, ...]
    unverified: Tuple[Reason, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "unverified", tuple(self.unverified))


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
) -> Tuple[list, list, list]:
    """Return ``(compatible, blocking_reasons, unverified)`` for one role.

    Reuses the existing ``_candidate_compatibility`` helper verbatim (no
    reimplementation of compatibility). Candidates whose semantic
    ``master_type`` differs from ``role`` are skipped (never trusted by key).
    """
    compatible = []
    blocking_reasons = []
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
        )
        blocking, unv = _split_reasons(reasons)
        if blocking:
            blocking_reasons.extend(blocking)
        else:
            compatible.append(c)
            unverified.extend(unv)
    return compatible, blocking_reasons, unverified


def _flat_prep_from_deps(deps: Mapping[str, Candidate]) -> str:
    """Map a ``_flat_dependency_options`` dependency binding to a flat-prep mode."""
    if "flat_dark" in deps:
        if deps["flat_dark"].descriptor.bias_state == "removed":
            return "flat_dark_bias_removed"
        return "flat_dark_incl_bias"
    if "bias_flat" in deps:
        return "bias_only_flat"
    return "already_normalized"  # unreachable for raw_response dependencies


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

    light_exposure = light.acquisition.exposure_s
    light_bias_range = light.acquisition.bias_exposure_max_s

    # ------------------------------------------------------------------ additive
    compatible_darks, dark_reject, dark_unv = _compatible_candidates(
        light, candidates, "dark", policy,
        check=_CHECK_DARK_EXPOSURE, reference_exposure=light_exposure, reference_bias_range=None,
        check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
    )
    compatible_biases, bias_reject, bias_unv = _compatible_candidates(
        light, candidates, "bias", policy,
        check=_CHECK_BIAS_RANGE, reference_exposure=None, reference_bias_range=light_bias_range,
        check_filter=False, check_optical=False, expected_units="ADU", flat_extra=False,
    )
    reasons.extend(dark_reject)
    reasons.extend(bias_reject)
    unverified.extend(dark_unv)
    unverified.extend(bias_unv)

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

    additive_options: list[tuple[str, bool, dict]] = []
    additive_ambiguous = False

    if not compatible_darks:
        if not compatible_biases:
            additive_options.append(("control", True, {}))
        else:
            for b in compatible_biases:
                additive_options.append(("bias_only", True, {"bias": b}))
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
            for d in included:
                additive_options.append(("dark_incl_bias", False, {"dark": d}))
            if compatible_biases:
                for d in removed:
                    for b in compatible_biases:
                        additive_options.append(("dark_bias_removed", False, {"dark": d, "bias": b}))
            else:
                reasons.append(
                    Reason(BIAS_REQUIRED, role="bias", parent="dark",
                           expected="compatible bias", observed=None)
                )
        elif included:
            for d in included:
                additive_options.append(("dark_incl_bias", False, {"dark": d}))
        else:  # removed only
            if not compatible_biases:
                reasons.append(
                    Reason(BIAS_REQUIRED, role="bias", parent="dark",
                           expected="compatible bias", observed=None)
                )
                additive_options = []
            else:
                for d in removed:
                    for b in compatible_biases:
                        additive_options.append(("dark_bias_removed", False, {"dark": d, "bias": b}))

    # ---------------------------------------------------------------------- flat
    compatible_flats = []
    flat_reject = []
    flat_unv = []
    for c in _sorted_candidates(_collapse_duplicates(candidates.get("flat", ()))):
        if _role_of(c.descriptor) != "flat":
            continue
        expected_units = "dimensionless" if c.descriptor.flat_form == "normalized_response" else "ADU"
        rs = _candidate_compatibility(
            light, c.descriptor, policy=policy,
            check=_CHECK_NONE, reference_exposure=None, reference_bias_range=None,
            check_filter=True, check_optical=True, expected_units=expected_units, flat_extra=True,
        )
        blocking, unv = _split_reasons(rs)
        if blocking:
            flat_reject.extend(blocking)
        else:
            compatible_flats.append(c)
            flat_unv.extend(unv)
    reasons.extend(flat_reject)
    unverified.extend(flat_unv)

    flat_options: list[tuple[str, Optional[str], dict]] = []
    flat_ambiguous = False
    if not compatible_flats:
        flat_options.append(("none", None, {}))
    else:
        flat_forms = {f.descriptor.flat_form for f in compatible_flats}
        if len(flat_forms) > 1:
            flat_ambiguous = True
        for f in compatible_flats:
            ff = f.descriptor.flat_form
            if ff in ("normalized_response", "corrected_unnormalized"):
                flat_options.append(("apply", "already_normalized", {"flat": f}))
            elif ff == "raw_response":
                opts, recs, struct, _audit, unv = _flat_dependency_options(
                    f.descriptor, f, candidates, policy
                )
                unverified.extend(unv)
                for rec in recs:
                    reasons.extend(rec.reasons)
                reasons.extend(struct)
                for deps in opts:
                    flat_options.append(("apply", _flat_prep_from_deps(deps), {"flat": f, **deps}))
            else:
                reasons.append(Reason("UNDOCUMENTED_PROCESSING", "flat_form", role="flat", observed=ff))

    flat_unusable = bool(compatible_flats) and not flat_options
    if flat_unusable:
        reasons.append(
            Reason(FLAT_UNUSABLE, "flat_dark", role="flat", parent="flat",
                   expected="flat_dark|bias_flat dependency", observed=None)
        )

    # ------------------------------------------------------------------- combine
    routes: list[Route] = []
    for amode, partial, am in additive_options:
        for fmode, fprep, fm in flat_options:
            masters = dict(am)
            masters.update(fm)
            routes.append(Route(amode, fmode, fprep, masters, partial))

    forced_ambiguous = additive_ambiguous or flat_ambiguous
    if not routes:
        outcome = OUTCOME_NEEDS_ATTENTION
    elif forced_ambiguous:
        outcome = OUTCOME_AMBIGUOUS
    elif len(routes) == 1:
        outcome = OUTCOME_READY if not routes[0].partial else OUTCOME_NEEDS_ATTENTION
    else:
        outcome = OUTCOME_AMBIGUOUS

    if outcome == OUTCOME_NEEDS_ATTENTION and routes and all(r.partial for r in routes):
        reasons.append(
            Reason(PARTIAL_ADDITIVE, "additive_mode",
                   observed=tuple(sorted({r.additive_mode for r in routes})))
        )

    reasons = _dedup_reasons(reasons)
    unverified = _dedup_unverified(unverified)
    return RouteEnumeration(outcome, tuple(routes), tuple(reasons), tuple(unverified))


__all__ = [
    "BIAS_REQUIRED",
    "BIAS_STATE_UNKNOWN",
    "FLAT_UNUSABLE",
    "OUTCOME_AMBIGUOUS",
    "OUTCOME_NEEDS_ATTENTION",
    "OUTCOME_READY",
    "PARTIAL_ADDITIVE",
    "Route",
    "RouteEnumeration",
    "enumerate_routes",
]
