"""Application-layer auto-route resolver (NOT public ``api.v1``).

This is the application-layer boundary the Standard UX uses to turn user intent
("use the compatible masters I supplied") into the existing
:class:`~zecalibrator.core.plans.CalibrationRequest` /
:class:`~zecalibrator.core.plans.CalibrationPlan` values. It runs the pure
:func:`zecalibrator.core.routes.enumerate_routes` and, for exactly one READY
route, binds the final plan through the EXISTING
:func:`zecalibrator.core.matching.match_calibration` (which performs the final
binding). There is no second matcher, and this module is deliberately **not**
exported from the public ``zecalibrator.api.v1`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from zecalibrator.core.descriptors import LightConstraints
from zecalibrator.core.matching import (
    OUTCOME_AMBIGUOUS as _MATCH_AMBIGUOUS,
    OUTCOME_MATCHED,
    OUTCOME_NO_MATCH,
    Reason,
    match_calibration,
)
from zecalibrator.core.plans import (
    CalibrationPlan,
    CalibrationRequest,
    Candidate,
    MatchPolicy,
)
from zecalibrator.core.routes import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_READY,
    Route,
    RouteEnumeration,
    enumerate_routes,
)
from zecalibrator.application.library import LibrarySnapshot


@dataclass(frozen=True)
class RouteResolution:
    """The auto-route decision for one light against a library snapshot."""

    outcome: str  # READY | NEEDS_ATTENTION | AMBIGUOUS
    request: Optional[CalibrationRequest]
    plan: Optional[CalibrationPlan]
    reasons: Tuple[Reason, ...]
    unverified: Tuple[Reason, ...]
    route: Optional[Route]  # the single resolved READY route (or None)
    routes: Tuple[Route, ...]  # every enumerated route (for AMBIGUOUS)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "unverified", tuple(self.unverified))
        object.__setattr__(self, "routes", tuple(self.routes))


def resolve_route(
    light: LightConstraints,
    snapshot: LibrarySnapshot,
    policy: MatchPolicy,
) -> RouteResolution:
    """Resolve the auto-route for ``light`` against ``snapshot``.

    For exactly one READY route, build the existing ``CalibrationRequest`` and
    bind the final ``CalibrationPlan`` via the existing ``match_calibration``.
    A partial route (``control``/``bias_only``) is never a READY outcome and is
    never materialized as a plan here.
    """
    enumeration = enumerate_routes(light, snapshot.candidates, policy)

    request: Optional[CalibrationRequest] = None
    plan: Optional[CalibrationPlan] = None
    route: Optional[Route] = None
    outcome = enumeration.outcome
    reasons = enumeration.reasons
    unverified = enumeration.unverified

    if enumeration.outcome == OUTCOME_READY and len(enumeration.routes) == 1:
        route = enumeration.routes[0]
        request = CalibrationRequest(
            additive_mode=route.additive_mode, flat_mode=route.flat_mode
        )
        result = match_calibration(light, request, snapshot.candidates, policy)
        unverified = _merge_unverified(enumeration.unverified, result.unverified)
        if result.outcome == OUTCOME_MATCHED and result.plan is not None:
            plan = result.plan
        else:
            # The enumerator and the matcher must agree; a disagreement must not
            # be silently masked as a resolved route.
            reasons = enumeration.reasons + tuple(result.reasons)
            outcome = (
                OUTCOME_NEEDS_ATTENTION
                if result.outcome == OUTCOME_NO_MATCH
                else OUTCOME_AMBIGUOUS
            )

    return RouteResolution(outcome, request, plan, reasons, unverified, route, enumeration.routes)


def _merge_unverified(*groups) -> Tuple[Reason, ...]:
    seen = set()
    out: list[Reason] = []
    for group in groups:
        for r in group:
            try:
                key = (r.code, r.field, hash(r.expected), hash(r.observed))
            except TypeError:
                key = (r.code, r.field, repr(r.expected), repr(r.observed))
            if key not in seen:
                seen.add(key)
                out.append(r)
    return tuple(out)


__all__ = ["RouteResolution", "resolve_route"]
