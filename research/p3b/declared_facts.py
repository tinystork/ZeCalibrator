"""Declared ground-truth facts (LOT4) — the declaration side of the harness.

Internal, non-public, never imported by ``zecalibrator.api.v1``.

This module builds **declared facts** from a scenario's ground truth (the LOT1
declaration, via :class:`research.p3b.model.ScenarioSpec`), and exposes a
**trivial bridge** from those declared facts to LOT2's
:class:`research.p3b.qualification_policy.EvidencePacket`.

The bridge exists *only* because ground truth is known in synthetic data. It is
a pure projection of declared facts — no detection, no threshold, no feature
measurement — and it is tested as such (``tests/p3b/test_features.py``): the
facts change when the *declaration* changes and do **not** change when the
*features* change.

**No feature → fact mapping exists here.** The computed features
(:mod:`research.p3b.features`) are never consulted to build a fact. In
particular the descriptive censored count/fraction is a *feature* and is
deliberately **not** fed back into the packet's non-decisive
``censored_measurement_count``.

The class→fact translation table below is a *declaration translation* (from the
catalogue's declared light/dark behaviours to LOT2's declared-fact vocabulary),
not detection logic and not a scientific claim: every entry is a literal
restatement of what ``catalog.py`` / ``model.py`` already declare. Whether
LOT2's decision then agrees with the LOT1 truth labels
(``expected_qualification_state`` / ``expected_action_state``) is a later-lot
cross-check concern (LOT5 metrics), not this lot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .catalog import resolve_class
from .generator import cfa_plane_label
from .model import ScenarioSpec, compute_bookkeeping
from .qualification_policy import (
    NO,
    NOT_REPRESENTATIVE,
    REPRESENTATIVE,
    RESIDUAL_INDETERMINATE,
    RESIDUAL_NONE,
    RESIDUAL_SYSTEMATIC_STABLE,
    RESIDUAL_VARIABLE,
    UNDETERMINED,
    YES,
    EvidencePacket,
)

# ---------------------------------------------------------------------------
# Class → declared-fact translation (auditable; a literal restatement of the
# catalogue's declared behaviours, never a measurement).
#
# Columns: (residual_behaviour, persisted_at_same_sensor_coord,
#           neighbourhood_residual_stable, transient_only, conflicting_evidence)
# ---------------------------------------------------------------------------

_CLASS_FACTS = {
    # residual absent -> NONE; no signal -> persistence UNDETERMINED.
    "NORMAL": (RESIDUAL_NONE, UNDETERMINED, YES, NO, NO),
    # anomaly present in light AND dark -> a representative dark corrects it -> NONE.
    "STABLE_ANOMALY_CORRECTED_BY_DARK": (RESIDUAL_NONE, YES, YES, NO, NO),
    # anomaly present in light only -> stable residual after a mismatched dark.
    "STABLE_ANOMALY_WITH_MISMATCHED_DARK": (RESIDUAL_SYSTEMATIC_STABLE, YES, YES, NO, NO),
    "INTERMITTENT_TWO_STATE": (RESIDUAL_VARIABLE, YES, YES, NO, NO),
    "INTERMITTENT_MULTI_STATE": (RESIDUAL_VARIABLE, YES, YES, NO, NO),
    "INTERMITTENT_CONTINUOUS": (RESIDUAL_VARIABLE, YES, YES, NO, NO),
    "RARE_HIGH_STATE": (RESIDUAL_VARIABLE, YES, YES, NO, NO),
    "RARE_LOW_STATE": (RESIDUAL_VARIABLE, YES, YES, NO, NO),
    # residual flips sign around the dark -> VARIABLE + contradictory evidence.
    "SIGN_CHANGING_POST_DARK": (RESIDUAL_VARIABLE, YES, YES, NO, YES),
    # censored residual cannot be characterised (§13.4).
    "CENSORED_ANOMALY": (RESIDUAL_INDETERMINATE, YES, YES, NO, NO),
    # one-frame transient -> unresolved residual, not a persistent sensor site.
    "SINGLE_TRANSIENT": (RESIDUAL_INDETERMINATE, NO, YES, YES, NO),
    # static optical structure -> stable residual, but a *sky* structure (not a
    # sensor site), and the disagreement is global (neighbourhood not stable).
    "OPTICAL_STRUCTURE": (RESIDUAL_SYSTEMATIC_STABLE, NO, NO, NO, NO),
    "STAR_CROSSING_SITE": (RESIDUAL_INDETERMINATE, NO, NO, YES, NO),
    "UNDERSAMPLED_STAR_CORE": (RESIDUAL_SYSTEMATIC_STABLE, NO, NO, NO, NO),
    "COSMIC_RAY": (RESIDUAL_INDETERMINATE, NO, YES, YES, NO),
    # extreme per-frame noise -> variable residual.
    "NOISE_EXTREME": (RESIDUAL_VARIABLE, YES, YES, NO, NO),
    "FLAT_STRUCTURE": (RESIDUAL_SYSTEMATIC_STABLE, NO, NO, NO, NO),
    "DUST_OR_VIGNETTING": (RESIDUAL_SYSTEMATIC_STABLE, NO, NO, NO, NO),
    # persistently elevated near the limit -> stable residual.
    "NEAR_SATURATION": (RESIDUAL_SYSTEMATIC_STABLE, YES, YES, NO, NO),
    # weak anomaly, insufficient evidence -> residual cannot be characterised.
    "AMBIGUOUS_INSUFFICIENT_EVIDENCE": (RESIDUAL_INDETERMINATE, YES, YES, NO, NO),
}


@dataclass(frozen=True)
class SiteDeclaredFacts:
    """Declared ground-truth facts for one site (never computed from features)."""

    site_id: str
    x: int
    y: int
    cfa_class: str
    cfa_plane: str
    # Explicitly declared on SiteSpec.
    calibration_representativeness: str
    censored_measurement_present: str
    # Declared-fact translation from the class (see _CLASS_FACTS).
    site_residual_behaviour: str
    persisted_at_same_sensor_coord: str
    neighbourhood_residual_stable: str
    transient_only: str
    conflicting_evidence: str
    # Structural declared facts (synthetic: identity/geometry fully declared).
    sensor_identity_resolved: str = YES
    geometry_compatible: str = YES
    # Not declared by LOT1 — defaulted, and *never* computed by LOT4.
    net_benefit_established: str = UNDETERMINED
    censored_measurement_count: int = 0
    persisted_basis: str = ""


@dataclass(frozen=True)
class DeclaredFacts:
    """The full declared-fact set for a scenario (never computed from features)."""

    scenario_name: str
    seed: int
    sensor_identity_resolved: str
    geometry_compatible: str
    calibration_present: str
    frame_count: int
    observation_count: int
    independent_group_count: int
    epoch_count: int
    sites: Tuple[SiteDeclaredFacts, ...]


def _representativeness(value: str) -> str:
    """Map a declared representativeness string to the LOT2 vocabulary."""
    return REPRESENTATIVE if value == "representative" else NOT_REPRESENTATIVE


def _calibration_present(scenario: ScenarioSpec) -> str:
    """Declared: is an additive (dark/bias) calibration present in the scenario?

    A structural fact read from the declared frame types and aggregates — never
    a feature, never a match decision.
    """
    additive_ids = {
        f.frame_id for f in scenario.all_frames() if f.frame_type in ("dark", "bias")
    }
    for agg in scenario.aggregates:
        if any(c in additive_ids for c in agg.constituent_frame_ids):
            return YES
    return YES if additive_ids else NO


def build_site_declared_facts(site, sensor) -> SiteDeclaredFacts:
    """Build the declared facts for one site (pure projection of the declaration)."""
    entry = resolve_class(site.cfa_class)  # validates the declared class name
    residual, persisted, neighbourhood, transient, conflicting = _CLASS_FACTS[site.cfa_class]
    return SiteDeclaredFacts(
        site_id=site.site_id,
        x=int(site.x),
        y=int(site.y),
        cfa_class=site.cfa_class,
        cfa_plane=cfa_plane_label(site.y, site.x, sensor.cfa_pattern),
        calibration_representativeness=_representativeness(site.calibration_representativeness),
        censored_measurement_present=YES if site.censored else NO,
        site_residual_behaviour=residual,
        persisted_at_same_sensor_coord=persisted,
        neighbourhood_residual_stable=neighbourhood,
        transient_only=transient,
        conflicting_evidence=conflicting,
        persisted_basis=f"cfa_class={site.cfa_class}",
    )


def build_declared_facts(scenario: ScenarioSpec) -> DeclaredFacts:
    """Build the declared-fact set from a scenario's ground truth (no frames read)."""
    bk = compute_bookkeeping(scenario)
    sites = tuple(build_site_declared_facts(site, scenario.sensor) for site in scenario.sites)
    return DeclaredFacts(
        scenario_name=scenario.name,
        seed=scenario.seed,
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=_calibration_present(scenario),
        frame_count=bk.frame_count,
        observation_count=bk.observation_count,
        independent_group_count=bk.independent_group_count,
        epoch_count=bk.epoch_count,
        sites=sites,
    )


def evidence_packet(site: SiteDeclaredFacts, declared: DeclaredFacts) -> EvidencePacket:
    """Trivial bridge: declared facts → LOT2 ``EvidencePacket`` (no detection).

    A pure field projection. It reads only declared facts, never a feature, and
    never calls ``qualification_policy.evaluate`` (the decision is LOT2's, not
    LOT4's). ``net_benefit_established`` is passed through as the declared
    default (``UNDETERMINED``): LOT4 computes no net benefit (that is LOT6).
    """
    return EvidencePacket(
        sensor_identity_resolved=declared.sensor_identity_resolved,
        geometry_compatible=declared.geometry_compatible,
        calibration_present=declared.calibration_present,
        calibration_representativeness=site.calibration_representativeness,
        persisted_at_same_sensor_coord=site.persisted_at_same_sensor_coord,
        independent_group_count=declared.independent_group_count,
        epoch_count=declared.epoch_count,
        censored_measurement_present=site.censored_measurement_present,
        site_residual_behaviour=site.site_residual_behaviour,
        neighbourhood_residual_stable=site.neighbourhood_residual_stable,
        transient_only=site.transient_only,
        conflicting_evidence=site.conflicting_evidence,
        net_benefit_established=site.net_benefit_established,
        persisted_basis=site.persisted_basis,
        censored_measurement_count=site.censored_measurement_count,
    )


__all__ = [
    "DeclaredFacts",
    "SiteDeclaredFacts",
    "build_declared_facts",
    "build_site_declared_facts",
    "evidence_packet",
]
