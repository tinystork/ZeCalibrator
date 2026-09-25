"""Qualification policy engine (LOT2) — internal, non-public, a pure function of facts.

This module turns a packet of *declared evidence facts* (:class:`EvidencePacket`)
into a structured, non-boolean decision (:class:`QualificationDecision`) across
the three orthogonal axes ratified in ``docs/SCIENCE_CONTRACT.md`` §13,
``docs/ARCHITECTURE.md`` §18 and the ratified three-axis state machine
``qualification_state_machine.md`` (P3A):

* **axis 1 — epistemic**      — what do we know about the site?
* **axis 2 — representativeness** — does the additive calibration represent this
  observation?  (exposed *verbatim* from the input, never re-derived)
* **axis 3 — action**         — what does the science permit?  (ordered ladder)

Boundary (enforced here, verified by ``tests/p3b/test_qualification_policy.py``):

* a **pure function of facts** — imports neither ``numpy`` nor ``astropy``,
  reads no FITS, opens no file, touches no filesystem, reads no pixel;
* computes **no amplitude**, performs **no detection**, applies **no threshold**;
* **never computes** a net benefit — it *consumes* ``net_benefit_established``
  as a declared fact (the numeric computation is LOT6, not this lot);
* **internal and non-public** — never imported by ``zecalibrator.api.v1``.

No ``detected -> corrected`` path exists: there is no detection field in this
vocabulary, and eligibility requires a *complete* evidence dossier (identity,
geometry, calibration, representativeness, **persistence at the sensor
coordinate**, independence, a local disagreement, and a demonstrated net
benefit). The four stages of SCIENCE §13.2 are never collapsed into a single
``detected -> corrected`` hop.

Persistence is the fact that separates a *sensor defect* from a *sky structure*
(a star crossing the coordinate, an optical structure): a genuine sensor site
persists at the same sensor coordinate. ``persisted_at_same_sensor_coord == NO``
means the signal is positively **not** a sensor site — never a reconstruction
candidate; ``UNDETERMINED`` means the sensor-site status cannot be established
— never ``CHARACTERISED_*`` and never eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Tuple

# ---------------------------------------------------------------------------
# Fact vocabularies (declared facts, not thresholds, not amplitudes)
# ---------------------------------------------------------------------------

# Binary facts (yes / no) — a fact with no "undetermined" value.
YES = "YES"
NO = "NO"

# Ternary facts (yes / no / undetermined).
UNDETERMINED = "UNDETERMINED"

# Axis 2 — calibration representativeness (SCIENCE §13.6).
REPRESENTATIVE = "REPRESENTATIVE"
NOT_REPRESENTATIVE = "NOT_REPRESENTATIVE"
REPRESENTATIVENESS_STATES: Tuple[str, ...] = (
    REPRESENTATIVE,
    NOT_REPRESENTATIVE,
    UNDETERMINED,
)

# Site residual behaviour (post-calibration, declared — never computed here).
RESIDUAL_NONE = "NONE"
RESIDUAL_SYSTEMATIC_STABLE = "SYSTEMATIC_STABLE"
RESIDUAL_VARIABLE = "VARIABLE"
RESIDUAL_INDETERMINATE = "INDETERMINATE"
RESIDUAL_STATES: Tuple[str, ...] = (
    RESIDUAL_NONE,
    RESIDUAL_SYSTEMATIC_STABLE,
    RESIDUAL_VARIABLE,
    RESIDUAL_INDETERMINATE,
)

# Net benefit (declared fact; the computation is LOT6's, not this lot's).
NET_BENEFIT_ESTABLISHED = "ESTABLISHED"
NET_BENEFIT_NOT_ESTABLISHED = "NOT_ESTABLISHED"
NET_BENEFIT_STATES: Tuple[str, ...] = (
    NET_BENEFIT_ESTABLISHED,
    NET_BENEFIT_NOT_ESTABLISHED,
    UNDETERMINED,
)

# ---------------------------------------------------------------------------
# Axis 1 — epistemic states (knowledge, never an action verdict)
# ---------------------------------------------------------------------------

EPISTEMIC_UNKNOWN = "UNKNOWN"
EPISTEMIC_OBSERVED_CANDIDATE = "OBSERVED_CANDIDATE"
EPISTEMIC_CHARACTERISED_STABLE = "CHARACTERISED_STABLE"
EPISTEMIC_CHARACTERISED_INTERMITTENT = "CHARACTERISED_INTERMITTENT"
EPISTEMIC_TRANSIENT_OR_UNRESOLVED = "TRANSIENT_OR_UNRESOLVED"
EPISTEMIC_CENSORED = "CENSORED"

EPISTEMIC_STATES: Tuple[str, ...] = (
    EPISTEMIC_UNKNOWN,
    EPISTEMIC_OBSERVED_CANDIDATE,
    EPISTEMIC_CHARACTERISED_STABLE,
    EPISTEMIC_CHARACTERISED_INTERMITTENT,
    EPISTEMIC_TRANSIENT_OR_UNRESOLVED,
    EPISTEMIC_CENSORED,
)

# ---------------------------------------------------------------------------
# Axis 3 — ordered action ladder (first satisfied condition decides; §13.7)
# ---------------------------------------------------------------------------

ACTION_REQUALIFY = "REQUALIFY_ADDITIVE_CALIBRATION"
ACTION_ABSTAIN_CENSORED = "ABSTAIN_CENSORED"
ACTION_ABSTAIN_INCONSISTENT = "ABSTAIN_INCONSISTENT"
ACTION_ABSTAIN_INSUFFICIENT = "ABSTAIN_INSUFFICIENT_EVIDENCE"
ACTION_ELIGIBLE = "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION"
ACTION_NO_ACTION = "NO_ACTION_REQUIRED"

ACTION_LADDER: Tuple[str, ...] = (
    ACTION_REQUALIFY,
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
    ACTION_NO_ACTION,
)

# ---------------------------------------------------------------------------
# Reason codes (explicit, versioned vocabulary — never free text as contract)
# ---------------------------------------------------------------------------

REASON_CODE_VERSION = "1"

RC_SENSOR_IDENTITY_UNRESOLVED = "SENSOR_IDENTITY_UNRESOLVED"
RC_GEOMETRY_INCOMPATIBLE = "GEOMETRY_INCOMPATIBLE"
RC_CALIBRATION_UNVERIFIED = "CALIBRATION_UNVERIFIED"
RC_CALIBRATION_NOT_REPRESENTATIVE = "CALIBRATION_NOT_REPRESENTATIVE"
RC_EVIDENCE_CENSORED = "EVIDENCE_CENSORED"
RC_INSUFFICIENT_EPOCHS = "INSUFFICIENT_EPOCHS"
RC_INSUFFICIENT_INDEPENDENT_GROUPS = "INSUFFICIENT_INDEPENDENT_GROUPS"
RC_TRANSIENT_ONLY = "TRANSIENT_ONLY"
RC_INCONSISTENT_EVIDENCE = "INCONSISTENT_EVIDENCE"
RC_NET_BENEFIT_NOT_ESTABLISHED = "NET_BENEFIT_NOT_ESTABLISHED"
RC_NO_ACTION_NEEDED_AFTER_CALIBRATION = "NO_ACTION_NEEDED_AFTER_CALIBRATION"
RC_ELIGIBLE = "ELIGIBLE"
# Justified additions beyond the mission minimum (see module docstring / report):
RC_SITE_RESIDUAL_INDETERMINATE = "SITE_RESIDUAL_INDETERMINATE"
RC_NEIGHBOURHOOD_RESIDUAL_UNSTABLE = "NEIGHBOURHOOD_RESIDUAL_UNSTABLE"
RC_NEIGHBOURHOOD_RESIDUAL_UNDETERMINED = "NEIGHBOURHOOD_RESIDUAL_UNDETERMINED"
# Rework-1 (M1): persistence at the sensor coordinate discriminates a *sensor
# defect* (persisted) from a *sky structure* (star crossing / optical structure,
# not persisted at the same sensor coordinate). Distinct codes, never merged
# (same discipline as INSUFFICIENT_EPOCHS / INSUFFICIENT_INDEPENDENT_GROUPS).
RC_NOT_PERSISTENT_IN_SENSOR_COORDINATES = "NOT_PERSISTENT_IN_SENSOR_COORDINATES"
RC_PERSISTENCE_UNDETERMINED = "PERSISTENCE_UNDETERMINED"

REASON_CODES: Tuple[str, ...] = (
    RC_SENSOR_IDENTITY_UNRESOLVED,
    RC_GEOMETRY_INCOMPATIBLE,
    RC_CALIBRATION_UNVERIFIED,
    RC_CALIBRATION_NOT_REPRESENTATIVE,
    RC_EVIDENCE_CENSORED,
    RC_INSUFFICIENT_EPOCHS,
    RC_INSUFFICIENT_INDEPENDENT_GROUPS,
    RC_TRANSIENT_ONLY,
    RC_INCONSISTENT_EVIDENCE,
    RC_NET_BENEFIT_NOT_ESTABLISHED,
    RC_NO_ACTION_NEEDED_AFTER_CALIBRATION,
    RC_ELIGIBLE,
    RC_SITE_RESIDUAL_INDETERMINATE,
    RC_NEIGHBOURHOOD_RESIDUAL_UNSTABLE,
    RC_NEIGHBOURHOOD_RESIDUAL_UNDETERMINED,
    RC_NOT_PERSISTENT_IN_SENSOR_COORDINATES,
    RC_PERSISTENCE_UNDETERMINED,
)

# Short, stable meaning for each code (versioned vocabulary documentation).
REASON_CODE_MEANINGS: Mapping[str, str] = {
    RC_SENSOR_IDENTITY_UNRESOLVED: "sensor identity could not be resolved",
    RC_GEOMETRY_INCOMPATIBLE: "geometry not established compatible",
    RC_CALIBRATION_UNVERIFIED: "additive calibration absent or representativeness undetermined",
    RC_CALIBRATION_NOT_REPRESENTATIVE: "additive calibration demonstrated not representative",
    RC_EVIDENCE_CENSORED: "relevant evidence truncated by the acquisition hard limit",
    RC_INSUFFICIENT_EPOCHS: "fewer than two independent epochs",
    RC_INSUFFICIENT_INDEPENDENT_GROUPS: "fewer than two independent groups",
    RC_TRANSIENT_ONLY: "transient-only signal; no persistent qualification",
    RC_INCONSISTENT_EVIDENCE: "contradictory evidence between groups/epochs",
    RC_NET_BENEFIT_NOT_ESTABLISHED: "net benefit not established",
    RC_NO_ACTION_NEEDED_AFTER_CALIBRATION: "known site already corrected by representative calibration",
    RC_ELIGIBLE: "eligible for targeted reconstruction",
    RC_SITE_RESIDUAL_INDETERMINATE: "site residual behaviour cannot be characterised",
    RC_NEIGHBOURHOOD_RESIDUAL_UNSTABLE: "disagreement is global (neighbourhood), not local to the site",
    RC_NEIGHBOURHOOD_RESIDUAL_UNDETERMINED: "local vs global disagreement cannot be decided",
    RC_NOT_PERSISTENT_IN_SENSOR_COORDINATES: "signal does not persist at the same sensor coordinate (sky structure, not a sensor defect)",
    RC_PERSISTENCE_UNDETERMINED: "sensor-site persistence cannot be decided",
}

# ---------------------------------------------------------------------------
# The 8 mission-brief names (preserved by documented projection — see below)
# ---------------------------------------------------------------------------

MISSION_NAMES: Tuple[str, ...] = (
    "UNKNOWN",
    "KNOWN_STABLE",
    "KNOWN_INTERMITTENT_CANDIDATE",
    "CENSORED",
    "TRANSIENT_OR_UNRESOLVED",
    "CALIBRATION_NOT_REPRESENTATIVE",
    "NO_ACTION_REQUIRED",
    "QUALIFIED_FOR_RECONSTRUCTION",
)


class CensoredInferenceError(ValueError):
    """Typed error: a quantitative inference is claimed from censored evidence.

    SCIENCE §13.4: a censored measurement cannot qualify amplitude, state,
    benefit or reconstruction. A packet that declares censored evidence *and*
    simultaneously asserts a characterised residual or an established net
    benefit violates that invariant. The engine refuses the packet instead of
    silently treating the censored value as trustworthy.
    """

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


@dataclass(frozen=True)
class EvidencePacket:
    """A packet of declared evidence facts (never measurements, never amplitudes).

    Every field is a *fact*, not a number: ``site_residual_behaviour``,
    ``calibration_representativeness`` and ``net_benefit_established`` are
    measured upstream and consumed here verbatim. There is no amplitude (ADU),
    no score, no threshold, no detection boolean anywhere in this model.
    """

    sensor_identity_resolved: str  # YES | NO
    geometry_compatible: str  # YES | NO | UNDETERMINED
    calibration_present: str  # YES | NO
    calibration_representativeness: str  # REPRESENTATIVE | NOT_REPRESENTATIVE | UNDETERMINED
    persisted_at_same_sensor_coord: str  # YES | NO | UNDETERMINED
    independent_group_count: int
    epoch_count: int
    censored_measurement_present: str  # YES | NO
    site_residual_behaviour: str  # NONE | SYSTEMATIC_STABLE | VARIABLE | INDETERMINATE
    neighbourhood_residual_stable: str  # YES | NO | UNDETERMINED
    transient_only: str  # YES | NO
    conflicting_evidence: str  # YES | NO
    net_benefit_established: str  # ESTABLISHED | NOT_ESTABLISHED | UNDETERMINED
    persisted_basis: str = ""  # short factual basis (contextual, never decisive)
    censored_measurement_count: int = 0  # contextual count, never decisive

    _BINARY = (YES, NO)
    _TERNARY = (YES, NO, UNDETERMINED)

    def __post_init__(self) -> None:
        _require(self.sensor_identity_resolved, self._BINARY, "sensor_identity_resolved")
        _require(self.geometry_compatible, self._TERNARY, "geometry_compatible")
        _require(self.calibration_present, self._BINARY, "calibration_present")
        _require(
            self.calibration_representativeness,
            REPRESENTATIVENESS_STATES,
            "calibration_representativeness",
        )
        _require(self.persisted_at_same_sensor_coord, self._TERNARY, "persisted_at_same_sensor_coord")
        _require(self.censored_measurement_present, self._BINARY, "censored_measurement_present")
        _require(self.site_residual_behaviour, RESIDUAL_STATES, "site_residual_behaviour")
        _require(self.neighbourhood_residual_stable, self._TERNARY, "neighbourhood_residual_stable")
        _require(self.transient_only, self._BINARY, "transient_only")
        _require(self.conflicting_evidence, self._BINARY, "conflicting_evidence")
        _require(self.net_benefit_established, NET_BENEFIT_STATES, "net_benefit_established")

        for name in ("independent_group_count", "epoch_count", "censored_measurement_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
            object.__setattr__(self, name, int(value))


def _require(value: str, allowed: Tuple[str, ...], field: str) -> None:
    if value not in allowed:
        raise ValueError(
            f"{field} must be one of {allowed!r}, got {value!r}"
        )


@dataclass(frozen=True)
class EvidenceFact:
    """One fact actually consulted in the decision, with its role."""

    field: str
    value: object
    role: str


@dataclass(frozen=True)
class QualificationDecision:
    """The structured, non-boolean decision (three axes + provenance of reasons)."""

    epistemic_state: str  # axis 1
    representativeness: str  # axis 2 (= the input fact, exposed verbatim)
    action: str  # axis 3 (ordered ladder)
    reason_codes: Tuple[str, ...]  # never empty
    evidence_summary: Tuple[EvidenceFact, ...]  # facts actually used


# ---------------------------------------------------------------------------
# Pure decision function
# ---------------------------------------------------------------------------


def evaluate(packet: EvidencePacket) -> QualificationDecision:
    """Turn an :class:`EvidencePacket` into a structured decision.

    Pure: no I/O, no numpy/astropy, no file access, no pixel, no amplitude.
    """
    _reject_censored_quantitative_claim(packet)

    epistemic, epistemic_facts = _derive_epistemic(packet)
    action, reason_codes, action_facts = _derive_action(packet)

    summary = (EvidenceFact("calibration_representativeness", packet.calibration_representativeness, "axis2"),)
    summary = summary + epistemic_facts + action_facts

    return QualificationDecision(
        epistemic_state=epistemic,
        representativeness=packet.calibration_representativeness,
        action=action,
        reason_codes=reason_codes,
        evidence_summary=summary,
    )


def _reject_censored_quantitative_claim(packet: EvidencePacket) -> None:
    """Refuse any quantitative inference from censored evidence (SCIENCE §13.4)."""
    if packet.censored_measurement_present != YES:
        return
    if packet.site_residual_behaviour in (RESIDUAL_SYSTEMATIC_STABLE, RESIDUAL_VARIABLE):
        raise CensoredInferenceError(
            "site_residual_behaviour",
            "censored evidence cannot support a characterised residual "
            f"({packet.site_residual_behaviour!r}); a censored site's residual "
            "must be NONE or INDETERMINATE",
        )
    if packet.net_benefit_established == NET_BENEFIT_ESTABLISHED:
        raise CensoredInferenceError(
            "net_benefit_established",
            "censored evidence cannot establish a net benefit",
        )


def _derive_epistemic(packet: EvidencePacket) -> Tuple[str, Tuple[EvidenceFact, ...]]:
    """Derive axis 1 (epistemic) from facts, in documented order."""
    if packet.censored_measurement_present == YES:
        return EPISTEMIC_CENSORED, (
            EvidenceFact("censored_measurement_present", YES, "axis1:censored"),
        )
    if packet.transient_only == YES:
        return EPISTEMIC_TRANSIENT_OR_UNRESOLVED, (
            EvidenceFact("transient_only", YES, "axis1:transient"),
        )
    if packet.conflicting_evidence == YES:
        return EPISTEMIC_TRANSIENT_OR_UNRESOLVED, (
            EvidenceFact("conflicting_evidence", YES, "axis1:unresolved"),
        )

    residual = packet.site_residual_behaviour
    if residual == RESIDUAL_NONE:
        facts = [EvidenceFact("site_residual_behaviour", RESIDUAL_NONE, "axis1:no_residual")]
        if packet.persisted_at_same_sensor_coord == YES:
            facts.append(
                EvidenceFact("persisted_at_same_sensor_coord", YES, "axis1:known_site")
            )
            return EPISTEMIC_CHARACTERISED_STABLE, tuple(facts)
        return EPISTEMIC_UNKNOWN, tuple(facts)

    if residual == RESIDUAL_INDETERMINATE:
        return EPISTEMIC_OBSERVED_CANDIDATE, (
            EvidenceFact("site_residual_behaviour", RESIDUAL_INDETERMINATE, "axis1:unresolved_residual"),
        )

    # residual present (SYSTEMATIC_STABLE or VARIABLE)
    facts = [EvidenceFact("site_residual_behaviour", residual, "axis1:residual_present")]

    # Persistence at the sensor coordinate is the fact that separates a *sensor
    # defect* from a *sky structure* (star crossing / optical structure).
    #   * NO          -> not a sensor-site candidate at all (transient/unresolved);
    #   * UNDETERMINED -> cannot be CHARACTERISED_* (candidate at most);
    #   * YES         -> a genuine sensor-site candidate (CHARACTERISED_* only here).
    if packet.persisted_at_same_sensor_coord == NO:
        facts.append(EvidenceFact("persisted_at_same_sensor_coord", NO, "axis1:not_sensor_site"))
        return EPISTEMIC_TRANSIENT_OR_UNRESOLVED, tuple(facts)
    if packet.persisted_at_same_sensor_coord == UNDETERMINED:
        facts.append(EvidenceFact("persisted_at_same_sensor_coord", UNDETERMINED, "axis1:persistence_undetermined"))
        return EPISTEMIC_OBSERVED_CANDIDATE, tuple(facts)

    # persisted == YES: a genuine sensor-site candidate.
    independent = packet.epoch_count >= 2 and packet.independent_group_count >= 2
    if independent:
        facts.append(EvidenceFact("persisted_at_same_sensor_coord", YES, "axis1:persistent_site"))
        facts.append(EvidenceFact("epoch_count", packet.epoch_count, "axis1:independence"))
        facts.append(
            EvidenceFact("independent_group_count", packet.independent_group_count, "axis1:independence")
        )
        if residual == RESIDUAL_VARIABLE:
            return EPISTEMIC_CHARACTERISED_INTERMITTENT, tuple(facts)
        return EPISTEMIC_CHARACTERISED_STABLE, tuple(facts)
    # persisted but not independently characterised -> candidate only
    return EPISTEMIC_OBSERVED_CANDIDATE, tuple(facts)


def _derive_action(
    packet: EvidencePacket,
) -> Tuple[str, Tuple[str, ...], Tuple[EvidenceFact, ...]]:
    """Derive axis 3 (action) with the ordered ladder — first match decides."""
    # P1 — representativeness dominates everything (SCIENCE §13.6, property 2)
    if packet.calibration_representativeness == NOT_REPRESENTATIVE:
        return (
            ACTION_REQUALIFY,
            (RC_CALIBRATION_NOT_REPRESENTATIVE,),
            (EvidenceFact("calibration_representativeness", NOT_REPRESENTATIVE, "action:p1"),),
        )

    # P2 — censoring (property 3)
    if packet.censored_measurement_present == YES:
        return (
            ACTION_ABSTAIN_CENSORED,
            (RC_EVIDENCE_CENSORED,),
            (EvidenceFact("censored_measurement_present", YES, "action:p2"),),
        )

    # P3 — contradictory evidence
    if packet.conflicting_evidence == YES:
        return (
            ACTION_ABSTAIN_INCONSISTENT,
            (RC_INCONSISTENT_EVIDENCE,),
            (EvidenceFact("conflicting_evidence", YES, "action:p3"),),
        )

    # P4 — transient only (property 7)
    if packet.transient_only == YES:
        return (
            ACTION_ABSTAIN_INCONSISTENT,
            (RC_TRANSIENT_ONLY,),
            (EvidenceFact("transient_only", YES, "action:p4"),),
        )

    # P5 — sensor identity
    if packet.sensor_identity_resolved != YES:
        return (
            ACTION_ABSTAIN_INSUFFICIENT,
            (RC_SENSOR_IDENTITY_UNRESOLVED,),
            (EvidenceFact("sensor_identity_resolved", packet.sensor_identity_resolved, "action:p5"),),
        )

    # P6 — geometry
    if packet.geometry_compatible != YES:
        return (
            ACTION_ABSTAIN_INSUFFICIENT,
            (RC_GEOMETRY_INCOMPATIBLE,),
            (EvidenceFact("geometry_compatible", packet.geometry_compatible, "action:p6"),),
        )

    # P7 — calibration present
    if packet.calibration_present != YES:
        return (
            ACTION_ABSTAIN_INSUFFICIENT,
            (RC_CALIBRATION_UNVERIFIED,),
            (EvidenceFact("calibration_present", packet.calibration_present, "action:p7"),),
        )

    # P8 — representativeness undetermined
    if packet.calibration_representativeness == UNDETERMINED:
        return (
            ACTION_ABSTAIN_INSUFFICIENT,
            (RC_CALIBRATION_UNVERIFIED,),
            (EvidenceFact("calibration_representativeness", UNDETERMINED, "action:p8"),),
        )

    residual = packet.site_residual_behaviour

    # P9 — no residual to act on: a known site already corrected by a
    # representative calibration is NO_ACTION (knowledge is not action,
    # property 5); a site with no observed anomaly is also NO_ACTION.
    if residual == RESIDUAL_NONE:
        return (
            ACTION_NO_ACTION,
            (RC_NO_ACTION_NEEDED_AFTER_CALIBRATION,),
            (EvidenceFact("site_residual_behaviour", RESIDUAL_NONE, "action:p9"),),
        )

    # P10 — residual cannot be characterised
    if residual == RESIDUAL_INDETERMINATE:
        return (
            ACTION_ABSTAIN_INSUFFICIENT,
            (RC_SITE_RESIDUAL_INDETERMINATE,),
            (EvidenceFact("site_residual_behaviour", RESIDUAL_INDETERMINATE, "action:p10"),),
        )

    # residual present (SYSTEMATIC_STABLE / VARIABLE): a reconstruction candidate
    facts = [EvidenceFact("site_residual_behaviour", residual, "action:residual_present")]

    # P11 — persistence at the sensor coordinate (qualification dossier item,
    # before independence/benefit: "is this a sensor site at all?" precedes
    # "do we have enough evidence?"). A sensor defect persists at the same
    # sensor coordinate; a signal that does NOT is a sky structure (star
    # crossing / optical structure) and is never a reconstruction candidate.
    # Distinct codes, never merged (same discipline as independence).
    if packet.persisted_at_same_sensor_coord == NO:
        facts.append(EvidenceFact("persisted_at_same_sensor_coord", NO, "action:p11"))
        return ACTION_ABSTAIN_INCONSISTENT, (RC_NOT_PERSISTENT_IN_SENSOR_COORDINATES,), tuple(facts)
    if packet.persisted_at_same_sensor_coord == UNDETERMINED:
        facts.append(EvidenceFact("persisted_at_same_sensor_coord", UNDETERMINED, "action:p11"))
        return ACTION_ABSTAIN_INSUFFICIENT, (RC_PERSISTENCE_UNDETERMINED,), tuple(facts)

    # persisted == YES continues: a genuine sensor-site candidate.
    facts.append(EvidenceFact("persisted_at_same_sensor_coord", YES, "action:p11"))

    # P12 / P13 — independence (distinct codes, never merged; property 4)
    insufficient: list = []
    if packet.epoch_count < 2:
        insufficient.append(RC_INSUFFICIENT_EPOCHS)
        facts.append(EvidenceFact("epoch_count", packet.epoch_count, "action:p12"))
    if packet.independent_group_count < 2:
        insufficient.append(RC_INSUFFICIENT_INDEPENDENT_GROUPS)
        facts.append(EvidenceFact("independent_group_count", packet.independent_group_count, "action:p13"))
    if insufficient:
        return ACTION_ABSTAIN_INSUFFICIENT, tuple(insufficient), tuple(facts)

    # P14 — net benefit (consumed, never computed; property 6)
    if packet.net_benefit_established != NET_BENEFIT_ESTABLISHED:
        facts.append(
            EvidenceFact("net_benefit_established", packet.net_benefit_established, "action:p14")
        )
        return ACTION_ABSTAIN_INSUFFICIENT, (RC_NET_BENEFIT_NOT_ESTABLISHED,), tuple(facts)

    # P15 / P16 — disagreement must be local to the site (not global)
    if packet.neighbourhood_residual_stable == NO:
        facts.append(
            EvidenceFact("neighbourhood_residual_stable", NO, "action:p15")
        )
        return ACTION_ABSTAIN_INCONSISTENT, (RC_NEIGHBOURHOOD_RESIDUAL_UNSTABLE,), tuple(facts)
    if packet.neighbourhood_residual_stable == UNDETERMINED:
        facts.append(
            EvidenceFact("neighbourhood_residual_stable", UNDETERMINED, "action:p16")
        )
        return ACTION_ABSTAIN_INSUFFICIENT, (RC_NEIGHBOURHOOD_RESIDUAL_UNDETERMINED,), tuple(facts)

    # P17 — eligible (not obligatory)
    facts.append(EvidenceFact("neighbourhood_residual_stable", YES, "action:p17"))
    facts.append(EvidenceFact("net_benefit_established", NET_BENEFIT_ESTABLISHED, "action:p17"))
    return ACTION_ELIGIBLE, (RC_ELIGIBLE,), tuple(facts)


# ---------------------------------------------------------------------------
# Projection to the 8 mission-brief names (documented, tested)
# ---------------------------------------------------------------------------
#
# The 8 names are overlapping concepts, not a partition of the decision space:
# a normal site is *both* UNKNOWN (axis 1) and NO_ACTION_REQUIRED (action), and
# a characterised site that qualifies is *both* KNOWN_STABLE (axis 1) and
# QUALIFIED_FOR_RECONSTRUCTION (action). A flat list would collapse independent
# axes; the three-axis model keeps them separate. The authoritative mapping of
# every name is the table :data:`MISSION_NAME_PROJECTIONS`; the convenience
# function :func:`project_mission_name` returns ONE name under a documented
# precedence (action first, then representativeness, then epistemic).


def _p_action_is(action: str) -> Callable[[QualificationDecision], bool]:
    return lambda d: d.action == action


def _p_epistemic_is(state: str) -> Callable[[QualificationDecision], bool]:
    return lambda d: d.epistemic_state == state


def _p_epistemic_in(states: Tuple[str, ...]) -> Callable[[QualificationDecision], bool]:
    return lambda d: d.epistemic_state in states


def _p_representativeness_not_representative(d: QualificationDecision) -> bool:
    return d.representativeness == NOT_REPRESENTATIVE


MISSION_NAME_PROJECTIONS: Mapping[str, Callable[[QualificationDecision], bool]] = {
    "UNKNOWN": _p_epistemic_is(EPISTEMIC_UNKNOWN),
    "KNOWN_STABLE": _p_epistemic_is(EPISTEMIC_CHARACTERISED_STABLE),
    "KNOWN_INTERMITTENT_CANDIDATE": _p_epistemic_in(
        (EPISTEMIC_OBSERVED_CANDIDATE, EPISTEMIC_CHARACTERISED_INTERMITTENT)
    ),
    "CENSORED": _p_epistemic_is(EPISTEMIC_CENSORED),
    "TRANSIENT_OR_UNRESOLVED": _p_epistemic_is(EPISTEMIC_TRANSIENT_OR_UNRESOLVED),
    "CALIBRATION_NOT_REPRESENTATIVE": _p_representativeness_not_representative,
    "NO_ACTION_REQUIRED": _p_action_is(ACTION_NO_ACTION),
    "QUALIFIED_FOR_RECONSTRUCTION": _p_action_is(ACTION_ELIGIBLE),
}


def project_mission_name(decision: QualificationDecision) -> str:
    """Return ONE of the 8 mission-brief names under a documented precedence.

    Precedence (first match wins): action-derived names first
    (``QUALIFIED_FOR_RECONSTRUCTION``, ``NO_ACTION_REQUIRED``), then
    representativeness (``CALIBRATION_NOT_REPRESENTATIVE``), then epistemic
    states. This is a lossy single-name view; the full 8-concept mapping is
    :data:`MISSION_NAME_PROJECTIONS`.
    """
    if decision.action == ACTION_ELIGIBLE:
        return "QUALIFIED_FOR_RECONSTRUCTION"
    if decision.action == ACTION_NO_ACTION:
        return "NO_ACTION_REQUIRED"
    if decision.representativeness == NOT_REPRESENTATIVE:
        return "CALIBRATION_NOT_REPRESENTATIVE"
    if decision.epistemic_state == EPISTEMIC_CENSORED:
        return "CENSORED"
    if decision.epistemic_state == EPISTEMIC_TRANSIENT_OR_UNRESOLVED:
        return "TRANSIENT_OR_UNRESOLVED"
    if decision.epistemic_state == EPISTEMIC_CHARACTERISED_STABLE:
        return "KNOWN_STABLE"
    if decision.epistemic_state in (EPISTEMIC_OBSERVED_CANDIDATE, EPISTEMIC_CHARACTERISED_INTERMITTENT):
        return "KNOWN_INTERMITTENT_CANDIDATE"
    if decision.epistemic_state == EPISTEMIC_UNKNOWN:
        return "UNKNOWN"
    raise AssertionError(f"unreachable epistemic state: {decision.epistemic_state!r}")


__all__ = [
    "ACTION_LADDER",
    "ACTION_ABSTAIN_CENSORED",
    "ACTION_ABSTAIN_INCONSISTENT",
    "ACTION_ABSTAIN_INSUFFICIENT",
    "ACTION_ELIGIBLE",
    "ACTION_NO_ACTION",
    "ACTION_REQUALIFY",
    "EPISTEMIC_STATES",
    "EPISTEMIC_CENSORED",
    "EPISTEMIC_CHARACTERISED_INTERMITTENT",
    "EPISTEMIC_CHARACTERISED_STABLE",
    "EPISTEMIC_OBSERVED_CANDIDATE",
    "EPISTEMIC_TRANSIENT_OR_UNRESOLVED",
    "EPISTEMIC_UNKNOWN",
    "MISSION_NAMES",
    "MISSION_NAME_PROJECTIONS",
    "NET_BENEFIT_STATES",
    "NET_BENEFIT_ESTABLISHED",
    "NET_BENEFIT_NOT_ESTABLISHED",
    "NOT_REPRESENTATIVE",
    "NO",
    "REASON_CODES",
    "REASON_CODE_MEANINGS",
    "REASON_CODE_VERSION",
    "REPRESENTATIVE",
    "REPRESENTATIVENESS_STATES",
    "RESIDUAL_STATES",
    "RESIDUAL_INDETERMINATE",
    "RESIDUAL_NONE",
    "RESIDUAL_SYSTEMATIC_STABLE",
    "RESIDUAL_VARIABLE",
    "UNDETERMINED",
    "YES",
    "CensoredInferenceError",
    "EvidenceFact",
    "EvidencePacket",
    "QualificationDecision",
    "evaluate",
    "project_mission_name",
]
