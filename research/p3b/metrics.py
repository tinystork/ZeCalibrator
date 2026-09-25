"""P3B metrics contract (LOT5) — calculable metrics on decisions vs ground truth.

Internal, non-public, never imported by ``zecalibrator.api.v1``.

Implements the *calculable* metrics of ``metrics_contract.md`` (§3–§9), each
with an explicit universe and unit, computed over *synthetic* results
(LOT2 decisions vs LOT1 ground-truth labels). This module:

* names the universe **before** the metric — three universes, never conflated:
  ``data`` (pixels / frames / megapixel-frames), ``decision`` (evaluated sites,
  ``(site, evaluation)`` units) and ``truth`` (actually abnormal / actually
  healthy sites);
* declares every metric's numerator **and** denominator in a precise universe,
  and its unit;
* produces **distributions** and **described candidate operating points** —
  never a chosen threshold, never an optimum, never a ranking (§28);
* treats numeric budgets as DEFERRED (§50): it reports descriptions, not
  decisions.

Threshold selection is forbidden here and enforced by tests: no ``argmax`` /
``argmin`` / ``max`` / ``min`` / ``sorted``, no "best"/"optimal" selection, no
numeric budget. Every numeric constant is registered through
:func:`research.p3b.parameters.param` (``EXPLORATORY``).

Censoring: a censored measurement can never support a quantitative inference
(SCIENCE §13.4). The ``censored_mis_inference_count`` invariant must stay 0 and
is tested as such. This module never reads pixels or raw series, so it never
calls a ``features.py`` primitive with an unfiltered censored series.

The two *definition-only* deliverables of §8–§9 (net-benefit pair, collateral
degradation) are declared here as typed value objects; their numeric
computation is the net-benefit harness (LOT6), not this lot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Tuple

from .declared_facts import build_declared_facts, evidence_packet
from .model import ScenarioSpec, compute_bookkeeping
from .parameters import EXPLORATORY, param
from .qualification_policy import (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    ACTION_ABSTAIN_INSUFFICIENT,
    ACTION_ELIGIBLE,
    ACTION_NO_ACTION,
    ACTION_REQUALIFY,
    EPISTEMIC_CHARACTERISED_INTERMITTENT,
    EPISTEMIC_CHARACTERISED_STABLE,
    EPISTEMIC_CENSORED,
    EPISTEMIC_OBSERVED_CANDIDATE,
    EPISTEMIC_TRANSIENT_OR_UNRESOLVED,
    EPISTEMIC_UNKNOWN,
    NET_BENEFIT_ESTABLISHED,
    NET_BENEFIT_NOT_ESTABLISHED,
    NOT_REPRESENTATIVE,
    RC_EVIDENCE_CENSORED,
    RC_GEOMETRY_INCOMPATIBLE,
    RC_INCONSISTENT_EVIDENCE,
    RC_INSUFFICIENT_EPOCHS,
    RC_INSUFFICIENT_INDEPENDENT_GROUPS,
    RC_NET_BENEFIT_NOT_ESTABLISHED,
    RC_NEIGHBOURHOOD_RESIDUAL_UNSTABLE,
    RC_NEIGHBOURHOOD_RESIDUAL_UNDETERMINED,
    RC_NOT_PERSISTENT_IN_SENSOR_COORDINATES,
    RC_PERSISTENCE_UNDETERMINED,
    RC_SENSOR_IDENTITY_UNRESOLVED,
    RC_SITE_RESIDUAL_INDETERMINATE,
    RC_TRANSIENT_ONLY,
    UNDETERMINED,
    YES,
    CensoredInferenceError,
    EvidencePacket,
    QualificationDecision,
    evaluate,
)

# ---------------------------------------------------------------------------
# Tagged numeric constants — EXPLORATORY (metric scaling, never a threshold)
# ---------------------------------------------------------------------------

# Scaling for "per million" rates. A unit conversion, not a product threshold.
RATE_PER_MILLION = param("RATE_PER_MILLION", 1_000_000, EXPLORATORY)


# ---------------------------------------------------------------------------
# Universes (named before every metric) and units
# ---------------------------------------------------------------------------

UNIVERSE_DATA = "data"           # pixels / frames / megapixel-frames
UNIVERSE_DECISION = "decision"   # evaluated sites, (site, evaluation) units
UNIVERSE_TRUTH = "truth"         # actually abnormal / actually healthy sites

# Metric names (stable, versioned vocabulary).
METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE = "site_level_false_promotion_rate"
METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE_PER_MILLION = "site_level_false_promotion_rate_per_million"
METRIC_FALSE_CANDIDATE_RATE = "false_candidate_rate"
METRIC_FALSE_CANDIDATE_RATE_PER_MEGAPIXEL_FRAME = "false_candidate_rate_per_megapixel_frame"
METRIC_ACTION_LEVEL_FALSE_APPLICATION_RATE = "action_level_false_application_rate"
METRIC_UNSAFE_ACTION_RATE = "unsafe_action_rate"
METRIC_CANDIDATE_RECALL = "candidate_recall"
METRIC_QUALIFICATION_RECALL = "qualification_recall"
METRIC_ACTION_ELIGIBILITY_RECALL = "action_eligibility_recall"
METRIC_ABSTENTION_RATE = "abstention_rate"
METRIC_CORRECT_ABSTENTION_RATE = "correct_abstention_rate"
METRIC_QUALIFICATION_PRECISION = "qualification_precision"
METRIC_GROUP_LEVEL_REPEATABILITY = "group_level_repeatability"
METRIC_EPOCH_STABILITY = "epoch_stability"
METRIC_CALIBRATION_REPRESENTATIVENESS_FAILURE_RATE = "calibration_representativeness_failure_rate"
METRIC_REPRESENTATIVENESS_UNDETERMINED_RATE = "representativeness_undetermined_rate"

# Truth label vocabularies (declared in LOT1, never inferred here).
TRUTH_UNQUALIFIED = "UNQUALIFIED"

# ---------------------------------------------------------------------------
# Abstention causes (§32) — decomposition by cause, never merged
# ---------------------------------------------------------------------------

ABSTENTION_CAUSES: Tuple[str, ...] = (
    "CENSORED",
    "INCONSISTENT",
    "INSUFFICIENT_EVIDENCE",
    "INSUFFICIENT_EPOCHS",
    "INSUFFICIENT_INDEPENDENT_GROUPS",
    "NET_BENEFIT_NOT_ESTABLISHED",
    "NOT_PERSISTENT_IN_SENSOR_COORDINATES",
    "PERSISTENCE_UNDETERMINED",
)

# Each cause is matched against the decision's LOT2 reason codes.
ABSTENTION_CAUSE_TO_CODES: Mapping[str, Tuple[str, ...]] = {
    "CENSORED": (RC_EVIDENCE_CENSORED,),
    "INCONSISTENT": (RC_INCONSISTENT_EVIDENCE, RC_TRANSIENT_ONLY, RC_NEIGHBOURHOOD_RESIDUAL_UNSTABLE),
    "INSUFFICIENT_EVIDENCE": (
        RC_SENSOR_IDENTITY_UNRESOLVED,
        RC_GEOMETRY_INCOMPATIBLE,
        RC_SITE_RESIDUAL_INDETERMINATE,
        RC_NEIGHBOURHOOD_RESIDUAL_UNDETERMINED,
    ),
    "INSUFFICIENT_EPOCHS": (RC_INSUFFICIENT_EPOCHS,),
    "INSUFFICIENT_INDEPENDENT_GROUPS": (RC_INSUFFICIENT_INDEPENDENT_GROUPS,),
    "NET_BENEFIT_NOT_ESTABLISHED": (RC_NET_BENEFIT_NOT_ESTABLISHED,),
    "NOT_PERSISTENT_IN_SENSOR_COORDINATES": (RC_NOT_PERSISTENT_IN_SENSOR_COORDINATES,),
    "PERSISTENCE_UNDETERMINED": (RC_PERSISTENCE_UNDETERMINED,),
}

_ABSTAIN_ACTIONS: Tuple[str, ...] = (
    ACTION_ABSTAIN_CENSORED,
    ACTION_ABSTAIN_INCONSISTENT,
    ACTION_ABSTAIN_INSUFFICIENT,
)

_CHARACTERISED: Tuple[str, ...] = (
    EPISTEMIC_CHARACTERISED_STABLE,
    EPISTEMIC_CHARACTERISED_INTERMITTENT,
)


# ---------------------------------------------------------------------------
# Truth / decision classifiers (pure, declarative, no threshold)
# ---------------------------------------------------------------------------


def is_reconstruction_target(outcome: "SiteOutcome") -> bool:
    """Truth axis 2: the declared expected action is targeted reconstruction."""
    return outcome.expected_action_state == ACTION_ELIGIBLE


def is_abnormal(outcome: "SiteOutcome") -> bool:
    """Truth axis 1: the declared qualification state is not ``UNQUALIFIED``."""
    return outcome.expected_qualification_state != TRUTH_UNQUALIFIED


def is_detected(outcome: "SiteOutcome") -> bool:
    """Detection stage: something was observed at the site (epistemic != UNKNOWN)."""
    return outcome.decision.epistemic_state != EPISTEMIC_UNKNOWN


def is_qualified(outcome: "SiteOutcome") -> bool:
    """Qualification stage: the site was characterised (stable or intermittent)."""
    return outcome.decision.epistemic_state in _CHARACTERISED


def is_promoted(outcome: "SiteOutcome") -> bool:
    """Promotion stage: the LOT2 decision reached action eligibility."""
    return outcome.decision.action == ACTION_ELIGIBLE


def is_abstained(outcome: "SiteOutcome") -> bool:
    """Abstention: the decision is one of the three abstain actions."""
    return outcome.decision.action in _ABSTAIN_ACTIONS


def is_applied(outcome: "SiteOutcome") -> bool:
    """Execution stage: the site was actually reconstructed (LOT3 applied flag).

    ``applied is None`` (execution not modelled) is *not* counted as applied:
    a metric that requires execution cannot be silently inflated by an absent
    execution fact.
    """
    return outcome.applied is True


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteOutcome:
    """One ``(site, evaluation)`` decision unit: truth + LOT2 decision + execution.

    ``expected_qualification_state`` / ``expected_action_state`` are the LOT1
    ground-truth labels (declared, never inferred). ``decision`` is the LOT2
    structured decision consumed verbatim. ``packet`` is the evidence packet
    that produced the decision (kept so repeatability can re-evaluate a reduced
    independence hypothesis). ``applied`` is the LOT3 execution fact
    (``True``/``False``/``None`` = not modelled).
    """

    site_id: str
    x: int
    y: int
    cfa_class: str
    expected_qualification_state: str
    expected_action_state: str
    censored: bool
    decision: QualificationDecision
    packet: Optional[EvidencePacket] = None
    applied: Optional[bool] = None


@dataclass(frozen=True)
class MetricValue:
    """One metric result: a named value with its explicit universe and unit.

    ``value is None`` means the denominator was empty (undefined), never a
    silent zero.
    """

    name: str
    universe: str
    unit: str
    value: Optional[float]


@dataclass(frozen=True)
class ConfusionMatrix:
    """Promotion-stage confusion matrix (truth: reconstruction target).

    ``true_positive``  = target ∧ promoted
    ``false_positive`` = ¬target ∧ promoted  (the false-promotion count)
    ``false_negative`` = target ∧ ¬promoted
    ``true_negative``  = ¬target ∧ ¬promoted
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


@dataclass(frozen=True)
class MetricsReport:
    """The full named-metric report over a set of evaluated sites."""

    metrics: Mapping[str, MetricValue]
    confusion: ConfusionMatrix
    abstention_decomposition: Mapping[str, int]
    censored_evaluation_count: int
    censored_mis_inference_count: int
    evaluation_count: int
    negative_count: int
    target_count: int


@dataclass(frozen=True)
class OperatingPoint:
    """A *described* candidate operating point (never selected).

    Each point names the fact setting that produced it and reports the
    resulting confusion matrix and the two headline rates as descriptions.
    There is no "best" field, no optimum, no ranking.
    """

    description: str
    net_benefit_setting: str
    confusion: ConfusionMatrix
    false_promotion_rate: Optional[float]
    action_eligibility_recall: Optional[float]


@dataclass(frozen=True)
class TruthDecisionComparison:
    """Descriptive cross-check of a LOT2 decision against LOT1 truth labels."""

    site_id: str
    cfa_class: str
    expected_qualification_state: str
    expected_action_state: str
    decision_epistemic_state: str
    decision_action: str
    action_agrees: bool


# ---------------------------------------------------------------------------
# Definition-only deliverables (§8–§9) — the harness is LOT6
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetBenefitPair:
    """DEFINITION (§8): net benefit is a paired couple, never a single score.

    ``residual_reduction_at_site`` — the residual reduction *at the site*
    (paired: same local reference, same registration transforms), in ADU and as
    a fraction of the baseline residual.
    ``collateral_degradation`` — the bounded collateral damage elsewhere.

    Both are measured on the *same* paired site, amortised over the image.
    The numeric harness is LOT6; this lot only fixes the shape.
    """

    residual_reduction_at_site: Optional[float] = None
    residual_reduction_fraction: Optional[float] = None
    collateral_degradation: Optional["CollateralDegradation"] = None


@dataclass(frozen=True)
class CollateralDegradation:
    """DEFINITION (§9): the minimum collateral controls, with units.

    Every field is a *required* control. ``nearby_star_*`` must be published
    with the peak↔site distance (``nearby_star_distance_px``).
    ``untouched_domain_invariance`` must be 0 outside the site neighbourhood.
    """

    local_background_bias: Optional[float] = None          # ADU (median bias)
    local_noise: Optional[float] = None                    # ADU (dispersion)
    nearby_star_peak_delta: Optional[float] = None         # ADU
    nearby_star_core_delta: Optional[float] = None         # ADU
    nearby_star_distance_px: Optional[float] = None        # published peak<->site distance
    cfa_structure_preserved: Optional[bool] = None
    photometric_impact: Optional[float] = None             # ADU
    new_artifact_created: Optional[bool] = None
    untouched_domain_invariance: Optional[float] = None    # ADU (must be 0)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _rate(num: int, den: int) -> Optional[float]:
    """A rate as a fraction; ``None`` (never 0.0) when the denominator is empty."""
    if den == 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# Builder: declared facts -> LOT2 decisions -> outcomes
# ---------------------------------------------------------------------------


def build_outcomes(
    scenario: ScenarioSpec,
    *,
    net_benefit_established: Optional[str] = None,
) -> Tuple[SiteOutcome, ...]:
    """Build the ``(site, evaluation)`` outcomes for a scenario (no pixel I/O).

    Declared facts (LOT4) → ``EvidencePacket`` → LOT2 ``evaluate``. The
    ``net_benefit_established`` override is an *exploration knob* for candidate
    operating points; it is never applied to a censored site (a censored
    measurement cannot establish a quantitative net benefit, SCIENCE §13.4).
    """
    declared = build_declared_facts(scenario)
    outcomes = []
    for site, sf in zip(scenario.sites, declared.sites):
        packet = evidence_packet(sf, declared)
        if net_benefit_established is not None and sf.censored_measurement_present != YES:
            packet = replace(packet, net_benefit_established=net_benefit_established)
        decision = evaluate(packet)
        outcomes.append(
            SiteOutcome(
                site_id=site.site_id,
                x=int(site.x),
                y=int(site.y),
                cfa_class=site.cfa_class,
                expected_qualification_state=site.expected_qualification_state,
                expected_action_state=site.expected_action_state,
                censored=(sf.censored_measurement_present == YES),
                decision=decision,
                packet=packet,
                applied=None,
            )
        )
    return tuple(outcomes)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix(outcomes: Tuple[SiteOutcome, ...]) -> ConfusionMatrix:
    """Promotion-stage confusion matrix (truth = reconstruction target)."""
    tp = fp = fn = tn = 0
    for o in outcomes:
        target = is_reconstruction_target(o)
        promoted = is_promoted(o)
        if target and promoted:
            tp += 1
        elif target and not promoted:
            fn += 1
        elif not target and promoted:
            fp += 1
        else:
            tn += 1
    return ConfusionMatrix(tp, fp, fn, tn)


# ---------------------------------------------------------------------------
# Repeatability / stability probes (re-evaluate a reduced independence hypothesis)
# ---------------------------------------------------------------------------


def _reproduces_without_one(o: SiteOutcome, drop_group: bool) -> bool:
    """Would the qualification reproduce with one independent unit removed?

    A qualified site already has ``epoch_count >= 2`` and
    ``independent_group_count >= 2`` (LOT2), so ``count - 1 >= 1 >= 0`` and no
    clamping is needed. This is *not* a threshold: it re-runs the actual LOT2
    engine on a reduced packet and asks whether the epistemic state is still
    characterised.
    """
    p = o.packet
    if p is None:
        return False
    if drop_group:
        reduced = replace(p, independent_group_count=p.independent_group_count - 1)
    else:
        reduced = replace(p, epoch_count=p.epoch_count - 1)
    try:
        d = evaluate(reduced)
    except CensoredInferenceError:
        return False
    return d.epistemic_state in _CHARACTERISED


def group_level_repeatability(outcomes: Tuple[SiteOutcome, ...]) -> Optional[float]:
    """§7: qualified sites whose qualification reproduces with one independent
    *group* removed, over qualified sites. Unit: per qualified site.

    This metric keys on the independent-group count, never on the frame count:
    adding frames to the same sequence cannot improve it (invariant §21).
    """
    qualified = [o for o in outcomes if is_qualified(o)]
    if not qualified:
        return None
    reproduced = 0
    for o in qualified:
        if _reproduces_without_one(o, drop_group=True):
            reproduced += 1
    return _rate(reproduced, len(qualified))


def epoch_stability(outcomes: Tuple[SiteOutcome, ...]) -> Optional[float]:
    """§7: qualified sites whose qualification reproduces with one *epoch*
    removed, over qualified sites. Unit: per qualified site."""
    qualified = [o for o in outcomes if is_qualified(o)]
    if not qualified:
        return None
    reproduced = 0
    for o in qualified:
        if _reproduces_without_one(o, drop_group=False):
            reproduced += 1
    return _rate(reproduced, len(qualified))


# ---------------------------------------------------------------------------
# The metric computation (pure; every metric names universe + unit)
# ---------------------------------------------------------------------------


def compute_metrics(
    outcomes: Tuple[SiteOutcome, ...],
    *,
    frame_count: int,
    megapixels_per_frame: float,
) -> MetricsReport:
    """Compute all named metrics over ``outcomes`` (pure, deterministic).

    ``frame_count`` is the number of *light* frames treated (the data-universe
    denominator of ``false_candidate_rate``). ``megapixels_per_frame`` is the
    sensor megapixel count (used for the per-megapixel-frame variant). Neither
    is a threshold; both are descriptive denominators supplied by the caller.
    """
    sites = list(outcomes)
    n = len(sites)
    negatives = [o for o in sites if not is_reconstruction_target(o)]
    targets = [o for o in sites if is_reconstruction_target(o)]
    n_neg = len(negatives)
    n_tar = len(targets)

    # False-positive family (each a *different* universe / denominator).
    false_promotions = 0
    false_candidates = 0
    false_applications = 0
    negative_promoted = 0
    for o in negatives:
        if is_promoted(o):
            false_promotions += 1
            negative_promoted += 1
        if is_applied(o):
            false_applications += 1
        # A false candidate is a *healthy* site flagged by detection (load).
    for o in sites:
        if (not is_abnormal(o)) and is_detected(o):
            false_candidates += 1

    # Recalls (separated, never aggregated) over reconstruction targets.
    candidate_recalled = 0
    qualification_recalled = 0
    eligibility_recalled = 0
    for o in targets:
        if is_detected(o):
            candidate_recalled += 1
        if is_qualified(o):
            qualification_recalled += 1
        if is_promoted(o):
            eligibility_recalled += 1

    # Qualification precision over qualified sites (truth confirmation).
    qualified = [o for o in sites if is_qualified(o)]
    qualified_confirmed = 0
    for o in qualified:
        if is_abnormal(o):
            qualified_confirmed += 1

    # Abstention.
    abstained = [o for o in sites if is_abstained(o)]
    correct_abstentions = 0
    for o in abstained:
        if not is_reconstruction_target(o):
            correct_abstentions += 1

    # Abstention decomposition by cause.
    decomposition: dict = {}
    for cause in ABSTENTION_CAUSES:
        codes = ABSTENTION_CAUSE_TO_CODES[cause]
        count = 0
        for o in abstained:
            if any(rc in o.decision.reason_codes for rc in codes):
                count += 1
        decomposition[cause] = count

    # Calibration representativeness (axis 2 of the decision).
    not_representative = 0
    representativeness_undetermined = 0
    for o in sites:
        if o.decision.representativeness == NOT_REPRESENTATIVE:
            not_representative += 1
        elif o.decision.representativeness == UNDETERMINED:
            representativeness_undetermined += 1

    # Censored evidence invariant (§10): a censored measurement must produce
    # exactly zero quantitative inferences (characterised residual / eligible).
    censored_evals = 0
    censored_mis_inferences = 0
    for o in sites:
        if o.censored:
            censored_evals += 1
            if is_qualified(o) or is_promoted(o):
                censored_mis_inferences += 1

    cm = confusion_matrix(tuple(sites))

    # Per-frame and per-megapixel-frame variants of the false candidate rate.
    false_candidate_per_megapixel_frame = _rate(
        false_candidates, int(round(frame_count * megapixels_per_frame))
    ) if megapixels_per_frame > 0 else None

    metrics: dict = {
        METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE: MetricValue(
            METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE, UNIVERSE_DECISION,
            "per negative site (fraction)", _rate(false_promotions, n_neg),
        ),
        METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE_PER_MILLION: MetricValue(
            METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE_PER_MILLION, UNIVERSE_DECISION,
            "per million negative sites",
            None if _rate(false_promotions, n_neg) is None
            else _rate(false_promotions, n_neg) * RATE_PER_MILLION,
        ),
        METRIC_FALSE_CANDIDATE_RATE: MetricValue(
            METRIC_FALSE_CANDIDATE_RATE, UNIVERSE_DATA,
            "per light frame", _rate(false_candidates, frame_count),
        ),
        METRIC_FALSE_CANDIDATE_RATE_PER_MEGAPIXEL_FRAME: MetricValue(
            METRIC_FALSE_CANDIDATE_RATE_PER_MEGAPIXEL_FRAME, UNIVERSE_DATA,
            "per megapixel-frame", false_candidate_per_megapixel_frame,
        ),
        METRIC_ACTION_LEVEL_FALSE_APPLICATION_RATE: MetricValue(
            METRIC_ACTION_LEVEL_FALSE_APPLICATION_RATE, UNIVERSE_DECISION,
            "per negative promoted site (conditional)",
            _rate(false_applications, negative_promoted),
        ),
        METRIC_UNSAFE_ACTION_RATE: MetricValue(
            METRIC_UNSAFE_ACTION_RATE, UNIVERSE_DECISION,
            "per negative site", _rate(false_applications, n_neg),
        ),
        METRIC_CANDIDATE_RECALL: MetricValue(
            METRIC_CANDIDATE_RECALL, UNIVERSE_DECISION,
            "per reconstruction target", _rate(candidate_recalled, n_tar),
        ),
        METRIC_QUALIFICATION_RECALL: MetricValue(
            METRIC_QUALIFICATION_RECALL, UNIVERSE_DECISION,
            "per reconstruction target", _rate(qualification_recalled, n_tar),
        ),
        METRIC_ACTION_ELIGIBILITY_RECALL: MetricValue(
            METRIC_ACTION_ELIGIBILITY_RECALL, UNIVERSE_DECISION,
            "per reconstruction target", _rate(eligibility_recalled, n_tar),
        ),
        METRIC_ABSTENTION_RATE: MetricValue(
            METRIC_ABSTENTION_RATE, UNIVERSE_DECISION,
            "per evaluation", _rate(len(abstained), n),
        ),
        METRIC_CORRECT_ABSTENTION_RATE: MetricValue(
            METRIC_CORRECT_ABSTENTION_RATE, UNIVERSE_DECISION,
            "per abstention", _rate(correct_abstentions, len(abstained)),
        ),
        METRIC_QUALIFICATION_PRECISION: MetricValue(
            METRIC_QUALIFICATION_PRECISION, UNIVERSE_DECISION,
            "per qualified site", _rate(qualified_confirmed, len(qualified)),
        ),
        METRIC_GROUP_LEVEL_REPEATABILITY: MetricValue(
            METRIC_GROUP_LEVEL_REPEATABILITY, UNIVERSE_DECISION,
            "per qualified site", group_level_repeatability(tuple(sites)),
        ),
        METRIC_EPOCH_STABILITY: MetricValue(
            METRIC_EPOCH_STABILITY, UNIVERSE_DECISION,
            "per qualified site", epoch_stability(tuple(sites)),
        ),
        METRIC_CALIBRATION_REPRESENTATIVENESS_FAILURE_RATE: MetricValue(
            METRIC_CALIBRATION_REPRESENTATIVENESS_FAILURE_RATE, UNIVERSE_DECISION,
            "per evaluation", _rate(not_representative, n),
        ),
        METRIC_REPRESENTATIVENESS_UNDETERMINED_RATE: MetricValue(
            METRIC_REPRESENTATIVENESS_UNDETERMINED_RATE, UNIVERSE_DECISION,
            "per evaluation", _rate(representativeness_undetermined, n),
        ),
    }

    return MetricsReport(
        metrics=metrics,
        confusion=cm,
        abstention_decomposition=decomposition,
        censored_evaluation_count=censored_evals,
        censored_mis_inference_count=censored_mis_inferences,
        evaluation_count=n,
        negative_count=n_neg,
        target_count=n_tar,
    )


# ---------------------------------------------------------------------------
# Candidate operating points (described, never selected)
# ---------------------------------------------------------------------------


def candidate_operating_points(scenario: ScenarioSpec) -> Tuple[OperatingPoint, ...]:
    """Sweep the net-benefit fact over its candidate settings and describe the
    resulting promotion-stage confusion matrices.

    This is a *description* of candidate operating points (an exploratory
    ROC-like set of ``(false_promotion_rate, action_eligibility_recall)``
    points). No point is chosen, no optimum is maximised, no algorithm is
    ranked (§28). The numeric budgets stay DEFERRED (§50).
    """
    points = []
    for setting in (UNDETERMINED, NET_BENEFIT_NOT_ESTABLISHED, NET_BENEFIT_ESTABLISHED):
        outcomes = build_outcomes(scenario, net_benefit_established=setting)
        cm = confusion_matrix(outcomes)
        fpr = _rate(cm.false_positive, cm.false_positive + cm.true_negative)
        tpr = _rate(cm.true_positive, cm.true_positive + cm.false_negative)
        points.append(
            OperatingPoint(
                description=f"net_benefit_established={setting}",
                net_benefit_setting=setting,
                confusion=cm,
                false_promotion_rate=fpr,
                action_eligibility_recall=tpr,
            )
        )
    return tuple(points)


# ---------------------------------------------------------------------------
# Cross-check: LOT2 decisions vs LOT1 truth labels (descriptive)
# ---------------------------------------------------------------------------


def compare_to_truth(outcomes: Tuple[SiteOutcome, ...]) -> Tuple[TruthDecisionComparison, ...]:
    """Compare LOT2 decisions to LOT1 truth labels and report the deviations.

    Purely descriptive: each row reports the declared truth, the decision, and
    whether the *action* agrees (exact string equality on the shared action
    vocabulary). This is one of the bench's cross-checks, not a decision.
    """
    rows = []
    for o in outcomes:
        rows.append(
            TruthDecisionComparison(
                site_id=o.site_id,
                cfa_class=o.cfa_class,
                expected_qualification_state=o.expected_qualification_state,
                expected_action_state=o.expected_action_state,
                decision_epistemic_state=o.decision.epistemic_state,
                decision_action=o.decision.action,
                action_agrees=(o.expected_action_state == o.decision.action),
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Convenience: full scenario -> report
# ---------------------------------------------------------------------------


def metrics_for_scenario(
    scenario: ScenarioSpec,
    *,
    net_benefit_established: Optional[str] = None,
) -> MetricsReport:
    """Build outcomes and compute the full named-metric report for a scenario."""
    bk = compute_bookkeeping(scenario)
    ny, nx = scenario.sensor.shape
    megapixels_per_frame = (ny * nx) / RATE_PER_MILLION
    outcomes = build_outcomes(scenario, net_benefit_established=net_benefit_established)
    return compute_metrics(
        outcomes,
        frame_count=bk.observation_count,
        megapixels_per_frame=megapixels_per_frame,
    )


__all__ = [
    "ABSTENTION_CAUSES",
    "ABSTENTION_CAUSE_TO_CODES",
    "CollateralDegradation",
    "ConfusionMatrix",
    "METRIC_ACTION_ELIGIBILITY_RECALL",
    "METRIC_ACTION_LEVEL_FALSE_APPLICATION_RATE",
    "METRIC_ABSTENTION_RATE",
    "METRIC_CALIBRATION_REPRESENTATIVENESS_FAILURE_RATE",
    "METRIC_CANDIDATE_RECALL",
    "METRIC_CORRECT_ABSTENTION_RATE",
    "METRIC_EPOCH_STABILITY",
    "METRIC_FALSE_CANDIDATE_RATE",
    "METRIC_FALSE_CANDIDATE_RATE_PER_MEGAPIXEL_FRAME",
    "METRIC_GROUP_LEVEL_REPEATABILITY",
    "METRIC_QUALIFICATION_PRECISION",
    "METRIC_QUALIFICATION_RECALL",
    "METRIC_REPRESENTATIVENESS_UNDETERMINED_RATE",
    "METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE",
    "METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE_PER_MILLION",
    "METRIC_UNSAFE_ACTION_RATE",
    "MetricsReport",
    "MetricValue",
    "NetBenefitPair",
    "OperatingPoint",
    "RATE_PER_MILLION",
    "SiteOutcome",
    "TRUTH_UNQUALIFIED",
    "TruthDecisionComparison",
    "UNIVERSE_DATA",
    "UNIVERSE_DECISION",
    "UNIVERSE_TRUTH",
    "build_outcomes",
    "candidate_operating_points",
    "compare_to_truth",
    "compute_metrics",
    "confusion_matrix",
    "epoch_stability",
    "group_level_repeatability",
    "is_abnormal",
    "is_abstained",
    "is_applied",
    "is_detected",
    "is_promoted",
    "is_qualified",
    "is_reconstruction_target",
    "metrics_for_scenario",
]
