"""P3B scenario fixtures (research/test only) — explicit declared evidence per class.

Internal, non-public, never imported by ``zecalibrator.api.v1`` and never imported
from ``src/``.

A synthetic class name from :mod:`research.p3b.catalog` does **not** determine a
scenario on its own: the ``default_expected_*`` truth labels assume declarations
that live *outside* the class name (net benefit, calibration representativeness,
censoring) and an acquisition structure (epochs / groups / frames) that the class
name does not specify. Without those, the policy conservatively abstains — the
"abstention-refuge" behaviour recorded in the P3B review addendum F1 (e.g.
``INTERMITTENT_TWO_STATE`` → ``ABSTAIN_INSUFFICIENT_EVIDENCE`` /
``NET_BENEFIT_NOT_ESTABLISHED``).

This module makes those declarations **explicit** and **machine-visible**:

* :func:`scenario_for_class` builds a coherent :class:`research.p3b.model.ScenarioSpec`
  for the requested class (pixels + acquisition structure) and a
  :class:`ScenarioFixture` that carries, for **every** non-pixel fact the policy
  consumes, a value **and** its provenance:

  ============================== =============================================
  provenance                    what it covers
  ============================== =============================================
  ``pixels``                    the generated scenario (frame content)
  ``acquisition_structure``     epochs / groups / frames and the independence
                                bookkeeping derived from them
  ``external_declared_evidence`` net benefit, representativeness, censoring and
                                the class→fact translation (residual / persistence /
                                neighbourhood / transient / conflict)
  ``ground_truth_only``         the expected labels (``default_expected_*``), never
                                consumed by the policy
  ============================== =============================================

The helper never *silently injects* the desired policy result: every declaration
is a first-class field of the fixture, and :func:`evaluate_fixture` runs the real
LOT2 policy on those declarations. A reader can therefore tell, per field, whether
a value came from pixels, from the acquisition structure, from a declaration, or
from ground truth.

Reachability is reported honestly. For the classes whose ``default_expected_*``
are consistent with their declared facts, supplying the full declared evidence
reaches the default. For the few classes whose catalogue default is *inconsistent*
with their declared facts (a pre-existing catalogue/facts inconsistency, out of
scope here), the fixture exposes ``default_reachable=False`` and the actual
conservative abstention, rather than faking the label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .catalog import resolve_class
from .metrics import build_outcomes
from .model import (
    Bookkeeping,
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SensorSpec,
    SiteSpec,
    compute_bookkeeping,
)
from .qualification_policy import (
    NET_BENEFIT_ESTABLISHED,
    UNDETERMINED,
    QualificationDecision,
)

# ---------------------------------------------------------------------------
# Provenance vocabulary (the four sources, kept separable and machine-visible)
# ---------------------------------------------------------------------------

PROVENANCE_PIXELS = "pixels"
PROVENANCE_ACQUISITION = "acquisition_structure"
PROVENANCE_DECLARED = "external_declared_evidence"
PROVENANCE_TRUTH = "ground_truth_only"

PROVENANCES: Tuple[str, ...] = (
    PROVENANCE_PIXELS,
    PROVENANCE_ACQUISITION,
    PROVENANCE_DECLARED,
    PROVENANCE_TRUTH,
)

# Provenance of each EvidencePacket field (the non-pixel facts the policy reads).
_PACKET_PROVENANCE: Mapping[str, str] = {
    # Structural synthetic facts (declared, never measured).
    "sensor_identity_resolved": PROVENANCE_DECLARED,
    "geometry_compatible": PROVENANCE_DECLARED,
    # Derived from the acquisition structure (darks present, group/epoch counts).
    "calibration_present": PROVENANCE_ACQUISITION,
    "independent_group_count": PROVENANCE_ACQUISITION,
    "epoch_count": PROVENANCE_ACQUISITION,
    # External declared evidence (net benefit, representativeness, censoring)
    # and the declared class→fact translation.
    "calibration_representativeness": PROVENANCE_DECLARED,
    "persisted_at_same_sensor_coord": PROVENANCE_DECLARED,
    "censored_measurement_present": PROVENANCE_DECLARED,
    "site_residual_behaviour": PROVENANCE_DECLARED,
    "neighbourhood_residual_stable": PROVENANCE_DECLARED,
    "transient_only": PROVENANCE_DECLARED,
    "conflicting_evidence": PROVENANCE_DECLARED,
    "net_benefit_established": PROVENANCE_DECLARED,
    "persisted_basis": PROVENANCE_DECLARED,
    "censored_measurement_count": PROVENANCE_DECLARED,
}

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredField:
    """One non-pixel fact the policy consumes, with its machine-visible provenance."""

    name: str
    value: object
    provenance: str


@dataclass(frozen=True)
class ExpectedTruth:
    """The catalogue's ``default_expected_*`` truth labels (never consumed by policy)."""

    qualification_state: str
    action_state: str


@dataclass(frozen=True)
class ScenarioFixture:
    """A coherent synthetic scenario plus every non-pixel declaration, with provenance.

    ``reachable_action`` is the LOT2 action reached when the full declared evidence
    is applied (computed by the real policy, never invented). ``default_reachable``
    is ``reachable_action == expected_truth.action_state``.
    """

    class_name: str
    scenario: ScenarioSpec
    declared_evidence: Tuple[DeclaredField, ...]
    independence_structure: Bookkeeping
    net_benefit_declaration: str
    expected_truth: ExpectedTruth
    reachable_action: str
    default_reachable: bool


# ---------------------------------------------------------------------------
# Per-class declared evidence (data, exposed — never silently applied)
# ---------------------------------------------------------------------------
#
# Columns: (net_benefit_declaration, calibration_representativeness, censored).
# representativeness uses the SiteSpec vocabulary ("representative" /
# "not_representative"); censored is the SiteSpec ``censored`` flag.
#
# net benefit is ``ESTABLISHED`` exactly for the classes whose declared default
# action is ``ELIGIBLE_FOR_TARGETED_RECONSTRUCTION`` (the promotion classes):
# to *reach* that default the reader must declare net benefit established. For
# every other class it is ``UNDETERMINED`` (not established / not needed).

_CLASS_EVIDENCE: Mapping[str, Tuple[str, str, bool]] = {
    "NORMAL": (UNDETERMINED, "representative", False),
    "STABLE_ANOMALY_CORRECTED_BY_DARK": (UNDETERMINED, "representative", False),
    "STABLE_ANOMALY_WITH_MISMATCHED_DARK": (UNDETERMINED, "not_representative", False),
    "INTERMITTENT_TWO_STATE": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "INTERMITTENT_MULTI_STATE": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "INTERMITTENT_CONTINUOUS": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "RARE_HIGH_STATE": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "RARE_LOW_STATE": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "SIGN_CHANGING_POST_DARK": (UNDETERMINED, "representative", False),
    "CENSORED_ANOMALY": (UNDETERMINED, "representative", True),
    "SINGLE_TRANSIENT": (UNDETERMINED, "representative", False),
    "OPTICAL_STRUCTURE": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "STAR_CROSSING_SITE": (UNDETERMINED, "representative", False),
    "UNDERSAMPLED_STAR_CORE": (UNDETERMINED, "representative", False),
    "COSMIC_RAY": (UNDETERMINED, "representative", False),
    "NOISE_EXTREME": (UNDETERMINED, "representative", False),
    "FLAT_STRUCTURE": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "DUST_OR_VIGNETTING": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "NEAR_SATURATION": (NET_BENEFIT_ESTABLISHED, "representative", False),
    "AMBIGUOUS_INSUFFICIENT_EVIDENCE": (UNDETERMINED, "representative", False),
}


# ---------------------------------------------------------------------------
# Scenario construction (deterministic, no RNG at construction time)
# ---------------------------------------------------------------------------


def _build_sensor() -> SensorSpec:
    return SensorSpec(
        instance_id="SYNTH-DET-0001",
        model="SYNTH-S50",
        shape=(96, 128),
        cfa_pattern="GRBG",
        binning=(1, 1),
        roi_origin=(0, 0),
        gain=80.0,
        offset_adu=800.0,
        saturation_limit_adu=60000.0,
        filter="NONE",
    )


def _build_scenario(
    name: str,
    seed: int,
    *,
    representativeness: str,
    censored: bool,
    light_epochs: int = 2,
    groups_per_epoch: int = 2,
    frames_per_group: int = 2,
) -> ScenarioSpec:
    """Build a coherent scenario: darks (calibration) + ``light_epochs`` ×
    ``groups_per_epoch`` independent light groups of ``frames_per_group`` frames.

    ``light_epochs=2, groups_per_epoch=2`` yields the independent structure the
    policy needs for qualification (``epoch_count >= 2`` and
    ``independent_group_count >= 2``); callers can reduce it to exercise the
    conservative abstention path.
    """
    sensor = _build_sensor()

    dark_frames = tuple(
        FrameSpec(
            frame_id=f"dark{i}",
            frame_type="dark",
            epoch_id="cal",
            group_id="calg",
            ordinal=i,
        )
        for i in range(4)
    )
    epochs = [EpochSpec("cal", (GroupSpec("calg", dark_frames),))]

    counter = 0
    for e in range(light_epochs):
        groups = []
        for g in range(groups_per_epoch):
            frames = tuple(
                FrameSpec(
                    frame_id=f"light{counter + i}",
                    frame_type="light",
                    epoch_id=f"e{e}",
                    group_id=f"g{e}_{g}",
                    ordinal=i,
                )
                for i in range(frames_per_group)
            )
            counter += frames_per_group
            groups.append(GroupSpec(group_id=f"g{e}_{g}", frames=frames))
        epochs.append(EpochSpec(epoch_id=f"e{e}", groups=tuple(groups)))

    site = SiteSpec(
        site_id="site0",
        x=8,
        y=8,
        cfa_class=name,
        calibration_representativeness=representativeness,
        censored=censored,
    )
    return ScenarioSpec(
        name=f"fixture:{name}",
        seed=seed,
        sensor=sensor,
        epochs=tuple(epochs),
        sites=(site,),
    )


def _reduce_independence(scenario: ScenarioSpec) -> ScenarioSpec:
    """Rebuild ``scenario`` with a single light epoch × single light group.

    Used to demonstrate the conservative-abstention path when the independence
    declaration is removed.
    """
    site = scenario.sites[0]
    return _build_scenario(
        site.cfa_class,
        scenario.seed,
        representativeness=site.calibration_representativeness,
        censored=site.censored,
        light_epochs=1,
        groups_per_epoch=1,
    )


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def scenario_for_class(name: str, *, seed: int = 0) -> ScenarioFixture:
    """Build a :class:`ScenarioFixture` for catalogue class ``name``.

    Raises the typed :class:`research.p3b.catalog.UnknownClassError` for an
    unknown name (never a bare ``KeyError``). Deterministic: the same seed and
    name produce an equal fixture (scenario + declarations).
    """
    entry = resolve_class(name)  # validates + raises UnknownClassError

    net_benefit, representativeness, censored = _CLASS_EVIDENCE[name]
    scenario = _build_scenario(
        name, seed, representativeness=representativeness, censored=censored
    )

    # Run the real policy on the declared evidence (never a hand-computed result).
    outcome = build_outcomes(scenario, net_benefit_established=net_benefit)[0]
    decision = outcome.decision
    packet = outcome.packet

    declared_evidence = tuple(
        DeclaredField(field, getattr(packet, field), _PACKET_PROVENANCE[field])
        for field in _PACKET_PROVENANCE
    )

    expected_truth = ExpectedTruth(
        qualification_state=entry.default_expected_qualification_state,
        action_state=entry.default_expected_action_state,
    )

    return ScenarioFixture(
        class_name=name,
        scenario=scenario,
        declared_evidence=declared_evidence,
        independence_structure=compute_bookkeeping(scenario),
        net_benefit_declaration=net_benefit,
        expected_truth=expected_truth,
        reachable_action=decision.action,
        default_reachable=(decision.action == expected_truth.action_state),
    )


# ---------------------------------------------------------------------------
# Evaluation (runs the real LOT2 policy on the fixture's declarations)
# ---------------------------------------------------------------------------


def evaluate_fixture(
    fixture: ScenarioFixture, *, remove: Optional[str] = None
) -> QualificationDecision:
    """Run the LOT2 policy on the fixture, optionally with one declaration removed.

    ``remove`` is one of:

    * ``"net_benefit"``     — revert the net-benefit declaration to ``UNDETERMINED``;
    * ``"independence"``    — reduce the acquisition structure to one light epoch
      × one light group (insufficient independence);
    * ``"representativeness"`` — flip the declared calibration representativeness;
    * ``"censoring"``       — drop the declared censoring flag.

    ``remove=None`` applies the full declared evidence (identical to the policy
    run recorded in ``fixture.reachable_action``).
    """
    scenario = fixture.scenario
    net_benefit = fixture.net_benefit_declaration

    if remove == "independence":
        scenario = _reduce_independence(scenario)
    elif remove == "net_benefit":
        net_benefit = UNDETERMINED
    elif remove == "representativeness":
        site = scenario.sites[0]
        flipped = "not_representative" if site.calibration_representativeness == "representative" else "representative"
        scenario = _build_scenario(
            site.cfa_class,
            scenario.seed,
            representativeness=flipped,
            censored=site.censored,
        )
    elif remove == "censoring":
        site = scenario.sites[0]
        scenario = _build_scenario(
            site.cfa_class,
            scenario.seed,
            representativeness=site.calibration_representativeness,
            censored=False,
        )
    elif remove is not None:
        raise ValueError(f"unknown evidence to remove: {remove!r}")

    return build_outcomes(scenario, net_benefit_established=net_benefit)[0].decision


__all__ = [
    "PROVENANCE_ACQUISITION",
    "PROVENANCE_DECLARED",
    "PROVENANCE_PIXELS",
    "PROVENANCE_TRUTH",
    "PROVENANCES",
    "DeclaredField",
    "ExpectedTruth",
    "ScenarioFixture",
    "evaluate_fixture",
    "scenario_for_class",
]
