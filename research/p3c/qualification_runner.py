"""P3C-3 — qualification campaign runner (measure frozen candidates, HOLDOUT sealed).

Research-only, internal, non-public. This is the **truth-side** campaign
harness: it freezes (via :mod:`research.p3c.candidate_freeze`) and then
**measures** the frozen candidates on DEVELOPMENT + QUALIFICATION only. The
HOLDOUT is sealed **structurally**, not merely "not done":

* This module (and every ``research/p3c`` module) never imports
  ``research.p3b.holdout``, never references ``DATASET_HOLDOUT``, and never
  receives a holdout path or manifest. There is no parameter, argument or
  constant anywhere in the campaign code that could name the holdout — it is
  *unrepresentable* at the API surface (enforced by
  tests/p3c/test_qualification_holdout_seal.py).
* :func:`run_qualification_campaign` takes a single :class:`CandidateFreeze`
  (which carries only the DEVELOPMENT and QUALIFICATION manifest hashes) and
  derives the QUALIFICATION scenarios from
  :mod:`research.p3c.qualification_corpus`. No holdout manifest can be passed.

The measurement path reuses the existing, accepted machinery **verbatim** — it
never re-implements an action path (§33–§34):

    infer_site (P3C candidate rules)
        -> evidence_bridge.to_evidence_packet (P3C adapter)
        -> qualification_policy.evaluate (LOT2, unchanged)
        -> metrics.compute_metrics (LOT5, unchanged)

``preparation_plan`` (P-A, site × run) is left untouched; the campaign produces
no reconstruction product, so ``applied`` stays ``None`` and
``unsafe_action_rate`` stays null by construction (§3). The P0 gate (no
inter-frame switching) is exercised by the unchanged qualification_policy and
preparation_plan modules.

Reproducibility (§44): every run records the source-code SHA, the candidate
config hashes, the dataset manifest hash, the seed set, the metrics-contract
version and the reason-code version. The campaign is a pure function of the
freeze, so the same freeze yields byte-identical JSON.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from research.p3b.features import compute_features
from research.p3b.generator import generate_corpus
from research.p3b.metrics import (
    METRIC_ABSTENTION_RATE,
    METRIC_CANDIDATE_RECALL,
    METRIC_CORRECT_ABSTENTION_RATE,
    METRIC_FALSE_CANDIDATE_RATE,
    METRIC_QUALIFICATION_RECALL,
    METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE,
    METRIC_UNSAFE_ACTION_RATE,
    PIXELS_PER_MEGAPIXEL,
    SiteOutcome,
    compute_metrics,
)
from research.p3b.qualification_policy import (
    ACTION_ELIGIBLE,
    ACTION_REQUALIFY,
    RESIDUAL_SYSTEMATIC_STABLE,
    RESIDUAL_VARIABLE,
    YES,
    evaluate,
)

from .candidate_freeze import CandidateFreeze, freeze_candidates, source_hashes
from .development_corpus import build_admission
from .evidence_bridge import to_evidence_packet
from .inference_candidates import CANDIDATES, infer_site
from .inference_contract import SENSOR_EVIDENCE_FACTS
from .oracle import (
    SENSOR_EVIDENCE_FACT_NAMES,
    oracle_entry,
    sensor_evidence_truth,
)
from .qualification_corpus import (
    QUALIFICATION_SEEDS,
    build_qualification_scenario,
    qualification_scenarios,
)

# Campaign artifacts schema.
CAMPAIGN_SCHEMA = "zecalibrator-p3c-qualification-campaign"
CAMPAIGN_VERSION = 1

# The headline metrics the mission names (§3) — a stable, ordered subset.
HEADLINE_METRIC_NAMES: Tuple[str, ...] = (
    METRIC_CANDIDATE_RECALL,
    METRIC_QUALIFICATION_RECALL,
    METRIC_FALSE_CANDIDATE_RATE,
    METRIC_SITE_LEVEL_FALSE_PROMOTION_RATE,
    METRIC_ABSTENTION_RATE,
    METRIC_CORRECT_ABSTENTION_RATE,
    METRIC_UNSAFE_ACTION_RATE,
)

# The five prioritised safety errors (§32). They "count more than a simple
# false negative" and are reported as dedicated lines per candidate. Two of them
# (censored->quantitative, non-representative dark->reconstruction) are ALSO
# §40 hard invariants; the other three (healthy/optical/sky structure->sensor
# site, transient->persistent defect, star crossing->fixed defect) are
# priority evidence-level errors that are reported but are *not* by themselves
# grounds for INVALID elimination.
SAFETY_ERROR_NAMES: Tuple[str, ...] = (
    "healthy_or_optical_structure_as_sensor_site",
    "transient_as_persistent_defect",
    "censored_quantitative_inference",
    "non_representative_dark_as_reconstruction_candidate",
    "star_crossing_as_fixed_sensor_defect",
)

# §40 hard invariants — the ONLY grounds for INVALID elimination (§8). The two
# measurable ones map to named safety errors above; the other two (truth leak,
# inter-frame switching) are enforced *structurally* by the truth-leak guard and
# the unchanged qualification_policy / preparation_plan modules, so they cannot
# be produced by any candidate and are reported as "structurally excluded".
HARD_INVARIANT_MEASURED: Tuple[str, ...] = (
    "censored_quantitative_inference",
    "non_representative_dark_as_reconstruction_candidate",
)
HARD_INVARIANT_STRUCTURAL: Tuple[str, ...] = (
    "truth_leak",
    "inter_frame_switching",
)

# Classes whose per-class failure must never be masked by a global rate (§31).
_MUST_NOT_MASK_CLASSES: Tuple[str, ...] = (
    "INTERMITTENT_TWO_STATE",
    "INTERMITTENT_MULTI_STATE",
    "INTERMITTENT_CONTINUOUS",
    "RARE_HIGH_STATE",
    "RARE_LOW_STATE",
    "CENSORED_ANOMALY",
    "STAR_CROSSING_SITE",
    "FLAT_STRUCTURE",
)

_DARK_FRAMES = 4
_SITE_X = 8
_SITE_Y = 8


class CandidateDriftError(RuntimeError):
    """The live candidates differ from the frozen snapshot — refuse to measure.

    Raised by :func:`run_qualification_campaign` when the current source hashes
    or candidate-parameter hashes do not match the freeze. This is the
    structural enforcement of §28: no candidate may be re-tuned on the strength
    of qualification results, because any such re-tuning makes the drift check
    fail and the campaign refuses to run.
    """


@dataclass(frozen=True)
class QualifiedSite:
    """One measured QUALIFICATION site (features + admission + truth keys)."""

    class_name: str
    seed: int
    features: object  # research.p3b.features.SiteFeatures (duck-typed)
    admission: object  # research.p3c.inference_contract.AdmissionFacts
    censored: bool


# ---------------------------------------------------------------------------
# Drift check (structural §28 enforcement)
# ---------------------------------------------------------------------------


def _assert_not_drifted(freeze: CandidateFreeze) -> None:
    """Raise :class:`CandidateDriftError` if live candidates differ from freeze."""
    if dict(source_hashes()) != dict(freeze.source_hashes):
        raise CandidateDriftError(
            "inference-side source code differs from the frozen snapshot; "
            "refusing to measure (no candidate may be re-tuned after freeze)"
        )
    frozen_by_id = {c["candidate_id"]: c for c in freeze.candidates}
    if set(frozen_by_id) != {c.candidate_id for c in CANDIDATES}:
        raise CandidateDriftError(
            "the frozen candidate set differs from the live candidate set"
        )
    for c in CANDIDATES:
        if c.as_dict() != frozen_by_id[c.candidate_id]:
            raise CandidateDriftError(
                f"candidate {c.candidate_id!r} parameters differ from the frozen "
                "snapshot; refusing to measure"
            )


# ---------------------------------------------------------------------------
# Materialisation (features computed once, shared by every candidate)
# ---------------------------------------------------------------------------


def _materialise_sites(scenarios) -> Tuple[QualifiedSite, ...]:
    """Compute LOT4 features + admission for every scenario (once).

    The features are candidate-independent: they are computed from the *same*
    materialised frames for every candidate, so the candidates differ only in
    their rules, never in the measured data they see.
    """
    sites = []
    for scenario in scenarios:
        class_name = scenario.sites[0].cfa_class
        censored = scenario.sites[0].censored
        with tempfile.TemporaryDirectory() as td:
            corpus = generate_corpus(scenario, td)
            dark = np.median(
                np.stack([corpus.frame_arrays[f"dark{i}"] for i in range(_DARK_FRAMES)]),
                axis=0,
            )
            cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
        sites.append(
            QualifiedSite(
                class_name=class_name,
                seed=int(scenario.seed),
                features=cf.sites[0],
                admission=build_admission(scenario),
                censored=censored,
            )
        )
    return tuple(sites)


def _light_frame_count(scenarios) -> int:
    """Total light (science) frames across the qualification set."""
    return sum(len(s.light_frames()) for s in scenarios)


def _megapixels_per_frame(scenarios) -> float:
    ny, nx = scenarios[0].sensor.shape
    return (ny * nx) / PIXELS_PER_MEGAPIXEL


# ---------------------------------------------------------------------------
# Per-site inference -> policy -> outcome (verbatim reuse, never reimplemented)
# ---------------------------------------------------------------------------


def _outcomes_for(candidate_id: str, sites: Tuple[QualifiedSite, ...]) -> Tuple[SiteOutcome, ...]:
    cfg = next(c for c in CANDIDATES if c.candidate_id == candidate_id)
    outcomes = []
    for site in sites:
        entry = oracle_entry(site.class_name)
        evidence = infer_site(cfg, site.features, site.admission)
        packet = to_evidence_packet(evidence)
        decision = evaluate(packet)
        outcomes.append(
            SiteOutcome(
                site_id=f"{site.class_name}:{site.seed}",
                x=_SITE_X,
                y=_SITE_Y,
                cfa_class=site.class_name,
                expected_qualification_state=entry.epistemic_state,
                expected_action_state=entry.expected_action,
                censored=site.censored,
                decision=decision,
                packet=packet,
                applied=None,  # no reconstruction product exists in this campaign
            )
        )
    return tuple(outcomes)


# ---------------------------------------------------------------------------
# Safety-error detection (§32) — hard invariants, not mere false negatives
# ---------------------------------------------------------------------------


def _safety_violations_for(
    class_name: str, evidence, decision
) -> Mapping[str, bool]:
    """The five prioritised safety-error flags for one (class, evidence, decision)."""
    truth = {t.fact: t.value for t in sensor_evidence_truth(class_name)}
    inferred = {f.field: f.value for f in evidence.sensor_evidence}
    entry = oracle_entry(class_name)

    persisted_truth = truth["persisted_at_same_sensor_coord"]
    persisted_inferred = inferred["persisted_at_same_sensor_coord"]
    residual_inferred = inferred["site_residual_behaviour"]
    censored_truth = truth["censored_measurement_present"]

    return {
        # sain / structure optique -> persistance capteur YES (interdit): a
        # healthy or optical/sky structure (truth persistence != YES) must never
        # be claimed as a persistent sensor site. Covers NORMAL (healthy) and
        # the optical/flat/dust/star sky structures (incl. an undersampled star
        # core misread as a fixed sensor site).
        "healthy_or_optical_structure_as_sensor_site": (
            persisted_truth != YES and persisted_inferred == YES
        ),
        # transitoire -> défaut persistant (interdit): a transient signal must
        # never be claimed as a persistent sensor defect.
        "transient_as_persistent_defect": (
            truth["transient_only"] == YES and persisted_inferred == YES
        ),
        # censuré -> inférence quantitative (interdit): a censored measurement
        # must never support a quantitative residual state or a promotion.
        "censored_quantitative_inference": (
            censored_truth == YES
            and (
                residual_inferred in (RESIDUAL_SYSTEMATIC_STABLE, RESIDUAL_VARIABLE)
                or decision.action == ACTION_ELIGIBLE
            )
        ),
        # dark non représentatif -> candidat à la reconstruction (interdit): a
        # non-representative calibration must requalify, never promote.
        "non_representative_dark_as_reconstruction_candidate": (
            entry.expected_action == ACTION_REQUALIFY
            and decision.action == ACTION_ELIGIBLE
        ),
        # étoile traversante -> défaut capteur fixe (interdit): a moving star
        # must never be claimed as a fixed sensor defect.
        "star_crossing_as_fixed_sensor_defect": (
            class_name == "STAR_CROSSING_SITE" and persisted_inferred == YES
        ),
    }


# ---------------------------------------------------------------------------
# Fact confusion matrices (§30): truth × inferred × UNDETERMINED, per candidate
# ---------------------------------------------------------------------------


def _fact_matrices_for(
    sites: Tuple[QualifiedSite, ...], evidence_by_site: Mapping[int, object]
) -> Mapping[str, Mapping[str, Mapping[str, int]]]:
    matrices: dict = {fact: {} for fact in SENSOR_EVIDENCE_FACT_NAMES}
    for idx, site in enumerate(sites):
        truth = {t.fact: t.value for t in sensor_evidence_truth(site.class_name)}
        inferred = {
            f.field: f.value for f in evidence_by_site[idx].sensor_evidence
        }
        for fact in SENSOR_EVIDENCE_FACT_NAMES:
            tv = truth[fact]
            iv = inferred[fact]
            cell = matrices[fact].setdefault(tv, {})
            cell[iv] = cell.get(iv, 0) + 1
    return matrices


# ---------------------------------------------------------------------------
# Class breakdown (§31): per-class results, never a single global rate
# ---------------------------------------------------------------------------


def _class_breakdown_for(
    sites: Tuple[QualifiedSite, ...],
    outcomes: Tuple[SiteOutcome, ...],
    evidence_by_site: Mapping[int, object],
) -> Mapping[str, Mapping]:
    by_class: dict = {}
    for idx, site in enumerate(sites):
        name = site.class_name
        outcome = outcomes[idx]
        truth = {t.fact: t.value for t in sensor_evidence_truth(name)}
        inferred = {f.field: f.value for f in evidence_by_site[idx].sensor_evidence}
        rec = by_class.setdefault(
            name,
            {
                "seed_count": 0,
                "expected_action": oracle_entry(name).expected_action,
                "expected_epistemic_state": oracle_entry(name).epistemic_state,
                "action_counts": {},
                "epistemic_counts": {},
                "fact_agreement": {f: {"correct": 0, "wrong": 0} for f in SENSOR_EVIDENCE_FACT_NAMES},
            },
        )
        rec["seed_count"] += 1
        action = outcome.decision.action
        rec["action_counts"][action] = rec["action_counts"].get(action, 0) + 1
        epi = outcome.decision.epistemic_state
        rec["epistemic_counts"][epi] = rec["epistemic_counts"].get(epi, 0) + 1
        for fact in SENSOR_EVIDENCE_FACT_NAMES:
            if truth[fact] == inferred[fact]:
                rec["fact_agreement"][fact]["correct"] += 1
            else:
                rec["fact_agreement"][fact]["wrong"] += 1
    return by_class


# ---------------------------------------------------------------------------
# Policy outcome distribution (§38)
# ---------------------------------------------------------------------------


def _policy_outcomes_for(outcomes: Tuple[SiteOutcome, ...]) -> Mapping[str, int]:
    return dict(Counter(o.decision.action for o in outcomes))


# ---------------------------------------------------------------------------
# Decision table (§38): descriptive, NO winner
# ---------------------------------------------------------------------------


def _decision_row(
    candidate_id: str,
    report,
    safety_counts: Mapping[str, int],
    class_breakdown: Mapping[str, Mapping],
    policy_outcomes: Mapping[str, int],
) -> dict:
    metrics = {}
    for name in HEADLINE_METRIC_NAMES:
        mv = report.metrics[name]
        metrics[name] = {"value": mv.value, "universe": mv.universe, "unit": mv.unit}

    # Per-class failure summary: a target that is not qualified, or a class that
    # carries a safety violation — reported per class, never collapsed.
    per_class = {}
    for cls_name, rec in sorted(class_breakdown.items()):
        expected = rec["expected_action"]
        actions = rec["action_counts"]
        per_class[cls_name] = {
            "expected_action": expected,
            "action_counts": actions,
            "epistemic_counts": rec["epistemic_counts"],
            "fact_wrong": {
                f: rec["fact_agreement"][f]["wrong"] for f in SENSOR_EVIDENCE_FACT_NAMES
            },
            "mask_guard": cls_name in _MUST_NOT_MASK_CLASSES,
        }

    # INVALID is reserved for a §40 hard-invariant violation (§8): only the two
    # *measurable* hard invariants (censored quantitative inference, non-
    # representative calibration promoted to reconstruction). The other three
    # §32 errors are reported but do not eliminate the candidate; the two
    # structural invariants cannot be produced by any candidate.
    hard_violations = {
        name: safety_counts[name] for name in HARD_INVARIANT_MEASURED
    }
    invalid = any(v > 0 for v in hard_violations.values())

    return {
        "candidate_id": candidate_id,
        "metrics": metrics,
        "abstained_target_count": report.abstained_target_count,
        "target_count": report.target_count,
        "negative_count": report.negative_count,
        "evaluation_count": report.evaluation_count,
        "safety_violations": dict(safety_counts),
        "hard_invariant_violations": hard_violations,
        "structural_invariants": {name: "structurally_excluded" for name in HARD_INVARIANT_STRUCTURAL},
        "invalid": invalid,
        "per_class": per_class,
        "policy_outcome_distribution": dict(policy_outcomes),
        "confusion": {
            "true_positive": report.confusion.true_positive,
            "false_positive": report.confusion.false_positive,
            "false_negative": report.confusion.false_negative,
            "true_negative": report.confusion.true_negative,
        },
    }


# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------


def run_qualification_campaign(
    freeze: Optional[CandidateFreeze] = None,
    *,
    scenarios=None,
) -> dict:
    """Run the qualification campaign over the frozen candidates.

    ``freeze`` defaults to a freshly computed freeze (so the campaign is a pure
    function of the current frozen state). ``scenarios`` is an optional subset
    of qualification scenarios (a test knob); the default is the full bounded
    QUALIFICATION set. No holdout is ever involved.
    """
    if freeze is None:
        freeze = freeze_candidates()
    _assert_not_drifted(freeze)

    scenarios = tuple(qualification_scenarios() if scenarios is None else scenarios)
    sites = _materialise_sites(scenarios)
    light_frames = _light_frame_count(scenarios)
    mpf = _megapixels_per_frame(scenarios)

    candidates = []
    fact_matrices_by_candidate = {}
    class_breakdown_by_candidate = {}
    policy_outcomes_by_candidate = {}

    for cfg in CANDIDATES:
        cid = cfg.candidate_id
        outcomes = _outcomes_for(cid, sites)
        # Evidence per site (recomputed from the same frozen candidate rules).
        evidence_by_site = {
            i: infer_site(cfg, site.features, site.admission)
            for i, site in enumerate(sites)
        }

        report = compute_metrics(
            outcomes,
            light_frame_count=light_frames,
            megapixels_per_frame=mpf,
        )

        # Safety violations aggregated over sites.
        safety_counts = {name: 0 for name in SAFETY_ERROR_NAMES}
        for i, site in enumerate(sites):
            flags = _safety_violations_for(
                site.class_name, evidence_by_site[i], outcomes[i].decision
            )
            for name, hit in flags.items():
                if hit:
                    safety_counts[name] += 1

        fm = _fact_matrices_for(sites, evidence_by_site)
        cb = _class_breakdown_for(sites, outcomes, evidence_by_site)
        po = _policy_outcomes_for(outcomes)

        candidates.append(_decision_row(cid, report, safety_counts, cb, po))
        fact_matrices_by_candidate[cid] = fm
        class_breakdown_by_candidate[cid] = cb
        policy_outcomes_by_candidate[cid] = po

    return {
        "campaign": {
            "campaign_schema": CAMPAIGN_SCHEMA,
            "campaign_version": CAMPAIGN_VERSION,
            "freeze_hash": freeze.freeze_hash,
            "metrics_contract_version": freeze.metrics_contract_version,
            "reason_code_version": freeze.reason_code_version,
            "adapter_version": freeze.adapter_version,
            "development_manifest_hash": freeze.development_manifest_hash,
            "qualification_manifest_hash": freeze.qualification_manifest_hash,
            "seeds": list(QUALIFICATION_SEEDS),
            "class_count": len({s.class_name for s in sites}),
            "scenario_count": len(sites),
            "light_frame_count": light_frames,
        },
        "candidates": candidates,
        "fact_confusion_matrices": fact_matrices_by_candidate,
        "class_breakdown": class_breakdown_by_candidate,
        "policy_outcomes": policy_outcomes_by_candidate,
    }


# ---------------------------------------------------------------------------
# Artifact writers (§9 / §43)
# ---------------------------------------------------------------------------


def build_artifacts(results: dict, freeze: CandidateFreeze) -> Mapping[str, dict]:
    """Split the campaign result into the five durable JSON artifacts."""
    return {
        "candidate_freeze.json": freeze.to_dict(),
        "qualification_results.json": {
            "campaign": results["campaign"],
            "candidates": results["candidates"],
        },
        "fact_confusion_matrices.json": {
            "campaign": results["campaign"],
            "fact_confusion_matrices": results["fact_confusion_matrices"],
        },
        "class_breakdown.json": {
            "campaign": results["campaign"],
            "class_breakdown": results["class_breakdown"],
        },
        "policy_outcomes.json": {
            "campaign": results["campaign"],
            "policy_outcomes": results["policy_outcomes"],
        },
    }


def _write_json(path, payload) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return str(path)


def write_artifacts(results: dict, freeze: CandidateFreeze, out_dir) -> Mapping[str, str]:
    """Write the five artifacts under ``out_dir``; return filename → path."""
    artifacts = build_artifacts(results, freeze)
    return {
        name: _write_json(Path(out_dir) / name, payload)
        for name, payload in artifacts.items()
    }


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_VERSION",
    "HEADLINE_METRIC_NAMES",
    "SAFETY_ERROR_NAMES",
    "HARD_INVARIANT_MEASURED",
    "HARD_INVARIANT_STRUCTURAL",
    "CandidateDriftError",
    "QualifiedSite",
    "build_artifacts",
    "run_qualification_campaign",
    "write_artifacts",
]
