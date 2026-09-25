"""P3C-4 LOT 5 — QUALIFICATION-2 campaign runner (measure frozen candidates, HOLDOUT sealed).

Research-only, internal, non-public. This is the **truth-side** campaign harness
for the P3C-4 temporal-persistence campaign. It freezes the temporal candidates
(via :mod:`research.p3c4.candidate_freeze`, GEL-2) and then **measures** them on
the fresh QUALIFICATION-2 corpus (the temporally-faithful 20-class corpus of
:mod:`research.p3c4.corpus`, seeds ``10..14``). The HOLDOUT is sealed
**structurally**, not merely "not done":

* This module never imports ``research.p3b.holdout``, never references
  ``DATASET_HOLDOUT``, and never receives a holdout path/manifest/seed set. There
  is no parameter, argument or constant here that could name the holdout
  (enforced by ``tests/p3c4/test_qualification2_holdout_seal.py``).
* :func:`run_qualification2_campaign` takes a single
  :class:`~research.p3c4.candidate_freeze.CandidateFreeze` (carrying only the
  DEVELOPMENT and QUALIFICATION-2 manifest hashes) and derives the QUALIFICATION-2
  scenarios from :mod:`research.p3c4.qualification2_corpus`. No holdout manifest
  can be passed.

The measurement path reuses the existing, accepted machinery **verbatim** — it
never re-implements an action path or a metric engine (§33–§34):

    infer_site (P3C-4 temporal candidate rule, only ``persisted`` re-derived)
        -> evidence_bridge.to_evidence_packet (P3C adapter, unchanged)
        -> qualification_policy.evaluate (P3B LOT2, unchanged)
        -> metrics.compute_metrics (P3B LOT5, unchanged)

The one *new* measurement this campaign owns is the headline safety metric
``sky_confounder_persistence_false_positive`` (§34): universe = the five §35
sky/optical confounder classes, numerator = confounders inferred
``persisted_at_same_sensor_coord = YES``. It is computed **separately per
confounder class** (§35) and never averaged away, and the five §36 rare /
intermittent classes are reported **separately** to prove the temporal rule did
not make *all* persistence impossible.

Promotion is **not exercised** (§37/§38): net benefit stays out of inference, so
no reconstruction product exists, ``applied`` stays ``None``, and the artifact
carries machine-readable ``promotion_stage_exercised = false`` /
``promotion_safety_measured = false`` — never a bare ``false_promotion = 0``.

Reproducibility (§44): every run records the freeze hash, the source-code SHA,
the candidate config hashes, the dataset manifest hash, the seed set, the
metrics-contract version, the reason-code version, the temporal-contract version
and the adapter version. The campaign is a pure function of the freeze, so the
same freeze yields byte-identical JSON.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from research.p3b.features import compute_features
from research.p3b.metrics import (
    METRIC_ABSTENTION_RATE,
    METRIC_CANDIDATE_RECALL,
    METRIC_CORRECT_ABSTENTION_RATE,
    METRIC_FALSE_CANDIDATE_RATE,
    METRIC_QUALIFICATION_RECALL,
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

from .candidate_freeze import CandidateFreeze, _assert_not_drifted, freeze_candidates
from .corpus import (
    CLASS_TO_KIND,
    CONFOUNDER_KINDS,
    INTERMITTENT_KINDS,
    KIND_TO_CLASS,
    materialise_frames,
)
from .qualification2_corpus import (
    QUALIFICATION2_SEEDS,
    qualification2_scenarios,
)
from .temporal_features import compute_temporal_features
from .temporal_inference import TEMPORAL_CANDIDATES, infer_site

from research.p3c.development_corpus import build_admission
from research.p3c.evidence_bridge import to_evidence_packet
from research.p3c.inference_contract import SENSOR_EVIDENCE_FACTS
from research.p3c.oracle import (
    SENSOR_EVIDENCE_FACT_NAMES,
    oracle_entry,
    sensor_evidence_truth,
)

# Campaign artifacts schema.
CAMPAIGN_SCHEMA = "zecalibrator-p3c4-qualification2-campaign"
CAMPAIGN_VERSION = 1

# The §33 headline metrics — a stable, ordered subset. ``abstained_target_count``
# is a raw counter (carried alongside ``target_count``), not a metric and not a
# score, exactly as in the P3C campaign.
HEADLINE_METRIC_NAMES: Tuple[str, ...] = (
    METRIC_CANDIDATE_RECALL,
    METRIC_QUALIFICATION_RECALL,
    METRIC_FALSE_CANDIDATE_RATE,
    METRIC_ABSTENTION_RATE,
    METRIC_CORRECT_ABSTENTION_RATE,
)

# The five §35 celestial confounder classes (the §34 metric universe), and the
# five §36 rare/intermittent classes, derived from the corpus kind sets — never
# a hand-typed second list.
_CONFOUNDER_CLASSES: Tuple[str, ...] = tuple(
    sorted({KIND_TO_CLASS[k] for k in CONFOUNDER_KINDS})
)
_INTERMITTENT_CLASSES: Tuple[str, ...] = tuple(
    sorted({KIND_TO_CLASS[k] for k in INTERMITTENT_KINDS})
)

# The headline safety metric name (§34) — reported separately per candidate with
# its own universe/unit, never folded into a global rate.
SAFETY_METRIC_SKY_CONFOUNDER = "sky_confounder_persistence_false_positive"

# Targeted safety-error flags (§33), reusing the P3C taxonomy. Two of them
# (censored->quantitative, non-representative dark->reconstruction) are §40 hard
# invariants; the others are priority evidence-level errors reported per
# candidate. The §34 headline is reported separately (see above), not here.
SAFETY_ERROR_NAMES: Tuple[str, ...] = (
    "healthy_or_optical_structure_as_sensor_site",
    "transient_as_persistent_defect",
    "censored_quantitative_inference",
    "non_representative_dark_as_reconstruction_candidate",
    "star_crossing_as_fixed_sensor_defect",
)

# §40 hard invariants — the ONLY grounds for INVALID elimination. The two
# measurable ones map to named safety errors above; the other two (truth leak,
# inter-frame switching) are enforced structurally by the truth-leak guard and
# the unchanged qualification_policy / preparation_plan modules.
HARD_INVARIANT_MEASURED: Tuple[str, ...] = (
    "censored_quantitative_inference",
    "non_representative_dark_as_reconstruction_candidate",
)
HARD_INVARIANT_STRUCTURAL: Tuple[str, ...] = (
    "truth_leak",
    "inter_frame_switching",
)

# Classes whose per-class result must never be masked by a global rate.
_MUST_NOT_MASK_CLASSES: Tuple[str, ...] = (
    "UNDERSAMPLED_STAR_CORE",
    "STAR_CROSSING_SITE",
    "OPTICAL_STRUCTURE",
    "FLAT_STRUCTURE",
    "DUST_OR_VIGNETTING",
    "RARE_HIGH_STATE",
    "RARE_LOW_STATE",
    "INTERMITTENT_TWO_STATE",
    "INTERMITTENT_MULTI_STATE",
    "INTERMITTENT_CONTINUOUS",
    "CENSORED_ANOMALY",
)

_SITE_X = 8
_SITE_Y = 8


@dataclass(frozen=True)
class QualifiedSite:
    """One measured QUALIFICATION-2 site (features + admission + truth keys)."""

    class_name: str
    seed: int
    temporal_kind: str
    features: object  # research.p3b.features.SiteFeatures (duck-typed)
    temporal_features: object  # research.p3c4.temporal_features.TemporalSiteFeatures
    admission: object  # research.p3c.inference_contract.AdmissionFacts
    censored: bool


# ---------------------------------------------------------------------------
# Materialisation (features computed once, shared by every candidate)
# ---------------------------------------------------------------------------


def _materialise_sites(scenarios) -> Tuple[QualifiedSite, ...]:
    """Compute spatial + temporal features + admission for every scenario (once).

    The features are candidate-independent: they are computed from the *same*
    materialised frames for every candidate, so the candidates differ only in
    their rules, never in the measured data they see. The temporal corpus is
    light-only (no dark reference), so the spatial features use the local
    residual as the canonical residual — a property of the corpus, not a choice
    made here.
    """
    sites = []
    for scenario in scenarios:
        class_name = scenario.sites[0].cfa_class
        temporal_kind = CLASS_TO_KIND[class_name]
        site = scenario.sites[0]
        censored = site.censored
        hard = (
            site.hard_limit_adu
            if site.hard_limit_adu is not None
            else scenario.sensor.saturation_limit_adu
        )
        frames = materialise_frames(scenario, temporal_kind, seed=scenario.seed)
        cf = compute_features(scenario, frames)
        tf = compute_temporal_features(
            frames,
            scenario.light_frames(),
            site.x,
            site.y,
            cfa_pattern=scenario.sensor.cfa_pattern,
            hard_limit=hard,
        )
        sites.append(
            QualifiedSite(
                class_name=class_name,
                seed=int(scenario.seed),
                temporal_kind=temporal_kind,
                features=cf.sites[0],
                temporal_features=tf,
                admission=build_admission(scenario),
                censored=censored,
            )
        )
    return tuple(sites)


def _light_frame_count(scenarios) -> int:
    """Total light (science) frames across the QUALIFICATION-2 set."""
    return sum(len(s.light_frames()) for s in scenarios)


def _megapixels_per_frame(scenarios) -> float:
    ny, nx = scenarios[0].sensor.shape
    return (ny * nx) / PIXELS_PER_MEGAPIXEL


# ---------------------------------------------------------------------------
# Per-site inference -> policy -> outcome (verbatim reuse, never reimplemented)
# ---------------------------------------------------------------------------


def _outcomes_for(
    candidate_id: str, sites: Tuple[QualifiedSite, ...]
) -> Tuple[SiteOutcome, ...]:
    from research.p3c4.temporal_inference import temporal_candidate

    cfg = temporal_candidate(candidate_id)
    outcomes = []
    for site in sites:
        entry = oracle_entry(site.class_name)
        evidence = infer_site(cfg, site.features, site.temporal_features, site.admission)
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
# Safety-error detection (§33) — hard invariants, not mere false negatives
# ---------------------------------------------------------------------------


def _safety_violations_for(
    class_name: str, evidence, decision
) -> Mapping[str, bool]:
    """The five targeted safety-error flags for one (class, evidence, decision)."""
    truth = {t.fact: t.value for t in sensor_evidence_truth(class_name)}
    inferred = {f.field: f.value for f in evidence.sensor_evidence}
    entry = oracle_entry(class_name)

    persisted_truth = truth["persisted_at_same_sensor_coord"]
    persisted_inferred = inferred["persisted_at_same_sensor_coord"]
    residual_inferred = inferred["site_residual_behaviour"]
    censored_truth = truth["censored_measurement_present"]

    return {
        "healthy_or_optical_structure_as_sensor_site": (
            persisted_truth != YES and persisted_inferred == YES
        ),
        "transient_as_persistent_defect": (
            truth["transient_only"] == YES and persisted_inferred == YES
        ),
        "censored_quantitative_inference": (
            censored_truth == YES
            and (
                residual_inferred in (RESIDUAL_SYSTEMATIC_STABLE, RESIDUAL_VARIABLE)
                or decision.action == ACTION_ELIGIBLE
            )
        ),
        "non_representative_dark_as_reconstruction_candidate": (
            entry.expected_action == ACTION_REQUALIFY
            and decision.action == ACTION_ELIGIBLE
        ),
        "star_crossing_as_fixed_sensor_defect": (
            class_name == "STAR_CROSSING_SITE" and persisted_inferred == YES
        ),
    }


# ---------------------------------------------------------------------------
# §34 headline safety metric + §35 decomposition (never averaged)
# ---------------------------------------------------------------------------


def _sky_confounder_metric(
    sites: Tuple[QualifiedSite, ...], evidence_by_site: Mapping[int, object]
) -> dict:
    """The §34 ``sky_confounder_persistence_false_positive`` metric.

    Universe = the five §35 sky/optical confounder classes (25 sites = 5 classes
    x 5 seeds). Numerator = confounders inferred ``persisted_at_same_sensor_coord
    = YES``. Reported as a fraction with explicit numerator/denominator and a
    per-class decomposition (§35) — never a bare average.
    """
    confounders = [
        (i, site) for i, site in enumerate(sites) if site.class_name in _CONFOUNDER_CLASSES
    ]
    num = 0
    decomposition = {cls: 0 for cls in _CONFOUNDER_CLASSES}
    for i, site in confounders:
        inferred = {
            f.field: f.value for f in evidence_by_site[i].sensor_evidence
        }
        if inferred["persisted_at_same_sensor_coord"] == YES:
            num += 1
            decomposition[site.class_name] += 1
    den = len(confounders)
    return {
        "metric": SAFETY_METRIC_SKY_CONFOUNDER,
        "value": (num / den) if den else None,
        "numerator": num,
        "denominator": den,
        "universe": "sky/optical confounder site (§35)",
        "unit": "per confounder site (fraction)",
        "decomposition_by_class": decomposition,
    }


def _intermittent_control(
    sites: Tuple[QualifiedSite, ...], evidence_by_site: Mapping[int, object]
) -> Mapping[str, Mapping[str, int]]:
    """§36 — rare/intermittent classes, separately, per inferred persisted value.

    The goal is to prove the temporal rule did **not** close the stellar
    persistence error by making *all* persistence impossible: each intermittent
    class must keep a useful temporal signal (YES, or at least not be wiped to
    NO). Reported as a per-class distribution of the inferred
    ``persisted_at_same_sensor_coord`` value — never a single global rate.
    """
    out: dict = {}
    for cls in _INTERMITTENT_CLASSES:
        dist = {"YES": 0, "NO": 0, "UNDETERMINED": 0}
        for i, site in enumerate(sites):
            if site.class_name != cls:
                continue
            inferred = {
                f.field: f.value for f in evidence_by_site[i].sensor_evidence
            }
            value = inferred["persisted_at_same_sensor_coord"]
            dist[value] = dist.get(value, 0) + 1
        out[cls] = dist
    return out


# ---------------------------------------------------------------------------
# Fact confusion matrices (§39): truth × inferred × UNDETERMINED, per candidate
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
    sky_metric: dict,
    intermittent_control: Mapping[str, Mapping[str, int]],
) -> dict:
    metrics = {}
    for name in HEADLINE_METRIC_NAMES:
        mv = report.metrics[name]
        metrics[name] = {"value": mv.value, "universe": mv.universe, "unit": mv.unit}

    per_class = {}
    for cls_name, rec in sorted(class_breakdown.items()):
        per_class[cls_name] = {
            "expected_action": rec["expected_action"],
            "action_counts": rec["action_counts"],
            "epistemic_counts": rec["epistemic_counts"],
            "fact_wrong": {
                f: rec["fact_agreement"][f]["wrong"] for f in SENSOR_EVIDENCE_FACT_NAMES
            },
            "mask_guard": cls_name in _MUST_NOT_MASK_CLASSES,
        }

    hard_violations = {
        name: safety_counts[name] for name in HARD_INVARIANT_MEASURED
    }
    invalid = any(v > 0 for v in hard_violations.values())

    # §37/§38 — promotion is not exercised: net benefit stays out of inference,
    # no reconstruction product exists. The false-promotion value is therefore
    # reported WITH the machine-readable flags, never bare.
    false_promotion_value = report.metrics["site_level_false_promotion_rate"].value

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
        "promotion": {
            "promotion_stage_exercised": False,
            "promotion_safety_measured": False,
            "site_level_false_promotion_rate": {
                "value": false_promotion_value,
                "status": "NOT_EXERCISED",
                "note": (
                    "no reconstruction product exists and net benefit is never "
                    "established; a bare 0 is not presented as a measured safety "
                    "figure"
                ),
            },
        },
        "sky_confounder_persistence_false_positive": sky_metric,
        "intermittent_control": intermittent_control,
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
# §40 minimal criterion (verified from the measured results, never declared)
# ---------------------------------------------------------------------------


def _section_40_status(results: dict) -> Mapping[str, object]:
    """Verify and report the §40 minimal criterion from measured results.

    Reported, not declared: the two persistence-invariant classes must show
    zero ``persisted=YES`` on QUALIFICATION-2 across every candidate; the two
    measurable hard invariants must be zero; the two structural invariants are
    excluded by construction. This is a *verification*, never a recommendation
    to open the holdout — the holdout stays sealed.
    """
    sky_yes = {}
    star_yes = {}
    for cand in results["candidates"]:
        cid = cand["candidate_id"]
        decomp = cand["sky_confounder_persistence_false_positive"]["decomposition_by_class"]
        sky_yes[cid] = decomp.get("UNDERSAMPLED_STAR_CORE", 0)
        star_yes[cid] = decomp.get("STAR_CROSSING_SITE", 0)
        for name in HARD_INVARIANT_MEASURED:
            if cand["hard_invariant_violations"][name] != 0:
                raise RuntimeError(
                    f"{cid} violates §40 hard invariant {name!r}; the criterion "
                    "is NOT satisfied"
                )
    return {
        "persistence_yes_zero_for_undersampled_star_core": all(v == 0 for v in sky_yes.values()),
        "persistence_yes_zero_for_star_crossing_site": all(v == 0 for v in star_yes.values()),
        "per_candidate_undersampled_star_core_yes": sky_yes,
        "per_candidate_star_crossing_site_yes": star_yes,
        "censored_quantitative_inference_zero": True,
        "non_representative_dark_as_reconstruction_candidate_zero": True,
        "truth_leak_structural": True,
        "inter_frame_switching_structural": True,
        "sealed": True,
        "opening_recommendation": "NONE",
    }


# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------


def run_qualification2_campaign(
    freeze: Optional[CandidateFreeze] = None,
    *,
    scenarios=None,
) -> dict:
    """Run the QUALIFICATION-2 campaign over the frozen temporal candidates.

    ``freeze`` defaults to a freshly computed freeze (so the campaign is a pure
    function of the current frozen state). ``scenarios`` is an optional subset
    of QUALIFICATION-2 scenarios (a test knob); the default is the full bounded
    QUALIFICATION-2 set. No holdout is ever involved.
    """
    if freeze is None:
        freeze = freeze_candidates()
    _assert_not_drifted(freeze)

    scenarios = tuple(qualification2_scenarios() if scenarios is None else scenarios)
    sites = _materialise_sites(scenarios)
    light_frames = _light_frame_count(scenarios)
    mpf = _megapixels_per_frame(scenarios)

    candidates = []
    fact_matrices_by_candidate = {}
    class_breakdown_by_candidate = {}
    policy_outcomes_by_candidate = {}

    for cfg in TEMPORAL_CANDIDATES:
        cid = cfg.candidate_id
        outcomes = _outcomes_for(cid, sites)
        evidence_by_site = {
            i: infer_site(cfg, site.features, site.temporal_features, site.admission)
            for i, site in enumerate(sites)
        }

        report = compute_metrics(
            outcomes,
            light_frame_count=light_frames,
            megapixels_per_frame=mpf,
        )

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
        sky_metric = _sky_confounder_metric(sites, evidence_by_site)
        intermittent = _intermittent_control(sites, evidence_by_site)

        candidates.append(
            _decision_row(cid, report, safety_counts, cb, po, sky_metric, intermittent)
        )
        fact_matrices_by_candidate[cid] = fm
        class_breakdown_by_candidate[cid] = cb
        policy_outcomes_by_candidate[cid] = po

    results = {
        "campaign": {
            "campaign_schema": CAMPAIGN_SCHEMA,
            "campaign_version": CAMPAIGN_VERSION,
            "freeze_hash": freeze.freeze_hash,
            "metrics_contract_version": freeze.metrics_contract_version,
            "reason_code_version": freeze.reason_code_version,
            "adapter_version": freeze.adapter_version,
            "temporal_contract_version": freeze.temporal_contract_version,
            "development_manifest_hash": freeze.development_manifest_hash,
            "qualification2_manifest_hash": freeze.qualification2_manifest_hash,
            "seeds": list(QUALIFICATION2_SEEDS),
            "class_count": len({s.class_name for s in sites}),
            "scenario_count": len(sites),
            "light_frame_count": light_frames,
            "verified_assertions": _verified_assertions(),
        },
        "candidates": candidates,
        "fact_confusion_matrices": fact_matrices_by_candidate,
        "class_breakdown": class_breakdown_by_candidate,
        "policy_outcomes": policy_outcomes_by_candidate,
    }
    results["campaign"]["section_40_status"] = _section_40_status(results)
    return results


def _verified_assertions() -> Mapping[str, bool]:
    """The four §54 assertions, verified (not declared), from the QUALIFICATION-2
    manifest builder — which computes them from actual signature sets."""
    from .qualification2_corpus import build_qualification2_manifest

    return build_qualification2_manifest()["verified_assertions"]


# ---------------------------------------------------------------------------
# Artifact writers (§52 / §43)
# ---------------------------------------------------------------------------


def build_artifacts(results: dict, freeze: CandidateFreeze) -> Mapping[str, dict]:
    """Split the campaign result into the durable JSON artifacts."""
    return {
        "P3C4-CANDIDATE-FREEZE.json": freeze.to_dict(),
        "qualification2_results.json": {
            "campaign": results["campaign"],
            "candidates": results["candidates"],
        },
        "temporal_fact_confusion.json": {
            "campaign": results["campaign"],
            "fact_confusion_matrices": results["fact_confusion_matrices"],
        },
        "class_breakdown.json": {
            "campaign": results["campaign"],
            "class_breakdown": results["class_breakdown"],
        },
    }


def _write_json(path, payload) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return str(path)


def write_artifacts(results: dict, freeze: CandidateFreeze, out_dir) -> Mapping[str, str]:
    """Write the four campaign artifacts under ``out_dir``; return filename → path."""
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
    "SAFETY_METRIC_SKY_CONFOUNDER",
    "HARD_INVARIANT_MEASURED",
    "HARD_INVARIANT_STRUCTURAL",
    "QualifiedSite",
    "build_artifacts",
    "run_qualification2_campaign",
    "write_artifacts",
]
