"""P3C-4 — corrected conflicting-evidence measurement harness (diagnosis closure).

Research-only, internal, non-public. This is the **truth-side** measurement
harness for the corrected ``conflicting_evidence`` semantics. It measures the
new ``p3c4-*-cefix`` candidates on the frozen QUALIFICATION-2 corpus (seeds
``10..14``) and reports, per candidate:

* the ``conflicting_evidence`` fact confusion matrix (truth × inferred) — the
  cell that the frozen P3C campaign left at 65 ``YES``-where-``NO``;
* the per-class ``conflicting_evidence`` agreement;
* the headline ``qualification_recall`` (to show the instrumentation artifact is
  removed — never to maximise anything);
* the gate §19 rows.

It reuses the existing, accepted machinery verbatim — it never re-implements an
action path or a metric engine:

    infer_site (corrected cefix candidate rule)
        -> evidence_bridge.to_evidence_packet (P3C adapter, unchanged)
        -> qualification_policy.evaluate (P3B LOT2, unchanged)
        -> metrics.compute_metrics (P3B LOT5, unchanged)

The HOLDOUT stays sealed: this module never imports ``research.p3b.holdout``,
never references ``DATASET_HOLDOUT`` and receives no holdout path/manifest/seed.
It derives the QUALIFICATION-2 scenarios from
:mod:`research.p3c4.qualification2_corpus` only.

This module is the truth side (like ``qualification2_runner``): it may read class
names and declared structure. It is deliberately **not** re-exported by
``research.p3c4/__init__.py`` so the corrected *contract* package still pulls no
truth into ``sys.modules``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from research.p3b.features import compute_features
from research.p3b.metrics import (
    METRIC_QUALIFICATION_RECALL,
    PIXELS_PER_MEGAPIXEL,
    SiteOutcome,
    compute_metrics,
)
from research.p3b.qualification_policy import evaluate
from research.p3c.development_corpus import build_admission
from research.p3c.evidence_bridge import to_evidence_packet
from research.p3c.inference_contract import SENSOR_EVIDENCE_FACTS
from research.p3c.oracle import (
    SENSOR_EVIDENCE_FACT_NAMES,
    oracle_entry,
    sensor_evidence_truth,
)

from .conflicting_inference import CONFLICTING_CANDIDATES, infer_site
from .corpus import CLASS_TO_KIND, materialise_frames
from .qualification2_corpus import QUALIFICATION2_SEEDS, qualification2_scenarios
from .temporal_features import compute_temporal_features

CAMPAIGN_SCHEMA = "zecalibrator-p3c4-conflicting-evidence-closure"
CAMPAIGN_VERSION = 1

_SITE_X = 8
_SITE_Y = 8


@dataclass(frozen=True)
class ClosureSite:
    """One measured QUALIFICATION-2 site (features + admission + truth key)."""

    class_name: str
    seed: int
    temporal_kind: str
    features: object
    temporal_features: object
    admission: object
    censored: bool


def _materialise_sites(scenarios) -> Tuple[ClosureSite, ...]:
    """Compute spatial + temporal features + admission for every scenario (once)."""
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
            ClosureSite(
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


def _fact_confusion_matrix(
    sites: Tuple[ClosureSite, ...], evidence_by_site: Mapping[int, object]
) -> Mapping[str, Mapping[str, Mapping[str, int]]]:
    """truth × inferred × UNDETERMINED, for every SENSOR-EVIDENCE fact."""
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


def _conflict_agreement_by_class(
    sites: Tuple[ClosureSite, ...], evidence_by_site: Mapping[int, object]
) -> Mapping[str, Mapping[str, object]]:
    out: dict = {}
    for idx, site in enumerate(sites):
        truth = {t.fact: t.value for t in sensor_evidence_truth(site.class_name)}
        inferred = {
            f.field: f.value for f in evidence_by_site[idx].sensor_evidence
        }
        rec = out.setdefault(
            site.class_name,
            {"truth": truth["conflicting_evidence"], "inferred": {}, "seeds": 0},
        )
        rec["seeds"] += 1
        iv = inferred["conflicting_evidence"]
        rec["inferred"][iv] = rec["inferred"].get(iv, 0) + 1
    return out


def run_conflicting_campaign(scenarios=None) -> dict:
    """Measure the corrected ``p3c4-*-cefix`` candidates on QUALIFICATION-2.

    ``scenarios`` is an optional subset (a test knob); the default is the full
    bounded QUALIFICATION-2 set. No holdout is ever involved.
    """
    scenarios = tuple(qualification2_scenarios() if scenarios is None else scenarios)
    sites = _materialise_sites(scenarios)
    light_frames = sum(len(s.light_frames()) for s in scenarios)
    ny, nx = scenarios[0].sensor.shape
    mpf = (ny * nx) / PIXELS_PER_MEGAPIXEL

    candidates = []
    matrices_by_candidate = {}
    conflict_agreement_by_candidate = {}

    for cfg in CONFLICTING_CANDIDATES:
        cid = cfg.candidate_id
        outcomes = []
        evidence_by_site = {}
        for i, site in enumerate(sites):
            entry = oracle_entry(site.class_name)
            evidence = infer_site(
                cfg, site.features, site.temporal_features, site.admission
            )
            packet = to_evidence_packet(evidence)
            decision = evaluate(packet)
            evidence_by_site[i] = evidence
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
                    applied=None,
                )
            )

        report = compute_metrics(
            tuple(outcomes),
            light_frame_count=light_frames,
            megapixels_per_frame=mpf,
        )

        candidates.append(
            {
                "candidate_id": cid,
                "base_temporal_candidate_id": cfg.base_temporal_candidate_id,
                "version": cfg.version,
                "qualification_recall": report.metrics[
                    METRIC_QUALIFICATION_RECALL
                ].value,
                "target_count": report.target_count,
                "abstained_target_count": report.abstained_target_count,
                "confusion": {
                    "true_positive": report.confusion.true_positive,
                    "false_positive": report.confusion.false_positive,
                    "false_negative": report.confusion.false_negative,
                    "true_negative": report.confusion.true_negative,
                },
                "policy_outcome_distribution": dict(
                    Counter(o.decision.action for o in outcomes)
                ),
            }
        )
        matrices_by_candidate[cid] = _fact_confusion_matrix(sites, evidence_by_site)
        conflict_agreement_by_candidate[cid] = _conflict_agreement_by_class(
            sites, evidence_by_site
        )

    return {
        "campaign": {
            "campaign_schema": CAMPAIGN_SCHEMA,
            "campaign_version": CAMPAIGN_VERSION,
            "seeds": list(QUALIFICATION2_SEEDS),
            "scenario_count": len(sites),
            "light_frame_count": light_frames,
        },
        "candidates": candidates,
        "fact_confusion_matrices": matrices_by_candidate,
        "conflicting_evidence_by_class": conflict_agreement_by_candidate,
    }


def write_artifacts(results: dict, out_dir) -> Mapping[str, str]:
    """Write the closure artifacts under ``out_dir``; return filename → path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, payload in {
        "conflicting_evidence_closure.json": results,
    }.items():
        p = Path(out_dir) / name
        p.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        paths[name] = str(p)
    return paths


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_VERSION",
    "ClosureSite",
    "run_conflicting_campaign",
    "write_artifacts",
]
