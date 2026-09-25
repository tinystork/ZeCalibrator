"""P3C-4 LOT 5 — QUALIFICATION-1 regression (KNOWN_DATA_REGRESSION_ONLY).

Research-only, internal, non-public. This module replays the **consumed**
QUALIFICATION-1 dataset (the P3B original qualification corpus, seeds ``5..9``)
through the new temporal candidates — **only** as a regression on known data,
never as validation. It is marked ``KNOWN_DATA_REGRESSION_ONLY = true``
(machine-readable, §27 / §53).

The regression answers exactly one question: *"is the bug we already saw
closed?"* It never answers *"does the fix generalise?"* — that question belongs
to QUALIFICATION-2, which is a fresh, out-of-sample measurement.

The subtlety (§27, §53) that must be stated honestly
-----------------------------------------------------

QUALIFICATION-1 rests on the **P3B original corpus**, in which
``UNDERSAMPLED_STAR_CORE`` is rendered at a **fixed sensor coordinate** — that
fixed rendering is the fidelity defect itself (SCIENCE §44). On *that* corpus a
temporal proof will legitimately conclude persistence: the star core's residual
signature recurs at the same coordinate across groups and epochs, so
``persisted_at_same_sensor_coord = YES`` is the **correct** answer **for those
non-physical data**. The defect was the **data**, not the rule.

Therefore this replay:

* **states which corpus it uses** — the P3B original qualification corpus
  (``research.p3c.qualification_corpus``, seeds ``5..9``);
* **does not claim the bug is closed** on the strength of ``YES``: it reports
  ``YES`` and explains that it is correct for non-physical (fixed-coordinate)
  data;
* points to **QUALIFICATION-2** (the temporally-faithful corpus) as the closure
  proof — where the same seeds on a faithful corpus would show ``NO``, and which
  is reported by :mod:`research.p3c4.qualification2_runner`, never here.

The HOLDOUT is never opened, built, imported or referenced here: this module
receives no holdout path, no holdout manifest and no holdout seed set (enforced
by ``tests/p3c4/test_qualification2_holdout_seal.py``).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from research.p3b.features import compute_features
from research.p3b.generator import generate_corpus

from research.p3c.development_corpus import build_admission
from research.p3c.oracle import sensor_evidence_truth
from research.p3c.qualification_corpus import (
    QUALIFICATION_SEEDS,
    qualification_scenarios,
)

from .temporal_features import compute_temporal_features
from .temporal_inference import TEMPORAL_CANDIDATES, infer_persisted

REGRESSION_SCHEMA = "zecalibrator-p3c4-qualification1-regression"
REGRESSION_VERSION = 1

# §27 / §53 — the regression is on known data only, never a fresh validation.
KNOWN_DATA_REGRESSION_ONLY = True

_DARK_FRAMES = 4


@dataclass(frozen=True)
class _RegressionSite:
    class_name: str
    seed: int
    temporal_features: object
    admission: object


def _materialise() -> Tuple[_RegressionSite, ...]:
    """Materialise the P3B original QUALIFICATION-1 corpus (seeds 5..9)."""
    out = []
    for scenario in qualification_scenarios():
        class_name = scenario.sites[0].cfa_class
        site = scenario.sites[0]
        hard = (
            site.hard_limit_adu
            if site.hard_limit_adu is not None
            else scenario.sensor.saturation_limit_adu
        )
        with tempfile.TemporaryDirectory() as td:
            corpus = generate_corpus(scenario, td)
            dark = np.median(
                np.stack(
                    [corpus.frame_arrays[f"dark{i}"] for i in range(_DARK_FRAMES)]
                ),
                axis=0,
            )
            # Spatial features use the representative dark (as in the P3C campaign);
            # the temporal proof consumes the temporal features below.
            compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
            tf = compute_temporal_features(
                corpus.frame_arrays,
                scenario.light_frames(),
                site.x,
                site.y,
                cfa_pattern=scenario.sensor.cfa_pattern,
                hard_limit=hard,
            )
        out.append(
            _RegressionSite(
                class_name=class_name,
                seed=int(scenario.seed),
                temporal_features=tf,
                admission=build_admission(scenario),
            )
        )
    return tuple(out)


def _persisted_truth_by_class(sites: Tuple[_RegressionSite, ...]) -> Mapping[str, str]:
    out: dict = {}
    for site in sites:
        truth = {t.fact: t.value for t in sensor_evidence_truth(site.class_name)}
        out[site.class_name] = truth["persisted_at_same_sensor_coord"]
    return out


def _persisted_distribution_by_class(
    candidate_id: str, sites: Tuple[_RegressionSite, ...]
) -> Mapping[str, Mapping[str, int]]:
    """Per-class distribution of inferred persisted values across the 5 seeds.

    Reported as a distribution (not a single value) because some classes are
    legitimately seed-dependent (e.g. a sign-flipping residual whose group median
    lands on either side of the presence bar depending on the seed's read noise).
    A seed-dependent persistence is made visible, never hidden behind a single
    mode and never raised as a failure — it is data, not an error.
    """
    from .temporal_inference import temporal_candidate

    cfg = temporal_candidate(candidate_id)
    out: dict = {}
    for site in sites:
        value = infer_persisted(cfg, site.temporal_features).value
        dist = out.setdefault(site.class_name, {"YES": 0, "NO": 0, "UNDETERMINED": 0})
        dist[value] = dist.get(value, 0) + 1
    return out


def run_qualification1_regression(
    freeze: Optional[object] = None,
) -> dict:
    """Replay QUALIFICATION-1 (P3B corpus) through the temporal candidates.

    ``freeze`` is accepted for symmetry with the QUALIFICATION-2 runner and is
    ignored here: the regression is a *replay on known data*, not a measurement
    of the frozen campaign. The result carries ``known_data_regression_only =
    true`` and the honest corpus note.

    Reproducible: same inputs => byte-identical JSON.
    """
    sites = _materialise()
    truth = _persisted_truth_by_class(sites)

    candidates = []
    for cfg in TEMPORAL_CANDIDATES:
        candidates.append(
            {
                "candidate_id": cfg.candidate_id,
                "persisted_distribution_by_class": _persisted_distribution_by_class(
                    cfg.candidate_id, sites
                ),
            }
        )

    return {
        "regression_schema": REGRESSION_SCHEMA,
        "regression_version": REGRESSION_VERSION,
        "known_data_regression_only": KNOWN_DATA_REGRESSION_ONLY,
        "role": "KNOWN_DATA_REGRESSION_ONLY",
        "corpus": (
            "P3B original QUALIFICATION-1 corpus "
            "(research.p3c.qualification_corpus, seeds 5..9)"
        ),
        "seeds": list(QUALIFICATION_SEEDS),
        "corpus_fidelity_note": (
            "UNDERSAMPLED_STAR_CORE is rendered at a FIXED sensor coordinate in this "
            "corpus — that fixed rendering is the fidelity defect itself (SCIENCE "
            "§44). A temporal proof therefore concludes persisted=YES here, which "
            "is CORRECT for these non-physical data; the defect was the data, not "
            "the rule."
        ),
        "persisted_truth_by_class": truth,
        "candidates": candidates,
        "headline": {
            "undersampled_star_core_inferred_persisted": candidates[0][
                "persisted_distribution_by_class"
            ].get("UNDERSAMPLED_STAR_CORE"),
            "reading": (
                "YES is the correct answer for the P3B corpus (fixed-coordinate "
                "non-physical rendering). This replay does NOT claim the bug is "
                "closed: the closure proof comes from QUALIFICATION-2 on the "
                "temporally-faithful corpus."
            ),
        },
    }


__all__ = [
    "REGRESSION_SCHEMA",
    "REGRESSION_VERSION",
    "KNOWN_DATA_REGRESSION_ONLY",
    "run_qualification1_regression",
]
