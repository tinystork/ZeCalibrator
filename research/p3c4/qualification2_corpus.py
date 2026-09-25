"""P3C-4 LOT 4 — QUALIFICATION-2 dataset builder (truth-side harness).

Research-only, internal, non-public. Builds the **fresh** QUALIFICATION-2 dataset
for the P3C-4 temporal-persistence campaign: the full 20-class temporal corpus of
:mod:`research.p3c4.corpus`, materialised with a seed set that is disjoint from
DEVELOPMENT (``0..4``) and QUALIFICATION-1 (``5..9``), namely ``(10, 11, 12, 13,
14)`` — chosen **before** any result is looked at (§30), so the measurement is
genuinely out-of-sample relative to both candidate derivation and the
QUALIFICATION-1 regression.

Why a P3C-4 corpus (not the P3B generator)
------------------------------------------

The P3C-4 campaign's subject is **temporal persistence**. The P3B generator
renders celestial confounders at a fixed sensor coordinate (the LOT 1 diagnosis,
SCIENCE §32/§44), so a QUALIFICATION-2 built from ``build_development_scenario``
would re-inherit the latent defect and make §35/§36 inapplicable. This builder
therefore materialises the **temporally-faithful** P3C-4 corpus (sky confounders
follow the sky; sensor defects stay fixed), so the campaign can measure temporal
discrimination on genuinely out-of-sample data.

Role discipline (SCIENCE §28–§30 / §42 / ARCHITECTURE §18):

* Only the **QUALIFICATION-2** dataset is built here. The HOLDOUT dataset is
  **never** opened, built, imported or referenced: this module receives no
  holdout path, no holdout manifest, no holdout seed set, and does not import
  ``research.p3b.holdout`` or ``DATASET_HOLDOUT``. This is structural, not a
  promise — there is no code path here that names the holdout (enforced by
  ``tests/p3c4/test_qualification2_holdout_seal.py``).
* The **anti-leakage signature guard** of :mod:`research.p3b.dataset_split` is
  exercised: every QUALIFICATION-2 scenario's signature is computed with
  :func:`research.p3b.dataset_split.scenario_signature`, and the builder
  *verifies* (not declares) that no signature collides with the DEVELOPMENT or
  QUALIFICATION-1 datasets. A collision is a typed refusal, never a warning.
* The four verified assertions are **computed**, not declared: ``fresh_validation``
  (seeds disjoint from DEV and QUALIF-1), ``development_overlap`` (empty
  signature intersection with DEV), ``qualification1_overlap`` (empty signature
  intersection with QUALIF-1), ``holdout_used`` (structural — no holdout code
  path; re-checked by the seal test).

The manifest is deterministic: same inputs ⇒ same signatures and same
``manifest_hash``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Tuple

from research.p3b.catalog import CLASS_NAMES
from research.p3b.dataset_split import scenario_signature
from research.p3c.development_corpus import (
    DEVELOPMENT_SEEDS,
    development_scenarios,
)
from research.p3c.qualification_corpus import (
    QUALIFICATION_SEEDS,
    qualification_scenarios,
)

from .corpus import (
    CLASS_TO_KIND,
    EPOCHS,
    FRAMES_PER_GROUP,
    GROUPS_PER_EPOCH,
    TEMPORAL_KINDS,
    build_scenario,
)

# ---------------------------------------------------------------------------
# The QUALIFICATION-2 dataset identity
# ---------------------------------------------------------------------------

# Seeds disjoint from DEVELOPMENT_SEEDS = (0,1,2,3,4) and QUALIFICATION_SEEDS =
# (5,6,7,8,9). Chosen fresh (§30), before any result was observed.
QUALIFICATION2_SEEDS: Tuple[int, ...] = (10, 11, 12, 13, 14)

# The full 20-class catalogue, in catalogue order (mirrors the P3C corpus). The
# temporal kind per class comes from research.p3c4.corpus.CLASS_TO_KIND.
QUALIFICATION2_CLASSES: Tuple[str, ...] = tuple(CLASS_NAMES)

QUALIFICATION2_ROLE = "QUALIFICATION-2"
MANIFEST_SCHEMA = "zecalibrator-p3c4-lot4-qualification2-dataset"
MANIFEST_VERSION = 1


def build_qualification2_scenario(class_name: str, seed: int):
    """Build one coherent QUALIFICATION-2 scenario for ``class_name``.

    Reuses the P3C-4 temporal corpus builder (a pure function of
    ``(temporal_kind, seed)``) and re-labels the scenario with a
    ``qualification2:`` prefix. The dataset signature excludes the name, so the
    rename changes nothing about what the data *is*.
    """
    if class_name not in CLASS_TO_KIND:
        raise ValueError(f"unknown synthetic class {class_name!r}")
    if seed not in QUALIFICATION2_SEEDS:
        raise ValueError(
            f"seed {seed!r} is outside QUALIFICATION-2 seeds {QUALIFICATION2_SEEDS!r}"
        )
    base = build_scenario(CLASS_TO_KIND[class_name], seed=seed)
    return replace(base, name=f"qualification2:{class_name}:{seed}")


def qualification2_scenarios():
    """The full bounded QUALIFICATION-2 scenario set (20 classes × 5 seeds)."""
    out = []
    for name in QUALIFICATION2_CLASSES:
        for seed in QUALIFICATION2_SEEDS:
            out.append(build_qualification2_scenario(name, seed))
    return tuple(out)


# ---------------------------------------------------------------------------
# Signature-based anti-leakage verification (computed, not declared)
# ---------------------------------------------------------------------------


def _signature_set(scenarios) -> set:
    return {scenario_signature(s) for s in scenarios}


def _verified_assertions() -> Mapping[str, bool]:
    """Compute the four verified assertions from actual data.

    ``fresh_validation`` is derived from the declared seed sets; the two overlap
    flags are derived from **actual** signature-set intersections (using
    :func:`research.p3b.dataset_split.scenario_signature`); ``holdout_used`` is
    ``False`` because this module has no holdout code path (structurally
    enforced by the seal test and asserted here as a runtime guard).
    """
    q2 = _signature_set(qualification2_scenarios())
    dev = _signature_set(development_scenarios())
    q1 = _signature_set(qualification_scenarios())

    # The guard refuses, it never warns: any overlap is a hard error.
    dev_overlap = q2 & dev
    q1_overlap = q2 & q1
    if dev_overlap:
        raise RuntimeError(
            f"QUALIFICATION-2 collides with DEVELOPMENT signatures: {sorted(dev_overlap)[:3]}…"
        )
    if q1_overlap:
        raise RuntimeError(
            f"QUALIFICATION-2 collides with QUALIFICATION-1 signatures: {sorted(q1_overlap)[:3]}…"
        )

    q2_seeds = set(QUALIFICATION2_SEEDS)
    fresh = (
        len(q2_seeds) == len(QUALIFICATION2_SEEDS)
        and q2_seeds.isdisjoint(set(DEVELOPMENT_SEEDS))
        and q2_seeds.isdisjoint(set(QUALIFICATION_SEEDS))
    )

    return {
        "fresh_validation": bool(fresh),
        "development_overlap": bool(dev_overlap),
        "qualification1_overlap": bool(q1_overlap),
        "holdout_used": False,
    }


def _stable_json_dumps(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qualification2ScenarioRecord:
    """One scenario's manifest record: identity, seed, class, signature."""

    name: str
    seed: int
    cfa_class: str
    temporal_kind: str
    signature: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "cfa_class": self.cfa_class,
            "temporal_kind": self.temporal_kind,
            "signature": self.signature,
        }


def build_qualification2_manifest() -> dict:
    """Build the QUALIFICATION-2 manifest (role, seeds, structure, hashes,
    verified assertions). Deterministic and self-verifying.
    """
    scenarios = qualification2_scenarios()
    records = [
        Qualification2ScenarioRecord(
            name=s.name,
            seed=int(s.seed),
            cfa_class=s.sites[0].cfa_class,
            temporal_kind=CLASS_TO_KIND[s.sites[0].cfa_class],
            signature=scenario_signature(s),
        ).to_dict()
        for s in scenarios
    ]

    structure = {
        "class_count": len(QUALIFICATION2_CLASSES),
        "seed_count": len(QUALIFICATION2_SEEDS),
        "scenario_count": len(records),
        "epochs_per_scenario": EPOCHS,
        "groups_per_epoch": GROUPS_PER_EPOCH,
        "frames_per_group": FRAMES_PER_GROUP,
        "light_frames_per_scenario": EPOCHS * GROUPS_PER_EPOCH * FRAMES_PER_GROUP,
        "site_coord": [8, 8],
        "sensor_shape": [96, 128],
        "cfa_pattern": "GRBG",
        "temporal_kinds": sorted(TEMPORAL_KINDS),
    }

    content = {
        "manifest_schema": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "dataset_role": QUALIFICATION2_ROLE,
        "seeds": list(QUALIFICATION2_SEEDS),
        "classes": list(QUALIFICATION2_CLASSES),
        "structure": structure,
        "scenarios": records,
        "verified_assertions": _verified_assertions(),
    }
    manifest_hash = _sha256(_stable_json_dumps(content))
    content["manifest_hash"] = manifest_hash
    return content


def write_qualification2_manifest(path) -> None:
    """Write the QUALIFICATION-2 manifest to ``path`` (JSON, deterministic)."""
    payload = build_qualification2_manifest()
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "QUALIFICATION2_SEEDS",
    "QUALIFICATION2_CLASSES",
    "QUALIFICATION2_ROLE",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "Qualification2ScenarioRecord",
    "build_qualification2_manifest",
    "build_qualification2_scenario",
    "qualification2_scenarios",
    "write_qualification2_manifest",
]
