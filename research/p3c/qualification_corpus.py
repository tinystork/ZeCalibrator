"""P3C-3 — QUALIFICATION dataset builder (truth-side harness).

Research-only, internal, non-public. Mirrors the DEVELOPMENT corpus
(:mod:`research.p3c.development_corpus`) but with a **distinct seed set**, so
the QUALIFICATION measurement runs on fresh scenarios — never the scenarios the
candidates were derived from.

Role discipline (SCIENCE §36–§37 / §42 / ARCHITECTURE §18):

* Only the ``QUALIFICATION`` dataset is built here. The ``HOLDOUT`` dataset is
  **never** opened, built, or referenced: this module receives no holdout path,
  no holdout manifest, no holdout seed set, and does not import
  ``DATASET_HOLDOUT`` or :mod:`research.p3b.holdout`. This is structural, not a
  promise: there is simply no code path in ``research/p3c`` that names the
  holdout (enforced by tests/p3c/test_qualification_holdout_seal.py).
* The scenarios share the same declared *shape* as DEVELOPMENT (sensor,
  acquisition structure, site coordinate, censored / non-representative
  declarations) — only the seeds differ, so the measurement is genuinely
  out-of-sample relative to candidate derivation while staying directly
  comparable. The scenario builder of the accepted DEVELOPMENT corpus is reused
  verbatim (it is a pure function of ``(class, seed)`` and does not know which
  role it is being materialised for); only the scenario ``name`` prefix is
  changed to ``qual:``. The dataset signature deliberately **excludes** the
  name, so this rename changes nothing about what the data *is*.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from research.p3b.dataset_split import DATASET_QUALIFICATION, DatasetRegistry
from research.p3b.model import ScenarioSpec

from .development_corpus import DEVELOPMENT_CLASSES, build_development_scenario

# The same 20 synthetic classes as DEVELOPMENT — never a second, drifting list.
QUALIFICATION_CLASSES: Tuple[str, ...] = tuple(DEVELOPMENT_CLASSES)

# Seeds disjoint from DEVELOPMENT_SEEDS = (0, 1, 2, 3, 4). A different seed set
# is what makes QUALIFICATION an independent measurement: the declared science
# (classes, structure) is identical, but every frame's read noise and behaviour
# draws a fresh sample. Seeds 5..9 are fixed so the manifest hash is stable.
QUALIFICATION_SEEDS: Tuple[int, ...] = (5, 6, 7, 8, 9)


def build_qualification_scenario(
    class_name: str,
    seed: int,
    **kwargs,
) -> ScenarioSpec:
    """Build one coherent QUALIFICATION scenario for ``class_name``.

    Reuses the accepted DEVELOPMENT scenario builder (a pure function of
    ``(class, seed)``) and re-labels the scenario with a ``qual:`` prefix. The
    dataset signature excludes the name, so the signature is governed solely by
    the (class, seed, declared structure) — which is exactly what makes the
    anti-leakage guard meaningful.
    """
    base = build_development_scenario(class_name, seed, **kwargs)
    return replace(base, name=f"qual:{class_name}:{seed}")


def qualification_scenarios() -> Tuple[ScenarioSpec, ...]:
    """The full bounded QUALIFICATION scenario set (20 classes × 5 seeds)."""
    out = []
    for name in QUALIFICATION_CLASSES:
        for seed in QUALIFICATION_SEEDS:
            out.append(build_qualification_scenario(name, seed))
    return tuple(out)


def build_qualification_manifest():
    """Register the QUALIFICATION scenarios under the ``QUALIFICATION`` role.

    Uses the existing :class:`~research.p3b.dataset_split.DatasetRegistry` so
    the signature-based anti-leakage guard is exercised. A fresh registry is
    used, and **only** the ``QUALIFICATION`` role is ever registered — no
    ``HOLDOUT`` manifest is built or touched anywhere in this module.
    """
    registry = DatasetRegistry()
    return registry.register(DATASET_QUALIFICATION, qualification_scenarios())


__all__ = [
    "QUALIFICATION_CLASSES",
    "QUALIFICATION_SEEDS",
    "build_qualification_manifest",
    "build_qualification_scenario",
    "qualification_scenarios",
]
