"""Frozen HOLDOUT dataset builder (LOT7) — independent of every evaluation artifact.

The HOLDOUT dataset is frozen, evaluated once, and never used to choose
anything. The code that produces it must therefore be runnable **without
consulting** any of the following:

* development results,
* a candidate threshold,
* an operating point,
* any evaluation artifact (a report, a metric, a qualification decision).

This module is kept deliberately thin and closed against the evaluation lots:
it imports nothing from :mod:`research.p3b.metrics`,
:mod:`research.p3b.net_benefit`, :mod:`research.p3b.qualification_policy`,
:mod:`research.p3b.features` or :mod:`research.p3b.preparation_plan`. Its public
function accepts only declared :class:`~research.p3b.model.ScenarioSpec` inputs
and returns a :class:`~research.p3b.dataset_split.DatasetManifest` for the
``HOLDOUT`` role. It reads no file, no pixel, no threshold and no report.

The actual frame materialisation (FITS) is LOT1's
:func:`research.p3b.generator.generate_corpus`, which is likewise a pure
function of the declared scenario and therefore also independent of every
evaluation artifact.
"""

from __future__ import annotations

from typing import Tuple

from .dataset_split import DATASET_HOLDOUT, DatasetManifest, build_manifest
from .model import ScenarioSpec


def build_holdout_manifest(scenarios: Tuple[ScenarioSpec, ...]) -> DatasetManifest:
    """Build the frozen HOLDOUT manifest from declared scenarios.

    Pure and independent: the only inputs are declared
    :class:`~research.p3b.model.ScenarioSpec` objects. Nothing about a
    development result, a candidate threshold, an operating point or any
    evaluation artifact is read — and the resulting manifest carries
    ``independent_real_validation == "NO"`` (``NOT_SATISFIED``), never a claim
    of independent real validation.
    """
    return build_manifest(DATASET_HOLDOUT, scenarios)


__all__ = [
    "DATASET_HOLDOUT",
    "DatasetManifest",
    "build_holdout_manifest",
]
