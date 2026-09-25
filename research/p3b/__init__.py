"""P3B synthetic-corpus harness (LOT1) — deterministic raw-frame generator.

This package is **internal and non-packaged** (a research harness, like
``research/phase0``, ``research/phase1``, ``research/phase7``). It is **not**
a capability, **not** public API, and is **never** imported by
``zecalibrator.api.v1``.

LOT1 scope (strictly bounded)
-----------------------------
Build the *deterministic synthetic-corpus infrastructure* that later lots will
exercise. This lot contains **no** qualification logic, **no** action logic,
**no** metrics, **no** detectors, **no** product thresholds and **no**
``SensorProfile`` promotion. It only *fabricates* raw frames and *declares*
ground truth.

See the mission ``ZC-SENSOR-P3B-LOT1-SYNTHETIC-CORPUS`` and
``docs/SCIENCE_CONTRACT.md`` §13 / ``docs/ARCHITECTURE.md`` §18 for the
normative invariants this harness must respect.
"""

from .catalog import (
    CLASS_CATALOG,
    CLASS_NAMES,
    ClassEntry,
    UnknownClassError,
    resolve_class,
)
from .generator import CorpusResult, generate_corpus
from .model import (
    AggregateSpec,
    Bookkeeping,
    DuplicateFrameIdError,
    EpochSpec,
    FrameSpec,
    GroupSpec,
    QUALIFICATION_STATES,
    ACTION_STATES,
    ScenarioSpec,
    SensorSpec,
    SiteSpec,
    UnknownConstituentError,
    compute_bookkeeping,
)
from .parameters import (
    EXPLORATORY,
    SYNTHETIC_GENERATOR_PARAMETER,
    all_params,
    get_param,
    param,
    param_kind,
)

__all__ = [
    "CLASS_CATALOG",
    "CLASS_NAMES",
    "ClassEntry",
    "UnknownClassError",
    "resolve_class",
    "CorpusResult",
    "generate_corpus",
    "AggregateSpec",
    "Bookkeeping",
    "DuplicateFrameIdError",
    "EpochSpec",
    "FrameSpec",
    "GroupSpec",
    "QUALIFICATION_STATES",
    "ACTION_STATES",
    "ScenarioSpec",
    "SensorSpec",
    "SiteSpec",
    "UnknownConstituentError",
    "compute_bookkeeping",
    "EXPLORATORY",
    "SYNTHETIC_GENERATOR_PARAMETER",
    "all_params",
    "get_param",
    "param",
    "param_kind",
]
