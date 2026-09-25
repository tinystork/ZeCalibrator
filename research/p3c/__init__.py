"""P3C inference contract + separate oracle (research-only, non-packaged).

P3C-1 lays the *interface* of evidence inference and *separates the truth* from
the inference. It implements **no inference rule** (that is P3C-2): no candidate,
no threshold, no score, no detector.

Two sides of a hard boundary (SCIENCE §16 / ARCHITECTURE §18):

* :mod:`research.p3c.inference_contract` — the **inference side** (imported here,
  by default). The typed ``InferredEvidence`` result, the two structurally-
  separated fact categories (EXOGENOUS/ADMISSION vs SENSOR-EVIDENCE), the
  explicit ``UNDETERMINED`` output, the censoring rule, and the non-inferable
  ``net_benefit_established``. It reads **no truth**.

* :mod:`research.p3c.oracle` — the **truth side** (imported *explicitly*, never
  by this package). The justified 20-class table + the per-fact SENSOR-EVIDENCE
  truth. It reuses ``research.p3b.declared_facts``.

This package imports **only the inference side**, so importing ``research.p3c``
(or ``research.p3c.inference_contract``) never pulls ``research.p3b.*`` into
``sys.modules`` — enforced by tests/p3c/test_truth_leak_guard.py. The oracle is
opt-in: ``from research.p3c.oracle import ORACLE``.
"""

from .inference_contract import (  # noqa: F401
    DETERMINED,
    NO,
    UNDETERMINED,
    YES,
    EXOGENOUS_FACTS,
    NON_INFERABLE_FACTS,
    SENSOR_EVIDENCE_FACTS,
    AdmissionFacts,
    CensoredInferenceError,
    ExogenousFactError,
    InferredEvidence,
    InferredField,
    NonInferableFactError,
    UnknownFactError,
    build_inferred_evidence,
)

__all__ = [
    # value vocabulary
    "YES",
    "NO",
    "UNDETERMINED",
    "DETERMINED",
    # fact categories
    "EXOGENOUS_FACTS",
    "SENSOR_EVIDENCE_FACTS",
    "NON_INFERABLE_FACTS",
    # typed result + errors
    "AdmissionFacts",
    "InferredField",
    "InferredEvidence",
    "build_inferred_evidence",
    "ExogenousFactError",
    "NonInferableFactError",
    "UnknownFactError",
    "CensoredInferenceError",
]
