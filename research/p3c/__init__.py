"""P3C inference contract + separate oracle (research-only, non-packaged).

P3C-1 lays the *interface* of evidence inference and *separates the truth* from
the inference. It implements **no inference rule** (that is P3C-2): no candidate,
no threshold, no score, no detector.

Two modules, two sides of a hard boundary (SCIENCE §16 / ARCHITECTURE §18):

* :mod:`research.p3c.inference_contract` — the **inference side**. The typed
  ``InferredEvidence`` result, the two structurally-separated fact categories
  (EXOGENOUS/ADMISSION vs SENSOR-EVIDENCE), the explicit ``UNDETERMINED``
  output, the censoring rule, and the non-inferable ``net_benefit_established``.
  It reads **no truth**: no ``declared_facts``, no catalogue expected labels, no
  ``cfa_class`` / ``expected_*`` / scenario name / truth manifest.

* :mod:`research.p3c.oracle` — the **truth side**. The explicit, justified
  20-class oracle table that the metrics consume. It corrects the reproduced
  P3B catalogue defect (7 inconsistent ``default_expected_action_state``
  entries + the near-saturation high-value anomaly), so that the oracle never
  declares a confounder as a reconstruction target.

The boundary is enforced structurally (AST + runtime) by
``tests/p3c/test_truth_leak_guard.py``. The metrics module (a later lot) may
compare inference vs truth *after the fact*; the inference module never can.
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
from .oracle import (  # noqa: F401
    CONFOUNDER_CLASSES,
    OracleEntry,
    ORACLE,
    expected_action,
    epistemic_state,
    oracle_entry,
    reconstruction_targets,
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
    # oracle
    "OracleEntry",
    "ORACLE",
    "CONFOUNDER_CLASSES",
    "oracle_entry",
    "expected_action",
    "epistemic_state",
    "reconstruction_targets",
]
