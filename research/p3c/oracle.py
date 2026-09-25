"""P3C-1 oracle — the truth side, a separate, corrected 20-class table.

This module is the **oracle** the metrics consume. It is deliberately **separate
from the inference side** (:mod:`research.p3c.inference_contract`): the metrics
may compare inference vs truth *after the fact*; the inference module can never
import this table (enforced by tests/p3c/test_truth_leak_guard.py).

Why a separate oracle, and why it corrects the P3B catalogue
------------------------------------------------------------------

The reproduced defect (mission plan §2): on **7 of the 20 classes**, the P3B
catalogue's ``default_expected_action_state`` is inconsistent with what the
qualification policy actually produces from the declared facts:

* ``OPTICAL_STRUCTURE``, ``FLAT_STRUCTURE``, ``DUST_OR_VIGNETTING`` were
  declared ``ELIGIBLE_FOR_TARGETED_RECONSTRUCTION`` — but they are **confounders**
  (optical/flat/vignetting structures, persistence ``NO``), and the policy
  abstains ``ABSTAIN_INCONSISTENT``. Declaring a confounder as an expected
  reconstruction target would *reward promoting a confounder* and *penalise
  correct abstention* — the instrument would measure upside-down.
* ``SINGLE_TRANSIENT``, ``COSMIC_RAY``, ``STAR_CROSSING_SITE``,
  ``UNDERSAMPLED_STAR_CORE`` were declared ``NO_ACTION_REQUIRED`` — but they are
  transient / sky structures (persistence ``NO`` or transient-only), and the
  policy abstains ``ABSTAIN_INCONSISTENT``.

On top of those 7, the guard list (§6.3) adds ``NEAR_SATURATION``: a
**high-value anomaly** persistently near (but below) the saturation limit. Its
residual amplitude lies in the sensor's non-linear response region and is not
trustworthy for quantitative reconstruction (§19: ``censored != high-value
anomaly``). It must therefore **not** be a reconstruction target either, even
though both catalogue and policy currently agree on ``ELIGIBLE`` for it.

Choice of correction site (justified): this oracle is corrected **here**, in a
dedicated P3C table, rather than editing ``research/p3b/catalog.py``. The
catalogue is accepted P3B LOT1 declaration *data* (its defaults are overridable
per site and are consumed as "declared defaults", not as the metrics oracle),
and editing it would ripple into LOT1/LOT8 tests and the accepted P3B.1 state.
The non-negotiable invariant — *the oracle used by the metrics never declares a
confounder as a reconstruction target* — is satisfied and guarded here.

The 8 confounder classes below can never be reconstruction targets. The 12
remaining classes keep their policy-consistent action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

# ---------------------------------------------------------------------------
# Truth-side vocabulary (mirrors research.p3b.model / qualification_policy)
# ---------------------------------------------------------------------------

# Action states (SCIENCE §13.7 ordered ladder).
ACTION_ELIGIBLE = "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION"
ACTION_NO_ACTION = "NO_ACTION_REQUIRED"
ACTION_REQUALIFY = "REQUALIFY_ADDITIVE_CALIBRATION"
ACTION_ABSTAIN_CENSORED = "ABSTAIN_CENSORED"
ACTION_ABSTAIN_INCONSISTENT = "ABSTAIN_INCONSISTENT"
ACTION_ABSTAIN_INSUFFICIENT = "ABSTAIN_INSUFFICIENT_EVIDENCE"

# Epistemic (qualification) states.
EPI_UNQUALIFIED = "UNQUALIFIED"
EPI_PERSISTENT = "QUALIFIED_PERSISTENT"
EPI_INTERMITTENT = "QUALIFIED_INTERMITTENT"
EPI_TRANSIENT = "QUALIFIED_TRANSIENT"
EPI_CENSORED = "CENSORED"

# ---------------------------------------------------------------------------
# The confounders that must never be reconstruction targets (§6.3)
# ---------------------------------------------------------------------------

CONFOUNDER_CLASSES: Tuple[str, ...] = (
    "OPTICAL_STRUCTURE",
    "FLAT_STRUCTURE",
    "DUST_OR_VIGNETTING",
    "COSMIC_RAY",
    "STAR_CROSSING_SITE",
    "UNDERSAMPLED_STAR_CORE",
    "SINGLE_TRANSIENT",
    "NEAR_SATURATION",
)


@dataclass(frozen=True)
class OracleEntry:
    """One explicit, justified oracle row.

    ``epistemic_state`` is the expected qualification (knowledge) state.
    ``expected_action`` is the corrected expected action the oracle must carry.
    ``catalog_default_action`` is what the P3B catalogue declared before, so the
    inconsistency is machine-visible, not silently papered over.
    """

    class_name: str
    epistemic_state: str
    expected_action: str
    justification: str
    catalog_default_action: str

    @property
    def corrected(self) -> bool:
        """True iff this entry fixes an inconsistent catalogue default."""
        return self.expected_action != self.catalog_default_action

    @property
    def is_reconstruction_target(self) -> bool:
        return self.expected_action == ACTION_ELIGIBLE


# ---------------------------------------------------------------------------
# The explicit 20-class table (class, epistemic, expected action, justification,
# prior catalogue default). Order follows research.p3b.catalog.CLASS_NAMES.
# ---------------------------------------------------------------------------

_ORACLE_TABLE: Tuple[OracleEntry, ...] = (
    OracleEntry(
        "NORMAL", EPI_UNQUALIFIED, ACTION_NO_ACTION,
        "No anomaly; the site is indistinguishable from its local background.",
        ACTION_NO_ACTION,
    ),
    OracleEntry(
        "STABLE_ANOMALY_CORRECTED_BY_DARK", EPI_PERSISTENT, ACTION_NO_ACTION,
        "Known persistent site already corrected by a representative dark — knowledge, not action.",
        ACTION_NO_ACTION,
    ),
    OracleEntry(
        "STABLE_ANOMALY_WITH_MISMATCHED_DARK", EPI_PERSISTENT, ACTION_REQUALIFY,
        "Stable residual from a non-representative dark — requalify the calibration, never reconstruct.",
        ACTION_REQUALIFY,
    ),
    OracleEntry(
        "INTERMITTENT_TWO_STATE", EPI_INTERMITTENT, ACTION_ELIGIBLE,
        "Genuine persistent intermittent sensor defect — a legitimate reconstruction target.",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "INTERMITTENT_MULTI_STATE", EPI_INTERMITTENT, ACTION_ELIGIBLE,
        "Genuine multi-state intermittent sensor defect — a legitimate reconstruction target.",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "INTERMITTENT_CONTINUOUS", EPI_INTERMITTENT, ACTION_ELIGIBLE,
        "Genuine continuously-drifting intermittent sensor defect — a legitimate reconstruction target.",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "RARE_HIGH_STATE", EPI_INTERMITTENT, ACTION_ELIGIBLE,
        "Genuine rare-high-state intermittent sensor defect — a legitimate reconstruction target.",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "RARE_LOW_STATE", EPI_INTERMITTENT, ACTION_ELIGIBLE,
        "Genuine rare-low-state intermittent sensor defect — a legitimate reconstruction target.",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "SIGN_CHANGING_POST_DARK", EPI_INTERMITTENT, ACTION_ABSTAIN_INCONSISTENT,
        "Residual flips sign after dark subtraction — contradictory evidence, abstain.",
        ACTION_ABSTAIN_INCONSISTENT,
    ),
    OracleEntry(
        "CENSORED_ANOMALY", EPI_CENSORED, ACTION_ABSTAIN_CENSORED,
        "Hits the acquisition hard limit — censored, no quantitative inference.",
        ACTION_ABSTAIN_CENSORED,
    ),
    OracleEntry(
        "SINGLE_TRANSIENT", EPI_TRANSIENT, ACTION_ABSTAIN_INCONSISTENT,
        "One-frame transient, not a persistent sensor site — abstain (transient-only), "
        "never NO_ACTION. [catalogue said NO_ACTION_REQUIRED]",
        ACTION_NO_ACTION,
    ),
    OracleEntry(
        "OPTICAL_STRUCTURE", EPI_PERSISTENT, ACTION_ABSTAIN_INCONSISTENT,
        "Static optical/sky structure (persistence NO) — a confounder, never a reconstruction "
        "target. [catalogue said ELIGIBLE_FOR_TARGETED_RECONSTRUCTION]",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "STAR_CROSSING_SITE", EPI_TRANSIENT, ACTION_ABSTAIN_INCONSISTENT,
        "A moving star PSF crossing the site (sky structure, persistence NO) — a confounder, "
        "never a sensor reconstruction target. [catalogue said NO_ACTION_REQUIRED]",
        ACTION_NO_ACTION,
    ),
    OracleEntry(
        "UNDERSAMPLED_STAR_CORE", EPI_PERSISTENT, ACTION_ABSTAIN_INCONSISTENT,
        "An undersampled star core (sky structure, persistence NO) — a confounder, never a "
        "sensor reconstruction target. [catalogue said NO_ACTION_REQUIRED]",
        ACTION_NO_ACTION,
    ),
    OracleEntry(
        "COSMIC_RAY", EPI_TRANSIENT, ACTION_ABSTAIN_INCONSISTENT,
        "A single-frame cosmic ray (transient, not a persistent sensor site) — a confounder, "
        "never a reconstruction target. [catalogue said NO_ACTION_REQUIRED]",
        ACTION_NO_ACTION,
    ),
    OracleEntry(
        "NOISE_EXTREME", EPI_UNQUALIFIED, ACTION_ABSTAIN_INSUFFICIENT,
        "Extreme per-frame noise — insufficient evidence to qualify.",
        ACTION_ABSTAIN_INSUFFICIENT,
    ),
    OracleEntry(
        "FLAT_STRUCTURE", EPI_PERSISTENT, ACTION_ABSTAIN_INCONSISTENT,
        "A flat-only structure (persistence NO) — a confounder, never a reconstruction target. "
        "[catalogue said ELIGIBLE_FOR_TARGETED_RECONSTRUCTION]",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "DUST_OR_VIGNETTING", EPI_PERSISTENT, ACTION_ABSTAIN_INCONSISTENT,
        "A dust/vignetting gradient (optical confounder, persistence NO) — never a "
        "reconstruction target. [catalogue said ELIGIBLE_FOR_TARGETED_RECONSTRUCTION]",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "NEAR_SATURATION", EPI_PERSISTENT, ACTION_ABSTAIN_INSUFFICIENT,
        "A high-value anomaly persistently near (but below) the saturation limit: its amplitude "
        "lies in the non-linear response region and is not trustworthy for quantitative "
        "reconstruction (§19, censored != high-value anomaly). Never a reconstruction target. "
        "[catalogue said ELIGIBLE_FOR_TARGETED_RECONSTRUCTION]",
        ACTION_ELIGIBLE,
    ),
    OracleEntry(
        "AMBIGUOUS_INSUFFICIENT_EVIDENCE", EPI_UNQUALIFIED, ACTION_ABSTAIN_INSUFFICIENT,
        "A weak anomaly — insufficient evidence to qualify.",
        ACTION_ABSTAIN_INSUFFICIENT,
    ),
)

ORACLE: Mapping[str, OracleEntry] = {e.class_name: e for e in _ORACLE_TABLE}


def oracle_entry(class_name: str) -> OracleEntry:
    """Return the oracle row for ``class_name`` (KeyError if unknown)."""
    try:
        return ORACLE[class_name]
    except KeyError:
        raise KeyError(f"unknown synthetic class in oracle: {class_name!r}") from None


def expected_action(class_name: str) -> str:
    """The corrected expected action for a class."""
    return oracle_entry(class_name).expected_action


def epistemic_state(class_name: str) -> str:
    """The expected epistemic (qualification) state for a class."""
    return oracle_entry(class_name).epistemic_state


def reconstruction_targets() -> Tuple[str, ...]:
    """The classes whose expected action is targeted reconstruction."""
    return tuple(e.class_name for e in _ORACLE_TABLE if e.is_reconstruction_target)


__all__ = [
    "ACTION_ELIGIBLE",
    "ACTION_NO_ACTION",
    "ACTION_REQUALIFY",
    "ACTION_ABSTAIN_CENSORED",
    "ACTION_ABSTAIN_INCONSISTENT",
    "ACTION_ABSTAIN_INSUFFICIENT",
    "EPI_UNQUALIFIED",
    "EPI_PERSISTENT",
    "EPI_INTERMITTENT",
    "EPI_TRANSIENT",
    "EPI_CENSORED",
    "CONFOUNDER_CLASSES",
    "OracleEntry",
    "ORACLE",
    "oracle_entry",
    "expected_action",
    "epistemic_state",
    "reconstruction_targets",
]
