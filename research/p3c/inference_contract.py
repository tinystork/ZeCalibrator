"""P3C-1 inference contract — the inference side, structurally separated from truth.

This module defines the **interface** of evidence inference. It implements
**no inference rule** (P3C-2): no candidate, no detector, no threshold, no
score. It only fixes the shape of the result and the invariants that a later
inference implementation must satisfy.

Boundary (SCIENCE §16, enforced structurally by tests/p3c/test_truth_leak_guard.py):
this module reads **no truth**. It never imports ``research.p3b.declared_facts``
(``_CLASS_FACTS``), never reads catalogue expected labels, never touches
``cfa_class`` / ``expected_*`` / scenario name / the truth manifest. It may
receive *measured features*, legitimate acquisition metadata, the independence
structure and admission facts as *inputs*; it never reaches for ground truth.

Design (the four non-negotiable properties, each tested):

* **No magic boolean, no confidence score.** Every inferred value is an
  explicit string state (``YES`` / ``NO`` / ``UNDETERMINED``, or a structured
  state). There is no boolean field and no numeric score/probability anywhere.

* **Two fact categories, structurally separated (§17).**
  ``EXOGENOUS_FACTS`` (admission/provenance: identity, geometry, calibration
  presence/representativeness, independence counts) can **never** be produced
  by pixel inference — they traverse the contract untouched, and attempting to
  infer one raises :class:`ExogenousFactError`. ``SENSOR_EVIDENCE_FACTS`` are
  the only facts an inference may produce.

* **``net_benefit_established`` is NOT predicted (§18).** It is exposed as
  :data:`NON_INFERABLE_FACTS` and carried as an always-``UNDETERMINED``
  separate proof on the result. Qualification is evaluated *before* net
  benefit, never forced to ``ESTABLISHED`` to inflate recall.

* **``UNDETERMINED`` is a first-class, normal output (§21).** Insufficient
  evidence produces an explicit ``UNDETERMINED`` (or the structured
  ``INDETERMINATE`` residual state), never an implicit YES/NO default.

* **Censoring (§19).** ``censored != high-value anomaly``. A censored sample
  forbids any quantitative state inference (characterised residual / amplitude);
  the contract raises :class:`CensoredInferenceError` if a censored fact is
  combined with a quantitative residual state. There is no amplitude field at
  all, so no "true amplitude beyond the hard limit" can be smuggled in.

The adversarial-probe structure (§49/§50, exercised in P3C-2) is made
expressible here: ``InferredEvidence`` is keyed only on measured inputs
(features/metadata/admission) plus a candidate identifier. It carries no
``cfa_class``, no ``expected_*``, no scenario name — so identical measured data
with different truth must yield identical inference, and the class *name* can
never drive the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Value vocabulary — explicit string states, never a boolean, never a score.
# ---------------------------------------------------------------------------

YES = "YES"
NO = "NO"
UNDETERMINED = "UNDETERMINED"

# Explicit uncertainty state of an inferred field (never an implicit default).
DETERMINED = "DETERMINED"
# UNDETERMINED is reused as the explicit "insufficient evidence" uncertainty.

# Structured residual states (mirror the qualification-policy vocabulary).
RESIDUAL_NONE = "NONE"
RESIDUAL_SYSTEMATIC_STABLE = "SYSTEMATIC_STABLE"
RESIDUAL_VARIABLE = "VARIABLE"
RESIDUAL_INDETERMINATE = "INDETERMINATE"

# Calibration representativeness (an exogenous fact; passed through).
REPRESENTATIVE = "REPRESENTATIVE"
NOT_REPRESENTATIVE = "NOT_REPRESENTATIVE"

# ---------------------------------------------------------------------------
# Fact categories (§17) — structural separation, never merged.
# ---------------------------------------------------------------------------

# EXOGENOUS / ADMISSION — never inferred from pixels. They come from
# admission/provenance and traverse the contract unmodified.
EXOGENOUS_FACTS: Tuple[str, ...] = (
    "sensor_identity_resolved",
    "geometry_compatible",
    "calibration_present",
    "calibration_representativeness",
    "independent_group_count",
    "epoch_count",
)

# SENSOR-EVIDENCE — the only facts a pixel inference may produce.
SENSOR_EVIDENCE_FACTS: Tuple[str, ...] = (
    "persisted_at_same_sensor_coord",
    "site_residual_behaviour",
    "neighbourhood_residual_stable",
    "transient_only",
    "conflicting_evidence",
    "censored_measurement_present",
)

# §18 — net benefit is a *separate proof*, never predicted by inference.
NON_INFERABLE_FACTS: Tuple[str, ...] = ("net_benefit_established",)


# ---------------------------------------------------------------------------
# Typed errors (contract violations, never silent fallbacks)
# ---------------------------------------------------------------------------


class ExogenousFactError(ValueError):
    """A pixel inference attempted to produce an EXOGENOUS/admission fact.

    Exogenous facts (identity, geometry, calibration, independence) are decided
    by admission/provenance and can never be overwritten by pixel inference.
    """

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(
            f"{field!r} is an EXOGENOUS/admission fact and cannot be inferred "
            "from pixels; it must come from admission/provenance"
        )


class NonInferableFactError(ValueError):
    """A pixel inference attempted to produce a fact that is never predicted."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(
            f"{field!r} is not inferable: it is a separate proof (§18), "
            "never produced by pixel inference"
        )


class UnknownFactError(ValueError):
    """A field name is not in any declared fact category."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"unknown fact name: {field!r}")


class CensoredInferenceError(ValueError):
    """A quantitative state inference was claimed from a censored sample (§19).

    A censored measurement cannot support a characterised residual or any
    amplitude/state inference beyond the hard limit.
    """

    def __init__(self, field: str, value: str) -> None:
        self.field = field
        self.value = value
        super().__init__(
            f"censored sample cannot support a quantitative residual state: "
            f"{field}={value!r} (§19)"
        )


# ---------------------------------------------------------------------------
# Per-field value vocabularies (no boolean, no number)
# ---------------------------------------------------------------------------

_TERNARY = (YES, NO, UNDETERMINED)
_RESIDUAL_STATES = (
    RESIDUAL_NONE,
    RESIDUAL_SYSTEMATIC_STABLE,
    RESIDUAL_VARIABLE,
    RESIDUAL_INDETERMINATE,
)

_SENSOR_EVIDENCE_VOCABULARY = {
    "persisted_at_same_sensor_coord": _TERNARY,
    "site_residual_behaviour": _RESIDUAL_STATES,
    "neighbourhood_residual_stable": _TERNARY,
    "transient_only": _TERNARY,
    "conflicting_evidence": _TERNARY,
    "censored_measurement_present": _TERNARY,
}


def undetermined_value_for(field: str) -> str:
    """The explicit "insufficient evidence" value for a sensor-evidence field."""
    if field == "site_residual_behaviour":
        return RESIDUAL_INDETERMINATE
    return UNDETERMINED


def _require_value(field: str, value: str) -> None:
    vocab = _SENSOR_EVIDENCE_VOCABULARY[field]
    if value not in vocab:
        raise ValueError(f"{field} value must be one of {vocab!r}, got {value!r}")


# ---------------------------------------------------------------------------
# Admission facts (exogenous) — immutable, pass-through
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionFacts:
    """EXOGENOUS/admission facts carried verbatim through the contract.

    These are decided upstream by admission/provenance and can never be
    modified by inference. Defaults are explicit ``UNDETERMINED`` (never an
    implicit YES/NO); a real admission supplies concrete values.
    """

    sensor_identity_resolved: str = UNDETERMINED
    geometry_compatible: str = UNDETERMINED
    calibration_present: str = UNDETERMINED
    calibration_representativeness: str = UNDETERMINED
    independent_group_count: int = 0
    epoch_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "sensor_identity_resolved",
            "geometry_compatible",
            "calibration_present",
        ):
            _require_value_loose(name, getattr(self, name))
        if self.calibration_representativeness not in (
            REPRESENTATIVE,
            NOT_REPRESENTATIVE,
            UNDETERMINED,
        ):
            raise ValueError(
                f"calibration_representativeness must be one of "
                f"{(REPRESENTATIVE, NOT_REPRESENTATIVE, UNDETERMINED)!r}"
            )
        for name in ("independent_group_count", "epoch_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")


def _require_value_loose(name: str, value: str) -> None:
    if value not in (YES, NO, UNDETERMINED):
        raise ValueError(f"{name} must be one of {(YES, NO, UNDETERMINED)!r}, got {value!r}")


# ---------------------------------------------------------------------------
# The typed inference result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferredField:
    """One inferred SENSOR-EVIDENCE fact with machine-visible provenance.

    ``value`` is an explicit string state (``YES``/``NO``/``UNDETERMINED`` or a
    structured residual state). ``uncertainty`` is explicit (``DETERMINED`` /
    ``UNDETERMINED``), never an implicit default. ``source`` describes which
    features/metadata produced the value; ``feature_refs`` lists the features
    actually consumed.
    """

    field: str
    value: str
    source: str
    uncertainty: str
    feature_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.field in EXOGENOUS_FACTS:
            raise ExogenousFactError(self.field)
        if self.field in NON_INFERABLE_FACTS:
            raise NonInferableFactError(self.field)
        if self.field not in SENSOR_EVIDENCE_FACTS:
            raise UnknownFactError(self.field)

        _require_value(self.field, self.value)
        if self.uncertainty not in (DETERMINED, UNDETERMINED):
            raise ValueError(
                f"uncertainty must be {DETERMINED!r} or {UNDETERMINED!r}, "
                f"got {self.uncertainty!r}"
            )

        # UNDETERMINED is explicit, never implicit: a field marked UNDETERMINED
        # must carry the field's "insufficient evidence" value, and vice versa.
        undetermined = undetermined_value_for(self.field)
        if self.uncertainty == UNDETERMINED:
            if self.value != undetermined:
                raise ValueError(
                    f"{self.field} uncertainty={UNDETERMINED!r} requires "
                    f"value={undetermined!r}, got {self.value!r}"
                )
        else:  # DETERMINED
            if self.value == undetermined:
                raise ValueError(
                    f"{self.field} value={undetermined!r} requires explicit "
                    f"uncertainty={UNDETERMINED!r}, got {self.uncertainty!r}"
                )

        if not isinstance(self.source, str) or not self.source:
            raise ValueError(f"{self.field} source must be a non-empty string")

        object.__setattr__(self, "feature_refs", tuple(self.feature_refs))


@dataclass(frozen=True)
class InferredEvidence:
    """The typed result of one candidate configuration's inference.

    ``candidate_id`` identifies the candidate configuration. ``admission`` is
    the immutable exogenous fact set. ``sensor_evidence`` carries only
    SENSOR-EVIDENCE facts. ``net_benefit_established`` is never a field here —
    it is exposed as an always-``UNDETERMINED`` separate proof (§18).
    """

    candidate_id: str
    admission: AdmissionFacts
    sensor_evidence: Tuple[InferredField, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.admission, AdmissionFacts):
            raise TypeError(f"admission must be an AdmissionFacts, got {type(self.admission)!r}")

        object.__setattr__(self, "sensor_evidence", tuple(self.sensor_evidence))

        seen = set()
        for f in self.sensor_evidence:
            if not isinstance(f, InferredField):
                raise TypeError(f"sensor_evidence items must be InferredField, got {type(f)!r}")
            if f.field in seen:
                raise ValueError(f"duplicate inferred field: {f.field!r}")
            seen.add(f.field)

        self._validate_censoring()

    def _validate_censoring(self) -> None:
        by_name = {f.field: f for f in self.sensor_evidence}
        censored = by_name.get("censored_measurement_present")
        residual = by_name.get("site_residual_behaviour")
        if (
            censored is not None
            and censored.value == YES
            and residual is not None
            and residual.value in (RESIDUAL_SYSTEMATIC_STABLE, RESIDUAL_VARIABLE)
        ):
            raise CensoredInferenceError(residual.field, residual.value)

    @property
    def net_benefit_established(self) -> str:
        """``net_benefit_established`` is a separate proof, never predicted (§18).

        Always ``UNDETERMINED`` here — the inference never forces ``ESTABLISHED``.
        """
        return UNDETERMINED

    def field(self, name: str) -> Optional[InferredField]:
        """Return the inferred field ``name``, or ``None`` if not inferred.

        ``None`` means "not inferred" — never a silent YES/NO.
        """
        for f in self.sensor_evidence:
            if f.field == name:
                return f
        return None


def build_inferred_evidence(
    candidate_id: str,
    admission: AdmissionFacts,
    sensor_evidence: Tuple[InferredField, ...],
) -> InferredEvidence:
    """Construct an :class:`InferredEvidence` with full invariant validation."""
    return InferredEvidence(
        candidate_id=candidate_id,
        admission=admission,
        sensor_evidence=sensor_evidence,
    )


__all__ = [
    "YES",
    "NO",
    "UNDETERMINED",
    "DETERMINED",
    "RESIDUAL_NONE",
    "RESIDUAL_SYSTEMATIC_STABLE",
    "RESIDUAL_VARIABLE",
    "RESIDUAL_INDETERMINATE",
    "REPRESENTATIVE",
    "NOT_REPRESENTATIVE",
    "EXOGENOUS_FACTS",
    "SENSOR_EVIDENCE_FACTS",
    "NON_INFERABLE_FACTS",
    "AdmissionFacts",
    "InferredField",
    "InferredEvidence",
    "build_inferred_evidence",
    "ExogenousFactError",
    "NonInferableFactError",
    "UnknownFactError",
    "CensoredInferenceError",
    "undetermined_value_for",
]
