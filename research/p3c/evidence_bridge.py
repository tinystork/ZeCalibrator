"""P3C-2 — thin adapter from inference output to the *existing* LOT2 policy.

Research-only, internal, non-public.

This module is **not** a reimplementation of qualification. It converts the P3C
inference result (:class:`~research.p3c.inference_contract.InferredEvidence`)
into the LOT2 :class:`~research.p3b.qualification_policy.EvidencePacket` that
the existing, already-accepted policy consumes. The decision itself stays in
``research.p3b.qualification_policy.evaluate``; the run plan stays in
``research.p3b.preparation_plan`` — both are imported and used verbatim, never
re-derived, never re-written.

Mapping rules (explicit and versioned, never silent):

* **EXOGENOUS / admission facts** traverse verbatim from ``AdmissionFacts``
  (identity, geometry, calibration presence/representativeness, independence
  counts). They are never inferred and never touched.
* **SENSOR-EVIDENCE facts** are copied by field name. The shared vocabulary is
  string-identical on both sides (``YES``/``NO``/``UNDETERMINED`` and the four
  residual states).
* **``net_benefit_established``** is the always-``UNDETERMINED`` separate proof
  carried by ``InferredEvidence``; it is passed through, never forced.
* **Three safety facts** (``censored_measurement_present``,
  ``conflicting_evidence``, ``transient_only``) are *binary* in the LOT2 packet
  (``YES`` / ``NO``) but *ternary* in the inference contract (``UNDETERMINED``
  allowed). For these three, ``YES`` is the **abstain** side (the safety guard
  that blocks action: P2 ``ABSTAIN_CENSORED``, P3/P4 ``ABSTAIN_INCONSISTENT``)
  and ``NO`` is the **permissive** side. The bridge is **fail-closed**: an
  ``UNDETERMINED`` inference for a safety fact maps to ``YES`` (abstain), never
  to the permissive ``NO``. This is a deliberate safety choice, not a claim
  that the fact is actually ``YES`` — the inference did not resolve it, so the
  bridge refuses to let the existing policy proceed past the guard.
"""

from __future__ import annotations

from typing import Mapping

from research.p3b.qualification_policy import (
    NO,
    YES,
    EvidencePacket,
)

from .inference_contract import (
    SENSOR_EVIDENCE_FACTS,
    UNDETERMINED as INFER_UNDETERMINED,
    InferredEvidence,
    undetermined_value_for,
)

# Version of the adapter convention (the fail-closed mapping above).
ADAPTER_VERSION = "p3c-evidence-bridge-2"


def _safety_fact(field: str, value: str) -> str:
    """Project a (possibly ternary) safety fact to the packet's binary fact, fail-closed.

    ``YES`` / ``NO`` pass through. ``UNDETERMINED`` maps to ``YES`` — the
    abstain side — so the existing policy abstains (``ABSTAIN_CENSORED`` /
    ``ABSTAIN_INCONSISTENT``) rather than proceeding past a safety guard on an
    unresolved fact. Never the permissive ``NO`` default.
    """
    if value in (YES, NO):
        return value
    if value == INFER_UNDETERMINED:
        return YES
    raise ValueError(
        f"{field} must be YES/NO/UNDETERMINED in the inference output, got {value!r}"
    )


def _sensor_evidence_values(evidence: InferredEvidence) -> Mapping[str, str]:
    """Return the inferred value for each SENSOR-EVIDENCE fact.

    A fact not inferred is carried as its explicit "insufficient evidence"
    value (``undetermined_value_for``) — never a silent YES/NO default. The
    candidate rules always produce all six, but the bridge stays total for any
    valid :class:`InferredEvidence`.
    """
    values = {f.field: f.value for f in evidence.sensor_evidence}
    for field in SENSOR_EVIDENCE_FACTS:
        if field not in values:
            values[field] = undetermined_value_for(field)
    return values


def to_evidence_packet(evidence: InferredEvidence) -> EvidencePacket:
    """Convert one :class:`InferredEvidence` into the LOT2 ``EvidencePacket``.

    Pure: reads only ``evidence``. It performs no decision and no I/O. The
    ``net_benefit_established`` fact is the separate proof, never predicted
    (always ``UNDETERMINED``).
    """
    admission = evidence.admission
    by_field = _sensor_evidence_values(evidence)

    return EvidencePacket(
        sensor_identity_resolved=admission.sensor_identity_resolved,
        geometry_compatible=admission.geometry_compatible,
        calibration_present=admission.calibration_present,
        calibration_representativeness=admission.calibration_representativeness,
        persisted_at_same_sensor_coord=by_field["persisted_at_same_sensor_coord"],
        independent_group_count=admission.independent_group_count,
        epoch_count=admission.epoch_count,
        censored_measurement_present=_safety_fact(
            "censored_measurement_present", by_field["censored_measurement_present"]
        ),
        site_residual_behaviour=by_field["site_residual_behaviour"],
        neighbourhood_residual_stable=by_field["neighbourhood_residual_stable"],
        transient_only=_safety_fact("transient_only", by_field["transient_only"]),
        conflicting_evidence=_safety_fact("conflicting_evidence", by_field["conflicting_evidence"]),
        net_benefit_established=evidence.net_benefit_established,
        persisted_basis=f"candidate={evidence.candidate_id}",
    )


__all__ = [
    "ADAPTER_VERSION",
    "to_evidence_packet",
]
