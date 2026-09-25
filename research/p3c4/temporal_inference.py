"""P3C-4 LOT 3 — temporal-persistence inference (new `persisted_at_same_sensor_coord`).

Research-only, internal, non-public. This module replaces the P3C-2 *spatial /
local* basis for ``persisted_at_same_sensor_coord`` with the **temporal proof**
of the LOT 1 contract (:mod:`research.p3c4.temporal_evidence`). It introduces a
new, **versioned** candidate family ``p3c4-baseline`` / ``p3c4-conservative`` /
``p3c4-sensitive``: the five *other* SENSOR-EVIDENCE facts are produced by the
**frozen, unchanged** P3C candidates (delegated verbatim), and only
``persisted_at_same_sensor_coord`` is re-derived from temporal evidence.

Boundary (SCIENCE §16 / ARCHITECTURE §18, enforced by
``tests/p3c4/test_truth_leak_guard.py``): this module reads **no truth**. It
imports only the temporal contract, the P3C inference contract + candidate
configs (the inference side, which is itself truth-free), and the Python standard
library. It never imports ``research.p3b.*``, never references a class label, a
scenario name, or an expected state, and never touches the corpus or the truth.

How temporal proof replaces spatial proof (SCIENCE §20)
--------------------------------------------------------

The frozen P3C candidate inferred ``persisted = YES`` on a **spatial / local**
basis (a local signal recurring across independent groups). That is exactly the
defect LOT 1 diagnosed: a celestial confounder (an undersampled star core) is a
local, high-SNR signal, so the spatial rule promoted it to a persistent sensor
site. The temporal rule forbids this: a high spatial SNR can **never** yield
``persisted = YES`` without temporal support.

The rule distinguishes four behaviours, using the LOT 1 contract as the single
basis and extending it only with a movement/departure discriminator (``NO``
reachable — see :class:`research.p3c4.temporal_evidence.NonPersistenceEvidence`):

* persistent sensor defect → the residual signature is **stationary** at the
  fixed coordinate across groups **and** epochs → ``YES`` (the LOT 1 contract's
  ≥2 groups AND ≥2 epochs recurrence);
* celestial confounder → the signature **departs** the fixed coordinate between
  epochs (its residual peak moves to a neighbouring coordinate) → ``NO``;
* low-occupancy intermittent → **low occupancy without an epoch departure** →
  ``UNDETERMINED`` (never ``NO``: a rare intermittent is not a non-persistent
  site, and not a transient merely for being rare);
* single excursion → one frame, never repeated → ``NO`` (unique occurrence).

Censoring is non-quantitative (§14): a censored frame contributes nothing, and a
site with insufficient valid samples is ``UNDETERMINED``, never ``NO``/``YES``.

Parameter discipline (§25): every new numeric constant is a
``RESEARCH_CANDIDATE_PARAMETER`` inside the ``p3c4-*`` configs — never a product
threshold, and never a re-tune of the frozen P3C budgets (``25/50/150 ADU``,
neighbourhood SNR, stable dispersion, ``transient_fraction`` are untouched).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Tuple

from research.p3c.inference_candidates import (
    RESEARCH_CANDIDATE_PARAMETER,
    CandidateParam,
    candidate as p3c_candidate,
    infer_site as p3c_infer_site,
)
from research.p3c.inference_contract import (
    DETERMINED,
    NO,
    UNDETERMINED,
    YES,
    AdmissionFacts,
    InferredEvidence,
    InferredField,
    build_inferred_evidence,
)
from research.p3c4.temporal_evidence import (
    GroupSignature,
    NonPersistenceEvidence,
    TemporalSignature,
    assess_temporal_persistence,
)

# Version of the whole temporal candidate-rule schema (fields/units of a config).
SCHEMA_VERSION = "p3c4-temporal-inference-candidates-schema-1"

# The one SENSOR-EVIDENCE fact this module owns. The other five come verbatim
# from the frozen P3C candidate named by ``base_candidate_id``.
PERSISTED_FACT = "persisted_at_same_sensor_coord"


@dataclass(frozen=True)
class TemporalCandidateConfig:
    """A versioned temporal candidate: a frozen P3C base + new temporal params.

    ``base_candidate_id`` names the frozen P3C candidate whose five non-persisted
    facts are reused unchanged (§24: the frozen ``p3c-*`` artefacts are never
    modified). ``parameters`` are the new temporal decision parameters.
    """

    candidate_id: str
    version: str
    description: str
    base_candidate_id: str
    parameters: Tuple[CandidateParam, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not isinstance(self.candidate_id, str):
            raise ValueError("candidate_id must be a non-empty string")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names in {self.candidate_id!r}")

    def param(self, name: str) -> float:
        for p in self.parameters:
            if p.name == name:
                return p.value
        raise KeyError(f"unknown temporal parameter {name!r}")

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "version": self.version,
            "description": self.description,
            "base_candidate_id": self.base_candidate_id,
            "parameters": {
                p.name: {
                    "value": p.value,
                    "unit": p.unit,
                    "kind": p.kind,
                    "rationale": p.rationale,
                }
                for p in self.parameters
            },
        }


def _p(name: str, value: float, unit: str, rationale: str) -> CandidateParam:
    return CandidateParam(name=name, value=value, unit=unit, rationale=rationale)


# ---------------------------------------------------------------------------
# Temporal candidate parameters (all RESEARCH_CANDIDATE_PARAMETER — never a
# product threshold, never a re-tune of the frozen P3C budgets).
#
#   presence_adu     — the residual magnitude (ADU) at the fixed coordinate at or
#                      above which the signature counts as "present" in a group.
#                      This is a temporal *presence* decision, not a spatial SNR.
#                      Measured residuals: read noise ~4-5 ADU; a weak anomaly
#                      ~60 ADU; hot pixel / intermittent / star ~1500-5400 ADU.
#                      The three candidates bracket the weak-anomaly band.
#   departure_offset — the peak-offset (pixels) beyond which the signature has
#                      "moved" to a neighbouring coordinate (celestial departure),
#                      versus a stationary fixed site (offset ~0). Sensitive
#                      uses 1 px, the others 2 px.
# ---------------------------------------------------------------------------

CANDIDATE_P3C4_BASELINE = TemporalCandidateConfig(
    candidate_id="p3c4-baseline",
    version="1",
    description=(
        "Anchor hypothesis: a signature present above 50 ADU counts as present; "
        "a residual peak that moves >= 2 px away from the fixed coordinate is a "
        "celestial departure (NO). Frozen p3c-baseline supplies the five other facts."
    ),
    base_candidate_id="p3c-baseline",
    parameters=(
        _p("presence_adu", 50.0, "ADU",
           "well above read-noise |residual| (~4-5 ADU); detects the weak-anomaly band (~60 ADU)"),
        _p("departure_offset_px", 2.0, "pixels",
           "a residual peak that shifts >= 2 px off the coordinate is a movement/departure"),
    ),
)

CANDIDATE_P3C4_CONSERVATIVE = TemporalCandidateConfig(
    candidate_id="p3c4-conservative",
    version="1",
    description=(
        "Higher presence bar: only residuals above 150 ADU count as present, so "
        "the weak-anomaly band is treated as absent. Otherwise identical to baseline."
    ),
    base_candidate_id="p3c-conservative",
    parameters=(
        _p("presence_adu", 150.0, "ADU",
           "above the weak-anomaly band (~60 ADU), below the hot-pixel scale (~1500 ADU)"),
        _p("departure_offset_px", 2.0, "pixels",
           "a residual peak that shifts >= 2 px off the coordinate is a movement/departure"),
    ),
)

CANDIDATE_P3C4_SENSITIVE = TemporalCandidateConfig(
    candidate_id="p3c4-sensitive",
    version="1",
    description=(
        "Lower presence bar and looser departure: anomalies above 25 ADU count, "
        "and a single-pixel peak shift is already a departure. Catches more, "
        "abstains less."
    ),
    base_candidate_id="p3c-sensitive",
    parameters=(
        _p("presence_adu", 25.0, "ADU",
           "well above read-noise |residual| (~4-5 ADU); detects the weak-anomaly band (~60 ADU)"),
        _p("departure_offset_px", 1.0, "pixels",
           "a single-pixel peak shift off the coordinate is already a departure"),
    ),
)

TEMPORAL_CANDIDATES: Tuple[TemporalCandidateConfig, ...] = (
    CANDIDATE_P3C4_BASELINE,
    CANDIDATE_P3C4_CONSERVATIVE,
    CANDIDATE_P3C4_SENSITIVE,
)

TEMPORAL_CANDIDATE_IDS: Tuple[str, ...] = tuple(c.candidate_id for c in TEMPORAL_CANDIDATES)


def temporal_candidate(candidate_id: str) -> TemporalCandidateConfig:
    for c in TEMPORAL_CANDIDATES:
        if c.candidate_id == candidate_id:
            return c
    raise KeyError(f"unknown temporal candidate {candidate_id!r}")


# ---------------------------------------------------------------------------
# Pure descriptive helpers (no truth, no I/O, deterministic)
# ---------------------------------------------------------------------------


def _finite(values) -> list:
    out = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return out


# ---------------------------------------------------------------------------
# Temporal signature assembly (measured presence per group)
# ---------------------------------------------------------------------------


def _build_temporal_signature(cfg: TemporalCandidateConfig, tf) -> TemporalSignature:
    """Build a LOT 1 :class:`TemporalSignature` from measured temporal features.

    ``tf`` is a :class:`~research.p3c4.temporal_features.TemporalSiteFeatures`
    (duck-typed). For each independent ``(epoch, group)``, ``signature_present``
    is ``YES`` when the group's median residual at the fixed coordinate is at/above
    the presence threshold (over valid samples only), ``NO`` when measured absent,
    and ``UNDETERMINED`` when the group has no valid sample (all censored).
    """
    presence = cfg.param("presence_adu")

    order = []
    groups = {}
    for i in range(len(tf.frame_epoch_ids)):
        key = (tf.frame_epoch_ids[i], tf.frame_group_ids[i])
        if key not in groups:
            groups[key] = {"residuals": [], "valid": 0, "censored": 0}
            order.append(key)
        if tf.frame_censored[i]:
            groups[key]["censored"] += 1
        else:
            groups[key]["valid"] += 1
            groups[key]["residuals"].append(tf.frame_coord_residual[i])

    signatures = []
    for key in order:
        epoch_id, group_id = key
        g = groups[key]
        if g["valid"] == 0:
            present = UNDETERMINED
        else:
            median = statistics.median(g["residuals"])
            present = YES if median >= presence else NO
        signatures.append(
            GroupSignature(
                group_id=group_id,
                epoch_id=epoch_id,
                valid_samples=g["valid"],
                censored_samples=g["censored"],
                signature_present=present,
            )
        )

    return TemporalSignature(
        coordinate_x=int(tf.coordinate_x),
        coordinate_y=int(tf.coordinate_y),
        groups=tuple(signatures),
    )


def _build_non_persistence_evidence(
    cfg: TemporalCandidateConfig, tf
) -> NonPersistenceEvidence:
    """Measure the movement/departure and unique-occurrence discriminators.

    * ``departed`` — the signature's residual peak moved to a neighbouring
      coordinate (offset >= ``departure_offset_px``) across at least two valid
      frames, while the signature is present. A stationary site keeps its peak
      at the fixed coordinate (offset ~0).
    * ``unique_occurrence`` — the signature appeared in exactly one valid frame
      and never recurred (a single excursion), regardless of amplitude.
    """
    presence = cfg.param("presence_adu")
    departure_offset = cfg.param("departure_offset_px")

    present_frames = 0
    peaks = []
    for i in range(len(tf.frame_coord_residual)):
        if tf.frame_censored[i]:
            continue
        r = tf.frame_coord_residual[i]
        if math.isfinite(r) and r >= presence:
            present_frames += 1
        pk = tf.frame_peak_residual[i]
        if math.isfinite(pk) and pk >= presence:
            peaks.append((tf.frame_peak_dy[i], tf.frame_peak_dx[i]))

    unique_occurrence = YES if present_frames == 1 else NO

    departed = NO
    if len(peaks) >= 2:
        for a in range(len(peaks)):
            for b in range(a + 1, len(peaks)):
                dist = math.hypot(
                    peaks[a][0] - peaks[b][0], peaks[a][1] - peaks[b][1]
                )
                if dist >= departure_offset:
                    departed = YES
                    break
            if departed == YES:
                break

    return NonPersistenceEvidence(
        departed=departed,
        unique_occurrence=unique_occurrence,
    )


def infer_persisted(
    cfg: TemporalCandidateConfig, tf
) -> InferredField:
    """Infer ``persisted_at_same_sensor_coord`` from temporal evidence only.

    This is the LOT 3 rule that replaces the spatial/local basis. It never reads
    a spatial SNR, a neighbourhood feature, or any truth; it consumes only the
    measured temporal features ``tf``.
    """
    signature = _build_temporal_signature(cfg, tf)
    non_persistence = _build_non_persistence_evidence(cfg, tf)
    result = assess_temporal_persistence(signature, non_persistence=non_persistence)

    uncertainty = DETERMINED if result.state in (YES, NO) else UNDETERMINED
    return InferredField(
        field=PERSISTED_FACT,
        value=result.state,
        source="temporal persistence evidence (LOT 3): stationary recurrence at the "
        "fixed coordinate vs movement/departure vs unique occurrence",
        uncertainty=uncertainty,
        feature_refs=(
            "fixed_sensor_coordinate",
            "per_frame_site_residual",
            "per_group_site_residual",
            "non_persistence_demonstrated",
        ),
    )


def infer_site(
    cfg: TemporalCandidateConfig,
    features,
    temporal_features,
    admission: AdmissionFacts,
) -> InferredEvidence:
    """Infer one site's SENSOR-EVIDENCE facts with temporal candidate ``cfg``.

    The five non-persisted facts come **verbatim** from the frozen P3C candidate
    named by ``cfg.base_candidate_id`` (the old spatial rules are untouched for
    those facts — §21/§24). Only ``persisted_at_same_sensor_coord`` is replaced
    by the temporal proof.

    ``features`` is the LOT4 ``SiteFeatures`` (measured, censored-filtered);
    ``temporal_features`` is the temporal feature set; ``admission`` traverses
    untouched. Pure and deterministic.
    """
    base_cfg = p3c_candidate(cfg.base_candidate_id)
    base_evidence = p3c_infer_site(base_cfg, features, admission)

    persisted = infer_persisted(cfg, temporal_features)

    fields = tuple(
        persisted if f.field == PERSISTED_FACT else f
        for f in base_evidence.sensor_evidence
    )
    return build_inferred_evidence(cfg.candidate_id, admission, fields)


__all__ = [
    "RESEARCH_CANDIDATE_PARAMETER",
    "SCHEMA_VERSION",
    "PERSISTED_FACT",
    "TemporalCandidateConfig",
    "CANDIDATE_P3C4_BASELINE",
    "CANDIDATE_P3C4_CONSERVATIVE",
    "CANDIDATE_P3C4_SENSITIVE",
    "TEMPORAL_CANDIDATES",
    "TEMPORAL_CANDIDATE_IDS",
    "temporal_candidate",
    "infer_persisted",
    "infer_site",
]
