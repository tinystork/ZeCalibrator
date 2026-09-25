"""P3C-2 — a bounded family of transparent, deterministic inference candidates.

Research-only, internal, non-public. This module implements the *inference
side* of the P3C evidence pipeline: a small, bounded family of **deterministic,
transparent rules** that transform measured LOT4 features
(:class:`research.p3b.features.SiteFeatures`) plus immutable admission facts
(:class:`research.p3c.inference_contract.AdmissionFacts`) into an
:class:`~research.p3c.inference_contract.InferredEvidence` conforming to the
P3C-1 contract.

Boundary (SCIENCE §16 / ARCHITECTURE §18, enforced by tests/p3c):

* **No truth is read.** This module imports only
  :mod:`research.p3c.inference_contract` and the Python standard library. It
  never imports ``research.p3b.declared_facts``, ``catalog``, ``fixtures``,
  ``metrics``, or the ``research.p3c.oracle`` truth table, and never references
  ``cfa_class`` / ``expected_*`` / a scenario name / the truth manifest.
* **Features only via LOT4.** It consumes the *already computed*
  ``SiteFeatures`` (the censored-filtered path). It never calls a
  ``research.p3b.features`` primitive on a raw, unfiltered series.
* **No ML, no scores.** Every rule is a closed-form, deterministic predicate
  over descriptive feature statistics. The output carries explicit string
  states (``YES``/``NO``/``UNDETERMINED`` or a structured residual state),
  never a boolean and never a confidence score.
* **No product thresholds.** Every numeric constant is a *candidate
  parameter*, versioned and marked ``RESEARCH_CANDIDATE_PARAMETER``. No numeric
  budget is fixed here; no candidate is frozen (that is P3C-3).
* **Censoring (§19).** A censored measurement forbids any quantitative residual
  state: a censored site yields ``site_residual_behaviour = INDETERMINATE``,
  never ``SYSTEMATIC_STABLE``/``VARIABLE``.
* **``net_benefit_established`` is NOT predicted** — it is the always-
  ``UNDETERMINED`` separate proof carried by ``InferredEvidence`` itself.

Derived from the DEVELOPMENT corpus only (see ``development_corpus.py`` and the
accompanying ``development_feature_summary.json``). The derivation justifications
are documented per candidate in :data:`CANDIDATE_JUSTIFICATIONS` and in the
mission report.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence, Tuple

from .inference_contract import (
    DETERMINED,
    NO,
    RESIDUAL_INDETERMINATE,
    RESIDUAL_NONE,
    RESIDUAL_SYSTEMATIC_STABLE,
    RESIDUAL_VARIABLE,
    UNDETERMINED,
    YES,
    AdmissionFacts,
    InferredEvidence,
    InferredField,
    build_inferred_evidence,
)

# ---------------------------------------------------------------------------
# Candidate-parameter marking (never a product threshold)
# ---------------------------------------------------------------------------

RESEARCH_CANDIDATE_PARAMETER = "RESEARCH_CANDIDATE_PARAMETER"

# Version of the whole candidate-rule *schema* (the fields/units of a config).
SCHEMA_VERSION = "p3c-inference-candidates-schema-1"

# The six SENSOR-EVIDENCE facts every candidate must produce (order is stable).
SENSOR_EVIDENCE_ORDER: Tuple[str, ...] = (
    "persisted_at_same_sensor_coord",
    "site_residual_behaviour",
    "neighbourhood_residual_stable",
    "transient_only",
    "conflicting_evidence",
    "censored_measurement_present",
)


@dataclass(frozen=True)
class CandidateParam:
    """One explicit, versioned candidate parameter.

    ``kind`` is always ``RESEARCH_CANDIDATE_PARAMETER`` — never a product
    threshold. ``unit`` and ``rationale`` make the choice auditable.
    """

    name: str
    value: float
    unit: str
    rationale: str
    kind: str = RESEARCH_CANDIDATE_PARAMETER


@dataclass(frozen=True)
class InferenceCandidateConfig:
    """A versioned, bounded candidate configuration.

    Holds a *handful* of explicit parameters (never a giant grid). Two configs
    that differ in any parameter differ in ``candidate_id``; the parameters are
    therefore auditable per candidate.
    """

    candidate_id: str
    version: str
    description: str
    parameters: Tuple[CandidateParam, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not isinstance(self.candidate_id, str):
            raise ValueError("candidate_id must be a non-empty string")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names in candidate {self.candidate_id!r}")

    def param(self, name: str) -> float:
        for p in self.parameters:
            if p.name == name:
                return p.value
        raise KeyError(f"unknown candidate parameter {name!r}")

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "version": self.version,
            "description": self.description,
            "parameters": {
                p.name: {"value": p.value, "unit": p.unit, "kind": p.kind, "rationale": p.rationale}
                for p in self.parameters
            },
        }


def _p(
    name: str, value: float, unit: str, rationale: str
) -> CandidateParam:
    return CandidateParam(name=name, value=value, unit=unit, rationale=rationale)


# ---------------------------------------------------------------------------
# The bounded candidate family (three candidates, three distinct hypotheses).
#
# Parameter derivation (DEVELOPMENT only; see development_feature_summary.json):
#
#   local_anomaly_adu  — the site-vs-neighbour residual magnitude that marks an
#                        anomaly present. Measured: read noise |residual| ~ 4–5
#                        ADU; weak anomaly ~ 60 ADU; hot pixel / intermittent
#                        states ~ 1500–2500 ADU. The three candidates bracket
#                        the weak-anomaly band (25 / 50 / 150) to expose how the
#                        "weak anomaly → INDETERMINATE vs NONE vs STABLE"
#                        boundary moves.
#   stable_disp_adu    — population std of the (post-calibration) residual that
#                        separates STABLE (constant) from VARIABLE. Measured:
#                        stable classes std ~ 4–5 ADU; intermittent/noise/star
#                        std ~ 150–1860 ADU. 25 sits cleanly between the two
#                        populations.
#   neighbourhood_snr  — median(|local residual|) / median(robust local scale)
#                        above which the disagreement is *local* to the site
#                        (neighbourhood stable) rather than *global* (an
#                        optical/flat/star structure). Measured: genuine sensor
#                        sites ≥ ~18; confounders ~ 3.3–4.2. 10 separates them;
#                        the sensitive candidate lowers it to 6.
#   transient_fraction — fraction of non-censored frames carrying a signal above
#                        which a signal is "persistent" rather than "transient".
#                        Measured: single-frame transients ~ 0.17 (per-group
#                        recurrence); intermittent on/off ~ 0.5; stable ~ 1.0.
#   persistence_min_groups — minimum distinct independent (epoch, group) pairs
#                        that must carry the signal for "persisted" = YES. 2
#                        mirrors the policy's ``independent_group_count >= 2``.
#   min_frames          — minimum non-censored frames required to characterise a
#                        residual; fewer ⇒ explicit UNDETERMINED (never a guess).
#   conflict_sign_changes — minimum strict sign crossings of the residual to
#                        declare conflicting evidence. Measured: 0 for every
#                        class except SIGN_CHANGING_POST_DARK (23); the
#                        sensitive candidate uses 1, the others 2.
# ---------------------------------------------------------------------------

CANDIDATE_BASELINE = InferenceCandidateConfig(
    candidate_id="p3c-baseline",
    version="1",
    description=(
        "Anchor hypothesis: median site-vs-neighbour residual above 50 ADU marks "
        "an anomaly; residual std above 25 ADU marks VARIABLE vs STABLE; "
        "neighbourhood stable when the residual exceeds 10x the local scale; "
        "persistence requires signal in >= 2 independent groups."
    ),
    parameters=(
        _p("local_anomaly_adu", 50.0, "ADU",
           "above read-noise |residual| (~4-5 ADU), below the weak-anomaly band (~60 ADU)"),
        _p("stable_disp_adu", 25.0, "ADU",
           "between stable classes' std (~4-5 ADU) and variable classes' std (>=~150 ADU)"),
        _p("neighbourhood_snr", 10.0, "dimensionless",
           "between sensor-site snr (>=~18) and confounder snr (~3.3-4.2)"),
        _p("transient_fraction", 0.25, "fraction",
           "between single-frame-transient duty (~0.17) and intermittent duty (~0.5)"),
        _p("persistence_min_groups", 2.0, "count",
           "mirrors the policy's independent_group_count >= 2 requirement"),
        _p("min_frames", 3.0, "count",
           "minimum non-censored frames before a residual may be characterised"),
        _p("conflict_sign_changes", 2.0, "count",
           ">=2 strict sign crossings => genuine contradictory evidence"),
    ),
)

CANDIDATE_CONSERVATIVE = InferenceCandidateConfig(
    candidate_id="p3c-conservative",
    version="1",
    description=(
        "Higher presence bar: only residuals above 150 ADU count as anomalies, "
        "so the weak-anomaly band (~60 ADU) is treated as no-anomaly (NONE). "
        "Otherwise identical to baseline."
    ),
    parameters=(
        _p("local_anomaly_adu", 150.0, "ADU",
           "above the weak-anomaly band (~60 ADU), below hot-pixel scale (~1500 ADU)"),
        _p("stable_disp_adu", 25.0, "ADU",
           "between stable classes' std (~4-5 ADU) and variable classes' std (>=~150 ADU)"),
        _p("neighbourhood_snr", 10.0, "dimensionless",
           "between sensor-site snr (>=~18) and confounder snr (~3.3-4.2)"),
        _p("transient_fraction", 0.25, "fraction",
           "between single-frame-transient duty (~0.17) and intermittent duty (~0.5)"),
        _p("persistence_min_groups", 2.0, "count",
           "mirrors the policy's independent_group_count >= 2 requirement"),
        _p("min_frames", 3.0, "count",
           "minimum non-censored frames before a residual may be characterised"),
        _p("conflict_sign_changes", 2.0, "count",
           ">=2 strict sign crossings => genuine contradictory evidence"),
    ),
)

CANDIDATE_SENSITIVE = InferenceCandidateConfig(
    candidate_id="p3c-sensitive",
    version="1",
    description=(
        "Lower presence bar and looser local-vs-global and conflict thresholds: "
        "anomalies above 25 ADU count, neighbourhood stable above 6x local scale, "
        "a single sign crossing is already conflicting. Catches more, abstains less."
    ),
    parameters=(
        _p("local_anomaly_adu", 25.0, "ADU",
           "well above read-noise |residual| (~4-5 ADU); detects the weak-anomaly band (~60 ADU)"),
        _p("stable_disp_adu", 25.0, "ADU",
           "between stable classes' std (~4-5 ADU) and variable classes' std (>=~150 ADU)"),
        _p("neighbourhood_snr", 6.0, "dimensionless",
           "still above confounder snr (~3.3-4.2), below sensor-site snr (>=~18)"),
        _p("transient_fraction", 0.5, "fraction",
           "splits intermittent duty (~0.5) from single-frame-transient duty (~0.17)"),
        _p("persistence_min_groups", 2.0, "count",
           "mirrors the policy's independent_group_count >= 2 requirement"),
        _p("min_frames", 3.0, "count",
           "minimum non-censored frames before a residual may be characterised"),
        _p("conflict_sign_changes", 1.0, "count",
           "a single strict sign crossing is already treated as conflicting"),
    ),
)

CANDIDATES: Tuple[InferenceCandidateConfig, ...] = (
    CANDIDATE_BASELINE,
    CANDIDATE_CONSERVATIVE,
    CANDIDATE_SENSITIVE,
)

CANDIDATE_IDS: Tuple[str, ...] = tuple(c.candidate_id for c in CANDIDATES)


def candidate(candidate_id: str) -> InferenceCandidateConfig:
    """Return the candidate config ``candidate_id`` (KeyError if unknown)."""
    for c in CANDIDATES:
        if c.candidate_id == candidate_id:
            return c
    raise KeyError(f"unknown inference candidate {candidate_id!r}")


# ---------------------------------------------------------------------------
# Pure descriptive helpers (no truth, no I/O, deterministic)
# ---------------------------------------------------------------------------


def _finite(values: Sequence[float]) -> list:
    """Drop non-finite (NaN / censored) entries from a feature series."""
    out = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return out


def _median_abs(values: Sequence[float]) -> float:
    return statistics.median([abs(v) for v in values])


def _peak(values: Sequence[float]) -> float:
    return max([abs(v) for v in values]) if values else 0.0


def _pstdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


# ---------------------------------------------------------------------------
# The deterministic rule engine
# ---------------------------------------------------------------------------


def _derived(features) -> dict:
    """Compute the transparent derived statistics over LOT4 features.

    ``features`` is a ``research.p3b.features.SiteFeatures`` (duck-typed: only
    the measured fields are read, never a truth label).
    """
    local_res = _finite(features.same_cfa_local_residual)
    local_scale = _finite(features.robust_local_scale)
    residual_series = _finite(features.temporal_residual_series)
    censored = features.censored_count > 0

    n = len(local_res)

    return {
        "local_res": local_res,
        "local_scale": local_scale,
        "residual_series": residual_series,
        "censored": censored,
        "n": n,
        "local_peak": _peak(local_res),
        "local_mag": _median_abs(local_res) if local_res else 0.0,
        "local_disp": _median_abs(local_scale) if local_scale else 0.0,
        "res_disp": _pstdev(residual_series),
        "sign_changes": int(features.sign_changes),
        "frame_epoch_ids": tuple(features.epoch_ids),
        "frame_group_ids": tuple(features.group_ids),
        "frame_local_res": tuple(features.same_cfa_local_residual),
    }


def _signal_statistics(cfg: InferenceCandidateConfig, d: dict) -> dict:
    """Compute per-frame signal flags and their group/epoch aggregation."""
    thr = cfg.param("local_anomaly_adu")
    groups: set = set()
    epochs: set = set()
    signal_count = 0
    frame_count = 0
    for lr, ep, grp in zip(d["frame_local_res"], d["frame_epoch_ids"], d["frame_group_ids"]):
        if isinstance(lr, (int, float)) and math.isfinite(lr):
            frame_count += 1
            if abs(lr) > thr:
                signal_count += 1
                groups.add((ep, grp))
                epochs.add(ep)
    fraction = (signal_count / frame_count) if frame_count else 0.0
    return {
        "signal_fraction": fraction,
        "signal_groups": len(groups),
        "signal_epochs": len(epochs),
    }


def _infer_sensor_evidence(
    cfg: InferenceCandidateConfig, features
) -> Tuple[InferredField, ...]:
    d = _derived(features)
    sig = _signal_statistics(cfg, d)

    thr = cfg.param("local_anomaly_adu")
    stable_disp = cfg.param("stable_disp_adu")
    nbh_snr = cfg.param("neighbourhood_snr")
    trans_frac = cfg.param("transient_fraction")
    min_groups = int(cfg.param("persistence_min_groups"))
    min_frames = int(cfg.param("min_frames"))
    conflict_sign = int(cfg.param("conflict_sign_changes"))

    # ------------------------------------------------------------------ 1.
    # censored_measurement_present — a measured acquisition fact (count of
    # frames at/above the hard limit), never a quantitative inference.
    censored_value = YES if d["censored"] else NO
    censored_field = InferredField(
        field="censored_measurement_present",
        value=censored_value,
        source="measured: censored_count > 0",
        uncertainty=DETERMINED,
        feature_refs=("censored_count",),
    )

    # ------------------------------------------------------------------ 2.
    # site_residual_behaviour — the post-calibration residual character.
    #   censored                    -> INDETERMINATE (no quantitative state, §19)
    #   too few non-censored frames -> INDETERMINATE (insufficient evidence)
    #   no local anomaly            -> NONE
    #   constant residual           -> SYSTEMATIC_STABLE
    #   varying residual            -> VARIABLE
    if d["censored"]:
        residual_field = InferredField(
            field="site_residual_behaviour",
            value=RESIDUAL_INDETERMINATE,
            source="censored sample: no quantitative residual state (§19)",
            uncertainty=UNDETERMINED,
            feature_refs=("censored_count",),
        )
    elif d["n"] < min_frames:
        residual_field = InferredField(
            field="site_residual_behaviour",
            value=RESIDUAL_INDETERMINATE,
            source="insufficient non-censored frames to characterise the residual",
            uncertainty=UNDETERMINED,
            feature_refs=("same_cfa_local_residual",),
        )
    elif d["local_peak"] <= thr:
        residual_field = InferredField(
            field="site_residual_behaviour",
            value=RESIDUAL_NONE,
            source="measured: no local residual above threshold",
            uncertainty=DETERMINED,
            feature_refs=("same_cfa_local_residual",),
        )
    elif d["res_disp"] <= stable_disp:
        residual_field = InferredField(
            field="site_residual_behaviour",
            value=RESIDUAL_SYSTEMATIC_STABLE,
            source="measured: stable (constant) residual above threshold",
            uncertainty=DETERMINED,
            feature_refs=("temporal_residual_series", "same_cfa_local_residual"),
        )
    else:
        residual_field = InferredField(
            field="site_residual_behaviour",
            value=RESIDUAL_VARIABLE,
            source="measured: residual dispersion above the stability threshold",
            uncertainty=DETERMINED,
            feature_refs=("temporal_residual_series", "state_spread"),
        )

    # ------------------------------------------------------------------ 3.
    # neighbourhood_residual_stable — is the disagreement local to the site
    # (neighbours stable) rather than global (an optical/flat/star structure)?
    if d["censored"]:
        neighbourhood_field = InferredField(
            field="neighbourhood_residual_stable",
            value=UNDETERMINED,
            source="censored sample: neighbourhood cannot be assessed",
            uncertainty=UNDETERMINED,
            feature_refs=("same_cfa_local_residual", "robust_local_scale"),
        )
    elif d["n"] < min_frames:
        neighbourhood_field = InferredField(
            field="neighbourhood_residual_stable",
            value=UNDETERMINED,
            source="insufficient non-censored frames to assess neighbourhood",
            uncertainty=UNDETERMINED,
            feature_refs=("same_cfa_local_residual", "robust_local_scale"),
        )
    elif d["local_peak"] <= thr:
        # No local disagreement at all -> the neighbourhood is trivially stable.
        neighbourhood_field = InferredField(
            field="neighbourhood_residual_stable",
            value=YES,
            source="measured: no local disagreement",
            uncertainty=DETERMINED,
            feature_refs=("same_cfa_local_residual", "robust_local_scale"),
        )
    else:
        local_vs_global = YES if d["local_mag"] >= nbh_snr * d["local_disp"] else NO
        neighbourhood_field = InferredField(
            field="neighbourhood_residual_stable",
            value=local_vs_global,
            source="measured: local residual vs robust local scale",
            uncertainty=DETERMINED,
            feature_refs=("same_cfa_local_residual", "robust_local_scale"),
        )

    # ------------------------------------------------------------------ 4.
    # transient_only — a signal confined to a small fraction of frames.
    if d["censored"]:
        transient_field = InferredField(
            field="transient_only",
            value=NO,
            source="measured: censored signal recurs every frame (not transient)",
            uncertainty=DETERMINED,
            feature_refs=("censored_count",),
        )
    elif d["n"] < min_frames:
        transient_field = InferredField(
            field="transient_only",
            value=UNDETERMINED,
            source="insufficient non-censored frames to assess transience",
            uncertainty=UNDETERMINED,
            feature_refs=("same_cfa_local_residual",),
        )
    elif d["local_peak"] <= thr:
        transient_field = InferredField(
            field="transient_only",
            value=NO,
            source="measured: no transient signal",
            uncertainty=DETERMINED,
            feature_refs=("same_cfa_local_residual",),
        )
    elif sig["signal_fraction"] <= trans_frac:
        transient_field = InferredField(
            field="transient_only",
            value=YES,
            source="measured: signal confined to a small fraction of frames",
            uncertainty=DETERMINED,
            feature_refs=("same_cfa_local_residual",),
        )
    else:
        transient_field = InferredField(
            field="transient_only",
            value=NO,
            source="measured: signal present across most frames (not transient)",
            uncertainty=DETERMINED,
            feature_refs=("same_cfa_local_residual",),
        )

    # ------------------------------------------------------------------ 5.
    # conflicting_evidence — strict sign crossings of the residual.
    conflicting_value = YES if d["sign_changes"] >= conflict_sign else NO
    conflicting_field = InferredField(
        field="conflicting_evidence",
        value=conflicting_value,
        source="measured: strict sign crossings of the residual",
        uncertainty=DETERMINED,
        feature_refs=("sign_changes",),
    )

    # ------------------------------------------------------------------ 6.
    # persisted_at_same_sensor_coord — a *local*, *recurring* signal at the
    # same coordinate is a sensor site; a global (sky) or transient signal is
    # not; no signal is undecidable.
    if d["censored"]:
        persisted_value = UNDETERMINED
        persisted_source = "censored sample: sensor-site persistence not assessed (§19)"
        persisted_unc = UNDETERMINED
    elif d["n"] < min_frames:
        persisted_value = UNDETERMINED
        persisted_source = "insufficient non-censored frames to assess persistence"
        persisted_unc = UNDETERMINED
    elif d["local_peak"] <= thr:
        persisted_value = UNDETERMINED
        persisted_source = "measured: no local residual, persistence undecidable"
        persisted_unc = UNDETERMINED
    elif sig["signal_fraction"] <= trans_frac:
        persisted_value = NO
        persisted_source = "measured: transient signal (not persisted at the coordinate)"
        persisted_unc = DETERMINED
    elif neighbourhood_field.value == NO:
        persisted_value = NO
        persisted_source = "measured: global disagreement (sky structure), not a sensor site"
        persisted_unc = DETERMINED
    elif sig["signal_groups"] >= min_groups:
        persisted_value = YES
        persisted_source = "measured: local signal recurs across independent groups"
        persisted_unc = DETERMINED
    else:
        persisted_value = UNDETERMINED
        persisted_source = "measured: signal does not recur across enough groups"
        persisted_unc = UNDETERMINED

    persisted_field = InferredField(
        field="persisted_at_same_sensor_coord",
        value=persisted_value,
        source=persisted_source,
        uncertainty=persisted_unc,
        feature_refs=("same_cfa_local_residual", "robust_local_scale"),
    )

    fields = (
        persisted_field,
        residual_field,
        neighbourhood_field,
        transient_field,
        conflicting_field,
        censored_field,
    )
    # Canonical order (matches the contract's declared fact order).
    by_name = {f.field: f for f in fields}
    return tuple(by_name[name] for name in SENSOR_EVIDENCE_ORDER)


def infer_site(
    cfg: InferenceCandidateConfig,
    features,
    admission: AdmissionFacts,
) -> InferredEvidence:
    """Infer one site's SENSOR-EVIDENCE facts with candidate ``cfg``.

    Pure and deterministic: a function of ``(cfg, features, admission)`` only.
    ``features`` is the LOT4 ``SiteFeatures`` (measured, censored-filtered);
    ``admission`` is the immutable exogenous fact set (traversed untouched).
    """
    facts = _infer_sensor_evidence(cfg, features)
    return build_inferred_evidence(cfg.candidate_id, admission, facts)


__all__ = [
    "RESEARCH_CANDIDATE_PARAMETER",
    "SCHEMA_VERSION",
    "SENSOR_EVIDENCE_ORDER",
    "CandidateParam",
    "InferenceCandidateConfig",
    "CANDIDATE_BASELINE",
    "CANDIDATE_CONSERVATIVE",
    "CANDIDATE_SENSITIVE",
    "CANDIDATES",
    "CANDIDATE_IDS",
    "candidate",
    "infer_site",
]
