"""P3C-2 — DEVELOPMENT corpus builder + descriptive feature summary.

Research-only, internal, non-public. This module is the **truth-side harness**
for P3C-2: it materialises a small, bounded DEVELOPMENT corpus covering all 20
synthetic classes, computes LOT4 features over it, and writes the descriptive
``development_feature_summary.json`` artifact. It is *not* the inference side
(which is :mod:`research.p3c.inference_candidates`); it may read class names and
declared structure because it is the declaration/measurement harness.

Role discipline (SCIENCE §15 / ARCHITECTURE §18, §36/§42):

* Only the ``DEVELOPMENT`` dataset is built here. The ``QUALIFICATION`` and
  ``HOLDOUT`` datasets are **never** opened, built, or referenced — this module
  receives no holdout path, no holdout manifest, and no qualification seed set.
* The corpus is registered under ``DATASET_DEVELOPMENT`` through
  :mod:`research.p3b.dataset_split` (roles) so the anti-leakage signature guard
  is exercised. Frame materialisation uses :mod:`research.p3b.generator`.

Corpus size justification (§27): 20 classes × 5 seeds = 100 scenarios; each
scenario has 2 epochs × 2 independent groups × 6 light frames (24 light frames,
ordinals 0–5) plus 4 dark frames, so the rare-state (ordinal 5), transient
(ordinal 3), star-crossing and sign-changing behaviours all manifest. The five
seeds give a stable descriptive distribution while keeping the whole build and
feature pass fast (a few seconds).
"""

from __future__ import annotations

import json
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

from research.p3b.dataset_split import (
    DATASET_DEVELOPMENT,
    DatasetRegistry,
)
from research.p3b.features import compute_features
from research.p3b.generator import generate_corpus
from research.p3b.model import (
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SensorSpec,
    SiteSpec,
    compute_bookkeeping,
)

from .inference_candidates import CANDIDATES
from .inference_contract import (
    NO,
    NOT_REPRESENTATIVE,
    REPRESENTATIVE,
    YES,
    AdmissionFacts,
)

# ---------------------------------------------------------------------------
# The declared, bounded DEVELOPMENT corpus (never QUALIFICATION / HOLDOUT)
# ---------------------------------------------------------------------------

DEVELOPMENT_CLASSES: Tuple[str, ...] = (
    "NORMAL",
    "STABLE_ANOMALY_CORRECTED_BY_DARK",
    "STABLE_ANOMALY_WITH_MISMATCHED_DARK",
    "INTERMITTENT_TWO_STATE",
    "INTERMITTENT_MULTI_STATE",
    "INTERMITTENT_CONTINUOUS",
    "RARE_HIGH_STATE",
    "RARE_LOW_STATE",
    "SIGN_CHANGING_POST_DARK",
    "CENSORED_ANOMALY",
    "SINGLE_TRANSIENT",
    "OPTICAL_STRUCTURE",
    "STAR_CROSSING_SITE",
    "UNDERSAMPLED_STAR_CORE",
    "COSMIC_RAY",
    "NOISE_EXTREME",
    "FLAT_STRUCTURE",
    "DUST_OR_VIGNETTING",
    "NEAR_SATURATION",
    "AMBIGUOUS_INSUFFICIENT_EVIDENCE",
)

# Censored / non-representative declarations (mirror research.p3b.fixtures).
_CENSORED_CLASSES = frozenset({"CENSORED_ANOMALY"})
_NOT_REPRESENTATIVE_CLASSES = frozenset({"STABLE_ANOMALY_WITH_MISMATCHED_DARK"})

DEVELOPMENT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

# Acquisition structure: 2 epochs × 2 independent groups × 6 light frames.
_EPOCHS = 2
_GROUPS_PER_EPOCH = 2
_FRAMES_PER_GROUP = 6
_DARK_FRAMES = 4

SUMMARY_SCHEMA = "zecalibrator-p3c-lot2-development-feature-summary"
SUMMARY_VERSION = 1


def _sensor() -> SensorSpec:
    return SensorSpec(
        instance_id="SYNTH-DET-0001",
        model="SYNTH-S50",
        shape=(96, 128),
        cfa_pattern="GRBG",
        binning=(1, 1),
        roi_origin=(0, 0),
        gain=80.0,
        offset_adu=800.0,
        saturation_limit_adu=60000.0,
        filter="NONE",
    )


def build_development_scenario(
    class_name: str,
    seed: int,
    *,
    epochs: int = _EPOCHS,
    groups_per_epoch: int = _GROUPS_PER_EPOCH,
    frames_per_group: int = _FRAMES_PER_GROUP,
    site_params: Sequence[Tuple[str, object]] = (),
) -> ScenarioSpec:
    """Build one coherent DEVELOPMENT scenario for ``class_name``.

    Deterministic: same (class, seed, structure) ⇒ same scenario. The site
    lives at native base-0 sensor coordinate (8, 8) in every scenario so that
    "persisted at the same sensor coordinate" has a fixed meaning.
    ``site_params`` overrides per-site declared behaviour parameters (used by
    the §50 adversarial probe to vary measured behaviour at a fixed class
    label).
    """
    sensor = _sensor()
    dark_frames = tuple(
        FrameSpec(
            frame_id=f"dark{i}", frame_type="dark", epoch_id="cal", group_id="calg", ordinal=i
        )
        for i in range(_DARK_FRAMES)
    )
    epochs_out = [EpochSpec("cal", (GroupSpec("calg", dark_frames),))]

    counter = 0
    for e in range(epochs):
        groups = []
        for g in range(groups_per_epoch):
            frames = tuple(
                FrameSpec(
                    frame_id=f"light{counter + i}",
                    frame_type="light",
                    epoch_id=f"e{e}",
                    group_id=f"g{e}_{g}",
                    ordinal=i,
                )
                for i in range(frames_per_group)
            )
            counter += frames_per_group
            groups.append(GroupSpec(group_id=f"g{e}_{g}", frames=frames))
        epochs_out.append(EpochSpec(epoch_id=f"e{e}", groups=tuple(groups)))

    site = SiteSpec(
        site_id="site0",
        x=8,
        y=8,
        cfa_class=class_name,
        calibration_representativeness=(
            "not_representative" if class_name in _NOT_REPRESENTATIVE_CLASSES else "representative"
        ),
        censored=(class_name in _CENSORED_CLASSES),
        params=tuple(site_params),
    )
    return ScenarioSpec(
        name=f"dev:{class_name}:{seed}",
        seed=seed,
        sensor=sensor,
        epochs=tuple(epochs_out),
        sites=(site,),
    )


def development_scenarios() -> Tuple[ScenarioSpec, ...]:
    """The full bounded DEVELOPMENT scenario set (20 classes × 5 seeds)."""
    out = []
    for name in DEVELOPMENT_CLASSES:
        for seed in DEVELOPMENT_SEEDS:
            out.append(build_development_scenario(name, seed))
    return tuple(out)


def build_development_manifest() -> "object":
    """Register the DEVELOPMENT scenarios under the ``DEVELOPMENT`` role.

    Uses the existing :class:`~research.p3b.dataset_split.DatasetRegistry` so
    the signature-based anti-leakage guard is exercised, and returns the
    manifest (role == DEVELOPMENT). No QUALIFICATION / HOLDOUT manifest is ever
    built or touched.
    """
    registry = DatasetRegistry()
    return registry.register(DATASET_DEVELOPMENT, development_scenarios())


# ---------------------------------------------------------------------------
# Per-site descriptive statistics over the DEVELOPMENT corpus
# ---------------------------------------------------------------------------


def _finite(values) -> list:
    out = []
    for v in values:
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.append(float(v))
    return out


def _describe(values: Sequence[float]) -> dict:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    if not vals:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "median": float(statistics.median(vals)),
    }


@dataclass(frozen=True)
class _SiteSummary:
    """One site's scalar feature digest (computed, never a decision)."""

    censored_count: float
    censored_fraction: float
    sign_changes: float
    cross_group_recurrence: float
    cross_epoch_recurrence: float
    local_residual_peak: float
    local_residual_median: float
    robust_local_scale_median: float
    residual_std: float
    residual_span: float
    residual_iqr: float
    snr: float
    post_cal_median: float


def _site_digest(site) -> _SiteSummary:
    local_res = [abs(v) for v in _finite(site.same_cfa_local_residual)]
    local_scale = _finite(site.robust_local_scale)
    residual = _finite(site.temporal_residual_series)
    post = site.post_calibration_residual
    post_abs = [abs(v) for v in _finite(post)] if post is not None else []

    local_median = statistics.median(local_res) if local_res else 0.0
    scale_median = statistics.median(local_scale) if local_scale else 0.0
    snr = (local_median / scale_median) if scale_median > 0 else 0.0

    return _SiteSummary(
        censored_count=float(site.censored_count),
        censored_fraction=float(site.censored_fraction),
        sign_changes=float(site.sign_changes),
        cross_group_recurrence=float(site.cross_group_recurrence),
        cross_epoch_recurrence=float(site.cross_epoch_recurrence),
        local_residual_peak=max(local_res) if local_res else 0.0,
        local_residual_median=local_median,
        robust_local_scale_median=scale_median,
        residual_std=float(statistics.pstdev(residual)) if len(residual) > 1 else 0.0,
        residual_span=float(site.state_spread.span) if site.state_spread.span == site.state_spread.span else 0.0,
        residual_iqr=float(site.state_spread.iqr) if site.state_spread.iqr == site.state_spread.iqr else 0.0,
        snr=snr,
        post_cal_median=(statistics.median(post_abs) if post_abs else float("nan")),
    )


_FEATURE_DESCRIPTIONS = {
    "censored_count": ("count", "number of frames whose value hit the declared hard limit"),
    "censored_fraction": ("fraction", "fraction of frames censored"),
    "sign_changes": ("count", "strict sign crossings of the canonical residual"),
    "cross_group_recurrence": ("count", "recurrence of group median signatures across independent groups"),
    "cross_epoch_recurrence": ("count", "recurrence of epoch median signatures across epochs"),
    "local_residual_peak": ("ADU", "max |site - median(same-CFA neighbours)| over light frames"),
    "local_residual_median": ("ADU", "median |site - median(same-CFA neighbours)| over light frames"),
    "robust_local_scale_median": ("ADU", "median raw MAD of the same-CFA local window"),
    "residual_std": ("ADU", "population std of the canonical residual series"),
    "residual_span": ("ADU", "max - min of the canonical residual series"),
    "residual_iqr": ("ADU", "interquartile range of the canonical residual series"),
    "snr": ("dimensionless", "local_residual_median / robust_local_scale_median"),
    "post_cal_median": ("ADU", "median |value - representative dark| (NaN when no dark reference)"),
}


def compute_feature_summary() -> dict:
    """Compute the descriptive per-class / per-feature summary over DEVELOPMENT.

    Pure and deterministic (fixed seeds, fixed structure). Never touches
    QUALIFICATION or HOLDOUT.
    """
    manifest = build_development_manifest()

    per_class_feature_values: dict = {name: {f: [] for f in _FEATURE_DESCRIPTIONS} for name in DEVELOPMENT_CLASSES}

    for name in DEVELOPMENT_CLASSES:
        for seed in DEVELOPMENT_SEEDS:
            scenario = build_development_scenario(name, seed)
            with tempfile.TemporaryDirectory() as td:
                corpus = generate_corpus(scenario, td)
                dark_ids = [f"dark{i}" for i in range(_DARK_FRAMES)]
                dark = np.median(np.stack([corpus.frame_arrays[i] for i in dark_ids]), axis=0)
                cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
            d = _site_digest(cf.sites[0])
            per_class_feature_values[name]["censored_count"].append(d.censored_count)
            per_class_feature_values[name]["censored_fraction"].append(d.censored_fraction)
            per_class_feature_values[name]["sign_changes"].append(d.sign_changes)
            per_class_feature_values[name]["cross_group_recurrence"].append(d.cross_group_recurrence)
            per_class_feature_values[name]["cross_epoch_recurrence"].append(d.cross_epoch_recurrence)
            per_class_feature_values[name]["local_residual_peak"].append(d.local_residual_peak)
            per_class_feature_values[name]["local_residual_median"].append(d.local_residual_median)
            per_class_feature_values[name]["robust_local_scale_median"].append(d.robust_local_scale_median)
            per_class_feature_values[name]["residual_std"].append(d.residual_std)
            per_class_feature_values[name]["residual_span"].append(d.residual_span)
            per_class_feature_values[name]["residual_iqr"].append(d.residual_iqr)
            per_class_feature_values[name]["snr"].append(d.snr)
            per_class_feature_values[name]["post_cal_median"].append(d.post_cal_median)

    features_summary = {}
    for feature, (unit, description) in _FEATURE_DESCRIPTIONS.items():
        per_class = {
            name: _describe(per_class_feature_values[name][feature])
            for name in DEVELOPMENT_CLASSES
        }
        features_summary[feature] = {"unit": unit, "description": description, "per_class": per_class}

    return {
        "summary_schema": SUMMARY_SCHEMA,
        "summary_version": SUMMARY_VERSION,
        "dataset_role": DATASET_DEVELOPMENT,
        "dataset_manifest_hash": manifest.manifest_hash,
        "corpus": {
            "class_count": len(DEVELOPMENT_CLASSES),
            "seed_count": len(DEVELOPMENT_SEEDS),
            "seeds": list(DEVELOPMENT_SEEDS),
            "scenario_count": len(DEVELOPMENT_CLASSES) * len(DEVELOPMENT_SEEDS),
            "epochs_per_scenario": _EPOCHS,
            "groups_per_epoch": _GROUPS_PER_EPOCH,
            "frames_per_group": _FRAMES_PER_GROUP,
            "light_frames_per_scenario": _EPOCHS * _GROUPS_PER_EPOCH * _FRAMES_PER_GROUP,
            "dark_frames_per_scenario": _DARK_FRAMES,
            "site_coord": [8, 8],
            "sensor_shape": [96, 128],
            "cfa_pattern": "GRBG",
            "size_justification": (
                "20 classes x 5 seeds; 2 epochs x 2 independent groups x 6 light "
                "frames manifest rare-state (ordinal 5), transient (ordinal 3), "
                "star-crossing and sign-changing behaviours while staying fast."
            ),
        },
        "candidate_parameters": {c.candidate_id: c.as_dict() for c in CANDIDATES},
        "features": features_summary,
    }


def build_admission(scenario: ScenarioSpec) -> AdmissionFacts:
    """Build the EXOGENOUS/admission fact set for a scenario.

    These are legitimate acquisition/provenance metadata (sensor identity,
    geometry, calibration presence/representativeness, independence counts) —
    decided upstream, never inferred from pixels. The inference receives them
    as immutable inputs and traverses them untouched.
    """
    bk = compute_bookkeeping(scenario)
    site = scenario.sites[0]
    representative = site.calibration_representativeness == "representative"
    has_additive = any(f.frame_type in ("dark", "bias") for f in scenario.all_frames())
    return AdmissionFacts(
        sensor_identity_resolved=YES,
        geometry_compatible=YES,
        calibration_present=YES if has_additive else NO,
        calibration_representativeness=(REPRESENTATIVE if representative else NOT_REPRESENTATIVE),
        independent_group_count=bk.independent_group_count,
        epoch_count=bk.epoch_count,
    )


def write_feature_summary(path) -> None:
    """Write the DEVELOPMENT feature summary artifact to ``path`` (JSON)."""
    payload = compute_feature_summary()
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "DEVELOPMENT_CLASSES",
    "DEVELOPMENT_SEEDS",
    "SUMMARY_SCHEMA",
    "SUMMARY_VERSION",
    "build_admission",
    "build_development_manifest",
    "build_development_scenario",
    "compute_feature_summary",
    "development_scenarios",
    "write_feature_summary",
]
