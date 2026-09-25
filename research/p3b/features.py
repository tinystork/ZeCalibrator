"""Descriptive feature harness (LOT4) — computed features only, no facts, no decisions.

Internal, non-public, never imported by ``zecalibrator.api.v1``.

This module computes **descriptive features** over materialised synthetic frames
(the physical-ADU arrays produced by :mod:`research.p3b.generator`). It is the
numeric substrate a future detector *would* consume; it neither detects nor
decides anything (``docs/SCIENCE_CONTRACT.md`` §13.2, §13.10).

The **declared** ground-truth facts live in a *separate* module
:mod:`research.p3b.declared_facts`. A feature is **never** mapped to a fact
here, and this module does not import :mod:`research.p3b.qualification_policy`
at all.

Boundary (enforced, tested in ``tests/p3b/test_features.py``):

* **no decision** — no qualification state, no action state, no call to
  ``qualification_policy.evaluate``; this module imports neither the decision
  engine nor the declared-facts module;
* **no threshold** — every numeric constant is registered through
  :func:`research.p3b.parameters.param` and marked ``EXPLORATORY`` (analysis);
  none is a product threshold;
* **no censored inference** — a censored measurement (site value at the declared
  hard limit) is *counted* but never used to derive a quantitative residual:
  those residuals are ``NaN`` and excluded from quantitative aggregates;
* **deterministic** — a pure function of the input arrays; no global RNG.

Feature universe (every feature is descriptive, with an explicit unit/scope):

* ``same_cfa_local_residual``     — per frame; ``site_value − median(same-CFA
  neighbours in a local window)``; ADU.
* ``robust_local_scale``          — per frame; raw median absolute deviation of
  the same-CFA local window; ADU.
* ``temporal_value_series``       — per site × frame; the site's physical value
  over the ordered light frames; ADU.
* ``temporal_residual_series``    — per site × frame (non-censored); the
  time-ordered canonical residual (see below); ADU.
* ``within_group_recurrence``     — per site × group; number of frames whose
  value recurs inside one independent group; count.
* ``cross_group_recurrence``      — per site; recurrence of group signatures
  across independent groups; count.
* ``cross_epoch_recurrence``      — per site; recurrence of epoch signatures
  across epochs; count.
* ``state_spread``                — per site; min/max/span/std/IQR of the
  non-censored canonical residual; ADU. No "states" are ever cut.
* ``sign_changes``                — per site; number of strict sign crossings of
  the non-censored canonical residual; count.
* ``censored_count`` / ``censored_fraction`` — per site; count and fraction of
  frames whose value hit the declared hard limit; count / dimensionless.
* ``post_calibration_residual``   — per frame; ``site_value − representative
  dark`` (NaN where censored; ``None`` when no dark reference is supplied); ADU.
* ``calibration_mismatch``        — per site; ``|old_dark − current_dark|`` at
  the site (``None`` unless both references are supplied); ADU.

**Canonical residual:** the post-calibration residual when a representative dark
reference is supplied, otherwise the same-CFA local residual. This is the series
that ``temporal_residual_series``, ``sign_changes`` and ``state_spread`` consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from .generator import cfa_plane_label
from .parameters import EXPLORATORY, param

# ---------------------------------------------------------------------------
# Tagged numeric constants — EXPLORATORY (analysis only, never a product threshold)
# ---------------------------------------------------------------------------

# Radius (in pixels, Chebyshev window) of the same-CFA local window used by the
# local residual and the robust local scale. An analysis parameter, not a
# product algorithm (SCIENCE §13.10: a future operator is chosen, not inherited).
LOCAL_WINDOW_RADIUS = param("LOCAL_WINDOW_RADIUS", 3, EXPLORATORY)

# Percentiles used to describe the state spread (IQR). Descriptive only.
SPREAD_PERCENTILE_LOW = param("SPREAD_PERCENTILE_LOW", 25.0, EXPLORATORY)
SPREAD_PERCENTILE_HIGH = param("SPREAD_PERCENTILE_HIGH", 75.0, EXPLORATORY)


# ---------------------------------------------------------------------------
# Local (spatial) features — same-CFA geometry, already validated in LOT1/P2
# ---------------------------------------------------------------------------


def _same_cfa_neighbor_values(
    frame: np.ndarray, y: int, x: int, pattern: str, radius: int = LOCAL_WINDOW_RADIUS
) -> np.ndarray:
    """Same-CFA neighbour values (physical ADU) inside the local window.

    Same-CFA = same Bayer plane as the site pixel (via ``cfa_plane_label``, the
    frozen CFA origin). The site pixel itself is excluded. ``radius`` is the
    Chebyshev window half-width.
    """
    ny, nx = frame.shape
    target = cfa_plane_label(y, x, pattern)
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            if cfa_plane_label(yy, xx, pattern) != target:
                continue
            out.append(float(frame[yy, xx]))
    return np.asarray(out, dtype=np.float64)


def same_cfa_local_residual(
    frame: np.ndarray, y: int, x: int, pattern: str, radius: int = LOCAL_WINDOW_RADIUS
) -> float:
    """Local residual: ``site_value − median(same-CFA neighbours)`` (ADU, per frame).

    Returns ``NaN`` when there are no same-CFA neighbours in the window.
    """
    neighbours = _same_cfa_neighbor_values(frame, y, x, pattern, radius)
    if neighbours.size == 0:
        return float("nan")
    reference = float(np.median(neighbours))
    return float(frame[y, x]) - reference


def robust_local_scale(
    frame: np.ndarray, y: int, x: int, pattern: str, radius: int = LOCAL_WINDOW_RADIUS
) -> float:
    """Robust local scale: raw MAD of the same-CFA local window (ADU, per frame).

    ``MAD = median(|neighbour − median(neighbourhood)|)``, unscaled (no Gaussian
    consistency factor), so it stays a pure dispersion descriptor with no hidden
    constant. Returns ``NaN`` with no same-CFA neighbours.
    """
    neighbours = _same_cfa_neighbor_values(frame, y, x, pattern, radius)
    if neighbours.size == 0:
        return float("nan")
    med = float(np.median(neighbours))
    mad = float(np.median(np.abs(neighbours - med)))
    return mad


# ---------------------------------------------------------------------------
# Temporal / recurrence / spread features
# ---------------------------------------------------------------------------


def recurrence_count(values) -> int:
    """Number of values that occur at least twice in ``values``.

    Exact equality over the (already discrete) observed ADU; no tolerance, no
    threshold. ``recurrence_count([5,5,7,5]) == 3``.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return 0
    uniq, counts = np.unique(arr, return_counts=True)
    return int(np.sum(counts[counts >= 2]))


def sign_changes(residuals) -> int:
    """Number of strict sign crossings in a residual series.

    A crossing is counted only between two consecutive **non-zero** residuals of
    opposite sign; zero residuals are ignored (never invented). Non-finite
    entries are dropped first.
    """
    arr = np.asarray(residuals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0
    signs = np.sign(arr)
    changes = 0
    for i in range(1, arr.size):
        if signs[i] != signs[i - 1] and signs[i] != 0.0 and signs[i - 1] != 0.0:
            changes += 1
    return int(changes)


@dataclass(frozen=True)
class StateSpread:
    """Descriptive spread of a series (no "states" are ever cut)."""

    min: float
    max: float
    span: float
    std: float
    iqr: float


def state_spread(values) -> StateSpread:
    """Descriptive spread statistics over the finite values of ``values``.

    min / max / span (max−min) / std / IQR. Purely descriptive; no clustering,
    no binning, no "state" boundary. All-``NaN`` when there are no finite values.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        nan = float("nan")
        return StateSpread(nan, nan, nan, nan, nan)
    lo = float(np.percentile(arr, SPREAD_PERCENTILE_LOW))
    hi = float(np.percentile(arr, SPREAD_PERCENTILE_HIGH))
    return StateSpread(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        span=float(np.max(arr) - np.min(arr)),
        std=float(np.std(arr)),
        iqr=hi - lo,
    )


def censored_count_and_fraction(values, hard_limit) -> Tuple[int, float]:
    """Count and fraction of censored measurements (value at/above ``hard_limit``).

    ``hard_limit`` is a *declared* acquisition boundary, not a detection
    threshold: a value at the limit means the acquisition path truncated it.
    Counting them is descriptive; nothing quantitative is derived from them.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0, 0.0
    censored = arr >= hard_limit
    count = int(np.count_nonzero(censored))
    return count, float(count) / arr.size


# ---------------------------------------------------------------------------
# Calibration-facing features (descriptive, not a calibration decision)
# ---------------------------------------------------------------------------


def post_calibration_residual(values, dark_values) -> np.ndarray:
    """Elementwise residual after representative calibration: ``value − dark``.

    Pure arithmetic; no threshold, no decision. The caller must mask censored
    frames (this function does not know the declared hard limit).
    """
    values = np.asarray(values, dtype=np.float64)
    dark = np.asarray(dark_values, dtype=np.float64)
    return values - dark


def calibration_mismatch(old_dark, current_dark, y: int, x: int) -> float:
    """Declared-evidence disagreement between two calibrations at one site.

    ``|old_dark[y, x] − current_dark[y, x]|`` (ADU). Descriptive only: it reports
    a measured disagreement, never decides that one calibration is wrong.
    """
    return float(abs(float(old_dark[y, x]) - float(current_dark[y, x])))


# ---------------------------------------------------------------------------
# Per-site feature container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteFeatures:
    """All descriptive features for one site (computed, never declared, never a decision)."""

    site_id: str
    x: int
    y: int
    cfa_plane: str
    # Ordered light frames (time order) — the scope of every per-frame feature.
    frame_ids: Tuple[str, ...]
    group_ids: Tuple[str, ...]
    epoch_ids: Tuple[str, ...]
    # Per-frame series (aligned with frame_ids).
    temporal_value_series: Tuple[float, ...]
    same_cfa_local_residual: Tuple[float, ...]  # NaN where censored / no neighbour
    robust_local_scale: Tuple[float, ...]
    post_calibration_residual: Optional[Tuple[float, ...]]  # None without dark ref; NaN where censored
    # Non-censored canonical residual + its frame scope.
    temporal_residual_series: Tuple[float, ...]
    temporal_residual_frame_ids: Tuple[str, ...]
    # Recurrence.
    within_group_recurrence: Tuple[int, ...]  # per independent group, in first-appearance order
    within_group_ids: Tuple[str, ...]
    cross_group_recurrence: int
    cross_epoch_recurrence: int
    # Spread / sign / censoring.
    state_spread: StateSpread
    sign_changes: int
    censored_count: int
    censored_fraction: float
    calibration_mismatch: Optional[float]


@dataclass(frozen=True)
class ComputedFeatures:
    """The full descriptive feature set for a corpus (computed, never a decision)."""

    scenario_name: str
    seed: int
    sites: Tuple[SiteFeatures, ...]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_features(
    scenario,
    frame_arrays: Mapping[str, np.ndarray],
    *,
    dark_reference: Optional[np.ndarray] = None,
    old_dark_reference: Optional[np.ndarray] = None,
    current_dark_reference: Optional[np.ndarray] = None,
) -> ComputedFeatures:
    """Compute descriptive features for every site of ``scenario``.

    ``frame_arrays`` maps ``frame_id`` → physical-ADU array (e.g.
    ``CorpusResult.frame_arrays``). ``dark_reference`` / ``old_dark_reference`` /
    ``current_dark_reference`` are optional representative-dark arrays (same
    shape as the frames) that enable the calibration-facing features; they are
    *inputs*, never computed or chosen here.

    Pure and deterministic: a function of the inputs only.
    """
    sensor = scenario.sensor
    pattern = sensor.cfa_pattern
    light = scenario.light_frames()

    frame_ids = tuple(f.frame_id for f in light)
    group_ids = tuple(f.group_id for f in light)
    epoch_ids = tuple(f.epoch_id for f in light)
    # Independent-group identity is the (epoch_id, group_id) PAIR (SCIENCE §13.5).
    group_keys = tuple((f.epoch_id, f.group_id) for f in light)

    dark = None
    if dark_reference is not None:
        dark = np.asarray(dark_reference, dtype=np.float64)

    sites_out = []
    for site in scenario.sites:
        y = site.y
        x = site.x
        plane = cfa_plane_label(y, x, pattern)
        hard = site.hard_limit_adu if site.hard_limit_adu is not None else sensor.saturation_limit_adu

        values_arr = np.asarray(
            [float(frame_arrays[fid][y, x]) for fid in frame_ids], dtype=np.float64
        )
        censored_mask = values_arr >= hard
        censored_count = int(np.count_nonzero(censored_mask))
        censored_fraction = float(censored_count) / values_arr.size if values_arr.size else 0.0

        local_res = []
        local_scale = []
        for i, fid in enumerate(frame_ids):
            arr = frame_arrays[fid]
            lr = same_cfa_local_residual(arr, y, x, pattern)
            if censored_mask[i]:
                lr = float("nan")
            local_res.append(lr)
            local_scale.append(robust_local_scale(arr, y, x, pattern))

        post_res = None
        if dark is not None:
            post = []
            for i, fid in enumerate(frame_ids):
                if censored_mask[i]:
                    post.append(float("nan"))
                else:
                    post.append(float(frame_arrays[fid][y, x]) - float(dark[y, x]))
            post_res = tuple(post)

        canonical = post_res if post_res is not None else tuple(local_res)

        temporal_residual = []
        temporal_residual_frame_ids = []
        for i in range(len(frame_ids)):
            v = canonical[i]
            if math.isfinite(v):
                temporal_residual.append(v)
                temporal_residual_frame_ids.append(frame_ids[i])

        # Recurrence — computed over non-censored values only (no quantitative
        # use of censored measurements).
        seen_groups = []
        for gk in group_keys:
            if gk not in seen_groups:
                seen_groups.append(gk)
        within_group_recurrence = []
        within_group_ids = []
        for gk in seen_groups:
            group_vals = [
                float(values_arr[i])
                for i in range(len(frame_ids))
                if group_keys[i] == gk and not censored_mask[i]
            ]
            within_group_recurrence.append(recurrence_count(group_vals))
            within_group_ids.append(f"{gk[0]}/{gk[1]}")

        group_signatures = []
        for gk in seen_groups:
            group_vals = [
                float(values_arr[i])
                for i in range(len(frame_ids))
                if group_keys[i] == gk and not censored_mask[i]
            ]
            if group_vals:
                group_signatures.append(float(np.median(np.asarray(group_vals))))
        cross_group_recurrence = recurrence_count(group_signatures)

        seen_epochs = []
        for e in epoch_ids:
            if e not in seen_epochs:
                seen_epochs.append(e)
        epoch_signatures = []
        for e in seen_epochs:
            epoch_vals = [
                float(values_arr[i])
                for i in range(len(frame_ids))
                if epoch_ids[i] == e and not censored_mask[i]
            ]
            if epoch_vals:
                epoch_signatures.append(float(np.median(np.asarray(epoch_vals))))
        cross_epoch_recurrence = recurrence_count(epoch_signatures)

        mismatch = None
        if old_dark_reference is not None and current_dark_reference is not None:
            mismatch = calibration_mismatch(old_dark_reference, current_dark_reference, y, x)

        sites_out.append(
            SiteFeatures(
                site_id=site.site_id,
                x=x,
                y=y,
                cfa_plane=plane,
                frame_ids=frame_ids,
                group_ids=group_ids,
                epoch_ids=epoch_ids,
                temporal_value_series=tuple(float(v) for v in values_arr),
                same_cfa_local_residual=tuple(local_res),
                robust_local_scale=tuple(local_scale),
                post_calibration_residual=post_res,
                temporal_residual_series=tuple(temporal_residual),
                temporal_residual_frame_ids=tuple(temporal_residual_frame_ids),
                within_group_recurrence=tuple(within_group_recurrence),
                within_group_ids=tuple(within_group_ids),
                cross_group_recurrence=cross_group_recurrence,
                cross_epoch_recurrence=cross_epoch_recurrence,
                state_spread=state_spread(temporal_residual),
                sign_changes=sign_changes(temporal_residual),
                censored_count=censored_count,
                censored_fraction=censored_fraction,
                calibration_mismatch=mismatch,
            )
        )

    return ComputedFeatures(scenario.name, scenario.seed, tuple(sites_out))


__all__ = [
    "ComputedFeatures",
    "LOCAL_WINDOW_RADIUS",
    "SiteFeatures",
    "SPREAD_PERCENTILE_HIGH",
    "SPREAD_PERCENTILE_LOW",
    "StateSpread",
    "calibration_mismatch",
    "censored_count_and_fraction",
    "compute_features",
    "post_calibration_residual",
    "recurrence_count",
    "robust_local_scale",
    "same_cfa_local_residual",
    "sign_changes",
    "state_spread",
]
