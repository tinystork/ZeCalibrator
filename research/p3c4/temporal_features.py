"""P3C-4 LOT 3 — temporal feature extraction (measurement side, no truth).

Research-only, internal, non-public. This module measures, at a **fixed sensor
coordinate**, the temporal evidence that the LOT 3 inference consumes. It is the
measurement side of the temporal proof: it reads pixels and acquisition
structure only, and implements **no threshold, no decision, no candidate and no
score** — those belong to :mod:`research.p3c4.temporal_inference`.

What is measured (SCIENCE §15, the *necessary* features — no 25-feature grab bag):

* ``frame_coord_residual`` — the residual at the fixed coordinate, per light
  frame. This is the established LOT4 local residual (``same_cfa_local_residual``:
  site value minus the median of same-CFA neighbours). ``NaN`` where censored.
* ``frame_peak_*`` — the residual peak **location and magnitude** in a square
  window around the coordinate, per light frame. This is the movement signal: a
  stationary signature keeps its peak at the coordinate; a celestial confounder
  that follows the sky moves its peak to a neighbouring coordinate. ``NaN`` where
  censored.
* ``frame_epoch_ids`` / ``frame_group_ids`` / ``frame_censored`` — acquisition
  structure and censoring bookkeeping, per light frame.

The residual here is deliberately the *same* primitive the LOT4
:func:`research.p3b.features.same_cfa_local_residual` uses, so the temporal proof
and the existing spatial features share one residual definition; the temporal
proof never introduces its own amplitude scale.

Boundary: this module reads **no truth**. It never imports ``declared_facts``,
``catalog``, ``fixtures``, ``metrics``, or any oracle; it never references a
class label, a scenario name or an expected state. It is measurement-side (like
:mod:`research.p3b.features`), which is why it may import the LOT4 residual
primitive and ``numpy`` — and why it is **not** re-exported by
``research.p3c4/__init__.py`` (so importing the *contract* package still pulls
no ``research.p3b.*`` into ``sys.modules``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from research.p3b.features import same_cfa_local_residual

# Measurement geometry (not a decision threshold, not a product threshold): the
# half-width, in pixels, of the square window scanned for the residual peak. It
# is sized to cover the corpus's per-epoch sky dither (``SKY_DITHER_PX = 6``)
# with margin, so a moved celestial confounder is still found inside the window.
DEPARTURE_WINDOW_RADIUS_PX = 8


@dataclass(frozen=True)
class TemporalSiteFeatures:
    """Measured temporal evidence at one fixed sensor coordinate (no decision).

    All series are aligned over the ordered light frames. ``NaN`` marks a
    censored frame (value at/above the hard limit): it contributes nothing
    quantitative (SCIENCE §13.4) and is excluded by the inference.
    """

    coordinate_x: int
    coordinate_y: int
    frame_epoch_ids: Tuple[str, ...]
    frame_group_ids: Tuple[str, ...]
    frame_censored: Tuple[bool, ...]
    frame_coord_residual: Tuple[float, ...]   # residual at the coordinate, per frame
    frame_peak_dy: Tuple[float, ...]          # residual-peak offset (dy), per frame
    frame_peak_dx: Tuple[float, ...]          # residual-peak offset (dx), per frame
    frame_peak_residual: Tuple[float, ...]    # residual-peak magnitude, per frame


def _window_peak(
    frame: np.ndarray, x: int, y: int, pattern: str, radius: int
) -> Tuple[float, float, float]:
    """Residual peak ``(value, dy, dx)`` in a ``(2*radius+1)`` window around ``(x, y)``.

    The peak is the largest local residual over the window (the coordinate itself
    included). The offset is relative to the coordinate, so a stationary site has
    peak offset ``(0, 0)`` and a moved confounder has a non-zero offset.
    """
    ny, nx = frame.shape
    best = float("-inf")
    best_dy = 0.0
    best_dx = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            r = same_cfa_local_residual(frame, yy, xx, pattern)
            if r != r:  # NaN (no same-CFA neighbours) — never a peak
                continue
            if r > best:
                best = r
                best_dy = float(dy)
                best_dx = float(dx)
    return best, best_dy, best_dx


def compute_temporal_features(
    frame_arrays: Mapping[str, np.ndarray],
    light_frames: Sequence,
    x: int,
    y: int,
    *,
    cfa_pattern: str,
    hard_limit: float,
    departure_radius_px: int = DEPARTURE_WINDOW_RADIUS_PX,
) -> TemporalSiteFeatures:
    """Measure the temporal evidence at ``(x, y)`` over the ordered light frames.

    ``light_frames`` is the ordered light-frame list (duck-typed: each entry has
    ``frame_id``, ``epoch_id``, ``group_id``). ``frame_arrays`` maps ``frame_id``
    to the physical-ADU array. ``hard_limit`` is the declared acquisition hard
    limit used to flag censored frames. Pure and deterministic.
    """
    epoch_ids = []
    group_ids = []
    censored = []
    coord_residual = []
    peak_dy = []
    peak_dx = []
    peak_residual = []

    for f in light_frames:
        arr = np.asarray(frame_arrays[f.frame_id], dtype=np.float64)
        value = float(arr[y, x])
        is_censored = value >= hard_limit

        epoch_ids.append(f.epoch_id)
        group_ids.append(f.group_id)
        censored.append(is_censored)

        if is_censored:
            coord_residual.append(float("nan"))
            peak_dy.append(float("nan"))
            peak_dx.append(float("nan"))
            peak_residual.append(float("nan"))
            continue

        coord_residual.append(
            float(same_cfa_local_residual(arr, y, x, cfa_pattern))
        )
        pk_val, pk_dy, pk_dx = _window_peak(
            arr, x, y, cfa_pattern, departure_radius_px
        )
        peak_residual.append(pk_val)
        peak_dy.append(pk_dy)
        peak_dx.append(pk_dx)

    return TemporalSiteFeatures(
        coordinate_x=int(x),
        coordinate_y=int(y),
        frame_epoch_ids=tuple(epoch_ids),
        frame_group_ids=tuple(group_ids),
        frame_censored=tuple(censored),
        frame_coord_residual=tuple(coord_residual),
        frame_peak_dy=tuple(peak_dy),
        frame_peak_dx=tuple(peak_dx),
        frame_peak_residual=tuple(peak_residual),
    )


__all__ = [
    "DEPARTURE_WINDOW_RADIUS_PX",
    "TemporalSiteFeatures",
    "compute_temporal_features",
]
