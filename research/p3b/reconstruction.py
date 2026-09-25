"""Frozen research-witness reconstruction operator (LOT6).

The single, frozen reconstruction operator used by the net-benefit harness. It
is the M74 research witness operator named in ``docs/SCIENCE_CONTRACT.md``
§13.10: a transparent, same-CFA 6×6 donor grid (offsets ±4 / ±6 / ±8 in each
axis, 36 donors) reduced by a median estimator.

This module:

* carries the explicit operator identifier ``RESEARCH_WITNESS_ONLY``;
* defines exactly ONE operator — it does **not** search over operators
  and does **not** tune its neighbourhood size;
* is internal and non-public — never imported by ``zecalibrator.api.v1``, never
  the product default algorithm, and never exposed publicly.

The operator reconstructs a single site *pixel*: it replaces the site value by
the median of its same-CFA donor values at the frozen offsets. Every offset is
even, which preserves CFA phase, so every donor is same-CFA by construction.
"""

from __future__ import annotations

import numpy as np

from .generator import cfa_plane_label
from .parameters import SYNTHETIC_GENERATOR_PARAMETER, param

# Explicit operator identifier — a *name*, not a threshold and not a numeric
# constant: the research witness ONLY, never the product operator.
RESEARCH_WITNESS_ONLY = "RESEARCH_WITNESS_ONLY"

# A stable, versioned identifier so the frozen construction can be referenced in
# provenance without ambiguity.
OPERATOR_VERSION = "research-witness-1"

# The frozen donor-grid base offsets (same-CFA by construction: all even).
# Marked SYNTHETIC_GENERATOR_PARAMETER: these numeric constants *define* the
# frozen operator's donor geometry; they are not a product threshold.
RESEARCH_WITNESS_BASE_OFFSETS = param(
    "RESEARCH_WITNESS_BASE_OFFSETS",
    (-8, -6, -4, 4, 6, 8),
    SYNTHETIC_GENERATOR_PARAMETER,
)

# The full 6×6 donor grid (36 offsets) — a structural cross-product of the
# frozen base offsets, not a new numeric constant.
DONOR_OFFSETS = tuple(
    (dy, dx)
    for dy in RESEARCH_WITNESS_BASE_OFFSETS
    for dx in RESEARCH_WITNESS_BASE_OFFSETS
)


class ReconstructionError(ValueError):
    """The frozen operator could not reconstruct (e.g. no same-CFA donor)."""


def donor_offsets_available(shape, y, x, pattern) -> bool:
    """Whether every frozen donor offset is in-bounds and same-CFA at the site.

    This is the *geometric* donor-availability fact consumed by the LOT3
    preflight as its default donor availability: a site is reconstructable on a
    frame only if all 36 donors exist. Pure; reads no pixel values.
    """
    ny, nx = shape
    target = cfa_plane_label(y, x, pattern)
    for dy, dx in DONOR_OFFSETS:
        yy, xx = y + dy, x + dx
        if not (0 <= yy < ny and 0 <= xx < nx):
            return False
        if cfa_plane_label(yy, xx, pattern) != target:
            return False
    return True


def reconstruct_site_pixel(frame, y, x, pattern, operator_id=RESEARCH_WITNESS_ONLY):
    """Reconstruct a single site pixel with the frozen research witness.

    ``frame`` is the post-calibration CFA (physical ADU, same geometry as the
    light). Returns the reconstructed value — the median of the same-CFA donor
    values at the frozen offsets. Does not mutate ``frame``.

    Only ``RESEARCH_WITNESS_ONLY`` is a valid operator id; any other id is
    refused (there is no second operator, and none is searched for).
    """
    if operator_id != RESEARCH_WITNESS_ONLY:
        raise ValueError(
            f"unknown reconstruction operator {operator_id!r}; only "
            f"{RESEARCH_WITNESS_ONLY!r} exists in the net-benefit harness"
        )
    ny, nx = frame.shape
    target = cfa_plane_label(y, x, pattern)
    donors = []
    for dy, dx in DONOR_OFFSETS:
        yy, xx = y + dy, x + dx
        if 0 <= yy < ny and 0 <= xx < nx and cfa_plane_label(yy, xx, pattern) == target:
            donors.append(float(frame[yy, xx]))
    if not donors:
        raise ReconstructionError(f"no same-CFA donor available for site ({y}, {x})")
    return float(np.median(np.asarray(donors, dtype=np.float64)))


__all__ = [
    "DONOR_OFFSETS",
    "OPERATOR_VERSION",
    "RESEARCH_WITNESS_BASE_OFFSETS",
    "RESEARCH_WITNESS_ONLY",
    "ReconstructionError",
    "donor_offsets_available",
    "reconstruct_site_pixel",
]
