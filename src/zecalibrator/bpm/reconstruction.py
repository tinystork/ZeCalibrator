"""Versioned internal reconstruction operator for targeted CFA preparation.

This module defines the **product** reconstruction operator behind a versioned
internal interface. It is deliberately **not** the M74 research witness operator
(`research/p3b/reconstruction.py`, the same-CFA 6×6 / ±4/±6/±8 / 36-donor median
witness): ``docs/SCIENCE_CONTRACT.md`` §13.10 states that the research witness is
"a validated research witness operator, NOT a product algorithm; a future
operator must be chosen, not inherited by default." The product therefore
*chooses* its own operator here, marks it with an explicit versioned identifier,
and never imports ``research/*`` (mission §85).

Contract (mission §38–§40):

* operates on the **post-calibration CFA** (the caller passes only the calibrated
  plane and a donor-usability mask — never the raw light, never the dark);
* **same-CFA donors only** (every donor offset is even, so CFA phase is preserved
  by construction; each donor is still re-verified against the target plane);
* **never recursively uses reconstructed values** — the operator reads *only* the
  immutable input ``cfa`` array and never the (not-yet-built) output; a
  reconstructed site is excluded from the donor population by the caller via the
  ``usable`` mask (an already-reconstructed value can therefore never be read);
* **excludes invalid/censored donors** — a donor is admitted only when
  ``usable[yy, xx]`` is true (the caller derives this from the post-calibration
  DQ, so ``DQ != 0`` — including ``SATURATED`` censoring — is excluded);
* returns :data:`NO_VALUE` when no valid donor exists (never a local fallback).

The chosen product operator is a **median of the 8 nearest same-CFA donors** (the
ring at CFA step 2). This is a *small* neighbourhood (a bounded constant), chosen
to keep the per-site cost ``O(|offsets|)`` (mission §71) and to stay strictly
smaller than the research witness — it is a versioned product choice, documented
here and cited in provenance, not a user setting and not an inherited default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from zecalibrator.core.geometry import array_plane

#: The versioned internal reconstruction-operator interface schema. Distinct from
#: the product version, BPM schema, provenance schema and science contract.
OPERATOR_SCHEMA_VERSION = "zecalibrator.bpm.reconstruction.v1"

#: The single product operator identifier (a name, not a threshold).
PRODUCT_OPERATOR_ID = "SAME_CFA_MEDIAN"

#: The estimator name (the only estimator the product operator implements).
ESTIMATOR_MEDIAN = "median"

#: Sentinel returned when no valid donor exists for a site (never a numeric
#: fallback and never a silent local improvisation).
NO_VALUE = "NO_VALUE"

# The frozen product donor neighbourhood: the 8 nearest same-CFA donors (ring at
# CFA step 2). Every offset is even, which preserves CFA phase, so each donor is
# same-CFA by construction (still re-verified per donor below). This is a bounded
# structural constant (the operator's geometry), never a user setting and never a
# numeric budget.
_DONOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-2, -2), (-2, 0), (-2, 2),
    (0, -2), (0, 2),
    (2, -2), (2, 0), (2, 2),
)


@dataclass(frozen=True)
class ReconstructionOperator:
    """A versioned internal reconstruction operator (not user-selectable).

    ``operator_id`` names the operator; ``version`` is the versioned interface
    schema; ``estimator`` is the estimator the operator applies to its donors;
    ``donor_offsets`` is the frozen same-CFA donor neighbourhood.
    """

    operator_id: str
    version: str
    estimator: str
    donor_offsets: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.operator_id or not isinstance(self.operator_id, str):
            raise ValueError("operator_id must be a non-empty string")
        if self.version != OPERATOR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported operator schema {self.version!r}; "
                f"expected {OPERATOR_SCHEMA_VERSION!r}"
            )
        if self.estimator != ESTIMATOR_MEDIAN:
            raise ValueError(
                f"unsupported estimator {self.estimator!r}; only "
                f"{ESTIMATOR_MEDIAN!r} is implemented"
            )
        offsets = tuple((int(dy), int(dx)) for dy, dx in self.donor_offsets)
        for dy, dx in offsets:
            # Same-CFA by construction: every donor offset must be even in both
            # axes (an odd offset would cross a CFA plane boundary).
            if dy % 2 != 0 or dx % 2 != 0:
                raise ValueError(
                    f"donor offset {(dy, dx)!r} is not same-CFA (must be even in both axes)"
                )
        object.__setattr__(self, "donor_offsets", offsets)


#: The single product operator (frozen). This is the operator the application
#: lot applies; it is never configurable by the user and never the research
#: witness default.
DEFAULT_OPERATOR = ReconstructionOperator(
    operator_id=PRODUCT_OPERATOR_ID,
    version=OPERATOR_SCHEMA_VERSION,
    estimator=ESTIMATOR_MEDIAN,
    donor_offsets=_DONOR_OFFSETS,
)


def _check_cfa(cfa: np.ndarray) -> np.ndarray:
    arr = np.asarray(cfa)
    if arr.ndim != 2:
        raise ValueError(f"cfa must be a 2D plane, got shape {arr.shape}")
    return arr


def donor_values(
    cfa: np.ndarray,
    y: int,
    x: int,
    *,
    cfa_phase: str,
    roi_origin: Tuple[int, int],
    usable: np.ndarray,
    operator: ReconstructionOperator = DEFAULT_OPERATOR,
) -> Tuple[float, ...]:
    """Return the valid same-CFA donor values for a site (in offset order).

    A donor is admitted only when it is in-bounds, same-CFA (re-verified against
    the target plane), and marked usable (``usable[yy, xx]`` true — the caller's
    donor-eligibility mask, which excludes invalid/censored/reconstructed
    positions). Reads only ``cfa`` (the immutable post-calibration plane); never
    a reconstructed output.
    """
    cfa = _check_cfa(cfa)
    ny, nx = cfa.shape
    usable_arr = np.asarray(usable, dtype=bool)
    if usable_arr.shape != cfa.shape:
        raise ValueError(f"usable mask shape {usable_arr.shape} vs cfa {cfa.shape}")
    target = array_plane(y, x, cfa_phase, roi_origin)
    donors: list = []
    for dy, dx in operator.donor_offsets:
        yy, xx = y + dy, x + dx
        if not (0 <= yy < ny and 0 <= xx < nx):
            continue
        if array_plane(yy, xx, cfa_phase, roi_origin) != target:
            continue
        if usable_arr[yy, xx]:
            donors.append(float(cfa[yy, xx]))
    return tuple(donors)


def donors_available(
    cfa: np.ndarray,
    y: int,
    x: int,
    *,
    cfa_phase: str,
    roi_origin: Tuple[int, int],
    usable: np.ndarray,
    operator: ReconstructionOperator = DEFAULT_OPERATOR,
) -> bool:
    """Whether at least one valid same-CFA donor exists for a site.

    The preflight's donor-availability fact: pure and pixel-light (reads only the
    small frozen neighbourhood, never a full-frame scan — mission §71).
    """
    return len(
        donor_values(
            cfa, y, x, cfa_phase=cfa_phase, roi_origin=roi_origin,
            usable=usable, operator=operator,
        )
    ) > 0


def reconstruct_site_pixel(
    cfa: np.ndarray,
    y: int,
    x: int,
    *,
    cfa_phase: str,
    roi_origin: Tuple[int, int],
    usable: np.ndarray,
    operator: ReconstructionOperator = DEFAULT_OPERATOR,
):
    """Reconstruct a single site pixel from its valid same-CFA donors.

    Returns the median (float64) of the valid donor values, or :data:`NO_VALUE`
    when no valid donor exists — never a local improvised fallback (mission §40).
    Does not mutate ``cfa``.
    """
    donors = donor_values(
        cfa, y, x, cfa_phase=cfa_phase, roi_origin=roi_origin,
        usable=usable, operator=operator,
    )
    if not donors:
        return NO_VALUE
    return float(np.median(np.asarray(donors, dtype=np.float64)))


__all__ = [
    "DEFAULT_OPERATOR",
    "ESTIMATOR_MEDIAN",
    "NO_VALUE",
    "OPERATOR_SCHEMA_VERSION",
    "PRODUCT_OPERATOR_ID",
    "ReconstructionOperator",
    "donor_values",
    "donors_available",
    "reconstruct_site_pixel",
]
