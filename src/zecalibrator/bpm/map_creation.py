"""Bad Pixel Map creation (LOT 1): the exact P4 detector recipe + storage.

This module implements the *creation* mechanism of a Bad Pixel Map from an
**already-selected** dark master. It is the exact P4 detector recipe — per CFA
plane median + ``MAD * 1.4826`` + threshold ``median + K * MAD`` with
``K = 30.0`` — combined with the v1 map policy (every detected site is
``QUALIFIED`` + ``ELIGIBLE_FOR_TARGETED_RECONSTRUCTION``) and the existing
immutable revision store.

It NEVER:

* reads a unit dark or builds a dark master (no median of darks) — it consumes
  an already-selected master (array + identity + metadata);
* exposes ``K`` (or any threshold) to the user (GUI/CLI/setting) — ``K`` is an
  **internal versioned constant**, not returned and not configurable (no grid
  search, no research campaign);
* mutates an existing revision — creation writes a NEW ``candidate`` revision
  then a NEW ``promoted`` revision; the previously-existing revisions remain;
* imports ``research/*`` or any GUI — it is headless.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .errors import BpmError
from .identity import SensorIdentity
from .revision import Revision, SiteRecord, make_revision
from .store import (
    BadPixelDatabase,
    STATE_MISSING,
    STATE_OPENED,
    create_bad_pixel_database,
    load_bad_pixel_database,
)
from .vocabulary import (
    ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION,
    KNOWLEDGE_STATE_QUALIFIED,
    REVISION_STATE_CANDIDATE,
)

#: The map-creation product version (distinct from BPM schema / revision schema).
MAP_CREATION_VERSION = "zecalibrator.bpm.map_creation.v1"

# The detector threshold multiplier K. INTERNAL and versioned: never returned and
# never exposed to any user surface (GUI/CLI/setting); no grid search, no new
# research campaign (§4).
_DETECTOR_K = 30.0
# The MAD -> sigma scale factor of the detector recipe (per CFA plane). INTERNAL.
_DETECTOR_MAD_SCALE = 1.4826

# The four CFA plane parities (py, px).
_CFA_PLANES: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


class MapCreationError(BpmError):
    """A map could not be created (base not openable, invalid input, …)."""


def detect_site_positions(dark_master) -> tuple[tuple[int, int], ...]:
    """Return the sorted sensor ``(y, x)`` positions detected by the exact P4 recipe.

    For each CFA plane ``(py, px)`` the sub-plane ``dark[py::2, px::2]`` is
    thresholded against ``median + K * (MAD * 1.4826)`` (``K = 30.0``, internal),
    where ``MAD = median(|sub - median(sub)|)``; a sub-plane coordinate ``(yy,
    xx)`` strictly above the threshold maps to the sensor coordinate
    ``(2*yy + py, 2*xx + px)``.

    Deterministic: the same master yields the same positions (sorted).
    """
    arr = np.asarray(dark_master)
    if arr.ndim != 2:
        raise MapCreationError(f"dark master must be a 2-D array, got shape {arr.shape}")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise MapCreationError(f"dark master must be at least 2x2, got shape {arr.shape}")

    positions: set[tuple[int, int]] = set()
    for py, px in _CFA_PLANES:
        sub = arr[py::2, px::2]
        plane_med = float(np.median(sub))
        plane_mad = float(np.median(np.abs(sub - plane_med))) * _DETECTOR_MAD_SCALE
        threshold = plane_med + _DETECTOR_K * plane_mad
        ys, xs = np.nonzero(sub > threshold)
        for yy, xx in zip(ys, xs):
            positions.add((int(2 * yy + py), int(2 * xx + px)))
    return tuple(sorted(positions))


def detect_sites(dark_master) -> tuple[SiteRecord, ...]:
    """Detect sites and wrap them with the v1 map policy.

    Every detected site is ``QUALIFIED`` (knowledge) and
    ``ELIGIBLE_FOR_TARGETED_RECONSTRUCTION`` (action) — the v1 map policy. The
    module never re-derives eligibility from light residuals and never degrades
    a detected site to ``NO_ACTION_REQUIRED``.
    """
    return tuple(
        SiteRecord(
            position=pos,
            action_state=ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION,
            knowledge_state=KNOWLEDGE_STATE_QUALIFIED,
        )
        for pos in detect_site_positions(dark_master)
    )


@dataclass(frozen=True)
class SelectedDarkMaster:
    """An already-selected dark master (array + identity + metadata).

    The map-creation module consumes this already-selected master; it never
    reads unit darks and never builds a master median. ``metadata`` is
    descriptive provenance (carried for callers; never persisted in the revision,
    whose content is identity + sites only).
    """

    data: np.ndarray
    identity: SensorIdentity
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim != 2:
            raise MapCreationError(f"dark master must be a 2-D array, got shape {arr.shape}")
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            raise MapCreationError(f"dark master must be at least 2x2, got shape {arr.shape}")
        if not isinstance(self.identity, SensorIdentity):
            raise MapCreationError("identity must be a SensorIdentity")
        geo_shape = tuple(self.identity.geometry.shape)
        if tuple(arr.shape) != geo_shape:
            raise MapCreationError(
                f"dark master shape {tuple(arr.shape)} does not match identity geometry {geo_shape}"
            )
        if not isinstance(self.metadata, Mapping):
            raise MapCreationError("metadata must be a Mapping")
        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class MapCreationResult:
    """The result of creating a map: the new candidate and promoted revisions."""

    candidate: Revision
    promoted: Revision

    @property
    def site_count(self) -> int:
        """The number of sites in the created map."""
        return len(self.promoted.sites)


def _open_or_create_database(root) -> BadPixelDatabase:
    load = load_bad_pixel_database(root)
    if load.state == STATE_MISSING:
        return create_bad_pixel_database(root)
    if load.state == STATE_OPENED:
        assert load.database is not None
        return load.database
    raise MapCreationError(
        f"cannot create a map: Bad Pixel Database is {load.state} ({load.reason_code})"
    )


def create_bad_pixel_map(root, master: SelectedDarkMaster) -> MapCreationResult:
    """Create + store a map from ``master`` as NEW immutable revisions.

    Detects sites with the exact P4 recipe, writes a NEW ``candidate`` revision
    (state ``candidate``) then promotes it to a NEW ``promoted`` revision; the
    previously-existing revisions remain untouched (no mutation). The promoted
    revision becomes immediately selectable by the product lookup (it is the
    latest promoted compatible revision).

    ``root`` is the ``bad_pixel_database_root``; an absent base is created, a
    present base is opened, and a corrupt/invalid/incompatible base is refused
    with a typed :class:`MapCreationError`.
    """
    if not isinstance(master, SelectedDarkMaster):
        raise MapCreationError("master must be a SelectedDarkMaster")
    db = _open_or_create_database(root)
    sites = detect_sites(master.data)
    candidate = make_revision(
        state=REVISION_STATE_CANDIDATE,
        sensor_identity=master.identity,
        sites=sites,
    )
    db.add_revision(candidate)
    promoted = db.promote(candidate)
    return MapCreationResult(candidate=candidate, promoted=promoted)


__all__ = [
    "MAP_CREATION_VERSION",
    "MapCreationError",
    "MapCreationResult",
    "SelectedDarkMaster",
    "create_bad_pixel_map",
    "detect_sites",
    "detect_site_positions",
]
