"""Internal, non-public data model for the P3B synthetic corpus (LOT1).

These types are *internal*: non-public, non-capability, and never imported by
``zecalibrator.api.v1``. They declare the ground truth of a scenario. Every
expected state is declared explicitly here (and serialised into the manifest);
**no** state is ever inferred from a filename.

The model mirrors the independence bookkeeping invariants of
``SCIENCE_CONTRACT.md`` §13.5:

* ``frame_count``            — raw FITS frames materialised (science + calibration
                               + derived aggregates).
* ``observation_count``      — science light frames (acquisitions of the sky).
* ``independent_group_count``— distinct science lineages (groups containing >= 1
                               light frame); 20 frames from one sequence is
                               *one* group, not 20.
* ``epoch_count``            — distinct epochs (acquisition campaigns).

A derived aggregate (e.g. a master median of frames) is a *frame* but **not** a
new observation and **not** a new independent group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .catalog import CLASS_NAMES, resolve_class
from .parameters import (
    DEFAULT_EXPOSURE_S,
    DEFAULT_GAIN,
    DEFAULT_OFFSET_ADU,
    DEFAULT_SATURATION_LIMIT_ADU,
    DEFAULT_TEMPERATURE_C,
)

# Declared ground-truth label vocabularies (truth labels, not product enums and
# not thresholds). Qualification states are a LOT1 truth vocabulary; the action
# states below follow the ordered ladder of SCIENCE §13.7.
QUALIFICATION_STATES: Tuple[str, ...] = (
    "UNQUALIFIED",
    "QUALIFIED_PERSISTENT",
    "QUALIFIED_INTERMITTENT",
    "QUALIFIED_TRANSIENT",
    "CENSORED",
)

ACTION_STATES: Tuple[str, ...] = (
    "REQUALIFY_ADDITIVE_CALIBRATION",
    "ABSTAIN_CENSORED",
    "ABSTAIN_INCONSISTENT",
    "ABSTAIN_INSUFFICIENT_EVIDENCE",
    "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
    "NO_ACTION_REQUIRED",
)

FRAME_TYPES: Tuple[str, ...] = ("light", "dark", "bias", "flat", "master")


@dataclass(frozen=True)
class SensorSpec:
    """Declared sensor identity and geometry (SCIENCE §4, §5).

    ``shape`` is ``(ny, nx)``; ``array[y, x]`` corresponds to sensor pixel
    ``(x, y)``. ``cfa_pattern`` is a Bayer phase or ``"mono"``. Coordinates are
    native, unbinned, base-0: ``binning=(1,1)`` and ``roi_origin=(0,0)``.
    """

    instance_id: str
    model: str
    shape: Tuple[int, int] = (96, 128)
    cfa_pattern: str = "GRBG"
    binning: Tuple[int, int] = (1, 1)
    roi_origin: Tuple[int, int] = (0, 0)
    gain: float = DEFAULT_GAIN
    offset_adu: float = DEFAULT_OFFSET_ADU
    saturation_limit_adu: float = DEFAULT_SATURATION_LIMIT_ADU
    filter: str = "NONE"

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(v <= 0 for v in self.shape):
            raise ValueError(f"shape must be a positive (ny, nx), got {self.shape!r}")
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        object.__setattr__(self, "binning", tuple(int(v) for v in self.binning))
        object.__setattr__(self, "roi_origin", tuple(int(v) for v in self.roi_origin))


@dataclass(frozen=True)
class SiteSpec:
    """Ground-truth declaration of one synthetic site.

    ``x`` / ``y`` are *native base-0 sensor coordinates*: the site lives at
    ``array[y, x]``. ``cfa_class`` must be a catalogue class name (validated).
    Expected qualification/action states and calibration representativeness are
    declared here, never derived from the filename.
    """

    site_id: str
    x: int
    y: int
    cfa_class: str
    expected_qualification_state: Optional[str] = None
    expected_action_state: Optional[str] = None
    calibration_representativeness: str = "representative"
    censored: bool = False
    hard_limit_adu: Optional[float] = None
    params: Tuple[Tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        entry = resolve_class(self.cfa_class)  # validates + raises UnknownClassError
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))
        object.__setattr__(self, "params", tuple((k, v) for k, v in self.params))
        # Default the expected states from the catalogue entry when not given.
        if self.expected_qualification_state is None:
            object.__setattr__(
                self, "expected_qualification_state",
                entry.default_expected_qualification_state,
            )
        if self.expected_action_state is None:
            object.__setattr__(
                self, "expected_action_state", entry.default_expected_action_state,
            )

    def param(self, key: str, default=None):
        for k, v in self.params:
            if k == key:
                return v
        return default


@dataclass(frozen=True)
class FrameSpec:
    """One raw frame to materialise."""

    frame_id: str
    frame_type: str  # one of FRAME_TYPES (light/dark/bias/flat/master)
    epoch_id: str
    group_id: str
    ordinal: int  # frame ordinal within the group/sequence
    exposure_s: float = DEFAULT_EXPOSURE_S
    temperature_c: float = DEFAULT_TEMPERATURE_C

    def __post_init__(self) -> None:
        if self.frame_type not in FRAME_TYPES:
            raise ValueError(
                f"frame_type must be one of {FRAME_TYPES}, got {self.frame_type!r}"
            )
        object.__setattr__(self, "ordinal", int(self.ordinal))


@dataclass(frozen=True)
class GroupSpec:
    """One acquisition lineage (a set of correlated frames)."""

    group_id: str
    frames: Tuple[FrameSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))


@dataclass(frozen=True)
class EpochSpec:
    """One acquisition campaign (a set of groups)."""

    epoch_id: str
    groups: Tuple[GroupSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))


@dataclass(frozen=True)
class AggregateSpec:
    """A derived aggregate (master) built from existing frames.

    It is a *frame* (counted in ``frame_count``) but **not** a new observation
    and **not** a new independent group (SCIENCE §13.5).
    """

    aggregate_id: str
    method: str  # "median" | "mean"
    constituent_frame_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method not in ("median", "mean"):
            raise ValueError(f"method must be 'median' or 'mean', got {self.method!r}")
        object.__setattr__(self, "constituent_frame_ids", tuple(self.constituent_frame_ids))


@dataclass(frozen=True)
class ScenarioSpec:
    """A complete deterministic synthetic scenario."""

    name: str
    seed: int
    sensor: SensorSpec
    epochs: Tuple[EpochSpec, ...] = ()
    sites: Tuple[SiteSpec, ...] = ()
    aggregates: Tuple[AggregateSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "epochs", tuple(self.epochs))
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "aggregates", tuple(self.aggregates))

    def all_frames(self) -> Tuple[FrameSpec, ...]:
        out: list[FrameSpec] = []
        for epoch in self.epochs:
            for group in epoch.groups:
                out.extend(group.frames)
        return tuple(out)

    def light_frames(self) -> Tuple[FrameSpec, ...]:
        return tuple(f for f in self.all_frames() if f.frame_type == "light")


@dataclass(frozen=True)
class Bookkeeping:
    """Independence bookkeeping counters (SCIENCE §13.5)."""

    frame_count: int
    observation_count: int
    independent_group_count: int
    epoch_count: int

    def to_dict(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "observation_count": self.observation_count,
            "independent_group_count": self.independent_group_count,
            "epoch_count": self.epoch_count,
        }


def compute_bookkeeping(scenario: ScenarioSpec) -> Bookkeeping:
    """Compute the four independence counters from a scenario's structure."""
    all_frames = scenario.all_frames()
    light_frames = scenario.light_frames()

    science_groups: set[str] = set()
    for epoch in scenario.epochs:
        for group in epoch.groups:
            if any(f.frame_type == "light" for f in group.frames):
                science_groups.add(group.group_id)

    frame_count = len(all_frames) + len(scenario.aggregates)
    observation_count = len(light_frames)
    independent_group_count = len(science_groups)
    epoch_count = len(scenario.epochs)

    return Bookkeeping(
        frame_count=frame_count,
        observation_count=observation_count,
        independent_group_count=independent_group_count,
        epoch_count=epoch_count,
    )


__all__ = [
    "ACTION_STATES",
    "FRAME_TYPES",
    "QUALIFICATION_STATES",
    "AggregateSpec",
    "Bookkeeping",
    "EpochSpec",
    "FrameSpec",
    "GroupSpec",
    "ScenarioSpec",
    "SensorSpec",
    "SiteSpec",
    "compute_bookkeeping",
]
