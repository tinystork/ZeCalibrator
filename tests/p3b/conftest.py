"""Shared fixtures/helpers for tests/p3b.

Adds the repository root to ``sys.path`` so ``research.p3b`` (non-packaged) is
importable, and provides small scenario builders for the LOT1 tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.p3b.model import (  # noqa: E402
    AggregateSpec,
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SensorSpec,
    SiteSpec,
)


def make_sensor(shape=(96, 128), **overrides) -> SensorSpec:
    base = dict(
        instance_id="SYNTH-DET-0001",
        model="SYNTH-S50",
        shape=shape,
        cfa_pattern="GRBG",
        binning=(1, 1),
        roi_origin=(0, 0),
        gain=80.0,
        offset_adu=800.0,
        saturation_limit_adu=60000.0,
        filter="NONE",
    )
    base.update(overrides)
    return SensorSpec(**base)


def make_scenario(name="s", seed=1234, sensor=None, **overrides) -> ScenarioSpec:
    base = dict(name=name, seed=seed, sensor=sensor or make_sensor())
    base.update(overrides)
    return ScenarioSpec(**base)


def single_light_scenario(seed=1234, n_frames=3, shape=(96, 128), **overrides) -> ScenarioSpec:
    """One epoch, one group, ``n_frames`` light frames."""
    sensor = make_sensor(shape=shape)
    frames = tuple(
        FrameSpec(
            frame_id=f"l{i}",
            frame_type="light",
            epoch_id="e0",
            group_id="g0",
            ordinal=i,
        )
        for i in range(n_frames)
    )
    kwargs = dict(
        name="s",
        seed=seed,
        sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
    )
    kwargs.update(overrides)
    return ScenarioSpec(**kwargs)


__all__ = [
    "REPO_ROOT",
    "make_sensor",
    "make_scenario",
    "single_light_scenario",
    "AggregateSpec",
    "EpochSpec",
    "FrameSpec",
    "GroupSpec",
    "ScenarioSpec",
    "SensorSpec",
    "SiteSpec",
]
