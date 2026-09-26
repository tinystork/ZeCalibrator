"""Shared fixtures for the Bad Pixel Database product-model tests.

Builds the product identity and site/revision value objects directly from the
frozen core types (no ``research/*`` import, no pixel application).
"""

from __future__ import annotations

import pytest

from zecalibrator.bpm.identity import ReadoutContext, SensorIdentity
from zecalibrator.bpm.revision import SiteRecord, make_revision
from zecalibrator.bpm.vocabulary import (
    ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION,
    ACTION_STATE_NO_ACTION_REQUIRED,
    REVISION_STATE_PROMOTED,
)
from zecalibrator.core.descriptors import DetectorIdentity
from zecalibrator.core.geometry import Geometry


def make_identity(
    *,
    detector_instance_id="DET-0001",
    detector_model="MODEL-X",
    serial=None,
    shape=(1080, 1920),
    sensor_dimensions=None,
    binning=(1, 1),
    roi_origin=(0, 0),
    cfa_phase="GRBG",
    gain=100,
    offset=50,
    readout_mode="MODE_A",
    adc_mode="MODE_16",
) -> SensorIdentity:
    return SensorIdentity(
        detector=DetectorIdentity(
            detector_instance_id=detector_instance_id,
            detector_model=detector_model,
            serial=serial,
        ),
        geometry=Geometry(
            shape=shape,
            # full-frame identity defaults to the fixture shape (SYNTH-BASE-1
            # convention); the normative geometry contract requires it known.
            sensor_dimensions=shape if sensor_dimensions is None else sensor_dimensions,
            binning=binning,
            roi_origin=roi_origin,
            roi_extent=shape,
            orientation="identity",
            cfa_phase=cfa_phase,
        ),
        readout=ReadoutContext(
            gain=gain, offset=offset, readout_mode=readout_mode, adc_mode=adc_mode,
        ),
    )


def make_site(y, x, action_state=ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION, knowledge_state="KNOWN") -> SiteRecord:
    return SiteRecord(position=(y, x), action_state=action_state, knowledge_state=knowledge_state)


def make_rev(identity, *, state=REVISION_STATE_PROMOTED, sites=(), sequence=0):
    return make_revision(state=state, sensor_identity=identity, sites=sites, sequence=sequence)


@pytest.fixture
def identity_factory():
    return make_identity


@pytest.fixture
def site_factory():
    return make_site


@pytest.fixture
def revision_factory():
    return make_rev
