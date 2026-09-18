"""Shared test fixtures for G3 science/contract tests.

Provides a synthetic SYNTH-BASE-1 frame factory with complete, qualified
acquisition evidence (never blesses unknown==unknown).
"""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.core.metadata import ImportDeclaration, build_sensor_metadata
from zecalibrator.io.raw_decoder import DecodedFrame


def make_declaration(**overrides) -> ImportDeclaration:
    base = dict(
        source="synthetic_fixture",
        identity="SYNTH-BASE-1",
        version="1.0",
        domain="raw",
        units="ADU",
        detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-MONO",
        gain=100,
        offset=50,
        readout_mode="MODE_A",
        adc_mode="MODE_16",
        binning=(1, 1),
        orientation="identity",
        cfa_phase="mono",
        roi_origin=(0, 0),
        exposure_s=10.0,
        temperature_c=20.0,
        filter="NONE",
        optical_train_id="SYNTH-TRAIN-1",
        bias_exposure_max_s=0.1,
        saturation_limit_adu=60000,
        saturation_evidence="qualified",
    )
    base.update(overrides)
    return ImportDeclaration(**base)


def make_frame(shape=(4, 4), value=0.0, units="ADU", **decl_overrides) -> DecodedFrame:
    """Build a fully qualified synthetic DecodedFrame."""
    decl = make_declaration(sensor_dimensions=shape, **decl_overrides)
    metadata = build_sensor_metadata(
        shape=shape, normalized={}, conflicts=(), cards=(), declaration=decl, units=units
    )
    data = np.full(shape, value, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint16)
    return DecodedFrame(
        data=data, mask=mask, metadata=metadata, stored_dtype="float32",
        bscale=1.0, bzero=0.0, blank=None, hdu=0, precision=None,
    )


@pytest.fixture
def make_frame_fixture():
    return make_frame


@pytest.fixture
def make_declaration_fixture():
    return make_declaration


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication for Qt tests (offscreen; skips without PySide6).

    Using a single QApplication everywhere avoids the Qt constraint that a
    QGuiApplication cannot later become a QApplication (test_icons_qt previously
    created a QGuiApplication; it now shares this QApplication).
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtWidgets
    except ImportError as exc:  # pragma: no cover - exercised without [gui]
        pytest.skip(f"PySide6 ([gui] extra) is not installed: {exc}")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app
