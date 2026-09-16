"""Cooperative cancellation and observer fault injection (ARCHITECTURE §3.7)."""

from __future__ import annotations

import numpy as np

from zecalibrator.application.cancellation import (
    CancellationToken,
    ProgressEvent,
    ProgressObserver,
)
from zecalibrator.application.executor import CalibrationRequest, MasterBinding, execute_calibration


def test_pre_cancelled_returns_cancelled_no_data(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=1.0)
    dark = make_frame((4, 4), value=0.0)
    token = CancellationToken()
    token.cancel()
    masters = {"dark": MasterBinding("dark", dark, bias_state="included")}
    r = execute_calibration(light, CalibrationRequest("dark_incl_bias"), masters, cancel=token)
    assert r.status == "CANCELLED"
    assert r.data is None and r.mask is None


def test_midflight_cancellation_no_partial_result(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=1.0)
    dark = make_frame((4, 4), value=0.0)
    token = CancellationToken()

    def _cancel_after_first(event: ProgressEvent):
        if event.phase == "geometry_validation":
            token.cancel()

    obs = ProgressObserver(_cancel_after_first)
    masters = {"dark": MasterBinding("dark", dark, bias_state="included")}
    r = execute_calibration(
        light, CalibrationRequest("dark_incl_bias"), masters,
        cancel=token, progress=obs,
    )
    assert r.status == "CANCELLED"
    assert r.data is None and r.mask is None


def test_failing_observer_does_not_change_arithmetic(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=110.0)
    dark = make_frame((2, 2), value=10.0)

    def _boom(event: ProgressEvent):
        raise RuntimeError("observer bug")

    obs = ProgressObserver(_boom)
    masters = {"dark": MasterBinding("dark", dark, bias_state="included")}
    r = execute_calibration(
        light, CalibrationRequest("dark_incl_bias"), masters, progress=obs,
    )
    assert r.status == "COMPLETED"
    assert np.allclose(r.data, np.full((2, 2), 100.0))
    assert len(obs.observer_errors) >= 1


def test_failed_result_does_not_emit_complete(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((2, 2), value=0.0)
    dark = make_frame((2, 2), value=0.0)
    finc = make_frame((2, 2), value=0.0, exposure_s=1.0)
    fdinc = make_frame((2, 2), value=0.0, exposure_s=1.0)
    events = []

    obs = ProgressObserver(events.append)
    masters = {
        "dark": MasterBinding("dark", dark, bias_state="included"),
        "flat": MasterBinding("flat", finc, flat_form="raw_response"),
        "flat_dark": MasterBinding("flat_dark", fdinc, bias_state="included"),
    }
    r = execute_calibration(
        light, CalibrationRequest("dark_incl_bias", "apply"), masters, progress=obs,
    )
    assert r.status == "FAILED"
    assert all(e.phase != "complete" for e in events)
