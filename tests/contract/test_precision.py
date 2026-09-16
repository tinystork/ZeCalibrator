"""Documented float32 precision envelope (SCIENCE §2.5)."""

from __future__ import annotations

import numpy as np

from zecalibrator.core.calibrate import calibrate_light
from zecalibrator.core.precision import (
    FLOAT32_EXACT_INT_BOUND,
    FLOAT32_REL_EPS,
    integer_exceeds_float32_exact_range,
    measure_float32_roundoff,
)


def test_float32_exact_integer_bound_fact():
    p24 = 2 ** 24
    assert float(np.float32(p24)) == float(p24)
    assert float(np.float32(p24 + 1)) != float(p24 + 1)
    assert FLOAT32_EXACT_INT_BOUND == p24


def test_integer_exceeds_range():
    assert not integer_exceeds_float32_exact_range(2 ** 24)
    assert integer_exceeds_float32_exact_range(2 ** 24 + 1)
    assert not integer_exceeds_float32_exact_range(1.5)  # non-integer


def test_precision_against_independent_float64_reference():
    # float64 reference computed independently (explicit Python arithmetic).
    L = np.array([[110.0, 50.0], [210.0, -2.0]])
    dark = np.full((2, 2), 10.0)
    R = np.array([[1.0, 0.5], [2.0, 1.0]])
    reference = (L - dark) / R  # float64
    result = calibrate_light(
        L, additive_mode="dark_incl_bias", flat_mode="apply",
        dark_inc=dark, flat_response=R, flat_valid=np.ones((2, 2), bool),
        saturation_limit=60000.0,
    )
    got = result.data.astype(np.float64)
    abs_err = np.abs(got - reference)
    rel_err = abs_err / np.maximum(np.abs(reference), 1e-30)
    assert np.max(abs_err) == 0.0  # these ADU are exactly representable
    assert np.max(rel_err) == 0.0


def test_precision_envelope_not_loosened():
    # Non-integer scaled values carry only float32 epsilon; assert the bound.
    ref = np.linspace(0.1, 1000.0, 1001)
    info = measure_float32_roundoff(ref)
    assert info.max_rel_error <= float(FLOAT32_REL_EPS) * 1.5
    assert info.max_abs_error < 0.1  # sane absolute bound for values <= 1000
