"""Hand/float64 equation witnesses (SCIENCE §7.9, ASTRA §4.3).

Independent numeric expectations, asserted against the pure ``core.equations``
functions and the full ``core.calibrate`` pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.core.calibrate import calibrate_light
from zecalibrator.core.equations import additive_numerator, final_plane, flat_correction

L = np.array([[110.0, 50.0], [210.0, -2.0]])
DINC = np.full((2, 2), 10.0)
BIAS = np.full((2, 2), 4.0)
D0 = np.full((2, 2), 6.0)
R = np.array([[1.0, 0.5], [2.0, 1.0]])


def test_light_branches_reference():
    assert np.array_equal(additive_numerator(L, "control"), L)
    assert np.array_equal(additive_numerator(L, "bias_only", bias=BIAS), [[106, 46], [206, -6]])
    assert np.array_equal(additive_numerator(L, "dark_incl_bias", dark_inc=DINC), [[100, 40], [200, -12]])
    assert np.array_equal(
        additive_numerator(L, "dark_bias_removed", bias=BIAS, dark_removed=D0),
        [[100, 40], [200, -12]],
    )


def test_coherent_dark_branches_equal_and_control():
    dark_incl = additive_numerator(L, "dark_incl_bias", dark_inc=DINC)
    removed = additive_numerator(L, "dark_bias_removed", bias=BIAS, dark_removed=D0)
    assert np.array_equal(dark_incl, removed)
    # Control / no additive correction is the explicit identity branch.
    assert np.array_equal(additive_numerator(L, "control"), L)


def test_double_subtraction_negative_control():
    c_dark = final_plane(additive_numerator(L, "dark_incl_bias", dark_inc=DINC), R)
    assert np.allclose(c_dark, [[100, 80], [100, -12]])
    double = (L - DINC - BIAS) / R
    assert np.allclose(double, [[96, 72], [98, -16]])
    assert not np.allclose(double, c_dark)


def test_full_pipeline_reference():
    result = calibrate_light(
        L,
        additive_mode="dark_incl_bias",
        flat_mode="apply",
        dark_inc=DINC,
        flat_response=R,
        flat_valid=np.ones((2, 2), dtype=bool),
        saturation_limit=60000.0,
    )
    assert result.status == "COMPLETED"
    assert result.data.dtype == np.float32
    assert np.allclose(result.data, [[100.0, 80.0], [100.0, -12.0]])
    assert np.count_nonzero(result.mask) == 0


def test_flat_only_partial_mode_reference():
    result = calibrate_light(
        L,
        additive_mode="control",
        flat_mode="apply",
        flat_response=R,
        flat_valid=np.ones((2, 2), dtype=bool),
        saturation_limit=60000.0,
    )
    assert np.allclose(result.data, [[110.0, 100.0], [105.0, -2.0]])


def test_flat_pedestal_branches():
    Finc = np.array([[100.0, 110.0], [120.0, 90.0]])
    FDinc = np.full((2, 2), 10.0)
    Bflat = np.full((2, 2), 4.0)
    FD0 = np.full((2, 2), 6.0)
    f_incl = flat_correction(Finc, "flat_dark_incl_bias", flat_dark_inc=FDinc)
    f_removed = flat_correction(Finc, "flat_dark_bias_removed", flat_dark_removed=FD0, bias_flat=Bflat)
    f_bias = flat_correction(Finc, "bias_only_flat", bias_flat=Bflat)
    assert np.array_equal(f_incl, [[90, 100], [110, 80]])
    assert np.array_equal(f_removed, [[90, 100], [110, 80]])
    assert np.array_equal(f_bias, [[96, 106], [116, 86]])
    assert np.array_equal(f_incl, f_removed)
    # Normalizing the uncorrected flat is not equivalent to correcting it first.
    s_corr = float(np.median(f_incl))
    s_wrong = float(np.median(Finc))
    assert abs(s_corr - 95.0) < 1e-12
    assert abs(s_wrong - 105.0) < 1e-12
    assert not np.allclose(f_incl / s_corr, Finc / s_wrong)


def test_invalid_request_modes_raise():
    with pytest.raises(ValueError):
        additive_numerator(L, "not_a_mode")
    from zecalibrator.core.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError):
        calibrate_light(L, additive_mode="bias_only", flat_mode="none")
