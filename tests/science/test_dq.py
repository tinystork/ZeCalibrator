"""DQ reason-bit and count semantics (SCIENCE §9)."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.core.calibrate import calibrate_light
from zecalibrator.core.dq import (
    ADDITIVE_INVALID,
    ARITH_NONFINITE,
    CountSummary,
    FLAT_INVALID,
    INPUT_INVALID,
    SATURATED,
    validate_mask,
)

ALL_ONES = np.ones((2, 2), dtype=bool)


def test_bit_values_frozen():
    assert INPUT_INVALID == 0x0001
    assert ADDITIVE_INVALID == 0x0002
    assert FLAT_INVALID == 0x0004
    assert SATURATED == 0x0008
    assert ARITH_NONFINITE == 0x0010


def test_count_summary_overlaps():
    mask = np.array([[0, INPUT_INVALID], [INPUT_INVALID | SATURATED, 0]], dtype=np.uint16)
    st = CountSummary.from_mask(mask)
    assert st.total == 4
    assert st.valid_count == 2
    assert st.invalid_count == 2
    assert st.per_bit["INPUT_INVALID"] == 2
    assert st.per_bit["SATURATED"] == 1
    assert st.invalid_count != sum(st.per_bit.values())


def test_saturated_only_is_invalid():
    mask = np.array([[SATURATED, 0], [0, 0]], dtype=np.uint16)
    st = CountSummary.from_mask(mask)
    assert st.valid_count == 3 and st.invalid_count == 1
    assert st.per_bit["SATURATED"] == 1


def test_valid_negative_output():
    L = np.array([[5.0, -12.0], [3.0, 7.0]], dtype=np.float32)
    r = calibrate_light(L, additive_mode="control", saturation_limit=100.0)
    assert r.status == "COMPLETED"
    assert np.count_nonzero(r.mask) == 0
    assert r.data[0, 1] < 0.0


def test_all_invalid_refusal():
    L = np.full((2, 2), np.nan, dtype=np.float32)
    mask = np.full((2, 2), INPUT_INVALID, dtype=np.uint16)
    r = calibrate_light(L, additive_mode="control", input_mask=mask, saturation_limit=100.0)
    assert r.status == "FAILED"
    assert r.data is None and r.mask is None
    assert r.counts.invalid_count == r.counts.total


def test_contributing_master_invalidity():
    L = np.ones((2, 2), dtype=np.float32)
    bias = np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32)
    bias_mask = np.zeros((2, 2), dtype=np.uint16)
    bias_mask[0, 1] = INPUT_INVALID
    r = calibrate_light(
        L, additive_mode="bias_only", bias=bias, bias_mask=bias_mask, saturation_limit=100.0
    )
    assert r.mask[0, 1] & ADDITIVE_INVALID
    assert np.isnan(r.data[0, 1])
    assert r.status == "COMPLETED_WITH_WARNINGS"


def test_saturation_unknown_is_frame_level_not_pixel_bit():
    L = np.ones((2, 2), dtype=np.float32)
    r = calibrate_light(L, additive_mode="control")
    assert r.frame_quality.saturation_evidence == "unknown"
    assert np.count_nonzero(r.mask) == 0
    assert r.status == "COMPLETED_WITH_WARNINGS"


def test_arithmetic_nonfinite_from_valid_inputs():
    # Division overflow: finite float32 light / tiny valid flat => Inf -> NaN + bit.
    L = np.array([[3e38, 0.0], [0.0, 0.0]], dtype=np.float32)
    R = np.array([[1e-5, 1.0], [1.0, 1.0]], dtype=np.float32)  # 1e-5 > 1e-6 floor
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r = calibrate_light(
            L, additive_mode="control", flat_mode="apply",
            flat_response=R, flat_valid=ALL_ONES, saturation_limit=None,
        )
    assert r.mask[0, 0] & ARITH_NONFINITE
    assert np.isnan(r.data[0, 0])


def test_float32_overflow_not_returned_as_success():
    # M6 repro: finite float64 1e39 overflows float32; must NOT return Inf + DQ0.
    with np.errstate(over="ignore", invalid="ignore"):
        r = calibrate_light(np.ones((2, 2)) * 1e39, additive_mode="control")
    assert r.status == "FAILED"  # all-invalid, not COMPLETED_WITH_WARNINGS
    assert r.data is None
    assert not np.isinf(r.counts.invalid_count)


def test_nan_input_is_input_invalid_not_arith():
    L = np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float32)
    mask = np.zeros((2, 2), dtype=np.uint16)
    mask[0, 0] = INPUT_INVALID
    r = calibrate_light(L, additive_mode="control", input_mask=mask, saturation_limit=100.0)
    assert r.mask[0, 0] & INPUT_INVALID
    assert not (r.mask[0, 0] & ARITH_NONFINITE)
    assert np.isnan(r.data[0, 0])


def test_reserved_bits_rejected():
    bad = np.array([[0x0020]], dtype=np.uint16)  # reserved bit 5
    with pytest.raises(ValueError):
        validate_mask(bad)


def test_non_integer_mask_rejected():
    with pytest.raises(ValueError):
        validate_mask(np.zeros((2, 2), dtype=np.float32))
