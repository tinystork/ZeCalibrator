"""Flat correction, contributor-invalidity propagation, quality screen (M3, §8)."""

from __future__ import annotations

import numpy as np

from zecalibrator.core.dq import FLAT_INVALID, INPUT_INVALID
from zecalibrator.core.equations import normalize_flat_response
from zecalibrator.application.executor import CalibrationRequest, MasterBinding, execute_calibration


def test_contributor_invalidity_propagates_to_invalid_response():
    F = np.ones((4, 4), dtype=np.float32) * 100.0
    inv = np.zeros((4, 4), dtype=bool)
    inv[0, 0] = True
    norm = normalize_flat_response(F, cfa_phase="mono", fcorr_invalid=inv)
    assert not norm.valid[0, 0]
    assert norm.plane_counts["mono"] == 15


def test_empty_cfa_plane_unusable():
    F = np.ones((6, 4), dtype=np.float32) * 100.0
    inv = np.zeros((6, 4), dtype=bool)
    for y in range(0, 6, 2):
        for x in range(0, 4, 2):
            inv[y, x] = True
    norm = normalize_flat_response(F, cfa_phase="GRBG", fcorr_invalid=inv)
    assert "G1" in norm.empty_planes
    assert not norm.usable
    assert norm.scalars["G1"] is None


def _mono_flat_with_valid_count(valid_count: int):
    F = np.ones((10, 10), dtype=np.float32) * 100.0
    inv = np.zeros((10, 10), dtype=bool)
    inv.flat[: 100 - valid_count] = True
    return normalize_flat_response(F, cfa_phase="mono", fcorr_invalid=inv)


def test_quality_screen_boundaries_89_90_91():
    assert not _mono_flat_with_valid_count(89).usable
    assert _mono_flat_with_valid_count(90).usable
    assert _mono_flat_with_valid_count(91).usable


def test_quality_population_excludes_floor():
    # E: 100 positive valid normalization samples, 20 of them 1e-8 (weak response).
    # The >=90% screen numerator is the median population (100), NOT post-floor
    # response validity. Screen passes; the 20 weak responses are masked.
    F = np.ones((10, 10), dtype=np.float32)
    F[:2, :] = 1e-8
    norm = normalize_flat_response(F, cfa_phase="mono")
    assert norm.plane_counts["mono"] == 100
    assert norm.plane_ratios["mono"] == 100.0
    assert norm.usable
    # The R <= 1e-6 floor masks the 20 weak responses (independent guard).
    assert int(norm.valid.sum()) == 80


def test_masked_flat_dark_propagates_to_flat_invalid(make_frame_fixture):
    make_frame = make_frame_fixture
    light = make_frame((4, 4), value=110.0)
    dark = make_frame((4, 4), value=10.0)
    finc = make_frame((4, 4), value=100.0, exposure_s=1.0)
    fdinc = make_frame((4, 4), value=10.0, exposure_s=1.0)
    fdinc.mask[0, 0] = INPUT_INVALID
    fdinc.data[0, 0] = 0.0

    masters = {
        "dark": MasterBinding(role="dark", frame=dark, bias_state="included"),
        "flat": MasterBinding(role="flat", frame=finc, flat_form="raw_response"),
        "flat_dark": MasterBinding(role="flat_dark", frame=fdinc, bias_state="included"),
    }
    result = execute_calibration(
        light, CalibrationRequest("dark_incl_bias", "apply"), masters,
        flat_prep_mode="flat_dark_incl_bias",
    )
    assert result.mask[0, 0] & FLAT_INVALID
    assert np.isnan(result.data[0, 0])
