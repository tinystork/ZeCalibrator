"""CFA four-plane normalization and frozen sensor phase (SCIENCE §7.4-7.5)."""

from __future__ import annotations

import numpy as np

from zecalibrator.core.equations import normalize_flat_response
from zecalibrator.core.geometry import array_plane, sensor_plane


def _cfa_flat(h=6, w=4):
    F = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            F[y, x] = 200.0 if sensor_plane(y, x, "GRBG") == "G1" else 100.0
    return F


def test_four_distinct_parity_planes():
    F = _cfa_flat()
    norm = normalize_flat_response(F, cfa_phase="GRBG")
    assert set(norm.scalars.keys()) == {"G1", "R", "B", "G2"}
    assert norm.scalars["G1"] == 200.0
    assert norm.scalars["R"] == 100.0
    assert norm.scalars["G2"] == 100.0
    assert norm.scalars["B"] == 100.0
    # Per-plane normalization yields a uniform unit response.
    assert np.allclose(norm.R[norm.valid], 1.0)


def test_global_median_overcorrects_g1():
    F = _cfa_flat()
    s_global = float(np.median(F))
    assert abs(s_global - 100.0) < 1e-12
    assert abs((F / s_global)[0, 0] - 2.0) < 1e-12


def test_shifted_origin_x():
    # +1 x-origin swaps G1<->R and B<->G2 (never G1<->G2).
    assert array_plane(0, 0, "GRBG", (0, 1)) == "R"
    assert array_plane(0, 1, "GRBG", (0, 1)) == "G1"
    assert array_plane(1, 0, "GRBG", (0, 1)) == "G2"
    assert array_plane(1, 1, "GRBG", (0, 1)) == "B"


def test_shifted_origin_y():
    # +1 y-origin swaps G1<->B and R<->G2.
    assert array_plane(0, 0, "GRBG", (1, 0)) == "B"
    assert array_plane(0, 1, "GRBG", (1, 0)) == "G2"
    assert array_plane(1, 0, "GRBG", (1, 0)) == "G1"
    assert array_plane(1, 1, "GRBG", (1, 0)) == "R"


def test_mono_uses_single_median():
    F = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm = normalize_flat_response(F, cfa_phase="mono")
    assert set(norm.scalars.keys()) == {"mono"}
    assert norm.scalars["mono"] == 2.5  # median of [1,2,3,4]
    assert np.allclose(norm.R[norm.valid], (F / 2.5)[norm.valid])


def test_zero_valid_population_unusable():
    F = np.array([[0.0, 0.0], [0.0, 0.0]])
    norm = normalize_flat_response(F, cfa_phase="GRBG")
    assert all(s is None for s in norm.scalars.values())
    assert not norm.valid.any()


def test_floor_masks_but_does_not_clamp():
    F = np.array([[1.0, 1e-9], [1.0, 1.0]])
    norm = normalize_flat_response(F, cfa_phase="mono", floor=1e-6)
    # The 1e-9 sample normalizes to a response <= floor => masked, not raised.
    assert not norm.valid[0, 1]
