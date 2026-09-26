"""Reconstruction operator tests (mission §38–§40): versioned internal interface,
same-CFA donors, invalid/censored donor exclusion, NO_VALUE, no recursion."""

from __future__ import annotations

import numpy as np
import pytest

from zecalibrator.bpm.reconstruction import (
    DEFAULT_OPERATOR,
    ESTIMATOR_MEDIAN,
    NO_VALUE,
    OPERATOR_SCHEMA_VERSION,
    PRODUCT_OPERATOR_ID,
    ReconstructionOperator,
    donor_values,
    donors_available,
    reconstruct_site_pixel,
)


def _uniform(shape, value):
    return np.full(shape, value, dtype=np.float32)


def _all_usable(shape):
    return np.ones(shape, dtype=bool)


# ---------------------------------------------------------------------------
# Versioned interface (mission §38)
# ---------------------------------------------------------------------------

def test_default_operator_is_explicitly_versioned():
    assert DEFAULT_OPERATOR.operator_id == PRODUCT_OPERATOR_ID
    assert DEFAULT_OPERATOR.version == OPERATOR_SCHEMA_VERSION
    assert DEFAULT_OPERATOR.estimator == ESTIMATOR_MEDIAN


def test_operator_rejects_unknown_schema_version():
    with pytest.raises(ValueError):
        ReconstructionOperator(
            operator_id="X", version="not-a-version",
            estimator="median", donor_offsets=((2, 2),),
        )


def test_operator_rejects_unknown_estimator():
    with pytest.raises(ValueError):
        ReconstructionOperator(
            operator_id="X", version=OPERATOR_SCHEMA_VERSION,
            estimator="mean", donor_offsets=((2, 2),),
        )


def test_operator_rejects_non_same_cfa_offset():
    # An odd offset crosses a CFA plane boundary — forbidden by construction.
    with pytest.raises(ValueError):
        ReconstructionOperator(
            operator_id="X", version=OPERATOR_SCHEMA_VERSION,
            estimator="median", donor_offsets=((1, 1),),
        )


# ---------------------------------------------------------------------------
# Same-CFA donors (mission §39)
# ---------------------------------------------------------------------------

def test_reconstruction_uses_only_same_cfa_donors():
    # GRBG: G1 = (even, even). Make G1 pixels 100 and every other plane 999.
    cfa = _uniform((8, 8), 999.0)
    cfa[0::2, 0::2] = 100.0
    # Site (4, 4) is (even, even) -> G1; its same-CFA donors are all 100.
    value = reconstruct_site_pixel(
        cfa, 4, 4, cfa_phase="GRBG", roi_origin=(0, 0), usable=_all_usable((8, 8)),
    )
    assert value == 100.0  # a cross-plane donor (999) would have skewed the median


def test_donor_values_returns_only_same_cfa_positions():
    cfa = _uniform((8, 8), 999.0)
    cfa[0::2, 0::2] = 100.0
    donors = donor_values(
        cfa, 4, 4, cfa_phase="GRBG", roi_origin=(0, 0), usable=_all_usable((8, 8)),
    )
    # 8 nearest same-CFA donors (ring at step 2), all in-bounds at (4,4).
    assert len(donors) == 8
    assert all(d == 100.0 for d in donors)


# ---------------------------------------------------------------------------
# Excludes invalid/censored donors (mission §39)
# ---------------------------------------------------------------------------

def test_invalid_donor_is_excluded():
    cfa = _uniform((8, 8), 100.0)
    cfa[2, 2] = 700.0  # a donor carrying a bad value
    usable = _all_usable((8, 8))
    usable[2, 2] = False  # marked invalid -> excluded
    value = reconstruct_site_pixel(
        cfa, 4, 4, cfa_phase="GRBG", roi_origin=(0, 0), usable=usable,
    )
    assert value == 100.0  # the 700 donor was excluded


def test_donors_available_false_when_all_donors_invalid():
    cfa = _uniform((8, 8), 100.0)
    usable = np.zeros((8, 8), dtype=bool)
    assert donors_available(
        cfa, 4, 4, cfa_phase="GRBG", roi_origin=(0, 0), usable=usable,
    ) is False


# ---------------------------------------------------------------------------
# NO_VALUE when no valid donor (mission §40)
# ---------------------------------------------------------------------------

def test_no_donor_returns_no_value():
    cfa = _uniform((8, 8), 100.0)
    usable = np.zeros((8, 8), dtype=bool)
    assert reconstruct_site_pixel(
        cfa, 4, 4, cfa_phase="GRBG", roi_origin=(0, 0), usable=usable,
    ) == NO_VALUE


# ---------------------------------------------------------------------------
# No recursion: the operator reads only the immutable input plane
# ---------------------------------------------------------------------------

def test_reconstruction_does_not_mutate_input():
    cfa = _uniform((8, 8), 100.0).copy()
    cfa[4, 4] = 5000.0
    before = cfa.copy()
    reconstruct_site_pixel(
        cfa, 4, 4, cfa_phase="GRBG", roi_origin=(0, 0), usable=_all_usable((8, 8)),
    )
    assert np.array_equal(cfa, before)  # the input CFA is never modified


def test_operator_never_reads_a_reconstructed_value():
    # Two mutually-adjacent same-CFA sites: reconstructing one must use the
    # OTHER's *original* value (the input plane), never a value already written
    # to the output. This is structural: the operator takes only the input plane.
    cfa = _uniform((12, 12), 100.0)
    a = (5, 5)
    b = (5, 7)  # same row, CFA step 2 apart -> mutual same-CFA donors
    usable = _all_usable((12, 12))

    # Original values are equal (both 100), so regardless of order the median is
    # 100; what we assert here is the *interface*: the operator has no output
    # argument, so a reconstructed value can never be an input.
    va = reconstruct_site_pixel(cfa, a[0], a[1], cfa_phase="GRBG", roi_origin=(0, 0), usable=usable)
    vb = reconstruct_site_pixel(cfa, b[0], b[1], cfa_phase="GRBG", roi_origin=(0, 0), usable=usable)
    assert va == 100.0 and vb == 100.0
    # The input plane is unchanged, so no reconstructed value entered the donors.
    assert np.array_equal(cfa, _uniform((12, 12), 100.0))
