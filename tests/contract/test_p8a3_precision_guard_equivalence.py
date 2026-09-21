"""P8-A3 — decoder precision-guard equivalence oracle (design §7, guards A/B/C).

Proves the static fast-path + vectorised predicate replace the per-pixel Python
predicate with EXACTLY identical acceptance/refusal semantics, reason codes,
exception types and messages, using a verbatim scalar reference as the oracle.

Covers: the exhaustive dtype x byte-order x scale matrix, the float32
exact-integer boundary values, refusal precedence (C1 magnitude wins over C2
exact-integer), BLANK interaction, non-finite/overflow filtering, static-proof
soundness (parameterised bound + fuzz) and the Guard A discriminating
``BSCALE==0`` + float NaN/Inf case.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from zecalibrator.api.v1._io import validate_array_precision
from zecalibrator.core.errors import PrecisionRefusalError
from zecalibrator.core.metadata import ImportDeclaration
from zecalibrator.core.precision import (
    FLOAT32_EXACT_INT_BOUND,
    exceeds_float32_exact_bound,
    float32_guard_provable_safe,
    integer_exceeds_float32_exact_range,
)
from zecalibrator.io.raw_decoder import decode_fits

F32MAX = float(np.finfo(np.float32).max)
NEXT_AFTER_F32MAX = float(np.nextafter(np.float64(np.finfo(np.float32).max), np.inf))

# ---------------------------------------------------------------------------
# Verbatim scalar oracle (the pre-change predicate, re-implemented in the test).
# ---------------------------------------------------------------------------


def _scalar_reference_predicate(value: float) -> bool:
    """Verbatim copy of the pre-change scalar predicate (the oracle)."""
    v = float(value)
    if not np.isfinite(v):
        return False
    if v != round(v):
        return False
    return abs(v) > FLOAT32_EXACT_INT_BOUND


def _oracle_array_precision(arr) -> tuple:
    """Verbatim reproduction of the pre-change ``validate_array_precision`` guard.

    Returns ``("accept", None, None)`` or ``("refuse", exc_type_name, message)``.
    """
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.integer) and a.dtype.itemsize == 8:
        if np.issubdtype(a.dtype, np.signedinteger):
            mag = np.abs(a.astype(np.int64))
        else:
            mag = a.astype(np.uint64)
        if np.any(mag > np.uint64(2 ** 53)):
            return ("refuse", "PrecisionRefusalError",
                    "stored 64-bit integer exceeds float64 exact-integer bound 2**53")

    f64 = a.astype(np.float64)
    finite = np.isfinite(f64)
    if finite.any():
        vals = f64[finite]
        if np.any(np.abs(vals) > float(np.finfo(np.float32).max)):
            return ("refuse", "PrecisionRefusalError",
                    "decoded magnitude exceeds float32 range (extreme scale)")
        if np.any(
            np.fromiter(
                (_scalar_reference_predicate(v) for v in vals),
                dtype=bool,
                count=int(vals.size),
            )
        ):
            return ("refuse", "PrecisionRefusalError",
                    "decoded integer ADU exceeds float32-exact bound 2**24")
    return ("accept", None, None)


def _oracle_decode_guard(stored, bscale, bzero, blank=None) -> tuple:
    """Verbatim reproduction of the pre-change ``decode_fits`` precision guard.

    Returns ``("accept", None, None)`` or ``("refuse", exc_type_name, message)``.
    """
    stored = np.asarray(stored)
    if stored.dtype.itemsize == 8 and np.issubdtype(stored.dtype, np.integer):
        if np.issubdtype(stored.dtype, np.signedinteger):
            mag = np.abs(stored.astype(np.int64))
        else:
            mag = stored.astype(np.uint64)
        if np.any(mag > np.uint64(2 ** 53)):
            return ("refuse", "PrecisionRefusalError",
                    "stored 64-bit integer exceeds float64 exact-integer bound 2**53")

    invalid = np.zeros(stored.shape, dtype=bool)
    if blank is not None and np.issubdtype(stored.dtype, np.integer):
        invalid |= stored == blank

    stored_f64 = stored.astype(np.float64)
    if np.issubdtype(stored.dtype, np.floating):
        invalid |= ~np.isfinite(stored_f64)

    physical = bscale * stored_f64 + bzero

    finite = np.isfinite(physical)
    if finite.any():
        vals = physical[finite]
        if np.any(np.abs(vals) > float(np.finfo(np.float32).max)):
            return ("refuse", "PrecisionRefusalError",
                    "decoded magnitude exceeds float32 range (extreme scale)")
        if np.any(
            np.fromiter(
                (_scalar_reference_predicate(v) for v in vals),
                dtype=bool,
                count=int(vals.size),
            )
        ):
            return ("refuse", "PrecisionRefusalError",
                    "decoded integer ADU exceeds float32-exact bound 2**24")
    return ("accept", None, None)


def _run_array_validate(arr) -> tuple:
    try:
        validate_array_precision(arr)
        return ("accept", None, None)
    except PrecisionRefusalError as exc:
        return ("refuse", type(exc).__name__, str(exc))


# ---------------------------------------------------------------------------
# FITS fixtures
# ---------------------------------------------------------------------------


def _decl(**overrides) -> ImportDeclaration:
    base = dict(
        source="synthetic_fixture", identity="SYNTH-P8A3-1", version="1.0",
        domain="raw", units="ADU",
        detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-MONO",
        gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16",
        binning=(1, 1), orientation="identity", cfa_phase="mono", roi_origin=(0, 0),
        exposure_s=10.0, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.1,
        saturation_limit_adu=60000, saturation_evidence="qualified",
    )
    base.update(overrides)
    return ImportDeclaration(**base)


def _write(path, data, header_cards=None):
    hdu = fits.PrimaryHDU(data)
    hdu.header["BUNIT"] = "ADU"
    for key, val in (header_cards or []):
        hdu.header[key] = val
    hdu.writeto(path, overwrite=True)
    return path


def _run_decode(path, declaration=None) -> tuple:
    try:
        decode_fits(path, declaration=declaration if declaration is not None else _decl())
        return ("accept", None, None)
    except PrecisionRefusalError as exc:
        return ("refuse", type(exc).__name__, str(exc))


def _read_fits(path):
    """Read back the actual stored plane + scale the decoder will see.

    astropy may remap storage (e.g. signed int8 -> unsigned BITPIX=8 with an
    auto BZERO), so the oracle must run on the values the decoder actually
    receives, not on the values the test intended to write.
    """
    with fits.open(path, do_not_scale_image_data=True, memmap=False) as hdul:
        hdu = hdul[0]
        stored = np.asarray(hdu.data)
        bscale = float(hdu.header.get("BSCALE", 1.0))
        bzero = float(hdu.header.get("BZERO", 0.0))
        blank = hdu.header.get("BLANK")
        if blank is not None:
            bf = float(blank)
            blank = int(bf) if bf.is_integer() else None
        return stored, bscale, bzero, blank


# ---------------------------------------------------------------------------
# Value domains per dtype (representable in that dtype)
# ---------------------------------------------------------------------------

_VALUES_BY_DTYPE = {
    "int8": [-128, -1, 0, 1, 127],
    "uint8": [0, 1, 127, 255],
    "int16": [-32768, -1, 0, 1, 100, 32767],
    "uint16": [0, 1, 100, 32768, 65535],
    "int32": [-(2 ** 31), -1, 0, 1, 2 ** 24, 2 ** 24 + 1, 2 ** 25, 2 ** 31 - 1],
    "uint32": [0, 1, 2 ** 24, 2 ** 24 + 1, 2 ** 32 - 1],
    "int64": [-(2 ** 53) - 1, -(2 ** 53), 2 ** 53, 2 ** 53 + 1, 2 ** 24, 2 ** 24 + 1, 2 ** 63 - 1],
    "uint64": [0, 2 ** 53, 2 ** 53 + 1, 2 ** 64 - 1],
    "float32": [0.0, 1.5, float(2 ** 24), float(2 ** 25), F32MAX, np.inf, -np.inf, np.nan],
    "float64": [
        0.0, 1.5, float(2 ** 24), float(2 ** 24 + 1), float(2 ** 25),
        float(2 ** 25 + 0.5), F32MAX, NEXT_AFTER_F32MAX, np.inf, -np.inf, np.nan,
    ],
}

# FITS-native storage dtypes exercised through the decoder (unsigned FITS has no
# native type; unsigned dtypes are exercised through validate_array_precision).
_FITS_DTYPES = ["int8", "int16", "int32", "int64", "float32", "float64"]

_SCALES = [
    (1.0, 0.0),
    (1.0, 32768.0),
    (1.0, -32768.0),
    (0.5, 0.0),
    (2.0, 0.0),
    (1e-3, 0.0),
    (32768.0, 0.0),
    (1.0, 1e30),
    (0.0, 1.0),
    (0.0, 1e300),
    (1e308, 0.0),
]


def _byte_order_variants(dtype_str):
    dt = np.dtype(dtype_str)
    if dt.itemsize == 1:
        return [dt]
    code = f"{dt.kind}{dt.itemsize}"
    return [np.dtype(code), np.dtype(">" + code), np.dtype("<" + code)]


# ---------------------------------------------------------------------------
# G1a — vectorised twin == verbatim scalar predicate (boundary values, exact)
# ---------------------------------------------------------------------------

BOUNDARY_VALUES = [
    2 ** 24 - 1, 2 ** 24, 2 ** 24 + 1, 2 ** 25, -(2 ** 24) - 1, 2 ** 25 + 0.5,
    0.0, -0.0, 1.5, F32MAX, NEXT_AFTER_F32MAX, np.inf, -np.inf, np.nan,
    float(np.float32(2 ** 24 + 1)),
]


@pytest.mark.parametrize("value", BOUNDARY_VALUES)
def test_vectorised_twin_matches_scalar_boundary(value):
    arr = np.array([value])
    assert exceeds_float32_exact_bound(arr) is _scalar_reference_predicate(value)
    # The canonical exported scalar must be unchanged and agree.
    assert integer_exceeds_float32_exact_range(value) is _scalar_reference_predicate(value)


def test_vectorised_twin_matches_scalar_mixed():
    vals = np.array(BOUNDARY_VALUES, dtype=np.float64)
    scalar = np.any(np.fromiter(
        (_scalar_reference_predicate(v) for v in vals), dtype=bool, count=vals.size,
    ))
    assert exceeds_float32_exact_bound(vals) is bool(scalar)


@pytest.mark.parametrize("order", [">f8", "<f8", ">f4", "<f4"])
def test_vectorised_twin_byte_order_invariant(order):
    vals = np.array(BOUNDARY_VALUES, dtype=np.float64).astype(order)
    scalar = np.any(np.fromiter(
        (_scalar_reference_predicate(float(v)) for v in vals), dtype=bool, count=vals.size,
    ))
    assert exceeds_float32_exact_bound(vals) is bool(scalar)


def test_vectorised_twin_empty_and_all_nonfinite():
    assert exceeds_float32_exact_bound(np.array([], dtype=np.float64)) is False
    assert exceeds_float32_exact_bound(np.array([np.nan, np.inf, -np.inf])) is False


# ---------------------------------------------------------------------------
# G1b — array-input path (validate_array_precision) equivalence, all 10 dtypes
# ---------------------------------------------------------------------------

_ARRAY_DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32",
                 "int64", "uint64", "float32", "float64"]


@pytest.mark.parametrize("dtype", _ARRAY_DTYPES)
def test_array_precision_equivalence_all_dtypes(dtype):
    values = _VALUES_BY_DTYPE[dtype]
    for dt in _byte_order_variants(dtype):
        arr = np.array(values, dtype=dt)
        assert _run_array_validate(arr) == _oracle_array_precision(arr), (dt, values)


@pytest.mark.parametrize("dtype", ["int16", "int32", "int64", "float32", "float64"])
def test_array_precision_equivalence_non_native(dtype):
    # Explicit non-native byte orders (Guard B) must be identical to native.
    values = _VALUES_BY_DTYPE[dtype]
    ref = None
    for dt in _byte_order_variants(dtype):
        arr = np.array(values, dtype=dt)
        got = _run_array_validate(arr)
        oracle = _oracle_array_precision(arr)
        assert got == oracle, (dt, got, oracle)
        if ref is None:
            ref = got
        else:
            assert got == ref, (dt, got, ref)


def test_array_precision_no_static_refusal():
    # int32 with an all-zero plane: the bound exceeds 2**24 but no value does —
    # the static proof must NOT refuse (Guard C).
    arr = np.zeros(8, dtype=np.int32)
    assert _run_array_validate(arr) == ("accept", None, None)
    assert _oracle_array_precision(arr) == ("accept", None, None)
    # but an int32 value beyond the bound must still be refused by C2
    arr2 = np.array([2 ** 24 + 1], dtype=np.int32)
    assert _run_array_validate(arr2) == ("refuse", "PrecisionRefusalError",
                                         "decoded integer ADU exceeds float32-exact bound 2**24")


def test_array_precision_fuzz_vectorised_matches_scalar():
    rng = np.random.default_rng(20260921)
    # Mix of integer-valued, fractional, boundary and huge finite values.
    for _ in range(50):
        n = int(rng.integers(1, 200))
        kind = rng.integers(0, 4)
        if kind == 0:
            vals = rng.integers(-(2 ** 26), 2 ** 26, size=n).astype(np.float64)
        elif kind == 1:
            vals = (rng.uniform(-2 ** 26, 2 ** 26, size=n) + 0.5).astype(np.float64)
        elif kind == 2:
            vals = rng.uniform(-1e38, 1e38, size=n)
        else:
            base = np.array([2 ** 24 - 1, 2 ** 24, 2 ** 24 + 1, 2 ** 25, F32MAX,
                             NEXT_AFTER_F32MAX, 0.0, 1.5, np.nan, np.inf, -np.inf])
            vals = rng.choice(base, size=n)
        arr = vals.astype(np.float64)
        assert _run_array_validate(arr) == _oracle_array_precision(arr)


# ---------------------------------------------------------------------------
# G1c — decode_fits equivalence matrix (FITS-native dtypes x scales)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _FITS_DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_decode_equivalence_matrix(tmp_path, dtype, scale):
    bscale, bzero = scale
    values = _VALUES_BY_DTYPE[dtype]
    stored = np.atleast_2d(np.array(values, dtype=dtype))
    cards = []
    if bscale != 1.0:
        cards.append(("BSCALE", bscale))
    if bzero != 0.0:
        cards.append(("BZERO", bzero))
    p = _write(tmp_path / "m.fits", stored, cards)
    actual, abscale, abzero, ablank = _read_fits(p)
    got = _run_decode(p)
    oracle = _oracle_decode_guard(actual, abscale, abzero, ablank)
    assert got == oracle, (dtype, scale, got, oracle, actual.dtype, abscale, abzero)


def test_decode_boundary_values_float64(tmp_path):
    stored = np.atleast_2d(np.array(BOUNDARY_VALUES, dtype=np.float64))
    p = _write(tmp_path / "boundary.fits", stored)
    got = _run_decode(p)
    oracle = _oracle_decode_guard(stored, 1.0, 0.0)
    assert got == oracle
    assert got[0] == "refuse"


def test_refusal_precedence_c1_wins(tmp_path):
    # A plane with both an over-float32max value and an integer > 2**24 must
    # raise the magnitude (C1) refusal first.
    stored = np.atleast_2d(np.array([NEXT_AFTER_F32MAX, 2 ** 24 + 1], dtype=np.float64))
    p = _write(tmp_path / "prec1.fits", stored)
    got = _run_decode(p)
    assert got == ("refuse", "PrecisionRefusalError",
                   "decoded magnitude exceeds float32 range (extreme scale)")
    # C2 alone when no value overflows float32.
    stored2 = np.atleast_2d(np.array([F32MAX, 2 ** 24 + 1], dtype=np.float64))
    p2 = _write(tmp_path / "prec2.fits", stored2)
    got2 = _run_decode(p2)
    assert got2 == ("refuse", "PrecisionRefusalError",
                    "decoded integer ADU exceeds float32-exact bound 2**24")


def test_blank_interaction(tmp_path):
    # Integer storage with BLANK: the guard still runs on the physical value and
    # the BLANK pixel is still marked invalid afterwards (order unchanged).
    stored = np.array([[-32768, -32668], [-32568, 32767]], dtype=np.int16)
    p = _write(tmp_path / "blank.fits", stored, [("BZERO", 32768), ("BLANK", -32768)])
    from zecalibrator.core.dq import INPUT_INVALID
    f = decode_fits(p, declaration=_decl())
    assert f.mask[0, 0] & INPUT_INVALID
    assert np.isnan(f.data[0, 0])
    assert f.data[0, 1] == 100.0
    assert f.data[1, 1] == 65535.0
    # guard equivalence (blank pixel's physical value is inside the guard, as
    # today, and BLANK = -32768 -> physical 0 does not trip the guard)
    oracle = _oracle_decode_guard(stored, 1.0, 32768.0, blank=-32768)
    assert oracle == ("accept", None, None)


def test_nonfinite_overflow_not_refusal(tmp_path):
    # BSCALE making every physical value overflow to inf is still NOT a refusal
    # (the finite filter excludes them), matching today's behaviour.
    from zecalibrator.core.dq import INPUT_INVALID
    stored = np.atleast_2d(np.array([4, 4, 4, 4], dtype=np.int16))
    p = _write(tmp_path / "ovf.fits", stored, [("BSCALE", 1e308)])
    got = _run_decode(p)
    actual, abscale, abzero, _ = _read_fits(p)
    oracle = _oracle_decode_guard(actual, abscale, abzero)
    assert got == oracle == ("accept", None, None)
    f = decode_fits(p, declaration=_decl())
    assert np.all(f.mask & INPUT_INVALID)
    assert np.all(np.isnan(f.data))


def test_guard_a_bscale_zero_float_nan_inf(tmp_path):
    # Guard A discriminating case: BSCALE==0 with float storage containing
    # NaN/Inf must retain exactly today's invalid-mask and output semantics.
    stored = np.atleast_2d(np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32))
    p = _write(tmp_path / "guard_a.fits", stored, [("BSCALE", 0.0), ("BZERO", 1.0)])
    from zecalibrator.core.dq import INPUT_INVALID
    f = decode_fits(p, declaration=_decl())
    # invalid mask built from the STORED plane (unchanged): NaN/Inf invalid, 0.5 valid.
    assert f.mask[0, 0] & INPUT_INVALID
    assert f.mask[0, 1] & INPUT_INVALID
    assert f.mask[0, 2] & INPUT_INVALID
    assert not (f.mask[0, 3] & INPUT_INVALID)
    # physical = 0*stored + 1 -> nan where stored was NaN/Inf, 1.0 where 0.5.
    assert np.isnan(f.data[0, 0]) and np.isnan(f.data[0, 1]) and np.isnan(f.data[0, 2])
    assert f.data[0, 3] == 1.0
    # the fast-path may skip C1/C2 here (provably safe: bound = |1| <= 2**24)
    assert float32_guard_provable_safe(np.dtype(np.float32), 0.0, 1.0) is True
    # and the guard verdict matches the oracle (accept).
    assert _oracle_decode_guard(stored, 0.0, 1.0) == ("accept", None, None)


# ---------------------------------------------------------------------------
# G2 — static-proof soundness
# ---------------------------------------------------------------------------


def _bound_formula(dtype, bscale, bzero) -> bool:
    dt = np.dtype(dtype)
    bs = float(bscale)
    bz = float(bzero)
    if bs == 0.0:
        bound = abs(bz)
    elif dt.kind in ("i", "u"):
        info = np.iinfo(dt)
        maxabs = float(max(abs(int(info.min)), abs(int(info.max))))
        bound = abs(bs) * maxabs + abs(bz)
    else:
        return False
    return bool(np.isfinite(bound)) and bound <= float(FLOAT32_EXACT_INT_BOUND)


@pytest.mark.parametrize("dtype", _ARRAY_DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_static_proof_parameterised_bound(dtype, scale):
    bscale, bzero = scale
    for dt in _byte_order_variants(dtype):
        assert float32_guard_provable_safe(dt, bscale, bzero) == _bound_formula(dt, bscale, bzero), (dt, scale)


def test_static_proof_byte_order_invariant():
    # Real FITS planes are big-endian: '>i2' lights, '>f4' masters. The static
    # bound must be identical for native / big-endian / little-endian.
    for code, kind in [("i2", "i"), ("i4", "i"), ("i8", "i"), ("u2", "u"),
                       ("u4", "u"), ("u8", "u")]:
        native = float32_guard_provable_safe(np.dtype(code), 1.0, 32768.0)
        big = float32_guard_provable_safe(np.dtype(">" + code), 1.0, 32768.0)
        little = float32_guard_provable_safe(np.dtype("<" + code), 1.0, 32768.0)
        assert native == big == little, code
    # float storage: bscale==0 provable safe regardless of byte order.
    for code in ("f4", "f8"):
        assert float32_guard_provable_safe(np.dtype(code), 0.0, 1.0) is True
        assert float32_guard_provable_safe(np.dtype(">" + code), 0.0, 1.0) is True
        assert float32_guard_provable_safe(np.dtype("<" + code), 0.0, 1.0) is True
        assert float32_guard_provable_safe(np.dtype(code), 1.0, 0.0) is False


# S-qualified representations: bound <= 2**24 => static fast-path skips C1/C2.
_S_QUALIFIED = [
    ("int8", 1.0, 0.0),
    ("int16", 1.0, 32768.0),
    ("int16", 1.0, -32768.0),
    ("int16", 0.5, 0.0),
    ("int32", 0.0, 1.0),
    ("float32", 0.0, 1.0),
    ("float64", 0.0, 1.0),
]


def test_static_proof_fuzz_invariant():
    rng = np.random.default_rng(20260921)
    for dtype, bscale, bzero in _S_QUALIFIED:
        dt = np.dtype(dtype)
        assert float32_guard_provable_safe(dt, bscale, bzero) is True, (dtype, bscale, bzero)
        info = np.iinfo(dt) if dt.kind in ("i", "u") else None
        for _ in range(20):
            n = int(rng.integers(1, 50))
            if info is not None:
                low = int(info.min) if dt.kind == "i" else 0
                high = int(info.max)
                stored = rng.integers(low, high + 1, size=n, dtype=dt)
            else:
                stored = rng.normal(0.0, 1000.0, size=n).astype(dt)
            oracle = _oracle_decode_guard(stored, bscale, bzero)
            assert oracle == ("accept", None, None), (dtype, bscale, bzero, stored)


def test_static_proof_never_statically_refuses(tmp_path):
    # int32 representation: bound = 2**31 > 2**24, so NOT statically safe. A
    # plane of zeros must still be ACCEPTED (fall back to the value predicate,
    # never refuse from the bound alone — Guard C).
    stored = np.atleast_2d(np.zeros(6, dtype=np.int32))
    p = _write(tmp_path / "int32zero.fits", stored)
    got = _run_decode(p)
    assert got == ("accept", None, None)
    assert float32_guard_provable_safe(np.dtype(np.int32), 1.0, 0.0) is False
    # and a value beyond the bound in the same representation still refuses.
    stored2 = np.atleast_2d(np.array([2 ** 24 + 1], dtype=np.int32))
    p2 = _write(tmp_path / "int32big.fits", stored2)
    assert _run_decode(p2) == ("refuse", "PrecisionRefusalError",
                               "decoded integer ADU exceeds float32-exact bound 2**24")


def test_static_proof_overflow_bound_not_safe():
    # A BSCALE whose product overflows float64 must NOT be claimed safe.
    assert float32_guard_provable_safe(np.dtype(np.int16), 1e308, 0.0) is False
    assert float32_guard_provable_safe(np.dtype(np.int64), 1e308, 0.0) is False
    assert float32_guard_provable_safe(np.dtype(np.int16), 1.0, 1e300) is False
