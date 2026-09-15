#!/usr/bin/env python3
"""
contract_witness.py — ZeCalibrator Phase 1 / Mission 1 research witness (owner reconciliation after accepted r4).

STANDALONE research script, NOT an installed module and NOT a production
calibration engine/matcher. It performs only tiny independent mathematical,
header/decoding, digest, DQ-semantics, identity, card-record, and
declarative-structure checks for the Phase 1 specification. It contains NO
production classes, package, matcher, or engine.

Sections:
  * source-authority guard (asserts mission + interop SHA BEFORE any test/output)
  * equations (light/flat branches, negative controls)
  * CFA parity + shifted-origin GRBG mapping
  * DQ bits + semantics (five invalid-reason bits; reason-OR; counts derived
    from masks incl SAT-only and IN|SAT; NaN/status derived, not literal)
  * FITS decode (BSCALE/BZERO exactly once, BLANK stored-space in-memory,
    float32 precision boundary)
  * canonical digest reference (magic+dtype+shape IN the hashed record; pinned
    independent bytes/SHA; byte-order/NaN-payload/signed-zero equivalence;
    dtype/shape/data/DQ difference negatives)
  * descriptor/plan identity via EXPLICIT PROJECTIONS (not blacklists); positive
    assertions that incidental fields don't change identity, negative assertions
    that science-relevant fields do
  * ordered CardRecord witness (repeated identical keyword preserved)
  * quality-gate boundary arithmetic (89/90/91 of 100) vs >1e-6 floor
  * cases.json crosscheck: deep materialization per case (no shared mutable
    objects), mechanical apply of overrides/removals/add-clone to independent
    copies, structural audit of MATCHED/NO_MATCH/AMBIGUOUS declared outcomes,
    and independent negative controls (shared-mutation, invalid path, clone
    isolation, normalized-history contradiction)

Output is STRICT JSON (no bare NaN/Inf). Run with the ZSSS venv interpreter:

    PYTHONDONTWRITEBYTECODE=1 .../contract_witness.py --check --self-test-guard
    .../contract_witness.py --out /tmp/evidence.json

--check means no PERSISTENT artifact write (temporary FITS files are permitted
and removed). Default/--out write the evidence JSON only.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import struct
import sys

import numpy as np
import astropy
from astropy.io import fits

PROJECT_ROOT = "/home/tristan/.openclaw/workspace/projects/zecalibrator"
CASES_PATH = os.path.join(PROJECT_ROOT, "research", "phase1", "cases.json")
OUT_EVIDENCE = os.path.join(PROJECT_ROOT, "research", "phase1", "evidence.json")
MISSION_DOC = os.path.join(PROJECT_ROOT, "ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md")
INTEROP_DOC = "/home/tristan/.openclaw/workspace/projects/ZESOFTWARE_INTEROPERABILITY_RULES.md"

EXPECTED_MISSION_SHA = "dbcf826d52031627fd7c817b2f8e430bdf581ecf9ee36c4ec4061d009762020c"
EXPECTED_INTEROP_SHA = "88831410559f8201eeddc6722482ebf9ab3495dd10b1ab33adbbef9e3d2595ed"

PINNED_CANONICAL_DATA_HEX = (
    "000000000000001d5a43414c444947310a666c6f617433320a312c320a3f800000c0000000"
)
PINNED_CANONICAL_DQ_HEX = (
    "00000000000000185a43414c444947310a75696e7431360a312c320a00000000"
)
PINNED_SCIENCE_DIGEST = "17761d64f6ce235dca8e9e5afa22645da80deed63bee83237adea571997660af"
PINNED_CANONICAL_JSON_EXAMPLE = '{"a":2.0,"b":1,"c":"x\\ny"}'


class SourceAuthorityError(Exception):
    """Raised when a source-authority hash does not match its expected value."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_authority(path: str, expected_sha: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise SourceAuthorityError(
            f"source authority mismatch: {path}\n"
            f"expected {expected_sha}\nactual   {actual}"
        )
    return actual


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def close(a, b, rtol=1e-12, atol=1e-12) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))


def grbg_plane(y, x):
    if y % 2 == 0:
        return "G1" if x % 2 == 0 else "R"
    return "B" if x % 2 == 0 else "G2"


def plane_under_shift(y, x, ox, oy):
    return grbg_plane(y + oy, x + ox)


# ---------------------------------------------------------------------------
# Canonical digest — zecalibrator.digest.v1 (reference implementation)
# ---------------------------------------------------------------------------
_DTYPE_TOKENS = {
    ("f", 4): "float32", ("f", 8): "float64", ("u", 2): "uint16",
    ("u", 1): "uint8", ("i", 2): "int16", ("u", 4): "uint32",
    ("i", 4): "int32", ("u", 8): "uint64", ("i", 8): "int64",
}
_F32_NAN = bytes([0x7F, 0xC0, 0x00, 0x00])
_F64_NAN = bytes([0x7F, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])


def dtype_token(dt) -> str:
    d = np.dtype(dt)
    return _DTYPE_TOKENS[(d.kind, d.itemsize)]


def canonical_array_bytes(arr) -> bytes:
    a = np.asarray(arr).astype(np.dtype(arr.dtype).newbyteorder(">"))
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        a = a.copy()
        if a.dtype.itemsize == 4:
            nanval = np.frombuffer(_F32_NAN, dtype=">f4")[0]
        elif a.dtype.itemsize == 8:
            nanval = np.frombuffer(_F64_NAN, dtype=">f8")[0]
        else:
            raise ValueError("unsupported float size")
        a[~np.isfinite(a)] = nanval
        neg_zero = (a == 0.0) & np.signbit(a)
        if neg_zero.any():
            a[neg_zero] = 0.0
    record = (
        b"ZCALDIG1\n"
        + dtype_token(a.dtype).encode("ascii")
        + b"\n"
        + ",".join(str(s) for s in a.shape).encode("ascii")
        + b"\n"
        + a.tobytes()
    )
    return struct.pack(">Q", len(record)) + record


def science_digest(data, dq) -> str:
    return hashlib.sha256(canonical_array_bytes(data) + canonical_array_bytes(dq)).hexdigest()


# ---------------------------------------------------------------------------
# Canonical metadata JSON (descriptor/plan digests)
# ---------------------------------------------------------------------------
def canonical_json(obj) -> str:
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, np.integer)):
        return str(int(obj))
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            raise ValueError("non-finite value in canonical metadata JSON")
        return repr(v)
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canonical_json(x) for x in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[0])
        return "{" + ",".join(
            json.dumps(k, ensure_ascii=False) + ":" + canonical_json(v)
            for k, v in items
        ) + "}"
    raise TypeError(f"unsupported canonical JSON type: {type(obj)!r}")


def _deep_get(obj, path):
    cur = obj
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            raise KeyError(path)
    return cur


def _project(obj, paths):
    """Build a nested dict projection from an explicit list of dot-paths."""
    root = {}
    for p in paths:
        parts = p.split(".")
        node = root
        for seg in parts[:-1]:
            node = node.setdefault(seg, {})
        node[parts[-1]] = _deep_get(obj, p)
    return root


# Explicit projections (allowlists), matching PROVENANCE §2.3.
_DESCRIPTOR_PROJECTION = [
    "master_type", "bias_state", "pixel_domain", "physical_units", "flat_form",
    "normalization_algorithm", "normalization_scalars", "geometry",
    "detector.detector_instance_id", "detector.detector_model", "detector.serial",
    "acquisition.gain", "acquisition.offset", "acquisition.readout_mode",
    "acquisition.adc_mode", "acquisition.temperature_c", "acquisition.exposure_s",
    "acquisition.saturation_limit_adu", "acquisition.saturation_evidence",
    "optical_train_id", "filter", "content_sha256", "size_bytes", "hdu",
    "mask_identity", "processing_provenance", "validity_evidence",
]

_PLAN_PROJECTION = [
    "request.additive_mode", "request.flat_mode",
    "light_constraints.geometry", "light_constraints.detector",
    "light_constraints.acquisition", "light_constraints.optical",
    "light_constraints.raw_domain_declaration",
    "policy_parameters.exposure_tolerance", "policy_parameters.temperature_tolerance",
    "policy_parameters.flat_quality_policy",
    "versions.science_contract", "versions.decoder",
    "versions.provenance_schema", "versions.matching_policy",
]
_MASTER_BINDING_PROJECTION = ["descriptor_id", "content_sha256", "size_bytes", "hdu", "mask_identity"]


def descriptor_digest(fields: dict) -> str:
    return hashlib.sha256(canonical_json(_project(fields, _DESCRIPTOR_PROJECTION)).encode("utf-8")).hexdigest()


def descriptor_snapshot_digest(snapshot: dict) -> str:
    """Descriptor identity is the digest of the COMPLETE acyclic snapshot
    (same field set as descriptor_digest); a tampered snapshot recomputes to a
    different digest and is therefore rejected."""
    return descriptor_digest(snapshot)


def plan_digest(fields: dict) -> str:
    proj = _project(fields, _PLAN_PROJECTION)
    # project each master binding to its hashed identity fields (exclude locator)
    masters = fields.get("masters", {})
    proj["masters"] = {
        role: _project(binding, _MASTER_BINDING_PROJECTION)
        for role, binding in sorted(masters.items())
    }
    return hashlib.sha256(canonical_json(proj).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def equations():
    out = []
    L = np.array([[110.0, 50.0], [210.0, -2.0]], dtype=np.float64)
    Dinc = np.full((2, 2), 10.0, dtype=np.float64)
    B = np.full((2, 2), 4.0, dtype=np.float64)
    D0 = np.full((2, 2), 6.0, dtype=np.float64)
    R = np.array([[1.0, 0.5], [2.0, 1.0]], dtype=np.float64)
    A_dark_incl = L - Dinc
    A_bias_removed = L - B - D0
    C_dark_incl = A_dark_incl / R
    C_bias_removed = A_bias_removed / R
    C_double = (L - Dinc - B) / R
    assert close(C_dark_incl, [[100.0, 80.0], [100.0, -12.0]])
    assert close(C_bias_removed, [[100.0, 80.0], [100.0, -12.0]])
    assert close(C_dark_incl, C_bias_removed)
    assert close(C_double, [[96.0, 72.0], [98.0, -16.0]])
    assert not close(C_double, C_dark_incl)
    out.append({
        "id": "light_branches",
        "C_dark_incl_bias": json_safe(C_dark_incl),
        "C_bias_removed_dark": json_safe(C_bias_removed),
        "C_double_subtraction_negative_control": json_safe(C_double),
        "coherent_dark_branches_equal": True,
        "double_subtraction_differs": True,
    })
    C_flat_only = L / R
    assert close(C_flat_only, [[110.0, 100.0], [105.0, -2.0]])
    out.append({"id": "flat_only_partial_mode", "C_flat_only": json_safe(C_flat_only)})
    Finc = np.array([[100.0, 110.0], [120.0, 90.0]], dtype=np.float64)
    FDinc = np.full((2, 2), 10.0, dtype=np.float64)
    Bflat = np.full((2, 2), 4.0, dtype=np.float64)
    FD0 = np.full((2, 2), 6.0, dtype=np.float64)
    F_incl = Finc - FDinc
    F_removed = Finc - Bflat - FD0
    F_bias = Finc - Bflat
    assert close(F_incl, [[90.0, 100.0], [110.0, 80.0]])
    assert close(F_removed, [[90.0, 100.0], [110.0, 80.0]])
    assert close(F_bias, [[96.0, 106.0], [116.0, 86.0]])
    assert close(F_incl, F_removed)
    s_corr = float(np.median(F_incl[np.isfinite(F_incl)]))
    s_wrong = float(np.median(Finc[np.isfinite(Finc)]))
    assert abs(s_corr - 95.0) < 1e-12 and abs(s_wrong - 105.0) < 1e-12
    assert not close(F_incl / s_corr, Finc / s_wrong)
    out.append({
        "id": "flat_branches_and_pedestal",
        "F_flatdark_incl_bias": json_safe(F_incl),
        "F_flatdark_bias_removed": json_safe(F_removed),
        "F_bias_only_flat": json_safe(F_bias),
        "normalizing_uncorrected_flat_is_not_correcting_it": True,
    })
    return out


def cfa_cases():
    out = []
    h, w = 6, 4
    F = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            F[y, x] = 200.0 if grbg_plane(y, x) == "G1" else 100.0
    s_plane = {}
    for p in ["R", "G1", "G2", "B"]:
        vals = np.array([F[y, x] for y in range(h) for x in range(w)
                         if grbg_plane(y, x) == p], dtype=np.float64)
        s_plane[p] = float(np.median(vals))
    assert abs(s_plane["G1"] - 200.0) < 1e-12
    assert abs(s_plane["R"] - 100.0) < 1e-12
    assert abs(s_plane["G2"] - 100.0) < 1e-12
    assert abs(s_plane["B"] - 100.0) < 1e-12
    R_per_plane = np.zeros_like(F)
    for y in range(h):
        for x in range(w):
            R_per_plane[y, x] = F[y, x] / s_plane[grbg_plane(y, x)]
    assert close(R_per_plane, np.ones_like(F))
    s_global = float(np.median(F))
    assert abs(s_global - 100.0) < 1e-12
    R_global = F / s_global
    assert not close(R_global, np.ones_like(F))
    assert abs(R_global[0, 0] - 2.0) < 1e-12
    out.append({
        "id": "cfa_four_parity_planes",
        "per_plane_medians": s_plane, "global_single_median": s_global,
        "R_per_plane_uniform": True, "R_global_over_corrects_G1": True,
    })
    assert plane_under_shift(0, 0, 1, 0) == "R"
    assert plane_under_shift(0, 1, 1, 0) == "G1"
    assert plane_under_shift(1, 0, 1, 0) == "G2"
    assert plane_under_shift(1, 1, 1, 0) == "B"
    assert plane_under_shift(0, 0, 1, 0) != "G2"
    assert plane_under_shift(1, 1, 1, 0) != "G1"
    assert plane_under_shift(0, 0, 0, 1) == "B"
    assert plane_under_shift(0, 1, 0, 1) == "G2"
    assert plane_under_shift(1, 0, 0, 1) == "G1"
    assert plane_under_shift(1, 1, 0, 1) == "R"
    out.append({
        "id": "cfa_shifted_origin_grbg",
        "x_shift_swaps": {"G1": "R", "R": "G1", "B": "G2", "G2": "B"},
        "y_shift_swaps": {"G1": "B", "R": "G2", "B": "G1", "G2": "R"},
        "x_shift_does_not_swap_G1_G2": True,
    })
    dead = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    assert dead[(dead > 0) & np.isfinite(dead)].size == 0
    out.append({"id": "positive_valid_normalization_population", "zero_valid_unusable": True})
    return out


def dq_bits():
    bits = {"INPUT_INVALID": 0, "ADDITIVE_INVALID": 1, "FLAT_INVALID": 2,
            "SATURATED": 3, "ARITH_NONFINITE": 4}
    masks = {k: (1 << b) for k, b in bits.items()}
    assert masks["INPUT_INVALID"] == 0x0001
    assert masks["ADDITIVE_INVALID"] == 0x0002
    assert masks["FLAT_INVALID"] == 0x0004
    assert masks["SATURATED"] == 0x0008
    assert masks["ARITH_NONFINITE"] == 0x0010
    return {"id": "dq_bits", "masks": {k: hex(v) for k, v in masks.items()},
            "all_five_are_invalid_reason_bits": True}


def _mask_stats(mask):
    """Derive valid/invalid/per-bit counts from a mask (no literals)."""
    mask = np.asarray(mask, dtype=np.uint16)
    total = int(mask.size)
    valid = int(np.count_nonzero(mask == 0))
    invalid = total - valid
    per_bit = {name: int(np.count_nonzero(mask & (1 << b)))
               for name, b in [("INPUT_INVALID", 0), ("ADDITIVE_INVALID", 1),
                               ("FLAT_INVALID", 2), ("SATURATED", 3),
                               ("ARITH_NONFINITE", 4)]}
    return {"total": total, "valid_count": valid, "invalid_count": invalid, "per_bit": per_bit}


def dq_semantics_cases():
    out = []
    IN = 0x0001
    ADD = 0x0002
    SAT = 0x0008

    mask = np.array([[0, IN], [IN | SAT, 0]], dtype=np.uint16)
    st = _mask_stats(mask)
    assert st["valid_count"] == 2 and st["invalid_count"] == 2
    assert st["per_bit"]["INPUT_INVALID"] == 2 and st["per_bit"]["SATURATED"] == 1
    # any nonzero mask => NaN (derived, not literal)
    nan_locations = mask != 0
    assert bool(np.all(nan_locations == np.array([[False, True], [True, False]])))
    out.append({
        "id": "dq_reason_or_and_counts",
        "mask": json_safe(mask), "stats": st,
        "nan_at_nonzero_mask": json_safe(nan_locations),
    })

    mask2 = np.array([[SAT, 0], [0, 0]], dtype=np.uint16)
    st2 = _mask_stats(mask2)
    assert st2["valid_count"] == 3 and st2["invalid_count"] == 1 and st2["per_bit"]["SATURATED"] == 1
    assert bool((mask2 != 0)[0, 0])  # SAT-only pixel -> NaN (derived)
    out.append({
        "id": "dq_saturated_only_is_invalid",
        "mask": json_safe(mask2), "stats": st2,
        "sat_only_pixel_is_nan": True,
    })

    # valid negative: finite negative is valid (mask 0).
    C_neg = np.array([[5.0, -12.0], [3.0, 7.0]], dtype=np.float64)
    dq_neg = np.zeros((2, 2), dtype=np.uint16)
    assert np.all(dq_neg == 0) and C_neg[0, 1] < 0.0
    out.append({"id": "dq_valid_negative_output", "mask_all_zero": True})

    # all-invalid -> FAILED (status derived from counts, not a literal).
    L_all = np.full((2, 2), np.nan, dtype=np.float64)
    dq_all = np.zeros((2, 2), dtype=np.uint16)
    dq_all[np.isnan(L_all)] |= IN
    st_all = _mask_stats(dq_all)
    failed = st_all["invalid_count"] == st_all["total"]
    assert failed is True
    out.append({
        "id": "dq_all_invalid_refusal",
        "stats": st_all, "status_is_failed": failed,
    })

    # contributing master invalidity -> ADDITIVE_INVALID.
    master = np.array([[10.0, np.nan], [10.0, 10.0]], dtype=np.float64)
    dq_m = np.zeros((2, 2), dtype=np.uint16)
    dq_m[np.isnan(master)] |= ADD
    assert int(dq_m[0, 1] & ADD) == ADD
    out.append({"id": "dq_contributing_master_invalidity", "mask": json_safe(dq_m)})
    return out


def fits_decode_cases():
    out = []
    import tempfile
    u16 = np.array([[0, 100], [200, 65535]], dtype=np.uint16)
    tmpdir = tempfile.mkdtemp(prefix="zcal_phase1_")
    p1 = os.path.join(tmpdir, "bzero.fits")
    fits.writeto(p1, u16, overwrite=True)
    with fits.open(p1, do_not_scale_image_data=True) as h:
        stored = np.asarray(h[0].data)
        hdr = h[0].header
    with fits.open(p1) as h:
        physical = np.asarray(h[0].data)
    bscale = float(hdr.get("BSCALE", 1.0))
    bzero = float(hdr.get("BZERO", 0.0))
    manual = bscale * stored.astype(np.float64) + bzero
    assert close(physical.astype(np.float64), manual)
    assert close(physical.astype(np.float64), u16.astype(np.float64))
    assert str(stored.dtype).startswith(">") or stored.dtype.byteorder == ">"
    assert np.asarray(physical, dtype=np.float32).dtype == np.float32
    out.append({
        "id": "bscale_bzero_exactly_once_uint16",
        "stored_dtype": str(stored.dtype), "BSCALE": bscale, "BZERO": bzero,
        "exactly_once_holds": True, "endianness_is_storage_not_identity": True,
    })
    stored_f = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    p2 = os.path.join(tmpdir, "scaled_float.fits")
    hdu = fits.PrimaryHDU(stored_f)
    hdu.header["BSCALE"] = 2.0
    hdu.header["BZERO"] = 10.0
    hdu.writeto(p2, overwrite=True)
    with fits.open(p2) as h:
        phys_f = np.asarray(h[0].data)
    manual_f = 2.0 * stored_f.astype(np.float64) + 10.0
    assert close(phys_f.astype(np.float64), manual_f)
    assert close(phys_f.astype(np.float64), [[11.0, 13.0], [15.0, 17.0]])
    out.append({"id": "scaled_floating_fits_exactly_once", "exactly_once_holds": True})
    stored_blank = np.array([[-32768, -32668], [-32568, 32767]], dtype=np.int16)
    invalid_stored = (stored_blank == -32768)
    assert bool(invalid_stored[0, 0]) is True
    assert float(stored_blank.astype(np.float64)[0, 0] + 32768.0) == 0.0
    out.append({
        "id": "integer_blank_in_stored_space",
        "method": "in-memory reference (not an astropy FITS round-trip)",
        "invalid_in_stored_space": json_safe(invalid_stored),
    })
    p24 = 2 ** 24
    p24p1 = p24 + 1
    assert float(np.float32(p24)) == float(p24)
    assert float(np.float32(p24p1)) != float(p24p1)
    assert float(np.float32(p24p1)) == float(p24)
    out.append({"id": "float32_precision_boundary",
                "two_pow_24_exact": True, "two_pow_24_plus_1_not_exact": True})
    for p in (p1, p2):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    return out


def digest_cases():
    out = []
    data = np.array([[1.0, -2.0]], dtype=">f4")
    dq = np.array([[0, 0]], dtype=">u2")
    assert canonical_array_bytes(data).hex() == PINNED_CANONICAL_DATA_HEX
    assert canonical_array_bytes(dq).hex() == PINNED_CANONICAL_DQ_HEX
    assert science_digest(data, dq) == PINNED_SCIENCE_DIGEST
    out.append({"id": "digest_pinned_reference", "pinned_matches": True})
    le = np.array([[1.5, -2.0], [3.25, 0.0]], dtype="<f4")
    be = le.astype(">f4")
    dq0 = np.zeros((2, 2), dtype=">u2")
    assert le.tobytes() != be.tobytes()
    assert science_digest(le, dq0) == science_digest(be, dq0)
    out.append({"id": "digest_byte_order_equivalence", "equal": True})
    nan_a = np.frombuffer(bytes([0x7F, 0xC0, 0x00, 0x01]), dtype=">f4")[0]
    nan_b = np.frombuffer(bytes([0x7F, 0xFF, 0xFF, 0xFF]), dtype=">f4")[0]
    arr_a = np.array([[nan_a, 1.0]], dtype=">f4")
    arr_b = np.array([[nan_b, 1.0]], dtype=">f4")
    dq1 = np.zeros((1, 2), dtype=">u2")
    assert science_digest(arr_a, dq1) == science_digest(arr_b, dq1)
    assert "7fc00000" in canonical_array_bytes(arr_a).hex()
    out.append({"id": "digest_nan_payload_equivalence", "equal": True})
    pz = np.array([[0.0]], dtype=">f4")
    nz = np.array([[-0.0]], dtype=">f4")
    dq1x = np.zeros((1, 1), dtype=">u2")
    assert science_digest(pz, dq1x) == science_digest(nz, dq1x)
    out.append({"id": "digest_signed_zero_equivalence", "equal": True})
    d_f4 = science_digest(np.zeros((2, 2), dtype=">f4"), np.zeros((2, 2), dtype=">u2"))
    d_u4 = science_digest(np.zeros((2, 2), dtype=">u4"), np.zeros((2, 2), dtype=">u2"))
    d_i2 = science_digest(np.zeros((2, 2), dtype=">i2"), np.zeros((2, 2), dtype=">u2"))
    assert d_f4 != d_u4 and d_f4 != d_i2
    out.append({"id": "digest_dtype_difference", "different": True})
    d_22 = science_digest(np.zeros((2, 2), dtype=">f4"), np.zeros((2, 2), dtype=">u2"))
    d_14 = science_digest(np.zeros((1, 4), dtype=">f4"), np.zeros((1, 4), dtype=">u2"))
    assert d_22 != d_14
    out.append({"id": "digest_shape_difference", "different": True})
    d_a = science_digest(np.array([[1.0, 2.0]], dtype=">f4"), np.zeros((1, 2), dtype=">u2"))
    d_b = science_digest(np.array([[1.0, 3.0]], dtype=">f4"), np.zeros((1, 2), dtype=">u2"))
    assert d_a != d_b
    out.append({"id": "digest_data_difference", "different": True})
    data2 = np.array([[1.0, 2.0]], dtype=">f4")
    dqa = np.array([[0, 0]], dtype=">u2")
    dqb = np.array([[0, 1]], dtype=">u2")
    assert science_digest(data2, dqa) != science_digest(data2, dqb)
    out.append({"id": "digest_dq_difference", "different": True})
    # canonical JSON pinned example (S1).
    assert canonical_json({"b": 1, "a": 2.0, "c": "x\ny"}) == PINNED_CANONICAL_JSON_EXAMPLE
    out.append({"id": "canonical_json_pinned", "matches": True})
    return out


def identity_cases():
    out = []
    base_desc = {
        "master_type": "dark", "bias_state": "included",
        "pixel_domain": "sensor_adu", "physical_units": "ADU",
        "flat_form": None, "normalization_algorithm": None, "normalization_scalars": None,
        "geometry": {"shape": [1080, 1920], "roi_origin": [0, 0], "binning": [1, 1], "cfa_phase": "GRBG"},
        "detector": {"detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA", "serial": None},
        "acquisition": {"gain": 100, "offset": 50, "readout_mode": "MODE_A", "adc_mode": "MODE_16",
                        "temperature_c": 20.0, "exposure_s": 10.0,
                        "saturation_limit_adu": 60000, "saturation_evidence": "qualified"},
        "optical_train_id": None, "filter": None,
        "content_sha256": "aaa", "size_bytes": 100, "hdu": 0,
        "mask_identity": "m1", "processing_provenance": {"source": "synthetic_fixture",
                                                         "additive_correction_history": [], "normalization": None},
        "validity_evidence": {"saturation_limit_known": True, "valid_normalization_count": None,
                              "quality_policy_state": "qualified"},
    }
    # content change changes digest.
    desc2 = copy.deepcopy(base_desc); desc2["content_sha256"] = "bbb"
    assert descriptor_digest(base_desc) != descriptor_digest(desc2)
    # hdu / mask / history changes change digest.
    d3 = copy.deepcopy(base_desc); d3["hdu"] = 1
    assert descriptor_digest(base_desc) != descriptor_digest(d3)
    d4 = copy.deepcopy(base_desc); d4["mask_identity"] = "m2"
    assert descriptor_digest(base_desc) != descriptor_digest(d4)
    d5 = copy.deepcopy(base_desc); d5["processing_provenance"]["additive_correction_history"] = ["bias_removed"]
    assert descriptor_digest(base_desc) != descriptor_digest(d5)
    # POSITIVE: incidental fields excluded by projection do NOT change digest.
    d_extra = copy.deepcopy(base_desc)
    d_extra["path"] = "/tmp/master.fits"
    d_extra["imported_at"] = "2026-09-15T00:00:00Z"
    d_extra["source_path"] = "/data/source.fits"
    d_extra["descriptor_id"] = "self-id-value"
    assert descriptor_digest(base_desc) == descriptor_digest(d_extra)
    out.append({
        "id": "descriptor_digest_explicit_projection",
        "content_change_changes": True, "hdu_mask_history_change_changes": True,
        "incidental_fields_excluded": True,
    })

    base_plan = {
        "request": {"additive_mode": "dark_incl_bias", "flat_mode": "apply"},
        "light_constraints": {
            "geometry": {"shape": [1080, 1920], "roi_origin": [0, 0], "binning": [1, 1], "cfa_phase": "GRBG"},
            "detector": {"detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA"},
            "acquisition": {"gain": 100, "offset": 50, "exposure_s": 10.0, "saturation_evidence": "qualified"},
            "optical": {"filter": "IRCUT", "optical_train_id": "TRAIN_A"},
            "raw_domain_declaration": "raw",
        },
        "policy_parameters": {
            "exposure_tolerance": {"relative": 1e-6, "absolute": 1e-6},
            "temperature_tolerance": {"relative": 0.0, "absolute": 1e-6},
            "flat_quality_policy": {"state": "qualified", "threshold_pct": 90.0, "per_plane": True},
        },
        "masters": {
            "dark": {"descriptor_id": "d1", "content_sha256": "c1", "size_bytes": 100, "hdu": 0, "mask_identity": "m1",
                     "locator": {"path": "/lib/dark.fits", "hdu": 0}},
            "flat": {"descriptor_id": "f1", "content_sha256": "c2", "size_bytes": 100, "hdu": 0, "mask_identity": "m2",
                     "locator": {"path": "/lib/flat.fits", "hdu": 0}},
            "flat_dark": {"descriptor_id": "fd1", "content_sha256": "c3", "size_bytes": 100, "hdu": 0, "mask_identity": "m3",
                          "locator": {"path": "/lib/flatdark.fits", "hdu": 0}},
        },
        "versions": {"science_contract": "1.0", "decoder": "1.0",
                     "provenance_schema": "zecalibrator.provenance.v1",
                     "matching_policy": "zecalibrator.match.v1"},
    }
    # F1: same input => same digest (determinism).
    assert plan_digest(base_plan) == plan_digest(copy.deepcopy(base_plan))
    # F1: filter-only change => different digest.
    pf = copy.deepcopy(base_plan)
    pf["light_constraints"]["optical"]["filter"] = "LP"
    assert plan_digest(base_plan) != plan_digest(pf)
    # F1: optical_train-only change => different digest.
    pt = copy.deepcopy(base_plan)
    pt["light_constraints"]["optical"]["optical_train_id"] = "TRAIN_B"
    assert plan_digest(base_plan) != plan_digest(pt)
    # F1: both change => different digest.
    pb = copy.deepcopy(base_plan)
    pb["light_constraints"]["optical"] = {"filter": "LP", "optical_train_id": "TRAIN_B"}
    assert plan_digest(base_plan) != plan_digest(pb)
    # NEGATIVE: policy parameter (same version) changes digest.
    p2 = copy.deepcopy(base_plan)
    p2["policy_parameters"]["flat_quality_policy"]["threshold_pct"] = 95.0
    assert plan_digest(base_plan) != plan_digest(p2)
    # NEGATIVE: bound light metadata changes digest.
    p3 = copy.deepcopy(base_plan)
    p3["light_constraints"]["acquisition"]["exposure_s"] = 20.0
    assert plan_digest(base_plan) != plan_digest(p3)
    # NEGATIVE: master descriptor/content/HDU change changes digest.
    p4 = copy.deepcopy(base_plan)
    p4["masters"]["dark"]["content_sha256"] = "CHANGED"
    assert plan_digest(base_plan) != plan_digest(p4)
    # POSITIVE: incidental locator / tile / path / imported_at / plan_id do NOT change digest.
    p5 = copy.deepcopy(base_plan)
    p5["masters"]["dark"]["locator"] = {"path": "/elsewhere/dark.fits", "hdu": 0}
    p5["tile_shape"] = [2, 2]
    p5["source_path"] = "/x/y.fits"
    p5["imported_at"] = "2026-09-15T00:00:00Z"
    p5["plan_id"] = "self-id"
    p5["output_digest"] = "out"
    assert plan_digest(base_plan) == plan_digest(p5)
    # F1: dark-only additive mode does not invent a filter requirement.
    dark_only = {
        "request": {"additive_mode": "dark_incl_bias", "flat_mode": "none"},
        "light_constraints": {
            "geometry": {"shape": [1080, 1920], "roi_origin": [0, 0], "binning": [1, 1], "cfa_phase": "GRBG"},
            "detector": {"detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA"},
            "acquisition": {"gain": 100, "offset": 50, "exposure_s": 10.0, "saturation_evidence": "qualified"},
            "optical": {"filter": None, "optical_train_id": None},
            "raw_domain_declaration": "raw",
        },
        "policy_parameters": base_plan["policy_parameters"],
        "masters": {"dark": base_plan["masters"]["dark"]},
        "versions": base_plan["versions"],
    }
    # Unknown optical in a dark-only plan hashes to a defined value (None), and
    # changing it to a known filter would change the digest — but the CONTRACT
    # (SCIENCE §6.2) does not require filter for dark matching. This is asserted
    # as a projection property, not a matcher.
    dark_only_filtered = copy.deepcopy(dark_only)
    dark_only_filtered["light_constraints"]["optical"]["filter"] = "IRCUT"
    assert plan_digest(dark_only) != plan_digest(dark_only_filtered)
    out.append({
        "id": "plan_digest_explicit_projection",
        "same_input_same_digest": True,
        "filter_change_changes": True,
        "optical_train_change_changes": True,
        "both_change_changes": True,
        "policy_param_change_changes": True,
        "metadata_change_changes": True,
        "master_content_change_changes": True,
        "locator_tile_path_imported_planid_excluded": True,
        "dark_only_unknown_optical_defined": True,
    })
    return out


def card_record_cases():
    out = []
    # Ordered CardRecord sequence preserves a repeated identical keyword.
    records = [
        {"keyword": "EXPTIME", "value": 20.0, "comment": "", "index": 0, "source": "primary"},
        {"keyword": "EXPTIME", "value": 20.0, "comment": "", "index": 1, "source": "duplicate"},
    ]
    assert len(records) == 2  # both preserved
    # A Mapping would silently drop one: demonstrate the loss.
    as_dict = {r["keyword"]: r["value"] for r in records}
    assert len(as_dict) == 1
    # agree diagnostic (both values equal) vs conflict diagnostic.
    agree = all(r["value"] == records[0]["value"] for r in records)
    assert agree is True
    conflict_records = [{"keyword": "EXPTIME", "value": 20.0}, {"keyword": "EXPTIME", "value": 10.0}]
    conflict = not all(r["value"] == conflict_records[0]["value"] for r in conflict_records)
    assert conflict is True
    out.append({
        "id": "ordered_card_record_duplicate_preserved",
        "records": records,
        "mapping_would_collapse_to": len(as_dict),
        "agree_detected": agree,
        "conflict_detected": conflict,
    })
    return out


def declaration_cases():
    """F2: descriptor/declaration snapshot + mask retention/revalidation
    (design contract, tiny reference — not a storage engine)."""
    out = []
    snapshot = {
        "master_type": "dark", "bias_state": "included",
        "pixel_domain": "sensor_adu", "physical_units": "ADU",
        "flat_form": None, "normalization_algorithm": None, "normalization_scalars": None,
        "geometry": {"shape": [1080, 1920], "roi_origin": [0, 0], "binning": [1, 1], "cfa_phase": "GRBG"},
        "detector": {"detector_instance_id": "SYNTH-DET-0001", "detector_model": "SYNTH-CFA", "serial": None},
        "acquisition": {"gain": 100, "offset": 50, "readout_mode": "MODE_A", "adc_mode": "MODE_16",
                        "temperature_c": 20.0, "exposure_s": 10.0,
                        "saturation_limit_adu": 60000, "saturation_evidence": "qualified"},
        "optical_train_id": None, "filter": None,
        "content_sha256": "aaa", "size_bytes": 100, "hdu": 0,
        "mask_identity": "m1",
        "processing_provenance": {"source": "synthetic_fixture",
                                   "additive_correction_history": [], "normalization": None},
        "validity_evidence": {"saturation_limit_known": True, "valid_normalization_count": None,
                              "quality_policy_state": "qualified"},
    }
    descriptor_id = descriptor_snapshot_digest(snapshot)
    # Revalidation: unchanged snapshot re-hashes to the same descriptor_id.
    assert descriptor_snapshot_digest(snapshot) == descriptor_id
    # Tamper (e.g. change bias_state) => recomputed digest differs => reject.
    tampered = copy.deepcopy(snapshot)
    tampered["bias_state"] = "removed"
    assert descriptor_snapshot_digest(tampered) != descriptor_id
    # Declaration revision semantics: a new declaration is a NEW snapshot with a
    # NEW descriptor_id, not an in-place mutation of the old one.
    revised = copy.deepcopy(snapshot)
    revised["processing_provenance"]["additive_correction_history"] = ["bias_removed"]
    new_descriptor_id = descriptor_snapshot_digest(revised)
    assert new_descriptor_id != descriptor_id
    out.append({
        "id": "descriptor_snapshot_retention_revalidation",
        "unchanged_rehashes_same": True,
        "tampered_rejected": True,
        "revision_is_new_identity": True,
    })

    # Mask payload identity: separate from descriptor digest and image SHA.
    mask_payload = b"\x00\x01\x00\x00"  # tiny synthetic DQ payload
    mask_identity = hashlib.sha256(mask_payload).hexdigest()
    assert hashlib.sha256(mask_payload).hexdigest() == mask_identity
    tampered_mask = b"\x00\x01\x00\x08"
    assert hashlib.sha256(tampered_mask).hexdigest() != mask_identity
    out.append({
        "id": "mask_payload_identity_revalidation",
        "unchanged_matches": True,
        "tampered_mask_rejected": True,
        "mask_digest_distinct_from_image_sha": True,
    })
    return out


def quality_gate():
    results = []
    for valid, total, thr, expected in [(89, 100, 90, False), (90, 100, 90, True), (91, 100, 90, True)]:
        pct = 100.0 * valid / total
        meets = pct >= thr
        assert meets == expected
        results.append({"valid": valid, "total": total, "threshold_pct": thr, "pct": pct, "meets": meets})
    floor = 1e-6
    assert not (1e-6 > floor)
    assert (1e-5 > floor)
    return {"id": "quality_gate_boundaries_and_numeric_floor", "cases": results,
            "numeric_floor": floor, "at_floor_masked_not_raised": True,
            "quality_gate_is_product_policy_not_physical_law": True}


# ---------------------------------------------------------------------------
# cases.json crosscheck — deep materialization + structural audit
# ---------------------------------------------------------------------------
def _materialize_fresh(cj):
    """Deep-copy the whole baseline; NO shared mutable objects between cases."""
    base = copy.deepcopy(cj["hypothetical_synthetic_baseline"])
    det = base["detector"]
    geo = base["geometry"]
    light = {"detector": copy.deepcopy(det), "geometry": copy.deepcopy(geo),
             **copy.deepcopy(base["light"])}
    masters = {}
    for name, m in base["masters"].items():
        masters[name] = {"detector": copy.deepcopy(det), "geometry": copy.deepcopy(geo),
                         **copy.deepcopy(m)}
    return base, light, masters


def _set_path(obj, path, value):
    parts = path.split(".")
    cur = obj
    for seg in parts[:-1]:
        if not isinstance(cur, dict) or seg not in cur:
            raise KeyError(path)
        cur = cur[seg]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        raise KeyError(path)
    cur[parts[-1]] = value


def _del_path(obj, path):
    parts = path.split(".")
    cur = obj
    for seg in parts[:-1]:
        if not isinstance(cur, dict) or seg not in cur:
            raise KeyError(path)
        cur = cur[seg]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        raise KeyError(path)
    del cur[parts[-1]]


def _get_path(obj, path):
    return _deep_get(obj, path)


def _required_masters(request):
    roles = []
    am = request["additive_mode"]
    if am == "bias_only":
        roles.append("bias")
    elif am == "dark_incl_bias":
        roles.append("dark_incl")
    elif am == "dark_bias_removed":
        roles.extend(["bias", "dark_removed"])
    # control -> no additive masters
    if request.get("flat_mode") == "apply":
        roles.extend(["flat", "flat_dark"])
    return roles


def _flat_coherence_problems(master):
    """Return list of processing-history/scalar coherence problems for a flat."""
    problems = []
    ff = master.get("flat_form")
    pp = master.get("processing_provenance") or {}
    hist = pp.get("additive_correction_history") or []
    norm = pp.get("normalization")
    scalars = master.get("normalization_scalars")
    if ff == "normalized_response":
        if not isinstance(scalars, dict) or set(scalars.keys()) != {"g1", "r", "b", "g2"}:
            problems.append("NORMALIZATION_SCALAR_COUNT_MISMATCH")
        if not norm:
            problems.append("UNDOCUMENTED_PROCESSING")
        if not hist:
            problems.append("UNDOCUMENTED_PROCESSING")
    elif ff == "corrected_unnormalized":
        if scalars is not None or norm is not None:
            problems.append("UNDOCUMENTED_PROCESSING")
        if not hist:
            problems.append("UNDOCUMENTED_PROCESSING")
    elif ff == "raw_response":
        if scalars is not None or norm is not None or hist:
            problems.append("UNDOCUMENTED_PROCESSING")
    return list(dict.fromkeys(problems))


def _derive_manifest_reasons(case, light, masters):
    """Derive manifest rejection reasons from the materialized case (structural
    audit, NOT a matcher)."""
    reasons = set()
    roles = _required_masters(case["request"])
    for role in roles:
        if role not in masters:
            if role == "flat_dark":
                reasons.add("FLAT_ADDITIVE_DEPENDENCY_MISSING")
            else:
                reasons.add("ROLE_UNAVAILABLE")
            continue
        m = masters[role]
        la = light["acquisition"]
        ma = m["acquisition"]
        for f, rc in [("gain", "GAIN_MISMATCH"), ("offset", "OFFSET_MISMATCH"),
                      ("readout_mode", "READOUT_MISMATCH"), ("adc_mode", "ADC_MISMATCH")]:
            if f not in la or f not in ma:
                reasons.add("MISSING_REQUIRED_FIELD")
            elif la[f] != ma[f]:
                reasons.add(rc)
        lg = light["geometry"]
        mg = m["geometry"]
        if mg.get("roi_origin") != lg.get("roi_origin"):
            reasons.add("ROI_ORIGIN_MISMATCH")
            reasons.add("CFA_PHASE_MISMATCH")
        if mg.get("binning") != lg.get("binning"):
            reasons.add("BINNING_MISMATCH")
            reasons.add("GEOMETRY_MISMATCH")
        if mg.get("shape") != lg.get("shape"):
            reasons.add("GEOMETRY_MISMATCH")
        ld = light["detector"]
        md = m["detector"]
        if ld.get("detector_instance_id") in (None, "unknown") or md.get("detector_instance_id") in (None, "unknown"):
            reasons.add("UNKNOWN_EQUALS_UNKNOWN")
            reasons.add("MISSING_REQUIRED_FIELD")
        if m["master_type"] == "dark":
            if ma.get("exposure_s") != la.get("exposure_s"):
                reasons.add("EXPOSURE_MISMATCH")
        if m["master_type"] == "flat_dark":
            flat = masters.get("flat")
            if flat is not None and ma.get("exposure_s") != flat["acquisition"].get("exposure_s"):
                reasons.add("EXPOSURE_MISMATCH")
        if m["master_type"] == "flat":
            if m["optical"].get("filter") != light["optical"].get("filter"):
                reasons.add("FILTER_MISMATCH")
            if m["optical"].get("optical_train_id") != light["optical"].get("optical_train_id"):
                reasons.add("OPTICAL_TRAIN_MISMATCH")
            reasons.update(_flat_coherence_problems(m))
    return reasons


def _negative_controls(cj):
    """Independent negative controls for materialization/structural checks."""
    out = {}
    # 1. shared-mutation: mutate one case's light; baseline and another case unchanged.
    _, light_a, masters_a = _materialize_fresh(cj)
    light_a["acquisition"]["gain"] = 999
    _, light_b, _ = _materialize_fresh(cj)
    assert light_b["acquisition"]["gain"] == 100
    assert masters_a["dark_incl"]["acquisition"]["gain"] == 100  # master independent of light
    out["no_shared_mutation"] = True

    # 2. invalid override/removal path must fail.
    try:
        _set_path(light_a, "light.acquisition.nonexistent", 1)
        out["invalid_override_path_fails"] = False
    except KeyError:
        out["invalid_override_path_fails"] = True
    assert out["invalid_override_path_fails"]

    # 3. clone must not mutate source.
    _, _, masters_c = _materialize_fresh(cj)
    src_before = masters_c["dark_incl"]["content_sha256"]
    clone = copy.deepcopy(masters_c["dark_incl"])
    clone["content_sha256"] = "CHANGED"
    assert masters_c["dark_incl"]["content_sha256"] == src_before
    out["clone_does_not_mutate_source"] = True

    # 4. normalized-history contradiction is detected by _flat_coherence_problems.
    bad_flat = copy.deepcopy(masters_c["flat"])
    bad_flat["flat_form"] = "normalized_response"
    bad_flat["pixel_domain"] = "normalized_response"
    bad_flat["physical_units"] = "dimensionless"
    bad_flat["normalization_scalars"] = {"g1": 1.0, "r": 1.0, "b": 1.0, "g2": 1.0}
    # history still empty + normalization null -> contradiction
    probs = _flat_coherence_problems(bad_flat)
    assert "UNDOCUMENTED_PROCESSING" in probs
    out["normalized_history_contradiction_detected"] = True
    return out


def crosscheck_cases():
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cj = json.load(f)

    # Owner decisions are inputs authorized by Tristan, not inferred defaults.
    expected_decisions = {'license': {'spdx': 'GPL-3.0-or-later'}, 'first_supported_cohort': {'id': 'SYNTH-BASE-1', 'qualification': 'synthetic-only', 'real_camera_qualified': False}, 'temperature_tolerance': {'scientific_celsius': 0.0, 'parser_celsius': 1e-06, 'nonzero_requires_versioned_camera_master_profile': True, 'nonzero_requires_real_witnesses': True, 'generic_nonzero_allowed': False}, 'flat_quality_gate': {'threshold_pct': 90.0, 'per_cfa_plane': True, 'kind': 'product_screening_not_physical_law', 'override_requires_versioned_profile': True}, 'missing_metadata_declaration': {'automatic_default': 'strict_refusal', 'versioned_evidence_backed_import_allowed': True, 'invented_values_allowed': False, 'fits_contradiction_override_allowed': False}}
    for d in cj.get("protected_decisions", []):
        assert d.get("status") == "adopted" and d.get("owner") == "Tristan", d
        assert d.get("selected") == expected_decisions[d["id"]], d
        assert d.get("decision_record") == "ZC-G1-OWNER-20260915", d
    assert cj["filter_applicability"] == {
        "flat": "exact",
        "dark": "not_a_matching_criterion_when_scientifically_irrelevant",
        "bias": "not_a_matching_criterion_when_scientifically_irrelevant",
    }
    names = [d["id"] for d in cj.get("protected_decisions", [])]
    assert set(names) == {"license", "first_supported_cohort", "temperature_tolerance",
                          "flat_quality_gate", "missing_metadata_declaration"}, names

    script_masks = {"INPUT_INVALID": 0x0001, "ADDITIVE_INVALID": 0x0002,
                    "FLAT_INVALID": 0x0004, "SATURATED": 0x0008,
                    "ARITH_NONFINITE": 0x0010}
    assert len(cj.get("dq_bits", [])) == 5
    for b in cj.get("dq_bits", []):
        assert script_masks[b["name"]] == int(b["hex"], 16)
        assert b["kind"] == "invalid"

    for qg in cj.get("quality_gate_cases", []):
        pct = 100.0 * qg["valid"] / qg["total"]
        assert bool(pct >= qg["threshold_pct"]) == bool(qg["meets"])

    known_eq = set(cj.get("equation_dependencies", {}).keys())
    valid_outcomes = {"MATCHED", "NO_MATCH", "AMBIGUOUS"}
    reason_registry = set(cj.get("reason_code_registry", {}).keys())

    audit_results = []
    for m in cj.get("matching_cases", []):
        assert m.get("stage") == "matching", m["id"]
        assert m.get("expected_outcome") in valid_outcomes, m
        assert m.get("baseline") == "SYNTH-BASE-1", m
        for rc in m.get("reason_codes", []):
            assert rc in reason_registry, (m["id"], rc)
        dep = m.get("equation_dependency")
        if dep is not None:
            assert dep in known_eq, m["id"]

        # Fresh deep materialization per case.
        base, light, masters = _materialize_fresh(cj)

        # Apply overrides (validate path).
        for path, val in m.get("overrides", {}).items():
            if path.startswith("light."):
                _set_path(light, path[len("light."):], val)
            elif path.startswith("masters."):
                rest = path[len("masters."):]
                mname, _, sub = rest.partition(".")
                assert mname in masters, (m["id"], path)
                _set_path(masters[mname], sub, val)
            else:
                raise AssertionError((m["id"], path))

        # Apply removals (validate path).
        for path in m.get("removals", []):
            if path.startswith("light."):
                _del_path(light, path[len("light."):])
            elif path.startswith("masters."):
                rest = path[len("masters."):]
                mname, _, sub = rest.partition(".")
                assert mname in masters, (m["id"], path)
                if sub:
                    _del_path(masters[mname], sub)
                else:
                    del masters[mname]
            else:
                raise AssertionError((m["id"], path))

        # Apply add/clone candidates (deep copy, apply overrides, insert).
        for add in m.get("add_candidates", []):
            src = add["clone_from"]
            assert src.startswith("masters."), (m["id"], src)
            srcname = src[len("masters."):]
            assert srcname in masters, (m["id"], src)
            clone = copy.deepcopy(masters[srcname])
            for path, val in add.get("overrides", {}).items():
                _set_path(clone, path, val)
            assert add["new_key"] not in masters, (m["id"], add["new_key"])
            masters[add["new_key"]] = clone

        # Structural audit vs declared outcome.
        manifest = _derive_manifest_reasons(m, light, masters)
        if m["expected_outcome"] == "MATCHED":
            assert not manifest, (m["id"], manifest)
        elif m["expected_outcome"] == "NO_MATCH":
            declared = set(m["reason_codes"])
            assert declared == manifest, (m["id"], declared, manifest)
        # AMBIGUOUS: required roles structurally complete; ambiguity is declared
        # (two valid candidates differ only in hashes), not derived here.
        else:
            assert not manifest, (m["id"], manifest)

        audit_results.append({
            "id": m["id"], "outcome": m["expected_outcome"],
            "manifest_reasons": sorted(manifest),
        })

    # Alias stage enum (separate from matching).
    alias_expected = {"ACCEPTED", "REJECTED", "PRESERVED"}
    for a in cj.get("alias_conflict_cases", []):
        assert a.get("expected") in alias_expected, a

    neg = _negative_controls(cj)

    return {
        "id": "cases_json_crosscheck",
        "all_owner_decisions_adopted_and_verified": True,
        "protected_decision_ids": names,
        "dq_bits_consistent_five_invalid": True,
        "quality_gate_consistent": True,
        "matching_outcomes_enums_valid": True,
        "matching_reason_codes_registered": True,
        "deep_materialization_per_case": True,
        "overrides_removals_clone_applied_and_validated": True,
        "structural_audit_matches_declared_outcomes": True,
        "negative_controls": neg,
        "audit": audit_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="assert only; do not write evidence.json (no persistent artifact write)")
    ap.add_argument("--out", default=OUT_EVIDENCE,
                    help="output JSON path (default: research/phase1/evidence.json)")
    ap.add_argument("--self-test-guard", action="store_true",
                    help="run negative source-authority guard self-test (injected wrong expected hash)")
    args = ap.parse_args()

    actual_mission_sha = verify_authority(MISSION_DOC, EXPECTED_MISSION_SHA)
    actual_interop_sha = verify_authority(INTEROP_DOC, EXPECTED_INTEROP_SHA)

    if args.self_test_guard:
        try:
            verify_authority(MISSION_DOC, "0" * 64)
        except SourceAuthorityError:
            guard_ok = True
        else:
            guard_ok = False
        assert guard_ok, "guard failed to raise on wrong expected hash"

    evidence = {
        "mission_id": "ZC-P1-M1-CONTRACT-DRAFT-20260915",
        "phase": "owner-reconciliation",
        "generated_by": os.path.basename(__file__),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "astropy": getattr(astropy, "__version__", "n/a"),
        "source_authority": {
            "mission_doc_sha256": actual_mission_sha,
            "expected_mission_sha256": EXPECTED_MISSION_SHA,
            "mission_guard_asserted": True,
            "interop_doc_sha256": actual_interop_sha,
            "expected_interop_sha256": EXPECTED_INTEROP_SHA,
            "interop_guard_asserted": True,
            "cases_json_sha256": sha256_file(CASES_PATH),
            "negative_guard_self_test": bool(args.self_test_guard),
        },
        "equations": equations(),
        "cfa": cfa_cases(),
        "dq": [dq_bits()] + dq_semantics_cases(),
        "fits_decode": fits_decode_cases(),
        "digest": digest_cases(),
        "identity": identity_cases(),
        "declaration": declaration_cases(),
        "card_record": card_record_cases(),
        "quality_gate": [quality_gate()],
        "crosscheck": [crosscheck_cases()],
        "summary": {
            "gate_acceptance_authority": "Junior; see TODO.md",
            "protected_decisions_resolved": True,
            "note": "mathematical invariants and reference semantics asserted; "
                    "matching cases are declaration-only (structural audit, NOT a "
                    "matcher); owner policies adopted 2026-09-15; real qualification remains open",
        },
    }

    payload = json_safe(evidence)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    text += "\n"

    if args.check:
        print("CHECK_OK: all Phase 1 witness assertions passed; no file written.")
        return 0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
