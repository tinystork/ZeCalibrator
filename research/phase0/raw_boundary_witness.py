#!/usr/bin/env python3
"""
raw_boundary_witness.py — ZeCalibrator Phase 0 / Mission 1 research witness.

STANDALONE research script, NOT an installed module and NOT an end-to-end
application test. It does not import the ZSSS application. It extracts the
*actual* function bodies of the ZSSS loader/sanitizer from the pinned source
file via AST and executes those bodies with injected dependencies (real NumPy,
real Astropy). This isolates the real arithmetic of the current loader while
avoiding any application import side effects.

It also:

  * re-reads the S50 / S30 acquisition-origin FITS and the synthetic
    Light_001 fixture, exporting ONLY the allowlisted matching-relevant header
    fields plus path/content identity, source qualification and missing fields;
  * inventories the supplied icons/ assets (SHA-256, bytes, container signature);
  * independently computes the ASTRA architecture document §4.3 equation
    witness and its negative controls;
  * writes research/phase0/evidence.json.

Run with the ZSSS venv interpreter (Python 3.13 + NumPy + Astropy present):

    /home/tristan/.openclaw/workspace/projects/zeseestarstacker/.venv/bin/python \
        research/phase0/raw_boundary_witness.py

It performs no writes outside research/phase0/evidence.json and its own
temporary directory. Real FITS files are opened read-only.
"""

from __future__ import annotations

import ast
import contextlib
import datetime as _dt
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import warnings
import traceback

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# Pinned source authority
# ---------------------------------------------------------------------------
ZSSS_ROOT = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
LOADER_PATH = os.path.join(ZSSS_ROOT, "seestar", "core", "image_processing.py")
ZSSS_HEAD = "77c92da0259ef7a41fa37d7aa6f3fab7a840735c"

MISSION_DOC = "/home/tristan/.openclaw/workspace/projects/zecalibrator/ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md"
ICON_DIR = "/home/tristan/.openclaw/workspace/projects/zecalibrator/icons"

S50_FITS = "/home/tristan/near_bench100_input/049_Light_mosaic_M 106_20.0s_IRCUT_20250518-233417.fit"
S30_FITS = "/home/tristan/near_bench100_input/058_Light_mosaic_M 31_10.0s_IRCUT_20250804-004954.fit"
SYNTH_FITS = os.path.join(ZSSS_ROOT, "fixture", "master", "Light_001.fit")

ALLOWED_FIELDS = [
    "INSTRUME", "CREATOR", "NAXIS1", "NAXIS2",
    "XBINNING", "YBINNING", "CCDXBIN", "CCDYBIN",
    "XORGSUBF", "YORGSUBF", "EXPTIME", "EXPOSURE",
    "CCD-TEMP", "GAIN", "FILTER", "BAYERPAT",
    "BITPIX", "BSCALE", "BZERO",
]

OUT_EVIDENCE = "/home/tristan/.openclaw/workspace/projects/zecalibrator/research/phase0/evidence.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 1. AST extraction of the ACTUAL ZSSS loader/sanitizer bodies
# ---------------------------------------------------------------------------
class _FakeCV2:
    """Stub only to satisfy the name binding of debayer_image; never called."""

    COLOR_BayerGR2RGB = 0
    COLOR_BayerRG2RGB = 0
    COLOR_BayerGB2RGB = 0
    COLOR_BayerBG2RGB = 0
    COLOR_BGR2RGB = 0

    class error(Exception):
        pass

    @staticmethod
    def cvtColor(*a, **k):
        raise NotImplementedError("cv2 stub: debayer_image must not run here")


def extract_functions(source_path: str, names):
    src = open(source_path, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            segments.append(ast.get_source_segment(src, node))
    if len(segments) != len(names):
        raise RuntimeError(
            f"Expected to extract {sorted(names)} but got {len(segments)} segments"
        )
    return "\n\n".join(segments)


def make_loader_namespace():
    ns = {
        "os": os,
        "np": np,
        "fits": fits,
        "warnings": warnings,
        "traceback": traceback,
        "cv2": _FakeCV2,
        "Image": None,
        "__name__": "extracted_image_processing",
    }
    code_src = extract_functions(
        LOADER_PATH,
        {"load_and_validate_fits", "sanitize_header_for_wcs", "debayer_image"},
    )
    exec(compile(code_src, "<extracted:image_processing.py>", "exec"), ns)
    return ns, code_src


def run_loader(load_fn, filepath, normalize=True, fix_nonfinite=True):
    """Call the ACTUAL extracted loader body, suppressing its debug prints.

    Returns (image_data, debug_log). The loader returns (image, header) or
    None; we surface only the image data (and header presence) here.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = load_fn(
            filepath,
            normalize_to_float32=normalize,
            attempt_fix_nonfinite=fix_nonfinite,
        )
    if result is None:
        return None, buf.getvalue()
    image_data = result[0] if isinstance(result, tuple) else result
    return image_data, buf.getvalue()


# ---------------------------------------------------------------------------
# 2. Loader witnesses (reproduce §3.2)
# ---------------------------------------------------------------------------
def loader_witnesses(ns):
    load_fn = ns["load_and_validate_fits"]
    out = []

    tmpdir = tempfile.mkdtemp(prefix="zcal_phase0_")

    # Witness A: uint16 physical data stored as int16 + BZERO=32768
    u16 = np.array([[0, 100], [200, 65535]], dtype=np.uint16)
    path_a = os.path.join(tmpdir, "witness_uint16.fits")
    fits.writeto(path_a, u16, overwrite=True)

    with fits.open(path_a, do_not_scale_image_data=True) as h:
        stored_a = np.array(h[0].data)
        hdr_a = h[0].header.copy()
    with fits.open(path_a) as h:  # default physical scaling
        physical_a = np.array(h[0].data)

    default_res, default_log = run_loader(load_fn, path_a, normalize=True, fix_nonfinite=True)
    raw_res, raw_log = run_loader(load_fn, path_a, normalize=False, fix_nonfinite=False)

    out.append({
        "id": "uint16_bzero_loader_witness",
        "input_physical": u16.tolist(),
        "header_bzero": int(hdr_a.get("BZERO")),
        "astropy_physical_read": physical_a.tolist(),
        "stored_dtype": str(stored_a.dtype),
        "zsss_default_loader": default_res.tolist(),
        "zsss_default_loader_dtype": str(default_res.dtype),
        "zsss_no_normalize_no_fix": raw_res.tolist(),
        "zsss_no_normalize_no_fix_dtype": str(raw_res.dtype),
        # loader performs min/max normalization in float32 after int16 storage
        # read; expected float32 values match §3.2 exactly.
        "expected_default_float32": (
            np.array([[0, 100], [200, 65535]], dtype=np.float32)
            / np.float32(65535.0)
        ).tolist(),
        "expected_no_normalize": [[-32768.0, -32668.0], [-32568.0, 32767.0]],
    })

    # Witness B: signed float + NaN destruction
    signed = np.array([[-2.0, 0.0], [2.0, np.nan]], dtype=np.float32)
    path_b = os.path.join(tmpdir, "witness_signed_nan.fits")
    fits.writeto(path_b, signed, overwrite=True)

    signed_res, signed_log = run_loader(load_fn, path_b, normalize=True, fix_nonfinite=True)
    out.append({
        "id": "signed_nan_loader_witness",
        "input": signed.tolist(),
        "zsss_default_loader": signed_res.tolist(),
        "expected": [[0.0, 0.5], [1.0, 0.5]],
    })

    # cleanup temp files
    for p in (path_a, path_b):
        with contextlib.suppress(OSError):
            os.remove(p)
    with contextlib.suppress(OSError):
        os.rmdir(tmpdir)

    return out


# ---------------------------------------------------------------------------
# 3. §4.3 equation witnesses
# ---------------------------------------------------------------------------
def equation_witnesses():
    out = []

    L = np.array([[110.0, 50.0], [210.0, -2.0]], dtype=np.float64)
    Dinc = np.full((2, 2), 10.0, dtype=np.float64)
    B = np.full((2, 2), 4.0, dtype=np.float64)
    D0 = np.full((2, 2), 6.0, dtype=np.float64)
    R = np.array([[1.0, 0.5], [2.0, 1.0]], dtype=np.float64)
    expected_C = np.array([[100.0, 80.0], [100.0, -12.0]], dtype=np.float64)

    A_dark_inc = L - Dinc
    C_dark_inc = A_dark_inc / R
    A_bias_removed = L - B - D0
    C_bias_removed = A_bias_removed / R
    # Negative control: subtract bias on top of a dark-that-includes-bias.
    C_double = (L - Dinc - B) / R

    assert np.array_equal(C_dark_inc, expected_C), C_dark_inc
    assert np.array_equal(C_bias_removed, expected_C), C_bias_removed
    assert not np.array_equal(C_double, expected_C), "double subtraction must change C"

    out.append({
        "id": "equation_main",
        "L": L.tolist(), "Dinc": Dinc.tolist(), "B": B.tolist(),
        "D0": D0.tolist(), "R": R.tolist(),
        "C_dark_inc": C_dark_inc.tolist(),
        "C_bias_removed": C_bias_removed.tolist(),
        "C_expected": expected_C.tolist(),
        "coherent_branches_equal": bool(np.array_equal(C_dark_inc, C_bias_removed)),
    })

    out.append({
        "id": "double_bias_subtraction_negative_control",
        "C_double_subtraction": C_double.tolist(),
        "differs_from_expected": bool(not np.array_equal(C_double, expected_C)),
        "note": "subtracting B in addition to Dinc (which already includes bias) changes the answer",
    })

    # Flat-pedestal witness: normalizing an uncorrected flat != correcting it.
    Finc = np.array([[100.0, 110.0], [120.0, 90.0]], dtype=np.float64)
    FDinc = np.full((2, 2), 10.0, dtype=np.float64)  # flat-dark incl. bias
    Fcorr = Finc - FDinc
    s_corr = float(np.median(Fcorr[np.isfinite(Fcorr)]))
    R_correct = Fcorr / s_corr
    s_wrong = float(np.median(Finc[np.isfinite(Finc)]))
    R_wrong = Finc / s_wrong

    A_light = np.array([[95.0, 105.0], [115.0, 85.0]], dtype=np.float64)
    C_correct = A_light / R_correct
    C_wrong = A_light / R_wrong

    out.append({
        "id": "flat_pedestal_witness",
        "Finc": Finc.tolist(), "FDinc": FDinc.tolist(),
        "Fcorr": Fcorr.tolist(),
        "s_correct": s_corr, "s_wrong": s_wrong,
        "R_correct": R_correct.tolist(),
        "R_wrong": R_wrong.tolist(),
        "C_with_corrected_flat": C_correct.tolist(),
        "C_with_uncorrected_flat": C_wrong.tolist(),
        "normalizing_uncorrected_flat_is_not_correcting_it": bool(
            not np.array_equal(R_correct, R_wrong)
        ),
    })

    # Four-CFA-plane scale witness (GRBG; separate G1/G2 planes).
    h = w = 4
    F = np.zeros((h, w), dtype=np.float64)
    parity = {}
    for y in range(h):
        for x in range(w):
            if y % 2 == 0 and x % 2 == 1:
                F[y, x] = 100.0   # R  (even, odd)
                parity[(y, x)] = "R"
            elif y % 2 == 0 and x % 2 == 0:
                F[y, x] = 200.0   # G1 (even, even)
                parity[(y, x)] = "G1"
            elif y % 2 == 1 and x % 2 == 1:
                F[y, x] = 100.0   # G2 (odd, odd)
                parity[(y, x)] = "G2"
            else:
                F[y, x] = 100.0   # B  (odd, even)
                parity[(y, x)] = "B"

    plane_names = ["R", "G1", "G2", "B"]
    s_plane = {}
    for p in plane_names:
        vals = [F[y, x] for (y, x), q in parity.items() if q == p]
        s_plane[p] = float(np.median(np.array(vals, dtype=np.float64)))

    R_per_plane = np.zeros_like(F)
    for (y, x), q in parity.items():
        R_per_plane[y, x] = F[y, x] / s_plane[q]

    s_global = float(np.median(F))
    R_global = F / s_global

    out.append({
        "id": "four_cfa_plane_scale_witness",
        "pattern": "GRBG",
        "F_flat": F.tolist(),
        "parity": {f"{y},{x}": q for (y, x), q in sorted(parity.items())},
        "per_plane_medians": s_plane,
        "global_single_median": s_global,
        "R_per_plane": R_per_plane.tolist(),
        "R_global_single_median": R_global.tolist(),
        "G1_plane_differs_under_global_median": bool(
            not np.array_equal(R_per_plane, R_global)
        ),
        "note": "G1 plane has scale 200 vs G2/R/B at 100; a single global median "
                "over-corrects G1 (2.0 vs 1.0), so four separate parity planes "
                "(including separate G1/G2) are required",
    })

    # Same-shape / different-ROI design rejection case (contract observation,
    # not a matcher implementation).
    out.append({
        "id": "same_shape_different_roi_design_rejection",
        "design_case": {
            "light_A": {"NAXIS1": 1080, "NAXIS2": 1920, "XORGSUBF": 0, "YORGSUBF": 0},
            "light_B": {"NAXIS1": 1080, "NAXIS2": 1920, "XORGSUBF": 1, "YORGSUBF": 0},
        },
        "observation": "Both frames have identical NAXIS shape (1080 x 1920) but "
                       "different ROI origin. A one-pixel x-origin shift flips the "
                       "Bayer CFA parity for every row (even<->odd column), changing "
                       "which plane is G1 vs G2 and R vs B. Same shape does NOT imply "
                       "same ROI, and translation by even one pixel changes CFA phase.",
        "required_rejection": "Detector instance/model, sensor dimensions, ROI origin/"
                              "extent, binning and CFA phase are all exact-match "
                              "criteria. A matcher MUST reject on ROI-origin mismatch "
                              "even when NAXIS shape is identical.",
        "is_code": False,
    })

    return out


# ---------------------------------------------------------------------------
# 4. Icon inventory
# ---------------------------------------------------------------------------
def icon_inventory():
    out = []
    for name in sorted(os.listdir(ICON_DIR)):
        path = os.path.join(ICON_DIR, name)
        if not os.path.isfile(path):
            continue
        sig = ""
        with contextlib.suppress(Exception):
            sig = subprocess.run(
                ["file", "-b", path], capture_output=True, text=True, check=True
            ).stdout.strip()
        # derive dimensions from `file` text for a stable numeric value where possible
        dims = None
        if "PNG" in sig:
            import re
            m = re.search(r"(\d+) x (\d+)", sig)
            if m:
                dims = [int(m.group(1)), int(m.group(2))]
        out.append({
            "name": name,
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
            "signature": sig,
            "dimensions": dims,
        })
    return out


# ---------------------------------------------------------------------------
# 5. FITS witness header extraction (allowlisted fields only)
# ---------------------------------------------------------------------------
def fits_witness(path: str, qualification: str):
    data = {
        "path": path,
        "source_qualification": qualification,
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
    }
    try:
        with fits.open(path, do_not_scale_image_data=True) as hdul:
            data["num_hdu"] = len(hdul)
            hdr = hdul[0].header
            fields = {}
            missing = []
            for k in ALLOWED_FIELDS:
                if k in hdr:
                    v = hdr[k]
                    # JSON-safe conversion
                    if hasattr(v, "item"):
                        v = v.item()
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", "replace")
                    fields[k] = v
                else:
                    missing.append(k)
            data["fields"] = fields
            data["missing"] = missing
            d = hdul[0].data
            data["stored_dtype"] = str(d.dtype)
            data["shape"] = list(d.shape)
    except Exception as e:  # noqa: BLE001 — record, never invent
        data["error"] = repr(e)
        data["blocked"] = True
    return data


# ---------------------------------------------------------------------------
# 6. Ecosystem baseline (read-only git inspection)
# ---------------------------------------------------------------------------
def git_state(repo_path: str):
    def run(args):
        try:
            return subprocess.run(
                ["git", "-C", repo_path] + args,
                capture_output=True, text=True, check=False,
            ).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"<error: {e!r}>"

    return {
        "path": repo_path,
        "toplevel": run(["rev-parse", "--show-toplevel"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": run(["rev-parse", "HEAD"]),
        "status_short": run(["status", "--short"]),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    ns, loader_code = make_loader_namespace()

    evidence = {
        "mission_id": "ZC-P0-M1-RAW-BOUNDARY-20260915",
        "phase": "implementation",
        "generated_by": os.path.basename(__file__),
        "generated_at": generated_at,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "astropy": fits.__version__ if hasattr(fits, "__version__") else "n/a",
        "source_authority": {
            "zsss_loader_source": LOADER_PATH,
            "zsss_loader_sha256": sha256_file(LOADER_PATH),
            "zsss_head_expected": ZSSS_HEAD,
            "mission_doc_sha256": sha256_file(MISSION_DOC),
            "loader_extraction_note": (
                "AST-extracted actual bodies of load_and_validate_fits, "
                "sanitize_header_for_wcs, debayer_image from the pinned source; "
                "executed with injected real numpy + astropy.io.fits + a cv2 stub. "
                "NOT an end-to-end application test."
            ),
            "loader_extracted_source_sha256": sha256_bytes(loader_code.encode("utf-8")),
        },
        "ecosystem_baseline": {
            "ZeAlfie": git_state("/home/tristan/.openclaw/workspace/projects/ZeAlfie"),
            "zeseestarstacker": git_state(ZSSS_ROOT),
            "zemosaic": git_state("/home/tristan/.openclaw/workspace/projects/zemosaic"),
            "ZeSolver-main": git_state("/home/tristan/.openclaw/workspace/projects/ZeSolver-main"),
            "ZeSolver": git_state("/home/tristan/.openclaw/workspace/projects/ZeSolver"),
            "zeanalyser": git_state("/home/tristan/.openclaw/workspace/projects/zeanalyser"),
            "zecalibrator": {
                "independent_git_repo": False,
                "git_toplevel": subprocess.run(
                    ["git", "-C", "/home/tristan/.openclaw/workspace/projects/zecalibrator",
                     "rev-parse", "--show-toplevel"],
                    capture_output=True, text=True, check=False,
                ).stdout.strip(),
                "note": "zecalibrator resolves to the enclosing workspace repo, "
                        "which itself has no committed HEAD (uncommitted repo).",
            },
        },
        "icons": icon_inventory(),
        "fits_witnesses": [
            fits_witness(S50_FITS, "real acquisition-origin light: Seestar S50"),
            fits_witness(S30_FITS, "real acquisition-origin light: Seestar S30"),
            fits_witness(SYNTH_FITS, "synthetic LIGHT fixture (INSTRUME=SYNTH); "
                                    "NOT a calibration master despite 'master' dir name"),
        ],
        "loader_witnesses": loader_witnesses(ns),
        "equation_witnesses": equation_witnesses(),
    }

    os.makedirs(os.path.dirname(OUT_EVIDENCE), exist_ok=True)
    with open(OUT_EVIDENCE, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
