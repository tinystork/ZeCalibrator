"""Deterministic synthetic-corpus generator (LOT1).

Turns a :class:`research.p3b.model.ScenarioSpec` into a materialised corpus in a
temporary/output directory:

* ``frames/<frame_id>.fits`` — raw science + calibration frames, written in the
  observed FITS convention (``BITPIX=16, BSCALE=1, BZERO=32768``), applied
  exactly once, native unbinned base-0 coordinates, ``array[y, x]`` ↔ sensor
  ``(x, y)``, explicit CFA phase/origin, no debayer/normalization/flip/rotate/
  crop/resample;
* ``manifest.json`` — the machine-readable ground-truth manifest. The pipeline
  under test must never see it (only the FITS frames).

Determinism: a single ``numpy.random.default_rng(seed)`` drives all randomness;
frames are generated in a fixed order; the FITS writer and JSON serialisation
are byte-stable. Same seed ⇒ byte-identical FITS + manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
from astropy.io import fits

from .behaviors import evaluate_site
from .catalog import CLASS_NAMES, resolve_class
from .model import (
    AggregateSpec,
    Bookkeeping,
    FrameSpec,
    ScenarioSpec,
    SensorSpec,
    SiteSpec,
    compute_bookkeeping,
)
from .parameters import (
    DARK_CURRENT_ADU_PER_S,
    FITS_BSCALE,
    FITS_BZERO,
    FLAT_PEDESTAL_ADU,
    QUICKLOOK_PERCENTILE_HIGH,
    QUICKLOOK_PERCENTILE_LOW,
    READ_NOISE_ADU,
    SKY_BACKGROUND_ADU_PER_S,
    get_param,
)

# CFA parity labels mirrored from zecalibrator.core.geometry (SCIENCE §7.5):
# parity index (y%2)*2 + (x%2) -> plane label.
_PHASE_LABELS: Mapping[str, Tuple[str, str, str, str]] = {
    "GRBG": ("G1", "R", "B", "G2"),
    "RGGB": ("R", "G1", "G2", "B"),
    "BGGR": ("B", "G1", "G2", "R"),
    "GBRG": ("G1", "B", "R", "G2"),
}


def cfa_plane_label(y: int, x: int, pattern: str) -> str:
    """Return the CFA plane label at *sensor/array* ``(y, x)``.

    Mirrors ``zecalibrator.core.geometry.sensor_plane`` for the four Bayer
    phases; ``"mono"`` returns ``"mono"``.
    """
    if pattern == "mono":
        return "mono"
    labels = _PHASE_LABELS[pattern]
    return labels[(y % 2) * 2 + (x % 2)]


def _add_patch(frame: np.ndarray, patch) -> None:
    """Add a SitePatch into ``frame`` (bounds-clipped)."""
    h, w = patch.values.shape
    y0 = patch.cy - h // 2
    x0 = patch.cx - w // 2
    ny, nx = frame.shape
    # Clip to the valid overlap.
    fy0 = max(0, y0)
    fx0 = max(0, x0)
    fy1 = min(ny, y0 + h)
    fx1 = min(nx, x0 + w)
    if fy1 <= fy0 or fx1 <= fx0:
        return
    frame[fy0:fy1, fx0:fx1] += patch.values[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0]


def _base_signal(sensor: SensorSpec, frame: FrameSpec, rng: np.random.Generator) -> np.ndarray:
    ny, nx = sensor.shape
    base = np.full((ny, nx), sensor.offset_adu, dtype=np.float64)
    if frame.frame_type == "dark":
        from .behaviors import _temp_factor  # local to avoid a top-level cycle

        dark = DARK_CURRENT_ADU_PER_S * frame.exposure_s * _temp_factor(frame.temperature_c)
        base += dark
    elif frame.frame_type == "light":
        base += SKY_BACKGROUND_ADU_PER_S * frame.exposure_s
    elif frame.frame_type == "flat":
        base += FLAT_PEDESTAL_ADU
    # read noise (deterministic via the seeded rng)
    base += rng.normal(0.0, READ_NOISE_ADU, size=(ny, nx))
    return base


def _quantize(arr: np.ndarray, sensor: SensorSpec) -> np.ndarray:
    return np.clip(np.rint(arr), 0.0, 65535.0).astype(np.uint16)


def _render_frame(
    scenario: ScenarioSpec,
    frame: FrameSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    sensor = scenario.sensor
    arr = _base_signal(sensor, frame, rng)
    for site in scenario.sites:
        patch = evaluate_site(site, frame.frame_type, frame.ordinal, rng, sensor)
        _add_patch(arr, patch)
    # Apply the declared hard limit (censoring): never let a censored site's
    # value exceed its hard limit.
    for site in scenario.sites:
        if site.censored:
            limit = site.hard_limit_adu if site.hard_limit_adu is not None else sensor.saturation_limit_adu
            # Only the site's own pixel is clipped to the hard limit; the rest
            # of the frame is untouched.
            if 0 <= site.y < sensor.shape[0] and 0 <= site.x < sensor.shape[1]:
                if arr[site.y, site.x] > limit:
                    arr[site.y, site.x] = limit
    return _quantize(arr, sensor)


def _write_fits(path: Path, stored_uint16: np.ndarray, sensor: SensorSpec, frame: FrameSpec) -> None:
    # FITS storage convention: physical = BSCALE * stored + BZERO (exactly once).
    # We store uint16 and encode as signed int16 with BZERO=32768 (observed on
    # the real corpus). The decoder applies the linear decode exactly once.
    stored = (stored_uint16.astype(np.int64) - int(FITS_BZERO)).astype(np.int16)
    hdu = fits.PrimaryHDU(stored)
    hdu.header["BSCALE"] = FITS_BSCALE
    hdu.header["BZERO"] = FITS_BZERO
    hdu.header["BUNIT"] = "ADU"
    hdu.header["EXPTIME"] = float(frame.exposure_s)
    hdu.header["CCD-TEMP"] = float(frame.temperature_c)
    hdu.header["GAIN"] = float(sensor.gain)
    hdu.header["INSTRUME"] = sensor.model
    hdu.header["FILTER"] = sensor.filter
    hdu.header["XBINNING"] = int(sensor.binning[0])
    hdu.header["YBINNING"] = int(sensor.binning[1])
    hdu.header["XORGSUBF"] = int(sensor.roi_origin[0])
    hdu.header["YORGSUBF"] = int(sensor.roi_origin[1])
    if sensor.cfa_pattern != "mono":
        hdu.header["BAYERPAT"] = sensor.cfa_pattern
    hdu.header["IMAGETYP"] = {
        "light": "Light Frame",
        "dark": "Dark Frame",
        "bias": "Bias Frame",
        "flat": "Flat Frame",
        "master": "Master Frame",
    }[frame.frame_type]
    hdu.writeto(path, overwrite=True)


def _frame_filename(frame_id: str) -> str:
    # Structural ID only — never encodes a class or an expected state.
    return f"{frame_id}.fits"


def _aggregate_content(
    agg: AggregateSpec,
    physical: Mapping[str, np.ndarray],
) -> np.ndarray:
    stacks = np.stack([physical[fid] for fid in agg.constituent_frame_ids], axis=0).astype(np.float64)
    if agg.method == "median":
        out = np.median(stacks, axis=0)
    else:
        out = np.mean(stacks, axis=0)
    return np.rint(out).astype(np.uint16)


@dataclass(frozen=True)
class CorpusResult:
    """The materialised corpus + ground truth."""

    scenario_name: str
    seed: int
    out_dir: str
    manifest: Mapping
    manifest_path: str
    frame_paths: Mapping[str, str]  # frame_id -> absolute FITS path
    frame_arrays: Mapping[str, np.ndarray]  # frame_id -> physical uint16 array
    bookkeeping: Bookkeeping

    def to_manifest_dict(self) -> dict:
        return dict(self.manifest)


def generate_corpus(scenario: ScenarioSpec, out_dir) -> CorpusResult:
    """Materialise ``scenario`` into ``out_dir`` and return the corpus result.

    Writes ``out_dir/frames/*.fits`` and ``out_dir/manifest.json``. Fully
    deterministic given ``scenario.seed``.
    """
    out_path = Path(out_dir)
    frames_dir = out_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(scenario.seed)

    physical: Dict[str, np.ndarray] = {}
    frame_paths: Dict[str, str] = {}
    frame_records: list = []

    # Deterministic iteration order: epochs -> groups -> frames.
    for epoch in scenario.epochs:
        for group in epoch.groups:
            for frame in group.frames:
                arr = _render_frame(scenario, frame, rng)
                path = frames_dir / _frame_filename(frame.frame_id)
                _write_fits(path, arr, scenario.sensor, frame)
                physical[frame.frame_id] = arr
                frame_paths[frame.frame_id] = str(path)
                frame_records.append(
                    {
                        "frame_id": frame.frame_id,
                        "frame_type": frame.frame_type,
                        "epoch_id": epoch.epoch_id,
                        "group_id": group.group_id,
                        "ordinal": frame.ordinal,
                        "exposure_s": frame.exposure_s,
                        "temperature_c": frame.temperature_c,
                        "path": f"frames/{_frame_filename(frame.frame_id)}",
                    }
                )

    # Derived aggregates (masters): frames, but not new observations/groups.
    aggregate_records: list = []
    for agg in scenario.aggregates:
        arr = _aggregate_content(agg, physical)
        path = frames_dir / _frame_filename(agg.aggregate_id)
        agg_frame = FrameSpec(
            frame_id=agg.aggregate_id,
            frame_type="master",
            epoch_id="",
            group_id="",
            ordinal=0,
        )
        _write_fits(path, arr, scenario.sensor, agg_frame)
        physical[agg.aggregate_id] = arr
        frame_paths[agg.aggregate_id] = str(path)
        aggregate_records.append(
            {
                "aggregate_id": agg.aggregate_id,
                "method": agg.method,
                "constituent_frame_ids": list(agg.constituent_frame_ids),
                "path": f"frames/{_frame_filename(agg.aggregate_id)}",
            }
        )

    bookkeeping = compute_bookkeeping(scenario)

    sensor_dict = {
        "instance_id": scenario.sensor.instance_id,
        "model": scenario.sensor.model,
        "shape": list(scenario.sensor.shape),
        "cfa_pattern": scenario.sensor.cfa_pattern,
        "binning": list(scenario.sensor.binning),
        "roi_origin": list(scenario.sensor.roi_origin),
        "gain": scenario.sensor.gain,
        "offset_adu": scenario.sensor.offset_adu,
        "saturation_limit_adu": scenario.sensor.saturation_limit_adu,
        "filter": scenario.sensor.filter,
    }

    sites = []
    for site in scenario.sites:
        sites.append(
            {
                "site_id": site.site_id,
                "x": site.x,
                "y": site.y,
                "cfa_class": site.cfa_class,
                "cfa_plane": cfa_plane_label(site.y, site.x, scenario.sensor.cfa_pattern),
                "expected_qualification_state": site.expected_qualification_state,
                "expected_action_state": site.expected_action_state,
                "calibration_representativeness": site.calibration_representativeness,
                "censored": site.censored,
                "hard_limit_adu": site.hard_limit_adu,
                "params": [list(kv) for kv in site.params],
            }
        )

    manifest = {
        "manifest_schema": "zecalibrator-p3b-lot1-truth",
        "manifest_version": 1,
        "scenario": {
            "name": scenario.name,
            "seed": scenario.seed,
            "sensor": sensor_dict,
            "classes_declared": list(CLASS_NAMES),
        },
        "bookkeeping": bookkeeping.to_dict(),
        "frames": frame_records,
        "aggregates": aggregate_records,
        "sites": sites,
    }

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    return CorpusResult(
        scenario_name=scenario.name,
        seed=scenario.seed,
        out_dir=str(out_path),
        manifest=manifest,
        manifest_path=str(manifest_path),
        frame_paths=frame_paths,
        frame_arrays=physical,
        bookkeeping=bookkeeping,
    )


def quicklook_stats(frame_arrays: Mapping[str, np.ndarray]) -> Mapping[str, float]:
    """Exploratory (not product) summary stats for visualising a corpus.

    Uses EXPLORATORY-marked constants only; never used by generation.
    """
    lo = QUICKLOOK_PERCENTILE_LOW
    hi = QUICKLOOK_PERCENTILE_HIGH
    out = {}
    for fid, arr in frame_arrays.items():
        a = arr.astype(np.float64)
        out[fid] = {
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "pct_low": float(np.percentile(a, lo)),
            "pct_high": float(np.percentile(a, hi)),
        }
    return out


__all__ = [
    "CorpusResult",
    "cfa_plane_label",
    "generate_corpus",
    "quicklook_stats",
]
