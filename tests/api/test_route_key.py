"""C26 — property test for the canonical ``light_route_key``.

Guarantee (ratified): two lights with the SAME key route to the SAME ``plan_id``
(``resolve_light``), and two lights the matcher distinguishes (different civil
day, different temperature, different roi_origin, …) have DIFFERENT keys.

Uses synthetic masters + synthetic light variants (header-only facts).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1

SHAPE = (40, 32)


def _write(path, imagetyp, exptime, *, filter_=None, mean=100.0,
           temp=-10.0, set_temp=-10.0, date_obs=None, gain=120.0, offset=50.0,
           xor=0, yor=0):
    rng = np.random.default_rng(20260927)
    data = np.clip(rng.normal(mean, mean * 0.02, size=SHAPE), 0, 65535).astype(np.int16)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["IMAGETYP"] = imagetyp
    h["EXPTIME"] = float(exptime)
    h["INSTRUME"] = "Seestar S50"
    h["GAIN"] = float(gain)
    h["OFFSET"] = float(offset)
    h["CCD-TEMP"] = float(temp)
    h["SET-TEMP"] = float(set_temp)
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["XORGSUBF"] = int(xor)
    h["YORGSUBF"] = int(yor)
    h["BAYERPAT"] = "MONO"
    h["BUNIT"] = "ADU"
    if filter_ is not None:
        h["FILTER"] = filter_
    if date_obs is not None:
        h["DATE-OBS"] = date_obs
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def _master_folder(tmp_path):
    _write(tmp_path / "dark_60s.fits", "Dark", 60.0, mean=200.0)
    _write(tmp_path / "bias.fits", "Bias", 0.0, mean=100.0)
    _write(tmp_path / "flat.fits", "Flat", 5.0, mean=20000.0, filter_="L")
    return str(tmp_path)


def _constraints(path):
    from zecalibrator.api.v1._io import _decode_light_once
    from zecalibrator.application.library import light_constraints_from_sensor_metadata

    decoded = _decode_light_once(v1.FitsFrameSource(path=path), token=None)
    return light_constraints_from_sensor_metadata(decoded.frame.metadata)


def _resolve_plan_id(folder, light_path):
    res = v1.open_session_library(folder, declaration=v1.SessionDeclaration(orientation="identity"))
    rr = res.handle.resolve_light(v1.FitsFrameSource(path=light_path))
    return rr.plan.plan_id if rr.plan is not None else None


def test_same_facts_same_key_same_plan(tmp_path):
    folder = _master_folder(tmp_path)
    _write(tmp_path / "light_a.fits", "Light", 60.0, filter_="L", date_obs="2026-09-15T01:00:00")
    _write(tmp_path / "light_b.fits", "Light", 60.0, filter_="L", date_obs="2026-09-15T02:00:00")

    ka = v1.light_route_key(_constraints(str(tmp_path / "light_a.fits")))
    kb = v1.light_route_key(_constraints(str(tmp_path / "light_b.fits")))
    assert ka == kb  # same civil day + same facts -> same key

    pa = _resolve_plan_id(folder, str(tmp_path / "light_a.fits"))
    pb = _resolve_plan_id(folder, str(tmp_path / "light_b.fits"))
    assert pa is not None and pa == pb


def test_different_civil_day_different_key(tmp_path):
    _write(tmp_path / "light_a.fits", "Light", 60.0, filter_="L", date_obs="2026-09-14T13:00:00")
    _write(tmp_path / "light_b.fits", "Light", 60.0, filter_="L", date_obs="2026-09-15T01:00:00")
    ka = v1.light_route_key(_constraints(str(tmp_path / "light_a.fits")))
    kb = v1.light_route_key(_constraints(str(tmp_path / "light_b.fits")))
    assert ka != kb


def test_different_temperature_different_key(tmp_path):
    _write(tmp_path / "light_a.fits", "Light", 60.0, filter_="L", temp=-10.0)
    _write(tmp_path / "light_b.fits", "Light", 60.0, filter_="L", temp=-9.9)
    ka = v1.light_route_key(_constraints(str(tmp_path / "light_a.fits")))
    kb = v1.light_route_key(_constraints(str(tmp_path / "light_b.fits")))
    assert ka != kb  # CCD-TEMP -10.0 vs -9.9 -> distinct (no tolerance)


def test_different_roi_origin_different_key(tmp_path):
    _write(tmp_path / "light_a.fits", "Light", 60.0, filter_="L", xor=0, yor=0)
    _write(tmp_path / "light_b.fits", "Light", 60.0, filter_="L", xor=10, yor=0)
    ka = v1.light_route_key(_constraints(str(tmp_path / "light_a.fits")))
    kb = v1.light_route_key(_constraints(str(tmp_path / "light_b.fits")))
    assert ka != kb


def test_missing_facts_distinct_keys(tmp_path):
    # DATE-OBS absent vs present, FILTER absent vs present, temp absent vs present:
    # never grouped by default.
    _write(tmp_path / "no_date.fits", "Light", 60.0, filter_="L", date_obs=None)
    _write(tmp_path / "with_date.fits", "Light", 60.0, filter_="L", date_obs="2026-09-15T01:00:00")
    _write(tmp_path / "no_filter.fits", "Light", 60.0, filter_=None, date_obs="2026-09-15T01:00:00")
    keys = {
        v1.light_route_key(_constraints(str(tmp_path / n)))
        for n in ("no_date.fits", "with_date.fits", "no_filter.fits")
    }
    assert len(keys) == 3  # three distinct keys


def test_key_is_deterministic_string(tmp_path):
    _write(tmp_path / "light.fits", "Light", 60.0, filter_="L", date_obs="2026-09-15T01:00:00")
    k1 = v1.light_route_key(_constraints(str(tmp_path / "light.fits")))
    k2 = v1.light_route_key(_constraints(str(tmp_path / "light.fits")))
    assert isinstance(k1, str)
    assert k1 == k2
