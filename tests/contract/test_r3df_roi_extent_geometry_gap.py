"""R3D-F — real-witness geometry-gap regression tests (P7-M3B).

The real auto-route blocked on ``MISSING_REQUIRED_FIELD geometry.roi_extent``
because the decode path hard-coded ``roi_extent=None`` while every managed
master descriptor records ``roi_extent=shape``. The fix is METADATA
REPRESENTATION (not matcher policy): a decoded 2-D FITS plane's stored extent
*is* its shape, so ``build_sensor_metadata`` now records ``roi_extent=shape``
(structural). ``roi_origin`` / ``sensor_dimensions`` / ``orientation`` remain
``None``-when-unknown (never invented).

These tests assert:

1. A Bayer light whose decoded ``roi_extent`` becomes ``shape`` + a master
   descriptor with ``roi_extent=shape`` resolves a route with **no**
   ``MISSING_REQUIRED_FIELD geometry.roi_extent``.
2. ``build_sensor_metadata`` records ``roi_extent == shape`` and leaves
   ``roi_origin`` / ``sensor_dimensions`` / ``orientation`` ``None`` when
   un-evidenced.
3. A mono frame (``cfa_phase="mono"``) is NOT gated by the Bayer ``roi_extent``
   rule (still routes even with an absent ``roi_extent`` on the matcher side).
"""

from __future__ import annotations

import pytest

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    detector,
    geo,
    policy,
)
from zecalibrator.application.library import (
    LibrarySnapshot,
    light_constraints_from_sensor_metadata,
)
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.metadata import ImportDeclaration, build_sensor_metadata
from zecalibrator.core.routes import OUTCOME_READY


def _decoded_bayer_light():
    """Decode a Bayer (RGGB) light via ``build_sensor_metadata`` (the decode path).

    No roi_origin / sensor_dimensions / orientation evidence is supplied — they
    must remain ``None``. The decoded ``roi_extent`` must equal ``shape``.
    """
    md = build_sensor_metadata(
        shape=(4, 4),
        normalized={
            "cfa": "RGGB",
            "bin_x": 1.0,
            "bin_y": 1.0,
            "gain": 100.0,
            "offset": 50.0,
            "exposure_seconds": 10.0,
            "temperature_c": 20.0,
            "detector_model": "SYNTH-CFA",
        },
        conflicts=(),
        cards=(),
        declaration=ImportDeclaration(
            source="standard_light_contract",
            identity="standard-light",
            version="1",
            domain="raw",
            units="ADU",
        ),
        units="ADU",
    )
    return md


def _bayer_master(shape=(4, 4)):
    """A managed-master descriptor with ``roi_extent=shape`` (Bayer RGGB).

    Mirrors the real managed master (``_descriptor_from_spec``): the
    disambiguator/declaration-only facts (readout_mode/adc_mode/sensor_dimensions/
    detector_instance_id) stay unknown so they do not inject spurious blocking
    reasons unrelated to ``roi_extent``.
    """
    return descriptor(
        "dark",
        "included",
        exposure_s=10.0,
        geometry=geo(
            shape=shape,
            cfa_phase="RGGB",
            roi_extent=shape,
            roi_origin=(0, 0),
            orientation="identity",
            sensor_dimensions=None,
            binning=(1, 1),
        ),
        detector_obj=detector(instance="unknown", model="SYNTH-CFA"),
        acquisition_obj=acquisition(
            gain=100.0, offset=None, readout_mode=None, adc_mode=None,
            temperature_c=20.0, exposure_s=10.0,
        ),
        optical_train_id=None,
        filter=None,
    )


# ---------------------------------------------------------------------------
# 1. Bayer light + roi_extent=shape master resolves a route (no roi_extent gate)
# ---------------------------------------------------------------------------
def test_bayer_decoded_roi_extent_shape_resolves_route():
    md = _decoded_bayer_light()
    assert md.geometry.roi_extent == (4, 4)  # structural: extent == shape

    light = light_constraints_from_sensor_metadata(md)
    snap = LibrarySnapshot(
        revision="r",
        schema_version="zecalibrator.library.v1",
        candidates={"dark": (candidate("d1", _bayer_master()),)},
    )
    res = resolve_route(light, snap, policy())

    assert res.outcome == OUTCOME_READY
    assert res.route is not None
    assert res.route.additive_mode == "dark_incl_bias"
    # roi_extent must NOT be a blocking reason.
    assert not any(
        r.code == "MISSING_REQUIRED_FIELD" and r.field == "geometry.roi_extent"
        for r in res.reasons
    )


# ---------------------------------------------------------------------------
# 2. build_sensor_metadata structural roi_extent + None-unknown neighbors
# ---------------------------------------------------------------------------
def test_build_sensor_metadata_roi_extent_equals_shape_and_neighbors_unknown():
    md = _decoded_bayer_light()
    g = md.geometry
    assert g.roi_extent == (4, 4) == g.shape
    # Never invented: the CFA-conditional / full-frame facts stay unknown.
    assert g.roi_origin is None
    assert g.sensor_dimensions is None
    assert g.orientation is None
    assert g.cfa_phase == "RGGB"
    assert g.binning == (1, 1)


def test_build_sensor_metadata_roi_extent_still_shape_when_no_cfa():
    # Even without a CFA phase, the structural extent equals the decoded shape.
    md = build_sensor_metadata(
        shape=(10, 7),
        normalized={"bin_x": 1.0, "bin_y": 1.0},
        conflicts=(),
        cards=(),
        declaration=ImportDeclaration(
            source="standard_light_contract", identity="standard-light",
            version="1", domain="raw", units="ADU",
        ),
        units="ADU",
    )
    assert md.geometry.roi_extent == (10, 7)
    assert md.geometry.roi_origin is None
    assert md.geometry.sensor_dimensions is None
    assert md.geometry.orientation is None


# ---------------------------------------------------------------------------
# 3. mono frame is NOT affected by the Bayer roi_extent gate
# ---------------------------------------------------------------------------
def test_mono_frame_unaffected_by_bayer_roi_extent_gate():
    # A mono master with an absent roi_extent still resolves (disambiguator, not
    # a CFA-conditional necessary fact). The decode path now sets roi_extent for
    # mono too, but the matcher must not *require* it for a mono pair.
    from _phase4_fixtures import light, pool, request
    from zecalibrator.core.matching import match_calibration

    lt = light(geometry=geo(cfa_phase="mono", roi_extent=None))
    dk = descriptor(
        "dark", "included",
        geometry=geo(cfa_phase="mono", roi_extent=None),
    )
    r = match_calibration(
        lt, request(), pool(dark=[candidate("d1", dk)]), policy()
    )
    assert r.outcome == "MATCHED"
    assert "MISSING_REQUIRED_FIELD" not in r.reason_codes


def test_mono_decoded_light_still_routes():
    # The decode path also records roi_extent=shape for mono; a mono master with
    # matching extent resolves a READY route (the Bayer gate never applies).
    md = build_sensor_metadata(
        shape=(4, 4),
        normalized={
            "cfa": "mono",
            "bin_x": 1.0,
            "bin_y": 1.0,
            "gain": 100.0,
            "offset": 50.0,
            "exposure_seconds": 10.0,
            "temperature_c": 20.0,
            "detector_model": "SYNTH-CFA",
        },
        conflicts=(),
        cards=(),
        declaration=ImportDeclaration(
            source="standard_light_contract", identity="standard-light",
            version="1", domain="raw", units="ADU",
        ),
        units="ADU",
    )
    light = light_constraints_from_sensor_metadata(md)
    snap = LibrarySnapshot(
        revision="r",
        schema_version="zecalibrator.library.v1",
        candidates={
            "dark": (
                candidate(
                    "d1",
                    descriptor(
                        "dark", "included", exposure_s=10.0,
                        geometry=geo(
                            cfa_phase="mono", roi_extent=(4, 4),
                            roi_origin=(0, 0), orientation="identity",
                            sensor_dimensions=None, binning=(1, 1),
                        ),
                        detector_obj=detector(instance="unknown", model="SYNTH-CFA"),
                        acquisition_obj=acquisition(
                            gain=100.0, offset=None, readout_mode=None,
                            adc_mode=None, temperature_c=20.0, exposure_s=10.0,
                        ),
                        optical_train_id=None,
                        filter=None,
                    ),
                ),
            )
        },
    )
    res = resolve_route(light, snap, policy())
    assert res.outcome == OUTCOME_READY
    assert not any(
        r.code == "MISSING_REQUIRED_FIELD" and r.field == "geometry.roi_extent"
        for r in res.reasons
    )
