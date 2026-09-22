"""Output-header builder policy tests (CFA-metadata preservation).

The builder (:func:`zecalibrator.io.output_header.build_output_header_fields`)
is a pure dict builder: these tests assert the exact policy table — canonical
cards, closed evidence whitelist, geometry guard, mono/unknown CFA handling and
the anti-stale guarantee (no BZERO/BSCALE/CHECKSUM/DATASUM/BLANK, no EGAIN) —
plus the writer-level round-trip (NAXIS=2, BITPIX=-32, BAYERPAT present, no
NAXIS3, pixels == calibrated science).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from zecalibrator.core.descriptors import (
    Acquisition,
    DetectorIdentity,
    LightConstraints,
    LightEvidence,
    OpticalIdentity,
)
from zecalibrator.core.geometry import Geometry
from zecalibrator.core.metadata import CardRecord
from zecalibrator.io.output_header import build_output_header_fields
from zecalibrator.io.output_writer import (
    CALPROV_EXTNAME,
    parse_calprov,
    write_standalone_output,
)

SHAPE = (8, 12)


def _card(keyword, value, index=0, source="primary", raw=None):
    return CardRecord(
        keyword=keyword, value=value, comment="", index=index,
        source=source, raw=raw,
    )


def _geometry(shape=SHAPE, cfa_phase="RGGB", binning=(1, 1), roi_origin=(0, 0)):
    return Geometry(
        shape=shape, sensor_dimensions=shape, binning=binning,
        roi_origin=roi_origin, roi_extent=shape, orientation="identity",
        cfa_phase=cfa_phase,
    )


def _detector(model="ZWO ASI294MC Pro"):
    return DetectorIdentity(detector_instance_id="ZWO-DET-1", detector_model=model)


def _acquisition(**kwargs):
    defaults = dict(
        gain=120.0, offset=None, readout_mode=None, adc_mode=None,
        temperature_c=-10.0, exposure_s=60.0, saturation_limit_adu=None,
        saturation_evidence="unknown",
    )
    defaults.update(kwargs)
    return Acquisition(**defaults)


def _optical(filt="irct"):
    return OpticalIdentity(filter=filt)


def _constraints(
    geometry=None, detector=None, acquisition=None, optical=None, cards=(),
):
    return LightConstraints(
        geometry=geometry or _geometry(),
        detector=detector or _detector(),
        acquisition=acquisition or _acquisition(),
        optical=optical or _optical(),
        raw_domain_declaration="raw",
        evidence=LightEvidence(original_cards=tuple(cards)),
    )


def _fields(lc, science_shape=SHAPE, status="COMPLETED", plan_id="p" * 64, schema="zecalibrator.provenance.v1"):
    return build_output_header_fields(
        light_constraints=lc,
        science_shape=science_shape,
        status=status,
        plan_id=plan_id,
        provenance_schema=schema,
    )


def _write(tmp_path, lc, data=None, science_shape=SHAPE):
    data = np.asarray(np.full(SHAPE, 42.0, dtype=np.float32) if data is None else data, dtype=np.float32)
    mask = np.zeros(data.shape, dtype=np.uint16)
    header_fields = _fields(lc, science_shape=tuple(data.shape))
    out = write_standalone_output(
        data, mask, {"schema_version": "zecalibrator.provenance.v1", "operation_id": "op"},
        input_identity={"kind": "fits", "path": "/light.fits", "hdu": 0},
        plan_id="p" * 64,
        destination=str(tmp_path),
        status="COMPLETED",
        header_fields=header_fields,
    )
    return out, data


# ---------------------------------------------------------------------------
# Category D — ZeCalibrator identity cards (byte-identical values)
# ---------------------------------------------------------------------------
def test_zecalibrator_identity_cards_byte_identical():
    fields = _fields(_constraints(), plan_id="abcd" * 16, status="COMPLETED",
                     schema="zecalibrator.provenance.v1")
    assert fields["ZECALCAL"] == "zecalibrator"
    assert fields["HIERARCH ZECALSCHEMA"] == "zecalibrator.provenance.v1"
    assert fields["HIERARCH ZECALPLAN"] == ("abcd" * 16)[:16]
    assert fields["HIERARCH ZECALSTAT"] == "COMPLETED"


# ---------------------------------------------------------------------------
# CFA round-trip through the real writer
# ---------------------------------------------------------------------------
def test_cfa_roundtrip_bayer_preserved(tmp_path):
    lc = _constraints(cards=[_card("BAYERPAT", "RGGB")])
    out, data = _write(tmp_path, lc)
    with fits.open(out.path, memmap=False) as hdul:
        hdr = hdul[0].header
        assert hdul[0].data.ndim == 2
        assert hdr["NAXIS"] == 2
        assert "NAXIS3" not in hdr
        assert hdr["BITPIX"] == -32
        assert hdr["BAYERPAT"] == "RGGB"
        assert hdr["INSTRUME"] == "ZWO ASI294MC Pro"
        assert hdr["FILTER"] == "irct"
        assert hdr["EXPTIME"] == 60.0
        assert hdr["GAIN"] == 120.0
        assert "EGAIN" not in hdr
        assert hdr["CCD-TEMP"] == -10.0
        assert np.array_equal(
            np.ascontiguousarray(hdul[0].data, dtype=np.float32), data, equal_nan=True
        )


@pytest.mark.parametrize("phase", ["GRBG", "RGGB", "BGGR", "GBRG"])
def test_all_bayer_phases_emitted_verbatim(phase):
    # A narrowing of _BAYER_PHASES (or a swapped phase mapping) must fail here:
    # every explicit Bayer phase is emitted verbatim as BAYERPAT.
    lc = _constraints(geometry=_geometry(cfa_phase=phase))
    fields = _fields(lc)
    assert fields["BAYERPAT"] == phase


def test_mono_input_no_bayer_invented(tmp_path):
    lc = _constraints(geometry=_geometry(cfa_phase="mono"))
    fields = _fields(lc)
    assert "BAYERPAT" not in fields
    out, data = _write(tmp_path, lc)
    with fits.open(out.path, memmap=False) as hdul:
        assert hdul[0].header["NAXIS"] == 2
        assert "BAYERPAT" not in hdul[0].header


def test_unknown_cfa_no_bayer_and_no_silent_assumption():
    lc = _constraints(geometry=_geometry(cfa_phase=None))
    fields = _fields(lc)
    assert "BAYERPAT" not in fields


# ---------------------------------------------------------------------------
# Realistic ASIAIR header policy
# ---------------------------------------------------------------------------
def test_realistic_asiair_policy_honoured():
    cards = (
        _card("INSTRUME", "ZWO ASI294MC Pro"),
        _card("BAYERPAT", "RGGB"),
        _card("FILTER", "irct"),
        _card("EXPTIME", 60.0),
        _card("GAIN", 120.0),
        _card("EGAIN", 0.5),  # must never leak through
        _card("OFFSET", 30.0),
        _card("XBINNING", 1),
        _card("YBINNING", 1),
        _card("CCD-TEMP", -10.0),
        _card("DATE-OBS", "2026-09-15T01:42:24"),
        _card("OBJECT", "M 74"),
        _card("FOCALLEN", 1829),
        _card("ROTATOR", 83),
        _card("IMAGETYP", "Light"),
    )
    lc = _constraints(cards=cards)
    fields = _fields(lc)
    assert fields["BAYERPAT"] == "RGGB"
    assert fields["INSTRUME"] == "ZWO ASI294MC Pro"
    assert fields["FILTER"] == "irct"
    assert fields["EXPTIME"] == 60.0
    assert fields["GAIN"] == 120.0  # gain setting, not EGAIN
    assert "EGAIN" not in fields
    assert fields["XBINNING"] == 1
    assert fields["YBINNING"] == 1
    assert fields["CCD-TEMP"] == -10.0
    # canonical offset is None -> evidence OFFSET surfaces
    assert fields["OFFSET"] == 30.0
    assert fields["DATE-OBS"] == "2026-09-15T01:42:24"
    assert fields["OBJECT"] == "M 74"
    assert fields["FOCALLEN"] == 1829
    assert fields["ROTATOR"] == 83
    assert fields["IMAGETYP"] == "Light"  # verbatim, no invention


def test_imagetyp_preserved_verbatim():
    lc = _constraints(cards=[_card("IMAGETYP", "Light")])
    fields = _fields(lc)
    assert fields["IMAGETYP"] == "Light"
    assert "Calibrated Light" not in fields.values()


# ---------------------------------------------------------------------------
# Anti-stale: no scaling/checksum/blank leakage, BITPIX regenerated
# ---------------------------------------------------------------------------
def test_anti_stale_no_scaling_or_checksum(tmp_path):
    cards = (
        _card("BITPIX", 16),
        _card("BZERO", 32768),
        _card("BSCALE", 1),
        _card("CHECKSUM", "abc"),
        _card("DATASUM", "def"),
        _card("BLANK", -999),
        _card("NAXIS", 2),
        _card("NAXIS1", 12),
        _card("NAXIS2", 8),
        _card("EXTEND", True),
    )
    lc = _constraints(cards=cards)
    fields = _fields(lc)
    for key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND",
                "BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
        assert key not in fields
    out, _ = _write(tmp_path, lc)
    with fits.open(out.path, memmap=False) as hdul:
        hdr = hdul[0].header
        assert hdr["BITPIX"] == -32
        for key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
            assert key not in hdr


# ---------------------------------------------------------------------------
# Geometry guard: science_shape != canonical geometry.shape -> no BAYERPAT
# ---------------------------------------------------------------------------
def test_geometry_guard_suppresses_bayer():
    lc = _constraints(geometry=_geometry(shape=SHAPE, cfa_phase="RGGB"))
    fields = _fields(lc, science_shape=(4, 4))  # shape mismatch
    assert "BAYERPAT" not in fields


# ---------------------------------------------------------------------------
# Binning / ROI round-trip conventions (geometry.binning is (bin_y, bin_x);
# roi_origin is (y, x)).
# ---------------------------------------------------------------------------
def test_binning_roundtrip_non_square():
    lc = _constraints(geometry=_geometry(binning=(2, 3)))
    fields = _fields(lc)
    assert fields["XBINNING"] == 3  # bin_x == binning[1]
    assert fields["YBINNING"] == 2  # bin_y == binning[0]


def test_roi_origin_roundtrip():
    lc = _constraints(geometry=_geometry(roi_origin=(5, 7)))
    fields = _fields(lc)
    assert fields["XORGSUBF"] == 7  # x == roi_origin[1]
    assert fields["YORGSUBF"] == 5  # y == roi_origin[0]


def test_binning_and_roi_omitted_when_unknown():
    lc = _constraints(geometry=_geometry(binning=None, roi_origin=None))
    fields = _fields(lc)
    for key in ("XBINNING", "YBINNING", "XORGSUBF", "YORGSUBF"):
        assert key not in fields


# ---------------------------------------------------------------------------
# Canonical OFFSET wins over evidence; evidence fallback only when unknown
# ---------------------------------------------------------------------------
def test_canonical_offset_wins_over_evidence():
    lc = _constraints(
        acquisition=_acquisition(offset=42.0),
        cards=[_card("OFFSET", 30.0)],
    )
    fields = _fields(lc)
    assert fields["OFFSET"] == 42.0


def test_evidence_offset_only_when_canonical_unknown():
    lc = _constraints(acquisition=_acquisition(offset=None), cards=[_card("OFFSET", 30.0)])
    assert _fields(lc)["OFFSET"] == 30.0
    # absent evidence -> no OFFSET card at all
    lc2 = _constraints(acquisition=_acquisition(offset=None), cards=())
    assert "OFFSET" not in _fields(lc2)


def test_ambiguous_evidence_card_is_dropped():
    lc = _constraints(cards=[_card("DATE-OBS", "2026-01-01T00:00:00"), _card("DATE-OBS", "2026-01-02T00:00:00")])
    fields = _fields(lc)
    assert "DATE-OBS" not in fields


# ---------------------------------------------------------------------------
# Science / CALPROV parity: header projection does not touch payload or record
# ---------------------------------------------------------------------------
def test_science_and_calprov_parity(tmp_path):
    data = np.full(SHAPE, 42.0, dtype=np.float32)
    mask = np.zeros(SHAPE, dtype=np.uint16)
    lc = _constraints(cards=[_card("BAYERPAT", "RGGB"), _card("OBJECT", "M 74")])

    # Distinct destination dirs: same logical id (deterministic), so two
    # no-clobber publishes of the same id into one dir would collide.
    dest_a = tmp_path / "a"
    dest_b = tmp_path / "b"
    dest_a.mkdir()
    dest_b.mkdir()

    with_header = write_standalone_output(
        data, mask, {"schema_version": "zecalibrator.provenance.v1", "operation_id": "op"},
        input_identity={"kind": "fits", "path": "/light.fits", "hdu": 0},
        plan_id="p" * 64, destination=str(dest_a), status="COMPLETED",
        header_fields=_fields(lc, science_shape=SHAPE),
    )
    without_header = write_standalone_output(
        data, mask, {"schema_version": "zecalibrator.provenance.v1", "operation_id": "op"},
        input_identity={"kind": "fits", "path": "/light.fits", "hdu": 0},
        plan_id="p" * 64, destination=str(dest_b), status="COMPLETED",
        header_fields={},
    )
    assert with_header.science_digest == without_header.science_digest
    assert with_header.logical_id == without_header.logical_id

    def _calprov(path):
        with fits.open(path, memmap=False) as hdul:
            payload = np.ascontiguousarray(hdul[CALPROV_EXTNAME].data, dtype=np.uint8).tobytes()
        return parse_calprov(payload)

    assert _calprov(with_header.path) == _calprov(without_header.path)
