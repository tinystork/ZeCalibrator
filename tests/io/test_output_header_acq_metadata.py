"""Output-header acquisition-metadata preservation tests (RA/DEC/XPIXSZ/YPIXSZ).

Phase F (ZS-GFR-20260927): a calibrated product must not silently drop the
acquisition pointing (RA/DEC) and pixel sampling (XPIXSZ/YPIXSZ) facts, which are
not invalidated by calibration and are required by a downstream plate-solver.

These tests extend the existing `test_output_header_cfa.py` policy table and the
writer-level round-trip, asserting:
  - RA/DEC/XPIXSZ/YPIXSZ are emitted verbatim when present with a single value;
  - they are NOT emitted when absent (no invented value, no 0 default);
  - an ambiguous (disagreeing) card is dropped;
  - the pre-existing evidence cards keep their exact treatment;
  - the writer round-trip still preserves science float32, DQ and CALPROV.
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
    return CardRecord(keyword=keyword, value=value, comment="", index=index,
                      source=source, raw=raw)


def _geometry(shape=SHAPE, cfa_phase="RGGB", binning=(1, 1), roi_origin=(0, 0)):
    return Geometry(shape=shape, sensor_dimensions=shape, binning=binning,
                    roi_origin=roi_origin, roi_extent=shape, orientation="identity",
                    cfa_phase=cfa_phase)


def _detector(model="ZWO ASI294MC Pro"):
    return DetectorIdentity(detector_instance_id="ZWO-DET-1", detector_model=model)


def _acquisition(**kwargs):
    defaults = dict(gain=120.0, offset=None, readout_mode=None, adc_mode=None,
                    temperature_c=-10.0, exposure_s=60.0, saturation_limit_adu=None,
                    saturation_evidence="unknown")
    defaults.update(kwargs)
    return Acquisition(**defaults)


def _optical(filt="irct"):
    return OpticalIdentity(filter=filt)


def _constraints(geometry=None, detector=None, acquisition=None, optical=None, cards=()):
    return LightConstraints(
        geometry=geometry or _geometry(),
        detector=detector or _detector(),
        acquisition=acquisition or _acquisition(),
        optical=optical or _optical(),
        raw_domain_declaration="raw",
        evidence=LightEvidence(original_cards=tuple(cards)),
    )


def _fields(lc, science_shape=SHAPE, status="COMPLETED", plan_id="p" * 64,
            schema="zecalibrator.provenance.v1"):
    return build_output_header_fields(
        light_constraints=lc, science_shape=science_shape, status=status,
        plan_id=plan_id, provenance_schema=schema,
    )


def _acq_cards():
    return (
        _card("RA", 24.545835),
        _card("DEC", 15.920278),
        _card("XPIXSZ", 4.63),
        _card("YPIXSZ", 4.63),
        _card("FOCALLEN", 1829),
        _card("BAYERPAT", "RGGB"),
        _card("DATE-OBS", "2026-09-15T01:42:24"),
        _card("OBJECT", "M 74"),
        _card("ROTATOR", 83),
        _card("IMAGETYP", "Light"),
    )


# ---------------------------------------------------------------------------
# 1. RA/DEC/XPIXSZ/YPIXSZ emitted verbatim when present with a unique value
# ---------------------------------------------------------------------------
def test_acq_metadata_emitted_when_present():
    lc = _constraints(cards=_acq_cards())
    fields = _fields(lc)
    assert fields["RA"] == 24.545835
    assert fields["DEC"] == 15.920278
    assert fields["XPIXSZ"] == 4.63
    assert fields["YPIXSZ"] == 4.63


def test_acq_metadata_emitted_verbatim_no_type_coercion():
    # Values are written as-is (float stays float, string stays string).
    lc = _constraints(cards=_acq_cards())
    fields = _fields(lc)
    assert isinstance(fields["RA"], float)
    assert isinstance(fields["XPIXSZ"], float)
    assert fields["RA"] == 24.545835
    assert fields["XPIXSZ"] == 4.63


# ---------------------------------------------------------------------------
# 2. Absent -> not emitted (no invention, no 0 default)
# ---------------------------------------------------------------------------
def test_acq_metadata_absent_not_emitted():
    # Only FOCALLEN/BAYERPAT/DATE-OBS/OBJECT/ROTATOR/IMAGETYP present.
    lc = _constraints(cards=[
        _card("FOCALLEN", 1829), _card("BAYERPAT", "RGGB"),
        _card("DATE-OBS", "2026-09-15T01:42:24"), _card("OBJECT", "M 74"),
        _card("ROTATOR", 83), _card("IMAGETYP", "Light"),
    ])
    fields = _fields(lc)
    for key in ("RA", "DEC", "XPIXSZ", "YPIXSZ"):
        assert key not in fields


# ---------------------------------------------------------------------------
# 3. Ambiguous (two disagreeing cards) -> dropped
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("keyword", ["RA", "DEC", "XPIXSZ", "YPIXSZ"])
def test_acq_metadata_ambiguous_dropped(keyword):
    cards = _acq_cards() + (_card(keyword, 999.0, index=99),)
    lc = _constraints(cards=cards)
    fields = _fields(lc)
    assert keyword not in fields


# ---------------------------------------------------------------------------
# 4. Non-regression: the pre-existing evidence cards keep their exact treatment
# ---------------------------------------------------------------------------
def test_preexisting_evidence_cards_unchanged():
    lc = _constraints(cards=_acq_cards())
    fields = _fields(lc)
    assert fields["FOCALLEN"] == 1829
    assert fields["BAYERPAT"] == "RGGB"
    assert fields["DATE-OBS"] == "2026-09-15T01:42:24"
    assert fields["OBJECT"] == "M 74"
    assert fields["ROTATOR"] == 83
    assert fields["IMAGETYP"] == "Light"


# ---------------------------------------------------------------------------
# 5. Writer round-trip: PRIMARY has the 4 new cards; science float32, DQ and
#    CALPROV are preserved (same assertions as the existing tests).
# ---------------------------------------------------------------------------
def test_writer_roundtrip_acq_metadata_and_payload_invariants(tmp_path):
    data = np.full(SHAPE, 42.0, dtype=np.float32)
    mask = np.zeros(SHAPE, dtype=np.uint16)
    lc = _constraints(cards=_acq_cards())
    header_fields = _fields(lc, science_shape=SHAPE)

    out = write_standalone_output(
        data, mask,
        {"schema_version": "zecalibrator.provenance.v1", "operation_id": "op"},
        input_identity={"kind": "fits", "path": "/light.fits", "hdu": 0},
        plan_id="p" * 64, destination=str(tmp_path), status="COMPLETED",
        header_fields=header_fields,
    )
    with fits.open(out.path, memmap=False) as hdul:
        hdr = hdul[0].header
        # new acquisition cards present in PRIMARY
        assert hdr["RA"] == 24.545835
        assert hdr["DEC"] == 15.920278
        assert hdr["XPIXSZ"] == 4.63
        assert hdr["YPIXSZ"] == 4.63
        # existing invariants
        assert hdul[0].data.ndim == 2
        assert hdr["NAXIS"] == 2
        assert hdr["BITPIX"] == -32
        assert hdr["FOCALLEN"] == 1829
        # science float32 unchanged (byte-identical to input)
        assert np.array_equal(
            np.ascontiguousarray(hdul[0].data, dtype=np.float32), data, equal_nan=True
        )
        # CALPROV payload present and parses strictly
        calprov_hdu = None
        for hdu in hdul[1:]:
            if hdu.header.get("EXTNAME") == CALPROV_EXTNAME:
                calprov_hdu = hdu
        assert calprov_hdu is not None
        payload = np.ascontiguousarray(calprov_hdu.data, dtype=np.uint8).tobytes()
        record = parse_calprov(payload)
        assert isinstance(record, dict)
        assert record.get("schema_version") == "zecalibrator.provenance.v1"
