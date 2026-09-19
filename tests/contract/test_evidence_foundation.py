"""R3A evidence-foundation invariants (no-derivation, additive).

These contract tests pin the provenance-carrying normalization foundation:

* unknown/vendor/HIERARCH cards are preserved in the collected audit;
* ``actual_plane_shape`` is structural evidence of the decoded plane and is
  never aliased to / derived from ``sensor_dimensions``;
* ``sensor_dimensions`` is never populated from NAXIS;
* a dubious partial card (``ROWORDER``) maps to nothing — no inferred fact.

None of these change any matching/selection result; they only pin the additive
provenance/structural-evidence boundary introduced in R3A.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from zecalibrator.core.metadata import (
    CardRecord,
    ImportDeclaration,
    build_sensor_metadata,
    collect_cards,
    resolve_aliases_with_provenance,
)
from zecalibrator.io.raw_decoder import decode_fits


def _decl(**overrides) -> ImportDeclaration:
    base = dict(
        source="synthetic_fixture",
        identity="SYNTH-EVIDENCE-1",
        version="1.0",
        domain="raw",
        units="ADU",
        detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-MONO",
        gain=100,
        offset=50,
        readout_mode="MODE_A",
        adc_mode="MODE_16",
        binning=(1, 1),
        orientation="identity",
        cfa_phase="mono",
        roi_origin=(0, 0),
        exposure_s=10.0,
        temperature_c=20.0,
        filter="NONE",
        optical_train_id="SYNTH-TRAIN-1",
    )
    base.update(overrides)
    return ImportDeclaration(**base)


def _write(path, data, extra_cards=(), with_bunit=True):
    hdu = fits.PrimaryHDU(data)
    if with_bunit:
        hdu.header["BUNIT"] = "ADU"
    for key, val in extra_cards:
        hdu.header[key] = val
    hdu.writeto(path, overwrite=True)
    return path


def _decode(data, extra_cards=(), **decl_overrides):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "frame.fits"
        _write(p, data, extra_cards)
        return decode_fits(p, declaration=_decl(**decl_overrides))


# ---------------------------------------------------------------------------
# TASK-4(a): unknown/vendor/HIERARCH cards are preserved in the collected audit.
# ---------------------------------------------------------------------------

def test_unknown_vendor_hierarch_cards_preserved():
    header = fits.Header()
    header["BUNIT"] = "ADU"
    header["EXPTIME"] = 10.0
    header["VENDORSN"] = "ABC-123"  # unknown vendor serial card
    header["ROWORDER"] = "TOP-DOWN"  # dubious partial card
    header.append(fits.Card("HIERARCH VENDOR CAMERA", "ASI1600MM"))
    cards = collect_cards(header, source="primary")
    keywords = [c.keyword for c in cards]
    assert "VENDORSN" in keywords
    assert "ROWORDER" in keywords
    assert any(k.startswith("HIERARCH") for k in keywords)


def test_unknown_vendor_hierarch_cards_preserved_through_decode():
    f = _decode(
        np.zeros((2, 2), dtype=np.int16),
        extra_cards=[
            ("VENDORSN", "ABC-123"),
            ("ROWORDER", "TOP-DOWN"),
        ],
    )
    keywords = [c.keyword for c in f.metadata.original_cards]
    assert "VENDORSN" in keywords
    assert "ROWORDER" in keywords


# ---------------------------------------------------------------------------
# TASK-2: actual_plane_shape is structural evidence of the decoded plane,
# separate from sensor_dimensions.
# ---------------------------------------------------------------------------

def test_actual_plane_shape_is_structural_evidence_of_decoded_plane():
    shape = (3, 5)
    f = _decode(np.zeros(shape, dtype=np.int16), sensor_dimensions=(2048, 2048))
    prov = f.metadata.provenance["actual_plane_shape"]
    assert prov.value == (3, 5)  # decoded plane, not the declaration's sensor dims
    assert prov.source == "structural"
    assert prov.confidence == "structural"
    # Geometry.shape continues to be the decoded plane.
    assert f.metadata.geometry.shape == (3, 5)


def test_actual_plane_shape_never_derived_from_sensor_dimensions():
    shape = (3, 5)
    f = _decode(np.zeros(shape, dtype=np.int16), sensor_dimensions=(2048, 2048))
    # actual_plane_shape tracks the decoded plane; sensor_dimensions stays distinct.
    assert f.metadata.provenance["actual_plane_shape"].value == (3, 5)
    assert f.metadata.geometry.sensor_dimensions == (2048, 2048)
    # The structural fact is never injected into the flat normalized mapping.
    assert "actual_plane_shape" not in f.metadata.normalized
    assert "sensor_dimensions" not in f.metadata.normalized


# ---------------------------------------------------------------------------
# TASK-4(c): sensor_dimensions is NEVER populated from NAXIS.
# ---------------------------------------------------------------------------

def test_sensor_dimensions_never_populated_from_naxis():
    # No declaration sensor_dimensions: geometry.sensor_dimensions stays None
    # even though NAXIS1/NAXIS2 describe the decoded plane.
    f = _decode(np.zeros((3, 5), dtype=np.int16))
    assert f.metadata.geometry.sensor_dimensions is None
    # The only structural fact is actual_plane_shape; sensor_dimensions is absent.
    assert "sensor_dimensions" not in f.metadata.provenance
    assert f.metadata.provenance["actual_plane_shape"].value == (3, 5)


# ---------------------------------------------------------------------------
# TASK-4(d): a dubious partial card (ROWORDER) maps to nothing.
# ---------------------------------------------------------------------------

def test_roworder_maps_to_nothing():
    cards = (
        CardRecord("BUNIT", "ADU", "", 0, "primary"),
        CardRecord("ROWORDER", "TOP-DOWN", "", 1, "primary"),
    )
    normalized, conflicts, malformed, provenance = resolve_aliases_with_provenance(cards)
    assert "roworder" not in normalized
    assert "roworder" not in provenance
    assert conflicts == ()
    assert malformed == ()


# ---------------------------------------------------------------------------
# TASK-1: provenance carries (value, source, confidence) for explicit facts.
# ---------------------------------------------------------------------------

def test_explicit_fact_carries_card_provenance():
    cards = (
        CardRecord("BUNIT", "ADU", "", 0, "primary"),
        CardRecord("EXPTIME", 10.0, "", 1, "primary"),
    )
    normalized, conflicts, malformed, provenance = resolve_aliases_with_provenance(cards)
    assert normalized["exposure_seconds"] == 10.0
    p = provenance["exposure_seconds"]
    assert p.value == 10.0
    assert p.source == "EXPTIME"
    assert p.confidence == "explicit"


def test_alias_group_provenance_records_all_keywords():
    cards = (
        CardRecord("BUNIT", "ADU", "", 0, "primary"),
        CardRecord("EXPTIME", 20.0, "", 1, "primary"),
        CardRecord("EXPOSURE", 20.0, "", 2, "primary"),
    )
    normalized, conflicts, malformed, provenance = resolve_aliases_with_provenance(cards)
    assert normalized["exposure_seconds"] == 20.0
    p = provenance["exposure_seconds"]
    assert p.source == ("EXPTIME", "EXPOSURE")
    assert p.confidence == "explicit"


def test_provenance_is_parallel_and_flat_normalized_unchanged():
    # The flat normalized dict keeps only canonical field -> value; provenance
    # is a separate structure keyed by the same canonical fields.
    f = _decode(np.zeros((2, 2), dtype=np.int16), exposure_s=10.0)
    md = f.metadata
    # The flat normalized dict keeps canonical field -> value for *card* facts
    # (BUNIT here); the declaration supplies exposure_s separately (unchanged).
    assert "bunit" in md.normalized
    assert md.normalized["bunit"] == "ADU"
    assert "actual_plane_shape" not in md.normalized
    assert set(md.provenance.keys()) >= {"actual_plane_shape", "bunit"}
    assert md.provenance["actual_plane_shape"].source == "structural"
    assert md.provenance["bunit"].source == "BUNIT"
    assert md.provenance["bunit"].confidence == "explicit"


# ---------------------------------------------------------------------------
# REWORK-1 (F1): HIERARCH-encoded standard keywords normalize EXACTLY like
# their plain counterparts; the collected audit keeps the HIERARCH prefix.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("plain", "value", "field"),
    [
        ("EXPTIME", 300, "exposure_seconds"),
        ("GAIN", 120, "gain"),
        ("BSCALE", 2, "bscale"),
    ],
)
def test_hierarch_standard_keyword_normalizes_like_plain(plain, value, field):
    plain_cards = (
        CardRecord("BUNIT", "ADU", "", 0, "primary"),
        CardRecord(plain, value, "", 1, "primary"),
    )
    hierarch_cards = (
        CardRecord("BUNIT", "ADU", "", 0, "primary"),
        CardRecord(f"HIERARCH {plain}", value, "", 1, "primary"),
    )
    pn, pc, pm, pp = resolve_aliases_with_provenance(plain_cards)
    hn, hc, hm, hp = resolve_aliases_with_provenance(hierarch_cards)
    assert pn == hn
    assert pm == hm
    assert pc == hc == ()
    assert hn.get(field) == float(value)


def test_hierarch_malformed_numeric_matches_plain():
    plain_cards = (
        CardRecord("BUNIT", "ADU", "", 0, "primary"),
        CardRecord("EXPTIME", "abc", "", 1, "primary"),
    )
    hierarch_cards = (
        CardRecord("BUNIT", "ADU", "", 0, "primary"),
        CardRecord("HIERARCH EXPTIME", "abc", "", 1, "primary"),
    )
    pn, pc, pm, pp = resolve_aliases_with_provenance(plain_cards)
    hn, hc, hm, hp = resolve_aliases_with_provenance(hierarch_cards)
    assert pn == hn
    assert pm == hm == ("EXPTIME",)
    assert pc == hc == ()


def test_collected_audit_keeps_hierarch_prefix():
    header = fits.Header()
    header["BUNIT"] = "ADU"
    header.append(fits.Card("HIERARCH EXPTIME", 300))
    header.append(fits.Card("HIERARCH GAIN", 120))
    cards = collect_cards(header, source="primary")
    keywords = [c.keyword for c in cards]
    assert "HIERARCH EXPTIME" in keywords
    assert "HIERARCH GAIN" in keywords
    assert "EXPTIME" not in keywords
    assert "GAIN" not in keywords
