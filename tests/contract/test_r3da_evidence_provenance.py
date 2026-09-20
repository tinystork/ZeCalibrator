"""R3D-A evidence/provenance foundation contract tests (P7-M3B).

Covers the two foundation deliverables:

A. D1d — explicit UNKNOWN vs KNOWN additive-processing-history discriminator on
   ``ProcessingProvenance`` (hard invariant, legacy deserialization, roundtrip).
B. Acquisition evidence adapters — ``CanonicalFact`` value object, producer
   fingerprint detection (exact match + generic fallback), per-producer
   keyword→canonical mapping with the REQUIRED gain ``semantic_domain``, and
   ordered HISTORY-card preservation (no generic reconstruction).

No producer derives processing state here (Siril stacking-normalization history
is NOT a ``raw_response``/``bias_state`` signal), and no route/GUI is touched.
"""

from __future__ import annotations

import pytest

from zecalibrator.core.descriptors import (
    ADDITIVE_HISTORY_STATES,
    DescriptorIntegrityError,
    MasterDescriptor,
    ProcessingProvenance,
    master_descriptor_from_dict,
)
from zecalibrator.core.metadata import (
    CanonicalFact,
    CardRecord,
    detect_producer,
    extract_acquisition_facts,
    history_cards,
)


def _card(keyword, value, index=0):
    return CardRecord(keyword=keyword, value=value, comment="", index=index, source="primary")


# ---------------------------------------------------------------------------
# A. D1d — additive_history_state discriminator
# ---------------------------------------------------------------------------

def test_unknown_with_nonempty_history_rejected():
    with pytest.raises(ValueError):
        ProcessingProvenance(
            source="synthetic_fixture",
            additive_history_state="unknown",
            additive_correction_history=("flat_dark_subtracted",),
        )


def test_unknown_with_empty_history_accepted():
    p = ProcessingProvenance(source="synthetic_fixture", additive_history_state="unknown")
    assert p.additive_history_state == "unknown"
    assert p.additive_correction_history == ()


def test_known_with_nonempty_history_accepted():
    p = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    assert p.additive_correction_history == ("flat_dark_subtracted",)


def test_known_with_empty_history_accepted():
    p = ProcessingProvenance(source="synthetic_fixture", additive_history_state="known")
    assert p.additive_correction_history == ()


def test_invalid_state_rejected():
    with pytest.raises(ValueError):
        ProcessingProvenance(source="synthetic_fixture", additive_history_state="maybe")


def test_known_history_entries_must_be_nonempty_strings():
    with pytest.raises(ValueError):
        ProcessingProvenance(
            source="synthetic_fixture",
            additive_history_state="known",
            additive_correction_history=("ok", ""),
        )


def test_state_vocabulary_frozen():
    assert ADDITIVE_HISTORY_STATES == ("unknown", "known")


def test_legacy_deserialization_without_state_reads_unknown():
    from zecalibrator.core.descriptors import _processing_from_dict

    p = _processing_from_dict({"source": "synthetic_fixture", "additive_correction_history": []})
    assert p.additive_history_state == "unknown"
    assert p.additive_correction_history == ()


def test_legacy_deserialization_nonempty_history_without_state_rejected():
    from zecalibrator.core.descriptors import _processing_from_dict

    # A legacy record carrying a non-empty history but no state discriminator is
    # NOT silently reinterpreted as known-empty; it is rejected (no bypass).
    with pytest.raises(ValueError):
        _processing_from_dict(
            {"source": "synthetic_fixture", "additive_correction_history": ["bias_removed"]}
        )


def test_to_dict_emits_additive_history_state():
    p = ProcessingProvenance(source="synthetic_fixture", additive_history_state="known")
    assert p.to_dict()["additive_history_state"] == "known"


def test_descriptor_roundtrip_preserves_state():
    # Rebuild a minimal master via the full serialization path and check the
    # discriminator survives exactly.
    base = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    d = {
        "master_type": "dark",
        "bias_state": "included",
        "pixel_domain": "sensor_adu",
        "physical_units": "ADU",
        "flat_form": None,
        "normalization_algorithm": None,
        "normalization_scalars": None,
        "geometry": {
            "shape": [2, 2],
            "sensor_dimensions": None,
            "binning": None,
            "roi_origin": None,
            "roi_extent": None,
            "orientation": None,
            "cfa_phase": "mono",
        },
        "detector": {
            "detector_instance_id": "SYNTH-DET-0001",
            "detector_model": None,
            "serial": None,
        },
        "acquisition": {
            "gain": 100.0,
            "offset": 50.0,
            "readout_mode": None,
            "adc_mode": None,
            "temperature_c": None,
            "exposure_s": 10.0,
            "saturation_limit_adu": None,
            "saturation_evidence": "unknown",
            "bias_exposure_max_s": None,
            "short_flat_profile": False,
        },
        "optical_train_id": None,
        "filter": None,
        "content_sha256": "a" * 64,
        "size_bytes": 16,
        "hdu": 0,
        "mask_identity": "b" * 64,
        "dq_state": "source_mask",
        "processing_provenance": dict(base.to_dict()),
        "validity_evidence": {
            "saturation_limit_known": True,
            "valid_normalization_count": None,
            "total_normalization_count": None,
            "quality_policy_state": "qualified",
            "illumination": None,
            "exposure_quality": None,
        },
    }
    rebuilt = master_descriptor_from_dict(d)
    pp = rebuilt.processing_provenance
    assert pp.additive_history_state == "known"
    assert pp.additive_correction_history == ("flat_dark_subtracted",)


def test_descriptor_roundtrip_unknown_state_identical():
    # Serialize then deserialize an "unknown" provenance; the discriminator must
    # round-trip as "unknown" (never fabricated known-empty).
    p = ProcessingProvenance(source="synthetic_fixture")
    assert p.to_dict()["additive_history_state"] == "unknown"
    from zecalibrator.core.descriptors import _processing_from_dict

    assert _processing_from_dict(dict(p.to_dict())).additive_history_state == "unknown"


# ---------------------------------------------------------------------------
# B. Producer fingerprint detection
# ---------------------------------------------------------------------------

def test_producer_fingerprint_siril():
    cards = (_card("PROGRAM", "Siril 1.2.1"),)
    assert detect_producer(cards) == "siril"


def test_producer_fingerprint_asiair():
    cards = (_card("CREATOR", "ZWO ASIAIR 1.9"),)
    assert detect_producer(cards) == "asiair"


def test_producer_fingerprint_nina():
    cards = (_card("SWCREATE", "N.I.N.A. 2.3"),)
    assert detect_producer(cards) == "nina"


def test_producer_fingerprint_sharpcap():
    cards = (_card("CREATOR", "SharpCap 4.1"),)
    assert detect_producer(cards) == "sharpcap"


def test_producer_fingerprint_indi():
    cards = (_card("CCD_GAIN", 120),)
    assert detect_producer(cards) == "indi"


def test_producer_fingerprint_fallback_generic():
    cards = (_card("INSTRUME", "SomeCamera"),)
    assert detect_producer(cards) == "generic"


def test_producer_fingerprint_no_substring_guessing():
    # "Siril" only matches PROGRAM *starting* with "Siril"; a different card is
    # never guessed. CREATOR mentioning "Siril" elsewhere is not a match.
    cards = (_card("CREATOR", "captured with Siril afterwards"),)
    assert detect_producer(cards) == "generic"


def test_producer_fingerprint_never_instrume_brand():
    # INSTRUME/camera brand is never a producer fingerprint.
    cards = (_card("INSTRUME", "ZWO ASIAIR"),)
    assert detect_producer(cards) == "generic"


# ---------------------------------------------------------------------------
# B. Per-producer keyword -> canonical mapping + semantic_domain
# ---------------------------------------------------------------------------

def test_generic_gain_fact_semantic_domain():
    cards = (_card("GAIN", 120),)
    facts = extract_acquisition_facts(cards, producer="generic")
    gain = [f for f in facts if f.field == "gain"]
    assert len(gain) == 1
    assert gain[0].semantic_domain == "producer_gain_units"
    assert gain[0].value == 120


def test_generic_egain_fact_semantic_domain():
    cards = (_card("EGAIN", 0.5),)
    facts = extract_acquisition_facts(cards, producer="generic")
    egain = [f for f in facts if f.field == "gain_e_per_adu"]
    assert len(egain) == 1
    assert egain[0].semantic_domain == "electrons_per_adu"
    assert egain[0].value == 0.5


def test_gain_and_egain_are_distinct_fields():
    cards = (_card("GAIN", 120), _card("EGAIN", 0.5))
    facts = extract_acquisition_facts(cards, producer="generic")
    fields = {f.field for f in facts}
    assert "gain" in fields
    assert "gain_e_per_adu" in fields


def test_sharpcap_offset_uses_blklevel():
    cards = (_card("BLKLEVEL", 256),)
    facts = extract_acquisition_facts(cards, producer="sharpcap")
    offset = [f for f in facts if f.field == "offset"]
    assert len(offset) == 1
    assert offset[0].value == 256
    assert offset[0].source_keyword == "BLKLEVEL"


def test_sharpcap_cfa_pattern_from_colortyp():
    cards = (_card("COLORTYP", "RGGB"),)
    facts = extract_acquisition_facts(cards, producer="sharpcap")
    cfa = [f for f in facts if f.field == "cfa_pattern"]
    assert len(cfa) == 1
    assert cfa[0].value == "RGGB"


def test_siril_exposure_single_exptime():
    cards = (_card("EXPTIME", 300.0),)
    facts = extract_acquisition_facts(cards, producer="siril")
    exp = [f for f in facts if f.field == "exposure_s"]
    assert len(exp) == 1
    assert exp[0].value == 300.0


def test_binning_is_a_pair_fact():
    cards = (_card("XBINNING", 2), _card("YBINNING", 2))
    facts = extract_acquisition_facts(cards, producer="generic")
    binning = [f for f in facts if f.field == "binning"]
    assert len(binning) == 1
    assert binning[0].value == (2, 2)
    assert binning[0].source_keyword == ("XBINNING", "YBINNING")


def test_pair_fact_requires_both_keywords():
    cards = (_card("XBINNING", 2),)
    facts = extract_acquisition_facts(cards, producer="generic")
    assert not [f for f in facts if f.field == "binning"]


def test_temperature_actual_vs_setpoint_distinct():
    cards = (_card("CCD-TEMP", -10.0), _card("SET-TEMP", -15.0))
    facts = extract_acquisition_facts(cards, producer="generic")
    actual = [f for f in facts if f.field == "temperature_actual"]
    setpoint = [f for f in facts if f.field == "temperature_setpoint"]
    assert len(actual) == 1 and actual[0].value == -10.0
    assert len(setpoint) == 1 and setpoint[0].value == -15.0


def test_canonical_fact_requires_gain_semantic_domain():
    with pytest.raises(ValueError):
        CanonicalFact(field="gain", value=1.0, source_keyword="GAIN", producer="generic")
    with pytest.raises(ValueError):
        CanonicalFact(
            field="gain_e_per_adu", value=1.0, source_keyword="EGAIN", producer="generic"
        )


def test_canonical_fact_to_dict():
    f = CanonicalFact(
        field="gain",
        value=120,
        source_keyword="GAIN",
        producer="generic",
        semantic_domain="producer_gain_units",
    )
    assert f.to_dict() == {
        "field": "gain",
        "value": 120,
        "source_keyword": "GAIN",
        "producer": "generic",
        "producer_version": None,
        "semantic_domain": "producer_gain_units",
        "confidence": "explicit",
    }


# ---------------------------------------------------------------------------
# C. Safe HISTORY handling (ordered, no generic reconstruction)
# ---------------------------------------------------------------------------

def test_history_cards_preserve_order_and_no_merge():
    cards = (
        _card("HISTORY", "first step", 0),
        _card("OTHER", 1, 1),
        _card("HISTORY", "second step", 2),
        _card("HISTORY", "third step", 3),
    )
    hist = history_cards(cards)
    assert [c.value for c in hist] == ["first step", "second step", "third step"]
    assert [c.index for c in hist] == [0, 2, 3]


def test_history_cards_never_generically_concatenate():
    # Two consecutive HISTORY cards remain two distinct records; the generic
    # accessor never joins them into a single reconstructed statement.
    cards = (
        _card("HISTORY", "…unnormalized outp", 0),
        _card("HISTORY", "ut, no image weighting", 1),
    )
    hist = history_cards(cards)
    assert len(hist) == 2
    assert hist[0].value == "…unnormalized outp"
    assert hist[1].value == "ut, no image weighting"


def test_history_cards_empty_when_no_history():
    cards = (_card("EXPTIME", 300.0),)
    assert history_cards(cards) == ()
