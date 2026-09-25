"""P3C-4 LOT 1 — temporal-persistence evidence contract tests.

These tests pin the *contract*, not any inference rule:

* the result is structured (explicit state, no magic boolean, no score);
* ``UNDETERMINED`` is a first-class, normal outcome and insufficient evidence
  is never promoted to ``YES`` or demoted to ``NO``;
* censoring: a censored sample cannot contribute quantitatively, only count as
  "a censored observation exists";
* the distinction from ``epoch_count >= 2`` (independence metadata is NOT
  temporal persistence evidence);
* no spatial threshold disguised as temporal evidence.
"""

from __future__ import annotations

import pytest

from research.p3c4.temporal_evidence import (
    ALLOWED_MEASUREMENT_SOURCES,
    NO,
    UNDETERMINED,
    YES,
    CensoredContributionError,
    DisallowedEvidenceSourceError,
    GroupSignature,
    InvalidTemporalStateError,
    TemporalPersistenceEvidence,
    TemporalSignature,
    assess_temporal_persistence,
)


def _group(group_id, epoch_id, *, valid=6, censored=0, present=YES):
    return GroupSignature(
        group_id=group_id,
        epoch_id=epoch_id,
        valid_samples=valid,
        censored_samples=censored,
        signature_present=present,
    )


def _sig(*groups, x=8, y=8):
    return TemporalSignature(coordinate_x=x, coordinate_y=y, groups=tuple(groups))


# ---------------------------------------------------------------------------
# 1) Structured result — no magic boolean, no score
# ---------------------------------------------------------------------------


def test_result_state_is_explicit_ternary_not_boolean():
    ev = assess_temporal_persistence(_sig(_group("g0", "e0"), _group("g1", "e1")))
    assert ev.state in (YES, NO, UNDETERMINED)
    assert isinstance(ev.state, str)
    assert ev.state == YES  # two groups across two epochs -> persistent


def test_no_numeric_score_field():
    from dataclasses import fields

    names = {f.name for f in fields(TemporalPersistenceEvidence)}
    # The result carries counts and a structured signature, never a probability.
    assert "confidence" not in names
    assert "score" not in names
    assert "probability" not in names
    assert "temporal_signature" in names
    assert "evidence_refs" in names


def test_invalid_state_rejected():
    with pytest.raises(InvalidTemporalStateError):
        GroupSignature("g", "e", 6, 0, "MAYBE")
    with pytest.raises(InvalidTemporalStateError):
        TemporalPersistenceEvidence(
            state="PERSISTENT",
            groups_examined=0,
            groups_supporting=0,
            epochs_examined=0,
            valid_samples=0,
            censored_samples=0,
            temporal_signature=_sig(),
        )


# ---------------------------------------------------------------------------
# 4) UNDETERMINED is normal — insufficient is never YES/NO
# ---------------------------------------------------------------------------


def test_insufficient_valid_samples_is_undetermined_not_yes():
    # Total valid samples below the minimum -> insufficient, so UNDETERMINED.
    # One group with one valid sample, the other censored-only (no valid samples).
    g0 = _group("g0", "e0", valid=1)
    g1 = GroupSignature("g1", "e1", valid_samples=0, censored_samples=6, signature_present=UNDETERMINED)
    ev = assess_temporal_persistence(_sig(g0, g1))
    assert ev.valid_samples == 1
    assert ev.state == UNDETERMINED
    assert ev.state != YES


def test_single_group_is_undetermined_not_yes():
    # One independent group can never prove persistence across groups/epochs.
    ev = assess_temporal_persistence(_sig(_group("g0", "e0")))
    assert ev.state == UNDETERMINED


def test_two_groups_same_epoch_is_not_yes():
    # Two groups but a single epoch: no temporal persistence across epochs.
    ev = assess_temporal_persistence(_sig(_group("g0", "e0"), _group("g1", "e0")))
    assert ev.state != YES


def test_insufficient_never_demoted_to_no():
    # No signal at all (all absent) is insufficient, not a positive "not persisted".
    ev = assess_temporal_persistence(
        _sig(_group("g0", "e0", present=NO), _group("g1", "e1", present=NO))
    )
    assert ev.state == UNDETERMINED
    assert ev.state != NO


def test_positive_non_persistence_is_no():
    # A residual observed in one group but not recurring across two groups
    # is positive non-persistence (transient / moving sky structure).
    ev = assess_temporal_persistence(
        _sig(_group("g0", "e0", present=YES), _group("g1", "e1", present=NO))
    )
    assert ev.state == NO


# ---------------------------------------------------------------------------
# 5) Censoring (§14)
# ---------------------------------------------------------------------------


def test_censored_only_group_cannot_claim_presence():
    with pytest.raises(CensoredContributionError):
        GroupSignature("g", "e", valid_samples=0, censored_samples=6, signature_present=YES)


def test_censored_samples_never_count_as_supporting():
    # A censored-only group has UNDETERMINED presence and never supports.
    g0 = _group("g0", "e0")  # valid support
    g1 = GroupSignature("g1", "e1", valid_samples=0, censored_samples=6, signature_present=UNDETERMINED)
    ev = assess_temporal_persistence(_sig(g0, g1))
    assert ev.censored_samples == 6
    # g1 does not support -> only one supporting group -> not YES.
    assert ev.groups_supporting == 1
    assert ev.state != YES


def test_censored_sample_counts_only_as_existence():
    ev = assess_temporal_persistence(_sig(_group("g0", "e0", censored=2)))
    assert ev.censored_samples == 2
    # The censored samples contribute nothing to valid_samples.
    assert ev.valid_samples == 6


# ---------------------------------------------------------------------------
# 6) Distinction from epoch_count >= 2 (independence metadata != evidence)
# ---------------------------------------------------------------------------


def test_epoch_count_is_not_a_field_of_the_result():
    from dataclasses import fields

    names = {f.name for f in fields(TemporalPersistenceEvidence)}
    assert "epoch_count" not in names
    assert "independent_group_count" not in names


def test_two_epochs_do_not_imply_persistence():
    # Two epochs examined, but the signature does not recur in either -> the
    # evaluator does NOT recopy epoch_count as evidence. UNDETERMINED, not YES.
    ev = assess_temporal_persistence(
        _sig(_group("g0", "e0", present=NO), _group("g1", "e1", present=NO))
    )
    assert ev.epochs_examined == 2
    assert ev.state == UNDETERMINED


def test_epochs_examined_counts_measured_epochs_not_metadata():
    # epochs_examined is derived from the measured signature, never an external
    # epoch_count. Two distinct epochs -> 2; duplicate epoch -> 1.
    ev2 = assess_temporal_persistence(_sig(_group("g0", "e0"), _group("g1", "e1")))
    ev1 = assess_temporal_persistence(_sig(_group("g0", "e0"), _group("g1", "e0")))
    assert ev2.epochs_examined == 2
    assert ev1.epochs_examined == 1


# ---------------------------------------------------------------------------
# 7) No spatial threshold disguised as temporal evidence
# ---------------------------------------------------------------------------


def test_contract_parameters_are_temporal_not_spatial():
    from research.p3c4.temporal_evidence import TEMPORAL_CONTRACT_PARAMETERS

    spatial_tokens = ("ADU", "adu", "snr", "SNR", "radius", "dispersion", "scale", "pixel")
    for name, value, rationale in TEMPORAL_CONTRACT_PARAMETERS:
        assert isinstance(value, int), f"{name} is a count, not a threshold"
        assert not any(tok in name for tok in spatial_tokens), f"{name} looks spatial"
        # The rationale must justify the parameter as temporal.
        assert "temporal" in rationale.lower(), f"{name} rationale not temporal: {rationale!r}"


def test_measurement_sources_are_temporal_identity_based():
    # Every allowed source is identity/temporal/residual-behaviour — none is a
    # spatial threshold. The list is fixed and auditable.
    assert "fixed_sensor_coordinate" in ALLOWED_MEASUREMENT_SOURCES
    assert "epoch_identity" in ALLOWED_MEASUREMENT_SOURCES
    assert "group_identity" in ALLOWED_MEASUREMENT_SOURCES
    assert "per_group_site_residual" in ALLOWED_MEASUREMENT_SOURCES
    assert "valid_sample_count" in ALLOWED_MEASUREMENT_SOURCES
    assert "censored_sample_existence" in ALLOWED_MEASUREMENT_SOURCES
    # No amplitude/scale/SNR source may appear.
    spatial = [s for s in ALLOWED_MEASUREMENT_SOURCES if any(t in s for t in ("adu", "snr", "scale", "amplitude"))]
    assert spatial == [], f"spatial sources leaked in: {spatial}"


def test_evidence_refs_restricted_to_allowed_sources():
    ev = assess_temporal_persistence(_sig(_group("g0", "e0"), _group("g1", "e1")))
    for ref in ev.evidence_refs:
        assert ref in ALLOWED_MEASUREMENT_SOURCES
    with pytest.raises(DisallowedEvidenceSourceError):
        TemporalPersistenceEvidence(
            state=YES,
            groups_examined=2,
            groups_supporting=2,
            epochs_examined=2,
            valid_samples=12,
            censored_samples=0,
            temporal_signature=_sig(_group("g0", "e0"), _group("g1", "e1")),
            evidence_refs=("same_cfa_local_residual",),  # not an allowed temporal source
        )
