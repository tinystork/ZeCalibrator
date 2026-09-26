"""Revision model tests: immutability, integrity digest, determinism, promotion,
and the knowledge/action distinction (a BPM is NOT a list of (x, y))."""

from __future__ import annotations

import dataclasses

import pytest

from zecalibrator.bpm.errors import BpmBaseCorrupted
from zecalibrator.bpm.revision import (
    Revision,
    action_eligible_sites,
    make_revision,
    promote_revision,
    revision_digest,
)
from zecalibrator.bpm.vocabulary import (
    ACTION_STATE_ABSTAIN_CENSORED,
    ACTION_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
    ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION,
    ACTION_STATE_NO_ACTION_REQUIRED,
    REVISION_STATE_CANDIDATE,
    REVISION_STATE_PROMOTED,
)

from conftest import make_identity, make_rev, make_site


def test_same_evidence_is_deterministic():
    ident = make_identity()
    r1 = make_rev(ident, sites=[make_site(10, 20), make_site(30, 40)])
    r2 = make_rev(ident, sites=[make_site(30, 40), make_site(10, 20)])
    assert r1.revision_id == r2.revision_id
    assert r1.integrity_digest == r2.integrity_digest
    assert r1.sites == r2.sites  # normalized order


def test_new_evidence_is_new_revision():
    ident = make_identity()
    r1 = make_rev(ident, sites=[make_site(10, 20)])
    r2 = make_rev(ident, sites=[make_site(10, 20), make_site(30, 40)])
    assert r1.revision_id != r2.revision_id


def test_revision_is_immutable():
    r = make_rev(make_identity(), sites=[make_site(10, 20)])
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.state = REVISION_STATE_CANDIDATE  # type: ignore[misc]


def test_verify_ok_then_tampered_rejected():
    r = make_rev(make_identity(), sites=[make_site(10, 20)])
    r.verify()  # no raise
    tampered = Revision(
        revision_id=r.revision_id,
        state=r.state,
        sensor_identity=r.sensor_identity,
        sites=r.sites,
        integrity_digest="0" * 64,
        schema_version=r.schema_version,
        sequence=r.sequence,
    )
    with pytest.raises(BpmBaseCorrupted):
        tampered.verify()


def test_promote_creates_new_revision_old_remains():
    ident = make_identity()
    candidate = make_rev(ident, state=REVISION_STATE_CANDIDATE, sites=[make_site(5, 5)], sequence=0)
    promoted = promote_revision(candidate, sequence=1)
    assert promoted.state == REVISION_STATE_PROMOTED
    assert promoted.revision_id != candidate.revision_id
    assert promoted.sites == candidate.sites
    # the candidate is untouched
    assert candidate.state == REVISION_STATE_CANDIDATE
    assert candidate.sequence == 0


def test_action_eligibility_preserves_distinction():
    ident = make_identity()
    eligible = make_site(1, 1, action_state=ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION)
    no_action = make_site(2, 2, action_state=ACTION_STATE_NO_ACTION_REQUIRED)
    abstain = make_site(3, 3, action_state=ACTION_STATE_ABSTAIN_CENSORED)
    abstain2 = make_site(4, 4, action_state=ACTION_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE)
    rev = make_rev(ident, sites=[eligible, no_action, abstain, abstain2])
    # All sites are preserved as records, not collapsed to (x, y).
    assert len(rev.sites) == 4
    # Only ACTION ELIGIBLE reaches reconstruction.
    assert action_eligible_sites(rev) == (eligible,)
    # The distinctions are preserved on each record.
    by_position = {s.position: s for s in rev.sites}
    assert by_position[(2, 2)].action_state == ACTION_STATE_NO_ACTION_REQUIRED
    assert by_position[(3, 3)].action_state == ACTION_STATE_ABSTAIN_CENSORED


def test_digest_excludes_sequence():
    ident = make_identity()
    r1 = make_rev(ident, sites=[make_site(1, 1)], sequence=0)
    r2 = make_rev(ident, sites=[make_site(1, 1)], sequence=7)
    # same content identity (sequence is store metadata, not part of the digest)
    assert r1.revision_id == r2.revision_id
    assert revision_digest(r1.state, r1.sensor_identity, r1.sites) == r1.integrity_digest
