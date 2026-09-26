"""BPM lookup: deterministic selection of a compatible revision + typed results.

The pure selection rule (documented and tested) is: among a base's **promoted**
revisions whose :class:`~zecalibrator.bpm.identity.SensorIdentity` is exactly
compatible with the run identity, select the one with the highest ``sequence``
(promotion order); a remaining tie is broken by ``revision_id`` (stable). This is
the "latest promoted compatible revision" rule (mission §26/§30).

Outcomes are structured, never fatal for the benign cases:

* no compatible revision          -> ``CALIBRATION_ONLY`` / ``NO_PROFILE``
* compatible but none promoted    -> ``CALIBRATION_ONLY`` / ``UNQUALIFIED_PROFILE``
* compatible promoted exists      -> ``SELECTED`` (the deterministic winner)

A corrupt / invalid / incompatible base is surfaced by the store layer as a
typed error and mapped (by the facade) to ``BASE_ERROR`` with provenance — never
a silent application and never a silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .identity import SensorIdentity, identity_matches
from .revision import Revision
from .vocabulary import (
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_SELECTED,
    REASON_NO_PROFILE,
    REASON_UNQUALIFIED_PROFILE,
    REVISION_STATE_PROMOTED,
)


@dataclass(frozen=True)
class ProvenanceNote:
    """A structured provenance note (code + detail) for a resolution."""

    code: str
    detail: str


@dataclass(frozen=True)
class BpmResolution:
    """The typed outcome of resolving a sensor/run identity against a base.

    ``outcome`` is ``SELECTED`` | ``CALIBRATION_ONLY`` | ``BASE_ERROR``;
    ``reason_code`` is the stable reason (empty for ``SELECTED``); ``revision``
    is present only when ``SELECTED``; ``provenance`` records what was inspected
    and why.
    """

    outcome: str
    reason_code: str
    revision: Optional[Revision] = None
    provenance: tuple[ProvenanceNote, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if self.outcome not in (OUTCOME_SELECTED, OUTCOME_CALIBRATION_ONLY, OUTCOME_BASE_ERROR):
            raise ValueError(f"BpmResolution.outcome invalid: {self.outcome!r}")
        if self.outcome == OUTCOME_SELECTED and self.revision is None:
            raise ValueError("BpmResolution outcome SELECTED requires a revision")

    @property
    def is_selected(self) -> bool:
        return self.outcome == OUTCOME_SELECTED

    @property
    def is_calibration_only(self) -> bool:
        return self.outcome == OUTCOME_CALIBRATION_ONLY

    @property
    def is_base_error(self) -> bool:
        return self.outcome == OUTCOME_BASE_ERROR


def select_revision(identity: SensorIdentity, revisions) -> BpmResolution:
    """Deterministic pure selection among an in-memory tuple of revisions.

    ``revisions`` must already be integrity-verified (the store verifies on
    load). Selection is the documented "latest promoted compatible" rule.
    """
    revs = tuple(revisions)
    compatible = [r for r in revs if identity_matches(identity, r.sensor_identity)]
    if not compatible:
        return BpmResolution(
            outcome=OUTCOME_CALIBRATION_ONLY,
            reason_code=REASON_NO_PROFILE,
            provenance=(
                ProvenanceNote(
                    "no-compatible-revision",
                    f"no revision binds a compatible sensor identity (checked {len(revs)} revision(s))",
                ),
            ),
        )

    promoted = [r for r in compatible if r.state == REVISION_STATE_PROMOTED]
    if not promoted:
        states = sorted({r.state for r in compatible})
        return BpmResolution(
            outcome=OUTCOME_CALIBRATION_ONLY,
            reason_code=REASON_UNQUALIFIED_PROFILE,
            provenance=(
                ProvenanceNote(
                    "unqualified-profile",
                    f"{len(compatible)} compatible revision(s) exist but none promoted (states: {', '.join(states)})",
                ),
            ),
        )

    # Deterministic: highest sequence (promotion order), then stable id tie-break.
    chosen = max(promoted, key=lambda r: (r.sequence, r.revision_id))
    return BpmResolution(
        outcome=OUTCOME_SELECTED,
        reason_code="",
        revision=chosen,
        provenance=(
            ProvenanceNote(
                "selected",
                f"selected promoted revision {chosen.revision_id} (sequence {chosen.sequence}, "
                f"of {len(promoted)} promoted compatible revision(s))",
            ),
        ),
    )


__all__ = [
    "BpmResolution",
    "ProvenanceNote",
    "select_revision",
]
