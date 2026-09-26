"""Immutable BPM revisions and sensor-site records.

A revision is the unit of update: it is **immutable** and content-addressed —
its ``revision_id`` defaults to its integrity digest, and its ``integrity_digest``
is the SHA-256 of its canonical content (reusing
``zecalibrator.core.digests.canonical_json`` / ``sha256_hex``). An update
(candidate -> qualified -> promoted) is a **new** revision; the old one remains.

A revision holds :class:`SiteRecord` entries, each of which preserves the
knowledge/action distinction (SCIENCE §13.3, §13.7): a site may be ``KNOWN``
while ``NO_ACTION_REQUIRED`` or ``ABSTAIN_*``. Only sites whose ``action_state``
is ``ELIGIBLE_FOR_TARGETED_RECONSTRUCTION`` may reach reconstruction — and that
application happens in P4-A2, never here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from zecalibrator.core.digests import canonical_json, sha256_hex

from .errors import BpmBaseCorrupted
from .identity import SensorIdentity
from .settings import DEFAULT_DETECTOR_K, resolved_detector_k
from .vocabulary import (
    ACTION_STATE_LADDER,
    BPM_SCHEMA_VERSION,
    KNOWLEDGE_STATES,
    REVISION_STATES,
    REVISION_STATE_PROMOTED,
    is_action_eligible,
)


@dataclass(frozen=True)
class SiteRecord:
    """One known sensor site — NOT a bare ``(x, y)``.

    ``position`` is the sensor coordinate ``(y, x)`` (array row, column) in the
    revision's bound geometry. ``action_state`` is a member of the
    :data:`~zecalibrator.bpm.vocabulary.ACTION_STATE_LADDER`; ``knowledge_state``
    records whether the site is merely known or qualified. A site that is
    ``KNOWN`` but ``NO_ACTION_REQUIRED`` (or ``ABSTAIN_*``) is preserved exactly
    as such — the BPM never collapses that distinction.
    """

    position: tuple[int, int]
    action_state: str
    knowledge_state: str = "KNOWN"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.position, (tuple, list))
            or len(self.position) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.position)
        ):
            raise ValueError(f"SiteRecord.position must be a pair of non-negative ints, got {self.position!r}")
        object.__setattr__(self, "position", (int(self.position[0]), int(self.position[1])))
        if self.action_state not in ACTION_STATE_LADDER:
            raise ValueError(
                f"SiteRecord.action_state must be one of {ACTION_STATE_LADDER}, got {self.action_state!r}"
            )
        if self.knowledge_state not in KNOWLEDGE_STATES:
            raise ValueError(
                f"SiteRecord.knowledge_state must be one of {KNOWLEDGE_STATES}, got {self.knowledge_state!r}"
            )

    @property
    def is_action_eligible(self) -> bool:
        """Whether this site may reach reconstruction (SCIENCE §13.2)."""
        return is_action_eligible(self.action_state)

    def to_dict(self) -> dict:
        return {
            "position": list(self.position),
            "action_state": self.action_state,
            "knowledge_state": self.knowledge_state,
        }


def site_record_from_dict(d: dict) -> SiteRecord:
    """Reconstruct a :class:`SiteRecord` from its canonical serialization."""
    return SiteRecord(
        position=(int(d["position"][0]), int(d["position"][1])),
        action_state=d["action_state"],
        knowledge_state=d.get("knowledge_state", "KNOWN"),
    )


def action_eligible_sites(revision: "Revision") -> tuple[SiteRecord, ...]:
    """Return the sites of ``revision`` that may reach reconstruction.

    This is the ONLY projection P4-A2 will act on; it never emits a site whose
    action state is ``NO_ACTION_REQUIRED`` or an ``ABSTAIN_*``.
    """
    return tuple(s for s in revision.sites if s.is_action_eligible)


@dataclass(frozen=True)
class Revision:
    """An immutable, content-addressed BPM revision.

    ``sequence`` is store-ordering metadata (the promotion order), used only for
    the deterministic "latest promoted compatible" selection; it is **not** part
    of the integrity digest (like ``acquired_at`` in the master library).

    ``detector_k`` records the detector threshold multiplier ``K`` actually used
    to create this revision's map (provenance, part of the integrity digest). A
    revision created by P4.2 without this field carries ``detector_k=None`` and
    resolves compatibly to :data:`DEFAULT_DETECTOR_K` (30.0) via
    :func:`zecalibrator.bpm.settings.resolved_detector_k`.
    """

    revision_id: str
    state: str
    sensor_identity: SensorIdentity
    sites: tuple[SiteRecord, ...]
    integrity_digest: str
    schema_version: str = BPM_SCHEMA_VERSION
    sequence: int = 0
    detector_k: Optional[float] = None

    def __post_init__(self) -> None:
        if self.schema_version != BPM_SCHEMA_VERSION:
            raise ValueError(f"unsupported BPM schema {self.schema_version!r}")
        if self.state not in REVISION_STATES:
            raise ValueError(f"Revision.state must be one of {REVISION_STATES}, got {self.state!r}")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError(f"Revision.sequence must be a non-negative int, got {self.sequence!r}")
        if not isinstance(self.revision_id, str) or not self.revision_id:
            raise ValueError("Revision.revision_id must be a non-empty string")
        if not isinstance(self.integrity_digest, str) or len(self.integrity_digest) != 64:
            raise ValueError("Revision.integrity_digest must be a 64-char hex string")
        if self.detector_k is not None:
            if isinstance(self.detector_k, bool) or not isinstance(self.detector_k, (int, float)):
                raise ValueError(
                    f"Revision.detector_k must be a finite positive float or None, got {self.detector_k!r}"
                )
            k = float(self.detector_k)
            if not math.isfinite(k) or k <= 0.0:
                raise ValueError(f"Revision.detector_k must be a finite positive float, got {k!r}")
            object.__setattr__(self, "detector_k", k)
        object.__setattr__(self, "sites", tuple(self.sites))

    @property
    def effective_detector_k(self) -> float:
        """The detector K this revision was created with (legacy None -> 30.0)."""
        return resolved_detector_k(self)

    def verify(self) -> None:
        """Reject a tampered revision (recompute the integrity digest).

        Raises :class:`BpmBaseCorrupted` with the revision id and provenance when
        the recomputed digest differs from ``integrity_digest``.
        """
        recomputed = revision_digest(
            self.state, self.sensor_identity, self.sites, self.schema_version,
            detector_k=self.detector_k,
        )
        if recomputed != self.integrity_digest:
            raise BpmBaseCorrupted(
                f"revision {self.revision_id!r} integrity digest mismatch: "
                f"recorded {self.integrity_digest!r}, recomputed {recomputed!r}",
                revision_id=self.revision_id,
            )

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "revision_id": self.revision_id,
            "state": self.state,
            "sequence": self.sequence,
            "sensor_identity": self.sensor_identity.to_dict(),
            "sites": [s.to_dict() for s in self.sites],
            "integrity_digest": self.integrity_digest,
        }
        if self.detector_k is not None:
            d["detector_k"] = self.detector_k
        return d


def _sorted_sites(sites) -> tuple[SiteRecord, ...]:
    return tuple(sorted(sites, key=lambda s: (s.position[0], s.position[1], s.action_state, s.knowledge_state)))


def revision_digest(
    state: str,
    sensor_identity: SensorIdentity,
    sites,
    schema_version: str = BPM_SCHEMA_VERSION,
    detector_k: Optional[float] = None,
) -> str:
    """Compute the canonical integrity digest of a revision's content.

    Site order is normalized (sorted by position) so identical evidence always
    yields the identical digest regardless of insertion order — this is what
    makes "same evidence -> deterministic profile" hold. When ``detector_k`` is
    present it participates in the digest (provenance is integrity-protected); a
    legacy revision (``detector_k=None``) omits it and keeps its historical
    digest byte-identical.
    """
    obj = {
        "schema_version": schema_version,
        "state": state,
        "sensor_identity": sensor_identity.to_dict(),
        "sites": [s.to_dict() for s in _sorted_sites(sites)],
    }
    if detector_k is not None:
        obj["detector_k"] = detector_k
    return sha256_hex(canonical_json(obj).encode("utf-8"))


def make_revision(
    *,
    state: str,
    sensor_identity: SensorIdentity,
    sites,
    sequence: int = 0,
    revision_id: Optional[str] = None,
    schema_version: str = BPM_SCHEMA_VERSION,
    detector_k: Optional[float] = None,
) -> Revision:
    """Build an immutable revision with a content-addressed identity.

    ``revision_id`` defaults to the integrity digest (deterministic: identical
    evidence -> identical id; new evidence -> new id). ``sites`` are normalized
    to a deterministic order. ``detector_k`` records the detector threshold used
    to create the map (optional; ``None`` for legacy P4.2 revisions).
    """
    canonical_sites = _sorted_sites(sites)
    digest = revision_digest(state, sensor_identity, canonical_sites, schema_version, detector_k=detector_k)
    return Revision(
        revision_id=revision_id or digest,
        state=state,
        sensor_identity=sensor_identity,
        sites=canonical_sites,
        integrity_digest=digest,
        schema_version=schema_version,
        sequence=sequence,
        detector_k=detector_k,
    )


def promote_revision(revision: Revision, *, sequence: int) -> Revision:
    """Return a **new** promoted revision carrying ``revision``'s knowledge.

    The original revision remains untouched; promotion is a new immutable
    revision (new id / digest, state=``promoted``, assigned ``sequence``). The
    detector K provenance is carried over unchanged.
    """
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("promote_revision.sequence must be a non-negative int")
    return make_revision(
        state=REVISION_STATE_PROMOTED,
        sensor_identity=revision.sensor_identity,
        sites=revision.sites,
        sequence=sequence,
        detector_k=revision.detector_k,
    )


__all__ = [
    "Revision",
    "SiteRecord",
    "action_eligible_sites",
    "make_revision",
    "promote_revision",
    "revision_digest",
    "site_record_from_dict",
]
