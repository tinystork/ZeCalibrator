"""Deterministic master selection and ranking (G2B — core selection policy).

This module is the **single** implementation of the acquisition-date ranking
policy. It is pure (stdlib only, no I/O, no astropy) and deterministic. It is
used by both :mod:`zecalibrator.core.matching` (the explicit-request matcher)
and :mod:`zecalibrator.core.routes` (the Standard auto-route enumerator) so
there is never a second ranking implementation.

Owner policy implemented here (exact, non-negotiable):

* **compatibility** (is a master scientifically usable?) and **ranking** (which
  compatible master to prefer?) are two DISTINCT concepts. Ranking runs strictly
  **after** the compatibility filter; a newer-but-incompatible master can never
  win over a compatible older one.
* **date is a RANKING criterion only** — never a compatibility criterion and
  never a "maximum validity age".
* A role with 0 or 1 compatible candidate is **never ranked** (no behaviour
  change): 0 ⇒ ``NO_CANDIDATE``, 1 ⇒ ``SELECTED`` with rule ``single_candidate``.

Ranking rules (see ``selection_policy_version``):

* **Additive** (``dark`` / ``bias`` / ``flat_dark`` / ``bias_flat``): among
  compatible collapsed candidates, order by (1) known date before unknown date,
  (2) most recent first, (3) ``candidate_id`` ascending. ``flat_dark`` is ranked
  by the additive rule — it is never ranked by the light's civil day.
* **Flat**: ``ld = day(light)``; key = ``(class, day_distance, -timestamp,
  tie_flag, candidate_id)`` with class 0 = same civil day as the light, class 1 =
  otherwise with a known flat date, class 2 = unknown flat date (always last);
  order class asc, day_distance asc, timestamp DESC, candidate_id asc.
  ``tie_flag`` encodes the documented "later date wins on equidistant days"
  tie-break (a later civil day is always a later timestamp, so this is a
  deterministic safety net).

``day(x)`` is the calendar date of DATE-OBS **as recorded (UTC)** — an explicit,
documented choice, not an implicit assumption. Naive DATE-OBS values are
interpreted as UTC; offset values are normalized to UTC before taking the date.

**Applicability before ranking:** ranking runs only among candidates that are
*applicable* for their role. Route-satisfiability (e.g. a ``raw_response`` flat
whose ``flat_dark``/``bias_flat`` dependency is unavailable) is an **applicability
filter applied before ranking** — NOT a compatibility criterion and NOT a
ranking peer. A compatible-but-route-unsatisfiable candidate is recorded with
``ROUTE_UNSATISFIABLE`` (never ``RANKED_BELOW_WINNER``), so it can never shadow a
satisfiable peer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Mapping, Optional, Sequence, Tuple

SELECTION_POLICY_VERSION = "zecalibrator.selection.v1"

# Selection statuses.
STATUS_SELECTED = "SELECTED"
STATUS_NO_CANDIDATE = "NO_CANDIDATE"
STATUS_AMBIGUOUS_TIE = "AMBIGUOUS_TIE"

# Ranked-out peer reason code (never a compatibility code — the peer WAS
# compatible; it simply lost the ranking).
RANKED_BELOW_WINNER = "RANKED_BELOW_WINNER"

# Applicability reason code: a compatible-but-route-unsatisfiable candidate that
# was excluded by the applicability filter before ranking (never a compatibility
# code, never RANKED_BELOW_WINNER).
ROUTE_UNSATISFIABLE = "ROUTE_UNSATISFIABLE"

# Rule labels recorded in the selection audit.
RULE_NO_CANDIDATE = "no_candidate"
RULE_SINGLE_CANDIDATE = "single_candidate"
RULE_ADDITIVE = "additive_most_recent"
RULE_FLAT = "flat_same_civil_day"
RULE_MANUAL = "manual_selection"

_HIERARCH = "HIERARCH "


# ---------------------------------------------------------------------------
# DATE-OBS parsing
# ---------------------------------------------------------------------------
_DATE_ONLY = "%Y-%m-%d"
_TIME_SS = "%Y-%m-%dT%H:%M:%S"
_TIME_SF = "%Y-%m-%dT%H:%M:%S.%f"
_TIME_MM = "%Y-%m-%dT%H:%M"
_FORMATS = (_TIME_SF, _TIME_SS, _TIME_MM, _DATE_ONLY)
_OFFSET_RE = re.compile(r"([+-]\d{2}:\d{2})$")


def parse_date_obs(value) -> Optional[datetime]:
    """Tolerantly parse a FITS ``DATE-OBS`` value into a UTC ``datetime``.

    Accepts ``YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+HH:MM]``,
    ``YYYY-MM-DDTHH:MM`` and ``YYYY-MM-DD``, with surrounding quotes/whitespace
    tolerated. Naive values are interpreted as UTC; offset values are normalized
    to UTC. Unparseable / empty / ``None`` input returns ``None`` (never raises,
    never invents a date).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    if not s:
        return None

    utc = False
    offset_minutes = None
    if s.endswith("Z"):
        utc = True
        s = s[:-1]
    m = _OFFSET_RE.search(s)
    if m:
        sign = 1 if m.group(1)[0] == "+" else -1
        hh, mm = m.group(1)[1:].split(":")
        offset_minutes = sign * (int(hh) * 60 + int(mm))
        s = s[: m.start()].rstrip()

    parsed: Optional[datetime] = None
    for fmt in _FORMATS:
        try:
            parsed = datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None

    if offset_minutes is not None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(minutes=offset_minutes)))
        parsed = parsed.astimezone(timezone.utc)
    else:
        # Naive or explicit ``Z``: interpret as UTC (documented choice).
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _card_keyword(card) -> Optional[str]:
    if isinstance(card, Mapping):
        return card.get("keyword")
    return getattr(card, "keyword", None)


def _card_value(card):
    if isinstance(card, Mapping):
        return card.get("value")
    return getattr(card, "value", None)


def _norm_keyword(keyword) -> Optional[str]:
    if keyword is None:
        return None
    kw = str(keyword)
    if kw.startswith(_HIERARCH):
        kw = kw[len(_HIERARCH):]
    return kw.strip().upper()


def light_acquisition_date(light) -> Optional[datetime]:
    """Derive the light's acquisition date from ``light.evidence.original_cards``.

    Reads the ``DATE-OBS`` card (HIERARCH-insensitive); a single unambiguous
    value is returned as a UTC ``datetime``, and multiple disagreeing values
    (parsing to different instants) return ``None``. This is NOT added to any
    descriptor, ``Acquisition``/``Geometry`` or plan-digest input.
    """
    evidence = getattr(light, "evidence", None)
    if evidence is None:
        return None
    cards = getattr(evidence, "original_cards", None)
    if not cards:
        return None
    parsed: list[datetime] = []
    for card in cards:
        if _norm_keyword(_card_keyword(card)) != "DATE-OBS":
            continue
        dt = parse_date_obs(_card_value(card))
        if dt is not None:
            parsed.append(dt)
    if not parsed:
        return None
    distinct = {p for p in parsed}
    if len(distinct) == 1:
        return parsed[0]
    return None


# ---------------------------------------------------------------------------
# Frozen audit value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SelectionKey:
    """The deterministic ordering key (plus the recorded date facts) for one
    candidate under one ranking rule.

    ``order_key`` is a sortable, hashable tuple of primitives (ascending order
    is the winning order); ``date_obs`` / ``day`` are the parsed DATE-OBS facts
    (``None`` when unknown); ``description`` is a human-readable rule note.
    """

    order_key: tuple
    date_obs: Optional[datetime]
    day: Optional[date]
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_key", tuple(self.order_key))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "order_key": list(self.order_key),
            "date_obs": self.date_obs.isoformat() if self.date_obs is not None else None,
            "day": self.day.isoformat() if self.day is not None else None,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "SelectionKey":
        date_obs = parse_date_obs(d.get("date_obs")) if d.get("date_obs") is not None else None
        day = None
        if d.get("day") is not None:
            try:
                day = date.fromisoformat(d["day"])
            except ValueError:
                day = None
        return cls(
            order_key=tuple(d.get("order_key", ())),
            date_obs=date_obs,
            day=day,
            description=d.get("description", ""),
        )


@dataclass(frozen=True)
class MasterSelectionRecord:
    """The chosen master for one role, with the rule + key that chose it."""

    role: str
    chosen_candidate_id: str
    chosen_content_sha256: str
    chosen_acquired_at: Optional[str]
    rule: str
    key: SelectionKey
    policy_version: str

    def to_dict(self) -> Mapping[str, object]:
        return {
            "role": self.role,
            "chosen_candidate_id": self.chosen_candidate_id,
            "chosen_content_sha256": self.chosen_content_sha256,
            "chosen_acquired_at": self.chosen_acquired_at,
            "rule": self.rule,
            "key": dict(self.key.to_dict()),
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "MasterSelectionRecord":
        return cls(
            role=d["role"],
            chosen_candidate_id=d["chosen_candidate_id"],
            chosen_content_sha256=d["chosen_content_sha256"],
            chosen_acquired_at=d.get("chosen_acquired_at"),
            rule=d["rule"],
            key=SelectionKey.from_dict(d["key"]),
            policy_version=d.get("policy_version", SELECTION_POLICY_VERSION),
        )


@dataclass(frozen=True)
class RankedOutRecord:
    """A compatible peer that lost the ranking (recorded, never silently dropped).

    ``role`` attributes the record to the role whose ranking produced it (the
    flattened ``MatchResult.ranked_out`` is otherwise ambiguous across roles).
    ``reason_code`` is ``RANKED_BELOW_WINNER`` for a peer that lost a ranking, or
    ``ROUTE_UNSATISFIABLE`` for a compatible-but-route-unsatisfiable peer that was
    excluded by the applicability filter before ranking (never a compatibility
    code).
    """

    role: str
    candidate_id: str
    content_sha256: str
    acquired_at: Optional[str]
    reason_code: str
    key: SelectionKey

    def to_dict(self) -> Mapping[str, object]:
        return {
            "role": self.role,
            "candidate_id": self.candidate_id,
            "content_sha256": self.content_sha256,
            "acquired_at": self.acquired_at,
            "reason_code": self.reason_code,
            "key": dict(self.key.to_dict()),
        }


@dataclass(frozen=True)
class RoleSelection:
    """The selection decision for one role."""

    role: str
    status: str  # SELECTED | NO_CANDIDATE | AMBIGUOUS_TIE
    winner: Optional[MasterSelectionRecord]
    ranked_out: tuple[RankedOutRecord, ...]
    rule: str
    tie_candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranked_out", tuple(self.ranked_out))
        object.__setattr__(self, "tie_candidates", tuple(self.tie_candidates))


# ---------------------------------------------------------------------------
# Ranking key construction
# ---------------------------------------------------------------------------
def _additive_key(candidate) -> SelectionKey:
    cid = candidate.candidate_id
    acquired = getattr(candidate, "acquired_at", None)
    dt = parse_date_obs(acquired)
    if dt is None:
        return SelectionKey((1, 0, cid), None, None, "additive-unknown-date")
    return SelectionKey(
        (0, -dt.timestamp(), cid), dt, dt.date(), "additive-most-recent"
    )


def _later_date_tie_flag(flat_day: Optional[date], light_day: Optional[date]) -> int:
    if flat_day is None or light_day is None or flat_day == light_day:
        return 0
    return -1 if flat_day > light_day else 1


def _flat_key(candidate, light_day: Optional[date]) -> SelectionKey:
    cid = candidate.candidate_id
    acquired = getattr(candidate, "acquired_at", None)
    dt = parse_date_obs(acquired)
    if dt is None:
        return SelectionKey((2, 0, 0, 0, cid), None, None, "flat-unknown-date")
    flat_day = dt.date()
    ts = -dt.timestamp()
    if light_day is not None and flat_day == light_day:
        return SelectionKey((0, 0, ts, 0, cid), dt, flat_day, "flat-same-civil-day")
    if light_day is not None:
        day_distance = abs((flat_day - light_day).days)
    else:
        day_distance = 0
    tie_flag = _later_date_tie_flag(flat_day, light_day)
    return SelectionKey(
        (1, day_distance, ts, tie_flag, cid), dt, flat_day,
        f"flat-known-date-day_distance={day_distance}",
    )


def _key_for(role: str, light, candidate) -> SelectionKey:
    if role == "flat":
        ld = light_acquisition_date(light)
        return _flat_key(candidate, ld.date() if ld is not None else None)
    return _additive_key(candidate)


def _rule_name(role: str) -> str:
    return RULE_FLAT if role == "flat" else RULE_ADDITIVE


def _rank(role: str, light, candidates: Sequence) -> list:
    """Return ``(candidate, key)`` pairs sorted ascending by ``order_key``."""
    keyed = [(c, _key_for(role, light, c)) for c in candidates]
    keyed.sort(key=lambda item: item[1].order_key)
    return keyed


def _winner_record(role: str, candidate, key: SelectionKey, rule: str) -> MasterSelectionRecord:
    return MasterSelectionRecord(
        role=role,
        chosen_candidate_id=candidate.candidate_id,
        chosen_content_sha256=candidate.descriptor.content_sha256,
        chosen_acquired_at=getattr(candidate, "acquired_at", None),
        rule=rule,
        key=key,
        policy_version=SELECTION_POLICY_VERSION,
    )


def _ranked_out(role: str, light, candidates: Sequence) -> tuple[RankedOutRecord, ...]:
    out = []
    for c in candidates:
        out.append(
            RankedOutRecord(
                role=role,
                candidate_id=c.candidate_id,
                content_sha256=c.descriptor.content_sha256,
                acquired_at=getattr(c, "acquired_at", None),
                reason_code=RANKED_BELOW_WINNER,
                key=_key_for(role, light, c),
            )
        )
    return tuple(out)


def selection_key(role: str, light, candidate) -> SelectionKey:
    """Return the deterministic :class:`SelectionKey` for ``candidate`` under ``role``.

    Public so callers (e.g. the matcher's applicability filter) can build audit
    records with the same key the ranking itself would have used.
    """
    return _key_for(role, light, candidate)


# ---------------------------------------------------------------------------
# Selection entry point
# ---------------------------------------------------------------------------
def select_role_candidates(
    role: str,
    light,
    compatible_candidates: Sequence,
    *,
    manual_choice: Optional[str] = None,
) -> RoleSelection:
    """Select the single winner for ``role`` among compatible collapsed candidates.

    ``compatible_candidates`` are already compatibility-filtered and
    duplicate-collapsed. ``manual_choice`` (a ``candidate_id``) overrides the
    ranking when it names a compatible candidate; a manual choice that is not
    among the compatible candidates is the caller's responsibility to refuse
    (incompatible/unknown), so it falls through to normal ranking here.

    Returns a :class:`RoleSelection` with ``status`` in ``SELECTED`` /
    ``NO_CANDIDATE`` / ``AMBIGUOUS_TIE``.
    """
    candidates = list(compatible_candidates)

    if not candidates:
        return RoleSelection(role, STATUS_NO_CANDIDATE, None, (), RULE_NO_CANDIDATE, ())

    if len(candidates) == 1:
        c = candidates[0]
        rec = _winner_record(role, c, _key_for(role, light, c), RULE_SINGLE_CANDIDATE)
        return RoleSelection(role, STATUS_SELECTED, rec, (), RULE_SINGLE_CANDIDATE, ())

    # Manual override (wins over ranking).
    if manual_choice is not None:
        matches = [c for c in candidates if c.candidate_id == manual_choice]
        if len(matches) == 1:
            winner = matches[0]
            peers = [c for c in candidates if c.candidate_id != winner.candidate_id]
            ranked_out = _ranked_out(role, light, peers)
            rec = _winner_record(role, winner, _key_for(role, light, winner), RULE_MANUAL)
            return RoleSelection(role, STATUS_SELECTED, rec, ranked_out, RULE_MANUAL, ())
        if len(matches) > 1:
            tie_ids = tuple(sorted({m.candidate_id for m in matches}))
            return RoleSelection(role, STATUS_AMBIGUOUS_TIE, None, (), RULE_MANUAL, tie_ids)
        # manual choice not compatible -> fall through to ranking; refusal is
        # handled by the caller.

    ranked = _rank(role, light, candidates)
    winner, winner_key = ranked[0]
    tied = [c for c in candidates if _key_for(role, light, c).order_key == winner_key.order_key]
    rule = _rule_name(role)
    if len(tied) > 1:
        tie_ids = tuple(sorted({c.candidate_id for c in tied}))
        return RoleSelection(role, STATUS_AMBIGUOUS_TIE, None, (), rule, tie_ids)

    ranked_out = _ranked_out(role, light, [c for c, _ in ranked[1:]])
    rec = _winner_record(role, winner, winner_key, rule)
    return RoleSelection(role, STATUS_SELECTED, rec, ranked_out, rule, ())


__all__ = [
    "MasterSelectionRecord",
    "RankedOutRecord",
    "RoleSelection",
    "RANKED_BELOW_WINNER",
    "ROUTE_UNSATISFIABLE",
    "RULE_ADDITIVE",
    "RULE_FLAT",
    "RULE_MANUAL",
    "RULE_NO_CANDIDATE",
    "RULE_SINGLE_CANDIDATE",
    "SELECTION_POLICY_VERSION",
    "SelectionKey",
    "STATUS_AMBIGUOUS_TIE",
    "STATUS_NO_CANDIDATE",
    "STATUS_SELECTED",
    "light_acquisition_date",
    "parse_date_obs",
    "select_role_candidates",
    "selection_key",
]
