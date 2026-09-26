"""Bad Pixel Database store: a root directory of immutable revisions.

One base = one root directory. Its internal layout is ZeCalibrator's property;
the user chooses only the root (``bad_pixel_database_root``). Layout:

```
<root>/
  manifest.json            # schema_version + ordered revision references
  revisions/<id>.json      # one immutable, content-addressed revision each
```

* **Immutable** — a revision file is written exactly once (no-clobber) and its
  integrity digest is recomputed on every load; a mismatch raises
  :class:`BpmBaseCorrupted`.
* **Update = new revision** — ``promote`` writes a *new* promoted revision; the
  old one remains.
* **Typed failures** — a corrupt / invalid / incompatible base raises a typed
  :class:`BpmError` (surfaced by ``resolve_bad_pixel_database`` as a
  ``BASE_ERROR`` resolution with provenance), never a silent application or a
  silent fallback.
* **Headless** — no Qt, no GUI, no ``research/*``, no ``api/v1`` import.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import BpmBaseCorrupted, BpmBaseIncompatible, BpmBaseInvalid, BpmError, BpmWriteError
from .identity import SensorIdentity, sensor_identity_from_dict
from .lookup import BpmResolution, ProvenanceNote, select_revision
from .revision import Revision, promote_revision, site_record_from_dict
from .vocabulary import (
    BPM_SCHEMA_VERSION,
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
    REASON_BASE_CORRUPTED,
    REASON_BASE_INCOMPATIBLE,
    REASON_BASE_INVALID,
    REASON_NO_BASE,
)

MANIFEST_FILENAME = "manifest.json"
REVISIONS_DIRNAME = "revisions"

STATE_OPENED = "OPENED"
STATE_MISSING = "MISSING"
STATE_CORRUPTED = "CORRUPTED"
STATE_INVALID = "INVALID"
STATE_INCOMPATIBLE = "INCOMPATIBLE"

_MANIFEST_KEYS = frozenset({"schema_version", "revisions"})
_REVISION_REF_KEYS = frozenset({"revision_id", "state", "sequence", "path"})


@dataclass(frozen=True)
class BpmLoadResult:
    """Structured load outcome: a valid opened base, or a typed load state."""

    state: str  # OPENED | MISSING | CORRUPTED | INVALID | INCOMPATIBLE
    database: Optional["BadPixelDatabase"] = None
    reason_code: str = ""
    provenance: tuple[ProvenanceNote, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", tuple(self.provenance))


def _manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_FILENAME


def _revisions_dir(root: Path) -> Path:
    return Path(root) / REVISIONS_DIRNAME


def _atomic_write(path: Path, data: str) -> None:
    parent = Path(path).parent
    tmp = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".zecalibrator-bpm-", suffix=".json.tmp", dir=str(parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, Path(path))
    except OSError as exc:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise BpmWriteError(f"could not write {path}: {exc}", path=str(path)) from exc


def _read_json(path: Path):
    """Return the parsed JSON object at ``path``, or raise a typed load error."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise BpmBaseInvalid(f"missing file {path}", path=str(path)) from exc
    except OSError as exc:
        raise BpmBaseInvalid(f"unreadable file {path}: {exc}", path=str(path)) from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BpmBaseInvalid(f"undecodable file {path}", path=str(path)) from exc
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BpmBaseInvalid(f"malformed JSON in {path}: {exc}", path=str(path)) from exc
    return obj


def _revision_from_file(obj: dict, path: str, *, revision_id: Optional[str], sequence: Optional[int]) -> Revision:
    unknown = set(obj.keys()) - {
        "schema_version", "revision_id", "state", "sequence", "sensor_identity", "sites", "integrity_digest", "detector_k",
    }
    if unknown:
        raise BpmBaseInvalid(f"unknown key(s) in revision file {path}: {sorted(unknown)}", path=path)
    if obj.get("schema_version") != BPM_SCHEMA_VERSION:
        raise BpmBaseIncompatible(
            f"revision file {path} has unsupported schema {obj.get('schema_version')!r}",
            schema_version=str(obj.get("schema_version")),
        )
    try:
        sites = tuple(site_record_from_dict(s) for s in obj["sites"])
        detector_k = obj.get("detector_k")
        if detector_k is not None and (isinstance(detector_k, bool) or not isinstance(detector_k, (int, float))):
            raise BpmBaseInvalid(f"revision file {path} detector_k must be a number or absent", path=path)
        revision = Revision(
            revision_id=obj["revision_id"],
            state=obj["state"],
            sensor_identity=sensor_identity_from_dict(obj["sensor_identity"]),
            sites=sites,
            integrity_digest=obj["integrity_digest"],
            schema_version=obj["schema_version"],
            sequence=obj.get("sequence", 0),
            detector_k=detector_k,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BpmBaseInvalid(f"malformed revision file {path}: {exc}", path=path) from exc

    if revision_id is not None and revision.revision_id != revision_id:
        raise BpmBaseCorrupted(
            f"revision index mismatch at {path}: manifest ids {revision_id!r}, file ids {revision.revision_id!r}",
            revision_id=revision_id,
            path=path,
        )
    if sequence is not None and revision.sequence != sequence:
        raise BpmBaseCorrupted(
            f"revision sequence mismatch at {path}: manifest {sequence}, file {revision.sequence}",
            revision_id=revision.revision_id,
            path=path,
        )
    revision.verify()
    return revision


def _load_revisions(root: Path) -> tuple[Revision, ...]:
    manifest = _read_json(_manifest_path(root))
    if not isinstance(manifest, dict):
        raise BpmBaseInvalid(f"manifest {_manifest_path(root)} must be a JSON object", path=str(_manifest_path(root)))
    unknown = set(manifest.keys()) - _MANIFEST_KEYS
    if unknown:
        raise BpmBaseInvalid(f"unknown key(s) in manifest: {sorted(unknown)}", path=str(_manifest_path(root)))
    if manifest.get("schema_version") != BPM_SCHEMA_VERSION:
        raise BpmBaseIncompatible(
            f"unsupported BPM schema {manifest.get('schema_version')!r}",
            schema_version=str(manifest.get("schema_version")),
        )
    refs = manifest.get("revisions")
    if refs is None:
        refs = []
    if not isinstance(refs, list):
        raise BpmBaseInvalid("manifest revisions must be a list", path=str(_manifest_path(root)))

    revisions: list[Revision] = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise BpmBaseInvalid("manifest revision refs must be objects", path=str(_manifest_path(root)))
        r_unknown = set(ref.keys()) - _REVISION_REF_KEYS
        if r_unknown:
            raise BpmBaseInvalid(f"unknown key(s) in manifest revision ref: {sorted(r_unknown)}", path=str(_manifest_path(root)))
        rev_path = Path(root) / str(ref["path"])
        rev_id = ref.get("revision_id")
        seq = ref.get("sequence")
        obj = _read_json(rev_path)
        if not isinstance(obj, dict):
            raise BpmBaseInvalid(f"revision file {rev_path} must be a JSON object", path=str(rev_path))
        revisions.append(_revision_from_file(obj, str(rev_path), revision_id=rev_id, sequence=seq))
    return tuple(revisions)


class BadPixelDatabase:
    """An opened Bad Pixel Database over a root directory (read + append)."""

    def __init__(self, root: Path):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def revisions(self) -> tuple[Revision, ...]:
        """Load and integrity-verify every revision (typed error on corruption)."""
        return _load_revisions(self._root)

    def resolve(self, identity: SensorIdentity) -> BpmResolution:
        """Select a revision for ``identity`` (deterministic; never fatal)."""
        return select_revision(identity, self.revisions())

    def add_revision(self, revision: Revision) -> Revision:
        """Persist a new immutable revision (no-clobber). Returns ``revision``.

        A revision whose ``revision_id`` already exists is refused (immutability:
        never overwrite), and a revision whose integrity digest does not match is
        refused up front.
        """
        revision.verify()
        rev_dir = _revisions_dir(self._root)
        rev_dir.mkdir(parents=True, exist_ok=True)
        rev_path = rev_dir / f"{revision.revision_id}.json"
        if rev_path.exists():
            raise BpmWriteError(
                f"revision {revision.revision_id!r} already exists (immutable; never overwritten)",
                path=str(rev_path),
            )
        _atomic_write(rev_path, json.dumps(revision.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        self._append_manifest(revision, rel_path=Path(REVISIONS_DIRNAME) / f"{revision.revision_id}.json")
        return revision

    def promote(self, revision: Revision) -> Revision:
        """Promote ``revision``: write a new promoted revision (old remains).

        The new revision is assigned the next ``sequence`` (max existing + 1).
        """
        existing = self.revisions()
        next_seq = max((r.sequence for r in existing), default=0) + 1
        promoted = promote_revision(revision, sequence=next_seq)
        self.add_revision(promoted)
        return promoted

    def _append_manifest(self, revision: Revision, *, rel_path) -> None:
        manifest_path = _manifest_path(self._root)
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if not isinstance(manifest, dict):
                raise BpmBaseInvalid(f"manifest {manifest_path} must be a JSON object", path=str(manifest_path))
            if manifest.get("schema_version") != BPM_SCHEMA_VERSION:
                raise BpmBaseIncompatible(
                    f"unsupported BPM schema {manifest.get('schema_version')!r}",
                    schema_version=str(manifest.get("schema_version")),
                )
            refs = manifest.get("revisions")
            if refs is None:
                refs = []
            if not isinstance(refs, list):
                raise BpmBaseInvalid("manifest revisions must be a list", path=str(manifest_path))
            refs = list(refs)
        else:
            manifest = {"schema_version": BPM_SCHEMA_VERSION, "revisions": []}
            refs = []
        if any(r.get("revision_id") == revision.revision_id for r in refs):
            return
        refs.append({
            "revision_id": revision.revision_id,
            "state": revision.state,
            "sequence": revision.sequence,
            "path": str(rel_path),
        })
        manifest["revisions"] = refs
        _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def create_bad_pixel_database(root: Path) -> BadPixelDatabase:
    """Create an empty Bad Pixel Database at ``root`` (fails if already present)."""
    root = Path(root)
    manifest_path = _manifest_path(root)
    if manifest_path.exists():
        raise BpmWriteError(f"a Bad Pixel Database already exists at {root}", path=str(manifest_path))
    _revisions_dir(root).mkdir(parents=True, exist_ok=True)
    doc = {"schema_version": BPM_SCHEMA_VERSION, "revisions": []}
    _atomic_write(manifest_path, json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return BadPixelDatabase(root)


def load_bad_pixel_database(root: Path) -> BpmLoadResult:
    """Load a base, returning a structured result (never raising for missing).

    * missing root / no manifest  -> ``MISSING`` (no BPM — benign)
    * corrupt / invalid / incompatible -> typed load states with provenance
    """
    root = Path(root)
    if not root.exists():
        return BpmLoadResult(state=STATE_MISSING, reason_code=REASON_NO_BASE)
    if not root.is_dir():
        return BpmLoadResult(
            state=STATE_INVALID, reason_code=REASON_BASE_INVALID,
            provenance=(ProvenanceNote("not-a-directory", f"{root} is not a directory"),),
        )
    manifest_path = _manifest_path(root)
    if not manifest_path.exists():
        # An existing directory without a manifest is "not yet a base" — benign.
        return BpmLoadResult(state=STATE_MISSING, reason_code=REASON_NO_BASE)
    try:
        revisions = _load_revisions(root)
    except BpmBaseCorrupted as exc:
        return BpmLoadResult(
            state=STATE_CORRUPTED, reason_code=REASON_BASE_CORRUPTED,
            provenance=(ProvenanceNote("revision-corrupted", exc.message),),
        )
    except BpmBaseIncompatible as exc:
        return BpmLoadResult(
            state=STATE_INCOMPATIBLE, reason_code=REASON_BASE_INCOMPATIBLE,
            provenance=(ProvenanceNote("incompatible-schema", exc.message),),
        )
    except BpmBaseInvalid as exc:
        return BpmLoadResult(
            state=STATE_INVALID, reason_code=REASON_BASE_INVALID,
            provenance=(ProvenanceNote("invalid-base", exc.message),),
        )
    return BpmLoadResult(state=STATE_OPENED, database=BadPixelDatabase(root))


def resolve_bad_pixel_database(root: Path, identity: SensorIdentity) -> BpmResolution:
    """Resolve a sensor/run identity against a base at ``root`` (headless facade).

    Missing base -> ``CALIBRATION_ONLY``/``NO_BASE``; corrupt/invalid/incompatible
    base -> ``BASE_ERROR`` (typed + provenance); otherwise the deterministic
    selection (``SELECTED`` / ``CALIBRATION_ONLY``).
    """
    load = load_bad_pixel_database(root)
    if load.state == STATE_MISSING:
        return BpmResolution(
            outcome=OUTCOME_CALIBRATION_ONLY,
            reason_code=REASON_NO_BASE,
            provenance=(ProvenanceNote("no-base", f"no Bad Pixel Database at {Path(root)}"),),
        )
    if load.state == STATE_CORRUPTED:
        return BpmResolution(outcome=OUTCOME_BASE_ERROR, reason_code=REASON_BASE_CORRUPTED, provenance=load.provenance)
    if load.state == STATE_INVALID:
        return BpmResolution(outcome=OUTCOME_BASE_ERROR, reason_code=REASON_BASE_INVALID, provenance=load.provenance)
    if load.state == STATE_INCOMPATIBLE:
        return BpmResolution(outcome=OUTCOME_BASE_ERROR, reason_code=REASON_BASE_INCOMPATIBLE, provenance=load.provenance)
    assert load.database is not None
    try:
        return load.database.resolve(identity)
    except BpmError as exc:
        return BpmResolution(
            outcome=OUTCOME_BASE_ERROR,
            reason_code=REASON_BASE_CORRUPTED,
            provenance=(ProvenanceNote("base-error", str(exc)),),
        )


__all__ = [
    "BpmLoadResult",
    "BadPixelDatabase",
    "STATE_CORRUPTED",
    "STATE_INCOMPATIBLE",
    "STATE_INVALID",
    "STATE_MISSING",
    "STATE_OPENED",
    "create_bad_pixel_database",
    "load_bad_pixel_database",
    "resolve_bad_pixel_database",
]
