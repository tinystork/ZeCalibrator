"""Versioned SQLite library index (ARCHITECTURE §5, PROVENANCE §2.6).

A stdlib ``sqlite3`` implementation behind the library protocol: it stores
references and descriptor snapshots (never master pixel arrays), supports one
serialized writer with short atomic transactions and safe concurrent readers
(WAL + busy timeout), and refuses unknown schema versions/migrations rather than
dropping or recreating an existing index.

The index metadata/stat is a **cache**, not a proof. Content/mask hashing happens
through the source adapter at scan/revalidation time, never from index mtime/stat.

Lifecycle guarantees (mission §6/§9):
* an empty index loads as a well-defined empty snapshot (revision ``""``);
* a requested missing revision is explicitly diagnosed, never masquerades as empty;
* ``open(initialize=False)`` on a missing path refuses **without** creating the
  file (existence is checked before connecting);
* unknown-schema refusal closes the local connection and mutates nothing;
* initialization is a single atomic ``BEGIN IMMEDIATE`` transaction.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

from zecalibrator.application.library import LibrarySnapshot
from zecalibrator.core.descriptors import DescriptorSnapshot, master_descriptor_from_dict
from zecalibrator.core.plans import Candidate, FitsFileLocator, MaskPayloadLocator

LIBRARY_INDEX_SCHEMA = "zecalibrator.library.v1"
_CREATE_STATEMENTS = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS revisions (revision TEXT PRIMARY KEY, ordinal INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS entries ("
    "revision TEXT NOT NULL, role TEXT NOT NULL, candidate_id TEXT NOT NULL, "
    "snapshot_json TEXT NOT NULL, locators_json TEXT NOT NULL, mask_locator_json TEXT, "
    "acquired_at TEXT, "
    "PRIMARY KEY (revision, role, candidate_id))",
)


class LibraryIndexError(Exception):
    """A library index open/schema/migration/publish failure."""


class UnsupportedSchemaError(LibraryIndexError):
    """An index with an unknown/unsupported schema version."""


def _json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _json_loads(text: str):
    return json.loads(text)


@dataclass(frozen=True)
class ScanResult:
    """Result of a scan/import orchestration run."""

    status: str  # "COMPLETED" | "CANCELLED" | "FAILED"
    revision: Optional[str]
    diagnostics: Tuple = ()
    candidate_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


class LibraryIndex:
    """Versioned SQLite library index behind the library protocol."""

    def __init__(self, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("LibraryIndex requires a non-empty index path")
        self._path = path
        self._conn: Optional[sqlite3.Connection] = None
        self._writer_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    def open(self, *, initialize: bool = False) -> "LibraryIndex":
        if self._conn is not None:
            return self
        exists = os.path.exists(self._path)
        if not exists and not initialize:
            # Never create the file/directories on a refused open.
            raise LibraryIndexError(f"index does not exist: {self._path}")

        parent = os.path.dirname(self._path)
        if initialize and parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

        conn = sqlite3.connect(self._path, timeout=10.0, isolation_level=None)

        if not exists:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for stmt in _CREATE_STATEMENTS:
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (LIBRARY_INDEX_SCHEMA,),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                conn.close()
                raise
        else:
            # Validate schema BEFORE mutating anything (no WAL pragma on unknown).
            try:
                self._check_schema(conn)
            except Exception:
                conn.close()
                raise

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._conn = conn
        return self

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "LibraryIndex":
        return self.open(initialize=False)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None

    def _check_schema(self, conn: sqlite3.Connection) -> None:
        try:
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.DatabaseError as exc:
            raise UnsupportedSchemaError(f"index schema unreadable: {exc}") from exc
        if row is None or row[0] != LIBRARY_INDEX_SCHEMA:
            got = row[0] if row else None
            raise UnsupportedSchemaError(f"unsupported library index schema {got!r}")

    @property
    def path(self) -> str:
        return self._path

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise LibraryIndexError("library index is not open")
        return self._conn

    # -- publish -------------------------------------------------------------
    def publish_revision(self, revision: str, entries: Mapping[str, Sequence[Candidate]]) -> None:
        """Transactionally publish a new revision (one serialized writer)."""
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision must be a non-empty string")
        conn = self._require_conn()
        with self._writer_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT COALESCE(MAX(ordinal), 0) + 1 FROM revisions").fetchone()
                ordinal = int(row[0])
                conn.execute("INSERT INTO revisions(revision, ordinal) VALUES (?, ?)", (revision, ordinal))
                for role, cands in entries.items():
                    for c in cands:
                        locators = [{"path": l.path, "hdu": l.hdu} for l in c.locators]
                        mask_loc = {"path": c.mask_locator.path} if c.mask_locator is not None else None
                        conn.execute(
                            "INSERT INTO entries(revision, role, candidate_id, snapshot_json, locators_json, mask_locator_json, acquired_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                revision,
                                role,
                                c.candidate_id,
                                _json_dumps(dict(c.descriptor_snapshot.to_dict())),
                                _json_dumps(locators),
                                _json_dumps(mask_loc) if mask_loc is not None else None,
                                getattr(c, "acquired_at", None),
                            ),
                        )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                raise

    def _latest_revision(self, conn: sqlite3.Connection) -> Optional[str]:
        row = conn.execute("SELECT revision FROM revisions ORDER BY ordinal DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def list_revisions(self) -> Sequence[str]:
        conn = self._require_conn()
        rows = conn.execute("SELECT revision FROM revisions ORDER BY ordinal DESC").fetchall()
        return tuple(r[0] for r in rows)

    # -- load ----------------------------------------------------------------
    def load_snapshot(self, revision: Optional[str] = None) -> LibrarySnapshot:
        """Reconstruct a frozen snapshot; descriptor ids are recomputed and
        verified on deserialize (tamper rejection). An empty index returns a
        well-defined empty snapshot (revision ``""``)."""
        conn = self._require_conn()
        rev = revision if revision is not None else self._latest_revision(conn)
        if rev is None:
            return LibrarySnapshot(revision="", schema_version=LIBRARY_INDEX_SCHEMA, candidates={})
        # ``acquired_at`` is additive: a revision written before this column
        # existed still loads with ``acquired_at=None``.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()}
        has_acquired = "acquired_at" in cols
        if has_acquired:
            rows = conn.execute(
                "SELECT role, candidate_id, snapshot_json, locators_json, mask_locator_json, acquired_at "
                "FROM entries WHERE revision = ? ORDER BY role, candidate_id",
                (rev,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, candidate_id, snapshot_json, locators_json, mask_locator_json "
                "FROM entries WHERE revision = ? ORDER BY role, candidate_id",
                (rev,),
            ).fetchall()
        if not rows and revision is not None:
            raise LibraryIndexError(f"requested revision not found: {revision}")
        by_role: dict[str, list[Candidate]] = {}
        for row in rows:
            role, candidate_id, snap_json, loc_json, mask_json = row[:5]
            acquired_at = row[5] if has_acquired else None
            snapshot = DescriptorSnapshot.from_dict(_json_loads(snap_json))
            locators = tuple(FitsFileLocator(path=l["path"], hdu=l["hdu"]) for l in _json_loads(loc_json))
            mask_loc = None
            if mask_json:
                m = _json_loads(mask_json)
                mask_loc = MaskPayloadLocator(path=m["path"])
            by_role.setdefault(role, []).append(
                Candidate(
                    candidate_id=candidate_id,
                    descriptor=snapshot.descriptor,
                    descriptor_snapshot=snapshot,
                    locators=locators,
                    mask_locator=mask_loc,
                    acquired_at=acquired_at,
                )
            )
        candidates = {role: tuple(cs) for role, cs in by_role.items()}
        return LibrarySnapshot(revision=rev, schema_version=LIBRARY_INDEX_SCHEMA, candidates=candidates)

    # -- scan orchestration --------------------------------------------------
    def scan_and_publish(
        self,
        paths: Iterable[str],
        *,
        build_candidate: Callable[[str], Candidate],
        source,
        revision: str,
        cancel=None,
        progress=None,
    ) -> ScanResult:
        """Scan/import orchestration over injected descriptor/source adapters.

        Hashing/verification happens **outside** the writer transaction; a
        successful completed revision is atomically published; cancellation or an
        operational failure leaves the last usable revision unchanged. Structured
        diagnostics record unreadable/invalid sources (never 'missing success').
        """
        diagnostics = []
        by_role: dict[str, list[Candidate]] = {}
        count = 0

        def emit(phase, completed, total):
            if progress is None:
                return
            try:
                progress({"phase": phase, "completed": completed, "total": total})
            except Exception:
                return

        paths = list(paths)
        total = len(paths)
        for i, path in enumerate(paths):
            if cancel is not None and cancel.is_cancelled():
                return ScanResult(status="CANCELLED", revision=None, diagnostics=tuple(diagnostics), candidate_count=count)
            emit("scan", i, total)
            try:
                cand = build_candidate(path)
            except Exception as exc:  # noqa: BLE001
                diagnostics.append(_diag(path, f"build: {exc}"))
                continue
            valid = True
            if not cand.locators:
                diagnostics.append(_diag(path, "no image locator"))
                valid = False
            for loc in cand.locators:
                try:
                    ident = source.image_identity(loc)
                except Exception as exc:  # noqa: BLE001
                    diagnostics.append(_diag(path, f"image_identity: {exc}"))
                    valid = False
                    break
                if ident.content_sha256 != cand.descriptor.content_sha256 or ident.size_bytes != cand.descriptor.size_bytes:
                    diagnostics.append(_diag(path, "content/size mismatch"))
                    valid = False
                    break
            if valid and cand.mask_locator is not None:
                try:
                    mask_id = source.mask_identity(cand.mask_locator)
                except Exception as exc:  # noqa: BLE001
                    diagnostics.append(_diag(path, f"mask_identity: {exc}"))
                    valid = False
                else:
                    if mask_id != cand.descriptor.mask_identity:
                        diagnostics.append(_diag(path, "mask mismatch"))
                        valid = False
            if valid:
                by_role.setdefault(cand.role, []).append(cand)
                count += 1

        if cancel is not None and cancel.is_cancelled():
            return ScanResult(status="CANCELLED", revision=None, diagnostics=tuple(diagnostics), candidate_count=count)

        try:
            self.publish_revision(revision, by_role)
        except Exception as exc:  # noqa: BLE001
            return ScanResult(status="FAILED", revision=None, diagnostics=tuple(diagnostics) + (_diag(revision, f"publish: {exc}"),), candidate_count=count)

        return ScanResult(status="COMPLETED", revision=revision, diagnostics=tuple(diagnostics), candidate_count=count)


def _diag(path: str, reason: str):
    from zecalibrator.io.master_source import ScanDiagnostic

    return ScanDiagnostic(path=path, reason=reason)


__all__ = [
    "LIBRARY_INDEX_SCHEMA",
    "LibraryIndex",
    "LibraryIndexError",
    "ScanResult",
    "UnsupportedSchemaError",
]
