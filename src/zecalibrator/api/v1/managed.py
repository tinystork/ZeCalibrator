"""Managed master ingestion ledger + derived library orchestration (P7-M3B).

This module adds an *additive* managed-ingestion capability on top of the frozen
``MasterImportSpec``/``index_library`` path. It does **not** change matching,
calibration, decoder, equation or plan semantics.

* :func:`load_managed_ledger` / :func:`save_managed_ledger` — a versioned atomic
  JSON ledger under ``StoragePaths.user_data_path`` (temp + ``os.replace``, with
  corruption/unsupported preservation mirroring the existing settings discipline).
* :func:`build_master_import_spec` — reconstructs a :class:`MasterImportSpec`
  **from the stored validated :class:`ImportDeclaration`** (R2: the declaration
  is the single canonical scientific-fact representation; evidence provenance is
  annotation only).
* :func:`managed_fingerprint` — the derived-index reuse fingerprint over content
  identities + roles + evidence versions + dq states + schema.
* :func:`build_managed_library` — builds (or reuses) the derived SQLite index
  under ``StoragePaths.user_cache_path``; the fingerprint is the index revision
  label, so reuse is validated lazily at open/build time (no watcher).

The SQLite index remains a derived, rebuildable materialization; the ledger is
the authoritative evidence store.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from zecalibrator.core.digests import canonical_json, sha256_hex

from . import _io
from .batch import index_library
from .errors import InvalidRequestError
from .library import open_library
from .models import (
    DQ_STATES,
    EvidenceFact,
    ImportDeclaration,
    LibrarySpec,
    MANAGED_LEDGER_SCHEMA,
    ManagedMasterRecord,
    MasterImportSpec,
)

MANAGED_LEDGER_FILENAME = "managed_masters.json"

# Load states (caller decides whether a save is allowed over existing bytes).
STATE_OK = "ok"
STATE_MISSING = "missing"
STATE_MALFORMED = "malformed"
STATE_UNSUPPORTED = "unsupported"


def managed_ledger_path(data_dir) -> Path:
    """Return the managed ledger JSON path under the injected data root."""
    return Path(data_dir) / MANAGED_LEDGER_FILENAME


def _to_jsonable(value):
    """Render a value into strict-JSON-safe Python (never reassemble science)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "to_mapping"):
        return _to_jsonable(value.to_mapping())
    import dataclasses

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return str(value)


def _declaration_to_dict(decl: ImportDeclaration) -> dict:
    import dataclasses

    d = {}
    for f in dataclasses.fields(ImportDeclaration):
        v = getattr(decl, f.name)
        if isinstance(v, tuple):
            v = list(v)
        d[f.name] = v
    return d


def _record_to_dict(record: ManagedMasterRecord) -> dict:
    return {
        "schema_version": record.schema_version,
        "role": record.role,
        "content_sha256": record.content_sha256,
        "size_bytes": record.size_bytes,
        "hdu": record.hdu,
        "bias_state": record.bias_state,
        "flat_form": record.flat_form,
        "declaration": _declaration_to_dict(record.declaration),
        "evidence": {fld: dict(fact.to_dict()) for fld, fact in record.evidence.items()},
        "dq_state": record.dq_state,
        "mask_path": record.mask_path,
        "last_seen_path": record.last_seen_path,
    }


@dataclass(frozen=True)
class ManagedLedgerLoad:
    """Result of loading the managed ledger: records + preservation state."""

    records: tuple
    state: str  # ok | missing | malformed | unsupported

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class ManagedLibraryResult:
    """Structured ``build_managed_library`` outcome envelope."""

    status: str  # "REUSED" | "COMPLETED" | "CANCELLED" | "FAILED"
    revision: Optional[str] = None
    candidate_count: int = 0
    diagnostics: tuple = ()
    reason_code: Optional[str] = None
    details: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


def load_managed_ledger(path) -> ManagedLedgerLoad:
    """Load the managed ledger, reporting state; never create directories or write.

    ``missing`` = file absent; ``malformed``/``unsupported`` preserve existing
    bytes (caller must not overwrite them).
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except FileNotFoundError:
        return ManagedLedgerLoad(records=(), state=STATE_MISSING)
    except OSError:
        return ManagedLedgerLoad(records=(), state=STATE_MALFORMED)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return ManagedLedgerLoad(records=(), state=STATE_MALFORMED)
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return ManagedLedgerLoad(records=(), state=STATE_MALFORMED)
    if not isinstance(obj, dict):
        return ManagedLedgerLoad(records=(), state=STATE_MALFORMED)
    if obj.get("schema_version") != MANAGED_LEDGER_SCHEMA:
        return ManagedLedgerLoad(records=(), state=STATE_UNSUPPORTED)
    raw_records = obj.get("records")
    if not isinstance(raw_records, list):
        return ManagedLedgerLoad(records=(), state=STATE_MALFORMED)
    try:
        records = tuple(ManagedMasterRecord.from_dict(r) for r in raw_records)
    except (TypeError, ValueError, KeyError, InvalidRequestError):
        return ManagedLedgerLoad(records=(), state=STATE_MALFORMED)
    return ManagedLedgerLoad(records=records, state=STATE_OK)


def save_managed_ledger(path, records: Iterable[ManagedMasterRecord]) -> None:
    """Atomically write the managed ledger (temp + ``os.replace``).

    The caller must not call this over an unsupported/malformed existing file
    (see :class:`ManagedLedgerLoad`); doing so would replace data it did not
    understand.
    """
    for r in records:
        if not isinstance(r, ManagedMasterRecord):
            raise InvalidRequestError("every record must be a ManagedMasterRecord")
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": MANAGED_LEDGER_SCHEMA,
        "records": [_record_to_dict(r) for r in records],
    }
    data = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fd, tmp = tempfile.mkstemp(
        prefix=".zecalibrator-managed-", suffix=".json.tmp", dir=str(parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, Path(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def content_identity(path, hdu=0) -> tuple:
    """Return the content identity ``(content_sha256, size_bytes)`` of a master
    FITS file via the public filesystem source (no pixel array loading)."""
    from zecalibrator.io.master_source import FilesystemSource
    from zecalibrator.core.plans import FitsFileLocator

    ident = FilesystemSource().image_identity(FitsFileLocator(path=str(path), hdu=hdu))
    return ident.content_sha256, ident.size_bytes


def build_master_import_spec(record: ManagedMasterRecord) -> MasterImportSpec:
    """Reconstruct a :class:`MasterImportSpec` from the stored validated
    :class:`ImportDeclaration` (R2: canonical facts only; evidence is annotation)."""
    if not isinstance(record, ManagedMasterRecord):
        raise InvalidRequestError("record must be a ManagedMasterRecord")
    if record.last_seen_path is None or not str(record.last_seen_path).strip():
        raise InvalidRequestError("ManagedMasterRecord.last_seen_path must be set to build a spec")
    return MasterImportSpec(
        path=str(record.last_seen_path),
        master_type=record.role,
        declaration=record.declaration,
        hdu=record.hdu,
        mask_path=record.mask_path,
        bias_state=record.bias_state,
        flat_form=record.flat_form,
        dq_state=record.dq_state,
    )


def managed_fingerprint(records: Iterable[ManagedMasterRecord]) -> str:
    """Compute the derived-index reuse fingerprint over content identities +
    roles + evidence versions + dq states + schema (add/remove/content/evidence/
    dq/source-set changes all change the fingerprint)."""
    items = []
    for r in sorted(records, key=lambda x: (x.role, x.content_sha256, x.size_bytes)):
        items.append({
            "role": r.role,
            "content_sha256": r.content_sha256,
            "size_bytes": r.size_bytes,
            "dq_state": r.dq_state,
            "mask_path": r.mask_path,
            "declaration": _declaration_to_dict(r.declaration),
            "evidence": {fld: fact.to_dict() for fld, fact in sorted(r.evidence.items())},
            "schema_version": r.schema_version,
        })
    return sha256_hex(canonical_json(items).encode("utf-8"))


def build_managed_library(
    spec: LibrarySpec,
    records: Iterable[ManagedMasterRecord],
    *,
    cancel=None,
    progress=None,
) -> ManagedLibraryResult:
    """Build (or reuse) the derived managed SQLite index.

    The fingerprint is the index revision label. When the existing index already
    carries that revision, the index is reused (no rebuild); otherwise it is
    rebuilt via :func:`index_library`. Validation is lazy (at build/open time);
    no filesystem watcher.
    """
    if not isinstance(spec, LibrarySpec):
        raise InvalidRequestError("spec must be a LibrarySpec")
    record_list = list(records)
    for r in record_list:
        if not isinstance(r, ManagedMasterRecord):
            raise InvalidRequestError("every record must be a ManagedMasterRecord")
    fp = managed_fingerprint(record_list)

    reuse = False
    count = 0
    opened = open_library(spec, cancel=cancel, progress=progress)
    if opened.operation_status == "OPENED":
        handle = opened.handle
        try:
            if handle.snapshot.revision == fp:
                reuse = True
                count = sum(len(cs) for cs in handle.snapshot.candidates.values())
        finally:
            handle.close()

    if reuse:
        return ManagedLibraryResult(
            status="REUSED", revision=fp, candidate_count=count, diagnostics=(),
        )

    specs = [build_master_import_spec(r) for r in record_list]
    result = index_library(spec, specs, revision=fp, cancel=cancel, progress=progress)
    return ManagedLibraryResult(
        status=result.operation_status,
        revision=result.revision if result.operation_status == "COMPLETED" else None,
        candidate_count=result.candidate_count,
        diagnostics=result.diagnostics,
        reason_code=result.reason_code,
        details=result.details,
    )


__all__ = [
    "MANAGED_LEDGER_FILENAME",
    "STATE_MALFORMED",
    "STATE_MISSING",
    "STATE_OK",
    "STATE_UNSUPPORTED",
    "ManagedLedgerLoad",
    "ManagedLibraryResult",
    "build_managed_library",
    "build_master_import_spec",
    "content_identity",
    "load_managed_ledger",
    "managed_fingerprint",
    "managed_ledger_path",
    "save_managed_ledger",
]
