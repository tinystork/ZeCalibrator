"""Public library opening: ``open_library``.

``open_library`` opens a versioned SQLite library index behind the library
protocol and returns a metadata-only :class:`LibraryHandle`. The SQLite import
happens only inside this function, never at ``zecalibrator.api.v1`` import time;
no sqlite3 connection/PRAGMA is ever exposed to consumers.
"""

from __future__ import annotations

import sqlite3

from zecalibrator.application.cancellation import CancellationToken, OperationCancelled
from zecalibrator.application.library import LibraryHandle

from . import _io
from .errors import InvalidRequestError
from .models import LibrarySpec, OpenLibraryResult

OPERATION_ID = "zecalibrator-open-library"


def open_library(
    spec: LibrarySpec,
    *,
    cancel=None,
    progress=None,
) -> OpenLibraryResult:
    """Open the versioned library index described by ``spec``.

    A bad spec raises :class:`InvalidRequestError`. Anticipated operational
    failures (missing index, unsupported schema, non-index path, sqlite/OS
    errors, cancellation) return a structured :class:`OpenLibraryResult` with no
    handle. An empty initialized index is a valid OPENED state (zero entries).
    """
    if not isinstance(spec, LibrarySpec):
        raise InvalidRequestError("spec must be a LibrarySpec")

    token = cancel or CancellationToken()
    obs = _io.normalize_progress(progress, OPERATION_ID)

    if token.is_cancelled():
        return OpenLibraryResult(operation_status="CANCELLED", reason_code="CANCELLED")

    _io.emit_progress(obs, OPERATION_ID, "start", 0, 2)
    try:
        from zecalibrator.io.library_index import (
            LibraryIndex,
            LibraryIndexError,
            UnsupportedSchemaError,
        )

        token.raise_if_cancelled()
        index = LibraryIndex(spec.resolved_index_path).open(initialize=False)
        try:
            snapshot = index.load_snapshot()
        finally:
            index.close()
    except UnsupportedSchemaError as exc:
        return OpenLibraryResult(
            operation_status="FAILED", reason_code="UNSUPPORTED_SCHEMA", details=str(exc)
        )
    except LibraryIndexError as exc:
        return OpenLibraryResult(
            operation_status="FAILED", reason_code="LIBRARY_ERROR", details=str(exc)
        )
    except (sqlite3.Error, OSError) as exc:
        return OpenLibraryResult(
            operation_status="FAILED", reason_code="LIBRARY_ERROR", details=str(exc)
        )
    except OperationCancelled:
        return OpenLibraryResult(operation_status="CANCELLED", reason_code="CANCELLED")

    handle = LibraryHandle(snapshot)
    _io.emit_progress(obs, OPERATION_ID, "complete", 2, 2)
    return OpenLibraryResult(operation_status="OPENED", handle=handle)


__all__ = ["open_library"]
