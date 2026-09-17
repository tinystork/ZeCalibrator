"""Public stable error hierarchy for ``zecalibrator.api.v1``.

The public API distinguishes *parameter/request* failures (which raise typed
stable exceptions) from *anticipated operational* failures (which are returned
as structured result envelopes).

The G3/G4 scientific error hierarchy is re-exported unchanged. The operational
errors below are **coherent aliases** of the exact classes the re-exported
source/handle/library boundaries raise, so ``except api.SourceError`` /
``except api.LibraryClosedError`` / ``except api.UnsupportedSchemaError`` catch
the real failure at those boundaries (no public/private identity trap).
"""

from __future__ import annotations

from zecalibrator.application.library import LibraryClosedError
from zecalibrator.core.errors import (
    DecodeError,
    GeometryMismatchError,
    InvalidRequestError,
    PrecisionRefusalError,
    ZeCalibratorError,
)
from zecalibrator.io.library_index import LibraryIndexError, UnsupportedSchemaError
from zecalibrator.io.master_source import SourceError

# Public library-failure categories are coherent aliases of the G4 library-index
# exception classes (UnsupportedSchemaError subclasses LibraryIndexError).
LibraryError = LibraryIndexError


__all__ = [
    "DecodeError",
    "GeometryMismatchError",
    "InvalidRequestError",
    "LibraryClosedError",
    "LibraryError",
    "PrecisionRefusalError",
    "SourceError",
    "UnsupportedSchemaError",
    "ZeCalibratorError",
]
