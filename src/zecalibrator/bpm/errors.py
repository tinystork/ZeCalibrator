"""Typed Bad Pixel Database (BPM) errors.

These are *anticipated operational* failures, raised at the point of detection by
the store/verify layer and caught into a structured :class:`BpmResolution` by the
lookup facade. They are never raised for the benign fallback states (no base, no
compatible profile, unqualified profile), which are first-class
``CALIBRATION_ONLY`` outcomes, not errors.

Every corruption/incompatibility carries explicit *provenance* (which revision,
which path, which schema) so the caller can surface a non-blocking warning and
fall back to ordinary calibration *knowingly* — never silently.
"""

from __future__ import annotations


class BpmError(Exception):
    """Base class for all Bad Pixel Database errors."""


class BpmBaseCorrupted(BpmError):
    """A revision's integrity digest does not match its recomputed digest.

    This is the "revision tampered / damaged / inconsistent index" case: the
    store refuses it explicitly and reports which revision is affected, never
    silently skipping it and never applying it.
    """

    def __init__(self, message: str, *, revision_id: str | None = None, path: str | None = None):
        super().__init__(message)
        self.message = message
        self.revision_id = revision_id
        self.path = path


class BpmBaseInvalid(BpmError):
    """The base structure is malformed (undecodable JSON, unexpected shape,
    unknown keys) — the path is not a well-formed Bad Pixel Database."""

    def __init__(self, message: str, *, path: str | None = None):
        super().__init__(message)
        self.message = message
        self.path = path


class BpmBaseIncompatible(BpmError):
    """The base schema version is not supported by this product build."""

    def __init__(self, message: str, *, schema_version: str | None = None):
        super().__init__(message)
        self.message = message
        self.schema_version = schema_version


class BpmWriteError(BpmError):
    """A write could not be committed (read-only base, permission/OS error, or
    an attempt to overwrite an existing immutable revision)."""

    def __init__(self, message: str, *, path: str | None = None):
        super().__init__(message)
        self.message = message
        self.path = path


__all__ = [
    "BpmBaseCorrupted",
    "BpmBaseIncompatible",
    "BpmBaseInvalid",
    "BpmError",
    "BpmWriteError",
]
