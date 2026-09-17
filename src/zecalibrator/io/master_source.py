"""Filesystem/source adapter for streamed byte identities and header metadata.

The library/matcher core is filesystem-independent: it consumes frozen
descriptors and candidates. This module provides the *source* boundary that
computes real content/mask identities by streaming bytes, without loading master
pixel arrays into the matcher, and a stable, symlink-loop-safe filesystem
enumeration for the scan/import orchestration.

* ``image_identity`` streams the FITS bytes and returns ``(content_sha256,
  size_bytes)`` with stable-byte verification (a double read that must hash
  identically, so a same-size edit or a change *during* hashing is detected —
  never a TOCTOU gap disguised as valid because mtime happened to agree).
* ``mask_identity`` hashes an external DQ/mask payload.
* ``read_header`` inspects FITS header cards without loading the image array.
* ``enumerate_fits_files`` walks a root with stable traversal, symlink-loop
  avoidance and structured inaccessible/invalid diagnostics.

An in-memory source is provided for tests and injected handles.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from zecalibrator.core.plans import FitsFileLocator, MaskPayloadLocator

_CHUNK = 1 << 20


class SourceError(Exception):
    """A source read/hash/identity failure with a structured reason."""


@dataclass(frozen=True)
class ImageIdentity:
    """A streamed image byte identity."""

    content_sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ScanDiagnostic:
    """A structured scan diagnostic (unreadable/invalid source, never 'missing success')."""

    path: str
    reason: str

    def to_dict(self) -> Mapping[str, str]:
        return {"path": self.path, "reason": self.reason}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stream_sha256(path: str) -> Tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


class FilesystemSource:
    """Streaming byte-identity source over a real filesystem."""

    def image_identity(self, locator: FitsFileLocator) -> ImageIdentity:
        first = self._stable_hash(locator.path)
        second = self._stable_hash(locator.path)
        if first != second:
            raise SourceError(f"file changed during read: {locator.path}")
        return ImageIdentity(content_sha256=first[0], size_bytes=first[1])

    def mask_identity(self, locator: MaskPayloadLocator) -> str:
        first = self._stable_hash(locator.path)
        second = self._stable_hash(locator.path)
        if first != second:
            raise SourceError(f"mask changed during read: {locator.path}")
        return first[0]

    def _stable_hash(self, path: str) -> Tuple[str, int]:
        try:
            before = os.stat(path)
        except OSError as exc:
            raise SourceError(f"cannot stat {path}: {exc}") from exc
        sha, size = _stream_sha256(path)
        try:
            after = os.stat(path)
        except OSError as exc:
            raise SourceError(f"cannot stat {path}: {exc}") from exc
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise SourceError(f"file changed during read: {path}")
        return sha, size

    def read_header(self, path: str, hdu=0) -> list:
        """Inspect FITS header cards without loading the image array."""
        from astropy.io import fits

        with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdulist:
            header = hdulist[hdu].header
            return list(header.cards)


class InMemorySource:
    """An injected in-memory byte source (tests / filesystem-independent handles)."""

    def __init__(self, images: Optional[Mapping[str, bytes]] = None, masks: Optional[Mapping[str, bytes]] = None) -> None:
        self._images = dict(images or {})
        self._masks = dict(masks or {})

    def put_image(self, path: str, data: bytes) -> None:
        self._images[path] = data

    def put_mask(self, path: str, data: bytes) -> None:
        self._masks[path] = data

    def image_identity(self, locator: FitsFileLocator) -> ImageIdentity:
        data = self._images.get(locator.path)
        if data is None:
            raise SourceError(f"image not found: {locator.path}")
        return ImageIdentity(content_sha256=_sha256_bytes(data), size_bytes=len(data))

    def mask_identity(self, locator: MaskPayloadLocator) -> str:
        data = self._masks.get(locator.path)
        if data is None:
            raise SourceError(f"mask not found: {locator.path}")
        return _sha256_bytes(data)


def enumerate_fits_files(root: str, *, follow_symlinks: bool = False) -> Tuple[list, list]:
    """Enumerate files under ``root`` with stable traversal and symlink-loop
    avoidance. Returns ``(files, diagnostics)`` where ``files`` are deduplicated by
    resolved realpath and sorted by path, and ``diagnostics`` record inaccessible
    directories (never silently skipped as success)."""
    files: list = []
    diagnostics: list = []
    seen: set = set()
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        diagnostics.append(ScanDiagnostic(path=root, reason="not a directory"))
        return files, diagnostics

    def onerror(err):
        diagnostics.append(ScanDiagnostic(path=getattr(err, "filename", str(err)), reason=str(err)))

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks, onerror=onerror):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            p = os.path.join(dirpath, name)
            real = os.path.realpath(p)
            if real in seen:
                continue
            seen.add(real)
            files.append(p)
    files.sort()
    return files, diagnostics


__all__ = [
    "FilesystemSource",
    "ImageIdentity",
    "InMemorySource",
    "ScanDiagnostic",
    "SourceError",
    "enumerate_fits_files",
]
