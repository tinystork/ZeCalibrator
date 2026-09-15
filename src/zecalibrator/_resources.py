"""Package-resource access via :mod:`importlib.resources`.

Resources are resolved strictly from the installed package (wheel/sdist),
never from the checkout, the current working directory, or a fixed drive path
(Interop Rule 18; ARCHITECTURE §9). This module never writes into the package
and never searches the top-level ``icons/`` directory at runtime.
"""

from __future__ import annotations

import json
from importlib.resources import files

_PACKAGE = "zecalibrator"
_ICONS_DIR = "resources/icons"
_MANIFEST = "resources/icon_manifest.json"


def _root():
    return files(_PACKAGE)


def load_icon_manifest() -> dict:
    """Return the decoded icon manifest (canonical → packaged hash map)."""
    raw = (_root() / _MANIFEST).read_bytes()
    return json.loads(raw.decode("utf-8"))


def icon_bytes(name: str) -> bytes:
    """Return byte-identical packaged icon bytes for a canonical filename."""
    if not name or not isinstance(name, str) or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"invalid icon name: {name!r}")
    return (_root() / _ICONS_DIR / name).read_bytes()


__all__ = ["load_icon_manifest", "icon_bytes"]
