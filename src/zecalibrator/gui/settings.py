"""Versioned, injected GUI settings with atomic, preservation-safe writes.

GUI settings live under ``StoragePaths.user_config_path`` (a single
``zecalibrator.storage`` adapter — never a second ``QStandardPaths`` owner). This
module performs no import-time directory creation or live-settings reads; all
roots are injected by the caller (the window receives its ``StoragePaths``).

Contract (ARCHITECTURE §8, prepared §10):

* Writes are atomic (temp + ``os.replace``).
* An unsupported on-disk schema or a malformed file is **never** silently
  overwritten: loading reports a ``state``, and the caller must preserve the
  existing bytes (skip or back up) rather than call ``save_settings`` over them.
* Unknown fields in a *valid* current-schema file are preserved (forward
  compatible), not silently dropped.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

SETTINGS_SCHEMA_VERSION = 1
SETTINGS_FILENAME = "gui_settings.json"

_KNOWN_KEYS = frozenset({
    "schema_version",
    "last_input_dir",
    "last_library_dir",
    "last_output_dir",
    "window_width",
    "window_height",
})


@dataclass(frozen=True)
class GuiSettings:
    """Immutable GUI preferences snapshot (window + last-used directories).

    ``extra`` preserves unknown forward-compatible keys from a valid
    current-schema file so they survive a round-trip unchanged.
    """

    schema_version: int = SETTINGS_SCHEMA_VERSION
    last_input_dir: Optional[str] = None
    last_library_dir: Optional[str] = None
    last_output_dir: Optional[str] = None
    window_width: int = 1280
    window_height: int = 800
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "last_input_dir": self.last_input_dir,
            "last_library_dir": self.last_library_dir,
            "last_output_dir": self.last_output_dir,
            "window_width": self.window_width,
            "window_height": self.window_height,
        }
        for key, value in self.extra.items():
            if key not in _KNOWN_KEYS:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GuiSettings":
        if not isinstance(d, dict):
            raise ValueError("settings must be a JSON object")
        if d.get("schema_version") != SETTINGS_SCHEMA_VERSION:
            raise ValueError(f"unsupported settings schema {d.get('schema_version')!r}")

        def _str_or_none(name):
            v = d.get(name)
            return v if isinstance(v, str) else None

        def _int_or_default(name, default):
            v = d.get(name)
            if isinstance(v, bool):
                return default
            if isinstance(v, (int, float)):
                try:
                    f = float(v)
                except (TypeError, ValueError, OverflowError):
                    return default
                if not (f > 0) or f != f or f in (float("inf"), float("-inf")):
                    return default
                if f > 1_000_000_000:
                    return default
                return int(f)
            return default

        extra = {k: v for k, v in d.items() if k not in _KNOWN_KEYS}
        return cls(
            schema_version=SETTINGS_SCHEMA_VERSION,
            last_input_dir=_str_or_none("last_input_dir"),
            last_library_dir=_str_or_none("last_library_dir"),
            last_output_dir=_str_or_none("last_output_dir"),
            window_width=_int_or_default("window_width", 1280),
            window_height=_int_or_default("window_height", 800),
            extra=extra,
        )


# Load states (caller decides whether a save is allowed over the existing file).
STATE_OK = "ok"
STATE_MISSING = "missing"
STATE_MALFORMED = "malformed"
STATE_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SettingsLoad:
    """The result of loading settings: value + why (for preservation decisions)."""

    settings: GuiSettings
    state: str  # ok | missing | malformed | unsupported


def default_settings() -> GuiSettings:
    return GuiSettings()


def settings_path(config_dir: Path) -> Path:
    return Path(config_dir) / SETTINGS_FILENAME


def _read_raw(config_dir: Path):
    """Return (state, text) where state ∈ missing|unreadable|undecodable|ok.

    A truly absent file is distinguishable from a file that exists but cannot be
    read/decoded, so callers can preserve corrupt bytes instead of overwriting.
    """
    path = settings_path(config_dir)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ("missing", None)
    except OSError:
        return ("unreadable", None)
    try:
        return ("ok", data.decode("utf-8"))
    except UnicodeDecodeError:
        return ("undecodable", None)


def load_settings(config_dir: Path) -> SettingsLoad:
    """Load settings; report state; never create directories or write.

    ``missing`` = file absent; ``unreadable``/``undecodable``/``malformed``/
    ``unsupported`` all preserve the existing bytes (the caller must not
    overwrite them).
    """
    state, text = _read_raw(config_dir)
    if state == "missing":
        return SettingsLoad(default_settings(), STATE_MISSING)
    if state == "unreadable":
        return SettingsLoad(default_settings(), STATE_MALFORMED)
    if state == "undecodable":
        return SettingsLoad(default_settings(), STATE_MALFORMED)
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return SettingsLoad(default_settings(), STATE_MALFORMED)
    if not isinstance(d, dict):
        return SettingsLoad(default_settings(), STATE_MALFORMED)
    if d.get("schema_version") != SETTINGS_SCHEMA_VERSION:
        return SettingsLoad(default_settings(), STATE_UNSUPPORTED)
    try:
        return SettingsLoad(GuiSettings.from_dict(d), STATE_OK)
    except (TypeError, ValueError, OverflowError):
        return SettingsLoad(default_settings(), STATE_MALFORMED)


def save_settings(config_dir: Path, settings: GuiSettings) -> None:
    """Atomically write settings into the injected config root.

    ``os.replace`` keeps the write atomic; only a task-owned temp file is removed
    on failure. The caller must not call this over an unsupported/malformed
    existing file (see ``SettingsLoad.state``); doing so would replace data it
    did not understand, so callers gate the write on a safe load state.
    """
    parent = Path(config_dir)
    parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        settings.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    fd, tmp = tempfile.mkstemp(
        prefix=".zecalibrator-gui-", suffix=".json.tmp", dir=str(parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, settings_path(parent))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


__all__ = [
    "SETTINGS_FILENAME",
    "SETTINGS_SCHEMA_VERSION",
    "STATE_MALFORMED",
    "STATE_MISSING",
    "STATE_OK",
    "STATE_UNSUPPORTED",
    "GuiSettings",
    "SettingsLoad",
    "default_settings",
    "load_settings",
    "save_settings",
    "settings_path",
]
