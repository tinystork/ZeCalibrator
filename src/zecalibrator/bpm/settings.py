"""The single Bad Pixel Database user setting: ``bad_pixel_database_root``.

Reuses the existing storage adapter (:class:`zecalibrator.storage.StoragePaths`)
and the existing settings *mechanism* (the versioned, atomic, preservation-safe
JSON discipline exemplified by ``zecalibrator.gui.settings``) — no new parallel
storage system and no second ``QStandardPaths``-style owner.

The default root is derived from ``StoragePaths.user_data_path`` (portable via
``pathlib``/``platformdirs`` — no hardcoded ``/home/...`` or ``C:\\...``).
``bad_pixel_database_root`` is the **only** user setting: no sigma, no threshold,
no radius, no duty cycle, no reconstruction operator.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

from zecalibrator.storage import StoragePaths

BPM_SETTINGS_SCHEMA_VERSION = 1
BPM_SETTINGS_FILENAME = "bpm_settings.json"

#: The default base subdirectory name under ``StoragePaths.user_data_path``.
BPM_DEFAULT_DIRNAME = "bad_pixel_database"

_KNOWN_KEYS = frozenset({"schema_version", "bad_pixel_database_root"})

STATE_OK = "ok"
STATE_MISSING = "missing"
STATE_MALFORMED = "malformed"
STATE_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class BpmSettings:
    """Immutable BPM settings snapshot (a single field)."""

    schema_version: int = BPM_SETTINGS_SCHEMA_VERSION
    bad_pixel_database_root: Optional[str] = None
    extra: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "bad_pixel_database_root": self.bad_pixel_database_root,
        }
        for key, value in self.extra.items():
            if key not in _KNOWN_KEYS:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "BpmSettings":
        if not isinstance(d, dict):
            raise ValueError("BPM settings must be a JSON object")
        if d.get("schema_version") != BPM_SETTINGS_SCHEMA_VERSION:
            raise ValueError(f"unsupported BPM settings schema {d.get('schema_version')!r}")
        root = d.get("bad_pixel_database_root")
        if root is not None and (not isinstance(root, str) or not root.strip()):
            raise ValueError("bad_pixel_database_root must be a non-empty string or null")
        extra = {k: v for k, v in d.items() if k not in _KNOWN_KEYS}
        return cls(
            schema_version=BPM_SETTINGS_SCHEMA_VERSION,
            bad_pixel_database_root=root if root is not None else None,
            extra=extra,
        )


@dataclass(frozen=True)
class BpmSettingsLoad:
    """The result of loading BPM settings: value + preservation state."""

    settings: BpmSettings
    state: str  # ok | missing | malformed | unsupported


def default_settings() -> BpmSettings:
    return BpmSettings()


def default_bad_pixel_database_root(storage: StoragePaths) -> Path:
    """Return the portable default base root derived from ``StoragePaths``.

    No directory is created and no hardcoded platform path is used; the value is
    ``<user_data_path>/bad_pixel_database`` (portable across Linux/macOS/Windows).
    """
    return Path(storage.user_data_path) / BPM_DEFAULT_DIRNAME


def resolve_bad_pixel_database_root(storage: StoragePaths, settings: BpmSettings) -> Path:
    """Return the effective base root: the configured value, else the default."""
    if settings.bad_pixel_database_root:
        return Path(settings.bad_pixel_database_root)
    return default_bad_pixel_database_root(storage)


def settings_path(config_dir: Path) -> Path:
    return Path(config_dir) / BPM_SETTINGS_FILENAME


def _read_raw(config_dir: Path):
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


def load_settings(config_dir: Path) -> BpmSettingsLoad:
    """Load BPM settings; report state; never create directories or write.

    ``missing`` = file absent; ``unreadable``/``undecodable``/``malformed``/
    ``unsupported`` preserve existing bytes (the caller must not overwrite).
    """
    state, text = _read_raw(config_dir)
    if state == "missing":
        return BpmSettingsLoad(default_settings(), STATE_MISSING)
    if state in ("unreadable", "undecodable"):
        return BpmSettingsLoad(default_settings(), STATE_MALFORMED)
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return BpmSettingsLoad(default_settings(), STATE_MALFORMED)
    if not isinstance(d, dict):
        return BpmSettingsLoad(default_settings(), STATE_MALFORMED)
    if d.get("schema_version") != BPM_SETTINGS_SCHEMA_VERSION:
        return BpmSettingsLoad(default_settings(), STATE_UNSUPPORTED)
    try:
        return BpmSettingsLoad(BpmSettings.from_dict(d), STATE_OK)
    except (TypeError, ValueError):
        return BpmSettingsLoad(default_settings(), STATE_MALFORMED)


def save_settings(config_dir: Path, settings: BpmSettings) -> None:
    """Atomically write BPM settings (temp + ``os.replace``).

    The caller must not call this over an unsupported/malformed existing file
    (see :class:`BpmSettingsLoad`); doing so would replace bytes it did not
    understand.
    """
    parent = Path(config_dir)
    parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(settings.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fd, tmp = tempfile.mkstemp(prefix=".zecalibrator-bpm-", suffix=".json.tmp", dir=str(parent))
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
    "BPM_DEFAULT_DIRNAME",
    "BPM_SETTINGS_FILENAME",
    "BPM_SETTINGS_SCHEMA_VERSION",
    "BpmSettings",
    "BpmSettingsLoad",
    "STATE_MALFORMED",
    "STATE_MISSING",
    "STATE_OK",
    "STATE_UNSUPPORTED",
    "default_bad_pixel_database_root",
    "default_settings",
    "load_settings",
    "resolve_bad_pixel_database_root",
    "save_settings",
    "settings_path",
]
