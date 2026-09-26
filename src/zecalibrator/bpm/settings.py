"""Bad Pixel Database user settings.

Reuses the existing storage adapter (:class:`zecalibrator.storage.StoragePaths`)
and the existing settings *mechanism* (the versioned, atomic, preservation-safe
JSON discipline exemplified by ``zecalibrator.gui.settings``) — no new parallel
storage system and no second ``QStandardPaths``-style owner.

The default root is derived from ``StoragePaths.user_data_path`` (portable via
``pathlib``/``platformdirs`` — no hardcoded ``/home/...`` or ``C:\\...``).

Two settings exist:

* ``bad_pixel_database_root`` — the Bad Pixel Database location (a path, not
  scientific).
* ``detector_k`` — the **single** scientific BPM setting (owner decision,
  2026-09-26): the detector threshold multiplier ``K`` used **only** when
  CREATING/REBUILDING an immutable Bad Pixel Map revision. Default ``30.0``.
  No other scientific control (no sigma, no radius, no duty cycle, no
  reconstruction operator, no per-frame threshold, no automatic retune) is
  exposed.
"""

from __future__ import annotations

import json
import math
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

#: The default detector threshold multiplier K (median + K * MAD * 1.4826).
DEFAULT_DETECTOR_K = 30.0

#: A conservative upper bound on the detector K setting. Values beyond this are
#: rejected calmly (never silently clamped); it is far above the default and
#: exists only to bound absurd input.
DETECTOR_K_MAX = 1000.0

_KNOWN_KEYS = frozenset({"schema_version", "bad_pixel_database_root", "detector_k"})

STATE_OK = "ok"
STATE_MISSING = "missing"
STATE_MALFORMED = "malformed"
STATE_UNSUPPORTED = "unsupported"


def validate_detector_k(value) -> float:
    """Validate the detector K setting; return it as a finite positive float.

    Rejects (raises ``ValueError``) non-numeric, non-finite, non-positive and
    out-of-bound values calmly — never silently clamps and never auto-retunes.
    """
    try:
        k = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Bad pixel detection threshold must be a number, got {value!r}"
        )
    if not math.isfinite(k) or k <= 0.0 or k > DETECTOR_K_MAX:
        raise ValueError(
            "Bad pixel detection threshold must be finite, positive and at most "
            f"{DETECTOR_K_MAX}, got {value!r}"
        )
    return k


def resolved_detector_k(revision) -> float:
    """Return a revision's effective detector K.

    A revision with no recorded ``detector_k`` (a legacy P4.2 revision) resolves
    explicitly and compatibly to :data:`DEFAULT_DETECTOR_K` (30.0) — never a
    silent fiction that it was created with a different K.
    """
    k = getattr(revision, "detector_k", None)
    return DEFAULT_DETECTOR_K if k is None else float(k)


@dataclass(frozen=True)
class BpmSettings:
    """Immutable BPM settings snapshot (location + the single detector K)."""

    schema_version: int = BPM_SETTINGS_SCHEMA_VERSION
    bad_pixel_database_root: Optional[str] = None
    detector_k: float = DEFAULT_DETECTOR_K
    extra: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))
        object.__setattr__(self, "detector_k", validate_detector_k(self.detector_k))

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "bad_pixel_database_root": self.bad_pixel_database_root,
            "detector_k": self.detector_k,
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
        detector_k = validate_detector_k(d.get("detector_k", DEFAULT_DETECTOR_K))
        extra = {k: v for k, v in d.items() if k not in _KNOWN_KEYS}
        return cls(
            schema_version=BPM_SETTINGS_SCHEMA_VERSION,
            bad_pixel_database_root=root if root is not None else None,
            detector_k=detector_k,
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
    "DEFAULT_DETECTOR_K",
    "DETECTOR_K_MAX",
    "STATE_MALFORMED",
    "STATE_MISSING",
    "STATE_OK",
    "STATE_UNSUPPORTED",
    "default_bad_pixel_database_root",
    "default_settings",
    "load_settings",
    "resolve_bad_pixel_database_root",
    "resolved_detector_k",
    "save_settings",
    "settings_path",
    "validate_detector_k",
]
