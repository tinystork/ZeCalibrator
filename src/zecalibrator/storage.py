"""Platform storage-path adapter.

Wraps ``platformdirs`` behind a single ``zecalibrator`` storage module.

Contract (ARCHITECTURE §8, Interop Rules 8/17/18):

* ``appname="ZeCalibrator"``, ``appauthor="ZeSoftware"``;
* version-independent roots (``version=None``);
* non-roaming mutable data (``roaming=False``);
* no import-time user-directory creation and no live user-settings reads —
  importing this module and resolving paths only compute :class:`Path` values;
* tests inject all roots explicitly.

The five location classes are ``user_config_path`` (versioned settings),
``user_data_path`` (library index/descriptors), ``user_cache_path``
(rebuildable materialization), ``user_state_path`` and ``user_log_path``
(bounded logs/recovery). No directory is created by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Union

APP_NAME = "ZeCalibrator"
APP_AUTHOR = "ZeSoftware"

StrPath = Union[str, PathLike]


@dataclass(frozen=True)
class StoragePaths:
    """Computed platform storage locations (path values only; nothing created)."""

    user_config_path: Path
    user_data_path: Path
    user_cache_path: Path
    user_state_path: Path
    user_log_path: Path

    def as_dict(self) -> dict[str, Path]:
        """Return the five paths as a stable ``name -> Path`` mapping."""
        return {
            "user_config_path": self.user_config_path,
            "user_data_path": self.user_data_path,
            "user_cache_path": self.user_cache_path,
            "user_state_path": self.user_state_path,
            "user_log_path": self.user_log_path,
        }


def _platform_defaults():
    """Construct platformdirs with the frozen ZeCalibrator identity."""
    from platformdirs import PlatformDirs

    return PlatformDirs(
        appname=APP_NAME,
        appauthor=APP_AUTHOR,
        version=None,
        roaming=False,
        multipath=False,
    )


def resolve_paths(
    *,
    config: StrPath | None = None,
    data: StrPath | None = None,
    cache: StrPath | None = None,
    state: StrPath | None = None,
    log: StrPath | None = None,
    base: StrPath | None = None,
) -> StoragePaths:
    """Resolve storage paths without creating directories or reading settings.

    Precedence, highest first:

    1. explicit per-root keyword overrides (``config``, ``data``, ``cache``,
       ``state``, ``log``);
    2. ``base`` — each unspecified root derives from it under a fixed
       subdirectory name, for injectable/portable roots;
    3. platformdirs platform conventions (Linux XDG, Windows AppData,
       macOS Application Support).

    ``version=None`` and ``roaming=False`` are enforced by the adapter; no
    filesystem mutation or settings discovery happens as a side effect.
    """

    default_map = _platform_defaults()

    def _pick(name: str, default_dir: str) -> Path:
        override = {"config": config, "data": data, "cache": cache, "state": state, "log": log}[name]
        if override is not None:
            return Path(override)
        if base is not None:
            return Path(base) / name
        return Path(default_dir)

    return StoragePaths(
        user_config_path=_pick("config", default_map.user_config_dir),
        user_data_path=_pick("data", default_map.user_data_dir),
        user_cache_path=_pick("cache", default_map.user_cache_dir),
        user_state_path=_pick("state", default_map.user_state_dir),
        user_log_path=_pick("log", default_map.user_log_dir),
    )


__all__ = ["APP_NAME", "APP_AUTHOR", "StoragePaths", "resolve_paths"]
