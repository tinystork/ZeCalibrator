"""Bootstrap tests: platform storage-path adapter."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from zecalibrator.storage import APP_AUTHOR, APP_NAME, StoragePaths, resolve_paths


def test_identity_constants():
    assert APP_NAME == "ZeCalibrator"
    assert APP_AUTHOR == "ZeSoftware"


def test_resolve_returns_storage_paths():
    p = resolve_paths()
    assert isinstance(p, StoragePaths)
    assert isinstance(p.user_config_path, Path)
    assert isinstance(p.user_data_path, Path)
    assert isinstance(p.user_cache_path, Path)
    assert isinstance(p.user_state_path, Path)
    assert isinstance(p.user_log_path, Path)


def test_matches_platformdirs_contract():
    """The adapter must faithfully wrap platformdirs with the frozen identity."""
    from platformdirs import PlatformDirs

    expected = PlatformDirs(
        appname="ZeCalibrator",
        appauthor="ZeSoftware",
        version=None,
        roaming=False,
        multipath=False,
    )
    p = resolve_paths()
    assert str(p.user_config_path) == expected.user_config_dir
    assert str(p.user_data_path) == expected.user_data_dir
    assert str(p.user_cache_path) == expected.user_cache_dir
    assert str(p.user_state_path) == expected.user_state_dir
    assert str(p.user_log_path) == expected.user_log_dir


def test_version_independent_roots():
    """Roots must be version-independent (no version subdirectory)."""
    from platformdirs import PlatformDirs

    no_version = PlatformDirs(
        appname="ZeCalibrator",
        appauthor="ZeSoftware",
        version=None,
        roaming=False,
        multipath=False,
    )
    with_version = PlatformDirs(
        appname="ZeCalibrator",
        appauthor="ZeSoftware",
        version="0.1.0",
        roaming=False,
        multipath=False,
    )
    assert no_version.user_data_dir != with_version.user_data_dir
    p = resolve_paths()
    assert str(p.user_data_path) == no_version.user_data_dir


@pytest.mark.skipif(sys.platform != "linux", reason="Linux XDG native test")
def test_linux_xdg_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    p = resolve_paths()
    assert p.user_config_path == Path(tmp_path) / "config" / "ZeCalibrator"
    assert p.user_data_path == Path(tmp_path) / "data" / "ZeCalibrator"
    assert p.user_cache_path == Path(tmp_path) / "cache" / "ZeCalibrator"
    assert p.user_state_path == Path(tmp_path) / "state" / "ZeCalibrator"
    assert p.user_log_path == Path(tmp_path) / "state" / "ZeCalibrator" / "log"


def test_base_override(tmp_path):
    p = resolve_paths(base=tmp_path / "base")
    assert p.user_config_path == Path(tmp_path) / "base" / "config"
    assert p.user_data_path == Path(tmp_path) / "base" / "data"
    assert p.user_cache_path == Path(tmp_path) / "base" / "cache"
    assert p.user_state_path == Path(tmp_path) / "base" / "state"
    assert p.user_log_path == Path(tmp_path) / "base" / "log"


def test_individual_override_wins_over_base(tmp_path):
    p = resolve_paths(base=tmp_path / "base", data=tmp_path / "custom-data")
    assert p.user_data_path == Path(tmp_path) / "custom-data"
    assert p.user_config_path == Path(tmp_path) / "base" / "config"


def test_individual_override_without_base(tmp_path):
    p = resolve_paths(config=tmp_path / "cfg")
    assert p.user_config_path == Path(tmp_path) / "cfg"
    # Other roots remain non-null platform defaults (not the config override).
    assert p.user_data_path != Path(tmp_path) / "cfg"
    assert str(p.user_data_path).strip()


def test_no_directory_creation(tmp_path):
    base = tmp_path / "base"
    resolve_paths(base=base)
    assert not base.exists()


def test_import_has_no_side_effects(tmp_path):
    code = textwrap.dedent(
        """
        import os
        import pathlib
        root = pathlib.Path(os.environ["ZC_TEST_ROOT"])
        os.environ["XDG_CONFIG_HOME"] = str(root / "config")
        os.environ["XDG_DATA_HOME"] = str(root / "data")
        os.environ["XDG_CACHE_HOME"] = str(root / "cache")
        os.environ["XDG_STATE_HOME"] = str(root / "state")
        import zecalibrator
        import zecalibrator.storage
        for name in ("config", "data", "cache", "state"):
            assert not (root / name).exists(), "directory created on import: " + name
        print("OK")
        """
    )
    env = dict(os.environ, ZC_TEST_ROOT=str(tmp_path))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
