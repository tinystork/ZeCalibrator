"""Settings tests: a single portable ``bad_pixel_database_root`` setting."""

from __future__ import annotations

from zecalibrator.bpm.settings import (
    BpmSettings,
    STATE_MISSING,
    STATE_OK,
    default_bad_pixel_database_root,
    default_settings,
    load_settings,
    resolve_bad_pixel_database_root,
    save_settings,
)
from zecalibrator.storage import StoragePaths


def _storage(tmp_path) -> StoragePaths:
    return StoragePaths(
        user_config_path=tmp_path / "config",
        user_data_path=tmp_path / "data",
        user_cache_path=tmp_path / "cache",
        user_state_path=tmp_path / "state",
        user_log_path=tmp_path / "log",
    )


def test_default_root_is_portable_and_derived(tmp_path):
    root = default_bad_pixel_database_root(_storage(tmp_path))
    # no hardcoded /home or C:\ — it is under the injected data path
    assert str(root).startswith(str(tmp_path))
    assert root.name == "bad_pixel_database"


def test_no_settings_yields_default_root(tmp_path):
    storage = _storage(tmp_path)
    resolved = resolve_bad_pixel_database_root(storage, default_settings())
    assert resolved == default_bad_pixel_database_root(storage)


def test_override_root_wins(tmp_path):
    storage = _storage(tmp_path)
    settings = BpmSettings(bad_pixel_database_root=str(tmp_path / "my-base"))
    assert resolve_bad_pixel_database_root(storage, settings) == tmp_path / "my-base"


def test_roundtrip_and_missing_state(tmp_path):
    config_dir = tmp_path / "config"
    assert load_settings(config_dir).state == STATE_MISSING
    settings = BpmSettings(bad_pixel_database_root=str(tmp_path / "my-base"))
    save_settings(config_dir, settings)
    loaded = load_settings(config_dir)
    assert loaded.state == STATE_OK
    assert loaded.settings.bad_pixel_database_root == str(tmp_path / "my-base")
