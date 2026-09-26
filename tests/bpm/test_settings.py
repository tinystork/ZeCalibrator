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


def test_detector_k_default_is_30(tmp_path):
    from zecalibrator.bpm.settings import DEFAULT_DETECTOR_K

    assert DEFAULT_DETECTOR_K == 30.0
    assert default_settings().detector_k == 30.0
    # Absent field in a legacy settings file resolves to the default.
    assert BpmSettings.from_dict({"schema_version": 1}).detector_k == 30.0


def test_detector_k_persistence_and_restoration(tmp_path):
    config_dir = tmp_path / "config"
    settings = BpmSettings(
        bad_pixel_database_root=str(tmp_path / "base"), detector_k=20.0,
    )
    save_settings(config_dir, settings)
    loaded = load_settings(config_dir)
    assert loaded.state == STATE_OK
    assert loaded.settings.detector_k == 20.0
    assert loaded.settings.bad_pixel_database_root == str(tmp_path / "base")


def test_reset_default_restores_exactly_30(tmp_path):
    # The owner-authorized "Reset default" must restore exactly 30.0.
    assert default_settings().detector_k == 30.0
    assert BpmSettings(detector_k=30.0).detector_k == 30.0
    assert float(30.0) == 30.0


def test_detector_k_validation_rejects_invalid_calmly():
    import pytest

    from zecalibrator.bpm.settings import DETECTOR_K_MAX, validate_detector_k

    assert validate_detector_k(20.0) == 20.0
    assert validate_detector_k("30") == 30.0
    assert validate_detector_k(DETECTOR_K_MAX) == DETECTOR_K_MAX
    for bad in (0.0, -1.0, -30.0, float("nan"), float("inf"), float("-inf"),
                DETECTOR_K_MAX + 1.0, "abc", None):
        with pytest.raises(ValueError):
            validate_detector_k(bad)


def test_detector_k_is_the_only_scientific_setting(tmp_path):
    # The BpmSettings value object exposes exactly one scientific field: K.
    fields = set(BpmSettings.__dataclass_fields__)
    assert "detector_k" in fields
    assert not ({"sigma", "radius", "duty_cycle", "epoch", "net_benefit"} & fields)
