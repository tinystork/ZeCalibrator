"""Headless tests for versioned atomic GUI settings (no PySide6 required)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from zecalibrator.gui import settings
from zecalibrator.gui.settings import (
    STATE_MALFORMED,
    STATE_MISSING,
    STATE_OK,
    STATE_UNSUPPORTED,
    GuiSettings,
)


def test_default_settings_roundtrip(tmp_path):
    s = settings.default_settings()
    settings.save_settings(tmp_path, s)
    loaded = settings.load_settings(tmp_path)
    assert loaded.state == STATE_OK
    assert loaded.settings == s


def test_load_missing_returns_defaults_without_creating(tmp_path):
    config = tmp_path / "config"
    loaded = settings.load_settings(config)
    assert loaded.state == STATE_MISSING
    assert loaded.settings == settings.default_settings()
    assert not config.exists()


def test_save_is_atomic_and_leaves_no_temp(tmp_path):
    settings.save_settings(tmp_path, GuiSettings(window_width=1000, window_height=700))
    assert settings.settings_path(tmp_path).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_roundtrip_preserves_values(tmp_path):
    s = GuiSettings(
        last_input_dir="/in put", last_library_dir="/lib", last_output_dir="/oüt",
        window_width=1400, window_height=900,
    )
    settings.save_settings(tmp_path, s)
    assert settings.load_settings(tmp_path).settings == s


def test_roundtrip_preserves_unknown_forward_compatible_fields(tmp_path):
    # Unknown fields in a valid current-schema file must survive (F5).
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "window_width": 1000, "future_field": "keep-me"}),
        encoding="utf-8",
    )
    loaded = settings.load_settings(tmp_path)
    assert loaded.state == STATE_OK
    assert loaded.settings.extra["future_field"] == "keep-me"
    settings.save_settings(tmp_path, loaded.settings)
    # Round-trip through save preserves the unknown field.
    reloaded = settings.load_settings(tmp_path)
    assert reloaded.settings.extra["future_field"] == "keep-me"


def test_malformed_json_reports_state_and_does_not_overwrite(tmp_path):
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    before = path.read_bytes()
    loaded = settings.load_settings(tmp_path)
    assert loaded.state == STATE_MALFORMED
    assert loaded.settings == settings.default_settings()
    assert path.read_bytes() == before  # untouched


def test_unknown_schema_reports_state_and_does_not_overwrite(tmp_path):
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"schema_version": 99, "preserve-me": "important"})
    path.write_text(original, encoding="utf-8")
    before = path.read_bytes()
    loaded = settings.load_settings(tmp_path)
    assert loaded.state == STATE_UNSUPPORTED
    assert loaded.settings == settings.default_settings()
    assert path.read_bytes() == before


def test_non_object_json_reports_malformed(tmp_path):
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1,2,3]", encoding="utf-8")
    assert settings.load_settings(tmp_path).state == STATE_MALFORMED


def test_non_utf8_bytes_reported_malformed_not_missing(tmp_path):
    """R3: corrupt non-UTF8 bytes must be malformed (not missing), so they are preserved."""
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"\xff\xfe\x80"
    path.write_bytes(raw)
    loaded = settings.load_settings(tmp_path)
    assert loaded.state == STATE_MALFORMED
    assert loaded.settings == settings.default_settings()
    assert path.read_bytes() == raw  # preserved, never overwritten


def test_unreadable_file_reported_malformed(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX permission model only")
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    os.chmod(path, 0o000)
    try:
        assert settings.load_settings(tmp_path).state == STATE_MALFORMED
    finally:
        os.chmod(path, 0o600)


def test_malformed_numeric_dimensions_safe(tmp_path):
    # int(Inf)/huge dimensions must not raise or produce invalid window sizes.
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for bad in (1e309, float("inf"), float("nan"), 1e20):
        path.write_text(
            json.dumps({"schema_version": 1, "window_width": bad, "window_height": 700}),
            encoding="utf-8",
        )
        loaded = settings.load_settings(tmp_path)
        assert loaded.settings.window_width == 1280
        assert loaded.settings.window_height == 700


def test_negative_window_dimensions_rejected_to_defaults(tmp_path):
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "window_width": -5, "window_height": -1}))
    loaded = settings.load_settings(tmp_path)
    assert loaded.settings.window_width == 1280
    assert loaded.settings.window_height == 800


def test_read_only_settings_directory_raises(tmp_path):
    if os.name == "nt":  # read-only semantics differ on Windows runners
        pytest.skip("POSIX permission model only")
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        with pytest.raises(OSError):
            settings.save_settings(ro, settings.default_settings())
    finally:
        os.chmod(ro, 0o700)


def test_settings_path_is_under_injected_config_root(tmp_path):
    assert settings.settings_path(tmp_path) == Path(tmp_path) / "gui_settings.json"


def test_default_settings_have_system_theme():
    assert settings.default_settings().appearance_theme == "system"


def test_theme_roundtrip(tmp_path):
    s = GuiSettings(appearance_theme="dark")
    settings.save_settings(tmp_path, s)
    assert settings.load_settings(tmp_path).settings.appearance_theme == "dark"


def test_theme_missing_defaults_to_system(tmp_path):
    # Old settings (schema 1, no theme field) load as System.
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "window_width": 1000}), encoding="utf-8")
    loaded = settings.load_settings(tmp_path)
    assert loaded.state == STATE_OK
    assert loaded.settings.appearance_theme == "system"


def test_theme_invalid_value_falls_back_to_system(tmp_path):
    for bad in ("neon", 123, None, True):
        path = settings.settings_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": 1, "appearance_theme": bad}), encoding="utf-8"
        )
        assert settings.load_settings(tmp_path).settings.appearance_theme == "system"


def test_theme_is_a_known_key_and_roundtrips_with_unknown_fields(tmp_path):
    path = settings.settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema_version": 1, "appearance_theme": "light", "future_field": "keep-me",
        }),
        encoding="utf-8",
    )
    loaded = settings.load_settings(tmp_path)
    assert loaded.state == STATE_OK
    assert loaded.settings.appearance_theme == "light"
    assert loaded.settings.extra["future_field"] == "keep-me"
    settings.save_settings(tmp_path, loaded.settings)
    reloaded = settings.load_settings(tmp_path)
    assert reloaded.settings.appearance_theme == "light"
    assert reloaded.settings.extra["future_field"] == "keep-me"
