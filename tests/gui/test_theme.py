"""Qt tests for the bounded System/Light/Dark appearance helper.

Asserts deterministic key-palette roles for Light/Dark and exact restoration for
System (never pixel appearance), plus theme normalization and Qt-free import
safety.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtGui

from zecalibrator.gui import theme


@pytest.fixture(autouse=True)
def _restore_system(qapp):
    """Capture the native palette once and always restore System after a test."""
    system_palette = theme.get_system_palette(qapp)
    yield
    theme.apply_theme(qapp, theme.THEME_SYSTEM, system_palette)


def _window_color(palette):
    return palette.color(QtGui.QPalette.ColorRole.Window).name()


def _base_color(palette):
    return palette.color(QtGui.QPalette.ColorRole.Base).name()


def test_theme_options_exact_and_default_system():
    assert theme.THEMES == ("system", "light", "dark")
    assert theme.normalize_theme("system") == theme.THEME_SYSTEM
    assert theme.normalize_theme(None) == theme.THEME_SYSTEM
    assert theme.normalize_theme("neon") == theme.THEME_SYSTEM
    assert theme.theme_label("system") == "System"
    assert theme.theme_label("light") == "Light"
    assert theme.theme_label("dark") == "Dark"


def test_light_palette_is_deterministic(qapp):
    sys_pal = theme.get_system_palette(qapp)
    theme.apply_theme(qapp, theme.THEME_LIGHT, sys_pal)
    first = _window_color(qapp.palette())
    theme.apply_theme(qapp, theme.THEME_DARK, sys_pal)
    theme.apply_theme(qapp, theme.THEME_LIGHT, sys_pal)
    assert _window_color(qapp.palette()) == first
    assert first == "#ffffff"


def test_dark_palette_is_deterministic(qapp):
    sys_pal = theme.get_system_palette(qapp)
    theme.apply_theme(qapp, theme.THEME_DARK, sys_pal)
    first = _window_color(qapp.palette())
    assert first == "#353535"
    assert _base_color(qapp.palette()) == "#232323"
    theme.apply_theme(qapp, theme.THEME_LIGHT, sys_pal)
    theme.apply_theme(qapp, theme.THEME_DARK, sys_pal)
    assert _window_color(qapp.palette()) == first


def test_system_restores_captured_native_palette(qapp):
    sys_pal = theme.get_system_palette(qapp)
    native_window = sys_pal.color(QtGui.QPalette.ColorRole.Window).name()
    theme.apply_theme(qapp, theme.THEME_DARK, sys_pal)
    assert _window_color(qapp.palette()) != native_window
    theme.apply_theme(qapp, theme.THEME_SYSTEM, sys_pal)
    assert _window_color(qapp.palette()) == native_window


def test_light_and_dark_are_distinct(qapp):
    sys_pal = theme.get_system_palette(qapp)
    theme.apply_theme(qapp, theme.THEME_LIGHT, sys_pal)
    light = _window_color(qapp.palette())
    theme.apply_theme(qapp, theme.THEME_DARK, sys_pal)
    dark = _window_color(qapp.palette())
    assert light != dark


def test_theme_module_import_is_qt_free():
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        "import zecalibrator.gui.theme\n"
        "assert 'PySide6' not in sys.modules\n"
        "print('OK')\n"
    )
    env = dict(__import__("os").environ)
    repo = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(repo / "src") + __import__("os").pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
