"""Bounded Qt palette helper for the System/Light/Dark appearance preference.

This is deliberately *not* a theme engine, style framework or OS probe:

* ``System`` restores the ``QApplication`` palette captured **before** any
  ZeCalibrator override and never forces a non-native style.
* ``Light`` / ``Dark`` are deterministic, bounded ``QPalette`` overrides (no
  stylesheet is applied).
* Exactly one original palette snapshot is kept for the whole process, so
  switching back to System after Light/Dark always restores the native palette.

No Qt import happens at module import time (PySide6 is imported lazily inside
each function), so importing this module is safe in a headless context.
"""

from __future__ import annotations

THEME_SYSTEM = "system"
THEME_LIGHT = "light"
THEME_DARK = "dark"

THEMES = (THEME_SYSTEM, THEME_LIGHT, THEME_DARK)

_THEME_LABELS = {
    THEME_SYSTEM: "System",
    THEME_LIGHT: "Light",
    THEME_DARK: "Dark",
}

_system_palette = None


def normalize_theme(value) -> str:
    """Return a canonical theme name; unknown/invalid values fall back to System."""
    return value if value in THEMES else THEME_SYSTEM


def theme_label(value) -> str:
    return _THEME_LABELS.get(value, value)


def get_system_palette(app):
    """Return the one original palette snapshot (captured before any override).

    The first caller captures ``app.palette()`` (the native Qt palette) and a
    frozen copy is reused for the rest of the process, so ``System`` always
    restores the pre-override palette even after Light/Dark has been applied.
    """
    global _system_palette
    from PySide6 import QtGui

    if _system_palette is None:
        _system_palette = QtGui.QPalette(app.palette())
    return QtGui.QPalette(_system_palette)


def reset_system_palette() -> None:
    """For tests only: forget the captured snapshot so the next capture is native."""
    global _system_palette
    _system_palette = None


def _palette(
    window,
    window_text,
    base,
    text,
    button,
    button_text,
    highlight,
    highlighted_text,
    tooltip_base,
    tooltip_text,
):
    from PySide6 import QtGui

    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, window)
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, window_text)
    palette.setColor(QtGui.QPalette.ColorRole.Base, base)
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, window)
    palette.setColor(QtGui.QPalette.ColorRole.Text, text)
    palette.setColor(QtGui.QPalette.ColorRole.Button, button)
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, button_text)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, highlighted_text)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, tooltip_base)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, tooltip_text)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("red"))
    palette.setColor(QtGui.QPalette.ColorRole.Link, highlight)
    palette.setColor(QtGui.QPalette.ColorRole.LinkVisited, highlight)
    return palette


def _light_palette():
    from PySide6 import QtGui

    return _palette(
        QtGui.QColor(255, 255, 255),  # window
        QtGui.QColor(0, 0, 0),        # window text
        QtGui.QColor(255, 255, 255),  # base
        QtGui.QColor(0, 0, 0),        # text
        QtGui.QColor(240, 240, 240),  # button
        QtGui.QColor(0, 0, 0),        # button text
        QtGui.QColor(48, 140, 198),   # highlight
        QtGui.QColor(255, 255, 255),  # highlighted text
        QtGui.QColor(255, 255, 220),  # tooltip base
        QtGui.QColor(0, 0, 0),        # tooltip text
    )


def _dark_palette():
    from PySide6 import QtGui

    return _palette(
        QtGui.QColor(53, 53, 53),     # window
        QtGui.QColor(255, 255, 255),  # window text
        QtGui.QColor(35, 35, 35),     # base
        QtGui.QColor(255, 255, 255),  # text
        QtGui.QColor(53, 53, 53),     # button
        QtGui.QColor(255, 255, 255),  # button text
        QtGui.QColor(42, 130, 218),   # highlight
        QtGui.QColor(0, 0, 0),        # highlighted text
        QtGui.QColor(53, 53, 53),     # tooltip base
        QtGui.QColor(255, 255, 255),  # tooltip text
    )


def apply_theme(app, theme_name, system_palette) -> None:
    """Apply a theme to ``app`` (never forces a non-native style)."""
    name = normalize_theme(theme_name)
    if name == THEME_SYSTEM:
        app.setPalette(system_palette)
    elif name == THEME_LIGHT:
        app.setPalette(_light_palette())
    elif name == THEME_DARK:
        app.setPalette(_dark_palette())
    else:  # pragma: no cover - normalize_theme already guards this
        app.setPalette(system_palette)


__all__ = [
    "THEME_DARK",
    "THEME_LIGHT",
    "THEME_SYSTEM",
    "THEMES",
    "apply_theme",
    "get_system_palette",
    "normalize_theme",
    "reset_system_palette",
    "theme_label",
]
