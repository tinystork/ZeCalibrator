"""Isolated GUI launcher.

PySide6 is imported lazily inside :func:`main`, so the GUI entry point can be
installed without the optional ``[gui]`` extra and still emit a precise
missing-extra diagnostic. The headless engine and CLI remain usable without
PySide6. The GUI itself (window/worker/service) lives in the sibling modules and
is imported only after PySide6 is confirmed present, so importing this module is
Qt-free and cheap.
"""

from __future__ import annotations

import sys

_MISSING_EXTRA_MESSAGE = (
    "ZeCalibrator GUI requires the optional 'gui' extra (PySide6), which is "
    "not installed in this environment.\n"
    "Install it with:  pip install 'ZeCalibrator[gui]'\n"
    "The headless engine and CLI remain usable without it."
)

_STARTUP_FAILED_MESSAGE = "ZeCalibrator GUI failed to start: {exc}"


def main(argv: list[str] | None = None) -> int:
    """GUI entry point. Returns a process exit code."""
    try:
        from PySide6 import QtWidgets  # noqa: F401  (import smoke only)
    except ImportError:
        print(_MISSING_EXTRA_MESSAGE, file=sys.stderr)
        return 1

    try:
        from zecalibrator.gui.window import run_application
    except Exception as exc:  # noqa: BLE001 - report a clean startup failure
        print(_STARTUP_FAILED_MESSAGE.format(exc=exc), file=sys.stderr)
        return 1

    return run_application(argv)


__all__ = ["main"]
