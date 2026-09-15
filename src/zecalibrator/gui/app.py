"""Isolated GUI launcher.

PySide6 is imported lazily inside :func:`main`, so the GUI entry point can be
installed without the optional ``[gui]`` extra and still emit a precise
missing-extra diagnostic. The GUI application itself is not implemented in this
bootstrap phase (it is a Phase 7 deliverable); the headless engine and CLI
remain usable without PySide6.
"""

from __future__ import annotations

import sys

_MISSING_EXTRA_MESSAGE = (
    "ZeCalibrator GUI requires the optional 'gui' extra (PySide6), which is "
    "not installed in this environment.\n"
    "Install it with:  pip install 'ZeCalibrator[gui]'\n"
    "The headless engine and CLI remain usable without it."
)

_NOT_IMPLEMENTED_MESSAGE = (
    "ZeCalibrator GUI is not implemented in this bootstrap release "
    "(planned for a later phase)."
)


def main(argv: list[str] | None = None) -> int:
    """GUI entry point. Returns a process exit code."""
    try:
        from PySide6 import QtWidgets  # noqa: F401  (import smoke only)
    except ImportError:
        print(_MISSING_EXTRA_MESSAGE, file=sys.stderr)
        return 1

    # PySide6 is present, but the GUI application is not implemented yet.
    print(_NOT_IMPLEMENTED_MESSAGE, file=sys.stderr)
    return 1


__all__ = ["main"]
