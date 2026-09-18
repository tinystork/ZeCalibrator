"""Process identity and packaged-icon helpers for the Qt application.

No Qt import happens at module import time (functions import PySide6 lazily), so
this module can be imported from a headless context without pulling Qt. The
process identity strings are the frozen ARCHITECTURE §9 stable identities.

Windows ``AppUserModelID`` must be set *before* ``QApplication`` construction
(Windows taskbar grouping); Linux ``desktopFileName`` is applied after the
application object exists; the macOS bundle id is reserved for G8 (no bundle is
created here, and no icon bundle identity is asserted on macOS at this phase).
"""

from __future__ import annotations

import sys

WINDOWS_APP_USER_MODEL_ID = "ZeSoftware.ZeCalibrator"
LINUX_DESKTOP_FILE_NAME = "io.github.tinystork.ZeCalibrator"
MACOS_BUNDLE_ID = "com.zesoftware.zecalibrator"

# A packaged PNG variant suitable for the app/window icon (byte-identical
# packaged copy, loaded via importlib.resources — never the top-level icons/).
_ICON_NAME = "zecalibrator_256x256.png"


def apply_process_identity() -> None:
    """Set the Windows AppUserModelID before QApplication (best-effort).

    Failures are non-fatal (a missing identity must not prevent the app from
    starting); resource tests still assert the canonical bytes are packaged.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                WINDOWS_APP_USER_MODEL_ID
            )
        except Exception:  # noqa: BLE001 - non-fatal identity failure
            return


def apply_linux_desktop_file(app) -> None:
    """Apply the Linux desktop id to a QApplication (matches installed desktop id)."""
    if sys.platform.startswith("linux"):
        try:
            app.setDesktopFileName(LINUX_DESKTOP_FILE_NAME)
        except AttributeError:  # older Qt without setDesktopFileName
            return


def load_app_icon():
    """Return a ``QIcon`` from packaged PNG bytes, or ``None`` on any failure.

    Qt import is lazy here; the caller (GUI thread) invokes this once and applies
    the result to the application and main window.
    """
    try:
        from PySide6 import QtGui

        from zecalibrator import _resources

        data = _resources.icon_bytes(_ICON_NAME)
        pixmap = QtGui.QPixmap()
        if not pixmap.loadFromData(data):
            return None
        if pixmap.isNull():
            return None
        return QtGui.QIcon(pixmap)
    except Exception:  # noqa: BLE001 - graceful runtime missing-icon
        return None


__all__ = [
    "LINUX_DESKTOP_FILE_NAME",
    "MACOS_BUNDLE_ID",
    "WINDOWS_APP_USER_MODEL_ID",
    "apply_linux_desktop_file",
    "apply_process_identity",
    "load_app_icon",
]
