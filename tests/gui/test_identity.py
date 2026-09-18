"""Qt tests for process identity and packaged-icon helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from zecalibrator.gui import identity


def test_identity_constants_frozen():
    assert identity.WINDOWS_APP_USER_MODEL_ID == "ZeSoftware.ZeCalibrator"
    assert identity.LINUX_DESKTOP_FILE_NAME == "io.github.tinystork.ZeCalibrator"
    assert identity.MACOS_BUNDLE_ID == "com.zesoftware.zecalibrator"


def test_apply_process_identity_does_not_raise():
    identity.apply_process_identity()  # no-op on non-Windows; never raises


def test_load_app_icon_returns_non_null_qicon(qapp):
    from PySide6 import QtGui

    icon = identity.load_app_icon()
    assert icon is not None
    assert isinstance(icon, QtGui.QIcon)
    assert not icon.isNull()


def test_apply_linux_desktop_file_does_not_raise(qapp):
    identity.apply_linux_desktop_file(qapp)


def test_missing_icon_graceful(monkeypatch):
    import zecalibrator._resources as res

    def boom(name):
        raise OSError("missing")

    monkeypatch.setattr(res, "icon_bytes", boom)
    assert identity.load_app_icon() is None
