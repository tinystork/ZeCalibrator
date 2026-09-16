"""Native Qt icon-decode witness (G3 platform evidence).

This is the mandatory "Qt native icon decoding" witness from the TODO.md
platform register (ASTRA_MISSION §13 and §18). It exercises Qt's actual image
decoders against every packaged icon asset, as opposed to the byte-hash
equality already covered by ``tests/test_resources.py``.

PySide6 is the optional ``[gui]`` extra. It is imported lazily inside the
tests, and the module skips cleanly when it is absent, so the no-PySide6 test
step still passes with a skip rather than a failure.

This test never writes into the package and never mutates the canonical icons;
it only reads packaged bytes via ``zecalibrator._resources.icon_bytes`` and
decodes them in memory.
"""

from __future__ import annotations

import os

import pytest

import zecalibrator._resources as res

# Force a headless offscreen platform before PySide6 is imported or any
# QGuiApplication is instantiated (required for CI/headless runners).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Pinned exact dimensions for the ten PNG assets: the nine square variants
# (16..1024) plus the canonical supplied source raster (ASTRA_MISSION §13).
PNG_DIMENSIONS = {
    "zecalibrator_16x16.png": (16, 16),
    "zecalibrator_24x24.png": (24, 24),
    "zecalibrator_32x32.png": (32, 32),
    "zecalibrator_48x48.png": (48, 48),
    "zecalibrator_64x64.png": (64, 64),
    "zecalibrator_128x128.png": (128, 128),
    "zecalibrator_256x256.png": (256, 256),
    "zecalibrator_512x512.png": (512, 512),
    "zecalibrator_1024x1024.png": (1024, 1024),
    "zecalibrator_icon.png": (1254, 1254),
}

# All twelve packaged assets must decode to a non-null image.
ALL_ASSETS = frozenset(PNG_DIMENSIONS) | {"zecalibrator.ico", "zecalibrator.icns"}


@pytest.fixture(scope="module")
def qt():
    """Return (QtCore, QtGui, QGuiApplication), skipping if PySide6 is absent.

    PySide6 is imported lazily here so this module can be collected (and skip)
    on hosts without the ``[gui]`` extra.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtCore, QtGui
    except ImportError as exc:  # pragma: no cover - exercised without [gui]
        pytest.skip(f"PySide6 ([gui] extra) is not installed: {exc}")
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    return QtCore, QtGui, app


def _decode(qtcore, qtgui, data: bytes):
    """Decode ``data`` with a Qt image reader; return (image|None, error|None)."""
    buf = qtcore.QBuffer()
    buf.setData(qtcore.QByteArray(data))
    if not buf.open(qtcore.QIODevice.OpenModeFlag.ReadOnly):
        return None, "buffer open failed"
    reader = qtgui.QImageReader(buf)
    if not reader.canRead():
        return None, reader.errorString()
    image = reader.read()
    if image.isNull():
        return None, reader.errorString()
    return image, None


def test_all_twelve_assets_decode_to_non_null_image(qt):
    qtcore, qtgui, _app = qt
    manifest = res.load_icon_manifest()
    names = {entry["name"] for entry in manifest["files"]}
    assert names == set(ALL_ASSETS)
    assert len(names) == 12
    for name in sorted(names):
        image, err = _decode(qtcore, qtgui, res.icon_bytes(name))
        assert image is not None, f"{name}: decode failed ({err})"
        assert image.width() > 0 and image.height() > 0, name


def test_png_assets_have_exact_pinned_dimensions(qt):
    qtcore, qtgui, _app = qt
    for name, (w, h) in sorted(PNG_DIMENSIONS.items()):
        image, err = _decode(qtcore, qtgui, res.icon_bytes(name))
        assert image is not None, f"{name}: decode failed ({err})"
        assert (image.width(), image.height()) == (w, h), name


def test_ico_has_at_least_one_decodable_frame(qt):
    qtcore, qtgui, _app = qt
    data = res.icon_bytes("zecalibrator.ico")
    buf = qtcore.QBuffer()
    buf.setData(qtcore.QByteArray(data))
    assert buf.open(qtcore.QIODevice.OpenModeFlag.ReadOnly)
    reader = qtgui.QImageReader(buf)
    assert reader.canRead(), reader.errorString()
    assert reader.imageCount() >= 1
    image = reader.read()
    assert not image.isNull()
    assert image.width() > 0 and image.height() > 0


def test_icns_decodes_to_non_null_image(qt):
    # Qt ships an `icns` image-format plugin; its first decodable frame is a
    # small representation (observed 16x16 on Linux), so only non-null is
    # asserted here — ICNS dimensions are not pinned. This asset is required
    # and is NOT skipped: a platform that cannot decode it will fail loudly.
    qtcore, qtgui, _app = qt
    image, err = _decode(qtcore, qtgui, res.icon_bytes("zecalibrator.icns"))
    assert image is not None, f"zecalibrator.icns: decode failed ({err})"
    assert image.width() > 0 and image.height() > 0
