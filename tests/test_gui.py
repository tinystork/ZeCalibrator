"""Bootstrap tests: GUI launcher diagnostic when PySide6 is absent."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _pyside6_absent() -> bool:
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return True
    return False


def test_gui_entrypoint_missing_extra_diagnostic():
    if not _pyside6_absent():
        pytest.skip("PySide6 present; missing-extra path is not applicable")
    code = "from zecalibrator.gui.app import main; import sys; sys.exit(main())"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 1
    assert "gui" in proc.stderr.lower()
    assert "PySide6" in proc.stderr
    assert "[gui]" in proc.stderr


def test_gui_module_import_does_not_import_pyside6():
    code = (
        "import sys\n"
        "import zecalibrator.gui.app\n"
        "assert 'PySide6' not in sys.modules\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
