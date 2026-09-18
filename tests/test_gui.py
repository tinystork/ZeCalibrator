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


def test_gui_missing_extra_diagnostic_deterministic_with_blocked_import():
    """The missing-extra diagnostic is deterministic even on a Qt-installed host.

    Block the ``PySide6`` import via a meta-path finder so the diagnostic is
    exercised regardless of whether the ``[gui]`` extra is present in this
    environment (isolated blocked import).
    """
    import textwrap

    code = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_module(self, fullname, path=None):
                if fullname == "PySide6" or fullname.startswith("PySide6."):
                    return self
                return None

            def find_spec(self, fullname, path=None, target=None):
                if fullname == "PySide6" or fullname.startswith("PySide6."):
                    raise ImportError("blocked PySide6 for test")
                return None

            def load_module(self, fullname):
                raise ImportError("blocked PySide6 for test")

        sys.meta_path.insert(0, Blocker())
        from zecalibrator.gui.app import main
        sys.exit(main())
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 1
    assert "gui" in proc.stderr.lower()
    assert "PySide6" in proc.stderr
    assert "[gui]" in proc.stderr
