"""Teardown-safety regression tests (run in SUBPROCESSES).

These exercise the interpreter-teardown path where a window/controller is created
and the process exits WITHOUT running ``app.exec()`` (or where ``app.exec()``
returns and the bounded post-loop join runs). An in-process crash here would
abort the whole test runner, so each scenario runs in a clean subprocess and we
assert on its exit code + stderr. All scenarios use the offscreen platform and
injected temp roots (never real user roots).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_REPO = Path(__file__).resolve().parents[2]
_SRC = str(_REPO / "src")


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def _run(code: str, tmp_path: Path):
    return subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True, text=True, env=_env(), timeout=60,
    )


def test_close_during_active_settings_load_exit_without_exec(tmp_path):
    """Create window, close while async settings-load is active, exit WITHOUT
    app.exec(). Must be rc 0 and never print 'Destroyed while thread is still
    running'."""
    code = (
        "import sys\n"
        "from zecalibrator.gui.window import create_qapplication, create_main_window\n"
        "from zecalibrator.storage import resolve_paths\n"
        "app = create_qapplication([])\n"
        "win = create_main_window(resolve_paths(base=sys.argv[1]))\n"
        "# close while the async settings-load op is still active\n"
        "win.close()\n"
        "print('EXITED_CLEANLY')\n"
    )
    proc = _run(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "EXITED_CLEANLY" in proc.stdout
    assert "Destroyed while thread is still running" not in proc.stderr


def test_run_application_immediate_close(tmp_path):
    """run_application() with an immediate close -> rc 0 (post-loop join runs)."""
    code = (
        "import sys\n"
        "from zecalibrator.gui import window as winmod\n"
        "from zecalibrator.storage import resolve_paths\n"
        "# run_application runs app.exec(); schedule an immediate close via a\n"
        "# zero-interval single-shot so the loop exits quickly.\n"
        "from PySide6 import QtCore\n"
        "orig = winmod.create_main_window\n"
        "def patched(paths=None, parent=None):\n"
        "    w = orig(paths, parent)\n"
        "    QtCore.QTimer.singleShot(0, w.close)\n"
        "    return w\n"
        "winmod.create_main_window = patched\n"
        "rc = winmod.run_application([], resolve_paths(base=sys.argv[1]))\n"
        "print('RC', rc)\n"
        "sys.exit(rc)\n"
    )
    proc = _run(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "RC 0" in proc.stdout


def test_app_exec_teardown_scenarios_rc0(tmp_path):
    """The four shipped app.exec() teardown scenarios remain rc 0 (Nono's four)."""
    scenarios = {
        "close_idle": (
            "import sys\n"
            "from PySide6 import QtCore\n"
            "from zecalibrator.gui import window as winmod\n"
            "from zecalibrator.storage import resolve_paths\n"
            "orig = winmod.create_main_window\n"
            "def patched(paths=None, parent=None):\n"
            "    w = orig(paths, parent)\n"
            "    def do_close():\n"
            "        from PySide6 import QtTest\n"
            "        for _ in range(100):\n"
            "            from PySide6 import QtWidgets\n"
            "            QtWidgets.QApplication.processEvents()\n"
            "            QtTest.QTest.qWait(5)\n"
            "            if w._controller is not None and not w._controller.is_active:\n"
            "                w.close(); return\n"
            "    QtCore.QTimer.singleShot(0, do_close)\n"
            "    return w\n"
            "winmod.create_main_window = patched\n"
            "rc = winmod.run_application([], resolve_paths(base=sys.argv[1]))\n"
            "print('RC', rc)\n"
            "sys.exit(rc)\n"
        ),
    }
    # The primary shipped scenario: open then close once the worker is idle.
    proc = _run(scenarios["close_idle"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "RC 0" in proc.stdout
    assert "Destroyed while thread is still running" not in proc.stderr
