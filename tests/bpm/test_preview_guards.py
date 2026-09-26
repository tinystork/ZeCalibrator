"""P4-P1 LOT 1 structural guards.

1. §35: no CLI/GUI caller can reach ``apply_preparation`` directly, bypassing the
   orchestrator — the only application-layer import surface for the BPM engine is
   the orchestrator (``application/bpm_preview.py``), never ``cli.py`` / ``gui/*``.
2. The new orchestrator module imports no ``research/*`` and is headless (no Qt).
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
APPLICATION_ROOT = SRC_ROOT / "zecalibrator" / "application"
CLI_PATH = SRC_ROOT / "zecalibrator" / "cli.py"
GUI_ROOT = SRC_ROOT / "zecalibrator" / "gui"


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").lstrip(".")
            if module:
                yield module


def _imports_bpm_apply(path: Path) -> bool:
    """Whether ``path`` imports ``apply_preparation`` from the BPM engine."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").lstrip(".")
            if module == "zecalibrator.bpm.preparation" or module.startswith("zecalibrator.bpm.preparation."):
                names = [a.name for a in node.names]
                if "apply_preparation" in names:
                    return True
            if module.startswith("zecalibrator.bpm"):
                names = [a.name for a in node.names]
                if "apply_preparation" in names:
                    return True
    return False


def _py_files(root: Path):
    return sorted(Path(root).rglob("*.py"))


def test_no_cli_or_gui_imports_apply_preparation():
    # §35: the only module permitted to touch the engine's apply entry is the
    # orchestrator; CLI and GUI must never reach it directly.
    offenders = []
    for path in [CLI_PATH] + _py_files(GUI_ROOT):
        if _imports_bpm_apply(path):
            offenders.append(str(path))
    assert not offenders, f"CLI/GUI import apply_preparation directly: {offenders}"


def test_cli_imports_only_api_v1():
    # The CLI is a thin facade over api.v1 only (mission §35 / frozen G6).
    offenders = set()
    for mod in _imported_modules(CLI_PATH):
        if mod.startswith("zecalibrator.") and not mod.startswith("zecalibrator.api"):
            offenders.add(mod)
    assert not offenders, f"cli.py imports non-api zecalibrator modules: {offenders}"


def test_gui_does_not_import_bpm_engine():
    # The GUI reaches calibration through api.v1; never the BPM engine directly.
    offenders = set()
    for path in _py_files(GUI_ROOT):
        for mod in _imported_modules(path):
            if mod == "zecalibrator.bpm" or mod.startswith("zecalibrator.bpm."):
                offenders.add(str(path))
    assert not offenders, f"gui/ imports the BPM engine directly: {offenders}"


def test_orchestrator_module_imports_no_research():
    path = APPLICATION_ROOT / "bpm_preview.py"
    offenders = set()
    for mod in _imported_modules(path):
        if mod == "research" or mod.startswith("research."):
            offenders.add(mod)
    assert not offenders, f"bpm_preview imports research: {offenders}"


def test_orchestrator_module_is_headless():
    path = APPLICATION_ROOT / "bpm_preview.py"
    qt_prefixes = ("PySide6", "PyQt5", "PyQt6", "PySide2", "tkinter")
    offenders = set()
    for mod in _imported_modules(path):
        top = mod.split(".")[0]
        if top in qt_prefixes or top == "zecalibrator.gui":
            offenders.add(mod)
    assert not offenders, f"bpm_preview imports Qt/GUI: {offenders}"


def test_gate_is_not_a_user_setting():
    # §23: the gate must NOT be a user setting. ``bpm/settings.py`` exposes only
    # ``bad_pixel_database_root``; no "enable BPM"/"force correction" field exists.
    settings_path = SRC_ROOT / "zecalibrator" / "bpm" / "settings.py"
    text = settings_path.read_text(encoding="utf-8")
    assert "BPM_AUTOMATIC_APPLICATION_ENABLED" not in text
    assert "bad_pixel_database_root" in text
