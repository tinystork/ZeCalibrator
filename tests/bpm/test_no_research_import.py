"""AST proof: no ``src/`` module imports ``research/*`` (mission §85/§86).

Also asserts the ``zecalibrator.bpm`` package imports neither ``research.*`` nor
Qt/GUI (headless, mission §6). These are structural (AST) checks, not runtime
imports, so they hold even if a module is never exercised.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
BPM_ROOT = SRC_ROOT / "zecalibrator" / "bpm"


def _py_files(root: Path):
    return sorted(Path(root).rglob("*.py"))


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


def _is_research(module: str) -> bool:
    return module == "research" or module.startswith("research.")


def _research_offenders(root: Path) -> dict:
    offenders: dict[str, set[str]] = {}
    for path in _py_files(root):
        for mod in _imported_modules(path):
            if _is_research(mod):
                offenders.setdefault(str(path), set()).add(mod)
    return offenders


def test_no_src_module_imports_research():
    """No ``src/`` module imports ``research`` or ``research.*`` at all."""
    offenders = _research_offenders(SRC_ROOT)
    assert not offenders, f"src/ modules import research: {offenders}"


def test_bpm_package_imports_no_research():
    """The product package ``zecalibrator.bpm`` imports no ``research.*``."""
    offenders = _research_offenders(BPM_ROOT)
    assert not offenders, f"zecalibrator.bpm imports research: {offenders}"


def test_bpm_is_headless():
    """The BPM core imports no Qt/GUI (headless, mission §6)."""
    qt_prefixes = ("PySide6", "PyQt5", "PyQt6", "PySide2", "tkinter")
    offenders: dict[str, set[str]] = {}
    for path in _py_files(BPM_ROOT):
        for mod in _imported_modules(path):
            top = mod.split(".")[0]
            if top in qt_prefixes or top == "zecalibrator.gui":
                offenders.setdefault(str(path), set()).add(mod)
    assert not offenders, f"zecalibrator.bpm imports Qt/GUI: {offenders}"


def test_bpm_package_has_expected_modules():
    """Sanity: the product package exposes its four clear modules."""
    expected = {"__init__.py", "errors.py", "vocabulary.py", "identity.py", "revision.py", "lookup.py", "store.py", "settings.py"}
    actual = {p.name for p in BPM_ROOT.glob("*.py")}
    assert expected <= actual
