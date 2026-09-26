"""P4-A2 structural guards (mission §85 + review carryover §7.3).

1. The new product modules (``reconstruction`` / ``preparation`` /
   ``prepared_result``) import no ``research/*`` and are headless (no Qt/GUI).
2. The revision-integrity guard is **named and exercised**: ``Revision.verify``
   (which recomputes :func:`~zecalibrator.bpm.revision.revision_digest`) rejects a
   tampered revision with :class:`~zecalibrator.bpm.errors.BpmBaseCorrupted` —
   the guard Nono's review could not name.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from zecalibrator.bpm.errors import BpmBaseCorrupted
from zecalibrator.bpm.revision import Revision, make_revision, revision_digest
from zecalibrator.bpm.vocabulary import REVISION_STATE_PROMOTED

from conftest import make_identity, make_site

BPM_ROOT = Path(__file__).resolve().parents[2] / "src" / "zecalibrator" / "bpm"
NEW_MODULES = ("reconstruction.py", "preparation.py", "prepared_result.py")


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


def test_new_modules_import_no_research():
    offenders: dict = {}
    for name in NEW_MODULES:
        path = BPM_ROOT / name
        for mod in _imported_modules(path):
            if mod == "research" or mod.startswith("research."):
                offenders.setdefault(name, set()).add(mod)
    assert not offenders, f"new product modules import research: {offenders}"


def test_new_modules_are_headless():
    qt_prefixes = ("PySide6", "PyQt5", "PyQt6", "PySide2", "tkinter")
    offenders: dict = {}
    for name in NEW_MODULES:
        path = BPM_ROOT / name
        for mod in _imported_modules(path):
            top = mod.split(".")[0]
            if top in qt_prefixes or top == "zecalibrator.gui":
                offenders.setdefault(name, set()).add(mod)
    assert not offenders, f"new product modules import Qt/GUI: {offenders}"


def test_revision_integrity_guard_named_and_rejects_tamper():
    # ``Revision.verify`` is the named integrity guard (it recomputes
    # ``revision_digest``). A tampered revision must be refused, never applied.
    ident = make_identity(shape=(12, 12))
    rev = make_revision(
        state=REVISION_STATE_PROMOTED, sensor_identity=ident, sites=[make_site(5, 5)], sequence=1,
    )
    assert isinstance(rev, Revision)
    rev.verify()  # the intact revision verifies

    # A byte-identical revision built from different evidence carries a different
    # digest; an altered record with a stale digest must fail verification.
    tampered = Revision(
        revision_id=rev.revision_id,
        state=REVISION_STATE_PROMOTED,
        sensor_identity=ident,
        sites=(make_site(1, 1),),  # different content, same recorded id/digest
        integrity_digest=rev.integrity_digest,
        schema_version=rev.schema_version,
        sequence=rev.sequence,
    )
    with pytest.raises(BpmBaseCorrupted):
        tampered.verify()

    # The named digest function is deterministic over identical evidence.
    assert revision_digest(REVISION_STATE_PROMOTED, ident, [make_site(5, 5)]) == rev.integrity_digest


def test_operator_version_is_documented_on_default():
    from zecalibrator.bpm.reconstruction import DEFAULT_OPERATOR, OPERATOR_SCHEMA_VERSION

    assert DEFAULT_OPERATOR.version == OPERATOR_SCHEMA_VERSION
