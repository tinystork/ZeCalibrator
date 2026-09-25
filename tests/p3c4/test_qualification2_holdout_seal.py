"""P3C-4 LOT 4 — HOLDOUT seal tests for ``research/p3c4``: structural, not a promise.

The QUALIFICATION-2 builder must never open, build, import or reference the
HOLDOUT dataset (§29). This mirrors the P3C seal
(``tests/p3c/test_qualification_holdout_seal.py``) for the new ``research/p3c4``
package: the seal is checked on **code**, not on prose. Docstrings/comments that
*describe* the prohibition are tolerated; a real leak is an ``import``, an
identifier, or a code string constant that names the holdout.

The QUALIFICATION-2 manifest also carries ``holdout_used = false``; this test
makes that assertion machine-verifiable (re-tested, not re-declared).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from research.p3c4.qualification2_corpus import (
    build_qualification2_manifest,
    qualification2_scenarios,
)

P3C4_DIR = Path(__file__).resolve().parents[2] / "research" / "p3c4"

_HOLDOUT_SUBSTRINGS = ("holdout", "HOLDOUT")

# The single negative assertion key the manifest MUST carry (§28/§30):
# ``holdout_used = false`` is the machine-readable claim that the holdout was
# NOT opened. It is a *negative assertion*, not a holdout access path, so it is
# whitelisted from the string scan. Every actual access vector (import, dataset
# role, builder, path/manifest/seed) remains forbidden.
_ALLOWED_ASSERTION_TOKENS = ("holdout_used",)


def _p3c4_python_sources():
    return sorted(P3C4_DIR.glob("*.py"))


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _docstring_values(tree: ast.Module):
    values = set()

    def _collect(body):
        if body and _is_docstring(body[0]):
            values.add(body[0].value.value)

    _collect(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect(node.body)
    return values


def _import_names(tree: ast.Module):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _identifier_tokens(tree: ast.Module):
    tokens = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            tokens.append(getattr(node, "id", None) or getattr(node, "attr", None))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            tokens.append(node.name)
        elif isinstance(node, ast.arg):
            tokens.append(node.arg)
    return [t for t in tokens if t]


def _code_string_constants(tree: ast.Module):
    docs = _docstring_values(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docs:
                out.append(node.value)
    return out


def _holdout_leaks(text: str) -> dict:
    tree = ast.parse(text)
    imports = [
        name
        for name in _import_names(tree)
        if name == "research.p3b.holdout" or name.endswith(".holdout")
    ]
    identifiers = [
        t for t in _identifier_tokens(tree) if any(s in t for s in _HOLDOUT_SUBSTRINGS)
    ]
    strings = [
        s
        for s in _code_string_constants(tree)
        if any(x in s for x in _HOLDOUT_SUBSTRINGS)
        and s not in _ALLOWED_ASSERTION_TOKENS
    ]
    return {"imports": imports, "identifiers": identifiers, "strings": strings}


def test_no_p3c4_module_code_references_the_holdout():
    offending = {}
    for path in _p3c4_python_sources():
        leaks = _holdout_leaks(path.read_text(encoding="utf-8"))
        if any(leaks.values()):
            offending[str(path.name)] = leaks
    assert not offending, (
        f"research/p3c4 module code references the holdout (seal is structural): {offending}"
    )


def test_no_p3c4_module_imports_the_holdout_builder():
    for path in _p3c4_python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _import_names(tree):
            assert imported != "research.p3b.holdout", f"{path.name} imports the holdout builder"
            assert not imported.endswith(".holdout"), f"{path.name} imports {imported!r}"


def test_qualification2_builder_has_no_holdout_parameter_or_reference():
    # The builder's public functions take no holdout path/manifest/seed set.
    for fn in (build_qualification2_manifest, qualification2_scenarios):
        sig = inspect.signature(fn)
        for name in sig.parameters:
            assert "holdout" not in name.lower(), f"{fn.__name__} parameter {name!r}"


def test_manifest_holdout_used_is_false():
    manifest = build_qualification2_manifest()
    assert manifest["verified_assertions"]["holdout_used"] is False


__all__ = []
