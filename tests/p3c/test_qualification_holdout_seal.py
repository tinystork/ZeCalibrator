"""P3C-3 — HOLDOUT seal tests (§36–§37 / §42): structural, not a promise.

The seal is checked the same way the P3C-1 truth-leak guard checks the inference
side: on **code**, not on prose. Docstrings and comments that *describe the
prohibition* are tolerated (they explain why the holdout is never touched); a
real leak is an ``import``, an identifier, or a code string constant that names
the holdout. That is exactly what a future re-opener would need to write to
touch the holdout, so this test makes the seal machine-verifiable.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from research.p3b.dataset_split import DATASET_HOLDOUT, DATASET_QUALIFICATION
from research.p3c.qualification_corpus import (
    QUALIFICATION_SEEDS,
    build_qualification_manifest,
    qualification_scenarios,
)
from research.p3c.qualification_runner import run_qualification_campaign

P3C_DIR = Path(__file__).resolve().parents[2] / "research" / "p3c"

_HOLDOUT_SUBSTRINGS = ("holdout", "HOLDOUT")


def _p3c_python_sources():
    return sorted(P3C_DIR.glob("*.py"))


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
    """Code string constants (docstrings excluded — they are tolerated prose)."""
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
        s for s in _code_string_constants(tree) if any(x in s for x in _HOLDOUT_SUBSTRINGS)
    ]
    return {"imports": imports, "identifiers": identifiers, "strings": strings}


# ---------------------------------------------------------------------------
# No research/p3c module *code* names the holdout (structural, token-level)
# ---------------------------------------------------------------------------


def test_no_p3c_module_code_references_the_holdout():
    offending = {}
    for path in _p3c_python_sources():
        leaks = _holdout_leaks(path.read_text(encoding="utf-8"))
        if any(leaks.values()):
            offending[str(path.name)] = leaks
    assert not offending, (
        f"research/p3c module code references the holdout (seal is structural): {offending}"
    )


def test_no_p3c_module_imports_the_holdout_builder():
    for path in _p3c_python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _import_names(tree):
            assert imported != "research.p3b.holdout", f"{path.name} imports the holdout builder"
            assert not imported.endswith(".holdout"), f"{path.name} imports {imported!r}"


# ---------------------------------------------------------------------------
# The runner's API cannot receive a holdout (§36–§37: even the path is refused)
# ---------------------------------------------------------------------------


def test_runner_signature_has_no_holdout_parameter():
    sig = inspect.signature(run_qualification_campaign)
    for name in sig.parameters:
        assert "holdout" not in name.lower(), f"runner parameter {name!r} names the holdout"
    names = set(sig.parameters)
    assert "holdout" not in names
    assert "holdout_manifest" not in names
    assert "holdout_path" not in names


def test_qualification_manifest_role_is_qualification_not_holdout():
    manifest = build_qualification_manifest()
    assert manifest.role == DATASET_QUALIFICATION
    assert manifest.role != DATASET_HOLDOUT


def test_qualification_seeds_are_disjoint_from_development():
    from research.p3c.development_corpus import DEVELOPMENT_SEEDS

    assert set(QUALIFICATION_SEEDS).isdisjoint(set(DEVELOPMENT_SEEDS))
    assert len(set(QUALIFICATION_SEEDS)) == len(QUALIFICATION_SEEDS)


def test_qualification_corpus_covers_all_20_classes():
    scenarios = qualification_scenarios()
    classes = {s.sites[0].cfa_class for s in scenarios}
    assert len(classes) == 20
    assert len(scenarios) == 20 * len(QUALIFICATION_SEEDS)
