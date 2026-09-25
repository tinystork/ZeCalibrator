"""P3C-4 LOT 1 — anti-truth-leak guard for ``research.p3c4`` (§16), four layers.

Extends the P3C-1 guard to the new temporal-evidence contract package. The
contract side reads **no truth**: it must never import a truth-bearing module,
never reference a truth symbol, never carry truth vocabulary in a code string,
and must not pull ``research.p3b.*`` into ``sys.modules`` at runtime.

Four layers (mirrors P3C-1, same blind spots closed):

* **AST — static imports**: no ``import`` / ``from ... import`` of a truth module
  (``research.p3b.declared_facts``, ``catalog``, ``fixtures``, ``metrics``, or
  ``research.p3c.oracle``).
* **AST — identifiers**: no ``Name``/``Attribute``/function/class/argument
  referencing a truth symbol (``_CLASS_FACTS``, ``cfa_class``, ``expected_*``,
  ``scenario_name``, ``compare_to_truth``, …).
* **AST — code strings** (F1): no *code* string carries the truth vocabulary
  (docstrings are prose that describe the prohibition, so they are excluded).
* **Runtime** (F1): importing the package must not pull ``research.p3b.*`` into
  ``sys.modules``, and no module attribute may be a string carrying truth
  vocabulary.

Witness tests (S1) prove the guard *detects* a string leak, a dynamic-import
leak and a static-import leak — so the blind spot is itself under test.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

# The P3C-4 contract side. ``research.p3c4`` (the package) is included so its
# re-exports are guarded too.
INFERENCE_MODULES = [
    "research.p3c4",
    "research.p3c4.temporal_evidence",
]

# Modules whose import would leak truth into the contract side.
FORBIDDEN_MODULES = (
    "research.p3b.declared_facts",   # _CLASS_FACTS
    "research.p3b.catalog",          # class labels / catalogue
    "research.p3b.fixtures",         # scenario_for_class / expected truth
    "research.p3b.metrics",          # compare_to_truth
    "research.p3c.oracle",           # the truth-side oracle table
)

# Truth symbols / substrings the contract source must never reference.
FORBIDDEN_SYMBOLS = (
    "_CLASS_FACTS",
    "cfa_class",
    "expected_",
    "scenario_name",
    "compare_to_truth",
    "CLASS_CATALOG",
    "resolve_class",
    "default_expected_",
    "build_declared_facts",
    "evidence_packet",
    "SiteDeclaredFacts",
    "DeclaredFacts",
    "scenario_for_class",
)

# Tokens the *string* layer scans for: truth symbols AND the forbidden module
# names (so a dynamic ``import_module("research.p3b.…")`` string is caught).
FORBIDDEN_STRING_TOKENS = FORBIDDEN_SYMBOLS + FORBIDDEN_MODULES


def _source_path(module: str) -> Path:
    mod = importlib.import_module(module)
    return Path(mod.__file__)


def _module_import_names(tree: ast.Module):
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


def _is_docstring_statement(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _docstring_values(tree: ast.Module):
    values = set()

    def _collect(body):
        if body and _is_docstring_statement(body[0]):
            values.add(body[0].value.value)

    _collect(tree.body)  # module docstring
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect(node.body)
    return values


def _code_string_constants(tree: ast.Module):
    docstrings = _docstring_values(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                out.append(node.value)
    return out


def _leak_report(source: str):
    """Return ``{kind: list}`` of leaks detected in ``source``.

    Pure and side-effect-free, so the witness can call it directly.
    """
    tree = ast.parse(source)
    imports = [
        name
        for name in _module_import_names(tree)
        if any(name == f or name.startswith(f + ".") for f in FORBIDDEN_MODULES)
    ]
    identifiers = [
        t for t in _identifier_tokens(tree) if any(s in t for s in FORBIDDEN_SYMBOLS)
    ]
    strings = [
        s for s in _code_string_constants(tree) if any(sym in s for sym in FORBIDDEN_STRING_TOKENS)
    ]
    return {"imports": imports, "identifiers": identifiers, "strings": strings}


# ---------------------------------------------------------------------------
# AST layer — static imports + identifiers + code strings (F1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_contract_module_has_no_truth_leaks(module):
    report = _leak_report(_source_path(module).read_text())
    assert report == {"imports": [], "identifiers": [], "strings": []}, report


def test_temporal_evidence_imports_nothing_truth_bearing():
    tree = ast.parse(_source_path("research.p3c4.temporal_evidence").read_text())
    for imported in _module_import_names(tree):
        assert not imported.startswith("research.p3b"), (
            f"temporal_evidence imports from research.p3b: {imported!r}"
        )
        assert not imported.startswith("research.p3c.oracle"), (
            f"temporal_evidence imports the oracle: {imported!r}"
        )


# ---------------------------------------------------------------------------
# Runtime layer (F1)
# ---------------------------------------------------------------------------


def test_importing_contract_pulls_no_p3b_into_sys_modules():
    code = (
        "import sys\n"
        "import research.p3c4.temporal_evidence\n"
        "leaks = sorted(m for m in sys.modules if m.startswith('research.p3b'))\n"
        "print('\\n'.join(leaks))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing the contract pulled P3B truth into sys.modules: {result.stdout!r}"
    )


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_contract_module_namespace_has_no_truth_symbols(module):
    mod = importlib.import_module(module)
    for name in dir(mod):
        if name.startswith("__"):
            continue
        assert not any(sym in name for sym in FORBIDDEN_SYMBOLS), (
            f"{module} exposes forbidden truth symbol {name!r} in its namespace"
        )
        value = getattr(mod, name)
        if isinstance(value, str):
            assert not any(sym in value for sym in FORBIDDEN_STRING_TOKENS), (
                f"{module}.{name} is a string carrying truth vocabulary: {value!r}"
            )


def test_result_carries_no_truth_fields():
    from research.p3c4.temporal_evidence import (
        TemporalPersistenceEvidence,
        TemporalSignature,
        GroupSignature,
    )

    for cls in (TemporalPersistenceEvidence, TemporalSignature, GroupSignature):
        fields = getattr(cls, "__dataclass_fields__", {}).keys()
        for fname in fields:
            assert "cfa_class" not in fname, f"{cls.__name__}.{fname} leaks the class label"
            assert "expected_" not in fname, f"{cls.__name__}.{fname} leaks expected truth"
            assert "scenario" not in fname, f"{cls.__name__}.{fname} leaks the scenario name"


# ---------------------------------------------------------------------------
# Witness (S1): the guard actually *detects* the blind-spot leaks
# ---------------------------------------------------------------------------


def test_witness_guard_detects_string_leak():
    report = _leak_report(
        'def infer(obj):\n    return getattr(obj, "cfa_class")\n'
    )
    assert report["strings"], "string leak not detected"


def test_witness_guard_detects_dynamic_import_leak():
    report = _leak_report(
        'import importlib\nimportlib.import_module("research.p3b.declared_facts")\n'
    )
    assert report["strings"], "dynamic-import string leak not detected by the AST string layer"


def test_witness_static_import_is_still_detected():
    report = _leak_report("from research.p3b.declared_facts import _CLASS_FACTS\n")
    assert report["imports"], "static truth import not detected"
