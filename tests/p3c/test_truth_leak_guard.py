"""P3C-1 guard: the inference module cannot read truth (§16) — structural, not declarative.

Four layers, two of which close the F1 blind spots (strings and dynamic imports):

* **AST — static imports** : the inference module must not ``import`` /
  ``from ... import`` any truth-bearing module (``research.p3b.declared_facts``,
  ``catalog``, ``fixtures``, ``metrics``, or the ``research.p3c.oracle`` table).

* **AST — identifiers** : no ``Name``/``Attribute``/function/class/argument may
  reference a truth symbol (``_CLASS_FACTS``, ``cfa_class``, ``expected_*``,
  ``scenario_name``, ``compare_to_truth``, …).

* **AST — string constants** (F1) : no *code* string (``ast.Constant`` of type
  ``str``) may contain the truth vocabulary. Docstrings are **prose that
  describes the prohibition**, so they are excluded — but a string in code
  (``getattr(obj, "cfa_class")``, ``{"cfa_class": …}``) is caught. This is the
  structural hook that a dynamic-import string would also hit.

* **Runtime** (F1) : importing the inference module must **not** pull any
  ``research.p3b.*`` module into ``sys.modules`` (catches ``import_module``),
  and no module attribute may be a string containing the truth vocabulary.

The witness test (S1) proves the guard *detects* both a string leak and a
dynamic-import leak — so the blind spot is itself under test and cannot reopen.

The metrics module (a later lot) may compare inference vs truth *after the
fact*; the inference module can never. That asymmetry is exactly what this guard
enforces for the inference side.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

INFERENCE_MODULES = [
    "research.p3c.inference_contract",
]

# Modules whose import would leak truth into the inference side.
FORBIDDEN_MODULES = (
    "research.p3b.declared_facts",   # _CLASS_FACTS
    "research.p3b.catalog",          # expected labels
    "research.p3b.fixtures",         # expected_truth / scenario_for_class
    "research.p3b.metrics",          # compare_to_truth
    "research.p3c.oracle",           # the truth-side oracle table
)

# Truth symbols / substrings the inference source must never reference.
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
    """True iff ``stmt`` is a docstring (an ``Expr`` wrapping a str ``Constant``)."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _docstring_values(tree: ast.Module):
    """Collect every string value that is a docstring (prose, tolerated)."""
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
    """All str constants in *code* (docstrings excluded)."""
    docstrings = _docstring_values(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                out.append(node.value)
    return out


def _leak_report(source: str):
    """Return ``{kind: list}`` of leaks detected in ``source``.

    Pure and side-effect-free, so the S1 witness can call it directly.
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
def test_inference_module_has_no_truth_leaks(module):
    report = _leak_report(_source_path(module).read_text())
    assert report == {"imports": [], "identifiers": [], "strings": []}, report


def test_inference_contract_imports_nothing_from_p3b_at_all():
    # Stronger than the module list: the inference contract is fully self-contained.
    tree = ast.parse(_source_path("research.p3c.inference_contract").read_text())
    for imported in _module_import_names(tree):
        assert not imported.startswith("research.p3b"), (
            f"inference contract imports from research.p3b: {imported!r}"
        )
        assert not imported.startswith("research.p3c.oracle"), (
            f"inference contract imports the oracle: {imported!r}"
        )


# ---------------------------------------------------------------------------
# Runtime layer (F1)
# ---------------------------------------------------------------------------


def test_importing_inference_module_pulls_no_p3b_into_sys_modules():
    # Import the inference contract in a fresh interpreter and assert no
    # research.p3b.* module is present. This is what catches a dynamic
    # ``importlib.import_module("research.p3b.…")`` that the static AST scan
    # cannot see (the module is only *executed* at runtime).
    code = (
        "import sys\n"
        "import research.p3c.inference_contract\n"
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
        f"importing the inference contract pulled P3B truth into sys.modules: {result.stdout!r}"
    )


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_inference_module_namespace_has_no_truth_symbols(module):
    mod = importlib.import_module(module)
    for name in dir(mod):
        # Docstrings (``__doc__`` and other dunder metadata) are tolerated prose:
        # they *describe* the prohibition. Only real data attributes are checked.
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


def test_inference_result_carries_no_truth_fields():
    from research.p3c.inference_contract import InferredEvidence, InferredField

    for cls in (InferredField, InferredEvidence):
        fields = getattr(cls, "__dataclass_fields__", {}).keys()
        for fname in fields:
            assert "cfa_class" not in fname, f"{cls.__name__}.{fname} leaks the class label"
            assert "expected_" not in fname, f"{cls.__name__}.{fname} leaks expected truth"
            assert "scenario" not in fname, f"{cls.__name__}.{fname} leaks the scenario name"


# ---------------------------------------------------------------------------
# Witness (S1): the guard actually *detects* the two blind-spot leaks
# ---------------------------------------------------------------------------


def test_witness_guard_detects_string_leak():
    # A code string carrying truth vocabulary — the exact getattr(...,"cfa_class")
    # form — must be flagged. Without this witness, the AST-string layer could
    # silently regress and the blind spot would reopen untested.
    report = _leak_report(
        'def infer(obj):\n    return getattr(obj, "cfa_class")\n'
    )
    assert report["strings"], "string leak not detected"


def test_witness_guard_detects_dynamic_import_leak():
    # A dynamic import of a truth module. The static-import scan misses it (it
    # is a string argument, not an Import node), so it must be caught by the
    # string layer + the runtime sys.modules layer.
    report = _leak_report(
        'import importlib\nimportlib.import_module("research.p3b.declared_facts")\n'
    )
    assert report["strings"], "dynamic-import string leak not detected by the AST string layer"


def test_witness_static_import_is_still_detected():
    report = _leak_report("from research.p3b.declared_facts import _CLASS_FACTS\n")
    assert report["imports"], "static truth import not detected"
