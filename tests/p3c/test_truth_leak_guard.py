"""P3C-1 guard: the inference module cannot read truth (§16) — structural, not declarative.

Two layers:

* **AST layer** — the inference module's source must not import any truth-bearing
  module nor reference any truth symbol (``_CLASS_FACTS``, ``cfa_class``,
  ``expected_*``, ``scenario_name``, ``compare_to_truth``, the truth manifest,
  catalogue expected labels). This is a static, structural check — a future
  P3C-2 candidate that reaches for truth fails at test time, before running.

* **Runtime layer** — the imported module must not expose any truth symbol in
  its namespace, and the inference result type must carry no ``cfa_class`` /
  ``expected_*`` / scenario name, so the §49/§50 probes (identical measured data
  => identical inference regardless of truth; the class name never drives the
  result) are structurally expressible.

The metrics module (a later lot) may compare inference vs truth *after the
fact*; the inference module can never. That asymmetry is exactly what this guard
enforces for the inference side.
"""

from __future__ import annotations

import ast
import importlib
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


# ---------------------------------------------------------------------------
# AST layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_inference_module_does_not_import_truth_modules(module):
    tree = ast.parse(_source_path(module).read_text())
    for imported in _module_import_names(tree):
        assert not any(
            imported == f or imported.startswith(f + ".")
            for f in FORBIDDEN_MODULES
        ), f"{module} imports forbidden truth module {imported!r}"


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_inference_module_does_not_reference_truth_symbols(module):
    tree = ast.parse(_source_path(module).read_text())
    for token in _identifier_tokens(tree):
        assert not any(sym in token for sym in FORBIDDEN_SYMBOLS), (
            f"{module} references forbidden truth symbol {token!r}"
        )


# ---------------------------------------------------------------------------
# Runtime layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_inference_module_namespace_has_no_truth_symbols(module):
    mod = importlib.import_module(module)
    for name in dir(mod):
        assert not any(sym in name for sym in FORBIDDEN_SYMBOLS), (
            f"{module} exposes forbidden truth symbol {name!r} in its namespace"
        )


def test_inference_result_carries_no_truth_fields():
    from research.p3c.inference_contract import InferredEvidence, InferredField

    for cls in (InferredField, InferredEvidence):
        fields = getattr(cls, "__dataclass_fields__", {}).keys()
        for fname in fields:
            assert "cfa_class" not in fname, f"{cls.__name__}.{fname} leaks the class label"
            assert "expected_" not in fname, f"{cls.__name__}.{fname} leaks expected truth"
            assert "scenario" not in fname, f"{cls.__name__}.{fname} leaks the scenario name"


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
