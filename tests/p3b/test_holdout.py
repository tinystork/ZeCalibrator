"""LOT7 tests: holdout independence (property 3).

The holdout-producing code must be runnable without consulting any evaluation
result: no development result, no candidate threshold, no operating point, no
evaluation artifact. Tested by function signature + AST (the module cannot read
a report, a threshold or an operating point).
"""

import ast
import inspect

import research.p3b.holdout as holdout_module
from research.p3b.dataset_split import DATASET_HOLDOUT
from research.p3b.model import (
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SiteSpec,
)
from tests.p3b.conftest import make_sensor


def _site_scenario(name, seed):
    sensor = make_sensor(shape=(48, 64))
    sites = (
        SiteSpec(site_id="s0", x=10, y=12, cfa_class="STABLE_ANOMALY_WITH_MISMATCHED_DARK"),
    )
    frames = tuple(
        FrameSpec(frame_id=f"l{i}", frame_type="light", epoch_id="e0", group_id="g0", ordinal=i)
        for i in range(3)
    )
    return ScenarioSpec(
        name=name,
        seed=seed,
        sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
        sites=sites,
    )


# The evaluation lots that the holdout module must never depend on.
EVALUATION_MODULES = {
    "qualification_policy",
    "features",
    "declared_facts",
    "metrics",
    "net_benefit",
    "preparation_plan",
    "reconstruction",
}


def _imported_modules(module):
    tree = ast.parse(inspect.getsource(module))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def test_holdout_module_imports_no_evaluation_module():
    imports = _imported_modules(holdout_module)
    overlap = imports & EVALUATION_MODULES
    assert overlap == set(), overlap


def test_holdout_function_accepts_only_declared_scenarios():
    sig = inspect.signature(holdout_module.build_holdout_manifest)
    assert set(sig.parameters) == {"scenarios"}
    annotation = str(sig.parameters["scenarios"].annotation)
    # Declared scenarios only — never a report, threshold or operating point.
    assert "ScenarioSpec" in annotation
    for token in ("report", "threshold", "operating_point", "metrics", "decision"):
        assert token not in annotation.lower()


def test_holdout_source_never_reads_report_threshold_or_operating_point():
    tree = ast.parse(inspect.getsource(holdout_module))
    forbidden = {"threshold", "report", "operating_point", "metric", "decision"}
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            offenders.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            offenders.append(node.attr)
    assert offenders == [], offenders


def test_build_holdout_manifest_produces_frozen_holdout_role():
    manifest = holdout_module.build_holdout_manifest((_site_scenario("hold", 17),))
    assert manifest.role == DATASET_HOLDOUT
    assert manifest.independent_real_validation == "NO"
    assert manifest.real_validation_status == "NOT_SATISFIED"
