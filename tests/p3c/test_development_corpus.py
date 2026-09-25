"""P3C-2 — DEVELOPMENT corpus + feature summary artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.p3b.dataset_split import DATASET_DEVELOPMENT
from research.p3c.development_corpus import (
    DEVELOPMENT_CLASSES,
    DEVELOPMENT_SEEDS,
    SUMMARY_SCHEMA,
    build_admission,
    build_development_manifest,
    build_development_scenario,
    compute_feature_summary,
    development_scenarios,
)
from research.p3c.inference_contract import AdmissionFacts


def test_covers_all_20_classes_and_multiple_seeds():
    assert len(DEVELOPMENT_CLASSES) == 20
    assert len(DEVELOPMENT_SEEDS) >= 3
    scenarios = development_scenarios()
    assert len(scenarios) == len(DEVELOPMENT_CLASSES) * len(DEVELOPMENT_SEEDS)
    classes = {s.sites[0].cfa_class for s in scenarios}
    assert classes == set(DEVELOPMENT_CLASSES)


def test_manifest_role_is_development_only():
    manifest = build_development_manifest()
    assert manifest.role == DATASET_DEVELOPMENT
    # Never QUALIFICATION or HOLDOUT.
    assert manifest.role not in ("QUALIFICATION", "HOLDOUT")


def test_scenario_is_deterministic():
    a = build_development_scenario("INTERMITTENT_TWO_STATE", 0)
    b = build_development_scenario("INTERMITTENT_TWO_STATE", 0)
    assert a == b


def test_admission_is_built_from_declared_structure():
    scenario = build_development_scenario("STABLE_ANOMALY_WITH_MISMATCHED_DARK", 0)
    adm = build_admission(scenario)
    assert isinstance(adm, AdmissionFacts)
    assert adm.calibration_representativeness == "NOT_REPRESENTATIVE"
    assert adm.calibration_present == "YES"
    # 2 epochs + 1 calibration epoch = 3 total epochs; 4 independent science groups.
    assert adm.epoch_count == 3
    assert adm.independent_group_count == 4


def test_summary_schema_and_per_class_coverage():
    summary = compute_feature_summary()
    assert summary["summary_schema"] == SUMMARY_SCHEMA
    assert summary["dataset_role"] == DATASET_DEVELOPMENT
    assert summary["dataset_manifest_hash"]
    # Every class and every feature is covered.
    for feature, info in summary["features"].items():
        assert set(info["per_class"]) == set(DEVELOPMENT_CLASSES)
        assert info["unit"]
        assert info["description"]


def test_summary_artifact_file_exists_and_parses():
    path = Path(__file__).resolve().parents[2] / "research" / "p3c" / "development_feature_summary.json"
    assert path.exists(), "development_feature_summary.json artifact must be committed"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary_schema"] == SUMMARY_SCHEMA
    assert payload["dataset_role"] == DATASET_DEVELOPMENT


def test_candidate_parameters_are_recorded_in_summary():
    summary = compute_feature_summary()
    assert summary["candidate_parameters"]
    for cid, cfg in summary["candidate_parameters"].items():
        assert cfg["candidate_id"] == cid
        assert cfg["version"]
        for p in cfg["parameters"].values():
            assert p["kind"] == "RESEARCH_CANDIDATE_PARAMETER"
