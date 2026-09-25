"""LOT1 tests: determinism + catalogue + autonomy + parameters."""

import hashlib

import pytest

from research.p3b.catalog import CLASS_NAMES, UnknownClassError, resolve_class
from research.p3b.generator import generate_corpus
from research.p3b.parameters import (
    EXPLORATORY,
    SYNTHETIC_GENERATOR_PARAMETER,
    all_params,
)
from tests.p3b.conftest import single_light_scenario


def _sha256(path):
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Determinism (LOT1 requirement 1)
# ---------------------------------------------------------------------------

def test_same_seed_byte_identical_fits_and_manifest(tmp_path):
    scenario = single_light_scenario(seed=42, n_frames=3)
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    r1 = generate_corpus(scenario, d1)
    r2 = generate_corpus(scenario, d2)

    assert _sha256(r1.manifest_path) == _sha256(r2.manifest_path)
    assert set(r1.frame_paths) == set(r2.frame_paths)
    for fid in r1.frame_paths:
        assert _sha256(r1.frame_paths[fid]) == _sha256(r2.frame_paths[fid])


def test_different_seeds_differ(tmp_path):
    s1 = single_light_scenario(seed=1, n_frames=3)
    s2 = single_light_scenario(seed=2, n_frames=3)
    r1 = generate_corpus(s1, tmp_path / "a")
    r2 = generate_corpus(s2, tmp_path / "b")
    assert _sha256(r1.frame_paths["l0"]) != _sha256(r2.frame_paths["l0"])


# ---------------------------------------------------------------------------
# Catalogue (LOT1 requirement 9)
# ---------------------------------------------------------------------------

EXPECTED_20 = [
    "NORMAL",
    "STABLE_ANOMALY_CORRECTED_BY_DARK",
    "STABLE_ANOMALY_WITH_MISMATCHED_DARK",
    "INTERMITTENT_TWO_STATE",
    "INTERMITTENT_MULTI_STATE",
    "INTERMITTENT_CONTINUOUS",
    "RARE_HIGH_STATE",
    "RARE_LOW_STATE",
    "SIGN_CHANGING_POST_DARK",
    "CENSORED_ANOMALY",
    "SINGLE_TRANSIENT",
    "OPTICAL_STRUCTURE",
    "STAR_CROSSING_SITE",
    "UNDERSAMPLED_STAR_CORE",
    "COSMIC_RAY",
    "NOISE_EXTREME",
    "FLAT_STRUCTURE",
    "DUST_OR_VIGNETTING",
    "NEAR_SATURATION",
    "AMBIGUOUS_INSUFFICIENT_EVIDENCE",
]


def test_catalog_declares_exactly_20_classes():
    assert len(CLASS_NAMES) == 20
    assert list(CLASS_NAMES) == EXPECTED_20


def test_unknown_class_raises_typed_error():
    with pytest.raises(UnknownClassError):
        resolve_class("NOT_A_CLASS")


def test_unknown_class_is_key_error_subtype():
    assert issubclass(UnknownClassError, KeyError)


def test_every_class_resolves_to_an_entry():
    for name in CLASS_NAMES:
        entry = resolve_class(name)
        assert entry.name == name
        assert entry.description


# ---------------------------------------------------------------------------
# Autonomy (LOT1 requirement 2, tested without network/ZeAlfie/ZSSS/GPU)
# ---------------------------------------------------------------------------

def test_module_imports_and_runs_without_zealfie_zsss_gpu(tmp_path):
    # The harness must run on CPU with numpy/astropy only. Importing research.p3b
    # must not pull in ZeAlfie/ZSSS/torch/tensorflow/network modules.
    import sys

    scenario = single_light_scenario(seed=0, n_frames=2)
    r = generate_corpus(scenario, tmp_path / "corpus")
    assert r.bookkeeping.frame_count == 2

    # No GPU/network/external-service imports anywhere in the harness package.
    forbidden = (
        "zealfie",
        "zsss",
        "torch",
        "tensorflow",
        "tensorrt",
        "cuda",
        "requests",
        "urllib",
        "socket",
        "aiohttp",
    )
    import research.p3b as pkg
    import inspect

    offenders = []
    for modname in ("research.p3b.catalog", "research.p3b.model", "research.p3b.parameters",
                    "research.p3b.behaviors", "research.p3b.generator"):
        if modname not in sys.modules:
            continue
        src = inspect.getsource(sys.modules[modname]).lower()
        for token in forbidden:
            if token in src:
                offenders.append((modname, token))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# Parameter marking (LOT1 requirement 10)
# ---------------------------------------------------------------------------

def test_parameters_are_marked_and_never_product_threshold():
    for name, (value, kind) in all_params().items():
        assert kind in (SYNTHETIC_GENERATOR_PARAMETER, EXPLORATORY), name
        # Nothing is presented or named as a PRODUCT THRESHOLD.
        assert "THRESHOLD" not in name.upper()
        assert kind != "PRODUCT THRESHOLD"
        assert "PRODUCT THRESHOLD" not in str(value).upper()
