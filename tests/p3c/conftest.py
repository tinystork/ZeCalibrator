"""Shared fixtures/helpers for tests/p3c.

Adds the repository root to ``sys.path`` so ``research.p3c`` (non-packaged) is
importable, and exposes a ``features_factory`` that materialises one DEVELOPMENT
scenario's LOT4 features for a given class.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def features_factory():
    """Return a callable ``features_factory(class_name, seed=0) -> SiteFeatures``.

    Builds a single DEVELOPMENT scenario for ``class_name``, materialises its
    frames, and returns the LOT4 ``SiteFeatures`` for its one site. Uses a
    temporary directory for FITS output (cleaned up automatically).
    """
    from research.p3c.development_corpus import build_development_scenario
    from research.p3b.features import compute_features
    from research.p3b.generator import generate_corpus

    def _factory(class_name: str, seed: int = 0):
        scenario = build_development_scenario(class_name, seed)
        with tempfile.TemporaryDirectory() as td:
            corpus = generate_corpus(scenario, td)
            dark_ids = [f"dark{i}" for i in range(4)]
            dark = np.median(np.stack([corpus.frame_arrays[i] for i in dark_ids]), axis=0)
            cf = compute_features(scenario, corpus.frame_arrays, dark_reference=dark)
        return cf.sites[0]

    return _factory


__all__ = ["REPO_ROOT"]
