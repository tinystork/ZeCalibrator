"""Shared fixtures/helpers for tests/p3c4.

Adds the repository root to ``sys.path`` so ``research.p3c4`` (non-packaged) is
importable, mirroring tests/p3c/conftest.py, and exposes a session-scoped frozen
QUALIFICATION-2 campaign fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def frozen_campaign():
    """Freeze the temporal candidates once and run the full QUALIFICATION-2
    campaign once.

    Session-scoped: the campaign is deterministic and ~15s, so it is computed a
    single time and shared by every P3C-4 test that needs the full result.
    """
    from research.p3c4.candidate_freeze import freeze_candidates
    from research.p3c4.qualification2_runner import run_qualification2_campaign

    freeze = freeze_candidates()
    results = run_qualification2_campaign(freeze)
    return freeze, results


@pytest.fixture(scope="session")
def qualification1_regression():
    """Run the QUALIFICATION-1 regression once (session-scoped)."""
    from research.p3c4.regression_qualification1 import run_qualification1_regression

    return run_qualification1_regression()


__all__ = ["REPO_ROOT"]
