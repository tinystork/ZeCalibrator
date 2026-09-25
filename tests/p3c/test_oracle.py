"""P3C-1 tests: the oracle (complete 20-class table, no confounder is a target)."""

from __future__ import annotations

import pytest

from research.p3b.catalog import CLASS_NAMES, resolve_class
from research.p3c.oracle import (
    ACTION_ELIGIBLE,
    CONFOUNDER_CLASSES,
    ORACLE,
    expected_action,
    epistemic_state,
    oracle_entry,
    reconstruction_targets,
)


# ---------------------------------------------------------------------------
# Completeness (§8: the oracle is complete — 20 classes)
# ---------------------------------------------------------------------------


def test_oracle_declares_exactly_the_20_catalogue_classes():
    assert len(ORACLE) == 20
    assert set(ORACLE) == set(CLASS_NAMES)


def test_every_class_has_an_epistemic_state_and_expected_action():
    for name in CLASS_NAMES:
        entry = oracle_entry(name)
        assert entry.epistemic_state
        assert entry.expected_action
        assert entry.justification


# ---------------------------------------------------------------------------
# No confounder is a reconstruction target (§6.3) — the guard
# ---------------------------------------------------------------------------


def test_no_confounder_is_a_reconstruction_target():
    for name in CONFOUNDER_CLASSES:
        entry = oracle_entry(name)
        assert not entry.is_reconstruction_target, (
            f"confounder {name!r} must never be a reconstruction target, "
            f"but the oracle declares {entry.expected_action!r}"
        )


def test_confounder_list_is_exactly_the_eight_safety_classes():
    assert set(CONFOUNDER_CLASSES) == {
        "OPTICAL_STRUCTURE",
        "FLAT_STRUCTURE",
        "DUST_OR_VIGNETTING",
        "COSMIC_RAY",
        "STAR_CROSSING_SITE",
        "UNDERSAMPLED_STAR_CORE",
        "SINGLE_TRANSIENT",
        "NEAR_SATURATION",
    }


# ---------------------------------------------------------------------------
# The three optical/flat/dust confounders are no longer ELIGIBLE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["OPTICAL_STRUCTURE", "FLAT_STRUCTURE", "DUST_OR_VIGNETTING"],
)
def test_optical_flat_dust_were_eligible_and_are_now_abstained(name):
    entry = oracle_entry(name)
    # They were the wrongly-ELIGIBLE confounders in the catalogue.
    assert entry.catalog_default_action == ACTION_ELIGIBLE
    assert entry.corrected
    assert entry.expected_action != ACTION_ELIGIBLE


# ---------------------------------------------------------------------------
# The transient/sky confounders are no longer NO_ACTION_REQUIRED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["SINGLE_TRANSIENT", "COSMIC_RAY", "STAR_CROSSING_SITE", "UNDERSAMPLED_STAR_CORE"],
)
def test_transient_sky_were_no_action_and_are_now_abstained(name):
    entry = oracle_entry(name)
    assert entry.catalog_default_action == "NO_ACTION_REQUIRED"
    assert entry.corrected
    assert entry.expected_action != ACTION_ELIGIBLE


def test_near_saturation_is_not_a_reconstruction_target():
    # §6.3 adds NEAR_SATURATION as a high-value anomaly confounder.
    entry = oracle_entry("NEAR_SATURATION")
    assert entry.expected_action != ACTION_ELIGIBLE
    assert entry.corrected


# ---------------------------------------------------------------------------
# The reconstruction targets are exactly the genuine intermittent sites
# ---------------------------------------------------------------------------


def test_reconstruction_targets_are_exactly_the_intermittent_defects():
    assert set(reconstruction_targets()) == {
        "INTERMITTENT_TWO_STATE",
        "INTERMITTENT_MULTI_STATE",
        "INTERMITTENT_CONTINUOUS",
        "RARE_HIGH_STATE",
        "RARE_LOW_STATE",
    }


def test_every_reconstruction_target_is_a_catalogue_class():
    for name in reconstruction_targets():
        assert name in CLASS_NAMES


# ---------------------------------------------------------------------------
# Corrected entries keep the catalogue's epistemic state (only action fixed)
# ---------------------------------------------------------------------------


def test_correction_does_not_touch_epistemic_state():
    for name in CONFOUNDER_CLASSES:
        entry = oracle_entry(name)
        cat = resolve_class(name)
        assert entry.epistemic_state == cat.default_expected_qualification_state


def test_expected_action_helpers_match_entries():
    for name in CLASS_NAMES:
        assert expected_action(name) == oracle_entry(name).expected_action
        assert epistemic_state(name) == oracle_entry(name).epistemic_state
