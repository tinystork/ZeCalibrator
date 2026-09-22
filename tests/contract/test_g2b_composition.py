"""G2B calibration-composition contract tests (core level).

Asserts the honest, availability-relative ``CalibrationComposition`` that the
matcher now records: applied/skipped/no-candidate roles, level (NONE/PARTIAL/
COMPLETE), the effective additive state and flat flag.
"""

from __future__ import annotations

import pytest

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    light,
    policy,
    pool,
    request,
)
from zecalibrator.core.descriptors import OpticalIdentity, ProcessingProvenance
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.plans import CalibrationComposition


def _compat_flat(cid, content_sha="a" * 64):
    pp = ProcessingProvenance(
        source="synthetic_fixture", additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    f = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp,
        content_sha256=content_sha, mask_identity="b" * 64,
    )
    return candidate(cid, f)


def test_composition_dark_bias_flat_complete():
    lt = light()
    dark = candidate("d1", descriptor("dark", "removed", content_sha256="a" * 64, mask_identity="b" * 64))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001, content_sha256="c" * 64, mask_identity="d" * 64))
    flat = _compat_flat("f1")
    r = match_calibration(
        lt, request("dark_bias_removed", "apply"),
        pool(dark=[dark], bias=[bias], flat=[flat]), policy(),
    )
    assert r.outcome == "MATCHED"
    comp = r.composition
    assert comp.level == "COMPLETE"
    assert comp.applied_roles == ("bias", "dark", "flat")
    assert comp.additive_state == "dark_bias_removed"
    assert comp.flat_applied is True
    assert comp.no_candidate_roles == ()
    assert comp.skipped_roles == ()
    assert r.plan.composition is comp


def test_composition_dark_bias_no_flat_complete():
    lt = light()
    dark = candidate("d1", descriptor("dark", "removed", content_sha256="a" * 64, mask_identity="b" * 64))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001))
    r = match_calibration(lt, request("dark_bias_removed"), pool(dark=[dark], bias=[bias]), policy())
    assert r.outcome == "MATCHED"
    comp = r.composition
    assert comp.level == "COMPLETE"
    assert comp.applied_roles == ("bias", "dark")
    assert comp.additive_state == "dark_bias_removed"
    assert comp.flat_applied is False


def test_composition_dark_only_complete():
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    r = match_calibration(lt, request("dark_incl_bias"), pool(dark=[dark]), policy())
    assert r.outcome == "MATCHED"
    assert r.composition.level == "COMPLETE"
    assert r.composition.applied_roles == ("dark",)
    assert r.composition.additive_state == "dark_incl_bias"


def test_composition_bias_only_complete():
    lt = light()
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001))
    r = match_calibration(lt, request("bias_only"), pool(bias=[bias]), policy())
    assert r.outcome == "MATCHED"
    assert r.composition.level == "COMPLETE"
    assert r.composition.applied_roles == ("bias",)
    assert r.composition.additive_state == "bias_only"


def test_composition_flat_only_complete():
    lt = light()
    flat = _compat_flat("f1")
    r = match_calibration(lt, request("control", "apply"), pool(flat=[flat]), policy())
    assert r.outcome == "MATCHED"
    assert r.composition.level == "COMPLETE"
    assert r.composition.applied_roles == ("flat",)
    assert r.composition.additive_state == "none"
    assert r.composition.flat_applied is True


def test_composition_no_master_at_all_none():
    lt = light()
    r = match_calibration(lt, request("control", "none"), pool(), policy())
    assert r.outcome == "MATCHED"
    assert set(r.plan.masters) == set()
    assert r.composition.level == "NONE"
    assert r.composition.applied_roles == ()
    assert r.composition.additive_state == "none"
    assert r.composition.flat_applied is False


def test_composition_all_incompatible_none_with_reasons():
    lt = light()
    bad = candidate("d1", descriptor("dark", "included", acquisition_obj=acquisition(gain=999)))
    r = match_calibration(lt, request("dark_incl_bias"), pool(dark=[bad]), policy())
    assert r.outcome == "NO_MATCH"
    assert "GAIN_MISMATCH" in r.reason_codes  # reasons preserved, never hidden
    comp = r.composition
    assert comp is not None
    assert comp.level == "NONE"
    assert comp.applied_roles == ()
    assert comp.no_candidate_roles == ("dark",)
    assert comp.skipped_roles == ()


def test_composition_level_vocabulary():
    # Frozen vocabulary sanity: only NONE/PARTIAL/COMPLETE are valid.
    for level in ("NONE", "PARTIAL", "COMPLETE"):
        CalibrationComposition((), (), level, "none", False, ())
    with pytest.raises(ValueError):
        CalibrationComposition((), (), "OTHER", "none", False, ())
    with pytest.raises(ValueError):
        CalibrationComposition((), (), "NONE", "control", False, ())
