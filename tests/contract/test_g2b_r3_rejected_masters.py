"""G2B rework-3 — considered-but-rejected masters are recorded in the composition.

D-4: nothing supplied may be invisible. Every master evaluated and rejected by
compatibility must appear in the persisted composition audit (``rejected_masters``)
with its structured reason codes, while ``no_candidate_roles`` keeps its meaning
"no compatible candidate existed".
"""

from __future__ import annotations

from _phase4_fixtures import acquisition, candidate, descriptor, geo, light, policy, pool, request
from zecalibrator.application.library import LibrarySnapshot
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.descriptors import ProcessingProvenance
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.plans import CalibrationComposition
from zecalibrator.core.routes import OUTCOME_READY


def snapshot(**roles) -> LibrarySnapshot:
    return LibrarySnapshot(
        revision="r",
        schema_version="zecalibrator.library.v1",
        candidates={k: tuple(v) for k, v in roles.items()},
    )


def _rejected_flat(cid="f1"):
    # A flat for a CFA Bayer sensor is incompatible with a mono light
    # (CFA_PHASE_MISMATCH — a scientific incompatibility, no role="flat").
    pp = ProcessingProvenance(
        source="synthetic_fixture", additive_history_state="known",
        additive_correction_history=("flat_dark_subtracted",),
    )
    f = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, exposure_s=1.0,
        geometry=geo(cfa_phase="RGGB"), content_sha256="a" * 64, mask_identity="b" * 64,
    )
    return candidate(cid, f)


def test_derived_passthrough_records_rejected_flat():
    lt = light()  # mono
    res = resolve_route(lt, snapshot(flat=[_rejected_flat()]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.plan is not None
    comp = res.plan.composition
    assert comp.level == "NONE"
    assert comp.applied_roles == ()
    # The supplied-but-rejected flat is recorded with its reason codes.
    assert len(comp.rejected_masters) == 1
    rec = comp.rejected_masters[0]
    assert rec.role == "flat"
    assert "CFA_PHASE_MISMATCH" in rec.reason_codes
    # no_candidate_roles keeps its meaning (the flat HAD a candidate, it was rejected).
    assert "flat" not in comp.no_candidate_roles


def test_derived_partial_records_rejected_flat_with_dark():
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="c" * 64, mask_identity="d" * 64))
    res = resolve_route(lt, snapshot(dark=[dark], flat=[_rejected_flat()]), policy())
    assert res.outcome == OUTCOME_READY
    comp = res.plan.composition
    assert comp.applied_roles == ("dark",)
    assert comp.additive_state == "dark_incl_bias"
    assert comp.flat_applied is False
    assert any(r.role == "flat" and "CFA_PHASE_MISMATCH" in r.reason_codes for r in comp.rejected_masters)


def test_explicit_rejected_master_recorded():
    lt = light()
    bad = candidate("d1", descriptor("dark", "included", acquisition_obj=acquisition(gain=999)))
    r = match_calibration(lt, request("dark_incl_bias"), pool(dark=[bad]), policy())
    assert r.outcome == "NO_MATCH"
    comp = r.composition
    assert comp.level == "NONE"
    assert len(comp.rejected_masters) == 1
    assert comp.rejected_masters[0].role == "dark"
    assert "GAIN_MISMATCH" in comp.rejected_masters[0].reason_codes


def test_backward_read_composition_without_rejected_masters():
    comp = CalibrationComposition(
        applied_roles=("dark",), skipped_roles=(), level="COMPLETE",
        additive_state="dark_incl_bias", flat_applied=False, no_candidate_roles=(),
    )
    d = comp.to_dict()
    del d["rejected_masters"]
    restored = CalibrationComposition.from_dict(d)
    assert restored.rejected_masters == ()
    assert restored.applied_roles == ("dark",)
    assert restored.level == "COMPLETE"


def test_rejected_masters_not_in_plan_digest():
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    r = match_calibration(lt, request("dark_incl_bias"), pool(dark=[dark]), policy())
    assert r.outcome == "MATCHED"
    # The composition (and its rejected_masters) is a NON-DIGEST audit block:
    # it never appears in the plan digest projection.
    digest = r.plan.plan_digest_dict()
    assert "composition" not in digest
    assert "rejected_masters" not in digest
