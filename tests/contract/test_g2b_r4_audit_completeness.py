"""G2B rework-4 (D-5) — audit completeness: every supplied role is classified.

A role present in the library must appear as applied / rejected (rejected_masters)
/ skipped(NOT_REQUIRED | NOT_APPLICABLE_FOR_ROUTE) / no-candidate — never silently
dropped.
"""

from __future__ import annotations

from _phase4_fixtures import candidate, descriptor, light, policy, pool, request
from zecalibrator.application.library import LibrarySnapshot
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.routes import OUTCOME_READY


def snapshot(**roles) -> LibrarySnapshot:
    return LibrarySnapshot(
        revision="r",
        schema_version="zecalibrator.library.v1",
        candidates={k: tuple(v) for k, v in roles.items()},
    )


def test_case1_compatible_bias_skipped_not_required():
    # Explicit dark_incl_bias with a compatible dark AND a compatible bias: the
    # bias is supplied and compatible but the route does not use it.
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001, content_sha256="c" * 64, mask_identity="d" * 64))
    r = match_calibration(lt, request("dark_incl_bias"), pool(dark=[dark], bias=[bias]), policy())
    assert r.outcome == "MATCHED"
    comp = r.composition
    assert comp.applied_roles == ("dark",)
    assert comp.level == "COMPLETE"  # the unused bias does not downgrade the level
    assert any(s.role == "bias" and s.reason_code == "NOT_REQUIRED" for s in comp.skipped_roles)


def test_case2_flat_dark_only_not_applicable_for_route():
    # Derived route with ONLY a flat_dark: the flat_dark is never evaluated
    # (no flat dependency) but must be recorded, not silently dropped.
    lt = light()
    fd = candidate("fd1", descriptor("flat_dark", "included", exposure_s=1.0, content_sha256="a" * 64, mask_identity="b" * 64))
    res = resolve_route(lt, snapshot(flat_dark=[fd]), policy())
    assert res.outcome == OUTCOME_READY
    comp = res.plan.composition
    assert comp.level == "NONE"
    assert comp.applied_roles == ()
    assert any(s.role == "flat_dark" and s.reason_code == "NOT_APPLICABLE_FOR_ROUTE" for s in comp.skipped_roles)


def test_case1_bias_rejected_is_in_rejected_masters():
    # A supplied-but-incompatible bias beside a dark route is rejected, not
    # silently dropped.
    lt = light()
    dark = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    bad_bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=5.0, content_sha256="c" * 64, mask_identity="d" * 64))
    r = match_calibration(lt, request("dark_incl_bias"), pool(dark=[dark], bias=[bad_bias]), policy())
    assert r.outcome == "MATCHED"
    comp = r.composition
    assert comp.applied_roles == ("dark",)
    assert any(rm.role == "bias" for rm in comp.rejected_masters)


def test_full_audit_explicit_all_roles_classified():
    # dark applied + flat applied + bias unused -> bias is NOT_REQUIRED.
    lt = light()
    from zecalibrator.core.descriptors import ProcessingProvenance
    dark = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64))
    bias = candidate("b1", descriptor("bias", "not_applicable", exposure_s=0.001, content_sha256="c" * 64, mask_identity="d" * 64))
    pp = ProcessingProvenance(source="synthetic_fixture", additive_history_state="known", additive_correction_history=("flat_dark_subtracted",))
    flat = candidate("f1", descriptor("flat", "not_applicable", flat_form="corrected_unnormalized", filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, exposure_s=1.0, content_sha256="e" * 64, mask_identity="f" * 64))
    # dark_incl_bias uses the dark (bias is unused -> NOT_REQUIRED) + flat apply.
    r = match_calibration(
        lt, request("dark_incl_bias", "apply"),
        pool(dark=[dark], bias=[bias], flat=[flat]), policy(),
    )
    assert r.outcome == "MATCHED"
    comp = r.composition
    assert set(comp.applied_roles) == {"dark", "flat"}
    assert any(s.role == "bias" and s.reason_code == "NOT_REQUIRED" for s in comp.skipped_roles)
