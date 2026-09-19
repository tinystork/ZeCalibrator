"""Rework-1 regression tests (mission §8 / contract A–G).

Discriminating negative/positive tests for the reproduced Nono/Junior findings
M1–M10, plus the supplementary probes. Each test asserts the intended
rejection/ambiguity, never the r0 defect.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3

import pytest

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    detector,
    flat_validity,
    geo,
    light,
    make_binding,
    policy,
    pool,
    profile,
    request,
)
from zecalibrator.core.descriptors import (
    DescriptorSchemaError,
    DescriptorSnapshot,
    NormalizationProvenance,
    NormalizationScalars,
    ProcessingProvenance,
    ValidityEvidence,
    master_descriptor_from_dict,
)
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.plans import (
    CalibrationRequest,
    Candidate,
    FlatQualityPolicy,
    MasterBinding,
    MatchPolicy,
    Tolerance,
    default_match_policy,
)
from zecalibrator.application.library import (
    LibraryHandle,
    LibrarySnapshot,
    light_constraints_from_sensor_metadata,
    plan_calibration_file_backed,
    resolve_calibration,
    validate_binding,
    validate_plan,
)
from zecalibrator.core.metadata import (
    ConflictDiagnostic,
    ImportDeclaration,
    build_sensor_metadata,
)
from zecalibrator.io.library_index import (
    LIBRARY_INDEX_SCHEMA,
    LibraryIndex,
    UnsupportedSchemaError,
)
from zecalibrator.io.master_source import InMemorySource, ScanDiagnostic


# --- M1: required facts never silently skip ---------------------------------
def test_missing_light_exposure_rejects_dark():
    lt = light(acquisition=acquisition(exposure_s=None))
    dk = descriptor("dark", "included")
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == "NO_MATCH"
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes


def test_missing_bias_qualified_range_rejects():
    lt = light(acquisition=acquisition(bias_exposure_max_s=None))
    bias = descriptor("bias", "not_applicable", exposure_s=5.0)
    r = match_calibration(lt, request("bias_only"), pool(bias=[candidate("b1", bias)]), policy())
    assert r.outcome == "NO_MATCH"
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes


def test_missing_roi_extent_rejects_for_cfa():
    # R3B: roi_extent is CFA-conditional — necessary for a Bayer sensor (a missing
    # ROI extent cannot be invented), a disambiguator for an explicit mono sensor.
    lt = light(geometry=geo(cfa_phase="GRBG", roi_extent=None))
    dk = descriptor("dark", "included", geometry=geo(cfa_phase="GRBG", roi_extent=None))
    r = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r.outcome == "NO_MATCH"
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes


# --- M2: frozen policy cannot be widened ------------------------------------
def test_widened_temperature_absolute_refused():
    with pytest.raises(ValueError):
        MatchPolicy(
            version="zecalibrator.match.v1",
            exposure_tolerance=Tolerance(relative=1e-6, absolute=1e-6),
            temperature_tolerance=Tolerance(relative=0.0, absolute=5.0),
            flat_quality_policy=FlatQualityPolicy(state="qualified", threshold_pct=90.0, per_plane=True),
        )


def test_widened_exposure_absolute_refused():
    with pytest.raises(ValueError):
        MatchPolicy(
            version="zecalibrator.match.v1",
            exposure_tolerance=Tolerance(relative=1e-6, absolute=100.0),
            temperature_tolerance=Tolerance(relative=0.0, absolute=1e-6),
            flat_quality_policy=FlatQualityPolicy(state="qualified", threshold_pct=90.0, per_plane=True),
        )


def test_nonzero_scientific_temperature_refused():
    with pytest.raises(ValueError):
        MatchPolicy(
            version="zecalibrator.match.v1",
            exposure_tolerance=Tolerance(relative=1e-6, absolute=1e-6),
            temperature_tolerance=Tolerance(relative=0.5, absolute=1e-6),
            flat_quality_policy=FlatQualityPolicy(state="qualified", threshold_pct=90.0, per_plane=True),
        )


def test_custom_flat_threshold_refused():
    with pytest.raises(ValueError):
        FlatQualityPolicy(state="qualified", threshold_pct=95.0, per_plane=True)


# --- M3: manual selection filters + recounts ---------------------------------
def test_empty_manual_changes_nothing():
    lt = light()
    two = pool(dark=[candidate("d1", descriptor("dark", "included", content_sha256="a" * 64)), candidate("d2", descriptor("dark", "included", content_sha256="c" * 64))])
    r = match_calibration(lt, request(), two, policy(), manual_selection={})
    assert r.outcome == "AMBIGUOUS"
    assert len(r.coherent_sets) == 2


def test_partial_manual_leaves_residual_ambiguity():
    lt = light()
    bias = candidate("b", descriptor("bias", "not_applicable", exposure_s=0.001))
    removed = [candidate("r1", descriptor("dark", "removed")), candidate("r2", descriptor("dark", "removed", content_sha256="3" * 64))]
    r = match_calibration(lt, request("dark_bias_removed"), pool(dark=removed, bias=[bias]), policy(), manual_selection={"bias": "b"})
    assert r.outcome == "AMBIGUOUS"
    assert len(r.coherent_sets) == 2


def test_manual_incompatible_choice_refuses():
    lt = light()
    d1 = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64))
    d2 = candidate("d2", descriptor("dark", "included", acquisition_obj=acquisition(gain=999)))
    r = match_calibration(lt, request(), pool(dark=[d1, d2]), policy(), manual_selection={"dark": "d2"})
    assert r.outcome == "NO_MATCH"
    assert "GAIN_MISMATCH" in r.reason_codes


def test_manual_unknown_candidate_refuses():
    lt = light()
    d1 = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64))
    r = match_calibration(lt, request(), pool(dark=[d1]), policy(), manual_selection={"dark": "nope"})
    assert r.outcome == "NO_MATCH"
    assert "ROLE_UNAVAILABLE" in r.reason_codes


def test_duplicate_candidate_id_manual_not_first_win():
    lt = light()
    a = descriptor("dark", "included", content_sha256="a" * 64)
    b = descriptor("dark", "included", content_sha256="c" * 64)
    cs = pool(dark=[candidate("same", a), candidate("same", b)])
    auto = match_calibration(lt, request(), cs, policy())
    assert auto.outcome == "AMBIGUOUS"
    manual = match_calibration(lt, request(), cs, policy(), manual_selection={"dark": "same"})
    assert manual.outcome == "AMBIGUOUS"  # both identities share the id -> residual ambiguity


# --- M4: role/master_type validation ----------------------------------------
def test_wrong_role_dark_used_as_bias_rejects():
    lt = light()
    wrong = candidate("wrong", descriptor("dark", "included", exposure_s=0.001))
    r = match_calibration(lt, request("bias_only"), pool(bias=[wrong]), policy())
    assert r.outcome == "NO_MATCH"
    assert "ROLE_UNAVAILABLE" in r.reason_codes


# --- M5: flat qualification evidence ----------------------------------------
def _flat_with_validity(valid, total):
    pp = ProcessingProvenance(source="synthetic_fixture", additive_correction_history=("flat_dark_subtracted",))
    return descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp,
        validity=flat_validity(phase="mono", valid=valid, total=total),
    )


def _flat_dark_matching_setup():
    lt = light()
    dk = descriptor("dark", "included")
    return lt, dk


def test_flat_zero_valid_rejects():
    lt, dk = _flat_dark_matching_setup()
    flat = _flat_with_validity(0, 100)
    r = match_calibration(lt, request("dark_incl_bias", "apply"), pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]), policy())
    assert r.outcome == "NO_MATCH"
    assert "FLAT_UNUSABLE" in r.reason_codes


def test_flat_89_of_100_rejects():
    lt, dk = _flat_dark_matching_setup()
    flat = _flat_with_validity(89, 100)
    r = match_calibration(lt, request("dark_incl_bias", "apply"), pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]), policy())
    assert r.outcome == "NO_MATCH"
    assert "SATURATED_FLAT" in r.reason_codes


def test_flat_90_of_100_matches():
    lt, dk = _flat_dark_matching_setup()
    flat = _flat_with_validity(90, 100)
    r = match_calibration(lt, request("dark_incl_bias", "apply"), pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]), policy())
    assert r.outcome == "MATCHED"


def test_flat_91_of_100_matches():
    lt, dk = _flat_dark_matching_setup()
    flat = _flat_with_validity(91, 100)
    r = match_calibration(lt, request("dark_incl_bias", "apply"), pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]), policy())
    assert r.outcome == "MATCHED"


def test_flat_unknown_saturation_rejects():
    lt, dk = _flat_dark_matching_setup()
    pp = ProcessingProvenance(source="synthetic_fixture", additive_correction_history=("flat_dark_subtracted",))
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp,
        validity=flat_validity(phase="mono", saturation_known=False),
    )
    r = match_calibration(lt, request("dark_incl_bias", "apply"), pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]), policy())
    assert r.outcome == "NO_MATCH"
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes


def test_flat_missing_illumination_rejects():
    lt, dk = _flat_dark_matching_setup()
    pp = ProcessingProvenance(source="synthetic_fixture", additive_correction_history=("flat_dark_subtracted",))
    ve = ValidityEvidence(saturation_limit_known=True, valid_normalization_count={"mono": 95}, total_normalization_count={"mono": 100}, quality_policy_state="qualified", illumination=None, exposure_quality="qualified")
    flat = descriptor("flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized", filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, validity=ve)
    r = match_calibration(lt, request("dark_incl_bias", "apply"), pool(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]), policy())
    assert r.outcome == "NO_MATCH"
    assert "MISSING_REQUIRED_FIELD" in r.reason_codes


def test_flat_conflicting_normalization_rejects():
    lt = light()
    dk = descriptor("dark", "included")
    sc = NormalizationScalars(g1=1.0, r=1.0, b=1.0, g2=1.0)
    sc2 = NormalizationScalars(g1=2.0, r=1.0, b=1.0, g2=1.0)
    pp = ProcessingProvenance(
        source="synthetic_fixture",
        additive_correction_history=("flat_dark_subtracted",),
        normalization=NormalizationProvenance(algorithm="cfa-median-per-plane", population="cfa-4-plane", scalars=sc2),
    )
    flat = descriptor(
        "flat", "not_applicable", exposure_s=1.0, flat_form="normalized_response",
        normalization_algorithm="cfa-median-per-plane", normalization_scalars=sc,
        pixel_domain="normalized_response", physical_units="dimensionless",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1", processing=pp, geometry=geo(cfa_phase="GRBG"),
    )
    r = match_calibration(lt, request("dark_incl_bias", "apply"), pool(dark=[candidate("d1", dk, )], flat=[candidate("f1", flat)]), policy())
    assert r.outcome == "NO_MATCH"
    assert "UNDOCUMENTED_PROCESSING" in r.reason_codes


# --- M6: identity integrity / tamper rejection ------------------------------
def test_profile_tamper_changes_descriptor_id():
    flat = descriptor("flat", "not_applicable", exposure_s=1.0, flat_form="corrected_unnormalized",
                      filter="NONE", optical_train_id="SYNTH-TRAIN-1",
                      processing=ProcessingProvenance(source="synthetic_fixture", additive_correction_history=("flat_dark_subtracted",), acquisition_profile=profile(bias_exposure_max_s=0.01, short_flat_profile=False)))
    d = dict(flat.to_dict())
    tampered = dict(d)
    tampered["processing_provenance"]["acquisition_profile"]["bias_exposure_max_s"] = 999
    tampered["processing_provenance"]["acquisition_profile"]["short_flat_profile"] = True
    with pytest.raises(Exception):
        master_descriptor_from_dict(tampered)  # recomputed id != recorded id


def test_unknown_descriptor_extension_rejected():
    dk = descriptor("dark", "included")
    d = dict(dk.to_dict())
    d["science_extension"] = {"contradiction": True}
    with pytest.raises(DescriptorSchemaError):
        master_descriptor_from_dict(d)


def test_mismatched_candidate_snapshot_rejected():
    a = descriptor("dark", "included")
    b = descriptor("dark", "included", content_sha256="a" * 64)
    with pytest.raises(ValueError):
        Candidate(candidate_id="bad", descriptor=a, descriptor_snapshot=DescriptorSnapshot(b))


def test_mismatched_binding_snapshot_rejected():
    a = descriptor("dark", "included")
    b = descriptor("dark", "included", content_sha256="a" * 64)
    with pytest.raises(ValueError):
        MasterBinding(
            descriptor_id=a.descriptor_id,
            descriptor_snapshot=DescriptorSnapshot(b),
            content_sha256=a.content_sha256,
            size_bytes=a.size_bytes,
            hdu=a.hdu,
            mask_identity=a.mask_identity,
        )


# --- M7: coherent sets frozen + plan roundtrip ------------------------------
def test_coherent_sets_are_frozen():
    lt = light()
    d1 = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64))
    d2 = candidate("d2", descriptor("dark", "included", content_sha256="c" * 64))
    r = match_calibration(lt, request(), pool(dark=[d1, d2]), policy())
    assert r.outcome == "AMBIGUOUS"
    with pytest.raises((TypeError, AttributeError)):
        r.coherent_sets[0].clear()


def test_plan_full_roundtrip_with_library_closed():
    lt = light()
    dk = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    snap = LibrarySnapshot(revision="r1", schema_version=LIBRARY_INDEX_SCHEMA, candidates={"dark": (candidate("d1", dk, locator_path="/d.fits", mask_path="/d.mask"),)})
    handle = LibraryHandle(snap)
    env = handle.resolve(lt, request(), policy())
    handle.close()
    plan = env.plan
    plan.verify_plan_id()
    for binding in plan.masters.values():
        binding.verify_snapshot()

    # JSON roundtrip with the library closed, no global lookup.
    import json

    raw = json.dumps(dict(plan.to_dict()), ensure_ascii=False, allow_nan=False)
    restored = type(plan).from_dict(json.loads(raw))
    assert restored.plan_id == plan.plan_id
    restored.verify_plan_id()


def test_adapter_refuses_conflicts():
    decl = ImportDeclaration(source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0")
    md = build_sensor_metadata(shape=(4, 4), normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU")
    from dataclasses import replace

    md = replace(md, conflicts=(ConflictDiagnostic("gain", ("GAIN", "EGAIN"), (1, 2)),))
    with pytest.raises(Exception):
        light_constraints_from_sensor_metadata(md)


def test_decision_envelope_json_roundtrip():
    from zecalibrator.application.library import DecisionEnvelope
    import json

    lt = light()
    d1 = candidate("d1", descriptor("dark", "included", content_sha256="a" * 64))
    d2 = candidate("d2", descriptor("dark", "included", content_sha256="c" * 64))
    snap = LibrarySnapshot(revision="r1", schema_version=LIBRARY_INDEX_SCHEMA, candidates={"dark": (d1, d2)})
    env = resolve_calibration(lt, request(), snap, policy())
    assert env.outcome == "AMBIGUOUS"
    raw = json.dumps(dict(env.to_dict()), ensure_ascii=False, allow_nan=False)
    restored = DecisionEnvelope.from_dict(json.loads(raw))
    assert restored.outcome == "AMBIGUOUS"
    assert restored.request.additive_mode == "dark_incl_bias"
    assert restored.policy.version == "zecalibrator.match.v1"
    assert len(restored.coherent_sets) == 2
    for s in restored.coherent_sets:
        for c in s.values():
            assert c.descriptor_snapshot.descriptor.descriptor_id


# --- M8: file-backed / detached validation ----------------------------------
def test_validate_binding_no_locators_fails():
    dk = descriptor("dark", "included")
    binding = MasterBinding(
        descriptor_id=dk.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(dk),
        content_sha256=dk.content_sha256,
        size_bytes=dk.size_bytes,
        hdu=dk.hdu,
        mask_identity=dk.mask_identity,
    )
    r = validate_binding(binding, InMemorySource())
    assert r.status == "FAILED"
    assert any("no image locator" in x for x in r.reasons)
    assert any("no mask locator" in x for x in r.reasons)


def test_detached_validation_after_close():
    data = b"ORIGINAL_BYTES"
    mask = b"mask-bytes"
    sha = hashlib.sha256(data).hexdigest()
    mask_sha = hashlib.sha256(mask).hexdigest()
    dk = descriptor("dark", "included", content_sha256=sha, size_bytes=len(data), mask_identity=mask_sha)
    binding = make_binding(dk)
    src = InMemorySource(images={"/d.fits": data}, masks={"/d.mask": mask})
    r = validate_binding(binding, src)  # no handle required
    assert r.status == "VALID", r.reasons


def test_file_backed_planning_verifies_hashes():
    data = b"ORIGINAL_BYTES"
    mask = b"mask-bytes"
    sha = hashlib.sha256(data).hexdigest()
    mask_sha = hashlib.sha256(mask).hexdigest()
    dk = descriptor("dark", "included", content_sha256=sha, size_bytes=len(data), mask_identity=mask_sha)
    snap = LibrarySnapshot(revision="r1", schema_version=LIBRARY_INDEX_SCHEMA, candidates={"dark": (candidate("d1", dk, locator_path="/d.fits", mask_path="/d.mask"),)})
    src = InMemorySource(images={"/d.fits": data}, masks={"/d.mask": mask})
    env = plan_calibration_file_backed(light(), request(), snap, policy(), source=src)
    assert env.plan is not None
    assert env.verification.status == "VALID"


def test_file_backed_planning_detects_stale_bytes():
    dk = descriptor("dark", "included", content_sha256="0" * 64, size_bytes=999, mask_identity="1" * 64)
    snap = LibrarySnapshot(revision="r1", schema_version=LIBRARY_INDEX_SCHEMA, candidates={"dark": (candidate("d1", dk, locator_path="/d.fits", mask_path="/d.mask"),)})
    src = InMemorySource(images={"/d.fits": b"STALE"}, masks={"/d.mask": b"stale-mask"})
    env = plan_calibration_file_backed(light(), request(), snap, policy(), source=src)
    assert env.plan is None
    assert env.verification.status == "FAILED"


# --- M9: index lifecycle ----------------------------------------------------
def test_empty_index_returns_empty_snapshot(tmp_path):
    idx = LibraryIndex(str(tmp_path / "lib.sqlite")).open(initialize=True)
    try:
        snap = idx.load_snapshot()
        assert snap.revision == ""
        assert snap.candidates == {}
    finally:
        idx.close()


def test_refused_open_does_not_create_file(tmp_path):
    p = tmp_path / "absent.sqlite"
    idx = LibraryIndex(str(p))
    with pytest.raises(Exception):
        idx.open(initialize=False)
    assert not p.exists()


def test_unsupported_schema_closes_connection(tmp_path):
    p = tmp_path / "bad.sqlite"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', 'unknown.v9')")
    conn.commit()
    conn.close()
    idx = LibraryIndex(str(p))
    with pytest.raises(UnsupportedSchemaError):
        idx.open()
    assert idx._conn is None


def test_missing_revision_diagnosed(tmp_path):
    idx = LibraryIndex(str(tmp_path / "lib.sqlite")).open(initialize=True)
    try:
        with pytest.raises(Exception):
            idx.load_snapshot("missing-revision")
    finally:
        idx.close()


def test_scan_publishes_and_verifies(tmp_path):
    data = b"ORIGINAL_BYTES"
    sha = hashlib.sha256(data).hexdigest()
    dk = descriptor("dark", "included", content_sha256=sha, size_bytes=len(data), mask_identity="1" * 64)

    def build(path):
        return candidate("d1", dk, locator_path=path)

    src = InMemorySource(images={"/m.fits": data})
    idx = LibraryIndex(str(tmp_path / "lib.sqlite")).open(initialize=True)
    try:
        result = idx.scan_and_publish(["/m.fits"], build_candidate=build, source=src, revision="r1")
        assert result.status == "COMPLETED"
        assert idx.load_snapshot().revision == "r1"
    finally:
        idx.close()


def test_scan_cancelled_preserves_last_revision(tmp_path):
    data = b"ORIGINAL_BYTES"
    sha = hashlib.sha256(data).hexdigest()
    dk = descriptor("dark", "included", content_sha256=sha, size_bytes=len(data), mask_identity="1" * 64)

    def build(path):
        return candidate("d1", dk, locator_path=path)

    src = InMemorySource(images={"/m.fits": data})
    idx = LibraryIndex(str(tmp_path / "lib.sqlite")).open(initialize=True)
    try:
        idx.publish_revision("r1", {"dark": (candidate("d1", dk),)})
        from zecalibrator.application.cancellation import CancellationToken

        token = CancellationToken()
        token.cancel()
        result = idx.scan_and_publish(["/m.fits", "/m2.fits"], build_candidate=build, source=src, revision="r2", cancel=token)
        assert result.status == "CANCELLED"
        assert idx.load_snapshot().revision == "r1"
    finally:
        idx.close()


# --- M10: unexplained refusal -----------------------------------------------
def test_wrong_bias_state_only_dark_explains_refusal():
    lt = light()
    only_removed = candidate("r1", descriptor("dark", "removed"))
    r = match_calibration(lt, request("dark_incl_bias"), pool(dark=[only_removed]), policy())
    assert r.outcome == "NO_MATCH"
    assert "ROLE_UNAVAILABLE" in r.reason_codes  # never empty
