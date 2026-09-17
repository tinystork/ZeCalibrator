"""Serialization/digest/plan contract tests (mission §7.C).

Canonical JSON, descriptor/plan explicit projections, optical identity binding,
location exclusion, snapshot retention after handle close, null/absent roles,
mask identity separation, tamper rejection, replay and no hash self-reference.
"""

from __future__ import annotations

import copy

import pytest

from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    detector,
    geo,
    light,
    policy,
    pool,
    request,
)
from zecalibrator.core.descriptors import (
    DescriptorIntegrityError,
    DescriptorSnapshot,
    MasterDescriptor,
    master_descriptor_from_dict,
)
from zecalibrator.core.digests import canonical_json, descriptor_digest, plan_digest
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.plans import (
    CalibrationPlan,
    CalibrationRequest,
    Candidate,
    FitsFileLocator,
    MasterBinding,
    PolicyParameters,
    VersionSet,
)


def _plan(masters, lt=None, req=None):
    lt = lt or light()
    req = req or request()
    return CalibrationPlan.build(
        request=req,
        light_constraints=lt,
        masters=masters,
        policy_parameters=PolicyParameters(
            exposure_tolerance=policy().exposure_tolerance,
            temperature_tolerance=policy().temperature_tolerance,
            flat_quality_policy=policy().flat_quality_policy,
        ),
        versions=VersionSet(),
    )


# --- canonical JSON / digest -------------------------------------------------
def test_canonical_json_pinned_example():
    assert canonical_json({"b": 1, "a": 2.0, "c": "x\ny"}) == '{"a":2.0,"b":1,"c":"x\\ny"}'


def test_descriptor_digest_changes_on_science_fields():
    d1 = descriptor("dark", "included")
    d2 = descriptor("dark", "included", content_sha256="e" * 64, mask_identity="a" * 64)
    d3 = descriptor("dark", "included", acquisition_obj=acquisition(exposure_s=11.0))
    assert d1.descriptor_id != d2.descriptor_id
    assert d1.descriptor_id != d3.descriptor_id


def test_descriptor_digest_excludes_incidental_fields():
    d = descriptor("dark", "included")
    base = dict(d.projection_dict())
    extra = dict(base)
    extra["path"] = "/tmp/master.fits"
    extra["imported_at"] = "2026-09-15T00:00:00Z"
    extra["source_path"] = "/data/source.fits"
    extra["descriptor_id"] = "self-id"
    assert descriptor_digest(base) == descriptor_digest(extra)


def test_plan_digest_optical_identity_bound():
    lt1 = light()
    lt2 = light()
    # Same additive (dark-only) plan but different light optical identity must
    # differ in plan digest (the light's optical identity is bound).
    b1 = _plan({"dark": _binding(descriptor("dark", "included"))}, lt=lt1)
    b2 = _plan({"dark": _binding(descriptor("dark", "included"))}, lt=lt2)
    assert b1.plan_id == b2.plan_id  # identical inputs -> identical digest
    lt3 = light()
    from zecalibrator.core.descriptors import OpticalIdentity

    lt3 = light(optical=OpticalIdentity(filter="IRCUT", optical_train_id=None))
    b3 = _plan({"dark": _binding(descriptor("dark", "included"))}, lt=lt3)
    assert b1.plan_id != b3.plan_id  # filter change changes plan identity


def test_plan_digest_excludes_locator_and_execution_fields():
    b1 = _plan({"dark": _binding(descriptor("dark", "included"), loc="/lib/dark.fits")})
    fields = dict(b1.plan_digest_dict())
    alt = copy.deepcopy(fields)
    alt["masters"]["dark"]["locator"] = {"path": "/elsewhere.fits", "hdu": 0}
    alt["tile_shape"] = [2, 2]
    alt["source_path"] = "/x/y.fits"
    alt["imported_at"] = "2026-09-15T00:00:00Z"
    alt["plan_id"] = "self-id"
    assert plan_digest(fields) == plan_digest(alt)


def test_plan_digest_policy_parameter_change_changes():
    b1 = _plan({"dark": _binding(descriptor("dark", "included"))})
    fields = dict(b1.plan_digest_dict())
    alt = copy.deepcopy(fields)
    alt["policy_parameters"]["flat_quality_policy"]["threshold_pct"] = 95.0
    assert plan_digest(fields) != plan_digest(alt)


def test_master_content_change_changes_plan_digest():
    b1 = _plan({"dark": _binding(descriptor("dark", "included"))})
    fields = dict(b1.plan_digest_dict())
    alt = copy.deepcopy(fields)
    alt["masters"]["dark"]["content_sha256"] = "e" * 64
    assert plan_digest(fields) != plan_digest(alt)


def _binding(desc, loc=None):
    from zecalibrator.core.plans import FitsFileLocator

    locs = (FitsFileLocator(path=loc, hdu=desc.hdu),) if loc else ()
    return MasterBinding(
        descriptor_id=desc.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(desc),
        content_sha256=desc.content_sha256,
        size_bytes=desc.size_bytes,
        hdu=desc.hdu,
        mask_identity=desc.mask_identity,
        locators=locs,
    )


# --- snapshot retention after handle close ----------------------------------
def test_snapshot_retained_after_handle_close():
    from zecalibrator.application.library import LibraryHandle, LibrarySnapshot

    d = descriptor("dark", "included")
    snap = LibrarySnapshot(revision="r1", schema_version="zecalibrator.library.v1", candidates={"dark": (candidate("d1", d, locator_path="/d.fits"),)})
    handle = LibraryHandle(snap)
    plan = handle.resolve(light(), request(), policy()).plan
    handle.close()
    # The plan binding's descriptor snapshot survives handle closure.
    assert plan.masters["dark"].descriptor_id == d.descriptor_id
    plan.masters["dark"].verify_snapshot()  # unchanged -> valid


def test_null_absent_roles_in_plan():
    lt = light()
    r = match_calibration(lt, request("control"), pool(), policy())
    assert r.outcome == "MATCHED"
    assert set(r.plan.masters) == set()  # no roles bound


# --- mask identity separated ------------------------------------------------
def test_mask_identity_distinct_from_image_and_descriptor():
    d = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    assert d.content_sha256 != d.mask_identity
    assert d.descriptor_id != d.content_sha256
    assert d.descriptor_id != d.mask_identity


# --- tamper rejection -------------------------------------------------------
def test_tampered_descriptor_snapshot_rejected():
    d = descriptor("dark", "included")
    fields = dict(d.to_dict())
    tampered = dict(fields)
    tampered["bias_state"] = "removed"
    with pytest.raises(DescriptorIntegrityError):
        master_descriptor_from_dict(tampered)


def test_tampered_descriptor_verify_rejected():
    d = descriptor("dark", "included")
    fields = dict(d.to_dict())
    del fields["descriptor_id"]  # recompute cleanly from the (tampered) snapshot
    fields["content_sha256"] = "e" * 64
    rebuilt = master_descriptor_from_dict(fields)
    assert rebuilt.descriptor_id != d.descriptor_id


def test_descriptor_replay_roundtrip_identical():
    d = descriptor("dark", "included")
    rebuilt = master_descriptor_from_dict(dict(d.to_dict()))
    assert rebuilt.descriptor_id == d.descriptor_id
    assert rebuilt.projection_dict() == d.projection_dict()


def test_plan_verify_id_rejects_tamper():
    b = _plan({"dark": _binding(descriptor("dark", "included"))})
    b.verify_plan_id()  # valid
    tampered = CalibrationPlan(
        plan_id="0" * 64,
        request=b.request,
        light_constraints=b.light_constraints,
        masters=b.masters,
        policy_parameters=b.policy_parameters,
        versions=b.versions,
    )
    with pytest.raises(ValueError):
        tampered.verify_plan_id()


def test_no_hash_self_reference():
    d = descriptor("dark", "included")
    proj = dict(d.projection_dict())
    assert "descriptor_id" not in proj  # descriptor_id is derived, never in its own projection
    b = _plan({"dark": _binding(d)})
    plan_fields = dict(b.plan_digest_dict())
    assert "plan_id" not in plan_fields


def test_replay_is_deterministic_across_instances():
    lt = light()
    dk = descriptor("dark", "included", content_sha256="a" * 64, mask_identity="b" * 64)
    r1 = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    r2 = match_calibration(lt, request(), pool(dark=[candidate("d1", dk)]), policy())
    assert r1.plan.plan_id == r2.plan.plan_id
    assert r1.plan.plan_digest_dict() == r2.plan.plan_digest_dict()
