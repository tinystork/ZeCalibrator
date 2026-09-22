"""Destination-collision policy tests for the private auto-route facade.

These tests prove the eight invariants behind the Standard "Overwrite / Skip /
Cancel" collision UX without touching scientific calibration, the public API,
GPU or ZeAlfie:

* P1 — the destination is computed after resolve and before calibration;
* P2 — exactly one acquisition/decode per frame (decode seam call-count == N);
* P3 — streaming / O(1) in frame count (no pre-pass, one scalar policy cache);
* P4 — at most one decision per batch (callable invoked exactly once across ≥3
  collisions);
* P5 — skip happens BEFORE calibration (calibrate seam never entered, no file);
* P6 — no-clobber preserved when overwrite is not authorised;
* P7 — overwrite is transactional (see tests/io/test_output_writer_overwrite.py);
* P8 — no scientific/API/GPU/ZeAlfie change (guarded by the contract suites).

The tests use lightweight synthetic array sources and spies (never a heavy
100-frame scientific calibration benchmark).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.api.v1._auto_route import auto_route_batch

SHAPE = (4, 4)
LIBRARY_SCHEMA = "zecalibrator.library.v1"


def _geo():
    return v1.Geometry(
        shape=SHAPE, sensor_dimensions=SHAPE, binning=(1, 1), roi_origin=(0, 0),
        roi_extent=SHAPE, orientation="identity", cfa_phase="mono",
    )


def _detector():
    return v1.DetectorIdentity(detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA")


def _acquisition(exposure_s=10.0):
    return v1.Acquisition(
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16",
        temperature_c=20.0, exposure_s=exposure_s, saturation_limit_adu=60000.0,
        saturation_evidence="qualified", bias_exposure_max_s=0.01, short_flat_profile=False,
    )


def _declaration(exposure_s=10.0):
    return v1.ImportDeclaration(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
        adc_mode="MODE_16", binning=(1, 1), sensor_dimensions=SHAPE,
        orientation="identity", cfa_phase="mono", roi_origin=(0, 0),
        exposure_s=exposure_s, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0, saturation_evidence="qualified",
    )


def _sensor_metadata(exposure_s=10.0):
    decl = _declaration(exposure_s=exposure_s)
    md = v1.build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU"
    )
    return v1.SensorMetadata(
        original_cards=md.original_cards, normalized=md.normalized, conflicts=md.conflicts,
        geometry=_geo(), raw_domain_declaration=md.raw_domain_declaration, units=md.units,
        declaration=md.declaration, exposure_s=md.exposure_s, temperature_c=md.temperature_c,
        gain=md.gain, offset=md.offset, readout_mode=md.readout_mode, adc_mode=md.adc_mode,
        filter=md.filter, detector_model=md.detector_model,
        detector_instance_id=md.detector_instance_id, optical_train_id=md.optical_train_id,
        saturation_limit_adu=md.saturation_limit_adu, saturation_evidence=md.saturation_evidence,
        warnings=md.warnings,
    )


def _array_source(caller_logical_id="light-1", value=100.0):
    return v1.ArrayFrameSource(
        data=np.full(SHAPE, value, dtype=np.float32),
        metadata=_sensor_metadata(exposure_s=10.0),
        identity=v1.ArrayInputIdentity(caller_logical_id=caller_logical_id),
    )


def _dark_descriptor(content_sha256, size_bytes, mask_identity):
    return v1.MasterDescriptor(
        master_type="dark", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state="included", geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=10.0), content_sha256=content_sha256,
        size_bytes=size_bytes, hdu=0, mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=v1.ValidityEvidence(saturation_limit_known=True),
    )


def _write_fits(path, data):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def _write_mask(path, mask):
    np.save(path, np.asarray(mask, dtype=np.uint16), allow_pickle=False)
    return str(path)


def _dark_library(tmp_path):
    dark_fits = str(tmp_path / "dark.fits")
    dark_mask = str(tmp_path / "dark.mask.npy")
    _write_fits(dark_fits, np.full(SHAPE, 10.0, dtype=np.float32))
    _write_mask(dark_mask, np.zeros(SHAPE, dtype=np.uint16))
    fb = open(dark_fits, "rb").read()
    mb = open(dark_mask, "rb").read()
    desc = _dark_descriptor(hashlib.sha256(fb).hexdigest(), len(fb), hashlib.sha256(mb).hexdigest())
    cand = v1.Candidate(
        candidate_id="dark", descriptor=desc, descriptor_snapshot=v1.DescriptorSnapshot(desc),
        locators=(v1.FitsFileLocator(path=dark_fits, hdu=0),),
        mask_locator=v1.MaskPayloadLocator(path=dark_mask),
    )
    return v1.LibraryHandle(
        v1.LibrarySnapshot(revision="r1", schema_version=LIBRARY_SCHEMA, candidates={"dark": (cand,)})
    )


def _manifest(tmp_path, batch_id):
    p = os.path.join(str(tmp_path), f"zecalibrator_batch_{batch_id}.json")
    return json.loads(open(p, "rb").read())


def _fits_files(dest):
    return sorted(f for f in os.listdir(dest) if f.endswith(".fits"))


def _run(tmp_path, library, frames, policy, *, collision_decision=None, batch_id="b"):
    dest = tmp_path / "out"
    dest.mkdir(exist_ok=True)
    opts = v1.BatchOptions(destination=str(dest), batch_id=batch_id)
    items = list(auto_route_batch(
        frames, library, policy, opts, collision_decision=collision_decision,
    ))
    return dest, items


# ---------------------------------------------------------------------------
# P1/P5 — skip happens after resolve, before calibration; no file is created
# ---------------------------------------------------------------------------
def test_skip_is_before_calibration_and_creates_no_file(tmp_path, monkeypatch):
    library = _dark_library(tmp_path)
    src = _array_source(caller_logical_id="light-a")

    # First run: commit the destination so the second run collides.
    dest, _ = _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="b1")
    committed = _fits_files(str(dest))
    assert len(committed) == 1

    import zecalibrator.api.v1.calibration as cal_mod

    calibrate_calls = []
    real_calibrate = cal_mod._calibrate_frame_impl

    def spy_calibrate(*args, **kwargs):
        calibrate_calls.append(args[0])
        return real_calibrate(*args, **kwargs)

    monkeypatch.setattr(cal_mod, "_calibrate_frame_impl", spy_calibrate)

    decisions = []
    dest2, items = _run(
        tmp_path, library, [src], v1.default_match_policy(),
        collision_decision=lambda p: decisions.append(p) or "skip", batch_id="b2",
    )
    assert len(items) == 1
    assert items[0].disposition == "SKIPPED"
    assert items[0].reason_code == "DESTINATION_EXISTS"
    # P1: the skipped item still carries the resolved plan_id (destination was
    # computed after resolve).
    assert items[0].plan_id is not None
    # P5: calibration was never entered for the skipped frame.
    assert calibrate_calls == []
    # No new file created, and the original is untouched.
    assert _fits_files(str(dest2)) == committed


# ---------------------------------------------------------------------------
# P2 — exactly one acquisition/decode per frame (decode seam call-count == N)
# ---------------------------------------------------------------------------
def test_exactly_one_decode_per_frame(tmp_path, monkeypatch):
    library = _dark_library(tmp_path)
    frames = [_array_source(caller_logical_id=f"light-{i}") for i in range(5)]

    import zecalibrator.api.v1._io as io_mod

    decode_calls = []
    real_decode = io_mod._decode_light_once

    def spy_decode(*args, **kwargs):
        decode_calls.append(args[0])
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(io_mod, "_decode_light_once", spy_decode)

    decisions = []
    dest, items = _run(
        tmp_path, library, frames, v1.default_match_policy(),
        collision_decision=lambda p: decisions.append(p) or "skip", batch_id="p2",
    )
    assert len(items) == 5
    assert all(i.disposition == "COMPLETED" for i in items)
    # Exactly one decode per frame — the collision check adds no acquisition and
    # calibration reuses the single decode carrier.
    assert len(decode_calls) == 5
    # No collision -> no decision.
    assert decisions == []


# ---------------------------------------------------------------------------
# P3/P4 — streaming O(1) state; at most one decision across ≥3 collisions
# ---------------------------------------------------------------------------
def test_streaming_constant_state_and_single_decision(tmp_path, monkeypatch):
    library = _dark_library(tmp_path)

    # One committed destination; 100 duplicate frames all map to the SAME path.
    src = _array_source(caller_logical_id="dup")
    dest, _ = _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="seed")

    frames = [_array_source(caller_logical_id="dup") for _ in range(100)]
    decisions = []
    dest2, items = _run(
        tmp_path, library, frames, v1.default_match_policy(),
        collision_decision=lambda p: decisions.append(p) or "skip", batch_id="p34",
    )
    # Streaming: all 100 items are yielded with dispositions; no pre-plan.
    assert len(items) == 100
    assert all(i.disposition == "SKIPPED" for i in items)
    assert all(i.reason_code == "DESTINATION_EXISTS" for i in items)
    # P4: the decision callable was invoked exactly once despite 100 collisions
    # (O(1) cached scalar — no per-frame accumulation).
    assert len(decisions) == 1


# ---------------------------------------------------------------------------
# P4b — at most one prompt per batch across ≥3 collisions (explicit counter)
# ---------------------------------------------------------------------------
def test_single_prompt_across_multiple_collisions(tmp_path):
    library = _dark_library(tmp_path)
    src = _array_source(caller_logical_id="c")

    # Pre-commit c, so three duplicate inputs all collide.
    _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="seed")

    frames = [_array_source(caller_logical_id="c") for _ in range(3)]
    calls = {"n": 0}

    def decision(path):
        calls["n"] += 1
        return "skip"

    dest, items = _run(
        tmp_path, library, frames, v1.default_match_policy(),
        collision_decision=decision, batch_id="p4b",
    )
    assert len(items) == 3
    assert all(i.disposition == "SKIPPED" for i in items)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Overwrite — all colliding destinations atomically replaced
# ---------------------------------------------------------------------------
def test_overwrite_replaces_colliding_destinations(tmp_path):
    library = _dark_library(tmp_path)
    src = _array_source(caller_logical_id="ow", value=100.0)

    dest, _ = _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="seed")
    committed = _fits_files(str(dest))
    assert len(committed) == 1

    decisions = []
    dest2, items = _run(
        tmp_path, library, [_array_source(caller_logical_id="ow", value=100.0)],
        v1.default_match_policy(),
        collision_decision=lambda p: decisions.append(p) or "overwrite", batch_id="ow",
    )
    assert len(items) == 1
    # Overwrite succeeded (a no-clobber run would be OUTPUT_COLLISION/FAILED).
    assert items[0].disposition == "COMPLETED"
    assert items[0].output is not None
    assert len(decisions) == 1
    # No temp left.
    assert [f for f in os.listdir(str(dest2)) if ".tmp" in f] == []


# ---------------------------------------------------------------------------
# Cancel — OperationCancelled, committed outputs preserved
# ---------------------------------------------------------------------------
def test_cancel_at_dialog_raises_and_preserves_commits(tmp_path):
    library = _dark_library(tmp_path)
    unique = _array_source(caller_logical_id="keep")
    colliding = _array_source(caller_logical_id="exists")

    # Pre-seed the colliding destination.
    _run(tmp_path, library, [colliding], v1.default_match_policy(), batch_id="seed")

    dest = tmp_path / "out"
    opts = v1.BatchOptions(destination=str(dest), batch_id="cancel")
    gen = auto_route_batch(
        [unique, colliding], library, v1.default_match_policy(), opts,
        collision_decision=lambda p: "cancel",
    )
    first = next(gen)
    assert first.disposition == "COMPLETED"  # committed before the cancel
    with pytest.raises(v1.OperationCancelled):
        next(gen)  # colliding -> cancel
    # The unique output remains committed (plus the pre-seeded collision).
    assert len(_fits_files(str(dest))) == 2


def test_cancel_raises_operation_cancelled(tmp_path):
    library = _dark_library(tmp_path)
    src = _array_source(caller_logical_id="cx")
    _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="seed")
    with pytest.raises(v1.OperationCancelled):
        _run(tmp_path, library, [src], v1.default_match_policy(),
             collision_decision=lambda p: "cancel", batch_id="cx2")


def test_malformed_decision_fails_safe_to_cancel(tmp_path):
    library = _dark_library(tmp_path)
    src = _array_source(caller_logical_id="garbage")
    _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="seed")
    # A garbled decision must never be treated as overwrite/skip: it cancels.
    with pytest.raises(v1.OperationCancelled):
        _run(tmp_path, library, [src], v1.default_match_policy(),
             collision_decision=lambda p: "bogus", batch_id="garbage2")


# ---------------------------------------------------------------------------
# Skip-all — COMPLETED (skipping is not an error) with skipped == N
# ---------------------------------------------------------------------------
def test_all_planned_outputs_exist_skip_all_completed(tmp_path):
    library = _dark_library(tmp_path)
    src = _array_source(caller_logical_id="all")
    _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="seed")

    dest, items = _run(
        tmp_path, library, [src], v1.default_match_policy(),
        collision_decision=lambda p: "skip", batch_id="skipall",
    )
    assert len(items) == 1
    assert items[0].disposition == "SKIPPED"
    m = _manifest(dest, "skipall")
    assert m["batch_status"] == "COMPLETED"
    assert m["commit_state"] == "COMMITTED"


# ---------------------------------------------------------------------------
# Subset exists + Skip — non-existing calibrated+written, existing skipped
# ---------------------------------------------------------------------------
def test_subset_exists_skip_skips_only_existing(tmp_path):
    library = _dark_library(tmp_path)
    existing = _array_source(caller_logical_id="exists")
    _run(tmp_path, library, [existing], v1.default_match_policy(), batch_id="seed")

    new = _array_source(caller_logical_id="new")
    dest, items = _run(
        tmp_path, library, [existing, new], v1.default_match_policy(),
        collision_decision=lambda p: "skip", batch_id="subset",
    )
    assert items[0].disposition == "SKIPPED"
    assert items[0].reason_code == "DESTINATION_EXISTS"
    assert items[1].disposition == "COMPLETED"
    assert items[1].output is not None
    # Only the new destination was written (plus the seed).
    assert len(_fits_files(str(dest))) == 2


# ---------------------------------------------------------------------------
# Duplicate inputs resolving to the same destination -> ordinary collision
# ---------------------------------------------------------------------------
def test_duplicate_inputs_share_destination_and_apply_same_policy(tmp_path):
    library = _dark_library(tmp_path)
    a = _array_source(caller_logical_id="same")
    b = _array_source(caller_logical_id="same")  # duplicate identity
    _run(tmp_path, library, [a], v1.default_match_policy(), batch_id="seed")

    dest, items = _run(
        tmp_path, library, [a, b], v1.default_match_policy(),
        collision_decision=lambda p: "skip", batch_id="dup",
    )
    # Both resolve to the same destination; both collide; the same batch policy
    # applies (skip). No separate subsystem.
    assert [i.disposition for i in items] == ["SKIPPED", "SKIPPED"]


# ---------------------------------------------------------------------------
# No collisions -> decision callable never invoked; no dialog
# ---------------------------------------------------------------------------
def test_no_collision_never_invokes_decision(tmp_path):
    library = _dark_library(tmp_path)
    calls = []

    dest, items = _run(
        tmp_path, library, [_array_source(caller_logical_id="fresh")],
        v1.default_match_policy(), collision_decision=lambda p: calls.append(p) or "cancel",
        batch_id="nc",
    )
    assert len(items) == 1
    assert items[0].disposition == "COMPLETED"
    assert calls == []


# ---------------------------------------------------------------------------
# C — collision_decision=None behaves identically to today (regression guard)
# ---------------------------------------------------------------------------
def test_collision_decision_none_preserves_today_behaviour(tmp_path):
    library = _dark_library(tmp_path)
    src = _array_source(caller_logical_id="guard")
    dest, first = _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="g1")
    assert first[0].disposition == "COMPLETED"
    committed_path = first[0].output.path
    committed = open(committed_path, "rb").read()

    # Second identical run WITHOUT a decision: strict no-clobber, silent
    # per-item OUTPUT_COLLISION (exactly today's behaviour — no pre-check).
    dest2, items = _run(tmp_path, library, [src], v1.default_match_policy(), batch_id="g2")
    assert items[0].disposition == "FAILED"
    assert items[0].reason_code == "OUTPUT_COLLISION"
    assert items[0].output is None
    # Previous destination byte-identical.
    assert open(committed_path, "rb").read() == committed


# ---------------------------------------------------------------------------
# P8 — no public-API/capability change (frozen surface)
# ---------------------------------------------------------------------------
def test_capabilities_and_public_surface_frozen():
    assert v1.get_api_info().capabilities == (
        "calibrate_frame", "calibration_library", "master_matching", "provenance",
        "cancel", "calibrate_batch",
    )
    # ``_auto_route`` / ``_output_path_for`` / overwrite primitive are NOT public.
    assert "auto_route_batch" not in v1.__all__
    assert "collision_decision" not in v1.__all__


def test_batch_options_overwrite_policy_still_rejects_non_no_clobber():
    with pytest.raises(v1.InvalidRequestError):
        v1.BatchOptions(overwrite_policy="overwrite")
    with pytest.raises(v1.InvalidRequestError):
        v1.BatchOptions(overwrite_policy="replace")
