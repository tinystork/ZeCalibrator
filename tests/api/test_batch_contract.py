"""Public ``calibrate_batch`` contract tests (Phase 6 / M1).

Public-API-only tests (no ``zecalibrator.core`` / ``zecalibrator.application`` /
``zecalibrator.io`` module-path contract). Synthetic SYNTH-BASE-1 facts are
explicit, never real detector defaults.

Covered: batch 1 frame == calibrate_frame (dark science), batch N frames,
deterministic order, failure before first commit / mid-batch, cancellation
before start / between frames / during a frame, no partial output as final,
destination collision + no-clobber, temporary-file cleanup, atomic promotion,
batch provenance, monotonic progress, API headless, and local cross-platform
path semantics.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
from zecalibrator.api.v1 import (
    Acquisition,
    ArrayFrameSource,
    ArrayInputIdentity,
    BatchOptions,
    CalibrationRequest,
    CancellationToken,
    DetectorIdentity,
    ExecutionOptions,
    FitsInputIdentity,
    FrameInspection,
    Geometry,
    ImportDeclaration,
    LibraryHandle,
    LibrarySnapshot,
    MasterDescriptor,
    OpticalIdentity,
    ProgressObserver,
    SensorMetadata,
    ValidityEvidence,
    build_sensor_metadata,
    calibrate_frame,
    calibrate_batch,
    default_match_policy,
    resolve_calibration,
)

SHAPE = (4, 4)
LIBRARY_SCHEMA = "zecalibrator.library.v1"


def _geo():
    return Geometry(
        shape=SHAPE, sensor_dimensions=SHAPE, binning=(1, 1), roi_origin=(0, 0),
        roi_extent=SHAPE, orientation="identity", cfa_phase="mono",
    )


def _detector():
    return DetectorIdentity(detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA")


def _acquisition(exposure_s=10.0):
    return Acquisition(
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16",
        temperature_c=20.0, exposure_s=exposure_s, saturation_limit_adu=60000.0,
        saturation_evidence="qualified", bias_exposure_max_s=0.01, short_flat_profile=False,
    )


def _optical():
    return OpticalIdentity(filter="NONE", optical_train_id="SYNTH-TRAIN-1")


def _declaration(exposure_s=10.0):
    return ImportDeclaration(
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
    md = build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU"
    )
    return SensorMetadata(
        original_cards=md.original_cards, normalized=md.normalized, conflicts=md.conflicts,
        geometry=_geo(), raw_domain_declaration=md.raw_domain_declaration, units=md.units,
        declaration=md.declaration, exposure_s=md.exposure_s, temperature_c=md.temperature_c,
        gain=md.gain, offset=md.offset, readout_mode=md.readout_mode, adc_mode=md.adc_mode,
        filter=md.filter, detector_model=md.detector_model,
        detector_instance_id=md.detector_instance_id, optical_train_id=md.optical_train_id,
        saturation_limit_adu=md.saturation_limit_adu, saturation_evidence=md.saturation_evidence,
        warnings=md.warnings,
    )


def _inspection(exposure_s=10.0):
    return FrameInspection(
        metadata=_sensor_metadata(exposure_s=exposure_s),
        identity=FitsInputIdentity(path="/light.fits", hdu=0),
        domain_finding="raw", hdu=0, shape=SHAPE, warnings=(),
    )


def _library(candidates):
    return LibraryHandle(LibrarySnapshot(revision="r1", schema_version=LIBRARY_SCHEMA, candidates=candidates))


def _control_plan():
    return resolve_calibration(
        _inspection(), CalibrationRequest("control"), _library({}), default_match_policy()
    ).plan


def _array_source(data=100.0, exposure_s=10.0):
    return ArrayFrameSource(
        data=np.full(SHAPE, data, dtype=np.float32),
        metadata=_sensor_metadata(exposure_s=exposure_s),
        identity=ArrayInputIdentity(caller_logical_id="light-1"),
    )


def _dark_descriptor(content_sha256, size_bytes, mask_identity):
    return MasterDescriptor(
        master_type="dark", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state="included", geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=10.0), content_sha256=content_sha256,
        size_bytes=size_bytes, hdu=0, mask_identity=mask_identity,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=ValidityEvidence(saturation_limit_known=True),
    )


def _write_fits(path, data):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)
    return str(path)


def _write_mask(path, mask):
    np.save(path, np.asarray(mask, dtype=np.uint16), allow_pickle=False)
    return str(path)


def _manifest(tmp_path, batch_id):
    p = os.path.join(str(tmp_path), f"zecalibrator_batch_{batch_id}.json")
    return json.loads(open(p, "rb").read())


# ---------------------------------------------------------------------------
# Science parity / ordering / memory
# ---------------------------------------------------------------------------
def test_batch_single_frame_matches_calibrate_frame(tmp_path):
    library = _dark_library(tmp_path)
    plan = resolve_calibration(
        _inspection(), CalibrationRequest("dark_incl_bias"), library, default_match_policy()
    ).plan
    src = _array_source(100.0)
    items = list(
        calibrate_batch(
            [src], CalibrationRequest("dark_incl_bias"), library, default_match_policy(), BatchOptions()
        )
    )
    direct = calibrate_frame(src, plan, ExecutionOptions())
    assert items[0].disposition == direct.status
    assert np.allclose(items[0].result.data, direct.data)
    assert np.count_nonzero(items[0].result.mask) == np.count_nonzero(direct.mask)


def _dark_library(tmp_path):
    from zecalibrator.api.v1 import Candidate as _C, DescriptorSnapshot as _D, FitsFileLocator as _F, MaskPayloadLocator as _M

    dark_fits = str(tmp_path / "dark.fits")
    dark_mask = str(tmp_path / "dark.mask.npy")
    _write_fits(dark_fits, np.full(SHAPE, 10.0, dtype=np.float32))
    _write_mask(dark_mask, np.zeros(SHAPE, dtype=np.uint16))
    fb = open(dark_fits, "rb").read()
    mb = open(dark_mask, "rb").read()
    desc = _dark_descriptor(hashlib.sha256(fb).hexdigest(), len(fb), hashlib.sha256(mb).hexdigest())
    cand = _C(
        candidate_id="dark", descriptor=desc, descriptor_snapshot=_D(desc),
        locators=(_F(path=dark_fits, hdu=0),), mask_locator=_M(path=dark_mask),
    )
    return _library({"dark": (cand,)})


def test_batch_n_frames_deterministic_order(tmp_path):
    plan = _control_plan()
    srcs = [_array_source(float(v)) for v in (10.0, 20.0, 30.0)]
    items = list(
        calibrate_batch(srcs, CalibrationRequest("control"), _library({}), default_match_policy(), BatchOptions())
    )
    assert [i.index for i in items] == [0, 1, 2]
    assert [i.result.data[0, 0] for i in items] == [10.0, 20.0, 30.0]
    assert all(i.disposition == "COMPLETED" for i in items)


def test_batch_validation_refusal_raises(tmp_path):
    with pytest.raises(v1.InvalidRequestError):
        calibrate_batch([], "not-a-request", _library({}), default_match_policy(), BatchOptions())
    with pytest.raises(v1.InvalidRequestError):
        calibrate_batch([object()], CalibrationRequest("control"), _library({}), default_match_policy(), BatchOptions())


def test_batch_destination_must_be_existing_directory(tmp_path):
    # existing file -> InvalidRequestError (validation refusal, no auto-create)
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(v1.InvalidRequestError):
        BatchOptions(destination=str(f))
    # missing directory -> InvalidRequestError
    with pytest.raises(v1.InvalidRequestError):
        BatchOptions(destination=str(tmp_path / "missing"))


def test_batch_manifest_write_failure_is_typed_not_bare_oserror(tmp_path, monkeypatch):
    dest = tmp_path / "out"
    dest.mkdir()
    import zecalibrator.api.v1.batch as batch_mod

    def boom(path, manifest):
        raise OSError("disk full")

    monkeypatch.setattr(batch_mod, "write_batch_manifest", boom)
    gen = calibrate_batch(
        [_array_source(100.0)], CalibrationRequest("control"), _library({}),
        default_match_policy(), BatchOptions(destination=str(dest), batch_id="mw"),
    )
    first = next(gen)
    assert first.disposition == "COMPLETED"
    assert first.output is not None
    with pytest.raises(v1.BatchManifestError):
        next(gen)


def test_index_library_diagnostics_are_structured_dicts(tmp_path):
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "lib.sqlite"))
    imports = [
        v1.MasterImportSpec(
            path="missing.fits", master_type="dark", declaration=_declaration(),
            mask_path="missing.mask.npy", bias_state="included",
        )
    ]
    result = v1.index_library(spec, imports)
    assert result.operation_status == "COMPLETED"
    d = result.to_dict()
    assert isinstance(d["diagnostics"], list)
    for diag in d["diagnostics"]:
        assert isinstance(diag, dict)
        assert "path" in diag and "reason" in diag


# ---------------------------------------------------------------------------
# Failure dispositions
# ---------------------------------------------------------------------------
def test_batch_failure_before_first_commit(tmp_path):
    # dark_incl_bias with an empty library -> NO_MATCH for every frame.
    src = _array_source(100.0)
    dest = tmp_path / "out"
    dest.mkdir()
    items = list(
        calibrate_batch(
            [src], CalibrationRequest("dark_incl_bias"), _library({}),
            default_match_policy(), BatchOptions(destination=str(dest), batch_id="b1"),
        )
    )
    assert items[0].disposition == "FAILED"
    assert items[0].reason_code == "NO_MATCH"
    assert items[0].output is None
    # no output committed
    assert [f for f in os.listdir(str(dest)) if f.endswith(".fits")] == []


def test_batch_failure_mid_batch(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    good = _array_source(100.0)
    bad = v1.FitsFrameSource(path=str(tmp_path / "missing.fits"), declaration=_declaration())
    items = list(
        calibrate_batch(
            [good, bad], CalibrationRequest("control"), _library({}),
            default_match_policy(), BatchOptions(destination=str(dest), batch_id="b2"),
        )
    )
    assert items[0].disposition == "COMPLETED"
    assert items[0].output is not None
    assert items[1].disposition == "FAILED"
    assert items[1].output is None
    # first output committed, no partial output for the failed frame
    fits_files = sorted(f for f in os.listdir(str(dest)) if f.endswith(".fits"))
    assert len(fits_files) == 1


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------
def test_batch_cancellation_before_start(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    token = CancellationToken()
    token.cancel()
    gen = calibrate_batch(
        [_array_source(100.0)], CalibrationRequest("control"), _library({}),
        default_match_policy(), BatchOptions(destination=str(dest), batch_id="c0"),
        cancel=token,
    )
    with pytest.raises(v1.OperationCancelled):
        next(gen)
    m = _manifest(tmp_path / "out", "c0")
    assert m["commit_state"] == "CANCELLED"
    assert m["items"] == []


def test_batch_cancellation_between_frames(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    token = CancellationToken()

    def obs(event):
        if event.phase == "frame_complete" and event.completed == 1:
            token.cancel()

    gen = calibrate_batch(
        [_array_source(10.0), _array_source(20.0)], CalibrationRequest("control"),
        _library({}), default_match_policy(),
        BatchOptions(destination=str(dest), batch_id="c1"), cancel=token,
        progress=ProgressObserver(obs),
    )
    first = next(gen)
    assert first.disposition == "COMPLETED"
    with pytest.raises(v1.OperationCancelled):
        next(gen)
    m = _manifest(tmp_path / "out", "c1")
    assert m["commit_state"] == "CANCELLED"
    # only the first frame is recorded/committed
    assert [i["index"] for i in m["items"]] == [0]
    assert len([f for f in os.listdir(str(dest)) if f.endswith(".fits")]) == 1


def test_batch_cancellation_during_frame(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    token = CancellationToken()

    def obs(event):
        if event.phase == "frame_start":
            token.cancel()

    gen = calibrate_batch(
        [_array_source(10.0)], CalibrationRequest("control"), _library({}),
        default_match_policy(), BatchOptions(destination=str(dest), batch_id="c2"),
        cancel=token, progress=ProgressObserver(obs),
    )
    with pytest.raises(v1.OperationCancelled):
        next(gen)
    # no committed output for the cancelled in-flight frame
    assert [f for f in os.listdir(str(dest)) if f.endswith(".fits")] == []


# ---------------------------------------------------------------------------
# No partial output / collision / temp cleanup
# ---------------------------------------------------------------------------
def test_batch_no_partial_output_as_final_and_temp_cleanup(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    items = list(
        calibrate_batch(
            [_array_source(100.0), v1.FitsFrameSource(path=str(tmp_path / "nope.fits"), declaration=_declaration())],
            CalibrationRequest("control"), _library({}),
            default_match_policy(), BatchOptions(destination=str(dest), batch_id="b3"),
        )
    )
    # no temp files survive
    assert [f for f in os.listdir(str(dest)) if ".tmp" in f] == []
    # the failed item presents no output
    assert items[1].output is None
    m = _manifest(tmp_path / "out", "b3")
    assert m["batch_status"] == "PARTIAL"
    assert m["commit_state"] == "COMMITTED"


def test_batch_destination_collision_refuses_overwrite(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    src = _array_source(100.0)
    opts = BatchOptions(destination=str(dest), batch_id="col")

    items = list(calibrate_batch([src], CalibrationRequest("control"), _library({}), default_match_policy(), opts))
    assert items[0].disposition == "COMPLETED"
    committed_path = items[0].output.path
    original_bytes = open(committed_path, "rb").read()

    # Second identical run -> deterministic collision -> OUTPUT_COLLISION, no overwrite.
    items2 = list(calibrate_batch([src], CalibrationRequest("control"), _library({}), default_match_policy(), opts))
    assert items2[0].disposition == "FAILED"
    assert items2[0].reason_code == "OUTPUT_COLLISION"
    assert open(committed_path, "rb").read() == original_bytes


# ---------------------------------------------------------------------------
# Provenance / progress
# ---------------------------------------------------------------------------
def test_batch_manifest_provenance(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    items = list(
        calibrate_batch(
            [_array_source(100.0)], CalibrationRequest("control"), _library({}),
            default_match_policy(), BatchOptions(destination=str(dest), batch_id="prov"),
        )
    )
    m = _manifest(tmp_path / "out", "prov")
    assert m["schema_version"] == "zecalibrator.batch_manifest.v1"
    assert m["operation_id"] == "zecalibrator-calibrate-batch"
    assert m["commit_state"] == "COMMITTED"
    assert m["batch_status"] == "COMPLETED"
    assert m["matching_policy"] == "zecalibrator.match.v1"
    assert m["science_contract"] == "1.0"
    item = m["items"][0]
    assert item["disposition"] == "COMPLETED"
    assert item["output"]["whole_file_sha256"] == items[0].output.whole_file_sha256
    assert item["output"]["science_digest"] == items[0].output.science_digest
    # strict parse round-trips through the public value object
    bm = v1.BatchManifest.from_dict(m)
    assert bm.batch_id == "prov"
    assert bm.to_dict() == m


def test_batch_monotonic_progress(tmp_path):
    events = []
    result = list(
        calibrate_batch(
            [_array_source(10.0), _array_source(20.0)], CalibrationRequest("control"),
            _library({}), default_match_policy(), BatchOptions(),
            progress=ProgressObserver(events.append),
        )
    )
    assert len(result) == 2
    completed = [e.completed for e in events]
    assert completed == sorted(completed)
    assert events[-1].phase == "complete"
    assert events[-1].total == 2


def test_batch_callable_progress_normalized():
    events = []
    items = list(
        calibrate_batch(
            [_array_source(10.0)], CalibrationRequest("control"), _library({}),
            default_match_policy(), BatchOptions(), progress=events.append,
        )
    )
    assert len(items) == 1
    assert len(events) > 0


def test_batch_local_path_semantics_with_spaces_and_unicode(tmp_path):
    dest = tmp_path / "out put" / "üräum"
    dest.mkdir(parents=True)
    items = list(
        calibrate_batch(
            [_array_source(100.0)], CalibrationRequest("control"), _library({}),
            default_match_policy(), BatchOptions(destination=str(dest), batch_id="paths"),
        )
    )
    assert items[0].output is not None
    assert os.path.exists(items[0].output.path)
    m = _manifest(dest, "paths")
    assert m["items"][0]["output"]["path"] == items[0].output.path


def test_batch_api_surface_is_public_and_advertised():
    # calibrate_batch is a public symbol and an ADVERTISED capability (P6-M2);
    # index_library is public but NOT a separate capability (it is an operation of
    # the calibration_library capability).
    assert "calibrate_batch" in v1.__all__
    assert "index_library" in v1.__all__
    assert "calibrate_batch" in v1.get_api_info().capabilities
    assert "index_library" not in v1.get_api_info().capabilities
    assert v1.get_api_info().capabilities == (
        "calibrate_frame", "calibration_library", "master_matching", "provenance", "cancel",
        "calibrate_batch",
    )
