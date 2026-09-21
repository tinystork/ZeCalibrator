"""P8-A4 — decode once: equivalence, snapshot gate, call-count and no-cache contract.

P8-A4 defines the scientific snapshot of a frame as the byte content acquired by
its single acquisition. Inspection, routing and calibration consume that same
decoded snapshot and identity. A concurrent/transient mutation AFTER that
acquisition is outside A4's frame snapshot.

These tests pin the single-acquisition carrier (private ``DecodedLight`` /
``_decode_light_once`` in ``api/v1/_io.py``):

1. equivalence oracle, FILE PATH: a test-local two-pass reference (today's
   independent ``inspect_frame`` + ``calibrate_frame`` decodes) vs the decode-once
   carrier path — identical data/mask/counts/scalars/precision/status/warnings/
   reason_code AND identical ``provenance.to_dict()``;
2. the same oracle over the ``ArrayFrameSource`` (in-memory) path;
3. ADDITIONAL DISCRIMINATING SNAPSHOT GATE (owner-required): an instrumented
   reader whose hypothetical second invocation returns DIFFERENT bytes proves the
   light is read exactly once, the second payload is never requested, and both
   stages + provenance consume that same snapshot (never a race on a real file);
4. failure + cancellation precedence parity (single-failure mapping);
5. call-count proof: exactly ONE light FITS read/decode per frame on the eligible
   batch path, with masters/flat unchanged;
6. no implicit/global/persistent cache (carrier is loop-local; no module state).

No public symbol, no ``__all__`` entry, no module-level state is introduced.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
import zecalibrator.api.v1._io as _io
from zecalibrator.api.v1.calibration import _calibrate_frame_impl
from zecalibrator.api.v1.frames import _decode_and_inspect
from zecalibrator.core.digests import science_digest

SHAPE = (4, 4)


# ---------------------------------------------------------------------------
# Synthetic fixture builders (public API)
# ---------------------------------------------------------------------------
def _geo():
    return v1.Geometry(
        shape=SHAPE, sensor_dimensions=SHAPE, binning=(1, 1),
        roi_origin=(0, 0), roi_extent=SHAPE, orientation="identity", cfa_phase="mono",
    )


def _detector():
    return v1.DetectorIdentity(detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA")


def _acquisition(exposure_s=10.0, **kw):
    base = dict(
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16",
        temperature_c=20.0, exposure_s=exposure_s, saturation_limit_adu=60000.0,
        saturation_evidence="qualified", bias_exposure_max_s=0.01, short_flat_profile=False,
    )
    base.update(kw)
    return v1.Acquisition(**base)


def _declaration(**kw):
    base = dict(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
        detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
        adc_mode="MODE_16", binning=(1, 1), sensor_dimensions=SHAPE,
        orientation="identity", cfa_phase="mono", roi_origin=(0, 0),
        exposure_s=10.0, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0, saturation_evidence="qualified",
    )
    base.update(kw)
    return v1.ImportDeclaration(**base)


def _sensor_metadata(exposure_s=10.0):
    decl = _declaration(exposure_s=exposure_s)
    md = v1.build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU"
    )
    geo = v1.Geometry(
        shape=md.geometry.shape, sensor_dimensions=md.geometry.sensor_dimensions,
        binning=md.geometry.binning, roi_origin=md.geometry.roi_origin,
        roi_extent=SHAPE, orientation=md.geometry.orientation, cfa_phase=md.geometry.cfa_phase,
    )
    return v1.SensorMetadata(
        original_cards=md.original_cards, normalized=md.normalized, conflicts=md.conflicts,
        geometry=geo, raw_domain_declaration=md.raw_domain_declaration, units=md.units,
        declaration=md.declaration, exposure_s=md.exposure_s, temperature_c=md.temperature_c,
        gain=md.gain, offset=md.offset, readout_mode=md.readout_mode, adc_mode=md.adc_mode,
        filter=md.filter, detector_model=md.detector_model,
        detector_instance_id=md.detector_instance_id, optical_train_id=md.optical_train_id,
        saturation_limit_adu=md.saturation_limit_adu,
        saturation_evidence=md.saturation_evidence, warnings=md.warnings,
    )


def _write_fits(path, data, bunit="ADU"):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    if bunit:
        hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)
    return str(path)


def _write_mask(path, mask):
    np.save(path, np.asarray(mask, dtype=np.uint16), allow_pickle=False)
    return str(path)


def _write_master(tmp_path, name, value, bunit="ADU", mask=None):
    fits_path = tmp_path / f"{name}.fits"
    _write_fits(fits_path, np.full(SHAPE, value, dtype=np.float32), bunit=bunit)
    fits_bytes = fits_path.read_bytes()
    mask = np.zeros(SHAPE, dtype=np.uint16) if mask is None else np.asarray(mask, dtype=np.uint16)
    mask_path = tmp_path / f"{name}.mask.npy"
    _write_mask(mask_path, mask)
    mask_bytes = mask_path.read_bytes()
    return (
        str(fits_path), str(mask_path),
        hashlib.sha256(fits_bytes).hexdigest(), len(fits_bytes),
        hashlib.sha256(mask_bytes).hexdigest(),
    )


def _flat_validity():
    return v1.ValidityEvidence(
        saturation_limit_known=True,
        valid_normalization_count={"mono": 95},
        total_normalization_count={"mono": 100},
        quality_policy_state="qualified", illumination="flat_field", exposure_quality="qualified",
    )


def _dark_descriptor(sha, size, mask_sha, bias_state="included"):
    return v1.MasterDescriptor(
        master_type="dark", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state=bias_state, geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=10.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=v1.ValidityEvidence(saturation_limit_known=True),
    )


def _corrected_flat_descriptor(sha, size, mask_sha):
    return v1.MasterDescriptor(
        master_type="flat", pixel_domain="sensor_adu", physical_units="ADU",
        bias_state="not_applicable", geometry=_geo(), detector=_detector(),
        acquisition=_acquisition(exposure_s=1.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            additive_correction_history=("bias_removed",),
        ),
        validity_evidence=_flat_validity(),
        flat_form="corrected_unnormalized", optical_train_id="SYNTH-TRAIN-1", filter="NONE",
    )


def _binding(desc, fits_path, mask_path):
    return v1.MasterBinding(
        descriptor_id=desc.descriptor_id,
        descriptor_snapshot=v1.DescriptorSnapshot(desc),
        content_sha256=desc.content_sha256,
        size_bytes=desc.size_bytes,
        hdu=desc.hdu,
        mask_identity=desc.mask_identity,
        locators=(v1.FitsFileLocator(path=fits_path, hdu=desc.hdu),),
        mask_locator=v1.MaskPayloadLocator(path=mask_path),
    )


def _build_plan(request, masters, metadata):
    policy = v1.default_match_policy()
    return v1.CalibrationPlan.build(
        request=request,
        light_constraints=v1.light_constraints_from_sensor_metadata(metadata),
        masters=masters,
        policy_parameters=v1.PolicyParameters(
            exposure_tolerance=policy.exposure_tolerance,
            temperature_tolerance=policy.temperature_tolerance,
            flat_quality_policy=policy.flat_quality_policy,
        ),
        versions=v1.VersionSet(),
    )


def _array_source(data, metadata=None):
    return v1.ArrayFrameSource(
        data=np.asarray(data, dtype=np.float32),
        metadata=metadata if metadata is not None else _sensor_metadata(),
        identity=v1.ArrayInputIdentity(caller_logical_id="light-1"),
    )


def _make_scenario(tmp_path):
    """Build a (plan, array_source) dark_incl_bias + corrected_unnormalized flat."""
    metadata = _sensor_metadata()
    masters = {}
    dark_fits, dark_mask, dsha, dsize, dmsha = _write_master(tmp_path, "dark", 10.0)
    masters["dark"] = _binding(_dark_descriptor(dsha, dsize, dmsha), dark_fits, dark_mask)
    flat_fits, flat_mask, fsha, fsize, fmsha = _write_master(tmp_path, "flat", 100.0)
    masters["flat"] = _binding(_corrected_flat_descriptor(fsha, fsize, fmsha), flat_fits, flat_mask)
    plan = _build_plan(v1.CalibrationRequest("dark_incl_bias", "apply"), masters, metadata)
    source = _array_source(np.full(SHAPE, 100.0, dtype=np.float32), metadata=metadata)
    return plan, source


def _roi():
    return v1.RoiExtentEvidence(
        extent=SHAPE, source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
    )


def _fits_light(tmp_path, value=100.0, name="light.fits", bunit="ADU"):
    light_path = tmp_path / name
    _write_fits(light_path, np.full(SHAPE, value, dtype=np.float32), bunit=bunit)
    return v1.FitsFrameSource(path=str(light_path), declaration=_declaration(), roi_extent=_roi()), light_path


def _dark_flat_library(tmp_path):
    dark_fits, dark_mask, dsha, dsize, dmsha = _write_master(tmp_path, "dark", 10.0)
    flat_fits, flat_mask, fsha, fsize, fmsha = _write_master(tmp_path, "flat", 100.0)
    dark = _dark_descriptor(dsha, dsize, dmsha)
    flat = _corrected_flat_descriptor(fsha, fsize, fmsha)

    def _candidate(cid, desc, p, mp):
        return v1.Candidate(
            candidate_id=cid, descriptor=desc, descriptor_snapshot=v1.DescriptorSnapshot(desc),
            locators=(v1.FitsFileLocator(path=p, hdu=desc.hdu),),
            mask_locator=v1.MaskPayloadLocator(path=mp),
        )

    handle = v1.LibraryHandle(v1.LibrarySnapshot(
        revision="r1", schema_version="zecalibrator.library.v1",
        candidates={
            "dark": (_candidate("dark", dark, dark_fits, dark_mask),),
            "flat": (_candidate("flat", flat, flat_fits, flat_mask),),
        },
    ))
    paths = {
        "dark": dark_fits, "dark_mask": dark_mask,
        "flat": flat_fits, "flat_mask": flat_mask,
    }
    return handle, paths


# ---------------------------------------------------------------------------
# Oracle equality helper
# ---------------------------------------------------------------------------
def _assert_equivalent(a, b):
    assert a.status == b.status
    assert a.reason_code == b.reason_code
    assert a.warnings == b.warnings
    assert a.scalars == b.scalars
    if a.data is None:
        assert b.data is None
    else:
        assert np.array_equal(a.data, b.data, equal_nan=True)
        assert np.array_equal(a.mask, b.mask)
    if a.counts is None:
        assert b.counts is None
    else:
        assert a.counts.total == b.counts.total
        assert a.counts.valid_count == b.counts.valid_count
        assert a.counts.invalid_count == b.counts.invalid_count
        assert dict(a.counts.per_bit) == dict(b.counts.per_bit)
    if a.precision is None:
        assert b.precision is None
    else:
        assert a.precision == b.precision
    assert a.provenance.to_dict() == b.provenance.to_dict()


def _decode_once(source, plan, token):
    """Decode once and calibrate from the SAME carrier (the P8-A4 path)."""
    insp_result, carrier = _decode_and_inspect(source, token=token, obs=None)
    assert insp_result.operation_status == "COMPLETED", insp_result.details
    assert carrier is not None
    result = _calibrate_frame_impl(
        source, plan, v1.ExecutionOptions(), token=token, obs=None, slot=None,
        decoded_light=carrier,
    )
    return insp_result, carrier, result


# ---------------------------------------------------------------------------
# 1/2. Equivalence oracle (file path + array source)
# ---------------------------------------------------------------------------
def test_file_path_equivalence_oracle(tmp_path):
    plan, _ = _make_scenario(tmp_path)
    source, _ = _fits_light(tmp_path)

    # Two-pass reference = today's independent inspect + calibrate decodes.
    ref_inspection = v1.inspect_frame(source)
    assert ref_inspection.operation_status == "COMPLETED"
    ref = v1.calibrate_frame(source, plan, v1.ExecutionOptions())

    # Decode-once: a single carrier shared by inspection and calibration.
    token = v1.CancellationToken()
    insp_result, carrier, once = _decode_once(source, plan, token)

    _assert_equivalent(ref, once)
    assert ref.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    # The inspection consumed by decode-once is byte-identical to the public one.
    assert insp_result.inspection.identity == ref_inspection.inspection.identity
    # Inspection and calibration consumed the SAME identity/snapshot.
    assert insp_result.inspection.identity == carrier.identity
    assert once.provenance.input_identity == carrier.identity


def test_array_source_equivalence_oracle(tmp_path):
    plan, source = _make_scenario(tmp_path)

    ref_inspection = v1.inspect_frame(source)
    assert ref_inspection.operation_status == "COMPLETED"
    ref = v1.calibrate_frame(source, plan, v1.ExecutionOptions())

    token = v1.CancellationToken()
    insp_result, carrier, once = _decode_once(source, plan, token)

    _assert_equivalent(ref, once)
    assert ref.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert isinstance(carrier.identity, v1.ArrayInputIdentity)
    assert insp_result.inspection.identity == ref_inspection.inspection.identity
    assert insp_result.inspection.identity == carrier.identity
    assert once.provenance.input_identity == carrier.identity


# ---------------------------------------------------------------------------
# 3. ADDITIONAL DISCRIMINATING SNAPSHOT GATE (owner-required)
# ---------------------------------------------------------------------------
def test_snapshot_gate_single_acquisition(tmp_path, monkeypatch):
    plan, _ = _make_scenario(tmp_path)

    # Two distinct VALID FITS payloads for the "same" light path (never raced or
    # mutated on disk: the stub returns A on the first call and B if ever invoked
    # again).
    path_a = tmp_path / "light_A.fits"
    path_b = tmp_path / "light_B.fits"
    _write_fits(path_a, np.full(SHAPE, 100.0, dtype=np.float32))
    _write_fits(path_b, np.full(SHAPE, 200.0, dtype=np.float32))
    bytes_a = path_a.read_bytes()
    bytes_b = path_b.read_bytes()
    assert bytes_a != bytes_b

    source = v1.FitsFrameSource(path=str(path_a), declaration=_declaration(), roi_extent=_roi())

    # Reference science + identity from the REAL single acquisition (no stub yet).
    ref = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    assert ref.status in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    ref_inspection = v1.inspect_frame(source).inspection
    whole_sha_a = hashlib.sha256(bytes_a).hexdigest()
    assert ref_inspection.identity.whole_fits_sha256 == whole_sha_a
    ref_decoded_digest = ref_inspection.identity.decoded_digest

    # Instrument the reader: first light read returns A; any SECOND read returns
    # DIFFERENT bytes and records that it was (wrongly) requested.
    calls = {"light": 0, "second_requested": False}
    real_read_bytes = _io.read_bytes

    def stub_read_bytes(path, *, cancel=None):
        p = os.path.abspath(os.fspath(path))
        if p == os.path.abspath(str(path_a)):
            calls["light"] += 1
            if calls["light"] == 1:
                return bytes_a
            calls["second_requested"] = True
            return bytes_b
        return real_read_bytes(path, cancel=cancel)

    monkeypatch.setattr(_io, "read_bytes", stub_read_bytes)

    token = v1.CancellationToken()
    insp_result, carrier, once = _decode_once(source, plan, token)

    # The reader was invoked exactly ONCE for the light; the second payload was
    # NEVER requested.
    assert calls["light"] == 1
    assert calls["second_requested"] is False

    # whole_fits_sha256 corresponds to the first/only byte snapshot.
    assert carrier.identity.whole_fits_sha256 == whole_sha_a
    # decoded_digest corresponds to the decoded first/only snapshot.
    assert carrier.identity.decoded_digest == ref_decoded_digest
    assert carrier.identity.decoded_digest == science_digest(carrier.frame.data, carrier.frame.mask)

    # Inspection/routing AND calibration consume that SAME carrier/identity.
    assert insp_result.inspection.identity == carrier.identity
    assert once.provenance.input_identity == carrier.identity
    assert once.provenance.input_identity.whole_fits_sha256 == whole_sha_a
    assert once.provenance.input_identity.decoded_digest == ref_decoded_digest

    # The calibrated science corresponds to that SAME (first) snapshot.
    assert np.array_equal(once.data, ref.data, equal_nan=True)
    assert np.array_equal(once.mask, ref.mask)
    assert once.provenance.to_dict() == ref.provenance.to_dict()


# ---------------------------------------------------------------------------
# 4. Failure + cancellation precedence parity (single-failure mapping)
# ---------------------------------------------------------------------------
def test_decode_failure_precedence_parity(tmp_path):
    plan, _ = _make_scenario(tmp_path)
    # A light the strict decoder refuses (Jy units -> DecodeError).
    source, _ = _fits_light(tmp_path, name="bad.fits", bunit="Jy")

    # Public inspection mapping (unchanged) and the decode-once seam agree.
    pub = v1.inspect_frame(source)
    assert pub.operation_status == "FAILED"

    token = v1.CancellationToken()
    insp_result, carrier = _decode_and_inspect(source, token=token, obs=None)
    assert insp_result.operation_status == "FAILED"
    assert carrier is None
    assert insp_result.reason_code == pub.reason_code

    # Standalone calibration surfaces the SAME single failure (documented mapping:
    # a decode failure surfaces through the inspection mapping; both agree for a
    # deterministic input).
    standalone = v1.calibrate_frame(source, plan, v1.ExecutionOptions())
    assert standalone.status == "FAILED"
    assert standalone.reason_code == insp_result.reason_code


def test_cancellation_precedence_parity(tmp_path):
    plan, _ = _make_scenario(tmp_path)
    source, _ = _fits_light(tmp_path)

    token = v1.CancellationToken()
    token.cancel()
    insp_result, carrier = _decode_and_inspect(source, token=token, obs=None)
    assert insp_result.operation_status == "CANCELLED"
    assert insp_result.reason_code == "CANCELLED"
    assert carrier is None
    # Public inspection parity.
    assert v1.inspect_frame(source, cancel=token).operation_status == "CANCELLED"


# ---------------------------------------------------------------------------
# 5. Call-count proof: exactly ONE light read/decode per frame on the batch path
# ---------------------------------------------------------------------------
def test_batch_single_light_read_decode(tmp_path, monkeypatch):
    handle, master_paths = _dark_flat_library(tmp_path)
    request = v1.CalibrationRequest("dark_incl_bias", "apply")

    light_dir = tmp_path / "lights"
    light_dir.mkdir()
    frames = []
    for i in range(5):
        p = light_dir / f"light{i}.fits"
        _write_fits(p, np.full(SHAPE, 100.0, dtype=np.float32))
        frames.append(v1.FitsFrameSource(path=str(p), declaration=_declaration(), roi_extent=_roi()))

    light_paths = {os.path.abspath(str(f.path)) for f in frames}
    master_fits_paths = {os.path.abspath(str(master_paths["dark"])), os.path.abspath(str(master_paths["flat"]))}

    read_calls = {"light": 0, "master": 0}
    decode_calls = {"light": 0, "master": 0}
    real_read = _io.read_bytes
    real_decode = _io.decode_fits_from_bytes

    def counting_read(path, *, cancel=None):
        p = os.path.abspath(os.fspath(path))
        if p in light_paths:
            read_calls["light"] += 1
        elif p in master_fits_paths or p.endswith(".mask.npy"):
            read_calls["master"] += 1
        return real_read(path, cancel=cancel)

    def counting_decode(data, hdu, declaration, *, cancel=None, progress=None,
                        admission="light", role=None, flat_form=None):
        if admission == "light":
            decode_calls["light"] += 1
        else:
            decode_calls["master"] += 1
        return real_decode(data, hdu, declaration, cancel=cancel, progress=progress,
                           admission=admission, role=role, flat_form=flat_form)

    monkeypatch.setattr(_io, "read_bytes", counting_read)
    monkeypatch.setattr(_io, "decode_fits_from_bytes", counting_decode)

    items = list(v1.calibrate_batch(frames, request, handle, v1.default_match_policy()))
    assert len(items) == 5
    assert all(it.disposition in ("COMPLETED", "COMPLETED_WITH_WARNINGS") for it in items)

    # Exactly ONE light read + ONE light decode per frame.
    assert read_calls["light"] == 5
    assert decode_calls["light"] == 5
    # Masters/flat unchanged: dark + flat decoded once per batch (prepared once),
    # never scaled by the number of frames.
    assert decode_calls["master"] == 2
    # Master reads: dark.fits + dark.mask + flat.fits + flat.mask, prepared once.
    assert read_calls["master"] == 4


# ---------------------------------------------------------------------------
# 6. No implicit/global/persistent cache
# ---------------------------------------------------------------------------
def test_carrier_is_private_and_no_module_state(tmp_path, monkeypatch):
    # No public symbol / __all__ entry for the private carrier or helper.
    assert "DecodedLight" not in v1.__all__
    assert "_decode_light_once" not in v1.__all__
    assert not hasattr(v1, "DecodedLight")
    assert not hasattr(v1, "_decode_light_once")

    # No module-level carrier leak (the carrier is loop-local, never module state).
    module_state = {k: val for k, val in vars(_io).items() if not k.startswith("__")}
    for k, val in module_state.items():
        assert not isinstance(val, _io.DecodedLight), f"module-level carrier leak: {k}"

    # Two independent acquisitions re-read the file (no cross-call cache) and
    # produce two DISTINCT carriers.
    plan, _ = _make_scenario(tmp_path)
    source, light_path = _fits_light(tmp_path)

    read_calls = []
    real_read = _io.read_bytes

    def counting_read(path, *, cancel=None):
        if os.path.abspath(os.fspath(path)) == os.path.abspath(str(light_path)):
            read_calls.append(1)
        return real_read(path, cancel=cancel)

    monkeypatch.setattr(_io, "read_bytes", counting_read)

    token1 = v1.CancellationToken()
    _, carrier1 = _decode_and_inspect(source, token=token1, obs=None)
    token2 = v1.CancellationToken()
    _, carrier2 = _decode_and_inspect(source, token=token2, obs=None)

    assert len(read_calls) == 2
    assert carrier1 is not carrier2
    assert carrier1.identity == carrier2.identity  # same snapshot, distinct carrier objects


# ---------------------------------------------------------------------------
# G4 identity/hash truthfulness (file path)
# ---------------------------------------------------------------------------
def test_identity_hash_truthfulness_file_path(tmp_path):
    plan, _ = _make_scenario(tmp_path)
    source, light_path = _fits_light(tmp_path)
    raw_bytes = light_path.read_bytes()

    token = v1.CancellationToken()
    _, carrier = _decode_and_inspect(source, token=token, obs=None)

    ident = carrier.identity
    assert isinstance(ident, v1.FitsInputIdentity)
    assert ident.path == str(light_path)
    assert ident.hdu == 0
    assert ident.whole_fits_sha256 == hashlib.sha256(raw_bytes).hexdigest()
    assert ident.decoded_digest == science_digest(carrier.frame.data, carrier.frame.mask)

    facts = carrier.facts
    assert facts["input_dtype"] == carrier.frame.stored_dtype
    assert facts["input_units"] == carrier.frame.metadata.units
    assert facts["input_scaling"] == {"bscale": carrier.frame.bscale, "bzero": carrier.frame.bzero}
    assert facts["roi_extent_evidence"] is carrier.roi_extent_evidence
    assert carrier.roi_extent_evidence is source.roi_extent
    assert carrier.domain_finding == "raw"
    assert carrier.warnings == tuple(carrier.frame.metadata.warnings)
