"""P8-A2 Stage 1 — no global/persistent state and import boundary (contract).

Proves the prepared calibration context is strictly batch-local and in-memory:

* two batches in one process share no context (each re-prepares its masters);
* no file is written under ``StoragePaths`` during a batch;
* importing the changed modules pulls in neither PySide6, nor ``multiprocessing``,
  nor ZeAlfie.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import zecalibrator.api.v1 as v1
import zecalibrator.api.v1._io as _io

SHAPE = (4, 4)


def _geo():
    return v1.Geometry(
        shape=SHAPE, sensor_dimensions=SHAPE, binning=(1, 1),
        roi_origin=(0, 0), roi_extent=SHAPE, orientation="identity", cfa_phase="mono",
    )


def _detector():
    return v1.DetectorIdentity(detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA")


def _acquisition(exposure_s=10.0):
    return v1.Acquisition(
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16",
        temperature_c=20.0, exposure_s=exposure_s, saturation_limit_adu=60000.0,
        saturation_evidence="qualified", bias_exposure_max_s=0.01, short_flat_profile=False,
    )


def _optical():
    return v1.OpticalIdentity(filter="NONE", optical_train_id="SYNTH-TRAIN-1")


def _declaration():
    return v1.ImportDeclaration(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0", domain="raw",
        units="ADU", detector_instance_id="SYNTH-DET-0001", detector_model="SYNTH-CFA",
        gain=100.0, offset=50.0, readout_mode="MODE_A", adc_mode="MODE_16", binning=(1, 1),
        sensor_dimensions=SHAPE, orientation="identity", cfa_phase="mono", roi_origin=(0, 0),
        exposure_s=10.0, temperature_c=20.0, filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        bias_exposure_max_s=0.01, saturation_limit_adu=60000.0, saturation_evidence="qualified",
    )


def _sensor_metadata():
    md = v1.build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=_declaration(), units="ADU"
    )
    geo = v1.Geometry(
        shape=md.geometry.shape, sensor_dimensions=md.geometry.sensor_dimensions,
        binning=md.geometry.binning, roi_origin=md.geometry.roi_origin, roi_extent=SHAPE,
        orientation=md.geometry.orientation, cfa_phase=md.geometry.cfa_phase,
    )
    return v1.SensorMetadata(
        original_cards=md.original_cards, normalized=md.normalized, conflicts=md.conflicts,
        geometry=geo, raw_domain_declaration=md.raw_domain_declaration, units=md.units,
        declaration=md.declaration, exposure_s=md.exposure_s, temperature_c=md.temperature_c,
        gain=md.gain, offset=md.offset, readout_mode=md.readout_mode, adc_mode=md.adc_mode,
        filter=md.filter, detector_model=md.detector_model,
        detector_instance_id=md.detector_instance_id, optical_train_id=md.optical_train_id,
        saturation_limit_adu=md.saturation_limit_adu, saturation_evidence=md.saturation_evidence,
        warnings=md.warnings,
    )


def _dark_descriptor(sha, size, mask_sha):
    return v1.MasterDescriptor(
        master_type="dark", pixel_domain="sensor_adu", physical_units="ADU", bias_state="included",
        geometry=_geo(), detector=_detector(), acquisition=_acquisition(exposure_s=10.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(source="synthetic_fixture"),
        validity_evidence=v1.ValidityEvidence(saturation_limit_known=True),
    )


def _flat_validity():
    return v1.ValidityEvidence(
        saturation_limit_known=True, valid_normalization_count={"mono": 95},
        total_normalization_count={"mono": 100}, quality_policy_state="qualified",
        illumination="flat_field", exposure_quality="qualified",
    )


def _corrected_flat_descriptor(sha, size, mask_sha):
    return v1.MasterDescriptor(
        master_type="flat", pixel_domain="sensor_adu", physical_units="ADU", bias_state="not_applicable",
        geometry=_geo(), detector=_detector(), acquisition=_acquisition(exposure_s=1.0),
        content_sha256=sha, size_bytes=size, hdu=0, mask_identity=mask_sha,
        processing_provenance=v1.ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            additive_correction_history=("bias_removed",),
        ),
        validity_evidence=_flat_validity(), flat_form="corrected_unnormalized",
        optical_train_id="SYNTH-TRAIN-1", filter="NONE",
    )


def _library(tmp_path):
    import hashlib

    from astropy.io import fits

    def _write(path, value):
        hdu = fits.PrimaryHDU(np.full(SHAPE, value, dtype=np.float32))
        hdu.header["BUNIT"] = "ADU"
        hdu.writeto(path, overwrite=True)

    def _mask(path):
        np.save(path, np.zeros(SHAPE, dtype=np.uint16), allow_pickle=False)

    dark_fits = tmp_path / "dark.fits"
    _write(dark_fits, 10.0)
    dark_mask = tmp_path / "dark.mask.npy"
    _mask(dark_mask)
    dsha = hashlib.sha256(dark_fits.read_bytes()).hexdigest()
    dsize = dark_fits.stat().st_size
    dmsha = hashlib.sha256(dark_mask.read_bytes()).hexdigest()

    flat_fits = tmp_path / "flat.fits"
    _write(flat_fits, 100.0)
    flat_mask = tmp_path / "flat.mask.npy"
    _mask(flat_mask)
    fsha = hashlib.sha256(flat_fits.read_bytes()).hexdigest()
    fsize = flat_fits.stat().st_size
    fmsha = hashlib.sha256(flat_mask.read_bytes()).hexdigest()

    dark = _dark_descriptor(dsha, dsize, dmsha)
    flat = _corrected_flat_descriptor(fsha, fsize, fmsha)

    def _candidate(cid, desc, p, mp):
        return v1.Candidate(
            candidate_id=cid, descriptor=desc, descriptor_snapshot=v1.DescriptorSnapshot(desc),
            locators=(v1.FitsFileLocator(path=str(p), hdu=desc.hdu),),
            mask_locator=v1.MaskPayloadLocator(path=str(mp)),
        )

    return v1.LibraryHandle(
        v1.LibrarySnapshot(
            revision="r1", schema_version="zecalibrator.library.v1",
            candidates={
                "dark": (_candidate("dark", dark, dark_fits, dark_mask),),
                "flat": (_candidate("flat", flat, flat_fits, flat_mask),),
            },
        )
    )


def _frames(n=10):
    metadata = _sensor_metadata()
    return [
        v1.ArrayFrameSource(
            data=np.full(SHAPE, 100.0, dtype=np.float32),
            metadata=metadata,
            identity=v1.ArrayInputIdentity(caller_logical_id=f"light-{i}"),
        )
        for i in range(n)
    ]


def test_two_batches_share_no_context(tmp_path, monkeypatch):
    handle = _library(tmp_path)
    request = v1.CalibrationRequest("dark_incl_bias", "apply")
    frames = _frames(10)

    decode_calls = []
    real_decode = _io.decode_fits_from_bytes

    def counting_decode(*args, **kwargs):
        decode_calls.append(1)
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(_io, "decode_fits_from_bytes", counting_decode)

    items1 = list(v1.calibrate_batch(frames, request, handle, v1.default_match_policy()))
    items2 = list(v1.calibrate_batch(frames, request, handle, v1.default_match_policy()))
    assert len(items1) == 10 and len(items2) == 10
    assert all(it.disposition in ("COMPLETED", "COMPLETED_WITH_WARNINGS") for it in items1)
    assert all(it.disposition in ("COMPLETED", "COMPLETED_WITH_WARNINGS") for it in items2)
    # dark ×1 + flat ×1 per batch ⇒ 4 total; a cross-batch shared context would
    # yield 2.
    assert len(decode_calls) == 4


def test_no_file_written_under_storage_paths(tmp_path, monkeypatch):
    from zecalibrator.storage import resolve_paths

    storage_base = tmp_path / "storage"
    storage_base.mkdir()
    paths = resolve_paths(base=str(storage_base))

    handle = _library(tmp_path)
    request = v1.CalibrationRequest("dark_incl_bias", "apply")
    frames = _frames(5)

    list(v1.calibrate_batch(frames, request, handle, v1.default_match_policy()))

    # Nothing was materialized under any StoragePaths root (the context is
    # strictly in-memory; only user-supplied master files live under tmp_path).
    created = [p for p in storage_base.rglob("*") if p != storage_base]
    assert created == []
    # No directory was created by resolving/using paths.
    assert not (paths.user_data_path).exists()
    assert not (paths.user_cache_path).exists()


def test_changed_modules_import_boundary():
    code = textwrap.dedent(
        """
        import sys
        import zecalibrator.api.v1.calibration
        import zecalibrator.api.v1.batch
        import zecalibrator.api.v1._auto_route
        forbidden = ("PySide6", "PySide2", "QtWidgets", "QtCore", "QtGui")
        for mod in forbidden:
            assert mod not in sys.modules, "unexpected import: " + mod
        assert "multiprocessing" not in sys.modules, "unexpected import: multiprocessing"
        for mod in list(sys.modules):
            low = mod.lower()
            assert not low.startswith("zealfie") and "zealfie" not in low, "unexpected import: " + mod
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
