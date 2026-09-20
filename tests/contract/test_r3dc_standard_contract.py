"""R3D-C Standard master contract correction tests.

Standard applies CONVENTIONAL master semantics (the act of supplying a
calibration master carries contract semantics), not forensic proof:

* a supplied dark/darkflat defaults to ``bias_state="included"`` (bias included)
  unless provenance explicitly proves bias removal;
* a supplied flat defaults to ``flat_form="corrected_unnormalized"`` (a
  ready-to-use master flat, normalized at execution);
* an explicit ``raw_response`` flat is unsupported in Standard (never
  auto-constructed into a flat_dark dependency);
* missing acquisition facts (gain/offset/temperature/orientation/roi_origin) on
  a session-supplied master are UNVERIFIED (non-blocking) in the auto-route,
  while a known comparable mismatch stays blocking;
* the audit records the semantic source ``standard_master_contract`` honestly,
  never claiming FITS evidence for a contract default.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import zecalibrator.api.v1 as v1
from _phase4_fixtures import (
    acquisition,
    candidate,
    descriptor,
    geo,
    light,
    policy,
)
from zecalibrator.application.library import LibrarySnapshot
from zecalibrator.application.routes import resolve_route
from zecalibrator.core.descriptors import ProcessingProvenance, ValidityEvidence
from zecalibrator.core.routes import (
    FLAT_UNSUPPORTED_RAW,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_READY,
    STANDARD_MASTER_CONTRACT,
)

SHAPE = (4, 4)


def snapshot(**roles) -> LibrarySnapshot:
    return LibrarySnapshot(
        revision="r",
        schema_version="zecalibrator.library.v1",
        candidates={k: tuple(v) for k, v in roles.items()},
    )


# ---------------------------------------------------------------------------
# Descriptor defaults via the public index_library path (A.1 / A.2).
# ---------------------------------------------------------------------------
def _write_fits(path, header_cards=()):
    hdu = fits.PrimaryHDU(np.full(SHAPE, 1.0, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    for kw, val in header_cards:
        hdu.header[kw] = val
    hdu.writeto(path, overwrite=True)
    return str(path)


_DECL = dict(
    source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
    domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
    detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
    adc_mode="MODE_16", binning=(1, 1), sensor_dimensions=SHAPE,
    orientation="identity", cfa_phase="mono", roi_origin=(0, 0),
    exposure_s=10.0, temperature_c=20.0, filter="NONE",
    optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
    saturation_limit_adu=60000.0, saturation_evidence="qualified",
)


def _index_descriptor(tmp_path, master_type, *, bias_state=None, flat_form=None,
                      processing=None, declaration_overrides=None):
    decl = dict(_DECL)
    if declaration_overrides:
        decl.update(declaration_overrides)
    import_spec = v1.MasterImportSpec(
        path="master.fits", master_type=master_type,
        declaration=v1.ImportDeclaration(**decl),
        hdu=0, mask_path=None, dq_state="no_source_dq",
        bias_state=bias_state, flat_form=flat_form,
        processing_provenance=processing,
    )
    _write_fits(tmp_path / "master.fits")
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "lib.sqlite"))
    result = v1.index_library(spec, [import_spec])
    assert result.operation_status == "COMPLETED", result.details
    opened = v1.open_library(spec)
    assert opened.operation_status == "OPENED"
    handle = opened.handle
    try:
        role_key = master_type
        cands = handle.snapshot.candidates.get(role_key, ())
        assert len(cands) == 1
        return cands[0].descriptor
    finally:
        handle.close()


def test_dark_default_bias_included(tmp_path):
    desc = _index_descriptor(tmp_path, "dark")
    assert desc.bias_state == "included"


def test_darkflat_default_bias_included(tmp_path):
    desc = _index_descriptor(tmp_path, "flat_dark")
    assert desc.bias_state == "included"


def test_dark_explicit_bias_removed(tmp_path):
    processing = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("bias_removed",),
    )
    desc = _index_descriptor(tmp_path, "dark", processing=processing)
    assert desc.bias_state == "removed"


def test_dark_bias_removed_membership(tmp_path):
    # F2: membership (not singleton-tuple equality) — a known history that
    # *contains* bias_removed among other entries still proves bias removal.
    processing = ProcessingProvenance(
        source="synthetic_fixture",
        additive_history_state="known",
        additive_correction_history=("bias_removed", "dark_scaled"),
    )
    desc = _index_descriptor(tmp_path, "dark", processing=processing)
    assert desc.bias_state == "removed"


def test_flat_default_corrected_unnormalized(tmp_path):
    desc = _index_descriptor(tmp_path, "flat")
    assert desc.flat_form == "corrected_unnormalized"
    assert desc.pixel_domain == "sensor_adu"
    assert desc.physical_units == "ADU"


# ---------------------------------------------------------------------------
# Route classification under the contract (A + B).
# ---------------------------------------------------------------------------
def _contract_dark(bias_state="included", **acq_kw):
    return descriptor(
        "dark", bias_state,
        processing=ProcessingProvenance(source="synthetic_fixture"),
        acquisition_obj=acquisition(**acq_kw),
    )


def test_dark_default_routes_dark_incl_bias():
    lt = light()
    dk = _contract_dark()
    res = resolve_route(lt, snapshot(dark=[candidate("d1", dk)]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_incl_bias"
    assert res.plan is not None


def test_dark_explicit_bias_removed_routes_dark_bias_removed():
    lt = light()
    dk = descriptor(
        "dark", "removed",
        processing=ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
            additive_correction_history=("bias_removed",),
        ),
    )
    bias = descriptor("bias", "not_applicable", exposure_s=0.001)
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], bias=[candidate("b1", bias)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_bias_removed"
    assert set(res.plan.masters) == {"dark", "bias"}


def test_flat_default_applied_not_rejected():
    lt = light()
    dk = _contract_dark()
    flat = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        processing=ProcessingProvenance(source="synthetic_fixture"),
        exposure_s=1.0,
    )
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.flat_mode == "apply"
    assert res.route.flat_prep_mode == "normalize_only"
    assert "flat" in res.plan.masters


def test_flat_explicit_raw_response_needs_attention():
    lt = light()
    dk = _contract_dark()
    flat = descriptor(
        "flat", "not_applicable", flat_form="raw_response",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        processing=ProcessingProvenance(
            source="synthetic_fixture", additive_history_state="known",
        ),
        exposure_s=1.0,
    )
    fd = descriptor("flat_dark", "included", exposure_s=1.0)
    res = resolve_route(
        lt,
        snapshot(
            dark=[candidate("d1", dk)],
            flat=[candidate("f1", flat)],
            flat_dark=[candidate("fd1", fd)],
        ),
        policy(),
    )
    assert res.outcome == OUTCOME_NEEDS_ATTENTION
    assert res.plan is None
    assert any(r.code == FLAT_UNSUPPORTED_RAW for r in res.reasons)


def test_darkflat_is_flat_dependency_only_never_direct_dark():
    lt = light()
    dk = _contract_dark()
    fd = descriptor("flat_dark", "included", exposure_s=1.0)
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], flat_dark=[candidate("fd1", fd)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_incl_bias"
    # flat_dark is never consumed as a direct light correction.
    assert res.route.masters["dark"].descriptor.master_type == "dark"
    assert "flat_dark" not in res.plan.masters


# ---------------------------------------------------------------------------
# B. UNVERIFIED acquisition facts (auto-route, non-blocking when missing).
# ---------------------------------------------------------------------------
def test_missing_acquisition_facts_unverified_and_ready():
    lt = light()
    dk = descriptor(
        "dark", "included",
        processing=ProcessingProvenance(source="synthetic_fixture"),
        acquisition_obj=acquisition(gain=None, offset=None, temperature_c=None),
        geometry=geo(orientation=None, roi_origin=None),
    )
    res = resolve_route(lt, snapshot(dark=[candidate("d1", dk)]), policy())
    assert res.outcome == OUTCOME_READY
    assert res.route.additive_mode == "dark_incl_bias"
    # The missing facts are recorded as non-blocking UNVERIFIED notes.
    fields = {r.field for r in res.unverified}
    assert "acquisition.gain" in fields
    assert "acquisition.offset" in fields
    assert "acquisition.temperature_c" in fields
    assert all(not r.blocking for r in res.unverified)


def test_known_gain_mismatch_stays_blocking():
    lt = light()  # gain=100
    dk = descriptor(
        "dark", "included",
        processing=ProcessingProvenance(source="synthetic_fixture"),
        acquisition_obj=acquisition(gain=999),
    )
    res = resolve_route(lt, snapshot(dark=[candidate("d1", dk)]), policy())
    assert res.outcome == OUTCOME_NEEDS_ATTENTION
    assert any(r.code == "GAIN_MISMATCH" for r in res.reasons)
    assert all(r.blocking for r in res.reasons if r.code == "GAIN_MISMATCH")


# ---------------------------------------------------------------------------
# A.4. Audit honesty: source=standard_master_contract, never FITS evidence.
# ---------------------------------------------------------------------------
def test_audit_records_standard_master_contract_source():
    lt = light()
    dk = _contract_dark()  # bias_state="included" + additive_history_state="unknown"
    res = resolve_route(lt, snapshot(dark=[candidate("d1", dk)]), policy())
    assert res.outcome == OUTCOME_READY
    records = [dict(r) for r in res.contract_defaults]
    assert records
    bias_records = [r for r in records if r.get("field") == "bias_state"]
    assert bias_records
    record = bias_records[0]
    assert record["role"] == "dark"
    assert record["value"] == "included"
    assert record["source"] == STANDARD_MASTER_CONTRACT
    assert record["provenance_note"] == "processing provenance from FITS = unavailable"


# ---------------------------------------------------------------------------
# S1. An unrelated flat (known-and-different filter) must not force attention.
# ---------------------------------------------------------------------------
def test_unrelated_flat_filter_does_not_force_needs_attention():
    lt = light()  # filter="NONE"
    dk = _contract_dark()
    # A flat for a DIFFERENT filter (HA) is unrelated to this light.
    flat = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="HA", optical_train_id="SYNTH-TRAIN-1",
        processing=ProcessingProvenance(source="synthetic_fixture"),
        exposure_s=1.0,
    )
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    # The unrelated flat neither forces NEEDS_ATTENTION nor pollutes the audit.
    assert res.outcome == OUTCOME_READY
    assert res.route.flat_mode == "none"
    assert "flat" not in res.plan.masters
    assert not any(r.code == "FILTER_MISMATCH" for r in res.reasons)


# ---------------------------------------------------------------------------
# S2. A conventional evidence-less third-party flat (like a real Siril flat:
# corrected_unnormalized, unknown history, no counts/illumination/saturation)
# is USED under Standard, with the missing R4 facts recorded as UNVERIFIED.
# ---------------------------------------------------------------------------
def test_evidence_less_flat_used_under_standard():
    lt = light()
    dk = _contract_dark()
    flat = descriptor(
        "flat", "not_applicable", flat_form="corrected_unnormalized",
        filter="NONE", optical_train_id="SYNTH-TRAIN-1",
        processing=ProcessingProvenance(source="synthetic_fixture"),
        acquisition_obj=acquisition(saturation_evidence="unknown", exposure_s=1.0),
        validity=ValidityEvidence(saturation_limit_known=False),
        exposure_s=1.0,
    )
    res = resolve_route(
        lt,
        snapshot(dark=[candidate("d1", dk)], flat=[candidate("f1", flat)]),
        policy(),
    )
    assert res.outcome == OUTCOME_READY
    assert res.route.flat_mode == "apply"
    assert "flat" in res.plan.masters
    # Missing R4 facts are recorded as non-blocking UNVERIFIED (not rejected).
    fields = {r.field for r in res.unverified}
    assert "validity_evidence.saturation_limit_known" in fields
    assert "validity_evidence.illumination" in fields
    assert "validity_evidence.exposure_quality" in fields
    assert "validity_evidence.valid_normalization_count" in fields
    assert "acquisition.saturation_evidence" in fields
    assert all(not r.blocking for r in res.unverified)
