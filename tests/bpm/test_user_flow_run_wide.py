"""LOT 3 user-flow tests (headless): the run-wide path wired into the normal
``calibrate_batch`` chain.

Verifies the two-time structure at the batch boundary: calibrate-all → freeze →
execute → corrected per-frame output. Covers:

* no base → CALIBRATION_ONLY (ordinary outputs, ``reconstructed_total == 0``);
* a compatible map → applied run-wide (``reconstructed_total > 0``), the site is
  reconstructed on every admissible frame, and the corrected CFA is what the
  prepared result carries;
* the plan is frozen once and no pixel outside ``reconstructed_mask`` changes;
* the seam reports a typed BASE_ERROR for a corrupt base without degrading the
  batch.
"""

from __future__ import annotations

import numpy as np

import zecalibrator.api.v1 as v1
from zecalibrator.api.v1 import _bpm
from zecalibrator.bpm.settings import BpmSettings
from zecalibrator.bpm.store import create_bad_pixel_database
from zecalibrator.bpm.vocabulary import REVISION_STATE_PROMOTED
from zecalibrator.storage import StoragePaths

from conftest import make_identity, make_rev, make_site

SHAPE = (12, 12)
CFA = "GRBG"
DET_MODEL = "SYNTH-MONO"


def _storage(tmp_path) -> StoragePaths:
    return StoragePaths(
        user_config_path=tmp_path / "config",
        user_data_path=tmp_path / "data",
        user_cache_path=tmp_path / "cache",
        user_state_path=tmp_path / "state",
        user_log_path=tmp_path / "log",
    )


def _base_with_site(tmp_path, sites):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    ident = make_identity(
        shape=SHAPE, cfa_phase=CFA,
        detector_instance_id="SYNTH-DET-0001", detector_model=DET_MODEL,
        gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16",
    )
    db.add_revision(make_rev(ident, state=REVISION_STATE_PROMOTED, sites=sites, sequence=1))
    return root, ident


def _decl():
    return v1.ImportDeclaration(
        source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
        domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
        detector_model=DET_MODEL, gain=100.0, offset=50.0,
        readout_mode="MODE_A", adc_mode="MODE_16", binning=(1, 1),
        sensor_dimensions=SHAPE, orientation="identity", cfa_phase=CFA,
        roi_origin=(0, 0), exposure_s=10.0, temperature_c=20.0, filter="NONE",
        optical_train_id="SYNTH-TRAIN-1", bias_exposure_max_s=0.01,
        saturation_limit_adu=60000.0, saturation_evidence="qualified",
    )


def _array_source(data, *, caller_id="light-1"):
    md = v1.build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=_decl(), units="ADU"
    )
    return v1.ArrayFrameSource(
        data=np.asarray(data, dtype=np.float32),
        metadata=md,
        identity=v1.ArrayInputIdentity(caller_logical_id=caller_id),
    )


def _empty_library():
    return v1.LibraryHandle(
        v1.LibrarySnapshot(revision="r1", schema_version="zecalibrator.library.v1", candidates={})
    )


def _control_plan():
    decl = _decl()
    md = v1.build_sensor_metadata(
        shape=SHAPE, normalized={}, conflicts=(), cards=(), declaration=decl, units="ADU"
    )
    inspection = v1.FrameInspection(
        metadata=md, identity=v1.ArrayInputIdentity(caller_logical_id="light"),
        domain_finding="raw", hdu=None, shape=SHAPE,
    )
    return v1.resolve_calibration(
        inspection, v1.CalibrationRequest("control"), _empty_library(),
        v1.default_match_policy(),
    ).plan


def _run_wide_batch(tmp_path, root, sources, *, destination=None):
    storage = _storage(tmp_path)
    settings = BpmSettings() if root is None else BpmSettings(bad_pixel_database_root=str(root))
    run_wide = _bpm.make_bpm_run_wide_seam(storage, settings, run_id="run-1")
    items = list(_bpm.calibrate_batch(
        sources, v1.CalibrationRequest("control"), _empty_library(),
        v1.default_match_policy(), v1.BatchOptions(destination=destination, batch_id="b1"),
        run_wide=run_wide,
    ))
    return items, run_wide


def _light(value=100.0, hot=()):
    data = np.full(SHAPE, value, dtype=np.float32)
    for y, x in hot:
        data[y, x] = 9000.0
    return data


def test_no_base_stays_calibration_only(tmp_path):
    sources = [_array_source(_light(hot=[(5, 5)]))]
    items, run_wide = _run_wide_batch(tmp_path, None, sources)
    assert run_wide.outcome is not None
    assert run_wide.outcome.outcome == "CALIBRATION_ONLY"
    assert run_wide.outcome.reconstructed_total == 0
    # The ordinary in-memory result is unchanged (no prepared substitution).
    assert items[0].disposition == "COMPLETED"
    assert run_wide.prepared(sources[0].identity.caller_logical_id) is None


def test_compatible_map_is_applied_run_wide(tmp_path):
    root, _ = _base_with_site(tmp_path, [make_site(5, 5)])
    sources = [_array_source(_light(hot=[(5, 5)]), caller_id="f0")]
    items, run_wide = _run_wide_batch(tmp_path, root, sources)
    assert run_wide.outcome.outcome == "PREPARED"
    assert run_wide.outcome.reconstructed_total == 1
    assert run_wide.outcome.reconstructed_per_frame == (1,)
    prepared = run_wide.prepared("f0")
    assert prepared is not None
    assert prepared.reconstructed_mask[5, 5] == 1
    assert prepared.prepared_data[5, 5] == 100.0
    assert items[0].disposition == "COMPLETED"


def test_site_reconstructed_on_every_admissible_frame(tmp_path):
    root, _ = _base_with_site(tmp_path, [make_site(5, 5)])
    sources = [
        _array_source(_light(hot=[(5, 5)]), caller_id="f0"),
        _array_source(_light(hot=[(5, 5)]), caller_id="f1"),
        _array_source(_light(hot=[(5, 5)]), caller_id="f2"),
    ]
    _items, run_wide = _run_wide_batch(tmp_path, root, sources)
    assert run_wide.outcome.reconstructed_per_frame == (1, 1, 1)
    assert run_wide.outcome.reconstructed_total == 3
    for cid in ("f0", "f1", "f2"):
        assert run_wide.prepared(cid).prepared_data[5, 5] == 100.0


def test_no_pixel_outside_reconstructed_mask_modified(tmp_path):
    root, _ = _base_with_site(tmp_path, [make_site(5, 5), make_site(7, 7)])
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, SHAPE).astype(np.float32)
    data[5, 5] = 9000.0
    data[7, 7] = 8500.0
    sources = [_array_source(data, caller_id="f0")]
    _items, run_wide = _run_wide_batch(tmp_path, root, sources)
    prepared = run_wide.prepared("f0")
    off_mask = prepared.reconstructed_mask == 0
    assert np.array_equal(prepared.prepared_data[off_mask], data[off_mask])
    assert np.count_nonzero(prepared.reconstructed_mask) == 2


def test_corrupt_base_is_typed_base_error_and_batch_preserved(tmp_path):
    import json

    root, _ = _base_with_site(tmp_path, [make_site(5, 5)])
    rev = root / "revisions"
    rev_path = list(rev.glob("*.json"))[0]
    obj = json.loads(rev_path.read_text(encoding="utf-8"))
    obj["sites"][0]["position"] = [1, 1]
    rev_path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    sources = [_array_source(_light(hot=[(5, 5)]), caller_id="f0")]
    items, run_wide = _run_wide_batch(tmp_path, root, sources)
    assert run_wide.outcome.outcome == "BASE_ERROR"
    # The ordinary calibration batch is preserved (no prepared substitution).
    assert items[0].disposition == "COMPLETED"
    assert run_wide.prepared("f0") is None


def test_plan_frozen_once_and_reported(tmp_path):
    root, _ = _base_with_site(tmp_path, [make_site(5, 5)])
    sources = [_array_source(_light(hot=[(5, 5)]), caller_id="f0")]
    _items, run_wide = _run_wide_batch(tmp_path, root, sources)
    assert run_wide.outcome.plan is not None
    assert run_wide.outcome.plan.frozen
    assert run_wide.outcome.bad_pixel_count == 1
