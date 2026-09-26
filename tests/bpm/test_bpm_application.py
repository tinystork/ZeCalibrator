"""LOT 2 batch-level run-wide BPM application tests.

Covers the run-wide application orchestration (``application.bpm_application``)
and the §11 synthesis invariant: the batch stays CALIBRATION_ONLY without a
base, a compatible map is applied automatically, a site is reconstructed on
every admissible frame, a donor-less frame abstains the site run-wide, no pixel
outside ``reconstructed_mask`` is modified, and the word "applied" appears only
when ``reconstructed_total > 0``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from zecalibrator.application.bpm_application import (
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_PREPARED,
    apply_bpm_run_wide,
    apply_resolution_run_wide,
    synthesis_lines,
)
from zecalibrator.bpm.identity import SensorIdentity
from zecalibrator.bpm.lookup import BpmResolution, select_revision
from zecalibrator.bpm.revision import make_revision
from zecalibrator.bpm.settings import BpmSettings
from zecalibrator.bpm.store import create_bad_pixel_database
from zecalibrator.bpm.vocabulary import REVISION_STATE_PROMOTED
from zecalibrator.storage import StoragePaths

from conftest import make_identity, make_rev, make_site

SHAPE = (12, 12)
CFA = "GRBG"

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def _storage(tmp_path) -> StoragePaths:
    return StoragePaths(
        user_config_path=tmp_path / "config",
        user_data_path=tmp_path / "data",
        user_cache_path=tmp_path / "cache",
        user_state_path=tmp_path / "state",
        user_log_path=tmp_path / "log",
    )


def _settings(root):
    if root is None:
        return BpmSettings()
    return BpmSettings(bad_pixel_database_root=str(root))


def make_calibration(data, mask=None):
    from zecalibrator.core.calibrate import CalibrationResult, FrameQuality
    from zecalibrator.core.dq import CountSummary

    data = np.asarray(data, dtype=np.float32)
    if mask is None:
        mask = np.zeros(data.shape, dtype=np.uint16)
    mask = np.asarray(mask, dtype=np.uint16)
    return CalibrationResult(
        status="COMPLETED",
        data=np.ascontiguousarray(data),
        mask=np.ascontiguousarray(mask),
        counts=CountSummary.from_mask(mask),
        frame_quality=FrameQuality(saturation_evidence="unknown"),
        precision=None,
        scalars={},
    )


def make_frame(frame_id, data, mask=None):
    from zecalibrator.bpm.preparation import CalibratedFrame

    return CalibratedFrame(frame_id=frame_id, calibration=make_calibration(data, mask))


def uniform(shape, value):
    return np.full(shape, value, dtype=np.float32)


def selected(identity, sites):
    rev = make_revision(
        state=REVISION_STATE_PROMOTED, sensor_identity=identity, sites=sites, sequence=1,
    )
    return select_revision(identity, [rev])


def _promoted_db(root, identity, sites):
    db = create_bad_pixel_database(root)
    db.add_revision(make_rev(identity, state=REVISION_STATE_PROMOTED, sites=sites, sequence=1))
    return db


# ---------------------------------------------------------------------------
# §14a — no BPM -> the batch stays CALIBRATION_ONLY
# ---------------------------------------------------------------------------

def test_no_base_is_calibration_only(tmp_path):
    storage = _storage(tmp_path)
    ident = make_identity(shape=SHAPE)
    frames = (make_frame("f0", uniform(SHAPE, 100.0)),)
    outcome = apply_bpm_run_wide(
        storage=storage, settings=_settings(None), identity=ident, frames=frames,
    )
    assert outcome.outcome == OUTCOME_CALIBRATION_ONLY
    assert outcome.reason_code == "NO_BASE"
    assert outcome.plan is None and outcome.results is None
    assert outcome.reconstructed_total == 0


def test_unqualified_profile_is_calibration_only(tmp_path):
    ident = make_identity(shape=SHAPE)
    resolution = BpmResolution(outcome=OUTCOME_CALIBRATION_ONLY, reason_code="UNQUALIFIED_PROFILE")
    outcome = apply_resolution_run_wide(resolution, (make_frame("f0", uniform(SHAPE, 100.0)),))
    assert outcome.outcome == OUTCOME_CALIBRATION_ONLY
    assert outcome.reconstructed_total == 0


def test_corrupt_base_is_typed_base_error(tmp_path):
    import json

    ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    rev = make_revision(
        state=REVISION_STATE_PROMOTED, sensor_identity=ident,
        sites=[make_site(5, 5)], sequence=1,
    )
    db.add_revision(rev)
    rev_path = root / "revisions" / f"{rev.revision_id}.json"
    obj = json.loads(rev_path.read_text(encoding="utf-8"))
    obj["sites"][0]["position"] = [1, 1]
    rev_path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    storage = _storage(tmp_path)
    frames = (make_frame("f0", uniform(SHAPE, 100.0)),)
    outcome = apply_bpm_run_wide(
        storage=storage, settings=_settings(root), identity=ident, frames=frames,
    )
    assert outcome.outcome == OUTCOME_BASE_ERROR
    assert outcome.provenance  # typed provenance, never a silent fallback


# ---------------------------------------------------------------------------
# §14 — a compatible map is used automatically; site reconstructed on all frames
# ---------------------------------------------------------------------------

def test_compatible_map_is_applied_automatically_and_reconstructed_on_all_frames(tmp_path):
    ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    _promoted_db(root, ident, [make_site(5, 5)])

    data = [uniform(SHAPE, 100.0) for _ in range(3)]
    for d in data:
        d[5, 5] = 9000.0  # the BPM site's anomalous value
    frames = (
        make_frame("f0", data[0]),
        make_frame("f1", data[1]),
        make_frame("f2", data[2]),
    )

    outcome = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=ident, frames=frames,
    )
    assert outcome.outcome == OUTCOME_PREPARED
    assert outcome.plan is not None and outcome.plan.frozen
    assert outcome.bad_pixel_count == 1
    # The site is reconstructed on EVERY admissible frame (never a mixed apply/skip).
    assert outcome.reconstructed_per_frame == (1, 1, 1)
    assert outcome.reconstructed_total == 3
    for result in outcome.results:
        assert result.reconstructed_mask[5, 5] == 1
        assert result.prepared_data[5, 5] == pytest.approx(100.0)


def test_site_abstains_run_wide_when_one_frame_lacks_donor(tmp_path):
    ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    _promoted_db(root, ident, [make_site(5, 5)])

    data = [uniform(SHAPE, 100.0) for _ in range(2)]
    for d in data:
        d[5, 5] = 9000.0
    mask1 = np.zeros(SHAPE, dtype=np.uint16)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if (dy, dx) == (0, 0):
                continue
            mask1[5 + dy, 5 + dx] = 0x0001  # frame 1 has no donors for the site

    frames = (make_frame("f0", data[0]), make_frame("f1", data[1], mask1))
    outcome = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=ident, frames=frames,
    )
    assert outcome.outcome == OUTCOME_PREPARED
    # Donor absence on ONE frame => the site abstains on the WHOLE run.
    assert outcome.reconstructed_per_frame == (0, 0)
    for result, expected in zip(outcome.results, data):
        assert result.reconstructed_mask.sum() == 0
        assert np.array_equal(result.prepared_data, expected)
        assert result.prepared_data[5, 5] == 9000.0


def test_blocked_site_does_not_disable_independent_site(tmp_path):
    ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    _promoted_db(root, ident, [make_site(3, 3), make_site(9, 9)])

    data0 = uniform(SHAPE, 100.0)
    data1 = uniform(SHAPE, 100.0)
    data0[3, 3] = data0[9, 9] = 9000.0
    data1[3, 3] = data1[9, 9] = 9000.0
    mask1 = np.zeros(SHAPE, dtype=np.uint16)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if (dy, dx) == (0, 0):
                continue
            mask1[3 + dy, 3 + dx] = 0x0001  # block site A on frame 1 only

    frames = (make_frame("f0", data0), make_frame("f1", data1, mask1))
    outcome = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=ident, frames=frames,
    )
    assert outcome.outcome == OUTCOME_PREPARED
    for result in outcome.results:
        assert result.reconstructed_mask[3, 3] == 0  # A blocked run-wide
        assert result.reconstructed_mask[9, 9] == 1  # B reconstructed everywhere
        assert result.prepared_data[3, 3] == 9000.0
        assert result.prepared_data[9, 9] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# §14 — no pixel outside reconstructed_mask is modified (B == A off-mask)
# ---------------------------------------------------------------------------

def test_no_pixel_outside_reconstructed_mask_is_modified(tmp_path):
    ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    _promoted_db(root, ident, [make_site(5, 5), make_site(7, 7)])

    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, SHAPE).astype(np.float32)
    data[5, 5] = 9000.0
    data[7, 7] = 8500.0
    frame = make_frame("f0", data)

    outcome = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=ident, frames=(frame,),
    )
    result = outcome.results[0]
    # B == A everywhere outside reconstructed_mask.
    off_mask = result.reconstructed_mask == 0
    assert np.array_equal(result.prepared_data[off_mask], data[off_mask])
    # On-mask pixels are the only modified pixels.
    assert np.count_nonzero(result.reconstructed_mask) == 2


# ---------------------------------------------------------------------------
# §14 — wrong sensor / wrong geometry -> map not used
# ---------------------------------------------------------------------------

def test_wrong_sensor_is_not_used(tmp_path):
    base_ident = make_identity(shape=SHAPE, detector_instance_id="DET-A")
    root = tmp_path / "base"
    _promoted_db(root, base_ident, [make_site(5, 5)])

    run_ident = make_identity(shape=SHAPE, detector_instance_id="DET-B")
    outcome = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=run_ident, frames=(make_frame("f0", uniform(SHAPE, 100.0)),),
    )
    assert outcome.outcome == OUTCOME_CALIBRATION_ONLY
    assert outcome.reason_code == "NO_PROFILE"


def test_wrong_geometry_is_not_used(tmp_path):
    base_ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    _promoted_db(root, base_ident, [make_site(5, 5)])

    run_ident = make_identity(shape=(14, 14))
    outcome = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=run_ident, frames=(make_frame("f0", uniform((14, 14), 100.0)),),
    )
    assert outcome.outcome == OUTCOME_CALIBRATION_ONLY
    assert outcome.reason_code == "NO_PROFILE"


# ---------------------------------------------------------------------------
# §14 — immutable revision; the next run finds the map automatically
# ---------------------------------------------------------------------------

def test_relaunch_finds_the_created_map_automatically(tmp_path):
    ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    rev = make_revision(
        state=REVISION_STATE_PROMOTED, sensor_identity=ident,
        sites=[make_site(5, 5)], sequence=1,
    )
    db.add_revision(rev)
    original_bytes = (root / "revisions" / f"{rev.revision_id}.json").read_bytes()

    data = uniform(SHAPE, 100.0)
    data[5, 5] = 9000.0
    frames = (make_frame("f0", data),)

    # Two independent runs (fresh resolution each time) find the same promoted map.
    first = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=ident, frames=frames,
    )
    second = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=ident, frames=frames,
    )
    assert first.outcome == OUTCOME_PREPARED
    assert second.outcome == OUTCOME_PREPARED
    assert first.resolution.revision.revision_id == rev.revision_id
    assert second.resolution.revision.revision_id == rev.revision_id
    # Immutability: the persisted revision bytes are unchanged by either run.
    assert (root / "revisions" / f"{rev.revision_id}.json").read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# §14 — the application path never fabricates a master (no file reading)
# ---------------------------------------------------------------------------

def test_application_module_never_reads_files():
    path = SRC_ROOT / "zecalibrator" / "application" / "bpm_application.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"open", "read_bytes", "read_text", "np.load", "glob", "listdir",
                 "fits", "astropy", "read_header", "FilesystemSource"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            pytest.fail(f"bpm_application.py references file-reading token {node.id!r}")
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in forbidden:
                pytest.fail(f"bpm_application.py calls file-reading token {name!r}")


# ---------------------------------------------------------------------------
# §14 — "applied" appears only when reconstructed_total > 0
# ---------------------------------------------------------------------------

def test_synthesis_applied_only_when_reconstructed_positive(tmp_path):
    ident = make_identity(shape=SHAPE)
    root = tmp_path / "base"
    _promoted_db(root, ident, [make_site(5, 5)])

    data = uniform(SHAPE, 100.0)
    data[5, 5] = 9000.0
    frames = (make_frame("f0", data),)

    applied = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(root),
        identity=ident, frames=frames,
    )
    assert applied.reconstructed_total > 0
    applied_text = "\n".join(synthesis_lines(applied, map_origin="created from <MasterDark>"))
    assert "BPM correction: applied" in applied_text
    assert "Bad Pixel Map: created from <MasterDark>" in applied_text
    assert "Bad pixels: 1" in applied_text
    assert "Reconstructed sites: 1" in applied_text

    # CALIBRATION_ONLY (no base) never says "applied".
    empty = apply_bpm_run_wide(
        storage=_storage(tmp_path), settings=_settings(None),
        identity=ident, frames=frames,
    )
    assert empty.reconstructed_total == 0
    empty_text = "\n".join(synthesis_lines(empty))
    assert "applied" not in empty_text
    assert "BPM correction: none" in empty_text
