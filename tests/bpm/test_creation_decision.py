"""LOT 3 map-creation decision tests (§3/§9 cases A–E, headless).

The GUI offers to create a Bad Pixel Map **at most once per run** and only when
a compatible Master Dark is selected; a compatible map is used automatically
with no question; a corrupt base never triggers an offer. These tests cover the
headless decision (:func:`zecalibrator.api.v1._bpm.bpm_creation_decision`).
"""

from __future__ import annotations

import json

import pytest

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


def _light_constraints(*, detector_instance_id="SYNTH-DET-0001", detector_model=DET_MODEL):
    from zecalibrator.core.descriptors import (
        Acquisition, DetectorIdentity, LightConstraints, OpticalIdentity,
    )
    from zecalibrator.core.geometry import Geometry

    return LightConstraints(
        geometry=Geometry(
            shape=SHAPE, sensor_dimensions=SHAPE, binning=(1, 1),
            roi_origin=(0, 0), roi_extent=SHAPE, orientation="identity", cfa_phase=CFA,
        ),
        detector=DetectorIdentity(
            detector_instance_id=detector_instance_id, detector_model=detector_model, serial=None,
        ),
        acquisition=Acquisition(gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16"),
        optical=OpticalIdentity(filter=None, optical_train_id=None),
    )


def _settings(root=None) -> BpmSettings:
    return BpmSettings() if root is None else BpmSettings(bad_pixel_database_root=str(root))


def _base_with_site(tmp_path, sites, *, detector_instance_id="SYNTH-DET-0001"):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    ident = make_identity(
        shape=SHAPE, cfa_phase=CFA,
        detector_instance_id=detector_instance_id, detector_model=DET_MODEL,
        gain=100, offset=50, readout_mode="MODE_A", adc_mode="MODE_16",
    )
    db.add_revision(make_rev(ident, state=REVISION_STATE_PROMOTED, sites=sites, sequence=1))
    return root


def test_compatible_map_is_used_without_question(tmp_path):
    root = _base_with_site(tmp_path, [make_site(5, 5)])
    decision = _bpm.bpm_creation_decision(
        _storage(tmp_path), _settings(root), _light_constraints(), dark_available=True,
    )
    assert decision["action"] == "use"


def test_no_map_with_dark_proposes_creation(tmp_path):
    # Empty base (no compatible revision) + a selected Master Dark -> propose.
    root = tmp_path / "base"
    create_bad_pixel_database(root)
    decision = _bpm.bpm_creation_decision(
        _storage(tmp_path), _settings(root), _light_constraints(), dark_available=True,
    )
    assert decision["action"] == "propose"
    assert decision["reason_code"] == "NO_PROFILE"


def test_no_map_without_dark_is_unavailable(tmp_path):
    root = tmp_path / "base"
    create_bad_pixel_database(root)
    decision = _bpm.bpm_creation_decision(
        _storage(tmp_path), _settings(root), _light_constraints(), dark_available=False,
    )
    # Never offer an impossible creation; the caller logs a discreet note.
    assert decision["action"] == "unavailable"
    assert decision["reason_code"] == "NO_PROFILE"


def test_no_base_is_missing_never_error(tmp_path):
    # No base at the configured root -> no compatible map, but never a typed error.
    decision = _bpm.bpm_creation_decision(
        _storage(tmp_path), _settings(tmp_path / "missing"), _light_constraints(), dark_available=True,
    )
    assert decision["action"] == "propose"
    assert decision["reason_code"] == "NO_BASE"


def test_corrupt_base_is_typed_error_never_propose(tmp_path):
    root = _base_with_site(tmp_path, [make_site(5, 5)])
    rev = list((root / "revisions").glob("*.json"))[0]
    obj = json.loads(rev.read_text(encoding="utf-8"))
    obj["sites"][0]["position"] = [999, 999]
    rev.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    decision = _bpm.bpm_creation_decision(
        _storage(tmp_path), _settings(root), _light_constraints(), dark_available=True,
    )
    assert decision["action"] == "error"
    assert decision["reason_code"] == "BASE_CORRUPTED"
