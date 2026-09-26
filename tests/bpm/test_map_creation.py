"""LOT 1 map-creation tests: exact P4 detector recipe, v1 map policy, no master
fabrication, immutable storage + immediate lookup, determinism, identity
confinement, and headless/no-research guards.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from zecalibrator.bpm import map_creation as mc
from zecalibrator.bpm.errors import BpmError
from zecalibrator.bpm.map_creation import (
    MapCreationError,
    MapCreationResult,
    SelectedDarkMaster,
    create_bad_pixel_map,
    detect_site_positions,
    detect_sites,
)
from zecalibrator.bpm.revision import make_revision
from zecalibrator.bpm.store import create_bad_pixel_database
from zecalibrator.bpm.vocabulary import (
    ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION,
    KNOWLEDGE_STATE_QUALIFIED,
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_SELECTED,
    REVISION_STATE_CANDIDATE,
    REVISION_STATE_PROMOTED,
)

from conftest import make_identity, make_site

BPM_ROOT = Path(__file__).resolve().parents[2] / "src" / "zecalibrator" / "bpm"
SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "zecalibrator"

SHAPE = (16, 16)
CFA = "GRBG"


def _synthetic_dark(*, shape=SHAPE, hot=(), seed=0) -> np.ndarray:
    """A hand-built dark master: a flat base plus small noise plus hot pixels.

    ``hot`` lists sensor ``(y, x)`` positions given an extreme value so the
    recipe (median + 30 * 1.4826 * MAD) flags them — and only them.
    """
    rng = np.random.default_rng(seed)
    dark = (100.0 + rng.normal(0.0, 1.0, size=shape)).astype(np.float32)
    for y, x in hot:
        dark[y, x] = 1_000_000.0
    return dark


def _reference_positions(dark) -> list[tuple[int, int]]:
    """Independent re-implementation of the P4 recipe (test oracle).

    Literally re-derives the recipe rather than calling the module under test.
    """
    dark = np.asarray(dark)
    positions = set()
    for py in (0, 1):
        for px in (0, 1):
            sub = dark[py::2, px::2]
            med = np.median(sub)
            mad = np.median(np.abs(sub - med)) * 1.4826
            thr = med + 30.0 * mad
            ys, xs = np.nonzero(sub > thr)
            for yy, xx in zip(ys, xs):
                positions.add((int(2 * yy + py), int(2 * xx + px)))
    return sorted(positions)


def _master(identity=None, *, hot=(), shape=SHAPE, cfa_phase=CFA):
    ident = identity or make_identity(shape=shape, cfa_phase=cfa_phase)
    return SelectedDarkMaster(
        data=_synthetic_dark(shape=shape, hot=hot),
        identity=ident,
        metadata={"master_type": "dark", "source": "synthetic_fixture"},
    )


# ---------------------------------------------------------------------------
# §1 detection recipe (K=30, MAD*1.4826 per plane) — exact reproduction.
# ---------------------------------------------------------------------------
def test_detection_matches_reference_exactly():
    hot = [(0, 0), (0, 1), (1, 0), (1, 1), (7, 5), (15, 14)]
    dark = _synthetic_dark(hot=hot)
    got = detect_site_positions(dark)
    expected = _reference_positions(dark)
    assert got == tuple(expected)
    assert set(got) == set(hot)
    assert len(got) == len(hot)


def test_detection_handles_no_hot_pixels():
    dark = _synthetic_dark(hot=())
    assert detect_site_positions(dark) == ()


def test_k_is_30_and_not_exposed():
    assert mc._DETECTOR_K == 30.0
    assert mc._DETECTOR_MAD_SCALE == 1.4826
    # The constant is internal: never exported, never a parameter, never a knob.
    assert "_DETECTOR_K" not in mc.__all__
    assert "_DETECTOR_MAD_SCALE" not in mc.__all__
    for fn in (detect_site_positions, detect_sites, create_bad_pixel_map):
        params = set(inspect.signature(fn).parameters)
        assert not ({"k", "sigma", "threshold", "seuil"} & params)
    for name in ("k", "sigma", "threshold"):
        assert name not in SelectedDarkMaster.__dataclass_fields__


def test_no_user_surface_references_detector_constant():
    # No user surface (CLI/GUI/settings) references the internal detector token.
    offenders = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name == "map_creation.py":
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("_DETECTOR_K", "_DETECTOR_MAD_SCALE", "DETECTOR_K"):
            if token in text:
                offenders.append((str(path), token))
    assert not offenders, f"user surface references internal detector constants: {offenders}"


# ---------------------------------------------------------------------------
# §5 v1 map policy: every detected site is QUALIFIED + ELIGIBLE.
# ---------------------------------------------------------------------------
def test_detected_sites_are_qualified_and_eligible():
    hot = [(3, 4), (10, 10), (12, 7)]
    sites = detect_sites(_synthetic_dark(hot=hot))
    assert len(sites) == len(hot)
    for site in sites:
        assert site.knowledge_state == KNOWLEDGE_STATE_QUALIFIED
        assert site.action_state == ACTION_STATE_ELIGIBLE_FOR_TARGETED_RECONSTRUCTION
    assert {s.position for s in sites} == set(hot)


# ---------------------------------------------------------------------------
# §2 no master fabrication: no unit-dark read, signature takes an array.
# ---------------------------------------------------------------------------
def test_module_never_reads_unit_darks():
    src = (BPM_ROOT / "map_creation.py").read_text(encoding="utf-8")
    forbidden = (
        "open(",
        "read_bytes",
        "read_text",
        "np.load",
        "np.loadtxt",
        "np.fromfile",
        "fits",
        "astropy",
        "read_header",
        "enumerate_fits_files",
        "FilesystemSource",
        "listdir",
        "glob",
    )
    for token in forbidden:
        assert token not in src, f"map_creation.py references {token!r}"


def test_signature_consumes_selected_master_array():
    import typing

    hints = typing.get_type_hints(SelectedDarkMaster)
    assert hints["data"] is np.ndarray
    sig = inspect.signature(create_bad_pixel_map)
    assert "master" in sig.parameters
    assert "root" in sig.parameters
    # No path list, no directory of darks, no unit-dark argument anywhere.
    assert not any("dark" in p for p in sig.parameters if p != "master")


# ---------------------------------------------------------------------------
# §6 storage: new immutable revision, old remains, immediate lookup.
# ---------------------------------------------------------------------------
def test_creation_writes_candidate_and_promoted(tmp_path):
    root = tmp_path / "base"
    result = create_bad_pixel_map(root, _master(hot=[(2, 2)]))
    assert isinstance(result, MapCreationResult)
    assert result.candidate.state == REVISION_STATE_CANDIDATE
    assert result.promoted.state == REVISION_STATE_PROMOTED
    assert result.site_count == 1

    from zecalibrator.bpm.store import load_bad_pixel_database, STATE_OPENED

    load = load_bad_pixel_database(root)
    assert load.state == STATE_OPENED
    revs = load.database.revisions()
    assert {r.state for r in revs} == {REVISION_STATE_CANDIDATE, REVISION_STATE_PROMOTED}


def test_creation_is_new_revision_and_old_remains(tmp_path):
    root = tmp_path / "base"
    first = create_bad_pixel_map(root, _master(hot=[(1, 1), (2, 2)]))
    first_promoted_bytes = (
        (root / "revisions" / f"{first.promoted.revision_id}.json").read_bytes()
    )

    # A second creation (different evidence -> different sites) must add NEW
    # revisions and leave the first promoted revision byte-identical.
    second = create_bad_pixel_map(root, _master(hot=[(5, 5), (6, 6), (7, 7)]))
    assert second.promoted.revision_id != first.promoted.revision_id

    still = (root / "revisions" / f"{first.promoted.revision_id}.json").read_bytes()
    assert still == first_promoted_bytes

    from zecalibrator.bpm.store import load_bad_pixel_database

    revs = load_bad_pixel_database(root).database.revisions()
    ids = {r.revision_id for r in revs}
    assert first.promoted.revision_id in ids
    assert second.promoted.revision_id in ids


def test_created_map_is_selected_by_lookup(tmp_path):
    root = tmp_path / "base"
    ident = make_identity(shape=SHAPE, cfa_phase=CFA)
    result = create_bad_pixel_map(root, _master(identity=ident, hot=[(3, 3)]))

    from zecalibrator.bpm.store import load_bad_pixel_database

    res = load_bad_pixel_database(root).database.resolve(ident)
    assert res.outcome == OUTCOME_SELECTED
    assert res.revision is not None
    assert res.revision.revision_id == result.promoted.revision_id


def test_determinism_same_master_same_digest(tmp_path):
    master = _master(hot=[(4, 4), (9, 9)])
    a = create_bad_pixel_map(tmp_path / "a", master)
    b = create_bad_pixel_map(tmp_path / "b", master)
    assert a.candidate.revision_id == b.candidate.revision_id
    assert a.promoted.integrity_digest == b.promoted.integrity_digest
    assert a.promoted.sites == b.promoted.sites


def test_wrong_sensor_or_geometry_is_not_selected(tmp_path):
    root = tmp_path / "base"
    ident = make_identity(shape=SHAPE, cfa_phase=CFA, detector_instance_id="DET-A")
    create_bad_pixel_map(root, _master(identity=ident, hot=[(3, 3)]))

    from zecalibrator.bpm.store import load_bad_pixel_database

    db = load_bad_pixel_database(root).database
    # wrong detector -> not selected
    wrong_detector = make_identity(
        shape=SHAPE, cfa_phase=CFA, detector_instance_id="DET-B"
    )
    assert db.resolve(wrong_detector).outcome == OUTCOME_CALIBRATION_ONLY
    # wrong geometry -> not selected
    wrong_geometry = make_identity(
        shape=(16, 18), cfa_phase=CFA, detector_instance_id="DET-A"
    )
    assert db.resolve(wrong_geometry).outcome == OUTCOME_CALIBRATION_ONLY


def test_creation_rejects_shape_mismatch(tmp_path):
    ident = make_identity(shape=(16, 16), cfa_phase=CFA)
    with pytest.raises(MapCreationError):
        SelectedDarkMaster(
            data=np.zeros((16, 18), dtype=np.float32),
            identity=ident,
        )


def test_creation_rejects_corrupt_base(tmp_path):
    root = tmp_path / "base"
    db = create_bad_pixel_database(root)
    db.add_revision(
        make_revision(
            state=REVISION_STATE_PROMOTED,
            sensor_identity=make_identity(shape=SHAPE, cfa_phase=CFA),
            sites=[make_site(1, 1)],
            sequence=1,
        )
    )
    # Corrupt the only revision: its integrity digest no longer matches.
    rev_path = root / "revisions" / f"{db.revisions()[0].revision_id}.json"
    import json

    obj = json.loads(rev_path.read_text(encoding="utf-8"))
    obj["sites"][0]["position"] = [999, 999]
    rev_path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BpmError):
        create_bad_pixel_map(root, _master(hot=[(3, 3)]))


# ---------------------------------------------------------------------------
# §7 headless + no research import (AST).
# ---------------------------------------------------------------------------
def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").lstrip(".")
            if module:
                yield module


def test_map_creation_imports_no_research():
    for mod in _imported_modules(BPM_ROOT / "map_creation.py"):
        assert not (mod == "research" or mod.startswith("research."))


def test_map_creation_is_headless():
    qt_prefixes = ("PySide6", "PyQt5", "PyQt6", "PySide2", "tkinter")
    for mod in _imported_modules(BPM_ROOT / "map_creation.py"):
        top = mod.split(".")[0]
        assert top not in qt_prefixes and top != "zecalibrator.gui"
