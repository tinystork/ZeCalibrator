"""Headless tests for the GUI service layer (no PySide6 required).

Covers request construction, JSON -> public value-object parsing, immutable
snapshots, operation/batch id generation, JSON-safe rendering, and the static
private-import boundary (``zecalibrator.gui`` modules never import
``zecalibrator.core`` / ``zecalibrator.application`` / ``zecalibrator.io`` /
``zecalibrator.cli``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service


# ---------------------------------------------------------------------------
# JSON -> public value objects
# ---------------------------------------------------------------------------
def test_parse_hdu_integer_and_name():
    assert service.parse_hdu("0") == 0
    assert service.parse_hdu(" 3 ") == 3
    assert service.parse_hdu("SCI") == "SCI"


def test_parse_declaration_builds_public_value():
    decl = service.parse_declaration({
        "source": "synthetic_fixture", "identity": "SYNTH-BASE-1", "version": "1.0",
        "domain": "raw", "units": "ADU", "detector_instance_id": "SYNTH-DET-0001",
        "detector_model": "SYNTH-CFA", "gain": 100.0, "offset": 50.0,
        "readout_mode": "MODE_A", "adc_mode": "MODE_16", "binning": [1, 1],
        "sensor_dimensions": [4, 4], "orientation": "identity", "cfa_phase": "mono",
        "roi_origin": [0, 0], "exposure_s": 10.0, "temperature_c": 20.0,
        "filter": "NONE", "optical_train_id": "SYNTH-TRAIN-1",
        "bias_exposure_max_s": 0.01, "saturation_limit_adu": 60000.0,
        "saturation_evidence": "qualified",
    })
    assert isinstance(decl, v1.ImportDeclaration)
    assert decl.detector_instance_id == "SYNTH-DET-0001"
    assert decl.gain == 100.0


def test_parse_declaration_rejects_missing_identity():
    with pytest.raises((ValueError, TypeError)):
        service.parse_declaration({"source": "synthetic_fixture"})


def test_parse_declaration_rejects_contrary_domain():
    with pytest.raises(ValueError):
        service.parse_declaration({
            "source": "s", "identity": "i", "version": "1", "domain": "processed",
        })


def test_parse_roi_builds_public_value():
    roi = service.parse_roi({
        "extent": [4, 4], "source": "synthetic_fixture",
        "identity": "SYNTH-BASE-1", "version": "1.0",
    })
    assert isinstance(roi, v1.RoiExtentEvidence)
    assert roi.extent == (4, 4)


def test_parse_imports_builds_master_import_specs(tmp_path):
    imports = service.parse_imports([{
        "path": "dark.fits", "master_type": "dark", "hdu": 0,
        "mask_path": "dark.mask.npy", "bias_state": "included",
        "declaration": {
            "source": "synthetic_fixture", "identity": "SYNTH-BASE-1", "version": "1.0",
            "domain": "raw", "units": "ADU", "detector_instance_id": "SYNTH-DET-0001",
            "detector_model": "SYNTH-CFA", "gain": 100.0, "offset": 50.0,
            "readout_mode": "MODE_A", "adc_mode": "MODE_16", "binning": [1, 1],
            "sensor_dimensions": [4, 4], "orientation": "identity", "cfa_phase": "mono",
            "roi_origin": [0, 0], "exposure_s": 10.0, "temperature_c": 20.0,
            "filter": "NONE", "optical_train_id": "SYNTH-TRAIN-1",
            "bias_exposure_max_s": 0.01, "saturation_limit_adu": 60000.0,
            "saturation_evidence": "qualified",
        },
    }])
    assert len(imports) == 1
    assert isinstance(imports[0], v1.MasterImportSpec)
    assert imports[0].mask_path == "dark.mask.npy"
    assert imports[0].bias_state == "included"


def test_parse_imports_requires_mask_path_preserved():
    # parse_imports must not silently strip the mask_path (index_library requires it).
    imports = service.parse_imports([{
        "path": "dark.fits", "master_type": "dark", "hdu": 0,
        "mask_path": "dark.mask.npy", "bias_state": "included",
        "declaration": {
            "source": "synthetic_fixture", "identity": "SYNTH-BASE-1", "version": "1.0",
        },
    }])
    assert imports[0].mask_path == "dark.mask.npy"


def test_load_json_object_and_array(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"a": 1}))
    assert service.load_json_object(str(p)) == {"a": 1}
    q = tmp_path / "a.json"
    q.write_text(json.dumps([1, 2]))
    assert service.load_json_array(str(q)) == [1, 2]
    with pytest.raises(ValueError):
        service.load_json_object(str(q))
    with pytest.raises(ValueError):
        service.load_json_array(str(p))


# ---------------------------------------------------------------------------
# Ids / snapshot immutability
# ---------------------------------------------------------------------------
def test_operation_and_batch_ids_are_unique():
    assert service.new_operation_id() != service.new_operation_id()
    assert service.new_batch_id() != service.new_batch_id()


def test_snapshot_is_frozen_and_immutable():
    light = service.LightInput(
        path="/tmp/l.fits", hdu=0, declaration=None, roi_extent=None, display_name="l.fits"
    )
    snap = service.OperationSnapshot(
        op_id="op-1", kind="preflight", library_spec=None, request=None, policy=None,
        lights=(light,),
    )
    with pytest.raises(Exception):
        snap.lights = ()
    with pytest.raises(Exception):
        snap.op_id = "other"


def test_light_input_requires_path():
    with pytest.raises(ValueError):
        service.LightInput(path="", hdu=0, declaration=None, roi_extent=None, display_name="x")


def test_light_input_defaults_display_name():
    light = service.LightInput(path="/tmp/abc.fits", hdu=0, declaration=None, roi_extent=None, display_name="")
    assert light.display_name == "abc.fits"


def test_to_jsonable_handles_public_value_objects(tmp_path):
    spec = v1.LibrarySpec(root=str(tmp_path), index_path=str(tmp_path / "i.sqlite"))
    rendered = service.to_jsonable(spec)
    assert rendered["root"] == str(tmp_path)


# ---------------------------------------------------------------------------
# Shared supported-input extension set + folder scan + dedup (LIGHTS)
# ---------------------------------------------------------------------------
def test_supported_input_extensions_are_the_known_fits_suffixes():
    assert service.SUPPORTED_INPUT_EXTENSIONS == (".fits", ".fit", ".fts")


def test_input_dialog_filter_derives_from_shared_set():
    f = service.input_dialog_filter()
    for ext in service.SUPPORTED_INPUT_EXTENSIONS:
        assert f"*{ext}" in f
    assert f.endswith(";;All files (*)")


def test_is_supported_input_file_case_insensitive_and_deterministic():
    assert service.is_supported_input_file("light.fits")
    assert service.is_supported_input_file("light.fit")
    assert service.is_supported_input_file("light.fts")
    # Case handled identically (deterministic lower-case suffix match).
    assert service.is_supported_input_file("light.FITS")
    assert service.is_supported_input_file("light.Fit")
    assert service.is_supported_input_file("light.FTS")
    assert not service.is_supported_input_file("light.fits.gz")
    assert not service.is_supported_input_file("light.txt")
    assert not service.is_supported_input_file("light")
    assert not service.is_supported_input_file("light.jpg")


def test_path_identity_normcase_and_abspath():
    base = os.path.abspath("/tmp")
    assert service.path_identity("/tmp/x.fits") == os.path.normcase(os.path.abspath("/tmp/x.fits"))
    # Relative and absolute spellings of the same file collapse to one identity.
    assert service.path_identity("x.fits") == service.path_identity(os.path.join(os.getcwd(), "x.fits"))


def test_dedup_input_paths_preserves_order_and_skips_existing(tmp_path):
    a = str(tmp_path / "a.fits")
    b = str(tmp_path / "b.fit")
    (tmp_path / "a.fits").write_bytes(b"")
    (tmp_path / "b.fit").write_bytes(b"")
    out = service.dedup_input_paths([a, b, a, a], existing_paths=[b])
    assert out == [a]
    out2 = service.dedup_input_paths([a, b], existing_paths=[])
    assert out2 == [a, b]


def test_scan_folder_inputs_top_level_only_and_sorted(tmp_path):
    (tmp_path / "b.fit").write_bytes(b"")
    (tmp_path / "a.FITS").write_bytes(b"")
    (tmp_path / "c.fts").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "image.jpg").write_bytes(b"")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.fits").write_bytes(b"")  # must NOT be added (non-recursive)
    paths, unsupported = service.scan_folder_inputs(str(tmp_path))
    assert paths == [
        str(tmp_path / "a.FITS"),
        str(tmp_path / "b.fit"),
        str(tmp_path / "c.fts"),
    ]
    assert unsupported == 2  # notes.txt + image.jpg (subdir is not a file)


def test_scan_folder_inputs_empty_folder_clean(tmp_path):
    paths, unsupported = service.scan_folder_inputs(str(tmp_path))
    assert paths == []
    assert unsupported == 0


def test_scan_folder_inputs_rejects_non_directory(tmp_path):
    f = tmp_path / "afile.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        service.scan_folder_inputs(str(f))


def test_scan_folder_inputs_default_is_top_level_only(tmp_path):
    """B1 — default (non-recursive) scan sees top-level files only."""
    (tmp_path / "a.fit").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.fit").write_bytes(b"")
    paths, unsupported = service.scan_folder_inputs(str(tmp_path))
    assert paths == [str(tmp_path / "a.fit")]
    assert unsupported == 0


def test_scan_folder_inputs_explicit_non_recursive_matches_default(tmp_path):
    """B2 — explicit ``recursive=False`` is identical to the default call."""
    (tmp_path / "a.fit").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.fit").write_bytes(b"")
    default = service.scan_folder_inputs(str(tmp_path))
    explicit = service.scan_folder_inputs(str(tmp_path), recursive=False)
    assert explicit == default


def test_scan_folder_inputs_recursive_deterministic_and_deduped(tmp_path):
    """B3 — recursive scan discovers supported FITS at every depth in a
    deterministic order, with no duplicate paths (call twice, same result)."""
    (tmp_path / "a.fit").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.fit").write_bytes(b"")
    deeper = nested / "deeper"
    deeper.mkdir()
    (deeper / "c.FITS").write_bytes(b"")

    first = service.scan_folder_inputs(str(tmp_path), recursive=True)
    second = service.scan_folder_inputs(str(tmp_path), recursive=True)
    assert first == second
    assert first == (
        [str(tmp_path / "a.fit"), str(nested / "b.fit"), str(deeper / "c.FITS")],
        0,
    )


def test_scan_folder_inputs_recursive_extension_filter_identical(tmp_path):
    """B5 — extension filter is identical under recursion: non-FITS nested
    files count as unsupported and never enter the input paths."""
    (tmp_path / "a.fit").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.fit").write_bytes(b"")
    (nested / "notes.txt").write_text("x")
    (nested / "image.jpg").write_bytes(b"")
    (nested / "c.fts").write_bytes(b"")

    paths, unsupported = service.scan_folder_inputs(str(tmp_path), recursive=True)
    assert paths == [
        str(tmp_path / "a.fit"),
        str(nested / "b.fit"),
        str(nested / "c.fts"),
    ]
    assert unsupported == 2  # notes.txt + image.jpg (directories not counted)


def test_scan_folder_inputs_recursive_does_not_follow_dir_symlink(tmp_path):
    """B6 — a directory symlink pointing back to an ancestor must not cause
    infinite recursion and must not be followed (os.walk followlinks=False).
    Skipped gracefully when the platform forbids symlink creation."""
    (tmp_path / "a.fit").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.fit").write_bytes(b"")
    try:
        (nested / "loop").symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink creation not permitted on this platform")

    paths, unsupported = service.scan_folder_inputs(str(tmp_path), recursive=True)
    assert paths == [str(tmp_path / "a.fit"), str(nested / "b.fit")]
    assert unsupported == 0


# ---------------------------------------------------------------------------
# Static boundary: no private module imports under gui/
# ---------------------------------------------------------------------------
def test_gui_modules_never_import_private_paths():
    gui_dir = Path(__file__).resolve().parents[2] / "src" / "zecalibrator" / "gui"
    for py in sorted(gui_dir.glob("*.py")):
        src = py.read_text(encoding="utf-8")
        for mod in ("zecalibrator.core", "zecalibrator.application", "zecalibrator.io", "zecalibrator.cli"):
            assert f"import {mod}" not in src, (py.name, mod)
            assert f"from {mod}" not in src, (py.name, mod)


def test_gui_import_is_qt_free():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import zecalibrator.gui\n"
        "import zecalibrator.gui.service\n"
        "import zecalibrator.gui.settings\n"
        "import zecalibrator.gui.presentation\n"
        "import zecalibrator.gui.identity\n"
        "import zecalibrator.gui.app\n"
        "assert 'PySide6' not in sys.modules\n"
        "print('OK')\n"
    )
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src") + __import__("os").pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
