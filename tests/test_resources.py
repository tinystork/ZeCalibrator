"""Bootstrap tests: package resources, icon manifest and interop declaration."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import zecalibrator._resources as res

# Canonical supplied-asset SHA-256 digests (pinned from the immutable mission
# handoff, ASTRA_MISSION §13). These are the byte-identical ground truth that
# the packaged copies and the manifest must match, independent of any checkout.
CANONICAL_SHA256 = {
    "zecalibrator_16x16.png": "7793ee217e847cc5b506f7d5bc56bb7a64bfd42ad309276cb6b0d592d78f8b20",
    "zecalibrator_24x24.png": "04bbb703d545b012bf63513662d1c5bd4197b92113427971e9f6a5fab5c9c8da",
    "zecalibrator_32x32.png": "2789d59d6cd2157c8f5d00c37fb8ef77cd2742fd2aa4945d6c407145d8be10a6",
    "zecalibrator_48x48.png": "cc66ea671658872b7c521e6bae4a7e8a4566b4b868a41819ee5ef8e98e3a1c87",
    "zecalibrator_64x64.png": "c5b050eff9b13c254e0901a5eca53fc05ebeb47700091f70acec70e4f3678cd5",
    "zecalibrator_128x128.png": "1be5ea466e395f59f52b445947e24daefcc7c1dea6cf2abaa015b6aba75f79dc",
    "zecalibrator_256x256.png": "205e6fb9260d0e19b71e37069aa753d975109cd36412624dae46eef67bdd4ece",
    "zecalibrator_512x512.png": "be09aa8b33ca0e17ee1ed269c22e48b3b345b7875edbe0e28e06d789199995d8",
    "zecalibrator_1024x1024.png": "c3582489faed7dfec694719618fd5661c8da2c9236d05a33edd6e0c1b9b18b51",
    "zecalibrator_icon.png": "7ad1589da6dd0d4b16780bd687ff79dec8fc999e4d0cc03431303117357b5ee4",
    "zecalibrator.ico": "717a04a871867de4cd32f90c3383d8414edbb2c960290ba346a1514e6bc584ca",
    "zecalibrator.icns": "a43b9b1c53aaedb98ee33c0591ce376a07947967490c67e7018ded2e2665dda6",
}


def test_manifest_loads_and_covers_all_canonical_files():
    manifest = res.load_icon_manifest()
    assert manifest["schema"] == "zecalibrator.resources.icons.v1"
    files_list = manifest["files"]
    assert len(files_list) == 12
    names = {entry["name"] for entry in files_list}
    assert names == set(CANONICAL_SHA256)


def test_manifest_hashes_match_canonical():
    manifest = res.load_icon_manifest()
    for entry in manifest["files"]:
        assert entry["sha256"] == CANONICAL_SHA256[entry["name"]], entry["name"]


def test_packaged_icon_bytes_match_manifest_and_canonical():
    manifest = res.load_icon_manifest()
    for entry in manifest["files"]:
        name = entry["name"]
        data = res.icon_bytes(name)
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], name
        assert hashlib.sha256(data).hexdigest() == CANONICAL_SHA256[name], name


def test_icon_bytes_rejects_path_traversal():
    for bad in ("", ".", "..", "a/b", "a\\b", "../x"):
        try:
            res.icon_bytes(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_exactly_one_interop_declaration_with_empty_provides():
    pkg = files("zecalibrator")
    matches = [p.name for p in pkg.iterdir() if p.name == "zesoftware_interop.json"]
    assert len(matches) == 1
    raw = (pkg / "zesoftware_interop.json").read_bytes()
    data = json.loads(raw.decode("utf-8"))
    assert data == {
        "schema": "zesoftware.interop.v1",
        "product_id": "zecalibrator",
        "distribution_name": "ZeCalibrator",
        "provides": [],
        "consumes": [],
    }
