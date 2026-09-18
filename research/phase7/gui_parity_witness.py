"""Reproducible SYNTH-BASE-1 GUI parity witness.

Builds the same synthetic dark fixture used by the GUI parity tests, then runs
the identical calibration through the public API path and reports deterministic
JSON (plan id, calibrated status, science digest) so a reviewer can replay the
"same plan/result/provenance via API" reference without the Qt GUI present.

This is a small synthetic-only witness; it makes no real-camera claim. Run from
the product root:

    PYTHONPATH=src python research/phase7/gui_parity_witness.py --out /tmp/gui-witness.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

import zecalibrator.api.v1 as v1

SHAPE = (4, 4)

DECL = dict(
    source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0",
    domain="raw", units="ADU", detector_instance_id="SYNTH-DET-0001",
    detector_model="SYNTH-CFA", gain=100.0, offset=50.0, readout_mode="MODE_A",
    adc_mode="MODE_16", binning=[1, 1], sensor_dimensions=[4, 4],
    orientation="identity", cfa_phase="mono", roi_origin=[0, 0], exposure_s=10.0,
    temperature_c=20.0, filter="NONE", optical_train_id="SYNTH-TRAIN-1",
    bias_exposure_max_s=0.01, saturation_limit_adu=60000.0, saturation_evidence="qualified",
)
ROI = dict(extent=[4, 4], source="synthetic_fixture", identity="SYNTH-BASE-1", version="1.0")


def _write_fits(path, value):
    hdu = fits.PrimaryHDU(np.full(SHAPE, value, dtype=np.float32))
    hdu.header["BUNIT"] = "ADU"
    hdu.writeto(path, overwrite=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args(argv)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dark = root / "dark.fits"
        _write_fits(dark, 10.0)
        (root / "dark.mask.npy").write_bytes(
            _npy_bytes(np.zeros(SHAPE, dtype=np.uint16))
        )
        light = root / "light.fits"
        _write_fits(light, 100.0)

        imports = [dict(
            path="dark.fits", master_type="dark", hdu=0, mask_path="dark.mask.npy",
            bias_state="included", declaration=DECL,
        )]
        spec = v1.LibrarySpec(root=str(root), index_path=str(root / "zecalibrator.library.sqlite"))
        index = v1.index_library(spec, _build_imports(imports))
        assert index.operation_status == "COMPLETED", index.details

        source = v1.FitsFrameSource(
            path=str(light), hdu=0,
            declaration=v1.ImportDeclaration(**DECL),
            roi_extent=v1.RoiExtentEvidence(**ROI),
        )
        inspection = v1.inspect_frame(source).inspection
        opened = v1.open_library(spec)
        handle = opened.handle
        try:
            resolve = v1.resolve_calibration(
                inspection, v1.CalibrationRequest("dark_incl_bias", "none"),
                handle, v1.default_match_policy(),
            )
        finally:
            handle.close()
        assert resolve.outcome == "MATCHED", resolve.outcome

        result = v1.calibrate_frame(source, resolve.plan, v1.ExecutionOptions())
        assert result.status == "COMPLETED", result.status

        from zecalibrator.core.digests import science_digest

        witness = {
            "cohort": "SYNTH-BASE-1",
            "qualification": "synthetic-only",
            "plan_id": resolve.plan.plan_id,
            "status": result.status,
            "science_digest": science_digest(result.data, result.mask),
            "provenance_schema": result.provenance.provenance_schema,
            "matching_policy": result.provenance.matching_policy,
            "science_contract": result.provenance.science_contract,
            "scalars": dict(result.scalars),
            "operation_id": result.provenance.operation_id,
        }
        Path(args.out).write_text(
            json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(witness, indent=2, sort_keys=True))
    return 0


def _npy_bytes(arr):
    import io

    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.uint16), allow_pickle=False)
    return buf.getvalue()


def _build_imports(value):
    return [
        v1.MasterImportSpec(
            path=d["path"], master_type=d["master_type"], declaration=v1.ImportDeclaration(**d["declaration"]),
            hdu=d.get("hdu", 0), mask_path=d["mask_path"], bias_state=d.get("bias_state"),
        )
        for d in value
    ]


if __name__ == "__main__":
    sys.exit(main())
