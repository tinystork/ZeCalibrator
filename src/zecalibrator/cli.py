"""ZeCalibrator command-line interface (Phase 6).

A **thin facade over ``zecalibrator.api.v1`` only**: this module never imports
``zecalibrator.core`` / ``zecalibrator.application`` / ``zecalibrator.io``
module paths directly. Commands: ``inspect``, ``index``, ``match``, ``calibrate``.

Exit contract (frozen ASTRA G6):
* ``0``   — all requested work committed;
* ``2``   — request/validation refusal;
* ``3``   — processing/partial failure;
* ``130`` — cancellation.

Output is deterministic, scriptable JSON (``sort_keys=True``, no timestamps, no
colors, no interactive wizard, no config/ZeAlfie discovery, no GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Iterable, Optional

import zecalibrator.api.v1 as v1

PROG = "zecalibrator"

_ADDITIVE_MODES = ("control", "bias_only", "dark_incl_bias", "dark_bias_removed")
_FLAT_MODES = ("none", "apply")

_DESCRIPTION = (
    "ZeCalibrator — raw/CFA FITS calibration with library matching, batch calibration and "
    "transactional outputs.\n\n"
    "Phase 6: bounded batch orchestration and transactional FITS output. No calibration "
    "science is added or modified beyond the frozen G3/G4/G5 engine; this CLI is a thin "
    "facade over zecalibrator.api.v1."
)


class _CliValidationError(Exception):
    """A request/validation refusal (mapped to exit code 2)."""


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    raise _CliValidationError(
        f"non-JSON value in CLI output: {type(value).__name__}"
    )


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(_json_safe(obj), sort_keys=True) + "\n")


def _parse_json(value: str, what: str):
    text = value
    if value.startswith("@"):
        text = Path(value[1:]).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _CliValidationError(f"invalid {what} JSON: {exc}") from exc


def _require_object(obj, what: str) -> dict:
    if not isinstance(obj, dict):
        raise _CliValidationError(f"{what} must be a JSON object")
    return obj


def _load_declaration(value: Optional[str]):
    if value is None:
        return None
    obj = _require_object(_parse_json(value, "declaration"), "declaration")
    return v1.ImportDeclaration(**obj)


def _load_roi(value: Optional[str]):
    if value is None:
        return None
    obj = _require_object(_parse_json(value, "roi-extent"), "roi-extent")
    return v1.RoiExtentEvidence(**obj)


def _parse_hdu(value: str):
    s = str(value).strip()
    try:
        return int(s)
    except ValueError:
        return s


def _build_scalars(value):
    if not isinstance(value, dict):
        raise _CliValidationError("normalization_scalars must be a JSON object")
    if set(value.keys()) == {"mono"}:
        return v1.NormalizationScalars(mono=float(value["mono"]))
    if set(value.keys()) == {"g1", "r", "b", "g2"}:
        return v1.NormalizationScalars(
            g1=float(value["g1"]), r=float(value["r"]), b=float(value["b"]), g2=float(value["g2"])
        )
    raise _CliValidationError("normalization_scalars must be {mono} or {g1,r,b,g2}")


def _build_norm_provenance(value):
    obj = _require_object(value, "normalization_provenance")
    return v1.NormalizationProvenance(
        algorithm=obj["algorithm"],
        population=obj["population"],
        scalars=_build_scalars(obj["scalars"]),
    )


def _build_validity(value):
    obj = _require_object(value, "validity_evidence")
    return v1.ValidityEvidence(
        saturation_limit_known=obj.get("saturation_limit_known", True),
        valid_normalization_count=obj.get("valid_normalization_count"),
        total_normalization_count=obj.get("total_normalization_count"),
        quality_policy_state=obj.get("quality_policy_state", "qualified"),
        illumination=obj.get("illumination"),
        exposure_quality=obj.get("exposure_quality"),
    )


def _build_processing(value):
    obj = _require_object(value, "processing_provenance")
    norm = obj.get("normalization")
    return v1.ProcessingProvenance(
        source=obj["source"],
        additive_correction_history=tuple(obj.get("additive_correction_history", ())),
        normalization=_build_norm_provenance(norm) if norm is not None else None,
        acquisition_profile=None,
    )


def _build_import_spec(value) -> v1.MasterImportSpec:
    obj = _require_object(value, "import spec")
    declaration = v1.ImportDeclaration(**obj["declaration"])
    kwargs = {
        "path": obj["path"],
        "master_type": obj["master_type"],
        "declaration": declaration,
        "hdu": obj.get("hdu", 0),
        "mask_path": obj.get("mask_path"),
        "bias_state": obj.get("bias_state"),
        "flat_form": obj.get("flat_form"),
    }
    if obj.get("normalization_algorithm") is not None:
        kwargs["normalization_algorithm"] = obj["normalization_algorithm"]
    if obj.get("normalization_scalars") is not None:
        kwargs["normalization_scalars"] = _build_scalars(obj["normalization_scalars"])
    if obj.get("normalization_provenance") is not None:
        kwargs["normalization_provenance"] = _build_norm_provenance(obj["normalization_provenance"])
    if obj.get("validity_evidence") is not None:
        kwargs["validity_evidence"] = _build_validity(obj["validity_evidence"])
    if obj.get("processing_provenance") is not None:
        kwargs["processing_provenance"] = _build_processing(obj["processing_provenance"])
    return v1.MasterImportSpec(**kwargs)


def _load_imports(value: Optional[str]) -> list:
    if value is None:
        raise _CliValidationError("index requires --declarations")
    arr = _parse_json(value, "declarations")
    if not isinstance(arr, list):
        raise _CliValidationError("declarations must be a JSON array")
    return [_build_import_spec(d) for d in arr]


def _open_library(index_path: str):
    index_path = os.path.abspath(index_path)
    root = os.path.dirname(index_path) or "."
    return v1.open_library(v1.LibrarySpec(root=root, index_path=index_path))


def _frame_sources(paths: Iterable[str], hdu, declaration, roi) -> list:
    return [
        v1.FitsFrameSource(path=p, hdu=hdu, declaration=declaration, roi_extent=roi)
        for p in paths
    ]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def _cmd_inspect(args, token) -> int:
    declaration = _load_declaration(args.declaration)
    roi = _load_roi(args.roi_extent)
    results = []
    for path in args.paths:
        source = v1.FitsFrameSource(
            path=path, hdu=args.hdu, declaration=declaration, roi_extent=roi
        )
        result = v1.inspect_frame(source, cancel=token)
        if result.operation_status == "CANCELLED":
            return 130
        results.append(
            {
                "path": path,
                "status": result.operation_status,
                "domain_finding": result.inspection.domain_finding if result.inspection else None,
                "hdu": result.inspection.hdu if result.inspection else None,
                "shape": list(result.inspection.shape) if result.inspection else None,
                "warnings": list(result.inspection.warnings) if result.inspection else [],
                "reason_code": result.reason_code,
                "details": result.details,
            }
        )
    _emit(results)
    return 3 if any(r["status"] == "FAILED" for r in results) else 0


def _cmd_index(args, token) -> int:
    imports = _load_imports(args.declarations)
    spec = v1.LibrarySpec(root=args.root, index_path=args.index_path)
    result = v1.index_library(spec, imports, revision=args.revision, cancel=token)
    _emit(result.to_dict())
    if result.operation_status == "CANCELLED":
        return 130
    if result.operation_status == "FAILED":
        return 3
    return 0


def _cmd_match(args, token) -> int:
    declaration = _load_declaration(args.declaration)
    roi = _load_roi(args.roi_extent)
    source = v1.FitsFrameSource(
        path=args.frame, hdu=args.hdu, declaration=declaration, roi_extent=roi
    )
    inspection = v1.inspect_frame(source, cancel=token)
    if inspection.operation_status == "CANCELLED":
        return 130
    if inspection.operation_status == "FAILED":
        _emit(
            {
                "status": "FAILED",
                "reason_code": inspection.reason_code,
                "details": inspection.details,
            }
        )
        return 3

    opened = _open_library(args.library)
    if opened.operation_status == "CANCELLED":
        return 130
    if opened.operation_status != "OPENED":
        _emit({"status": "FAILED", "reason_code": opened.reason_code, "details": opened.details})
        return 3

    library = opened.handle
    try:
        request = v1.CalibrationRequest(args.additive_mode, args.flat_mode)
        result = v1.resolve_calibration(
            inspection.inspection, request, library, v1.default_match_policy(), cancel=token
        )
    finally:
        library.close()

    if result.operation_status == "CANCELLED":
        return 130
    if result.operation_status == "FAILED":
        _emit(
            {
                "status": "FAILED",
                "reason_code": result.reason_code,
                "details": result.details,
            }
        )
        return 3

    _emit(
        {
            "status": "COMPLETED",
            "outcome": result.outcome,
            "plan_id": result.plan.plan_id if result.plan is not None else None,
            "reason_codes": list(result.decision.reason_codes),
            "details": result.details,
        }
    )
    return 0


def _cmd_calibrate(args, token) -> int:
    declaration = _load_declaration(args.declaration)
    roi = _load_roi(args.roi_extent)
    frames = _frame_sources(args.frames, args.hdu, declaration, roi)
    batch_id = args.batch_id or uuid.uuid4().hex
    # Eager destination validation: a standalone destination that is not an
    # existing directory raises InvalidRequestError (mapped to exit 2) before
    # any library open or per-item work.
    options = v1.BatchOptions(destination=args.destination, batch_id=batch_id)

    opened = _open_library(args.library)
    if opened.operation_status == "CANCELLED":
        return 130
    if opened.operation_status != "OPENED":
        _emit({"status": "FAILED", "reason_code": opened.reason_code, "details": opened.details})
        return 3

    library = opened.handle
    request = v1.CalibrationRequest(args.additive_mode, args.flat_mode)

    items = []
    cancelled = False
    manifest_error = None
    try:
        for item in v1.calibrate_batch(
            frames, request, library, v1.default_match_policy(), options, cancel=token
        ):
            items.append(item.to_dict())
    except v1.OperationCancelled:
        cancelled = True
    except v1.BatchManifestError as exc:
        manifest_error = str(exc)
    finally:
        library.close()

    if cancelled:
        return 130
    if manifest_error is not None:
        _emit(
            {
                "status": "FAILED",
                "reason_code": "MANIFEST_WRITE_FAILED",
                "details": manifest_error,
                "items": items,
            }
        )
        return 3

    batch_status = "PARTIAL" if any(i["disposition"] == "FAILED" for i in items) else "COMPLETED"
    manifest = (
        os.path.join(args.destination, f"zecalibrator_batch_{batch_id}.json")
        if args.destination is not None
        else None
    )
    _emit({"status": batch_status, "items": items, "manifest": manifest})
    return 3 if batch_status == "PARTIAL" else 0


# ---------------------------------------------------------------------------
# Parser / entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=_DESCRIPTION)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {v1.get_api_info().product_version}",
        help="show program's version number and exit",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    p_inspect = sub.add_parser("inspect", help="inspect FITS frames (raw decode + metadata)")
    p_inspect.add_argument("paths", nargs="+", help="FITS file path(s)")
    p_inspect.add_argument("--hdu", type=_parse_hdu, default=0, help="HDU index or name (default 0)")
    p_inspect.add_argument("--declaration", help="JSON object or @path for the raw-domain import declaration")
    p_inspect.add_argument("--roi-extent", help="JSON object or @path for the ROI-extent evidence")

    p_index = sub.add_parser("index", help="scan a library root and publish a versioned index")
    p_index.add_argument("root", help="library root directory")
    p_index.add_argument("--index-path", help="index path (default <root>/zecalibrator.library.sqlite)")
    p_index.add_argument("--declarations", help="JSON array (or @path) of master import specs")
    p_index.add_argument("--revision", help="revision label (default deterministic)")

    p_match = sub.add_parser("match", help="inspect a frame and resolve calibration")
    p_match.add_argument("frame", help="FITS light frame path")
    p_match.add_argument("--library", required=True, help="library index path")
    p_match.add_argument("--additive-mode", required=True, choices=_ADDITIVE_MODES)
    p_match.add_argument("--flat-mode", default="none", choices=_FLAT_MODES)
    p_match.add_argument("--hdu", type=_parse_hdu, default=0)
    p_match.add_argument("--declaration", help="JSON object or @path for the raw-domain import declaration")
    p_match.add_argument("--roi-extent", help="JSON object or @path for the ROI-extent evidence")

    p_cal = sub.add_parser("calibrate", help="calibrate one or more FITS frames (bounded batch)")
    p_cal.add_argument("frames", nargs="+", help="FITS light frame path(s)")
    p_cal.add_argument("--library", required=True, help="library index path")
    p_cal.add_argument("--additive-mode", required=True, choices=_ADDITIVE_MODES)
    p_cal.add_argument("--flat-mode", default="none", choices=_FLAT_MODES)
    p_cal.add_argument("--hdu", type=_parse_hdu, default=0)
    p_cal.add_argument("--declaration", help="JSON object or @path for the raw-domain import declaration")
    p_cal.add_argument("--roi-extent", help="JSON object or @path for the ROI-extent evidence")
    p_cal.add_argument("--destination", help="output directory for standalone FITS (omit for in-memory)")
    p_cal.add_argument("--batch-id", help="explicit batch id (default generated)")

    return parser


def main(argv: Optional[list] = None, *, cancel=None) -> int:
    """CLI entry point. Returns a process exit code (0/2/3/130)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse validation refusal (bad flags/choices) -> exit code 2.
        return int(exc.code) if exc.code not in (None, 0) else 0

    token = cancel if cancel is not None else v1.CancellationToken()

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "inspect":
            return _cmd_inspect(args, token)
        if args.command == "index":
            return _cmd_index(args, token)
        if args.command == "match":
            return _cmd_match(args, token)
        if args.command == "calibrate":
            return _cmd_calibrate(args, token)
    except _CliValidationError as exc:
        sys.stderr.write(f"zecalibrator: validation refusal: {exc}\n")
        return 2
    except v1.InvalidRequestError as exc:
        sys.stderr.write(f"zecalibrator: validation refusal: {exc}\n")
        return 2
    except v1.OperationCancelled:
        return 130
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 130

    return 2


__all__ = ["PROG", "build_parser", "main"]
