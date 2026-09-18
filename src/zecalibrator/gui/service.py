"""GUI API client — pure, Qt-free request construction and value-object parsing.

This module is the *only* place under ``zecalibrator.gui`` that constructs the
public ``zecalibrator.api.v1`` request/value objects and parses user-supplied
JSON (import declarations, ROI-extent evidence, master import specs) into those
public value objects. It imports **only** ``zecalibrator.api.v1`` (the exported
stable contract) plus the standard library — never ``zecalibrator.core`` /
``zecalibrator.application`` / ``zecalibrator.io`` / ``zecalibrator.cli``.

The worker owns the actual expensive API calls; this module is deliberately
import-safe on a headless install (``zecalibrator.api.v1`` cold import is cheap
and Qt/NumPy-free until first lazy use).
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import zecalibrator.api.v1 as v1

# ---------------------------------------------------------------------------
# Identity helpers (GUI operation ids are distinct from the fixed engine ids)
# ---------------------------------------------------------------------------
def new_operation_id() -> str:
    """Return a fresh GUI operation id (128-bit hex, independent of engine ids)."""
    return uuid.uuid4().hex


def new_batch_id() -> str:
    """Return a fresh standalone batch id (128-bit hex)."""
    return uuid.uuid4().hex


def parse_hdu(value: str) -> Union[int, str]:
    """Parse an HDU selector exactly like the CLI (integer when numeric, else name)."""
    s = str(value).strip()
    try:
        return int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Immutable request snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LightInput:
    """One selected light frame, frozen for a worker operation.

    ``declaration`` / ``roi_extent`` are public immutable value objects (or
    ``None`` when the user did not supply evidence — never synthesized).
    ``row_id`` is the GUI's stable per-row identity so a cached plan can never
    be bound to another row with the same path/HDU but different evidence.
    """

    path: str
    hdu: object  # int | str
    declaration: object  # ImportDeclaration | None
    roi_extent: object  # RoiExtentEvidence | None
    display_name: str
    row_id: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("LightInput.path must be non-empty")
        if not self.display_name:
            object.__setattr__(self, "display_name", Path(self.path).name)

    def to_source(self) -> "v1.FitsFrameSource":
        return v1.FitsFrameSource(
            path=self.path, hdu=self.hdu,
            declaration=self.declaration, roi_extent=self.roi_extent,
        )


@dataclass(frozen=True)
class OperationSnapshot:
    """An immutable snapshot of everything one worker operation needs.

    All scientific value objects carried here are frozen public API value
    objects; the snapshot itself is frozen, so the worker cannot observe a
    half-constructed request and the GUI cannot mutate an in-flight request.
    """

    op_id: str
    kind: str  # open_library | index_library | preflight | calibrate_in_memory | export | load_declaration | load_roi
    library_spec: Optional["v1.LibrarySpec"]
    request: Optional["v1.CalibrationRequest"]
    policy: Optional["v1.MatchPolicy"]
    lights: Tuple[LightInput, ...]
    plan: Optional["v1.CalibrationPlan"] = None
    destination: Optional[str] = None
    batch_id: Optional[str] = None
    index_imports: Tuple["v1.MasterImportSpec", ...] = ()
    imports_path: Optional[str] = None
    index_revision: Optional[str] = None
    evidence_path: Optional[str] = None  # declaration/ROI JSON path
    target_indices: Tuple[int, ...] = ()  # light rows a declaration/ROI applies to
    config_dir: Optional[str] = None  # settings root (load_settings/save_settings)
    settings_payload: Optional[dict] = None  # GuiSettings.to_dict() for save

    def __post_init__(self) -> None:
        object.__setattr__(self, "lights", tuple(self.lights))
        object.__setattr__(self, "index_imports", tuple(self.index_imports))
        object.__setattr__(self, "target_indices", tuple(self.target_indices))


# ---------------------------------------------------------------------------
# JSON -> public value objects (fail-closed; no facts invented)
# ---------------------------------------------------------------------------
def _require_object(obj, what: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{what} must be a JSON object")
    return obj


def parse_declaration(value) -> "v1.ImportDeclaration":
    """Build an :class:`ImportDeclaration` from a decoded JSON object.

    The public model validates source/identity/version/domain/units/geometry and
    raises a ``ValueError`` on any invalid/contrary fact (never a silent default).
    """
    obj = _require_object(value, "declaration")
    try:
        return v1.ImportDeclaration(**dict(obj))
    except TypeError as exc:
        raise ValueError(f"invalid declaration: {exc}") from exc


def parse_roi(value) -> "v1.RoiExtentEvidence":
    """Build a :class:`RoiExtentEvidence` from a decoded JSON object."""
    obj = _require_object(value, "roi-extent")
    try:
        return v1.RoiExtentEvidence(**dict(obj))
    except TypeError as exc:
        raise ValueError(f"invalid roi-extent: {exc}") from exc


def _build_scalars(value) -> "v1.NormalizationScalars":
    if not isinstance(value, dict):
        raise ValueError("normalization_scalars must be a JSON object")
    if set(value.keys()) == {"mono"}:
        return v1.NormalizationScalars(mono=float(value["mono"]))
    if set(value.keys()) == {"g1", "r", "b", "g2"}:
        return v1.NormalizationScalars(
            g1=float(value["g1"]), r=float(value["r"]),
            b=float(value["b"]), g2=float(value["g2"]),
        )
    raise ValueError("normalization_scalars must be {mono} or {g1,r,b,g2}")


def _build_norm_provenance(value) -> "v1.NormalizationProvenance":
    obj = _require_object(value, "normalization_provenance")
    return v1.NormalizationProvenance(
        algorithm=obj["algorithm"],
        population=obj["population"],
        scalars=_build_scalars(obj["scalars"]),
    )


def _build_validity(value) -> "v1.ValidityEvidence":
    obj = _require_object(value, "validity_evidence")
    return v1.ValidityEvidence(
        saturation_limit_known=obj.get("saturation_limit_known", True),
        valid_normalization_count=obj.get("valid_normalization_count"),
        total_normalization_count=obj.get("total_normalization_count"),
        quality_policy_state=obj.get("quality_policy_state", "qualified"),
        illumination=obj.get("illumination"),
        exposure_quality=obj.get("exposure_quality"),
    )


def _build_processing(value) -> "v1.ProcessingProvenance":
    obj = _require_object(value, "processing_provenance")
    norm = obj.get("normalization")
    return v1.ProcessingProvenance(
        source=obj["source"],
        additive_correction_history=tuple(obj.get("additive_correction_history", ())),
        normalization=_build_norm_provenance(norm) if norm is not None else None,
        acquisition_profile=None,
    )


def _build_import_spec(value) -> "v1.MasterImportSpec":
    obj = _require_object(value, "import spec")
    if "declaration" not in obj:
        raise ValueError("import spec requires a 'declaration' object")
    declaration = v1.ImportDeclaration(**dict(obj["declaration"]))
    if "path" not in obj or "master_type" not in obj:
        raise ValueError("import spec requires 'path' and 'master_type'")
    kwargs: dict = {
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


def parse_imports(value) -> Tuple["v1.MasterImportSpec", ...]:
    """Build an ordered tuple of :class:`MasterImportSpec` from a decoded JSON array."""
    if not isinstance(value, list):
        raise ValueError("declarations must be a JSON array")
    return tuple(_build_import_spec(d) for d in value)


def load_json_object(path: str) -> dict:
    """Read and decode a small JSON file into a ``dict`` (used for evidence/imports)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path!r}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path!r} must decode to a JSON object")
    return obj


def load_json_array(path: str) -> list:
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path!r}: {exc}") from exc
    if not isinstance(obj, list):
        raise ValueError(f"{path!r} must decode to a JSON array")
    return obj


# ---------------------------------------------------------------------------
# JSON-safe rendering of public value objects (display only, never science)
# ---------------------------------------------------------------------------
def to_jsonable(value):
    """Render a public value object / container into strict-JSON-safe Python.

    Uses each object's public ``to_dict`` / ``to_mapping`` / dataclass fields;
    never assembles replacement scientific provenance.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "to_mapping"):
        return to_jsonable(value.to_mapping())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return str(value)


__all__ = [
    "LightInput",
    "OperationSnapshot",
    "load_json_array",
    "load_json_object",
    "new_batch_id",
    "new_operation_id",
    "parse_declaration",
    "parse_hdu",
    "parse_imports",
    "parse_roi",
    "to_jsonable",
]
