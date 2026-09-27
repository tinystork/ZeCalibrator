"""Managed master ingestion value objects (P7-M3B): EvidenceFact + ManagedMasterRecord.

These immutable value objects define the *managed ingestion* evidence model —
per-field provenance annotations (:class:`EvidenceFact`) and the managed-master
record (:class:`ManagedMasterRecord`).  They live in the ``core`` layer (like
:class:`~zecalibrator.core.metadata.ImportDeclaration`) so that both the public
API facade (``zecalibrator.api.v1.models`` re-exports them unchanged) and the
application-layer admission logic (:mod:`zecalibrator.application.masters`) can
depend on them without inverting the layering.

This module imports only the standard library, NumPy, and sibling ``core``
modules — never ``api.v1``, never Qt/ZeAlfie/ZSSS.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Union

import numpy as np

from zecalibrator.core.descriptors import DQ_STATES
from zecalibrator.core.errors import InvalidRequestError
from zecalibrator.core.metadata import ImportDeclaration

MANAGED_LEDGER_SCHEMA = "zecalibrator.managed_ledger.v1"

# Evidence origin types (frozen vocabulary).
EVIDENCE_ORIGIN_TYPES: tuple[str, ...] = ("fits_header", "user", "sidecar", "native")


def _freeze_evidence_value(value):
    """Recursively coerce a value into an immutable JSON-safe structure."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_evidence_value(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return _freeze_evidence_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_evidence_value(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class EvidenceFact:
    """Per-field provenance/confirmation annotation (R2: annotative only).

    Records *how* a scientific fact was obtained (``origin_type``/``origin_field``/
    ``confirmed_by``/``version``). It is **never** an independent competing copy of
    the scientific facts: the validated :class:`ImportDeclaration` remains the
    canonical scientific-fact representation.
    """

    field: str
    value: object
    origin_type: str
    origin_field: Optional[str] = None
    confirmed_by: Optional[str] = None
    version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise InvalidRequestError("EvidenceFact.field must be a non-empty string")
        if self.origin_type not in EVIDENCE_ORIGIN_TYPES:
            raise InvalidRequestError(
                f"EvidenceFact.origin_type invalid: {self.origin_type!r}"
            )
        if self.origin_field is not None and not isinstance(self.origin_field, str):
            raise InvalidRequestError("EvidenceFact.origin_field must be a string or None")
        if self.confirmed_by is not None and not isinstance(self.confirmed_by, str):
            raise InvalidRequestError("EvidenceFact.confirmed_by must be a string or None")
        if not isinstance(self.version, str) or not self.version.strip():
            raise InvalidRequestError("EvidenceFact.version must be a non-empty string")
        object.__setattr__(self, "field", self.field)
        object.__setattr__(self, "value", _freeze_evidence_value(self.value))
        object.__setattr__(self, "origin_type", self.origin_type)
        object.__setattr__(self, "origin_field", self.origin_field)
        object.__setattr__(self, "confirmed_by", self.confirmed_by)
        object.__setattr__(self, "version", self.version)

    def to_dict(self) -> Mapping[str, object]:
        value = self.value
        if isinstance(value, tuple):
            value = list(value)
        return {
            "field": self.field,
            "value": value,
            "origin_type": self.origin_type,
            "origin_field": self.origin_field,
            "confirmed_by": self.confirmed_by,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "EvidenceFact":
        value = d.get("value")
        if isinstance(value, list):
            value = tuple(value)
        return cls(
            field=d["field"],
            value=value,
            origin_type=d["origin_type"],
            origin_field=d.get("origin_field"),
            confirmed_by=d.get("confirmed_by"),
            version=d.get("version", "1"),
        )


@dataclass(frozen=True)
class ManagedMasterRecord:
    """Immutable managed-master record (P7-M3B, R2).

    ``content_sha256``/``size_bytes`` are the content identity of the master
    FITS bytes; ``role``/``master_type`` the semantic role; ``declaration`` the
    validated canonical :class:`ImportDeclaration` (the single source of
    scientific truth); ``evidence`` the per-field provenance annotations;
    ``dq_state`` the structural DQ state; ``mask_path``/``last_seen_path`` are
    location/provenance only, never scientific identity.
    """

    role: str
    content_sha256: str
    size_bytes: int
    declaration: ImportDeclaration
    hdu: Union[int, str] = 0
    bias_state: Optional[str] = None
    flat_form: Optional[str] = None
    evidence: Mapping[str, EvidenceFact] = field(default_factory=dict)
    dq_state: str = "source_mask"
    mask_path: Optional[str] = None
    last_seen_path: Optional[str] = None
    schema_version: str = MANAGED_LEDGER_SCHEMA
    acquired_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ("bias", "dark", "flat", "flat_dark"):
            raise InvalidRequestError(f"ManagedMasterRecord.role invalid: {self.role!r}")
        if self.acquired_at is not None and (not isinstance(self.acquired_at, str) or not self.acquired_at.strip()):
            raise InvalidRequestError("ManagedMasterRecord.acquired_at must be a non-empty string or None")
        if not _is_hex64(self.content_sha256):
            raise InvalidRequestError(
                "ManagedMasterRecord.content_sha256 must be a 64-char lowercase hex string"
            )
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise InvalidRequestError("ManagedMasterRecord.size_bytes must be a non-negative int")
        if not isinstance(self.declaration, ImportDeclaration):
            raise InvalidRequestError("ManagedMasterRecord.declaration must be an ImportDeclaration")
        if isinstance(self.hdu, bool) or not isinstance(self.hdu, (int, str)):
            raise InvalidRequestError("ManagedMasterRecord.hdu must be int or str")
        if isinstance(self.hdu, int) and self.hdu < 0:
            raise InvalidRequestError("ManagedMasterRecord.hdu must be a non-negative int or str")
        if self.dq_state not in DQ_STATES:
            raise InvalidRequestError(f"ManagedMasterRecord.dq_state invalid: {self.dq_state!r}")
        if self.dq_state == "no_source_dq":
            if self.mask_path is not None:
                raise InvalidRequestError(
                    "ManagedMasterRecord with dq_state='no_source_dq' must have mask_path=None"
                )
        else:
            if self.mask_path is None or not isinstance(self.mask_path, str) or not self.mask_path.strip():
                raise InvalidRequestError(
                    "ManagedMasterRecord with dq_state='source_mask' requires a non-empty mask_path"
                )
        if self.schema_version != MANAGED_LEDGER_SCHEMA:
            raise InvalidRequestError(
                f"ManagedMasterRecord.schema_version unsupported: {self.schema_version!r}"
            )
        for fld, fact in self.evidence.items():
            if not isinstance(fld, str) or not fld:
                raise InvalidRequestError("ManagedMasterRecord.evidence keys must be non-empty strings")
            if not isinstance(fact, EvidenceFact):
                raise InvalidRequestError("ManagedMasterRecord.evidence values must be EvidenceFact")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(self, "role", self.role)
        object.__setattr__(self, "content_sha256", self.content_sha256)
        object.__setattr__(self, "bias_state", self.bias_state)
        object.__setattr__(self, "flat_form", self.flat_form)
        object.__setattr__(self, "dq_state", self.dq_state)
        object.__setattr__(self, "mask_path", self.mask_path)
        object.__setattr__(self, "last_seen_path", self.last_seen_path)

    def to_dict(self) -> Mapping[str, object]:
        decl = self.declaration
        decl_dict = {f.name: (list(getattr(decl, f.name)) if isinstance(getattr(decl, f.name), tuple) else getattr(decl, f.name)) for f in dataclasses.fields(ImportDeclaration)}
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "hdu": self.hdu,
            "bias_state": self.bias_state,
            "flat_form": self.flat_form,
            "declaration": decl_dict,
            "evidence": {fld: dict(fact.to_dict()) for fld, fact in self.evidence.items()},
            "dq_state": self.dq_state,
            "mask_path": self.mask_path,
            "last_seen_path": self.last_seen_path,
            "acquired_at": self.acquired_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "ManagedMasterRecord":
        decl = ImportDeclaration(**dict(d["declaration"]))
        evidence = {
            fld: EvidenceFact.from_dict(fact) for fld, fact in d.get("evidence", {}).items()
        }
        return cls(
            role=d["role"],
            content_sha256=d["content_sha256"],
            size_bytes=d["size_bytes"],
            declaration=decl,
            hdu=d.get("hdu", 0),
            bias_state=d.get("bias_state"),
            flat_form=d.get("flat_form"),
            evidence=evidence,
            dq_state=d.get("dq_state", "source_mask"),
            mask_path=d.get("mask_path"),
            last_seen_path=d.get("last_seen_path"),
            schema_version=d.get("schema_version", MANAGED_LEDGER_SCHEMA),
            acquired_at=d.get("acquired_at"),
        )


__all__ = [
    "EVIDENCE_ORIGIN_TYPES",
    "EvidenceFact",
    "MANAGED_LEDGER_SCHEMA",
    "ManagedMasterRecord",
]
