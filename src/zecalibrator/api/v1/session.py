"""Public session-library capability (``session_library`` + ``auto_route``).

``open_session_library`` performs folder admission + role identification (top-
level, FITS-only, non-recursive) and returns a :class:`SessionLibrary` handle
that auto-routes each light and calibrates in-memory while **reusing** the
prepared masters across lights (never reloading them per light).

This module is the public entry point for the ZSSS « folder -> library -> route
by light -> in-memory calibration » flow. It imports only ``zecalibrator.api.v1``
value objects, the application-layer admission logic
(:mod:`zecalibrator.application.masters`), the existing private auto-route
resolver (``._auto_route``) and the existing private calibration execution seam
(``.calibration``) — never Qt/ZeAlfie/ZSSS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import MappingProxyType
from pathlib import Path
from typing import Mapping, Optional, Union

from zecalibrator.application.cancellation import CancellationToken, OperationCancelled
from zecalibrator.application.library import (
    LIBRARY_SCHEMA,
    LibrarySnapshot,
    light_constraints_from_sensor_metadata,
)
from zecalibrator.io.master_source import FilesystemSource

from . import _io
from zecalibrator.application.masters import (
    build_declaration,
    detect_header_candidates,
    detect_imagetyp_role,
    make_managed_record,
    master_evidence_status,
    master_incompatibility,
)
from ._auto_route import _AUTO_ROUTE_TO_MATCH, resolve_route
from .batch import _build_candidate
from .calibration import _PreparedContextSlot, _calibrate_frame_impl
from .errors import InvalidRequestError, LibraryClosedError, PlanSourceMismatchError
from .managed import build_master_import_spec, managed_fingerprint
from .models import (
    ArrayFrameSource,
    CalibrationPlan,
    Candidate,
    EvidenceFact,
    ExecutionOptions,
    FitsFrameSource,
    FrameSource,
    ImportDeclaration,
    ManagedMasterRecord,
    MatchPolicy,
    default_match_policy,
)

OPERATION_ID = "zecalibrator-open-session-library"

# The ImportDeclaration source/identity/version used for masters admitted by the
# session-library path (analogous to STANDARD_LIGHT_CONTRACT_SOURCE in ``_io``).
# Header-derived facts are the evidence; the declaration supplies the canonical
# raw/ADU domain and a stable source identity (never invented acquisition facts).
SESSION_LIBRARY_CONTRACT_SOURCE = "session_library_contract"

_FITS_SUFFIXES = (".fits", ".fit", ".fts")


@dataclass(frozen=True)
class MasterAdmission:
    """One admitted master: identified role + content identity + retained evidence.

    ``evidence`` holds the header-derived :class:`EvidenceFact` values that were
    retained for this master (origin ``fits_header``); the :class:`ImportDeclaration`
    is the canonical scientific-fact representation built from them.
    """

    role: str  # bias | dark | flat | flat_dark
    path: str
    content_sha256: str
    size_bytes: int
    hdu: Union[int, str]
    declaration: ImportDeclaration
    evidence: Mapping[str, EvidenceFact]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True)
class RejectionDiagnostic:
    """A master that was refused with a structured reason (never an exception)."""

    path: str
    reason_code: str
    detail: str = ""


@dataclass(frozen=True)
class SessionLibraryResult:
    """Structured ``open_session_library`` outcome envelope.

    ``handle`` is ``None`` exactly when zero masters were admissible (C3a), which
    is a completed, informative result — never a raised error. ``rejected`` names
    every refused master with a structured reason; the rest of the library is
    never invalidated by a refusal.

    ``context_preparations`` exposes the §18 audit counter (see also
    :attr:`SessionLibrary.context_preparation_count`): the number of master-context
    preparations performed so far. It grows by exactly one per *distinct*
    calibrated plan, so a consumer can verify that the masters are **not**
    reloaded per light (reusing one plan across N lights keeps it at 1).
    """

    operation_status: str  # "COMPLETED" | "CANCELLED"
    handle: Optional["SessionLibrary"] = None
    fingerprint: str = ""
    admissions: tuple = ()
    rejected: tuple = ()
    counts_by_role: Mapping[str, int] = MappingProxyType({})
    warnings: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "admissions", tuple(self.admissions))
        object.__setattr__(self, "rejected", tuple(self.rejected))
        object.__setattr__(self, "counts_by_role", MappingProxyType(dict(self.counts_by_role)))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def context_preparations(self) -> int:
        """Number of master-context preparations so far (S-a / §18 audit).

        Delegates to the live counter on the handle (``0`` when there is no
        handle). One preparation per distinct calibrated plan; reusing the same
        plan across lights never re-prepares.
        """
        return self.handle.context_preparation_count if self.handle is not None else 0


@dataclass(frozen=True)
class RouteResolution:
    """Public auto-route decision for one light (``SessionLibrary.resolve_light``).

    ``outcome`` is ``MATCHED``/``NO_MATCH``/``AMBIGUOUS`` (mapped from the
    resolver's READY/NEEDS_ATTENTION/AMBIGUOUS) and is only meaningful when
    ``operation_status == "COMPLETED"``. ``plan`` is the ZeCalibrator-chosen
    :class:`CalibrationPlan` (``None`` unless MATCHED). ``reasons`` /
    ``unverified`` carry the structured decision audit.
    """

    operation_status: str  # "COMPLETED" | "CANCELLED" | "FAILED"
    outcome: Optional[str] = None
    plan: Optional[CalibrationPlan] = None
    reasons: tuple = ()
    unverified: tuple = ()
    details: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "unverified", tuple(self.unverified))


class _CountingContextSlot(_PreparedContextSlot):
    """A :class:`_PreparedContextSlot` that also counts context preparations (S-a).

    Each ``put`` is exactly one master-context preparation (the executor only
    ``put``s when it prepares a *new* context, never on reuse), so the counter is
    the audit trace for the §18 no-per-light-reload invariant.
    """

    __slots__ = ("preparation_count",)

    def __init__(self) -> None:
        super().__init__()
        self.preparation_count = 0

    def put(self, plan: CalibrationPlan, context) -> None:
        super().put(plan, context)
        self.preparation_count += 1


def _scan_top_level_fits(root: str) -> list:
    """Scan ``root`` for top-level FITS files only (deterministic, no recursion)."""
    p = Path(root)
    if not p.is_dir():
        raise InvalidRequestError(f"root is not a directory: {root!r}")
    out: list = []
    for entry in sorted(p.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _FITS_SUFFIXES:
            out.append(str(entry))
    return out


class SessionLibrary:
    """An open calibration session over an admitted master library.

    **Threading contract (D5/S3):** this object is **mono-thread** — one instance
    per worker, never shared, never re-entrant. Concurrent ``resolve_light``/
    ``calibrate`` calls on the same instance are not supported.

    ``resolve_light`` auto-routes a light to a :class:`CalibrationPlan` (the
    scientific route is chosen by ZeCalibrator). ``calibrate`` runs in-memory
    calibration using that plan and **reuses** the prepared masters between
    lights (no per-light master reload — see :attr:`context_preparation_count`).
    A ``plan`` passed to ``calibrate`` must originate from ``resolve_light`` of
    this same session (C1): a foreign plan raises :class:`PlanSourceMismatchError`.

    **Fingerprint scope (RW-3):** :attr:`fingerprint` is computed with
    :func:`zecalibrator.api.v1.managed_fingerprint` over records built by the
    session admission with **fixed** structural values (``dq_state="no_source_dq"``,
    ``bias_state=None``, ``flat_form=None``). A *curated* managed index that
    carries richer states may therefore produce a **different** fingerprint for
    the same folder. The fingerprint is reliable for freeze/resume **as long as a
    single ingestion path is used** — ZSSS uses only :func:`open_session_library`.
    """

    def __init__(self, *, fingerprint: str, snapshot: LibrarySnapshot,
                 policy: MatchPolicy, options: ExecutionOptions) -> None:
        self._fingerprint = fingerprint
        self._snapshot = snapshot
        self._policy = policy
        self._options = options
        self._slot = _CountingContextSlot()
        self._issued_plan_ids: set = set()
        self._closed = False

    @property
    def fingerprint(self) -> str:
        """The derived library fingerprint (roles + content + evidence + schema)."""
        return self._fingerprint

    @property
    def context_preparation_count(self) -> int:
        """Number of master-context preparations performed by ``calibrate`` (S-a).

        Equal to the number of *distinct* plans calibrated (one preparation per
        plan); reusing the same plan across lights does not re-prepare.
        """
        return self._slot.preparation_count

    def resolve_light(self, source: FrameSource, *, cancel=None) -> RouteResolution:
        """Auto-route one light against this session's library.

        ``source`` is a :class:`FitsFrameSource` or :class:`ArrayFrameSource`. The
        scientific route is decided by ZeCalibrator (never by the caller): returns
        a :class:`RouteResolution` with outcome MATCHED/NO_MATCH/AMBIGUOUS and, on
        MATCHED, the chosen :class:`CalibrationPlan`.
        """
        self._ensure_open()
        if not isinstance(source, (FitsFrameSource, ArrayFrameSource)):
            raise InvalidRequestError("source must be a FitsFrameSource or ArrayFrameSource")
        token = cancel or CancellationToken()
        if token.is_cancelled():
            return RouteResolution(operation_status="CANCELLED")

        from .frames import inspect_frame

        ins = inspect_frame(source, cancel=token)
        if ins.operation_status == "CANCELLED":
            return RouteResolution(operation_status="CANCELLED")
        if ins.operation_status == "FAILED":
            return RouteResolution(
                operation_status="FAILED",
                details=f"{ins.reason_code}: {ins.details}" if ins.reason_code else ins.details,
            )
        inspection = ins.inspection
        if inspection.domain_finding != "raw":
            return RouteResolution(
                operation_status="FAILED",
                details=f"domain_finding={inspection.domain_finding} (raw required)",
            )

        try:
            light = light_constraints_from_sensor_metadata(inspection.metadata)
        except ValueError as exc:  # MetadataAdapterError (unresolved conflicts/units)
            return RouteResolution(operation_status="FAILED", details=str(exc))

        token.raise_if_cancelled()
        resolution = resolve_route(light, self._snapshot, self._policy)
        outcome = _AUTO_ROUTE_TO_MATCH[resolution.outcome]
        if resolution.plan is not None:
            self._issued_plan_ids.add(resolution.plan.plan_id)
        return RouteResolution(
            operation_status="COMPLETED",
            outcome=outcome,
            plan=resolution.plan,
            reasons=resolution.reasons,
            unverified=resolution.unverified,
        )

    def calibrate(self, source: FrameSource, plan: CalibrationPlan, *, cancel=None) -> "CalibrationResult":
        """Calibrate one light in-memory using a plan from this session (C1).

        ``plan`` must come from ``resolve_light`` of **this same** session; a
        foreign plan raises :class:`PlanSourceMismatchError` (never a silent
        calibration). Masters are reused between lights of the same session
        (§18); the light is decoded once per call.
        """
        self._ensure_open()
        if not isinstance(plan, CalibrationPlan):
            raise InvalidRequestError("plan must be a CalibrationPlan")
        if plan.plan_id not in self._issued_plan_ids:
            raise PlanSourceMismatchError(
                "plan does not originate from this session's resolve_light"
            )
        if not isinstance(source, (FitsFrameSource, ArrayFrameSource)):
            raise InvalidRequestError("source must be a FitsFrameSource or ArrayFrameSource")
        token = cancel or CancellationToken()
        obs = _io.normalize_progress(None, OPERATION_ID)
        return _calibrate_frame_impl(
            source, plan, self._options, token=token, obs=obs, slot=self._slot,
        )

    def close(self) -> None:
        """Close the session. Idempotent (C3c): repeated calls are no-ops."""
        self._closed = True

    def __enter__(self) -> "SessionLibrary":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None

    def _ensure_open(self) -> None:
        if self._closed:
            raise LibraryClosedError("session library is closed")


def open_session_library(
    root,
    *,
    policy: Optional[MatchPolicy] = None,
    options: Optional[ExecutionOptions] = None,
    cancel=None,
    progress=None,
) -> SessionLibraryResult:
    """Admit a top-level folder of masters into a usable :class:`SessionLibrary`.

    Scans ``root`` top-level only (FITS suffixes only, no recursion). Each FITS
    master is admitted only when its role (``IMAGETYP``) is unambiguous, it is
    structurally admissible, its header evidence is non-conflicting, and every
    matching-blocking fact is present. A refused master yields a structured
    :class:`RejectionDiagnostic` and never invalidates the rest (C3a: zero
    admissible masters returns ``handle=None`` — not an exception).

    ``policy``/``options`` default to :func:`default_match_policy` /
    :class:`ExecutionOptions`. Cancellation is honoured at every stage and returns
    a ``CANCELLED`` result (C3b); a partially-built session is never promoted.

    ``fingerprint`` is derived from :func:`zecalibrator.api.v1.managed_fingerprint`
    over records admitted with **fixed** structural values (``dq_state="no_source_dq"``,
    ``bias_state=None``, ``flat_form=None``); see :class:`SessionLibrary` for the
    consequence (single-ingestion-path assumption for freeze/resume, RW-3).
    """
    if not isinstance(root, (str, os.PathLike)) or str(root) == "":
        raise InvalidRequestError("root must be a non-empty path")
    if policy is not None and not isinstance(policy, MatchPolicy):
        raise InvalidRequestError("policy must be a MatchPolicy or None")
    if options is not None and not isinstance(options, ExecutionOptions):
        raise InvalidRequestError("options must be an ExecutionOptions or None")

    token = cancel or CancellationToken()
    obs = _io.normalize_progress(progress, OPERATION_ID)
    decided_policy = policy if policy is not None else default_match_policy()
    decided_options = options if options is not None else ExecutionOptions()

    if token.is_cancelled():
        return SessionLibraryResult(
            operation_status="CANCELLED", warnings=("cancelled",),
        )

    _io.emit_progress(obs, OPERATION_ID, "start", 0, 1)
    files = _scan_top_level_fits(str(root))
    source = FilesystemSource()

    admissions: list = []
    rejected: list = []
    warnings: list = []
    records: list = []

    for path in files:
        token.raise_if_cancelled()
        try:
            cards = source.read_header(path, hdu=0)
        except OperationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            rejected.append(RejectionDiagnostic(path=path, reason_code="HEADER_READ_ERROR", detail=str(exc)))
            continue
        card_pairs = [(c.keyword, c.value) for c in cards]

        role = detect_imagetyp_role(card_pairs)
        if role is None:
            rejected.append(RejectionDiagnostic(
                path=path, reason_code="AMBIGUOUS_ROLE",
                detail="no (or multiple distinct) IMAGETYP role; filename/folder layout never produce a role",
            ))
            continue

        incompat = master_incompatibility(card_pairs, role=role)
        if incompat:
            rejected.append(RejectionDiagnostic(
                path=path, reason_code="INCOMPATIBLE", detail="; ".join(incompat),
            ))
            continue

        candidates, conflicts = detect_header_candidates(card_pairs)
        if conflicts:
            rejected.append(RejectionDiagnostic(
                path=path, reason_code="CONFLICTING_HEADER",
                detail="; ".join(f"{f} has conflicting values" for f in sorted(conflicts)),
            ))
            continue

        identity = f"session-library-{role}"
        declaration = build_declaration(
            SESSION_LIBRARY_CONTRACT_SOURCE, identity, "1", candidates,
        )
        status, reasons = master_evidence_status(role, declaration)
        if status != "ready":
            code = "MISSING_REQUIRED_FIELDS"
            if role == "flat":
                code = "FLAT_QUALITY_EVIDENCE_INSUFFICIENT"
            rejected.append(RejectionDiagnostic(
                path=path, reason_code=code, detail="; ".join(reasons),
            ))
            continue

        try:
            from zecalibrator.core.plans import FitsFileLocator

            ident = source.image_identity(FitsFileLocator(path=path, hdu=0))
        except OperationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            rejected.append(RejectionDiagnostic(
                path=path, reason_code="CONTENT_IDENTITY_ERROR", detail=str(exc),
            ))
            continue

        record = make_managed_record(
            role=role,
            content_sha256=ident.content_sha256,
            size_bytes=ident.size_bytes,
            declaration=declaration,
            hdu=0,
            bias_state=None,
            flat_form=None,
            evidence=candidates,
            dq_state="no_source_dq",
            mask_path=None,
            last_seen_path=path,
        )
        records.append(record)
        admissions.append(MasterAdmission(
            role=role, path=path, content_sha256=ident.content_sha256,
            size_bytes=ident.size_bytes, hdu=0, declaration=declaration,
            evidence=candidates,
        ))

    fingerprint = managed_fingerprint(records)
    counts: dict = {}
    for r in records:
        counts[r.role] = counts.get(r.role, 0) + 1

    if not records:
        warnings.append("no admissible masters; handle is None (C3a)")
        handle = None
    else:
        candidates_by_role: dict = {}
        for record in records:
            spec = build_master_import_spec(record)
            cand = _build_candidate(record.last_seen_path, spec, source, None)
            candidates_by_role.setdefault(cand.role, []).append(cand)
        snapshot = LibrarySnapshot(
            revision=fingerprint,
            schema_version=LIBRARY_SCHEMA,
            candidates=candidates_by_role,
        )
        handle = SessionLibrary(
            fingerprint=fingerprint,
            snapshot=snapshot,
            policy=decided_policy,
            options=decided_options,
        )

    _io.emit_progress(obs, OPERATION_ID, "complete", 1, 1)
    return SessionLibraryResult(
        operation_status="COMPLETED",
        handle=handle,
        fingerprint=fingerprint,
        admissions=tuple(admissions),
        rejected=tuple(rejected),
        counts_by_role=counts,
        warnings=tuple(warnings),
    )


__all__ = [
    "MasterAdmission",
    "RejectionDiagnostic",
    "RouteResolution",
    "SessionLibrary",
    "SessionLibraryResult",
    "open_session_library",
]
