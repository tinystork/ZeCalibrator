"""Qt worker for the ZeCalibrator GUI.

One active operation at a time: a single :class:`_OperationWorker` (a ``QObject``)
lives on a dedicated ``QThread`` that runs its own event loop for the lifetime of
the application. Each operation is dispatched as an immutable
:class:`~zecalibrator.gui.service.OperationSnapshot` + a shared
:class:`~zecalibrator.api.v1.CancellationToken` + a bounded progress mailbox
through a queued signal, so the request crosses the thread boundary by reference
and never races the previous operation (the controller's ``_active`` guard
prevents overlap).

Threading/lifetime invariants (ARCHITECTURE §3.7, ASTRA §11, prepared §5-§8):

* All scientific/expensive work and I/O (library open, FITS inspection, matching,
  calibration, indexing, output writing, JSON file reading/parsing, large audit
  serialization, settings read/fsync/write, path/stat checks) run inside
  ``_run`` on the worker thread.
* The worker owns open/use/close of ``LibraryHandle`` (closed in ``finally``);
  no GUI-thread DB calls and no shared mutable global active library.
* Progress uses a single-slot latest-value mailbox, drained by a main-thread
  timer: the producer queue is strictly bounded to one event (latest wins),
  even if the GUI stalls. Only engine-sourced ``ProgressEvent`` values are
  forwarded, never invented percentages. Per-item summaries are lightweight
  public dicts (no scientific arrays); their count is bounded by the input list.
* ``CancellationToken.cancel`` is called directly by the GUI (thread-safe) and is
  never queued behind blocking work. No ``QThread.terminate``, no blocking GUI
  wait: shutdown is finished-driven (``quit()`` + ``deleteLater`` on
  ``thread.finished``). The terminal result is authoritative: the mailbox is
  drained on **every** terminal path so a late progress event can never overwrite
  a success/failure/cancel summary.
* Terminal delivery carries the GUI operation id on every event; stale events
  from a prior operation are rejected by the receiving window (op-id guard).
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from pathlib import Path

from PySide6 import QtCore

import zecalibrator.api.v1 as v1
from zecalibrator.gui import service, settings as settings_mod

# ---------------------------------------------------------------------------
# Interpreter-teardown safety net (programmatic/embedding callers).
#
# A GUI library must not abort the interpreter when a caller creates a window and
# exits WITHOUT running ``app.exec()``. We keep a strong reference to every live
# worker QThread and, at interpreter exit, request a non-blocking quit and do a
# *bounded* join. This is never invoked from a GUI slot/signal/aboutToQuit (those
# paths use the non-blocking ``shutdown()``); it only runs at process teardown.
# ---------------------------------------------------------------------------
_LIVE_THREADS: set = set()
_ATEXIT_REGISTERED = False
_ATEXIT_LOGGED = False


def _register_thread(thread: QtCore.QThread) -> None:
    global _ATEXIT_REGISTERED
    _LIVE_THREADS.add(thread)
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_join_workers)
        _ATEXIT_REGISTERED = True


def _unregister_thread(thread: QtCore.QThread) -> None:
    _LIVE_THREADS.discard(thread)


def _atexit_join_workers() -> None:
    """Bounded, deadlock-free join of any still-live worker threads at exit.

    Requests a non-blocking ``quit()`` and waits up to a bounded timeout per
    thread; a wedged thread is not waited on indefinitely (we proceed and log
    once). No ``QThread.terminate`` is ever used.
    """
    global _ATEXIT_LOGGED
    for thread in list(_LIVE_THREADS):
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(2000)
        except Exception:  # noqa: BLE001 - teardown best effort
            pass
        finally:
            _LIVE_THREADS.discard(thread)
        if thread.isRunning() and not _ATEXIT_LOGGED:
            print(
                "zecalibrator: worker thread did not stop before interpreter exit",
                file=sys.stderr,
            )
            _ATEXIT_LOGGED = True


class ProgressMailbox:
    """A bounded, single-slot latest-progress mailbox (thread-safe).

    ``post`` overwrites the latest event (latest wins); ``take`` atomically
    drains it. This bounds the producer queue to exactly one in-flight event
    regardless of how slowly the consumer drains it (no unbounded Qt queue).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None

    def post(self, event) -> None:
        with self._lock:
            self._latest = event

    def take(self):
        with self._lock:
            event = self._latest
            self._latest = None
            return event


def _tuple_or_none(value):
    if value is None:
        return None
    return [value[0], value[1]]


def _metadata_dict(md) -> dict:
    """Render a public ``SensorMetadata`` into a JSON-safe display summary."""
    geo = md.geometry
    return {
        "geometry": {
            "shape": list(geo.shape),
            "sensor_dimensions": _tuple_or_none(geo.sensor_dimensions),
            "binning": _tuple_or_none(geo.binning),
            "roi_origin": _tuple_or_none(geo.roi_origin),
            "roi_extent": _tuple_or_none(geo.roi_extent),
            "orientation": geo.orientation,
            "cfa_phase": geo.cfa_phase,
        },
        "raw_domain_declaration": md.raw_domain_declaration,
        "units": md.units,
        "exposure_s": md.exposure_s,
        "temperature_c": md.temperature_c,
        "gain": md.gain,
        "offset": md.offset,
        "readout_mode": md.readout_mode,
        "adc_mode": md.adc_mode,
        "filter": md.filter,
        "detector_model": md.detector_model,
        "detector_instance_id": md.detector_instance_id,
        "optical_train_id": md.optical_train_id,
        "saturation_limit_adu": md.saturation_limit_adu,
        "saturation_evidence": md.saturation_evidence,
        "warnings": list(md.warnings),
        "conflicts": [dict(c.__dict__) for c in md.conflicts],
    }


def _reason_dict(reason) -> dict:
    return reason.to_dict()


def _rejection_dict(rec) -> dict:
    d = rec.to_dict()
    return {
        "candidate_id": d["candidate_id"],
        "role": d["role"],
        "reasons": d.get("reasons", ()),
    }


def _resolve_summary(index: int, light: "service.LightInput", inspection, decision):
    outcome = decision.outcome
    plan = decision.plan
    summary = {
        "index": index,
        "row_id": light.row_id,
        "path": light.path,
        "display": light.display_name,
        "hdu": light.hdu,
        "inspect_status": "COMPLETED",
        "domain_finding": inspection.domain_finding,
        "shape": list(inspection.shape),
        "warnings": list(inspection.warnings),
        "metadata": _metadata_dict(inspection.metadata),
        "outcome": outcome,
        "plan_id": plan.plan_id if plan is not None else None,
        "reason_codes": list(decision.reason_codes),
        "reasons": [_reason_dict(r) for r in decision.reasons],
        "rejected_candidates": [_rejection_dict(r) for r in decision.rejected_candidates],
        "coherent_sets": [
            {role: c.candidate_id for role, c in s.items()}
            for s in decision.coherent_sets
        ],
        "selection_decisions": [list(kv) for kv in decision.selection_decisions],
        # Full public decision/plan audit (worker-side serialization; no science
        # reassembly). Exposes nested master identities, policy, input identity,
        # reasons/role/parent and the selected plan.
        "decision_audit": service.to_jsonable(decision.to_dict()),
        "plan_audit": service.to_jsonable(plan.to_dict()) if plan is not None else None,
    }
    return summary, plan


def _inspect_failed_summary(index: int, light: "service.LightInput", insp) -> dict:
    return {
        "index": index,
        "row_id": light.row_id,
        "path": light.path,
        "display": light.display_name,
        "hdu": light.hdu,
        "inspect_status": insp.operation_status,
        "reason_code": insp.reason_code,
        "details": insp.details,
        "outcome": None,
        "plan_id": None,
        "reason_codes": [],
        "reasons": [],
        "rejected_candidates": [],
        "coherent_sets": [],
        "selection_decisions": [],
        "decision_audit": None,
        "plan_audit": None,
    }


def _resolve_failed_summary(index: int, light: "service.LightInput", inspection, res) -> dict:
    return {
        "index": index,
        "row_id": light.row_id,
        "path": light.path,
        "display": light.display_name,
        "hdu": light.hdu,
        "inspect_status": "COMPLETED",
        "domain_finding": inspection.domain_finding,
        "shape": list(inspection.shape),
        "warnings": list(inspection.warnings),
        "metadata": _metadata_dict(inspection.metadata),
        "outcome": None,
        "plan_id": None,
        "reason_code": res.reason_code,
        "details": res.details,
        "reason_codes": [],
        "reasons": [],
        "rejected_candidates": [],
        "coherent_sets": [],
        "selection_decisions": [],
        "decision_audit": None,
        "plan_audit": None,
    }


class _OperationWorker(QtCore.QObject):
    """The single worker QObject (lives on the worker thread)."""

    start_op = QtCore.Signal(object, object, object)  # (snapshot, token, mailbox)
    started = QtCore.Signal(str)
    preflight_light = QtCore.Signal(str, object, object)  # op_id, summary, plan
    batch_item = QtCore.Signal(str, object)  # op_id, item dict
    finished = QtCore.Signal(str, object)  # op_id, summary
    failed = QtCore.Signal(str, str, str)  # op_id, reason_code, details
    ended = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_op.connect(self._run)

    @QtCore.Slot(object, object, object)
    def _run(self, snapshot, token, mailbox):
        op_id = snapshot.op_id
        self.started.emit(op_id)

        def on_progress(event):
            mailbox.post(event)

        try:
            summary = self._execute(snapshot, token, on_progress)
        except v1.OperationCancelled:
            summary = {
                "kind": snapshot.kind, "status": "CANCELLED",
                "details": "cancellation requested",
            }
        except v1.InvalidRequestError as exc:
            self.failed.emit(op_id, "INVALID_REQUEST", str(exc))
            self.ended.emit()
            return
        except Exception as exc:  # noqa: BLE001 - terminal worker boundary
            self.failed.emit(op_id, "UNEXPECTED", f"{type(exc).__name__}: {exc}")
            self.ended.emit()
            return
        # Terminal authority (R1) is enforced window-side via the terminal-seen
        # guard: _on_operation_finished/_on_operation_failed set it before the
        # controller's final drain runs, so a late "complete" can never overwrite
        # the truthful terminal summary. Engine-sourced progress is preserved.
        self.finished.emit(op_id, summary)
        self.ended.emit()

    # -- operations ----------------------------------------------------------
    def _execute(self, snap, token, progress) -> dict:
        if snap.kind == "open_library":
            return self._open_library(snap, token, progress)
        if snap.kind == "index_library":
            return self._index_library(snap, token, progress)
        if snap.kind == "preflight":
            return self._preflight(snap, token, progress)
        if snap.kind == "calibrate_in_memory":
            return self._calibrate_in_memory(snap, token, progress)
        if snap.kind == "export":
            return self._export(snap, token, progress)
        if snap.kind == "load_declaration":
            return self._load_declaration(snap)
        if snap.kind == "load_roi":
            return self._load_roi(snap)
        if snap.kind == "load_settings":
            return self._load_settings(snap)
        if snap.kind == "save_settings":
            return self._save_settings(snap)
        if snap.kind == "scan_masters":
            return self._scan_masters(snap)
        if snap.kind == "load_ledger":
            return self._load_ledger(snap)
        if snap.kind == "confirm_evidence":
            return self._confirm_evidence(snap)
        if snap.kind == "build_managed_library":
            return self._build_managed_library(snap)
        raise RuntimeError(f"unknown operation kind {snap.kind!r}")

    def _open_library(self, snap, token, progress) -> dict:
        result = v1.open_library(snap.library_spec, cancel=token, progress=progress)
        handle = result.handle
        if handle is None:
            return {
                "kind": "open_library",
                "status": result.operation_status,
                "reason_code": result.reason_code,
                "details": result.details,
            }
        try:
            snap_shot = handle.snapshot
            roles = snap_shot.roles()
            counts = {role: len(snap_shot.candidates[role]) for role in roles}
            return {
                "kind": "open_library",
                "status": "OPENED",
                "revision": snap_shot.revision,
                "roles": list(roles),
                "candidate_counts": counts,
            }
        finally:
            handle.close()

    def _index_library(self, snap, token, progress) -> dict:
        imports = snap.index_imports
        if snap.imports_path is not None:
            imports = service.parse_imports(service.load_json_array(snap.imports_path))
        result = v1.index_library(
            snap.library_spec, imports,
            revision=snap.index_revision, cancel=token, progress=progress,
        )
        return {
            "kind": "index_library",
            "status": result.operation_status,
            "revision": result.revision,
            "candidate_count": result.candidate_count,
            "diagnostics": [
                d.to_dict() if hasattr(d, "to_dict") else str(d)
                for d in result.diagnostics
            ],
            "reason_code": result.reason_code,
            "details": result.details,
        }

    def _preflight(self, snap, token, progress) -> dict:
        opened = v1.open_library(snap.library_spec, cancel=token, progress=progress)
        if opened.operation_status == "CANCELLED":
            return {"kind": "preflight", "status": "CANCELLED", "count": 0}
        if opened.operation_status != "OPENED":
            return {
                "kind": "preflight", "status": "FAILED",
                "reason_code": opened.reason_code, "details": opened.details,
                "count": 0,
            }
        handle = opened.handle
        try:
            for index, light in enumerate(snap.lights):
                token.raise_if_cancelled()
                source = light.to_source()
                insp = v1.inspect_frame(source, cancel=token, progress=progress)
                if insp.operation_status == "CANCELLED":
                    return {"kind": "preflight", "status": "CANCELLED", "count": index}
                if insp.operation_status == "FAILED":
                    self.preflight_light.emit(
                        snap.op_id, _inspect_failed_summary(index, light, insp), None
                    )
                    continue
                inspection = insp.inspection
                res = v1.resolve_calibration(
                    inspection, snap.request, handle, snap.policy,
                    cancel=token, progress=progress,
                )
                if res.operation_status == "CANCELLED":
                    return {"kind": "preflight", "status": "CANCELLED", "count": index}
                if res.operation_status == "FAILED":
                    self.preflight_light.emit(
                        snap.op_id, _resolve_failed_summary(index, light, inspection, res), None
                    )
                    continue
                summary, plan = _resolve_summary(index, light, inspection, res.decision)
                self.preflight_light.emit(snap.op_id, summary, plan)
            return {"kind": "preflight", "status": "COMPLETED", "count": len(snap.lights)}
        finally:
            handle.close()

    def _calibrate_in_memory(self, snap, token, progress) -> dict:
        light = snap.lights[0]
        source = light.to_source()
        result = v1.calibrate_frame(
            source, snap.plan, v1.ExecutionOptions(), cancel=token, progress=progress
        )
        if result.status == "CANCELLED":
            return {
                "kind": "calibrate_in_memory", "status": "CANCELLED",
                "display": light.display_name,
            }
        provenance = dict(result.provenance.to_dict())
        return {
            "kind": "calibrate_in_memory",
            "status": result.status,
            "display": light.display_name,
            "plan_id": result.provenance.plan.plan_id,
            "reason_code": result.reason_code,
            "warnings": list(result.warnings),
            "scalars": dict(result.scalars),
            "counts": result.to_dict().get("counts"),
            "saturation_evidence": result.frame_quality.saturation_evidence,
            "provenance": provenance,
            # Full execution audit (nested master identities/policy/input scaling).
            "provenance_audit": provenance,
        }

    def _export(self, snap, token, progress) -> dict:
        frames = [light.to_source() for light in snap.lights]
        options = v1.BatchOptions(destination=snap.destination, batch_id=snap.batch_id)
        opened = v1.open_library(snap.library_spec, cancel=token, progress=progress)
        if opened.operation_status == "CANCELLED":
            return {"kind": "export", "status": "CANCELLED", "batch_id": snap.batch_id,
                    "items": [], "total_inputs": len(snap.lights)}
        if opened.operation_status != "OPENED":
            return {
                "kind": "export", "status": "FAILED",
                "reason_code": opened.reason_code, "details": opened.details,
                "batch_id": snap.batch_id, "items": [], "total_inputs": len(snap.lights),
            }
        handle = opened.handle
        items = []
        cancelled = False
        manifest_error = None
        try:
            for item in v1.calibrate_batch(
                frames, snap.request, handle, snap.policy, options,
                cancel=token, progress=progress,
            ):
                d = item.to_dict()
                items.append(d)
                self.batch_item.emit(snap.op_id, d)
        except v1.OperationCancelled:
            cancelled = True
        except v1.BatchManifestError as exc:
            # Items were already yielded and delivered; retain them and report the
            # manifest finalization failure truthfully (never discard the items).
            manifest_error = str(exc)
        finally:
            handle.close()

        manifest = None
        manifest_exists = False
        manifest_audit = None
        if snap.destination is not None:
            manifest = os.path.join(snap.destination, f"zecalibrator_batch_{snap.batch_id}.json")
            manifest_exists = os.path.exists(manifest)
            if manifest_exists:
                try:
                    with open(manifest, "rb") as f:
                        manifest_audit = json.loads(f.read().decode("utf-8"))
                except Exception:  # noqa: BLE001 - audit is best-effort
                    manifest_audit = None

        if manifest_error is not None:
            status = "FAILED"
            reason_code = "MANIFEST_WRITE_FAILED"
            details = manifest_error
        elif cancelled:
            status = "CANCELLED"
            reason_code = "CANCELLED"
            details = "cancellation requested"
        elif any(i["disposition"] == "FAILED" for i in items):
            status = "PARTIAL"
            reason_code = None
            details = ""
        else:
            status = "COMPLETED"
            reason_code = None
            details = ""

        return {
            "kind": "export",
            "status": status,
            "reason_code": reason_code,
            "details": details,
            "batch_id": snap.batch_id,
            "destination": snap.destination,
            "manifest": manifest,
            "manifest_exists": manifest_exists,
            "manifest_audit": manifest_audit,
            "items": items,
            "total_inputs": len(snap.lights),
            "input_displays": [light.display_name for light in snap.lights],
        }

    def _load_declaration(self, snap) -> dict:
        obj = service.load_json_object(snap.evidence_path)
        declaration = service.parse_declaration(obj)
        return {
            "kind": "load_declaration",
            "status": "COMPLETED",
            "declaration": declaration,
            "target_indices": list(snap.target_indices),
            "display": os.path.basename(snap.evidence_path),
        }

    def _load_roi(self, snap) -> dict:
        obj = service.load_json_object(snap.evidence_path)
        roi = service.parse_roi(obj)
        return {
            "kind": "load_roi",
            "status": "COMPLETED",
            "roi": roi,
            "target_indices": list(snap.target_indices),
            "display": os.path.basename(snap.evidence_path),
        }

    def _load_settings(self, snap) -> dict:
        result = settings_mod.load_settings(Path(snap.config_dir))
        return {
            "kind": "load_settings",
            "status": "COMPLETED",
            "state": result.state,
            "settings": result.settings.to_dict(),
        }

    def _save_settings(self, snap) -> dict:
        # Re-check the on-disk state (preserve unsupported/malformed bytes; never
        # overwrite data we did not understand).
        current = settings_mod.load_settings(Path(snap.config_dir))
        if current.state in (settings_mod.STATE_MALFORMED, settings_mod.STATE_UNSUPPORTED):
            return {
                "kind": "save_settings",
                "status": "PRESERVED",
                "state": current.state,
                "details": "existing settings preserved (not overwritten)",
            }
        settings_mod.save_settings(
            Path(snap.config_dir),
            settings_mod.GuiSettings.from_dict(snap.settings_payload),
        )
        return {"kind": "save_settings", "status": "COMPLETED", "state": current.state}

    # -- managed master ingestion (P7-M3B) ----------------------------------
    def _scan_masters(self, snap) -> dict:
        masters = []
        targets = list(snap.master_targets)
        if not targets:
            targets = [(p, snap.master_type) for p in snap.master_paths]
        for path, role in targets:
            try:
                entry = dict(service.scan_master_header(path, selected_role=role))
                # Keep the legacy ``master_type`` key for the presentation path;
                # the effective role is also exposed as ``role``.
                entry["master_type"] = entry.get("role")
                masters.append(entry)
            except Exception as exc:  # noqa: BLE001 - per-master scan failure
                masters.append({
                    "path": path,
                    "master_type": role,
                    "status": "FAILED",
                    "reason": str(exc),
                    "candidates": {},
                    "conflicts": {},
                    "incompatible": [],
                    "admissible": False,
                })
        return {
            "kind": "scan_masters",
            "status": "COMPLETED",
            "masters": masters,
            "count": len(masters),
        }

    def _load_ledger(self, snap) -> dict:
        result = v1.load_managed_ledger(v1.managed_ledger_path(snap.ledger_dir))
        return {
            "kind": "load_ledger",
            "status": "COMPLETED",
            "state": result.state,
            "records": [r.to_dict() for r in result.records],
            "count": len(result.records),
        }

    def _confirm_evidence(self, snap) -> dict:
        payload = snap.confirm_payload or {}
        path = payload.get("path")
        role = payload.get("role")
        if not path or role not in ("bias", "dark", "flat", "flat_dark"):
            raise v1.InvalidRequestError("confirm_evidence requires a path and a valid role")
        hdu = payload.get("hdu", 0)
        content_sha256, size_bytes = v1.content_identity(path, hdu=hdu)

        evidence: dict = {}
        for field, e in (payload.get("evidence") or {}).items():
            evidence[field] = v1.EvidenceFact(
                field=field,
                value=e["value"],
                origin_type=e.get("origin_type", "user"),
                origin_field=e.get("origin_field"),
                confirmed_by="user",
                version="1",
            )

        declaration = service.build_declaration(
            payload.get("declaration_source", "user"),
            payload.get("declaration_identity", "managed-import"),
            payload.get("declaration_version", "1"),
            evidence,
            payload.get("extra"),
        )
        record = service.make_managed_record(
            role=role,
            content_sha256=content_sha256,
            size_bytes=size_bytes,
            declaration=declaration,
            hdu=hdu,
            bias_state=payload.get("bias_state"),
            flat_form=payload.get("flat_form"),
            evidence=evidence,
            dq_state=payload.get("dq_state", "no_source_dq"),
            mask_path=payload.get("mask_path"),
            last_seen_path=path,
        )

        ledger_path = v1.managed_ledger_path(snap.ledger_dir)
        loaded = v1.load_managed_ledger(ledger_path)
        if loaded.state in (v1.STATE_MALFORMED, v1.STATE_UNSUPPORTED):
            return {
                "kind": "confirm_evidence",
                "status": "PRESERVED",
                "state": loaded.state,
                "details": "existing managed ledger preserved (not overwritten)",
            }
        status, missing = service.master_evidence_status(role, declaration)
        # Reuse known master: an unchanged record (same content identity + role +
        # evidence + declaration) is reused without re-asking; never duplicated.
        for existing in loaded.records:
            if (
                existing.content_sha256 == content_sha256
                and existing.size_bytes == size_bytes
                and existing.role == role
                and existing.declaration == declaration
                and dict(existing.evidence) == evidence
            ):
                return {
                    "kind": "confirm_evidence",
                    "status": "REUSED",
                    "record": existing.to_dict(),
                    "count": len(loaded.records),
                    "path": path,
                    "evidence_status": status,
                    "missing": list(missing),
                }
        # Content identity + role is authoritative over path (same bytes+role =
        # same record; same path+new bytes = new record).
        records = [
            r for r in loaded.records
            if not (r.content_sha256 == content_sha256 and r.role == role)
        ]
        records.append(record)
        v1.save_managed_ledger(ledger_path, records)
        return {
            "kind": "confirm_evidence",
            "status": "COMPLETED",
            "record": record.to_dict(),
            "count": len(records),
            "evidence_status": status,
            "missing": list(missing),
        }

    def _build_managed_library(self, snap) -> dict:
        result = v1.load_managed_ledger(v1.managed_ledger_path(snap.ledger_dir))
        if result.state in (v1.STATE_MALFORMED, v1.STATE_UNSUPPORTED):
            return {
                "kind": "build_managed_library",
                "status": "PRESERVED",
                "state": result.state,
                "details": "existing managed ledger preserved (not overwritten)",
            }
        # F3: build the derived session index ONLY from the current session
        # selection (content identity + role), reusing evidence from the ledger
        # by content identity. ``None`` = no filter (legacy full-ledger build);
        # ``()`` = explicit empty selection (build nothing).
        records = list(result.records)
        if snap.session_selection is not None:
            keys = set(snap.session_selection)
            records = [r for r in records if (r.content_sha256, r.role) in keys]
        ready = []
        attention = []
        for r in records:
            status, missing = service.master_evidence_status(r.role, r.declaration)
            if status == "ready":
                ready.append(r)
            else:
                attention.append({
                    "role": r.role,
                    "path": r.last_seen_path,
                    "missing": list(missing),
                })
        built = v1.build_managed_library(snap.managed_spec, ready)
        return {
            "kind": "build_managed_library",
            "status": built.status,
            "revision": built.revision,
            "candidate_count": built.candidate_count,
            "diagnostics": [
                d.to_dict() if hasattr(d, "to_dict") else str(d)
                for d in built.diagnostics
            ],
            "reason_code": built.reason_code,
            "details": built.details,
            "needs_attention": attention,
        }


class WorkerController(QtCore.QObject):
    """Main-thread controller owning the worker thread (one active op at a time).

    Progress is delivered through a bounded mailbox drained by a main-thread
    timer (coalesced wakeup); no progress signals are emitted from the worker
    thread, so the queued-event count cannot grow unbounded if the GUI stalls.

    Shutdown is finished-driven and non-blocking: ``shutdown()`` posts ``quit()``
    and the thread/worker delete themselves on ``thread.finished`` (the canonical
    Qt worker pattern), so no live QThread is ever destroyed and no GUI-thread
    ``wait()`` is performed.
    """

    operation_started = QtCore.Signal(str)
    progress = QtCore.Signal(str, object)
    preflight_light = QtCore.Signal(str, object, object)
    batch_item = QtCore.Signal(str, object)
    operation_finished = QtCore.Signal(str, object)
    operation_failed = QtCore.Signal(str, str, str)
    worker_ended = QtCore.Signal()
    shutdown_finished = QtCore.Signal()

    _DRAIN_INTERVAL_MS = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QtCore.QThread()  # no parent: Python-owned; released on main thread after the thread stops
        self._worker = _OperationWorker()  # no parent: Python-owned; released after the thread stops
        self._worker.moveToThread(self._thread)

        self._worker.started.connect(self.operation_started)
        self._worker.preflight_light.connect(self.preflight_light)
        self._worker.batch_item.connect(self.batch_item)
        self._worker.finished.connect(self.operation_finished)
        self._worker.failed.connect(self.operation_failed)
        self._worker.ended.connect(self._on_worker_ended)

        # Worker/thread lifetime (PySide6-safe; see ``_on_thread_finished`` and
        # ``finalize``). The worker C++ QObject is thread-affine to the worker
        # thread; the QThread object is affine to the creating/main thread and
        # must never be destroyed while running. Both Python references are kept
        # on the controller and released ONLY after the worker thread has fully
        # stopped (in ``finalize``, or when the controller is collected after
        # ``shutdown``) — never from ``_on_thread_finished``, which runs while
        # ``QThreadPrivate::finish`` has not yet cleared ``running``.
        #
        # The canonical Qt cleanup (Qt 6 QThread class reference, Detailed
        # Description — the ``worker->moveToThread`` example — connects
        # ``thread.finished`` to ``worker.deleteLater``) was considered but NOT
        # used here. Under this PySide6 6.11.2 build we observed an occasional
        # native crash (double-free) when a Python-owned ``moveToThread``-ed
        # QObject is ``deleteLater``-ed during shutdown; that instability has not
        # been isolated to a stable in-repo reproducer, so it is treated as
        # uncertain rather than a proven Shiboken defect. Retaining the
        # references until the thread has stopped, then dropping them on the main
        # thread, is the safer ownership pattern for Python-owned workers and is
        # what this implementation uses.
        self._thread.finished.connect(self._on_thread_finished)

        self._drain_timer = QtCore.QTimer(self)
        self._drain_timer.setInterval(self._DRAIN_INTERVAL_MS)
        self._drain_timer.timeout.connect(self._drain_progress)

        self._active = False
        self._shutting_down = False
        self._finished = False
        self._mailbox = None
        self._active_op_id = None
        self._thread.start()
        _register_thread(self._thread)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_finished(self) -> bool:
        return self._finished

    def start(self, snapshot: "service.OperationSnapshot", token) -> None:
        if self._active:
            raise RuntimeError("an operation is already active")
        if self._shutting_down:
            raise RuntimeError("controller is shutting down")
        self._active = True
        self._active_op_id = snapshot.op_id
        self._mailbox = ProgressMailbox()
        self._worker.start_op.emit(snapshot, token, self._mailbox)
        self._drain_timer.start()

    @QtCore.Slot()
    def _drain_progress(self) -> None:
        if self._mailbox is None:
            return
        event = self._mailbox.take()
        if event is not None:
            self.progress.emit(self._active_op_id, event)

    @QtCore.Slot()
    def _on_worker_ended(self) -> None:
        self._drain_timer.stop()
        self._drain_progress()  # mailbox is empty on terminal paths (no-op)
        self._active = False
        self._active_op_id = None
        self._mailbox = None
        self.worker_ended.emit()

    @QtCore.Slot()
    def _on_thread_finished(self) -> None:
        self._finished = True
        self.shutdown_finished.emit()
        # Unregister the thread from the atexit safety net. Deliberately do NOT
        # drop the ``self._worker`` / ``self._thread`` Python references here:
        #
        # * ``QThread::finished`` is emitted from the worker thread *before*
        #   ``QThreadPrivate::finish`` clears ``running``, so dropping the QThread
        #   reference here could destroy a still-running thread (a native abort:
        #   "QThread: Destroyed while thread is still running").
        # * the worker C++ QObject is thread-affine to the worker thread; it must
        #   not be destroyed on the main thread while that thread is still
        #   finishing.
        #
        # Both references are released later, only after the thread has fully
        # stopped (``finalize`` after ``wait()``, or controller collection after
        # ``shutdown``).
        if self._thread is not None:
            _unregister_thread(self._thread)

    def shutdown(self) -> None:
        """Finished-driven, non-blocking shutdown (no GUI-thread wait).

        Requests the thread to quit; the thread releases its references on
        ``thread.finished`` (see ``_on_thread_finished``). Never blocks the GUI
        and never terminates a thread.
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        self._drain_timer.stop()
        if self._thread is not None:
            self._thread.quit()

    def finalize(self, timeout_ms: int = 2000) -> None:
        """Bounded post-event-loop join for teardown (NOT for GUI slots/signals).

        Must never be invoked from a GUI slot/signal/aboutToQuit (those use the
        non-blocking ``shutdown()``). It is only safe after the event loop has
        stopped, or from an atexit/teardown handler. Idempotent; captures a LOCAL
        strong reference to the thread before joining; requests a non-blocking
        quit if not already; waits up to ``timeout_ms`` (a wedged thread is not
        waited on indefinitely).

        The worker/thread references are released ONLY if the thread has actually
        stopped. If ``wait()`` times out (a wedged thread), the references are
        deliberately retained (leak-not-crash): the atexit ``_LIVE_THREADS`` net
        performs a bounded join and logs once, but a still-running Python-owned
        QThread must never be destroyed here (native abort: "QThread: Destroyed
        while thread is still running"). No ``QThread.terminate`` anywhere.
        """
        thread = self._thread
        if thread is None:
            return
        if not self._shutting_down:
            self.shutdown()
        stopped = True
        if thread.isRunning():
            stopped = thread.wait(timeout_ms)  # False on timeout
        if stopped and not thread.isRunning():
            _unregister_thread(thread)
            self._worker = None
            self._thread = None
        # else: keep both references (the thread is still running); do not drop
        # them here.


__all__ = ["ProgressMailbox", "WorkerController"]
