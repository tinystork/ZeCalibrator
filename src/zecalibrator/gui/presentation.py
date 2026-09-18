"""Pure presentation helpers for GUI display (stdlib only, no science).

These functions format already-computed public value-object summaries into
human-readable strings and table rows. They never perform arithmetic, matching,
parsing or provenance assembly — widgets render these strings only.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Human labels for the frozen status/disposition/outcome vocabulary.
_STATUS_LABELS = {
    "OPENED": "Opened",
    "COMPLETED": "Completed",
    "COMPLETED_WITH_WARNINGS": "Completed with warnings",
    "SKIPPED": "Skipped",
    "FAILED": "Failed",
    "CANCELLED": "Cancelled",
    "PARTIAL": "Partial",
    "MATCHED": "Matched",
    "NO_MATCH": "No match",
    "AMBIGUOUS": "Ambiguous",
}

_MODE_LABELS = {
    "control": "Control (no additive correction — uncalibrated/partial)",
    "bias_only": "Bias only (partial)",
    "dark_incl_bias": "Dark including bias",
    "dark_bias_removed": "Bias-removed dark",
    "none": "No flat",
    "apply": "Apply flat",
}

_ADDITIVE_MODES = ("control", "bias_only", "dark_incl_bias", "dark_bias_removed")
_FLAT_MODES = ("none", "apply")


def status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


def additive_mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


def flat_mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


def additive_modes() -> tuple:
    return _ADDITIVE_MODES


def flat_modes() -> tuple:
    return _FLAT_MODES


def is_partial_mode(additive_mode: str, flat_mode: str) -> bool:
    """True when the selected modes are explicitly partial (never full calibration)."""
    return additive_mode == "control" or flat_mode == "none"


def _fmt_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_value(v) for v in value)
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={_fmt_value(v)}" for k, v in value.items())
    return str(value)


def format_rejection_table(rejected_candidates: Sequence[Mapping]) -> str:
    """Render structured candidate rejection records as a readable table.

    Each record is ``{candidate_id, role, reasons:[{code,field,role,parent,
    expected,observed}, ...]}`` (public ``RejectionRecord.to_dict`` shape).
    """
    lines: list[str] = []
    for rec in rejected_candidates:
        cid = rec.get("candidate_id", "?")
        role = rec.get("role", "?")
        lines.append(f"[{role}] {cid}")
        for reason in rec.get("reasons", ()):
            code = reason.get("code", "?")
            field = reason.get("field")
            expected = _fmt_value(reason.get("expected"))
            observed = _fmt_value(reason.get("observed"))
            loc = f" @ {field}" if field else ""
            lines.append(f"    - {code}{loc}: expected {expected}, observed {observed}")
    return "\n".join(lines) if lines else "(no rejected candidates)"


def format_coherent_sets(coherent_sets: Sequence[Mapping]) -> str:
    """Render coherent candidate sets (role -> candidate_id)."""
    if not coherent_sets:
        return "(no coherent sets)"
    lines = []
    for i, s in enumerate(coherent_sets, 1):
        body = ", ".join(f"{role}={cid}" for role, cid in sorted(s.items()))
        lines.append(f"set {i}: {body}")
    return "\n".join(lines)


def format_metadata(metadata: Mapping) -> str:
    """Render the inspected metadata summary (already JSON-safe) as key=value lines."""
    if not metadata:
        return "(no metadata)"
    lines: list[str] = []
    for key in (
        "raw_domain_declaration", "units", "exposure_s", "temperature_c", "gain",
        "offset", "readout_mode", "adc_mode", "filter", "detector_model",
        "detector_instance_id", "optical_train_id", "saturation_limit_adu",
        "saturation_evidence",
    ):
        lines.append(f"{key}: {_fmt_value(metadata.get(key))}")
    geo = metadata.get("geometry")
    if isinstance(geo, Mapping):
        for gkey in ("shape", "sensor_dimensions", "binning", "roi_origin",
                     "roi_extent", "orientation", "cfa_phase"):
            lines.append(f"geometry.{gkey}: {_fmt_value(geo.get(gkey))}")
    warnings = metadata.get("warnings")
    if warnings:
        lines.append("warnings:")
        for w in warnings:
            lines.append(f"    - {w}")
    conflicts = metadata.get("conflicts")
    if conflicts:
        lines.append("conflicts:")
        for c in conflicts:
            lines.append(f"    - {_fmt_value(c)}")
    return "\n".join(lines)


def reason_codes_text(reason_codes: Sequence[str]) -> str:
    return ", ".join(reason_codes) if reason_codes else "(none)"


__all__ = [
    "additive_mode_label",
    "additive_modes",
    "flat_mode_label",
    "flat_modes",
    "format_coherent_sets",
    "format_metadata",
    "format_rejection_table",
    "is_partial_mode",
    "reason_codes_text",
    "status_label",
]
