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

# Standard (progressive-disclosure) labels. These are presentation-only: each
# label maps 1:1 to the exact frozen runtime value carried by the Advanced
# technical combos (``control`` / ``bias_only`` / ``dark_incl_bias`` /
# ``dark_bias_removed`` and ``none`` / ``apply``). No value is invented.
_STANDARD_MODE_LABELS = {
    "control": "None",
    "bias_only": "Bias only",
    "dark_incl_bias": "Standard dark",
    "dark_bias_removed": "Dark already bias-corrected",
    "none": "None",
    "apply": "Use flat",
}

# Human-first outcomes for the Standard summary. Exact technical outcomes remain
# accessible in Advanced; these never re-rank, re-guess or pick a candidate.
_HUMAN_OUTCOME_LABELS = {
    "MATCHED": "Ready",
    "NO_MATCH": "Needs attention",
    "AMBIGUOUS": "Ambiguous calibration set",
}

# Human names for the matcher's structural roles (presentation fallback only).
_ROLE_HUMAN_NAMES = {
    "dark": "dark",
    "bias": "bias",
    "flat": "flat",
    "flat_dark": "flat dark",
}

# Inspect-failure human explanations (presentation fallback only). Exact
# ``reason_code``/``details`` remain in Advanced; these never expose raw
# filesystem exception prose.
_INSPECT_SOURCE_UNREADABLE = "Could not read this image."
_INSPECT_NEEDS_EVIDENCE = "This image needs valid raw-sensor import evidence."
_INSPECT_DECODE_FAILED = "This file could not be decoded as a supported raw FITS image."
_INSPECT_NOT_RAW = "This image is not raw 2-D sensor data (it looks processed or colour)."
_INSPECT_HDU = "The selected HDU has no usable 2-D image — pick another HDU in Advanced."
_INSPECT_GENERIC = "This image could not be inspected."

# Reason codes that mean the source file was missing/unreadable.
_SOURCE_READ_CODES = frozenset({"SOURCE_ERROR"})

# Reason codes that mean the frame lacks valid raw-domain/unit import evidence.
_EVIDENCE_CODES = frozenset({"UNKNOWN_DOMAIN", "UNKNOWN_UNITS", "UNITS_UNSUPPORTED"})

# Reason codes that mean the file/data could not be decoded as a supported raw
# FITS image (precision/range refusal, malformed cards, invalid/conflicting
# metadata or source).
_DECODE_FAILED_CODES = frozenset({
    "PRECISION_REFUSAL",
    "MALFORMED_CARD",
    "METADATA_ADAPTER",
    "INVALID_SOURCE",
    "ALIAS_CONFLICT",
    "DECLARATION_CONFLICT",
})

# Reason codes that mean the input is processed/colour rather than raw 2-D data.
_NOT_RAW_CODES = frozenset({"PROCESSED_HISTORY", "RGB_UNSUPPORTED"})

# Reason codes that mean the selected HDU has no usable 2-D image.
_HDU_CODES = frozenset({
    "HDU_NOT_FOUND", "HDU_NOT_2D_IMAGE", "HDU_AMBIGUOUS", "NO_2D_IMAGE", "NOT_2D_PLANE",
})

_ADDITIVE_MODES = ("control", "bias_only", "dark_incl_bias", "dark_bias_removed")
_FLAT_MODES = ("none", "apply")


def status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


def additive_mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


def flat_mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


def standard_additive_mode_label(mode: str) -> str:
    """Standard (human) label for an additive mode value (presentation only)."""
    return _STANDARD_MODE_LABELS.get(mode, mode)


def standard_flat_mode_label(mode: str) -> str:
    """Standard (human) label for a flat mode value (presentation only)."""
    return _STANDARD_MODE_LABELS.get(mode, mode)


def human_outcome_label(outcome) -> str:
    """Map a technical outcome to its human-first Standard label.

    ``None`` (inspection failed / not resolved) is reported truthfully as
    "Needs attention" rather than collapsed into a technical token.
    """
    if outcome is None:
        return "Needs attention"
    return _HUMAN_OUTCOME_LABELS.get(outcome, outcome)


def human_reason_text(summary: Mapping) -> str:
    """Concise, truthful human explanation for one preflight summary.

    This is a presentation mapping/fallback only: it never selects, ranks or
    guesses a candidate, and never exposes exact codes/expected/observed values
    (those remain in Advanced). Auto-route reasons (Standard) are mapped to
    plain-language sentences; indeterminate masters are reported truthfully,
    never as a formula/mode questionnaire.

    The auto-route-only sentences are **outcome-aware**: a failure sentence is
    never shown on a resolved/ready (``MATCHED``) row, and an ``AMBIGUOUS`` row
    reports conflicting/indeterminate master declarations rather than a single
    bias-required note. The durable ``reason_codes``/``reasons`` audit channels
    are left unchanged.
    """
    outcome = summary.get("outcome")
    codes = tuple(summary.get("reason_codes") or ())
    if outcome == "MATCHED":
        # A resolved route never renders a failure sentence (the audit reasons
        # may still carry non-blocking/informative notes such as an incompatible
        # dark with an indeterminate bias state).
        return "A compatible calibration set was found."
    if outcome == "AMBIGUOUS":
        if "BIAS_STATE_UNKNOWN" in codes or "BIAS_REQUIRED" in codes:
            return (
                "Multiple calibration routes are possible: the supplied masters "
                "have conflicting or indeterminate declarations."
            )
        return "More than one compatible calibration set was found."
    # Needs attention (NO_MATCH) or unresolved (None): explain the concrete
    # reason; the failure sentences apply only here.
    if "BIAS_STATE_UNKNOWN" in codes:
        return (
            "Master dark cannot be used automatically: its processing history "
            "does not establish whether bias has already been removed."
        )
    if "FLAT_UNSUPPORTED_RAW" in codes:
        return (
            "A raw (unprocessed) flat master was supplied, which cannot be used "
            "automatically in Standard; a ready-to-use flat is required."
        )
    if "FLAT_UNUSABLE" in codes or "FLAT_ADDITIVE_DEPENDENCY_MISSING" in codes:
        return (
            "A flat master was supplied but cannot be used automatically: no "
            "compatible flat-dark or bias dependency was found."
        )
    if "PARTIAL_ADDITIVE" in codes:
        return (
            "Only a partial correction is available (no compatible dark master); "
            "this is not a full calibration."
        )
    if "BIAS_REQUIRED" in codes:
        return (
            "A bias master is required for this dark master (bias already removed) "
            "but no compatible bias was found."
        )
    if outcome == "NO_MATCH":
        role = _unavailable_role(summary.get("reasons", ()))
        if role:
            return f"No compatible {_ROLE_HUMAN_NAMES.get(role, role)} found."
        return "No compatible calibration set was found."
    # outcome is None: the frame could not be inspected/resolved. Explain in
    # plain language from the existing reason-code category (never raw details).
    return _inspect_failure_text(summary.get("reason_code"))


def _inspect_failure_text(reason_code) -> str:
    """Map an inspect/resolve failure reason code to a concise human sentence."""
    if reason_code in _SOURCE_READ_CODES:
        return _INSPECT_SOURCE_UNREADABLE
    if reason_code in _EVIDENCE_CODES:
        return _INSPECT_NEEDS_EVIDENCE
    if reason_code in _DECODE_FAILED_CODES:
        return _INSPECT_DECODE_FAILED
    if reason_code in _NOT_RAW_CODES:
        return _INSPECT_NOT_RAW
    if reason_code in _HDU_CODES:
        return _INSPECT_HDU
    return _INSPECT_GENERIC


def _unavailable_role(reasons) -> str | None:
    """Return the first structurally unavailable role, or ``None``.

    Only the manifest ``ROLE_UNAVAILABLE`` reason is interpreted, and only to
    name a *missing* required role — never to choose a candidate.
    """
    for reason in reasons:
        if reason.get("code") == "ROLE_UNAVAILABLE" and reason.get("role"):
            return reason.get("role")
    return None


def summarize_outcomes(summaries: Sequence[Mapping]) -> tuple[int, int, int]:
    """Count (ready, needs_attention, ambiguous) from per-light summaries."""
    ready = attention = ambiguous = 0
    for summary in summaries:
        outcome = summary.get("outcome")
        if outcome == "MATCHED":
            ready += 1
        elif outcome == "AMBIGUOUS":
            ambiguous += 1
        else:
            attention += 1
    return ready, attention, ambiguous


def format_outcome_summary(ready: int, attention: int, ambiguous: int) -> str:
    """Render a truthful human outcome-count summary (empty until verified)."""
    parts = []
    if ready:
        parts.append(f"{ready} {_plural(ready, 'image', 'images')} ready")
    if attention:
        parts.append(f"{attention} {_plural(attention, 'needs', 'need')} attention")
    if ambiguous:
        parts.append(
            f"{ambiguous} {_plural(ambiguous, 'ambiguous calibration set', 'ambiguous calibration sets')}"
        )
    return "; ".join(parts) or "No images verified."


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def format_folder_add_feedback(added: int, unsupported: int) -> str:
    """Truthful feedback after a folder add (presentation only).

    Always reports how many images were added; when non-input files were skipped
    it appends the ignored count so the user knows nothing was silently dropped.
    """
    text = f"{added} {_plural(added, 'image', 'images')} added"
    if unsupported:
        text += f" ({unsupported} {_plural(unsupported, 'unsupported file', 'unsupported files')} ignored)"
    return text


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


# ---------------------------------------------------------------------------
# Managed master ingestion presentation (P7-M3B)
# ---------------------------------------------------------------------------
_ORIGIN_TYPE_LABELS = {
    "fits_header": "FITS header",
    "user": "user",
    "sidecar": "sidecar",
    "native": "native",
}

_DQ_STATE_LABELS = {
    "source_mask": "Source DQ mask",
    "no_source_dq": "No source DQ",
}


def dq_state_label(dq_state) -> str:
    """Human label for a DQ state (presentation only)."""
    return _DQ_STATE_LABELS.get(dq_state, dq_state)


def format_evidence_fact(fact: Mapping) -> str:
    """Render one detected/confirmed evidence fact for the confirmation UI."""
    field = fact.get("field", "?")
    value = _fmt_value(fact.get("value"))
    origin = fact.get("origin_field")
    otype = _ORIGIN_TYPE_LABELS.get(fact.get("origin_type"), fact.get("origin_type", "?"))
    loc = f" @ {origin}" if origin else ""
    return f"{field}: {value} — source {otype}{loc}"


def format_conflict(field: str, facts: Sequence[Mapping]) -> str:
    """Render a detected evidence conflict (never auto-resolved)."""
    vals = ", ".join(_fmt_value(f.get("value")) for f in facts)
    return f"{field}: conflicting detected values ({vals}) — review, not auto-resolved"


def format_incompatibility(reason: str) -> str:
    """Render one master-incompatibility reason (presentation only)."""
    return f"incompatible: {reason}"


def format_role_conflict(conflict: Mapping) -> str:
    """Render a selected-vs-detected role conflict (never auto-resolved)."""
    selected = conflict.get("selected_role")
    detected = conflict.get("detected_role")
    source = conflict.get("source", "FITS IMAGETYP")
    return (
        f"Selected role: {selected} / Detected role: {detected} / "
        f"Source: {source} / Needs confirmation"
    )


__all__ = [
    "additive_mode_label",
    "additive_modes",
    "dq_state_label",
    "flat_mode_label",
    "flat_modes",
    "format_coherent_sets",
    "format_conflict",
    "format_evidence_fact",
    "format_incompatibility",
    "format_metadata",
    "format_outcome_summary",
    "format_rejection_table",
    "format_role_conflict",
    "format_folder_add_feedback",
    "human_outcome_label",
    "human_reason_text",
    "is_partial_mode",
    "reason_codes_text",
    "standard_additive_mode_label",
    "standard_flat_mode_label",
    "status_label",
    "summarize_outcomes",
]
