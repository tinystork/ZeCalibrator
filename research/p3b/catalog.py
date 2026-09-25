"""P3B class catalogue (data, not executed code) — SCIENCE §15.

Declares the **20** synthetic site classes. This module is *data*: the entries
are declarative records, never executed. The generator knows how to
*materialise* any declared class (see :mod:`research.p3b.behaviors`), but the
catalogue itself only names and describes them.

An unknown class name raises the typed :class:`UnknownClassError` (never a bare
``KeyError`` / ``ValueError``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class ClassEntry:
    """One declarative catalogue entry for a synthetic site class."""

    name: str
    description: str
    # Declared ground-truth defaults (overridable per site). These are *truth
    # labels*, not product states, not thresholds.
    default_expected_qualification_state: str
    default_expected_action_state: str
    default_light_behavior: str
    default_dark_behavior: str


class UnknownClassError(KeyError):
    """Typed error for a synthetic class name that is not in the catalogue."""

    def __init__(self, name: object) -> None:
        self.name = name
        super().__init__(f"unknown synthetic class: {name!r}")


# The 20 classes, in the exact order given by the mission (SCIENCE §15).
_CLASS_NAMES: Tuple[str, ...] = (
    "NORMAL",
    "STABLE_ANOMALY_CORRECTED_BY_DARK",
    "STABLE_ANOMALY_WITH_MISMATCHED_DARK",
    "INTERMITTENT_TWO_STATE",
    "INTERMITTENT_MULTI_STATE",
    "INTERMITTENT_CONTINUOUS",
    "RARE_HIGH_STATE",
    "RARE_LOW_STATE",
    "SIGN_CHANGING_POST_DARK",
    "CENSORED_ANOMALY",
    "SINGLE_TRANSIENT",
    "OPTICAL_STRUCTURE",
    "STAR_CROSSING_SITE",
    "UNDERSAMPLED_STAR_CORE",
    "COSMIC_RAY",
    "NOISE_EXTREME",
    "FLAT_STRUCTURE",
    "DUST_OR_VIGNETTING",
    "NEAR_SATURATION",
    "AMBIGUOUS_INSUFFICIENT_EVIDENCE",
)

_CLASS_CATALOG: Mapping[str, ClassEntry] = {
    "NORMAL": ClassEntry(
        "NORMAL",
        "No anomaly; the site is indistinguishable from the local background.",
        "UNQUALIFIED",
        "NO_ACTION_REQUIRED",
        "absent",
        "absent",
    ),
    "STABLE_ANOMALY_CORRECTED_BY_DARK": ClassEntry(
        "STABLE_ANOMALY_CORRECTED_BY_DARK",
        "A stable additive anomaly present identically in light and dark; a "
        "representative dark corrects it.",
        "QUALIFIED_PERSISTENT",
        "NO_ACTION_REQUIRED",
        "present_stable",
        "present_stable",
    ),
    "STABLE_ANOMALY_WITH_MISMATCHED_DARK": ClassEntry(
        "STABLE_ANOMALY_WITH_MISMATCHED_DARK",
        "A stable additive anomaly in the light that the supplied dark does "
        "not contain (mismatched / non-representative dark).",
        "QUALIFIED_PERSISTENT",
        "REQUALIFY_ADDITIVE_CALIBRATION",
        "present_stable",
        "absent",
    ),
    "INTERMITTENT_TWO_STATE": ClassEntry(
        "INTERMITTENT_TWO_STATE",
        "Toggles deterministically between two levels across frames.",
        "QUALIFIED_INTERMITTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "intermittent",
        "absent",
    ),
    "INTERMITTENT_MULTI_STATE": ClassEntry(
        "INTERMITTENT_MULTI_STATE",
        "Cycles deterministically among more than two levels.",
        "QUALIFIED_INTERMITTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "intermittent",
        "absent",
    ),
    "INTERMITTENT_CONTINUOUS": ClassEntry(
        "INTERMITTENT_CONTINUOUS",
        "Drifts continuously across frames (a ramp, not discrete states).",
        "QUALIFIED_INTERMITTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "intermittent",
        "absent",
    ),
    "RARE_HIGH_STATE": ClassEntry(
        "RARE_HIGH_STATE",
        "Usually low, rarely and deterministically high.",
        "QUALIFIED_INTERMITTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "rare_high",
        "absent",
    ),
    "RARE_LOW_STATE": ClassEntry(
        "RARE_LOW_STATE",
        "Usually high, rarely and deterministically low.",
        "QUALIFIED_INTERMITTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "rare_low",
        "absent",
    ),
    "SIGN_CHANGING_POST_DARK": ClassEntry(
        "SIGN_CHANGING_POST_DARK",
        "Residual after dark subtraction changes sign across frames.",
        "QUALIFIED_INTERMITTENT",
        "ABSTAIN_INCONSISTENT",
        "sign_changing",
        "present_stable",
    ),
    "CENSORED_ANOMALY": ClassEntry(
        "CENSORED_ANOMALY",
        "Hits the acquisition hard limit; the value is censored and the "
        "quantitative residual is not trustworthy.",
        "CENSORED",
        "ABSTAIN_CENSORED",
        "censored",
        "absent",
    ),
    "SINGLE_TRANSIENT": ClassEntry(
        "SINGLE_TRANSIENT",
        "A one-frame transient spike at the site.",
        "QUALIFIED_TRANSIENT",
        "NO_ACTION_REQUIRED",
        "transient",
        "absent",
    ),
    "OPTICAL_STRUCTURE": ClassEntry(
        "OPTICAL_STRUCTURE",
        "A static optical structure (e.g. a ring) in the light.",
        "QUALIFIED_PERSISTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "spatial",
        "absent",
    ),
    "STAR_CROSSING_SITE": ClassEntry(
        "STAR_CROSSING_SITE",
        "A moving star whose PSF crosses the site across frames.",
        "QUALIFIED_TRANSIENT",
        "NO_ACTION_REQUIRED",
        "transient",
        "absent",
    ),
    "UNDERSAMPLED_STAR_CORE": ClassEntry(
        "UNDERSAMPLED_STAR_CORE",
        "A star core smaller than a pixel (energy concentrated in one pixel).",
        "QUALIFIED_PERSISTENT",
        "NO_ACTION_REQUIRED",
        "spatial",
        "absent",
    ),
    "COSMIC_RAY": ClassEntry(
        "COSMIC_RAY",
        "A bright cosmic-ray hit in a single frame.",
        "QUALIFIED_TRANSIENT",
        "NO_ACTION_REQUIRED",
        "transient",
        "absent",
    ),
    "NOISE_EXTREME": ClassEntry(
        "NOISE_EXTREME",
        "Extreme per-frame noise at the site.",
        "UNQUALIFIED",
        "ABSTAIN_INSUFFICIENT_EVIDENCE",
        "noise",
        "absent",
    ),
    "FLAT_STRUCTURE": ClassEntry(
        "FLAT_STRUCTURE",
        "A structure present only in the flat (e.g. a dust donut).",
        "QUALIFIED_PERSISTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "spatial",
        "absent",
    ),
    "DUST_OR_VIGNETTING": ClassEntry(
        "DUST_OR_VIGNETTING",
        "Large-scale dust/vignetting gradient affecting light and flat.",
        "QUALIFIED_PERSISTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "spatial",
        "absent",
    ),
    "NEAR_SATURATION": ClassEntry(
        "NEAR_SATURATION",
        "Persistently near (but below) the saturation limit.",
        "QUALIFIED_PERSISTENT",
        "ELIGIBLE_FOR_TARGETED_RECONSTRUCTION",
        "near_saturation",
        "absent",
    ),
    "AMBIGUOUS_INSUFFICIENT_EVIDENCE": ClassEntry(
        "AMBIGUOUS_INSUFFICIENT_EVIDENCE",
        "A weak anomaly with insufficient evidence to qualify.",
        "UNQUALIFIED",
        "ABSTAIN_INSUFFICIENT_EVIDENCE",
        "weak",
        "absent",
    ),
}


def resolve_class(name: str) -> ClassEntry:
    """Return the catalogue entry for ``name`` or raise :class:`UnknownClassError`."""
    if not isinstance(name, str):
        raise UnknownClassError(name)
    try:
        return _CLASS_CATALOG[name]
    except KeyError as exc:
        raise UnknownClassError(name) from exc


# Public read-only views (tuples / frozensets are immutable).
CLASS_NAMES: Tuple[str, ...] = _CLASS_NAMES
CLASS_CATALOG: Mapping[str, ClassEntry] = _CLASS_CATALOG


__all__ = [
    "CLASS_CATALOG",
    "CLASS_NAMES",
    "ClassEntry",
    "UnknownClassError",
    "resolve_class",
]
