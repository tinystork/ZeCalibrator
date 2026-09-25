"""Dataset separation + anti-leakage guard (LOT7) — internal, non-public.

Splits the P3B synthetic corpus into three datasets with distinct roles and
distinct seeds, each carrying a **reproducible manifest**, and enforces a
**signature-based anti-leakage guard** at the dataset level:

    DEVELOPMENT     build / verify the harness
    QUALIFICATION   the current measurement
    HOLDOUT         frozen; evaluated once; never used to choose anything

The guard is **not** statistical similarity and **not** an invented threshold.
It is a deterministic, identifiable signature of the *declared* scenario — a
fingerprint of the declared parameters (sensor identity, geometry, CFA phase,
site classes + coordinates + params, seed, group/epoch structure). A signature
collision between two datasets is a **typed refusal**
(:class:`SignatureCollisionError`), never a silent warning.

Nothing here reads pixels, computes a metric, chooses a threshold, selects an
operating point, or claims *independent real validation*. That field stays
explicitly ``NO`` / ``NOT_SATISFIED`` (the M74 research witness does **not**
satisfy it — a separate dataset, and preferably a separate sensor family,
remains required for any later gate).

The signature deliberately **excludes** the scenario ``name`` and the dataset
role: renaming a scenario or re-presenting it under another dataset label does
not change what the scenario *is*, so the same declared science always yields
the same signature and is caught as a leak.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from dataclasses import dataclass
from typing import Tuple

from .generator import cfa_plane_label
from .model import ScenarioSpec, compute_bookkeeping

# ---------------------------------------------------------------------------
# Dataset roles (distinct, each with its own seed discipline)
# ---------------------------------------------------------------------------

DATASET_DEVELOPMENT = "DEVELOPMENT"
DATASET_QUALIFICATION = "QUALIFICATION"
DATASET_HOLDOUT = "HOLDOUT"
DATASET_ROLES: Tuple[str, ...] = (
    DATASET_DEVELOPMENT,
    DATASET_QUALIFICATION,
    DATASET_HOLDOUT,
)

# ---------------------------------------------------------------------------
# Real-validation status — never claimed by this harness (property 5)
# ---------------------------------------------------------------------------

# The field value carried by every manifest: independent real validation is NOT
# satisfied. M74 does not satisfy it; another dataset (preferably another sensor
# family) remains required for a later gate.
INDEPENDENT_REAL_VALIDATION = "NO"
REAL_VALIDATION_NOT_SATISFIED = "NOT_SATISFIED"

MANIFEST_SCHEMA = "zecalibrator-p3b-lot7-dataset-split"
MANIFEST_VERSION = 1


# ---------------------------------------------------------------------------
# Typed errors (the guard refuses, it never warns silently)
# ---------------------------------------------------------------------------


class DatasetSeparationError(ValueError):
    """Base class for dataset-separation guard violations."""


class UnknownDatasetRoleError(DatasetSeparationError):
    """Typed error: a role is not one of the three dataset roles."""

    def __init__(self, role: object) -> None:
        self.role = role
        super().__init__(
            f"unknown dataset role {role!r}; expected one of {DATASET_ROLES!r}"
        )


class SignatureCollisionError(DatasetSeparationError):
    """Typed error: the same scenario signature appears in two different datasets.

    This is the anti-leakage guard. Two scenarios of the **same signature**
    (same declared sensor/geometry/CFA phase/site classes+coordinates+params/
    seed/group-epoch structure) presented as belonging to two different datasets
    is a leak, and it is refused here — not downweighted, not warned about.
    """

    def __init__(
        self,
        signature: str,
        first_role: str,
        second_role: str,
        scenario_name: str,
    ) -> None:
        self.signature = signature
        self.first_role = first_role
        self.second_role = second_role
        self.scenario_name = scenario_name
        super().__init__(
            f"signature collision: scenario {scenario_name!r} (signature "
            f"{signature!r}) already belongs to dataset {first_role!r}; it cannot "
            f"also belong to {second_role!r}"
        )


class DuplicateScenarioError(DatasetSeparationError):
    """Typed error: the same signature was registered twice in the SAME dataset.

    Distinct from :class:`SignatureCollisionError`: this is a within-dataset
    duplicate (double counting), not a cross-dataset leak.
    """

    def __init__(self, signature: str, role: str, scenario_name: str) -> None:
        self.signature = signature
        self.role = role
        self.scenario_name = scenario_name
        super().__init__(
            f"duplicate scenario: {scenario_name!r} (signature {signature!r}) was "
            f"already registered in dataset {role!r}"
        )


# ---------------------------------------------------------------------------
# Deterministic, identifiable scenario signature (the core of the guard)
# ---------------------------------------------------------------------------


def _canonical_value(value):
    """Canonicalise one declared parameter value to a JSON-native form.

    Distinguishes bool / int / float / str / None / sequence; any other type
    (a numpy scalar, for example) falls back to ``repr``. Declared parameters
    are Python literals in this harness, so this is deterministic.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [_canonical_value(v) for v in value]
    return repr(value)


def _canonical_params(params) -> list:
    out = []
    for key, value in params:
        out.append([str(key), _canonical_value(value)])
    return out


def _canonical_scenario(scenario: ScenarioSpec) -> dict:
    """The canonical declared structure a signature is computed over.

    Captures exactly the declared parameters the mission names: sensor identity,
    geometry, CFA phase, site classes + coordinates + params, seed, and the
    group/epoch structure. The scenario ``name`` and the dataset role are
    **excluded** on purpose (a label is not part of what the data *is*).
    """
    sensor = scenario.sensor
    sensor_dict = {
        "instance_id": sensor.instance_id,
        "model": sensor.model,
        "shape": list(sensor.shape),
        "cfa_pattern": sensor.cfa_pattern,
        "binning": list(sensor.binning),
        "roi_origin": list(sensor.roi_origin),
        "gain": float(sensor.gain),
        "offset_adu": float(sensor.offset_adu),
        "saturation_limit_adu": float(sensor.saturation_limit_adu),
        "filter": sensor.filter,
    }

    sites = []
    for site in scenario.sites:
        sites.append(
            {
                "site_id": site.site_id,
                "x": int(site.x),
                "y": int(site.y),
                "cfa_class": site.cfa_class,
                "cfa_plane": cfa_plane_label(site.y, site.x, sensor.cfa_pattern),
                "expected_qualification_state": site.expected_qualification_state,
                "expected_action_state": site.expected_action_state,
                "calibration_representativeness": site.calibration_representativeness,
                "censored": bool(site.censored),
                "hard_limit_adu": (
                    None if site.hard_limit_adu is None else float(site.hard_limit_adu)
                ),
                "params": _canonical_params(site.params),
            }
        )

    epochs = []
    for epoch in scenario.epochs:
        groups = []
        for group in epoch.groups:
            frames = []
            for frame in group.frames:
                frames.append(
                    {
                        "frame_id": frame.frame_id,
                        "frame_type": frame.frame_type,
                        "ordinal": int(frame.ordinal),
                        "exposure_s": float(frame.exposure_s),
                        "temperature_c": float(frame.temperature_c),
                    }
                )
            groups.append({"group_id": group.group_id, "frames": frames})
        epochs.append({"epoch_id": epoch.epoch_id, "groups": groups})

    aggregates = []
    for agg in scenario.aggregates:
        aggregates.append(
            {
                "aggregate_id": agg.aggregate_id,
                "method": agg.method,
                "constituent_frame_ids": list(agg.constituent_frame_ids),
            }
        )

    return {
        "seed": int(scenario.seed),
        "sensor": sensor_dict,
        "sites": sites,
        "epochs": epochs,
        "aggregates": aggregates,
    }


def _stable_json_dumps(payload: dict) -> str:
    """Byte-stable JSON serialisation (key-sorted, compact, ASCII)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scenario_signature(scenario: ScenarioSpec) -> str:
    """Return the deterministic, identifiable signature of a declared scenario.

    A SHA-256 fingerprint of the canonical declared structure (sensor identity,
    geometry, CFA phase, site classes + coordinates + params, seed, group/epoch
    structure). Same declaration ⇒ same signature; any declared difference ⇒ a
    different signature. Not a statistical similarity, not a threshold.
    """
    return _sha256(_stable_json_dumps(_canonical_scenario(scenario)))


# ---------------------------------------------------------------------------
# Reproducible manifest records
# ---------------------------------------------------------------------------


def _group_structure(scenario: ScenarioSpec) -> Tuple[Tuple[str, str, int], ...]:
    """The (epoch_id, group_id, frame_count) listing — the group/epoch structure."""
    out = []
    for epoch in scenario.epochs:
        for group in epoch.groups:
            out.append((epoch.epoch_id, group.group_id, len(group.frames)))
    return tuple(out)


@dataclass(frozen=True)
class ScenarioRecord:
    """One scenario's manifest record: identity, seed, signature, structure."""

    name: str
    seed: int
    signature: str
    frame_count: int
    observation_count: int
    independent_group_count: int
    epoch_count: int
    group_structure: Tuple[Tuple[str, str, int], ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "signature": self.signature,
            "bookkeeping": {
                "frame_count": self.frame_count,
                "observation_count": self.observation_count,
                "independent_group_count": self.independent_group_count,
                "epoch_count": self.epoch_count,
            },
            "group_structure": [list(item) for item in self.group_structure],
        }


def _record(scenario: ScenarioSpec) -> ScenarioRecord:
    bk = compute_bookkeeping(scenario)
    return ScenarioRecord(
        name=scenario.name,
        seed=int(scenario.seed),
        signature=scenario_signature(scenario),
        frame_count=bk.frame_count,
        observation_count=bk.observation_count,
        independent_group_count=bk.independent_group_count,
        epoch_count=bk.epoch_count,
        group_structure=_group_structure(scenario),
    )


@dataclass(frozen=True)
class DatasetManifest:
    """The reproducible manifest of one dataset (one of the three roles).

    ``independent_real_validation`` and ``real_validation_status`` are always
    ``NO`` / ``NOT_SATISFIED`` — the harness never produces or implies that it
    satisfies independent real validation (property 5).
    """

    role: str
    manifest_version: int
    independent_real_validation: str
    real_validation_status: str
    scenarios: Tuple[ScenarioRecord, ...]
    manifest_hash: str

    def to_dict(self) -> dict:
        return {
            "manifest_schema": MANIFEST_SCHEMA,
            "manifest_version": self.manifest_version,
            "dataset_role": self.role,
            "independent_real_validation": self.independent_real_validation,
            "real_validation_status": self.real_validation_status,
            "manifest_hash": self.manifest_hash,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


def _require_role(role: str) -> None:
    if not isinstance(role, str) or role not in DATASET_ROLES:
        raise UnknownDatasetRoleError(role)


def build_manifest(role: str, scenarios) -> DatasetManifest:
    """Build a reproducible manifest for ``role`` from a sequence of scenarios.

    Pure and deterministic: the same ``(role, scenarios)`` yields the same
    ``manifest_hash`` and the same per-scenario signatures. The ``manifest_hash``
    covers the role + records (never the hash itself).
    """
    _require_role(role)
    records = tuple(_record(s) for s in scenarios)
    content = {
        "manifest_schema": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "dataset_role": role,
        "independent_real_validation": INDEPENDENT_REAL_VALIDATION,
        "real_validation_status": REAL_VALIDATION_NOT_SATISFIED,
        "scenarios": [r.to_dict() for r in records],
    }
    return DatasetManifest(
        role=role,
        manifest_version=MANIFEST_VERSION,
        independent_real_validation=INDEPENDENT_REAL_VALIDATION,
        real_validation_status=REAL_VALIDATION_NOT_SATISFIED,
        scenarios=records,
        manifest_hash=_sha256(_stable_json_dumps(content)),
    )


# ---------------------------------------------------------------------------
# The dataset-level guard (signature collision ⇒ typed refusal)
# ---------------------------------------------------------------------------


class DatasetRegistry:
    """Tracks scenario signatures across the three datasets and refuses leaks.

    A signature is registered under the role it first appears in. Registering a
    scenario whose signature already belongs to a **different** role raises
    :class:`SignatureCollisionError`; re-registering the same signature in the
    **same** role raises :class:`DuplicateScenarioError`. This is the whole
    anti-leakage mechanism at the dataset level.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_signature_roles", {})
        object.__setattr__(self, "_manifests", {})

    def register(self, role: str, scenarios) -> DatasetManifest:
        """Register ``scenarios`` under ``role`` and return the manifest.

        Raises :class:`SignatureCollisionError` on a cross-dataset collision and
        :class:`DuplicateScenarioError` on a within-dataset duplicate.
        """
        _require_role(role)
        items = tuple(scenarios)
        pending = []
        for scenario in items:
            signature = scenario_signature(scenario)
            previous = self._signature_roles.get(signature)
            if previous is not None and previous != role:
                raise SignatureCollisionError(signature, previous, role, scenario.name)
            if previous == role:
                raise DuplicateScenarioError(signature, role, scenario.name)
            pending.append((scenario, signature))
        for scenario, signature in pending:
            self._signature_roles[signature] = role
        manifest = build_manifest(role, items)
        self._manifests[role] = manifest
        return manifest

    def manifest(self, role: str) -> DatasetManifest:
        """Return the last manifest built for ``role`` (KeyError when absent)."""
        _require_role(role)
        return self._manifests[role]

    def manifests(self) -> Tuple[DatasetManifest, ...]:
        """Return the manifests, in role insertion order."""
        return tuple(self._manifests[role] for role in DATASET_ROLES if role in self._manifests)


__all__ = [
    "DATASET_DEVELOPMENT",
    "DATASET_HOLDOUT",
    "DATASET_QUALIFICATION",
    "DATASET_ROLES",
    "INDEPENDENT_REAL_VALIDATION",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "REAL_VALIDATION_NOT_SATISFIED",
    "DatasetManifest",
    "DatasetRegistry",
    "DatasetSeparationError",
    "DuplicateScenarioError",
    "ScenarioRecord",
    "SignatureCollisionError",
    "UnknownDatasetRoleError",
    "build_manifest",
    "scenario_signature",
]
