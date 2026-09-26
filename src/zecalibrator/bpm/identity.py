"""Bad Pixel Database sensor identity / liaison.

The BPM selection identity *reuses* the normative sensor identity value objects
(:class:`~zecalibrator.core.descriptors.DetectorIdentity`,
:class:`~zecalibrator.core.geometry.Geometry`, and the readout-context projection
of :class:`~zecalibrator.core.descriptors.Acquisition`) — it does **not** invent
a parallel, simplified identity.

A :class:`SensorIdentity` binds the facts a site ``(y, x)`` is meaningful in:

* physical identity — detector instance/model (serial carried, never matched),
* geometry — shape / sensor dims / binning / ROI origin & extent / orientation /
  CFA phase (the exact geometry contract, reused via
  :func:`~zecalibrator.core.geometry.geometry_matches`),
* readout context — gain / offset / readout_mode / adc_mode.

Matching is **strict and conservative** (SCIENCE §6.4 "unknown != unknown; missing
required fields block automatic selection"): a revision is compatible with a run
only when every identity field is known on both sides and exactly equal. Applying
persistent sensor-site knowledge across an unknown or mismatched detector /
geometry / readout context is never allowed; an insufficient or mismatched
identity therefore degrades to ``NO_PROFILE`` (a benign ``CALIBRATION_ONLY``),
never a guessed application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from zecalibrator.core.descriptors import Acquisition, DetectorIdentity, LightConstraints
from zecalibrator.core.digests import canonical_json, sha256_hex
from zecalibrator.core.geometry import Geometry, geometry_matches

#: Identity reason codes (stable, for provenance/diagnostics).
MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
DETECTOR_MISMATCH = "DETECTOR_MISMATCH"
READOUT_MISMATCH = "READOUT_MISMATCH"
#: The literal string that denotes an unknown detector instance (mirrors
#: ``application.library.light_constraints_from_sensor_metadata``).
UNKNOWN_INSTANCE = "unknown"


@dataclass(frozen=True)
class ReadoutContext:
    """The readout-context projection of :class:`Acquisition` (exact facts).

    ``None`` means unknown — never a default, never matched against another
    unknown (SCIENCE §6.4).
    """

    gain: Optional[float] = None
    offset: Optional[float] = None
    readout_mode: Optional[str] = None
    adc_mode: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("gain", "offset"):
            v = getattr(self, name)
            if v is not None:
                f = float(v)
                if not math.isfinite(f):
                    raise ValueError(f"ReadoutContext.{name} must be finite, got {v!r}")
                object.__setattr__(self, name, f)

    def to_dict(self) -> dict:
        return {
            "gain": self.gain,
            "offset": self.offset,
            "readout_mode": self.readout_mode,
            "adc_mode": self.adc_mode,
        }


@dataclass(frozen=True)
class SensorIdentity:
    """Immutable, exact sensor identity for BPM selection (reuses core objects)."""

    detector: DetectorIdentity
    geometry: Geometry
    readout: ReadoutContext

    def to_dict(self) -> dict:
        return {
            "detector": self.detector.to_dict(),
            "geometry": _geometry_to_dict(self.geometry),
            "readout": self.readout.to_dict(),
        }

    def canonical(self) -> str:
        """Return the canonical serialization (for identity digests)."""
        return canonical_json(self.to_dict())

    def identity_digest(self) -> str:
        """Return the SHA-256 of the canonical identity serialization."""
        return sha256_hex(self.canonical().encode("utf-8"))


def _geometry_to_dict(geo: Geometry) -> dict:
    def tup(v):
        return None if v is None else list(v)

    return {
        "shape": list(geo.shape),
        "sensor_dimensions": tup(geo.sensor_dimensions),
        "binning": tup(geo.binning),
        "roi_origin": tup(geo.roi_origin),
        "roi_extent": tup(geo.roi_extent),
        "orientation": geo.orientation,
        "cfa_phase": geo.cfa_phase,
    }


def _geometry_from_dict(d: dict) -> Geometry:
    def tup(v):
        return None if v is None else (int(v[0]), int(v[1]))

    return Geometry(
        shape=(int(d["shape"][0]), int(d["shape"][1])),
        sensor_dimensions=tup(d.get("sensor_dimensions")),
        binning=tup(d.get("binning")),
        roi_origin=tup(d.get("roi_origin")),
        roi_extent=tup(d.get("roi_extent")),
        orientation=d.get("orientation"),
        cfa_phase=d.get("cfa_phase"),
    )


def sensor_identity_from_dict(d: dict) -> SensorIdentity:
    """Reconstruct a :class:`SensorIdentity` from its canonical serialization."""
    det = d["detector"]
    return SensorIdentity(
        detector=DetectorIdentity(
            detector_instance_id=det["detector_instance_id"],
            detector_model=det.get("detector_model"),
            serial=det.get("serial"),
        ),
        geometry=_geometry_from_dict(d["geometry"]),
        readout=ReadoutContext(**d["readout"]),
    )


def readout_from_acquisition(acq: Acquisition) -> ReadoutContext:
    """Project the readout-context facts of an :class:`Acquisition` (no invention)."""
    return ReadoutContext(
        gain=acq.gain,
        offset=acq.offset,
        readout_mode=acq.readout_mode,
        adc_mode=acq.adc_mode,
    )


def sensor_identity_from_light_constraints(light: LightConstraints) -> SensorIdentity:
    """Build the BPM selection identity from the normative matching liaison.

    This is the *existing* identity (``LightConstraints``), projected onto the
    BPM-relevant facts — physical / geometry / binning / readout / CFA — with no
    simplified parallel identity.
    """
    return SensorIdentity(
        detector=light.detector,
        geometry=light.geometry,
        readout=readout_from_acquisition(light.acquisition),
    )


def _unknown_instance(v) -> bool:
    if v is None:
        return True
    return isinstance(v, str) and v.strip().lower() == UNKNOWN_INSTANCE


def _unknown(v) -> bool:
    return v is None


def _detector_reasons(run: DetectorIdentity, base: DetectorIdentity) -> list[str]:
    reasons: list[str] = []
    if _unknown_instance(run.detector_instance_id) or _unknown_instance(base.detector_instance_id):
        reasons.append(MISSING_REQUIRED_FIELD)
    elif run.detector_instance_id != base.detector_instance_id:
        reasons.append(DETECTOR_MISMATCH)

    if _unknown(run.detector_model) or _unknown(base.detector_model):
        reasons.append(MISSING_REQUIRED_FIELD)
    elif run.detector_model != base.detector_model:
        reasons.append(DETECTOR_MISMATCH)
    # serial is carried (and hashed) but is NOT a selection criterion, mirroring
    # the normative ``_detector_reasons`` (which never matches on serial).
    return reasons


def _readout_reasons(run: ReadoutContext, base: ReadoutContext) -> list[str]:
    reasons: list[str] = []
    for name in ("gain", "offset", "readout_mode", "adc_mode"):
        rv, bv = getattr(run, name), getattr(base, name)
        if _unknown(rv) or _unknown(bv):
            reasons.append(MISSING_REQUIRED_FIELD)
        elif rv != bv:
            reasons.append(READOUT_MISMATCH)
    return reasons


def identity_reasons(run: SensorIdentity, base: SensorIdentity) -> tuple[str, ...]:
    """Return the reason codes for every incompatibility (empty == compatible).

    Strict and conservative: unknown never matches unknown; any unknown required
    fact yields ``MISSING_REQUIRED_FIELD`` (SCIENCE §6.4). Geometry is delegated
    to the frozen exact contract :func:`zecalibrator.core.geometry.geometry_matches`.
    """
    reasons: list[str] = []
    reasons.extend(_detector_reasons(run.detector, base.detector))
    reasons.extend(geometry_matches(run.geometry, base.geometry))
    reasons.extend(_readout_reasons(run.readout, base.readout))
    return tuple(dict.fromkeys(reasons))


def identity_matches(run: SensorIdentity, base: SensorIdentity) -> bool:
    """Return ``True`` iff ``base`` is compatible with ``run`` (exact identity)."""
    return not identity_reasons(run, base)


__all__ = [
    "DETECTOR_MISMATCH",
    "MISSING_REQUIRED_FIELD",
    "READOUT_MISMATCH",
    "ReadoutContext",
    "SensorIdentity",
    "UNKNOWN_INSTANCE",
    "identity_matches",
    "identity_reasons",
    "readout_from_acquisition",
    "sensor_identity_from_dict",
    "sensor_identity_from_light_constraints",
]
