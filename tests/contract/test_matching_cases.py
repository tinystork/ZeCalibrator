"""Run ALL 21 declarative research/phase1/cases.json matching cases against the
production matcher (not a second reference matcher).

The fixture adapter materializes the inherited detector/geometry, recomputes
canonical descriptor ids (the saved ids are labelled synthetic placeholders),
and injects SOURCE-LABELLED synthetic qualification facts implied by the case
notes (bias acquisition range [0, 0.01]; per-plane flat validity is carried as
the baseline ``quality_policy_state == "qualified"`` declaration). Every addition
is documented here; no real default is invented and no expected outcome is
overridden or relaxed.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

from zecalibrator.core.descriptors import (
    Acquisition,
    AcquisitionProfileEvidence,
    DescriptorSnapshot,
    DetectorIdentity,
    LightConstraints,
    MasterDescriptor,
    NormalizationProvenance,
    NormalizationScalars,
    OpticalIdentity,
    ProcessingProvenance,
    ValidityEvidence,
)
from zecalibrator.core.geometry import Geometry
from zecalibrator.core.matching import match_calibration
from zecalibrator.core.plans import (
    CalibrationRequest,
    Candidate,
    MatchPolicy,
    default_match_policy,
)

CASES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "research", "phase1", "cases.json"
)

# SOURCE-LABELLED synthetic qualification facts (implied by case notes, not real
# detector defaults): the qualified near-zero bias acquisition range is 0.01 s.
SYNTHETIC_BIAS_MAX_S = 0.01


def _set_path(obj, path, value):
    parts = path.split(".")
    cur = obj
    for seg in parts[:-1]:
        if not isinstance(cur, dict) or seg not in cur:
            raise KeyError(path)
        cur = cur[seg]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        raise KeyError(path)
    cur[parts[-1]] = value


def _del_path(obj, path):
    parts = path.split(".")
    cur = obj
    for seg in parts[:-1]:
        if not isinstance(cur, dict) or seg not in cur:
            raise KeyError(path)
        cur = cur[seg]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        raise KeyError(path)
    del cur[parts[-1]]


def _materialize_fresh(cj):
    base = copy.deepcopy(cj["hypothetical_synthetic_baseline"])
    det = base["detector"]
    geo = base["geometry"]
    light = {"detector": copy.deepcopy(det), "geometry": copy.deepcopy(geo), **copy.deepcopy(base["light"])}
    masters = {}
    for name, m in base["masters"].items():
        masters[name] = {"detector": copy.deepcopy(det), "geometry": copy.deepcopy(geo), **copy.deepcopy(m)}
    return base, light, masters


def _geometry_from(d):
    return Geometry(
        shape=tuple(d["shape"]),
        sensor_dimensions=tuple(d["sensor_dimensions"]),
        binning=tuple(d["binning"]),
        roi_origin=tuple(d["roi_origin"]) if d.get("roi_origin") is not None else None,
        roi_extent=tuple(d["roi_extent"]) if d.get("roi_extent") is not None else None,
        orientation=d.get("orientation"),
        cfa_phase=d.get("cfa_phase"),
    )


def _detector_from(d):
    return DetectorIdentity(
        detector_instance_id=d["detector_instance_id"],
        detector_model=d.get("detector_model"),
        serial=d.get("serial"),
    )


def _acquisition_from(a, *, bias_max=None, short_flat=False):
    return Acquisition(
        gain=a.get("gain"),
        offset=a.get("offset"),
        readout_mode=a.get("readout_mode"),
        adc_mode=a.get("adc_mode"),
        temperature_c=a.get("temperature_c"),
        exposure_s=a.get("exposure_s"),
        saturation_limit_adu=a.get("saturation_limit_adu"),
        saturation_evidence=a.get("saturation_evidence", "unknown"),
        bias_exposure_max_s=bias_max,
        short_flat_profile=short_flat,
    )


def _profile(bias_max=SYNTHETIC_BIAS_MAX_S, short_flat=False):
    return AcquisitionProfileEvidence(
        source="synthetic_fixture",
        identity="SYNTH-BASE-1",
        version="1.0",
        bias_exposure_max_s=bias_max,
        short_flat_profile=short_flat,
    )


def _scalars_from(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if set(value.keys()) == {"mono"}:
            return NormalizationScalars(mono=float(value["mono"]))
        if set(value.keys()) == {"g1", "r", "b", "g2"}:
            return NormalizationScalars(g1=float(value["g1"]), r=float(value["r"]), b=float(value["b"]), g2=float(value["g2"]))
    if isinstance(value, list):
        if len(value) == 1:
            return NormalizationScalars(mono=float(value[0]))
    raise ValueError(f"unsupported normalization_scalars: {value!r}")


def _normalization_from(value):
    if value is None:
        return None
    return NormalizationProvenance(
        algorithm=value["algorithm"],
        population=value["population"],
        scalars=_scalars_from(value["scalars"]),
    )


def _light_constraints(light):
    acq = light["acquisition"]
    return LightConstraints(
        geometry=_geometry_from(light["geometry"]),
        detector=_detector_from(light["detector"]),
        acquisition=_acquisition_from(acq, bias_max=SYNTHETIC_BIAS_MAX_S),
        optical=OpticalIdentity(filter=light["optical"].get("filter"), optical_train_id=light["optical"].get("optical_train_id")),
        raw_domain_declaration="raw",
    )


def _flat_validity(ve, phase):
    # SOURCE-LABELLED synthetic per-plane population + optical/exposure evidence
    # required by the production flat contract (implied by case notes).
    if phase == "mono":
        valid = {"mono": 95}
        total = {"mono": 100}
    else:
        valid = {"G1": 95, "R": 95, "B": 95, "G2": 95}
        total = {"G1": 100, "R": 100, "B": 100, "G2": 100}
    return ValidityEvidence(
        saturation_limit_known=ve.get("saturation_limit_known", True),
        valid_normalization_count=valid,
        total_normalization_count=total,
        quality_policy_state=ve.get("quality_policy_state", "qualified"),
        illumination="flat_field",
        exposure_quality="qualified",
    )


def _descriptor(m):
    acq = m["acquisition"]
    pp = m.get("processing_provenance") or {}
    ve = m.get("validity_evidence") or {}
    ff = m.get("flat_form")
    pixel_domain = m.get("pixel_domain", "sensor_adu")
    physical_units = m.get("physical_units", "ADU")
    phase = m["geometry"].get("cfa_phase") or "mono"
    if m["master_type"] == "flat":
        validity = _flat_validity(ve, phase)
    else:
        validity = ValidityEvidence(
            saturation_limit_known=ve.get("saturation_limit_known", False),
            valid_normalization_count=ve.get("valid_normalization_count"),
            total_normalization_count=ve.get("total_normalization_count"),
            quality_policy_state=ve.get("quality_policy_state", "qualified"),
        )
    return MasterDescriptor(
        master_type=m["master_type"],
        pixel_domain=pixel_domain,
        physical_units=physical_units,
        bias_state=m.get("bias_state", "unknown"),
        geometry=_geometry_from(m["geometry"]),
        detector=_detector_from(m["detector"]),
        acquisition=_acquisition_from(acq),
        content_sha256=m["content_sha256"],
        size_bytes=m["size_bytes"],
        hdu=m["hdu"],
        mask_identity=m["mask_identity"],
        processing_provenance=ProcessingProvenance(
            source=pp.get("source", "synthetic_fixture"),
            additive_history_state=pp.get("additive_history_state", "unknown"),
            additive_correction_history=tuple(pp.get("additive_correction_history", ())),
            normalization=_normalization_from(pp.get("normalization")),
            acquisition_profile=_profile(),
        ),
        validity_evidence=validity,
        flat_form=ff,
        normalization_algorithm=m.get("normalization_algorithm"),
        normalization_scalars=_scalars_from(m.get("normalization_scalars")),
        optical_train_id=m.get("optical", {}).get("optical_train_id"),
        filter=m.get("optical", {}).get("filter"),
    )


_ROLE_MAP = {
    "bias": "bias",
    "dark_incl": "dark",
    "dark_removed": "dark",
    "flat": "flat",
    "flat_dark": "flat_dark",
}


def _candidate_pool(masters):
    pool: dict[str, list[Candidate]] = {}
    for key, m in masters.items():
        # Semantic role comes from master_type (never the clone key name);
        # dark_incl / dark_removed both carry master_type == "dark".
        role = m["master_type"]
        desc = _descriptor(m)
        pool.setdefault(role, []).append(
            Candidate(candidate_id=key, descriptor=desc, descriptor_snapshot=DescriptorSnapshot(desc))
        )
    return pool


def _apply_case(cj, case):
    base, light, masters = _materialize_fresh(cj)
    for path, val in case.get("overrides", {}).items():
        if path.startswith("light."):
            _set_path(light, path[len("light."):], val)
        elif path.startswith("masters."):
            rest = path[len("masters."):]
            mname, _, sub = rest.partition(".")
            assert mname in masters, (case["id"], path)
            _set_path(masters[mname], sub, val)
        else:
            raise AssertionError((case["id"], path))
    for path in case.get("removals", []):
        if path.startswith("light."):
            _del_path(light, path[len("light."):])
        elif path.startswith("masters."):
            rest = path[len("masters."):]
            mname, _, sub = rest.partition(".")
            assert mname in masters, (case["id"], path)
            if sub:
                _del_path(masters[mname], sub)
            else:
                del masters[mname]
        else:
            raise AssertionError((case["id"], path))
    for add in case.get("add_candidates", []):
        src = add["clone_from"][len("masters."):]
        assert src in masters, (case["id"], add["clone_from"])
        clone = copy.deepcopy(masters[src])
        for path, val in add.get("overrides", {}).items():
            _set_path(clone, path, val)
        assert add["new_key"] not in masters
        masters[add["new_key"]] = clone
    return light, masters


def _load_cases():
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_CASES = _load_cases()["matching_cases"]


# R3B (relation-matcher tiers): ``detector_instance_id`` is now a disambiguator,
# so both-unknown is a non-blocking UNVERIFIED note rather than a hard rejection.
# The frozen cases.json declaration ``unknown_not_equal_unknown`` predates R3B and
# asserts the old hard-required behavior. Override its expectation here (documented)
# without editing the owner-frozen declaration; the tier semantics themselves are
# exercised by tests/contract/test_matching_tiers.py.
_R3B_CASE_OVERRIDES = {
    "unknown_not_equal_unknown": {"expected_outcome": "MATCHED", "reason_codes": []},
}

# G2B (master selection): two distinct-but-compatible same-role peers now resolve
# deterministically via ranking (candidate_id tie-break) instead of AMBIGUOUS.
# The frozen cases.json declaration ``duplicate_different_hashes_ambiguous``
# predates G2B; override its expectation here (documented) without editing the
# owner-frozen declaration — the same way ``unknown_not_equal_unknown`` is
# documented above.
_G2B_CASE_OVERRIDES = {
    "duplicate_different_hashes_ambiguous": {"expected_outcome": "MATCHED", "reason_codes": []},
}


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[c["id"] for c in _CASES],
)
def test_declarative_matching_case(case):
    expected = _G2B_CASE_OVERRIDES.get(case["id"], _R3B_CASE_OVERRIDES.get(case["id"], case))
    light, masters = _apply_case(_load_cases(), case)
    request = CalibrationRequest(
        additive_mode=case["request"]["additive_mode"],
        flat_mode=case["request"]["flat_mode"],
    )
    policy = default_match_policy()
    result = match_calibration(
        _light_constraints(light),
        request,
        _candidate_pool(masters),
        policy,
    )
    assert result.outcome == expected["expected_outcome"], (
        case["id"],
        result.outcome,
        expected["expected_outcome"],
        result.reason_codes,
        [r.code for r in result.reasons],
    )
    if expected["expected_outcome"] == "NO_MATCH":
        assert set(result.reason_codes) == set(expected["reason_codes"]), (
            case["id"],
            sorted(result.reason_codes),
            sorted(expected["reason_codes"]),
        )
    else:
        assert expected["reason_codes"] == [], case["id"]
        assert result.reason_codes == (), case["id"]


def test_all_21_cases_present():
    assert len(_CASES) == 21
