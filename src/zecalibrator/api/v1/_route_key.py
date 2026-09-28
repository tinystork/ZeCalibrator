"""C26 — canonical, conservative light route-class key (header-only).

The route key groups lights by the facts the matcher actually consumes for
compatibility + ranking.  It is the single authoritative class key; a consumer
(ZSSS) must never recompute an approximate key of its own.

Safety rule (ratified): **over-grouping is dangerous; under-grouping is only
costly.**  The key is therefore *conservative and exact*: it includes every fact
the matcher consumes, with **no tolerance** — two lights differing on any
relevant fact (``CCD-TEMP`` -10.0 vs -9.9, ``roi_origin``, ``SET-TEMP``, filter,
or the civil day) produce **different keys**.  Unknown/conflicting facts are
never grouped with known ones.
"""

from __future__ import annotations

from zecalibrator.core.digests import canonical_json
from zecalibrator.core.selection import light_acquisition_date

# The matcher functions that consume each fact (documented source, not a guess):
#   _geometry_reasons           geometry.* (necessary / disambiguator tiers)
#   _detector_reasons           detector.*
#   _acquisition_reasons        acquisition.gain / offset / readout_mode / adc_mode
#   _temperature_reason / _standard_temperature_reasons   temperature_setpoint_c / temperature_c
#   _exposure_reason / _bias_exposure_reason              exposure_s / bias_exposure_max_s
#   _string_reason (FILTER)     optical.filter
#   _disambiguator_reasons      optical.optical_train_id
#   _flat_key / _additive_key   DATE-OBS civil day (ranking)


def light_route_key(light) -> str:
    """Return the canonical route-class key for ``light`` (deterministic string).

    Header-only: derived from ``light`` (a ``LightConstraints`` already built
    from header facts, no pixel decode) plus the ``DATE-OBS`` **civil day**.  The
    key is conservative and exact (no tolerance): lights that the matcher can
    distinguish must produce different keys.

    The ``plan_id`` returned by ``resolve_light`` remains the canonical identity
    **after** resolution; this key is for **preflight grouping** only.
    """
    a = light.acquisition
    g = light.geometry
    d = light.detector
    o = light.optical
    civil = light_acquisition_date(light)
    facts = {
        "geometry": {
            "shape": g.shape,
            "binning": g.binning,
            "sensor_dimensions": g.sensor_dimensions,
            "orientation": g.orientation,
            "roi_origin": g.roi_origin,
            "roi_extent": g.roi_extent,
            "cfa_phase": g.cfa_phase,
        },
        "detector": {
            "detector_instance_id": d.detector_instance_id,
            "detector_model": d.detector_model,
        },
        "acquisition": {
            "gain": a.gain,
            "offset": a.offset,
            "readout_mode": a.readout_mode,
            "adc_mode": a.adc_mode,
            "temperature_setpoint_c": a.temperature_setpoint_c,
            "temperature_c": a.temperature_c,
            "exposure_s": a.exposure_s,
            "bias_exposure_max_s": a.bias_exposure_max_s,
        },
        "optical": {
            "filter": o.filter,
            "optical_train_id": o.optical_train_id,
        },
        "date": {
            "civil_day": civil.date().isoformat() if civil is not None else None,
        },
    }
    return canonical_json(facts)


__all__ = ["light_route_key"]
