"""Phase4-only fixture helpers for library/matching/plan contract tests.

Builds synthetic SYNTH-BASE-1-style descriptors, candidates, light constraints
and policies. All facts are explicit synthetic fixture values, never real
detector defaults. SOURCE-LABELLED synthetic qualification facts (bias range
[0, 0.01], per-plane valid/total populations, illumination/exposure-quality) are
injected here; every supplied fact is documented.
"""

from __future__ import annotations

from zecalibrator.core.descriptors import (
    Acquisition,
    AcquisitionProfileEvidence,
    DescriptorSnapshot,
    DetectorIdentity,
    LightConstraints,
    MasterDescriptor,
    OpticalIdentity,
    ProcessingProvenance,
    ValidityEvidence,
)
from zecalibrator.core.geometry import Geometry
from zecalibrator.core.plans import (
    Candidate,
    CalibrationRequest,
    default_match_policy,
)

# SOURCE-LABELLED synthetic facts (implied by the frozen case notes; not real
# detector defaults).
SYNTHETIC_BIAS_MAX_S = 0.01
SYNTHETIC_ILLUMINATION = "flat_field"


def geo(
    shape=(4, 4),
    sensor_dimensions=(4, 4),
    binning=(1, 1),
    roi_origin=(0, 0),
    roi_extent=(4, 4),
    orientation="identity",
    cfa_phase="mono",
):
    return Geometry(
        shape=shape,
        sensor_dimensions=sensor_dimensions,
        binning=binning,
        roi_origin=roi_origin,
        roi_extent=roi_extent,
        orientation=orientation,
        cfa_phase=cfa_phase,
    )


def detector(instance="SYNTH-DET-0001", model="SYNTH-CFA", serial=None):
    return DetectorIdentity(detector_instance_id=instance, detector_model=model, serial=serial)


def acquisition(
    gain=100,
    offset=50,
    readout_mode="MODE_A",
    adc_mode="MODE_16",
    temperature_c=20.0,
    exposure_s=10.0,
    saturation_limit_adu=60000.0,
    saturation_evidence="qualified",
    bias_exposure_max_s=SYNTHETIC_BIAS_MAX_S,
    short_flat_profile=False,
):
    return Acquisition(
        gain=gain,
        offset=offset,
        readout_mode=readout_mode,
        adc_mode=adc_mode,
        temperature_c=temperature_c,
        exposure_s=exposure_s,
        saturation_limit_adu=saturation_limit_adu,
        saturation_evidence=saturation_evidence,
        bias_exposure_max_s=bias_exposure_max_s,
        short_flat_profile=short_flat_profile,
    )


def profile(bias_exposure_max_s=SYNTHETIC_BIAS_MAX_S, short_flat_profile=False, identity="SYNTH-BASE-1"):
    return AcquisitionProfileEvidence(
        source="synthetic_fixture",
        identity=identity,
        version="1.0",
        bias_exposure_max_s=bias_exposure_max_s,
        short_flat_profile=short_flat_profile,
    )


def flat_validity(phase="mono", valid=95, total=100, saturation_known=True):
    if phase == "mono":
        planes = {"mono": valid}
        totals = {"mono": total}
    else:
        planes = {"G1": valid, "R": valid, "B": valid, "G2": valid}
        totals = {"G1": total, "R": total, "B": total, "G2": total}
    return ValidityEvidence(
        saturation_limit_known=saturation_known,
        valid_normalization_count=planes,
        total_normalization_count=totals,
        quality_policy_state="qualified",
        illumination=SYNTHETIC_ILLUMINATION,
        exposure_quality="qualified",
    )


def light(**kw):
    defaults = dict(
        geometry=geo(),
        detector=detector(),
        acquisition=acquisition(),
        optical=OpticalIdentity(filter="NONE", optical_train_id="SYNTH-TRAIN-1"),
        raw_domain_declaration="raw",
    )
    defaults.update(kw)
    return LightConstraints(**defaults)


def descriptor(
    master_type="dark",
    bias_state="included",
    exposure_s=10.0,
    geometry=None,
    detector_obj=None,
    acquisition_obj=None,
    flat_form=None,
    normalization_algorithm=None,
    normalization_scalars=None,
    optical_train_id=None,
    filter=None,
    content_sha256=None,
    mask_identity=None,
    processing=None,
    validity=None,
    pixel_domain="sensor_adu",
    physical_units="ADU",
    acquisition_profile=None,
    size_bytes=16,
):
    geo_obj = geometry or geo()
    det_obj = detector_obj or detector()
    acq_obj = acquisition_obj or acquisition(exposure_s=exposure_s)
    if content_sha256 is None:
        content_sha256 = "0" * 64 if master_type != "flat" else "f" * 64
    if mask_identity is None:
        mask_identity = "1" * 64
    if processing is None:
        processing = ProcessingProvenance(
            source="synthetic_fixture",
            acquisition_profile=acquisition_profile,
        )
    if validity is None and master_type == "flat":
        validity = flat_validity(phase=geo_obj.cfa_phase or "mono")
    elif validity is None:
        validity = ValidityEvidence(saturation_limit_known=True)
    return MasterDescriptor(
        master_type=master_type,
        pixel_domain=pixel_domain,
        physical_units=physical_units,
        bias_state=bias_state,
        geometry=geo_obj,
        detector=det_obj,
        acquisition=acq_obj,
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        hdu=0,
        mask_identity=mask_identity,
        processing_provenance=processing,
        validity_evidence=validity,
        flat_form=flat_form,
        normalization_algorithm=normalization_algorithm,
        normalization_scalars=normalization_scalars,
        optical_train_id=optical_train_id,
        filter=filter,
    )


def candidate(candidate_id, desc, locator_path=None, mask_path=None):
    from zecalibrator.core.plans import FitsFileLocator, MaskPayloadLocator

    locators = (FitsFileLocator(path=locator_path, hdu=desc.hdu),) if locator_path else ()
    mask_loc = MaskPayloadLocator(path=mask_path) if mask_path else None
    return Candidate(
        candidate_id=candidate_id,
        descriptor=desc,
        descriptor_snapshot=DescriptorSnapshot(desc),
        locators=locators,
        mask_locator=mask_loc,
    )


def pool(**roles):
    return {k: list(v) for k, v in roles.items()}


def request(additive_mode="dark_incl_bias", flat_mode="none"):
    return CalibrationRequest(additive_mode=additive_mode, flat_mode=flat_mode)


def policy():
    return default_match_policy()


def make_binding(desc, *, locator_path="/d.fits", mask_path="/d.mask"):
    from zecalibrator.core.plans import FitsFileLocator, MaskPayloadLocator, MasterBinding

    return MasterBinding(
        descriptor_id=desc.descriptor_id,
        descriptor_snapshot=DescriptorSnapshot(desc),
        content_sha256=desc.content_sha256,
        size_bytes=desc.size_bytes,
        hdu=desc.hdu,
        mask_identity=desc.mask_identity,
        locators=(FitsFileLocator(path=locator_path, hdu=desc.hdu),),
        mask_locator=MaskPayloadLocator(path=mask_path),
    )
