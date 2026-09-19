"""Public API v1 namespace — stable lazy facade over the frozen G3/G4 engine.

``API_VERSION`` is the frozen public API target, independent of the product
version, the science-contract version, the provenance schema, the library schema
and the matching-policy version (ARCHITECTURE §2).

``__all__`` is the explicit public-symbol contract. Anything not listed is
internal and may change. **Cold import and ``get_api_info()`` are cheap**: they
import no NumPy/Astropy, no Qt/ZeAlfie/ZSSS/CuPy, no SQLite and no filesystem/
network activity. Scientific value objects/functions are loaded lazily (PEP 562
``__getattr__``) on first use, which may then import the scientific modules.
"""

from __future__ import annotations

import importlib

from zecalibrator import _version

from ._meta import ApiInfo, CAPABILITIES, PROVENANCE_SCHEMA

API_VERSION = "1.0"


def get_api_info() -> ApiInfo:
    """Return the static API info (cheap; no file/network/Qt/GPU side effects)."""
    return ApiInfo(
        api_version=API_VERSION,
        product_version=_version.__version__,
        capabilities=CAPABILITIES,
    )


_LAZY = {
    # errors
    "ZeCalibratorError": ("errors", "ZeCalibratorError"),
    "InvalidRequestError": ("errors", "InvalidRequestError"),
    "GeometryMismatchError": ("errors", "GeometryMismatchError"),
    "PrecisionRefusalError": ("errors", "PrecisionRefusalError"),
    "DecodeError": ("errors", "DecodeError"),
    "LibraryError": ("errors", "LibraryError"),
    "UnsupportedSchemaError": ("errors", "UnsupportedSchemaError"),
    "SourceError": ("errors", "SourceError"),
    "LibraryClosedError": ("errors", "LibraryClosedError"),
    # models
    "BATCH_MANIFEST_SCHEMA": ("models", "BATCH_MANIFEST_SCHEMA"),
    "BatchManifestError": ("batch", "BatchManifestError"),
    "Acquisition": ("models", "Acquisition"),
    "AcquisitionProfileEvidence": ("models", "AcquisitionProfileEvidence"),
    "ArrayFrameSource": ("models", "ArrayFrameSource"),
    "ArrayInputIdentity": ("models", "ArrayInputIdentity"),
    "BatchItem": ("models", "BatchItem"),
    "BatchManifest": ("models", "BatchManifest"),
    "BatchOptions": ("models", "BatchOptions"),
    "BatchOutputRecord": ("models", "BatchOutputRecord"),
    "CalibrationPlan": ("models", "CalibrationPlan"),
    "CalibrationRequest": ("models", "CalibrationRequest"),
    "CalibrationResult": ("models", "CalibrationResult"),
    "CancellationToken": ("models", "CancellationToken"),
    "Candidate": ("models", "Candidate"),
    "CardRecord": ("models", "CardRecord"),
    "ConflictDiagnostic": ("models", "ConflictDiagnostic"),
    "CountSummary": ("models", "CountSummary"),
    "DecisionEnvelope": ("models", "DecisionEnvelope"),
    "DQ_STATES": ("models", "DQ_STATES"),
    "DqState": ("models", "DqState"),
    "DescriptorSnapshot": ("models", "DescriptorSnapshot"),
    "DetectorIdentity": ("models", "DetectorIdentity"),
    "EVIDENCE_ORIGIN_TYPES": ("models", "EVIDENCE_ORIGIN_TYPES"),
    "EvidenceFact": ("models", "EvidenceFact"),
    "ExecutionOptions": ("models", "ExecutionOptions"),
    "FilesystemSource": ("models", "FilesystemSource"),
    "FitsFileLocator": ("models", "FitsFileLocator"),
    "FitsFrameSource": ("models", "FitsFrameSource"),
    "FitsInputIdentity": ("models", "FitsInputIdentity"),
    "FlatQualityPolicy": ("models", "FlatQualityPolicy"),
    "FrameInspection": ("models", "FrameInspection"),
    "FrameQuality": ("models", "FrameQuality"),
    "FrameSource": ("models", "FrameSource"),
    "Geometry": ("models", "Geometry"),
    "ImportDeclaration": ("models", "ImportDeclaration"),
    "IndexLibraryResult": ("models", "IndexLibraryResult"),
    "InMemorySource": ("models", "InMemorySource"),
    "InputIdentity": ("models", "InputIdentity"),
    "InspectResult": ("models", "InspectResult"),
    "LibraryHandle": ("models", "LibraryHandle"),
    "LibrarySnapshot": ("models", "LibrarySnapshot"),
    "LibrarySpec": ("models", "LibrarySpec"),
    "LightConstraints": ("models", "LightConstraints"),
    "LightEvidence": ("models", "LightEvidence"),
    "MANAGED_LEDGER_SCHEMA": ("models", "MANAGED_LEDGER_SCHEMA"),
    "ManagedMasterRecord": ("models", "ManagedMasterRecord"),
    "MaskPayloadLocator": ("models", "MaskPayloadLocator"),
    "MasterBinding": ("models", "MasterBinding"),
    "MasterDescriptor": ("models", "MasterDescriptor"),
    "MatchPolicy": ("models", "MatchPolicy"),
    "MatchResult": ("models", "MatchResult"),
    "MasterImportSpec": ("models", "MasterImportSpec"),
    "NormalizationProvenance": ("models", "NormalizationProvenance"),
    "NormalizationScalars": ("models", "NormalizationScalars"),
    "OpenLibraryResult": ("models", "OpenLibraryResult"),
    "OperationCancelled": ("models", "OperationCancelled"),
    "OpticalIdentity": ("models", "OpticalIdentity"),
    "PolicyParameters": ("models", "PolicyParameters"),
    "PrecisionInfo": ("models", "PrecisionInfo"),
    "ProcessingProvenance": ("models", "ProcessingProvenance"),
    "ProgressEvent": ("models", "ProgressEvent"),
    "ProgressObserver": ("models", "ProgressObserver"),
    "ProvenanceRecord": ("models", "ProvenanceRecord"),
    "Reason": ("models", "Reason"),
    "RejectionRecord": ("models", "RejectionRecord"),
    "ResolveResult": ("models", "ResolveResult"),
    "RoiExtentEvidence": ("models", "RoiExtentEvidence"),
    "SensorMetadata": ("models", "SensorMetadata"),
    "Tolerance": ("models", "Tolerance"),
    "ValidationResult": ("models", "ValidationResult"),
    "ValidityEvidence": ("models", "ValidityEvidence"),
    "VersionSet": ("models", "VersionSet"),
    "build_sensor_metadata": ("models", "build_sensor_metadata"),
    "default_match_policy": ("models", "default_match_policy"),
    "light_constraints_from_sensor_metadata": ("models", "light_constraints_from_sensor_metadata"),
    # functions
    "calibrate_batch": ("batch", "calibrate_batch"),
    "calibrate_frame": ("calibration", "calibrate_frame"),
    "index_library": ("batch", "index_library"),
    "inspect_frame": ("frames", "inspect_frame"),
    "open_library": ("library", "open_library"),
    "resolve_calibration": ("matching", "resolve_calibration"),
    "validate_binding": ("matching", "validate_binding"),
    "validate_plan": ("matching", "validate_plan"),
    "calibrate_frame": ("calibration", "calibrate_frame"),
    # managed (P7-M3B)
    "load_managed_ledger": ("managed", "load_managed_ledger"),
    "save_managed_ledger": ("managed", "save_managed_ledger"),
    "build_master_import_spec": ("managed", "build_master_import_spec"),
    "managed_fingerprint": ("managed", "managed_fingerprint"),
    "build_managed_library": ("managed", "build_managed_library"),
    "content_identity": ("managed", "content_identity"),
    "managed_ledger_path": ("managed", "managed_ledger_path"),
    "ManagedLedgerLoad": ("managed", "ManagedLedgerLoad"),
    "ManagedLibraryResult": ("managed", "ManagedLibraryResult"),
    "STATE_MALFORMED": ("managed", "STATE_MALFORMED"),
    "STATE_MISSING": ("managed", "STATE_MISSING"),
    "STATE_OK": ("managed", "STATE_OK"),
    "STATE_UNSUPPORTED": ("managed", "STATE_UNSUPPORTED"),
}


def __getattr__(name):
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = entry
    module = importlib.import_module(f"zecalibrator.api.v1.{module_name}")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "API_VERSION",
    "BATCH_MANIFEST_SCHEMA",
    "BatchManifestError",
    "CAPABILITIES",
    "PROVENANCE_SCHEMA",
    "Acquisition",
    "AcquisitionProfileEvidence",
    "ApiInfo",
    "ArrayFrameSource",
    "ArrayInputIdentity",
    "BatchItem",
    "BatchManifest",
    "BatchOptions",
    "BatchOutputRecord",
    "CalibrationPlan",
    "CalibrationRequest",
    "CalibrationResult",
    "CancellationToken",
    "Candidate",
    "CardRecord",
    "ConflictDiagnostic",
    "CountSummary",
    "DecisionEnvelope",
    "DecodeError",
    "DQ_STATES",
    "DqState",
    "DescriptorSnapshot",
    "DetectorIdentity",
    "EVIDENCE_ORIGIN_TYPES",
    "EvidenceFact",
    "ExecutionOptions",
    "FilesystemSource",
    "FitsFileLocator",
    "FitsFrameSource",
    "FitsInputIdentity",
    "FlatQualityPolicy",
    "FrameInspection",
    "FrameQuality",
    "FrameSource",
    "Geometry",
    "GeometryMismatchError",
    "ImportDeclaration",
    "IndexLibraryResult",
    "InMemorySource",
    "InputIdentity",
    "InspectResult",
    "InvalidRequestError",
    "LibraryClosedError",
    "LibraryError",
    "LibraryHandle",
    "LibrarySnapshot",
    "LibrarySpec",
    "LightConstraints",
    "LightEvidence",
    "MANAGED_LEDGER_SCHEMA",
    "ManagedMasterRecord",
    "MaskPayloadLocator",
    "MasterBinding",
    "MasterDescriptor",
    "MatchPolicy",
    "MatchResult",
    "MasterImportSpec",
    "NormalizationProvenance",
    "NormalizationScalars",
    "OpenLibraryResult",
    "OperationCancelled",
    "OpticalIdentity",
    "PolicyParameters",
    "PrecisionInfo",
    "PrecisionRefusalError",
    "ProcessingProvenance",
    "ProgressEvent",
    "ProgressObserver",
    "ProvenanceRecord",
    "Reason",
    "RejectionRecord",
    "ResolveResult",
    "RoiExtentEvidence",
    "SensorMetadata",
    "SourceError",
    "Tolerance",
    "UnsupportedSchemaError",
    "ValidationResult",
    "ValidityEvidence",
    "VersionSet",
    "ZeCalibratorError",
    "build_sensor_metadata",
    "build_master_import_spec",
    "build_managed_library",
    "calibrate_batch",
    "calibrate_frame",
    "content_identity",
    "default_match_policy",
    "get_api_info",
    "index_library",
    "inspect_frame",
    "light_constraints_from_sensor_metadata",
    "load_managed_ledger",
    "managed_fingerprint",
    "managed_ledger_path",
    "ManagedLedgerLoad",
    "ManagedLibraryResult",
    "open_library",
    "resolve_calibration",
    "save_managed_ledger",
    "STATE_MALFORMED",
    "STATE_MISSING",
    "STATE_OK",
    "STATE_UNSUPPORTED",
    "validate_binding",
    "validate_plan",
]
