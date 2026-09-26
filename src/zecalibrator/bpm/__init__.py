"""Bad Pixel Database (BPM) product model — model + store + lookup + fallbacks.

This package delivers the *product model* of the Bad Pixel Database and the
application lot (P4-A2) that links it to the images: the sensor identity/liaison,
the store of immutable revisions, the deterministic lookup, the typed
errors/returns and benign fallbacks, the versioned internal reconstruction
operator, the run-wide :class:`~zecalibrator.bpm.preparation.PreparationPlan`
(preflight → freeze → execute) and the separate
:class:`~zecalibrator.bpm.prepared_result.PreparedCalibrationResult`. It is
headless (no Qt), imports no ``research/*``, and exposes no public ``api/v1``
surface.

Only the vocabulary promoted into product terms lives here; the retained
contracts are versioned as ``zecalibrator.bpm.v1`` and tested independently.
"""

from __future__ import annotations

from .errors import (
    BpmBaseCorrupted,
    BpmBaseIncompatible,
    BpmBaseInvalid,
    BpmError,
    BpmWriteError,
)
from .identity import (
    ReadoutContext,
    SensorIdentity,
    identity_matches,
    identity_reasons,
    readout_from_acquisition,
    sensor_identity_from_light_constraints,
)
from .lookup import BpmResolution, ProvenanceNote, select_revision
from .preparation import (
    OUTCOME_PREPARED,
    PREPARATION_SCHEMA_VERSION,
    RC_DONORS_UNAVAILABLE_RUN_WIDE,
    RUN_WIDE_ABSTAIN,
    RUN_WIDE_ELIGIBLE,
    CalibratedFrame,
    DuplicateSiteError,
    EmptyFramesError,
    PlanFrozenError,
    PlanInvalidError,
    PlanNotFrozenError,
    PreparationError,
    PreparationOutcome,
    PreparationPlan,
    SitePlanEntry,
    UnsupportedCfaError,
    apply_preparation,
    derive_run_wide,
    execute,
    preflight,
)
from .prepared_result import PreparedCalibrationResult
from .reconstruction import (
    DEFAULT_OPERATOR,
    ESTIMATOR_MEDIAN,
    NO_VALUE,
    OPERATOR_SCHEMA_VERSION,
    PRODUCT_OPERATOR_ID,
    ReconstructionOperator,
    donor_values,
    donors_available,
    reconstruct_site_pixel,
)
from .revision import (
    Revision,
    SiteRecord,
    action_eligible_sites,
    make_revision,
    promote_revision,
    revision_digest,
)
from .settings import (
    BpmSettings,
    default_bad_pixel_database_root,
    default_settings,
    load_settings,
    resolve_bad_pixel_database_root,
    save_settings,
)
from .store import (
    BadPixelDatabase,
    create_bad_pixel_database,
    load_bad_pixel_database,
    resolve_bad_pixel_database,
)
from .vocabulary import (
    ACTION_ELIGIBLE,
    ACTION_STATE_LADDER,
    BPM_SCHEMA_VERSION,
    KNOWLEDGE_STATES,
    OUTCOME_BASE_ERROR,
    OUTCOME_CALIBRATION_ONLY,
    OUTCOME_SELECTED,
    REASON_BASE_CORRUPTED,
    REASON_BASE_INCOMPATIBLE,
    REASON_BASE_INVALID,
    REASON_NO_BASE,
    REASON_NO_PROFILE,
    REASON_UNQUALIFIED_PROFILE,
    REVISION_STATES,
)

__all__ = [
    "ACTION_ELIGIBLE",
    "ACTION_STATE_LADDER",
    "BPM_SCHEMA_VERSION",
    "BadPixelDatabase",
    "BpmBaseCorrupted",
    "BpmBaseIncompatible",
    "BpmBaseInvalid",
    "BpmError",
    "BpmResolution",
    "BpmSettings",
    "BpmWriteError",
    "CalibratedFrame",
    "DEFAULT_OPERATOR",
    "DuplicateSiteError",
    "EmptyFramesError",
    "ESTIMATOR_MEDIAN",
    "KNOWLEDGE_STATES",
    "NO_VALUE",
    "OPERATOR_SCHEMA_VERSION",
    "OUTCOME_BASE_ERROR",
    "OUTCOME_CALIBRATION_ONLY",
    "OUTCOME_PREPARED",
    "OUTCOME_SELECTED",
    "PREPARATION_SCHEMA_VERSION",
    "PRODUCT_OPERATOR_ID",
    "PlanFrozenError",
    "PlanInvalidError",
    "PlanNotFrozenError",
    "PreparedCalibrationResult",
    "PreparationError",
    "PreparationOutcome",
    "PreparationPlan",
    "ProvenanceNote",
    "REASON_BASE_CORRUPTED",
    "RC_DONORS_UNAVAILABLE_RUN_WIDE",
    "RUN_WIDE_ABSTAIN",
    "RUN_WIDE_ELIGIBLE",
    "ReconstructionOperator",
    "SitePlanEntry",
    "UnsupportedCfaError",
    "apply_preparation",
    "derive_run_wide",
    "donor_values",
    "donors_available",
    "execute",
    "preflight",
    "reconstruct_site_pixel",
    "REASON_BASE_INCOMPATIBLE",
    "REASON_BASE_INVALID",
    "REASON_NO_BASE",
    "REASON_NO_PROFILE",
    "REASON_UNQUALIFIED_PROFILE",
    "REVISION_STATES",
    "ReadoutContext",
    "Revision",
    "SensorIdentity",
    "SiteRecord",
    "action_eligible_sites",
    "create_bad_pixel_database",
    "default_bad_pixel_database_root",
    "default_settings",
    "identity_matches",
    "identity_reasons",
    "load_bad_pixel_database",
    "load_settings",
    "make_revision",
    "promote_revision",
    "readout_from_acquisition",
    "resolve_bad_pixel_database",
    "resolve_bad_pixel_database_root",
    "revision_digest",
    "save_settings",
    "select_revision",
    "sensor_identity_from_light_constraints",
]
