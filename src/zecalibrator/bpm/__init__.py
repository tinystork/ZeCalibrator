"""Bad Pixel Database (BPM) product model — model + store + lookup + fallbacks.

This package delivers the *product model* of the Bad Pixel Database: the sensor
identity/liaison, the store of immutable revisions, the deterministic lookup, and
the typed errors/returns and benign fallbacks. It contains **no pixel
application, no ``PreparationPlan``, no mask, and no reconstruction** (that is
P4-A2). It is headless (no Qt), imports no ``research/*``, and exposes no public
``api/v1`` surface.

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
    "KNOWLEDGE_STATES",
    "OUTCOME_BASE_ERROR",
    "OUTCOME_CALIBRATION_ONLY",
    "OUTCOME_SELECTED",
    "ProvenanceNote",
    "REASON_BASE_CORRUPTED",
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
