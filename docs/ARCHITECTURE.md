# ZeCalibrator — Architecture (specified v1)

**Status:** Phase 1 specification — owner decisions reconciled 2026-09-15;
**G1 ACCEPTED by Junior, 2026-09-15.** Design/API targets only: no implemented
package or advertised capability is claimed by this document.

Authority: `ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md` §6, §9–§12 and
`ZESOFTWARE_INTEROPERABILITY_RULES.md` v1.0.

---

## 1. Dependency direction (specified)

```
scientific/core ← application/services ← public API / CLI / GUI adapters
                                                   ↑
                                               PySide6 GUI
```

Internal responsibility split (specified):

- **core** — metadata/units/geometry, pure equations, DQ semantics. No I/O.
- **io** — FITS decode, library indexing, provenance, storage adapters.
- **application** — orchestration, plans, progress, cancellation, transactions.
- **api.v1** — exported stable models/functions only.
- **gui** — presentation only; PySide6 import isolated.

Hard boundaries (INVARIANT from mission §6.1):

- No Qt, ZSSS, ZeAlfie, CuPy, or camera-device dependencies in scientific
  modules.
- No GUI import from package root or `api.v1`.
- Optional integrations are lazy.
- No generic top-level installed `utils`/`core`/`config`/`helpers` modules
  (shared-runtime safety, Interop Rule 17).

CPU is the scientific reference; GPU may change execution strategy only, and is
unsupported/unadvertised until a future gate.

---

## 2. Names, versions, capabilities (specified)

| Item | Specified value | Note |
| --- | --- | --- |
| distribution | `ZeCalibrator` | |
| product_id / package | `zecalibrator` | |
| public module | `zecalibrator.api.v1` | |
| product version target | `0.1.0` | |
| public API version target | `1.0` | independent of product version |
| consumer compatibility range | `>=1,<2` + capabilities | |
| provenance schema | `zecalibrator.provenance.v1` | |
| library index schema | `zecalibrator.library.v1` | specified |
| matching-policy version | `zecalibrator.match.v1` | specified |
| science-contract version | `1.0` | specified |
| digest canonicalization | `zecalibrator.digest.v1` | specified (PROVENANCE.md §2) |

Product version, API version, provenance schema, library schema, matching-policy
version, science-contract version, and digest scheme are **distinct**. Breaking
API semantics require a new major. Additive compatible changes require
capability negotiation and documented deprecation (Interop Rules 3–5).

Initial capability identifiers to freeze:

```
calibrate_frame, calibration_library, master_matching, provenance, cancel
```

`calibrate_batch` is added at Phase 6 (implemented and advertised since its G6
acceptance). `GPU` and `master_building` are **not** advertised. Static wheel
capabilities describe implemented public behavior, not hardware present; dynamic
probe distinguishes supported code from currently available resources. The six
implemented identifiers are the five above plus `calibrate_batch` (Phase 6).

---

## 3. Public API v1 — concrete specified model

This is the **design contract** for the public API, delivered in G5 (public facade) and G6
(batch). The field tables below remain the authoritative contract.

### 3.1 Explicit request/roles routing (specified — resolves sketch gap)

The mission sketch's `resolve_calibration(FrameInspection, LibraryHandle,
MatchPolicy)` lacks an obvious request argument. The specified contract fixes
this by making the **requested calibration roles an explicit parameter**, never
an implicit global state, and by **removing any optionality about whether roles
are required**: each supported mode fixes its exact required roles, and missing
a role is `NO_MATCH` (never a silent downgrade).

- `CalibrationRequest` carries an explicit `additive_mode` and `flat_mode` (the
  exact roles required), enumerated in §3.3.
- `resolve_calibration` consumes that request; matching is a pure function of
  `(frame, request, library, policy)`.
- The returned `CalibrationPlan` binds the request **and** the exact resolved
  master identities/geometry; `calibrate_frame` takes only the plan (plus a
  source whose geometry/metadata provably match the plan's constraints).

There is no module-level "active calibration set", no ambient default mode, and
**no "roles are optional" flag**.

### 3.2 Signatures (specified) — operation outcome envelopes

Cheap discovery ops return plain values; expensive/cancellable ops return a
**structured operation outcome envelope** with an `operation_status` field, so
anticipated operational failures (cancellation, I/O, unhealthy provider) are
structured results, not exceptions, and `MATCHED`/`NO_MATCH`/`AMBIGUOUS` remain
pure **selection** states.

```python
API_VERSION: str  # "1.0"

def get_api_info() -> ApiInfo: ...
# cheap; no file scans, Qt/GPU init, network, or writes

def probe(check_library: bool = False, check_gpu: bool = False, *,
          library_spec: LibrarySpec | None = None) -> RuntimeProbe: ...
# side-effect-free by default; bounded optional checks with explicit reasons

def open_library(spec: LibrarySpec, *, cancel: CancellationToken | None = None,
                 progress: ProgressObserver | None = None) -> OpenLibraryResult: ...
# OpenLibraryResult.operation_status ∈ {OPENED, CANCELLED, FAILED};
# handle present ONLY when OPENED; no ZeAlfie config discovery

def inspect_frame(source: FrameSource, *,
                  cancel: CancellationToken | None = None,
                  progress: ProgressObserver | None = None) -> InspectResult: ...
# InspectResult.operation_status ∈ {COMPLETED, CANCELLED, FAILED};
# inspection present ONLY when COMPLETED

def resolve_calibration(frame: FrameInspection,
                        request: CalibrationRequest,
                        library: LibraryHandle,
                        policy: MatchPolicy, *,
                        cancel: CancellationToken | None = None,
                        progress: ProgressObserver | None = None) -> ResolveResult: ...
# ResolveResult.operation_status ∈ {COMPLETED, CANCELLED, FAILED};
# when COMPLETED, match.outcome ∈ {MATCHED, NO_MATCH, AMBIGUOUS} (selection only)

def calibrate_frame(source: FrameSource,
                    plan: CalibrationPlan,
                    options: ExecutionOptions, *,
                    cancel: CancellationToken | None = None,
                    progress: ProgressObserver | None = None) -> CalibrationResult: ...

def calibrate_batch(frames: Iterable[FrameSource], request: CalibrationRequest,
                    library: LibraryHandle, policy: MatchPolicy,
                    options: ExecutionOptions, *, cancel=None,
                    progress=None) -> Iterator[BatchItem]: ...
```

**Operation/error mapping (specified, explicit):**

| Operation | Parameter validation | Anticipated operational failure |
| --- | --- | --- |
| `open_library` | raises `InvalidRequestError` (bad spec) | returns `OpenLibraryResult(operation_status=FAILED/CANCELLED)`; **no handle** |
| `inspect_frame` | raises `InvalidRequestError` (bad source) | returns `InspectResult(operation_status=FAILED/CANCELLED)`; no inspection |
| `resolve_calibration` | raises `InvalidRequestError` (bad request) | returns `ResolveResult(operation_status=FAILED/CANCELLED)`; no match |
| `calibrate_frame` | raises `InvalidRequestError` (plan/source mismatch) | returns `CalibrationResult(status=FAILED/CANCELLED)` |
| `get_api_info` / `probe` | n/a | never raises for capability absence; `probe` reports states |

`MATCHED`/`NO_MATCH`/`AMBIGUOUS` are selection states only, never used to report
cancellation or failure. Callback (observer) exceptions are isolated and logged;
they never change arithmetic and never surface as operation failure. Consumers
still catch ordinary unexpected exceptions to isolate unhealthy providers; never
swallow process interrupts.

### 3.3 Concrete data models (specified)

All models are immutable value objects (frozen dataclasses / `NamedTuple`s).
`None` means **unknown/not-present**, never a silent default.

#### `CardRecord` and `SensorMetadata`

```python
CardRecord = (keyword: str, value: object, comment: str, index: int, source: str)
```

`original_cards` is an **ordered sequence** `tuple[CardRecord, ...]` preserving
**duplicate candidate cards and their original order/index/source** (a `Mapping`
cannot — it dedupes keys). Normalized values and conflicts are derived, never
used to discard the original sequence.

| Field | Type | Required | Notes / invariant |
| --- | --- | --- | --- |
| original_cards | `tuple[CardRecord, ...]` | required | ordered, duplicate-preserving, never pruned |
| normalized | `Mapping[str, NormalizedValue]` | required | trimmed/padded-normalized values + units |
| units | `Mapping[str, str]` | required | explicit unit per normalized field |
| alias_version | `str` | required | profile/alias table version used |
| conflicts | `tuple[ConflictDiagnostic, ...]` | required | empty when none; disagreement never silently resolved |
| raw_domain_declaration | `Literal["raw","processed","unknown"]` | required | unknown ⇒ reject unless import declaration |
| geometry | `Geometry` | required | may contain unknown subfields during inspection |
| detector | `DetectorIdentity` | required | unknown is explicit |
| acquisition | `Acquisition` | required | unknown fields explicit, never defaulted |
| optical | `OpticalIdentity` | required | filter + optical train; unknown explicit (see below) |

#### `OpticalIdentity`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| filter | `str \| None` | optional | `None` = unknown; flat-vs-light matching rejects unknown required filter |
| optical_train_id | `str \| None` | optional | `None` = unknown; flat-vs-light matching rejects unknown required train |

`OpticalIdentity` is an explicit constraint on `SensorMetadata`, not merely an
entry in the `normalized` map. It participates in the plan identity projection
(§3.3 `CalibrationPlan`, PROVENANCE §2.3), so changing the light's `filter` or
`optical_train_id` changes the plan digest. For additive-only (dark/bias) modes
with no flat correction, filter/train are **not** required and an unknown value
is **not** a rejection; only flat-vs-light matching requires them (SCIENCE §6.2).

#### `Geometry`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| sensor_dimensions | `(int, int)` | required | full-frame sensor `(ny, nx)`, distinct from ROI shape |
| shape | `(int, int)` | required | array/ROI `(ny, nx)` (row-major) |
| binning | `(int, int)` | required | exact |
| roi_origin | `(int, int) \| None` | optional | `None` = unknown (inspection); must be known before matching |
| roi_extent | `(int, int) \| None` | optional | `None` = unknown |
| orientation | `Literal["identity"]` | required | v1 supports identity only |
| cfa_phase | `Literal["GRBG","RGGB","BGGR","GBRG","mono"] \| None` | optional | `None` = unknown CFA (never treated as mono); `"mono"` = explicit mono sensor |

Allowed CFA patterns are enumerated explicitly (no ellipsis). `"mono"` (explicit
mono sensor) is distinct from `None` (unknown CFA). A frame with unknown ROI
origin/extent is representable during inspection; matching rejects incomplete
required geometry (`MISSING_REQUIRED_FIELD` / `GEOMETRY_MISMATCH`). No automatic
transformations.

#### `DetectorIdentity`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| detector_instance_id | `str` | required | user-assigned when header cannot identify; `"unknown"` is explicit, never `""` default |
| detector_model | `str \| None` | optional | `None` = unknown at inspection; matching rejects unknown model (never merge S30/S50) |
| serial | `str \| None` | optional | when known |

#### `Acquisition`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| gain | `float \| None` | optional | never inferred as e-/ADU from GAIN card; `None` = unknown |
| offset | `float \| None` | optional | unknown explicit; never default 0 |
| readout_mode | `str \| None` | optional | unknown explicit |
| adc_mode | `str \| None` | optional | unknown explicit; never inferred from BITPIX |
| temperature_c | `float \| None` | optional | measured, not setpoint; `None` = unknown |
| exposure_s | `float \| None` | optional | `None` = unknown at inspection; matching rejects unknown exposure |
| saturation_limit_adu | `float \| None` | optional | qualified only; None ⇒ unknown |
| saturation_evidence | `Literal["qualified","unknown"]` | required | frame-quality diagnostic (SCIENCE §9.3) |

**Inspection vs matching (S4):** during `inspect_frame`, any acquisition or
identity field may be `None`/`"unknown"` (missing header value is representable,
never fabricated). Matching rejects any **required** fact that is still unknown
(`MISSING_REQUIRED_FIELD`), without user-policy assumption.

#### `InputIdentity` (discriminated)

| Variant | Fields | Invariant |
| --- | --- | --- |
| `FitsIdentity` | `path: str`, `hdu: int \| str`, `whole_fits_sha256: str \| None`, `decoded_digest: str` | whole-FITS hash only when computable |
| `ArrayIdentity` | `caller_logical_id: str`, `decoded_digest: str` | **no hdu, no path, no fake FITS hash** |

An array source never carries a fabricated FITS hdu/hash; its identity is the
caller's logical id + the canonical decoded-data digest.

#### `MasterDescriptor`

Binds matching-relevant metadata, not just role.

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| master_type | `Literal["bias","dark","flat","flat_dark"]` | required | semantic role, never filename; `flat_dark` is **additive**, not a response-flat |
| pixel_domain | `Literal["sensor_adu","normalized_response"]` | required | |
| physical_units | `Literal["ADU","dimensionless"]` | required | ADU for raw/corrected-unnormalized; dimensionless for normalized `R` |
| bias_state | `Literal["included","removed","not_applicable","unknown"]` | required | |
| flat_form | `Literal["raw_response","corrected_unnormalized","normalized_response"] \| None` | conditional | only `master_type=="flat"`; `None` for non-response roles |
| normalization_algorithm | `str \| None` | conditional | required when `flat_form=="normalized_response"` |
| normalization_scalars | `NormalizationScalars \| None` | conditional | required when normalized; mono = 1 named scalar, CFA = **4 named scalars** `{G1,R,B,G2}` |
| geometry | `Geometry` | required | bound |
| detector | `DetectorIdentity` | required | bound |
| acquisition | `Acquisition` | required | bound |
| optical_train_id | `str \| None` | optional | required for flat; unknown explicit |
| filter | `str \| None` | optional | required for flat; unknown explicit |
| content_sha256 | `str` | required | content identity |
| size_bytes | `int` | required | |
| hdu | `int \| str` | required | |
| mask_identity | `str` | required | mask/descriptor identity |
| processing_provenance | `ProcessingProvenance` | required | import/processing evidence |
| validity_evidence | `ValidityEvidence` | required | for flats |
| descriptor_id | `str` | derived | acyclic digest (PROVENANCE §2.3), excludes itself |

`NormalizationScalars` is a discriminated value: `MonoScalar(s: float)` or
`CfaScalars(g1: float, r: float, b: float, g2: float)`. A normalized CFA flat
requires all four labelled scalars and a qualified normalization history; a
corrected-unnormalized flat must still be normalized by the engine (never
consumed directly as `R`).

#### `ProcessingProvenance` and `NormalizationProvenance`

Structured evidence record (not free text) describing what was done to a master
before import, so additive-correction history and normalization source/population/
scalars are explicit and auditable.

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| source | `Literal["synthetic_fixture","user_import","observed"]` | required | origin of the master |
| additive_correction_history | `tuple[str, ...]` | required | ordered; e.g. `()` for raw, `("bias_removed",)` or `("flat_dark_subtracted",)` |
| normalization | `NormalizationProvenance \| None` | conditional | required when `flat_form=="normalized_response"`; else `None` |

`NormalizationProvenance`:

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| algorithm | `str` | required | e.g. `"median"` or `"cfa-median-per-plane"` |
| population | `str` | required | e.g. `"mono-valid"` or `"cfa-4-plane"` |
| scalars | `NormalizationScalars` | required | mono 1 or CFA 4 named scalars |

#### `ValidityEvidence`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| saturation_limit_known | `bool` | required | |
| valid_normalization_count | `dict[str, int] \| None` | optional | per-plane valid population count |
| quality_policy_state | `Literal["pending","qualified"]` | required | `pending` ⇒ flat execution/matching cannot proceed as if qualified |

#### `MatchPolicy` and `FlatQualityPolicy`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| version | `str` | required | `zecalibrator.match.v1` |
| exposure_tolerance | `Tolerance` | required | ENGINEERING BOUND (SCIENCE §6.3) |
| temperature_tolerance | `Tolerance` | required | zero scientific + parser 1e-6 °C |
| flat_quality_policy | `FlatQualityPolicy` | required | see below; **no `None ⇒ not enforced`** |

`FlatQualityPolicy` is a discriminated value — it is **forbidden** to encode
"no quality screening" as a silent `None` default. The adopted product default
is FlatQualityQualified(90.0, per_plane=True); this qualifies the policy value,
not the master or its missing quality evidence:

```python
FlatQualityPolicy = FlatQualityPending            # incomplete/unrecognized policy; never implicit no-screening
                  | FlatQualityQualified(threshold_pct: float, per_plane: bool)
```

- `FlatQualityPending` ⇒ flat-mode matching/execution **refuses** with reason
  `QUALITY_POLICY_PENDING`; it never silently proceeds as if a threshold were
  accepted. Control/no-flat request modes do not require a flat policy.
- `FlatQualityQualified(t, per_plane)` carries the **actual** threshold value
  (owner-frozen or a qualified profile), and that value is frozen into the plan
  digest (§3.3 `CalibrationPlan`).

#### `Tolerance`

| Field | Type | Required |
| --- | --- | --- |
| relative | `float` | required |
| absolute | `float` | required |

#### `CalibrationRequest`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| additive_mode | `Literal["control","bias_only","dark_incl_bias","dark_bias_removed"]` | required | exact role; no downgrade |
| flat_mode | `Literal["none","apply"]` | required | exact role |
| required_roles | `tuple[str, ...]` | derived | computed from modes, never user-optional |

#### `CalibrationPlan`

Freezes **validated light metadata constraints** and **actual policy parameters**,
not merely version strings.

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| plan_id | `str` | derived | acyclic digest (PROVENANCE §2.3), excludes itself + output digests |
| request | `CalibrationRequest` | required | frozen |
| light_constraints | `SensorMetadata` | required | frozen validated light geometry/detector/acquisition/optical/domain |
| masters | `Mapping[str, MasterBinding]` | required | role → bound master (descriptor + content hash + HDU) |
| policy_parameters | `PolicyParameters` | required | actual tolerances + flat-quality threshold (or pending) |
| versions | `VersionSet` | required | science/decoder/provenance-schema/matching-policy versions |

`MasterBinding` is an immutable value object with two disjoint concerns —
**science identity** (hashed) and **retrieval locator** (excluded from hashing):

| Field | Type | Required | In identity digest? |
| --- | --- | --- | --- |
| descriptor_id | `str` | required | yes |
| descriptor_snapshot | `DescriptorSnapshot \| None` | required | **no** (retrieval; see below) |
| content_sha256 | `str` | required | yes |
| size_bytes | `int` | required | yes |
| hdu | `int \| str` | required | yes |
| mask_identity | `str` | required | yes |
| mask_locator | `MaskLocator \| None` | required | **no** (retrieval) |
| locator | `MasterLocator \| None` | optional | **no** (retrieval only) |

`MasterLocator` is a narrow immutable retrieval locator for local FITS bytes,
separate from hashed identity:

```python
MasterLocator = FitsFileLocator(path: str, hdu: int | str)
MaskLocator = MaskPayloadLocator(path: str)          # external DQ/mask bytes
DescriptorSnapshot = immutable complete MasterDescriptor + ProcessingProvenance
                     + ValidityEvidence + import declarations (full acyclic field set)
```

**Imported declaration retention/revalidation (F2, design only — no storage
engine implemented).** Imported `ProcessingProvenance`/`ValidityEvidence`/import
declarations are **not** assumed to live inside the FITS image bytes. The plan
therefore carries, per master, an **immutable `descriptor_snapshot`** — the
complete descriptor record (acyclic field set of §2.3 item 2) whose own digest is
`descriptor_id` — and an explicit **`mask_locator`** for the DQ mask payload.

- `descriptor_id` is `SHA-256(CANONICAL_JSON(descriptor_snapshot))`; the snapshot
  is retained with the plan and re-hashed at execution to prove it is unchanged.
- `mask_identity` is the mask-payload digest; `mask_locator` points at the mask
  bytes, re-hashed before use.
- `content_sha256` is the **image-byte** SHA, distinct from the descriptor digest
  and the mask digest; the three are never conflated.
- No global library lookup, no ZeAlfie dependency, no assumption that user
  declarations exist in FITS. A declaration change (tampered snapshot) makes the
  recomputed descriptor digest differ from `descriptor_id` ⇒ structured `FAILED`,
  never a substitution or a silent reinterpretation. Declaration **revision**
  semantics are explicit: a new declaration is a new snapshot + new
  `descriptor_id` (a different binding), not an in-place mutation of an old one.
- A `LibraryHandle` closed with unchanged identities (snapshot bytes, mask bytes,
  image bytes) remains usable by design; changed/missing required bytes or mask
  ⇒ structured `FAILED`.

A plan is reusable only for a frame whose validated metadata equals
`light_constraints` and whose policy parameters equal `policy_parameters`; a
same-version-but-different-threshold policy produces a **different** plan digest
(asserted in `contract_witness.py` `identity_cases`).

#### `PolicyParameters` and `VersionSet`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| exposure_tolerance | `Tolerance` | required | ENGINEERING BOUND (SCIENCE §6.3) |
| temperature_tolerance | `Tolerance` | required | zero scientific + parser 1e-6 °C |
| flat_quality_policy | `FlatQualityPolicy` | required | actual threshold or `pending` |

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| science_contract | `str` | required | `"1.0"` |
| decoder | `str` | required | decoder/domain contract version |
| provenance_schema | `str` | required | `"zecalibrator.provenance.v1"` |
| matching_policy | `str` | required | `"zecalibrator.match.v1"` |

#### `ExecutionOptions`

| Field | Type | Required | Invariant |
| --- | --- | --- | --- |
| backend_requested | `Literal["cpu","gpu"]` | required | gpu not advertised |
| memory_budget | `MemoryBudget \| None` | optional | |
| tile_shape | `(int, int) \| None` | optional | cooperative cancel latency bound |
| output_path | `str \| None` | optional | standalone FITS only |

#### `ApiInfo` / `RuntimeProbe`

| Field | Type | Required |
| --- | --- | --- |
| api_version | `str` | required |
| product_version | `str` | required |
| capabilities | `tuple[str, ...]` | required |

```python
CapabilitySupport     = SUPPORTED | NOT_SUPPORTED      # static code capability
CapabilityAvailability = NOT_CHECKED | AVAILABLE | UNAVAILABLE | UNHEALTHY
```

| Field | Type | Required |
| --- | --- | --- |
| capabilities | `Mapping[str, CapabilityEntry]` | required |
| reasons | `tuple[str, ...]` | required |

`CapabilityEntry = (support: CapabilitySupport, availability: CapabilityAvailability)`.
An **unsupported** GPU is representable truthfully as `support=NOT_SUPPORTED,
availability=UNAVAILABLE` (never advertised as `SUPPORTED`). Defaults are
side-effect-free: `probe()` performs no scans/writes/imports.

#### `FrameInspection`

| Field | Type | Required |
| --- | --- | --- |
| metadata | `SensorMetadata` | required |
| identity | `InputIdentity` | required |
| domain_finding | `Literal["raw","processed","unknown","unsupported"]` | required |
| hdu | `int \| str \| None` | required | `None` for array sources (no invented FITS identity) |
| shape | `(int, int)` | required |
| warnings | `tuple[str, ...]` | required |

#### `MatchResult`

| Field | Type | Required |
| --- | --- | --- |
| outcome | `Literal["MATCHED","NO_MATCH","AMBIGUOUS"]` | required |
| plan | `CalibrationPlan \| None` | conditional (MATCHED) |
| rejected_candidates | `tuple[RejectionRecord, ...]` | required |
| reason_codes | `tuple[str, ...]` | required (stable codes) |

`RejectionRecord` = `(candidate_id, reason_codes, details)`.

#### `CalibrationResult`

| Field | Type | Required |
| --- | --- | --- |
| status | `Literal["COMPLETED","COMPLETED_WITH_WARNINGS","SKIPPED","CANCELLED","FAILED"]` | required |
| data | `float32 ndarray \| None` | conditional |
| mask | `uint16 ndarray \| None` | conditional |
| counts | `CountSummary` | required |
| frame_quality | `FrameQuality` | required |
| matching_report | `MatchResult \| None` | conditional |
| execution_report | `ExecutionReport` | required |
| provenance | `ProvenanceRecord` | required |
| warnings | `tuple[str, ...]` | required |
| reason_code | `str \| None` | optional |

`CountSummary` = `(total, valid_count, invalid_count, per_bit)` where
`invalid_count = total − valid_count` and `per_bit` counts overlap independently
(SCIENCE §9.2).

`FrameQuality` = `(saturation_evidence: Literal["qualified","unknown"], …)`
(SCIENCE §9.3).

#### Status semantics (specified, stable)

- `COMPLETED` — full success, `mask == 0` everywhere.
- `COMPLETED_WITH_WARNINGS` — partial invalidity or unknown quality evidence;
  data + exact counts returned.
- `SKIPPED` — a frame intentionally not processed at batch stage (e.g. already
  calibrated/non-raw/domain-unsupported), recorded as a skip disposition, not an
  error and distinct from `NO_MATCH` (matching) and `FAILED` (error).
- `CANCELLED` — cooperative cancellation; no committed partial output.
- `FAILED` — all-invalid frame, or an unrecoverable error; no data.

### 3.4 FrameSource — tagged union (specified, INVARIANT)

```python
FitsFrameSource(path: Path, hdu: int | str)            # product performs raw decode
ArrayFrameSource(data: ArrayLike, metadata: SensorMetadata,
                 identity: ArrayIdentity, invalid_mask: ArrayLike | None,
                 domain: str = "sensor_adu",
                 scaling_applied: bool = True)
```

- `FitsFrameSource` — correct raw decode inside the product.
- `ArrayFrameSource` — caller guarantees the decoded physical domain; reject
  ambiguous scaled/normalized arrays. `scaling_applied=True` means the caller
  has already applied BSCALE/BZERO; anything else must be explicit.
- A mutable header is never accepted as the sole proof of domain.
- Public metadata preserves original card evidence, units, and explicit decoder
  facts.
- Array input is read-only to the callee; no in-place modification, even on
  failure. Output owns its buffer; masks and provenance outlive library-context
  closure. At least one float32 result allocation is expected (no zero-copy
  promise).

### 3.5 Handles / lifetime / no source mutation (specified, INVARIANT)

- `LibraryHandle` is context-managed; `close()` is idempotent. Not blindly
  shared across threads/processes; expose documented thread-safe reads or one
  handle per worker.
- Plans bind input geometry/metadata and exact master identities; reusable only
  for frames proven to match the same constraints.
- No global mutable active calibration set.
- **In-memory result semantics:** `calibrate_frame` returning a `CalibrationResult`
  is committed the moment the result object is produced (data + mask + provenance
  are complete and immutable); progress `100%` is reported at that point. A
  standalone FITS output, by contrast, is committed only after the transactional
  no-clobber publication completes (PROVENANCE §5); `100%` is reported only then.

### 3.6 Result / status / errors (specified)

- API validation errors raise typed stable exceptions
  (`ZeCalibratorApiError`, `InvalidRequestError`, `LibraryClosedError`, …).
- Anticipated **operational** failures return structured results (operation
  envelopes §3.2, `CalibrationResult` status `FAILED`/`CANCELLED`), not
  exceptions.
- `NO_MATCH` / `AMBIGUOUS` belong to matching (`MatchResult`), not numerical
  success and not cancellation/failure.
- Consumers still catch ordinary unexpected exceptions to isolate unhealthy
  providers; never swallow process interrupts.

### 3.7 Cancellation and progress (specified)

- `CancellationToken` is Qt-independent; check before expensive read/hash/
  normalization, between tiles, between frames, and before commit.
- A blocking filesystem/native call cannot promise instantaneous cancellation;
  cooperative latency is defined by tile size. No unsafe thread kill.
- `ProgressEvent`: `operation_id`, `phase`, `completed`, `total-or-null`, `unit`,
  optional `frame_id`. Immutable, monotonically ordered; unknown total remains
  unknown; only committed completion is `100%` (in-memory vs filesystem per §3.5).
- User callbacks run outside library/database locks; throttle delivery. A
  failing observer must not change arithmetic — isolate/log observer failures;
  cancellation remains a separate explicit token.
- No unbounded producer queues. GUI owns a worker `QObject`/`QThread` (or
  bounded pool) with queued signals; widgets accessed only on the Qt main
  thread.
- Cancellation surface is present on **expensive** operations: `open_library`
  (indexing), `inspect_frame`, `resolve_calibration`, `calibrate_frame`,
  `calibrate_batch`. `get_api_info`/`probe` are cheap and have no cancellation
  surface (no expensive work to cancel).

### 3.8 In-memory provenance (specified, INVARIANT)

Embedded results carry complete structured provenance **in memory**; no implicit
log/sidecar write by the calibration core. No compulsory intermediate FITS file
for the array path. The consumer attaches provenance to its own ledger.

---

## 4. Wheel interoperability declaration (specified, Phase 5)

Bootstrap may ship an empty `provides` list. At Phase 5 acceptance, package one
declaration at `zecalibrator/zesoftware_interop.json`:

```json
{
  "schema": "zesoftware.interop.v1",
  "product_id": "zecalibrator",
  "distribution_name": "ZeCalibrator",
  "provides": [{
    "api_module": "zecalibrator.api.v1",
    "api_version": "1.0",
    "capabilities": [
      "calibrate_frame", "calibration_library",
      "master_matching", "provenance", "cancel"
    ]
  }],
  "consumes": []
}
```

Metadata must agree with installed runtime API tests. No ZeAlfie import is
needed to generate/read it. ZeAlfie catalog admission is a separate deployment
integration, not an application dependency.

---

## 5. Library indexing (specified)

Read-only over user masters; store references and normalized metadata, not owned
copies. Scanner: stable sort for display, cancellation support (observes
`CancellationToken`), symlink-loop avoidance, records unreadable files (not
"missing success"), invalidates cache on change. Start with a versioned SQLite
index (stdlib `sqlite3`), short transactions, one serialized writer, filesystem
abstraction so a smaller alternative can be justified before implementation.
Network libraries / distributed locking deferred.

---

## 6. Execution contract (specified)

- NumPy CPU, float32 frame arithmetic with documented float64 decode/statistical
  intermediates. No mandatory CUDA/CuPy; no GUI-thread GPU probing.
- Distinguish eligible / requested / attempted / executed / CPU fallback with
  reason. CPU fallback only if feasible; otherwise fail explicitly.
- Budget RAM/VRAM for arrays, masks, masters, output, scratch — not only input
  dimensions.
- On OOM shrink spatial tile/worker concurrency; never change input population,
  masters, equations, or precision silently. Bound retries; preserve original
  buffers; transactional output.
- Global flat normalization computed once from the canonical population before
  tile execution; exact median cannot be replaced by a streaming approximation
  without a new science contract. If it does not fit, use exact out-of-core
  selection or refuse; do not average tile medians.
- Backend report includes requested/supported/available/eligible/attempted/
  executed/effective backend, tile shape/count, memory budget, retries, fallback,
  reason, device/backend/library versions. Mixed executed tiles are reported
  explicitly, not relabelled wholesale.

---

## 7. GUI architecture (design only, Phase 7)

PySide6 is the only primary GUI toolkit. GUI widgets never perform calibration
equations, master selection, FITS parsing, or provenance assembly. Headless
engine installs without Qt.

```
ZeCalibrator          → engine / public API / CLI
ZeCalibrator[gui]     → PySide6 application
official desktop artifacts and ZeAlfie GUI install → always include [gui]
```

No other toolkit. No Qt import for scientific tests or storage-path resolution.
Initial GUI scope: inputs, library roots, matching explanations, explicit mode,
output destination, preflight summary, progress/cancel, per-file result, audit
link. Never mask a no-match with a convenient master.

### 7.1 Implementation status (Phase 7 / M1, NOT YET ACCEPTED)

**Status:** implemented as a bounded public-API-only PySide6 client (mission
`ZC-P7-M1-GUI-20260918`, branch `feat/zc-p7-gui`). G7 is **NOT ACCEPTED**;
Windows/macOS interactive native witnesses remain NOT_RUN and are a mandatory
qualification criterion, never a reason to block the technical implementation.

Modules (all under `src/zecalibrator/gui/`):

- `app.py` — isolated launcher; PySide6 imported lazily; precise missing-`[gui]`
  diagnostic; headless import stays Qt-free.
- `service.py` — pure, Qt-free API client: request construction, JSON -> public
  value-object parsing, immutable `OperationSnapshot`/`LightInput`.
- `settings.py` — versioned atomic injected GUI settings under
  `StoragePaths.user_config_path` (single storage adapter, no second
  `QStandardPaths` owner).
- `presentation.py` — pure formatting of public summaries (stdlib only).
- `identity.py` — Windows AppUserModelID / Linux desktop id / packaged icon
  (lazy Qt).
- `worker.py` — one `QObject` worker on a dedicated `QThread`; queued immutable
  events; bounded/coalesced progress; shared `CancellationToken`; no terminate.
- `window.py` — main window: progressive-disclosure top-level tabs
  (`Standard | Advanced | Settings`, Standard default) as three views of one
  application/scientific state. Standard shows the nominal workflow (images,
  library, dark/flat in human labels, verify, calibrate/export); Advanced
  retains all technical controls/outcomes (HDU/declaration/ROI evidence,
  library open/index, exact technical mode values, in-memory calibration,
  technical preflight/results/details/audit, qualification label); Settings
  holds the Appearance/Theme preference (System/Light/Dark). Progress/status/
  Cancel live in a global footer below the tabs. Safe close, finished-driven
  teardown and the single worker controller are unchanged.
- `theme.py` — bounded Qt palette helper for System/Light/Dark (System restores
  the pre-override palette and never forces a non-native style; Light/Dark are
  deterministic ``QPalette`` overrides). Not a theme engine or style framework.

P7-M2 UX simplification keeps Standard and Advanced bound to the same
``_lights`` / ``_library_spec`` / request / generation / plans / results and
controller: tab changes never mutate or re-bump scientific state, and a real
mode change invalidates cached plans exactly once. The GUI-only
``appearance_theme`` preference round-trips through the existing versioned
``GuiSettings`` persistence (schema 1, backward compatible; missing/invalid
value falls back to System). There is no ``QSettings`` and no Language control:
the repository has no ``QTranslator`` / translation resources, so a
`Settings > Language` selector is recorded as a future separate micro-phase and
is not implemented here.

Boundaries enforced: widgets assemble presentation requests only; all
science/inspection/library/matching/calibration/indexing/output I/O run on the
worker thread through `zecalibrator.api.v1`; no `zecalibrator.core` /
`zecalibrator.application` / `zecalibrator.io` / `zecalibrator.cli` imports
under `gui/`. `index_library` requires explicit `MasterImportSpec` + `mask_path`;
`calibrate_batch` has no `manual_selection`, so the GUI exposes ambiguity and
refuses ambiguous execution (no manual-pick that exports would ignore).

Witness procedure (ASTRA §§17 G7 / 18):

1. Offscreen Qt CI step (authored, not executed here) runs `tests/gui` +
   `tests/test_gui.py` + `tests/test_icons_qt.py` on Python 3.13 on all three OS
   (ubuntu-latest, windows-latest, macos-latest); the full Qt offscreen GUI suite
   now runs on Python 3.13.
2. Interactive native Linux/Windows/macOS worker/event-loop/dialog witnesses
   (launch from unrelated CWD, native dialogs, selected-input/preflight/export
   flow, progress/cancel, safe close while active, scaling and app identity)
   are **NOT_RUN** until provided by Tristan/Junior; local offscreen Linux PASS
   is not native GUI qualification and says nothing about Windows/macOS.

G7 full cross-platform ACCEPT remains HOLD pending those witnesses plus Nono
review. G8 packaging stays deferred.

---

## 8. Storage and portability (design only, §12)

- Supported Python `3.13.x`; qualify NumPy/Astropy/PySide6 wheels before freezing
  minimum/upper. Platform-neutral `pathlib`/`os.fspath`/package resources; no
  CWD/checkout-layout/fixed-drive/GNU-tool/shell dependency for ordinary
  behavior.
- **platformdirs** behind one `zecalibrator` storage adapter:
  `appname="ZeCalibrator"`, `appauthor="ZeSoftware"`, version-independent roots,
  non-roaming mutable library/cache data. GUI uses the same adapter; no second
  `QStandardPaths` owner of persisted locations.
- Location classes: `user_config_path` (versioned settings, atomic writes,
  migration backups), `user_data_path` (library index/descriptors), `user_cache_path`
  (rebuildable thumbnails/materialization), `user_state_path`/`user_log_path`
  (bounded logs/recovery), explicit user destination (output), explicit user
  library roots (masters), destination filesystem (staging). No import-time
  directory creation or reads of live user settings. Tests inject all roots.
- Windows long-path failures get explicit diagnostics, not silent truncation.
  Subprocess adapters (if later needed) use argument arrays, `shell=False`,
  explicit timeouts/cancellation.

---

## 9. Icons and resources (design only, §13)

Three categories: (1) canonical supplied artwork under `icons/` (untouched);
(2) packaged runtime resources under `src/zecalibrator/resources/icons/`
(byte-identical copies, package-data); (3) release artifacts under `build/dist`
only. A manifest maps canonical files to packaged copies and verifies hashes.
Qt loading follows the observed ZSSS pattern
(`importlib.resources.files("zecalibrator")…` → `QPixmap.loadFromData` → `QIcon`,
GUI thread). Runtime never searches the top-level `icons/` directory or creates
package files. Stable specification identities for packaging: Windows
AppUserModelID `ZeSoftware.ZeCalibrator`; Linux desktop id
`io.github.tinystork.ZeCalibrator`; macOS bundle id `com.zesoftware.zecalibrator`.

---

## 10. ZSSS integration seam (deferred, §7)

A consumer-owned `RawCalibrationPort` + `NoCalibrationAdapter` +
`ZeCalibratorPublicAdapter` (lazy importlib discovery, API range/capability
checks) is introduced at **HDU selection + original header capture + physical
raw decoding**, before the sanitizer/shape/nonfinite/minmax path. This is a
separate later ZSSS mission (§7.2 signed raw-domain prerequisite gate). Not
implemented in Phase 1. No actual ZSSS changes are made.

---

## 11. Future master-building and ZeMosaic seams (deferred, §8)

No public versioned ZSSS raw-sensor N-frame reduction API exists. A conditional
`seestar.api.v1` public capability (`raw_sensor_reduce`) is a separate ZSSS
contract mission; ZeCalibrator owns source selection/acquisition checks/flat
normalization/master semantics. ZeMosaic gets the same public `RawCalibrationPort`
semantics at its own raw FITS seam in Phase 12. Both deferred; do not import
sibling private code. **No ZeStackCore.**

---

## 12. Provenance (summary)

Full schema, identity/hash canonicalization (acyclic science/descriptor/plan
digests, fixed byte order, no-TOCTOU, no hash self-reference), and transactional
no-clobber FITS/DQ/CALPROV design are specified in `PROVENANCE.md`. The
architecture guarantees in-memory provenance with no compulsory FITS (§3.8).

---

## 13. Specified bootstrap skeleton (Phase 2, not created now)

See mission §14. No `pyproject.toml`, `src/`, package, CI, GUI, integration,
backend, or master-builder files are created in Phase 1. Entry points and
`_version.py` are Phase 2+. License is an explicit owner decision before
distribution; GPL-3.0-or-later was explicitly selected by Tristan (§14.1).

---

## 14. Owner policies — decided by Tristan, 2026-09-15

Decision authority: Tristan's explicit HUMAN_GATE reply, reconciled by Junior.
These are adopted specification policies, not claims of implemented behavior.
The immutable architecture handoff remains historical; this dated decision
supersedes its open choices. No publication or Phase 2 dispatch is authorized.

### 14.1 Project license

**GPL-3.0-or-later** for ZeCalibrator. The future Phase 2 bootstrap must add the
license text and accurate package metadata. This decision does not license
unrelated artwork or authorize copying third-party source without checking its
terms; supplied artwork is preserved unchanged. No license/package file is
created during this Phase 1 closure.

### 14.2 First supported acquisition cohort/profile

**SYNTH-BASE-1**, explicitly labelled **synthetic-only qualification**.
The reference case geometry is array shape (ny,nx)=(1080,1920), CFA GRBG,
signed int16 FITS storage with BSCALE=1/BZERO=32768, decoded physical uint16 ADU;
fixture gain=100, offset=50, readout=MODE_A, adc_mode=MODE_16, light/dark=10 s,
flat/flat_dark=1 s, temperature=20 °C, saturation=60000 ADU, filter=NONE,
optical_train_id=SYNTH-TRAIN-1. These values are synthetic fixture facts,
NEVER defaults for a real detector. No calibration engine exists yet.

Real-camera qualification remains open until real compatible masters and raw
acquisition/firmware provenance are qualified. In particular S50/S30 headers
alone neither establish unprocessed raw data nor supply missing detector-instance,
offset/readout/ADC facts. Future real witnesses must qualify masters, acquisition
semantics, exact geometry and decoding; no real scientific release claim now.

### 14.3 Temperature

Default scientific tolerance **0 °C**, after unit conversion. The separate
serialization/parser equality tolerance remains **1e-6 °C**, not a scientific
thermal window. Temperature is an important compatibility criterion; do not
drop it to improve match rate.
Any nonzero window requires a **versioned camera/master-specific profile** and
**real witnesses** validating that window. No generic nonzero tolerance is
invented or adopted by this decision.

### 14.4 Flat quality

Default **at least 90% valid samples in EACH CFA parity plane** (G1/R/B/G2
separately; mono uses its single plane), as a **product screening threshold,
not a physical law**. Numerator = valid normalization population in SCIENCE
§7.4; denominator = total geometric sample count of that plane.
A later versioned profile may replace this threshold; record both its version
and actual value in the plan/provenance. The threshold alone does not establish
spatial/scientific quality, saturation evidence, or master compatibility.
The separate R > 1e-6 division guard is unchanged.

### 14.5 Missing metadata and role-specific filter applicability

**Strict refusal is the automatic default** for missing required facts.
Versioned, evidence-backed import declarations are authorized to supply missing
detector-instance/offset/readout facts with visible source/evidence identity.
Never invent a value, silently default a real detector, or permit a declaration
to hide/override any contradiction with the FITS evidence. Conflicts reject.

**Filter is exact for flats**, together with optical-train identity and the
other applicable criteria. Filter is **not an artificial matching requirement
for dark/bias** when scientifically irrelevant; additive-only modes accept
unknown optical metadata without inventing it. Recording a known filter in
plan/provenance is not a dark/bias filter-equality requirement.
Unknown saturation remains a frame-quality diagnostic; required flat quality
evidence cannot be inferred merely from adoption of the 90% product threshold.

---

## 15. Independence guarantees (INVARIANT, Interop Rules 1–25)

- ZeCalibrator works without ZeAlfie, ZSSS, or GPU.
- ZSSS works without ZeCalibrator.
- No mandatory `import zealfie`; no sibling-repository/checkout-name assumptions;
  public contracts only.
- Optional integration failures are isolated at the adapter boundary with
  distinct states (absent / incompatible / unhealthy / capability unavailable /
  operation failed).
- User data never in package/runtime/ZeAlfie slots; versioned migrations.
- Immutable provenance and rollback per `PROVENANCE.md`.

---

## 16. Versioning summary

See §2. Versions and names are specification targets frozen upon G1 acceptance,
not claims of installed implementations. Breaking changes move to a new major;
capability identifiers are stable within a major.

---

## 17. Bootstrap implementation evidence (Phase 2 / Mission 1, 2026-09-16)

**Scope:** `ZC-P2-M1-BOOTSTRAP-20260915`. Independent repository, src-layout
package, GPL metadata, help/version entry points, package-resource manifest,
storage adapter, tests and authored CI. No scientific arithmetic was
implemented; this section records implementation evidence only, not G2
acceptance (G2 acceptance is Junior's alone).

Created (all under the product root, none in the parent workspace):

- `README.md`, `LICENSE` (GPL-3.0 text + `GPL-3.0-or-later` SPDX grant,
  neutral owner-confirmed copyright placeholder), `pyproject.toml`
  (setuptools, `src` layout, `requires-python >=3.13,<3.14`, `license =
  "GPL-3.0-or-later"`), `.gitignore`, `.gitattributes`.
- `src/zecalibrator/` package: `__init__.py` (version only), `_version.py`
  (literal `0.1.0`), `__main__.py` (CLI only), `cli.py` (`--help`/`--version`
  only), `storage.py` (platformdirs adapter, `appname="ZeCalibrator"`,
  `appauthor="ZeSoftware"`, `version=None`, `roaming=False`, no import-time
  directory creation), `_resources.py` (importlib.resources), `api/v1/`
  (headless namespace, `API_VERSION="1.0"`, no capability), `gui/app.py`
  (isolated PySide6 import; precise missing-`[gui]` diagnostic),
  `resources/icons/` (12 byte-identical packaged icons) + `icon_manifest.json`,
  and exactly one `zesoftware_interop.json` (`provides`/`consumes` empty).
- `tests/` (pytest, 29 tests) and `.github/workflows/ci.yml` (authored locally
  only; remote runs NOT triggered).

Observed validation on Linux x86_64 (Python 3.13.5, setuptools 84.0.0, wheel
0.48.0, build 1.6.1, pytest 9.1.1, platformdirs 4.11.8):

- `python -m build --sdist` → `zecalibrator-0.1.0.tar.gz`
  (SHA-256 `75cd8200281349cbeefdb1c606c86d5c1f62d6653458b498a49ee84cd336fc48`).
- Wheel built from that sdist → `zecalibrator-0.1.0-py3-none-any.whl`
  (SHA-256 `20e5e7c7046e2c0effcd7d7b09406555ed4cef1c7c5688a528580d405ba95129`).
- Installed wheel into a clean venv; `zecalibrator --help`/`--version` and
  `python -m zecalibrator` pass from an unrelated CWD; `zecalibrator-gui`
  emits the missing-`[gui]` diagnostic (exit 1) with no PySide6.
- Headless `import zecalibrator.api.v1` / engine import verified with no
  Qt/ZeAlfie/ZSSS/CuPy/NumPy/Astropy imports; no calibration capability
  advertised.
- 29 pytest tests pass against both source (`PYTHONPATH=src`) and the
  installed wheel. Packaged icon bytes match manifest + canonical SHA-256
  digests in both source and wheel.

**NOT_RUN at this stage (do not claim):** Windows/macOS native filesystem and
GUI qualification, Qt icon decode (PySide6 absent on this host), Python 3.11
interpreter execution (only 3.13.5 available here), and any remote CI run.
The three-OS CI matrix is authored but not executed.


### 17.1 G2 acceptance (2026-09-16)

Phase 2 / Mission 1 (`ZC-P2-M1-BOOTSTRAP-20260915`) was accepted by Junior on
2026-09-16 at branch `chore/zc-p2-bootstrap`, HEAD
`2270536f8567b6a1d9d99db0acd32ec3a76207f5` (tree `21ee3376`), after Coco r0, Junior
independent verification, and Nono review-0 ACCEPT. The bootstrap is an
engineering/package skeleton with no scientific arithmetic; the five G1 owner
policies and the Phase 1 science contract remain the frozen scientific authority.
Windows/macOS native filesystem and Qt icon-decode evidence in this environment,
Python 3.11 execution, and remote CI remain NOT_RUN; they are not claimed as a
three-OS PASS. No push/merge/tag/release/deploy was performed. Phase 3 is prepared,
not dispatched.

### 17.2 Owner-approved G2 platform amendment — 2026-09-16

Tristan explicitly permits **local/bootstrap G2 evidence to remain ACCEPTED**.
This amendment supersedes any earlier interpretation that merely authoring a
three-OS CI matrix satisfied executed native/CI evidence. It is not retroactive
validation: NOT_RUN must never be represented as PASS.

Windows/macOS native execution, Qt native icon decoding, Python 3.11 execution
and remote CI execution remain **NOT_RUN**. No Windows/macOS qualification claim
is allowed. These witnesses remain mandatory before any corresponding platform
support/release claim; TODO.md retains their explicit open register. A local
supplementary offscreen byte decode is not native shell/icon qualification.

The owner authorized a small local commit containing ONLY TODO.md and this
file's G2 closure/amendment, then Phase3 launch from the clean closure revision.
No remote/push/merge/tag/release/deploy/external CI is authorized. G3 implements
the frozen science contract (strict physical raw FITS decode, immutable evidence,
DQ, signed float32 and CPU primitives), with no matching/UI/integrations/GPU/
master-building or Phase4 work. Stop at G3 acceptance/HOLD; do not infer this
G2-only amendment as permission to weaken later gates.
