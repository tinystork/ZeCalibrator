# ZeCalibrator — Provenance (specified v1)

**Status:** Phase 1 specification — owner decisions reconciled 2026-09-15;
**G1 ACCEPTED by Junior, 2026-09-15.** Schema/digest/transaction specifications
do not claim a delivered runtime or output writer.

Authority: `ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md` §10 and
`ZESOFTWARE_INTEROPERABILITY_RULES.md` (Rules 12, 16, 25 — immutable revisions
and artifact identity).

---

## 1. Schema and versioning

- Specified provenance-projection schema: `zecalibrator.provenance.v2`
  (P7-M3B R3D-A; see §9.1 old-data rule). This is distinct from the record
  *document* schema (`schema_version`), which remains `zecalibrator.provenance.v1`.
- Specified library index schema: `zecalibrator.library.v1`.
- Specified matching-policy version: `zecalibrator.match.v1`.
- Specified digest canonicalization scheme: `zecalibrator.digest.v1` (§2.4).
- These are **distinct** from product version, API version, and
  science-contract version.
- The record carries its own `schema_version`; a parser must refuse an
  unsupported major.

### 1.1 Required fields (specified)

- `operation_id`; product/API/schema versions; installed artifact/source
  revision when available (unknown is explicit, never fabricated).
- Request mode and requested roles; input identity, HDU, shape, dtype, physical
  units, original scaling facts.
- Selected library index revision, calibration-plan id/hash, policy/profile
  versions.
- Master content SHA-256, size, HDU, descriptor identity, optional
  original-source population, semantic processing states.
- Allowlisted original header values, normalized values, alias decisions, user
  declarations, tolerances, rejected-candidate reasons.
- Equations/mode, flat normalization algorithm/population/scalars,
  invalid-mask meanings/counts.
- Requested/available/matched/executed/skipped/failed/cancelled facts, warnings,
  reason codes.
- CPU/GPU actual execution and fallbacks.
- Output identity, units, dtype, invalid counts, commit status, timestamps.

---

## 2. Identity and hash canonicalization

### 2.1 Content identity is not a path (INVARIANT)

A path+mtime is a cache hint, not content identity. Selected masters require
content-hash validation or an explicitly equally strong immutable identity.

### 2.2 Named, distinct identities (INVARIANT)

- **Whole-FITS hash** (raw bytes) and **decoded-payload digest** are distinct,
  named identities.
- Include mask/descriptor identities, not image pixels alone.
- For caller-owned arrays, accept the caller identity plus its declared strength;
  compute a canonical decoded-data digest where needed. **Never claim a FITS
  byte hash for an array.**

### 2.3 Acyclic digest scopes (INVARIANT — explicit projections, not blacklists)

Three acyclic digests, each over an **explicit projection** (an enumerated
allowlist of field paths). No digest is an input to itself and no digest of an
output appears inside that output. The reference helpers in
`contract_witness.py` implement these exact projections (not
"all-fields-minus-blacklist").

1. **Science digest** — `SHA-256(CANONICAL(data) ‖ CANONICAL(DQ))` over the
   calibrated float32 data plane and the uint16 DQ mask only. It never includes
   the provenance record.
2. **Descriptor digest** — `SHA-256(CANONICAL_JSON(projected_descriptor))` over
   this **explicit projection** (field paths, in this order):

   ```
   master_type, bias_state, pixel_domain, physical_units, flat_form,
   normalization_algorithm, normalization_scalars, geometry,
   detector.detector_instance_id, detector.detector_model, detector.serial,
   acquisition.gain, acquisition.offset, acquisition.readout_mode,
   acquisition.adc_mode, acquisition.temperature_c, acquisition.exposure_s,
   acquisition.saturation_limit_adu, acquisition.saturation_evidence,
   optical_train_id, filter, content_sha256, size_bytes, hdu, mask_identity,
   processing_provenance, validity_evidence
   ```

   It **excludes** `descriptor_id` (derived from it) and any retrieval
   locator/path/timestamp/imported_at field.
3. **Plan digest** — `SHA-256(CANONICAL_JSON(projected_plan))` over this
   **explicit projection**:

   ```
   request.additive_mode, request.flat_mode,
   light_constraints.{geometry, detector, acquisition, optical, raw_domain_declaration},
   policy_parameters.{exposure_tolerance, temperature_tolerance, flat_quality_policy},
   masters.{role}.{descriptor_id, content_sha256, size_bytes, hdu, mask_identity},
   versions.{science_contract, decoder, provenance_schema, matching_policy}
   ```

   It **excludes** `plan_id`, any output digest, any provenance record, and —
   critically — `masters.*.locator` (retrieval locator) and execution fields
   (`tile_shape`, `source_path`, `imported_at`, backend/execution strategy).

**What counts vs what is incidental (documented):** science-relevant metadata
(geometry, detector/acquisition identity, semantic processing history, validity
intervals, exposure times, normalization scalars, content/HDU/mask identity, and
actual policy parameters) **does** affect identity. Incidental retrieval/location
facts (path, imported_at, source_path, tile strategy, locator) and derived
self-ids/output digests **do not**. Note that a scientifically meaningful
*validity interval* or *exposure time* is **not** excluded merely because its
name contains "time"; only explicitly enumerated incidental fields are excluded.

Consequences (asserted in `contract_witness.py` `identity_cases`): adding/changing
`path`, `imported_at`, `source_path`, `tile_shape`, or a nested master `locator`
does **not** change descriptor/plan digest; changing a policy parameter (same
version), geometry/acquisition, **the light's optical identity (filter /
optical_train_id)**, descriptor content/HDU/mask/history **does** change it. Two
policies with the same version but different thresholds produce different plan
digests.

**Descriptor-snapshot and mask retention (F2):** the plan also carries, per
master, an immutable **descriptor snapshot** (the complete acyclic descriptor
record, whose digest is `descriptor_id`) and a **mask locator/payload identity**
(`mask_identity`). These are retained for revalidation across `LibraryHandle`
closure and are **excluded from the plan digest** as retrieval artifacts; their
**digests** (`descriptor_id`, `mask_identity`) are part of the identity. The
image-byte `content_sha256`, the descriptor `descriptor_id`, and the mask
`mask_identity` are three distinct hashes, never conflated. Imported processing/
metadata declarations are **not** assumed to live inside the FITS bytes; they
survive via the retained snapshot. A tampered snapshot recomputes to a different
`descriptor_id` ⇒ rejected; declaration revision is a new snapshot + new id, not
an in-place mutation.

### 2.4 Canonical byte serialization — `zecalibrator.digest.v1`

For an array `A` of shape `S = (s0, s1, …)` and dtype `d`:

1. **Order:** C order (row-major, last axis fastest). The array is treated as
   contiguous.
2. **Byte order:** fixed **big-endian** (`>`) for every element, never native.
3. **NaN canonicalization:** replace every NaN and ±Inf with a single fixed
   big-endian quiet-NaN **bit pattern given as literal bytes**, not derived from
   host `float('nan')` encoding:
   - float32 → `7F C0 00 00`;
   - float64 → `7F F8 00 00 00 00 00 00`.
4. **Signed zero:** `-0.0` normalizes to `+0.0` (all zero bits); the digest does
   not distinguish them.
5. **Record and length prefix:**

```
dtype-token  ∈ {float32, float64, uint16, uint8, int16, int32, uint32, int64, uint64}
             (a fixed literal ASCII token per dtype; byte order never appears in the token)
record = "ZCALDIG1\n" ‖ dtype-token ‖ "\n" ‖ "s0,s1,…" (ASCII, comma-sep, no spaces) ‖ "\n"
         ‖ big-endian normalized data bytes
CANONICAL(A) = uint64_be(len(record)) ‖ record
```

The **magic, dtype token, and shape are part of the hashed bytes** (they are in
`record`, which is length-prefixed). Science digest concatenates two
`CANONICAL` buffers, each self-delimiting via its length prefix.

Consequently (asserted in `contract_witness.py`):

- Two arrays differing only in **native byte order** hash equal (both convert to
  big-endian).
- Two arrays differing only in **NaN payload bits** hash equal (fixed pattern).
- Two arrays differing in **dtype** or **shape** hash **differently** (dtype
  token and shape ASCII are in the record) — these are regression negatives that
  must fail the prior omitted-header behavior.
- Two arrays differing in any **data value** or in the **DQ mask** hash
  differently.

**Pinned independent reference example** (asserted verbatim, not derived from
the implementation under test):

- `CANONICAL(float32 [[1.0, -2.0]])` =
  `000000000000001d5a43414c444947310a666c6f617433320a312c320a3f800000c0000000`
- `CANONICAL(uint16 [[0, 0]])` =
  `00000000000000185a43414c444947310a75696e7431360a312c320a00000000`
- `science_digest(float32 [[1.0,-2.0]], uint16 [[0,0]])` =
  `17761d64f6ce235dca8e9e5afa22645da80deed63bee83237adea571997660af`

### 2.5 Canonical metadata JSON (descriptor/plan digests)

`CANONICAL_JSON` is a deterministic subset of JSON used only for descriptor and
plan digests. The exact rules (pinned, matching the reference implementation in
`contract_witness.py`):

- **Separators:** compact `(item_separator=",", key_separator=":")` — no
  spaces; an object is `{"k":v,"k2":v2}` and an array `[v1,v2]`.
- **Key ordering:** object keys sorted by Unicode code point (byte order),
  ascending.
- **Strings:** UTF-8, `ensure_ascii=False`, JSON-escaped (embedded quotes,
  backslashes, control chars, and non-ASCII preserved as UTF-8).
- **Numbers:** finite only (no NaN/Inf — they raise); integers serialized as
  decimal integers; floats via Python `repr` (shortest round-trip).
- **Lists/tuples** preserve declared order.
- **`null`** for explicit unknown; **no trailing newline** on the final string.
- No serializer framework is introduced; the reference routine is a hand-written
  recursive function that must remain byte-identical to this spec.
- `null` for explicit unknown; **never** a silent default value.

### 2.6 No TOCTOU ambiguity (INVARIANT)

Hash at planning and revalidate at execution to detect edits after planning.
Protect read/hash with one of: read stable bytes once and hash those exact
bytes; or verify metadata/content did not change while hashing/reading (size,
mtime, content digest). A stale index is not evidence of unchanged data.

### 2.7 No hash self-reference (INVARIANT)

A file cannot truthfully contain its own final whole-file hash.

- Embedded provenance records an output **logical id** and the **science digest**
  (§2.3 item 1), never the final whole-file SHA-256 and never a digest of the
  provenance record itself.
- The final whole-file SHA-256 goes in the **external batch manifest after
  output closure**.
- The logical id is computed from input identity + plan id (collision-safe),
  never from the output's own final bytes.

---

## 3. CALPROV extension byte representation (concrete)

The `CALPROV` extension is a **1D `uint8` ImageHDU** whose data array is the
**UTF-8 bytes of the strict-JSON provenance record**:

- The record is serialized with `json.dumps(record, ensure_ascii=False,
  allow_nan=False, separators=(",", ":"))` and encoded UTF-8, then stored as a
  `uint8` array (one byte per element).
- `allow_nan=False` guarantees strict JSON (no bare `NaN`/`Inf`); non-finite
  values are stringified before serialization.
- `EXTNAME = 'CALPROV'`, no BSCALE/BZERO/BLANK, no CHECKSUM/DATASUM at write time
  (whole-file checksums recomputed after closure).
- The record contains the **science digest** (§2.3) and the output **logical
  id**, but **not** a hash of the CALPROV extension itself and **not** the final
  whole-file hash. No circular hash reference.
- `CALPROV` decodes by reading the uint8 array, joining to bytes, and
  `json.loads` with strict parsing (`parse_constant` rejects bare non-finite).

Full matching reports are not truncated into FITS keyword strings; the full
structured record lives in `CALPROV`, while short `HIERARCH ZECAL…` fields carry
only human-readable summaries.

---

## 4. FITS / standalone / embedded persistence

- **Embedded (array API):** complete structured provenance in memory; no
  implicit log/sidecar write by the core; no compulsory intermediate FITS.
- **Standalone output FITS:** short `HIERARCH ZECAL…` fields/history, a **DQ
  image extension**, and a **CALPROV UTF-8 JSON payload extension** (§3).
- **Batch manifest** indexes outputs, committed files, failures,
  pending/skipped/cancelled dispositions, and final whole-file hashes.

---

## 5. Transactional no-clobber output (INVARIANT)

- Default output uses a distinct user-selected directory and deterministic
  collision-safe names based on input identity + plan id.
- Never overwrite a light/master or an existing unrelated output, including
  symlink/hardlink aliases and case-insensitive collisions.
- Write into an **exclusive temporary file in the destination filesystem**,
  flush/close, validate, then commit via a platform-appropriate **no-clobber
  publication primitive**. `os.replace` alone is not a no-overwrite guarantee.
  Close FITS handles before rename (Windows).
- Single output FITS with embedded provenance is the atomic unit. Commit
  manifest **last**; restart can reconcile verified committed files.
  Cross-filesystem rename and network-filesystem atomicity are not assumed.
- On cancellation/crash remove only task-owned incomplete temporary files;
  preserve committed outputs and user masters. Disk-full/provenance-serialization
  failure is **not** success.
- Science-critical provenance is part of the output transaction. Optional human
  logs may fail gracefully; required provenance may not silently disappear.

> No native publication primitive is implemented in this phase; the above is the
> design contract for Phase 6.

---

## 6. Logs and redaction

- Logs default to bounded rotating local files; UI uses basenames/logical IDs.
- Detailed local audit records may contain explicit user paths. Export/redaction
  avoids leaking site coordinates or unrelated header personal data. No
  telemetry/cloud uploads.

---

## 7. Schema-version and policy-decision linkage (specified)

Each provenance record links:

- the **schema version** it was written with;
- the **science-contract version** and **matching-policy version** that governed
  the equations and selection;
- the **product/API versions** of the code that produced it;
- the **policy decisions** in force (flat quality threshold, temperature
  tolerance, missing-metadata declaration policy, license-visible cohort label)
  by named version and **by their actual parameter values** (not version alone),
  so a later policy change cannot silently re-interpret old outputs.

Tristan adopted the five owner policies on 2026-09-15 (SCIENCE §11,
ARCHITECTURE §14; machine-readable selections in research/phase1/cases.json).
Persist actual effective values and their decision/profile source: synthetic-only
cohort SYNTH-BASE-1, zero scientific temperature tolerance (parser tolerance
separate), 90% per-plane product screening, and strict automatic refusal with
versioned evidence-backed declarations that cannot override FITS contradictions.
A future unresolved/unrecognized profile remains pending/refused, never silently
interpreted as an adopted default. Real-camera qualification is not implied.
The selected license is GPL-3.0-or-later; this record is not publication approval.

---

## 8. Transaction / resume identity

- Checkpoint/resume fingerprint includes selected identities, equations,
  matching policy **parameters**, decoder/domain contract, and mask semantics.
- Resume refuses changed masters or a changed scientific calibration plan.
- Backend/tile strategy is recorded separately as execution (parity proven).

---

## 9. Versioning summary

`zecalibrator.provenance.v2` is the current specified schema target (bumped from
`v1` by P7-M3B R3D-A D1d). Breaking schema changes require a new major; additive
optional fields must not change semantics of existing required fields.
Immutability of committed records is a hard requirement (Interop Rule 16).

### 9.1 Old-data rule (v1 -> v2)

`ProcessingProvenance` gained the explicit discriminator `additive_history_state`
(`"unknown"` | `"known"`). Legacy records WITHOUT that key read back as
`"unknown"`, never a fabricated known-empty state. A legacy record carrying a
non-empty `additive_correction_history` but no state is rejected at
reconstruction (NO compatibility bypass). Because `additive_history_state` joins
the hashed descriptor projection and `VersionSet.provenance_schema` joins the
plan projection, old descriptor/plan ids may fail strict re-verification against
a v2 schema — that is expected and accepted for this pre-beta branch.
