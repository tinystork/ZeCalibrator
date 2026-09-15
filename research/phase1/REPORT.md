# ZeCalibrator — Phase 1 / Mission 1 research report (rework-4)

**mission_id:** ZC-P1-M1-CONTRACT-DRAFT-20260915
**phase:** owner-reconciliation (Junior; technical base rework-4)
**date:** 2026-09-15 (Europe/Paris)
**worker:** Coco
**authority:** `ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md` (ZC-ARCH-20260915 rev 1),
`ZESOFTWARE_INTEROPERABILITY_RULES.md` (v1.0)

This report documents the corrected, decision-ready, technically consistent
contracts and research evidence (technical base rework-4, supersedes r3). No
production code, package, GUI, integration, backend, or master-builder code was
created. All five owner decisions were explicitly adopted by Tristan on
2026-09-15 and reconciled by Junior; **G1 ACCEPTED by Junior, 2026-09-15.**
Nono review-4 ACCEPT covers the technical r4 draft; it is not a separate
review of these administrative owner-decision reconciliation edits.

---

## 1. Summary

Single supplemental correction resolving the two confirmed blockers from
Nono review-3 / Junior r3: **F1** (plan identity omitted the light's optical
identity) and **F2** (imported descriptor/mask declaration retention and
revalidation across library-close). Verified the ecosystem baseline unchanged
and executed the witness with all assertions passing.

---

## 2. Baseline (re-verified)

- ZeCalibrator is **not** an independent Git repo; `git rev-parse
  --show-toplevel` → `/home/tristan/.openclaw/workspace` (parent, no HEAD).
  Branch/SHA **N/A**.
- Ecosystem six repos branch/HEAD/status unchanged from the Junior baseline
  (ZeAlfie 5 M + 2 ??; ZSSS 0 M + 9 ??; others clean).

---

## 3. Correction traceability (rework-4)

### 3.0 F1 + F2 (this iteration)

| Id | Defect (Nono r3 / Junior r3) | Resolution |
| --- | --- | --- |
| F1 | plan identity omitted the light's optical identity (filter / optical_train) | `ARCHITECTURE.md` adds `SensorMetadata.optical: OpticalIdentity` (`filter`, `optical_train_id`, both `known \| unknown`; matching rejects unknown required for flat; dark-only mode does not require them); `PROVENANCE.md` §2.3 adds `light_constraints.optical` to the plan projection; `SCIENCE_CONTRACT.md` §5.1 documents optical binding into plan identity; `contract_witness.py` `_PLAN_PROJECTION` includes `light_constraints.optical`, and `identity_cases` asserts same-input-same-digest, filter-only/train-only/both change digest, incidental locator/tile/path/imported_at/plan_id excluded, and dark-only unknown-optical is defined (no invented filter requirement). |
| F2 | imported descriptor/mask declarations not retrievable/revalidatable after library close | `ARCHITECTURE.md` `MasterBinding` adds immutable `descriptor_snapshot` + `mask_locator` (retrieval, excluded from identity) with `descriptor_id = SHA-256(snapshot)`, mask payload digest distinct from image-byte SHA; no global library lookup / ZeAlfie / FITS-embedded-declaration assumption; tampered snapshot ⇒ recomputed digest ≠ `descriptor_id` ⇒ structured `FAILED`; revision = new snapshot + new id. `PROVENANCE.md` §2.3 documents snapshot/mask retention. `contract_witness.py` adds `declaration_cases` (unchanged rehashes same, tamper rejected, revision-is-new-identity, mask payload identity/tamper rejection). |

### 3.1 Prior corrections (r3, retained)

| Id | Defect (Nono r2 / Junior r1) | Resolution |
| --- | --- | --- |
| M1 | baseline masters lacked gain/offset/readout/adc_mode; light used `adc_depth_bits`; no validity evidence; raw_response contradiction in normalized override; non-hex identities | `cases.json` masters now carry full `acquisition` (gain/offset/readout_mode/adc_mode matching light), `validity_evidence`, structured `ProcessingProvenance` with `additive_correction_history` + `normalization`; light uses `adc_mode`; all content/descriptor/mask identities are valid 64-hex synthetic fixture ids; `normalized_flat_four_scalars_matched` updates history+domain+units+scalars coherently; `flat_corrected_unnormalized` coherent (bias-removed history, no normalization); added missing-required-acquisition negative and normalized-history-contradiction negative. |
| M2 | crosscheck never applied/cloned; shared `det`/`geo` objects aliased | `contract_witness.py` `_materialize_fresh` deep-copies the whole baseline per case (light and each master get independent detector/geometry); `_set_path`/`_del_path`/clone apply operations to actual copies with path validation; `_derive_manifest_reasons` performs a structural audit; independent `_negative_controls` prove no shared mutation, invalid path fails, clone isolation, and normalized-history contradiction detection. |
| M3 | digest helpers were blacklists; reproduced path/imported_at/tile/source_path changing digest | `PROVENANCE.md` §2.3 lists explicit projections; `contract_witness.py` `descriptor_digest`/`plan_digest` use `_project(...)` allowlists; `identity_cases` adds positive assertions (path/imported_at/source_path/tile/locator/plan_id excluded) and negative assertions (content/HDU/mask/history/policy-param/metadata change identity). |
| M4 | `PolicyParameters`/`VersionSet`/`ProcessingProvenance` undefined; `MasterBinding` had no locator; size vs size_bytes | `ARCHITECTURE.md` defines all three types + `NormalizationProvenance`; `MasterBinding` now has `size_bytes` (unified) + `locator: MasterLocator | None` (retrieval-only, excluded from identity); `calibrate_frame` locates masters via plan-bound locator, revalidates content/HDU/mask, structured `FAILED` on stale/missing/changed. |
| S1 | canonical JSON details unpinned | `PROVENANCE.md` §2.5 pins separators/key-order/escaping/number-repr/no-trailing-newline; `digest_cases` asserts a pinned `canonical_json` example. |
| S2 | DQ witness recorded literal FAILED | `dq_semantics_cases` derives counts/NaN/status from masks (`_mask_stats`; `status_is_failed = invalid_count == total`); no literal FAILED. |
| S3 | no repeated-identical-keyword card case | `cases.json` `alias_conflict_cases` adds `repeated_identical_keyword_preserved`; `card_record_cases` in the witness asserts both duplicate cards preserved + agree/conflict diagnostics. |
| S4 | inspection can't represent missing exposure/model | `ARCHITECTURE.md` `Acquisition.exposure_s` and `DetectorIdentity.detector_model` now `\| None` (unknown at inspection); matching rejects unknown required facts. |
| S5 | unknown==unknown relied on shared aliasing | `cases.json` uses explicit independent overrides on `light.detector.detector_instance_id` and `masters.dark_incl.detector.detector_instance_id`. |

---

## 4. Research evidence summary

### 4.1 Equations (executed)
Coherent dark branches `C=[[100,80],[100,-12]]`; double subtraction
`[[96,72],[98,-16]]`; flat-only `[[110,100],[105,-2]]`; flat branches
`[[90,100],[110,80]]` vs bias-only `[[96,106],[116,86]]`; CFA four-plane uniform
vs global-median G1→2.0; shifted-origin GRBG (x: G1↔R + B↔G2, not G1↔G2; y:
G1↔B + R↔G2).

### 4.2 DQ (executed, derived not literal)
Five invalid-reason bits; counts/NaN/status derived from masks; SAT-only and
IN|SAT overlap; all-invalid status derived (`invalid_count == total`).

### 4.3 FITS decode / precision (executed)
BSCALE/BZERO exactly once (uint16-BZERO + scaled-float); integer BLANK stored
space (in-memory); float32 `2^24` boundary.

### 4.4 Digest (executed)
Pinned independent bytes/SHA; byte-order/NaN/signed-zero equivalence; dtype/
shape/data/DQ difference negatives; pinned canonical JSON example.

### 4.5 Identity (executed, explicit projections)
Descriptor: content/HDU/mask/history changes alter digest; path/imported_at/
source_path/descriptor_id excluded. Plan: policy-param (same version), light
metadata, master content changes alter digest; locator/tile/path/imported_at/
plan_id/output excluded.

### 4.6 Matching cases (declaration-only, structural audit — NOT a matcher)
21 matching cases (20 technical r4 cases + bias filter applicability), each deep-materialized, operations applied, and audited
against its declared `MATCHED`/`NO_MATCH`/`AMBIGUOUS` outcome. `NO_MATCH` cases
derive exactly their declared reason codes; `MATCHED`/`AMBIGUOUS` derive none.
A matcher is a Phase 4 deliverable; this is structural reference evidence only.

---

## 5. Owner decisions — adopted 2026-09-15

1. **License:** GPL-3.0-or-later.
2. **Cohort:** SYNTH-BASE-1, qualification **synthetic-only**; real qualification
   deferred until real compatible masters and acquisition evidence are available.
3. **Temperature:** scientific tolerance 0 °C; separate parser tolerance 1e-6 °C.
   Any nonzero window requires a versioned camera/master-specific profile and
   real validation witnesses; no generic nonzero window.
4. **Flat quality:** at least 90% valid normalization samples in EACH CFA parity
   plane as the default **product screening threshold**, never a physical law;
   later replacement requires a versioned profile.
5. **Missing metadata:** automatic strict refusal; versioned evidence-backed
   import declarations may supply unknown facts, never invent values or hide
   FITS contradictions.

**Filter applicability:** exact for flats; no artificial filter criterion for
dark/bias when scientifically irrelevant. All other applicable compatibility
constraints remain unchanged.

---

## 6. G1 checklist traceability

| G1 criterion | State | Evidence |
| --- | --- | --- |
| Equations all branches | specified + reference witnesses passed | SCIENCE §7; witness `equations`, `cfa` |
| DQ bits, CFA scalars/quality, units, precision, saturation | specified + reference witnesses passed | SCIENCE §3,§8,§9; witness `dq`, `fits_decode`, `quality_gate`, `digest`, `identity`, `card_record` |
| Matching matrix + declarative cases | declaration-only (structural audit) | cases.json `matching_cases` |
| Header/domain/HDU, API ownership/errors/cancel/progress, versions | specified design (no production implementation) | ARCHITECTURE §3; PROVENANCE §2 |
| First supported profile | SYNTH-BASE-1; synthetic-only | §5 |
| No invented temp tolerance / detector defaults | adopted zero default | SCIENCE §6.3, §11 item 3 |
| Nono ACCEPT + Junior sign-off | Nono review-4 ACCEPT; Junior closure verified and accepted | owner registry + TODO |

**Real/native NOT_RUN:** no real masters; no native build/GUI/GPU/packaging/ZSSS
tests — explicitly NOT_RUN, not PASS.

---

## 7. Validation executed (exact commands and results)

```bash
cd /home/tristan/.openclaw/workspace/projects/zecalibrator

# 1. witness check + negative guard self-test (no bytecode)
PYTHONDONTWRITEBYTECODE=1 /home/tristan/.openclaw/workspace/projects/zeseestarstacker/.venv/bin/python \
    research/phase1/contract_witness.py --check --self-test-guard
# → CHECK_OK: all Phase 1 witness assertions passed; no file written. (exit 0)

# 2. generate evidence
PYTHONDONTWRITEBYTECODE=1 /home/tristan/.openclaw/workspace/projects/zeseestarstacker/.venv/bin/python \
    research/phase1/contract_witness.py
# → wrote research/phase1/evidence.json (exit 0)

# 3. strict JSON (parse_constant rejects bare NaN/Inf)
# evidence.json + cases.json → OK.

# 4. non-mutating out-path replay
PYTHONDONTWRITEBYTECODE=1 .../contract_witness.py --out /tmp/zcal_p1_r4_evidence.json
# → exit 0; json.load OK; removed.

# 5. source authority asserted by witness guard (mission + interop) → True.

# 6. no __pycache__ / *.pyc produced.

# 7. ecosystem six repos unchanged; immutability hashes match Junior baseline.
```

All witness assertions passed; none failed/skipped. Matching cases are
declaration-only (structural audit, not a matcher); no production tests exist.

New F1/F2 positive+negative cases are in `evidence.json` `identity` (plan-digest
optical) and `declaration` (snapshot/mask revalidation); no matcher/engine test
is claimed.

---

## 8. Files changed (rework-4)

Modified (allowed): `docs/SCIENCE_CONTRACT.md`, `docs/ARCHITECTURE.md`,
`docs/PROVENANCE.md`, `research/phase1/contract_witness.py`,
`research/phase1/evidence.json` (regenerated), `research/phase1/REPORT.md` (this
file), `TODO.md`.

`research/phase1/cases.json` was **not** modified in rework-4 (F1/F2 are model/
projection/retention changes; the case fixture already models filter/optical
and master identities consistently).

`AGENTS.md`, mission doc, `icons/`, `research/phase0/*` byte-unchanged. No commit
created; no ecosystem file touched. r0/r1/r2/r3 reports preserved.

---

## 9. Blockers, uncertainties, next step

**Blockers:** none for G1. Owner decisions reconciled; final independent checks passed.
Real-camera qualification and later implementation/native gates remain separate.

**Uncertainties:** no qualified real masters; no
real/native integration tested.

**Next step:** Phase 2 mission prepared, NOT dispatched. G1 is accepted based
on Nono review-4, explicit owner decisions and final Junior checks; this does
not claim a production engine or real-camera qualification. Present G1 closure
before any Phase 2 launch.

## 11. Owner-decision reconciliation scope and verification

Junior changed policy/status clauses, the five machine-readable selections,
the witness's decision-registry assertions and evidence metadata, and added a
bias-filter declaration. Scientific arithmetic, DQ, FITS reference logic,
identity projections/revalidation and structural audit implementation remain
the reviewed r4 bodies. No new production mechanism or extra Coco rework.
Nono's nonblocking research-helper limitations (normalized-flat dependency /
unknown-acquisition audit coverage; unknown nested extension policy) remain
visible, not claimed as full matcher qualification.

Run: PYTHONDONTWRITEBYTECODE=1 <ZSSS .venv>/bin/python
research/phase1/contract_witness.py --check --self-test-guard;
generate evidence, replay --out temporary path, compare strict JSON except
generated_at. The script validates evidence and cannot accept a gate itself.
Final Junior report and hashes: canonical .a2a-reports/
ZC-P1-M1-CONTRACT-DRAFT-20260915.g1-acceptance.md (final acceptance record).
