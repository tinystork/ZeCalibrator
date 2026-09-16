# ZeCalibrator — project state

## Current gate / mission

- **Status: G3 TECHNICAL ACCEPTED / GATE HOLD (Phase 3 / Mission 1)** — `ZC-P3-M1-RAW-CPU-20260916`,
  implementation on branch `feat/zc-p3-raw-cpu` (worker DONE, **not** self-accepted).
  Technical work ACCEPTED by Junior after Nono review-3 ACCEPT; G3 gate is HOLD,
  not ACCEPTED, because the mandatory three-OS scientific matrix is NOT_RUN.
  Uncommitted G3 work (no commit authorized); no remote.
- Previous accepted gate: **G2 ACCEPTED (Phase 2 / Mission 1)**, accepting architect: Junior.
- Closed mission: **ZC-P1-M1-CONTRACT-DRAFT-20260915** (G1).
- Previous accepted gate: G0, 2026-09-15 (baseline/raw-boundary witnesses).
- Closed mission: **ZC-P2-M1-BOOTSTRAP-20260915** — independent repository, src-layout
  package, GPL metadata, entry points, resources, storage adapter, tests and CI.
  G2 ACCEPTED 2026-09-16 (Nono review-0 ACCEPT + Junior independent verification).
- Implementation evidence: [Coco r0 report](../../.a2a-reports/ZC-P2-M1-BOOTSTRAP-20260915.coco.r0.md).
  Commit SHA recorded in that report; G2 acceptance is Junior's alone.

## Baseline

- Product root: /home/tristan/.openclaw/workspace/projects/zecalibrator.
- **Independent Git repository: YES** (initialized during ZC-P2-M1-BOOTSTRAP-20260915;
  parent workspace repository was NOT used).
- **Branch:** `chore/zc-p2-bootstrap`. **Full HEAD:** recorded in the Coco r0
  durable report (avoiding a self-referential in-ledger hash).
- Git toplevel is now the product root itself, not the enclosing workspace.
- Source authority: immutable ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md,
  ZC-ARCH-20260915 rev1; ZeSoftware Interoperability Rules v1.0.
- Tracked bootstrap: 61 files, including the independent repository/package,
  tests and authored CI. Closure changes are limited to TODO.md and
  docs/ARCHITECTURE.md; the authorized closure commit provides a clean G3 base.
- 17 protected baseline files (AGENTS, handoff, icons, Phase0 evidence) unchanged.
  All six ecosystem branch/HEAD/status/diff hashes independently unchanged.

| Repository | Branch | Full HEAD | Inherited state |
| --- | --- | --- | --- |
| ZeAlfie | chore/za-tmpfs-cleanup-gate | dd74f8475c8bf8baa35f920b304f58bc6bb3f4e6 | 5 modified + 2 untracked |
| zeseestarstacker | main | 77c92da0259ef7a41fa37d7aa6f3fab7a840735c | tracked clean; research/drizzle_contract untracked |
| zemosaic | beta | c03d0bb965d073b12ad9978094327829f0d0c366 | clean |
| ZeSolver-main | main | 091fbb7a3621a585a45dc110af8a52f870f1fcc9 | clean |
| ZeSolver | test | 47bfa4cb7aabbdd0a486099b06557c16dea681ae | clean |
| zeanalyser | za-perf-p2.0-instrumentation | 38184521997b77553fd854c454065c17ed1ead2a | clean |

## Previously ACCEPTED gate — G1

- Date / authority: 2026-09-15, Junior.
- Technical evidence: Coco r4 DONE + **Nono review-4 ACCEPT**, F1/F2 resolved;
  r4 supplemental correction explicitly authorized by Tristan.
- Owner decision evidence: explicit direct HUMAN_GATE reply selecting all five
  policies; registry ZC-G1-OWNER-20260915 in research/phase1/cases.json.
- Final Junior verification: witness --check --self-test-guard exit 0; strict
  --out replay equals saved evidence except generated_at. Functions for science,
  DQ, FITS, canonical digests, identity and declaration retention unchanged from
  reviewed r4. Only crosscheck_cases and main AST bodies changed for owner
  registry/status reconciliation. Existing 20 matching declarations preserved;
  added bias_ignores_filter = 21 declaration-only cases, not matcher tests.
- Required filter reference checks: dark_ignores_filter => no rejection;
  bias_ignores_filter => no rejection; flat_filter_mismatch_rejects =>
  FILTER_MISMATCH. Registry mutation to generic scientific tolerance 5 °C
  refused by validation (not a claim of implemented temperature matching).
- No unexpected files, no ecosystem changes; no product Git SHA to report.
- Final acceptance report and exact final artifact hashes:
  [G1 acceptance](../../.a2a-reports/ZC-P1-M1-CONTRACT-DRAFT-20260915.g1-acceptance.md).
- Detailed [verification record](../../.a2a-reports/ZC-P1-M1-CONTRACT-DRAFT-20260915.g1-verification.json).
- Nono's r4 ACCEPT covers the technical draft, not a newly executed review of
  Junior's subsequent administrative decision reconciliation. Final reconciliation
  checked and accepted independently by Junior with explicit owner authorization.

## Adopted owner decisions — 2026-09-15

| Policy | Adopted specification |
| --- | --- |
| License | **GPL-3.0-or-later**; no publication authorization |
| First cohort | **SYNTH-BASE-1, synthetic-only qualification** explicitly labelled |
| Temperature | **0 °C scientific tolerance**, parser equality 1e-6 °C separate; any nonzero window requires versioned camera/master profile and real witnesses |
| Flat screening | **≥90% valid normalization samples in each CFA plane** (separate G1/G2), product screening threshold, not physical law; later override by versioned profile |
| Missing metadata | Automatic strict refusal; versioned evidence-backed imports allowed; never invented values or declarations hiding FITS contradictions |
| Filter clarification | Exact for flats; no artificial filter matching requirement for dark/bias when scientifically irrelevant |

Full contracts: [SCIENCE_CONTRACT](docs/SCIENCE_CONTRACT.md),
[ARCHITECTURE](docs/ARCHITECTURE.md), [PROVENANCE](docs/PROVENANCE.md).
These specify future implementation; no executable scientific capability exists yet.

## Last ACCEPTED gate — G2

- Date / authority: 2026-09-16, Junior.
- Branch: `chore/zc-p2-bootstrap`. Full HEAD: `2270536f8567b6a1d9d99db0acd32ec3a76207f5`,
  tree `21ee3376a74ded8958bae8f569d3062297da3afd` (implementation revision).
  A second, explicitly authorized documentation-only closure commit follows; no remote.
- Evidence: Coco r0 DONE + Nono review-0 ACCEPT + Junior independent verification.
- **G2 local/bootstrap ACCEPTED under Tristan's explicit platform amendment
  of 2026-09-16.** This supersedes the earlier overbroad claim that all seven
  original gate criteria passed. Authored CI is not executed CI.
- Windows/macOS native execution, Qt native icon decoding, Python 3.11 execution
  and remote CI remain **NOT_RUN**, never PASS. No Windows/macOS qualification
  claim is permitted; mandatory witness register below remains open.
- 29 pytest tests pass against source and installed wheel; sdist → wheel-from-sdist →
  clean venv install; entry points and `python -m` from unrelated CWD; headless import
  without Qt/ZeAlfie/ZSSS/CuPy/NumPy/Astropy; 12/12 resource bytes match canonical +
  manifest; exactly one interop declaration with empty provides; storage adapter
  overrides + no import-time creation.
- Protected baseline byte-identical to G1-accepted hashes; only docs/ARCHITECTURE.md
  (§17) and TODO.md changed. Six ecosystem HEADs unchanged.
- Non-blocking findings recorded (F-A build-hash reproducibility → Phase 8; F-B gui
  `-m` guard; N-1..N-5). Acceptance report:
  [G2 acceptance](../../.a2a-reports/ZC-P2-M1-BOOTSTRAP-20260915.g2-acceptance.md).

## Implemented / delivered

### Phase 1 (documentation and research, G1 ACCEPTED)

- AGENTS.md durable working contract.
- research/phase0/{REPORT.md,evidence.json,raw_boundary_witness.py}.
- docs/{SCIENCE_CONTRACT,ARCHITECTURE,PROVENANCE}.md.
- research/phase1/{REPORT.md,cases.json,contract_witness.py,evidence.json}.
- Owner-policy reconciliation and gate records. No production code.

### Phase 2 / Mission 1 (local/bootstrap G2 ACCEPTED; platform witnesses open)

- Independent Git repository (product root), branch `chore/zc-p2-bootstrap`.
- README.md, LICENSE (GPL-3.0-or-later), pyproject.toml, .gitignore, .gitattributes.
- src/zecalibrator/ package: version, __main__/cli (--help/--version only),
  storage adapter (platformdirs), _resources (importlib.resources),
  api/v1 (headless namespace), gui/app (isolated PySide6 launcher),
  resources/icons/ + icon_manifest.json, zesoftware_interop.json (provides empty).
- tests/ (29 pytest tests) and .github/workflows/ci.yml (authored, not executed).

## Independently verified

G0: physical BZERO hazard and signed/NaN destruction in actual ZSSS AST-isolated
loader, coherent calibration equations and double-subtraction negative control,
flat-pedestal/four-plane CFA observations, real vs synthetic qualification,
icons/source hashes, unchanged ecosystem state.
G1: tiny equations/DQ/decode/precision/digest/declaration witnesses; F1 binds
light optical facts in plan identity; F2 retains descriptor snapshot and mask
locator independently of LibraryHandle, with distinct byte/declaration/mask
hashes. All-invalid refusal/signed finite values and no double subtraction
specified. Matching remains deterministic/conservative design, not implemented.

Representative arithmetic: both coherent dark branches yield
C=[[100,80],[100,-12]]; double subtraction yields [[96,72],[98,-16]].
No-calibration and partial modes are explicit, not silent role downgrades.

## Remaining / known limitations

- Real camera/master scientific qualification remains OPEN; SYNTH-BASE-1 is
  synthetic-only, not qualification of Seestar S50/S30 or other actual cameras.
- Observed real lights lack qualified detector-instance/offset/readout/ADC facts;
  firmware preprocessing remains uncertain. No defaults inferred from fixture.
- Phase2 local/bootstrap witnesses passed; the mandatory platform register below
  remains NOT_RUN. Local PASS does not establish cross-platform qualification.
- Nono nonblocking research limitations retained: structural helper is not a
  complete matcher (normalized-flat dependency and None/None acquisition checks);
  unknown nested-schema extension policy to make explicit in implementation.
  Cover these at appropriate Phase3/4 contract tests, not by claiming full matching.
- generated_at is volatile; compare semantic replay excluding ONLY this field.
- Original r3 evidence-hash discrepancy is historical; r4 report matched then.
  Closure hashes are in the new final acceptance record, not backdated into r4.
- G0 ROI prose erratum: +1 x shift swaps G1↔R and B↔G2; +1 y swaps G1↔B and
  R↔G2. Immutable Phase0 evidence preserved; SCIENCE §7.5 is authoritative.
- No current G1 blocker; no unchosen owner policy among the five.

## Mandatory platform witness register — owner amendment 2026-09-16

| Required witness | Status | Gate before claim |
| --- | --- | --- |
| Windows native execution/filesystem behavior | **NOT_RUN** | Mandatory before Windows support/release claim |
| macOS native execution/filesystem behavior | **NOT_RUN** | Mandatory before macOS support/release claim |
| Qt native icon decoding | **NOT_RUN** | Mandatory before corresponding GUI/platform support/release claim |
| Python 3.11 execution | **NOT_RUN** | Mandatory before verified minimum-interpreter support/release claim |
| Remote CI execution | **NOT_RUN** | Mandatory before corresponding CI-backed platform support/release claim |

The amendment permits **local/bootstrap G2 ACCEPTED only** with these witnesses
outstanding. NOT_RUN is never PASS; no Windows/macOS qualification is claimed.
No platform obligation is waived or removed, and no G3/G4 requirement is silently
weakened. No external CI/push/publication was authorized by this amendment.

## Next exact mission — Phase 3 authorized (now ACTIVE)

**ZC-P3-M1-RAW-CPU-20260916**: launched from the clean local G2 closure commit
(`84060e945e3307d1dd7138688f5fa6543385ca0d`, tree `fa0f5ca69594e32df328c7a0db7af7db395c0f62`).
G3 remains **not accepted**; the worker's implementation/validation record is below,
never a self-accept. No Phase3 implementation commit is authorized (no commit).
Strict physical FITS decode, immutable metadata/card evidence, uint16 DQ,
signed raw float32 and CPU primitives with explicitly supplied validated masters;
minimal cooperative cancellation/progress. Implement frozen G1, do not redesign.
No library matching/index, GUI, ZSSS/ZeAlfie integration, GPU, master building,
Phase4 work or public capability promotion. Stop at G3 ACCEPTED/HOLD/BLOCKED;
never start Phase4 automatically.

### Phase 3 implementation/validation (worker evidence, NOT acceptance; rework-3)

Branch `feat/zc-p3-raw-cpu` (created from the clean baseline, no commit).
Rework-1 resolved Nono M1–M7 / L1–L3; rework-2 resolved Nono/Junior A–G;
rework-3 (final corrective iteration) resolved R1 (flat bias-range) / M1 (exposure
domain) / M2 (declaration tuple domain) / S5 (normalization-proof coherence).

- `src/zecalibrator/core/`: `dq.py` (five frozen reason bits + reserved-bit
  validation + exact counts), `geometry.py` (exact geometry with explicit
  unknowns, never invented; unknown ≠ unknown; nested tuples frozen),
  `metadata.py` (deep-immutable CardRecord/SensorMetadata/ImportDeclaration with
  recursive freeze incl. numpy→tuple; validated source/identity/version + raw/
  ADU domain; comprehensive alias/duplicate/scaling conflict detection with
  numeric-aware equality + FITS-vs-declaration contradiction enforcement),
  `precision.py` (float32 envelope + float64 reference error), `equations.py`
  (**float32 frame arithmetic** + float64 median + four-plane CFA normalization
  + ≥90% per-plane quality screen computed from the §7.4 median population, with
  the R>1e-6 floor as an independent response guard), `calibrate.py` (float32
  frame pipeline + uint16 DQ + floor on supplied responses + post-arithmetic
  nonfinite canonicalization).
- `src/zecalibrator/io/raw_decoder.py`: strict FITS decode requiring an
  evidence-backed raw-domain declaration (BUNIT=ADU alone is not raw proof);
  ADU-or-declaration units; Jy/electron/rate/normalized rejected; integer BLANK
  stored-space + integrality; NaN/Inf; endian→float32; actual-HDU card source;
  RGB/processed rejection; alias/duplicate/scaling + declaration-contradiction
  rejection; 2**24 integer + 2**53 int64 + float32-magnitude refusal; decode
  progress with isolated observer.
- `src/zecalibrator/application/`: `cancellation.py` (Qt-free cooperative token +
  isolated progress observer) and `executor.py` (explicit master role/bias_state/
  flat_form + normalization proof; exact detector/geometry/ROI/binning/orientation/
  CFA/gain/offset/readout/ADC/units/thermal/exposure validation with NaN/None
  refusal; bias vs qualified bias range; dark vs light, flat-dark vs parent flat;
  per-frame saturation from each frame's own qualified evidence; separate light
  vs flat bias dependencies; already_normalized requires normalization proof;
  no partial mutation on cancel/failure; FAILED never emits committed 100%).
- `pyproject.toml`: added already-accepted scientific dependencies
  `numpy>=1.26` and `astropy>=6` (NumPy/Astropy frozen in G1/G2).
- `tests/science/` + `tests/contract/`: focused science/contract tests
  (**146 total tests pass** incl. the 29 bootstrap tests, Linux x86_64,
  Python 3.13.5, numpy 2.2.4, astropy 7.0.1). Exact commands/results in
  `ZC-P3-M1-RAW-CPU-20260916.coco.{r0,r1,r2,r3}.md`.
- Durable implementation/validation records:
  [Coco r0](../../.a2a-reports/ZC-P3-M1-RAW-CPU-20260916.coco.r0.md) (historical),
  [Coco r1](../../.a2a-reports/ZC-P3-M1-RAW-CPU-20260916.coco.r1.md) (historical),
  [Coco r2](../../.a2a-reports/ZC-P3-M1-RAW-CPU-20260916.coco.r2.md) (historical),
  [Coco r3](../../.a2a-reports/ZC-P3-M1-RAW-CPU-20260916.coco.r3.md) (rework-3, current).



Prepared scope: [Phase3 mission](../../.a2a-reports/ZC-P3-M1-RAW-CPU-20260916.prepared.md).
Activation contract/report baseline records the actual post-closure full SHA.

## Phase 3 / Mission 1 — G3 HOLD (technical ACCEPTED)

- Date: 2026-09-16. Branch `feat/zc-p3-raw-cpu`, HEAD `84060e945e3307d1dd7138688f5fa6543385ca0d`
  unchanged; G3 work uncommitted (no commit authorized); no remote.
- Technical outcome: Nono review-3 **ACCEPT** (no material defect) + Junior independent
  verification. 146 pytest tests pass (isolated roots, one intentional malformed-BLANK warning);
  all prior findings (M1-M7, L1-L3, A-G, R1, M1, M2, S5) resolved. Frozen raw FITS decode,
  explicit-master validation, immutable metadata evidence, uint16 DQ, signed float32 CPU
  primitives, per-frame saturation, quality screen vs response floor, cancellation/progress
  implemented per the frozen G1 contract.
- Gate state: **HOLD**, not ACCEPTED. The mandatory three-OS scientific matrix remains
  **NOT_RUN**: Windows native, macOS native, Qt native icon decode, Python 3.11, remote CI.
  The G2-only owner amendment does not waive G3's three-OS requirement. Qualification remains
  SYNTH-BASE-1 synthetic-only; no real-camera/master claim.
- Nono non-blocking suggestions recorded (G1-G5): light-bias range helper symmetry; non-negative
  finite guard for remaining declaration numerics (gain/offset/saturation/limit); a negative
  test for non-finite qualified saturation limit; optional ARCHITECTURE/PROVENANCE wording for
  bias-range governing declaration and bias_flat key role; keep bias_flat documented as a lookup
  key, not a fifth role.
- Next: provide/execute native Windows/macOS scientific evidence (and Python 3.11 + Qt native
  icon decode + remote CI) to move G3 from HOLD to ACCEPTED; then Phase 4 only with a separate
  activation. No Phase 4 was started. No push/merge/tag/release/deploy.

## DEFERRED

- Raw decoder/CPU primitive G3, matching/library G4, public API G5,
  batch/transactional outputs G6, GUI G7, native packaging G8.
- ZSSS signed preparation + adapter / ZeAlfie admission G9.
- GPU G10; raw reducer/master generation G11; ZeMosaic adapter G12.
- Shared ZeStackCore (unscheduled); destructive in-place calibration (not v1);
  fuzzy matching, dark scaling, arbitrary cropping/rotation/binning;
  cloud libraries, device control, cosmetics/denoise/stretch, uncertainty
  propagation, cross-executable transport, artwork regeneration, signing,
  publication and auto-update.
- Nonzero temperature profile only after real evidence; real scientific support
  only after matched real masters. Neither silently enabled by G1 acceptance.

## Reproducible commands / durable report index

From product root:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tristan/.openclaw/workspace/projects/zeseestarstacker/.venv/bin/python research/phase1/contract_witness.py --check --self-test-guard
```

Use --out <task-owned-temporary-path> for nonmutating regeneration; compare
strict JSON ignoring generated_at only. Default mode regenerates evidence.json.

Reports under /home/tristan/.openclaw/workspace/.a2a-reports/:
- ZC-P0-M1-RAW-BOUNDARY-20260915.coco.r0.md / .nono.r0.md (G0).
- ZC-P1-M1-CONTRACT-DRAFT-20260915.coco.r4.md / .junior.r4.md / .nono.r4.md.
- ZC-P1-M1-CONTRACT-DRAFT-20260915.g1-acceptance.md / .g1-verification.json.
- ZC-P1-M1-CONTRACT-DRAFT-20260915.junior.pre-g1-closure.json archives the
  pre-closure state, including historical BLOCKED/HUMAN_GATE ledger.
- Earlier .final.md records historical r3 BLOCKED, superseded by authorized r4
  technical ACCEPT and this G1 closure. Never treat it as current status.

No git init/commit/push/merge/tag/release/deploy was performed during G1.
