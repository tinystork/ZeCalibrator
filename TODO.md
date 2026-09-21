# ZeCalibrator — project state

## Current gate / mission

- **Current: G7 LOCAL TECHNICAL CANDIDATE ACCEPT (technical only)** —
  `ZC-P7-M1-GUI-20260918`, branch `feat/zc-p7-gui`, base/HEAD
  `de6daf862fa5c662d6814d5b31dc322ae6605895` (verified beta == origin/beta;
  P6-M2 closed). G0–G6 remain CLOSED/ACCEPTED. Phase 7 NOT YET ACCEPTED;
  Phase 8 NOT STARTED. No commit/push/promotion/version bump authorized.
  [Prepared mission](../../.a2a-reports/ZC-P7-M1-GUI-20260918.prepared.md).
  Coco implementation complete (rework-3, 3/3 corrective iterations used);
  Junior independent verification + Nono read-only reviews r0→r3 (final ACCEPT).
  Full Qt 497 passed/1 skipped, headless 440 passed/13 skipped, teardown 3/3,
  wheel+source smoke rc 0. Candidate: 27 GUI/tests/docs/CI files, no frozen-layer
  change, no commit yet. Awaiting owner authorization for bounded commit + feature
  push + GitHub Actions. New three-OS CI and interactive GUI witnesses NOT_RUN;
  no platform GUI qualification claimed from old icon/G6 witnesses. G7 final
  platform gate remains HOLD. Local technical acceptance is not G7 acceptance.

- **Status: G6 ACCEPTED + COMMITTED + PUSHED (Phase 6 / Mission 1)** — `ZC-P6-M1-BATCH-TRANSACTIONAL-OUTPUTS-20260917`,
  on branch `feat/zc-p6-batch-transactional` (base `ef2a9f133ee9a957586849da5db7517ddbea5db5` = G5_SHA).
  Functional G6 SHA `9066daa3b86a727d1a64113316aed3e80e32faaf` (Windows mkstemp-fd fix, Nono review-2
  ACCEPT) + G6 closure SHA `a4398d1b2043c2c5f8930e0e5ee73236066d2098` (docs: record G6 acceptance, beta tip).
  402 passed/4 skipped local; three-OS CI run 35235411591 6/6 PASS (ubuntu/macos/windows x 3.11/3.13);
  batch == unitary G5 science, transactional no-clobber outputs + batch manifest + CLI 0/2/3/130.
  Pushed; beta promoted (fast-forward) to a4398d1b. calibrate_batch admitted as capability (P6-M2).
  G1/G3/G4/G5 unchanged/closed. Phase 7 was NOT STARTED at G6 closure.
- **Status: G5 ACCEPTED (Phase 5 / Mission 1)** — `ZC-P5-M1-PUBLIC-API-20260917`, on branch
  `feat/zc-p5-public-api` (base `760324b692c98d07f834a2b3b8ad34e4337ca416`).
  Technical + evidence ACCEPTED 2026-09-17 after Nono review-3 ACCEPT + Junior independent
  verification; 351 passed/4 skipped; five capabilities exposed at G5 (six since P6-M2);
  G1/G3/G4 unchanged/closed. Committed `ef2a9f133ee9a957586849da5db7517ddbea5db5` + pushed; beta
  promoted (fast-forward). (Uncommitted only at acceptance time.)
- **Status: G4 ACCEPTED (Phase 4 / Mission 1)** — `ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916`,
  on branch `feat/zc-p4-library-matching-plan` (base `e22678ceb734052363af79cc0e52e754466ff502`).
  Technical + evidence ACCEPTED 2026-09-17 after Nono review-2 ACCEPT + Junior independent
  verification; 289 passed/4 skipped; G1 projection allowlists byte-identical; G3 stays
  closed/accepted. Committed `760324b692c98d07f834a2b3b8ad34e4337ca416` + pushed; beta promoted
  (fast-forward). (Uncommitted only at acceptance time.)
- **Status: G3 ACCEPTED (Phase 3 / Mission 1)** — `ZC-P3-M1-RAW-CPU-20260916`,
  implementation on branch `feat/zc-p3-raw-cpu`.
  Technical work ACCEPTED (Nono review-3) and platform matrix ACCEPTED (Nono review-0
  of the Qt CI witness). G3 ACCEPTED 2026-09-16. Commit cd05f64 (implementation) +
  e22678c (Qt CI witness).
- Previous accepted gate: **G2 ACCEPTED (Phase 2 / Mission 1)**, accepting architect: Junior.
- Closed mission: **ZC-P1-M1-CONTRACT-DRAFT-20260915** (G1).
- Previous accepted gate: G0, 2026-09-15 (baseline/raw-boundary witnesses).
- Closed mission: **ZC-P2-M1-BOOTSTRAP-20260915** — independent repository, src-layout
  package, GPL metadata, entry points, resources, storage adapter, tests and CI.
  G2 ACCEPTED 2026-09-16 (Nono review-0 ACCEPT + Junior independent verification).
- Implementation evidence: [Coco r0 report](../../.a2a-reports/ZC-P2-M1-BOOTSTRAP-20260915.coco.r0.md).
  Commit SHA recorded in that report; G2 acceptance is Junior's alone.

## Historical baseline — Phase 2 bootstrap

This section preserves the Phase 2 snapshot, including ecosystem states; it is not
the current branch/HEAD. See the Phase 4 activation record for the live baseline.

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
These froze the implementation contracts at G1; the raw decoder/CPU subset is now
implemented and accepted at G3. Library matching is the separately activated G4 work.

## Previously ACCEPTED gate — G2 (historical closure snapshot)

The NOT_RUN statements below describe G2 closure, not current platform evidence;
all five witnesses were subsequently satisfied at G3 (register below).

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

### Phase 2 / Mission 1 (historical: G2 ACCEPTED; platform witnesses then open)

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
- Phase2 local/bootstrap witnesses passed; the mandatory platform witnesses were
  subsequently SATISFIED at G3 by run 35145764088. This does not qualify real cameras.
- Nono nonblocking research limitations retained: structural helper is not a
  complete matcher (normalized-flat dependency and None/None acquisition checks);
  unknown nested-schema extension policy to make explicit in implementation.
  Cover these at appropriate Phase3/4 contract tests, not by claiming full matching.
- R3B (relation matcher): the owner-frozen declarative case
  `unknown_not_equal_unknown` (research/phase1/cases.json) is overridden in
  `tests/contract/test_matching_cases.py` (`_R3B_CASE_OVERRIDES`) — under R3B,
  `detector_instance_id` is a disambiguator, so both-unknown is MATCHED with a
  non-blocking UNVERIFIED note rather than the old hard NO_MATCH. cases.json is
  NOT edited; the divergence is localized to the test adapter.
- generated_at is volatile; compare semantic replay excluding ONLY this field.
- Original r3 evidence-hash discrepancy is historical; r4 report matched then.
  Closure hashes are in the new final acceptance record, not backdated into r4.
- G0 ROI prose erratum: +1 x shift swaps G1↔R and B↔G2; +1 y swaps G1↔B and
  R↔G2. Immutable Phase0 evidence preserved; SCIENCE §7.5 is authoritative.
- No current G1 blocker; no unchosen owner policy among the five.

## Mandatory platform witness register — owner amendment 2026-09-16

| Required witness | Status | Gate before claim |
| --- | --- | --- |
| Windows native execution/filesystem behavior | **SATISFIED** (GitHub windows-latest, run 35145764088) | Mandatory before Windows support/release claim |
| macOS native execution/filesystem behavior | **SATISFIED** (GitHub macos-latest, run 35145764088) | Mandatory before macOS support/release claim |
| Qt native icon decoding | **SATISFIED** (offscreen witness success ubuntu/windows/macos 3.11) | Mandatory before corresponding GUI/platform support/release claim |
| Supported Python runtime | CPython 3.13.x only (owner decision 2026-09-19) | Python 3.11 unsupported due to reproducible native instability in the Qt-worker/Astropy workload |
| Remote CI execution | **SATISFIED** (GitHub Actions run 35145764088) | Mandatory before corresponding CI-backed platform support/release claim |

Runtime-boundary decision 2026-09-19: ZeCalibrator now supports CPython 3.13.x
only. The former "Python 3.11 execution" witness row is retired; Python 3.11 is
unsupported due to reproducible native instability in the Qt-worker/Astropy
workload. This is not a new "Python 3.13 execution SATISFIED" witness: the
three-OS Python 3.13 Qt offscreen GUI suite has not yet passed in CI.

Historically, the amendment permitted **local/bootstrap G2 ACCEPTED only** while
these witnesses were outstanding. NOT_RUN was never PASS. Subsequent G3 CI evidence
satisfies the register above; it does not retroactively change the G2 evidence.
No platform obligation is waived or removed, and no G3/G4 requirement is silently
weakened. No external CI/push/publication was authorized by this amendment.

## Historical activation — Phase 3 (superseded by G3 ACCEPTED below)

The following preserves the original activation constraints and pre-acceptance
worker state. It is not the current gate, commit authorization, or next mission.

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

## Phase 3 / Mission 1 — G3 ACCEPTED

- Date: 2026-09-16. Branch `feat/zc-p3-raw-cpu`.
  Implementation commit `cd05f64e3cd3a9ebbd15cdce6161a8611c306182`;
  Qt CI witness commit `e22678ceb734052363af79cc0e52e754466ff502` (pushed, remote==local).
- Technical: Nono review-3 ACCEPT (no material defect) + Junior independent verification;
  146 tests pass. Frozen raw FITS decode, explicit-master validation, immutable metadata
  evidence, uint16 DQ, signed float32 CPU primitives, per-frame saturation, quality screen
  vs response floor, cancellation/progress per frozen G1.
- Platform: Nono review-0 ACCEPT of the native Qt icon-decode CI witness; GitHub Actions run
  35145764088 all 6 jobs success, Qt witness success on ubuntu/windows/macos Python 3.11.
  All five mandatory platform witnesses SATISFIED (runner-native CI evidence, not Tristan's
  own machines). Qualification SYNTH-BASE-1 synthetic-only; no real-camera/master claim.
- Nono non-blocking suggestions recorded (S1-S4 of the Qt-witness review, G1-G5 of the G3
  science review): ICNS imageCount, document the 3.11-only witness choice, apt-get narrowing,
  optional PySide6 CI pin, plus earlier helper-symmetry/numeric-guard/negative-test notes.
- At G3 closure, Phase 4 was not started. The separate owner-authorized activation
  of 2026-09-16 is recorded below.


## Phase 4 / Mission 1 — G4 ACCEPTED

- Mission: `ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916`, separately authorized by Tristan
  on 2026-09-16. Junior owns architecture/gate; Coco implementation; Nono read-only review.
- Branch: `feat/zc-p4-library-matching-plan`; accepted base and origin Phase3 ref both
  `e22678ceb734052363af79cc0e52e754466ff502`. Only inherited change: this G3-acceptance ledger.
- Preflight: required contracts and G1/G3 acceptance/review reports read; no material
  contradiction with G3 acceptance. Historical sections labelled, not rewritten.
  Original ledger + tracked-file hashes archived under the mission's `.preflight.*` reports.
- Baseline: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3 -m pytest
  tests/contract/test_master_validation.py tests/contract/test_rework3.py
  tests/contract/test_immutability.py -q -p no:cacheprovider` => **36 passed**.
- Scope: immutable descriptors, filesystem-independent library snapshots, pure strict
  coherent-set matching, canonical CalibrationPlan/decision provenance, local versioned
  SQLite adapter and read-only identity/revalidation boundaries. Implement frozen G1.
- Qualification: SYNTH-BASE-1 synthetic-only. No new scientific policy or real-camera claim.
- Design, allowlist, evidence gaps and gates:
  [prepared mission](../../.a2a-reports/ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916.prepared.md).
- Coco implementation-r0 DONE received; 7 new source + 5 new test files, no commit.
  Junior reproduced all 96 new tests plus targeted G3 regression: **158 passed**.
  Independent counterexamples confirm blocking strictness/manual-selection/provenance
  and library lifecycle gaps; **G4 NOT ACCEPTED**.
  Nono review-0 **FINDINGS** (M1-M10), independently confirmed by Junior on 2026-09-17.
  **Active: Coco REWORK-1 DONE** (first corrective iteration), 281 passed/4 skipped.
  Junior independent verification: M1-M10 RESOLVED, one minor residual R1
  (manual selection to a known wrong-role / non-required role -> unexplained NO_MATCH).
  [Junior r1 findings](../../.a2a-reports/ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916.junior.r1.md).
  Nono review-1 **FINDINGS**: M1-M10 resolved; two new material findings N-A
  (light acquisition-profile outside plan identity) + N-B (unroutable flat_dark -> empty
  reason_codes) and confirmed minor R1; all three independently reproduced by Junior.
  [Nono r1](../../.a2a-reports/ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916.nono.r1.md).
  **Active: Coco REWORK-2 DONE** (2nd corrective iteration), 289 passed/4 skipped.
  Junior independent verification: N-A (light profile in plan identity), N-B (unroutable
  flat_dark reason), R1 (manual wrong-role reason) all RESOLVED; G1 projection allowlists
  byte-identical to reference; no new residual defect found.
  [Junior r2 findings](../../.a2a-reports/ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916.junior.r2.md).
  Nono review-2 **ACCEPT** (no material defect); Junior final acceptance checks PASS.
  **G4 ACCEPTED 2026-09-17.** [G4 acceptance](../../.a2a-reports/ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916.g4-acceptance.md).
  [Bounded r2 contract](../../.a2a-reports/ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916.coco.r2.contract.md).
  G1 projections stay frozen; G3 unchanged.
  [Bounded r1 contract](../../.a2a-reports/ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916.coco.r1.contract.md).
  Frozen G1 projection/policies and G3 remain unchanged.
  [Junior r0 findings](../../.a2a-reports/ZC-P4-M1-LIBRARY-MATCHING-PLAN-20260916.junior.r0.md).
  G3 code/tests/contracts remain byte-identical. No Phase5.
- Dispatch: Coco implementation-r0 completed; transport recorded in activation report.
- Next: bounded Coco implementation; independent Junior verification, Nono review,
  corrections if needed, G4 decision and STOP. No commit/push/merge/tag/release/deploy.

## Phase 5 / Mission 1 — G5 ACCEPTED

- Mission: `ZC-P5-M1-PUBLIC-API-20260917`, separately authorized by Tristan on 2026-09-17.
  Junior owns architecture/gate; Coco implementation; Nono read-only review.
- Branch: `feat/zc-p5-public-api`; base `760324b692c98d07f834a2b3b8ad34e4337ca416`
  (== G4 closure commit == beta). Owner A-D: G4 CalibrationRequest/MasterBinding + LightConstraints
  public; get_api_info included; probe deferred; FrameSource + inspect_frame public, DecodedFrame private.
- Scope: public `zecalibrator.api.v1` facade over frozen G3/G4 (models/errors/versioned serialization,
  FrameSource/inspect_frame, open_library, resolve_calibration MATCHED/NO_MATCH/AMBIGUOUS,
  CalibrationPlan/validate_plan/validate_binding, calibrate_frame, public cancellation/progress,
  five interop provides). No SQLite/internal-path contract; core Qt-free; no ZeAlfie/ZSSS/CuPy.
- Cycle: Coco r0 -> Nono review-0 F1-F7 -> rework-1 -> Nono review-1 M1-M6/D1-D4 -> rework-2 ->
  Nono review-2 M7/M8/D5-D8 -> rework-3 -> Nono review-3 ACCEPT. 3/3 corrective rounds.
- Evidence: 351 passed/4 skipped (full), 60 API tests, cold import cheap + five caps, protected
  baseline byte-identical, no G1/G3/G4 edit.
- Qualification: SYNTH-BASE-1 synthetic-only. Committed `ef2a9f13…` + pushed; beta promoted.
  [G5 acceptance](../../.a2a-reports/ZC-P5-M1-PUBLIC-API-20260917.g5-acceptance.md).

## Phase 6 / Mission 1 — G6 ACCEPTED

- Mission: `ZC-P6-M1-BATCH-TRANSACTIONAL-OUTPUTS-20260917`, separately authorized by Tristan on
  2026-09-17 (owner option 2: CLI INCLUDED). Branch `feat/zc-p6-batch-transactional`, base
  `ef2a9f133ee9a957586849da5db7517ddbea5db5`.
- Scope: public `calibrate_batch` (bounded iterator, batch science == unitary G5), transactional
  standalone FITS output (DQ uint16 + CALPROV JSON + no-clobber publication + manifest last),
  batch manifest/provenance, and complete ASTRA CLI (inspect/index/match/calibrate, exit 0/2/3/130)
  as a thin facade over `zecalibrator.api.v1`.
- Cycle: Coco r0 -> Nono review-0 (F-1) -> rework-1 -> Nono review-1 ACCEPT (technical, local Linux)
  -> three-OS witness FAIL (Windows mkstemp-fd leak) -> rework-2 (os.close(fd)) -> three-OS witness
  PASS (run 35235411591 6/6) -> Nono review-2 ACCEPT (closure).
- Evidence: 402 passed/4 skipped (full), 50+ G6 tests, batch parity verified, F-1 resolved (typed
  InvalidRequestError/BatchManifestError, CLI 2/3 no traceback), protected baseline unchanged,
  6 capabilities (calibrate_batch advertised since P6-M2).
- Gate: three-OS filesystem/resource/clean-env witnesses PASS (run 35235411591 6/6). Historical:
  first witness run 35230700912 FAILED on Windows (mkstemp-fd leak); 9066daa fixed it.
- Qualification: SYNTH-BASE-1 synthetic-only. Committed + pushed + beta promoted (functional
  9066daa + closure a4398d1b).

## DEFERRED

- **P8 PERFORMANCE = CLOSED (owner decision, 2026-09-21).** Do not pursue further
  timing improvements. Specifically do NOT reopen: A2 precision instrumentation;
  light-side defensive checks; V2 identity validation; output I/O; GPU/CuPy;
  parallelism — unless a future *measured product requirement* independently
  justifies it.
  **Committed state:** A1 + S1 in `74c72e88aa9709e111094628b1d30d84d370ed6d`
  (feature branch pushed; `origin/beta` untouched; no beta promotion, no tag).
  A1 = prevalidated prepared master forms; S1 = the float32 precondition the fast
  path depends on is now enforced fail-loudly in `_build_prepared_master_forms`
  (never a silent conversion).
  **Final wall-clock (real 10-light dataset, this host, load-dependent):**
  ~610–646 s at the start of P8 → **52 s** after A2 (prepared masters) + A3
  (decoder guard prove-or-vectorise) + A3B (prepared flat) + A4 (decode once) +
  A1 (prevalidated master forms).
  **Final profile (10 lights, ~52 s):** acquisition/decode 15.9 s · calibration
  16.8 s (majority = the precision instrumentation deliberately KEPT) · V2
  identity revalidation 4.2 s · output writing 4.5 s · one-time preparation
  (masters + flat + forms) ~10.3 s · decoder guard on the 2 master planes ~1.2 s.
  **Why we stop:** the remaining time is *consciously chosen* work — real per-frame
  science, instrumentation we decided to keep, safety revalidation (V2) and I/O —
  not architectural waste. There is no remaining structural redundancy; chasing
  further seconds would start trading simplicity for timing.
  **Also recorded:** P8-A1 owner decisions (A2 CLOSED/status quo; ZSSS residency
  RETAIN HOST-CENTRIC) and the P8 STOP CONDITION are in the entries below.
- **CPU in-memory integration — VERDICT A: EXISTING PUBLIC CONTRACT SUFFICIENT**
  (archaeology only, no implementation). An in-memory NumPy consumer can already
  calibrate already-decoded raw frames with no FITS serialisation via the public
  surface: `ArrayFrameSource` (+ public `SensorMetadata`/`ArrayInputIdentity`) →
  `inspect_frame` → `resolve_calibration` → `calibrate_frame` / `calibrate_batch`
  (the latter already prepares masters/flat once per call). This path is exercised
  by the *public* contract tests (`tests/api/test_api_v1_contract.py`,
  `test_batch_contract.py`), so it is public by construction.
  Deliverable = documentation only. Recorded candidates, necessity NOT proven:
  **G1** the Standard zero-question auto-route is private (`api/v1/_auto_route.py`)
  so a consumer must pass explicit modes (still safe: incompatible combinations
  are refused, never downgraded); **G2** `SensorMetadata` construction ergonomics
  (public but verbose; no public "describe my in-memory frame" helper);
  **G3** an explicit in-memory capability identifier (would advertise something
  already true). Do not expose `DecodedLight`/`PreparedCalibrationContext` merely
  because they exist, and do not create prepare/apply public APIs without a proven
  requirement. Report: `.a2a-reports/ZC-P8-CPU-INMEMORY-MINIMUM-CONTRACT-20260921.junior.r0.md`.
- P8 post-A4 owner decisions (2026-09-21):
  **A1 — prepared master defensive work: AUTHORISED (bounded implementation).**
  Masters and the prepared flat response are float32 by contract and frozen since
  A2/A3B, so per-frame master-side work is a per-master invariant. Scope: extend the
  EXISTING `PreparedCalibrationContext` with frozen prepared master forms
  (float32 frame + validated uint16 mask + proven invariants); `calibrate_light`
  keeps ALL its existing checks unchanged for arbitrary/direct callers; only the
  batch path consumes the established invariants. Shape validations stay per frame.
  Estimated ~24 % of `calibrate_light` (~0.65 s/frame, ~6.5 s per 10-light batch)
  with no contract change. Gates: equivalence, direct-caller checks intact,
  reserved-bit/overflow parity, no aliasing, read-only prepared forms, reuse
  call-count, real byte-identical witness, reprofile, frozen/public surfaces
  unchanged; owner acceptance before commit/push.
  **A2 — precision instrumentation: CLOSED with STATUS QUO.** Keep the exhaustive
  per-plane float64 reference, the measured float32 error envelope and the existing
  `PrecisionInfo` semantics. Do NOT add a bound, sampling, an opt-out,
  `precision_measured`, persistence into FITS/CALPROV or a new provenance contract:
  the object is an in-memory diagnostic, no product requirement for persisting or
  consuming it has been demonstrated, and no new contract will be created merely to
  optimise an internal diagnostic. Future science/product decision if a real
  consumer or persistence requirement appears: decide the scientific statement
  first, then its measurement/proof mechanism.
  **ZSSS GPU residency: RETAIN HOST-CENTRIC** (accepted as the ZSSS architectural
  position; recorded in the ZSSS ledger at
  `zeseestarstacker/docs/gpu_residency_decision.md` with the three reopen triggers
  and the mandatory preconditions). No GPU-resident pipeline state for
  ZeCalibrator; no CuPy backend on an assumption of future ZSSS residency.
- P8 STOP CONDITION (owner, 2026-09-21): after A1 is accepted and reprofiled, STOP —
  do NOT automatically start another performance optimisation. At that point report
  the complete new wall-clock and cost decomposition so the owner can decide whether
  P8 performance work is finished and the project returns to product functionality /
  the CPU in-memory integration capability admission. Rationale: ~610–646 s at the
  start of P8, ~67 s before A1; chasing a further few seconds would turn a successful
  optimisation campaign into a distraction.
- P8-A4 follow-ups (deferred/informational by owner — no A4 rework): **O1** the
  `DecodedLight` carrier is built even when a later stage fails (routing
  `NO_MATCH`), costing exactly the acquisition inspection already paid; **O2**
  array `facts` (`scaling_applied`/`domain`) and FITS `facts` (`{bscale,bzero}`)
  keep their pre-A4 shapes (no diagnostic drift); **O3** with the carrier
  supplied, the calibrate-side decode-failure mapping is unreachable on the batch
  path by construction (standalone `calibrate_frame` unchanged); **O4** the
  reprofile re-ranking.
- P8 GPU-RESIDENT INTEGRATION FEASIBILITY (owner-recorded 2026-09-21 —
  ARCHAEOLOGY/DESIGN ONLY, no implementation, no CuPy, no public API, no ZSSS or
  ZeCalibrator change). Question: can ZeCalibrator expose a backend-neutral
  calibration boundary letting an already-resident NumPy **or** CuPy raw frame
  stay resident through calibration, with NumPy as the canonical scientific
  reference? Two related areas:
  **A — precision/validation cost contract**: for `_float64_reference`,
  `measure_float32_error`, `_f32_checked`, `_master_mask` and the related
  defensive copies/conversions, characterise the scientific requirement, the
  provenance/diagnostic requirement, the mutation-safety requirement, whether it
  must run per frame, whether it can be proven once or structurally, and whether
  changing it would weaken an existing contract. **Do not optimise it yet.**
  **B — ZSSS / GPU-resident boundary**: study the current public/private array
  path and the actual ZSSS GPU frame representation; design a future public,
  versioned integration API accepting CPU-resident NumPy and GPU-resident CuPy
  raw frames with no mandatory CPU<->GPU round trips.
  Requirements: ZeCalibrator independently installable; no dependency on ZSSS;
  ZSSS consumes only a documented public contract; CPU/NumPy canonical
  reference; CuPy optional; no silent CPU fallback when GPU execution is
  explicitly requested; prepared masters/flat/masks transferable/preparable once
  and reusable on device; the calibrated result may remain GPU-resident for
  downstream ZSSS processing; DQ/mask/NaN/scalar/provenance semantics equivalent
  to CPU; no FITS serialise/re-read boundary; no ZeAlfie dependency;
  Windows/Linux/macOS CPU behaviour unchanged when CuPy is unavailable.
  Distinguish explicitly: (1) standalone ZeCalibrator performance, (2) CPU
  in-memory integration, (3) GPU-resident ZSSS integration — and estimate the
  transfer/allocation boundaries for each. Return current data-flow diagrams,
  proposed CPU and GPU-resident flows, the contract/API decisions that would
  eventually be required, parity gates, dependency/packaging consequences,
  expected transfer counts, which measured costs could disappear or move on GPU,
  and whether a GPU backend is architecturally justified.
  Context (owner): ZeCalibrator standalone almost certainly does not need the GPU;
  ZeCalibrator as the scientific engine embedded in ZSSS could benefit greatly
  precisely because the GPU is already there — this distinction avoids
  complicating the application while preparing a real high-performance backend
  where it makes sense.
- P8-A3B follow-ups (deferred/informational by owner — no A3B rework): **O1**
  `_build_prepared_flat` captures only `InvalidRequestError`/`GeometryMismatchError`,
  so a genuinely unexpected exception type would surface at frame-1 context build
  rather than at the per-frame flat stage (today it would also propagate; the
  timing differs); **O2** the eager build runs before `_freeze_master` (verified
  harmless: copies + freshly allocated `norm` arrays, no aliasing into a master
  frame); **O3** the seam's route guard can only mismatch on caller error (the
  facade always passes `context.flat_prep_mode`); **O4** the reprofile re-ranking.
- P8-A4 — DECODE ONCE (owner-authorised for DESIGN/ARCHAEOLOGY only, 2026-09-21):
  remove the redundant per-light FITS read/decode between `inspect_frame` and
  calibration in the batch path. Target shape: `file -> decode once -> canonical
  private DecodedFrame -> {inspection/routing, calibration}`. Preserves the
  public `calibrate_frame` API, standalone behaviour, raw-decoder semantics, light
  identity/hash/provenance truthfulness, validation/failure/cancellation
  precedence; no implicit/global/persistent cache; no science/DQ/equation change;
  no GPU; no parallelism; no public API/capability change. Study whether the
  existing `ArrayFrameSource`/decoded representation can serve as the internal
  seam instead of a competing representation. Anticipate (do NOT implement) a
  future integration boundary where a consumer (e.g. ZSSS) supplies an
  already-decoded in-memory frame with no FITS re-serialise/re-read cycle.
- P8-GPU-FEASIBILITY gate (recorded by owner 2026-09-21 — analysis only, NOT
  authorised for implementation): after A4 + reprofile, decompose the remaining
  per-frame calibration cost into actual calibration equations vs DQ/mask work vs
  float32/float64 conversions and copies vs float64 reference computation vs
  precision/error measurement vs other material array operations. Only then
  decide between further CPU/vectorised work, redesigning the precision
  instrumentation without weakening contracts, or an optional CuPy backend.
  Constraints for any future GPU design: CPU remains the canonical scientific
  reference; analyse specifically the ZSSS GPU-resident case (frame and prepared
  masters already in VRAM) to avoid unnecessary VRAM<->RAM round trips; a ~6.5
  Mpx single-frame transfer/allocation overhead must not be assumed away. Do not
  put a GPU under work that can still be deleted (hence A4 first).
- P8-A3 follow-ups (deferred by owner, non-blocking — do NOT rework the accepted
  A3 commit for these): **O1** `core/precision.exceeds_float32_exact_bound`
  re-applies `np.isfinite` to an array both callers already filtered (idempotent;
  one redundant pass on the fallback path only); **O2** the
  `RuntimeWarning: invalid value encountered in multiply` under `BSCALE=0` with
  non-finite storage is pre-existing (raised in `physical = bscale*stored_f64 +
  bzero`, upstream of the guard) and unchanged; **O3** characterisation: the C1
  magnitude check is unreachable from float32 storage (it needs `BITPIX=-64` or a
  large `BSCALE`), since float32 storage cannot exceed `float32max`.
- P8 follow-up candidates (recorded after the A3 reprofile, NOT authorised for
  optimisation yet): duplicate light decode per frame (`inspect_frame` +
  `calibrate_frame::_decode_light` decode the same bytes — the former "A4");
  precision roundoff/error measurement (`measure_float32_roundoff` /
  `measure_float32_error`); repeated per-frame master mask validation
  (`core/dq.validate_mask` in `calibrate_light`); output/write cost. Measured
  shares of the 10-light batch after A3: flat normalization 27.6 s (28.8 %),
  light decode 23.3 s, calibration arithmetic 23.1 s, output write 6.3 s,
  `validate_plan` 4.8 s, masters 3.6 s (total ~95.8 s wall; host-load dependent).
  Note: `tottime` under cProfile is NOT reliable wall-clock attribution after the
  A3 call-count collapse — prefer the instrumented per-stage timings.
- P8 trajectory note (owner, 2026-09-21): parallelism is **removed from the
  mandatory roadmap** (kept as a tool only). Sequence: A3 commit -> Stage 2
  prepared flat response (design gate first) -> reprofile -> A4 -> reprofile ->
  product decision ("is ZeCalibrator fast enough yet?"). 10 real lights went from
  ~10 min to ~96 s over A2+A3 on this host.
- P8-A2 Stage 1 follow-up (deferred by owner, cosmetic): in
  `api/v1/calibration.py::_calibrate_frame_impl` the `load_masters` progress
  event is emitted *before* `_flat_prep_mode(plan)` is evaluated, so a plan with
  an unsupported flat form emits one extra (harmless) `load_masters` event
  before the propagating `InvalidRequestError`. Error path only; failure
  precedence and `executed_processing` are unchanged. Do **not** modify the
  accepted Stage 1 commit for this; fold into a future UX/cleanup pass (fix is a
  one-line hoist of the emit after the `_flat_prep_mode` call). Recorded from the
  Nono review-0 observation O-new-1 of mission `ZC-P8A2-PREPARED-CONTEXT-STAGE1-20260921`.
- Native packaging G8 (G7 now separately activated; see current mission).
  G3/G4/G5/G6 are ACCEPTED.
- GUI `Settings > Language` selector — future separate micro-phase only (the
  repository has no ``QTranslator`` / ``.qm``/``.ts`` translation resources or
  ``tr()`` routing; no i18n framework is built).
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
