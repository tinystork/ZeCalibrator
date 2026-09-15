# ZeCalibrator — project state

## Current gate / mission

- **Status: G1 ACCEPTED (Phase 1)**; **G2 (Phase 2 / M1) IMPLEMENTED, AWAITING ACCEPTANCE**, accepting architect: Junior.
- Closed mission: **ZC-P1-M1-CONTRACT-DRAFT-20260915** (G1).
- Previous accepted gate: G0, 2026-09-15 (baseline/raw-boundary witnesses).
- Active/closed mission: **ZC-P2-M1-BOOTSTRAP-20260915** — independent repository,
  src-layout package, GPL metadata, entry points, resources, storage adapter,
  tests and authored CI. Implementation complete; G2 NOT self-accepted.
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
- Working tree: 25 product files; only the eight allowed Phase1 docs/research/
  ledger files changed. No production package, GUI, backend, CI or .git created.
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

## Last ACCEPTED gate — G1

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

## Implemented / delivered

### Phase 1 (documentation and research, G1 ACCEPTED)

- AGENTS.md durable working contract.
- research/phase0/{REPORT.md,evidence.json,raw_boundary_witness.py}.
- docs/{SCIENCE_CONTRACT,ARCHITECTURE,PROVENANCE}.md.
- research/phase1/{REPORT.md,cases.json,contract_witness.py,evidence.json}.
- Owner-policy reconciliation and gate records. No production code.

### Phase 2 / Mission 1 (bootstrap, G2 AWAITING ACCEPTANCE)

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
- Phase2 has no implementation/native/CI witnesses yet. No OS qualification claim.
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

## Next exact mission — G2 acceptance (implementation complete)

Implementation for **ZC-P2-M1-BOOTSTRAP-20260915** is complete; G2 acceptance is
Junior's alone (with Nono review). See
[ZC-P2-M1-BOOTSTRAP-20260915.coco.r0.md](../../.a2a-reports/ZC-P2-M1-BOOTSTRAP-20260915.coco.r0.md)
for the full implementation report, exact commands/results, artifact hashes and
NOT_RUN items.

Pass conditions (G2, not self-accepted): independent Git toplevel/branch/SHA;
sdist and wheel-from-sdist built and clean-installed; help/version/CLI from
unrelated CWD; headless engine/API import without Qt/ZeAlfie/ZSSS/GPU; resource
bytes match manifest/canonical hashes; storage roots injectable with no import
writes; three-OS CI authored (native Windows/macOS execution and Qt decode are
NOT_RUN on this Linux host).

Next phase (Phase 3) is **not** started; it requires accepted G2.

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
