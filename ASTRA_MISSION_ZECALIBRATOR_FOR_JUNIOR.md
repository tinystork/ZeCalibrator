# ASTRA → JUNIOR — ZeCalibrator architecture and sequential implementation mission

**Document:** ZC-ARCH-20260915, revision 1  
**Archaeology date:** 2026-09-15, Europe/Paris  
**Recipient / downstream owner:** Junior  
**Product owner:** Tristan Nauleau  
**Status:** architecture handoff; NO ZeCalibrator implementation performed  
**Workspace:** /home/tristan/.openclaw/workspace/projects/zecalibrator

This document is self-contained. Conversation history is not a prerequisite or an authoritative project ledger. Paths to ecosystem checkouts below are archaeology locations, NEVER application runtime paths. This mission does not authorize publication, remote pushes, releases, deployments, destructive changes, or unsolicited communication.

## 1. Executive objective and non-negotiable boundaries

Create **ZeCalibrator**, an independently installable application and reusable scientific calibration engine for **raw astronomical sensor FITS frames**. The initial useful release applies existing masters; it does not build them.

Two clients use the same application services:

1. Standalone CLI / **PySide6** GUI: select lights and a user-owned calibration library, inspect proposed matches, produce new calibrated FITS outputs.
2. Embedded consumer: ZSSS calls a versioned public Python API and receives calibrated raw/CFA float32 arrays and provenance without requiring intermediate calibrated FITS files.

ZeAlfie remains installer, updater, channel/compatibility/activation/runtime/launch/provenance/rollback orchestrator. It is **not a runtime data broker**. ZeCalibrator works without ZeAlfie or ZSSS; ZSSS works without ZeCalibrator. Products communicate directly through public contracts.

Scientific ordering:

    FITS storage → physical raw sensor values + original metadata
        → calibration master resolution → sensor calibration
        → signed calibrated raw/CFA float32 + validity + provenance
        → consumer-owned debayer / colour / registration / stacking

Never calibrate already-normalized or debayered data. Do not rotate, flip, transpose, crop, resample, register, reproject, replace hot pixels, white-balance, stretch or clip negatives within calibration. Preserve the correspondence array[y, x] ↔ physical sensor pixel (x, y).

CPU is the scientific reference. GPU may change execution strategy only. Matching is deterministic and conservative. Missing compatible masters is preferable to a dubious match. Preserve original files by default.

**Execution discipline:** one bounded mission and one accepted gate at a time. A roadmap entry is not authorization to implement it early. Phase 0 below is documentation and research evidence only.

## 2. Evidence authority and repositories inspected

### 2.1 Exact baseline

All ecosystem paths in this table are under /home/tristan/.openclaw/workspace/projects/.

| Repository | Branch examined | Exact local HEAD | Product version observed | Working-tree qualification |
| --- | --- | --- | --- | --- |
| ZeAlfie | chore/za-tmpfs-cleanup-gate | dd74f8475c8bf8baa35f920b304f58bc6bb3f4e6 | 0.1.1, pyproject.toml | Dirty unrelated testing/witness work; see below |
| zeseestarstacker | main | 77c92da0259ef7a41fa37d7aa6f3fab7a840735c | 8.5.2, seestar/__init__.py | Tracked source clean; untracked research/drizzle_contract work |
| zemosaic | beta | c03d0bb965d073b12ad9978094327829f0d0c366 | 4.7.0, src/zemosaic/__init__.py | Clean |
| ZeSolver-main | main | 091fbb7a3621a585a45dc110af8a52f870f1fcc9 | 1.2.1; public API 1.2 | Clean; published-branch API convention used |
| ZeSolver | test | 47bfa4cb7aabbdd0a486099b06557c16dea681ae | Not used as published API authority | Clean; development branch differs from main |
| zeanalyser | za-perf-p2.0-instrumentation | 38184521997b77553fd854c454065c17ed1ead2a | 3.4.0 | Clean; packaging/resource conventions inspected |
| zecalibrator | No independent Git repository yet | N/A | None | Existing icons directory only before this document |

Observed local remote-tracking refs (NOT a live upstream attestation):

- ZeAlfie origin/main equals examined HEAD; origin/beta = 56782113932ed1efdca0bdab2d0abfda878dbff8.
- ZSSS origin/main and origin/beta both equal examined HEAD.
- ZeMosaic origin/main and origin/beta both equal examined HEAD.
- ZeSolver origin/test equals its test HEAD; origin/main equals ZeSolver-main HEAD.
- ZeAnalyser origin/beta equals examined HEAD; origin/main = d3e8c79d670c1f2cb1ca322a1589204756a7dc3e.

No fetch, pull, branch switch, remote mutation or worker delegation was performed. **Authority for this architecture is the inspected local source at these SHAs**, plus the normative contract and explicitly identified local assets. Cached refs may be stale. Junior must record drift before implementation; never silently substitute a moving branch or a historical report.

ZeAlfie dirty paths at inspection:

    M docs/testing.md
    M packaging/macos/gui_smoke_offscreen.py
    M packaging/macos/witnesses.py
    M packaging/windows/gui_smoke_offscreen.py
    M tests/witness/posix_lock_ci_witness.py
    ?? docs/tmpfs-cleanup-gate.md
    ?? tests/test_tmpfs_cleanup_gate.py

These changes are not this mission's work. Preserve them. Do not attribute uncommitted witness changes to dd74f847. The architecture, catalog, compatibility parser, packaging definitions and cited contracts are the relevant baseline sources. ZSSS's untracked research directories are likewise not release evidence.

### 2.2 Normative contract and current orchestration evidence

Read the entire **projects/ZESOFTWARE_INTEROPERABILITY_RULES.md** before work.

- Title: ZeSoftware Interoperability Rules.
- Version **1.0**, dated **2026-08-13**.
- SHA-256: **88831410559f8201eeddc6722482ebf9ab3495dd10b1ab33adbbef9e3d2595ed**.
- This is a workspace-level normative file, not a version inferred from a product release.
- Apply all 25 rules, especially independence, public API version/capabilities, optional failure isolation, persistent storage, dedicated namespaces, package resources, compatible dependency closure, immutable provenance and rollback.
- Interoperability gates A–H are incorporated in Phase 9 below.

The current workspace **/home/tristan/.openclaw/workspace/AGENTS.md** was checked directly:

- SHA-256 **418bb0a040a90196593df81ed77fe03e2071d21d2b597c303507411fa67c18b4**.
- Its canonical Direct Agent-to-Agent Delegation Policy specifies persistent Coco/Nono sessions, fire-and-forget sessions_send(timeoutSeconds=0), durable reports, exact-source-session callbacks and REPLY_SKIP.
- It documents OpenClaw 2026.9.1-beta.1 as the targeted build. This archaeology did not independently attest the installed gateway binary version or test callback delivery.
- The runtime tool catalog exposes sessions, sessions_history and sessions_send. Tool availability does not prove A2A authorization or successful delivery.
- Preserve this documented workflow; revalidate the live operational contract at handoff. Do not change runtime configuration to make the mission work.

### 2.3 Source evidence index

Paths are relative to their repository; line numbers are anchors at the inspected revision, not permanent API contracts.

| Evidence | Location | Finding |
| --- | --- | --- |
| ZeAlfie product selection | docs/architecture.md, especially around 871; src/zealfie/manifests/products.toml | Known catalog, desired selection, installed probe and launchable state are distinct; registry is derived |
| ZeAlfie compatibility | src/zealfie/compatibility/parser.py:51 and model.py | Scans wheel package-data zesoftware_interop.json without importing product code; checks distribution identity; schema zesoftware.interop.v1 |
| API precedent | ZeSolver-main/zesolver/api/v1/{__init__,models,probe,readiness}.py | API_VERSION 1.2, get_api_info, cheap probe, separately requested resource/GPU readiness |
| Consumer precedent | ZSSS seestar/alignment/zesolver_adapter.py:1; ZeMosaic solver_port.py and zesolver_adapter.py under src/zemosaic | Lazy public-only adapter; absent provider differs from broken internal dependency |
| ZSSS raw entry | seestar/core/image_processing.py:62,97,130 | load_and_validate_fits selects HDU with do_not_scale_image_data=True; copies and sanitizes header |
| ZSSS destructive preparation | same file:143,190,195,209,215,224 | Shape heuristics; non-finite→0 repair; float64 statistics; float32 cast; min/max normalization and clip; constant→0.5 |
| ZSSS debayer | same file:251,269 | Clips to [0,1], casts to uint16, OpenCV conversion, returns RGB float32 /65535 |
| ZSSS normal science frame route | seestar/queuep/queue_manager.py:11065,11130,11192,11212,11225 | _process_file: load, variance gate, debayer, apply_wb_basique, hot-pixel correction |
| ZSSS registration entry | same file:11308 onward; seestar/core/alignment.py:134 | Alignment/WCS follows preparation; _align_image branches include local and global alignment |
| ZSSS references | seestar/core/alignment.py:430,470,569; geometry_reference.py:208 | _get_reference_image manual/automatic and geometry candidate preparation separately load/debayer/correct hot pixels |
| ZSSS colour / rescaling | seestar/core/drizzle_background.py:148; queue_manager.py:11284 | Median-ratio WB; rescale_01_to_adu has a nonnegative clip and range heuristic |
| ZSSS hot pixels | seestar/core/hot_pixels.py:9,62,85 | Local-median threshold/replacement cosmetic mechanism; not dark subtraction |
| ZSSS execution policy | seestar/core/gpu.py:70,134,295 | GpuCapabilities, explicit probe, frozen AccelerationPolicy; CuPy execution, OpenCV CUDA diagnostic-only |
| Exact-N reducers | seestar/core/{stack_methods,stack_gpu,cpu_winsor_exact_n,cpu_memory_planner,cpu_memory_policy}.py | Whole-population spatial tiling; bounded smaller-tile retries; refusal rather than changing N |
| CPU fallback seam | queue_manager.py:3744 onward, _run_cpu_winsor_policy | Both ordinary CPU and GPU fallback obey current RAM policy; FULL_CPU / SPATIAL_TILED_CPU / CPU_MEMORY_REFUSAL |
| Provenance | queue_manager.py:22593,22861,22962,23117,23242 | RUN_REQUEST/RUN_EFFECTIVE, GPU decision/execution facts, run_config persistence |
| Resume identity | seestar/core/drizzle_checkpoint.py; seestar/run_contract.py | Scientific fingerprint, transactional writer/reader, source disposition and continuation contracts |
| ZeMosaic input | src/zemosaic/zemosaic_utils.py:3903,3976,4228 | Explicit BZERO/BSCALE treatment for signed integers before preparation; still a normalized/debayer-oriented utility, not a calibration API |
| Qt resources | ZSSS seestar/gui_qt/resources.py | importlib.resources bytes → QPixmap/QIcon, no CWD dependency |
| ZeMosaic resources | src/zemosaic/_resources.py | Package-aware/frozen resolver, but zip-resource cache falls back to ~/.cache: do NOT copy that Linux-centric policy |
| Packaging | each pyproject.toml; ZeMosaic ZeMosaic.spec and zemosaic_installer.iss | Dedicated namespaces, wheel package-data; ZeMosaic PyInstaller plus Inno |
| Native packaging lessons | ZeAlfie packaging/windows, packaging/macos and corresponding docs/workflows | Locked private Python/wheelhouse, explicit native validation, deployment distinct from runtime product behavior |
| App identity | ZeAnalyser src/zeanalyser resources and pyproject; ZeAlfie GUI identity/docs/windows-taskbar-icon.md | Native shell identity is more than Qt window icon; managed pythonw launch needs its own witness |

**Important stale documentation:** ZSSS docs/gpu_acceleration_architecture.md describes an older non-tiled GPU limitation. Current stack_gpu.py has stack_winsorized_sigma_gpu_tiled (around 1135); cpu_winsor_exact_n.py implements its CPU twin. Source and focused tests supersede stale summary prose. ZeAlfie products.toml also retains an obsolete comment attributing ZSSS CuPy to drizzle. Current Classic reducers and direct Drizzle accumulation must not be conflated.

No full ecosystem test suites, native builds or GPU benchmarks were rerun for this documentation-only task. Tests named here are inventory, not fresh PASS claims.

## 3. FITS archaeology and scientific witnesses

### 3.1 What was actually found

A read-only filename search found 5,647 FITS-like paths under /home/tristan, excluding caches/Git/site-packages. A bounded, order-dependent sample of 450 primary headers showed seven camera/type/shape signatures: real Seestar S50/S30 lights, synthetic fixtures and processed outputs. This is **not** a statistically representative camera survey. No filenames containing dark/bias/flat were found in that inventory; absence of such filenames does not prove absence of masters.

Representative raw-header observations:

| Field | S50 witness | S30 witness | Interpretation |
| --- | --- | --- | --- |
| INSTRUME | Seestar S50 | Seestar S30 | Models differ; same geometry is not detector identity |
| CREATOR | ZWO Seestar S50 | ZWO Seestar S30 | Capture software context, not an unconditional detector alias |
| NAXIS1 × NAXIS2 | 1080 × 1920 | 1080 × 1920 | NumPy shape is (1920,1080) |
| XBINNING / YBINNING | 1 / 1 | 1 / 1 | Also CCDXBIN / CCDYBIN = 1 / 1 |
| XORGSUBF / YORGSUBF | 0 / 0 | 0 / 0 | Header comments explicitly say binned-pixel origins |
| EXPTIME / EXPOSURE | 20 / 20 s | 10 / 10 s | Two actual aliases requiring conflict checks |
| CCD-TEMP | 30.5625 °C | 28.125 °C | Measured temperature, not assumed setpoint |
| GAIN | 80 | 200 | Vendor setting; not automatically e-/ADU |
| FILTER | IRCUT | IRCUT | Separate inspected M16 S50 acquisition has LP |
| BAYERPAT | GRBG | GRBG | Pattern alone does not prove sensor/raw status |
| BITPIX / BSCALE / BZERO | 16 / 1 / 32768 | 16 / 1 / 32768 | FITS signed storage representing unsigned physical values |
| OFFSET / readout mode / ADC depth | Absent in sampled examples | Absent | Unknown, not zero/default/equal |

Witness locations:

- /home/tristan/near_bench100_input/049_Light_mosaic_M 106_20.0s_IRCUT_20250518-233417.fit
- /home/tristan/near_bench100_input/058_Light_mosaic_M 31_10.0s_IRCUT_20250804-004954.fit
- ZSSS C:/data/input_folder/rejected_low_snr/Light_M 16_10.0s_LP_20250530-035340.fit: real acquisition-origin header, already augmented with WCS/ASTAP metadata; not claimed pristine acquisition.
- ZSSS fixture/master/Light_001.fit: explicitly **synthetic LIGHT**, INSTRUME=SYNTH, seed 20260821, 1920×1080. The directory name “master” does not make it a calibration master.

Processed three-dimensional outputs can retain BAYERPAT=GRBG. Never classify raw CFA solely from that card. Do not identify master type from a filename or directory name alone.

No independent manufacturer's real master library was qualified. Candidate aliases for other cameras below remain unqualified until witnessed. Real witness files stay local and read-only; export only allowlisted, redacted header fields into project research evidence, not site coordinates or full personal capture headers.

### 3.2 Narrow executed loader witness

A temporary FITS was written/read in a temporary directory using ZSSS's existing .venv Python, NumPy and Astropy. The actual loader and sanitizer function ASTs were evaluated without editing/importing the full application. This proves those functions' arithmetic, **not** whole-pipeline integration.

Input uint16:

    [[0, 100],
     [200, 65535]]

Observed:

- Astropy physical FITS read: same values.
- Header BZERO: 32768.
- ZSSS default loader: [[0, 0.0015259021893143654], [0.0030518043786287308, 1]].
- ZSSS loader with normalize_to_float32=False and attempt_fix_nonfinite=False: [[-32768, -32668], [-32568, 32767]].
- Separate float32 [[-2,0],[2,NaN]] with default loader became [[0,0.5],[1,0.5]].

Therefore “normalize=False” is NOT a valid scientific raw decoder. Original header, storage scaling and invalid mask must be addressed before this seam can carry calibration. Temporary witness files were removed; Phase 0 recreates a durable reproducible research witness.

## 4. Normative scientific calibration contract

### 4.1 Frame domain and decoding

Initial supported domain: one explicitly selected **2D mono or 2D Bayer/CFA sensor plane**. No RGB cubes, stacked science images, compressed camera RAW formats, multisensor assembly or implicit mosaic detector handling. Multi-HDU FITS must select a unique supported sensor plane or require an explicit HDU identifier; do not silently choose the first plausible image.

FITS decoding rules:

1. Preserve the original header/card evidence and storage descriptor before Astropy updates scaling cards.
2. Decode physical sample = BSCALE × stored sample + BZERO **exactly once**, including scaled floating FITS where applicable. Defaults follow FITS, not manufacturer conventions.
3. Treat integer BLANK, NaN and ±Inf as invalid samples; detect BLANK in stored space before scaling.
4. Resolve units explicitly: physical acquisition ADU for v1. A unitless normalized response flat is a separate declared role. Do not infer gain conversion from a GAIN card. Electron/rate data require an explicitly supported conversion contract, otherwise reject.
5. Byte order is storage, not camera identity. Convert deliberately to native-endian contiguous float32 only after decoding. Use float64 for scaling/statistical intermediates where needed; return float32 scientific buffers.
6. Float32 exactly represents integer ADU through 2^24. Accept usual 8–16-bit acquisition witnesses; larger integer ranges, extreme BSCALE or catastrophic cancellation need an explicit error-bound witness or refusal, never a silent precision loss claim.
7. No data-derived normalization, clipping, orientation heuristics, arbitrary channel axes or invalid→zero replacement.
8. Require a raw-domain declaration backed by header/profile/import metadata. Recognized processed-history markers cause rejection. Uncertain legacy master semantics require an explicit persisted import declaration, never a guess.

Geometry is an exact contract: shape, binning, ROI origin/extent, orientation, CFA phase and detector identity. Same shape does not imply same ROI. Do not crop a full-frame master to match a light in v1. Translation by even one pixel changes CFA phase. Preserve WCS cards if valid, but never solve or use WCS to align masters.

### 4.2 Master semantic roles, not just filenames

Every imported master has a versioned descriptor:

- master_type: bias / dark / flat / flat_dark;
- pixel_domain and physical units;
- detector/acquisition/geometry metadata;
- bias_state: included / removed / not_applicable / unknown;
- for flats: raw_response or normalized_response; additive-correction history, normalization algorithm, scalar(s) and validity evidence;
- source population/exposure/temperature metadata when known;
- stable content identity, HDU and mask identity;
- provenance/import declaration source.

Unknown processing history is not interchangeable with raw masters. A manual declaration fills documented metadata; it does not silently override contradictory measured facts or bypass incompatible geometry.

### 4.3 Equations and no double subtraction

Let L be the raw light in physical ADU; B a bias master; Dinc a same-exposure dark including bias; D0 a demonstrably bias-removed dark; Finc an unnormalized flat including its additive pedestal; FDinc a flat-dark including bias, exposure-matched to the flat; FD0 its bias-removed equivalent.

Choose exactly one coherent light additive branch:

| Mode | Corrected numerator A | Preconditions |
| --- | --- | --- |
| Control / no additive correction | L | Explicit requested identity/partial mode; not “fully calibrated” |
| Bias only | L − B | Bias matched to light; reports bias-only |
| Dark including bias | L − Dinc | Same exposure, compatible thermal/acquisition conditions; **do not subtract B again** |
| Bias-removed dark | L − B − D0 | Compatible B required; D0 descriptor explicitly bias_removed |

Darks are not exposure-scaled in v1. Flat-dark is for flat preparation, not an extra subtraction from the light.

Flat preparation chooses exactly one additive branch:

- Flat-dark including bias: Fcorr = Finc − FDinc.
- Bias-removed flat-dark: Fcorr = Finc − Bflat − FD0.
- Bias only: Fcorr = Finc − Bflat; allowed only by an explicit scientifically qualified short-flat profile that declares dark current negligible at that exposure/temperature. It is not a generic automatic fallback.
- Already corrected/normalized response: consume only when the descriptor proves its processing/normalization semantics; never subtract bias or flat-dark a second time.

Normalize a valid corrected flat to a dimensionless response R:

- Mono: s = exact median of finite, positive, non-saturated, explicitly valid Fcorr pixels; R = Fcorr / s.
- Bayer CFA: four parity planes (including separate G1/G2) each get their own globally computed median s[p]. Apply R[y,x] = Fcorr[y,x] / s[parity(y,x)] using the frozen CFA origin. This preserves raw colour-plane scales; it is not white balance or debayer. Never use a per-tile median.
- Record the normalization mask, counts and four scalars. Do not guess saturation from the largest observed pixel or from BITPIX; use qualified acquisition saturation metadata or report the missing quality evidence.
- Default numerical division floor: R > 1e-6, finite. Pixels at/below the floor are masked, not raised to the floor. This is a numerical guard, not a claim that arbitrarily weak flats are scientifically good. Additional low-response quality limits must be explicit profile parameters, frozen and reported.
- At least one valid normalization sample per mono/CFA plane is necessary; zero makes the master unusable. v1 default automatic-quality gate additionally requires ≥90% valid samples per plane; this conservative product quality threshold must be reviewed in Phase 1 and must not be presented as a universal physical law.
- Do not silently repair a dead flat pixel by interpolation. A bad master may be rejected rather than “made usable”.

Final calibrated sensor plane:

    C = A / R       when a compatible flat response is requested and available
    C = A           when the explicit requested mode has no flat correction

Flat-only operation is an explicit partial mode, not a substitute for additive correction. The request fixes required roles before matching; failure to satisfy a role never silently downgrades the equation.

Small independent arithmetic witness, not dependent on implementation:

    L       = [[110,  50], [210, -2]]
    Dinc    = [[ 10,  10], [ 10, 10]]
    B       = [[  4,   4], [  4,  4]]
    D0      = [[  6,   6], [  6,  6]]
    R       = [[  1, 0.5], [  2,  1]]
    C       = [[100,  80], [100,-12]]

Both coherent dark branches yield C. Subtracting B in addition to Dinc must fail a negative-control test. Add flat-dark/pedestal witnesses showing that normalizing an uncorrected flat is not equivalent.

### 4.4 Invalidity, signed data and output

Return a uint16 reason-bit mask with stable v1 meanings: input invalid, additive master invalid, flat invalid/too-small, saturated input, arithmetic nonfinite. Multiple reasons can coexist. Negative finite C is valid and is never itself a masking condition.

Initial policy: any invalid contributing sample yields output NaN and mask bits. Inf is canonicalized to NaN with a reason, not zero. All-invalid frame → FAILED, not empty success. Partial invalidity returns completed-with-warning and exact counts. Explicit client quality thresholds can reject a frame. No fabricated uncertainty array; variance propagation is a separately versioned future capability.

Standalone writes a new FITS with BITPIX=-32, physical ADU BUNIT and no stale integer BSCALE/BZERO/BLANK or CHECKSUM/DATASUM. Recompute relevant FITS checksums; preserve geometry/CFA/acquisition metadata, append namespaced calibration history. Store a DQ image extension and full structured provenance as described below. Never use ZSSS save_fits_image: it is a uint16/[0,1] export path.

Identity mode returns equal decoded float32 values and mask, without modifying the caller's array or the input file; “no calibration” does not mean byte-identical FITS encoding. Byte hashes of original files must remain unchanged.

## 5. Deterministic calibration-library matching

### 5.1 Metadata normalization and qualification

Normalize into an immutable SensorMetadata value object. Keep **all original candidate cards**, normalized values, units, alias/profile version and conflict diagnostics. Never silently take one of two disagreeing aliases.

Confirmed aliases:

- exposure_seconds: EXPTIME / EXPOSURE.
- bin_x: XBINNING / CCDXBIN; bin_y: YBINNING / CCDYBIN.
- roi_origin: XORGSUBF / YORGSUBF, using the witnessed binned-pixel convention.
- detector model: INSTRUME; CFA: BAYERPAT; temperature: CCD-TEMP; gain: GAIN; filter: FILTER.

Possible future profile aliases (NOT globally enabled merely by this document): CAMERA/DETECTOR, CCDGAIN/EGAIN, BLACKLEV/OFFSET, XBIN/YBIN, CCDTEMP, READMODE/READOUTM, XBAYROFF/YBAYROFF. EGAIN can mean conversion factor, not vendor gain; SET-TEMP is not measured CCD-TEMP; TELESCOP is not universally a camera serial. Validate each mapping using header evidence and manufacturer/driver semantics.

Strings: trim FITS padding and Unicode-normalize; case-fold only fields declared case-insensitive. Do not merge S30/S50, different cameras of the same model, arbitrary filter aliases or optical trains. Use user-assigned detector_instance_id and flat optical_train_id when headers cannot identify them reliably. Explicit profile evidence may supply fixed fields; unknown==unknown is not an exact match.

### 5.2 Field applicability

E = exact/qualified identity; T = thermal/exposure policy; Q = quality context, not blind light-to-master equality; — = not a normal criterion.

| Field | Bias vs light | Dark vs light | Flat vs light | Flat-dark vs parent flat |
| --- | --- | --- | --- | --- |
| Detector instance/model | E | E | E | E |
| Sensor dimensions, ROI, orientation | E | E | E | E |
| Binning / CFA phase | E | E | E | E |
| Gain / offset / readout mode | E | E | E by default | E |
| Acquisition ADU representation / ADC mode | E | E | E; stored float master allowed | E |
| Temperature | T | T | Q; exact by default unless qualified stable response | T |
| Exposure | Bias-class near-zero, NOT equal to light | T | Flat-quality range, NOT equal to light | T against flat |
| Filter | — | — | E | — |
| Optical train / illumination geometry | — | — | E | — |
| Observation date | Report only | Report only | Report/explicit validity interval | Report only |

BITPIX is storage type, not acquisition ADC depth. A float32 master can match a uint16 light if its documented physical units/acquisition mode match. Unknown ADC depth is not silently inferred as 16 because BITPIX=16. Geometry and units mismatches are hard rejections.

### 5.3 Conservative v1 policy

- **Detector, ROI, binning, gain, offset, readout and CFA are exact**, after qualified normalization. No nearest-neighbor or weighted “similarity score”.
- Exposure equality permits only numerical serialization tolerance: absolute difference ≤ max(1e-6 s, 1e-6 × max(abs(t1),abs(t2))). This is not scientific exposure scaling. Darks compare to lights; flat-darks compare to flats. Bias master exposure must satisfy a declared qualified bias acquisition range, not an invented 0-second universal rule.
- Temperature: default zero scientific tolerance after unit conversion; allow only tiny parser equality tolerance (1e-6 °C). Any nonzero thermal window must be named, versioned and validated for the detector/master type in Phase 1 or a later profile mission. No unqualified “within 5 °C is fine”. Store master population min/max/median where available; do not treat the first source temperature as the whole population.
- Flat filter is exact, but filter equality alone is insufficient: optical train, sensor orientation, illumination mode and explicit validity interval matter. Binning/filter/ROI are not automatically adaptable.
- Missing required fields block automatic selection. A reviewed user library/profile declaration can supply an unknown field with visible provenance. Two unknown values must not match automatically. Explicitly “not applicable” requires a profile reason.
- Malformed numerics, duplicate conflicting cards, incompatible HDUs, undocumented master processing or saturated/unusable flat normalization produce reason-coded rejection.
- Do not select “newest”, closest temperature, most frames, first directory entry or filename sort as a scientific tie-break.

Selection procedure:

1. Freeze request, matching policy, library index revision and role requirements.
2. Inspect raw metadata; enumerate masters by semantic role.
3. Reject incompatible candidates with per-field reason records.
4. Construct complete coherent sets, including the flat's own additive correction dependencies.
5. Zero sets → NO_MATCH; more than one distinct set → AMBIGUOUS; exactly one → MATCHED.
6. Byte-identical duplicate masters with identical descriptors can be collapsed to a single content identity, preserving all locations. Different hashes remain ambiguous.
7. Manual choice among valid candidates may resolve ambiguity and is recorded; it cannot bypass hard science rejections.
8. Revalidate selected content/HDU/descriptor identities at execution to detect edits after planning. A stale index is not evidence of unchanged data.

Library indexing is read-only over user masters. Store references and normalized metadata, not owned copies. Scanner sorts results for stable display, supports cancellation, avoids symlink loops, records unreadable files without treating them as missing success, and invalidates cache on change. Start with a versioned SQLite index using stdlib sqlite3, short transactions and one serialized writer; use a filesystem abstraction so a smaller alternative can be justified before implementation. Network libraries/distributed locking are deferred.

## 6. Architecture and public API

### 6.1 Dependency direction

    scientific/core ← application/services ← public API / CLI / GUI adapters
                                                   ↑
                                               PySide6 GUI

Suggested internal responsibilities: metadata/units/geometry and pure equations in core; FITS/library/provenance/storage adapters in io; orchestration, plans, progress, cancellation and transactions in application; exported stable models/functions in api.v1; presentation only in gui.

No Qt, ZSSS, ZeAlfie, CuPy or camera-device dependencies in scientific modules. No GUI import from package root or api.v1. Optional integrations are lazy. No generic top-level installed utils/core/config/helpers modules.

### 6.2 Smallest useful versioned boundary

Use the established ecosystem style:

    distribution: ZeCalibrator
    product_id / package: zecalibrator
    public module: zecalibrator.api.v1
    product version initial target: 0.1.0
    public API version target: 1.0
    compatibility range for initial consumer: >=1,<2 plus explicit capabilities

These are proposed ZeCalibrator names to freeze at Phase 1, not claims that an API exists today. Product version, API version, provenance schema, library schema and matching-policy version are distinct. Breaking API semantics require a new major. Additive compatible changes require capability negotiation and documented deprecation.

Minimum exported surface:

| Public operation/type | Contract |
| --- | --- |
| API_VERSION, get_api_info() → ApiInfo | Product/API versions, stable supported capabilities; cheap, no file scans, Qt/GPU init, network or writes |
| probe(check_library=False, check_gpu=False, ...) → RuntimeProbe | Separate supported / available / not_checked / unhealthy with reasons; bounded optional checks; defaults side-effect-free |
| open_library(LibrarySpec) → LibraryHandle | Explicit library/index location or platform user defaults; context-managed, close idempotent; no configuration discovery via ZeAlfie |
| inspect_frame(FrameSource) → FrameInspection | Selected HDU, raw metadata, decoding/domain/geometry findings; no normalization |
| resolve_calibration(FrameInspection, LibraryHandle, MatchPolicy) → MatchResult | Explainable complete-set selection; returns an immutable CalibrationPlan when matched |
| calibrate_frame(FrameSource, CalibrationPlan, ExecutionOptions, *, cancel=None, progress=None) → CalibrationResult | Same plan/units/masks/equations for Python and standalone; returns NumPy float32 raw sensor array |
| calibrate_batch(iterable, library, policy, options, ...) | Added only at its gate as a bounded iterator of per-frame results, not an eager list of image buffers |

FrameSource is a tagged union, not an untyped path-or-array guess:

- FitsFrameSource(path, hdu): product performs a correct raw decode.
- ArrayFrameSource(data, SensorMetadata, InputIdentity, invalid_mask, domain="sensor_adu", scaling_applied=True): caller guarantees the decoded physical domain; reject ambiguous scaled/normalized arrays.
- Do not accept a mutable header as the sole proof of domain. Public metadata preserves original card evidence, units and explicit decoder facts.
- Array input is read-only to the callee; no in-place modification, even on failure. Output owns its buffer; masks and provenance outlive library context closure. At least one float32 result allocation is expected; “no intermediate FITS” is not a zero-copy promise.
- Plans bind input geometry/metadata and exact master identities. They can be reused only for frames proven to match the same constraints.
- Library handles are not blindly shared across threads/processes; expose documented thread-safe reads or one handle per worker. No global mutable active calibration set.

Initial capability identifiers to freeze:

    calibrate_frame, calibration_library, master_matching, provenance, cancel

Add calibrate_batch only when implemented. GPU is unsupported/unavailable until a backend passes its gate; master_building is not advertised. Static wheel capabilities describe implemented public behavior, not hardware present. Dynamic probe distinguishes supported code from currently available resources.

CalibrationResult contains status, raw data or no data, mask, matching report, execution report, provenance, warnings and reason code. Expected statuses: COMPLETED / COMPLETED_WITH_WARNINGS / SKIPPED / CANCELLED / FAILED. NO_MATCH and AMBIGUOUS belong to matching, not numerical success. API validation errors have typed stable exceptions; anticipated operational failures return structured results. Consumer adapters still catch ordinary unexpected exceptions to isolate unhealthy providers; do not swallow process interrupts.

### 6.3 Failure, cancellation and progress

- CancellationToken is independent of Qt; check before expensive read/hash/normalization, between tiles, between frames and before commit.
- A blocking filesystem call/native kernel cannot promise instantaneous cancellation. Define cooperative cancellation latency by tile size; no unsafe thread kill.
- ProgressEvent: operation_id, phase, completed, total-or-null, unit, optional frame_id. Immutable, monotonically ordered sequence; unknown total remains unknown. Only committed completion is 100%.
- User callbacks run outside library/database locks; throttle delivery. A failing observer must not change arithmetic: isolate/log observer failures; cancellation remains a separate explicit token.
- No unbounded producer queues. Initial application uses bounded worker execution; GUI owns a worker QObject/QThread or bounded pool adapter and queued signals. Widgets are accessed only on Qt main thread.
- Close/cancel during a write leaves no committed partial output. Completed earlier batch files remain valid; batch manifest records partial/cancelled state. Cancellation is not CPU fallback and not a successful no-op.

### 6.4 Wheel interoperability declaration

Bootstrap can ship an empty provides list; do not claim executable API capability before implementation. At Phase 5 acceptance, package exactly one declaration at zecalibrator/zesoftware_interop.json:

~~~json
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
~~~

After batch acceptance add calibrate_batch. Metadata must agree with installed runtime API tests. No ZeAlfie import is needed to generate/read it. ZeAlfie catalog admission later is a separate deployment integration, with GUI extra/entry point and channel mapping, not an application dependency.

## 7. ZSSS integration: exact seam and prerequisite gate

### 7.1 Seam to introduce, not implement now

Introduce a consumer-owned **RawCalibrationPort** plus NoCalibrationAdapter and ZeCalibratorPublicAdapter. The optional adapter imports only zecalibrator.api.v1 through lazy importlib discovery and checks API range/capabilities.

The insertion is at **HDU selection + original header capture + physical raw decoding**, before image_processing.py's sanitizer/shape conversion/nonfinite repair/minmax path. At the normal _process_file call around queue_manager.py:11130, calibration must already be inside a raw preparation service; inserting a call after that function returns is too late.

Do not add a global callback to every legacy FITS reader: reducers, saved stacks, previews and resumed artifacts are not necessarily raw. Explicit request domain and per-run immutable service routing must govern calls.

Consumers requiring review coverage:

- Ordinary light _process_file.
- Manual/frozen reference _get_reference_image around alignment.py:470.
- Automatic reference candidates around alignment.py:569.
- geometry_reference.py reference_quality_metric (around line 205), select_legacy_reference and select_geometry_reference paths.
- Boring subprocess path and its serializable run configuration.
- Classic, Drizzle and mosaic preparation, plus resume and calibration-plan persistence.
- Direct FITS reads of reduced outputs/checkpoints must remain marked non-raw and never recalibrated.
- Preview/analysis-only loaders must not accidentally alter scientific source selection or claim calibration.

### 7.2 New prerequisite discovered during archaeology

Current ZSSS debayer clips signed ADU through [0,1]→uint16 and Drizzle input rescale clips negative values. Simply moving subtraction earlier would execute calibration but then destroy valid scientific values.

Before enabling the integration, a **bounded ZSSS raw-domain preparation gate** must:

1. Preserve legacy calibration-OFF outputs exactly for declared witnesses.
2. Provide an explicitly calibrated/signed path through consumer-owned float-capable debayer, subsequent processing and science export; no convenience clipping, unsigned wrap or data-dependent scale heuristics.
3. Freeze and document colour/variance/invalid-mask handling, reference preparation and registration inputs; legacy normalized variance thresholds cannot be applied directly in ADU without qualification.
4. Keep display-only normalization on a copy. Per-plane flat normalization is not permission for consumer min/max normalization of calibrated science.
5. Qualify the existing colour-demosaicing dependency as a possible float debayer implementation, not assume it is acceptable because installed. Preserve CFA orientation with impulse/ramp witnesses.
6. If this cannot be bounded safely, stop at BLOCKED with an exact missing-mechanism report; do not enable a scientifically misleading integration.

This prerequisite is a separate later ZSSS mission, not permission to refactor ZSSS now.

### 7.3 Safe optional behavior and homogeneous science

ZSSS startup and ordinary use must survive absent/disabled/incompatible/unhealthy provider and no compatible set. Disabled is no import/no probe and runs the established legacy path.

Freeze calibration policy before science starts:

- off: legacy processing, calibration requested=false.
- optional (default when opted in): preflight all planned science/reference inputs; if no coherent supported calibration plan, run **the whole run uncalibrated**, with prominent skip reasons and truthful provenance. Do not silently mix calibrated and uncalibrated frames.
- required (explicit user choice): no match/incompatible/failure refuses that run, not application startup.

If a provider fails after calibrated frames have already been accepted, never silently append uncalibrated frames. Stop the affected run safely; offer a fresh uncalibrated run via the normal user action. The product remains usable. Any future skip-frame policy requires an explicit acceptance change and exact disposition accounting.

A disabled/failed call cannot corrupt the source buffer. At execution failures discard the entire partial provider output. UI diagnostics distinguish absence, API mismatch, dependency failure, no match, ambiguous match, stale master, cancellation and numerical failure.

Persist calibration requested/available/matched/executed/skipped/failed state per frame and aggregated run. Checkpoint/resume fingerprint includes selected identities, equations, matching policy, decoder/domain contract and mask semantics; backend/tile strategy is separately recorded as execution where parity is proven. Resume refuses changed masters or a changed scientific calibration plan. Avoid double calibration of already calibrated intermediates.

## 8. Future master-building and ZeMosaic seams

### 8.1 Conditional ZSSS public raw reduction

**Deferred.** No public versioned raw-sensor N-frame reduction API was found. Existing Python exports/private reducers are not that contract.

When explicitly scheduled, first test feasibility of exposing a ZSSS-owned public capability, provisionally raw_sensor_reduce under a versioned public namespace such as seestar.api.v1. Final naming and implementation belong to a separate ZSSS contract mission.

ZeCalibrator owns source selection, all acquisition/thermal checks, bias/flat-dark preparation, flat normalization, master semantics and publication. ZSSS provides only N-frame reduction execution, with input identity/order, masks, parameters, memory policy, cancellation, progress and reduction provenance.

The dedicated request must make these non-overridable invariants:

    registration OFF; WCS OFF; reprojection OFF
    debayer OFF; white balance OFF; geometric transformations OFF
    min/max normalization OFF; cosmetic hot-pixel replacement OFF
    ZeCalibrator callback OFF

Do not call the ordinary queue preparation pipeline with a fragile collection of flags. The provider must expose a mechanically bounded raw reducer path that cannot discover/call calibration. A recursion guard is defense-in-depth, not the primary architecture. Calibration capability must remain functional without this provider; master_building becomes unavailable when the optional reducer is absent/incompatible.

Reuse lessons, not private imports:

- Winsorized sigma CPU authority in stack_methods.py.
- CPU/GPU spatial tiling holds the full N at every pixel.
- Global two-pass rejection schedule coordination preserves untiled z_eff.
- Exact placement, not blending; global counts, not means of tile percentages.
- Bounded spatial retry, truthful memory refusal, no subset/hierarchical nonlinear shortcut.
- Transaction/commit and source-disposition patterns may inform the new engine contract; current Drizzle checkpoint is not automatically a reusable master-builder transaction.
- Source Frames × pixels are fixed before execution. Scientific rejection is explicit; VRAM/RAM must never secretly reduce N.
- Current reducer comments/code include all-masked zero fix-ups. The future public raw reducer must expose valid/survivor counts and masks so ZeCalibrator never mistakes an all-invalid zero for a measured master pixel. Define this adaptation explicitly and qualify it without changing the legacy stacking contract.
- Flats' differing brightness need a separate scientifically qualified normalization policy before robust reduction; don't inherit sky_mean/noise_variance stacking defaults.

Verify source licensing before copying any code. ZSSS/ZeMosaic/ZeAnalyser declare GPL-3.0-or-later; API reuse is preferred over duplicated code but does not eliminate licensing obligations. ZeCalibrator's own license is not yet specified and must be settled before distribution.

**Do not create ZeStackCore.** Require demonstrated duplicated independent requirements before considering extraction.

### 8.2 ZeMosaic

**Deferred.** Future consumer gets the same public RawCalibrationPort semantics at its own raw FITS seam before zemosaic_utils.py normalization/debayer. Inspect its current pipeline afresh then; do not import its utilities or rely on sibling checkouts. Preserve per-light raw geometry and ensure calibration precedes master-tile construction/WCS reprojection. API tests can demonstrate geometry/CFA portability without implementing a ZeMosaic adapter.

## 9. CPU/GPU execution contract

Initial implementation: NumPy CPU, float32 frame arithmetic with documented float64 decode/statistical intermediates. No mandatory CUDA/CuPy dependency and no GUI-thread GPU probing.

Applicable ZSSS lessons:

- Freeze backend intent and science parameters separately.
- Distinguish eligible, requested, attempted, executed and CPU fallback with reason.
- Query actual supported backend readiness; hardware/model-name detection alone is insufficient.
- Budget current RAM/VRAM, arrays, masks, masters, output and scratch, not only input dimensions.
- On OOM shrink spatial tile/worker concurrency, never change input population, masters, equations or precision silently.
- Bound retries, preserve original buffers and transactional output. CPU fallback is attempted only if feasible; otherwise fail explicitly.
- Do not copy nvidia-smi/shell probing as normal ZeCalibrator behavior; use a library backend probe behind an optional adapter.

Calibration is mostly per-pixel subtraction/division, not an N-frame nonlinear reducer. Tiles require no blend and normally no halo. Global flat normalization is computed once using the canonical population before tile execution; exact median cannot be replaced by a streaming approximation without a new science contract. If it does not fit, use exact out-of-core selection or refuse; do not average tile medians.

For v1, document CPU memory limits honestly; do not promise arbitrary sizes until tiled/out-of-core paths are qualified. The public array-return mode necessarily requires enough memory for output+mask. Standalone tiled FITS output may later handle larger frames, but cannot claim the array API eliminates that allocation.

Backend report includes backend_requested, gpu_supported, gpu_available/not_checked, gpu_eligible, gpu_attempted, gpu_executed, effective_backend, tile shape/count, memory budget, retries, fallback and reason, device/backend/library versions. Mixed executed tiles are explicitly reported, not relabeled “CPU only” or “GPU used” wholesale.

GPU gate: same master choices, decoded units, flat scalars, masks and status; exact masks and geometry; compare finite output against CPU with a preregistered float32 envelope (initial bound: atol=1e-4 ADU plus rtol=5e-6, to be scientifically accepted before kernels). Include near-zero negative values, high dynamic range, cancellation and OOM fault injection. Record max absolute/relative/ULP error; do not broaden tolerance post hoc to hide defects. Benchmark total read/transfer/compute/write time on named hardware; no universal speedup claim. macOS CPU support does not imply NVIDIA CUDA support; other GPU vendors/backends remain unadvertised.

## 10. Provenance and transactions

### 10.1 Record schema

Use a versioned JSON-compatible record, initially zecalibrator.provenance.v1. Required fields:

- operation_id; product/API/schema versions; installed artifact/source revision when available (unknown is explicit);
- request mode and requested roles; input identity, HDU, shape, dtype, physical units and original scaling facts;
- selected library index revision, calibration-plan id/hash and policy/profile versions;
- master content SHA-256, size, HDU, descriptor identity, optional original-source population, semantic processing states;
- allowlisted original header values, normalized values, alias decisions, user declarations, tolerances and rejected candidate reasons;
- equations/mode, flat normalization algorithm/population/scalars, invalid-mask meanings/counts;
- requested/available/matched/executed/skipped/failed/cancelled facts, warnings and reason codes;
- CPU/GPU actual execution and fallbacks;
- output identity, units, dtype, invalid counts, commit status and timestamps.

A path+mtime is a cache hint, not content identity. Selected masters require content hash validation or an explicitly equally strong immutable identity. Hash at planning/execution with TOCTOU protection: read stable bytes or verify metadata/content did not change while hashing/reading. Include mask/descriptor identities, not image pixels alone. A whole FITS hash and a decoded-payload hash are distinct, named identities.

For caller-owned arrays accept caller identity plus its declared strength; compute a canonical decoded-data digest where needed. Never claim a FITS byte hash for an array. Canonical hashes specify shape, dtype, byte order and NaN canonicalization.

### 10.2 FITS / standalone / embedded persistence

- Embedded result carries complete structured provenance in memory; no implicit log/sidecar write by the calibration core. Consumer attaches it to its run ledger.
- Standalone output FITS includes short HIERARCH ZECAL fields/history, a DQ extension and a CALPROV UTF-8 JSON payload extension using a documented byte representation. Full matching reports must not be truncated into FITS keyword strings.
- A batch manifest indexes outputs, committed files, failures and pending/skipped/cancelled dispositions.
- Avoid hash self-reference: embedded provenance records an output logical id and science-payload digest; final whole-file SHA-256 goes in the external batch manifest after output closure. A file cannot truthfully contain its own final whole-file hash.
- Default output uses a distinct user-selected directory and deterministic collision-safe names based on input identity + plan id. Never overwrite a light/master or an existing unrelated output, including symlink/hardlink aliases and case-insensitive collisions.
- Write into an exclusive temporary file in the destination filesystem, flush/close and validate, then commit via a platform-appropriate no-clobber publication primitive. os.replace alone is not a no-overwrite guarantee. Close FITS handles before rename on Windows.
- Single output FITS with embedded provenance is the atomic unit. Commit manifest last; restart can reconcile verified committed files. Cross-filesystem rename and network-filesystem atomicity are not assumed.
- On cancellation/crash remove only task-owned incomplete temporary files, preserve committed outputs and user masters. Disk-full/provenance-serialization failure is not success.
- Keep science-critical provenance with the output transaction. Optional human logs may fail gracefully; required provenance may not silently disappear.

Logs default to bounded rotating local files; ordinary UI uses basenames/logical IDs. Detailed local audit records can contain explicit user paths. Export/redaction policy avoids leaking site coordinates or unrelated header personal data. No telemetry/cloud uploads.

## 11. PySide6 architecture

PySide6 is the only primary GUI toolkit. GUI widgets never perform calibration equations, master selection, FITS parsing or provenance assembly.

Core package supports headless installation. Adopt **PySide6 as a mandatory dependency of the GUI extra/application artifact**, not of the engine-only wheel install:

    ZeCalibrator          → engine / public API / CLI
    ZeCalibrator[gui]     → PySide6 application
    official desktop artifacts and ZeAlfie GUI install → always include [gui]

This follows the existing ZeSolver GUI-extra convention while other ecosystem desktop apps use base PySide6 dependencies. It does not permit another toolkit. No Qt import is required for scientific tests or storage path resolution.

GUI initial scope: inputs, library roots, matching explanations, explicit mode, output destination, preflight summary, progress/cancel, per-file result and audit link. Display missing metadata/ambiguity; never mask a no-match with a convenient master. Image preview, if added, uses a display copy with a visibly separate stretch.

Worker emits immutable application events via queued Qt signals. Disable conflicting actions during an operation. Close requests cancellation and coordinated shutdown; no QThread.terminate. Prevent stale signals from an older operation updating the new run (operation_id). No calibration under a Qt slot, and no busy-loop processEvents workaround.

## 12. Linux / Windows / macOS portability and persistent storage

### 12.1 Platform-neutral runtime

Proposed baseline Python >=3.11 (compatible with ZeAlfie minimum); qualify actual NumPy/Astropy/PySide6 wheel availability before freezing upper/minimum support. Use pathlib/os.fspath, package resources, and platform-neutral APIs. Do not depend on CWD, checkout layout, fixed drives, /home paths, GNU tools or shell commands for ordinary application behavior. All subprocess adapters, if later needed, use argument arrays, shell=False, explicit timeouts/cancellation and platform-aware executable discovery.

Test actual filesystem behavior on each OS. PureWindowsPath on Linux tests lexical parsing, not Windows file I/O. Windows backslashes are not path separators on POSIX; preserve native path semantics. Include spaces, Unicode (including composed/decomposed names), case collisions and practical long names without demanding a filename beyond filesystem component limits. Windows long-path failures receive explicit diagnostics rather than silent truncation.

### 12.2 Headless user-path policy

Use **platformdirs** behind one zecalibrator storage adapter, appname="ZeCalibrator", appauthor="ZeSoftware", version-independent roots and non-roaming mutable library/cache data. GUI uses the same adapter; do not independently derive different QStandardPaths defaults. QStandardPaths may be used for dialogs/download defaults, not as a second owner of persisted locations.

Logical locations:

| Data | Location class | Lifetime |
| --- | --- | --- |
| settings with schema version | user_config_path | Persistent; atomic writes, migration backups |
| library index/descriptors | user_data_path | Persistent metadata only |
| disposable decode/hash thumbnails/resource materialization | user_cache_path | Rebuildable; quota/eviction; not sole provenance store |
| logs and operation recovery metadata | user_state_path / user_log_path | Bounded, versioned |
| standalone output | Explicit user destination | Owned output, not package resources |
| calibration masters | Explicit user library roots | User-owned; no rename/delete/overwrite |
| staging file for output | Destination filesystem | Task-owned, removed only if incomplete |

Concrete roots are those returned by the adapter on each OS, not hard-coded strings. Linux follows XDG; Windows AppData policy and macOS Application Support/Caches are validated by native tests. Prefer absolute resolved storage overrides supplied through public settings/CLI; tests inject all roots. No import-time directory creation or reads of live user settings. Test versioned migration, concurrent index writers and read-only permissions. Never put state in source, site-packages, frozen resources or ZeAlfie runtime slots. Updates/rollback leave library data intact.

## 13. Icons and application resources

The supplied artwork was inspected. It is an illustrated circular astronomical/calibration emblem with transparent outer area. **Preserve it. Do not redraw, rename destructively or regenerate unnecessarily.**

Existing files in icons/:

- zecalibrator_icon.png: **1254×1254**, 2,459,037 bytes, SHA-256 **7ad1589da6dd0d4b16780bd687ff79dec8fc999e4d0cc03431303117357b5ee4**. Treat as canonical supplied source raster unless Tristan later identifies a different source.
- PNG variants: 16, 24, 32, 48, 64, 128, 256, 512 and 1024 square.
- zecalibrator.ico: 218,054 bytes, SHA-256 **717a04a871867de4cd32f90c3383d8414edbb2c960290ba346a1514e6bc584ca**; ICO header declares seven images.
- zecalibrator.icns: 660,414 bytes, SHA-256 **a43b9b1c53aaedb98ee33c0591ce376a07947967490c67e7018ded2e2665dda6**; icns signature/size inspected.

**These ICO/ICNS assets already exist.** Do not schedule gratuitous regeneration. Phase 0 inventories/hashes every file; native decoding remains a packaging gate, not an assumed PASS.

Three distinct categories:

1. Canonical supplied artwork: icons/ preserved at repository root, source provenance and hashes recorded.
2. Packaged runtime resources: selected byte-identical copies under src/zecalibrator/resources/icons/, declared package-data and included in wheel/sdist.
3. Release artifacts: .exe icon embedding, .app Resources/*.icns, Linux desktop/hicolor installation, generated build outputs under build/dist only; do not check in newly generated binary duplicates without need.

A single manifest maps canonical files to packaged copies and verifies byte hashes. Editable checkout preparation may synchronize these assets once via a documented development/build command; runtime NEVER searches the top-level icons directory or creates package files.

Qt resource loading follows the observed good ZSSS pattern: importlib.resources.files("zecalibrator").joinpath(...).read_bytes() → QPixmap.loadFromData → QIcon, on the GUI thread. Frozen build collects the same package tree. If an OS API requires a real path, use importlib.resources.as_file with lifetime held for the native use, or a content-addressed user-cache materialization; never retain a path after its context is gone. Native persistent shortcuts must not point at ephemeral extraction or removable ZeAlfie slot paths.

Stable proposed identities to freeze before packaging:

- Windows AppUserModelID: ZeSoftware.ZeCalibrator.
- Linux desktop id: io.github.tinystork.ZeCalibrator.
- macOS bundle id: com.zesoftware.zecalibrator.

Set process identity before QApplication where required; apply Qt app/window icon; Linux desktopFileName must match installed desktop id. Validate Windows taskbar/window/shortcut behavior under ZeAlfie pythonw separately from native frozen launch. A QIcon alone is not proof of shell integration.

Source/editable, installed wheel, unrelated CWD, managed runtime and frozen resource tests are all required. A missing icon may degrade gracefully at runtime, but package/release resource tests must fail on omission.

## 14. Bootstrap target (after scientific specification gate)

No package files are to be created in the architecture task or Phase 0. Phase 1 is specification/research only; production bootstrap starts at Phase 2.

Proposed skeleton by Phase 2:

    zecalibrator/
      ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md
      AGENTS.md
      TODO.md
      README.md
      pyproject.toml
      .gitignore
      .gitattributes
      icons/                         # supplied artwork, untouched
      docs/
        ARCHITECTURE.md               # boundaries + interoperability + portability
        SCIENCE_CONTRACT.md           # equations/units/matching/invalidity
        PROVENANCE.md                 # record/identity/transactions
      research/
        phase0/                      # small evidence, not production modules
      src/zecalibrator/
        __init__.py                  # lightweight
        _version.py                  # one literal version source
        __main__.py                  # CLI only
        api/
          __init__.py
          v1/
            __init__.py              # expose only accepted contracts
        core/
        application/
        io/
        cli.py
        gui/
          app.py                     # PySide6 import isolated
        resources/icons/
        zesoftware_interop.json
      tests/
        science/
        contract/
        filesystem/
        gui/
        integration/
      .github/workflows/ci.yml

Directories/modules are created only as required for the current phase; no speculative empty framework implementation. Start with standard setuptools src-layout build, pytest, and build/wheel tooling. Product version from _version.py; API_VERSION belongs to api.v1, not derived from product version. Scientific dependencies: NumPy/Astropy; metadata range parsing: packaging; platform storage: platformdirs. Do not inherit ZSSS's full dependency list. PySide6>=6.6 is a candidate GUI floor to qualify on the chosen Python/OS matrix. GPU remains an optional future dependency with a proven closure.

Entrypoints:

- console_scripts: zecalibrator = zecalibrator.cli:main.
- gui_scripts: zecalibrator-gui = zecalibrator.gui.app:main.
- python -m zecalibrator routes to CLI; no implicit GUI import.
- GUI entry without [gui] emits a precise missing-extra diagnostic; engine/CLI remain usable.

Phase 2 CLI is only --help/--version and resource/storage smoke. API declarations are not promises that calibration works. Build sdist→wheel→clean venv, launch outside checkout; verify wheel namespaces and package-data. No scientific stubs returning success. License must be an explicit project-owner decision before distribution; don't copy a repository license as a “packaging default”.

## 15. Durable AGENTS.md and TODO.md contracts

Avoid document proliferation: ARCHITECTURE includes interoperability/GUI/packaging; SCIENCE_CONTRACT includes matching; PROVENANCE owns schemas/transactions; this immutable mission is handoff context, not a second live ledger. Add docs/INTEROPERABILITY.md only if actual cross-product contracts later outgrow ARCHITECTURE.

### AGENTS.md permanent instructions

Every coding/review agent must, before edits:

1. Read AGENTS.md and TODO.md.
2. Inspect the actual repository root, branch, full HEAD and working tree. Do not mistake the enclosing workspace Git repository for ZeCalibrator.
3. Read current phase instructions, last ACCEPTED gate and evidence.
4. Read relevant existing tests/contracts before editing.
5. Preserve unrelated work; understand every unexpected diff.
6. Respect DEFERRED and explicit non-goals; do not implement future phases.
7. Do not equate implementation completion with accepted gate.
8. Use public inter-product interfaces only; preserve raw science, portability and no in-place defaults.
9. Report exact validation commands/results and limitations.
10. Update durable state when Junior actually accepts a gate; workers may record evidence, not self-promote the project.
11. Require current operational A2A policy; never guess transport/session keys.
12. No publication/destructive actions without explicit user authorization; local commits only when the bounded mission authorizes them.

### TODO.md living ledger

Required first-screen fields:

    Baseline: branch / full SHA / source authority / working-tree condition
    Active phase and mission_id
    Last ACCEPTED gate: date / accepting Junior / exact evidence
    Status: NOT_STARTED | ACTIVE | REVIEW | REWORK | ACCEPTED | BLOCKED | HUMAN_GATE
    Implemented (not automatically verified)
    Independently verified
    Remaining
    Known issues and inherited failures
    Blockers / decisions needed
    DEFERRED (with future gate, not a vague wish list)
    Next exact mission (scope + pass conditions + exclusions)
    Reports and reproducible witness/test commands

Before repository initialization, branch/SHA explicitly N/A, never guessed main. After commits, record the implementation SHA being accepted; do not create a circular demand that the ledger contains its own future commit hash. Include report content summary/project-relative evidence so an inaccessible .a2a-report path is not the only continuity.

Phase acceptance updates the ledger and relevant contract docs, not a sprawling conversation log. Preserve this architecture document as a baseline; amendments need dated rationale and links in TODO.

## 16. Junior / Coco / Nono protocol

Junior is the sole downstream architect and acceptance authority; no callback to Astra or dependency on this conversation. Coco implements the current bounded mission. Nono independently reviews and issues **ACCEPT or REWORK**, with evidence, confirmed defects distinguished from suggestions. BLOCKED/FAILED may report an inability to complete review but never mean acceptance.

### 16.1 Mission lifecycle

1. Junior reads state, checks real branch/HEAD/status and baseline, chooses only active authorized phase, defines measurable acceptance and allowed files.
2. Before a **new independent** Coco mission, establish that no run is active, then reset agent:coco:main with sessions. Zeroed context/token counters establish reset, even if sessionId display persists.
3. Assign one stable mission_id; iteration-specific report under /home/tristan/.openclaw/workspace/.a2a-reports/.
4. Dispatch with sessions_send(agentId="coco", timeoutSeconds=0, message=bounded_contract). Require status=accepted; queued/steered disposition is not completion.
5. Return control; no synchronous waits, polling, repeated history reads, shell sleeps, native subagent fallback or automatic announce-based completion.
6. Coco completes work and validation, writes full durable report, explicitly callbacks the originating Junior source session.
7. Junior validates callback provenance/mission/path, reads report, independently inspects repo/diff/unexpected files and verifies essential tests/artifacts.
8. Delegate bounded Nono review asynchronously with same mission_id and exact branch/SHA/diff. Nono does not modify the repository.
9. Junior independently validates material findings; send confirmed defects as bounded REWORK to the same Coco session, no reset.
10. At most three corrective Coco iterations per task; then preserve work and report BLOCKED if unresolved.
11. No phase advances solely on a worker report. Junior records ACCEPT, BLOCKED or HUMAN_GATE and the next exact step.

Initial report paths, then increment r suffix each iteration:

    <mission_id>.coco.r0.md
    <mission_id>.nono.r0.md
    <mission_id>.coco.r1.md

Coco report includes mission/phase/status, implementation summary, files changed, commit if authorized, exact tests/results, branch/HEAD/status, artifacts, blockers and recommended next step. Nono report includes the same reviewed revision, verdict ACCEPT/REWORK, evidence and severity for findings.

### 16.2 Mandatory callback text in EVERY worker assignment

Include this complete routing contract, including REWORK:

> Read the incoming OpenClaw [Inter-session message] provenance: sourceSession=<session-key> is the EXACT Junior callback target. Never replace it with agent:main:main unless it literally is that source. Finish work and validation, write the entire durable report, then call sessions_send(sessionKey=<exact sourceSession>, timeoutSeconds=0, message=<callback>). Use sessionKey only for routing; no label. After status=accepted, final assistant output must be exactly REPLY_SKIP. Do not use automatic announce delivery as the report. On explicit callback failure, update the report; at most one same-target retry if plausibly transient. Never guess another session or reset one. If still failed, finish CALLBACK_FAILED with the error.

Coco wake-up:

    COCO_REPORT_READY
    mission=<mission_id>
    phase=<implementation|rework-N>
    report=<absolute-report-path>
    status=<DONE|BLOCKED|FAILED>

Nono wake-up (transport vocabulary preserved):

    NONO_REVIEW_READY
    mission=<mission_id>
    phase=<review-N>
    report=<absolute-report-path>
    status=<ACCEPT|FINDINGS|BLOCKED|FAILED>

Nono durable **REWORK** maps to callback status FINDINGS; do not silently change current callback parser vocabulary.

Junior intermediate callback turns that launch another worker finish REPLY_SKIP. Terminal acceptance/blocker/human-gate gets a normal concise result for Tristan. Reply-step suppression: REPLY_SKIP; intermediate announce-step: ANNOUNCE_SKIP. Workers never carry substantive reports through the automatic ping-pong loop.

Recovery only after user request, explicit delivery failure or concrete interruption evidence. Read expected report first; inspect worker history at most once per recovery step with sessionKey/limit and optional offset/includeTools. Do not invent messageId/sessionId. Never reset first or blindly resend. If activity is uncertain preserve the session and report ambiguity.

Runtime mismatch/missing required tool: stop affected delegation and report the exact error. Do not silently enable Code Mode, invent transport, or spawn Coco/Nono natively. No workers were started during this architecture mission.

## 17. Sequential roadmap with acceptance gates

This order follows archaeology: first demonstrate raw-domain hazards and missing metadata, then freeze science, then bootstrap, headless CPU/library/provenance/API, desktop and packaging, and only then repair the ZSSS seam. Provenance/cancellation are implemented with services, not bolted on after the GUI.

**The gates below define future work, not parallel authorization.** Junior splits a phase into small bounded Coco missions when needed, accepts each before starting the next. A phase cannot be marked ACCEPTED with skipped required evidence.

### Phase 0 — Mission 1: baseline and scientific boundary witnesses

**Scope:** read-only ecosystem verification and local documentation/research evidence. Establish AGENTS/TODO continuity immediately, even before Git bootstrap.

**Deliver:** source-authority inventory, redacted representative headers, supplied-asset hashes, independently calculated equation tables, loader negative-control witness, proposed phase-1 decision list.

**Gate G0:**

- Revalidate exact sources/drift and no external repo changes.
- Distinguish raw, processed and synthetic samples; demonstrate BZERO error and negative/invalid destruction in the actual current ZSSS loader.
- Explain why normalize=False and post-loader insertion are insufficient.
- At least one documented coherent calibration equation and double-subtraction negative control.
- Every unknown (masters, offset/readout, optical train, thermal tolerance, license) remains visible.
- Coco report + Junior independent replay/inspection + Nono ACCEPT.
- No production package, GUI, integration, backend or master-builder code.

See the final section for the exact executable assignment.

### Phase 1 — Freeze science, metadata and API design

**Dependency:** G0.

**Scope:** SCIENCE_CONTRACT, architecture and provenance/API model specification; tiny independent mathematical/header fixtures only. Resolve profiles needed for first usable cohort.

**Gate G1:**

- Equations for dark-including-bias / bias-removed dark / flat-dark / qualified bias-flat / normalized-flat / explicit control fixed.
- DQ bits, flat CFA normalization/scalars/quality, units, precision and saturation/missing metadata rules unambiguous.
- Matching matrix and ambiguity resolution tested via declarative expected cases, including observed aliases/conflicts.
- Header/domain/HDU specification, API ownership/errors/cancel/progress and version schemas fixed.
- First supported acquisition/profile is explicit; if no real masters, synthetic-only qualification is labelled and real scientific acceptance stays open.
- No invented nonzero temperature tolerance or detector defaults.
- Nono ACCEPT and Junior sign-off; scope-changing science/identity choices escalated with concrete alternatives.

### Phase 2 — Repository/package/resource/storage bootstrap

**Dependency:** G1.

**Scope:** independent Git repo/branch if authorized by the bounded mission, src namespace, version, packaging, entry-point stubs that only report help/version, resource manifest and user-path adapter, test/CI skeleton. Preserve artwork.

**Gate G2:**

- Verify independent Git toplevel (not parent workspace), branch/SHA/status.
- Wheel and sdist built; install wheel generated from sdist into clean environment.
- Package/version/--help and CLI work from unrelated CWD; no scientific success stubs.
- Resource hashes match canonical supplied assets; GUI icon decoder smoke where Qt available.
- Engine/API import works without Qt, ZeAlfie, ZSSS or GPU.
- Linux/Windows/macOS basic CI jobs active for namespace/version/resources/user paths; tests isolate user roots.
- No production scientific arithmetic yet. Desktop artifacts are not platform-qualified releases.

### Phase 3 — Headless raw decoder and CPU calibration primitive

**Dependency:** G2.

**Scope:** strict raw I/O and pure equations with explicitly supplied validated masters; immutable outputs and DQ; basic application cancellation/progress. No automatic library matching or UI.

**Gate G3:**

- Arithmetic against hand/float64 witnesses; identity and dark/bias double-subtraction negative controls.
- FITS unsigned BZERO, nontrivial BSCALE (integer and floating), BLANK, NaN/Inf, endian, malformed/conflicting headers and HDU ambiguity.
- Float32 output, valid negatives, CFA four-plane normalization/phase, shape/ROI exact rejection.
- RGB/processed input rejection; no mutation of arrays/headers/files; all-invalid refusal.
- Cancellation and fault injection leave no success/partial mutation.
- Scientific tests run headlessly on three-OS matrix; quantify precision envelope.
- Nono ACCEPT; real master quality remains separately labelled if unavailable.

### Phase 4 — Library matching and auditable application plans

**Dependency:** G3.

**Scope:** indexes, profile declarations, deterministic matcher, immutable plans, full provenance and TOCTOU guards.

**Gate G4:**

- Reordering directories/candidates leaves selection/ambiguity invariant.
- Correct role-specific rules: filter ignored for dark, required for flat; flat-dark exposure compared to flat.
- Missing camera instance/ROI/offset/readout rejects unless qualified declaration supplies it.
- Alias agreement/conflict, no-match, ambiguity, manual valid selection, master semantic mismatch and changed-file detection.
- Same equation/masks as Phase 3; full required provenance serializes/replays; no hash self-reference.
- Read-only masters, atomic versioned index/migration, concurrent/cancelled scan and inaccessible files.
- Same operation without ZeAlfie and ZSSS; Nono ACCEPT.

### Phase 5 — Public API 1.0 contract and consumer isolation harness

**Dependency:** G4.

**Scope:** export only accepted functions/types through api.v1; capability discovery and wheel metadata.

**Gate G5:**

- Tests install built wheel outside checkout; import only public API.
- FitsFrameSource vs ArrayFrameSource parity without temporary calibrated FITS; ownership/lifetime tests.
- Cheap discovery has no Qt/CuPy/process/network/config-write effects; capability static vs readiness states accurate.
- Minimum API contract and current package compatibility; malformed version, missing module, missing internal dependency, probe exception and missing capability cases.
- Dummy consumer port remains operational with absent/incompatible/unhealthy provider. No actual ZSSS changes.
- API_VERSION and product version demonstrably independent; wheel interop declaration agrees.
- Cancellation/progress/typed failure behavior documented and tested; Nono ACCEPT.

### Phase 6 — Standalone CLI, bounded batch and transactional FITS

**Dependency:** G5.

**Scope:** calibrate_batch capability, user commands for inspect/index/match/calibrate, deterministic per-file outputs, DQ/CALPROV and batch manifest.

**Gate G6:**

- End-to-end small raw+masters produces expected signed float32 output, DQ and complete provenance; original hashes unchanged.
- No-calibration control and requested-role failure correctly differentiated.
- CLI exit contract: 0 all requested work committed; 2 request/validation refusal; 3 processing/partial failure; 130 cancellation, with machine-readable statuses.
- Input/master/output alias collision rejection, no-clobber publication, disk-full/interrupted-write/cancel tests.
- Bounded iterator memory, handles closed, partial batch manifest truthful.
- Standalone clean-environment install without ZeAlfie/ZSSS; three OS filesystem/resource gates; Nono ACCEPT.
- Add calibrate_batch to advertised capability only after acceptance.

### Phase 7 — PySide6 minimum application

**Dependency:** G6.

**Scope:** GUI client of existing services; no new scientific algorithms.

**Gate G7:**

- Same plan/result/provenance via CLI/API/GUI on identical fixture.
- Long operation off event loop; progress/cancel responsive, closing safe, stale events rejected.
- Inspectable matches/rejection reasons and explicit uncalibrated/partial/failed states.
- Resource and application identity tests from unrelated CWD/source/wheel.
- Qt offscreen tests on CI plus interactive Linux/Windows/macOS worker/event-loop/dialog witness before claiming GUI qualification on each.
- No scientific code in widgets/slots; headless imports still independent; Nono ACCEPT.

### Phase 8 — Independent cross-platform application packaging

**Dependency:** G7.

**Scope:** reproducible desktop artifacts and platform qualification. Proposed default: PyInstaller onedir per-platform proof, then Windows installer/macOS app/Linux packaged launch; managed runtime remains wheel-based.

**Gate G8:**

- Native OS/architecture dependency closure, pinned build inputs and hashes/licenses.
- Windows x64 executable/installer, macOS arm64 .app, Linux x86_64 onedir/archive launch in clean native environments. Other architectures only if separately witnessed.
- No host Python, source checkout, CWD or shell-dependent normal app behavior.
- Icons in app/window/taskbar/Dock/desktop, paths with spaces/Unicode, real dialogs, cancellation, scientific sample and output provenance.
- Settings/library data outside bundle/runtime; upgrade and uninstall preserve user masters.
- Frozen tests cover Qt plugins, NumPy/Astropy binaries and package resources, not just build success.
- Signing/notarization/installer publication require product-owner choices and external release authorization. Unsigned engineering artifacts are labelled, not marketed as fully release-ready.
- No remote release during the gate without prior authorization.

### Phase 9 — Qualified ZSSS public integration and ZeAlfie admission

**Dependency:** G8 plus compatible public API delivered to a reviewable artifact.

Split strictly:

- **9A:** bounded ZSSS signed raw-preparation prerequisite from §7.2; no calibration enabled yet. G9A requires legacy OFF control, signed float debayer/science path, reference/mask/units and export witnesses.
- **9B:** ZSSS optional public adapter; reference/batch/Boring/Classic/Drizzle/resume coverage. G9B requires absence/disabled/incompatible/unhealthy/no-match/preflight fallback, no mixed unannounced population, exact per-frame provenance and changed-plan resume rejection.
- **9C:** separate ZeAlfie catalog/deployment contract admission. G9C requires machine metadata, supported GUI extra/entry point, immutable artifact/SHA and shared dependency closure; no runtime broker.

Interoperability A–H evidence:

| Gate | Required witness |
| --- | --- |
| A standalone producer | ZeCalibrator without ZeAlfie/ZSSS |
| B standalone consumer | ZSSS without ZeCalibrator |
| C independent interoperability | Both installed together without ZeAlfie, public API only |
| D absence fallback | Remove provider, ZSSS startup/run works |
| E incompatibility fallback | Unsupported major and broken dependency isolated |
| F shared runtime | ZeAlfie-managed compatible dependency closure and real launch/calibration |
| G immutable provenance | Product/API/artifact/SHA identities retained |
| H rollback | Failed candidate does not replace known-good runtime; user library unaffected |

Nono reviews each bounded change separately; Junior verifies actual artifacts. Do not mark interoperability complete using only mocked producer/consumer tests.

### Phase 10 — Optional GPU calibration capability

**Dependency:** accepted CPU/application/public integration gates and explicit scheduling.

**Gate G10:** parity contract in §9, actual hardware witnesses, OOM/CPU fallback, cancellation and transaction safety, same matching/population and global flat scalars, honest throughput/memory report. GPU flag never implies execution. No GPU support claim on unwitnessed OS/backend.

### Phase 11 — Conditional master-building feasibility and implementation

**Dependency:** explicit product-owner scheduling; does not block v0.1.

- **11A feasibility gate:** exact missing public ZSSS reducer blocks inventoried; licensing, API, raw-only and recursion-free proof; compare reuse cost against a narrow alternative. No new reducer implementation in feasibility.
- **11B provider gate, only if adopted:** ZSSS dedicated public raw reducer, immutable N, CPU full/tiled equivalence, GPU parity where claimed, malformed/cancel/memory refusal and callback-OFF mechanical tests; no ZeCalibrator dependency.
- **11C ZeCalibrator master builder gate:** each of Bias/Dark/Flat/Flat-Dark separately qualified with real/synthetic witnesses, coherent preprocessing, population provenance, transactional output and standalone absence isolation.
- Nono ACCEPT each; no speculative shared core.

### Phase 12 — Future ZeMosaic adapter

**Dependency:** explicit scheduling after a new ZeMosaic archaeology.

**Gate G12:** public-only optional adapter at raw seam; geometry/masks/negatives, no master-tile population changes, ZeAlfie-independent operation and A–H relevant interoperability witnesses. No implementation merely because the API is already convenient.

## 18. CI and release qualification matrix

Do not defer portability until packaging. Separate jobs/artifacts, so skip is never reported as PASS:

| Evidence family | Start gate | Automation | Native/real-machine requirement |
| --- | --- | --- | --- |
| Unit/science | G3 | Linux/Windows/macOS hosted runners, pinned seeds, minimum + current Python/dependency endpoints | CPU arithmetic fixtures on each claimed architecture |
| API contracts | G5 | Installed wheel, no optional apps; mock versions and actual minimum/current compatible wheels | Real producer+consumer at G9 |
| Filesystem/resources | G2 onward | Three OS, unrelated CWD, isolated roots, Unicode/spaces/case/collisions/read-only | Long-path and desktop behavior where runner differs |
| GUI smoke | G2 limited / G7 full | Qt offscreen, imports, worker events, icon decode | Interactive native dialogs, cancel/close, scaling and shell identity |
| Packaging | G2 wheel/sdist; G8 native | GitHub Actions per-OS build and clean-install/frozen smoke | Clean end-user-like launch without developer tools/Python |
| Interoperability | G5 harness / G9 real | absence/version/failure matrices; public import audit | Shared ZeAlfie runtime + rollback witness |
| GPU | G10 | CPU-only fallback and injected failures on regular CI | Named CUDA hardware/driver/OS for executed parity/performance |
| Master reduction | G11 | exact-N full/tiled synthetic tests and recursion sentinel | Representative real calibration populations |

Initial Python matrix: minimum 3.11 plus the actual current supported ecosystem interpreter (3.13 observed in ZeAlfie packaging); later newest Python only after dependency wheels are validated. Record numpy/astropy/PySide6/platformdirs versions and OS/architecture, not only python --version. GitHub Actions definitions can be authored locally; triggering/publishing external runs follows the active user's authorization.

Observed release conventions are heterogeneous, not one ecosystem mandate:

- ZSSS: standard wheel/source install and Qt GUI entry; README shows Windows/Linux/macOS pip/venv routes; no native ZSSS freeze spec was found in the inspected inventory.
- ZeMosaic: PyInstaller spec and Inno installer; version.txt explicitly describes Windows x64/CuPy CUDA 12.x bundle. Not evidence of equivalent current macOS/Linux release qualification.
- ZeAlfie: Windows Inno wrapping a pinned private python-build-standalone runtime/wheelhouse, not freezing; macOS ARM64 private-runtime .app workflow. Its macOS document has an old “not executed” status plus later native-run references: treat current qualification as requiring evidence review, not as a fresh platform PASS.
- ZeAnalyser: src-layout resource packaging and macOS test workflow.
- None grants ZeCalibrator platform support by inheritance.

Early freeze choice is an engineering proposal to qualify, not a license to copy all ZeMosaic hidden-import/runtime hacks. If PyInstaller fails a demonstrable requirement, present a bounded alternative based on native evidence. No new major dependency or custom launcher architecture without a concrete reviewed need.

## 19. Explicit DEFERRED register

Unless moved by an accepted, explicitly scheduled gate:

| Deferred item | Earliest reconsideration |
| --- | --- |
| ZeMosaic integration | Phase 12 |
| Automatic master generation | Phase 11C |
| ZSSS raw-reduction API implementation | Phase 11B after feasibility acceptance |
| Shared ZeStackCore extraction | No scheduled gate; demonstrated duplicated requirements required |
| Advanced GPU optimization / non-CUDA backend | Phase 10 or later named extension |
| Destructive in-place calibration | Not in v1; separate explicit authorization/design |
| Sophisticated exposure/temperature dark scaling | Separate science qualification |
| Automatic approximate/fuzzy master matching | Separate profile/science qualification |
| Arbitrary master cropping/rotation/binning conversion | Separate geometry contract |
| Cloud/shared remote calibration libraries | Separate data/security/consistency architecture |
| Acquisition-device control | Outside current product role |
| Cosmetic hot-pixel replacement/denoise/stretch features | Outside calibration science |
| Uncertainty propagation | Separate versioned capability, not fabricated v1 variance |
| Cross-frozen-executable in-memory transport | Later public process/shared-memory protocol; no repository probing |
| New artwork or needless ICO/ICNS generation | Only on explicit resource requirement |
| Signing/notarization/publication/auto-update subsystem | Explicit release mission; ZeAlfie remains manager |

Deferred does not mean rejected. TODO must retain these items visibly; workers cannot use them as convenient adjacent scope.

## 20. Risks, unresolved questions and decision policy

1. **Real masters missing from qualified evidence.** This is the largest scientific qualification gap. Do not promise useful automatic calibration of the current S50/S30 libraries yet. Phase 0/1 can pass with honest synthetic-only scope; a real-calibration release claim requires matched real masters.
2. **Acquisition provenance may be insufficient.** Offset, readout and detector-instance data are absent in observed lights. Choose explicit profile/import declarations backed by evidence; never fill zero/model defaults to improve match rate.
3. **Seestar firmware preprocessing is not established here.** “Light” and CFA do not prove no in-camera calibration. Obtain documented firmware/acquisition provenance or keep double-calibration risk visible and reject uncertain cohorts.
4. **Flat optical-train state and illumination stability.** Filter/shape alone cannot establish response compatibility. Freeze user setup identifiers and validity policy; do not infer optical equivalence.
5. **Thermal/bias policies.** No nonzero temperature tolerance was qualified. Exact matching may reduce usability; that is preferable to false science. A nonzero range requires detector-specific witnesses.
6. **Flat statistics/invalidity.** Four-plane median and default 90% validity are explicit proposed science/quality choices requiring G1 acceptance; no per-tile normalization shortcut.
7. **ZSSS signed-domain migration is nontrivial.** Legacy normalized variance/debayer/rescale/reference assumptions make integration a separately gated consumer change, not a one-line hook.
8. **Stale docs and moving repositories.** Recheck current HEAD/source; this document names exact baselines. Existing dirty work is preserved.
9. **License and distribution policy.** Product license, target macOS minimum/architectures, installer signing and notarization remain owner decisions. Recommend familiar GPL-compatible policy only after explicit choice; do not assert approval.
10. **Frozen independence vs shared-environment API.** Python array API is for a compatible Python environment. Two isolated standalone executables cannot import one another's embedded modules. Do not promise duplication-free cross-process interoperability without a later protocol.
11. **Memory/precision.** Float32 appropriate for witnessed 16-bit ADU; wider acquisitions and exact flat statistics need bounded memory/error evidence.
12. **Native artifacts unqualified.** This mission inspected source definitions, not the installed Windows/macOS apps or GPU behavior. Mark missing witnesses NOT_RUN, not PASS.

Resolve routine implementation choices inside the accepted contract. Escalate only true blocking scientific/product decisions or changes outside authorized scope, with a concrete evidence-backed recommendation. No permission request is required to perform Phase 0's reversible local evidence/documentation work.

## 21. FIRST EXACT JUNIOR ASSIGNMENT — Phase 0 / Mission 1

**Mission id:** ZC-P0-M1-RAW-BOUNDARY-20260915  
**Objective:** establish a resumable, verified architecture/science baseline before any production code exists.

### Authorized locations and actions

- Read this document and the current workspace AGENTS/interoperability contract.
- Read the named ecosystem repositories; verify real toplevel/branch/full SHA/status.
- Under ZeCalibrator only, create:
  - AGENTS.md with the permanent contract in §15;
  - TODO.md with active Phase 0 and last accepted gate NONE;
  - research/phase0/REPORT.md containing baseline, science tables, decisions and commands;
  - research/phase0/evidence.json with redacted header/asset/source identities and executed witness results;
  - research/phase0/raw_boundary_witness.py as a **standalone research script**, not an installed module.
- Existing icons and this mission document must remain unchanged.
- Durable worker reports go under the canonical .a2a-reports directory.
- Do not initialize Git, create pyproject/package/GUI/CI files, commit, edit ecosystem code or copy full real FITS into the repository in this first mission. Independent Git bootstrap belongs to Phase 2.

### Exact work

1. Junior checks current facts and records drift versus §2; do not overwrite/clean/rebase any repository. Verify the ZeCalibrator directory is not already an independent repo before recording N/A.
2. Junior establishes Coco idle/reset for this independent mission, dispatches the bounded scope asynchronously with the full callback contract.
3. Coco inventories all supplied icons by SHA-256, bytes and dimensions/container signature; no generated artwork.
4. Coco re-reads at least one S50 and one S30 acquisition-origin FITS and the synthetic Light_001 fixture. Save only the matching-relevant allowlisted fields, path/content identity, source qualification and missing fields. If paths moved, locate replacements read-only; if unavailable, record BLOCKED for that witness, never invent headers.
5. Reproduce the temporary uint16 BZERO witness and signed/NaN witness against the **actual loader function bodies at the recorded current ZSSS SHA**, not a reimplementation claimed to be the loader. A source-AST extraction with explicit dependency injection is acceptable research isolation; record that it is not an end-to-end application test.
6. Compute §4.3's hand/reference equations independently; prove double bias subtraction changes the answer. Add a flat-pedestal witness, a four-CFA-plane scale witness and a same-shape/different-ROI **design rejection case**. These are mathematical/contract observations, not implementation of a matcher/calibration engine.
7. REPORT explains:
   - raw/HDU/physical scaling seam and every identified consumer path;
   - why post-loader calibration and normalize=False are invalid;
   - why signed float debayer/rescale is a later prerequisite;
   - dark/flat/bias/flat-dark ownership and no-double-subtraction;
   - actual aliases and missing required metadata;
   - precise Phase 1 unresolved decisions and DEFERRED boundaries.
8. Run only the research script and documentation/asset checks. No full ZSSS suites, GPU benchmarks, GUI or packaging build is needed to pass G0.
9. Coco writes **/home/tristan/.openclaw/workspace/.a2a-reports/ZC-P0-M1-RAW-BOUNDARY-20260915.coco.r0.md**, then callbacks exact Junior source and finishes REPLY_SKIP.
10. Junior independently reruns the small witness, checks calculations against expected tables, verifies all changed/created paths and hashes, and verifies ecosystem repository states are unchanged by this mission.
11. Junior sends Nono a read-only review of the exact artifacts/source identities; report **/home/tristan/.openclaw/workspace/.a2a-reports/ZC-P0-M1-RAW-BOUNDARY-20260915.nono.r0.md** with ACCEPT/REWORK and the callback protocol.
12. Only after material findings are resolved, Junior marks G0 ACCEPTED in TODO with exact evidence and writes the next bounded Phase 1 specification mission. End this Mission 1 at that boundary; do not start production implementation as part of closing G0.

### Mission 1 acceptance checklist

- [ ] Real baseline and authority recorded; parent workspace Git not mistaken for product Git.
- [ ] AGENTS/TODO allow a new Junior session to resume without this conversation.
- [ ] Canonical icons byte-identical; no generated assets.
- [ ] Real vs synthetic vs processed FITS distinguished; missing metadata not invented.
- [ ] Actual loader witnesses reproduced with expected BZERO and signed/NaN failures.
- [ ] Independent equation/CFA/pedestal witnesses and negative controls correct.
- [ ] Raw geometry/matching rejection cases stated without implementing later mechanisms.
- [ ] ZSSS entry and reference seams + signed-path blocker explicitly documented.
- [ ] No production code, package, GUI, integration, master generation or GPU implementation.
- [ ] No ecosystem edits, commits, push, merge, release or deployment.
- [ ] Junior independent verification and Nono ACCEPT recorded.
- [ ] TODO last accepted gate G0 and next exact Phase 1 mission are truthful.

**Passing Phase 0 requires only baseline, contracts and reproducible research evidence. Nothing from a later phase needs to be implemented to pass it.**
