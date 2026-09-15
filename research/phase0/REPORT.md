# ZeCalibrator — Phase 0 / Mission 1 research report

**mission_id:** ZC-P0-M1-RAW-BOUNDARY-20260915
**phase:** implementation
**date:** 2026-09-15 (Europe/Paris)
**worker:** Coco
**authority:** `ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md` (ZC-ARCH-20260915 rev 1),
`ZESOFTWARE_INTEROPERABILITY_RULES.md` (v1.0)

This report is documentation and read-only research evidence only. No production
code, package, GUI, integration, backend, or master-builder code was created.

---

## 1. Summary

Established a resumable, verified architecture/science baseline for ZeCalibrator
before any production code. Verified the §2.1 ecosystem baseline (branch/HEAD/
status) is unchanged; inventoried all `icons/` assets byte-for-byte; re-read the
S50/S30 acquisition FITS and the synthetic `Light_001` fixture; reproduced the
loader BZERO and signed/NaN witnesses against the **actual** ZSSS loader bodies
at HEAD `77c92da0259ef7a41fa37d7aa6f3fab7a840735c` (AST-extracted, dependency-
injected, not end-to-end); and independently computed the §4.3 equation contract
with negative controls.

Status of the requested deliverables:

| Deliverable | Path | State |
| --- | --- | --- |
| Permanent contract | `AGENTS.md` | created |
| Living ledger | `TODO.md` | created |
| Report | `research/phase0/REPORT.md` | this file |
| Evidence | `research/phase0/evidence.json` | created (script-generated) |
| Research script | `research/phase0/raw_boundary_witness.py` | created, executed |
| Duplicate report | `.a2a-reports/ZC-P0-M1-RAW-BOUNDARY-20260915.coco.r0.md` | created |

`icons/` and the mission document remain byte-unchanged (see §8).

---

## 2. Baseline and source authority

### 2.1 ZeCalibrator repository identity

`/home/tristan/.openclaw/workspace/projects/zecalibrator` is **not** an
independent Git repository. `git rev-parse --show-toplevel` resolves to
`/home/tristan/.openclaw/workspace`, which itself has no committed HEAD
(uncommitted repo). Before this mission the directory contained only the
mission document and `icons/`. Branch/SHA are therefore **N/A**, never guessed.

### 2.2 Ecosystem baseline (verified, unchanged)

| Repository | Branch | HEAD | Working tree |
| --- | --- | --- | --- |
| ZeAlfie | chore/za-tmpfs-cleanup-gate | dd74f8475c8bf8baa35f920b304f58bc6bb3f4e6 | 5 M + 2 ?? (exactly §2.1 list) |
| zeseestarstacker | main | 77c92da0259ef7a41fa37d7aa6f3fab7a840735c | tracked clean; ?? research/drizzle_contract/* |
| zemosaic | beta | c03d0bb965d073b12ad9978094327829f0d0c366 | clean |
| ZeSolver-main | main | 091fbb7a3621a585a45dc110af8a52f870f1fcc9 | clean |
| ZeSolver | test | 47bfa4cb7aabbdd0a486099b06557c16dea681ae | clean |
| zeanalyser | za-perf-p2.0-instrumentation | 38184521997b77553fd854c454065c17ed1ead2a | clean |

No drift from the Junior-verified baseline was observed. No ecosystem file was
modified, committed, pushed, merged, or deleted.

---

## 3. Icon asset inventory

All 12 assets under `icons/` inventoried by SHA-256, byte size and container
signature. The three canonical assets match §13 exactly; no generated artwork.

| File | Bytes | SHA-256 (full) | Container / dimensions |
| --- | --- | --- | --- |
| zecalibrator_icon.png | 2,459,037 | 7ad1589da6dd0d4b16780bd687ff79dec8fc999e4d0cc03431303117357b5ee4 | PNG RGBA 1254×1254 |
| zecalibrator.ico | 218,054 | 717a04a871867de4cd32f90c3383d8414edbb2c960290ba346a1514e6bc584ca | ICO, 7 images (16×16, 24×24, …) |
| zecalibrator.icns | 660,414 | a43b9b1c53aaedb98ee33c0591ce376a07947967490c67e7018ded2e2665dda6 | ICNS ("is32" type) |
| zecalibrator_16x16.png | 1,560 | 7793ee217e847cc5b506f7d5bc56bb7a64bfd42ad309276cb6b0d592d78f8b20 | PNG 16×16 (8-bit colormap) |
| zecalibrator_24x24.png | 2,089 | 04bbb703d545b012bf63513662d1c5bd4197b92113427971e9f6a5fab5c9c8da | PNG RGBA 24×24 |
| zecalibrator_32x32.png | 3,465 | 2789d59d6cd2157c8f5d00c37fb8ef77cd2742fd2aa4945d6c407145d8be10a6 | PNG RGBA 32×32 |
| zecalibrator_48x48.png | 7,027 | cc66ea671658872b7c521e6bae4a7e8a4566b4b868a41819ee5ef8e98e3a1c87 | PNG RGBA 48×48 |
| zecalibrator_64x64.png | 11,519 | c5b050eff9b13c254e0901a5eca53fc05ebeb47700091f70acec70e4f3678cd5 | PNG RGBA 64×64 |
| zecalibrator_128x128.png | 39,782 | 1be5ea466e395f59f52b445947e24daefcc7c1dea6cf2abaa015b6aba75f79dc | PNG RGBA 128×128 |
| zecalibrator_256x256.png | 135,324 | 205e6fb9260d0e19b71e37069aa753d975109cd36412624dae46eef67bdd4ece | PNG RGBA 256×256 |
| zecalibrator_512x512.png | 471,181 | be09aa8b33ca0e17ee1ed269c22e48b3b345b7875edbe0e28e06d789199995d8 | PNG RGBA 512×512 |
| zecalibrator_1024x1024.png | 1,716,483 | c3582489faed7dfec694719618fd5661c8da2c9236d05a33edd6e0c1b9b18b51 | PNG RGBA 1024×1024 |

The `16x16.png` is the only variant with an 8-bit colormap container; the others
are 8-bit/color RGBA. Native decoding of ICO/ICNS remains a packaging gate, not
an assumed PASS in Phase 0.

---

## 4. FITS witnesses and the raw/HDU/physical-scaling seam

### 4.1 Raw / HDU / physical-scaling seam

Three distinct domains exist in FITS and must never be conflated:

1. **Stored samples** — what `do_not_scale_image_data=True` yields: the on-disk
   integer/float array as written (for these Seestar files, big-endian int16
   `>i2`).
2. **Physical samples** — `BSCALE × stored + BZERO` applied **exactly once**:
   the unsigned 16-bit ADU values (physical = stored + 32768 for BZERO=32768).
3. **Normalized / processed** — data-dependent min/max, debayer, white balance,
   stretch, clip, etc. This is consumer-owned presentation, **not** raw.

The ZSSS loader opens with `do_not_scale_image_data=True`, selects the first
image HDU, copies and sanitizes the header, applies shape heuristics, then a
non-finite→0 repair and min/max normalization. It never applies BSCALE/BZERO.
Therefore its output is **stored-or-normalized**, never physical ADU.

### 4.2 Witnesses read (allowlisted fields only)

| Field | S50 (049) | S30 (058) | Synthetic Light_001 |
| --- | --- | --- | --- |
| Qualification | real acquisition light | real acquisition light | synthetic LIGHT, **not** a master |
| INSTRUME | Seestar S50 | Seestar S30 | SYNTH |
| CREATOR | ZWO Seestar S50 | ZWO Seestar S30 | **missing** |
| NAXIS1 × NAXIS2 | 1080 × 1920 | 1080 × 1920 | 1920 × 1080 |
| XBINNING / YBINNING | 1 / 1 | 1 / 1 | missing / missing |
| CCDXBIN / CCDYBIN | 1 / 1 | 1 / 1 | missing / missing |
| XORGSUBF / YORGSUBF | 0 / 0 | 0 / 0 | missing / missing |
| EXPTIME / EXPOSURE | 20 / 20 | 10 / 10 | 10 / missing |
| CCD-TEMP | 30.5625 °C | 28.125 °C | missing |
| GAIN | 80 | 200 | missing |
| FILTER | IRCUT | IRCUT | missing |
| BAYERPAT | GRBG | GRBG | missing |
| BITPIX / BSCALE / BZERO | 16 / 1 / 32768 | 16 / 1 / 32768 | 16 / 1 / 32768 |
| stored dtype / shape | >i2 (1920,1080) | >i2 (1920,1080) | >i2 (1080,1920) |

**Qualification notes (never invent headers):**

- S50/S30 are acquisition-origin; all 19 allowlisted fields present. No
  OFFSET/readout-mode/ADC-depth/detector-instance field exists in either
  header — these remain **unknown, not zero/default**.
- The synthetic fixture has INSTRUME=SYNTH, NAXIS 1920×1080 (landscape,
  transposed relative to the real 1080×1920 portrait lights), and is missing
  CREATOR, all binning/ROI fields, EXPOSURE, CCD-TEMP, GAIN, FILTER, BAYERPAT.
  The directory name `master` does **not** make it a calibration master.

### 4.3 Actual loader witnesses (reproduced against real bodies)

The loader bodies (`load_and_validate_fits`, `sanitize_header_for_wcs`,
`debayer_image`) were AST-extracted from
`seestar/core/image_processing.py` at HEAD
`77c92da0259ef7a41fa37d7aa6f3fab7a840735c` and executed with injected real
NumPy + real `astropy.io.fits` and a `cv2` stub. **This is source-AST isolation,
not an end-to-end application test.**

**Witness A — uint16 BZERO.** Physical input `[[0,100],[200,65535]]`, header
BZERO=32768, stored int16 `[[-32768,-32668],[-32568,32767]]`.

- Astropy physical read: `[[0,100],[200,65535]]` (correct physical decode).
- ZSSS default loader: `[[0, 0.0015259021893143654], [0.0030518043786287308, 1.0]]`
  (float32) — a min/max normalization, matching §3.2 exactly.
- ZSSS `normalize=False, attempt_fix_nonfinite=False`:
  `[[-32768,-32668],[-32568,32767]]` — the **signed storage offsets**, not
  physical ADU.

**Witness B — signed float + NaN.** Input `[[-2,0],[2,NaN]]` (float32). Default
loader → `[[0,0.5],[1,0.5]]`. The `-2` is mapped to 0 by min/max, and NaN is
replaced by 0 by `nan_to_num` before normalization; valid negative/NaN science
is silently destroyed.

**Conclusion:** `normalize=False` is **not** a valid scientific raw decoder
(it returns signed storage, not physical ADU), and the default loader is a
data-dependent min/max normalization that destroys physical scale and negatives.

---

## 5. Why post-loader calibration and normalize=False are invalid

**normalize=False is invalid** because it returns `data_raw.astype(np.float32)`
with `do_not_scale_image_data=True` still in effect: the on-disk int16 offsets
(`-32768…32767`) are surfaced directly, without the `BSCALE/BZERO` physical
decode. It is neither physical ADU nor a normalization; it is a signed storage
leak. Two acquisitions with different `BZERO`/`BSCALE` would be silently
incomparable, and `-32768`/`32767` are not sensor ADU values.

**Post-loader calibration is invalid** because by the time
`load_and_validate_fits` returns, the physical scale and invalid mask are gone:

1. non-finite → 0 repair has already replaced NaNs/±Inf with a plausible number;
2. min/max normalization has mapped the data into `[0,1]` (or 0.5 for a
   constant image, or 0.0 for all-non-finite), and clip discards anything
   outside that range;
3. the header copy and original scaling descriptor (BITPIX/BSCALE/BZERO/BLANK)
   are no longer the authoritative input to calibration.

Calibrating `[0,1]` normalized data is mathematically a different operation
than calibrating physical ADU, and any invalid mask reconstructed after the
repair is necessarily a guess. The calibration seam must therefore sit at
**HDU selection + original header capture + physical raw decoding**, *before*
the sanitizer/shape/nonfinite/minmax path (mission §7.1), around
`queue_manager.py:11130`.

---

## 6. Signed float debayer/rescale is a later prerequisite

The current ZSSS debayer (`debayer_image`) does:

```python
img_uint16 = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)
```

i.e. it clips signed ADU to `[0,1]`, scales to uint16, OpenCV-demosaics, and
returns `RGB_uint16 / 65535`. The Drizzle input rescale (`rescale_01_to_adu`)
similarly clips negatives. If calibration were merely moved earlier, it would
compute correct signed `C` values that the consumer then destroys at debayer or
rescale. Therefore §7.2's bounded ZSSS raw-domain preparation gate — a
consumer-owned float-capable debayer, a signed science path with no convenience
clipping/unsigned wrap/data-dependent scale heuristics, and preservation of
colour/variance/invalid-mask handling — is a **separate later ZSSS mission**,
not part of ZeCalibrator Phase 0 and not something this mission implements.

---

## 7. Master semantics, ownership, and no double subtraction

### 7.1 Roles (not filenames)

| Master type | Meaning | Additive branch used against |
| --- | --- | --- |
| bias | additive pedestal/offset | light, or to remove bias from a dark/flat |
| dark | thermal signal; descriptor must state bias_state | light (exposure-matched) |
| flat | response map; descriptor must state raw vs normalized, additive history | light (division) |
| flat_dark | additive for the flat's own correction | flat (exposure-matched to flat) |

A `dark` "including bias" (`Dinc`) and a "bias-removed" dark (`D0`) are
different descriptors. Subtracting `B` on top of `Dinc` double-subtracts bias.
Flat-dark corrects the **flat**, never the light directly.

### 7.2 Coherent additive branches and the equation witness

Choose exactly one light additive branch:

- Control / no additive correction: `A = L`.
- Bias only: `A = L − B`.
- Dark including bias: `A = L − Dinc` (do **not** subtract B again).
- Bias-removed dark: `A = L − B − D0`.

`C = A / R` when a compatible flat response is requested; `C = A` otherwise.

Reference witness (independently computed, matches §4.3):

```
L     = [[110, 50], [210, -2]]
Dinc  = [[ 10, 10], [ 10, 10]]
B     = [[  4,  4], [  4,  4]]
D0    = [[  6,  6], [  6,  6]]
R     = [[  1,0.5], [  2,  1]]
C     = [[100, 80], [100,-12]]
```

- Dark-including-bias: `A = L − Dinc = [[100,40],[200,-12]]`, `C = [[100,80],[100,-12]]`. ✓
- Bias-removed dark: `A = L − B − D0 = [[100,40],[200,-12]]`, `C = [[100,80],[100,-12]]`. ✓
- **Double subtraction (negative control):** `C_double = (L − Dinc − B)/R = [[96,72],[98,-16]]`, which differs — proving subtracting bias on top of an already-bias-inclusive dark changes the answer.

### 7.3 Flat-pedestal witness

`Finc = [[100,110],[120,90]]`, `FDinc = 10` (pedestal). Correcting first
(`Fcorr = Finc − FDinc`, then `R = Fcorr/median(Fcorr)`) is **not** equal to
normalizing the uncorrected flat (`R_wrong = Finc/median(Finc)`). The medians
differ (95 vs 105), so the two response maps differ, and a light divided by them
gives different `C`. An uncorrected flat's pedestal must be removed before
normalization; normalizing an uncorrected flat is not a substitute for
correction.

### 7.4 Four-CFA-plane scale witness (GRBG)

GRBG parities: R=(even,odd), G1=(even,even), G2=(odd,odd), B=(odd,even).
With a flat whose G1 plane is 200 while G2/R/B are 100, four per-plane medians
(100/200/100/100) yield a uniform response `R≡1.0`, whereas a single global
median (100) over-corrects G1 to 2.0. This proves four parity planes
**including separate G1/G2** are required, not one global median, and not a
per-tile median.

### 7.5 Same-shape / different-ROI design rejection

Two frames can both be NAXIS 1080×1920 yet differ in `XORGSUBF`/`YORGSUBF`.
A one-pixel origin shift flips Bayer CFA parity on every row, changing which
plane is G1 vs G2 and R vs B. Same shape does **not** imply same ROI. Detector
instance/model, sensor dimensions, ROI origin/extent, binning, gain/offset/
readout and CFA phase are all exact-match criteria; a matcher must reject on
ROI-origin mismatch even when NAXIS shape is identical, and must not crop a
full-frame master to match a light (v1).

---

## 8. Validation executed

Commands (exact) and results:

```bash
# 1. research script (only scientific/asset code run)
cd /home/tristan/.openclaw/workspace/projects/zecalibrator
/home/tristan/.openclaw/workspace/projects/zeseestarstacker/.venv/bin/python \
    research/phase0/raw_boundary_witness.py
# → exit 0; evidence.json written; all assertions pass
#   (loader witness A/B match §3.2; equation C=[[100,80],[100,-12]];
#    double-subtraction differs; flat-pedestal, CFA, ROI witnesses recorded)

# 2. asset / icon hashes (byte-unchanged checks)
sha256sum icons/* ; file icons/*
# → canonical icon.png/.ico/.icns hashes match §13; 12 assets present

# 3. mission document byte-unchanged
sha256sum ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md
# → dbcf826d52031627fd7c817b2f8e430bdf581ecf9ee36c4ec4061d009762020c (unchanged)

# 4. ecosystem repo state re-check (read-only)
git -C <each repo> rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git status --short
# → all match §2.1 baseline; no changes attributable to this mission
```

No full ZSSS suites, GPU benchmarks, GUI, or packaging builds were run (per
G0 scope). No Git init/commit/push/merge/release/deploy performed.

---

## 9. Files changed

Created under `/home/tristan/.openclaw/workspace/projects/zecalibrator` only:

- `AGENTS.md`
- `TODO.md`
- `research/phase0/REPORT.md`
- `research/phase0/evidence.json`
- `research/phase0/raw_boundary_witness.py`

Duplicate report (required):
`/home/tristan/.openclaw/workspace/.a2a-reports/ZC-P0-M1-RAW-BOUNDARY-20260915.coco.r0.md`

No commit was created (none authorized). No ecosystem file was touched. `icons/`
and the mission document remain byte-unchanged.

---

## 10. Blockers, uncertainties, and Phase 1 decisions

**Blockers:** none for G0 (documentation/research scope).

**Open scientific unknowns (kept visible, not invented):**

- No qualified real bias/dark/flat/flat-dark masters exist in evidence.
- Offset, readout mode, ADC depth, and detector-instance are absent in observed
  lights — cannot be defaulted to zero or inferred from BITPIX.
- Seestar firmware preprocessing (in-camera calibration) not established; CFA +
  "Light" do not prove raw.

**Phase 1 unresolved decisions (for Junior, with evidence-backed recommendation):**

1. **License** — must be an explicit product-owner decision before distribution.
2. **First supported cohort/profile** — real S50/S30 lights currently have no
   matching masters; synthetic-only qualification must stay labelled.
3. **Temperature tolerance** — default zero scientific tolerance; no invented
   nonzero window without detector-specific witnesses.
4. **Flat normalization quality gate** — default ≥90% valid samples per plane
   is a product threshold to be accepted, not a physical law.
5. **Detector-instance / offset / readout** — explicit profile/import declaration
   backed by evidence vs hard reject on missing required fields.
6. **Alias conflict policy** — EXPTIME/EXPOSURE, XBINNING/CCDXBIN, etc. must
   require agreement; two disagreeing aliases never silently resolve.

**DEFERRED boundaries:** full register in mission §19; see TODO.md for the
list. None were implemented.

---

## 11. Recommended next step

Junior: independently re-run `research/phase0/raw_boundary_witness.py`, verify
the equation/negative-control tables, verify created paths + hashes, confirm
ecosystem repo states are unchanged, then dispatch Nono read-only review for
`ACCEPT`/`REWORK`. On ACCEPT, mark G0 ACCEPTED in TODO.md and write the next
bounded Phase 1 specification mission. Do not begin production implementation
as part of closing G0.
