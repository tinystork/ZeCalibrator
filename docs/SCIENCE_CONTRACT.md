# ZeCalibrator — Science Contract (specified v1)

**Status:** Phase 1 specification — owner decisions reconciled 2026-09-15;
**G1 ACCEPTED by Junior, 2026-09-15.** This specifies future product behavior,
not an implemented or real-camera-qualified calibration engine.

Label legend (three distinct classes, kept separable):

- **INVARIANT** — mathematical necessity and/or a rule authorized verbatim by
  `ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md` (ZC-ARCH-20260915 rev 1) §§4–5.
- **ENGINEERING BOUND** — a mission-authorized numeric convention/constant that
  is a design bound, **not** a mathematical necessity and not a physical law
  (e.g. `1e-6` serialization tolerances, the `R > 1e-6` division floor, the
  `2^24` float32 integer-exactness bound).
- **ADOPTED PRODUCT POLICY** — Tristan's explicit decisions dated 2026-09-15;
  product screening choices are not physical laws.

Authority: `ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md` §4–§5 and the workspace
`ZESOFTWARE_INTEROPERABILITY_RULES.md` v1.0.

---

## 1. Scope and ordering

ZeCalibrator calibrates **raw astronomical sensor FITS frames** by applying
existing masters. Scientific ordering (INVARIANT):

```
FITS storage → physical raw sensor values + original metadata
    → calibration master resolution → sensor calibration
    → signed calibrated raw/CFA float32 + validity + provenance
    → consumer-owned debayer / colour / registration / stacking
```

The engine owns: raw decode, master resolution, subtraction of additive
pedestal (bias/dark), division by a normalized flat response, validity/DQ, and
provenance. Everything downstream (debayer, colour, registration, stacking,
stretch, white balance, hot-pixel cosmetics) is **consumer-owned** and outside
calibration.

The engine MUST NOT rotate, flip, transpose, crop, resample, register,
reproject, replace hot pixels, white-balance, stretch, or clip negatives within
calibration. `array[y, x]` corresponds to physical sensor pixel `(x, y)`
(INVARIANT).

---

## 2. Frame domain and decoding

### 2.1 Supported input domain (INVARIANT)

One explicitly selected **2D mono** or **2D Bayer/CFA sensor plane**. No RGB
cubes, stacked science images, compressed camera RAW, multisensor assembly, or
implicit mosaic handling. Multi-HDU FITS must select a unique supported sensor
plane or require an explicit HDU identifier; never silently choose the first
plausible image.

### 2.2 Storage vs physical vs processed domains (INVARIANT)

Three domains are distinct and must never be conflated:

1. **Stored samples** — the on-disk array as written (what
   `do_not_scale_image_data=True` yields).
2. **Physical samples** — `BSCALE × stored + BZERO` applied **exactly once**.
3. **Normalized / processed** — any data-dependent min/max, debayer, white
   balance, stretch, clip, etc.

Calibration operates on **physical ADU** (see §3 for units). It never calibrates
already-normalized or debayered data.

### 2.3 FITS scaling — apply BSCALE/BZERO exactly once (INVARIANT)

```
physical_sample = BSCALE × stored_sample + BZERO
```

- Apply **exactly once**, including scaled floating FITS where applicable.
- Defaults follow the FITS standard (BSCALE=1, BZERO=0), **not** manufacturer
  conventions.
- Detect integer `BLANK` in **stored space before scaling**. A stored sample
  equal to the integer `BLANK` value is invalid; do not scale it and then test.
- Treat `NaN` and `±Inf` as invalid samples.
- A floating FITS with non-trivial BSCALE/BZERO is a scaled floating case: still
  apply the linear decode exactly once; never assume float storage implies
  already-physical values.

### 2.4 Endianness (INVARIANT)

Byte order is **storage**, not camera identity. Convert deliberately to
native-endian contiguous float32 **only after** physical decoding. Use float64
for decode/statistical intermediates where needed; the scientific return buffer
is float32.

### 2.5 Precision and error refusal (ENGINEERING BOUND)

- float32 exactly represents integer ADU through `2^24` (16777216). This is an
  IEEE-754 fact, stated as an engineering bound for the supported range.
- Accept the witnessed 8–16-bit acquisition range.
- Larger integer ranges, extreme BSCALE, or catastrophic cancellation require an
  explicit error-bound witness **or refusal**; never a silent precision-loss
  claim. There is no implicit ADC-depth inference.

### 2.6 Duplicate cards and contradictory aliases (INVARIANT)

Keep **all original candidate cards** and normalized values. Never silently take
one of two disagreeing aliases. Confirmed v1 aliases (§5.2) that disagree
produce a **reason-coded rejection**, not a pick.

### 2.7 Domain / HDU / processed-history rejection (INVARIANT)

- Require a **raw-domain declaration** backed by header/profile/import metadata.
- Recognized processed-history markers cause rejection.
- Uncertain legacy master semantics require an explicit persisted import
  declaration, never a guess.
- Ambiguous HDU (no unique supported 2D sensor plane) → require explicit HDU
  identifier or reject.

### 2.8 No firmware/ADC inference (INVARIANT)

Never derive ADC saturation or raw status from `BITPIX` or `BAYERPAT`. Never
independently qualify Seestar firmware preprocessing; that is not established in
this project's evidence. A "Light" file with CFA does **not** prove unprocessed
raw. This project makes **no** claim about in-camera calibration.

---

## 3. Units

### 3.1 Physical units (INVARIANT)

Physical acquisition **ADU** for v1. A unitless normalized response flat is a
separate declared role. Do not infer gain conversion from a `GAIN` card.
Electron/rate data require an explicitly supported conversion contract,
otherwise reject. Output physical units are ADU (`BUNIT`), calibrated but still
in the sensor ADU domain.

### 3.2 Exposure/temperature units (INVARIANT)

Exposure in seconds; temperature in °C after any declared unit conversion.
Equality tolerances are stated in §6.3.

---

## 4. Geometry contract (INVARIANT)

Geometry is an exact contract: shape, binning, ROI origin/extent, orientation,
CFA phase, and detector identity.

- Same shape does **not** imply same ROI.
- Do not crop a full-frame master to match a light in v1.
- Translation by even one pixel changes CFA phase (see §7.5).
- Preserve WCS cards if valid, but never solve or use WCS to align masters.

---

## 5. Metadata normalization

### 5.1 Value object

Normalize into an immutable `SensorMetadata` value object carrying: all original
candidate cards, normalized values, units, alias/profile version, and conflict
diagnostics. (Full field/type definition in `ARCHITECTURE.md` §3.)

`SensorMetadata` carries an explicit **optical identity** (`OpticalIdentity`:
`filter` and `optical_train_id`, each `known | unknown`). The light's optical
identity is a matching criterion for flat-vs-light (§6.2) and is **bound into the
calibration-plan identity** (`ARCHITECTURE.md` §3.3, `PROVENANCE.md` §2.3):
changing the light's filter or optical train changes the recorded plan identity.
Flat matching/reuse requires exact applicable optical facts. This recording is
not a dark/bias filter-equality criterion: a new plan may select the same
compatible additive masters for lights with different filters. For additive-only
(dark/bias) modes with no flat correction, filter/train are not required and
unknown is not a rejection.

### 5.2 Confirmed v1 aliases (INVARIANT)

| Field | Aliases |
| --- | --- |
| exposure_seconds | `EXPTIME` / `EXPOSURE` |
| bin_x | `XBINNING` / `CCDXBIN` |
| bin_y | `YBINNING` / `CCDYBIN` |
| roi_origin | `XORGSUBF` / `YORGSUBF` (binned-pixel convention) |
| detector model | `INSTRUME` |
| CFA | `BAYERPAT` |
| temperature | `CCD-TEMP` |
| gain | `GAIN` |
| filter | `FILTER` |

Possible future aliases (`CAMERA/DETECTOR`, `CCDGAIN/EGAIN`, `BLACKLEV/OFFSET`,
`XBIN/YBIN`, `CCDTEMP`, `READMODE/READOUTM`, `XBAYROFF/YBAYROFF`) are **NOT**
enabled merely by appearing here; each must be validated with header evidence.
`EGAIN` may mean conversion factor; `SET-TEMP` is not measured `CCD-TEMP`;
`TELESCOP` is not a camera serial.

### 5.3 String normalization (INVARIANT)

Trim FITS padding, Unicode-normalize, case-fold only fields declared
case-insensitive. Never merge S30/S50, different cameras of the same model,
arbitrary filter aliases, or optical trains.

---

## 6. Matching matrix (conservative, role-specific)

### 6.1 Legend

- `E` = exact/qualified identity.
- `T` = thermal/exposure policy.
- `Q` = quality context, not blind light-to-master equality.
- `—` = not a normal criterion.

### 6.2 Field applicability matrix (INVARIANT)

| Field | Bias vs light | Dark vs light | Flat vs light | Flat-dark vs parent flat |
| --- | --- | --- | --- | --- |
| Detector instance/model | E | E | E | E |
| Sensor dims, ROI, orientation | E | E | E | E |
| Binning / CFA phase | E | E | E | E |
| Gain / offset / readout mode | E | E | E by default | E |
| Acquisition ADU repr / ADC mode | E | E | E; stored-float master allowed | E |
| Temperature | T | T | Q; exact by default unless qualified stable response | T |
| Exposure | bias-class near-zero, NOT equal to light | T | flat-quality range, NOT equal to light | T against flat |
| Filter | — | — | E | — |
| Optical train / illumination geometry | — | — | E | — |
| Observation date | report only | report only | report/explicit validity interval | report only |

Role-specific rules that fall out of the matrix:

- **Dark ignores filter** (`—`): a dark master matches by detector/geometry/
  gain/readout/CFA and thermal/exposure policy, not by the light's filter.
- **Flat uses filter AND optical train** (`E`): filter equality alone is
  insufficient; optical train, sensor orientation, illumination mode, and an
  explicit validity interval matter.
- **Flat-dark compares exposure to the parent flat**, not to the light.
- **Bias exposure** is not an invented 0-second universal rule; it must satisfy
  a declared qualified bias acquisition range (near-zero for the detector).
- `BITPIX` is storage type, not acquisition ADC depth. A float32 master may
  match a uint16 light if documented physical units/acquisition mode match.
  Unknown ADC depth is never silently inferred as 16.

### 6.3 Exactness and tolerances (INVARIANT + ENGINEERING BOUND)

- Detector, ROI, binning, gain, offset, readout, and CFA are **exact** after
  qualified normalization. No nearest-neighbor or weighted similarity score.
- **ENGINEERING BOUND** — exposure equality permits only numerical serialization
  tolerance: `|t1 − t2| ≤ max(1e-6 s, 1e-6 × max(|t1|, |t2|))`. Not scientific
  exposure scaling.
- **ENGINEERING BOUND** — temperature default: **zero scientific tolerance**
  after unit conversion, with only a tiny parser equality tolerance `1e-6 °C`.
  Any nonzero thermal window must be named, versioned, and validated for the
  camera/master type in a future profile mission, supported by real witnesses.
  Temperature remains an important compatibility criterion. No generic nonzero
  window is permitted. See §11 item 3.

### 6.4 Unknown and missing fields (INVARIANT)

- Missing required fields block automatic selection.
- A reviewed user library/profile declaration can supply an unknown field with
  visible provenance.
- **unknown ≠ unknown**: two unknown values must not match automatically.
- Explicit "not applicable" requires a profile reason.
- Malformed numerics, duplicate conflicting cards, incompatible HDUs,
  undocumented master processing, or saturated/unusable flat normalization
  produce reason-coded rejection.

### 6.5 Selection procedure and outcomes (INVARIANT)

1. Freeze request, matching policy, library index revision, and role
   requirements **before** inspection.
2. Inspect raw metadata; enumerate masters by semantic role.
3. Reject incompatible candidates with per-field reason records.
4. Construct complete coherent sets, including the flat's own additive
   correction dependencies.
5. Zero sets → `NO_MATCH`; more than one distinct set → `AMBIGUOUS`; exactly one
   → `MATCHED`.
6. Byte-identical duplicate masters with identical descriptors may collapse to a
   single content identity, preserving all locations. Different hashes remain
   `AMBIGUOUS`.
7. Manual choice among valid candidates may resolve ambiguity and is recorded;
   it **cannot bypass hard science rejections**.
8. Revalidate selected content/HDU/descriptor identities at execution to detect
   edits after planning.

**Frozen request prevents implicit downgrade (INVARIANT):** the request fixes
required roles before matching. Failure to satisfy a requested role is
`NO_MATCH` or a typed failure, never a silent downgrade to a weaker equation.
There is **no** "roles are optional" flag in the request model; the supported
additive/flat modes enumerated in §7 each fix their exact required roles.

**Stage separation (INVARIANT):** request validation (malformed/unsupported
mode) is a distinct, earlier stage from library matching. Manual hard refusal of
a candidate happens at matching stage (a candidate is excluded by reason code),
never at request-validation stage and never as a bypass of a hard rejection.

No scientific tie-break on "newest", closest temperature, most frames, first
directory entry, or filename sort.

### 6.6 Declarative case tables

See `research/phase1/cases.json` — `matching_cases`. Each case references the
fully-qualified `hypothetical_synthetic_baseline` plus explicit overrides/
removals, states an exact outcome enum (`MATCHED`/`NO_MATCH`/`AMBIGUOUS`) and
stable reason codes, and is `declaration_only: true` (no matcher implementation
exists; a matcher is a Phase 4 deliverable). Alias normalization uses a separate
enum (`ACCEPTED`/`REJECTED`) because it is a different stage.

---

## 7. Equations (INVARIANT)

### 7.1 Notation

| Symbol | Meaning |
| --- | --- |
| `L` | raw light, physical ADU |
| `B` | bias master |
| `Dinc` | dark including bias (exposure-matched to light) |
| `D0` | bias-removed dark |
| `Bflat` | bias master used to correct the flat |
| `Finc` | unnormalized flat including its additive pedestal |
| `FDinc` | flat-dark including bias, exposure-matched to the flat |
| `FD0` | bias-removed flat-dark |
| `Fcorr` | corrected flat (pedestal removed), **unnormalized** |
| `s` / `s[p]` | normalization scalar (mono) / per-plane scalar (CFA) |
| `R` | normalized flat response (dimensionless) |
| `A` | additive-corrected numerator |
| `C` | final calibrated sensor plane (physical ADU) |

### 7.2 Light additive branches — choose exactly one (INVARIANT)

| Mode | Equation | Preconditions |
| --- | --- | --- |
| Control / no additive correction | `A = L` | explicitly requested identity/partial mode; never "fully calibrated" |
| Bias only | `A = L − B` | bias matched to light |
| Dark including bias | `A = L − Dinc` | same exposure, compatible thermal/acquisition; **do not subtract B again** |
| Bias-removed dark | `A = L − B − D0` | compatible B required; D0 descriptor explicitly bias_removed |

### 7.3 Flat preparation — choose exactly one (INVARIANT)

| Branch | Equation | Preconditions |
| --- | --- | --- |
| Flat-dark including bias | `Fcorr = Finc − FDinc` | FDinc exposure-matched to flat |
| Bias-removed flat-dark | `Fcorr = Finc − Bflat − FD0` | FD0 bias_removed; Bflat compatible |
| Bias-only flat | `Fcorr = Finc − Bflat` | **only** by explicit qualified short-flat profile declaring dark current negligible at that exposure/temperature |
| Already-corrected / normalized response | consume `R` directly | descriptor proves processing/normalization semantics; never subtract again |

`Fcorr` is the **corrected but unnormalized** flat. `R` is the normalized
response. They are distinct descriptors (see `ARCHITECTURE.md` §3.3
`MasterDescriptor.flat_form`): a corrected-unnormalized flat must not be confused
with a normalized response, and neither may be consumed without its declared
form.

### 7.4 Flat normalization (INVARIANT)

- **Mono:** `s = median( finite, positive, non-saturated, valid Fcorr )`;
  `R = Fcorr / s`.
- **Bayer CFA (GRBG):** four parity planes (separate `G1`, `G2`, `R`, `B`) each
  get their own globally computed median `s[p]`;
  `R[y,x] = Fcorr[y,x] / s[parity(y,x)]` using the **frozen CFA origin**. Never
  a per-tile median.

**Normalization count population (INVARIANT):** the median population is the
per-plane set of `Fcorr` samples that are finite, strictly positive, not
saturated (when a qualified saturation limit exists), and not invalid via the
flat's own additive correction. This population is computed once, globally,
before any tiled execution.

### 7.5 CFA phase and shifted origin (INVARIANT)

For GRBG the 2×2 tile is:

```
G R
B G
```

with parities `G1=(even,even)`, `R=(even,odd)`, `B=(odd,even)`, `G2=(odd,odd)`.

A **+1 x-origin shift** (XORGSUBF parity change) remaps the array parities to
the sensor tile as:

```
R G
G B
```

i.e. it swaps **G1 ↔ R** and **B ↔ G2**. It does **not** swap G1 ↔ G2. A +1
y-origin shift swaps G1 ↔ B and R ↔ G2. This is asserted in
`contract_witness.py` (`cfa_shifted_origin_grbg`).

> **Erratum (supersedes Phase 0 prose):** `research/phase0/evidence.json`
> `same_shape_different_roi_design_rejection.observation` states an x-shift
> changes "which plane is G1 vs G2 and R vs B". That wording is imprecise: the
> correct mapping is G1↔R and B↔G2 (not G1↔G2), as fixed here. Phase 0 files
> remain byte-unchanged; this note is the authoritative correction.

### 7.6 Final calibrated plane (INVARIANT)

```
C = A / R    when a compatible flat response is requested and available
C = A        when the explicit requested mode has no flat correction
```

### 7.7 Flat-only explicit partial mode (INVARIANT)

Flat-only is an explicit partial mode: `A = L` (no additive correction),
`C = L / R`. It is not a substitute for additive correction and not a generic
fallback; it is only produced when explicitly requested.

### 7.8 No double subtraction (INVARIANT)

- Do not subtract `B` in addition to `Dinc` (which already includes bias).
- Do not subtract bias or flat-dark a second time from an already-corrected or
  normalized flat response.
- The flat-dark corrects the **flat**, never the light directly.

### 7.9 Reference witnesses (executed)

All numeric expectations below are independently stated and asserted by
`research/phase1/contract_witness.py` (see `evidence.json`).

- `L=[[110,50],[210,-2]]`, `Dinc=[[10,10],[10,10]]`, `B=[[4,4],[4,4]]`,
  `D0=[[6,6],[6,6]]`, `R=[[1,0.5],[2,1]]`.
  - Control `A = L`.
  - Bias-only `A = [[106,46],[206,-6]]`.
  - Dark-incl `A = [[100,40],[200,-12]]`.
  - Bias-removed dark `A = [[100,40],[200,-12]]` (same).
  - `C(dark) = C(bias-removed) = [[100,80],[100,-12]]`.
  - Double subtraction negative control `(L−Dinc−B)/R = [[96,72],[98,-16]]` ≠ C.
- Flat pedestal: `Finc=[[100,110],[120,90]]`, `FDinc=10`, `Bflat=4`, `FD0=6`.
  Both coherent flat-dark branches give `Fcorr=[[90,100],[110,80]]`;
  bias-only flat gives `[[96,106],[116,86]]`. Normalizing the uncorrected flat
  is not equivalent to correcting it first.
- CFA four-parity medians and shifted-origin GRBG mapping (see §7.5).

---

## 8. Flat normalization population, plane count, floor vs quality

### 8.1 Population and plane count (INVARIANT)

- Normalization scalar is computed **once**, globally, over the canonical valid
  population for the plane (§7.4), before any tiled execution.
- Mono needs **at least one valid normalization sample**; CFA needs at least one
  valid sample **per plane**. Zero valid samples in any required plane makes the
  flat unusable.
- The valid population is: finite, `> 0`, non-saturated (when saturation is
  qualified), and explicitly valid (`Fcorr` not masked by additive/input
  invalidity).

### 8.2 Numerical floor (ENGINEERING BOUND)

Default numerical division floor: `R > 1e-6`, finite. Pixels at/below the floor
are **masked**, not raised to the floor. This is a numerical guard, **not** a
claim that arbitrarily weak flats are scientifically good.

### 8.3 Quality gate (ADOPTED PRODUCT POLICY, 2026-09-15)

Default acceptance requires **≥90% valid normalization samples in EACH CFA
parity plane**, including separate G1 and G2 (mono: its single plane).
This is a product screening threshold, **not a physical law**, and is separate
from subtraction/division arithmetic. A later versioned profile may replace
the threshold, with its actual value/version frozen in the plan and provenance.

Denominator = total geometric sample count per plane; numerator = valid samples
in the §7.4 median population. Boundary arithmetic (89/90/91 of 100) remains in
cases.json. Counts alone do not assess the spatial distribution of bad pixels
or qualify missing saturation/processing evidence. The R > 1e-6 response floor
is an independent numerical guard, not a substitute for this product screen.

---

## 9. Data quality (DQ) bits and count semantics (specified v1)

### 9.1 Mask representation (specified v1)

DQ is a `uint16` reason-bit mask with stable v1 meanings. Multiple reasons
coexist by bitwise OR. `mask == 0` means VALID.

**Five baseline pixel reason bits only — all five are invalid-reason bits.**
There is **no** per-pixel "missing evidence" bit and **no** quality-flag-only
bit (see §9.3 for the separate frame-quality diagnostic).

| Bit | Hex | Name | Output effect |
| --- | --- | --- | --- |
| 0 | `0x0001` | `INPUT_INVALID` | light sample invalid in stored space (BLANK/NaN/±Inf) or decode failure → output `NaN` |
| 1 | `0x0002` | `ADDITIVE_INVALID` | contributing bias/dark master sample invalid → output `NaN` |
| 2 | `0x0004` | `FLAT_INVALID` | flat response invalid (non-positive, non-finite, or ≤ floor) → output `NaN` |
| 3 | `0x0008` | `SATURATED` | sample at/above a **qualified** saturation limit → output `NaN` (evaluated on original acquisition values, never clipped) |
| 4 | `0x0010` | `ARITH_NONFINITE` | arithmetic produced NaN/±Inf (±Inf canonicalized to NaN + this bit) → output `NaN` |
| 5–15 | `0x0020`…`0x8000` | reserved | MUST be 0 in v1 |

### 9.2 Count semantics (specified v1)

- Per-sample mask = bitwise OR of all reasons for that sample.
- **Any nonzero invalidity mask ⇒ output `NaN`.** This is the conservative
  baseline contract: all five bits are invalid-reason bits.
- `valid_count = count(mask == 0)`; `invalid_count = total_count − valid_count`.
- Per-bit counts are **independent with overlaps**: a sample contributes to every
  bit it has set. In particular `saturated_count = count(mask & SATURATED)`
  (this includes a sample that is both saturated and input-invalid).
- `invalid_count` equals the number of samples with at least one bit set; it is
  not the sum of per-bit counts (which double-count overlaps).

### 9.3 Saturation and missing evidence (specified v1)

- `SATURATED` is set **only** when a **qualified acquisition saturation limit**
  is available, and is evaluated on the **original acquisition samples** (the
  light's decoded physical ADU), never on the calibrated output and never guessed
  from the largest observed pixel, `BITPIX`, or `BAYERPAT`. The engine never
  clips a saturated value; a saturated sample is masked (`NaN` + `SATURATED`).
- When the saturation limit is **unknown**, no `SATURATED` bit is set (the
  engine cannot determine it). This is surfaced as a **frame-quality
  diagnostic/assessment status** (`saturation_evidence = qualified | unknown`),
  **not** as a per-pixel invalidity bit and **not** as an automatic per-pixel
  invalidation.
- Consequence (specified): identity mode returns equal decoded float32 values and
  `mask == 0` when all pixels are finite and no invalidity applies, **even if
  saturation quality is UNKNOWN**. Unknown saturation quality is a frame-level
  assessment, not a pixel mask.
- **Missing required matching identity** (detector-instance/offset/readout)
  ⇒ **no automatic match** (`NO_MATCH` with reason code). **Unknown quality
  evidence** (e.g. saturation limit, or an unqualified flat) ⇒ an explicit
  frame-quality assessment/refusal with an explicit reason; required flat quality
  must have evidence. Versioned evidence-backed import declarations are allowed,
  but cannot fabricate facts or hide contradictions with FITS (§11 item 5).

### 9.4 Invalidity, all-invalid refusal, valid negatives (INVARIANT)

- Any sample with any invalidity bit set yields output `NaN` for `C`. `Inf` is
  canonicalized to `NaN` with `ARITH_NONFINITE`, never zero.
- **All-invalid frame → `FAILED`**, not empty success. "All-invalid" means every
  sample has at least one invalid bit (`invalid_count == total_count`).
  A frame that is saturated-only (every pixel `SATURATED`) is also all-invalid
  because `SATURATED` is an invalid-reason bit.
- Partial invalidity returns `COMPLETED_WITH_WARNINGS` with exact counts.
- **Negative finite `C` is valid** and never itself a masking condition.
- Contributing master invalidity reaches output bits as follows: an invalid
  bias/dark master sample at `(y,x)` ⇒ output `NaN` + `ADDITIVE_INVALID`; an
  invalid flat response at `(y,x)` ⇒ output `NaN` + `FLAT_INVALID`. Reference
  witnesses for reason-OR, counts (including SAT-only and IN|SAT overlap), NaN
  propagation, valid-negative, and all-invalid refusal are asserted in
  `contract_witness.py` (`dq_semantics_cases`) — these are reference arithmetic
  on tiny arrays, not a production engine.
- No fabricated uncertainty array (variance propagation is a separately
  versioned future capability).

---

## 10. Output

- Return signed calibrated raw/CFA **float32** + uint16 DQ + provenance.
- Standalone FITS output is `BITPIX=-32`, physical ADU `BUNIT`, no stale integer
  `BSCALE`/`BZERO`/`BLANK`, no stale `CHECKSUM`/`DATASUM`. Recompute relevant
  checksums. Store DQ as an image extension and full provenance per
  `PROVENANCE.md`.
- Identity mode returns equal decoded float32 values and mask without modifying
  the caller's array or the input file.

---

## 11. Owner policies — adopted 2026-09-15

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

Full operational details and evidence: ARCHITECTURE §14.

---

## 12. Versioning

- This contract targets **science-contract version `1.0`**, distinct
  from product version, public API version, provenance schema, library schema,
  and matching-policy version. Breaking science changes require a new major.
- The matching policy version is separate from the science version so a profile
  threshold change (e.g. a flat quality gate) does not silently change the
  equations.
