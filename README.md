# ZeCalibrator

ZeCalibrator is an independently installable application and reusable scientific
calibration engine for **raw astronomical sensor FITS frames**. The initial
useful release applies existing calibration masters; it does not build them.

> **Status — G6 ACCEPTED.** The public `zecalibrator.api.v1` surface exposes six
> implemented capabilities: `calibrate_frame`, `calibration_library`,
> `master_matching`, `provenance`, `cancel`, and `calibrate_batch` (Phase 6
> batch + transactional FITS outputs + CLI). Qualification remains SYNTH-BASE-1
> synthetic-only. See [`TODO.md`](TODO.md) for the accepted gate ledger.

## License

**GPL-3.0-or-later** (GNU General Public License v3, or at your option any later
version). See [`LICENSE`](LICENSE).

The copyright-holder name in `LICENSE` is a neutral, owner-confirmed placeholder
pending the product owner's final attribution; it is not an invented legal
identity and is not copied from any other repository.

## Scope label

The first supported acquisition cohort is **SYNTH-BASE-1** and is **synthetic-only
qualification**. Real-camera (e.g. Seestar S50/S30) qualification is **not**
claimed and remains open until real compatible masters and acquisition/firmware
evidence are qualified.

## Install (from a source checkout)

```bash
python -m pip install .
# optional GUI extra (PySide6) — not required for the engine/CLI:
python -m pip install '.[gui]'
```

## Entry points

```bash
zecalibrator --help          # console entry point
zecalibrator --version
python -m zecalibrator       # routes to the CLI
zecalibrator-gui             # GUI launcher; emits a precise diagnostic without [gui]
```

The CLI exposes `inspect`, `index`, `match` and `calibrate` (deterministic
scriptable JSON output, exit codes 0/2/3/130) as a thin facade over
`zecalibrator.api.v1`, plus `--help`/`--version`.

The GUI is a bounded public-API-only PySide6 client (Phase 7, **G7 NOT YET
ACCEPTED**). With the optional `[gui]` extra installed, `zecalibrator-gui`
launches a desktop client for selecting raw FITS lights + evidence/HDU, opening
or indexing a calibration library, running an off-thread inspection/match
preflight (MATCHED / NO_MATCH / AMBIGUOUS with structured rejection reasons),
choosing explicit modes, calibrating in memory or exporting standalone FITS, and
showing per-file results with progress/cancel and safe close. Without the `[gui]`
extra, `zecalibrator-gui` emits a precise missing-extra diagnostic and exits
non-zero while the engine and CLI remain usable.

## Storage paths

Persistent user state is resolved through a single `zecalibrator.storage`
adapter over `platformdirs` (`appname="ZeCalibrator"`, `appauthor="ZeSoftware"`,
version-independent, non-roaming roots). Importing the package and resolving
paths never create directories or read live user settings; tests inject all
roots explicitly. See `src/zecalibrator/storage.py`.

## Icons and resources

Supplied artwork lives under [`icons/`](icons) and is preserved byte-for-byte.
Selected byte-identical copies are packaged under
`src/zecalibrator/resources/icons/` and listed in
`src/zecalibrator/resources/icon_manifest.json` (canonical path → packaged path
→ SHA-256). Runtime loading uses `importlib.resources` and never searches the
top-level `icons/` directory or writes into the package.

Development/build synchronization command (byte-identical copy + hash re-check):

```bash
# from the product root
mkdir -p src/zecalibrator/resources/icons
for f in icons/*; do cp "$f" "src/zecalibrator/resources/icons/"; done
python - <<'PY'
import hashlib, json
m = json.load(open("src/zecalibrator/resources/icon_manifest.json"))
for e in m["files"]:
    src = hashlib.sha256(open(e["canonical_path"], "rb").read()).hexdigest()
    pkg = hashlib.sha256(open("src/zecalibrator/" + e["packaged_dir"] + "/" + e["name"], "rb").read()).hexdigest()
    assert src == e["sha256"] == pkg, e["name"]
print("icons synchronized and verified")
PY
```

## Development

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel build pytest
.venv/bin/python -m build          # sdist + wheel into dist/
.venv/bin/python -m pytest -ra     # bootstrap test suite
```

## Disclaimers

- Scientific arithmetic (FITS decode, calibration equations, deterministic
  matching, batch orchestration, transactional FITS output) is implemented and
  accepted through G6, with SYNTH-BASE-1 synthetic-only qualification.
- No real-camera or master-building qualification is claimed.
- No publication, release, signing or deployment is authorized by this source.
