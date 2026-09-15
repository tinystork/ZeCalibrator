# AGENTS.md — ZeCalibrator

Permanent working contract. Authority: `ASTRA_MISSION_ZECALIBRATOR_FOR_JUNIOR.md`
(ZC-ARCH-20260915 revision 1) §15, plus the workspace normative contract
`/home/tristan/.openclaw/workspace/projects/ZESOFTWARE_INTEROPERABILITY_RULES.md`
(version 1.0). This file is a durable instructions contract, not a second
architecture ledger.

## Working rules

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

## Context

- Product owner: Tristan Nauleau. Architect/reviewer: Junior. Reviewer: Nono.
  Worker: Coco.
- ZeCalibrator is an independently installable application and reusable
  scientific calibration engine for raw astronomical sensor FITS frames. The
  initial useful release applies existing masters; it does not build them.
- Scientific ordering: FITS storage → physical raw sensor values + original
  metadata → master resolution → sensor calibration → signed calibrated
  raw/CFA float32 + validity + provenance → consumer-owned debayer / colour /
  registration / stacking.
- Never calibrate already-normalized or debayered data. Do not rotate, flip,
  transpose, crop, resample, register, reproject, replace hot pixels,
  white-balance, stretch or clip negatives within calibration. Preserve
  array[y, x] ↔ physical sensor pixel (x, y).
- CPU is the scientific reference; GPU may change execution strategy only.
  Matching is deterministic and conservative. Missing compatible masters is
  preferable to a dubious match. Preserve original files by default.
- One bounded mission and one accepted gate at a time. A roadmap entry is not
  authorization to implement it early.
