# Changelog

All notable changes to ZeCalibrator are documented in this file.

## [0.1.0] — 2026-09-28

- **Canonical route-class key (C26)**: new public ``light_route_key(light)`` in the
  ``1.1`` contract (additive, no capability/version bump). Returns a deterministic,
  conservative, header-only class key that includes every fact the matcher consumes
  for compatibility + ranking (geometry/detector/acquisition/optical + the ``DATE-OBS``
  civil day), with no tolerance — so ZSSS never recomputes an approximate key.
- **Flat admission aligned with the Standard contract (C24)**: `open_session_library`
  no longer filters flats on `master_evidence_status` — the flat quality evidence
  (R4) is now informational `needs_attention`, resolved as traced UNVERIFIED by the
  single compatibility authority (Standard matcher) at route time. The necessary
  fields tier (orientation, filter, …) stays an eliminatory refusal. No science added,
  no tolerance extended to other fields.
- **Public session-library API**: `open_session_library` / `SessionLibrary` /
  `SessionLibraryResult` (admission + role identification from a folder, in-memory
  calibration with master reuse, freeze/resume provenance).
- **Auto-route capability**: per-light `resolve_light` auto-routes each light to its
  exposure/temperature-matched master plan.
- **Session orientation declaration**: `SessionDeclaration(orientation=…)` supplies a
  fallback-only sensor-orientation fact; a master's own header evidence always wins.
- **GAIN/EGAIN evidence fix**: `GAIN` (camera gain) and `EGAIN` (e⁻/ADU) are decoded as
  distinct physical quantities instead of being conflated on one field.
- **roi_origin evidence fix**: `XORGSUBF`/`YORGSUBF` are decoded into the `roi_origin`
  fact so Bayer masters carrying both cards are no longer refused.
- **ZSSS interoperability**: `zesoftware.interop.json` declares the `zecalibrator.api.v1`
  surface (API 1.1, eight capabilities) for optional ZSSS discovery.
- **RAW/master discrimination** (evidence-based, bounded): a producer that stamps its
  masters with a stacking-count card (`STACKCNT`/`NCOMBINE`/`NFRAMES`, value >= 2)
  refuses an identifiable-role file of the same producer that carries no such proof
  (`NOT_A_MASTER`). This is a proof-based admission rule, not a contractual entry limit.

**API version**: 1.1 (unchanged).
