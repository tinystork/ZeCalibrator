"""ZeCalibrator I/O package (Phase 3 / G3 scope).

Strict raw FITS decoding only. No library indexing, no ZSSS import, no GUI.
"""

from __future__ import annotations

from zecalibrator.io.raw_decoder import DecodedFrame, decode_fits

__all__ = ["DecodedFrame", "decode_fits"]
