"""Numpy-free meta constants/types for cheap API discovery.

Imported at cold ``import zecalibrator.api.v1`` time without pulling NumPy,
Astropy, Qt or SQLite, so ``get_api_info()`` stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The five implemented public capability identifiers (ARCHITECTURE §2).
#: ``calibrate_batch`` / ``GPU`` / ``master_building`` are NOT advertised.
CAPABILITIES: tuple[str, ...] = (
    "calibrate_frame",
    "calibration_library",
    "master_matching",
    "provenance",
    "cancel",
)

PROVENANCE_SCHEMA = "zecalibrator.provenance.v1"


@dataclass(frozen=True)
class ApiInfo:
    """Cheap API discovery result (no filesystem/network/Qt side effects)."""

    api_version: str
    product_version: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))


__all__ = ["CAPABILITIES", "PROVENANCE_SCHEMA", "ApiInfo"]
