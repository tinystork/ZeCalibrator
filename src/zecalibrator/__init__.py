"""ZeCalibrator — raw astronomical sensor FITS calibration engine.

Bootstrap skeleton (Phase 2): this build provides package layout, a literal
product version, CLI ``--help``/``--version`` entry points, package resources
and a storage-path adapter only. No scientific calibration, matching or other
numerical capability is implemented or advertised in this release.

Importing this package performs no Qt / ZeAlfie / ZSSS / CuPy / GPU imports.
"""

from __future__ import annotations

from zecalibrator._version import __version__

__all__ = ["__version__"]
