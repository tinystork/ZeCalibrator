"""Public API v1 namespace (bootstrap skeleton).

``API_VERSION`` is the frozen public API target. It is independent of the
product version (``zecalibrator._version.__version__``), the science-contract
version, the provenance schema, the library schema and the matching-policy
version (ARCHITECTURE §2).

No scientific calibration capability is implemented or advertised in this
release, and importing this module performs no Qt / ZeAlfie / ZSSS / CuPy /
GPU import and no filesystem or network activity.
"""

API_VERSION = "1.0"

__all__ = ["API_VERSION"]
