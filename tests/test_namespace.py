"""Bootstrap tests: namespace and version."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib import metadata

import pytest


def test_package_version():
    import zecalibrator

    assert zecalibrator.__version__ == "0.0.1"


def test_version_literal_single_source():
    import zecalibrator._version as v

    assert v.__version__ == "0.0.1"
    assert isinstance(v.__version__, str)


def test_package_and_version_module_agree():
    import zecalibrator
    import zecalibrator._version as v

    assert zecalibrator.__version__ == v.__version__


def test_installed_package_metadata_matches_version_literal():
    """The installed distribution metadata derives from the same literal source.

    Skips when the distribution is not installed (e.g. a bare ``PYTHONPATH=src``
    run): the metadata is a build artefact, not a second version source.
    """
    import zecalibrator

    try:
        installed = metadata.version("ZeCalibrator")
    except metadata.PackageNotFoundError:  # pragma: no cover - depends on the env
        pytest.skip("ZeCalibrator distribution metadata is not installed")
    assert installed == zecalibrator.__version__ == "0.0.1"


def test_clean_import_has_no_optional_heavy_dependencies():
    code = textwrap.dedent(
        """
        import sys
        import zecalibrator
        import zecalibrator.api.v1
        import zecalibrator.storage
        import zecalibrator._resources
        assert zecalibrator.__version__ == "0.0.1"
        # Cold import of the package + lazy API facade must not pull heavy or
        # optional modules (NumPy/Astropy are loaded lazily on model use).
        for mod in ("PySide6", "QtWidgets", "QtCore", "QtGui", "zealfie", "seestar", "cupy", "numpy", "astropy"):
            assert mod not in sys.modules, "unexpected import: " + mod
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
