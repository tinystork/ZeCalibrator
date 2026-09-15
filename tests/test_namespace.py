"""Bootstrap tests: namespace and version."""

from __future__ import annotations

import textwrap
import subprocess
import sys


def test_package_version():
    import zecalibrator

    assert zecalibrator.__version__ == "0.1.0"


def test_version_literal_single_source():
    import zecalibrator._version as v

    assert v.__version__ == "0.1.0"
    assert isinstance(v.__version__, str)


def test_package_and_version_module_agree():
    import zecalibrator
    import zecalibrator._version as v

    assert zecalibrator.__version__ == v.__version__


def test_clean_import_has_no_optional_heavy_dependencies():
    code = textwrap.dedent(
        """
        import sys
        import zecalibrator
        import zecalibrator.api.v1
        import zecalibrator.storage
        import zecalibrator._resources
        assert zecalibrator.__version__ == "0.1.0"
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
