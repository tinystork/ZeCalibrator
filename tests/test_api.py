"""Bootstrap tests: headless public-API namespace smoke."""

from __future__ import annotations


def test_api_version_literal():
    import zecalibrator.api.v1 as v1

    assert v1.API_VERSION == "1.0"


def test_api_version_independent_of_product_version():
    import zecalibrator
    import zecalibrator.api.v1 as v1

    assert v1.API_VERSION != zecalibrator.__version__
    assert v1.API_VERSION == "1.0"
    assert zecalibrator.__version__ == "0.1.0"


def test_v1_namespace_exposes_no_capability():
    import zecalibrator.api.v1 as v1

    public = sorted(n for n in dir(v1) if not n.startswith("_"))
    assert public == ["API_VERSION"]


def test_api_import_is_headless():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import zecalibrator.api.v1 as v1
        assert v1.API_VERSION == "1.0"
        for mod in ("PySide6", "QtWidgets", "QtCore", "zealfie", "seestar", "cupy", "numpy", "astropy"):
            assert mod not in sys.modules, mod
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
