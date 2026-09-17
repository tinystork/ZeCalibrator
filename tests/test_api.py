"""Public-API namespace contract tests (bootstrap smoke, updated for G5).

The public facade exposes the approved G5 surface (ARCHITECTURE §2/§3 +
owner-approved A-D). Cold import and ``get_api_info()`` are cheap: no NumPy/
Astropy/Qt/ZeAlfie/ZSSS/CuPy/SQLite import at discovery time.
"""

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


def test_get_api_info_matches_static_surface():
    import zecalibrator
    import zecalibrator.api.v1 as v1

    info = v1.get_api_info()
    assert info.api_version == "1.0"
    assert info.product_version == zecalibrator.__version__
    assert info.capabilities == (
        "calibrate_frame",
        "calibration_library",
        "master_matching",
        "provenance",
        "cancel",
    )


def test_explicit_public_symbols_equal_all():
    import inspect

    import zecalibrator.api.v1 as v1

    public = {n for n in dir(v1) if not n.startswith("_")}
    extras = {
        n
        for n in public - set(v1.__all__)
        if not inspect.ismodule(getattr(v1, n)) and n != "annotations"
    }
    assert extras == set()
    assert set(v1.__all__) <= public


def test_cold_import_and_get_api_info_are_cheap():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import zecalibrator.api.v1 as v1
        assert v1.API_VERSION == "1.0"
        info = v1.get_api_info()
        assert info.capabilities == (
            "calibrate_frame", "calibration_library", "master_matching",
            "provenance", "cancel",
        )
        for mod in ("numpy", "astropy", "sqlite3", "PySide6", "QtWidgets",
                    "QtCore", "zealfie", "seestar", "cupy"):
            assert mod not in sys.modules, mod
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_lazy_access_loads_scientific_modules_on_use():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import zecalibrator.api.v1 as v1
        assert "numpy" not in sys.modules
        _ = v1.Geometry  # scientific model use loads numpy/astropy
        assert "numpy" in sys.modules
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
