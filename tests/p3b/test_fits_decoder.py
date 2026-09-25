"""LOT1 tests: FITS/CFA conformity + decoder interop (requirements 4, 5)."""

import numpy as np
from astropy.io import fits

from research.p3b.generator import cfa_plane_label, generate_corpus
from tests.p3b.conftest import make_sensor, single_light_scenario


# ---------------------------------------------------------------------------
# FITS conformity (LOT1 requirement 5)
# ---------------------------------------------------------------------------

def test_bscale_bzero_applied_exactly_once(tmp_path):
    scenario = single_light_scenario(seed=9, n_frames=1)
    r = generate_corpus(scenario, tmp_path / "c")

    path = r.frame_paths["l0"]
    # Raw on-disk storage (before any scaling) is signed 16-bit.
    with fits.open(path, do_not_scale_image_data=True) as hdul:
        hdr = hdul[0].header
        stored = hdul[0].data
        assert hdr["BITPIX"] == 16
        assert hdr["BSCALE"] == 1
        assert hdr["BZERO"] == 32768
        assert stored.dtype in (np.dtype(">i2"), np.dtype("int16"))

    # The physical array is exactly stored + BZERO (with BSCALE=1), once.
    physical_expected = stored.astype(np.float64) + 32768.0
    assert np.array_equal(physical_expected, r.frame_arrays["l0"].astype(np.float64))


def test_native_unbinned_base0_coordinates(tmp_path):
    scenario = single_light_scenario(seed=9, n_frames=1)
    r = generate_corpus(scenario, tmp_path / "c")
    with fits.open(r.frame_paths["l0"]) as hdul:
        hdr = hdul[0].header
        assert hdr["XBINNING"] == 1
        assert hdr["YBINNING"] == 1
        assert hdr["XORGSUBF"] == 0
        assert hdr["YORGSUBF"] == 0


def test_cfa_phase_and_origin_explicit(tmp_path):
    scenario = single_light_scenario(seed=9, n_frames=1)
    r = generate_corpus(scenario, tmp_path / "c")
    with fits.open(r.frame_paths["l0"]) as hdul:
        hdr = hdul[0].header
        assert hdr["BAYERPAT"] == "GRBG"


def test_array_y_x_corresponds_to_sensor_x_y(tmp_path):
    # A site declared at sensor (x, y) must land at array[y, x].
    from research.p3b.model import SiteSpec

    scenario = single_light_scenario(
        seed=11, n_frames=1,
        sites=(SiteSpec(site_id="s0", x=7, y=9, cfa_class="STABLE_ANOMALY_CORRECTED_BY_DARK"),),
    )
    r = generate_corpus(scenario, tmp_path / "c")
    arr = r.frame_arrays["l0"]

    # The anomaly is a strong positive bump at (y=9, x=7); the symmetric pixel
    # (y=7, x=9) is just background. Compare against the neighbourhood median.
    neighbourhood = arr[5:13, 3:13].astype(np.float64)
    background = float(np.median(neighbourhood))
    assert arr[9, 7] > background + 1000  # the hot pixel lives at (y,x)=(9,7)
    assert abs(arr[7, 9] - background) < 200  # (7,9) is ordinary background


def test_cfa_plane_label_matches_repo_geometry():
    # Mirror check against the repo's authoritative geometry module.
    from zecalibrator.core.geometry import sensor_plane

    for pattern in ("GRBG", "RGGB", "BGGR", "GBRG"):
        for y in range(8):
            for x in range(8):
                assert cfa_plane_label(y, x, pattern) == sensor_plane(y, x, pattern)
    assert cfa_plane_label(3, 4, "mono") == "mono"


# ---------------------------------------------------------------------------
# Decoder interop (LOT1 requirement: existing decoder reads generated frames)
# ---------------------------------------------------------------------------

def test_existing_decoder_reads_generated_frames_with_expected_physical_values(tmp_path):
    from zecalibrator.core.metadata import ImportDeclaration
    from zecalibrator.io.raw_decoder import decode_fits

    scenario = single_light_scenario(seed=13, n_frames=2)
    r = generate_corpus(scenario, tmp_path / "c")

    decl = ImportDeclaration(
        source="zecalibrator-synthetic-p3b", identity="SYNTH-BASE-1", version="1.0"
    )
    for fid in ("l0", "l1"):
        decoded = decode_fits(r.frame_paths[fid], declaration=decl)
        assert decoded.data.dtype == np.float32
        assert np.array_equal(decoded.data, r.frame_arrays[fid].astype(np.float32))
        assert decoded.mask.sum() == 0
        assert decoded.bscale == 1.0
        assert decoded.bzero == 32768.0
        assert decoded.metadata.cfa_phase == "GRBG"
        assert decoded.metadata.geometry.binning == (1, 1)
        assert decoded.metadata.geometry.roi_origin == (0, 0)


def test_generated_dark_and_bias_decode(tmp_path):
    from zecalibrator.core.metadata import ImportDeclaration
    from zecalibrator.io.raw_decoder import decode_fits
    from research.p3b.model import EpochSpec, FrameSpec, GroupSpec, ScenarioSpec

    sensor = make_sensor(shape=(64, 64))
    darks = tuple(
        FrameSpec(frame_id=f"d{i}", frame_type="dark", epoch_id="e0", group_id="gd", ordinal=i)
        for i in range(2)
    )
    bias = FrameSpec(frame_id="b0", frame_type="bias", epoch_id="e0", group_id="gb", ordinal=0)
    scenario = ScenarioSpec(
        name="calib", seed=5, sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("gd", darks), GroupSpec("gb", (bias,)))),),
    )
    r = generate_corpus(scenario, tmp_path / "c")
    decl = ImportDeclaration(
        source="zecalibrator-synthetic-p3b", identity="SYNTH-BASE-1", version="1.0"
    )
    for fid in ("d0", "d1", "b0"):
        decoded = decode_fits(r.frame_paths[fid], declaration=decl)
        assert np.array_equal(decoded.data, r.frame_arrays[fid].astype(np.float32))
        assert decoded.mask.sum() == 0
