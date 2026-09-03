"""Tests for utils photometry and limiting-magnitude helpers."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from astropy.table import Table

from helpers import (
    ensure_stub_package,
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
    stub_plot_process,
)


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _photometry():
    root = pkg_src() / "ost_photometry"
    ensure_stub_package("ost_photometry", path=root)
    term = ensure_stub_package("ost_photometry.terminal_output")
    if getattr(term, "__file__", None) is None:
        term.print_to_terminal = lambda *a, **k: None
        term.TerminalLog = object
    return load_module_from_path(
        "ost_photometry.analyze.utils.photometry",
        pkg_src() / "ost_photometry" / "analyze" / "utils" / "photometry.py",
    )


def _limmag_helpers():
    """Load limiting_magnitude with stubs so pure helpers need no photutils."""
    root = pkg_src() / "ost_photometry"
    analyze = root / "analyze"
    ensure_stub_package("ost_photometry", path=root)
    ensure_stub_package("ost_photometry.terminal_output")
    stub_plot_process()
    ensure_stub_package("ost_photometry.analyze", path=analyze)
    ensure_stub_package("ost_photometry.analyze.plots")
    ensure_stub_package(
        "ost_photometry.analyze.post_processing",
        path=analyze / "post_processing",
    )
    ensure_stub_package("ost_photometry.analyze.post_processing.adapters")
    ensure_stub_package("ost_photometry.analyze.post_processing.imaging")
    ensure_stub_package("photutils")
    ensure_stub_package("photutils.utils")

    term = sys.modules["ost_photometry.terminal_output"]
    if getattr(term, "__file__", None) is None:
        term.print_to_terminal = lambda *a, **k: None

    plots = sys.modules["ost_photometry.analyze.plots"]
    if getattr(plots, "__file__", None) is None:
        plots.starmap = object
        plots.plot_limiting_mag_sky_apertures = object

    adapters = sys.modules["ost_photometry.analyze.post_processing.adapters"]
    if getattr(adapters, "__file__", None) is None:
        adapters.ensure_epoch_native_photometry_table = lambda t: t

    imaging = sys.modules["ost_photometry.analyze.post_processing.imaging"]
    if getattr(imaging, "__file__", None) is None:

        class ImagingPlotContext:
            pass

        imaging.ImagingPlotContext = ImagingPlotContext

    photutils_utils = sys.modules["photutils.utils"]
    if getattr(photutils_utils, "__file__", None) is None:
        photutils_utils.ImageDepth = object

    return load_module_from_path(
        "ost_photometry.analyze.utils.limiting_magnitude",
        analyze / "utils" / "limiting_magnitude.py",
    )


def _flag_comparison():
    return load_module_from_path(
        "ost_photometry.analyze.calibration_sources.flags",
        pkg_src() / "ost_photometry" / "analyze" / "calibration_sources" / "flags.py",
    )


def test_flux_to_magnitudes():
    mod = _photometry()
    m, e = mod.flux_to_magnitudes(np.array([100.0, 25.0]), np.array([-1.0, 2.0]))
    np.testing.assert_allclose(m, -2.5 * np.log10([100.0, 25.0]), rtol=1e-12)
    expected = (2.5 / np.log(10)) * np.array([0.01, 0.08])
    np.testing.assert_allclose(e, expected, rtol=1e-12)
    assert np.all(np.asarray(e) > 0)


def test_attach_finder_quality_by_nearest_xy():
    pytest.importorskip("scipy")
    mod = _photometry()
    photometry = Table(
        {
            "x_fit": np.array([10.0, 20.0, 50.0]),
            "y_fit": np.array([10.0, 20.0, 50.0]),
            "qfit": np.array([0.11, 0.22, 0.33]),
        }
    )
    finder = Table(
        {
            "x_centroid": np.array([10.2, 20.1]),
            "y_centroid": np.array([10.1, 19.9]),
            "sharpness": np.array([0.4, 0.8]),
            "roundness": np.array([0.1, -0.2]),
        }
    )
    out = mod.attach_finder_quality(photometry, finder, max_sep_pix=3.0)
    np.testing.assert_allclose(out["qfit"], [0.11, 0.22, 0.33])
    np.testing.assert_allclose(out["sharpness"][:2], [0.4, 0.8])
    np.testing.assert_allclose(out["roundness"][:2], [0.1, -0.2])
    assert np.isnan(out["sharpness"][2])


def test_attach_finder_quality_skips_when_column_already_present():
    mod = _photometry()
    photometry = Table(
        {
            "x_fit": np.array([1.0, 2.0]),
            "y_fit": np.array([1.0, 2.0]),
            "sharpness": np.array([9.0, 8.0]),
        }
    )
    finder = Table(
        {
            "x_centroid": np.array([1.0, 2.0]),
            "y_centroid": np.array([1.0, 2.0]),
            "sharpness": np.array([0.1, 0.2]),
        }
    )
    out = mod.attach_finder_quality(photometry, finder)
    np.testing.assert_allclose(out["sharpness"], [9.0, 8.0])


def test_flag_comparison_stars_from_std_and_separation():
    mod = _flag_comparison()
    tbl = Table(
        {
            "mag_std_V": np.array([12.0, np.nan, 13.0]),
            "match_sep_arcsec": np.array([np.nan, 0.4, np.nan]),
        }
    )
    out = mod.flag_comparison_stars(tbl)
    np.testing.assert_array_equal(out["is_comparison"], [True, True, True])

    empty = Table({"id": np.array([1, 2])})
    flagged = mod.flag_comparison_stars(empty)
    np.testing.assert_array_equal(flagged["is_comparison"], [False, False])


def test_mark_used_calibrators_prefers_exact_mask_and_clip():
    mod = _flag_comparison()

    class _TC:
        color_term = 0.0
        zero_point = 1.0
        rms_residual = 0.05
        color_index_filters = ("B", "V")

    tbl = Table(
        {
            "mag_V": np.array([11.0, 12.0, 13.0]),
            "mag_std_V": np.array([12.0, 13.0, 14.2]),
        }
    )
    exact = np.array([True, False, False])
    out = mod.mark_used_calibrators(
        tbl.copy(),
        ["V"],
        transformations={"V": _TC()},
        sigma_clip=2.5,
        exact_masks={"V": exact},
    )
    np.testing.assert_array_equal(out["is_calibrator_V"], exact)
    assert np.all(out["is_comparison"])

    reconstructed = mod.mark_used_calibrators(
        tbl.copy(),
        ["V"],
        transformations={"V": _TC()},
        sigma_clip=2.5,
    )
    # residuals vs ZP=1 are 0, 0, 0.2; 0.2 > 2.5*0.05 so the last star is clipped
    np.testing.assert_array_equal(reconstructed["is_calibrator_V"], [True, True, False])


def test_rm_edge_objects_drops_border_sources():
    mod = _photometry()
    tbl = Table(
        {
            "x_fit": np.array([5.0, 50.0, 95.0]),
            "y_fit": np.array([50.0, 50.0, 50.0]),
        }
    )
    data = np.zeros((100, 100))
    out = mod.rm_edge_objects(tbl, data, border=10)
    assert len(out) == 1
    assert out["x_fit"][0] == 50.0


def test_resolve_limiting_mag_column_prefers_calibrated():
    mod = _limmag_helpers()
    tbl = Table({"mag_cal_V": [12.0], "mag_inst_V": [11.0]})
    assert mod._resolve_limiting_mag_column(tbl, "V") == "mag_cal_V"


def test_subset_photometry_by_epoch():
    mod = _limmag_helpers()
    tbl = Table({"epoch_id": ["a", "b", "a"], "mag_cal_V": [1.0, 2.0, 3.0]})
    sub = mod._subset_photometry_by_epoch(tbl, "a")
    assert len(sub) == 2
    np.testing.assert_array_equal(sub["mag_cal_V"], [1.0, 3.0])


def test_image_and_mask_for_depth_masks_nonfinite():
    mod = _limmag_helpers()
    data = np.ones((4, 4), dtype=float)
    data[0, 0] = np.nan
    data[1, 2] = np.inf
    mask = np.zeros((4, 4), dtype=bool)
    mask[3, 3] = True
    cleaned, out_mask = mod._image_and_mask_for_depth(data, mask)
    assert cleaned[0, 0] == 0.0
    assert cleaned[1, 2] == 0.0
    assert out_mask[0, 0] and out_mask[1, 2] and out_mask[3, 3]
    assert not out_mask[2, 2]
    np.testing.assert_allclose(cleaned[2, 2], 1.0)


def test_image_and_mask_for_depth_passthrough_when_finite():
    mod = _limmag_helpers()
    data = np.arange(9, dtype=float).reshape(3, 3)
    mask = np.zeros((3, 3), dtype=bool)
    cleaned, out_mask = mod._image_and_mask_for_depth(data, mask)
    np.testing.assert_array_equal(cleaned, data)
    np.testing.assert_array_equal(out_mask, mask)
