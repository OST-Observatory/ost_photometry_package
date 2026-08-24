"""Tests for Image / AnalysisImage split."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from helpers import ensure_stub_package, isolated_sys_modules, load_module_from_path, pkg_src


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _ensure_regions_stub() -> None:
    if "regions" in sys.modules:
        return
    regions = types.ModuleType("regions")

    class PixCoord:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class RectanglePixelRegion:
        def __init__(self, center, width, height):
            self.center = center
            self.width = width
            self.height = height

    regions.PixCoord = PixCoord
    regions.RectanglePixelRegion = RectanglePixelRegion
    sys.modules["regions"] = regions


def _load_image_modules():
    _ensure_regions_stub()
    src = pkg_src()
    load_module_from_path("ost_photometry.style", src / "ost_photometry" / "style.py")
    load_module_from_path(
        "ost_photometry.terminal_output",
        src / "ost_photometry" / "terminal_output.py",
    )
    load_module_from_path(
        "ost_photometry.calibration_parameters",
        src / "ost_photometry" / "calibration_parameters.py",
    )
    image_mod = load_module_from_path(
        "ost_photometry.image",
        src / "ost_photometry" / "image.py",
    )
    analyze_dir = src / "ost_photometry" / "analyze"
    ensure_stub_package("ost_photometry.analyze", path=analyze_dir)
    analysis_mod = load_module_from_path(
        "ost_photometry.analyze.image",
        src / "ost_photometry" / "analyze" / "image.py",
    )
    return image_mod, analysis_mod


def _write_minimal_fits(path: Path) -> None:
    data = np.ones((32, 32), dtype=np.float32)
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 32
    header["NAXIS2"] = 32
    header["FOCALLEN"] = 3454.0
    header["OBJCTRA"] = "00 00 00"
    header["OBJCTDEC"] = "+00 00 00"
    header["INSTRUME"] = "TESTCAM"
    header["XPIXSZ"] = 3.76
    header["JD"] = 2460000.5
    header["AIRMASS"] = 1.2
    fits.writeto(path, data, header, overwrite=True)


def test_analysis_image_is_subclass_of_image():
    image_mod, analysis_mod = _load_image_modules()
    assert issubclass(analysis_mod.AnalysisImage, image_mod.Image)


def test_base_image_has_metadata_not_analysis_fields(tmp_path: Path):
    image_mod, _ = _load_image_modules()
    fits_path = tmp_path / "img.fit"
    _write_minimal_fits(fits_path)

    image = image_mod.Image(0, "V", fits_path, tmp_path)

    assert image.jd == 2460000.5
    assert image.air_mass == 1.2
    assert image.field_of_view_x is not None
    assert image.pixel_scale is not None
    assert not hasattr(image, "photometry")
    assert not hasattr(image, "epsf")
    assert not hasattr(image, "zp")
    assert not hasattr(image, "positions")
    assert not hasattr(image, "residual_image")


def test_analysis_image_adds_photometry_fields(tmp_path: Path):
    _, analysis_mod = _load_image_modules()
    fits_path = tmp_path / "img.fit"
    _write_minimal_fits(fits_path)

    image = analysis_mod.AnalysisImage(0, "V", fits_path, tmp_path)

    assert image.photometry is None
    assert image.epsf is None
    assert image.positions is None
    assert image.zp is None
    assert image.residual_image is None
    assert image.jd == 2460000.5


def test_utilities_reexports_base_image():
    pytest.importorskip("yaml")
    _ensure_regions_stub()
    src = pkg_src()
    image_mod, _ = _load_image_modules()

    # Minimal stubs so utilities.py can import without full wcs stack.
    wcs_mod = types.ModuleType("ost_photometry.wcs")
    for name in (
        "check_wcs_exists",
        "find_wcs_astap",
        "find_wcs_astrometry",
        "find_wcs_twirl",
        "persist_wcs_to_fits",
        "sync_image_coordinates_from_wcs",
    ):
        setattr(wcs_mod, name, lambda *a, **k: None)
    sys.modules["ost_photometry.wcs"] = wcs_mod
    load_module_from_path("ost_photometry.checks", src / "ost_photometry" / "checks.py")

    utilities_mod = load_module_from_path(
        "ost_photometry.utilities",
        src / "ost_photometry" / "utilities.py",
    )
    assert utilities_mod.Image is image_mod.Image
