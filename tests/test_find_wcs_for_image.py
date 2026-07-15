"""Tests for single-image WCS resolution and HiPS subtraction wiring."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from astropy import wcs as astropy_wcs
from astropy.io import fits

from helpers import load_module_from_path, pkg_src
from ost_photometry.wcs import (
    _wcs_maps_distinct_sky_positions,
    find_wcs_for_image,
)


def _write_science_fits(path: Path) -> astropy_wcs.WCS:
    data = np.ones((64, 64), dtype=np.float32)
    header = fits.Header()
    header["BITPIX"] = -32
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 11.776
    header["CRVAL2"] = 85.29
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CDELT1"] = 0.00018
    header["CDELT2"] = 0.00018
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["BUNIT"] = "adu"
    fits.writeto(path, data, header, overwrite=True)
    return astropy_wcs.WCS(header)


def test_find_wcs_for_image_reuses_header_wcs(tmp_path: Path):
    fits_path = tmp_path / "science.fit"
    expected = _write_science_fits(fits_path)

    image = MagicMock()
    image.path = fits_path
    image.out_path = tmp_path

    resolved = find_wcs_for_image(image, method="astrometry", indent=2)

    assert _wcs_maps_distinct_sky_positions(resolved, (64, 64))
    assert resolved.wcs.crval[0] == expected.wcs.crval[0]
    assert resolved.wcs.crval[1] == expected.wcs.crval[1]


def _hips_module():
    src = pkg_src()
    load_module_from_path("ost_photometry.style", src / "ost_photometry" / "style.py")
    load_module_from_path(
        "ost_photometry.terminal_output",
        src / "ost_photometry" / "terminal_output.py",
    )
    load_module_from_path("ost_photometry.checks", src / "ost_photometry" / "checks.py")
    load_module_from_path("ost_photometry.wcs", src / "ost_photometry" / "wcs.py")

    utilities_mod = types.ModuleType("ost_photometry.utilities")

    class _Image:
        def __init__(self, image_id, filter_, path, output_dir):
            self.image_id = image_id
            self.filter_ = filter_
            self.path = Path(path)
            self.out_path = Path(output_dir)

    utilities_mod.Image = _Image
    utilities_mod.get_basename = lambda path: Path(path).name
    sys.modules["ost_photometry.utilities"] = utilities_mod

    analyze_pkg = types.ModuleType("ost_photometry.analyze")
    sys.modules.setdefault("ost_photometry.analyze", analyze_pkg)

    plots_mod = types.ModuleType("ost_photometry.analyze.plots")
    plots_mod.compare_images = MagicMock()
    sys.modules["ost_photometry.analyze.plots"] = plots_mod

    subtraction_mod = types.ModuleType("ost_photometry.analyze.subtraction")
    subtraction_mod.run_hotpants = MagicMock(
        return_value=Path("/tmp/hotpants_diff.fits")
    )
    sys.modules["ost_photometry.analyze.subtraction"] = subtraction_mod

    models_mod = types.ModuleType("ost_photometry.analyze.models")
    models_mod.ImageSeries = MagicMock
    sys.modules["ost_photometry.analyze.models"] = models_mod

    hips_query_mod = types.ModuleType("astroquery.hips2fits")
    hips_query_mod.hips2fitsClass = MagicMock
    sys.modules["astroquery.hips2fits"] = hips_query_mod

    ccdproc_mod = types.ModuleType("ccdproc")
    ccdproc_mod.trim_image = lambda ccd: ccd
    sys.modules["ccdproc"] = ccdproc_mod

    return load_module_from_path(
        "ost_photometry.analyze.post_processing.hips_reference_subtract",
        src
        / "ost_photometry"
        / "analyze"
        / "post_processing"
        / "hips_reference_subtract.py",
    )


def test_run_hips_reference_subtraction_reuses_image_wcs(
    monkeypatch,
    tmp_path: Path,
):
    hips_mod = _hips_module()
    run_hips = hips_mod.run_hips_reference_subtraction

    science_path = tmp_path / "science.fit"
    _write_science_fits(science_path)
    workdir = tmp_path / "work"
    workdir.mkdir()

    reused_wcs = astropy_wcs.WCS(naxis=2)
    reused_wcs.wcs.crpix = [32.5, 32.5]
    reused_wcs.wcs.crval = [180.0, 45.0]
    reused_wcs.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    reused_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    reuse_image = MagicMock()
    reuse_image.wcs = reused_wcs

    calls: list[str] = []

    def _fake_find_wcs_for_image(image, **kwargs):
        calls.append("find_wcs_for_image")
        return astropy_wcs.WCS(naxis=2)

    class _FakeHips:
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            primary = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32))
            return fits.HDUList([primary])

    monkeypatch.setattr(hips_mod, "find_wcs_for_image", _fake_find_wcs_for_image)
    monkeypatch.setattr(hips_mod, "hips2fitsClass", _FakeHips)
    monkeypatch.setattr(
        hips_mod.subtraction,
        "run_hotpants",
        lambda *args, **kwargs: workdir / "hotpants_diff.fits",
    )

    result = run_hips(
        "B",
        str(science_path),
        workdir,
        reuse_wcs_image_series=reuse_image,
        plot_comp=False,
    )

    assert calls == []
    assert result.difference_fits == workdir / "hotpants_diff.fits"


def test_run_hips_reference_subtraction_solves_wcs_for_single_image(
    monkeypatch,
    tmp_path: Path,
):
    hips_mod = _hips_module()
    run_hips = hips_mod.run_hips_reference_subtraction

    science_path = tmp_path / "science.fit"
    _write_science_fits(science_path)
    workdir = tmp_path / "work"
    workdir.mkdir()

    calls: list[str] = []

    def _fake_find_wcs_for_image(image, **kwargs):
        calls.append(image.path.name)
        solved = astropy_wcs.WCS(naxis=2)
        solved.wcs.crpix = [32.5, 32.5]
        solved.wcs.crval = [180.0, 45.0]
        solved.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
        solved.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return solved

    class _FakeHips:
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            primary = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32))
            return fits.HDUList([primary])

    monkeypatch.setattr(hips_mod, "find_wcs_for_image", _fake_find_wcs_for_image)
    monkeypatch.setattr(hips_mod, "hips2fitsClass", _FakeHips)
    monkeypatch.setattr(
        hips_mod.subtraction,
        "run_hotpants",
        lambda *args, **kwargs: workdir / "hotpants_diff.fits",
    )

    run_hips(
        "B",
        str(science_path),
        workdir,
        reuse_wcs_image_series=None,
        plot_comp=False,
    )

    assert calls == ["science.fit"]
