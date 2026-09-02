"""Alard–Lupton kernel subtractor and backend dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData
from scipy.ndimage import gaussian_filter

from ost_photometry.analyze.subtraction import (
    resolve_subtract_backend,
    subtract_science_template,
)
from ost_photometry.analyze.subtraction_alard_lupton import (
    alard_lupton_difference,
    kernel_basis,
    run_alard_lupton,
)

_STAR_XY = np.array(
    [
        [40.0, 40.0],
        [90.0, 50.0],
        [55.0, 100.0],
        [125.0, 110.0],
        [70.0, 145.0],
        [155.0, 75.0],
    ]
)


def _star_field(shape: tuple[int, int], xy: np.ndarray, fwhm: float) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.indices(shape)
    sigma = fwhm / 2.355
    img = np.zeros(shape, dtype=np.float64)
    amps = np.linspace(800.0, 1800.0, len(xy))
    for (x, y), amp in zip(xy, amps, strict=True):
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    return img


def test_kernel_basis_shape_and_normalization():
    bases = kernel_basis(15)
    # degrees (2, 1, 0) → 6 + 3 + 1 basis images
    assert bases.shape == (10, 15, 15)
    assert np.all(np.isfinite(bases))


def test_alard_lupton_cancels_convolved_template():
    tmpl = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    sci = 1.25 * gaussian_filter(tmpl, sigma=1.1) + 12.0
    diff, method = alard_lupton_difference(sci, tmpl, star_xy=_STAR_XY, fwhm=4.0)
    assert method == "alard_lupton"
    naive = sci - tmpl
    assert float(np.std(diff)) < 0.2 * float(np.std(naive))
    assert float(np.std(diff)) < 15.0


def test_alard_lupton_falls_back_when_stamps_are_missing():
    xy = np.array([[40.0, 40.0], [55.0, 50.0], [65.0, 60.0]])
    tmpl = _star_field((80, 80), xy, fwhm=3.0)
    sci = 1.1 * tmpl + 5.0
    diff, method = alard_lupton_difference(
        sci, tmpl, star_xy=np.array([[2.0, 2.0]]), fwhm=3.0
    )
    assert method == "flux_scale"
    assert diff.shape == sci.shape
    assert float(np.std(diff)) < 1.0


def test_run_alard_lupton_writes_fits(tmp_path: Path):
    tmpl = _star_field((120, 120), _STAR_XY[:4], fwhm=3.0)
    sci = 1.2 * gaussian_filter(tmpl, sigma=0.8) + 4.0
    ccd = CCDData(sci, unit="adu")
    hdu = fits.PrimaryHDU(tmpl)
    out = run_alard_lupton(
        ccd,
        hdu,
        workdir=tmp_path,
        output_filename="hotpants_diff.fits",
        star_xy=_STAR_XY[:4],
    )
    assert out == tmp_path / "hotpants_diff.fits"
    assert out.is_file()
    written = CCDData.read(out)
    assert written.data.shape == sci.shape


def test_resolve_subtract_backend_auto(monkeypatch):
    import ost_photometry.analyze.subtraction as sub

    monkeypatch.setattr(sub.shutil, "which", lambda name: None)
    assert resolve_subtract_backend("auto") == "alard_lupton"
    assert resolve_subtract_backend("alard_lupton") == "alard_lupton"
    with pytest.raises(RuntimeError, match="hotpants"):
        resolve_subtract_backend("hotpants")

    monkeypatch.setattr(sub.shutil, "which", lambda name: "/usr/bin/hotpants")
    assert resolve_subtract_backend("auto") == "hotpants"
    assert resolve_subtract_backend("alard_lupton") == "alard_lupton"
    assert resolve_subtract_backend("hotpants") == "hotpants"


def test_subtract_science_template_dispatches_to_alard_lupton(tmp_path: Path):
    tmpl = _star_field((120, 120), _STAR_XY[:4], fwhm=3.0)
    sci = 1.2 * gaussian_filter(tmpl, sigma=0.8) + 4.0
    path = subtract_science_template(
        CCDData(sci, unit="adu"),
        fits.PrimaryHDU(tmpl),
        workdir=tmp_path,
        backend="alard_lupton",
        star_xy=_STAR_XY[:4],
    )
    assert path.is_file()
    assert path.name == "hotpants_diff.fits"
