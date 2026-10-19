"""PSF photometry of stars next to brighter neighbours (sky, grouping, weights)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table

FWHM = 4.0
BETA = 3.0
EXPTIME = 60.0
SKY = 20.0  # e-/s per pixel


def _moffat(xx, yy, x0, y0, flux):
    alpha = FWHM / (2 * np.sqrt(2 ** (1 / BETA) - 1))
    return flux * (BETA - 1) / (np.pi * alpha**2) * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / alpha**2) ** (-BETA)


def _scene(rng):
    shape = (400, 400)
    stars = []
    # targets with a 10x brighter neighbour at 2.2-4.5 FWHM, on a grid
    for gy in range(40, 380, 60):
        for gx in range(40, 200, 60):
            f = 10 ** rng.uniform(2.5, 3.2)
            d = rng.uniform(2.2, 4.5) * FWHM
            a = rng.uniform(0, 2 * np.pi)
            x, y = gx + rng.uniform(-1, 1), gy + rng.uniform(-1, 1)
            stars.append((x, y, f, "target"))
            stars.append((x + d * np.cos(a), y + d * np.sin(a), 10 * f, "neighbour"))
    # isolated stars over 4 decades
    for gy in range(40, 380, 50):
        for gx in range(240, 380, 50):
            stars.append((gx + rng.uniform(-1, 1), gy + rng.uniform(-1, 1), 10 ** rng.uniform(2.0, 5.0), "isolated"))
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    model = np.zeros(shape)
    for x, y, f, _ in stars:
        model += _moffat(xx, yy, x, y, f)
    counts = (model + SKY) * EXPTIME
    data = rng.poisson(counts) / EXPTIME - SKY  # background-subtracted rate image
    sigma = np.sqrt(np.maximum(data + SKY, 0) * EXPTIME + 25.0) / EXPTIME  # from the noisy data
    tbl = Table(rows=[(x, y, f, k) for x, y, f, k in stars], names=["x", "y", "flux", "kind"])
    return data, sigma, tbl


def _run(data, sigma, tbl, tmp_path, rng):
    pytest.importorskip("photutils")
    from photutils.psf import MoffatPSF

    from ost_photometry.analyze.extraction import extraction_epsf

    ccd = CCDData(data, unit="electron/s", uncertainty=StdDevUncertainty(sigma), meta={"EXPTIME": EXPTIME})
    positions = Table({"x_centroid": tbl["x"] + rng.normal(0, 0.1, len(tbl)),
                       "y_centroid": tbl["y"] + rng.normal(0, 0.1, len(tbl)),
                       "flux": tbl["flux"]})
    image = SimpleNamespace(out_path=tmp_path, read_image=lambda: ccd, positions=positions,
                            epsf=MoffatPSF(alpha=FWHM / (2 * np.sqrt(2 ** (1 / BETA) - 1)), beta=BETA), fwhm=FWHM, filter_="V", image_id=0)
    extraction_epsf(image, background_rms=np.sqrt(SKY / EXPTIME), size_extraction_region=17)
    res = image.photometry
    xr, yr = np.asarray(res["x_fit"], float), np.asarray(res["y_fit"], float)
    out = np.full(len(tbl), np.nan)
    for i, (x, y) in enumerate(zip(tbl["x"], tbl["y"], strict=True)):
        d = np.hypot(xr - x, yr - y)
        j = int(np.argmin(d))
        if d[j] < 1.0:
            out[i] = float(res["flux_fit"][j])
    return out, res


def test_psf_targets_next_to_bright_neighbours(tmp_path):
    rng = np.random.default_rng(11)
    data, sigma, tbl = _scene(rng)
    flux, _res = _run(data, sigma, tbl, tmp_path, rng)
    dm = -2.5 * np.log10(flux / np.asarray(tbl["flux"]))
    kind = np.asarray(tbl["kind"])
    target = dm[kind == "target"]
    assert np.isfinite(target).all()
    assert abs(np.median(target)) < 0.005
    iso = kind == "isolated"
    bright = iso & (np.asarray(tbl["flux"]) > 1e4)
    # constant (sky-only) weights: no flux-dependent bias of bright stars
    assert abs(np.median(dm[bright])) < 0.002


def test_psf_errors_match_the_scatter(tmp_path):
    """flux_err (background fit + star photons) describes the real scatter."""
    rng = np.random.default_rng(5)
    pulls = []
    for _ in range(3):
        data, sigma, tbl = _scene(rng)
        flux, res = _run(data, sigma, tbl, tmp_path, rng)
        xr, yr = np.asarray(res["x_fit"], float), np.asarray(res["y_fit"], float)
        iso = np.asarray(tbl["kind"]) == "isolated"
        for i in np.flatnonzero(iso):
            j = int(np.argmin(np.hypot(xr - tbl["x"][i], yr - tbl["y"][i])))
            pulls.append((flux[i] - tbl["flux"][i]) / float(res["flux_err"][j]))
    pulls = np.asarray(pulls)
    # ~1.15-1.2 (also before the change): the fit covariance misses a little
    assert 0.8 < 1.4826 * np.median(np.abs(pulls - np.median(pulls))) < 1.35
