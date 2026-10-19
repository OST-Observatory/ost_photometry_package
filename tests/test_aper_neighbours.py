"""Neighbour-subtracted aperture photometry in a crowded synthetic field."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table

FWHM, BETA, EXPTIME, SKY = 6.0, 2.5, 60.0, 30.0


def _moffat(xx, yy, x0, y0, flux):
    alpha = FWHM / (2 * np.sqrt(2 ** (1 / BETA) - 1))
    return flux * (BETA - 1) / (np.pi * alpha**2) * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / alpha**2) ** (-BETA)


def _scene(seed):
    rng = np.random.default_rng(seed)
    shape = (500, 500)
    stars = []
    for gy in range(50, 460, 70):
        for gx in range(50, 460, 70):
            f = 10 ** rng.uniform(2.5, 3.0)
            d, a = rng.uniform(2.0, 4.0) * FWHM, rng.uniform(0, 2 * np.pi)
            stars.append((gx, gy, f, True))
            stars.append((gx + d * np.cos(a), gy + d * np.sin(a), 10 * f, False))
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    model = np.zeros(shape)
    for x, y, f, _ in stars:
        model += _moffat(xx, yy, x, y, f)
    data = rng.poisson((model + SKY) * EXPTIME) / EXPTIME - SKY
    sigma = np.sqrt((model + SKY) * EXPTIME) / EXPTIME
    return data, sigma, Table(rows=stars, names=["x", "y", "flux", "target"])


@pytest.mark.parametrize("subtract", [False, True])
def test_faint_stars_next_to_bright_ones(tmp_path, subtract):
    from ost_photometry.analyze.extraction import extraction_aperture

    biases = []
    for seed in (1, 2):
        data, sigma, tbl = _scene(seed)
        ccd = CCDData(data, unit="electron/s", uncertainty=StdDevUncertainty(sigma), meta={"EXPTIME": EXPTIME})
        image = SimpleNamespace(
            out_path=tmp_path, read_image=lambda c=ccd: c, image_id=0, filter_="V", fwhm=FWHM, pixel_scale=1.0,
            positions=Table({"x_centroid": tbl["x"], "y_centroid": tbl["y"], "flux": tbl["flux"]}),
        )
        extraction_aperture(image, 1.0 * FWHM, 2.5 * FWHM, 4.0 * FWHM, radii_unit="pixel",
                            neighbour_subtraction=subtract)
        res = image.photometry
        alpha = FWHM / (2 * np.sqrt(2 ** (1 / BETA) - 1))
        ee = 1 - (1 + (FWHM / alpha) ** 2) ** (1 - BETA)
        t = np.asarray(tbl["target"], bool)
        ratio = np.asarray(res["flux_fit"], float)[t] / (np.asarray(tbl["flux"], float)[t] * ee)
        biases.append(-2.5 * np.log10(np.median(ratio)))
    bias = float(np.mean(biases))
    if subtract:
        assert abs(bias) < 0.01
    else:
        # neighbour light in aperture and annulus (sign depends on the radii)
        assert abs(bias) > 0.02
