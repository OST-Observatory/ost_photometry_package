"""Aperture-to-table flux ratio used to shift the ImageDepth limit."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

pytest.importorskip("photutils")

from ost_photometry.analyze.utils.limiting_magnitude import (  # noqa: E402
    aperture_to_table_flux_ratio,
)


def _gaussian_field(rng, sigma=2.0, n=40, shape=(400, 400)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    img = rng.normal(100.0, 3.0, shape)
    x = rng.uniform(30, shape[1] - 30, n)
    y = rng.uniform(30, shape[0] - 30, n)
    total = rng.uniform(2e4, 1e5, n)
    for xi, yi, fi in zip(x, y, total, strict=True):
        img += fi / (2 * np.pi * sigma**2) * np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sigma**2))
    return img, Table({"x": x, "y": y, "mag_V": -2.5 * np.log10(total), "err_V": np.full(n, 0.005)})


def test_ratio_is_enclosed_fraction_of_total_flux():
    rng = np.random.default_rng(0)
    sigma, radius = 2.0, 3.0
    img, tbl = _gaussian_field(rng, sigma=sigma)
    rho, rho_err, n = aperture_to_table_flux_ratio(img, tbl, "mag_V", radius, err_col="err_V")
    expected = 1.0 - np.exp(-(radius**2) / (2 * sigma**2))  # Gaussian enclosed fraction
    assert n >= 5
    assert rho == pytest.approx(expected, abs=max(0.01, 3 * rho_err))


def test_ratio_nan_without_enough_stars():
    rng = np.random.default_rng(1)
    img, tbl = _gaussian_field(rng, n=3)
    rho, _err, n = aperture_to_table_flux_ratio(img, tbl, "mag_V", 3.0)
    assert np.isnan(rho) and n == 0
