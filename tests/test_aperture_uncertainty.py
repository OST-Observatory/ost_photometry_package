"""APER flux errors must hypot photutils σ with the scaled sky, not DAOPHOT-rescale it."""

from __future__ import annotations

import numpy as np
import pytest

from ost_photometry.analyze.extraction import (
    compute_aperture_photometry_uncertainties,
    sky_subtraction_error,
)


def test_aperture_uncertainty_is_hypot_of_stddevs():
    out = compute_aperture_photometry_uncertainties(300.0, 60.0)
    assert out == pytest.approx(np.hypot(300.0, 60.0))


def test_aperture_uncertainty_does_not_treat_error_as_variance():
    """Regression: σ_sum was used as variance and sky was multiplied by n_pix."""
    sigma_sum = 300.0
    sigma_sky = 60.0
    n_pix = 78.5
    n_sky = 160.0
    out = compute_aperture_photometry_uncertainties(sigma_sum, sigma_sky)
    old = np.sqrt(
        sigma_sum + n_pix * sigma_sky**2 * (1.0 + n_pix / n_sky)
    )
    assert out == pytest.approx(np.hypot(sigma_sum, sigma_sky))
    assert out < 0.6 * old


def test_aperture_uncertainty_broadcasts():
    src = np.array([100.0, 200.0, 400.0])
    sky = np.array([10.0, 20.0, 0.0])
    out = compute_aperture_photometry_uncertainties(src, sky)
    np.testing.assert_allclose(out, np.hypot(src, sky))


def test_sky_subtraction_error_passthrough_when_already_scaled():
    err = np.array([12.0, 8.5])
    np.testing.assert_allclose(
        sky_subtraction_error(err, per_pixel=False),
        err,
    )


def test_sky_subtraction_error_converts_per_pixel_annulus_std():
    sigma_pix = 5.0
    n_pix = 78.5
    n_sky = 160.0
    out = sky_subtraction_error(
        sigma_pix,
        per_pixel=True,
        aperture_area=n_pix,
        annulus_area=n_sky,
    )
    assert out == pytest.approx(sigma_pix * n_pix / np.sqrt(n_sky))
