"""APER sky, flux errors, empty-sky noise factor, saturation, aperture correction.

The pull tests simulate frames with known star fluxes (Poisson + read
noise) and require ``(measured - true) / error`` to scatter by ~1.
"""

from __future__ import annotations

import numpy as np
import pytest
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

from ost_photometry.analyze.photometry_quality import (
    MEDIAN_ERROR_FACTOR,
    annulus_sky,
    aperture_correction,
    aperture_flux_errors,
    background_noise_factor,
    saturated_sources,
    source_disk_mask,
)

SIZE = 200
SKY = 400.0  # e- per pixel
READ_NOISE = 8.0
FWHM = 4.0
R_AP, R_IN, R_OUT = 6.0, 10.0, 15.0


def _gaussian(x0, y0, flux, fwhm=FWHM, size=SIZE):
    sig = fwhm / 2.3548
    yy, xx = np.mgrid[:size, :size]
    return flux / (2 * np.pi * sig**2) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig**2))


def _frame(stars, rng, extra=None):
    """Electrons, sky subtracted like the analysis frames; σ map as UNCERT."""
    model = np.full((SIZE, SIZE), SKY)
    for x0, y0, flux in stars:
        model += _gaussian(x0, y0, flux)
    data = rng.poisson(model).astype(float) + rng.normal(0.0, READ_NOISE, model.shape)
    if extra is not None:
        data += extra
    sigma = np.sqrt(model + READ_NOISE**2)
    return data - SKY, sigma


def _measure(data, sigma, pos, factor=1.0, detected=None):
    ap = CircularAperture(pos, R_AP)
    ann = CircularAnnulus(pos, R_IN, R_OUT)
    phot = aperture_photometry(data, ap, error=sigma)
    src = None
    if detected is not None:
        src = source_disk_mask(data.shape, detected[:, 0], detected[:, 1], 2 * FWHM)
    sky = annulus_sky(data, sigma, ann, source_mask=src)
    area = ap.area
    flux = np.asarray(phot["aperture_sum"]) - sky["sky"] * area
    err = aperture_flux_errors(
        np.asarray(phot["aperture_sum_err"]), sky["sigma_map"], area, sky["n_sky"], factor
    )
    return flux, err


def _enclosed(r, fwhm=FWHM):
    sig = fwhm / 2.3548
    return 1.0 - np.exp(-(r**2) / (2 * sig**2))


def test_aperture_flux_errors_source_and_background_parts():
    # Source-only part of the map error stays; background part is scaled.
    a, s, n = 100.0, 10.0, 400.0
    total = np.sqrt(5000.0 + a * s**2)
    out = aperture_flux_errors(total, s, a, n, noise_factor=2.0)
    expected = np.sqrt(5000.0 + 4.0 * (a * s**2 + MEDIAN_ERROR_FACTOR**2 * a**2 * s**2 / n))
    assert out == pytest.approx(expected)


def _pulls(with_neighbour: bool, n_trials: int = 500) -> np.ndarray:
    rng = np.random.default_rng(1)
    flux_true = np.array([2.0e4, 5.0e3, 1.5e3])
    xs = np.array([50.0, 100.0, 150.0])
    pulls = []
    for _ in range(n_trials):
        stars = [(x, 100.0, f) for x, f in zip(xs, flux_true, strict=True)]
        if with_neighbour:
            # 8x brighter neighbour at 3 FWHM, inside the annulus of the middle star
            stars.append((100.0 + 12.0, 100.0, 4.0e4))
        detected = np.array([[x, y] for x, y, _f in stars])
        data, sigma = _frame(stars, rng)
        flux, err = _measure(
            data, sigma, np.column_stack([xs, np.full(3, 100.0)]), detected=detected
        )
        pulls.append((flux - flux_true * _enclosed(R_AP)) / err)
    return np.array(pulls)


def test_aperture_pulls_are_unit_with_neighbours_in_annulus():
    """Errors match the scatter; a neighbour in the annulus does not bias."""
    isolated = _pulls(False)
    crowded = _pulls(True)
    assert np.std(isolated) == pytest.approx(1.0, abs=0.1)
    assert np.std(crowded) == pytest.approx(1.0, abs=0.1)
    # Masked detected sources + clipped median: the neighbour shifts the
    # middle star by < 0.2 σ (the former unclipped annulus mean: about -40 σ).
    shift = np.mean(crowded[:, 1]) - np.mean(isolated[:, 1])
    assert abs(shift) < 0.2


def test_background_noise_factor_white_noise_and_structure():
    rng = np.random.default_rng(2)
    stars = [(40.0 + 30 * i, 40.0 + 25 * (i % 4), 1.0e4) for i in range(5)]
    data, sigma = _frame(stars, rng)
    src = np.array([[x, y] for x, y, _ in stars])
    f_white, n = background_noise_factor(data, sigma, src, R_AP, R_IN, R_OUT, n_apertures=300)
    assert n >= 30
    assert f_white == pytest.approx(1.0, abs=0.15)
    # Unmodelled structure (correlated on a few pixels) raises the factor.
    from scipy.ndimage import gaussian_filter

    blobs = gaussian_filter(rng.normal(0.0, 60.0, (SIZE, SIZE)), 2.0)
    data2, sigma2 = _frame(stars, rng, extra=blobs)
    f_struct, _ = background_noise_factor(data2, sigma2, src, R_AP, R_IN, R_OUT, n_apertures=300)
    assert f_struct > 1.3


def test_saturated_sources_on_frame_with_sky():
    rng = np.random.default_rng(3)
    data, _sigma = _frame([(50.0, 50.0, 5.0e5), (150.0, 150.0, 2.0e4)], rng)
    raw = data + SKY  # saturation is judged with the sky
    sat_level = 0.8 * float(np.max(raw))
    flags = saturated_sources(raw, [50.0, 150.0], [50.0, 150.0], R_AP, sat_level, fraction=0.9)
    assert list(flags) == [True, False]


def test_aperture_correction_recovers_enclosed_fraction():
    rng = np.random.default_rng(4)
    xs = np.linspace(25.0, 175.0, 6)
    ys = np.tile([40.0, 120.0], 3)
    stars = [(x, y, 3.0e5) for x, y in zip(xs, ys, strict=True)]
    data, sigma = _frame(stars, rng)
    pos = np.column_stack([xs, ys])
    flux, err = _measure(data, sigma, pos)
    value, error, n = aperture_correction(
        data, sigma, xs, ys, flux, err, R_IN, R_IN, R_OUT, isolation_radius=R_OUT + R_AP
    )
    assert n == 6
    assert value == pytest.approx(_enclosed(R_IN) / _enclosed(R_AP), abs=3 * error + 1e-3)


def test_estimate_pixel_error_without_uncert_extension():
    import astropy.units as u

    from ost_photometry.analyze.extraction import estimate_pixel_error

    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 0.8, (200, 200))  # background-subtracted sky, e-/s
    data[100, 100] += 500.0  # star peak
    sigma = estimate_pixel_error(data, u.electron / u.s, exposure_time=20.0)
    assert np.median(sigma) == pytest.approx(0.8, rel=0.05)
    assert sigma[100, 100] == pytest.approx(np.sqrt(0.8**2 + 500.8 / 20.0), rel=0.05)
    sigma_e = estimate_pixel_error(data * 20.0, u.electron, exposure_time=20.0)
    assert sigma_e[100, 100] == pytest.approx(np.sqrt(16.0**2 + 500.8 * 20.0), rel=0.05)
    with pytest.raises(ValueError, match="unit"):
        estimate_pixel_error(data, u.adu, exposure_time=20.0)
