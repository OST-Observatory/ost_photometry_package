"""Tests for flux normalization helpers extracted from calibration/_legacy."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import uncertainty as unc
from astropy.stats import sigma_clipped_stats

from helpers import load_module_from_path, pkg_src


def _load_flux_normalize():
    return load_module_from_path(
        "ost_photometry.analyze.calibration.flux_normalize",
        pkg_src() / "ost_photometry" / "analyze" / "calibration" / "flux_normalize.py",
    )


def test_quasi_flux_calibration_preserves_constant_star_ratios():
    """Complete detections: object flux ratios stay, epoch common mode is removed."""
    mod = _load_flux_normalize()
    flux = np.array(
        [
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
        ],
        dtype=float,
    )
    err = np.full_like(flux, 0.1)
    result = mod.quasi_flux_calibration_flux_arrays(flux, err, distribution_samples=5000)
    med = result.pdf_median()
    np.testing.assert_allclose(med[0], med[1], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(med[0, 1] / med[0, 0], 2.0, rtol=5e-3)
    np.testing.assert_allclose(med[0, 2] / med[0, 0], 3.0, rtol=5e-3)


def test_quasi_flux_calibration_ignores_faint_dropout_in_epoch_median():
    """Faint stars appearing only at high transparency must not tilt a bright OOI."""
    mod = _load_flux_normalize()
    rng = np.random.default_rng(0)
    n_ep = 40
    x = np.linspace(-1.0, 1.0, n_ep)
    airmass = 1.15 + 1.0 * x**2
    transparency = 10 ** (-0.4 * 0.35 * (airmass - 1.0))
    n_bright, n_faint = 8, 120
    scale = np.concatenate(
        [
            [5000.0],
            rng.uniform(800.0, 3000.0, n_bright),
            rng.lognormal(mean=np.log(25.0), sigma=0.8, size=n_faint),
        ]
    )
    true = transparency[:, None] * scale[None, :]
    flux = np.where(true > 20.0, true, np.nan)
    err = np.where(np.isfinite(flux), np.sqrt(np.maximum(flux, 1.0)), np.nan)

    work = np.where(np.isfinite(flux), flux, 0.0)
    _, raw_epoch_med, _ = sigma_clipped_stats(
        work, axis=1, sigma=1.5, mask_value=0.0
    )
    old = flux[:, 0] / raw_epoch_med
    old_norm = old / np.nanmedian(old)

    result = mod.quasi_flux_calibration_flux_arrays(
        flux, err, distribution_samples=800, min_ensemble_fraction=0.5
    )
    quasi = result.pdf_median()[:, 0]
    new_norm = quasi / np.nanmedian(quasi)

    assert np.nanmax(old_norm) / np.nanmin(old_norm) > 1.2
    np.testing.assert_allclose(new_norm, 1.0, rtol=0.03, atol=0.03)


def test_flux_normalization_flux_distribution_scales_by_object_median():
    mod = _load_flux_normalize()
    flux = np.array(
        [
            [2.0, 4.0],
            [4.0, 8.0],
            [6.0, 12.0],
        ],
        dtype=float,
    )
    dist = unc.normal(flux, std=np.full_like(flux, 0.01), n_samples=5000)
    result = mod.flux_normalization_flux_distribution(dist)
    med = result.pdf_median()
    _, obj_med, _ = sigma_clipped_stats(flux, axis=0, sigma=1.5, mask_value=0.0)
    expected = flux / obj_med
    np.testing.assert_allclose(med, expected, rtol=5e-3, atol=5e-3)


def test_calibration_package_exports_array_helpers():
    pytest.importorskip("photutils")
    pytest.importorskip("regions")
    from ost_photometry.analyze import calibration

    assert callable(calibration.quasi_flux_calibration_flux_arrays)
    assert callable(calibration.flux_normalization_flux_distribution)
    assert callable(calibration.quasi_flux_calibration_image_series)
    assert callable(calibration.flux_normalization_image_series)
    assert not hasattr(calibration, "apply_calibration")
