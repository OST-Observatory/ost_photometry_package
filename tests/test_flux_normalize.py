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


def test_quasi_flux_calibration_flux_arrays_scales_by_epoch_median():
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
    _, epoch_med, _ = sigma_clipped_stats(flux, axis=1, sigma=1.5, mask_value=0.0)
    expected = flux / epoch_med[:, np.newaxis]
    np.testing.assert_allclose(med, expected, rtol=5e-3, atol=5e-3)


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


def test_deprecated_legacy_entry_points_warn():
    pytest.importorskip("photutils")
    pytest.importorskip("regions")
    from ost_photometry.analyze import calibration

    with pytest.warns(DeprecationWarning, match="apply_calibration is deprecated"):
        # Call will fail on bad args after the warning — that is fine.
        try:
            calibration.apply_calibration()
        except TypeError:
            pass
