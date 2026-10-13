"""Relative-flux fallback for filters without catalog calibration."""

from __future__ import annotations

import numpy as np

from helpers import (
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
    stub_analyze_package,
)

_PP = pkg_src() / "ost_photometry" / "analyze" / "post_processing"


def _diff():
    stub_analyze_package("post_processing")
    root = pkg_src() / "ost_photometry"
    load_module_from_path("ost_photometry.terminal_output", root / "terminal_output.py")
    load_module_from_path("ost_photometry.output_layout", root / "output_layout.py")
    load_module_from_path("ost_photometry.analyze.post_processing.light_curve", _PP / "light_curve.py")
    return load_module_from_path(
        "ost_photometry.analyze.post_processing.differential_light_curve",
        _PP / "differential_light_curve.py",
    )


def _field(rng, n_ep=150, n_bright=8, n_faint=120):
    x = np.linspace(-1.0, 1.0, n_ep)
    airmass = 1.15 + 1.0 * x**2
    transparency = 10 ** (-0.4 * 0.35 * (airmass - 1.0))
    scale = np.concatenate(
        [
            [5000.0],
            rng.uniform(800.0, 3000.0, n_bright),
            rng.lognormal(mean=np.log(25.0), sigma=0.8, size=n_faint),
        ]
    )
    true = transparency[:, None] * scale[None, :]
    err = np.sqrt(true + 100.0)
    flux = true + rng.normal(0.0, 1.0, true.shape) * err
    flux = np.where(true > 20.0, flux, np.nan)  # faint stars drop out at high airmass
    return flux, np.where(np.isfinite(flux), err, np.nan)


def test_constant_stars_are_flat_with_honest_errors():
    with isolated_sys_modules():
        mod = _diff()
        rng = np.random.default_rng(0)
        flux, err = _field(rng)
        rel, rel_err, info = mod.relative_flux_light_curves(flux, err)
        # target (column 0) and bright stars: flat, pulls ~1
        for j in range(0, 9):
            ok = np.isfinite(rel[:, j])
            assert abs(np.median(rel[ok, j]) - 1.0) < 1e-4
            pull = (rel[ok, j] - 1.0) / rel_err[ok, j]
            assert 0.85 < np.std(pull) < 1.15
        assert info["ensemble"][0]


def test_common_mode_does_not_follow_dropouts_or_a_variable():
    with isolated_sys_modules():
        mod = _diff()
        rng = np.random.default_rng(1)
        flux, err = _field(rng)
        n_ep = flux.shape[0]
        flux[:, 3] *= 1.0 + 0.3 * np.sin(np.linspace(0, 6, n_ep))  # variable ensemble star
        rel, rel_err, _info = mod.relative_flux_light_curves(flux, err)
        pull = (rel[:, 0] - 1.0) / rel_err[:, 0]
        assert 0.85 < np.nanstd(pull) < 1.15
        assert abs(np.nanmean(pull)) < 0.3


def test_missing_errors_stay_nan_and_zero_flux_is_ignored():
    with isolated_sys_modules():
        mod = _diff()
        rng = np.random.default_rng(2)
        flux, err = _field(rng, n_ep=40)
        err[5, 0] = np.nan
        flux[6, 0] = 0.0
        rel, rel_err, _info = mod.relative_flux_light_curves(flux, err)
        assert np.isfinite(rel[5, 0]) and np.isnan(rel_err[5, 0])
        assert np.isnan(rel[6, 0])
