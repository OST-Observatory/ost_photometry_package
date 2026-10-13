"""Isochrone fit statistics: fiducials, profiled nuisance offsets, age interval."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import isolated_sys_modules, load_module_from_path, pkg_src


def _mod():
    return load_module_from_path(
        "ost_photometry.analyze.isochrone_fit",
        pkg_src() / "ost_photometry" / "analyze" / "isochrone_fit.py",
    )


def _isochrone(age, n=400):
    mag = np.linspace(0.0, 8.0, n)
    color = 0.4 + 0.1 * mag + 0.12 * age * np.exp(-((mag - 2.0) ** 2) / 1.5)
    return mag, color


A_PER_E, C_PER_E = 3.1, 1.0


def _stars(rng, age, dm, de, n=600, sigma=0.03):
    iso_m, iso_c = _isochrone(age, n=4000)
    k = rng.integers(0, iso_m.size, n)
    mag = iso_m[k] + dm + de * A_PER_E + rng.normal(0, sigma, n)
    color = iso_c[k] + de * C_PER_E + rng.normal(0, sigma, n)
    return mag, color, np.full(n, sigma)


def test_fiducial_sigma_takes_the_larger_term():
    with isolated_sys_modules():
        m = _mod()
        assert m.fiducial_sigma(0.01, 0.2, 16) == pytest.approx(1.2533 * 0.2 / 4)
        assert m.fiducial_sigma(0.1, 0.2, 16) == 0.1
        assert np.isnan(m.fiducial_sigma(np.nan, np.nan, 3))


def test_fiducial_covariance_keeps_the_colour_magnitude_anticorrelation():
    with isolated_sys_modules():
        m = _mod()
        rng = np.random.default_rng(0)
        n = 4000
        e1, e2 = 0.03, 0.04
        m1 = rng.normal(0, e1, n)
        m2 = rng.normal(0, e2, n) + np.repeat(np.arange(4), n // 4)
        color = m1 - (m2 - np.repeat(np.arange(4), n // 4))
        fid = m.bin_fiducials(
            m2, color, mag_err=np.full(n, e2), color_err=np.full(n, np.hypot(e1, e2)),
            color_mag_cov=np.full(n, -(e2**2)), bins=np.array([-0.5, 0.5, 1.5, 2.5, 3.5]),
        )
        rho = fid.cov / (fid.mag_err * fid.color_err)
        assert np.allclose(rho, -e2 / np.hypot(e1, e2), atol=0.05)


def test_profiled_offsets_recover_distance_and_reddening():
    with isolated_sys_modules():
        m = _mod()
        rng = np.random.default_rng(1)
        mag, color, err = _stars(rng, 3.0, dm=0.12, de=0.03)
        fid = m.bin_fiducials(mag, color, mag_err=err, color_err=err, bins=np.linspace(0.3, 7.7, 25))
        iso_m, iso_c = _isochrone(3.0)
        res = m.profile_chi_square(fid, iso_m, iso_c, a_per_e=A_PER_E, c_per_e=C_PER_E,
                                   sigma_m=0.3, sigma_e=0.1)
        assert res.delta_e == pytest.approx(0.03, abs=0.015)
        assert res.delta_m + res.delta_e * A_PER_E == pytest.approx(0.12 + 0.03 * A_PER_E, abs=0.05)


@pytest.mark.parametrize("true_age", [3.0, 3.1])
def test_age_interval_coverage(true_age):
    """1σ age interval covers the true age in ~68 % of the realisations.

    3.1 lies between the grid ages (step 0.25): the best age is not tied to
    the grid.
    """
    with isolated_sys_modules():
        m = _mod()
        rng = np.random.default_rng(2)
        ages = np.arange(1.0, 5.01, 0.25)
        grid = [_isochrone(a) for a in ages]
        hits, nus = [], []
        for _ in range(60):
            dm, de = rng.normal(0, 0.1), rng.normal(0, 0.02)
            mag, color, err = _stars(rng, true_age, dm, de, n=250, sigma=0.05)
            fid = m.bin_fiducials(mag, color, mag_err=err, color_err=err, bins=np.linspace(0.3, 7.7, 20))
            chi2 = [
                m.profile_chi_square(fid, im, ic, a_per_e=A_PER_E, c_per_e=C_PER_E,
                                     sigma_m=0.1, sigma_e=0.02).chi_square
                for im, ic in grid
            ]
            iv = m.age_interval(ages, np.asarray(chi2), n_fiducials=len(fid), n_free=2)
            lo = iv.best_age - (iv.age_err_lo if np.isfinite(iv.age_err_lo) else 0.0)
            hi = iv.best_age + (iv.age_err_hi if np.isfinite(iv.age_err_hi) else 0.0)
            hits.append(lo <= true_age <= hi)
            nus.append(iv.chi_square_nu)
        assert 0.55 <= np.mean(hits) <= 0.82
        assert 0.75 <= np.median(nus) <= 1.15


def test_data_scale_keeps_offsets_near_the_prior():
    """Overconfident fiducials (χ²_ν ≫ 1) must not drag the offsets away."""
    with isolated_sys_modules():
        m = _mod()
        rng = np.random.default_rng(4)
        mag, color, err = _stars(rng, 3.0, dm=0.0, de=0.0, sigma=0.05)
        # errors claimed 5x too small, isochrone of the wrong age
        fid = m.bin_fiducials(mag, color, mag_err=err / 5, color_err=err / 5,
                              bins=np.linspace(0.3, 7.7, 25))
        iso_m, iso_c = _isochrone(2.5)
        kw = dict(a_per_e=A_PER_E, c_per_e=C_PER_E, sigma_m=0.05, sigma_e=0.01)
        raw = m.profile_chi_square(fid, iso_m, iso_c, **kw)
        scale = raw.chi_square_data / (len(fid) - 1)
        scaled = m.profile_chi_square(fid, iso_m, iso_c, data_scale=scale, **kw)
        assert scale > 3
        assert abs(scaled.delta_e) <= abs(raw.delta_e) + 1e-9
        assert abs(scaled.delta_m) <= abs(raw.delta_m) + 1e-9
        assert scaled.chi_square_data >= raw.chi_square_data - 1e-6
