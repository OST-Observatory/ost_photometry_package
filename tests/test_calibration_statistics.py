"""Coverage of zero-point / colour-term errors (analyze.calibration.statistics)."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import isolated_sys_modules, load_module_from_path, pkg_src


def _stats():
    return load_module_from_path(
        "ost_photometry.analyze.calibration.statistics",
        pkg_src() / "ost_photometry" / "analyze" / "calibration" / "statistics.py",
    )


def _simulate(rng, n, *, zp=25.0, t=0.08, sigma_int=0.03):
    color = rng.uniform(0.2, 1.5, n)
    s_inst = rng.uniform(0.005, 0.04, n)
    s_cat = rng.uniform(0.01, 0.04, n)
    delta = zp + t * color + rng.normal(0, 1, n) * np.sqrt(s_inst**2 + s_cat**2 + sigma_int**2)
    return delta, color, s_inst, s_cat


@pytest.mark.parametrize(("n", "fit_color"), [(25, True), (8, True), (4, False)])
def test_one_sigma_intervals_cover_68_percent(n, fit_color):
    with isolated_sys_modules():
        st = _stats()
        rng = np.random.default_rng(n)
        hits_zp, hits_t, sig = [], [], []
        for _ in range(600):
            delta, color, s_inst, s_cat = _simulate(rng, n, t=0.08 if fit_color else 0.0)
            fit = st.fit_zero_point(delta, color, s_inst, s_cat, fit_color=fit_color, sigma_clip=None)
            hits_zp.append(abs(fit.zero_point - 25.0) <= fit.zero_point_err)
            if fit_color:
                hits_t.append(abs(fit.color_term - 0.08) <= fit.color_term_err)
            sig.append(fit.sigma_int)
        assert np.mean(hits_zp) == pytest.approx(0.683, abs=0.06)
        if fit_color:
            assert np.mean(hits_t) == pytest.approx(0.683, abs=0.06)
        if n >= 25:
            assert np.median(sig) == pytest.approx(0.03, abs=0.012)


@pytest.mark.parametrize("n", [40, 15])
def test_overestimated_catalog_errors_are_scaled_down(n):
    """Catalog errors twice the true scatter: s ≈ 0.5, coverage stays 68 %."""
    with isolated_sys_modules():
        st = _stats()
        rng = np.random.default_rng(10 + n)
        hits, scales = [], []
        for _ in range(600):
            delta, color, s_inst, s_cat = _simulate(rng, n, sigma_int=0.0)
            fit = st.fit_zero_point(delta, color, s_inst, 2.0 * s_cat, fit_color=True, sigma_clip=None)
            hits.append(abs(fit.zero_point - 25.0) <= fit.zero_point_err)
            scales.append(fit.catalog_error_scale)
            assert fit.sigma_int == 0.0 or fit.catalog_error_scale == 1.0
        assert np.mean(hits) == pytest.approx(0.683, abs=0.06)
        if n >= 40:
            assert np.median(scales) == pytest.approx(0.5, abs=0.1)


def test_clipped_target_keeps_coverage():
    """Default 2.5σ clipping: σ_int unbiased, coverage 68 %."""
    with isolated_sys_modules():
        st = _stats()
        assert st.clipped_variance(2.5) == pytest.approx(0.912, abs=0.002)
        assert st.clipped_variance(None) == 1.0
        rng = np.random.default_rng(7)
        hits, sig = [], []
        for _ in range(600):
            delta, color, s_inst, s_cat = _simulate(rng, 60)
            fit = st.fit_zero_point(delta, color, s_inst, s_cat, fit_color=True)
            hits.append(abs(fit.zero_point - 25.0) <= fit.zero_point_err)
            sig.append(fit.sigma_int)
        assert np.mean(hits) == pytest.approx(0.683, abs=0.05)
        assert np.median(sig) == pytest.approx(0.03, abs=0.004)


def test_catalog_errors_not_scaled_with_few_stars():
    with isolated_sys_modules():
        st = _stats()
        rng = np.random.default_rng(11)
        delta, color, s_inst, s_cat = _simulate(rng, st.MIN_STARS_CATALOG_SCALE - 1, sigma_int=0.0)
        fit = st.fit_zero_point(delta, color, s_inst, 3.0 * s_cat, fit_color=True, sigma_clip=None)
        assert fit.catalog_error_scale == 1.0


def test_few_calibrators_flagged_and_t_inflated():
    with isolated_sys_modules():
        st = _stats()
        rng = np.random.default_rng(1)
        delta, color, s_inst, s_cat = _simulate(rng, 4, sigma_int=0.0)
        fit = st.fit_zero_point(delta, color, s_inst, s_cat, fit_color=False, sigma_clip=None)
        assert "few_calibrators" in fit.quality_flag
        assert st.small_sample_factor(3) == pytest.approx(1.20, abs=0.02)


def test_outlier_is_clipped_on_normalised_residuals():
    with isolated_sys_modules():
        st = _stats()
        rng = np.random.default_rng(2)
        delta, color, s_inst, s_cat = _simulate(rng, 30, sigma_int=0.0)
        delta[3] += 0.8  # variable or blended calibrator
        fit = st.fit_zero_point(delta, color, s_inst, s_cat, fit_color=True)
        assert not fit.used[3]
        assert abs(fit.zero_point - 25.0) < 3 * fit.zero_point_err


def test_fixed_colour_term_keeps_its_error():
    with isolated_sys_modules():
        st = _stats()
        rng = np.random.default_rng(3)
        delta, color, s_inst, s_cat = _simulate(rng, 20, t=0.08, sigma_int=0.0)
        fit = st.fit_zero_point(
            delta, color, s_inst, s_cat, fit_color=False,
            fixed_color_term=0.08, fixed_color_term_err=0.01, sigma_clip=None,
        )
        assert fit.color_term == 0.08 and fit.color_term_err == 0.01
        assert abs(fit.zero_point - 25.0) < 4 * fit.zero_point_err
        assert "fixed_color_term" in fit.quality_flag


def test_combine_epochs_catalog_part_does_not_average_down():
    with isolated_sys_modules():
        st = _stats()
        rng = np.random.default_rng(4)
        n_ep = 30
        # transparency: epoch zero points scatter by 0.02 mag beyond their errors
        zp = 25.0 + rng.normal(0, 0.02, n_ep) + rng.normal(0, 0.005, n_ep)
        t = 0.08 + rng.normal(0, 0.01, n_ep)
        cov_phot = np.zeros((n_ep, 2, 2))
        cov_phot[:, 0, 0] = 0.005**2
        cov_phot[:, 1, 1] = 0.01**2
        cov_cat = np.zeros((n_ep, 2, 2))
        cov_cat[:, 0, 0] = 0.01**2  # same calibrators every epoch
        out = st.combine_epoch_fits(zp, t, cov_phot, cov_cat, color_ref=0.0)
        assert out["birge"] > 4.0
        assert out["zero_point_scatter"] == pytest.approx(0.02, abs=0.008)
        # the 0.01 mag catalog part stays in the night error
        assert out["zero_point_err"] >= 0.01
        assert abs(out["zero_point"] - 25.0) < 3 * out["zero_point_err"]


def test_fit_splits_photometric_and_catalog_covariance():
    with isolated_sys_modules():
        st = _stats()
        rng = np.random.default_rng(5)
        delta, color, s_inst, s_cat = _simulate(rng, 30)
        fit = st.fit_zero_point(delta, color, s_inst, s_cat, fit_color=True, sigma_clip=None)
        total = fit.cov_phot + fit.cov_cat
        assert total[0, 0] == pytest.approx(fit.zero_point_err**2, rel=1e-6)
        assert total[1, 1] == pytest.approx(fit.color_term_err**2, rel=1e-6)
        assert total[0, 1] == pytest.approx(fit.cov_tz, rel=1e-6, abs=1e-12)
        assert 0 < fit.cov_phot[0, 0] < fit.cov_cat[0, 0]


def test_calibration_epochs_table_rows():
    from ost_photometry.analyze.calibration.result import (
        CalibrationResult,
        TransformationCoefficients,
        calibration_epochs_table,
    )

    tc = TransformationCoefficients(
        filter_name="V", color_term=0.05, color_term_err=0.01, zero_point=24.0,
        zero_point_err=0.02, cov_phot=np.diag([0.0001, 0.00005]), cov_cat=np.diag([0.0003, 0.00005]),
        sigma_int=0.03, quality_flag="few_calibrators",
    )
    res = CalibrationResult(identifier="e0", transformation={"V": tc})
    tbl = calibration_epochs_table({"e0": res, "e1": res})
    assert len(tbl) == 2
    assert tbl["zero_point_err_phot"][0] == pytest.approx(0.01)
    assert tbl["zero_point_err_cat"][0] == pytest.approx(np.sqrt(0.0003))
    assert tbl["quality_flag"][0] == "few_calibrators"
