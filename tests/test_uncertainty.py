"""Tests for calibration uncertainty_mode propagation."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from helpers import load_module_from_path, pkg_src


def _uncertainty_mod():
    pkg = pkg_src() / "ost_photometry" / "analyze"
    load_module_from_path(
        "ost_photometry.analyze.warnings_types",
        pkg / "warnings_types.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.extinction",
        pkg / "extinction.py",
    )
    _result_mod()
    return load_module_from_path(
        "ost_photometry.analyze.calibration.uncertainty",
        pkg / "calibration" / "uncertainty.py",
    )


def _result_mod():
    return load_module_from_path(
        "ost_photometry.analyze.calibration.result",
        pkg_src() / "ost_photometry" / "analyze" / "calibration" / "result.py",
    )


def _synthetic_calibrated_table() -> Table:
    tbl = Table()
    tbl["epoch_id"] = ["night_a", "night_a", "night_b", "night_b"]
    tbl["mag_cal_V"] = [12.0, 13.0, 12.1, 13.1]
    tbl["err_cal_V"] = [0.03, 0.03, 0.04, 0.04]
    tbl["flux_V"] = [1000.0, 500.0, 950.0, 480.0]
    tbl["flux_err_V"] = [20.0, 15.0, 25.0, 18.0]
    return tbl


def _calibration_result():
    result_mod = _result_mod()
    TransformationCoefficients = result_mod.TransformationCoefficients
    CalibrationResult = result_mod.CalibrationResult
    tc = TransformationCoefficients(
        filter_name="V",
        color_term=0.0,
        zero_point=0.1,
        zero_point_err=0.01,
    )
    return CalibrationResult(identifier="night_a", transformation={"V": tc})


def test_flux_monte_carlo_changes_err_cal():
    unc = _uncertainty_mod()
    tbl = _synthetic_calibrated_table()
    cal = _calibration_result()
    results = {"night_a": cal, "night_b": cal}

    out = unc.apply_uncertainty_mode_to_calibrated_table(
        tbl,
        results,
        ["V"],
        uncertainty_mode="flux_monte_carlo",
        distribution_samples=500,
        random_seed=7,
    )

    assert not np.allclose(out["err_cal_V"], tbl["err_cal_V"])
    assert np.all(np.isfinite(out["err_cal_V"]))


def test_both_combines_fit_and_mc_errors():
    unc = _uncertainty_mod()
    tbl = _synthetic_calibrated_table()
    cal = _calibration_result()
    results = {"night_a": cal, "night_b": cal}

    mc_only = unc.apply_uncertainty_mode_to_calibrated_table(
        tbl,
        results,
        ["V"],
        uncertainty_mode="flux_monte_carlo",
        distribution_samples=500,
        random_seed=7,
    )
    both = unc.apply_uncertainty_mode_to_calibrated_table(
        tbl,
        results,
        ["V"],
        uncertainty_mode="both",
        distribution_samples=500,
        random_seed=7,
    )

    assert np.all(both["err_cal_V"] >= mc_only["err_cal_V"] - 1e-12)


def test_fit_errors_leaves_table_unchanged():
    unc = _uncertainty_mod()
    tbl = _synthetic_calibrated_table()
    cal = _calibration_result()

    out = unc.apply_uncertainty_mode_to_calibrated_table(
        tbl,
        {"night_a": cal},
        ["V"],
        uncertainty_mode="fit_errors",
    )
    assert np.allclose(out["err_cal_V"], tbl["err_cal_V"])


def test_propagate_flux_monte_carlo_applies_zp():
    unc = _uncertainty_mod()
    flux = np.array([1000.0, 500.0])
    ferr = np.array([20.0, 15.0])
    mag, err = unc.propagate_flux_monte_carlo(
        flux, ferr, zp=0.25, n_samples=1000, seed=1
    )
    assert np.all(np.isfinite(err))
    expected = -2.5 * np.log10(flux) + 0.25
    assert np.allclose(mag, expected)
