"""uncertainty_mode: analytic errors vs the Monte Carlo cross-check."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

pytest.importorskip("photutils")
pytest.importorskip("regions")

from ost_photometry.analyze.calibration import DifferentialPhotometer  # noqa: E402
from ost_photometry.analyze.calibration.result import (  # noqa: E402
    CalibrationResult,
    TransformationCoefficients,
)
from ost_photometry.analyze.calibration.uncertainty import (  # noqa: E402
    apply_uncertainty_mode_to_calibrated_table,
)


def _calibrated(snr: float):
    n = 4
    flux_v = np.full(n, 1.0e4)
    flux_b = np.full(n, 8.0e3)
    k = 2.5 / np.log(10)
    tbl = Table({
        "epoch_id": ["e0"] * n,
        "mag_B": -2.5 * np.log10(flux_b) + np.linspace(0, 1, n),
        "mag_V": -2.5 * np.log10(flux_v) + np.linspace(0, 0.8, n),
        "flux_B": flux_b, "flux_err_B": flux_b / snr,
        "flux_V": flux_v, "flux_err_V": flux_v / snr,
        "err_B": np.full(n, k / snr), "err_V": np.full(n, k / snr),
    })

    def tc(f, t, zp):
        return TransformationCoefficients(
            filter_name=f, color_term=t, color_term_err=0.01, zero_point=zp,
            zero_point_err=0.015, cov_tz=0.00005, color_index_filters=("B", "V"),
        )
    cal = CalibrationResult(identifier="e0", transformation={"B": tc("B", 0.1, 25.0), "V": tc("V", -0.05, 24.6)})
    out = DifferentialPhotometer().apply_transform_to_table(tbl, cal, filters=["B", "V"])
    out["epoch_id"] = "e0"
    return out, {"e0": cal}


def test_monte_carlo_agrees_with_analytic_at_high_snr():
    out, results = _calibrated(snr=200.0)
    mc = apply_uncertainty_mode_to_calibrated_table(
        out, results, ["B", "V"], uncertainty_mode="monte_carlo",
        distribution_samples=20000, random_seed=1,
    )
    assert np.allclose(mc["err_cal_V"], out["err_cal_V"], rtol=0.05)
    assert np.allclose(mc["err_cal_B"], out["err_cal_B"], rtol=0.05)
    assert not np.any(mc["cal_low_snr_V"])


def test_monte_carlo_flags_low_snr():
    out, results = _calibrated(snr=2.0)
    mc = apply_uncertainty_mode_to_calibrated_table(
        out, results, ["B", "V"], uncertainty_mode="monte_carlo",
        distribution_samples=4000, random_seed=2,
    )
    assert np.all(mc["cal_low_snr_V"])


@pytest.mark.parametrize("mode", ["both", "flux_monte_carlo"])
def test_removed_modes_raise(mode):
    out, results = _calibrated(snr=100.0)
    with pytest.raises(ValueError, match="removed"):
        apply_uncertainty_mode_to_calibrated_table(out, results, ["V"], uncertainty_mode=mode)


def test_fit_errors_leaves_table_unchanged():
    out, results = _calibrated(snr=100.0)
    same = apply_uncertainty_mode_to_calibrated_table(out, results, ["V"])
    assert same is out
