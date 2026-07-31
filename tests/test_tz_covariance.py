"""Tests for T/ZP fit covariance in calibrated magnitude errors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from helpers import load_module_from_path, pkg_src

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


def _deps_available() -> bool:
    try:
        import photutils  # noqa: F401
        import regions  # noqa: F401
        return True
    except ImportError:
        return False


def _transform_mod():
    return load_module_from_path(
        "ost_photometry.analyze.calibration.transform",
        pkg_src() / "ost_photometry" / "analyze" / "calibration" / "transform.py",
    )


def test_weighted_linear_fit_returns_cov_tz():
    mod = _transform_mod()
    rng = np.random.default_rng(0)
    x = np.linspace(-0.5, 1.5, 40)
    true_t, true_zp = 0.12, 1.7
    y = true_t * x + true_zp + rng.normal(0.0, 0.02, size=x.size)
    w = np.ones_like(x)

    t, zp, t_err, zp_err, cov_tz = mod.weighted_linear_fit(x, y, w)

    assert abs(t - true_t) < 0.02
    assert abs(zp - true_zp) < 0.02
    assert t_err > 0.0 and zp_err > 0.0
    assert np.isfinite(cov_tz)

    sw = np.sqrt(w)
    a = np.column_stack([x * sw, sw])
    b = y * sw
    coeffs, residuals, _, _ = np.linalg.lstsq(a, b, rcond=None)
    s_sq = residuals[0] / (len(x) - 2)
    cov = s_sq * np.linalg.inv(a.T @ a)
    assert np.isclose(t, coeffs[0])
    assert np.isclose(zp, coeffs[1])
    assert np.isclose(t_err, np.sqrt(cov[0, 0]))
    assert np.isclose(zp_err, np.sqrt(cov[1, 1]))
    assert np.isclose(cov_tz, cov[0, 1])


def test_calibrated_magnitude_variance_includes_cov_term():
    mod = _transform_mod()
    color = np.array([0.0, 0.5, 1.0])
    inst_err = np.array([0.01, 0.01, 0.01])
    base = mod.calibrated_magnitude_variance(
        inst_err,
        color,
        color_term=0.1,
        color_term_err=0.02,
        zero_point_err=0.03,
        cov_tz=0.0,
    )
    with_cov = mod.calibrated_magnitude_variance(
        inst_err,
        color,
        color_term=0.1,
        color_term_err=0.02,
        zero_point_err=0.03,
        cov_tz=0.001,
    )
    assert np.isclose(with_cov[0], base[0])
    assert np.isclose(with_cov[1] - base[1], 2 * 0.5 * 0.001)
    assert np.isclose(with_cov[2] - base[2], 2 * 1.0 * 0.001)


@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_apply_transform_uses_cov_tz():
    from ost_photometry.analyze.calibration.result import (
        CalibrationResult,
        TransformationCoefficients,
    )
    from ost_photometry.analyze.differential_photometry import DifferentialPhotometer

    tc = TransformationCoefficients(
        filter_name="V",
        color_term=0.0,
        color_term_err=0.0,
        zero_point=1.0,
        zero_point_err=0.0,
        cov_tz=0.002,
        color_index_filters=("B", "V"),
    )
    assert tc.cov_tz == 0.002
    assert "cov(T,ZP)" in repr(tc)

    tbl = Table()
    tbl["mag_V"] = np.array([12.0, 12.5, 13.0])
    tbl["err_V"] = np.array([0.01, 0.01, 0.01])
    tbl["mag_B"] = np.array([12.5, 13.2, 14.0])
    tbl["err_B"] = np.array([0.01, 0.01, 0.01])

    cal = CalibrationResult(identifier="e0", transformation={"V": tc})
    phot = DifferentialPhotometer()
    out = phot.apply_transform_to_table(tbl, cal, filters=["V"])

    color = tbl["mag_B"] - tbl["mag_V"]
    # T=0 → no color-noise or σ_T terms; only inst + 2·color·cov_tz
    expected = np.sqrt(np.maximum(0.01**2 + 2.0 * color * 0.002, 0.0))
    assert np.allclose(out["err_cal_V"], expected, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_fit_epoch_stores_cov_tz(synthetic_calibration_epoch_table):
    from ost_photometry.analyze.differential_photometry import DifferentialPhotometer

    tbl = synthetic_calibration_epoch_table
    mask = np.asarray(tbl["is_comparison"], dtype=bool)
    phot = DifferentialPhotometer()
    result = phot.fit_transformation_epoch(
        tbl,
        epoch_id="epoch_000",
        filters=["B", "V"],
        comparison_mask=mask,
        determine_color_terms=True,
        color_term_fit="always",
    )
    for f in ("B", "V"):
        tc = result.transformation[f]
        assert hasattr(tc, "cov_tz")
        assert np.isfinite(tc.cov_tz)
