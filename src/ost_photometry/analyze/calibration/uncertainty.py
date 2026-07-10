"""Uncertainty propagation for calibrated magnitudes."""

from __future__ import annotations

from typing import Literal

import numpy as np
from astropy.table import Table

UncertaintyMode = Literal["fit_errors", "flux_monte_carlo", "both"]


def propagate_fit_errors(
    table: Table,
    filter_: str,
    *,
    inst_err_col: str | None = None,
    cal_err_col: str | None = None,
    zp_err: float = 0.0,
    color_term_err: float = 0.0,
) -> None:
    """Set calibrated magnitude errors from instrumental errors + fit parameter errors."""
    inst_err_col = inst_err_col or f"err_{filter_}"
    cal_err_col = cal_err_col or f"err_cal_{filter_}"
    if inst_err_col not in table.colnames:
        return
    inst_err = np.asarray(table[inst_err_col], dtype=float)
    total = np.sqrt(inst_err**2 + zp_err**2 + color_term_err**2)
    table[cal_err_col] = total


def propagate_flux_monte_carlo(
    flux: np.ndarray,
    flux_err: np.ndarray,
    zp: float,
    *,
    n_samples: int = 1000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo magnitude errors from flux uncertainties (legacy ``distribution_from_table``).

    Returns (mag_cal, mag_cal_err) arrays.
    """
    rng = np.random.default_rng(seed)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    valid = np.isfinite(flux) & (flux > 0) & np.isfinite(flux_err)
    mag = np.full(flux.shape, np.nan)
    mag_err = np.full(flux.shape, np.nan)
    if not np.any(valid):
        return mag, mag_err

    n = int(np.sum(valid))
    samples = rng.normal(
        flux[valid],
        np.maximum(flux_err[valid], 1e-12),
        size=(n_samples, n),
    )
    samples = np.maximum(samples, 1e-12)
    mag_samples = -2.5 * np.log10(samples)
    mag[valid] = -2.5 * np.log10(flux[valid])
    mag_err[valid] = np.nanstd(mag_samples, axis=0)
    mag[valid] = mag[valid] + zp
    return mag, mag_err


__all__ = ["UncertaintyMode", "propagate_fit_errors", "propagate_flux_monte_carlo"]
