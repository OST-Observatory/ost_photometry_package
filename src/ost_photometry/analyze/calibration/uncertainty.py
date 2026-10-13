"""Monte Carlo cross-check of calibrated magnitude errors.

``uncertainty_mode``:

``fit_errors`` (default)
    ``err_cal_*`` from the analytic propagation of
    :meth:`~ost_photometry.analyze.calibration.photometer.DifferentialPhotometer.apply_transform_to_table`
    (``err_cal_rand_*`` / ``err_cal_sys_*``, joint solution of the colour
    filters).
``monte_carlo``
    ``err_cal_*`` replaced by half the 16–84 % range of a Monte Carlo:
    zero point and colour term drawn once per epoch and filter (shared by all
    stars, with their covariance and ``zero_point_sys``), instrumental flux
    and catalog colour per star; the two colour filters are solved jointly
    per draw. Non-Gaussian low-S/N errors come out right; stars with
    S/N < 3 are flagged (``cal_low_snr_<f>``) and get NaN where more than
    16 % of the draws have non-positive flux. ``err_cal_rand_*`` /
    ``err_cal_sys_*`` stay analytic.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy.table import Table

if TYPE_CHECKING:
    from .result import CalibrationResult

UncertaintyMode = Literal["fit_errors", "monte_carlo"]
_LOW_SNR = 3.0


def _analysis_warning_category():
    from ..warnings_types import OstPhotometryAnalyzeWarning

    return OstPhotometryAnalyzeWarning


def _resolve_calibration_result(
    results: dict[str, CalibrationResult],
    epoch_id: str,
) -> CalibrationResult | None:
    if epoch_id in results:
        return results[epoch_id]
    for key in ("night_combined", "ensemble", "fixed"):
        if key in results:
            return results[key]
    return None


def _param_cov(tc) -> np.ndarray:
    if tc.cov_phot is not None and tc.cov_cat is not None:
        cov = np.asarray(tc.cov_phot, dtype=float) + np.asarray(tc.cov_cat, dtype=float)
    else:
        cov = np.array([[tc.zero_point_err**2, tc.cov_tz], [tc.cov_tz, tc.color_term_err**2]])
    cov = np.where(np.isfinite(cov), cov, 0.0)
    cov[0, 0] += tc.zero_point_sys**2
    return cov


def monte_carlo_errors(
    table: Table,
    cal: CalibrationResult,
    filters: list[str],
    *,
    n_samples: int = 1000,
    seed: int | None = None,
    std_col_prefix: str = "mag_std_",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{filter: (err_mc, low_snr)}`` for one epoch's rows of a calibrated table."""
    from .photometer import _closed_system

    rng = np.random.default_rng(seed)
    n = len(table)
    use = [f for f in filters if f in cal.transformation and f"mag_{f}" in table.colnames]
    if not use or n == 0:
        return {}
    # instrumental (extinction-corrected) magnitudes per draw, from the flux
    inst = {}
    low = {}
    for f in use:
        mag = np.asarray(table[f"mag_{f}"], dtype=float)
        fc, fe = f"flux_{f}", f"flux_err_{f}"
        if fc in table.colnames and fe in table.colnames:
            flux = np.asarray(table[fc], dtype=float)
            ferr = np.abs(np.asarray(table[fe], dtype=float))
            draws = flux[None, :] + rng.normal(size=(n_samples, n)) * ferr[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(draws > 0, draws / flux[None, :], np.nan)
                inst[f] = mag[None, :] - 2.5 * np.log10(ratio)
                low[f] = ~(np.abs(flux / ferr) >= _LOW_SNR)
        else:
            err = np.abs(np.asarray(table[f"err_{f}"], dtype=float))
            inst[f] = mag[None, :] + rng.normal(size=(n_samples, n)) * err[None, :]
            low[f] = np.zeros(n, dtype=bool)
    # zero point and colour term per draw, shared by all stars
    zp = {}
    ct = {}
    for f in use:
        tc = cal.transformation[f]
        draws = rng.multivariate_normal([tc.zero_point, tc.color_term], _param_cov(tc), n_samples)
        zp[f] = draws[:, 0]
        ct[f] = draws[:, 1]
    calibrated = set(use)
    system = [f for f in _closed_system(cal, calibrated) if f in use]
    out_mags = {f: np.full((n_samples, n), np.nan) for f in use}
    if system:
        idx = {f: k for k, f in enumerate(system)}
        k = len(system)
        a = np.broadcast_to(np.eye(k), (n_samples, k, k)).copy()
        for f, i in idx.items():
            c1, c2 = cal.transformation[f].color_index_filters
            a[:, i, idx[c1]] -= ct[f]
            a[:, i, idx[c2]] += ct[f]
        a_inv = np.linalg.inv(a)
        rhs = np.stack([inst[f] + zp[f][:, None] for f in system], axis=2)  # (s, n, k)
        sol = np.einsum("sij,snj->sni", a_inv, rhs)
        for f, i in idx.items():
            out_mags[f] = sol[:, :, i]
    for f in use:
        tc = cal.transformation[f]
        c1, c2 = tc.color_index_filters
        s1, s2 = f"{std_col_prefix}{c1}", f"{std_col_prefix}{c2}"
        if s1 in table.colnames and s2 in table.colnames:
            cat = np.asarray(table[s1], dtype=float) - np.asarray(table[s2], dtype=float)
            e1 = std_col_prefix.replace("mag_", "err_") + c1
            e2 = std_col_prefix.replace("mag_", "err_") + c2
            ec = np.hypot(
                np.asarray(table[e1], dtype=float) if e1 in table.colnames else 0.02,
                np.asarray(table[e2], dtype=float) if e2 in table.colnames else 0.02,
            )
            color = cat[None, :] + rng.normal(size=(n_samples, n)) * np.nan_to_num(ec, nan=0.02)
        else:
            color = np.full((n_samples, n), np.nan)
        if abs(tc.color_term) <= 1e-12 and tc.color_term_err <= 0:
            color = np.zeros((n_samples, n))
        simple = inst[f] + ct[f][:, None] * color + zp[f][:, None]
        # rows where the joint solution has no value use the catalog colour
        out_mags[f] = np.where(np.isfinite(out_mags[f]), out_mags[f], simple)
    result = {}
    for f in use:
        draws = out_mags[f]
        good = np.isfinite(draws)
        frac_bad = 1.0 - good.mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p16, p84 = np.nanpercentile(draws, [16, 84], axis=0)
        err = 0.5 * (p84 - p16)
        err = np.where(frac_bad > 0.16, np.nan, err)
        result[f] = (err, low[f])
    return result


def apply_uncertainty_mode_to_calibrated_table(
    calibrated: Table,
    results: dict[str, CalibrationResult],
    filters: list[str],
    *,
    uncertainty_mode: UncertaintyMode = "fit_errors",
    distribution_samples: int = 1000,
    random_seed: int | None = None,
) -> Table:
    """Replace ``err_cal_*`` by the Monte Carlo errors (``monte_carlo``)."""
    if uncertainty_mode not in ("fit_errors", "monte_carlo"):
        raise ValueError(
            f"uncertainty_mode={uncertainty_mode!r} was removed: 'both' counted the "
            "instrumental error twice and 'flux_monte_carlo' dropped the zero-point and "
            "colour-term errors. Use 'fit_errors' (default) or 'monte_carlo'."
        )
    if uncertainty_mode == "fit_errors" or len(calibrated) == 0:
        return calibrated
    out = calibrated.copy()
    epoch_ids = (
        np.asarray(out["epoch_id"]).astype(str) if "epoch_id" in out.colnames else np.full(len(out), "")
    )
    for epoch_id in np.unique(epoch_ids):
        cal = _resolve_calibration_result(results, str(epoch_id))
        if cal is None:
            warnings.warn(
                f"No calibration result for epoch_id={epoch_id!r}; Monte Carlo skipped.",
                category=_analysis_warning_category(),
                stacklevel=2,
            )
            continue
        rows = np.flatnonzero(epoch_ids == epoch_id)
        mc = monte_carlo_errors(
            out[rows], cal, filters, n_samples=distribution_samples, seed=random_seed
        )
        for f, (err, low) in mc.items():
            col = f"err_cal_{f}"
            if col not in out.colnames:
                continue
            values = np.asarray(out[col], dtype=float)
            values[rows] = err
            out[col] = values
            flag = (
                np.asarray(out[f"cal_low_snr_{f}"], dtype=bool)
                if f"cal_low_snr_{f}" in out.colnames
                else np.zeros(len(out), dtype=bool)
            )
            flag[rows] = low
            out[f"cal_low_snr_{f}"] = flag
    return out


__all__ = [
    "UncertaintyMode",
    "apply_uncertainty_mode_to_calibrated_table",
    "monte_carlo_errors",
]
