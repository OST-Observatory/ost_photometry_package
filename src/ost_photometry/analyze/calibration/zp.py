"""Median zero-point fitting."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from .result import CalibrationResult, TransformationCoefficients
from .statistics import catalog_errors, fit_zero_point


def comparison_mask_from_std_columns(table: Table, filters: list[str]) -> np.ndarray:
    """True rows with finite ``mag_std_<filter>`` for all requested filters."""
    mask = np.ones(len(table), dtype=bool)
    for filter_ in filters:
        std_col = f"mag_std_{filter_}"
        if std_col in table.colnames:
            mask &= np.isfinite(np.asarray(table[std_col], dtype=float))
        else:
            mask &= False
    return mask


def fit_median_zero_point_epoch(
    data: Table,
    epoch_id: str,
    filters: list[str],
    comparison_mask: np.ndarray,
    *,
    mag_col_prefix: str = "mag_",
    std_col_prefix: str = "mag_std_",
    err_col_prefix: str = "err_",
    min_comparisons: int = 3,
    color_index_filters: dict[str, tuple[str, str]] | None = None,
    sigma_clip: float | None = None,
    max_clip_iterations: int = 5,
    fixed_color_terms: dict[str, tuple[float, float]] | None = None,
) -> CalibrationResult:
    """
    Zero point per filter without a fitted colour term.

    Weighted by instrumental and catalog errors plus the intrinsic scatter
    that makes χ²_ν = 1, with clipping of normalised residuals at
    ``sigma_clip`` (:func:`~ost_photometry.analyze.calibration.statistics.fit_zero_point`).
    ``fixed_color_terms`` (``{filter: (T, σ_T)}``) subtracts a known colour
    term first (``calibration_grouping="fixed"``); else T = 0.
    """
    color_indices = color_index_filters or {}
    result = CalibrationResult(identifier=epoch_id)
    cand = np.asarray(comparison_mask, dtype=bool)
    comp_idx = np.flatnonzero(cand)
    comps = data[cand]
    n_table = len(data)

    for filter_ in filters:
        inst_col = f"{mag_col_prefix}{filter_}"
        std_col = f"{std_col_prefix}{filter_}"
        if inst_col not in comps.colnames or std_col not in comps.colnames:
            continue
        m_inst = np.asarray(comps[inst_col], dtype=float)
        m_std = np.asarray(comps[std_col], dtype=float)
        err_col = f"{err_col_prefix}{filter_}"
        sigma_inst = (
            np.abs(np.asarray(comps[err_col], dtype=float))
            if err_col in comps.colnames
            else np.full(len(comps), np.nan)
        )
        cat_col = f"{std_col_prefix.replace('mag_', 'err_')}{filter_}"
        sigma_cat = catalog_errors(
            np.asarray(comps[cat_col], dtype=float) if cat_col in comps.colnames else None,
            len(comps),
        )
        ci = color_indices.get(filter_, ("B", "V"))
        fixed = (fixed_color_terms or {}).get(filter_)
        color = np.zeros(len(comps))
        if fixed is not None:
            c1, c2 = f"{std_col_prefix}{ci[0]}", f"{std_col_prefix}{ci[1]}"
            if c1 in comps.colnames and c2 in comps.colnames:
                color = np.asarray(comps[c1], dtype=float) - np.asarray(comps[c2], dtype=float)
        fit = fit_zero_point(
            m_std - m_inst,
            color,
            sigma_inst,
            sigma_cat,
            fit_color=False,
            fixed_color_term=0.0 if fixed is None else float(fixed[0]),
            fixed_color_term_err=0.0 if fixed is None else float(fixed[1]),
            sigma_clip=sigma_clip,
            max_iterations=max_clip_iterations,
            min_stars=min_comparisons,
        )
        if fit.n_used < min_comparisons or not np.isfinite(fit.zero_point):
            continue
        result.transformation[filter_] = TransformationCoefficients(
            filter_name=filter_,
            color_term=fit.color_term,
            color_term_err=fit.color_term_err,
            zero_point=fit.zero_point,
            zero_point_err=fit.zero_point_err,
            color_index_filters=ci,
            n_stars_used=fit.n_used,
            rms_residual=fit.rms,
            sigma_int=fit.sigma_int,
            catalog_error_scale=fit.catalog_error_scale,
            chi2_nu=fit.chi2_nu,
            quality_flag=fit.quality_flag,
            cov_phot=fit.cov_phot,
            cov_cat=fit.cov_cat,
        )
        full_mask = np.zeros(n_table, dtype=bool)
        full_mask[comp_idx[fit.used]] = True
        result.calibrator_mask_by_filter[filter_] = full_mask

    result.n_comparison_stars = int(np.sum(cand))
    return result


def zp_subsample_statistic(
    m_std: np.ndarray,
    m_inst: np.ndarray,
    *,
    n_subsamples: int = 1000,
    fraction: float = 0.6,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Legacy-style subsample median statistic for QC (no plotting).

    Returns median-of-medians and spread across subsamples.
    """
    valid = np.isfinite(m_std) & np.isfinite(m_inst)
    zp_all = m_std[valid] - m_inst[valid]
    n = zp_all.size
    if n < 5:
        return {"median": float(np.median(zp_all)), "subsample_spread": 0.0}

    rng = np.random.default_rng(seed)
    n_sample = max(3, int(n * fraction))
    idx = rng.integers(0, high=n, size=(n_subsamples, n_sample))
    medians = np.median(zp_all[idx], axis=1)
    return {
        "median": float(np.median(medians)),
        "subsample_spread": float(np.std(medians)),
    }


__all__ = [
    "comparison_mask_from_std_columns",
    "fit_median_zero_point_epoch",
    "zp_subsample_statistic",
]
