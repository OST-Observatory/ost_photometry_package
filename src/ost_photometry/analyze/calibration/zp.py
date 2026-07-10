"""Median zero-point fitting (legacy-style)."""

from __future__ import annotations

from typing import List

import numpy as np
from astropy.table import Table

from .result import CalibrationResult, TransformationCoefficients


def comparison_mask_from_std_columns(table: Table, filters: List[str]) -> np.ndarray:
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
    filters: List[str],
    comparison_mask: np.ndarray,
    *,
    mag_col_prefix: str = "mag_",
    std_col_prefix: str = "mag_std_",
    min_comparisons: int = 3,
    color_index_filters: dict[str, tuple[str, str]] | None = None,
) -> CalibrationResult:
    """
    Fit zero points as median(m_std - m_inst) per filter; color term fixed at 0.

    Mirrors legacy ``prepare_zero_point`` / median ZP without iterative sigma-clip.
    """
    color_indices = color_index_filters or {}
    result = CalibrationResult(identifier=epoch_id)
    comps = data[comparison_mask]

    for filter_ in filters:
        inst_col = f"{mag_col_prefix}{filter_}"
        std_col = f"{std_col_prefix}{filter_}"
        if inst_col not in comps.colnames or std_col not in comps.colnames:
            continue

        m_inst = np.asarray(comps[inst_col], dtype=float)
        m_std = np.asarray(comps[std_col], dtype=float)
        valid = np.isfinite(m_inst) & np.isfinite(m_std)
        if np.sum(valid) < min_comparisons:
            continue

        residuals = m_std[valid] - m_inst[valid]
        zp = float(np.median(residuals))
        zp_err = float(np.std(residuals) / np.sqrt(np.sum(valid)))
        ci = color_indices.get(filter_, ("B", "V"))

        result.transformation[filter_] = TransformationCoefficients(
            filter_name=filter_,
            color_term=0.0,
            color_term_err=0.0,
            zero_point=zp,
            zero_point_err=zp_err,
            color_index_filters=ci,
            n_stars_used=int(np.sum(valid)),
            rms_residual=float(np.nanstd(residuals)),
        )

    result.n_comparison_stars = int(np.sum(comparison_mask))
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
