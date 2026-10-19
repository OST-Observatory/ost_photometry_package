"""
Catalog-color derive-transform for two-filter epoch tables.

Ports the former ``derive_transformation_onthefly`` recipe: comparison stars
are selected jointly in both filters (clip on the summed zero points, then
on the residuals of ``m_std - m_inst`` vs catalog colour). The colour terms
and zero points are then fitted per filter with
:func:`~ost_photometry.analyze.calibration.statistics.fit_zero_point`
(weighted, intrinsic scatter, errors) and applied by the standard
:meth:`~ost_photometry.analyze.calibration.photometer.DifferentialPhotometer.apply_transform_to_table`,
which solves the two filters jointly. The former row-wise apply mixed the
instrumental colour (with its zero-point difference) into non-catalog stars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.stats import sigma_clip as astropy_sigma_clip
from astropy.table import Table

from .result import CalibrationResult, TransformationCoefficients
from .statistics import catalog_errors, fit_zero_point, shared_catalog_covariance
from .transform import unweighted_linear_fit
from .zp import comparison_mask_from_std_columns


@dataclass
class DeriveTransformFit:
    """Raw slopes and intercepts from catalog-color derive-transform fits."""

    c_slope_f0: float
    c_slope_f1: float
    z_intercept_f0: float
    z_intercept_f1: float
    n_stars_used: int
    comparison_mask: np.ndarray


def derive_transform_c_factor(
    filter_index: int,
    c_slope_f0: float,
    c_slope_f1: float,
) -> float:
    """Applied ``c`` factor from fitted slopes (former ``transformation_core`` derive mode)."""
    denom = 1.0 - c_slope_f0 + c_slope_f1
    if abs(denom) < 1e-12:
        return 0.0
    if filter_index == 0:
        return float(c_slope_f0 / denom)
    return float(c_slope_f1 / denom)


def _comparison_mask_with_instrumental(
    table: Table,
    filters: list[str],
    comparison_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(comparison_mask, dtype=bool).copy()
    for f in filters:
        inst_col = f"mag_{f}"
        std_col = f"mag_std_{f}"
        if inst_col not in table.colnames or std_col not in table.colnames:
            return np.zeros(len(table), dtype=bool)
        mask &= np.isfinite(np.asarray(table[inst_col], dtype=float))
        mask &= np.isfinite(np.asarray(table[std_col], dtype=float))
    return mask


def fit_color_corrections_epoch(
    table: Table,
    filter_f0: str,
    filter_f1: str,
    comparison_mask: np.ndarray,
    *,
    sigma_clip: float = 2.5,
    min_comparisons: int = 5,
    max_clip_iterations: int = 5,
) -> DeriveTransformFit | None:
    """
    Fit catalog-color slopes for a two-filter epoch table.

    1. Sigma-clip on ``zp_sum = (m_inst−m_std)_0 + (m_inst−m_std)_1``.
    2. Iterative linear fits of ``m_std−m_inst`` vs catalog color; reject stars
       whose residual in either filter exceeds ``sigma_clip`` × RMS.

    ``sigma_clip`` is typically :attr:`PipelineConfig.fit_sigma_clip` (lower =
    more aggressive outlier rejection).
    """
    filters = [filter_f0, filter_f1]
    mask = _comparison_mask_with_instrumental(table, filters, comparison_mask)
    if np.sum(mask) < min_comparisons:
        return None

    m_std_0 = np.asarray(table[f"mag_std_{filter_f0}"], dtype=float)[mask]
    m_std_1 = np.asarray(table[f"mag_std_{filter_f1}"], dtype=float)[mask]
    m_inst_0 = np.asarray(table[f"mag_{filter_f0}"], dtype=float)[mask]
    m_inst_1 = np.asarray(table[f"mag_{filter_f1}"], dtype=float)[mask]

    color_literature = m_std_0 - m_std_1
    diff_0 = m_std_0 - m_inst_0
    diff_1 = m_std_1 - m_inst_1
    zp_sum = (m_inst_1 - m_std_1) + (m_inst_0 - m_std_0)

    clipped = astropy_sigma_clip(zp_sum, sigma=sigma_clip, masked=True)
    keep = ~np.asarray(clipped.mask, dtype=bool)
    keep &= np.isfinite(zp_sum) & np.isfinite(color_literature)
    keep &= np.isfinite(diff_0) & np.isfinite(diff_1)
    if np.sum(keep) < min_comparisons:
        return None

    c0 = c1 = z0 = z1 = 0.0
    for _ in range(max_clip_iterations):
        if np.sum(keep) < min_comparisons:
            return None
        c0, z0 = unweighted_linear_fit(color_literature[keep], diff_0[keep])
        c1, z1 = unweighted_linear_fit(color_literature[keep], diff_1[keep])
        res0 = diff_0 - (c0 * color_literature + z0)
        res1 = diff_1 - (c1 * color_literature + z1)
        rms0 = float(np.nanstd(res0[keep]))
        rms1 = float(np.nanstd(res1[keep]))
        if rms0 <= 0.0 and rms1 <= 0.0:
            break
        new_keep = keep.copy()
        if rms0 > 0.0:
            new_keep &= np.abs(res0) < sigma_clip * rms0
        if rms1 > 0.0:
            new_keep &= np.abs(res1) < sigma_clip * rms1
        if int(np.sum(new_keep)) == int(np.sum(keep)) or int(np.sum(new_keep)) < min_comparisons:
            break
        keep = new_keep

    # Final coefficients on the retained set
    c0, z0 = unweighted_linear_fit(color_literature[keep], diff_0[keep])
    c1, z1 = unweighted_linear_fit(color_literature[keep], diff_1[keep])

    full_mask = np.zeros(len(table), dtype=bool)
    idx = np.nonzero(mask)[0][keep]
    full_mask[idx] = True

    return DeriveTransformFit(
        c_slope_f0=float(c0),
        c_slope_f1=float(c1),
        z_intercept_f0=float(z0),
        z_intercept_f1=float(z1),
        n_stars_used=int(np.sum(keep)),
        comparison_mask=full_mask,
    )


def calibration_result_from_derive_transform_fit(
    epoch_id: str,
    filters: list[str],
    derive_fit: DeriveTransformFit,
    table: Table,
    *,
    color_index_filters: tuple[str, str] | None = None,
) -> CalibrationResult:
    """Weighted T / ZP per filter on the stars selected by ``derive_fit``."""
    if len(filters) != 2:
        raise ValueError("derive_transform_from_data requires exactly two filters")
    ci = color_index_filters or (filters[0], filters[1])
    used = np.asarray(derive_fit.comparison_mask, dtype=bool)
    sub = table[used]
    color = np.asarray(sub[f"mag_std_{ci[0]}"], dtype=float) - np.asarray(
        sub[f"mag_std_{ci[1]}"], dtype=float
    )
    sigma_color = np.hypot(
        catalog_errors(_col(sub, f"err_std_{ci[0]}"), len(sub)),
        catalog_errors(_col(sub, f"err_std_{ci[1]}"), len(sub)),
    )
    result = CalibrationResult(identifier=epoch_id)
    for f in filters:
        sigma_inst = _col(sub, f"err_{f}")
        sigma_cat = catalog_errors(_col(sub, f"err_std_{f}"), len(sub))
        fit = fit_zero_point(
            np.asarray(sub[f"mag_std_{f}"], dtype=float) - np.asarray(sub[f"mag_{f}"], dtype=float),
            color,
            np.full(len(sub), np.nan) if sigma_inst is None else np.abs(sigma_inst),
            sigma_cat,
            fit_color=True,
            sigma_color=sigma_color,
            delta_color_cov=shared_catalog_covariance(f, ci, sigma_cat),
            sigma_clip=None,
        )
        if not np.isfinite(fit.zero_point):
            continue
        result.transformation[f] = TransformationCoefficients(
            filter_name=f,
            color_term=fit.color_term,
            color_term_err=fit.color_term_err,
            zero_point=fit.zero_point,
            zero_point_err=fit.zero_point_err,
            cov_tz=fit.cov_tz,
            color_index_filters=ci,
            n_stars_used=fit.n_used,
            rms_residual=fit.rms,
            sigma_int=fit.sigma_int,
            catalog_error_scale=fit.catalog_error_scale,
            chi2_nu=fit.chi2_nu,
            color_ref=fit.color_ref,
            quality_flag=fit.quality_flag,
            cov_phot=fit.cov_phot,
            cov_cat=fit.cov_cat,
        )
    result.notes = (
        f"derive_transform slopes=({derive_fit.c_slope_f0:.5f}, {derive_fit.c_slope_f1:.5f})"
    )
    result.n_comparison_stars = int(np.sum(used))
    result.calibrator_mask_by_filter = {f: used for f in result.transformation}
    return result


def _col(table: Table, name: str) -> np.ndarray | None:
    return np.asarray(table[name], dtype=float) if name in table.colnames else None


def diagnostic_transformation_for_plots(
    result: CalibrationResult,
    filters: list[str],
    derive_fit: DeriveTransformFit,
) -> dict[str, TransformationCoefficients]:
    """T/ZP for diagnostic plots (``m_std - m_inst`` vs catalog colour)."""
    return {f: result.transformation[f] for f in filters if f in result.transformation}


def fit_epoch_derive_transform(
    table: Table,
    epoch_id: str,
    filters: list[str],
    *,
    color_indices: dict[str, tuple[str, str]] | None = None,
    min_comparisons: int = 5,
    sigma_clip: float = 2.5,
    zp_subsample_statistic: bool = False,
    distribution_samples: int = 1000,
    quality_config: Any | None = None,
) -> tuple[CalibrationResult, DeriveTransformFit] | None:
    """Fit one epoch; returns calibration result and raw derive-transform fit for plots."""
    if len(filters) != 2:
        return None
    f0, f1 = filters[0], filters[1]
    if quality_config is not None:
        from .quality import calibrator_candidate_mask

        comp_mask = calibrator_candidate_mask(table, filters, quality_config)
    else:
        comp_mask = comparison_mask_from_std_columns(table, filters)
    derive_fit = fit_color_corrections_epoch(
        table,
        f0,
        f1,
        comp_mask,
        sigma_clip=sigma_clip,
        min_comparisons=min_comparisons,
    )
    if derive_fit is None:
        return None

    ci = (f0, f1)
    if color_indices:
        ci = color_indices.get(f1, color_indices.get(f0, ci))

    result = calibration_result_from_derive_transform_fit(
        epoch_id,
        filters,
        derive_fit,
        table,
        color_index_filters=ci,
    )
    if not result.transformation:
        return None
    if zp_subsample_statistic:
        from .zp import zp_subsample_statistic

        for f in filters:
            if f not in result.transformation:
                continue
            inst_col, std_col = f"mag_{f}", f"mag_std_{f}"
            m = derive_fit.comparison_mask
            stats = zp_subsample_statistic(
                np.asarray(table[std_col][m], dtype=float),
                np.asarray(table[inst_col][m], dtype=float),
                n_subsamples=distribution_samples,
            )
            result.notes = (
                f"{result.notes}; {f} subsample_median={stats['median']:.4f}"
            )
    return result, derive_fit


__all__ = [
    "DeriveTransformFit",
    "calibration_result_from_derive_transform_fit",
    "derive_transform_c_factor",
    "diagnostic_transformation_for_plots",
    "fit_color_corrections_epoch",
    "fit_epoch_derive_transform",
]
