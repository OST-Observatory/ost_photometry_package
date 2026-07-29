"""
Catalog-color derive-transform for two-filter epoch tables.

Ports the former ``derive_transformation_onthefly`` recipe: fit color-correction
slopes from catalog colors, median ZP per filter, then apply the differential
color-term transformation row-wise on multi-band epoch tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table, vstack

from .result import CalibrationResult, TransformationCoefficients
from .transform import unweighted_linear_fit
from .zp import comparison_mask_from_std_columns, fit_median_zero_point_epoch


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
    filters: List[str],
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
) -> Optional[DeriveTransformFit]:
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

    clipped = sigma_clip(zp_sum, sigma=sigma_clip, masked=True)
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
    filters: List[str],
    derive_fit: DeriveTransformFit,
    zp_result: CalibrationResult,
    *,
    color_index_filters: tuple[str, str] | None = None,
) -> CalibrationResult:
    """Build :class:`CalibrationResult` with derived ``c`` factors and median ZPs."""
    if len(filters) != 2:
        raise ValueError("derive_transform_from_data requires exactly two filters")
    ci = color_index_filters or (filters[0], filters[1])
    result = CalibrationResult(identifier=epoch_id)
    slopes = (derive_fit.c_slope_f0, derive_fit.c_slope_f1)
    intercepts = (derive_fit.z_intercept_f0, derive_fit.z_intercept_f1)
    for idx, f in enumerate(filters):
        zp_tc = zp_result.transformation.get(f)
        if zp_tc is None:
            continue
        c_apply = derive_transform_c_factor(
            idx, derive_fit.c_slope_f0, derive_fit.c_slope_f1
        )
        result.transformation[f] = TransformationCoefficients(
            filter_name=f,
            color_term=c_apply,
            zero_point=zp_tc.zero_point,
            zero_point_err=zp_tc.zero_point_err,
            color_index_filters=ci,
            n_stars_used=derive_fit.n_stars_used,
            rms_residual=zp_tc.rms_residual,
        )
        result.notes = (
            f"derive_transform slopes=({derive_fit.c_slope_f0:.5f}, "
            f"{derive_fit.c_slope_f1:.5f}) intercepts=({intercepts[0]:.4f}, "
            f"{intercepts[1]:.4f}) c_apply=({derive_transform_c_factor(0, *slopes):.5f}, "
            f"{derive_transform_c_factor(1, *slopes):.5f})"
        )
    result.n_comparison_stars = int(np.sum(derive_fit.comparison_mask))
    return result


def diagnostic_transformation_for_plots(
    result: CalibrationResult,
    filters: List[str],
    derive_fit: DeriveTransformFit,
) -> Dict[str, TransformationCoefficients]:
    """T/ZP for diagnostic plots (linear ``m_std-m_inst`` vs catalog color)."""
    intercepts = (derive_fit.z_intercept_f0, derive_fit.z_intercept_f1)
    slopes = (derive_fit.c_slope_f0, derive_fit.c_slope_f1)
    out: Dict[str, TransformationCoefficients] = {}
    for idx, f in enumerate(filters):
        base = result.transformation.get(f)
        if base is None:
            continue
        out[f] = TransformationCoefficients(
            filter_name=f,
            color_term=slopes[idx],
            zero_point=intercepts[idx],
            color_term_err=base.color_term_err,
            zero_point_err=base.zero_point_err,
            color_index_filters=base.color_index_filters,
            n_stars_used=base.n_stars_used,
            rms_residual=base.rms_residual,
        )
    return out


def apply_derive_transform_to_table(
    data: Table,
    calibration: CalibrationResult,
    filters: List[str],
    comparison_mask: np.ndarray,
    *,
    mag_col_prefix: str = "mag_",
    output_prefix: str = "mag_cal_",
    err_col_prefix: str = "err_",
    output_err_prefix: str = "err_cal_",
) -> Table:
    """
    Apply derive-transform on a multi-band epoch row table.

    ``m_cal = m_inst + ZP + c * color - c * color`` on comparison rows (per-star),
    ``m_cal = m_inst + ZP + c * color`` elsewhere, with ``color = mag_f0 - mag_f1``.
    """
    if len(filters) != 2:
        raise ValueError("derive_transform apply requires exactly two filters")
    f0, f1 = filters[0], filters[1]
    out = data.copy()
    m0 = np.asarray(out[f"{mag_col_prefix}{f0}"], dtype=float)
    m1 = np.asarray(out[f"{mag_col_prefix}{f1}"], dtype=float)
    color = m0 - m1
    cal_mask = np.asarray(comparison_mask, dtype=bool)

    for f in filters:
        if f not in calibration.transformation:
            continue
        tc = calibration.transformation[f]
        c = tc.color_term
        zp = tc.zero_point
        m_inst = np.asarray(out[f"{mag_col_prefix}{f}"], dtype=float)
        correction = np.zeros(len(out), dtype=float)
        correction[cal_mask] = c * color[cal_mask]
        m_cal = m_inst + zp + c * color - correction
        out[f"{output_prefix}{f}"] = m_cal
        err_col = f"{err_col_prefix}{f}"
        if err_col in out.colnames:
            inst_err = np.asarray(out[err_col], dtype=float)
            out[f"{output_err_prefix}{f}"] = np.sqrt(
                np.maximum(inst_err**2 + tc.zero_point_err**2, 0.0)
            )
    return out


def apply_derive_transform_epochs(
    epochs: Dict[str, Table],
    results: Dict[str, CalibrationResult],
    filters: List[str],
    *,
    output_prefix: str = "mag_cal_",
) -> Table:
    """Apply derive-transform calibration per epoch and vstack."""
    tables: list[Table] = []
    for epoch_id, data in epochs.items():
        cal = results.get(epoch_id)
        if cal is None:
            continue
        mask = comparison_mask_from_std_columns(data, filters)
        out = apply_derive_transform_to_table(
            data,
            cal,
            filters,
            mask,
            output_prefix=output_prefix,
        )
        out["epoch_id"] = epoch_id
        tables.append(out)
    return vstack(tables) if tables else Table()


def fit_epoch_derive_transform(
    table: Table,
    epoch_id: str,
    filters: List[str],
    *,
    color_indices: dict[str, tuple[str, str]] | None = None,
    min_comparisons: int = 5,
    sigma_clip: float = 2.5,
    zp_subsample_statistic: bool = False,
    distribution_samples: int = 1000,
) -> Optional[tuple[CalibrationResult, DeriveTransformFit]]:
    """Fit one epoch; returns calibration result and raw derive-transform fit for plots."""
    if len(filters) != 2:
        return None
    f0, f1 = filters[0], filters[1]
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

    zp_result = fit_median_zero_point_epoch(
        table,
        epoch_id,
        filters,
        derive_fit.comparison_mask,
        color_index_filters=color_indices,
        min_comparisons=min_comparisons,
    )
    if not zp_result.transformation:
        return None

    ci = (f0, f1)
    if color_indices:
        ci = color_indices.get(f1, color_indices.get(f0, ci))

    result = calibration_result_from_derive_transform_fit(
        epoch_id,
        filters,
        derive_fit,
        zp_result,
        color_index_filters=ci,
    )
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
    "apply_derive_transform_epochs",
    "apply_derive_transform_to_table",
    "calibration_result_from_derive_transform_fit",
    "derive_transform_c_factor",
    "diagnostic_transformation_for_plots",
    "fit_color_corrections_epoch",
    "fit_epoch_derive_transform",
]
