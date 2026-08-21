"""Derive-transform backend (``derive_transform_from_data`` within ``linear_fit``)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from ...warnings_types import OstPhotometryAnalyzeWarning
from ..derive_transform import (
    diagnostic_transformation_for_plots,
    fit_epoch_derive_transform,
)
from ..result import CalibrationResult

if TYPE_CHECKING:
    from astropy.table import Table

    from ...pipeline.config import PipelineConfig


def _fit_qc_metrics_for_epoch(
    table: Table,
    filters: list[str],
    diagnostic_coeffs: dict,
    used_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-filter RMS and n_used for catalog-color derive-transform fits."""
    out: dict[str, dict[str, float]] = {}
    used = np.asarray(used_mask, dtype=bool)
    for f in filters:
        tc = diagnostic_coeffs.get(f)
        if tc is None:
            continue
        inst_col, std_col = f"mag_{f}", f"mag_std_{f}"
        if inst_col not in table.colnames or std_col not in table.colnames:
            continue
        m_inst = np.asarray(table[inst_col], dtype=float)
        m_std = np.asarray(table[std_col], dtype=float)
        delta = m_std - m_inst
        ci_f1, ci_f2 = tc.color_index_filters
        c1, c2 = f"mag_std_{ci_f1}", f"mag_std_{ci_f2}"
        if c1 in table.colnames and c2 in table.colnames:
            color = (
                np.asarray(table[c1], dtype=float)
                - np.asarray(table[c2], dtype=float)
            )
        else:
            color = np.zeros(len(table), dtype=float)
        sel = used & np.isfinite(delta) & np.isfinite(color)
        n_used = int(np.sum(sel))
        if n_used == 0:
            out[f] = {"rms": float("nan"), "n_used": 0.0}
            continue
        resid = delta[sel] - (
            float(tc.color_term) * color[sel] + float(tc.zero_point)
        )
        out[f] = {"rms": float(np.nanstd(resid)), "n_used": float(n_used)}
    return out


def fit_epochs(
    epochs: dict[str, Table],
    filters: list[str],
    config: PipelineConfig,
    *,
    color_indices: dict[str, tuple[str, str]] | None = None,
    output_dir: str | None = None,
    file_type: str = "pdf",
) -> dict[str, CalibrationResult] | None:
    """
    Fit catalog-color derive-transform per epoch.

    Returns ``None`` when fewer than two filters are requested (caller should
    fall back to the standard linear backend).
    """
    if len(filters) != 2:
        warnings.warn(
            "derive_transform_from_data requires exactly two filters; "
            "falling back to standard linear_fit.",
            category=OstPhotometryAnalyzeWarning,
            stacklevel=2,
        )
        return None

    results: dict[str, CalibrationResult] = {}
    plot_epochs: dict = {}
    plot_coeffs: dict = {}
    plot_masks: dict = {}
    plot_metrics: dict = {}

    for epoch_id, table in epochs.items():
        fitted = fit_epoch_derive_transform(
            table,
            epoch_id,
            filters,
            color_indices=color_indices,
            min_comparisons=5,
            sigma_clip=config.fit_sigma_clip,
            zp_subsample_statistic=config.zp_subsample_statistic,
            distribution_samples=config.distribution_samples,
        )
        if fitted is None:
            warnings.warn(
                f"[{epoch_id}] derive_transform fit failed; epoch skipped.",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )
            continue
        result, derive_fit = fitted
        results[epoch_id] = result
        plot_epochs[epoch_id] = table
        coeffs = diagnostic_transformation_for_plots(result, filters, derive_fit)
        plot_coeffs[epoch_id] = coeffs
        plot_masks[epoch_id] = derive_fit.comparison_mask
        plot_metrics[epoch_id] = _fit_qc_metrics_for_epoch(
            table, filters, coeffs, derive_fit.comparison_mask
        )

    if not results:
        return None

    if output_dir:
        from ... import plots
        from ..engine import prepare_calibration_check_plots

        # Per-epoch fit QC: catalog-color slopes (not applied c factors).
        for epoch_id, coeffs in plot_coeffs.items():
            prepare_calibration_check_plots(
                output_dir,
                {epoch_id: plot_epochs[epoch_id]},
                {
                    epoch_id: CalibrationResult(
                        identifier=epoch_id,
                        transformation=coeffs,
                    )
                },
                filters,
                file_type=file_type,
                filename_prefix="derive_transform",
                title_prefix="Derive-transform fit (catalog color)",
                fit_masks={epoch_id: plot_masks[epoch_id]},
            )

        ordered_ids = list(results.keys())
        if len(ordered_ids) >= 1:
            # Fit quality across epochs (slopes / RMS / n).
            plots.plot_derive_transform_fit_overview(
                output_dir,
                ordered_ids,
                [plot_coeffs[k] for k in ordered_ids],
                [plot_metrics[k] for k in ordered_ids],
                filters,
                file_type=file_type,
                output_basename="derive_transform_fit_overview",
            )
            # Stability of applied c factors + median ZPs across epochs.
            plots.plot_calibration_night_summary(
                output_dir,
                ordered_ids,
                [results[k].transformation for k in ordered_ids],
                filters,
                file_type=file_type,
                output_basename="derive_transform_summary",
            )

    return results


__all__ = ["fit_epochs"]
