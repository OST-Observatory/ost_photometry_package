"""Derive-transform backend (``derive_transform_from_data`` within ``linear_fit``)."""

from __future__ import annotations

import warnings
from typing import Dict, List, TYPE_CHECKING

from ...warnings_types import OstPhotometryAnalyzeWarning
from ..derive_transform import (
    diagnostic_transformation_for_plots,
    fit_epoch_derive_transform,
)
from ..result import CalibrationResult

if TYPE_CHECKING:
    from astropy.table import Table

    from ...pipeline.config import PipelineConfig


def fit_epochs(
    epochs: Dict[str, "Table"],
    filters: List[str],
    config: "PipelineConfig",
    *,
    color_indices: dict[str, tuple[str, str]] | None = None,
    output_dir: str | None = None,
    file_type: str = "pdf",
) -> Dict[str, CalibrationResult] | None:
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

    results: Dict[str, CalibrationResult] = {}
    plot_epochs: dict = {}
    plot_coeffs: dict = {}

    for epoch_id, table in epochs.items():
        fitted = fit_epoch_derive_transform(
            table,
            epoch_id,
            filters,
            color_indices=color_indices,
            min_comparisons=5,
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
        plot_coeffs[epoch_id] = diagnostic_transformation_for_plots(
            result, filters, derive_fit
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
            )

        # Stability of applied c factors + median ZPs across epochs.
        ordered_ids = list(results.keys())
        if len(ordered_ids) >= 1:
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
