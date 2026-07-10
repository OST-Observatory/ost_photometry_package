"""Median-ZP calibration backend."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from ..result import CalibrationResult
from ..zp import comparison_mask_from_std_columns, fit_median_zero_point_epoch

if TYPE_CHECKING:
    from astropy.table import Table

    from ...pipeline.config import PipelineConfig


def fit_epochs(
    epochs: Dict[str, "Table"],
    filters: List[str],
    config: "PipelineConfig",
    *,
    color_indices: dict[str, tuple[str, str]] | None = None,
) -> Dict[str, CalibrationResult]:
    """Fit median ZP per epoch (``grouping=per_image``) or one ZP for all (``ensemble``)."""
    grouping = config.calibration_grouping
    min_comparisons = 3

    if grouping == "ensemble":
        from astropy.table import vstack

        combined = vstack(list(epochs.values()))
        mask = comparison_mask_from_std_columns(combined, filters)
        result = fit_median_zero_point_epoch(
            combined,
            "ensemble",
            filters,
            mask,
            color_index_filters=color_indices,
            min_comparisons=min_comparisons,
        )
        return {eid: result for eid in epochs}

    if grouping == "per_night":
        from astropy.table import vstack

        combined = vstack(list(epochs.values()))
        mask = comparison_mask_from_std_columns(combined, filters)
        night_result = fit_median_zero_point_epoch(
            combined,
            "night_combined",
            filters,
            mask,
            color_index_filters=color_indices,
            min_comparisons=min_comparisons,
        )
        return {eid: night_result for eid in epochs}

    # per_image (default for n2_stack) and fixed (same as per_image without preset coeffs)
    results: Dict[str, CalibrationResult] = {}
    for epoch_id, data in epochs.items():
        mask = comparison_mask_from_std_columns(data, filters)
        results[epoch_id] = fit_median_zero_point_epoch(
            data,
            epoch_id,
            filters,
            mask,
            color_index_filters=color_indices,
            min_comparisons=min_comparisons,
        )
    return results


__all__ = ["fit_epochs"]
