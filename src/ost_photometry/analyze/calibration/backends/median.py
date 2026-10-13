"""Median-ZP calibration backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...extinction import extinction_airmass_ready
from ..quality import calibrator_candidate_mask
from ..result import CalibrationResult
from ..zp import fit_median_zero_point_epoch

if TYPE_CHECKING:
    from astropy.table import Table

    from ...pipeline.config import PipelineConfig


def fit_epochs(
    epochs: dict[str, Table],
    filters: list[str],
    config: PipelineConfig,
    *,
    color_indices: dict[str, tuple[str, str]] | None = None,
    extinction_corrector=None,
) -> dict[str, CalibrationResult]:
    """Fit median ZP per epoch (``grouping=per_image``) or one ZP for all (``ensemble``).

    ``extinction_corrector`` must be the one used when the result is applied:
    the zero point is fitted on extinction-corrected magnitudes, so the
    ``k·X`` subtracted on apply is not missing from the fit.
    """
    grouping = config.calibration_grouping
    min_comparisons = 3
    if extinction_corrector is not None:
        epochs = {
            eid: _extinction_corrected(tbl, filters, extinction_corrector)
            for eid, tbl in epochs.items()
        }

    if grouping == "ensemble":
        from astropy.table import vstack

        combined = vstack(list(epochs.values()))
        mask = calibrator_candidate_mask(combined, filters, config)
        result = fit_median_zero_point_epoch(
            combined,
            "ensemble",
            filters,
            mask,
            color_index_filters=color_indices,
            min_comparisons=min_comparisons,
            sigma_clip=config.fit_sigma_clip,
        )
        return {eid: result for eid in epochs}

    if grouping == "per_night":
        from astropy.table import vstack

        combined = vstack(list(epochs.values()))
        mask = calibrator_candidate_mask(combined, filters, config)
        night_result = fit_median_zero_point_epoch(
            combined,
            "night_combined",
            filters,
            mask,
            color_index_filters=color_indices,
            min_comparisons=min_comparisons,
            sigma_clip=config.fit_sigma_clip,
        )
        return {eid: night_result for eid in epochs}

    # per_image (default for median_zp_per_image) and fixed (same as per_image without preset coeffs)
    results: dict[str, CalibrationResult] = {}
    for epoch_id, data in epochs.items():
        mask = calibrator_candidate_mask(data, filters, config)
        results[epoch_id] = fit_median_zero_point_epoch(
            data,
            epoch_id,
            filters,
            mask,
            color_index_filters=color_indices,
            min_comparisons=min_comparisons,
            sigma_clip=config.fit_sigma_clip,
        )
    return results


def _extinction_corrected(data: Table, filters: list[str], corrector) -> Table:
    if not extinction_airmass_ready(data, filters, "airmass"):
        return data
    return corrector.correct(
        data,
        fallback_airmass_col="airmass",
        mag_col_prefix="mag_",
        output_prefix="mag_",
        filters=filters,
        inplace=False,
        catalog_color_prefix="mag_std_",
    )


__all__ = ["fit_epochs"]
