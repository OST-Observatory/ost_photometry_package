"""Median-ZP calibration backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...extinction import extinction_airmass_ready
from ..quality import calibrator_candidate_mask
from ..result import CalibrationResult
from ..statistics import (
    average_epochs_per_star,
    mark_shared_zero_point,
    parse_fixed_color_terms,
)
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
    if extinction_corrector is not None and getattr(config, "extinction_mode", None) == "from_comparison_stars":
        #   k from the comparison stars, as on the linear path (before, the
        #   tabulated values were used without a word). The corrector is the
        #   one that applies the result, so the fitted k is used there too.
        from ...extinction import fit_extinction_from_comparison_stars

        fitted = fit_extinction_from_comparison_stars(epochs)
        if fitted:
            extinction_corrector.coefficients.update(fitted)
        else:
            import warnings

            from ...warnings_types import OstPhotometryAnalyzeWarning

            warnings.warn(
                "extinction_mode='from_comparison_stars': no coefficient could be fitted "
                "(airmass range or comparison stars); tabulated values are used.",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )
    if extinction_corrector is not None:
        epochs = {
            eid: _extinction_corrected(tbl, filters, extinction_corrector)
            for eid, tbl in epochs.items()
        }

    if grouping == "ensemble":
        #   One point per star (mean over the already extinction-corrected
        #   epochs): each star's catalog error counts once.
        combined = average_epochs_per_star(list(epochs.values()), filters)
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
        mark_shared_zero_point(result, list(epochs.values()), filters)
        return {eid: result for eid in epochs}

    if grouping == "per_night":
        #   Per-epoch fits combined like the linear path: the catalog part
        #   does not average down, the epochs' extra scatter (transparency)
        #   is kept as zero_point_sys.
        from ..photometer import DifferentialPhotometer

        per_epoch = []
        for epoch_id, data in epochs.items():
            mask = calibrator_candidate_mask(data, filters, config)
            per_epoch.append(
                fit_median_zero_point_epoch(
                    data,
                    epoch_id,
                    filters,
                    mask,
                    color_index_filters=color_indices,
                    min_comparisons=min_comparisons,
                    sigma_clip=config.fit_sigma_clip,
                )
            )
        night = CalibrationResult(
            identifier="night_combined",
            transformation=DifferentialPhotometer().combine_epoch_calibration_results(
                per_epoch, filters
            ),
        )
        mark_shared_zero_point(night, list(epochs.values()), filters)
        return {eid: night for eid in epochs}

    fixed = (
        parse_fixed_color_terms(config.transformation_coefficients_dict)
        if grouping == "fixed"
        else None
    )
    if grouping == "fixed" and not fixed:
        raise ValueError(
            "calibration_grouping='fixed' needs transformation_coefficients_dict "
            "with the colour terms"
        )

    # per_image (default for median_zp_per_image) and fixed (known colour term)
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
            fixed_color_terms=fixed,
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
