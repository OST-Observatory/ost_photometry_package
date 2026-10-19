"""Linear-fit calibration backend (PhotometryCalibrator)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ...extinction import CoefficientMode, ExtinctionCoefficients, ExtinctionOrder
from ...extinction_io import (
    finalize_pipeline_extinction_coefficients,
    resolve_pipeline_extinction_coefficients,
    resolve_pipeline_extinction_order,
)
from ..calibrator import PhotometryCalibrator
from ..quality import calibrator_candidate_mask
from ..result import CalibrationResult
from ..statistics import parse_fixed_color_terms

if TYPE_CHECKING:
    from astropy.coordinates import EarthLocation
    from astropy.table import Table

    from ...pipeline.config import PipelineConfig


def _coefficient_mode(grouping: str) -> CoefficientMode:
    return {
        "per_image": CoefficientMode.PER_IMAGE,
        "per_night": CoefficientMode.PER_NIGHT,
        "fixed": CoefficientMode.FIXED,
        "ensemble": CoefficientMode.ENSEMBLE,
    }.get(grouping, CoefficientMode.PER_NIGHT)


def build_calibrator(
    config: PipelineConfig,
    *,
    observatory_location: EarthLocation | None = None,
    color_indices: dict[str, tuple[str, str]] | None = None,
    extinction_coefficients: dict[str, ExtinctionCoefficients] | None = None,
) -> PhotometryCalibrator:
    grouping = config.calibration_grouping
    ext_order = resolve_pipeline_extinction_order(config)
    raw = resolve_pipeline_extinction_coefficients(
        config, fitted=extinction_coefficients
    )
    coeffs = finalize_pipeline_extinction_coefficients(config, raw)
    calibrator = PhotometryCalibrator(
        mode=_coefficient_mode(grouping),
        extinction_order=ext_order,
        observatory_location=observatory_location or config.observatory_location,
        color_indices=color_indices,
        extinction_coefficients=coeffs,
        match_radius=config.calibration_match_radius,
    )
    if grouping == "fixed":
        calibrator.set_fixed_color_terms(
            parse_fixed_color_terms(config.transformation_coefficients_dict)
        )
    return calibrator


def _apply_second_order_after_fit(
    calibrator: PhotometryCalibrator,
    config: PipelineConfig,
    fitted: dict | None = None,
) -> None:
    """Merge tabulated / user k″ onto fitted k′ when ``extinction_order="second"``.

    A k′ fitted from the calibrators without the k″ term holds
    ``k″·C̄`` (C̄ the mean standard colour of the calibrators); adding the
    k″ term on top would count it twice, so ``k′ ← k′ − k″·C̄`` for the
    filters in ``fitted``.
    """
    if resolve_pipeline_extinction_order(config) != ExtinctionOrder.SECOND:
        # Still allow explicit k_second overrides on FIRST-order runs (no-op for correct()).
        if not getattr(config, "k_second", None):
            return
    finalized = finalize_pipeline_extinction_coefficients(
        config, dict(calibrator.extinction.coefficients)
    )
    if not finalized:
        return
    if resolve_pipeline_extinction_order(config) == ExtinctionOrder.SECOND:
        for filter_ in fitted or {}:
            coeff = finalized.get(filter_)
            if coeff is None or coeff.k_second == 0:
                continue
            c_mean = _mean_catalog_color(calibrator.epochs, coeff.color_filter_1, coeff.color_filter_2)
            if np.isfinite(c_mean):
                finalized[filter_] = replace(coeff, k_prime=coeff.k_prime - coeff.k_second * c_mean)
    calibrator.extinction.coefficients.update(finalized)
    calibrator.extinction.order = resolve_pipeline_extinction_order(config)


def _mean_catalog_color(epochs: dict, filter_1: str, filter_2: str) -> float:
    """Median catalog colour of the catalog-matched stars over all epochs."""
    values = []
    for tbl in epochs.values():
        c1, c2 = f"mag_std_{filter_1}", f"mag_std_{filter_2}"
        if c1 in tbl.colnames and c2 in tbl.colnames:
            color = np.asarray(tbl[c1], dtype=float) - np.asarray(tbl[c2], dtype=float)
            values.append(color[np.isfinite(color)])
    values = np.concatenate(values) if values else np.array([])
    return float(np.median(values)) if values.size else float("nan")


def fit_epochs(
    calibrator: PhotometryCalibrator,
    epochs: dict[str, Table],
    filters: list[str],
    config: PipelineConfig,
    *,
    output_dir: str | None = None,
    file_type: str = "pdf",
    calibration_summary_x_jd: dict[str, float] | None = None,
) -> dict[str, CalibrationResult]:
    """Populate calibrator epochs and fit T/ZP via PhotometryCalibrator."""
    for epoch_id, tbl in epochs.items():
        calibrator.epochs[epoch_id] = tbl

    if config.extinction_mode == "from_comparison_stars":
        fitted = calibrator.fit_extinction_from_epochs(
            output_dir=output_dir,
            file_type=file_type,
        )
        _apply_second_order_after_fit(calibrator, config, fitted)

    def _comparison_selector(table):
        return calibrator_candidate_mask(table, filters, config)

    calibrator.fit_transformation_parameters(
        filters=filters,
        comparison_selector=_comparison_selector,
        determine_color_terms=True,
        min_comparisons=5,
        sigma_clip=config.fit_sigma_clip,
        output_dir=output_dir,
        file_type=file_type,
        per_image_rolling_median_color_term=config.per_image_rolling_median_color_term,
        per_image_rolling_median_zero_point=config.per_image_rolling_median_zero_point,
        per_image_rolling_mean_color_term=config.per_image_rolling_mean_color_term,
        per_image_rolling_mean_zero_point=config.per_image_rolling_mean_zero_point,
        per_image_rolling_window=config.per_image_rolling_window,
        calibration_summary_x_jd=calibration_summary_x_jd,
        calibration_summary_use_jd_x=config.calibration_summary_use_jd_x,
        color_term_fit=config.color_term_fit,
    )
    return dict(calibrator.calib_parameters)


__all__ = ["build_calibrator", "fit_epochs"]
