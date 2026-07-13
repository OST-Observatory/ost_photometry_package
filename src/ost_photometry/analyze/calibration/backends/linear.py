"""Linear-fit calibration backend (PhotometryCalibrator)."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING, Optional

from ...differential_photometry import PhotometryCalibrator
from ...extinction import CoefficientMode, ExtinctionCoefficients, ExtinctionOrder
from ..result import CalibrationResult

if TYPE_CHECKING:
    from astropy.table import Table
    from astropy.coordinates import EarthLocation

    from ...pipeline.config import PipelineConfig


def _extinction_order(mode: str) -> ExtinctionOrder:
    if mode == "none":
        return ExtinctionOrder.NONE
    return ExtinctionOrder.FIRST


def _coefficient_mode(grouping: str) -> CoefficientMode:
    return {
        "per_image": CoefficientMode.PER_IMAGE,
        "per_night": CoefficientMode.PER_NIGHT,
        "fixed": CoefficientMode.FIXED,
        "ensemble": CoefficientMode.ENSEMBLE,
    }.get(grouping, CoefficientMode.PER_NIGHT)


def build_calibrator(
    config: "PipelineConfig",
    *,
    observatory_location: "EarthLocation | None" = None,
    color_indices: dict[str, tuple[str, str]] | None = None,
    extinction_coefficients: Optional[Dict[str, ExtinctionCoefficients]] = None,
) -> PhotometryCalibrator:
    grouping = config.calibration_grouping
    ext_order = _extinction_order(config.extinction_mode)
    coeffs = None
    if config.extinction_mode == "from_value_airmass":
        coeffs = extinction_coefficients
    return PhotometryCalibrator(
        mode=_coefficient_mode(grouping),
        extinction_order=ext_order,
        observatory_location=observatory_location or config.observatory_location,
        color_indices=color_indices,
        extinction_coefficients=coeffs,
    )


def fit_epochs(
    calibrator: PhotometryCalibrator,
    epochs: Dict[str, "Table"],
    filters: List[str],
    config: "PipelineConfig",
    *,
    output_dir: str | None = None,
    file_type: str = "pdf",
    calibration_summary_x_jd: dict[str, float] | None = None,
) -> Dict[str, CalibrationResult]:
    """Populate calibrator epochs and fit T/ZP via PhotometryCalibrator."""
    for epoch_id, tbl in epochs.items():
        calibrator.epochs[epoch_id] = tbl

    if config.extinction_mode == "from_comparison_stars":
        calibrator.fit_extinction_from_epochs(
            output_dir=output_dir,
            file_type=file_type,
        )

    calibrator.fit_transformation_parameters(
        filters=filters,
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
