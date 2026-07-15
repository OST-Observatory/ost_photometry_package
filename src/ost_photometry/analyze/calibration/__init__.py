"""Unified calibration engine, backends, and legacy re-exports."""

from .engine import CalibrationEngine, prepare_calibration_check_plots
from .mk_calib import (
    FieldTransformationRecord,
    TransformCoefficient,
    calibrate_mk_calib_filter_pair,
    load_field_transformation_record,
    write_trans_para_table,
)
from .result import CalibrationResult, TransformationCoefficients
from .second_order_extinction import SecondOrderFitResult, run_second_order_campaign
from ._legacy import (
    apply_calibration,
    apply_magnitude_transformation,
    calibrate_magnitudes_transformation,
    calibrate_magnitudes_zero_point,
    calibrate_simple,
    determine_transformation_coefficients,
    find_best_comparison_image_second_filter,
    flux_normalization_image_series,
    prepare_zero_point,
    quasi_flux_calibration_image_series,
    transformation_core,
)

__all__ = [
    "CalibrationEngine",
    "CalibrationResult",
    "FieldTransformationRecord",
    "SecondOrderFitResult",
    "TransformCoefficient",
    "TransformationCoefficients",
    "calibrate_mk_calib_filter_pair",
    "load_field_transformation_record",
    "run_second_order_campaign",
    "write_trans_para_table",
    "apply_calibration",
    "apply_magnitude_transformation",
    "calibrate_magnitudes_transformation",
    "calibrate_magnitudes_zero_point",
    "calibrate_simple",
    "determine_transformation_coefficients",
    "find_best_comparison_image_second_filter",
    "flux_normalization_image_series",
    "prepare_calibration_check_plots",
    "prepare_zero_point",
    "quasi_flux_calibration_image_series",
    "transformation_core",
]
