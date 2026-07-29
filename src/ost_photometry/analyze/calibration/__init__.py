"""Unified calibration engine, backends, and public re-exports."""

from __future__ import annotations

from .engine import CalibrationEngine, prepare_calibration_check_plots
from .flux_normalize import (
    flux_normalization_flux_distribution,
    flux_normalization_image_series,
    quasi_flux_calibration_flux_arrays,
    quasi_flux_calibration_image_series,
)
from .mk_calib import (
    FieldTransformationRecord,
    TransformCoefficient,
    calibrate_mk_calib_filter_pair,
    load_field_transformation_record,
    write_trans_para_table,
)
from .result import CalibrationResult, TransformationCoefficients
from .second_order_extinction import SecondOrderFitResult, run_second_order_campaign

__all__ = [
    "CalibrationEngine",
    "CalibrationResult",
    "FieldTransformationRecord",
    "SecondOrderFitResult",
    "TransformCoefficient",
    "TransformationCoefficients",
    "calibrate_mk_calib_filter_pair",
    "flux_normalization_flux_distribution",
    "flux_normalization_image_series",
    "load_field_transformation_record",
    "prepare_calibration_check_plots",
    "quasi_flux_calibration_flux_arrays",
    "quasi_flux_calibration_image_series",
    "run_second_order_campaign",
    "write_trans_para_table",
]
