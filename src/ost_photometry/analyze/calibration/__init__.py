"""Unified calibration engine, backends, and compatibility re-exports."""

from __future__ import annotations

import warnings
from typing import Any, Callable

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

_LEGACY_DEPRECATION = (
    "{name} is deprecated and retained only for compatibility. "
    "Prefer CalibrationEngine, derive_transform, or mk_calib APIs. "
    "See docs/TODO.md (analyze calibration/_legacy)."
)


def _deprecated_legacy(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name=name),
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = __name__
    return wrapper


# Image-based ZP/transform stack — no in-repo callers; deprecate on use.
from . import _legacy as _legacy_mod

apply_calibration = _deprecated_legacy(
    "apply_calibration", _legacy_mod.apply_calibration
)
apply_magnitude_transformation = _deprecated_legacy(
    "apply_magnitude_transformation", _legacy_mod.apply_magnitude_transformation
)
calibrate_magnitudes_transformation = _deprecated_legacy(
    "calibrate_magnitudes_transformation",
    _legacy_mod.calibrate_magnitudes_transformation,
)
calibrate_magnitudes_zero_point = _deprecated_legacy(
    "calibrate_magnitudes_zero_point",
    _legacy_mod.calibrate_magnitudes_zero_point,
)
calibrate_simple = _deprecated_legacy(
    "calibrate_simple", _legacy_mod.calibrate_simple
)
determine_transformation_coefficients = _deprecated_legacy(
    "determine_transformation_coefficients",
    _legacy_mod.determine_transformation_coefficients,
)
find_best_comparison_image_second_filter = _deprecated_legacy(
    "find_best_comparison_image_second_filter",
    _legacy_mod.find_best_comparison_image_second_filter,
)
prepare_zero_point = _deprecated_legacy(
    "prepare_zero_point", _legacy_mod.prepare_zero_point
)
transformation_core = _deprecated_legacy(
    "transformation_core", _legacy_mod.transformation_core
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
    "flux_normalization_flux_distribution",
    "flux_normalization_image_series",
    "prepare_calibration_check_plots",
    "prepare_zero_point",
    "quasi_flux_calibration_flux_arrays",
    "quasi_flux_calibration_image_series",
    "transformation_core",
]
