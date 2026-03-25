"""Calibration apply step (apply_calibration)."""

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ... import calibration, utilities


class CalibrationApplyStep(base.PipelineStep):
    """Apply zero points and magnitude transformation for each filter combination."""

    name = "calibration_apply"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        return config.skip_calibration or config.calibration_module == "differential"

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        obs = context._observation
        if obs is None or obs.calib_parameters is None:
            raise RuntimeError(
                "CalibrationApplyStep requires context._observation with calib_parameters"
            )

        calibration_filters = obs.calib_parameters.column_names
        _, usable_filter_combinations = (
            utilities.find_filter_for_magnitude_transformation(
                context.filter_list,
                calibration_filters,
            )
        )

        for filter_combination in usable_filter_combinations:
            calibration.apply_calibration(
                obs,
                filter_combination,
                apply_transformation=config.apply_transformation,
                transformation_coefficients_dict=config.transformation_coefficients_dict,
                derive_transformation_coefficients=config.derive_transformation_coefficients,
                photometry_extraction_method=config.photometry_extraction_method,
                calculate_zero_point_statistic=config.calculate_zero_point_statistic,
                distribution_samples=config.distribution_samples,
                file_type_plots=config.file_type_plots,
                add_progress_bar=False,
            )

        return context
