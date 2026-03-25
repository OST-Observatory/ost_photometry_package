"""Calibration data step (derive_calibration)."""

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ... import calibration_data


class CalibrationDataStep(base.PipelineStep):
    """Load calibration information (APASS etc.)."""

    name = "calibration_data"

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
        if obs is None:
            raise RuntimeError(
                "CalibrationDataStep requires context._observation"
            )

        calibration_data.derive_calibration(
            obs,
            context.filter_list,
            calibration_method=config.calibration_method,
            max_pixel_between_objects=config.max_pixel_between_objects,
            own_correlation_option=config.own_correlation_option,
            vizier_dict=config.vizier_dict,
            path_calibration_file=config.path_calibration_file,
            magnitude_range=config.magnitude_range,
            region_to_select_calibration_stars=config.region_to_select_calibration_stars,
            file_type_plots=config.file_type_plots,
            use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
        )

        return context
