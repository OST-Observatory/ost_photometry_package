"""Legacy calibration step: load catalog via calibration_sources, then derive_calibration."""

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
            calibration_source=config.calibration_source,
            max_pixel_between_objects=config.max_pixel_between_objects,
            ooi_correlation_strategy=config.ooi_correlation_strategy,
            vizier_dict=config.vizier_dict,
            path_calibration_file=config.path_calibration_file,
            calibration_catalog_mag_range=config.calibration_catalog_mag_range,
            region_to_select_calibration_stars=config.region_to_select_calibration_stars,
            file_type_plots=config.file_type_plots,
            use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
        )

        from ...diagnostic_plot_hooks import run_diagnostic_plots_phase

        run_diagnostic_plots_phase(context, config, "calibration_data")

        return context
