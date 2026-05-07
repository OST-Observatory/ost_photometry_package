"""Inter-filter correlation step (correlate_image_series)."""

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ... import correlate


class CorrelationInterStep(base.PipelineStep):
    """Correlate star lists across filters to find common objects."""

    name = "correlation_inter"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        if config.skip_correlation_inter:
            return True
        if len(context.filter_list) < 2:
            return True
        return False

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        from .... import terminal_output
        obs = context._observation
        if obs is None:
            raise RuntimeError(
                "CorrelationInterStep requires context._observation (run from Observation.run_pipeline)"
            )

        terminal_output.print_to_terminal(
            "Correlate and calibrate image series",
            style_name="HEADER",
        )

        correlate.correlate_image_series(
            obs,
            context.filter_list,
            max_pixel_between_objects=config.max_pixel_between_objects,
            ooi_correlation_strategy=config.ooi_correlation_strategy,
            correlation_method=config.correlation_method,
            separation_limit=config.separation_limit,
            file_type_plots=config.file_type_plots,
            duplicate_handling_object_identification=config.duplicate_handling_object_identification,
            debug_verify_ooi_global_ids=config.debug_verify_ooi_global_ids,
        )

        if len(context.filter_list) > 1:
            from ...utilities import prepare_and_plot_starmap_from_observation

            prepare_and_plot_starmap_from_observation(
                obs,
                context.filter_list,
                use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
                file_type_plots=config.file_type_plots,
            )

        from ...diagnostic_plot_hooks import run_diagnostic_plots_phase

        run_diagnostic_plots_phase(context, config, "correlation_inter")

        context.correlation_inter_done = True
        return context
