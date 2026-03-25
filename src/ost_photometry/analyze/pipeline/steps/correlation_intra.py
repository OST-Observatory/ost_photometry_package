"""Intra-filter correlation step (correlate_preserve_variable)."""

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ... import correlate


class CorrelationIntraStep(base.PipelineStep):
    """Correlate results within each filter's image series, preserving variable objects."""

    name = "correlation_intra"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        if config.skip_correlation_intra:
            return True
        # Only 1 image per filter: no intra-correlation needed
        if not context.has_multiple_images_per_filter():
            return True
        return False

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        obs = context._observation
        if obs is None:
            raise RuntimeError(
                "CorrelationIntraStep requires context._observation (run from Observation.run_pipeline)"
            )

        for filter_ in context.filter_list:
            image_series = context.image_series_dict[filter_]
            if len(image_series.image_list) <= 1:
                continue

            correlate.correlate_preserve_variable(
                obs,
                filter_,
                max_pixel_between_objects=config.max_pixel_between_objects,
                own_correlation_option=config.own_correlation_option,
                cross_identification_limit=config.cross_identification_limit,
                reference_image_id=config.reference_image_id,
                n_allowed_non_detections_object=config.n_allowed_non_detections_object,
                expected_bad_image_fraction=config.expected_bad_image_fraction,
                protect_reference_obj=config.protect_reference_obj,
                verbose=config.verbose,
                duplicate_handling_object_identification=config.duplicate_handling_object_identification,
                plots_for_all_images=config.plots_for_all_images,
                correlation_method=config.correlation_method,
                separation_limit=config.separation_limit,
                use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
                file_type_plots=config.file_type_plots,
            )

        context.correlation_intra_done = True
        return context
