"""Intra-filter correlation step (correlate_preserve_objects)."""

from ... import correlate
from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext


class CorrelationIntraStep(base.PipelineStep):
    """Correlate results within each filter's image series, preserving selected objects."""

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
        obs = context.require_observation()

        for filter_ in context.filter_list:
            image_series = context.image_series_dict[filter_]
            if len(image_series.image_list) <= 1:
                continue

            correlate.correlate_preserve_objects(
                obs,
                filter_,
                context.filter_list,
                max_pixel_between_objects=config.max_pixel_between_objects,
                ooi_correlation_strategy=config.ooi_correlation_strategy,
                cross_identification_limit=config.cross_identification_limit,
                reference_image_index=config.reference_image_index,
                n_allowed_non_detections_object=config.n_allowed_non_detections_object,
                expected_bad_image_fraction=config.expected_bad_image_fraction,
                protected_object_ids=config.protected_object_ids,
                protect_ooi=config.protect_ooi,
                protect_calibration_objects=config.protect_calibration_objects,
                calibration_source=config.calibration_source,
                calibration_catalog_mag_range=config.calibration_catalog_mag_range,
                vizier_dict=config.vizier_dict,
                path_calibration_file=config.path_calibration_file,
                plot_only_reference_starmap=not config.plots_for_all_images,
                correlation_method=config.correlation_method,
                separation_limit=config.separation_limit,
                use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
                file_type_plots=config.file_type_plots,
                verbose=config.verbose,
                duplicate_handling_object_identification=config.duplicate_handling_object_identification,
                plots_for_all_images=config.plots_for_all_images,
            )

        context.correlation_intra_done = True
        return context
