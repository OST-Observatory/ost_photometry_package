"""Post-process step (post_process_results, derive_limiting_magnitude)."""

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ... import utilities


class PostProcessStep(base.PipelineStep):
    """Restrict results, filter by Gaia, and derive limiting magnitude."""

    name = "post_process"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        return (
            config.skip_post_process
            or config.skip_calibration
            or config.calibration_module == "differential"
        )

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        obs = context._observation
        if obs is None or obs.calib_parameters is None:
            raise RuntimeError(
                "PostProcessStep requires context._observation with calib_parameters"
            )

        calibration_filters = obs.calib_parameters.column_names
        _, usable_filter_combinations = (
            utilities.find_filter_for_magnitude_transformation(
                context.filter_list,
                calibration_filters,
            )
        )

        for filter_combination in usable_filter_combinations:
            utilities.post_process_results(
                obs,
                filter_combination,
                id_object=config.object_id,
                extraction_method=config.photometry_extraction_method,
                extract_only_circular_region=config.extract_only_circular_region,
                region_radius=config.region_radius,
                identify_cluster_gaia_data=config.identify_cluster_gaia_data,
                clean_objects_using_proper_motion=config.clean_objs_using_pm,
                max_distance_cluster=config.max_distance_cluster,
                find_cluster_para_set=config.find_cluster_para_set,
                convert_magnitudes=config.convert_magnitudes,
                target_filter_system=config.target_filter_system,
                distribution_samples=config.distribution_samples,
                use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
                file_type_plots=config.file_type_plots,
            )

            utilities.derive_limiting_magnitude(
                obs,
                filter_combination,
                config.reference_image_id,
                aperture_radius=config.aperture_radius,
                radii_unit=config.radii_unit,
                file_type_plots=config.file_type_plots,
                use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
            )

        return context
