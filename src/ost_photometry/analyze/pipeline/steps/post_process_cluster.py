"""Cluster-field post-processing split into ordered pipeline steps."""

from __future__ import annotations

import warnings

from ... import utilities
from ...post_processing.cluster_field import (
    apply_cluster_field_phase,
    write_post_processed_cluster_field_table,
)
from ...post_processing.imaging import imaging_context_from_image_series
from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext


def _sync_table_to_context(context: AnalysisContext, obs) -> None:
    context.table_magnitudes = obs.table_magnitudes


def _reference_image_series(context: AnalysisContext, obs) -> object:
    """Reference band for WCS / Gaia FoV (first pipeline filter)."""
    fl = context.filter_list
    if not fl:
        raise RuntimeError("context.filter_list is empty")
    k0 = fl[0]
    if k0 not in obs.image_series_dict:
        raise RuntimeError(
            f"Reference filter {k0!r} missing from observation.image_series_dict"
        )
    return obs.image_series_dict[k0]


def _usable_filter_combinations(
    context: AnalysisContext, config: PipelineConfig, obs
) -> list[list[str]]:
    tbl = obs.table_magnitudes
    if tbl is None or len(tbl) == 0:
        raise RuntimeError(
            "Post-process steps require observation.table_magnitudes "
            "after calibration."
        )
    calibration_filters = utilities.transformation_keys_for_table_magnitudes(
        tbl, context.filter_list
    )

    _, usable = utilities.find_filter_for_magnitude_transformation(
        context.filter_list,
        calibration_filters,
    )
    if len(usable) > 1:
        warnings.warn(
            "Multiple usable filter combinations: cluster phases run once using the "
            "reference image series; only the save step loops per combination. For "
            "per-combo ordering, call post_process_cluster_field in a loop.",
            UserWarning,
            stacklevel=2,
        )
    return usable


def _phase_kwargs(config: PipelineConfig) -> dict:
    return dict(
        region_radius=config.region_radius,
        max_distance_cluster=config.max_distance_cluster,
        find_cluster_para_set=config.find_cluster_para_set,
        cluster_selection_id=config.cluster_selection_id,
        cluster_ruwe_max=config.cluster_ruwe_max,
        cluster_plx_snr_min=config.cluster_plx_snr_min,
        cluster_pmem_min=config.cluster_pmem_min,
        cluster_membership_method=config.cluster_membership_method,
        cluster_component_id=config.cluster_component_id,
        use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
        file_type_plots=config.file_type_plots,
    )


class PostProcessRegionStep(base.PipelineStep):
    """Circular region cut around objects of interest (first epoch for multi-epoch tables)."""

    name = "post_process_region"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_calibration:
            return True
        if config.skip_cluster_region_step:
            return True
        return not config.extract_only_circular_region

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        obs = context.require_observation()
        if obs is None:
            raise RuntimeError("PostProcessRegionStep requires context._observation")
        img = _reference_image_series(context, obs)
        kw = {
            **_phase_kwargs(config),
            "plot_context": imaging_context_from_image_series(img),
        }
        _usable_filter_combinations(context, config, obs)
        apply_cluster_field_phase(obs, "region", **kw)
        _sync_table_to_context(context, obs)
        return context


class PostProcessClusterGaiaStep(base.PipelineStep):
    """Identify cluster members in Gaia (μ_α*, μ_δ, ϖ)."""

    name = "post_process_cluster_gaia"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_calibration:
            return True
        if config.skip_cluster_gaia_step:
            return True
        return not config.identify_cluster_gaia_data

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        obs = context.require_observation()
        if obs is None:
            raise RuntimeError("PostProcessClusterGaiaStep requires context._observation")
        img = _reference_image_series(context, obs)
        kw = {
            **_phase_kwargs(config),
            "plot_context": imaging_context_from_image_series(img),
        }
        _usable_filter_combinations(context, config, obs)
        apply_cluster_field_phase(obs, "gaia", **kw)
        _sync_table_to_context(context, obs)
        return context


class PostProcessProperMotionStep(base.PipelineStep):
    """Filter sources using Gaia proper motions."""

    name = "post_process_proper_motion"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_calibration:
            return True
        if config.skip_cluster_pm_step:
            return True
        return not config.clean_objs_using_pm

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        obs = context.require_observation()
        if obs is None:
            raise RuntimeError("PostProcessProperMotionStep requires context._observation")
        img = _reference_image_series(context, obs)
        kw = {
            **_phase_kwargs(config),
            "plot_context": imaging_context_from_image_series(img),
        }
        _usable_filter_combinations(context, config, obs)
        apply_cluster_field_phase(obs, "pm", **kw)
        _sync_table_to_context(context, obs)
        return context


class PostProcessSaveMagnitudesStep(base.PipelineStep):
    """Write post-processed ``table_magnitudes`` as ECSV (per usable filter combination)."""

    name = "post_process_save_magnitudes"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_calibration:
            return True
        return config.skip_save_post_processed_magnitudes

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        obs = context.require_observation()
        if obs is None:
            raise RuntimeError(
                "PostProcessSaveMagnitudesStep requires context._observation"
            )
        for fc in _usable_filter_combinations(context, config, obs):
            write_post_processed_cluster_field_table(
                obs,
                fc,
                object_id=config.object_id,
                extraction_method=config.photometry_extraction_method,
            )
        _sync_table_to_context(context, obs)
        return context
