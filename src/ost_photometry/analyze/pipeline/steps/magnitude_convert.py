"""Convert calibrated magnitudes to another photometric system / filter set."""

from __future__ import annotations

from ...post_processing.magnitude_convert import (
    apply_magnitude_system_convert_on_observation,
)
from ...post_processing.magnitude_systems import validate_magnitude_output_request
from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext


class PostProcessMagnitudeConvertStep(base.PipelineStep):
    """Map magnitudes to ``output_filter_set`` / ``output_magnitude_system``."""

    name = "magnitude_system_convert"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_calibration:
            return True
        if config.skip_magnitude_convert_step:
            return True
        # Always run at least to annotate meta when calibration produced a table;
        # numerical conversion only if convert_magnitudes is True.
        return context.table_magnitudes is None and (
            context.observation is None
            or getattr(context.observation, "table_magnitudes", None) is None
        )

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        validate_magnitude_output_request(
            output_filter_set=config.output_filter_set,
            output_magnitude_system=config.output_magnitude_system,
        )
        obs = context.require_observation()
        if obs is None:
            raise RuntimeError(
                "PostProcessMagnitudeConvertStep requires context._observation"
            )
        apply_magnitude_system_convert_on_observation(
            obs,
            target_filter_system=config.target_filter_system,
            output_filter_set=config.output_filter_set,
            output_magnitude_system=config.output_magnitude_system,
            convert_magnitudes=config.convert_magnitudes,
            distribution_samples=config.distribution_samples,
            calibration_source=config.calibration_source,
        )
        context.table_magnitudes = obs.table_magnitudes
        return context
