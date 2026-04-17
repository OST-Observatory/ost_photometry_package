"""Convert calibrated magnitudes to another photometric system (legacy tables)."""

from __future__ import annotations

from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext
from ...post_processing.magnitude_convert import (
    apply_magnitude_system_convert_on_observation,
)


class PostProcessMagnitudeConvertStep(base.PipelineStep):
    """Map magnitudes to ``target_filter_system`` (not part of cluster-field logic)."""

    name = "magnitude_system_convert"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_calibration:
            return True
        if config.calibration_module == "differential":
            return True
        if config.skip_magnitude_convert_step:
            return True
        return not config.convert_magnitudes

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        obs = context._observation
        if obs is None:
            raise RuntimeError(
                "PostProcessMagnitudeConvertStep requires context._observation"
            )
        apply_magnitude_system_convert_on_observation(
            obs,
            target_filter_system=config.target_filter_system,
            distribution_samples=config.distribution_samples,
        )
        context.table_magnitudes = obs.table_magnitudes
        return context
