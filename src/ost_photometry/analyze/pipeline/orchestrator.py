"""Pipeline orchestrator."""

from .config import PipelineConfig
from .context import AnalysisContext
from .base import PipelineStep
from .steps import (
    WcsStep,
    ExtractionStep,
    CorrelationIntraStep,
    CorrelationInterStep,
    ExtinctionFitStep,
    CalibrationStep,
    PostProcessClusterGaiaStep,
    PostProcessProperMotionStep,
    PostProcessRegionStep,
    PostProcessMagnitudeConvertStep,
    PostProcessSaveMagnitudesStep,
    DeriveLimitingMagnitudeStep,
    HipsReferenceSubtractStep,
    LightCurveStep,
)


class AnalysisPipeline:
    """Orchestrator that runs pipeline steps in sequence."""

    def __init__(
        self,
        steps: list[PipelineStep] | None = None,
        config: PipelineConfig | None = None,
    ):
        self.steps = steps or self._default_steps()
        self.config = config or PipelineConfig()

    def _default_steps(self) -> list[PipelineStep]:
        return [
            WcsStep(),
            ExtractionStep(),
            CorrelationIntraStep(),
            CorrelationInterStep(),
            ExtinctionFitStep(),
            CalibrationStep(),
            PostProcessRegionStep(),
            PostProcessClusterGaiaStep(),
            PostProcessProperMotionStep(),
            PostProcessMagnitudeConvertStep(),
            PostProcessSaveMagnitudesStep(),
            DeriveLimitingMagnitudeStep(),
            HipsReferenceSubtractStep(),
            LightCurveStep(),
        ]

    def _validate_magnitude_config(self) -> None:
        from ..post_processing.magnitude_systems import (
            apply_target_filter_system_alias,
            validate_magnitude_output_request,
        )

        cfg = self.config
        if cfg.target_filter_system:
            ofs, oms = apply_target_filter_system_alias(cfg.target_filter_system)
            if ofs is not None and cfg.output_filter_set == "auto":
                cfg.output_filter_set = ofs
            if oms is not None and cfg.output_magnitude_system == "auto":
                cfg.output_magnitude_system = oms
        validate_magnitude_output_request(
            output_filter_set=cfg.output_filter_set,
            output_magnitude_system=cfg.output_magnitude_system,
        )

    def run(self, context: AnalysisContext) -> AnalysisContext:
        self._validate_magnitude_config()
        for step in self.steps:
            if step.skip(context, self.config):
                continue
            context = step.run(context, self.config)
        return context
