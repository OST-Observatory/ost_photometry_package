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
    CalibrationDataStep,
    CalibrationApplyStep,
    CalibrationDifferentialStep,
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
            CalibrationDataStep(),
            CalibrationDifferentialStep(),
            CalibrationApplyStep(),
            PostProcessRegionStep(),
            PostProcessClusterGaiaStep(),
            PostProcessProperMotionStep(),
            PostProcessMagnitudeConvertStep(),
            PostProcessSaveMagnitudesStep(),
            DeriveLimitingMagnitudeStep(),
            HipsReferenceSubtractStep(),
            LightCurveStep(),
        ]

    def run(self, context: AnalysisContext) -> AnalysisContext:
        for step in self.steps:
            if step.skip(context, self.config):
                continue
            context = step.run(context, self.config)
        return context
