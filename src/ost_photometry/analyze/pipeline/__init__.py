"""Modular pipeline for photometry analysis."""

from .config import (
    CALIBRATION_PRESETS,
    CalibrationConfig,
    CorrelationConfig,
    DiagnosticPlots,
    ExtractionConfig,
    ExtinctionConfig,
    HipsConfig,
    LightCurveConfig,
    PipelineConfig,
    PostProcessConfig,
    WcsConfig,
)
from .context import AnalysisContext
from .base import PipelineStep
from .orchestrator import AnalysisPipeline
from .bridge import (
    observation_to_calibration_epochs,
    observation_to_epoch_tables,
)

__all__ = [
    "AnalysisContext",
    "AnalysisPipeline",
    "CALIBRATION_PRESETS",
    "CalibrationConfig",
    "CorrelationConfig",
    "DiagnosticPlots",
    "ExtractionConfig",
    "ExtinctionConfig",
    "HipsConfig",
    "LightCurveConfig",
    "PipelineConfig",
    "PipelineStep",
    "PostProcessConfig",
    "WcsConfig",
    "observation_to_calibration_epochs",
    "observation_to_epoch_tables",
]
