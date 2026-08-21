"""Modular pipeline for photometry analysis."""

from .base import PipelineStep
from .bridge import (
    observation_to_calibration_epochs,
    observation_to_epoch_tables,
)
from .config import (
    CALIBRATION_PRESETS,
    CalibrationConfig,
    CorrelationConfig,
    DiagnosticPlots,
    ExtinctionConfig,
    ExtractionConfig,
    HipsConfig,
    LightCurveConfig,
    PipelineConfig,
    PostProcessConfig,
    WcsConfig,
)
from .context import AnalysisContext
from .orchestrator import AnalysisPipeline

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
