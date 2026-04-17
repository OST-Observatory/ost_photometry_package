"""Modular pipeline for photometry analysis."""

from .config import DiagnosticPlots, PipelineConfig
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
    "DiagnosticPlots",
    "PipelineConfig",
    "PipelineStep",
    "observation_to_calibration_epochs",
    "observation_to_epoch_tables",
]
