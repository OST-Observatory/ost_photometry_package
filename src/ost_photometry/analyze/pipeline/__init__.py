"""Modular pipeline for photometry analysis."""

from .config import PipelineConfig
from .context import AnalysisContext
from .base import PipelineStep
from .orchestrator import AnalysisPipeline
from .bridge import (
    observation_to_calibration_epochs,
    observation_to_epoch_tables,
    observation_to_frame_tables,
)

__all__ = [
    "AnalysisContext",
    "AnalysisPipeline",
    "PipelineConfig",
    "PipelineStep",
    "observation_to_calibration_epochs",
    "observation_to_epoch_tables",
    "observation_to_frame_tables",
]
