"""Photometry analysis module: extraction, calibration, light curves."""

from .differential_photometry import PhotometryCalibrator
from .extinction import (
    CoefficientMode,
    ExtinctionCorrector,
    ExtinctionOrder,
    calculate_airmass,
)
from .extraction import main_extract
from .image import AnalysisImage
from .models import ImageSeries, ObjectOfInterest
from .observation import Observation
from .pipeline import AnalysisContext, AnalysisPipeline, DiagnosticPlots, PipelineConfig

# Registers Observation.run_pipeline; keep after extinction so that module can finish first.
from . import analyze as _analyze_module  # noqa: F401  # isort: skip

__all__ = [
    "AnalysisContext",
    "AnalysisImage",
    "AnalysisPipeline",
    "DiagnosticPlots",
    "CoefficientMode",
    "ExtinctionCorrector",
    "ExtinctionOrder",
    "PhotometryCalibrator",
    "ImageSeries",
    "ObjectOfInterest",
    "Observation",
    "PipelineConfig",
    "calculate_airmass",
    "main_extract",
]
