"""Photometry analysis module: extraction, calibration, light curves."""

from .observation import Observation
from . import analyze as _analyze_module  # noqa: F401 — registers Observation.run_pipeline
from .extraction import main_extract
from .extinction import (
    CoefficientMode,
    ExtinctionCorrector,
    ExtinctionOrder,
    calculate_airmass,
)
from .differential_photometry import PhotometryCalibrator
from .models import ImageSeries, ObjectOfInterest
from .pipeline import AnalysisContext, AnalysisPipeline, DiagnosticPlots, PipelineConfig

__all__ = [
    "AnalysisContext",
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
