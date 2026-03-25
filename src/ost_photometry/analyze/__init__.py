"""Photometry analysis module: extraction, calibration, light curves."""

from .analyze import Observation, main_extract
from .extinction import (
    CoefficientMode,
    ExtinctionCorrector,
    ExtinctionOrder,
    calculate_airmass,
)
from .calibration_differential_catalog import (
    APASSCatalog,
    PhotometryCalibrator,
)
from .models import ImageSeries, ObjectOfInterest
from .pipeline import AnalysisContext, AnalysisPipeline, PipelineConfig

__all__ = [
    "APASSCatalog",
    "AnalysisContext",
    "AnalysisPipeline",
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
