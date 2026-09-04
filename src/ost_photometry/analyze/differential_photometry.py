"""Deprecated shim — import from :mod:`ost_photometry.analyze.calibration` instead.

``PhotometryCalibrator`` lives in :mod:`.calibration.calibrator`;
``DifferentialPhotometer`` in :mod:`.calibration.photometer`.
"""

from __future__ import annotations

import warnings

from .calibration.calibrator import PhotometryCalibrator
from .calibration.photometer import DifferentialPhotometer

warnings.warn(
    "ost_photometry.analyze.differential_photometry is deprecated; "
    "import PhotometryCalibrator / DifferentialPhotometer from "
    "ost_photometry.analyze.calibration instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DifferentialPhotometer", "PhotometryCalibrator"]
