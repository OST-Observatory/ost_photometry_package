"""Unified calibration result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from astropy.time import Time

from ..extinction import ExtinctionCoefficients

if TYPE_CHECKING:
    import numpy as np


@dataclass
class TransformationCoefficients:
    """Transformation coefficients for one filter."""

    filter_name: str
    color_term: float
    color_term_err: float = 0.0
    zero_point: float = 0.0
    zero_point_err: float = 0.0
    #: Off-diagonal covariance ``Cov(T, ZP)`` from the same linear fit (mag²).
    cov_tz: float = 0.0
    color_index_filters: tuple[str, str] = ("B", "V")
    n_stars_used: int = 0
    rms_residual: float = 0.0

    def __repr__(self) -> str:
        ci = f"({self.color_index_filters[0]}-{self.color_index_filters[1]})"
        return (
            f"{self.filter_name}: T={self.color_term:.4f}±{self.color_term_err:.4f}, "
            f"ZP={self.zero_point:.4f}±{self.zero_point_err:.4f}, "
            f"cov(T,ZP)={self.cov_tz:.4g}, CI={ci}"
        )


@dataclass
class CalibrationResult:
    """Container for calibration results."""

    identifier: str
    timestamp: Time | None = None
    extinction: dict[str, ExtinctionCoefficients] = field(default_factory=dict)
    transformation: dict[str, TransformationCoefficients] = field(default_factory=dict)
    n_comparison_stars: int = 0
    quality_flag: str = "OK"
    notes: str = ""
    #: Per-filter boolean mask of stars used in the fit (aligned to the epoch table).
    calibrator_mask_by_filter: dict[str, np.ndarray] = field(default_factory=dict)


__all__ = ["CalibrationResult", "TransformationCoefficients"]
