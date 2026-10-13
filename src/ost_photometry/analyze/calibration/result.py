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
    #: Intrinsic scatter of the calibrators (mag) that makes χ²_ν = 1 with
    #: instrumental and catalog errors (catalog systematics, colour-term
    #: mismatch); part of every calibrator's weight.
    sigma_int: float = 0.0
    #: χ²_ν of the fit with ``sigma_int`` = 0 (1 or below: errors explain it).
    chi2_nu: float = float("nan")
    #: Factor applied to the catalog errors: below 1 when they over-explained
    #: the calibrator scatter (χ²_ν < 1, enough calibrators).
    catalog_error_scale: float = 1.0
    #: Weighted mean colour of the calibrators; ZP and T are nearly
    #: uncorrelated around it.
    color_ref: float = 0.0
    #: Extra systematic error of this epoch's zero point (mag) that the fit
    #: does not contain, e.g. the scatter of epoch zero points about a
    #: per-night value, or σ_k·(X - X_ref) of a shared extinction.
    zero_point_sys: float = 0.0
    #: ``few_calibrators`` (n < 5: errors t-inflated), ``fixed_color_term``.
    quality_flag: str = "OK"
    #: 2×2 covariance of (ZP, T): photometric part (independent between
    #: epochs) and catalog part (catalog errors and σ_int of the same
    #: calibrators, common to all epochs). ``None`` when unknown.
    cov_phot: np.ndarray | None = None
    cov_cat: np.ndarray | None = None
    #: Mean airmass of the epochs that share this zero point (per night /
    #: ensemble): the extinction error contributes σ_k·(X - airmass_ref).
    airmass_ref: float | None = None

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


CALIBRATION_EPOCHS_FILENAME = "calibration_epochs.ecsv"


def calibration_epochs_table(results: dict[str, CalibrationResult]):
    """One row per epoch and filter with the fitted zero point / colour term.

    ``zero_point_err`` / ``color_term_err`` / ``cov_tz`` are the full fit
    errors; ``zero_point_err_phot`` / ``zero_point_err_cat`` split the zero
    point error into the part that is independent from epoch to epoch
    (photon, sky, scintillation of the calibrators) and the part common to
    all epochs with the same calibrators (catalog errors, σ_int).
    """
    import numpy as np
    from astropy.table import Table

    rows = []
    for epoch_id in sorted(results, key=str):
        res = results[epoch_id]
        for f, tc in sorted(res.transformation.items()):
            phot = None if tc.cov_phot is None else float(np.sqrt(max(tc.cov_phot[0, 0], 0.0)))
            cat = None if tc.cov_cat is None else float(np.sqrt(max(tc.cov_cat[0, 0], 0.0)))
            rows.append({
                "epoch_id": str(epoch_id),
                "calibration_id": str(res.identifier),
                "filter": str(f),
                "color_index": f"{tc.color_index_filters[0]}-{tc.color_index_filters[1]}",
                "zero_point": float(tc.zero_point),
                "zero_point_err": float(tc.zero_point_err),
                "zero_point_err_phot": np.nan if phot is None else phot,
                "zero_point_err_cat": np.nan if cat is None else cat,
                "zero_point_sys": float(tc.zero_point_sys),
                "color_term": float(tc.color_term),
                "color_term_err": float(tc.color_term_err),
                "cov_tz": float(tc.cov_tz),
                "color_ref": float(tc.color_ref),
                "sigma_int": float(tc.sigma_int),
                "chi2_nu": float(tc.chi2_nu),
                "catalog_error_scale": float(tc.catalog_error_scale),
                "n_stars_used": int(tc.n_stars_used),
                "rms_residual": float(tc.rms_residual),
                "airmass_ref": np.nan if tc.airmass_ref is None else float(tc.airmass_ref),
                "quality_flag": str(tc.quality_flag),
            })
    return Table(rows=rows) if rows else Table()


__all__ = [
    "CALIBRATION_EPOCHS_FILENAME",
    "CalibrationResult",
    "TransformationCoefficients",
    "calibration_epochs_table",
]
