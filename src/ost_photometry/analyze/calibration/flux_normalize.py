"""Quasi-ZP and per-object flux normalization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy import uncertainty as unc
from astropy.stats import sigma_clipped_stats

if TYPE_CHECKING:
    from ..models import ImageSeries


def _clipped_median(
    values: np.ndarray,
    axis: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """Sigma-clipped median; non-finite and non-positive values are ignored."""
    work = np.asarray(values, dtype=float).copy()
    work[~np.isfinite(work)] = 0.0
    work[work <= 0.0] = 0.0
    _, median, _stddev = sigma_clipped_stats(
        work,
        axis=axis,
        sigma=sigma,
        mask_value=0.0,
    )
    return np.asarray(median, dtype=float)


def _sanitize_std(flux: np.ndarray, flux_error: np.ndarray) -> np.ndarray:
    """Positive finite stddev for :func:`astropy.uncertainty.normal`."""
    fl = np.asarray(flux, dtype=float)
    err = np.asarray(flux_error, dtype=float)
    floor = np.maximum(np.abs(np.where(np.isfinite(fl), fl, 1.0)) * 1e-6, 1e-12)
    return np.where(np.isfinite(err) & (err > 0.0), err, floor)


def _ensemble_mask(
    finite: np.ndarray,
    min_ensemble_fraction: float,
) -> np.ndarray:
    """Columns detected often enough to trace the common mode."""
    n_epochs = int(finite.shape[0])
    n_det = np.asarray(finite, dtype=bool).sum(axis=0)
    frac = min(max(float(min_ensemble_fraction), 0.0), 1.0)
    min_det = max(int(np.ceil(frac * n_epochs)), 1)
    ensemble = n_det >= min_det
    if np.any(ensemble):
        return ensemble
    return n_det > 0


def _epoch_common_mode(
    flux: np.ndarray,
    *,
    min_ensemble_fraction: float = 0.5,
    sigma: float = 1.5,
) -> np.ndarray:
    """Per-epoch scale of the field, from relative fluxes of a stable ensemble.

    Dividing raw flux by the clipped median *flux* of whoever is detected that
    epoch follows the luminosity function of the detections: at high airmass
    faint stars drop out, the median jumps toward the bright end, and a
    constantly detected target is left with an airmass-shaped continuum
    (e.g. 0.6 → 1.4). Using each star's own median first, then the epoch
    median of those ratios, keeps the common mode even when membership
    changes.
    """
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(flux) & (flux > 0.0)
    obj_med = _clipped_median(flux, axis=0, sigma=sigma)
    obj_med = np.where(np.isfinite(obj_med) & (obj_med > 0.0), obj_med, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = flux / obj_med[np.newaxis, :]
    rel[~finite] = np.nan
    ensemble = _ensemble_mask(finite, min_ensemble_fraction)
    rel[:, ~ensemble] = np.nan
    factor = _clipped_median(rel, axis=1, sigma=sigma)
    return np.where(np.isfinite(factor) & (factor > 0.0), factor, np.nan)


def _as_flux_distribution(
    flux: np.ndarray,
    flux_error: np.ndarray,
    distribution_samples: int,
) -> unc.core.NdarrayDistribution:
    fl = np.asarray(flux, dtype=float)
    # Keep missing cells as NaN in the samples (pdf_median stays NaN).
    return unc.normal(
        fl,
        std=_sanitize_std(fl, flux_error),
        n_samples=distribution_samples,
    )


def quasi_flux_calibration_flux_arrays(
    flux: np.ndarray,
    flux_error: np.ndarray,
    *,
    distribution_samples: int = 1000,
    min_ensemble_fraction: float = 0.5,
) -> unc.core.NdarrayDistribution:
    """
    Quasi flux calibration on a 2D array ``(n_epochs, n_objects)``.

    Removes the epoch common mode (transparency, airmass, clouds) using the
    sigma-clipped median of *relative* fluxes of stars detected in at least
    ``min_ensemble_fraction`` of epochs. Missing or non-positive flux is
    ignored (NaN or 0).
    """
    flux = np.asarray(flux, dtype=float)
    factor = _epoch_common_mode(
        flux,
        min_ensemble_fraction=min_ensemble_fraction,
    )
    factor = np.where(np.isfinite(factor) & (factor > 0.0), factor, np.nan)
    flux_distribution = _as_flux_distribution(
        flux, flux_error, distribution_samples
    )
    return flux_distribution / factor[:, np.newaxis]


def flux_normalization_flux_distribution(
    flux_distribution: unc.core.NdarrayDistribution,
) -> unc.core.NdarrayDistribution:
    """
    Per-object normalization: divide by sigma-clipped median over epochs (axis 0).

    Matches :func:`flux_normalization_image_series` when given quasi-calibrated
    flux, or raw flux wrapped in a normal distribution.
    """
    flux = flux_distribution.pdf_median()
    median = _clipped_median(flux, axis=0)
    median = np.where(np.isfinite(median) & (median > 0.0), median, np.nan)
    return flux_distribution / median


def quasi_flux_calibration_image_series(
    image_series: ImageSeries,
    distribution_samples: int = 1000,
    min_ensemble_fraction: float = 0.5,
) -> unc.core.NdarrayDistribution:
    """
    Simple calibration for flux values: divide out the field common mode.

    Parameters
    ----------
    image_series
        Image series with flux of all objects in all images.
    distribution_samples
        Number of samples used for distributions. Default is ``1000``.
    min_ensemble_fraction
        Stars must be detected in at least this fraction of epochs to enter
        the common-mode ensemble. Default is ``0.5``.

    Returns
    -------
    flux_calibrated
        Quasi-calibrated flux distribution.
    """
    flux, flux_error = image_series.get_flux_array()
    return quasi_flux_calibration_flux_arrays(
        flux,
        flux_error,
        distribution_samples=distribution_samples,
        min_ensemble_fraction=min_ensemble_fraction,
    )


def flux_normalization_image_series(
    image_series: ImageSeries,
    quasi_calibrated_flux: unc.core.NdarrayDistribution | None = None,
    distribution_samples: int = 1000,
) -> unc.core.NdarrayDistribution:
    """
    Normalize flux of each object.

    Parameters
    ----------
    image_series
        Image series with flux of all objects in all images.
    quasi_calibrated_flux
        Quasi-calibrated object flux (common mode already removed). If
        ``None``, raw flux from ``image_series`` is used.
    distribution_samples
        Number of samples used for distributions. Default is ``1000``.

    Returns
    -------
    normalized_flux
        Normalized flux distribution.
    """
    if quasi_calibrated_flux is not None:
        flux_distribution = quasi_calibrated_flux
    else:
        flux, flux_error = image_series.get_flux_array()
        flux_distribution = _as_flux_distribution(
            flux, flux_error, distribution_samples
        )
    return flux_normalization_flux_distribution(flux_distribution)


__all__ = [
    "flux_normalization_flux_distribution",
    "flux_normalization_image_series",
    "quasi_flux_calibration_flux_arrays",
    "quasi_flux_calibration_image_series",
]
