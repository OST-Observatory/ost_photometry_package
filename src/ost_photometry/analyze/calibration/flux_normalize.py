"""Quasi-ZP and per-object flux normalization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy import uncertainty as unc
from astropy.stats import sigma_clipped_stats

if TYPE_CHECKING:
    from ..models import ImageSeries


def quasi_flux_calibration_flux_arrays(
    flux: np.ndarray,
    flux_error: np.ndarray,
    *,
    distribution_samples: int = 1000,
) -> unc.core.NdarrayDistribution:
    """
    Quasi flux calibration on a 2D array ``(n_epochs, n_objects)``.

    Per epoch, divide flux by the sigma-clipped median flux over objects (same
    idea as :func:`quasi_flux_calibration_image_series`). Zero flux is masked
    like in the image-series path.
    """
    _, median, _stddev = sigma_clipped_stats(
        flux,
        axis=1,
        sigma=1.5,
        mask_value=0.0,
    )
    flux_distribution = unc.normal(
        flux,
        std=flux_error,
        n_samples=distribution_samples,
    )
    return flux_distribution / median[:, np.newaxis]


def flux_normalization_flux_distribution(
    flux_distribution: unc.core.NdarrayDistribution,
) -> unc.core.NdarrayDistribution:
    """
    Per-object normalization: divide by sigma-clipped median over epochs (axis 0).

    Matches :func:`flux_normalization_image_series` when given quasi-calibrated
    flux, or raw flux wrapped in a normal distribution.
    """
    flux = flux_distribution.pdf_median()
    _, median, _stddev = sigma_clipped_stats(
        flux,
        axis=0,
        sigma=1.5,
        mask_value=0.0,
    )
    return flux_distribution / median


def quasi_flux_calibration_image_series(
    image_series: ImageSeries,
    distribution_samples: int = 1000,
) -> unc.core.NdarrayDistribution:
    """
    Simple calibration for flux values. Assuming the median over all
    objects in an image as a quasi ZP.

    Parameters
    ----------
    image_series
        Image series with flux of all objects in all images.
    distribution_samples
        Number of samples used for distributions. Default is ``1000``.

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
        Quasi-calibrated object flux: the median over all objects is used as
        the quasi ZP. If ``None``, raw flux from ``image_series`` is used.
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
        flux_distribution = unc.normal(
            flux,
            std=flux_error,
            n_samples=distribution_samples,
        )
    return flux_normalization_flux_distribution(flux_distribution)


__all__ = [
    "flux_normalization_flux_distribution",
    "flux_normalization_image_series",
    "quasi_flux_calibration_flux_arrays",
    "quasi_flux_calibration_image_series",
]
