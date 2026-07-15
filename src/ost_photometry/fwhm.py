"""Shared PSF FWHM estimation helpers."""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table


def select_sources_for_fwhm_fit(
    source_table: Table,
    *,
    min_objects: int = 40,
    slice_start: int = 20,
    slice_end: int = 40,
) -> Table:
    """Select unsaturated sources for a robust FWHM fit."""
    table = source_table.copy()
    table.sort("flux")
    if len(table) >= min_objects:
        return table[slice_start:slice_end]
    return table


def source_positions_from_table(source_table: Table) -> list[tuple[float, float]]:
    """Return ``(x, y)`` positions from a photutils source table."""
    if "x_centroid" in source_table.colnames:
        x_column, y_column = "x_centroid", "y_centroid"
    elif "x" in source_table.colnames:
        x_column, y_column = "x", "y"
    else:
        raise ValueError(
            "Source table must contain x/y or x_centroid/y_centroid columns."
        )
    return list(zip(source_table[x_column], source_table[y_column]))


def estimate_fwhm_from_positions(
    data: np.ndarray,
    xypos: list[tuple[float, float]] | np.ndarray,
    *,
    mask: np.ndarray | None = None,
    error: np.ndarray | None = None,
    default_fwhm: float,
    fit_shape: int = 25,
    min_fwhm: float = 2.0,
    max_fwhm: float = 9.0,
) -> tuple[float, str | None]:
    """
    Estimate PSF FWHM from source positions using ``photutils.psf.fit_fwhm``.

    Returns
    -------
    fwhm
        Estimated FWHM in pixels, or ``default_fwhm`` on failure.
    error_message
        ``None`` on success, otherwise a short reason string.
    """
    if len(xypos) == 0:
        return default_fwhm, "no sources available for FWHM fit"

    from astropy.modeling.fitting import NonFiniteValueError
    from photutils.psf import fit_fwhm

    try:
        fwhm_values = fit_fwhm(
            data,
            xypos=xypos,
            fit_shape=fit_shape,
            mask=mask,
            error=error,
        )
        median_fwhm = float(sigma_clipped_stats(fwhm_values)[1])
    except (ValueError, NonFiniteValueError) as exc:
        return default_fwhm, str(exc)

    if median_fwhm < min_fwhm or median_fwhm > max_fwhm:
        return default_fwhm, (
            f"estimated FWHM {median_fwhm:.3f} outside [{min_fwhm}, {max_fwhm}]"
        )

    return median_fwhm, None
