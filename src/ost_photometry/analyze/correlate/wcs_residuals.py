"""Reproject matched sources onto a reference WCS."""

from __future__ import annotations

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord


def residual_vectors_on_reference_wcs(
    x_ref,
    y_ref,
    wcs_ref: wcs.WCS,
    x_other,
    y_other,
    wcs_other: wcs.WCS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel residuals on the reference WCS, plus on-sky separation (arcsec).

    The other-filter sky position is reprojected onto ``wcs_ref``. ``dx, dy``
    are then ``(x_other_on_ref - x_ref, y_other_on_ref - y_ref)``.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    c_ref = SkyCoord.from_pixel(x_ref, y_ref, wcs_ref, origin=0)
    c_other = SkyCoord.from_pixel(
        np.asarray(x_other, dtype=float),
        np.asarray(y_other, dtype=float),
        wcs_other,
        origin=0,
    )
    x_proj, y_proj = c_other.to_pixel(wcs_ref, origin=0)
    dx = np.asarray(x_proj, dtype=float) - x_ref
    dy = np.asarray(y_proj, dtype=float) - y_ref
    sep = np.asarray(c_ref.separation(c_other).arcsec, dtype=float)
    return dx, dy, sep
