"""Series-level WCS helpers (thin wrappers around ``ost_photometry.wcs``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ... import wcs as wcs_utilities

if TYPE_CHECKING:
    from ..models import ImageSeries


def find_wcs(
    image_series: ImageSeries,
    reference_image_index: int | None = None,
    method: str = "astrometry",
    cosmics_removed: bool = False,
    image_path_cosmics_removed: str | None = None,
    object_x_coordinates: np.ndarray | None = None,
    object_y_coordinates: np.ndarray | None = None,
    force_wcs_determination: bool = False,
    indent: int = 2,
) -> None:
    """
    Resolve WCS for an image series via :func:`~ost_photometry.wcs.find_wcs_for_image`.

    If ``reference_image_index`` is set, only that image is solved and the
    series WCS is set from it. Otherwise each image is solved and the series
    WCS is taken from the first image.
    """
    wcs_kwargs = {
        "method": method,
        "cosmics_removed": cosmics_removed,
        "image_path_cosmics_removed": image_path_cosmics_removed,
        "object_x_coordinates": object_x_coordinates,
        "object_y_coordinates": object_y_coordinates,
        "force_wcs_determination": force_wcs_determination,
        "indent": indent,
    }

    if reference_image_index is not None:
        img = image_series.image_list[reference_image_index]
        image_series.set_wcs(
            wcs_utilities.find_wcs_for_image(img, **wcs_kwargs)
        )
        return

    for i, img in enumerate(image_series.image_list):
        resolved_wcs = wcs_utilities.find_wcs_for_image(img, **wcs_kwargs)
        if i == 0:
            image_series.set_wcs(resolved_wcs)


__all__ = ["find_wcs"]
