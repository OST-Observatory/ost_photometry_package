"""Series-level WCS helpers (thin wrappers around ``ost_photometry.wcs``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ... import wcs as wcs_utilities

if TYPE_CHECKING:
    from ...image import Image
    from ..models import ImageSeries


def ensure_image_wcs(
    image: Image,
    *,
    series_wcs=None,
    aligned_grid: bool = False,
    method: str = "astap",
    force_wcs_determination: bool = False,
    indent: int = 2,
    **wcs_kwargs,
):
    """Return a current WCS for ``image``, solving or copying once if missing."""
    if getattr(image, "wcs", None) is not None:
        return image.wcs
    if aligned_grid and series_wcs is not None:
        image.wcs = series_wcs
        wcs_utilities.sync_image_coordinates_from_wcs(image, series_wcs)
        return image.wcs
    resolved = wcs_utilities.find_wcs_for_image(
        image,
        method=method,
        force_wcs_determination=force_wcs_determination,
        indent=indent,
        **wcs_kwargs,
    )
    image.wcs = resolved
    wcs_utilities.sync_image_coordinates_from_wcs(image, resolved)
    return resolved


def find_wcs(
    image_series: ImageSeries,
    reference_image_index: int | None = None,
    method: str = "astap",
    cosmics_removed: bool = False,
    image_path_cosmics_removed: str | None = None,
    object_x_coordinates: np.ndarray | None = None,
    object_y_coordinates: np.ndarray | None = None,
    force_wcs_determination: bool = False,
    indent: int = 2,
    solve_all_images: bool = False,
) -> None:
    """
    Resolve WCS for an image series via :func:`~ost_photometry.wcs.find_wcs_for_image`.

    If ``solve_all_images`` is True, each frame is solved and
    ``image_series.wcs`` is the reference (or first) solution without
    broadcasting onto other frames.

    If ``reference_image_index`` is set and ``solve_all_images`` is False,
    only that image is solved and the solution is broadcast (aligned-grid
    assumption).

    Otherwise each image is solved and the series WCS is taken from the
    first image without overwriting per-frame solutions.
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

    def _solve(img):
        return wcs_utilities.find_wcs_for_image(img, **wcs_kwargs)

    if solve_all_images:
        series_wcs = None
        ref_idx = 0 if reference_image_index is None else int(reference_image_index)
        for i, img in enumerate(image_series.image_list):
            resolved_wcs = _solve(img)
            img.wcs = resolved_wcs
            wcs_utilities.sync_image_coordinates_from_wcs(img, resolved_wcs)
            if series_wcs is None or i == ref_idx:
                series_wcs = resolved_wcs
        if series_wcs is not None:
            image_series.set_wcs(series_wcs, broadcast=False)
        return

    if reference_image_index is not None:
        img = image_series.image_list[int(reference_image_index)]
        image_series.set_wcs(_solve(img), broadcast=True)
        return

    series_wcs = None
    for img in image_series.image_list:
        resolved_wcs = _solve(img)
        img.wcs = resolved_wcs
        wcs_utilities.sync_image_coordinates_from_wcs(img, resolved_wcs)
        if series_wcs is None:
            series_wcs = resolved_wcs
    if series_wcs is not None:
        image_series.set_wcs(series_wcs, broadcast=False)


__all__ = ["ensure_image_wcs", "find_wcs"]
