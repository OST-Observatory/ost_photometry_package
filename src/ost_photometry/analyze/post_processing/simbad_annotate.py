"""Simbad cone query and image annotation (optional post-processing)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.exceptions import TableParseError
from astroquery.simbad import Simbad

from ... import terminal_output
from .. import plots

# TAP / Simbad cone queries reject radii outside (0, 90] deg.
_SIMBAD_MAX_RADIUS_DEG = 90.0


def search_cone_from_wcs(
    wcs_image: wcs.WCS,
    image_shape: tuple[int, int],
) -> tuple[SkyCoord, u.Quantity]:
    """
    Image-center SkyCoord and circumradius covering all four corners.

    Uses angular separation so RA wrap-around (e.g. fields near 0 h) does not
    produce a ~180 deg radius from ``ra_max - ra_min``.
    """
    height, width = image_shape
    x0, x1 = 0.0, float(max(width - 1, 0))
    y0, y1 = 0.0, float(max(height - 1, 0))
    center = wcs_image.pixel_to_world(0.5 * (x0 + x1), 0.5 * (y0 + y1))
    corners = wcs_image.pixel_to_world(
        [x0, x1, x0, x1],
        [y0, y0, y1, y1],
    )
    radius = corners.separation(center).max()
    return center, radius


def _simbad_query_radius_deg(radius: u.Quantity) -> float | None:
    """Return a TAP-legal radius in degrees, or ``None`` if the cone is unusable."""
    radius_deg = float(radius.to_value(u.deg))
    if not np.isfinite(radius_deg) or radius_deg <= 0.0:
        terminal_output.print_to_terminal(
            "Simbad search radius is not positive; skipping query.",
            style_name="WARNING",
        )
        return None
    if radius_deg > _SIMBAD_MAX_RADIUS_DEG:
        terminal_output.print_to_terminal(
            f"Simbad search radius {radius_deg:.1f} deg exceeds the "
            f"{_SIMBAD_MAX_RADIUS_DEG:.0f} deg TAP limit (check WCS). "
            f"Clipping to {_SIMBAD_MAX_RADIUS_DEG:.0f} deg.",
            style_name="WARNING",
        )
        return _SIMBAD_MAX_RADIUS_DEG
    return radius_deg


def query_simbad_objects(
        wcs_image: wcs.WCS, image_shape: tuple[int, int],
        filter_mag: str | None = None,
    ) -> Table:
    """
    Retrieves objects from the Simbad database that are within
    the field of view.

    Parameters
    ----------
    wcs_image
       WCS object of the FITS file

    image_shape
        Tuple (height, width) of the image

    filter_mag
        Name of the filter (e.g. 'V')
        Default is ``None``.

    Returns
    -------
        Table of objects found
    """
    center_coord, radius = search_cone_from_wcs(wcs_image, image_shape)
    radius_deg = _simbad_query_radius_deg(radius)
    if radius_deg is None:
        return Table()

    #   Adjust Simbad query
    custom_simbad = Simbad()
    custom_simbad.TIMEOUT = 120
    if filter_mag is not None:
        custom_simbad.add_votable_fields('otype', f'flux({filter_mag})', 'dimensions')
    else:
        custom_simbad.add_votable_fields('otype', 'dimensions')

    #   Query Simbad
    try:
        result = custom_simbad.query_region(center_coord, radius=radius_deg * u.deg)
    except TimeoutError:
        terminal_output.print_to_terminal(
            "The connection to the Simbad database for retrieving object "
            "information has timed out. Return an empty table.",
            style_name='WARNING',
        )
        return Table()
    except TableParseError as e:
        terminal_output.print_to_terminal(
            f"Simbad request to retrieve object information failed. Most "
            f"likely because the requested magnitude is not available. "
            f" The error message was {e}.\n Remove magnitude from request...",
            style_name='WARNING',
        )
        custom_simbad = Simbad()
        custom_simbad.TIMEOUT = 120
        custom_simbad.add_votable_fields('otype', 'dimensions')
        result = custom_simbad.query_region(center_coord, radius=radius_deg * u.deg)

    return result


def mark_simbad_objects_on_image(
        image_data: np.ndarray, image_wcs: wcs.WCS, output_dir: Path,
        filter_: str, file_type: str = 'pdf', filter_mag: str | None = None,
        mag_limit: float | None = None,
    ) -> None:
    """
    Retrieves all known objects from Simbad for the current field of view
    and marks them on the image.

    Parameters
    ----------
    image_data
        Array with the image data

    image_wcs
       WCS object of the FITS file

    output_dir
        Output directory

    filter_
        Filter identifier

    file_type
        Type of plot file to be created
        Default is ``pdf``.

    filter_mag
        Name of the filter (e.g. 'V')
        Default is ``None``.

    mag_limit
        Limiting magnitude, only objects brighter as this limit will be shown
        Default is ``None``.
    """
    #   Retrieve objects from the Simbad database
    simbad_objects = query_simbad_objects(
        image_wcs,
        image_data.shape,
        filter_mag=filter_mag,
    )
    if simbad_objects is None or len(simbad_objects) == 0:
        terminal_output.print_to_terminal(
            "Simbad returned no objects for this field; skipping annotated starmap.",
            style_name="WARNING",
        )
        return

    #   Marks all known objects in the image
    plots.plot_annotated_image(
        image_data,
        image_wcs,
        simbad_objects,
        output_dir,
        filter_=filter_,
        file_type=file_type,
        filter_mag=filter_mag,
        mag_limit=mag_limit,
    )


def annotate_reference_image_with_simbad(
    image: Any,
    *,
    file_type: str = "pdf",
    filter_mag: str | None = None,
    mag_limit: float | None = None,
    indent: int = 2,
) -> None:
    """Overlay Simbad objects on a reference image, skipping on missing WCS or query errors."""
    if getattr(image, "wcs", None) is None:
        terminal_output.print_to_terminal(
            "Skipping Simbad annotated starmap: no WCS on the reference image.",
            indent=indent,
            style_name="WARNING",
        )
        return
    try:
        mark_simbad_objects_on_image(
            image.get_data(),
            image.wcs,
            image.out_path,
            image.filter_,
            file_type=file_type,
            filter_mag=filter_mag,
            mag_limit=mag_limit,
        )
    except Exception as exc:
        terminal_output.print_to_terminal(
            f"Simbad annotated starmap failed (network / query issue?): {exc}",
            indent=indent,
            style_name="WARNING",
        )


__all__ = [
    "annotate_reference_image_with_simbad",
    "mark_simbad_objects_on_image",
    "query_simbad_objects",
    "search_cone_from_wcs",
]
