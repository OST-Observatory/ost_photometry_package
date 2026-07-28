"""Simbad cone query and image annotation."""

from __future__ import annotations

import astropy.units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.exceptions import TableParseError
from astroquery.simbad import Simbad
from regions import CirclePixelRegion, PixCoord, Regions

from ... import terminal_output
from .. import plots

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
    #   Determine the limits of the image in the celestial coordinate system
    height, width = image_shape
    coordinates = wcs_image.pixel_to_world(
        [0, width, 0, width], [0, 0, height, height]
    )
    ra_min, ra_max = coordinates.ra.degree.min(), coordinates.ra.degree.max()
    dec_min, dec_max = coordinates.dec.degree.min(), coordinates.dec.degree.max()

    #   Calculate image center and search radius
    center_ra = (ra_min + ra_max) / 2
    center_dec = (dec_min + dec_max) / 2
    radius_deg = max(ra_max - ra_min, dec_max - dec_min) / 2

    center_coord = SkyCoord(ra=center_ra, dec=center_dec, unit="deg")

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
            f"The connection to the Simbad database for retrieving object "
            f"information has timed out. Return an empty table.",
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
        custom_simbad.add_votable_fields('otype')
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


__all__ = [
    "mark_simbad_objects_on_image",
    "query_simbad_objects",
]
