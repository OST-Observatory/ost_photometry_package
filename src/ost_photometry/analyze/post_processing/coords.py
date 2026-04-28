"""Sky coordinates from photometry tables; starmaps via ImagingPlotContext."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

if TYPE_CHECKING:
    from astropy.wcs import WCS

    from .imaging import ImagingPlotContext


def table_object_sky_coords(tbl: Table, wcs_fallback: "WCS") -> SkyCoord:
    """
    ICRS coordinates for each row: prefer ``ra``/``dec`` or ``ra (deg)``/``dec (deg)``,
    otherwise ``wcs_fallback.all_pix2world(tbl['x'], tbl['y'], 0)``.
    """
    cols = tbl.colnames
    if "ra" in cols and "dec" in cols:
        ra = u.Quantity(tbl["ra"], u.deg)
        dec = u.Quantity(tbl["dec"], u.deg)
        return SkyCoord(ra=ra, dec=dec, frame="icrs")
    if "ra (deg)" in cols and "dec (deg)" in cols:
        ra = u.Quantity(tbl["ra (deg)"], u.deg)
        dec = u.Quantity(tbl["dec (deg)"], u.deg)
        return SkyCoord(ra=ra, dec=dec, frame="icrs")
    lon, lat = wcs_fallback.all_pix2world(tbl["x"], tbl["y"], 0)
    return SkyCoord(lon, lat, unit=u.deg, frame="icrs")


def plot_starmap_from_imaging_context(
    ctx: "ImagingPlotContext",
    tbl: Table,
    *,
    filter_: str,
    x_name: str = "x",
    y_name: str = "y",
    rts_pre: str = "image",
    label: str = "Stars with photometric extractions",
    add_image_id: bool = True,
    use_wcs_projection_for_star_maps: bool = True,
    file_type_plots: str = "pdf",
) -> None:
    """Overlay ``tbl`` positions on the reference image (same contract as ``prepare_and_plot_starmap``)."""
    from .. import plots

    data = ctx.reference_image
    n_stars = len(tbl)
    tbl_xy = Table(
        names=["id", "x_centroid", "y_centroid"],
        data=[np.arange(n_stars, dtype=int), tbl[x_name], tbl[y_name]],
    )
    
    rts = rts_pre
    if add_image_id and ctx.plot_reference_image_id is not None:
        rts += f": {ctx.plot_reference_image_id}"
    plots.starmap(
        str(ctx.out_path_stub),
        data,
        filter_,
        tbl_xy,
        label=label,
        rts=rts,
        wcs_image=ctx.wcs,
        use_wcs_projection=use_wcs_projection_for_star_maps,
        file_type=file_type_plots,
    )
