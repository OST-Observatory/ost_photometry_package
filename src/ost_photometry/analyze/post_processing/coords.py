"""Sky coordinates from photometry tables; starmaps via ImagingPlotContext."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

if TYPE_CHECKING:
    from astropy.wcs import WCS

    from .imaging import ImagingPlotContext


def table_object_sky_coords(tbl: Table, wcs_fallback: WCS) -> SkyCoord:
    """
    ICRS coordinates for each row: prefer ``ra``/``dec`` or ``ra (deg)``/``dec (deg)``,
    otherwise ``wcs_fallback.all_pix2world(tbl['x'], tbl['y'], 0)``.
    """
    cols = tbl.colnames
    if "ra" in cols and "dec" in cols:
        ra = u.Quantity(tbl["ra"].ravel(), u.deg)
        dec = u.Quantity(tbl["dec"].ravel(), u.deg)
        return SkyCoord(ra=ra, dec=dec, frame="icrs")
    if "ra (deg)" in cols and "dec (deg)" in cols:
        ra = u.Quantity(tbl["ra (deg)"].ravel(), u.deg)
        dec = u.Quantity(tbl["dec (deg)"].ravel(), u.deg)
        return SkyCoord(ra=ra, dec=dec, frame="icrs")
    lon, lat = wcs_fallback.all_pix2world(tbl["x"].ravel(), tbl["y"].ravel(), 0)
    return SkyCoord(lon, lat, unit=u.deg, frame="icrs")


def _starmap_xy_table(tbl: Table, x_name: str, y_name: str) -> Table:
    n_stars = len(tbl)
    if "id" in tbl.colnames:
        ids = np.asarray(tbl["id"])
    else:
        ids = np.arange(n_stars, dtype=int)
    return Table(
        names=["id", "x_centroid", "y_centroid"],
        data=[ids, tbl[x_name], tbl[y_name]],
    )


def plot_starmap_from_imaging_context(
    ctx: ImagingPlotContext,
    tbl: Table,
    *,
    filter_: str,
    x_name: str = "x",
    y_name: str = "y",
    rts_pre: str = "image",
    label: str = "Stars with photometric extractions",
    label_2: str = "Identified stars (set 2)",
    tbl_2: Table | None = None,
    extra_patches: list | None = None,
    covariance_on_tbl_2: bool = False,
    add_image_id: bool = True,
    use_wcs_projection_for_star_maps: bool = True,
    file_type_plots: str = "pdf",
) -> None:
    """Overlay ``tbl`` (and optional ``tbl_2``) on the reference image."""
    from ...output_layout import diagnostics_dir
    from .. import plots

    data = ctx.reference_image
    tbl_xy = _starmap_xy_table(tbl, x_name, y_name)
    tbl_xy_2 = None
    if tbl_2 is not None:
        tbl_xy_2 = _starmap_xy_table(tbl_2, x_name, y_name)

    patches = list(extra_patches or [])
    if covariance_on_tbl_2 and tbl_2 is not None and len(tbl_2) > 0:
        ellipse = plots.covariance_ellipse_pixels(
            tbl_2[x_name],
            tbl_2[y_name],
            image_shape=np.asarray(data).shape,
        )
        if ellipse is not None:
            patches.append(ellipse)

    title_rts = rts_pre
    filename_suffix = rts_pre
    if add_image_id and ctx.plot_reference_image_id is not None:
        filename_suffix = f"{rts_pre}: {ctx.plot_reference_image_id}"
        fname = getattr(ctx, "plot_reference_filename", None)
        if fname:
            title_rts = f"{rts_pre}: {ctx.plot_reference_image_id} ({fname})"
        else:
            title_rts = filename_suffix
    plots.starmap(
        str(diagnostics_dir(ctx.out_path_stub, "cluster")),
        data,
        filter_,
        tbl_xy,
        tbl_2=tbl_xy_2,
        label=label,
        label_2=label_2,
        rts=title_rts,
        filename_suffix=filename_suffix,
        wcs_image=ctx.wcs,
        use_wcs_projection=use_wcs_projection_for_star_maps,
        file_type=file_type_plots,
        extra_patches=patches or None,
    )
