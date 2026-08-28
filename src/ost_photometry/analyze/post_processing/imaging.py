"""Minimal context for imaging-based plots in post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    import numpy as np
    from astropy.wcs import WCS


@dataclass
class ImagingPlotContext:
    """
    WCS, reference image, filter label, output naming, and optional Vizier cone.

    Use for star maps, pixel overlays, and Gaia ``query_region`` (field center +
    radius in arcminutes). Table-based cuts should prefer ``ra``/``dec`` on the table
    when present.
    """

    wcs: WCS
    reference_image: np.ndarray
    #: Campaign output directory (not a basename). Plot writers nest under this.
    out_path_stub: Path | str
    #: Band / filter name (e.g. for plot labels); mirrors ``ImageSeries.filter_``.
    filter_name: str
    image_shape: tuple[int, int] | None = None
    #: ``Image.image_id`` of the reference exposure for starmap filename suffix; optional.
    plot_reference_image_id: int | None = None
    #: Basename of the reference FITS file (for starmap titles); optional.
    plot_reference_filename: str | None = None
    #: Field center for Vizier cone queries; optional if only plotting / region cuts.
    field_center_icrs: SkyCoord | None = None
    #: Cone radius in **arcminutes** (same convention as ``ImageSeries.field_of_view_x``).
    field_radius_arcmin: float | None = None


def imaging_context_from_image_series(image_series: object) -> ImagingPlotContext:
    """
    Build context from an :class:`~ost_photometry.analyze.models.ImageSeries`.

    Fills WCS, reference image data, ``filter_name``, output path stub, optional Vizier
    cone fields from the reference image / series, and ``plot_reference_image_id``.
    """
    import numpy as np

    ref = image_series.reference_image
    arr = np.asarray(ref.get_data())
    return ImagingPlotContext(
        wcs=ref.wcs,
        reference_image=arr,
        out_path_stub=ref.out_path,
        filter_name=image_series.filter_,
        image_shape=tuple(arr.shape),
        plot_reference_image_id=ref.image_id,
        plot_reference_filename=getattr(ref, "filename", None),
        field_center_icrs=getattr(image_series, "coordinates_image_center", None),
        field_radius_arcmin=getattr(image_series, "field_of_view_x", None),
    )
