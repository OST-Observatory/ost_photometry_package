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


def _allfluxes_adql_column(band: str) -> str:
    """TAP ``allfluxes`` column for a photometric band (``V`` → ``allfluxes.V``)."""
    b = str(band).strip()
    if len(b) == 1 and b.islower():
        return f"allfluxes.{b}_"
    return f"allfluxes.{b}"


def simbad_query_criteria(
    *,
    filter_mag: str | None = None,
    mag_limit: float | None = None,
    otypes: list[str] | None = None,
    require_common_name: bool = False,
    extra_criteria: str | None = None,
) -> str | None:
    """
    Build a TAP ``criteria`` string for :meth:`~astroquery.simbad.SimbadClass.query_region`.

    Magnitude uses ``allfluxes.<band>`` (joined when ``flux(V)`` is requested).
    Object types use SIMBAD hierarchical ``..`` suffixes. Common names are the
    ``NAME `` identifier catalog (``ids LIKE '%NAME %'``).
    """
    parts: list[str] = []
    if mag_limit is not None:
        band = (filter_mag or "V").strip() or "V"
        parts.append(f"{_allfluxes_adql_column(band)} < {float(mag_limit)}")
    if otypes:
        clauses: list[str] = []
        for raw in otypes:
            ot = str(raw).strip()
            if not ot:
                continue
            if not ot.endswith(".."):
                ot = f"{ot}.."
            escaped = ot.replace("'", "''")
            clauses.append(f"otype = '{escaped}'")
        if clauses:
            parts.append("(" + " OR ".join(clauses) + ")")
    if require_common_name:
        parts.append("ids.ids LIKE '%NAME %'")
    extra = (extra_criteria or "").strip()
    if extra:
        parts.append(f"({extra})")
    if not parts:
        return None
    return " AND ".join(parts)


def simbad_magnitude_column(table: Table, filter_mag: str) -> str | None:
    """Column holding the requested Simbad flux (TAP ``V`` vs legacy ``FLUX_V``)."""
    band = str(filter_mag).strip()
    upper = band.upper()
    candidates = (
        f"FLUX_{upper}",
        upper,
        band,
        f"{upper}mag",
        f"{band}mag",
        f"flux_{upper}",
        f"FLUX_{band}",
    )
    lookup = {str(name).upper(): name for name in table.colnames}
    for name in candidates:
        hit = lookup.get(name.upper())
        if hit is not None:
            return hit
    return None


def _table_has_common_name(table: Table) -> np.ndarray:
    """True where a row carries a SIMBAD ``NAME …`` identifier."""
    n = len(table)
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    for col in ("ids", "IDS", "main_id", "MAIN_ID"):
        if col not in table.colnames:
            continue
        values = np.asarray(table[col], dtype=str)
        keep |= np.char.find(values, "NAME ") >= 0
    return keep


def filter_simbad_objects(
    table: Table,
    *,
    filter_mag: str | None = None,
    mag_limit: float | None = None,
    otypes: list[str] | None = None,
    require_common_name: bool = False,
) -> Table:
    """Client-side safety net after the TAP query (magnitude / type / common name)."""
    if table is None or len(table) == 0:
        return table if table is not None else Table()
    keep = np.ones(len(table), dtype=bool)
    if mag_limit is not None:
        band = (filter_mag or "V").strip() or "V"
        mag_col = simbad_magnitude_column(table, band)
        if mag_col is not None:
            mag = np.asarray(table[mag_col], dtype=float)
            keep &= np.isfinite(mag) & (mag <= float(mag_limit))
    if otypes:
        otype_col = None
        for name in ("OTYPE", "otype"):
            if name in table.colnames:
                otype_col = name
                break
        if otype_col is not None:
            labels = [str(ot).strip().rstrip(".") for ot in otypes if str(ot).strip()]
            types = np.asarray(table[otype_col], dtype=str)
            type_ok = np.zeros(len(table), dtype=bool)
            for label in labels:
                type_ok |= np.char.find(types, label) >= 0
            keep &= type_ok
    if require_common_name:
        keep &= _table_has_common_name(table)
    if np.all(keep):
        return table
    return table[keep]


def query_simbad_objects(
        wcs_image: wcs.WCS, image_shape: tuple[int, int],
        filter_mag: str | None = None,
        mag_limit: float | None = None,
        otypes: list[str] | None = None,
        require_common_name: bool = False,
        extra_criteria: str | None = None,
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
        Photometric band for a magnitude cut (e.g. ``V``). Defaults to ``V``
        when ``mag_limit`` is set.

    mag_limit
        Bright-end TAP cut: keep objects with ``band < mag_limit``.

    otypes
        Hierarchical SIMBAD types (``Star``, ``Galaxy``, ``Cl*``, …). Each
        value is queried as ``otype = 'Star..'``.

    require_common_name
        If True, only objects with a SIMBAD ``NAME …`` identifier.

    extra_criteria
        Extra TAP/ADQL fragment, ANDed with the other cuts.
    """
    center_coord, radius = search_cone_from_wcs(wcs_image, image_shape)
    radius_deg = _simbad_query_radius_deg(radius)
    if radius_deg is None:
        return Table()

    band = filter_mag
    if mag_limit is not None and not (band or "").strip():
        band = "V"

    criteria = simbad_query_criteria(
        filter_mag=band,
        mag_limit=mag_limit,
        otypes=otypes,
        require_common_name=require_common_name,
        extra_criteria=extra_criteria,
    )

    custom_simbad = Simbad()
    custom_simbad.TIMEOUT = 120
    fields = ["otype", "dimensions"]
    if band:
        fields.append(f"flux({band})")
    if require_common_name:
        fields.append("ids")
    custom_simbad.add_votable_fields(*fields)

    query_kw: dict[str, Any] = {}
    if criteria:
        query_kw["criteria"] = criteria

    try:
        result = custom_simbad.query_region(
            center_coord, radius=radius_deg * u.deg, **query_kw
        )
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
        fallback = simbad_query_criteria(
            otypes=otypes,
            require_common_name=require_common_name,
            extra_criteria=extra_criteria,
        )
        fallback_kw: dict[str, Any] = {}
        if fallback:
            fallback_kw["criteria"] = fallback
        result = custom_simbad.query_region(
            center_coord, radius=radius_deg * u.deg, **fallback_kw
        )

    if result is None:
        return Table()
    return filter_simbad_objects(
        result,
        filter_mag=band,
        mag_limit=mag_limit,
        otypes=otypes,
        require_common_name=require_common_name,
    )


def mark_simbad_objects_on_image(
        image_data: np.ndarray, image_wcs: wcs.WCS, output_dir: Path,
        filter_: str, file_type: str = 'pdf', filter_mag: str | None = None,
        mag_limit: float | None = None,
        otypes: list[str] | None = None,
        require_common_name: bool = False,
        extra_criteria: str | None = None,
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

    otypes, require_common_name, extra_criteria
        Passed to :func:`query_simbad_objects`.
    """
    simbad_objects = query_simbad_objects(
        image_wcs,
        image_data.shape,
        filter_mag=filter_mag,
        mag_limit=mag_limit,
        otypes=otypes,
        require_common_name=require_common_name,
        extra_criteria=extra_criteria,
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
    otypes: list[str] | None = None,
    require_common_name: bool = False,
    extra_criteria: str | None = None,
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
            otypes=otypes,
            require_common_name=require_common_name,
            extra_criteria=extra_criteria,
        )
    except Exception as exc:
        terminal_output.print_to_terminal(
            f"Simbad annotated starmap failed (network / query issue?): {exc}",
            indent=indent,
            style_name="WARNING",
        )


__all__ = [
    "annotate_reference_image_with_simbad",
    "filter_simbad_objects",
    "mark_simbad_objects_on_image",
    "query_simbad_objects",
    "search_cone_from_wcs",
    "simbad_magnitude_column",
    "simbad_query_criteria",
]
