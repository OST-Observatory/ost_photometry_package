"""Drop calibration-catalog rows that coincide with known variable stars."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, matching
from astropy.table import Table

from ... import terminal_output
from .vizier_query import get_vizier_catalog

VSX_VIZIER_ID = "B/vsx/vsx"
KNOWN_VARIABLES_EXCLUDED_META = "ost_photometry.known_variables_excluded"
DEFAULT_EXCLUDE_RADIUS = 1.0 * u.arcsec
_ROW_COL = "ost_cat_row"


def _as_angle(radius: u.Quantity | float) -> u.Quantity:
    if isinstance(radius, u.Quantity):
        return radius
    return float(radius) * u.arcsec


def query_vsx_xmatch(
    positions: Table,
    radius: u.Quantity,
    catalog_identifier: str = VSX_VIZIER_ID,
) -> Table:
    """CDS xMatch: catalog positions against VSX (only those rows, not a FOV dump)."""
    from astroquery.xmatch import XMatch

    cat2 = catalog_identifier
    if not str(cat2).lower().startswith("vizier:"):
        cat2 = f"vizier:{cat2}"
    service = XMatch()
    # Cone-searching the whole VSX table via Vizier is often very slow; xMatch is not.
    timeout = getattr(service, "TIMEOUT", 60)
    try:
        if timeout is None or float(timeout) > 120:
            service.TIMEOUT = 60
    except (TypeError, ValueError):
        service.TIMEOUT = 60
    return service.query(
        cat1=positions,
        cat2=cat2,
        max_distance=radius,
        colRA1="ra",
        colDec1="dec",
    )


def _drop_via_xmatch(
    catalog: Table,
    radius: u.Quantity,
    catalog_identifier: str,
) -> np.ndarray | None:
    """Return integer catalog-row indices that match VSX, or ``None`` if xMatch failed."""
    work = Table()
    work[_ROW_COL] = np.arange(len(catalog), dtype=np.int64)
    work["ra"] = np.asarray(catalog["ra"], dtype=float)
    work["dec"] = np.asarray(catalog["dec"], dtype=float)
    matched = query_vsx_xmatch(work, radius, catalog_identifier)
    if matched is None or len(matched) == 0:
        return np.array([], dtype=int)
    if _ROW_COL not in matched.colnames:
        return None
    return np.unique(np.asarray(matched[_ROW_COL], dtype=int))


def _drop_via_cone_search(
    catalog: Table,
    center: SkyCoord,
    field_of_view_arcmin: float,
    radius: u.Quantity,
    catalog_identifier: str,
    indent: int,
) -> np.ndarray | None:
    """FOV cone on VSX (fallback). ``None`` means the query failed."""
    vsx_tbl, column_dict, ra_unit = get_vizier_catalog(
        [],
        center,
        field_of_view_arcmin,
        catalog_identifier,
        cleanup_magnitudes=False,
        print_infos=True,
        indent=indent,
    )
    if vsx_tbl is None or len(vsx_tbl) == 0 or not column_dict:
        return np.array([], dtype=int)
    ra_col = column_dict.get("ra")
    dec_col = column_dict.get("dec")
    if not ra_col or not dec_col or ra_col not in vsx_tbl.colnames:
        return np.array([], dtype=int)
    if not ra_unit:
        ra_unit = u.deg
    var_coords = SkyCoord(
        vsx_tbl[ra_col],
        vsx_tbl[dec_col],
        unit=(ra_unit, u.deg),
        frame="icrs",
    )
    cat_coords = SkyCoord(
        np.asarray(catalog["ra"], dtype=float),
        np.asarray(catalog["dec"], dtype=float),
        unit="deg",
        frame="icrs",
    )
    idx_cat, _idx_var, _sep, _ = matching.search_around_sky(
        cat_coords, var_coords, radius
    )
    if len(idx_cat) == 0:
        return np.array([], dtype=int)
    return np.unique(np.asarray(idx_cat, dtype=int))


def drop_catalog_rows_near_known_variables(
    catalog: Table,
    center: SkyCoord,
    field_of_view_arcmin: float,
    *,
    radius: u.Quantity | float = DEFAULT_EXCLUDE_RADIUS,
    catalog_identifier: str = VSX_VIZIER_ID,
    indent: int = 2,
) -> Table:
    """
    Remove standard-schema catalog rows within ``radius`` of a VSX (or similar) source.

    Prefers CDS xMatch of the catalog positions (typically ~10² APASS stars) over a
    Vizier cone of the whole field: ``B/vsx/vsx`` cone queries are often very slow
    and used to stall anlyses steps right after the APASS download with no log line.

    A failed query leaves ``catalog`` unchanged (calibration must not abort).
    Sets ``catalog.meta[KNOWN_VARIABLES_EXCLUDED_META]`` when the query succeeds.
    """
    if catalog is None or len(catalog) == 0:
        return catalog
    if "ra" not in catalog.colnames or "dec" not in catalog.colnames:
        return catalog
    if catalog.meta.get(KNOWN_VARIABLES_EXCLUDED_META):
        return catalog

    match_radius = _as_angle(radius)
    terminal_output.print_to_terminal(
        f"Excluding known variables ({catalog_identifier}) via CDS xMatch "
        f"({len(catalog)} catalog positions, {match_radius.to_value(u.arcsec):.1f}\") ...",
        indent=indent,
        style_name="NORMAL",
    )

    drop_idx: np.ndarray | None = None
    try:
        drop_idx = _drop_via_xmatch(catalog, match_radius, catalog_identifier)
    except Exception as exc:
        terminal_output.print_to_terminal(
            f"CDS xMatch failed ({exc}); falling back to a VSX cone query",
            indent=indent,
            style_name="WARNING",
        )

    if drop_idx is None:
        try:
            drop_idx = _drop_via_cone_search(
                catalog,
                center,
                field_of_view_arcmin,
                match_radius,
                catalog_identifier,
                indent,
            )
        except Exception as exc:
            terminal_output.print_to_terminal(
                f"Known-variable catalog ({catalog_identifier}) query failed; "
                f"keeping all comparison stars ({exc})",
                indent=indent,
                style_name="WARNING",
            )
            return catalog

    if drop_idx is None:
        return catalog

    drop = np.zeros(len(catalog), dtype=bool)
    if drop_idx.size:
        valid = drop_idx[(drop_idx >= 0) & (drop_idx < len(catalog))]
        drop[valid] = True
    n_drop = int(np.count_nonzero(drop))
    out = catalog[~drop] if n_drop else catalog
    out.meta[KNOWN_VARIABLES_EXCLUDED_META] = True
    if n_drop == 0:
        terminal_output.print_to_terminal(
            "No known variables among comparison stars (VSX); list unchanged",
            indent=indent,
            style_name="NORMAL",
        )
    else:
        terminal_output.print_to_terminal(
            f"{len(out)} calibration stars remain after excluding {n_drop} "
            f"known variable{'s' if n_drop != 1 else ''} "
            f"({catalog_identifier}, {match_radius.to_value(u.arcsec):.1f}\")",
            indent=indent,
            style_name="GOOD",
        )
    return out


__all__ = [
    "DEFAULT_EXCLUDE_RADIUS",
    "KNOWN_VARIABLES_EXCLUDED_META",
    "VSX_VIZIER_ID",
    "drop_catalog_rows_near_known_variables",
    "query_vsx_xmatch",
]
