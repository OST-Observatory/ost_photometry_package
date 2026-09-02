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


def _as_angle(radius: u.Quantity | float) -> u.Quantity:
    if isinstance(radius, u.Quantity):
        return radius
    return float(radius) * u.arcsec


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

    Same idea as the legacy calibration path: a star that is a known variable is
    not a comparison star. A failed or empty query leaves ``catalog`` unchanged
    (calibration must not abort because Vizier is down). Sets
    ``catalog.meta[KNOWN_VARIABLES_EXCLUDED_META]`` when the query succeeds.
    """
    if catalog is None or len(catalog) == 0:
        return catalog
    if "ra" not in catalog.colnames or "dec" not in catalog.colnames:
        return catalog
    if catalog.meta.get(KNOWN_VARIABLES_EXCLUDED_META):
        return catalog

    try:
        vsx_tbl, column_dict, ra_unit = get_vizier_catalog(
            [],
            center,
            field_of_view_arcmin,
            catalog_identifier,
            cleanup_magnitudes=False,
            print_infos=False,
            indent=indent,
        )
    except Exception as exc:
        terminal_output.print_to_terminal(
            f"Known-variable catalog ({catalog_identifier}) query failed; "
            f"keeping all comparison stars ({exc})",
            indent=indent,
            style_name="WARNING",
        )
        return catalog

    if vsx_tbl is None or len(vsx_tbl) == 0 or not column_dict:
        catalog.meta[KNOWN_VARIABLES_EXCLUDED_META] = True
        terminal_output.print_to_terminal(
            "No known variables in the field (VSX); comparison-star list unchanged",
            indent=indent,
            style_name="NORMAL",
        )
        return catalog

    ra_col = column_dict.get("ra")
    dec_col = column_dict.get("dec")
    if not ra_col or not dec_col or ra_col not in vsx_tbl.colnames:
        catalog.meta[KNOWN_VARIABLES_EXCLUDED_META] = True
        return catalog
    if not ra_unit:
        ra_unit = u.deg

    try:
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
    except Exception as exc:
        terminal_output.print_to_terminal(
            f"Could not match known variables to the catalog ({exc}); "
            "keeping all comparison stars",
            indent=indent,
            style_name="WARNING",
        )
        return catalog

    match_radius = _as_angle(radius)
    idx_cat, _idx_var, _sep, _ = matching.search_around_sky(
        cat_coords, var_coords, match_radius
    )
    drop = np.zeros(len(catalog), dtype=bool)
    if len(idx_cat):
        drop[np.asarray(idx_cat, dtype=int)] = True
    n_drop = int(np.count_nonzero(drop))
    out = catalog[~drop] if n_drop else catalog
    out.meta[KNOWN_VARIABLES_EXCLUDED_META] = True
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
]
