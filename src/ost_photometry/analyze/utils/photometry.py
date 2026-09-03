"""Photometry helpers used by extraction."""

from __future__ import annotations

import numpy as np
from astropy.table import Column, Table

from ... import terminal_output


def flux_to_magnitudes(
    flux: np.ndarray | Column,
    flux_error: np.ndarray | Column,
) -> tuple[np.ndarray | Column, np.ndarray | Column]:
    """
    Calculate magnitudes from flux.

    Parameters
    ----------
    flux
        Flux values.
    flux_error
        Flux uncertainties.

    Returns
    -------
    magnitudes, magnitudes_error
        Object magnitudes (``-2.5 log10``) and **positive** 1σ uncertainties
        ``(2.5 / ln(10)) · σ(F)/|F|``.
    """
    magnitudes = -2.5 * np.log10(flux)
    # Pogson: σ(m) = (2.5 / ln(10)) · σ(F)/|F|.
    magnitudes_error = (2.5 / np.log(10)) * np.abs(flux_error / flux)
    return magnitudes, magnitudes_error


def rm_edge_objects(
    table: Table,
    data_array: np.ndarray,
    border: int = 10,
    terminal_logger: terminal_output.TerminalLog | None = None,
    indent: int = 3,
) -> Table:
    """
    Remove detected objects that are too close to the image edges.

    Parameters
    ----------
    table
        Object data with ``x_fit`` / ``y_fit`` columns.
    data_array
        Image data (2D).
    border
        Distance to the edge (pixels) inside which objects are discarded.
        Default is ``10``.
    terminal_logger
        Optional logger; otherwise prints to the terminal.
    indent
        Indentation for console output. Default is ``3``.
    """
    hsize = border + 1
    x = table["x_fit"].value
    y = table["y_fit"].value
    mask = (
        (x > hsize)
        & (x < (data_array.shape[1] - 1 - hsize))
        & (y > hsize)
        & (y < (data_array.shape[0] - 1 - hsize))
    )

    out_str = (
        f"Removed {np.count_nonzero(np.invert(mask))} objects "
        f"that were too close to the edges of the image."
    )
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            out_str,
            style_name="ITALIC",
            indent=indent + 1,
        )
    else:
        terminal_output.print_to_terminal(
            out_str,
            style_name="ITALIC",
            indent=indent + 1,
        )

    return table[mask]


_XY_CANDIDATES = (
    ("x_fit", "y_fit"),
    ("x_centroid", "y_centroid"),
    ("xcentroid", "ycentroid"),
    ("xfit", "yfit"),
    ("x", "y"),
)
_FINDER_QUALITY_COLUMNS = (
    "sharpness",
    "roundness",
    "roundness1",
    "roundness2",
    "fwhm",
    "npix",
    "peak",
    "sky",
)


def xy_column_names(table: Table) -> tuple[str, str] | None:
    """Return the first available pixel-coordinate column pair."""
    for xname, yname in _XY_CANDIDATES:
        if xname in table.colnames and yname in table.colnames:
            return xname, yname
    return None


def attach_finder_quality(
    photometry: Table,
    finder_table: Table | None,
    *,
    max_sep_pix: float = 3.0,
) -> Table:
    """Copy finder quality columns onto photometry rows by nearest (x, y).

    Photutils PSF tables already carry ``qfit`` / ``cfit`` when the fitter
    provides them; this adds DAO/IRAF sharpness (and related columns) that
    would otherwise stay only on ``image.positions``.
    """
    if finder_table is None or len(photometry) == 0 or len(finder_table) == 0:
        return photometry
    phot_xy = xy_column_names(photometry)
    find_xy = xy_column_names(finder_table)
    if phot_xy is None or find_xy is None:
        return photometry
    quality_cols = [
        c
        for c in _FINDER_QUALITY_COLUMNS
        if c in finder_table.colnames and c not in photometry.colnames
    ]
    if not quality_cols:
        return photometry

    px = np.asarray(photometry[phot_xy[0]], dtype=float)
    py = np.asarray(photometry[phot_xy[1]], dtype=float)
    fx = np.asarray(finder_table[find_xy[0]], dtype=float)
    fy = np.asarray(finder_table[find_xy[1]], dtype=float)
    phot_ok = np.isfinite(px) & np.isfinite(py)
    find_ok = np.isfinite(fx) & np.isfinite(fy)
    if not np.any(phot_ok) or not np.any(find_ok):
        return photometry
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return photometry

    find_idx = np.flatnonzero(find_ok)
    tree = cKDTree(np.column_stack([fx[find_ok], fy[find_ok]]))
    dist, local = tree.query(np.column_stack([px[phot_ok], py[phot_ok]]), k=1)
    matched = np.zeros(len(photometry), dtype=bool)
    idx = np.zeros(len(photometry), dtype=int)
    matched[phot_ok] = np.isfinite(dist) & (dist <= float(max_sep_pix))
    idx[phot_ok] = find_idx[np.asarray(local, dtype=int)]
    out = photometry.copy()
    for col in quality_cols:
        src = np.asarray(finder_table[col], dtype=float)
        dest = np.full(len(out), np.nan, dtype=float)
        dest[matched] = src[idx[matched]]
        out[col] = dest
    return out


def attach_sky_coords_from_wcs(photometry: Table, wcs) -> Table:
    """Add ``ra`` / ``dec`` (deg) from a WCS if they are not already present."""
    if wcs is None or "ra" in photometry.colnames:
        return photometry
    xy = xy_column_names(photometry)
    if xy is None:
        return photometry
    x = np.asarray(photometry[xy[0]], dtype=float)
    y = np.asarray(photometry[xy[1]], dtype=float)
    try:
        sky = wcs.pixel_to_world(x, y)
    except (ValueError, TypeError, AttributeError):
        return photometry
    out = photometry.copy()
    out["ra"] = np.asarray(sky.ra.deg, dtype=float)
    out["dec"] = np.asarray(sky.dec.deg, dtype=float)
    return out


__all__ = [
    "attach_finder_quality",
    "attach_sky_coords_from_wcs",
    "flux_to_magnitudes",
    "rm_edge_objects",
    "xy_column_names",
]
