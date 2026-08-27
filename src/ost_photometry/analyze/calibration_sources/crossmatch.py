"""
Sky cross-match between extracted sources and a standard-schema calibration catalog.

The reference catalog must contain ``ra``, ``dec`` in degrees and arbitrary
numeric columns (e.g. ``mag_std_B``, …) that are copied onto the source table for
matched rows. Non-numeric columns (e.g. string IDs) are skipped to avoid dtype issues.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, matching
from astropy.table import Table

from .. import utilities
from .flags import flag_comparison_stars


def second_nearest_separations(
    n_sources: int,
    idx_src,
    sep_arcsec,
) -> np.ndarray:
    """Second-smallest pairing separation per source (NaN if only one catalog hit)."""
    out = np.full(int(n_sources), np.nan)
    if n_sources <= 0:
        return out
    idx = np.asarray(idx_src, dtype=int)
    seps = np.asarray(sep_arcsec, dtype=float)
    if idx.size == 0:
        return out
    order = np.argsort(seps, kind="mergesort")
    seen = np.zeros(n_sources, dtype=np.uint8)
    for i, s in zip(idx[order], seps[order], strict=True):
        if i < 0 or i >= n_sources or not np.isfinite(s):
            continue
        if seen[i] == 0:
            seen[i] = 1
        elif seen[i] == 1:
            out[i] = float(s)
            seen[i] = 2
    return out


def crossmatch_standard_catalog(
    sources: Table,
    catalog: Table,
    ra_col: str = "ra",
    dec_col: str = "dec",
    match_radius: u.Quantity = 2.0 * u.arcsec,
) -> Table:
    """
    One-to-one sky match: each source at most one catalog row (closest wins).

    Algorithm (same idea as the former ``APASSCatalog.crossmatch``):

    1. ``search_around_sky`` finds all pairs within ``match_radius``.
    2. ``clear_duplicates`` first by source index (keep smallest separation per source).
    3. ``clear_duplicates`` again by catalog index (keep smallest separation per catalog star).
       This yields a **one-to-one** assignment when possible.

    Copies all **numeric** catalog columns onto ``sources`` (except duplicate ``ra``/``dec``).
    Adds ``match_sep_arcsec`` (NaN = no match), ``match_sep2_arcsec`` (second-nearest
    catalog star before the one-to-one assignment), ``ra_cat`` / ``dec_cat`` (matched
    catalog coordinates, deg), and ``is_comparison``.

    Parameters
    ----------
    sources
        Extracted objects; must have ``ra_col``, ``dec_col`` in degrees.
    catalog
        Standard calibration table from :func:`fetch_standard_calibration_catalog`.
    match_radius
        Astropy quantity (default 2 arcsec), passed to ``search_around_sky``.
    """
    # Empty catalog: preserve row count, flag no matches
    if len(catalog) == 0:
        result = sources.copy()
        result["match_sep_arcsec"] = np.full(len(result), np.nan)
        result["match_sep2_arcsec"] = np.full(len(result), np.nan)
        result["ra_cat"] = np.full(len(result), np.nan)
        result["dec_cat"] = np.full(len(result), np.nan)
        return flag_comparison_stars(result)

    source_coords = SkyCoord(sources[ra_col].ravel(), sources[dec_col].ravel(), unit="deg")
    cat_coords = SkyCoord(catalog["ra"].ravel(), catalog["dec"].ravel(), unit="deg")

    idx_src, idx_cat, sep, _ = matching.search_around_sky(
        source_coords, cat_coords, match_radius
    )
    sep_arcsec = np.asarray(sep.arcsec, dtype=float)

    # No pairs: add NaN columns for every numeric catalog field (downstream code may expect them)
    if len(idx_src) == 0:
        result = sources.copy()
        result["match_sep_arcsec"] = np.full(len(result), np.nan)
        result["match_sep2_arcsec"] = np.full(len(result), np.nan)
        result["ra_cat"] = np.full(len(result), np.nan)
        result["dec_cat"] = np.full(len(result), np.nan)
        for col in catalog.colnames:
            if col in ("ra", "dec") or not np.issubdtype(
                catalog[col].dtype, np.number
            ):
                continue
            result[col] = np.full(len(result), np.nan, dtype=float)
        return flag_comparison_stars(result)

    sep2_for_source = second_nearest_separations(len(sources), idx_src, sep_arcsec)

    # Enforce one-to-one: closest catalog star per source, then closest source per star
    idx_src, sep_arcsec, idx_cat = utilities.clear_duplicates(
        idx_src, sep_arcsec, idx_cat
    )
    idx_cat, sep_arcsec, idx_src = utilities.clear_duplicates(
        idx_cat, sep_arcsec, idx_src
    )

    # Dense mapping: for each source row, catalog index or -1
    cat_idx_for_source = np.full(len(sources), -1, dtype=np.intp)
    sep_for_source = np.full(len(sources), np.nan)
    cat_idx_for_source[idx_src] = idx_cat
    sep_for_source[idx_src] = sep_arcsec
    good_match = cat_idx_for_source >= 0

    result = sources.copy()
    result["match_sep_arcsec"] = sep_for_source
    result["match_sep2_arcsec"] = sep2_for_source
    ra_cat = np.full(len(result), np.nan, dtype=float)
    dec_cat = np.full(len(result), np.nan, dtype=float)
    if np.any(good_match):
        cat_rows = cat_idx_for_source[good_match]
        ra_vals = catalog["ra"][cat_rows]
        dec_vals = catalog["dec"][cat_rows]
        if hasattr(ra_vals, "value"):
            ra_vals = ra_vals.value
        if hasattr(dec_vals, "value"):
            dec_vals = dec_vals.value
        ra_cat[good_match] = np.asarray(ra_vals, dtype=float)
        dec_cat[good_match] = np.asarray(dec_vals, dtype=float)
    result["ra_cat"] = ra_cat
    result["dec_cat"] = dec_cat

    for col in catalog.colnames:
        if col in ("ra", "dec"):
            continue
        if not np.issubdtype(catalog[col].dtype, np.number):
            continue
        matched_vals = catalog[col][cat_idx_for_source[good_match]]
        if hasattr(matched_vals, "value"):
            matched_vals = np.asarray(matched_vals.value, dtype=float)
        else:
            matched_vals = np.asarray(matched_vals, dtype=float)
        new_col = np.full(len(result), np.nan, dtype=float)
        new_col[good_match] = matched_vals
        result[col] = new_col
    return flag_comparison_stars(result)
