"""Comparison-star flags on photometry tables."""

from __future__ import annotations

import numpy as np
from astropy.table import Table


def flag_comparison_stars(table: Table) -> Table:
    """Set ``is_comparison`` where a catalog match or any ``mag_std_*`` is finite."""
    n = len(table)
    flag = np.zeros(n, dtype=bool)
    if "match_sep_arcsec" in table.colnames:
        flag |= np.isfinite(np.asarray(table["match_sep_arcsec"], dtype=float))
    for col in table.colnames:
        if str(col).startswith("mag_std_"):
            flag |= np.isfinite(np.asarray(table[col], dtype=float))
    table["is_comparison"] = flag
    return table


def mark_used_calibrators(
    table: Table,
    filters: list[str],
    *,
    transformations: dict | None = None,
    sigma_clip: float | None = None,
    exact_masks: dict[str, np.ndarray] | None = None,
) -> Table:
    """Flag catalog candidates and the stars actually used in the calibration fit.

    Writes ``is_comparison`` (catalog match / finite ``mag_std_*``) and
    ``is_calibrator_<filter>`` (used in that band's fit). Exact masks from the
    fitter win when they match the table length; otherwise residuals vs the
    adopted T/ZP are clipped with ``sigma_clip`` × RMS when those are available.
    """
    flag_comparison_stars(table)
    n = len(table)
    transformations = transformations or {}
    exact_masks = exact_masks or {}
    for filter_ in filters:
        used = np.zeros(n, dtype=bool)
        exact = exact_masks.get(filter_)
        if exact is not None:
            arr = np.asarray(exact, dtype=bool).ravel()
            if arr.size == n:
                table[f"is_calibrator_{filter_}"] = arr
                continue
        inst_col, std_col = f"mag_{filter_}", f"mag_std_{filter_}"
        if inst_col not in table.colnames or std_col not in table.colnames:
            table[f"is_calibrator_{filter_}"] = used
            continue
        m_inst = np.asarray(table[inst_col], dtype=float)
        m_std = np.asarray(table[std_col], dtype=float)
        cand = np.isfinite(m_inst) & np.isfinite(m_std)
        tc = transformations.get(filter_)
        if (
            tc is None
            or sigma_clip is None
            or float(getattr(tc, "rms_residual", 0.0) or 0.0) <= 0.0
        ):
            table[f"is_calibrator_{filter_}"] = cand
            continue
        color = np.zeros(n, dtype=float)
        ci = getattr(tc, "color_index_filters", None)
        if ci and len(ci) == 2:
            c1, c2 = f"mag_std_{ci[0]}", f"mag_std_{ci[1]}"
            if c1 in table.colnames and c2 in table.colnames:
                color = np.asarray(table[c1], dtype=float) - np.asarray(
                    table[c2], dtype=float
                )
                cand &= np.isfinite(color)
        residual = (m_std - m_inst) - (
            float(tc.color_term) * color + float(tc.zero_point)
        )
        rms = float(tc.rms_residual)
        clipped = cand & np.isfinite(residual) & (np.abs(residual) < float(sigma_clip) * rms)
        table[f"is_calibrator_{filter_}"] = clipped if np.any(clipped) else cand
    return table


__all__ = ["flag_comparison_stars", "mark_used_calibrators"]
