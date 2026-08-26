"""Shared PSF FWHM estimation helpers."""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table


def select_sources_for_fwhm_fit(
    source_table: Table,
    *,
    data_shape: tuple[int, int] | None = None,
    fit_shape: int = 25,
    min_objects: int = 40,
    n_select: int = 25,
    flux_percentile_lo: float = 70.0,
    flux_percentile_hi: float = 97.0,
    min_separation_pix: float | None = None,
) -> Table:
    """
    Select unsaturated, preferably isolated sources for a robust FWHM fit.

    Prefer a high-but-not-saturated flux window (default 70–97 percentile) rather
    than a fixed index slice of the flux-sorted table. When ``data_shape`` is
    given, drop sources closer than ``fit_shape/2`` to the edge. When
    ``min_separation_pix`` is set, greedily keep only mutually isolated stars
    (brightest first within the flux window).
    """
    if source_table is None or len(source_table) == 0:
        return Table()

    table = source_table.copy()
    if "flux" not in table.colnames:
        return table

    # Column names for positions
    if "x_centroid" in table.colnames:
        x_col, y_col = "x_centroid", "y_centroid"
    elif "x" in table.colnames:
        x_col, y_col = "x", "y"
    else:
        return table

    x = np.asarray(table[x_col], dtype=float)
    y = np.asarray(table[y_col], dtype=float)
    flux = np.asarray(table["flux"], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(flux) & (flux > 0)
    table = table[ok]
    if len(table) == 0:
        return table

    x = np.asarray(table[x_col], dtype=float)
    y = np.asarray(table[y_col], dtype=float)
    flux = np.asarray(table["flux"], dtype=float)

    if data_shape is not None:
        ny, nx = int(data_shape[0]), int(data_shape[1])
        half = max(fit_shape, 3) / 2.0
        edge = (
            (x > half)
            & (x < nx - 1 - half)
            & (y > half)
            & (y < ny - 1 - half)
        )
        table = table[edge]
        if len(table) == 0:
            return table
        x = np.asarray(table[x_col], dtype=float)
        y = np.asarray(table[y_col], dtype=float)
        flux = np.asarray(table["flux"], dtype=float)

    # Prefer stars with finder quality flags when present (IRAF/DAO)
    round_col = _finder_roundness_column(table)
    if "sharpness" in table.colnames and round_col is not None:
        sharp = np.asarray(table["sharpness"], dtype=float)
        rnd = np.asarray(table[round_col], dtype=float)
        if round_col == "roundness1":
            round_ok = np.abs(rnd) < 0.5
        else:
            # IRAF moment ellipticity is ~[0, 1], not DAO's signed roundness.
            round_ok = (rnd >= 0.0) & (rnd < 0.5)
        quality = (
            np.isfinite(sharp)
            & np.isfinite(rnd)
            & (sharp > 0.2)
            & (sharp < 1.0)
            & round_ok
        )
        if np.sum(quality) >= max(8, n_select // 2):
            table = table[quality]
            x = np.asarray(table[x_col], dtype=float)
            y = np.asarray(table[y_col], dtype=float)
            flux = np.asarray(table["flux"], dtype=float)

    if len(table) >= min_objects:
        lo = np.percentile(flux, flux_percentile_lo)
        hi = np.percentile(flux, flux_percentile_hi)
        in_win = (flux >= lo) & (flux <= hi)
        if np.sum(in_win) >= max(8, n_select // 2):
            table = table[in_win]
            x = np.asarray(table[x_col], dtype=float)
            y = np.asarray(table[y_col], dtype=float)
            flux = np.asarray(table["flux"], dtype=float)

    # Brightest-first within window for isolation / truncation
    order = np.argsort(flux)[::-1]
    table = table[order]
    x = x[order]
    y = y[order]

    sep = min_separation_pix
    if sep is None:
        sep = float(max(fit_shape * 0.75, 8.0))

    if len(table) > 1 and sep > 0:
        keep: list[int] = []
        for i in range(len(table)):
            if len(keep) >= n_select:
                break
            if not keep:
                keep.append(i)
                continue
            dx = x[i] - x[keep]
            dy = y[i] - y[keep]
            if np.min(np.hypot(dx, dy)) >= sep:
                keep.append(i)
        if len(keep) >= max(5, n_select // 3):
            table = table[keep]
        else:
            table = table[:n_select]
    else:
        table = table[:n_select]

    return table


def _finder_roundness_column(table: Table) -> str | None:
    """DAO uses ``roundness1``; IRAF uses ``roundness``."""
    if "roundness1" in table.colnames:
        return "roundness1"
    if "roundness" in table.colnames:
        return "roundness"
    return None


def roundness_range_for_finder(
    method: str,
    roundness_range: tuple[float, float],
) -> tuple[float, float]:
    """
    Map a configured roundness window onto the finder that will use it.

    DAO roundness is signed (about ``[-1, 1]``). IRAF roundness is moment
    ellipticity (about ``[0, 1]``). Passing the DAO window to IRAF disables
    the cut and lets noise peaks through at a low detection threshold.
    """
    lo, hi = float(roundness_range[0]), float(roundness_range[1])
    if str(method).upper() != "IRAF":
        return (lo, hi)
    if lo < 0.0:
        # DAO-style signed range on IRAF → keep stellar, slightly looser than
        # photutils' default ``(0.0, 0.2)`` so faint stars with noisy moments
        # still pass.
        hi = 0.5 if hi >= 1.0 else max(hi, 0.0)
        return (0.0, hi)
    return (lo, hi)


def filter_finder_table_by_fwhm_scale(
    source_table: Table,
    fwhm_pix: float,
    *,
    scale_range: tuple[float, float] | None = (0.5, 2.0),
) -> tuple[Table, int]:
    """
    Keep finder rows whose ``fwhm`` is within ``scale_range × fwhm_pix``.

    Returns ``(filtered_table, n_removed)``. No-op when the table has no
    ``fwhm`` column or ``scale_range`` is ``None``. An empty filtered table
    means every row failed the cut; the caller should decide whether to keep
    the original catalog.
    """
    if (
        source_table is None
        or len(source_table) == 0
        or scale_range is None
        or "fwhm" not in source_table.colnames
    ):
        return source_table, 0
    lo_s, hi_s = float(scale_range[0]), float(scale_range[1])
    if not np.isfinite(fwhm_pix) or fwhm_pix <= 0.0 or hi_s <= lo_s:
        return source_table, 0
    vals = np.asarray(source_table["fwhm"], dtype=float)
    keep = np.isfinite(vals) & (vals >= lo_s * fwhm_pix) & (vals <= hi_s * fwhm_pix)
    n_keep = int(np.count_nonzero(keep))
    n_removed = int(len(source_table) - n_keep)
    if n_removed == 0:
        return source_table, 0
    return source_table[keep], n_removed


def source_positions_from_table(source_table: Table) -> list[tuple[float, float]]:
    """Return ``(x, y)`` positions from a photutils source table."""
    x_column, y_column = _table_xy_columns(source_table)
    return list(zip(source_table[x_column], source_table[y_column], strict=True))


def _table_xy_columns(source_table: Table) -> tuple[str, str]:
    if "x_centroid" in source_table.colnames:
        return "x_centroid", "y_centroid"
    if "x" in source_table.colnames:
        return "x", "y"
    raise ValueError(
        "Source table must contain x/y or x_centroid/y_centroid columns."
    )


def finite_cutout_star_mask(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    size: int,
    *,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    True where a ``size``×``size`` stamp around ``(x, y)`` is on-image and finite.

    Matches photutils ``extract_stars`` / ``Cutout2D`` windows (rounded center).
    Non-finite centroids, incomplete stamps, non-finite pixels, or ``True`` in
    ``extra_mask`` are rejected.
    """
    size = int(size)
    if size < 1:
        raise ValueError(f"cutout size must be >= 1, got {size}")

    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.size != ys.size:
        raise ValueError("x and y must have the same length.")

    data_f = np.asarray(data, dtype=float)
    ny, nx = data_f.shape[:2]
    bad = None if extra_mask is None else np.asarray(extra_mask, dtype=bool)
    if bad is not None and bad.shape != data_f.shape[:2]:
        raise ValueError(
            f"extra_mask shape {bad.shape} does not match data shape {data_f.shape[:2]}."
        )

    ok = np.zeros(xs.size, dtype=bool)
    half = size // 2
    for i in range(xs.size):
        if not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
            continue
        x0 = int(np.round(xs[i])) - half
        y0 = int(np.round(ys[i])) - half
        x1 = x0 + size
        y1 = y0 + size
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            continue
        stamp = data_f[y0:y1, x0:x1]
        if stamp.shape != (size, size) or not np.all(np.isfinite(stamp)):
            continue
        if bad is not None and np.any(bad[y0:y1, x0:x1]):
            continue
        ok[i] = True
    return ok


def filter_table_finite_cutouts(
    source_table: Table,
    data: np.ndarray,
    size: int,
    *,
    extra_mask: np.ndarray | None = None,
) -> tuple[Table, int]:
    """Drop stars whose ``extract_stars`` stamps would contain non-finite pixels.

    Returns ``(filtered_table, n_rejected)``.
    """
    if source_table is None or len(source_table) == 0:
        return source_table, 0
    x_column, y_column = _table_xy_columns(source_table)
    ok = finite_cutout_star_mask(
        data,
        source_table[x_column],
        source_table[y_column],
        size,
        extra_mask=extra_mask,
    )
    n_rejected = int(np.size(ok) - np.count_nonzero(ok))
    if n_rejected == 0:
        return source_table, 0
    return source_table[ok], n_rejected


def _odd_fit_shape(fwhm_guess: float, base: int = 25) -> int:
    """Choose an odd fit stamp size covering ~5×FWHM, at least ``base``."""
    need = int(np.ceil(max(base, 5.0 * float(fwhm_guess))))
    if need % 2 == 0:
        need += 1
    return min(need, 51)


def _aggregate_fwhm(
    fwhm_values: np.ndarray,
    *,
    min_fwhm: float,
    max_fwhm: float,
    min_valid: int,
) -> tuple[float | None, dict[str, float | int]]:
    """
    Aggregate per-star FWHMs: keep in-range values, sigma-clip, require ``min_valid``.
    """
    vals = np.asarray(fwhm_values, dtype=float)
    finite = vals[np.isfinite(vals) & (vals > 0)]
    in_range = finite[(finite >= min_fwhm) & (finite <= max_fwhm)]
    stats = {
        "n_fit": int(finite.size),
        "n_in_range": int(in_range.size),
        "raw_median": float(np.nanmedian(finite)) if finite.size else float("nan"),
    }
    if in_range.size >= min_valid:
        _, med, _ = sigma_clipped_stats(in_range, sigma=2.5, maxiters=5)
        return float(med), stats
    return None, stats


def estimate_fwhm_from_finder_table(
    source_table: Table,
    *,
    default_fwhm: float,
    min_fwhm: float = 2.0,
    max_fwhm: float = 15.0,
    min_valid: int = 5,
    data_shape: tuple[int, int] | None = None,
    fit_shape: int = 25,
) -> tuple[float, str | None, dict[str, float | int | str]]:
    """
    Estimate FWHM from an IRAFStarFinder ``fwhm`` column when available.

    Aggregates only quality-selected stars (same cuts as
    :func:`select_sources_for_fwhm_fit`). Returns
    ``(default_fwhm, reason, meta)`` if the column is missing or unusable.
    """
    meta: dict[str, float | int | str] = {
        "source": "default",
        "n_in_range": 0,
    }
    if source_table is None or "fwhm" not in source_table.colnames:
        return default_fwhm, "no finder fwhm column", meta

    selected = select_sources_for_fwhm_fit(
        source_table,
        data_shape=data_shape,
        fit_shape=fit_shape,
    )
    if len(selected) == 0 or "fwhm" not in selected.colnames:
        return default_fwhm, "no quality-selected stars with fwhm", meta

    vals = np.asarray(selected["fwhm"], dtype=float)
    med, stats = _aggregate_fwhm(
        vals, min_fwhm=min_fwhm, max_fwhm=max_fwhm, min_valid=min_valid
    )
    meta["n_in_range"] = int(stats["n_in_range"])
    if med is None:
        return default_fwhm, (
            f"finder FWHM unusable (n_fit={stats['n_fit']}, "
            f"n_in_range={stats['n_in_range']}, "
            f"raw_median={stats['raw_median']:.3f}, "
            f"allowed=[{min_fwhm}, {max_fwhm}])"
        ), meta
    # Pile-up at the allowed minimum is typical of noise/cosmic cores, not seeing.
    if med < float(min_fwhm) + 1.0:
        return default_fwhm, (
            f"finder FWHM {med:.2f} px is too close to the lower limit "
            f"({min_fwhm}); likely compact noise rather than stellar seeing"
        ), meta
    meta["source"] = "finder_column"
    return med, None, meta


def estimate_fwhm_from_positions(
    data: np.ndarray,
    xypos: list[tuple[float, float]] | np.ndarray,
    *,
    mask: np.ndarray | None = None,
    error: np.ndarray | None = None,
    default_fwhm: float,
    fit_shape: int = 25,
    min_fwhm: float = 2.0,
    max_fwhm: float = 15.0,
    min_valid: int = 5,
) -> tuple[float, str | None]:
    """
    Estimate PSF FWHM from source positions using ``photutils.psf.fit_fwhm``.

    Individual star fits outside ``[min_fwhm, max_fwhm]`` are discarded before
    aggregating (sigma-clipped median). Softens the previous behaviour that
    rejected the *entire* estimate when the raw median was slightly outside range.

    Returns
    -------
    fwhm
        Estimated FWHM in pixels, or ``default_fwhm`` on failure.
    error_message
        ``None`` on success, otherwise a short reason string.
    """
    if len(xypos) == 0:
        return default_fwhm, "no sources available for FWHM fit"

    from astropy.modeling.fitting import NonFiniteValueError
    from photutils.psf import fit_fwhm

    shape = _odd_fit_shape(default_fwhm, base=fit_shape)

    try:
        fwhm_values = fit_fwhm(
            data,
            xypos=xypos,
            fit_shape=shape,
            mask=mask,
            error=error,
        )
        fwhm_values = np.asarray(fwhm_values, dtype=float)
    except (ValueError, NonFiniteValueError) as exc:
        return default_fwhm, str(exc)

    med, stats = _aggregate_fwhm(
        fwhm_values, min_fwhm=min_fwhm, max_fwhm=max_fwhm, min_valid=min_valid
    )
    if med is None:
        return default_fwhm, (
            f"estimated FWHM unusable (n_fit={stats['n_fit']}, "
            f"n_in_range={stats['n_in_range']}, "
            f"raw_median={stats['raw_median']:.3f}, "
            f"allowed=[{min_fwhm}, {max_fwhm}])"
        )

    return med, None


def estimate_image_fwhm(
    data: np.ndarray,
    source_table: Table,
    *,
    mask: np.ndarray | None = None,
    error: np.ndarray | None = None,
    default_fwhm: float,
    fit_shape: int = 25,
    min_fwhm: float = 2.0,
    max_fwhm: float = 15.0,
    min_valid: int = 5,
) -> tuple[float, str | None, dict[str, float | int | str]]:
    """
    Robust image FWHM: prefer quality-filtered finder ``fwhm``, else PSF fits.

    Returns ``(fwhm, error_or_none, meta)`` where ``meta`` has ``source``
    (``finder_column`` / ``psf_fit`` / ``default``) and ``n_in_range``.
    """
    data_shape = tuple(int(s) for s in np.asarray(data).shape[:2])

    # 1) IRAF (or similar) per-source FWHM on quality-selected stars
    fwhm_finder, err_finder, meta = estimate_fwhm_from_finder_table(
        source_table,
        default_fwhm=default_fwhm,
        min_fwhm=min_fwhm,
        max_fwhm=max_fwhm,
        min_valid=min_valid,
        data_shape=data_shape,
        fit_shape=fit_shape,
    )
    if err_finder is None:
        return fwhm_finder, None, meta

    # 2) PSF fit on cleaned star sample
    selected = select_sources_for_fwhm_fit(
        source_table,
        data_shape=data_shape,
        fit_shape=fit_shape,
    )
    if len(selected) == 0:
        meta["source"] = "default"
        return default_fwhm, err_finder, meta

    xy_pos = source_positions_from_table(selected)
    fwhm_fit, err_fit = estimate_fwhm_from_positions(
        data,
        xy_pos,
        mask=mask,
        error=error,
        default_fwhm=default_fwhm,
        fit_shape=fit_shape,
        min_fwhm=min_fwhm,
        max_fwhm=max_fwhm,
        min_valid=min_valid,
    )
    if err_fit is None:
        # Count in-range from a fresh aggregate is not returned by positions
        # helper; report selected-star count as a proxy for n used in the fit.
        meta = {"source": "psf_fit", "n_in_range": len(selected)}
        return fwhm_fit, None, meta

    meta = {"source": "default", "n_in_range": 0}
    return default_fwhm, f"{err_finder}; {err_fit}", meta
