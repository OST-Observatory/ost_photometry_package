"""Light-curve plots from the long ``light_curves.ecsv`` table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries

from ...output_layout import diagnostics_dir, results_dir
from ..post_processing.magnitude_systems import magnitude_system_axis_suffix

plt.switch_backend("Agg")

JD_MINUS_OFFSET = 2450000.0

_FIGSIZE = (10.0, 4.8)
_TITLE_FS = 12
_LABEL_FS = 11
_TICK_FS = 10
_NIGHT_MARKERS = (".", "o", "s", "D", "^", "v", "P", "X")


def _sanitize_filename(name: str) -> str:
    text = str(name).strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in text) or "object"


def _lightcurve_dir(output_dir: str, subdirectory: str = "") -> Path:
    extra = subdirectory.strip("/")
    if extra:
        return results_dir(output_dir, "lightcurves", extra)
    return results_dir(output_dir, "lightcurves")


def _nonnegative_errorbar_yerr(values) -> np.ndarray:
    return np.abs(np.asarray(values, dtype=float))


def fold_phase(time_jd: np.ndarray, t0_jd: float, period: float) -> np.ndarray:
    """Phase in ``[0, 1)``: ``((t - t0) / P) mod 1``."""
    t = np.asarray(time_jd, dtype=float)
    p = float(period)
    if not np.isfinite(p) or p <= 0.0:
        return np.full(t.shape, np.nan)
    return np.mod((t - float(t0_jd)) / p, 1.0)


def y_limits_for_quantity(
    y: np.ndarray,
    *,
    quantity: str = "magnitude",
    n_sigma: float = 5.5,
    min_half: float = 0.05,
) -> tuple[float, float]:
    """
    Axis limits from median ± MAD (not min/max outliers).

    Magnitudes return ``(hi, lo)`` so callers can ``set_ylim`` inverted.
    Flux returns ``(lo, hi)``.
    """
    arr = np.asarray(y, dtype=float)
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return (1.0, 0.0) if quantity != "flux" else (0.0, 1.0)
    center = float(np.median(fin))
    mad = float(np.median(np.abs(fin - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        lo_c, hi_c = np.percentile(fin, [10.0, 90.0])
        scale = max(0.5 * (float(hi_c) - float(lo_c)), 0.0)
    half = max(float(min_half), float(n_sigma) * scale)
    lo, hi = center - half, center + half
    if quantity == "flux":
        return lo, hi
    return hi, lo


def _series_y(tbl: Table) -> tuple[np.ndarray, np.ndarray, str]:
    qty = "magnitude"
    if "quantity" in tbl.colnames:
        qarr = np.asarray(tbl["quantity"]).astype(str)
        if np.any(qarr == "flux"):
            qty = "flux"
    if qty == "flux":
        y = np.asarray(tbl["flux"], dtype=float)
        e = np.abs(np.asarray(tbl["flux_err"], dtype=float))
    else:
        y = np.asarray(tbl["mag"], dtype=float)
        e = np.abs(np.asarray(tbl["mag_err"], dtype=float))
    return y, e, qty


def _plot_time_and_label(
    tbl: Table,
    time_scale: str = "bjd_tdb",
) -> tuple[np.ndarray, str]:
    jd = np.asarray(tbl["jd"], dtype=float)
    bjd = (
        np.asarray(tbl["bjd_tdb"], dtype=float)
        if "bjd_tdb" in tbl.colnames
        else np.full(len(tbl), np.nan)
    )
    if time_scale == "bjd_tdb" and np.any(np.isfinite(bjd)):
        x = np.where(np.isfinite(bjd), bjd, jd) - JD_MINUS_OFFSET
        return x, "BJD_TDB − 2450000"
    return jd - JD_MINUS_OFFSET, "JD − 2450000"


def _draw_night_bands(ax, x: np.ndarray, night_id: np.ndarray | None) -> None:
    if night_id is None:
        return
    nid = np.asarray(night_id)
    xx = np.asarray(x, dtype=float)
    ok = np.isfinite(xx)
    if not np.any(ok):
        return
    for night in np.unique(nid[ok]):
        if int(night) < 0:
            continue
        m = ok & (nid == night)
        if not np.any(m):
            continue
        ax.axvspan(float(np.min(xx[m])), float(np.max(xx[m])), color="0.92", zorder=0)


def _ylabel(filter_: str, quantity: str, magnitude_system: str) -> str:
    if quantity == "flux":
        return f"{filter_} [flux] (normalized)"
    return f"{filter_}{magnitude_system_axis_suffix(magnitude_system)}"


def _scatter_with_flags(
    ax,
    x,
    y,
    yerr,
    flag,
    *,
    quantity: str,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    e = _nonnegative_errorbar_yerr(yerr)
    if flag is None:
        flag = np.zeros(x.size, dtype=bool)
    else:
        flag = np.asarray(flag, dtype=bool)
    good = ~flag
    if np.any(good):
        ax.errorbar(
            x[good],
            y[good],
            yerr=e[good],
            marker=".",
            markersize=4,
            linestyle="none",
            capsize=2,
            ecolor="dodgerblue",
            color="darkred",
            zorder=3,
        )
    if np.any(flag):
        ax.errorbar(
            x[flag],
            y[flag],
            yerr=e[flag],
            marker="o",
            markersize=4,
            linestyle="none",
            capsize=2,
            ecolor="0.7",
            color="0.55",
            markerfacecolor="none",
            zorder=2,
            label="outlier",
        )
    lo_hi = y_limits_for_quantity(y[good] if np.any(good) else y, quantity=quantity)
    ax.set_ylim(*lo_hi)


def night_date_label(jd_values: np.ndarray) -> str:
    """UTC calendar date of the median finite JD (``YYYY-MM-DD``)."""
    arr = np.asarray(jd_values, dtype=float)
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return "unknown"
    t = Time(float(np.median(fin)), format="jd", scale="utc")
    return t.iso[:10]


def unique_night_ids(tbl: Table) -> np.ndarray:
    """Sorted ``night_id`` values, skipping the missing-night sentinel ``-1``."""
    if "night_id" not in tbl.colnames:
        return np.array([], dtype=np.int64)
    nid = np.asarray(tbl["night_id"]).astype(np.int64)
    vals = np.unique(nid[nid >= 0])
    return vals


def _night_color(index: int) -> tuple:
    cmap = plt.get_cmap("tab10")
    return cmap(int(index) % 10)


def _night_marker(index: int) -> str:
    return _NIGHT_MARKERS[int(index) % len(_NIGHT_MARKERS)]


def _scatter_nights(
    ax,
    x,
    y,
    yerr,
    night_id,
    flag,
    *,
    quantity: str,
    night_order: np.ndarray,
    labels: dict[int, str],
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    e = _nonnegative_errorbar_yerr(yerr)
    nid = np.asarray(night_id).astype(np.int64)
    if flag is None:
        flag = np.zeros(x.size, dtype=bool)
    else:
        flag = np.asarray(flag, dtype=bool)
    for i, night in enumerate(night_order):
        m = nid == int(night)
        if not np.any(m):
            continue
        color = _night_color(i)
        marker = _night_marker(i)
        good = m & ~flag
        bad = m & flag
        if np.any(good):
            ax.errorbar(
                x[good],
                y[good],
                yerr=e[good],
                marker=marker,
                markersize=5 if marker != "." else 4,
                linestyle="none",
                capsize=2,
                color=color,
                ecolor=color,
                zorder=3,
                label=labels.get(int(night), str(int(night))),
            )
        if np.any(bad):
            ax.errorbar(
                x[bad],
                y[bad],
                yerr=e[bad],
                marker=marker,
                markersize=4,
                linestyle="none",
                capsize=2,
                color="0.55",
                ecolor="0.7",
                markerfacecolor="none",
                zorder=2,
                label="outlier" if i == 0 else None,
            )
    good_y = y[~flag] if np.any(~flag) else y
    ax.set_ylim(*y_limits_for_quantity(good_y, quantity=quantity))


def _night_labels_from_table(tbl: Table, night_order: np.ndarray) -> dict[int, str]:
    jd = np.asarray(tbl["jd"], dtype=float)
    nid = np.asarray(tbl["night_id"]).astype(np.int64)
    out: dict[int, str] = {}
    for night in night_order:
        out[int(night)] = night_date_label(jd[nid == int(night)])
    return out


def light_curve_nights_jd_from_table(
    tbl: Table,
    output_dir: str,
    *,
    name_object: str,
    filter_: str,
    file_type: str = "pdf",
    subdirectory: str = "",
    time_scale: str = "bjd_tdb",
    magnitude_system: str = "vega",
    ylabel: str | None = None,
) -> Path:
    """JD light curve with one colour/marker per local night."""
    y, e, qty = _series_y(tbl)
    x, xlabel = _plot_time_and_label(tbl, time_scale)
    flag = np.asarray(tbl["flag_outlier"], dtype=bool) if "flag_outlier" in tbl.colnames else None
    nights = unique_night_ids(tbl)
    if nights.size == 0:
        return light_curve_jd_from_table(
            tbl,
            output_dir,
            name_object=name_object,
            filter_=filter_,
            file_type=file_type,
            subdirectory=subdirectory,
            time_scale=time_scale,
            show_airmass=False,
            magnitude_system=magnitude_system,
        )
    labels = _night_labels_from_table(tbl, nights)
    nid = np.asarray(tbl["night_id"])

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    _scatter_nights(
        ax, x, y, e, nid, flag, quantity=qty, night_order=nights, labels=labels
    )
    ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
    ax.set_ylabel(ylabel or _ylabel(filter_, qty, magnitude_system), fontsize=_LABEL_FS)
    ax.set_title(
        f"Nights — {_sanitize_filename(name_object)} ({filter_})",
        fontsize=_TITLE_FS,
    )
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(True, color="lightgray", linestyle="--")
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=min(4, int(nights.size)))
    fig.tight_layout()
    plot_dir = _lightcurve_dir(output_dir, subdirectory)
    path = plot_dir / (
        f"lightcurve_nights_jd_{_sanitize_filename(name_object)}"
        f"_{_sanitize_filename(filter_)}.{file_type}"
    )
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


def light_curve_nights_fold_from_table(
    tbl: Table,
    output_dir: str,
    *,
    transit_time: str,
    period: float,
    name_object: str,
    filter_: str,
    file_type: str = "pdf",
    subdirectory: str = "",
    time_scale: str = "bjd_tdb",
    phase_cycles: int = 1,
    magnitude_system: str = "vega",
    ylabel: str | None = None,
) -> Path:
    """Folded light curve with one colour/marker per local night."""
    y, e, qty = _series_y(tbl)
    jd = np.asarray(tbl["jd"], dtype=float)
    bjd = (
        np.asarray(tbl["bjd_tdb"], dtype=float)
        if "bjd_tdb" in tbl.colnames
        else np.full(len(tbl), np.nan)
    )
    if time_scale == "bjd_tdb" and np.any(np.isfinite(bjd)):
        t_use = np.where(np.isfinite(bjd), bjd, jd)
    else:
        t_use = jd
    t0 = Time(transit_time, format="isot", scale="utc")
    t0_jd = float(t0.tdb.jd) if time_scale == "bjd_tdb" else float(t0.jd)
    phase = fold_phase(t_use, t0_jd, period)
    flag = np.asarray(tbl["flag_outlier"], dtype=bool) if "flag_outlier" in tbl.colnames else None
    nights = unique_night_ids(tbl)
    labels = _night_labels_from_table(tbl, nights) if nights.size else {}
    nid = np.asarray(tbl["night_id"]) if "night_id" in tbl.colnames else np.zeros(len(tbl), dtype=int)
    cycles = 2 if int(phase_cycles) >= 2 else 1

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.axvline(0.0, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
    if cycles == 2:
        ax.axvline(1.0, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
    if nights.size == 0:
        _scatter_with_flags(ax, phase, y, e, flag, quantity=qty)
        if cycles == 2:
            _scatter_with_flags(ax, phase + 1.0, y, e, flag, quantity=qty)
    else:
        for wrap in range(cycles):
            _scatter_nights(
                ax,
                phase + wrap,
                y,
                e,
                nid,
                flag,
                quantity=qty,
                night_order=nights,
                labels=labels if wrap == 0 else {int(n): "_nolegend_" for n in nights},
            )
    ax.set_xlim(-0.02, float(cycles) + 0.02)
    ax.set_xlabel("Phase", fontsize=_LABEL_FS)
    ax.set_ylabel(ylabel or _ylabel(filter_, qty, magnitude_system), fontsize=_LABEL_FS)
    ax.set_title(
        f"Nights folded — {_sanitize_filename(name_object)} ({filter_}); "
        f"P = {float(period):.6g} d",
        fontsize=_TITLE_FS,
    )
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(True, color="lightgray", linestyle="--")
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=min(4, max(1, int(nights.size))))
    fig.tight_layout()
    plot_dir = _lightcurve_dir(output_dir, subdirectory)
    path = plot_dir / (
        f"lightcurve_nights_folded_{_sanitize_filename(name_object)}"
        f"_{_sanitize_filename(filter_)}.{file_type}"
    )
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


def light_curve_nights_panels_from_table(
    tbl: Table,
    output_dir: str,
    *,
    name_object: str,
    filter_: str,
    file_type: str = "pdf",
    subdirectory: str = "",
    time_scale: str = "bjd_tdb",
    magnitude_system: str = "vega",
    ylabel: str | None = None,
) -> Path:
    """One JD panel per night, shared magnitude/flux limits."""
    y, e, qty = _series_y(tbl)
    nights = unique_night_ids(tbl)
    if nights.size == 0:
        return light_curve_nights_jd_from_table(
            tbl,
            output_dir,
            name_object=name_object,
            filter_=filter_,
            file_type=file_type,
            subdirectory=subdirectory,
            time_scale=time_scale,
            magnitude_system=magnitude_system,
            ylabel=ylabel,
        )
    labels = _night_labels_from_table(tbl, nights)
    nid = np.asarray(tbl["night_id"]).astype(np.int64)
    flag_all = (
        np.asarray(tbl["flag_outlier"], dtype=bool)
        if "flag_outlier" in tbl.colnames
        else np.zeros(len(tbl), dtype=bool)
    )
    ylim = y_limits_for_quantity(y[~flag_all] if np.any(~flag_all) else y, quantity=qty)
    n = int(nights.size)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(10.0, max(3.0, 2.6 * n)),
        sharey=True,
        squeeze=False,
    )
    y_lab = ylabel or _ylabel(filter_, qty, magnitude_system)
    x_label = None
    for ax, night in zip(axes[:, 0], nights, strict=True):
        m = nid == int(night)
        sub = tbl[m]
        x, x_label = _plot_time_and_label(sub, time_scale)
        yi, ei, _qty = _series_y(sub)
        fl = np.asarray(sub["flag_outlier"], dtype=bool) if "flag_outlier" in sub.colnames else None
        _scatter_with_flags(ax, x, yi, ei, fl, quantity=qty)
        ax.set_ylim(*ylim)
        ax.set_ylabel(y_lab, fontsize=9)
        ax.set_title(labels[int(night)], fontsize=10, loc="left")
        ax.tick_params(labelsize=9)
        ax.grid(True, color="lightgray", linestyle="--")
    axes[-1, 0].set_xlabel(x_label or "", fontsize=_LABEL_FS)
    fig.suptitle(
        f"Nights — {_sanitize_filename(name_object)} ({filter_})",
        fontsize=_TITLE_FS,
        y=1.01,
    )
    fig.tight_layout()
    plot_dir = _lightcurve_dir(output_dir, subdirectory)
    path = plot_dir / (
        f"lightcurve_nights_panels_{_sanitize_filename(name_object)}"
        f"_{_sanitize_filename(filter_)}.{file_type}"
    )
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


def light_curve_jd_from_table(
    tbl: Table,
    output_dir: str,
    *,
    name_object: str,
    filter_: str,
    file_type: str = "pdf",
    subdirectory: str = "",
    time_scale: str = "bjd_tdb",
    show_airmass: bool = True,
    magnitude_system: str = "vega",
) -> Path:
    """JD (or BJD) light curve for one source/filter slice of the long table."""
    y, e, qty = _series_y(tbl)
    x, xlabel = _plot_time_and_label(tbl, time_scale)
    flag = np.asarray(tbl["flag_outlier"], dtype=bool) if "flag_outlier" in tbl.colnames else None
    night = np.asarray(tbl["night_id"]) if "night_id" in tbl.colnames else None
    am = np.asarray(tbl["airmass"], dtype=float) if "airmass" in tbl.colnames else None

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    _draw_night_bands(ax, x, night)
    _scatter_with_flags(ax, x, y, e, flag, quantity=qty)
    ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
    ax.set_ylabel(_ylabel(filter_, qty, magnitude_system), fontsize=_LABEL_FS)
    ax.set_title(f"Light curve — {_sanitize_filename(name_object)} ({filter_})", fontsize=_TITLE_FS)
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(True, color="lightgray", linestyle="--")
    if show_airmass and am is not None and np.any(np.isfinite(am)):
        ax2 = ax.twinx()
        ax2.plot(x, am, color="0.45", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.set_ylabel("Airmass", fontsize=_LABEL_FS, color="0.35")
        ax2.tick_params(labelsize=_TICK_FS, colors="0.35")
        ax2.set_ylim(bottom=max(0.9, float(np.nanmin(am)) - 0.05))
    fig.tight_layout()
    plot_dir = _lightcurve_dir(output_dir, subdirectory)
    path = plot_dir / (
        f"lightcurve_jd_{_sanitize_filename(name_object)}_{_sanitize_filename(filter_)}.{file_type}"
    )
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


def _phase_bin_centers_and_means(
    phase: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    binning_factor: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if binning_factor is None or not np.isfinite(binning_factor) or binning_factor <= 0:
        return None
    ok = np.isfinite(phase) & np.isfinite(y)
    n_ok = int(np.count_nonzero(ok))
    if n_ok < 4:
        return None
    if float(binning_factor) < 1.0:
        n_bins = max(2, int(round(1.0 / float(binning_factor))))
    else:
        n_bins = max(2, int(round(float(binning_factor))))
    n_bins = min(n_bins, n_ok)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(phase[ok], edges) - 1, 0, n_bins - 1)
    yv = y[ok]
    ev = _nonnegative_errorbar_yerr(yerr[ok])
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    errs = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        means[b] = float(np.nanmean(yv[m]))
        if np.any(np.isfinite(ev[m])) and int(np.count_nonzero(m)) > 1:
            errs[b] = float(np.nanstd(yv[m]) / np.sqrt(np.count_nonzero(m)))
        else:
            errs[b] = float(np.nanmedian(ev[m])) if np.any(np.isfinite(ev[m])) else 0.0
    keep = np.isfinite(means)
    if not np.any(keep):
        return None
    return centers[keep], means[keep], errs[keep]


def light_curve_fold_from_table(
    tbl: Table,
    output_dir: str,
    *,
    transit_time: str,
    period: float,
    name_object: str,
    filter_: str,
    file_type: str = "pdf",
    subdirectory: str = "",
    binning_factor: float | None = None,
    time_scale: str = "bjd_tdb",
    phase_cycles: int = 1,
    magnitude_system: str = "vega",
) -> Path:
    """Folded light curve vs phase (0–1, optionally wrapped to 0–2)."""
    y, e, qty = _series_y(tbl)
    jd = np.asarray(tbl["jd"], dtype=float)
    bjd = (
        np.asarray(tbl["bjd_tdb"], dtype=float)
        if "bjd_tdb" in tbl.colnames
        else np.full(len(tbl), np.nan)
    )
    if time_scale == "bjd_tdb" and np.any(np.isfinite(bjd)):
        t_use = np.where(np.isfinite(bjd), bjd, jd)
    else:
        t_use = jd
    t0 = Time(transit_time, format="isot", scale="utc")
    t0_jd = float(t0.tdb.jd) if time_scale == "bjd_tdb" else float(t0.jd)
    phase = fold_phase(t_use, t0_jd, period)
    flag = np.asarray(tbl["flag_outlier"], dtype=bool) if "flag_outlier" in tbl.colnames else None
    cycles = 2 if int(phase_cycles) >= 2 else 1

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.axvline(0.0, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
    if cycles == 2:
        ax.axvline(1.0, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
    for wrap in range(cycles):
        _scatter_with_flags(
            ax,
            phase + wrap,
            y,
            e,
            flag,
            quantity=qty,
        )
    binned = _phase_bin_centers_and_means(phase, y, e, binning_factor)
    if binned is not None:
        cx, cy, ce = binned
        for wrap in range(cycles):
            ax.errorbar(
                cx + wrap,
                cy,
                yerr=ce,
                marker="o",
                markersize=5,
                linestyle="none",
                color="black",
                ecolor="black",
                capsize=2,
                zorder=4,
                label="phase bins" if wrap == 0 else None,
            )
    ax.set_xlim(-0.02, float(cycles) + 0.02)
    ax.set_xlabel("Phase", fontsize=_LABEL_FS)
    ax.set_ylabel(_ylabel(filter_, qty, magnitude_system), fontsize=_LABEL_FS)
    ax.set_title(
        f"Folded — {_sanitize_filename(name_object)} ({filter_}); "
        f"P = {float(period):.6g} d",
        fontsize=_TITLE_FS,
    )
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(True, color="lightgray", linestyle="--")
    fig.tight_layout()
    plot_dir = _lightcurve_dir(output_dir, subdirectory)
    path = plot_dir / (
        f"lightcurve_folded_{_sanitize_filename(name_object)}_{_sanitize_filename(filter_)}"
        f".{file_type}"
    )
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


def plot_check_star_qc(
    lc: Table,
    output_dir: str,
    *,
    filter_: str,
    ooi_ids: list[tuple[int, str]],
    calibrator_ids: list[int],
    file_type: str = "pdf",
    time_scale: str = "bjd_tdb",
    show_airmass: bool = False,
    magnitude_system: str = "vega",
) -> Path | None:
    """OOI plus the most variable catalog calibrators (OOI ids are not repeated)."""
    from ..post_processing.light_curve import build_check_star_qc_panels

    panels = build_check_star_qc_panels(lc, filter_, ooi_ids, calibrator_ids)
    if not panels:
        return None
    n_cal = sum(1 for title, _sub in panels if title.startswith("catalog calibrator"))
    n = len(panels)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(10.0, max(3.2, 2.4 * n)),
        sharex=True,
        squeeze=False,
    )
    x_label = None
    for ax, (title, sub) in zip(axes[:, 0], panels, strict=True):
        y, e, qty = _series_y(sub)
        x, x_label = _plot_time_and_label(sub, time_scale)
        night = np.asarray(sub["night_id"]) if "night_id" in sub.colnames else None
        flag = (
            np.asarray(sub["flag_outlier"], dtype=bool)
            if "flag_outlier" in sub.colnames
            else None
        )
        _draw_night_bands(ax, x, night)
        _scatter_with_flags(ax, x, y, e, flag, quantity=qty)
        ax.set_ylabel(_ylabel(filter_, qty, magnitude_system), fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.tick_params(labelsize=9)
        ax.grid(True, color="lightgray", linestyle="--")
        if qty != "flux" and np.any(np.isfinite(y)):
            fin = y[np.isfinite(y)]
            rms = float(np.sqrt(np.mean((fin - np.median(fin)) ** 2)))
            ax.text(
                0.99,
                0.05,
                f"RMS = {rms:.4f} mag",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )
        if show_airmass and "airmass" in sub.colnames:
            am = np.asarray(sub["airmass"], dtype=float)
            if np.any(np.isfinite(am)):
                ax2 = ax.twinx()
                ax2.plot(x, am, color="0.5", ls="--", lw=0.7, alpha=0.6)
                ax2.set_ylabel("X", fontsize=8, color="0.4")
    if x_label:
        axes[-1, 0].set_xlabel(x_label, fontsize=_LABEL_FS)
    if n_cal <= 0:
        fig.suptitle(
            f"Check-star QC ({filter_}): object of interest "
            "(no independent catalog calibrators)",
            fontsize=_TITLE_FS,
        )
    else:
        fig.suptitle(
            f"Check-star QC ({filter_}): object of interest vs "
            f"{n_cal} catalog calibrator"
            f"{'s' if n_cal != 1 else ''} with largest excess RMS",
            fontsize=_TITLE_FS,
        )
    fig.tight_layout()
    out = diagnostics_dir(output_dir, "lightcurves")
    path = out / f"check_star_qc_{_sanitize_filename(filter_)}.{file_type}"
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


def plot_calibrator_variability(
    lc: Table,
    stats: Table,
    output_dir: str,
    *,
    filter_: str,
    top_ids: list[int],
    file_type: str = "pdf",
    time_scale: str = "bjd_tdb",
    magnitude_system: str = "vega",
) -> Path | None:
    """All calibrators: excess-RMS vs mag, plus overplotted residual LCs."""
    from ..post_processing.light_curve import slice_light_curve

    if stats is None or len(stats) == 0:
        return None
    top = {int(i) for i in top_ids}
    fig, (ax_sc, ax_res) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    med = np.asarray(stats["med_mag"], dtype=float)
    exc = np.asarray(stats["excess_rms"], dtype=float)
    sids = np.asarray(stats["id"]).astype(int)
    ax_sc.scatter(med, exc, c="0.55", s=18, zorder=2, label="calibrators")
    if "rms" in stats.colnames:
        # Photometric-error floor proxy: rms - excess is not plotted; med err via chi2.
        pass
    for i, sid in enumerate(sids):
        if int(sid) in top:
            ax_sc.scatter(med[i], exc[i], c="C3", s=36, zorder=3)
            ax_sc.annotate(
                str(int(sid)),
                (med[i], exc[i]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    ax_sc.set_xlabel(_ylabel(filter_, "magnitude", magnitude_system), fontsize=_LABEL_FS)
    ax_sc.set_ylabel(r"Excess RMS (mag)", fontsize=_LABEL_FS)
    ax_sc.set_title("Calibrator excess scatter", fontsize=10)
    ax_sc.grid(True, color="lightgray", linestyle="--")

    x_label = None
    for sid in sids:
        sub = slice_light_curve(lc, int(sid), filter_)
        if len(sub) == 0:
            continue
        y, _e, qty = _series_y(sub)
        if qty == "flux":
            continue
        x, x_label = _plot_time_and_label(sub, time_scale)
        flag = (
            np.asarray(sub["flag_outlier"], dtype=bool)
            if "flag_outlier" in sub.colnames
            else np.zeros(len(sub), dtype=bool)
        )
        ok = np.isfinite(y) & (~flag)
        if not np.any(ok):
            continue
        res = y - np.nanmedian(y[ok])
        is_top = int(sid) in top
        ax_res.plot(
            x[ok],
            res[ok],
            color="C3" if is_top else "0.75",
            lw=1.0 if is_top else 0.6,
            alpha=0.95 if is_top else 0.45,
            zorder=3 if is_top else 1,
        )
    ax_res.axhline(0.0, color="0.3", lw=0.8)
    ax_res.set_xlabel(x_label or "JD − 2450000", fontsize=_LABEL_FS)
    ax_res.set_ylabel("mag − median", fontsize=_LABEL_FS)
    ax_res.invert_yaxis()
    ax_res.set_title("Residual light curves", fontsize=10)
    ax_res.grid(True, color="lightgray", linestyle="--")
    fig.suptitle(f"Calibrator variability ({filter_})", fontsize=_TITLE_FS)
    fig.tight_layout()
    out = diagnostics_dir(output_dir, "lightcurves")
    path = out / f"calibrator_variability_{_sanitize_filename(filter_)}.{file_type}"
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


def plot_light_curve_overview(
    lc: Table,
    output_dir: str,
    *,
    filter_: str,
    ooi_ids: list[tuple[int, str]],
    extra_ids: list[int],
    file_type: str = "pdf",
    time_scale: str = "bjd_tdb",
    magnitude_system: str = "vega",
) -> Path | None:
    """Raster of OOI plus random field stars (not one PDF per object)."""
    from ..post_processing.light_curve import slice_light_curve

    panels: list[tuple[str, Table]] = []
    seen: set[int] = set()
    for oid, name in ooi_ids:
        sub = slice_light_curve(lc, oid, filter_)
        if len(sub) > 0:
            panels.append((f"OOI {name} (id={oid})", sub))
            seen.add(int(oid))
    for sid in extra_ids:
        if int(sid) in seen:
            continue
        sub = slice_light_curve(lc, sid, filter_)
        if len(sub) > 0:
            panels.append((f"id={int(sid)}", sub))
    if not panels:
        return None
    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.2 * nrows),
        squeeze=False,
        sharex=True,
    )
    x_label = None
    for i, (title, sub) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        y, e, qty = _series_y(sub)
        x, x_label = _plot_time_and_label(sub, time_scale)
        night = np.asarray(sub["night_id"]) if "night_id" in sub.colnames else None
        flag = (
            np.asarray(sub["flag_outlier"], dtype=bool)
            if "flag_outlier" in sub.colnames
            else None
        )
        _draw_night_bands(ax, x, night)
        _scatter_with_flags(ax, x, y, e, flag, quantity=qty)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, color="lightgray", linestyle="--")
        ax.set_ylabel(_ylabel(filter_, qty, magnitude_system), fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    if x_label:
        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel(x_label, fontsize=9)
    fig.suptitle(f"Light-curve overview ({filter_})", fontsize=_TITLE_FS)
    fig.tight_layout()
    out = diagnostics_dir(output_dir, "lightcurves")
    path = out / f"lightcurve_overview_{_sanitize_filename(filter_)}.{file_type}"
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


def _table_from_timeseries(
    ts: TimeSeries,
    data_column: str,
    err_column: str,
    y_axis_style: str,
) -> Table:
    jd = np.asarray(ts.time.jd, dtype=float)
    y = np.asarray(ts[data_column], dtype=float)
    e = np.abs(np.asarray(ts[err_column], dtype=float))
    n = jd.size
    mag = np.full(n, np.nan)
    mag_err = np.full(n, np.nan)
    flux = np.full(n, np.nan)
    flux_err = np.full(n, np.nan)
    qty = "flux" if y_axis_style == "flux" else "magnitude"
    if qty == "flux":
        flux[:] = y
        flux_err[:] = e
    else:
        mag[:] = y
        mag_err[:] = e
    nights = np.full(n, -1, dtype=np.int64)
    ok = np.isfinite(jd)
    nights[ok] = np.floor(jd[ok] - 0.5).astype(np.int64)
    return Table(
        {
            "id": np.zeros(n, dtype=np.int64),
            "object_name": np.full(n, "", dtype="U64"),
            "filter": np.full(n, str(data_column), dtype="U32"),
            "epoch_id": np.full(n, "", dtype="U64"),
            "jd": jd,
            "bjd_tdb": np.full(n, np.nan),
            "airmass": np.full(n, np.nan),
            "night_id": nights,
            "mag": mag,
            "mag_err": mag_err,
            "flux": flux,
            "flux_err": flux_err,
            "quantity": np.full(n, qty, dtype="U16"),
            "flag_outlier": np.zeros(n, dtype=bool),
            "ra": np.full(n, np.nan),
            "dec": np.full(n, np.nan),
            "is_calibrator": np.zeros(n, dtype=bool),
        }
    )


def light_curve_jd(
        ts: TimeSeries, data_column: str, err_column: str, output_dir: str,
        error_bars: bool = True, name_object: str | None = None,
        file_name_suffix: str = '', subdirectory: str = '',
        file_type: str = 'pdf', own_scaling: bool = True,
        invert_axis: bool = True,
        y_axis_style: str = "magnitude",
        magnitude_system: str = "vega") -> None:
    """Backward-compatible JD plot from a :class:`~astropy.timeseries.TimeSeries`."""
    _ = error_bars, own_scaling, invert_axis, file_name_suffix
    tbl = _table_from_timeseries(ts, data_column, err_column, y_axis_style)
    light_curve_jd_from_table(
        tbl,
        output_dir,
        name_object=name_object or "object",
        filter_=data_column,
        file_type=file_type,
        subdirectory=subdirectory,
        time_scale="jd",
        show_airmass=False,
        magnitude_system=magnitude_system,
    )


def light_curve_fold(
        time_series: TimeSeries, data_column: str, err_column: str,
        output_dir: str, transit_time: str, period: float,
        binning_factor: float | None = None, error_bars: bool = True,
        name_object: str | None = None, file_name_suffix: str = '',
        subdirectory: str = '', file_type: str = 'pdf',
        y_axis_style: str = "magnitude",
        magnitude_system: str = "vega") -> None:
    """Backward-compatible folded plot from a TimeSeries (phase, not folded JD)."""
    _ = error_bars, file_name_suffix
    tbl = _table_from_timeseries(time_series, data_column, err_column, y_axis_style)
    light_curve_fold_from_table(
        tbl,
        output_dir,
        transit_time=transit_time,
        period=float(period),
        name_object=name_object or "object",
        filter_=data_column,
        file_type=file_type,
        subdirectory=subdirectory,
        binning_factor=binning_factor,
        time_scale="jd",
        phase_cycles=1,
        magnitude_system=magnitude_system,
    )


__all__ = [
    "fold_phase",
    "light_curve_fold",
    "light_curve_fold_from_table",
    "light_curve_jd",
    "light_curve_jd_from_table",
    "light_curve_nights_fold_from_table",
    "light_curve_nights_jd_from_table",
    "light_curve_nights_panels_from_table",
    "night_date_label",
    "plot_calibrator_variability",
    "plot_check_star_qc",
    "plot_light_curve_overview",
    "unique_night_ids",
    "y_limits_for_quantity",
]
