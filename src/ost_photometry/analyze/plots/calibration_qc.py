"""Calibration, extinction-fit, and diagnostic QC plots."""
from __future__ import annotations

from math import ceil
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.visualization import ImageNormalize, ZScaleInterval
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ... import checks

plt.switch_backend("Agg")

def plot_extinction_fit_value_airmass(
    output_dir: str | Path,
    data_by_filter: dict[str, tuple[np.ndarray, np.ndarray]],
    coefficients: dict[str, object],
    use_magnitude: bool = True,
    y_label: str | None = None,
    file_type: str = "pdf",
) -> None:
    """
    Plot extinction fit from flux/magnitude vs airmass (cat-star.org method).

    For each filter: scatter of y vs airmass with regression line and k ± err in title.
    Supports per-star fits (multiple series) or single overall fit.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory. Plots saved to output_dir/extinction_fit/.
    data_by_filter : dict
        {filter_name: (airmass_arr, y_arr)} with airmass and ln(flux) or magnitude.
    coefficients : dict
        {filter_name: ExtinctionCoefficients} from fit_extinction_from_value_airmass.
    use_magnitude : bool
        If True, y is magnitude (slope = k). If False, y is ln(flux) (slope = -k).
    y_label : str, optional
        Override y-axis label (e.g. "m [mag]" or "ln(flux)").
    file_type : str
        Plot file format (pdf, png, etc.). Default is ``pdf``.
    """

    out = Path(output_dir) / "extinction_fit"
    checks.check_output_directories(out)

    for filter_, (airmass, y) in data_by_filter.items():
        ec = coefficients.get(filter_)
        if ec is None:
            continue
        k = ec.k_prime
        k_err = ec.k_prime_err

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(airmass, y, alpha=0.7, s=20, color="C0", edgecolors="none")

        # Regression line
        slope = k if use_magnitude else -k
        intercept = float(np.nanmean(y) - slope * np.nanmean(airmass))
        airmass_line = np.linspace(airmass.min(), airmass.max(), 50)
        ax.plot(airmass_line, slope * airmass_line + intercept, "C1-", lw=2, label="Fit")

        ax.set_xlabel("Airmass X")
        if y_label is not None:
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel("m [mag]" if use_magnitude else "ln(flux)")
        ax.set_title(f"Filter {filter_}: k' = {k:.4f} ± {k_err:.4f} mag/airmass")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.savefig(
            out / f"extinction_value_airmass_{filter_}.{file_type}",
            bbox_inches="tight",
            format=file_type,
        )
        plt.close()


def plot_extinction_fit_comparison_stars(
    output_dir: str | Path,
    data_by_filter: dict[str, tuple[np.ndarray, np.ndarray]],
    coefficients: dict[str, object],
    file_type: str = "pdf",
) -> None:
    """
    Plot extinction fit from comparison stars (mean(m_obs - m_std) vs airmass).

    For each filter: scatter of delta vs X with regression line and k ± err in title.
    One point per frame.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory. Plots saved to output_dir/extinction_fit/.
    data_by_filter : dict
        {filter_name: (X_arr, delta_arr)} with airmass and mean(m_obs - m_std).
    coefficients : dict
        {filter_name: ExtinctionCoefficients} from fit_extinction_from_comparison_stars.
    file_type : str
        Plot file format (pdf, png, etc.). Default is ``pdf``.
    """

    out = Path(output_dir) / "extinction_fit"
    checks.check_output_directories(out)

    for filter_, (airmass, delta) in data_by_filter.items():
        ec = coefficients.get(filter_)
        if ec is None:
            continue
        k = ec.k_prime
        k_err = ec.k_prime_err

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(airmass, delta, alpha=0.7, s=40, color="C0", edgecolors="none")

        # Regression line
        slope = k
        intercept = float(np.nanmean(delta) - slope * np.nanmean(airmass))
        airmass_line = np.linspace(airmass.min(), airmass.max(), 50)
        ax.plot(airmass_line, slope * airmass_line + intercept, "C1-", lw=2, label="Fit")

        ax.set_xlabel("Airmass X")
        ax.set_ylabel(r"$\langle m_{\mathrm{obs}} - m_{\mathrm{std}} \rangle$ [mag]")
        ax.set_title(f"Filter {filter_}: k' = {k:.4f} ± {k_err:.4f} mag/airmass")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.savefig(
            out / f"extinction_comparison_stars_{filter_}.{file_type}",
            bbox_inches="tight",
            format=file_type,
        )
        plt.close()


def plot_calibration_transformation(
    output_dir: str | Path,
    epoch_id: str,
    data_by_filter: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    coefficients: dict[str, object],
    file_type: str = "pdf",
    *,
    filename_prefix: str = "calibration",
    title_prefix: str | None = None,
) -> None:
    """
    Plot calibration transformation fit: m_std - m_inst vs color.

    For each filter: scatter of (m_std - m_inst) vs color index with fit line
    T*color + ZP, and residuals panel. Allows checking fit quality and outliers.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory. Plots saved to output_dir/calibration/.
    epoch_id : str
        Identifier for the calibration epoch (e.g. ``epoch_000``).
    data_by_filter : dict
        {filter: (color_arr, delta_arr, mask)} with color index, m_std - m_inst,
        and boolean mask of stars used in fit.
    coefficients : dict
        {filter: TransformationCoefficients} with T, ZP, color_index_filters.
    file_type : str
        Plot file format. Default is ``pdf``.
    filename_prefix : str
        Filename stem before ``_{epoch}_{filter}`` (e.g. ``derive_transform``).
    title_prefix : str, optional
        Prepended to the left-panel title (e.g. ``Derive-transform fit``).
    """

    out = Path(output_dir) / "calibration"
    checks.check_output_directories(out)

    for filter_, (color, delta, mask) in data_by_filter.items():
        tc = coefficients.get(filter_)
        if tc is None:
            continue
        T, ZP = tc.color_term, tc.zero_point
        ci = f"({tc.color_index_filters[0]}-{tc.color_index_filters[1]})"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Left: m_std - m_inst vs color with fit
        used = np.asarray(mask, dtype=bool)
        finite = np.isfinite(color) & np.isfinite(delta)
        excl = (~used) & finite
        n_excl = int(np.sum(excl))
        if n_excl > 0:
            ax1.scatter(color[excl], delta[excl], alpha=0.4, s=15, c="gray", label="excluded")
        ax1.scatter(color[used], delta[used], alpha=0.7, s=25, c="C0", label="used")
        c_used = np.asarray(color, dtype=float)[used]
        c_finite = c_used[np.isfinite(c_used)]
        if len(c_finite) > 0 and np.nanmax(c_finite) - np.nanmin(c_finite) > 0.01:
            c_min, c_max = float(np.nanmin(c_finite)), float(np.nanmax(c_finite))
            c_line = np.linspace(c_min, c_max, 50)
            ax1.plot(c_line, T * c_line + ZP, "C1-", lw=2, label="Fit")
        else:
            ax1.axhline(ZP, color="C1", ls="-", lw=2, label="Fit (ZP only)")
        ax1.set_xlabel(f"Color {ci} [mag]")
        ax1.set_ylabel(r"$m_{\mathrm{std}} - m_{\mathrm{inst}}$ [mag]")
        head = f"{title_prefix}: " if title_prefix else ""
        ax1.set_title(f"{head}{epoch_id} {filter_}: T={T:.4f}, ZP={ZP:.4f}")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Right: residuals
        residuals = delta - (T * color + ZP)
        if n_excl > 0:
            ax2.scatter(color[excl], residuals[excl], alpha=0.4, s=15, c="gray")
        ax2.scatter(color[used], residuals[used], alpha=0.7, s=25, c="C0")
        ax2.axhline(0, color="C1", ls="--", lw=1)
        ax2.set_xlabel(f"Color {ci} [mag]")
        ax2.set_ylabel("Residual [mag]")
        rms_val = np.nanstd(residuals[used]) if np.sum(used) > 0 else 0.0
        n_used = int(np.sum(used))
        ax2.set_title(f"RMS = {rms_val:.4f} mag (n={n_used})")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_id = str(epoch_id).replace("/", "_").replace(":", "_")
        stem = str(filename_prefix).strip() or "calibration"
        plt.savefig(
            out / f"{stem}_{safe_id}_{filter_}.{file_type}",
            bbox_inches="tight",
            format=file_type,
        )
        plt.close()


def plot_derive_transform_fit_overview(
    output_dir: str | Path,
    epoch_ids: list[str],
    diagnostic_coeffs_per_epoch: list[dict],
    fit_metrics_per_epoch: list[dict[str, dict[str, float]]],
    filters: list[str],
    file_type: str = "pdf",
    *,
    output_basename: str = "derive_transform_fit_overview",
) -> None:
    """
    Overview of catalog-color derive-transform fits across epochs.

    Panels per filter: fitted slope T and intercept ZP, residual RMS, and
    number of stars used. Complements ``derive_transform_summary`` (applied
    ``c`` / median ZP) and the per-epoch scatter plots.
    """

    if not epoch_ids or not filters:
        return

    out = Path(output_dir) / "calibration"
    checks.check_output_directories(out)

    n_ep = len(epoch_ids)
    n_filt = len(filters)
    x = np.arange(n_ep, dtype=float)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=n_filt,
        figsize=(4.2 * n_filt, 9.0),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        "Derive-transform fit overview (catalog-color slopes)",
        fontsize=13,
    )

    for j, filter_ in enumerate(filters):
        T = np.full(n_ep, np.nan)
        ZP = np.full(n_ep, np.nan)
        rms = np.full(n_ep, np.nan)
        n_used = np.full(n_ep, np.nan)
        for i, coeffs in enumerate(diagnostic_coeffs_per_epoch):
            tc = coeffs.get(filter_)
            if tc is not None:
                T[i] = float(tc.color_term)
                ZP[i] = float(tc.zero_point)
            met = fit_metrics_per_epoch[i].get(filter_) if i < len(fit_metrics_per_epoch) else None
            if met:
                rms[i] = float(met.get("rms", np.nan))
                n_used[i] = float(met.get("n_used", np.nan))

        ax_t, ax_zp, ax_rms, ax_n = (axes[0, j], axes[1, j], axes[2, j], axes[3, j])
        ax_t.plot(x, T, "o-", ms=3, lw=1.0, color="C0")
        ax_t.set_ylabel("Slope T")
        ax_t.set_title(f"{filter_}")
        ax_t.grid(True, alpha=0.3)

        ax_zp.plot(x, ZP, "o-", ms=3, lw=1.0, color="C1")
        ax_zp.set_ylabel("Intercept ZP [mag]")
        ax_zp.grid(True, alpha=0.3)

        ax_rms.plot(x, rms, "o-", ms=3, lw=1.0, color="C2")
        ax_rms.set_ylabel("RMS residual [mag]")
        ax_rms.grid(True, alpha=0.3)

        ax_n.plot(x, n_used, "o-", ms=3, lw=1.0, color="C3")
        ax_n.set_ylabel("n stars used")
        ax_n.grid(True, alpha=0.3)

    ax_bottom_row = axes[-1, :]
    max_ticks = 22
    if n_ep > max_ticks:
        tick_idx = np.unique(np.linspace(0, n_ep - 1, num=max_ticks, dtype=int))
    else:
        tick_idx = np.arange(n_ep, dtype=int)

    def _short_epoch_label(s: str) -> str:
        s = str(s)
        if s.startswith("epoch_"):
            s = s[6:]
        return s if len(s) <= 16 else s[:13] + "…"

    labels = [_short_epoch_label(epoch_ids[k]) for k in tick_idx]
    for ax in ax_bottom_row:
        ax.set_xticks(x[tick_idx])
        ax.set_xticklabels(labels, rotation=68, ha="right", fontsize=7)
        ax.set_xlabel("Epoch")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    filt_tag = "_".join(str(f) for f in filters)
    stem = output_basename
    if filt_tag and not stem.endswith(filt_tag):
        stem = f"{output_basename}_{filt_tag}"
    fig.savefig(out / f"{stem}.{file_type}", bbox_inches="tight", format=file_type)
    plt.close(fig)


def plot_calibration_night_summary(
    output_dir: str | Path,
    epoch_ids: list[str],
    coefficients_per_epoch: list[dict],
    filters: list[str],
    file_type: str = "pdf",
    *,
    combined_per_filter: dict | None = None,
    output_basename: str = "calibration_night_summary",
    x_jd: list[float] | None = None,
) -> None:
    """
    Plot T and ZP across calibration epochs (night- or per-image-style fits).

    Shows stability of transformation coefficients over the run.
    One column of subplots: per filter, color term (T) above zero point (ZP);
    epoch index or Julian Date on the shared x-axis (better on narrow displays).

    Filters without any finite fitted zero point across epochs are omitted. If
    none remain (e.g. Clear-only without catalog standards), no file is written.
    Remaining filters share **one** figure (stacked panels); the filename includes
    the filter names (e.g. ``calibration_per_image_summary_B_V.pdf``).

    Parameters
    ----------
    output_dir : str or Path
        Base output directory. Plots saved to output_dir/calibration/.
    epoch_ids : list
        Epoch identifiers in order.
    coefficients_per_epoch : list of dict
        Each element: {filter: TransformationCoefficients} for that epoch.
    filters : list
        Filter names to plot.
    file_type : str
        Plot file format. Default is ``pdf``.
    combined_per_filter : dict, optional
        ``{filter: TransformationCoefficients}`` with inverse-variance mean T/ZP
        and errors (same as night-combined coefficients). Drawn as horizontal
        line ±1σ band on each panel.
    output_basename : str
        Filename stem (without extension). Use ``calibration_per_image_summary``
        for :class:`CoefficientMode.PER_IMAGE` runs. Filter names are appended.
    x_jd : list of float, optional
        If given, same length as ``epoch_ids`` and all finite: x-axis is these
        Julian Dates. Otherwise x is 0..N-1 with thinned / shortened epoch labels.
    """
    import warnings

    from ..warnings_types import OstPhotometryAnalyzeWarning

    def _filter_has_finite_zp(filter_name: str) -> bool:
        for cf in coefficients_per_epoch:
            tc = cf.get(filter_name)
            if tc is None:
                continue
            if np.isfinite(float(tc.zero_point)):
                return True
        return False

    filters = [f for f in filters if _filter_has_finite_zp(f)]
    n_filt = len(filters)
    if n_filt == 0:
        return

    out = Path(output_dir) / "calibration"
    checks.check_output_directories(out)

    n_ep = len(epoch_ids)
    use_jd_axis = False
    if x_jd is not None and len(x_jd) == n_ep:
        xa = np.asarray(x_jd, dtype=float)
        if np.all(np.isfinite(xa)):
            x_plot = xa
            use_jd_axis = True
        else:
            warnings.warn(
                "plot_calibration_night_summary: x_jd contains non-finite values; "
                "using epoch index on x-axis.",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )
            x_plot = np.arange(n_ep, dtype=float)
    else:
        if x_jd is not None and len(x_jd) != n_ep:
            warnings.warn(
                "plot_calibration_night_summary: len(x_jd) != len(epoch_ids); "
                "using epoch index on x-axis.",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )
        x_plot = np.arange(n_ep, dtype=float)

    # Single column: T then ZP for each filter (stacked vertically)
    fig_w = max(8.0, min(14.0, 6.0 + 0.22 * n_ep))
    n_ax = 2 * n_filt
    fig_h = min(42.0, 2.65 * n_ax)
    fig, axes = plt.subplots(n_ax, 1, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_1d(axes)

    for i, filter_ in enumerate(filters):
        T_vals = []
        ZP_vals = []
        for cf in coefficients_per_epoch:
            tc = cf.get(filter_)
            if tc is not None:
                T_vals.append(tc.color_term)
                ZP_vals.append(tc.zero_point)
            else:
                T_vals.append(np.nan)
                ZP_vals.append(np.nan)

        ax_t = axes[2 * i]
        ax_zp = axes[2 * i + 1]

        ax_t.plot(x_plot, T_vals, "o-", color="C0", markersize=6, label="per epoch")
        ax_t.set_ylabel(f"T ({filter_})")
        ax_t.grid(True, alpha=0.3)
        ax_t.set_title("Color term")

        ax_zp.plot(x_plot, ZP_vals, "s-", color="C1", markersize=6, label="per epoch")
        ax_zp.set_ylabel(f"ZP ({filter_})")
        ax_zp.grid(True, alpha=0.3)
        ax_zp.set_title("Zero point")

        if combined_per_filter:
            tc_mean = combined_per_filter.get(filter_)
            if tc_mean is not None:
                tm = float(tc_mean.color_term)
                zm = float(tc_mean.zero_point)
                te = float(tc_mean.color_term_err)
                ze = float(tc_mean.zero_point_err)
                if np.isfinite(tm):
                    ax_t.axhline(
                        tm,
                        color="C0",
                        linestyle="--",
                        linewidth=1.4,
                        label=(
                            f"T_mean = {tm:.4f} ± {te:.4f}"
                            if np.isfinite(te)
                            else f"T_mean = {tm:.4f}"
                        ),
                    )
                    if np.isfinite(te) and te > 0:
                        ax_t.axhspan(
                            tm - te,
                            tm + te,
                            color="C0",
                            alpha=0.15,
                            linewidth=0,
                        )
                if np.isfinite(zm):
                    ax_zp.axhline(
                        zm,
                        color="C1",
                        linestyle="--",
                        linewidth=1.4,
                        label=(
                            f"ZP_mean = {zm:.4f} ± {ze:.4f}"
                            if np.isfinite(ze)
                            else f"ZP_mean = {zm:.4f}"
                        ),
                    )
                    if np.isfinite(ze) and ze > 0:
                        ax_zp.axhspan(
                            zm - ze,
                            zm + ze,
                            color="C1",
                            alpha=0.15,
                            linewidth=0,
                        )
                ax_t.legend(loc="best", fontsize=7)
                ax_zp.legend(loc="best", fontsize=7)

    ax_bottom = axes[-1]
    if use_jd_axis:
        ax_bottom.xaxis.set_major_locator(MaxNLocator(nbins=9, prune="both"))
        ax_bottom.ticklabel_format(axis="x", useOffset=True, style="plain")
        ax_bottom.tick_params(axis="x", labelsize=8)
        ax_bottom.set_xlabel("Julian Date (JD)")
    else:
        max_ticks = 22
        if n_ep > max_ticks:
            tick_idx = np.unique(
                np.linspace(0, n_ep - 1, num=max_ticks, dtype=int)
            )
        else:
            tick_idx = np.arange(n_ep, dtype=int)

        def _short_epoch_label(s: str) -> str:
            s = str(s)
            if s.startswith("epoch_"):
                s = s[6:]
            return s if len(s) <= 20 else s[:17] + "…"

        tick_x = x_plot[tick_idx]
        labels = [_short_epoch_label(epoch_ids[j]) for j in tick_idx]
        ax_bottom.set_xticks(tick_x)
        ax_bottom.set_xticklabels(labels, rotation=68, ha="right", fontsize=7)
        ax_bottom.set_xlabel("Epoch")

    fig.align_xlabels()
    plt.tight_layout(rect=[0, 0.06, 1, 0.99])

    filt_tag = "_".join(str(f) for f in filters)
    stem = output_basename
    if filt_tag and not stem.endswith(filt_tag):
        stem = f"{output_basename}_{filt_tag}"

    plt.savefig(
        out / f"{stem}.{file_type}",
        bbox_inches="tight",
        format=file_type,
    )
    plt.close()


# ---------------------------------------------------------------------------
# Diagnostic plots (written under a caller-provided directory, typically
# ``<pipeline_output>/diagnostics/``; toggles in ``DiagnosticPlots``).
# ---------------------------------------------------------------------------


def _diagnostic_plot_path(
    output_dir: str | Path,
    stem: str,
    file_type: str,
) -> Path:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    ft = file_type.lstrip(".")
    return base / f"{stem}.{ft}"


_POGSON = 2.5 / np.log(10)
_SNR_GUIDES = (10.0, 5.0)
# Distinct from each other and from the cyan median / orange photon curve.
_SNR_GUIDE_STYLE = {
    10.0: dict(color="#0072B2", ls="-.", lw=1.45),
    5.0: dict(color="#333333", ls=(0, (1.0, 1.35)), lw=1.8),
}
_MEDIAN_COLOR = "#00C2D1"
_PHOTON_COLOR = "#E69F00"
_QUALITY_PLOT_COLUMNS = (
    ("qfit", "qfit (PSF residual)"),
    ("cfit", "cfit (PSF residual)"),
    ("sharpness", "sharpness"),
    ("roundness", "roundness"),
    ("roundness1", "roundness1"),
    ("roundness2", "roundness2"),
    ("fwhm", "finder FWHM [pix]"),
)


def _positive_magnitude_uncertainty(err) -> np.ndarray:
    """Absolute 1σ magnitude error; non-finite and non-positive values → NaN."""
    e = np.abs(np.asarray(err, dtype=float))
    e[~np.isfinite(e) | (e <= 0.0)] = np.nan
    return e


def _finite_mag_and_positive_err(
    mag,
    err,
) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mag, dtype=float)
    e = _positive_magnitude_uncertainty(err)
    ok = np.isfinite(m) & np.isfinite(e)
    return m[ok], e[ok]


def _mag_err_valid_mask(mag, err) -> np.ndarray:
    m = np.asarray(mag, dtype=float)
    e = _positive_magnitude_uncertainty(err)
    return np.isfinite(m) & np.isfinite(e)


def _column_float(table: Table, name: str) -> np.ndarray:
    col = table[name]
    return np.asarray(col.value if hasattr(col, "value") else col, dtype=float)


def _binned_error_percentiles(
    mag: np.ndarray,
    err: np.ndarray,
    *,
    percentiles: tuple[float, float, float] = (16.0, 50.0, 84.0),
    min_per_bin: int = 8,
    n_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-count magnitude bins → centers and (p16, p50, p84) of σ."""
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)
    if mag.size < min_per_bin * 2:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    order = np.argsort(mag)
    mag_s = mag[order]
    err_s = err[order]
    n_bins = int(min(n_bins, mag.size // min_per_bin))
    n_bins = max(n_bins, 2)
    edges = np.linspace(0, mag.size, n_bins + 1, dtype=int)
    centers: list[float] = []
    p_lo: list[float] = []
    p_mid: list[float] = []
    p_hi: list[float] = []
    lo_p, mid_p, hi_p = percentiles
    for i0, i1 in zip(edges[:-1], edges[1:], strict=True):
        if i1 - i0 < min_per_bin:
            continue
        chunk = err_s[i0:i1]
        centers.append(float(np.median(mag_s[i0:i1])))
        p_lo.append(float(np.percentile(chunk, lo_p)))
        p_mid.append(float(np.percentile(chunk, mid_p)))
        p_hi.append(float(np.percentile(chunk, hi_p)))
    return (
        np.asarray(centers),
        np.asarray(p_lo),
        np.asarray(p_mid),
        np.asarray(p_hi),
    )


def _photon_noise_sigma(mag: np.ndarray, floor: float, faint_scale: float) -> np.ndarray:
    """Photon + additive-noise envelope: σ(m) = √(σ₀² + (c · 10^{0.4 m})²).

    Valid for CCD and CMOS: Poisson photon statistics plus a flux-independent
    term (sky in the aperture, read noise, residual floor).
    """
    return np.sqrt(floor**2 + (faint_scale * 10.0 ** (0.4 * mag)) ** 2)


def _fit_photon_noise_envelope(
    mag: np.ndarray,
    err: np.ndarray,
) -> tuple[float, float] | None:
    """Fit a constant floor plus background/read term (∝ 10^{0.4 m}) to the binned median."""
    centers, _lo, mid, _hi = _binned_error_percentiles(mag, err)
    if centers.size < 3:
        return None
    positive = (mid > 0) & np.isfinite(mid) & np.isfinite(centers)
    if np.count_nonzero(positive) < 3:
        return None
    m_fit = centers[positive]
    s_fit = mid[positive]

    def _model(m, floor, faint_scale):
        return _photon_noise_sigma(m, floor, faint_scale)

    p0 = (
        float(np.nanmin(s_fit)),
        float(np.median(s_fit) / max(np.median(10.0 ** (0.4 * m_fit)), 1e-12)),
    )
    try:
        from scipy.optimize import curve_fit

        popt, _cov = curve_fit(
            _model,
            m_fit,
            s_fit,
            p0=p0,
            bounds=(0.0, np.inf),
            maxfev=4000,
        )
    except (ImportError, RuntimeError, ValueError, TypeError):
        return None
    floor, faint_scale = float(popt[0]), float(popt[1])
    if not np.isfinite(floor) or not np.isfinite(faint_scale):
        return None
    if floor <= 0.0 and faint_scale <= 0.0:
        return None
    return floor, faint_scale


def _quality_series_for_plot(
    photometry: Table,
    ok: np.ndarray,
    image_shape: tuple[int, int] | None = None,
) -> list[tuple[np.ndarray, str]]:
    """Every finite finder/PSF quality column, not only the first match."""
    series: list[tuple[np.ndarray, str]] = []
    for col, label in _QUALITY_PLOT_COLUMNS:
        if col not in photometry.colnames:
            continue
        values = _column_float(photometry, col)[ok]
        if np.any(np.isfinite(values)):
            series.append((values, label))
    if series:
        return series
    if "x_fit" not in photometry.colnames or "y_fit" not in photometry.colnames:
        return []
    x = _column_float(photometry, "x_fit")[ok]
    y = _column_float(photometry, "y_fit")[ok]
    if image_shape is not None and image_shape[0] > 1 and image_shape[1] > 1:
        naxis1, naxis2 = image_shape
        dist = np.minimum.reduce([x, y, naxis1 - 1.0 - x, naxis2 - 1.0 - y])
        return [(dist, "edge distance [pix]")]
    r = np.hypot(x - np.nanmedian(x), y - np.nanmedian(y))
    return [(r, "offset from field centre [pix]")]


def _comparison_mask(photometry: Table, ok: np.ndarray) -> np.ndarray | None:
    if "is_comparison" in photometry.colnames:
        flag = np.asarray(photometry["is_comparison"])[ok]
        return np.asarray(flag, dtype=bool)
    std_cols = [c for c in photometry.colnames if str(c).startswith("mag_std_")]
    if not std_cols:
        return None
    flag = np.zeros(int(np.count_nonzero(ok)), dtype=bool)
    for col in std_cols:
        vals = _column_float(photometry, col)[ok]
        flag |= np.isfinite(vals)
    return flag if np.any(flag) else None


def _calibrator_overlays(
    photometry: Table,
    ok: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Stars used in the fit, and catalog matches that were later rejected."""
    used = None
    if "is_calibrator" in photometry.colnames:
        used = np.asarray(photometry["is_calibrator"], dtype=bool)[ok]
        if not np.any(used):
            used = None
    candidates = _comparison_mask(photometry, ok)
    rejected = None
    if used is not None and candidates is not None:
        rejected = candidates & ~used
        if not np.any(rejected):
            rejected = None
    elif used is None:
        used = candidates
    return used, rejected


def _snr_sigma(snr: float) -> float:
    return _POGSON / float(snr)


def _draw_mag_err_density(ax, mag: np.ndarray, err: np.ndarray):
    """2D histogram in (mag, σ) with log-spaced σ bins and log y-axis."""
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)
    e_min = float(np.min(err))
    e_max = float(np.max(err))
    if not np.isfinite(e_min) or e_max <= e_min:
        e_max = e_min * 3.0 if e_min > 0 else 1.0
        e_min = e_min / 3.0 if e_min > 0 else 1e-3
    e_min = max(e_min * 0.8, 1e-5)
    e_max = max(e_max * 1.25, e_min * 1.01)
    n = mag.size
    n_x = int(np.clip(np.sqrt(n), 12, 40))
    n_y = int(np.clip(np.sqrt(n), 12, 36))
    xbins = np.linspace(float(np.min(mag)), float(np.max(mag)), n_x + 1)
    ybins = np.logspace(np.log10(e_min), np.log10(e_max), n_y + 1)
    counts, xedges, yedges = np.histogram2d(mag, err, bins=[xbins, ybins])
    mesh = np.ma.masked_less_equal(counts.T, 0)
    positive = counts[counts > 0]
    vmin = float(np.min(positive)) if positive.size else 1.0
    pcm = ax.pcolormesh(
        xedges,
        yedges,
        mesh,
        norm=LogNorm(vmin=max(vmin, 1.0)),
        cmap="viridis",
        shading="auto",
    )
    ax.set_yscale("log")
    ax.set_ylim(e_min, e_max)
    return pcm


def _draw_snr_guides(ax) -> None:
    for snr in _SNR_GUIDES:
        sigma = _snr_sigma(snr)
        style = _SNR_GUIDE_STYLE[snr]
        ax.axhline(
            sigma,
            color=style["color"],
            ls=style["ls"],
            lw=style["lw"],
            zorder=3,
            label=rf"{snr:.0f}$\sigma$  ($\sigma_m$={sigma:.2f})",
        )


def _expand_ylim_for_snr_guides(ax) -> None:
    """Keep 5σ / 10σ inside the frame so they are not both clipped to the top."""
    y0, y1 = ax.get_ylim()
    top = y1
    for snr in _SNR_GUIDES:
        sigma = _snr_sigma(snr)
        if np.isfinite(sigma) and sigma > 0:
            top = max(top, float(sigma) * 1.2)
    if top > y1:
        ax.set_ylim(y0, top)


def _draw_trend_and_photon_model(ax, mag: np.ndarray, err: np.ndarray) -> None:
    centers, p16, p50, p84 = _binned_error_percentiles(mag, err)
    if centers.size:
        # Light band on viridis; legend uses a black-edged proxy (white fill
        # is invisible on a white legend frame).
        ax.fill_between(
            centers,
            p16,
            p84,
            color="white",
            alpha=0.72,
            zorder=4,
            linewidth=0,
        )
        ax.plot(centers, p16, color="k", lw=1.5, zorder=5)
        ax.plot(centers, p84, color="k", lw=1.5, zorder=5)
        ax.plot(
            centers,
            p50,
            color="k",
            lw=4.4,
            zorder=6,
            solid_capstyle="round",
        )
        ax.plot(
            centers,
            p50,
            color=_MEDIAN_COLOR,
            lw=2.8,
            zorder=7,
            solid_capstyle="round",
            label="binned median",
        )
    params = _fit_photon_noise_envelope(mag, err)
    if params is None:
        return
    floor, faint_scale = params
    m_line = np.linspace(float(np.min(mag)), float(np.max(mag)), 120)
    ax.plot(
        m_line,
        _photon_noise_sigma(m_line, floor, faint_scale),
        color=_PHOTON_COLOR,
        lw=1.6,
        ls="--",
        zorder=8,
        label=r"photon+sky/read  $\sqrt{\sigma_0^2+(c\,10^{0.4m})^2}$",
    )


def _finalize_mag_err_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if "binned median" in labels:
        band = Patch(
            facecolor="white",
            edgecolor="0.15",
            linewidth=1.2,
            alpha=0.95,
            label="16–84% range",
        )
        i = labels.index("binned median") + 1
        handles.insert(i, band)
        labels.insert(i, band.get_label())
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.92)


def _draw_calibrator_markers(
    ax,
    mag: np.ndarray,
    err: np.ndarray,
    used: np.ndarray | None,
    rejected: np.ndarray | None,
    *,
    legend: bool,
    used_edge: str = "C1",
) -> None:
    n_rej = int(np.count_nonzero(rejected)) if rejected is not None else 0
    n_comp = int(np.count_nonzero(used)) if used is not None else 0
    if rejected is not None and n_rej:
        label = f"catalog, not used ({n_rej})" if legend else None
        ax.scatter(
            mag[rejected],
            err[rejected],
            s=22,
            marker="x",
            c="0.35",
            linewidths=0.7,
            zorder=8,
            label=label,
        )
    if used is not None and n_comp:
        label = f"used in calibration ({n_comp})" if legend else None
        ax.scatter(
            mag[used],
            err[used],
            s=28 if legend else 36,
            marker="*",
            facecolors="none",
            edgecolors=used_edge,
            linewidths=0.8 if legend else 0.9,
            zorder=9,
            label=label,
        )


def _mag_err_stats_text(
    mag: np.ndarray,
    err: np.ndarray,
    n_comparison: int = 0,
    n_rejected: int = 0,
) -> str:
    lines = [
        f"N = {mag.size}",
        f"median σ = {np.median(err):.3f} mag",
        f"p90 σ = {np.percentile(err, 90):.3f} mag",
    ]
    if n_comparison:
        lines.append(f"used in calibration = {n_comparison}")
    if n_rejected:
        lines.append(f"catalog, not used = {n_rejected}")
    return "\n".join(lines)


def _stats_text(sep: np.ndarray) -> str:
    x = np.asarray(sep, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "N=0"
    return (
        f"N = {x.size}\n"
        f"median = {np.median(x):.3f}\"\n"
        f"p90 = {np.percentile(x, 90):.3f}\"\n"
        f"max = {np.max(x):.3f}\""
    )


def plot_calibration_crossmatch_separations(
    separations_arcsec: np.ndarray,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    title: str = "Calibration cross-match separations",
    filename_stem: str = "calibration_crossmatch_separations",
) -> Path:
    """Histogram of on-sky separations (arcsec) for catalog matches."""
    x = np.asarray(separations_arcsec, dtype=float)
    x = x[np.isfinite(x)]
    fig, ax = plt.subplots(figsize=(6, 4))
    if x.size:
        ax.hist(x, bins="auto", color="C0", alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Separation [arcsec]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.text(
        0.97,
        0.97,
        _stats_text(x),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def _xy_columns_for_match_plot(table: Table) -> tuple[str, str] | None:
    for xname, yname in (("x", "y"), ("x_fit", "y_fit")):
        if xname in table.colnames and yname in table.colnames:
            return xname, yname
    return None


def _first_inst_mag_column(table: Table) -> str | None:
    preferred = []
    others = []
    for name in table.colnames:
        text = str(name)
        if not text.startswith("mag_"):
            continue
        if text.startswith("mag_std_") or text.startswith("mag_cal"):
            continue
        (preferred if text in {"mag_V", "mag_B", "mag_R"} else others).append(text)
    if preferred:
        for band in ("mag_V", "mag_B", "mag_R"):
            if band in preferred:
                return band
        return preferred[0]
    return others[0] if others else None


def _first_std_mag_column(table: Table) -> str | None:
    names = [str(c) for c in table.colnames if str(c).startswith("mag_std_")]
    for band in ("mag_std_V", "mag_std_B", "mag_std_R"):
        if band in names:
            return band
    return names[0] if names else None


def _calibrator_flag_for_plot(table: Table) -> np.ndarray | None:
    if "is_calibrator" in table.colnames:
        flag = np.asarray(table["is_calibrator"], dtype=bool)
        return flag if np.any(flag) else None
    cols = [c for c in table.colnames if str(c).startswith("is_calibrator_")]
    if not cols:
        return None
    flag = np.zeros(len(table), dtype=bool)
    for col in cols:
        flag |= np.asarray(table[col], dtype=bool)
    return flag if np.any(flag) else None


def catalog_match_pixel_residuals(
    table: Table,
    wcs_image,
    *,
    x_col: str | None = None,
    y_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pixel residuals of catalog sky positions vs detections (matched rows only)."""
    if (
        wcs_image is None
        or "ra_cat" not in table.colnames
        or "dec_cat" not in table.colnames
    ):
        return None
    xy = (x_col, y_col) if x_col and y_col else _xy_columns_for_match_plot(table)
    if xy is None:
        return None
    x = _column_float(table, xy[0])
    y = _column_float(table, xy[1])
    ra_cat = _column_float(table, "ra_cat")
    dec_cat = _column_float(table, "dec_cat")
    ok = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(ra_cat)
        & np.isfinite(dec_cat)
    )
    sep_all = None
    if "match_sep_arcsec" in table.colnames:
        sep_all = _column_float(table, "match_sep_arcsec")
        ok &= np.isfinite(sep_all)
    if not np.any(ok):
        return None
    try:
        coord = SkyCoord(ra=ra_cat[ok] * u.deg, dec=dec_cat[ok] * u.deg)
        x_cat, y_cat = wcs_image.world_to_pixel(coord)
    except (AttributeError, TypeError, ValueError):
        return None
    dx = np.asarray(x_cat, dtype=float) - x[ok]
    dy = np.asarray(y_cat, dtype=float) - y[ok]
    if sep_all is not None:
        sep = sep_all[ok]
    elif "ra" in table.colnames and "dec" in table.colnames:
        det = SkyCoord(
            ra=_column_float(table, "ra")[ok] * u.deg,
            dec=_column_float(table, "dec")[ok] * u.deg,
        )
        sep = np.asarray(det.separation(coord).arcsec, dtype=float)
    else:
        sep = np.hypot(dx, dy)
    return x[ok], y[ok], dx, dy, sep


def plot_calibration_crossmatch_diagnostics(
    table: Table,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    filename_stem: str = "calibration_crossmatch_diagnostics",
    title: str = "Catalog cross-match diagnostics",
) -> Path | None:
    """
    Why a separation histogram has a core plus a tail: mag, radius, and
    second-nearest catalog star (ambiguous matches sit near the 1:1 line).
    """
    if "match_sep_arcsec" not in table.colnames:
        return None
    sep = _column_float(table, "match_sep_arcsec")
    matched = np.isfinite(sep)
    if not np.any(matched):
        return None
    sep_m = sep[matched]
    xy = _xy_columns_for_match_plot(table)
    if xy is not None:
        x = _column_float(table, xy[0])[matched]
        y = _column_float(table, xy[1])[matched]
        x0, y0 = float(np.nanmedian(x)), float(np.nanmedian(y))
        radius = np.hypot(x - x0, y - y0)
    else:
        radius = np.full(int(np.count_nonzero(matched)), np.nan)

    inst_col = _first_inst_mag_column(table)
    std_col = _first_std_mag_column(table)
    inst = _column_float(table, inst_col)[matched] if inst_col else None
    std = _column_float(table, std_col)[matched] if std_col else None
    sep2 = None
    if "match_sep2_arcsec" in table.colnames:
        sep2 = _column_float(table, "match_sep2_arcsec")[matched]

    used = _calibrator_flag_for_plot(table)
    used_m = used[matched] if used is not None else None

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.6, 8.8),
        gridspec_kw={"wspace": 0.28, "hspace": 0.32},
    )
    ax_h, ax_mag = axes[0]
    ax_r, ax_amb = axes[1]

    bins = "auto" if sep_m.size < 80 else min(40, max(12, int(np.sqrt(sep_m.size))))
    ax_h.hist(
        sep_m,
        bins=bins,
        color="C0",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.3,
    )
    ax_h.set_yscale("log")
    med = float(np.median(sep_m))
    p90 = float(np.percentile(sep_m, 90))
    ax_h.axvline(med, color="#E69F00", ls="--", lw=1.3, label=f"median {med:.2f}\"")
    ax_h.axvline(p90, color="#0072B2", ls="-.", lw=1.2, label=f"p90 {p90:.2f}\"")
    ax_h.set_xlabel("Separation [arcsec]")
    ax_h.set_ylabel("Count (log)")
    ax_h.set_title("Separation histogram (log y)", fontsize=10)
    ax_h.legend(fontsize=8, loc="upper right")
    ax_h.grid(True, which="both", alpha=0.3)

    def _scatter_sep(ax, xvals, xlabel, title_txt):
        if xvals is None or not np.any(np.isfinite(xvals)):
            ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center")
            ax.set_title(title_txt, fontsize=10)
            return
        ok = np.isfinite(xvals)
        if used_m is not None:
            other = ok & ~used_m
            ax.scatter(
                xvals[other],
                sep_m[other],
                s=10,
                alpha=0.45,
                c="0.45",
                edgecolors="none",
                label="catalog match",
            )
            ax.scatter(
                xvals[ok & used_m],
                sep_m[ok & used_m],
                s=28,
                marker="*",
                facecolors="none",
                edgecolors="C1",
                linewidths=0.8,
                label="used in calibration",
            )
            ax.legend(fontsize=7, loc="upper left")
        else:
            ax.scatter(xvals[ok], sep_m[ok], s=10, alpha=0.45, c="C0", edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Separation [arcsec]")
        ax.set_title(title_txt, fontsize=10)
        ax.grid(True, alpha=0.3)

    mag_vals = inst if inst is not None and np.any(np.isfinite(inst)) else std
    mag_label = (
        f"{inst_col} [mag]"
        if inst is not None and np.any(np.isfinite(inst))
        else (f"{std_col} [mag]" if std_col else "Magnitude [mag]")
    )
    _scatter_sep(ax_mag, mag_vals, mag_label, "|offset| vs magnitude")
    _scatter_sep(ax_r, radius, "Radius from field centre [pix]", "|offset| vs radius")

    if sep2 is not None and np.any(np.isfinite(sep2)):
        ok2 = np.isfinite(sep2)
        ax_amb.scatter(sep_m[ok2], sep2[ok2], s=10, alpha=0.45, c="C3", edgecolors="none")
        hi = float(np.nanmax([np.nanmax(sep_m[ok2]), np.nanmax(sep2[ok2]), 0.2]))
        ax_amb.plot([0.0, hi], [0.0, hi], "k--", lw=1.0, alpha=0.7, label="sep2 = sep")
        ax_amb.legend(fontsize=7, loc="upper left")
        ax_amb.set_xlabel("Nearest catalog star [arcsec]")
        ax_amb.set_ylabel("Second-nearest [arcsec]")
        ax_amb.set_title("Ambiguous matches (near 1:1)", fontsize=10)
        ax_amb.grid(True, alpha=0.3)
    else:
        _scatter_sep(
            ax_amb,
            std,
            f"{std_col} [mag]" if std_col else "Catalog mag",
            "|offset| vs catalog magnitude",
        )

    n_tail = int(np.count_nonzero(sep_m > max(med * 3.0, 1.0)))
    box = (
        f"N match = {sep_m.size}\n"
        f"median = {med:.3f}\"\n"
        f"p90 = {p90:.3f}\"\n"
        f"max = {float(np.max(sep_m)):.3f}\"\n"
        f"N(sep > max(3×med, 1\")) = {n_tail}"
    )
    ax_h.text(
        0.03,
        0.97,
        box,
        transform=ax_h.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(top=0.92)
    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_combined_separation_histograms(
    separations_inter_filter_arcsec: np.ndarray,
    separations_calibration_crossmatch_arcsec: np.ndarray,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    reference_filter: str = "",
    other_filters: list[str] | None = None,
    filename_stem: str = "combined_separation_histograms",
) -> Path | None:
    """
    Two stacked histograms (shared separation axis): inter-filter correlations vs.
    calibration catalog cross-match. Each panel uses the same summary box as the
    standalone separation plots (N, median, p90, max in arcsec).
    """
    s_if = np.asarray(separations_inter_filter_arcsec, dtype=float)
    s_if = s_if[np.isfinite(s_if)]
    s_cal = np.asarray(separations_calibration_crossmatch_arcsec, dtype=float)
    s_cal = s_cal[np.isfinite(s_cal)]
    if s_if.size == 0 and s_cal.size == 0:
        return None

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(6.5, 6.9),
        sharex=True,
        gridspec_kw={"hspace": 0.14},
    )
    txt_kw = dict(
        transform=ax0.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    title_if = "Inter-filter correlation separations"
    if reference_filter:
        title_if += f" (ref={reference_filter}"
        if other_filters:
            title_if += f", vs {','.join(other_filters)}"
        title_if += ")"

    if s_if.size:
        ax0.hist(
            s_if,
            bins="auto",
            color="C1",
            alpha=0.85,
            edgecolor="k",
            linewidth=0.3,
        )
        ax0.text(0.97, 0.97, _stats_text(s_if), **txt_kw)
    else:
        ax0.text(
            0.5,
            0.5,
            "No inter-filter data\n(single filter or unavailable)",
            transform=ax0.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            alpha=0.75,
        )
    ax0.set_ylabel("Count")
    ax0.set_title(title_if, fontsize=10)

    txt_kw_cal = dict(txt_kw) | dict(transform=ax1.transAxes)
    if s_cal.size:
        ax1.hist(
            s_cal,
            bins="auto",
            color="C0",
            alpha=0.85,
            edgecolor="k",
            linewidth=0.3,
        )
        ax1.text(0.97, 0.97, _stats_text(s_cal), **txt_kw_cal)
    else:
        ax1.text(
            0.5,
            0.5,
            "No calibration cross-match data",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            alpha=0.75,
        )
    ax1.set_xlabel("Separation [arcsec]")
    ax1.set_ylabel("Count")
    ax1.set_title("Calibration cross-match separations", fontsize=10)

    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_photometry_mag_vs_error(
    photometry: Table,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    band_label: str = "",
    filename_stem: str | None = None,
    image_shape: tuple[int, int] | None = None,
    x_label: str = "Instrumental magnitude [mag]",
) -> Path | None:
    """
    QC figure: density of mag vs σ (log y), trend, photon-noise model, SNR guides.

    Extra panels are added for each available quality column (``qfit``, ``cfit``,
    sharpness, roundness*, finder FWHM) and/or comparison-star flags.
    """
    if "mags_fit" not in photometry.colnames or "mags_unc" not in photometry.colnames:
        return None
    mag_all = _column_float(photometry, "mags_fit")
    err_all = _column_float(photometry, "mags_unc")
    ok = _mag_err_valid_mask(mag_all, err_all)
    mag, err = mag_all[ok], _positive_magnitude_uncertainty(err_all)[ok]
    if mag.size == 0:
        return None

    used, rejected = _calibrator_overlays(photometry, ok)
    quality_series = _quality_series_for_plot(photometry, ok, image_shape=image_shape)
    n_comp = int(np.count_nonzero(used)) if used is not None else 0
    n_rej = int(np.count_nonzero(rejected)) if rejected is not None else 0
    has_cal = used is not None or rejected is not None
    extra_panels: list[tuple[np.ndarray | None, str]] = list(quality_series)
    if has_cal and not extra_panels:
        extra_panels = [(None, "Stars used in calibration")]

    stem = filename_stem or (
        f"photometry_mag_vs_error_{band_label}" if band_label else "photometry_mag_vs_error"
    )
    n_extra = len(extra_panels)
    extra_axes: list = []
    if n_extra:
        n_cols = 1 if n_extra == 1 else 2
        n_q_rows = ceil(n_extra / n_cols)
        fig = plt.figure(figsize=(6.8 if n_cols == 1 else 11.2, 5.2 + 3.5 * n_q_rows))
        gs = fig.add_gridspec(
            1 + n_q_rows,
            n_cols,
            height_ratios=[1.45] + [1.0] * n_q_rows,
            hspace=0.38,
            wspace=0.28,
        )
        ax0 = fig.add_subplot(gs[0, :])
        for i in range(n_extra):
            row = 1 + i // n_cols
            col = i % n_cols
            span_row = n_extra % n_cols == 1 and i == n_extra - 1 and n_cols == 2
            ax = fig.add_subplot(gs[row, :] if span_row else gs[row, col], sharex=ax0)
            extra_axes.append(ax)
        plt.setp(ax0.get_xticklabels(), visible=False)
    else:
        fig, ax0 = plt.subplots(figsize=(6.4, 4.8))

    pcm = _draw_mag_err_density(ax0, mag, err)
    fig.colorbar(pcm, ax=ax0, label="N per bin")
    _draw_trend_and_photon_model(ax0, mag, err)
    _draw_snr_guides(ax0)
    _expand_ylim_for_snr_guides(ax0)
    _draw_calibrator_markers(ax0, mag, err, used, rejected, legend=True)
    ax0.set_ylabel(r"$\sigma_m$ [mag]")
    ax0.grid(True, which="both", alpha=0.3)
    _finalize_mag_err_legend(ax0)
    ax0.text(
        0.97,
        0.03,
        _mag_err_stats_text(mag, err, n_comparison=n_comp, n_rejected=n_rej),
        transform=ax0.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )
    ttl = "Magnitude vs. uncertainty"
    if band_label:
        ttl += f" ({band_label})"
    ax0.set_title(ttl)

    last_row_start = 0
    if extra_axes:
        n_cols_extra = 1 if n_extra <= 1 else 2
        last_row_start = ((n_extra - 1) // n_cols_extra) * n_cols_extra
    for i, ax1 in enumerate(extra_axes):
        quality, quality_label = extra_panels[i]
        if quality is not None:
            finite_q = np.isfinite(quality)
            sc = ax1.scatter(
                mag[finite_q],
                err[finite_q],
                c=quality[finite_q],
                s=10,
                cmap="plasma",
                alpha=0.75,
                edgecolors="none",
                zorder=2,
            )
            fig.colorbar(sc, ax=ax1, label=quality_label)
        else:
            ax1.scatter(mag, err, s=8, alpha=0.35, c="C0", edgecolors="none", zorder=2)
        _draw_calibrator_markers(
            ax1,
            mag,
            err,
            used,
            rejected,
            legend=i == 0 and has_cal,
            used_edge="k",
        )
        if i == 0 and has_cal:
            ax1.legend(loc="upper left", fontsize=8)
        ax1.set_yscale("log")
        ax1.set_ylim(ax0.get_ylim())
        ax1.set_ylabel(r"$\sigma_m$ [mag]")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.set_title(quality_label, fontsize=10, pad=8)
        if i >= last_row_start:
            ax1.set_xlabel(x_label)
        else:
            plt.setp(ax1.get_xticklabels(), visible=False)
    if not extra_axes:
        ax0.set_xlabel(x_label)

    path = _diagnostic_plot_path(output_dir, stem, file_type)
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_photometry_mag_vs_error_overview(
    mag_by_image: list[np.ndarray],
    err_by_image: list[np.ndarray],
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    band_label: str = "",
    image_labels: list[str] | None = None,
    image_jd: list[float] | None = None,
    image_airmass: list[float] | None = None,
    filename_stem: str | None = None,
) -> Path | None:
    """
    Overview: pooled mag–σ density (log y) plus per-image error level vs JD
    and/or airmass (image index if neither is available).
    """
    if not mag_by_image or len(mag_by_image) != len(err_by_image):
        return None
    pooled_m: list[float] = []
    pooled_e: list[float] = []
    med_err: list[float] = []
    for mag, err in zip(mag_by_image, err_by_image, strict=True):
        m, e = _finite_mag_and_positive_err(mag, err)
        if m.size:
            pooled_m.extend(m.tolist())
            pooled_e.extend(e.tolist())
            bright = m < np.nanpercentile(m, 40)
            med_err.append(
                float(np.median(e[bright])) if np.any(bright) else float(np.median(e))
            )
        else:
            med_err.append(np.nan)

    if not pooled_m:
        return None

    mag = np.asarray(pooled_m, dtype=float)
    err = np.asarray(pooled_e, dtype=float)
    med = np.asarray(med_err, dtype=float)
    n_img = len(mag_by_image)

    jd = None if image_jd is None else np.asarray(image_jd, dtype=float)
    airmass = None if image_airmass is None else np.asarray(image_airmass, dtype=float)
    has_jd = jd is not None and jd.size == n_img and np.any(np.isfinite(jd))
    has_am = (
        airmass is not None and airmass.size == n_img and np.any(np.isfinite(airmass))
    )
    extra_axes: list[tuple[str, np.ndarray | None]] = []
    if has_jd:
        extra_axes.append(("jd", jd))
    if has_am:
        extra_axes.append(("airmass", airmass))
    if not extra_axes:
        extra_axes.append(("index", np.arange(n_img, dtype=float)))

    n_rows = 1 + len(extra_axes)
    height = 4.6 + 3.1 * len(extra_axes)
    ratios = [1.45] + [1.0] * len(extra_axes)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(7.0, height),
        gridspec_kw={"height_ratios": ratios, "hspace": 0.28},
    )
    if n_rows == 1:
        axes = [axes]
    ax0 = axes[0]

    pcm = _draw_mag_err_density(ax0, mag, err)
    fig.colorbar(pcm, ax=ax0, label="N per bin")
    _draw_trend_and_photon_model(ax0, mag, err)
    _draw_snr_guides(ax0)
    _expand_ylim_for_snr_guides(ax0)
    ax0.set_xlabel("Instrumental magnitude [mag]")
    ax0.set_ylabel(r"$\sigma_m$ [mag]")
    ax0.grid(True, which="both", alpha=0.3)
    _finalize_mag_err_legend(ax0)
    ax0.text(
        0.97,
        0.03,
        _mag_err_stats_text(mag, err) + f"\nimages = {n_img}",
        transform=ax0.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )
    ttl = "Magnitude vs. uncertainty (all images)"
    if band_label:
        ttl += f" ({band_label})"
    ax0.set_title(ttl, fontsize=11)

    for ax, (kind, xvals) in zip(axes[1:], extra_axes, strict=True):
        y = med
        finite = np.isfinite(y)
        if xvals is not None:
            finite &= np.isfinite(xvals)
        if kind == "jd" and xvals is not None:
            jd0 = float(np.nanmin(xvals[finite])) if np.any(finite) else 0.0
            x = xvals - jd0
            xlabel = f"JD − {jd0:.5f}"
            title = "Per-image photometric error vs time"
        elif kind == "airmass" and xvals is not None:
            x = xvals
            xlabel = "Airmass"
            title = "Per-image photometric error vs airmass"
        else:
            x = np.arange(n_img, dtype=float) if xvals is None else xvals
            xlabel = "Image index"
            title = "Per-image photometric error level"
            if (
                image_labels is not None
                and len(image_labels) == n_img
                and n_img <= 25
            ):
                ax.set_xticks(x)
                ax.set_xticklabels(image_labels, rotation=90, fontsize=7)
        ax.plot(x[finite], y[finite], "o-", ms=4, lw=1.0, color="C0")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Median $\sigma_m$ (bright 40%) [mag]")
        ax.set_title(title, fontsize=10)
        if np.any(np.isfinite(y) & (y > 0)):
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

    stem = filename_stem or (
        f"photometry_mag_vs_error_overview_{band_label}"
        if band_label
        else "photometry_mag_vs_error_overview"
    )
    path = _diagnostic_plot_path(output_dir, stem, file_type)
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_exposure_pairing_overview(
    pair_records: Table,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    reference_filter: str = "",
    pairing_mode: str = "",
    filename_stem: str = "exposure_pairing_overview",
) -> Path | None:
    """
    Plot max |ΔJD| vs exposure-pair index and annotate image ids.

    Expects columns: ``pair_index``, ``max_abs_delta_jd_day``, and optionally
    ``image_ids`` (string) / ``ref_jd``.
    """
    if len(pair_records) == 0 or "pair_index" not in pair_records.colnames:
        return None
    idx = np.asarray(pair_records["pair_index"], dtype=float)
    if "max_abs_delta_jd_day" in pair_records.colnames:
        dj = np.asarray(pair_records["max_abs_delta_jd_day"], dtype=float)
    else:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(idx, dj * 1440.0, "o-", ms=3, lw=1.0, color="C0")
    ax.set_xlabel("Exposure pair index")
    ax.set_ylabel(r"max $|\Delta\mathrm{JD}|$ [minutes]")
    title = "Exposure pairing time offsets"
    if pairing_mode:
        title += f" ({pairing_mode}"
        if reference_filter:
            title += f", ref={reference_filter}"
        title += ")"
    elif reference_filter:
        title += f" (ref={reference_filter})"
    title += f"\nn_pairs={len(pair_records)}"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_aperture_growth_curve(
    image_data: np.ndarray,
    x_pixel: float,
    y_pixel: float,
    radii_pixel: np.ndarray,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    normalize: bool = False,
    filename_stem: str = "photometry_radial_growth_curve",
) -> Path:
    """
    Encircled flux vs. aperture radius (pixels) using circular apertures.

    Parameters
    ----------
    image_data
        2D array (background not subtracted here; use a background-subtracted
        cutout when possible).
    """
    from photutils.aperture import CircularAperture, aperture_photometry

    r = np.asarray(radii_pixel, dtype=float)
    r = r[r > 0]
    fluxes = []
    for rad in r:
        ap = CircularAperture((x_pixel, y_pixel), r=rad)
        phot = aperture_photometry(image_data, ap)
        fluxes.append(float(phot["aperture_sum"][0]))
    fluxes = np.asarray(fluxes, dtype=float)
    yplot = fluxes
    if normalize and np.nanmax(np.abs(fluxes)) > 0:
        yplot = fluxes / np.nanmax(fluxes)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(r, yplot, "o-", ms=3, lw=1)
    ax.set_xlabel("Radius [pixel]")
    ax.set_ylabel("Encircled flux" + (" (normalized)" if normalize else ""))
    ax.set_title("Aperture growth curve")
    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_zeropoint_residual_vs_color(
    color_index: np.ndarray | None,
    zp_residuals: np.ndarray | None,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    band_label: str = "",
    color_label: str | None = None,
    filename_stem: str | None = None,
    residuals_by_band: dict[str, np.ndarray] | None = None,
) -> Path | None:
    """
    Scatter of ZP residuals vs. a color index (e.g. catalog ``B-V``) for cal stars.

    Pass either ``(color_index, zp_residuals)`` for one band, or ``color_index`` plus
    ``residuals_by_band`` (filter name → residual array, same row order as ``color_index``)
    to overlay several bands with distinct colors and a legend.

    Residual convention matches :func:`plot_zeropoint_residual_distribution`
    (``m_cat - m_inst - ZP`` per band).
    """
    clab = (
        color_label
        if color_label is not None
        else r"$(B-V)_\mathrm{lit}$ [mag]"
    )
    ylabel = r"$m_\mathrm{cat} - m_\mathrm{inst} - \mathrm{ZP}$ [mag]"

    if residuals_by_band:
        if color_index is None:
            return None
        ci_full = np.asarray(color_index, dtype=float)
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        any_pts = False
        for i, (lab, rz_raw) in enumerate(residuals_by_band.items()):
            rz = np.asarray(rz_raw, dtype=float)
            k = min(ci_full.size, rz.size)
            if k == 0:
                continue
            ci, rz = ci_full[:k], rz[:k]
            ok = np.isfinite(ci) & np.isfinite(rz)
            if not np.any(ok):
                continue
            any_pts = True
            ax.scatter(
                ci[ok],
                rz[ok],
                s=18,
                alpha=0.65,
                label=str(lab),
                color=f"C{i % 10}",
                edgecolors="none",
            )
        if not any_pts:
            plt.close(fig)
            return None
        ax.axhline(0.0, color="k", linestyle="--", lw=1, alpha=0.55)
        ax.legend(title="Band", fontsize=9, loc="best")
        ax.set_xlabel(clab)
        ax.set_ylabel(ylabel)
        keys = list(residuals_by_band.keys())
        stem = filename_stem or (
            "zeropoint_residual_vs_color_" + "_".join(keys)
            if len(keys) > 1
            else f"zeropoint_residual_vs_color_{keys[0]}"
        )
        ttl = "ZP residuals vs. color"
        if len(keys) > 1:
            ttl += f" ({', '.join(keys)})"
        elif band_label:
            ttl += f" ({band_label})"
        ax.set_title(ttl)
        path = _diagnostic_plot_path(output_dir, stem, file_type)
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
        plt.close(fig)
        return path

    if color_index is None or zp_residuals is None:
        return None
    ci = np.asarray(color_index, dtype=float)
    rz = np.asarray(zp_residuals, dtype=float)
    ok = np.isfinite(ci) & np.isfinite(rz)
    if not np.any(ok):
        return None
    ci = ci[ok]
    rz = rz[ok]
    stem = filename_stem or (
        f"zeropoint_residual_vs_color_{band_label}"
        if band_label
        else "zeropoint_residual_vs_color"
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(ci, rz, s=14, alpha=0.55, c="C0", edgecolors="none")
    ax.axhline(0.0, color="k", linestyle="--", lw=1, alpha=0.55)
    ax.set_xlabel(clab)
    ax.set_ylabel(ylabel)
    ttl = "ZP residuals vs. color"
    if band_label:
        ttl += f" ({band_label})"
    ax.set_title(ttl)
    path = _diagnostic_plot_path(output_dir, stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_instrumental_vs_catalog_magnitudes(
    m_instrumental: np.ndarray,
    m_catalog: np.ndarray,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    band_label: str = "",
    filename_stem: str | None = None,
    show_one_to_one: bool = False,
    x_label: str | None = None,
    title: str | None = None,
) -> Path | None:
    """
    Scatter observed vs. catalog magnitudes for calibration stars.

    For raw instrumental magnitudes, leave ``show_one_to_one=False`` (default): a
    1:1 line is misleading because of the zero-point offset. Set
    ``show_one_to_one=True`` when the x values are calibrated (e.g. ``m_cal``);
    then a **second panel** below shows a scatter of signed offsets from the
    1:1 relation, ``m_cat - m_obs``, vs. ``m_obs`` (x-axis magnitude, e.g. ``m_cal``).
    """
    mi = np.asarray(m_instrumental, dtype=float)
    mc = np.asarray(m_catalog, dtype=float)
    ok = np.isfinite(mi) & np.isfinite(mc)
    if not np.any(ok):
        return None
    xi = mi[ok]
    yc = mc[ok]
    stem = filename_stem or (
        f"instrumental_vs_catalog_{band_label}"
        if band_label
        else "instrumental_vs_catalog_magnitudes"
    )
    xl = x_label if x_label is not None else r"$m_\mathrm{inst}$ [mag]"
    ttl = title
    if ttl is None:
        ttl = "Cal stars: instrumental vs. catalog" + (
            f" ({band_label})" if band_label else ""
        )

    if show_one_to_one:
        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            figsize=(5, 7.2),
            gridspec_kw={"height_ratios": [2.4, 1.15], "hspace": 0.3},
        )
        ax0.scatter(xi, yc, s=12, alpha=0.5, c="C0", edgecolors="none")
        lo = float(np.nanmin([np.nanmin(xi), np.nanmin(yc)]))
        hi = float(np.nanmax([np.nanmax(xi), np.nanmax(yc)]))
        ax0.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax0.set_aspect("equal", adjustable="box")
        ax0.set_xlabel(xl)
        ax0.set_ylabel(r"$m_\mathrm{cat}$ [mag]")
        ax0.set_title(ttl)

        res = yc - xi
        ax1.scatter(
            xi,
            res,
            s=14,
            alpha=0.55,
            c="C2",
            edgecolors="none",
        )
        ax1.axhline(0.0, color="k", linestyle="--", lw=1, alpha=0.55)
        ax1.set_xlabel(xl)
        ax1.set_ylabel(r"$m_\mathrm{cat} - m_\mathrm{obs}$ [mag]")
        ax1.set_title("Residuals from 1:1 line", fontsize=10)
    else:
        fig, ax0 = plt.subplots(figsize=(5, 5))
        ax0.scatter(xi, yc, s=12, alpha=0.5, c="C0", edgecolors="none")
        ax0.set_aspect("auto")
        ax0.set_xlabel(xl)
        ax0.set_ylabel(r"$m_\mathrm{cat}$ [mag]")
        ax0.set_title(ttl)

    path = _diagnostic_plot_path(output_dir, stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_zeropoint_residual_distribution(
    residuals: np.ndarray | None,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    band_label: str = "",
    filename_stem: str | None = None,
    residuals_by_band: dict[str, np.ndarray] | None = None,
) -> Path | None:
    """
    Histogram of ZP residuals :math:`m_\\mathrm{cat} - m_\\mathrm{inst} - \\mathrm{ZP}`.

    Pass either a single ``residuals`` array or ``residuals_by_band`` (filter name →
    residuals) to overlay several filters in **one** figure with a common binning and
    a legend (fewer files for reports).
    """
    ax_xlabel = r"$m_\mathrm{cat} - m_\mathrm{inst} - \mathrm{ZP}$ [mag]"

    if residuals_by_band:
        series: dict[str, np.ndarray] = {}
        for label, arr in residuals_by_band.items():
            x = np.asarray(arr, dtype=float)
            x = x[np.isfinite(x)]
            if x.size > 0:
                series[str(label)] = x
        if not series:
            return None
        combined = np.concatenate(list(series.values()))
        edges = np.histogram_bin_edges(combined, bins="auto")
        keys = list(series.keys())
        stem = filename_stem or (
            "zeropoint_residuals_" + "_".join(keys)
            if len(keys) > 1
            else f"zeropoint_residuals_{keys[0]}"
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (lab, x) in enumerate(series.items()):
            ax.hist(
                x,
                bins=edges,
                alpha=0.55,
                label=lab,
                color=f"C{(i + 2) % 10}",
                edgecolor="k",
                linewidth=0.25,
            )
        ax.legend(title="Filter", fontsize=9)
        ax.set_xlabel(ax_xlabel)
        ax.set_ylabel("Count")
        ttl = "ZP residual distribution"
        if len(keys) > 1:
            ttl += f" ({', '.join(keys)})"
        elif len(keys) == 1:
            ttl += f" ({keys[0]})"
        ax.set_title(ttl)
        path = _diagnostic_plot_path(output_dir, stem, file_type)
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
        plt.close(fig)
        return path

    if residuals is None:
        return None
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    stem = filename_stem or (
        f"zeropoint_residuals_{band_label}" if band_label else "zeropoint_residual_distribution"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins="auto", color="C2", alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(ax_xlabel)
    ax.set_ylabel("Count")
    ax.set_title("ZP residual distribution" + (f" ({band_label})" if band_label else ""))
    path = _diagnostic_plot_path(output_dir, stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_calibration_color_color_cal_stars(
    color_literature: np.ndarray,
    color_observed: np.ndarray,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    filename_stem: str = "calibration_color_color_cal_stars",
    show_residual_panel: bool = True,
) -> Path | None:
    """
    Literature vs. observed color (e.g. ``B-V``) for calibration stars.

    With ``show_residual_panel=True`` (default), a second axes below shows
    ``(B-V)_lit - (B-V)_obs`` vs. ``(B-V)_lit`` and a horizontal line at zero,
    analogous to the residual panel in :func:`plot_instrumental_vs_catalog_magnitudes`
    when ``show_one_to_one=True``.
    """
    cl = np.asarray(color_literature, dtype=float)
    co = np.asarray(color_observed, dtype=float)
    ok = np.isfinite(cl) & np.isfinite(co)
    if not np.any(ok):
        return None
    cl_ok = cl[ok]
    co_ok = co[ok]
    xlit = r"$(B-V)_\mathrm{lit}$ [mag]"
    yobs = r"$(B-V)_\mathrm{obs}$ [mag]"

    if show_residual_panel:
        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            figsize=(5, 7.2),
            gridspec_kw={"height_ratios": [2.4, 1.15], "hspace": 0.3},
        )
        ax0.scatter(cl_ok, co_ok, s=14, alpha=0.55, c="C3", edgecolors="none")
        lo = float(np.nanmin([np.nanmin(cl_ok), np.nanmin(co_ok)]))
        hi = float(np.nanmax([np.nanmax(cl_ok), np.nanmax(co_ok)]))
        ax0.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax0.set_xlabel(xlit)
        ax0.set_ylabel(yobs)
        ax0.set_title("Calibration stars: color-color")
        ax0.set_aspect("equal", adjustable="box")

        res = cl_ok - co_ok
        ax1.scatter(
            cl_ok,
            res,
            s=16,
            alpha=0.55,
            c="C2",
            edgecolors="none",
        )
        ax1.axhline(0.0, color="k", linestyle="--", lw=1, alpha=0.55)
        ax1.set_xlabel(xlit)
        ax1.set_ylabel(r"$(B-V)_\mathrm{lit} - (B-V)_\mathrm{obs}$ [mag]")
        ax1.set_title("Residuals from 1:1 line", fontsize=10)
    else:
        fig, ax0 = plt.subplots(figsize=(5, 5))
        ax0.scatter(cl_ok, co_ok, s=14, alpha=0.55, c="C3", edgecolors="none")
        lo = float(np.nanmin([np.nanmin(cl_ok), np.nanmin(co_ok)]))
        hi = float(np.nanmax([np.nanmax(cl_ok), np.nanmax(co_ok)]))
        ax0.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax0.set_xlabel(xlit)
        ax0.set_ylabel(yobs)
        ax0.set_title("Calibration stars: color-color")
        ax0.set_aspect("equal", adjustable="box")

    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def _pixel_radial_tangential(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    x0: float,
    y0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radius and residual components parallel / perpendicular to the radius vector.

    ``d_radial`` > 0 is outward (plate-scale / pincushion-like).
    ``d_tangential`` > 0 is counterclockwise (rotation).
    """
    vx = np.asarray(x, dtype=float) - x0
    vy = np.asarray(y, dtype=float) - y0
    radius = np.hypot(vx, vy)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        urx = np.where(radius > 0, vx / radius, 0.0)
        ury = np.where(radius > 0, vy / radius, 0.0)
    d_radial = dx * urx + dy * ury
    d_tangential = dx * (-ury) + dy * urx
    return radius, d_radial, d_tangential


def _geometry_field_center(
    x: np.ndarray,
    y: np.ndarray,
    image_data: np.ndarray | None,
) -> tuple[float, float]:
    if image_data is not None and image_data.ndim >= 2:
        ny, nx = image_data.shape[-2], image_data.shape[-1]
        return (nx - 1) / 2.0, (ny - 1) / 2.0
    return float(np.nanmedian(x)), float(np.nanmedian(y))


def residual_geometry_summary(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    x0: float | None = None,
    y0: float | None = None,
    min_radius_pix: float = 5.0,
) -> dict[str, float]:
    """Rotation (arcmin) and scale (fraction) implied by median d/r.

    A median ``(dx, dy)`` translation is removed before the radial/tangential
    split so a bulk WCS offset is not mistaken for rotation or plate-scale.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    nan = float("nan")
    summary = {
        "n": 0.0,
        "median_dx_pix": nan,
        "median_dy_pix": nan,
        "median_abs_pix": nan,
        "rms_radial_pix": nan,
        "rms_tangential_pix": nan,
        "rotation_arcmin": nan,
        "scale_fraction": nan,
    }
    if not np.any(ok):
        return summary
    if x0 is None or y0 is None:
        x0, y0 = _geometry_field_center(x[ok], y[ok], None)
    summary["median_dx_pix"] = float(np.median(dx[ok]))
    summary["median_dy_pix"] = float(np.median(dy[ok]))
    dx_c = dx - summary["median_dx_pix"]
    dy_c = dy - summary["median_dy_pix"]
    radius, d_rad, d_tan = _pixel_radial_tangential(x, y, dx_c, dy_c, x0, y0)
    ok = ok & np.isfinite(radius) & (radius >= min_radius_pix)
    summary["n"] = float(np.count_nonzero(ok))
    if not np.any(ok):
        return summary
    abs_res = np.hypot(dx, dy)
    summary["median_abs_pix"] = float(np.median(abs_res[ok]))
    summary["rms_radial_pix"] = float(np.sqrt(np.mean(d_rad[ok] ** 2)))
    summary["rms_tangential_pix"] = float(np.sqrt(np.mean(d_tan[ok] ** 2)))
    summary["rotation_arcmin"] = float(
        np.degrees(np.median(d_tan[ok] / radius[ok])) * 60.0
    )
    summary["scale_fraction"] = float(np.median(d_rad[ok] / radius[ok]))
    return summary


def plot_inter_filter_correlation_geometry(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    image_data: np.ndarray | None = None,
    sep_arcsec: np.ndarray | None = None,
    reference_filter: str = "",
    other_filter: str = "",
    filename_stem: str = "inter_filter_correlation_geometry",
    title_suffix: str = "",
    title: str | None = None,
) -> Path | None:
    """
    Diagnose match tails: quiver on the image plus radial vs tangential residuals
    (scale/distortion vs rotation). Used for inter-filter pairs and catalog matches.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    if sep_arcsec is not None:
        sep = np.asarray(sep_arcsec, dtype=float)
        ok &= np.isfinite(sep)
    else:
        sep = np.hypot(dx, dy)
    if not np.any(ok):
        return None
    x, y, dx, dy, sep = x[ok], y[ok], dx[ok], dy[ok], sep[ok]
    if image_data is not None:
        image_data = np.asarray(image_data, dtype=float)
        while image_data.ndim > 2:
            image_data = image_data[0]
        if image_data.ndim != 2:
            image_data = None
    x0, y0 = _geometry_field_center(x, y, image_data)
    stats = residual_geometry_summary(x, y, dx, dy, x0=x0, y0=y0)
    dx_c = dx - stats["median_dx_pix"]
    dy_c = dy - stats["median_dy_pix"]
    radius, d_rad, d_tan = _pixel_radial_tangential(x, y, dx_c, dy_c, x0, y0)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 9.2),
        gridspec_kw={"wspace": 0.28, "hspace": 0.32},
    )
    ax_img, ax_r = axes[0]
    ax_rad, ax_tan = axes[1]

    if image_data is not None and np.isfinite(image_data).any():
        finite = image_data[np.isfinite(image_data)]
        try:
            norm = ImageNormalize(image_data, interval=ZScaleInterval())
        except Exception:
            lo, hi = np.percentile(finite, (1.0, 99.5))
            norm = ImageNormalize(vmin=lo, vmax=max(hi, lo + 1e-6))
        ax_img.imshow(
            image_data,
            origin="lower",
            cmap="gray",
            norm=norm,
            interpolation="nearest",
        )
    ax_img.set_aspect("equal")

    n_show = min(x.size, 400)
    if x.size > n_show:
        rng = np.random.default_rng(0)
        show = rng.choice(x.size, n_show, replace=False)
    else:
        show = np.arange(x.size)
    abs_res = np.hypot(dx, dy)
    med_len = float(np.median(abs_res[abs_res > 0])) if np.any(abs_res > 0) else 1.0
    span = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    magnify = (0.06 * span) / max(med_len, 1e-6)
    q = ax_img.quiver(
        x[show],
        y[show],
        dx[show] * magnify,
        dy[show] * magnify,
        sep[show],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        cmap="plasma",
        pivot="tail",
    )
    fig.colorbar(q, ax=ax_img, label="Separation [arcsec]")
    ax_img.plot(x0, y0, "c+", ms=10, mew=1.2)
    ax_img.set_xlabel("x [pix]")
    ax_img.set_ylabel("y [pix]")
    ax_img.set_title("Residual vectors on reference image", fontsize=10)
    ax_img.text(
        0.02,
        0.02,
        f"arrows ×{magnify:.1f}",
        transform=ax_img.transAxes,
        fontsize=8,
        color="w" if image_data is not None else "k",
        bbox=dict(boxstyle="round", facecolor="k", alpha=0.35),
    )

    ax_r.scatter(radius, sep, s=8, alpha=0.4, c="C1", edgecolors="none")
    ax_r.set_xlabel("Radius from centre [pix]")
    ax_r.set_ylabel("Separation [arcsec]")
    ax_r.set_title("|offset| vs radius", fontsize=10)
    ax_r.grid(True, alpha=0.3)

    r_line = np.array(
        [float(np.nanmin(radius)), float(np.nanmax(radius))], dtype=float
    )
    if not np.all(np.isfinite(r_line)) or r_line[1] <= r_line[0]:
        r_line = np.array([0.0, float(np.nanmax(radius)) if np.any(radius) else 1.0])

    ax_rad.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_rad.scatter(radius, d_rad, s=8, alpha=0.4, c="C0", edgecolors="none")
    if np.isfinite(stats["scale_fraction"]) and abs(stats["scale_fraction"]) > 1e-12:
        ax_rad.plot(
            r_line,
            stats["scale_fraction"] * r_line,
            "k--",
            lw=1.0,
            alpha=0.75,
            label="median d/r",
        )
        ax_rad.legend(fontsize=7, loc="upper left")
    ax_rad.set_xlabel("Radius from centre [pix]")
    ax_rad.set_ylabel(r"Radial residual [pix]")
    ax_rad.set_title("Scale / distortion (radial, translation removed)", fontsize=10)
    ax_rad.grid(True, alpha=0.3)

    theta_rad = np.radians(stats["rotation_arcmin"] / 60.0)
    ax_tan.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_tan.scatter(radius, d_tan, s=8, alpha=0.4, c="C2", edgecolors="none")
    if np.isfinite(theta_rad) and abs(theta_rad) > 1e-12:
        ax_tan.plot(
            r_line,
            theta_rad * r_line,
            "k--",
            lw=1.0,
            alpha=0.75,
            label="median d/r",
        )
        ax_tan.legend(fontsize=7, loc="upper left")
    ax_tan.set_xlabel("Radius from centre [pix]")
    ax_tan.set_ylabel(r"Tangential residual [pix]")
    ax_tan.set_title("Rotation (tangential, translation removed)", fontsize=10)
    ax_tan.grid(True, alpha=0.3)

    box = (
        f"N = {int(stats['n'])}\n"
        f"median (dx, dy) = ({stats['median_dx_pix']:.3f}, "
        f"{stats['median_dy_pix']:.3f}) pix\n"
        f"median |res| = {stats['median_abs_pix']:.3f} pix\n"
        f"rms radial = {stats['rms_radial_pix']:.3f} pix\n"
        f"rms tangential = {stats['rms_tangential_pix']:.3f} pix\n"
        f"implied rotation = {stats['rotation_arcmin']:.2f}'\n"
        f"implied scale = {100.0 * stats['scale_fraction']:.3f} %"
    )
    ax_r.text(
        0.97,
        0.97,
        box,
        transform=ax_r.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )

    plot_title = title
    if plot_title is None:
        plot_title = "Inter-filter residual geometry"
        if reference_filter or other_filter:
            plot_title += f" ({reference_filter} → {other_filter})"
    if title_suffix:
        plot_title += f"\n{title_suffix}"
    fig.suptitle(plot_title, fontsize=12)
    fig.subplots_adjust(top=0.90)

    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_inter_filter_correlation_geometry_overview(
    pair_summaries: list[dict],
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    pair_labels: list[str] | None = None,
    filename_stem: str = "inter_filter_correlation_geometry_overview",
    title_suffix: str = "",
) -> Path | None:
    """Per-pair rms radial vs tangential: distortion-like vs rotation-like."""
    if not pair_summaries:
        return None
    rms_r = np.array([float(s["rms_radial_pix"]) for s in pair_summaries], dtype=float)
    rms_t = np.array(
        [float(s["rms_tangential_pix"]) for s in pair_summaries], dtype=float
    )
    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.25, 0.32, 1.0],
        hspace=0.12,
    )
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])
    ax0.scatter(rms_r, rms_t, s=28, c="C1", edgecolors="k", linewidths=0.3)
    hi = float(np.nanmax([np.nanmax(rms_r), np.nanmax(rms_t), 0.05]))
    ax0.plot([0, hi], [0, hi], "k--", lw=1, alpha=0.55)
    ax0.set_xlabel("RMS radial residual [pix]")
    ax0.set_ylabel("RMS tangential residual [pix]")
    ax0.set_title("Per pair: distortion-like vs rotation-like", fontsize=10)
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, alpha=0.3)
    ax0.text(
        0.03,
        0.97,
        "above 1:1 → rotation\nbelow 1:1 → scale/distortion",
        transform=ax0.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="w", alpha=0.7),
    )

    xi = np.arange(len(pair_summaries))
    ax1.plot(xi, rms_r, "o-", ms=4, lw=1.0, color="C0", label="rms radial")
    ax1.plot(xi, rms_t, "s-", ms=4, lw=1.0, color="C2", label="rms tangential")
    ax1.set_xlabel("Exposure pair")
    ax1.set_ylabel("RMS residual [pix]")
    ax1.set_title("Field pattern vs pair (translation removed)", fontsize=10, pad=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    if pair_labels is not None and len(pair_labels) == len(xi) and len(xi) <= 25:
        ax1.set_xticks(xi)
        ax1.set_xticklabels(pair_labels, rotation=90, fontsize=7)

    title = "Inter-filter residual geometry (all pairs)"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.92, bottom=0.12)
    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_inter_filter_correlation_separations(
    separations_arcsec: np.ndarray,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    reference_filter: str = "",
    other_filters: list[str] | None = None,
    filename_stem: str = "inter_filter_correlation_separations",
    title_suffix: str = "",
) -> Path:
    """Histogram of inter-filter position separations with summary statistics."""
    x = np.asarray(separations_arcsec, dtype=float)
    x = x[np.isfinite(x)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if x.size:
        ax.hist(x, bins="auto", color="C1", alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Separation [arcsec]")
    ax.set_ylabel("Count")
    title = "Inter-filter correlation separations"
    if reference_filter:
        title += f" (ref={reference_filter}"
        if other_filters:
            title += f", vs {','.join(other_filters)}"
        title += ")"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=11)
    ax.text(
        0.97,
        0.97,
        _stats_text(x),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path


def plot_inter_filter_correlation_separations_overview(
    separations_by_pair: list[np.ndarray],
    pair_labels: list[str],
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    reference_filter: str = "",
    other_filters: list[str] | None = None,
    filename_stem: str = "inter_filter_correlation_separations_overview",
    pairing_mode: str = "",
) -> Path:
    """
    Overview for many exposure pairs: pooled histogram + median separation vs pair.
    """
    medians: list[float] = []
    pooled: list[float] = []
    for sep in separations_by_pair:
        x = np.asarray(sep, dtype=float)
        x = x[np.isfinite(x)]
        pooled.extend(x.tolist())
        medians.append(float(np.nanmedian(x)) if x.size else np.nan)

    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(7.5, 7.0), gridspec_kw={"height_ratios": [1.2, 1.0]}
    )
    pooled_arr = np.asarray(pooled, dtype=float)
    if pooled_arr.size:
        ax0.hist(
            pooled_arr,
            bins="auto",
            color="C1",
            alpha=0.85,
            edgecolor="k",
            linewidth=0.3,
        )
    ax0.set_xlabel("Separation [arcsec]")
    ax0.set_ylabel("Count")
    title = "Inter-filter separations (all exposure pairs)"
    if reference_filter:
        title += f" (ref={reference_filter}"
        if other_filters:
            title += f", vs {','.join(other_filters)}"
        title += ")"
    if pairing_mode:
        title += f"\npairing={pairing_mode}, n_pairs={len(separations_by_pair)}"
    ax0.set_title(title, fontsize=11)
    ax0.text(
        0.97,
        0.97,
        _stats_text(pooled_arr),
        transform=ax0.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    xi = np.arange(len(medians))
    ax1.plot(xi, medians, "o-", ms=3, lw=1.0, color="C0")
    ax1.set_xlabel("Exposure pair index")
    ax1.set_ylabel("Median separation [arcsec]")
    ax1.set_title("Median separation per exposure pair")
    ax1.grid(True, alpha=0.3)
    if len(pair_labels) <= 20 and pair_labels:
        ax1.set_xticks(xi)
        ax1.set_xticklabels(pair_labels, rotation=90, fontsize=7)
    path = _diagnostic_plot_path(output_dir, filename_stem, file_type)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", format=file_type.lstrip("."))
    plt.close(fig)
    return path
