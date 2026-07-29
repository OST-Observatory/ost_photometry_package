"""Calibration, extinction-fit, and diagnostic QC plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

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
    from ... import checks

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
    from ... import checks

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
    """
    from ... import checks

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
        n_excl = np.sum(~mask)
        if n_excl > 0:
            ax1.scatter(color[~mask], delta[~mask], alpha=0.4, s=15, c="gray", label="excluded")
        ax1.scatter(color[mask], delta[mask], alpha=0.7, s=25, c="C0", label="used")
        c_used = np.asarray(color, dtype=float)[mask]
        c_finite = c_used[np.isfinite(c_used)]
        if len(c_finite) > 0 and np.nanmax(c_finite) - np.nanmin(c_finite) > 0.01:
            c_min, c_max = float(np.nanmin(c_finite)), float(np.nanmax(c_finite))
            c_line = np.linspace(c_min, c_max, 50)
            ax1.plot(c_line, T * c_line + ZP, "C1-", lw=2, label="Fit")
        else:
            ax1.axhline(ZP, color="C1", ls="-", lw=2, label="Fit (ZP only)")
        ax1.set_xlabel(f"Color {ci} [mag]")
        ax1.set_ylabel(r"$m_{\mathrm{std}} - m_{\mathrm{inst}}$ [mag]")
        ax1.set_title(f"{epoch_id} {filter_}: T={T:.4f}, ZP={ZP:.4f}")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Right: residuals
        residuals = delta - (T * color + ZP)
        if n_excl > 0:
            ax2.scatter(color[~mask], residuals[~mask], alpha=0.4, s=15, c="gray")
        ax2.scatter(color[mask], residuals[mask], alpha=0.7, s=25, c="C0")
        ax2.axhline(0, color="C1", ls="--", lw=1)
        ax2.set_xlabel(f"Color {ci} [mag]")
        ax2.set_ylabel("Residual [mag]")
        rms_val = np.nanstd(residuals[mask]) if np.sum(mask) > 0 else 0.0
        ax2.set_title(f"RMS = {rms_val:.4f} mag")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_id = str(epoch_id).replace("/", "_").replace(":", "_")
        plt.savefig(
            out / f"calibration_{safe_id}_{filter_}.{file_type}",
            bbox_inches="tight",
            format=file_type,
        )
        plt.close()


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
        for :class:`CoefficientMode.PER_IMAGE` runs.
    x_jd : list of float, optional
        If given, same length as ``epoch_ids`` and all finite: x-axis is these
        Julian Dates. Otherwise x is 0..N-1 with thinned / shortened epoch labels.
    """
    import warnings

    from ... import checks
    from ..warnings_types import OstPhotometryAnalyzeWarning

    out = Path(output_dir) / "calibration"
    checks.check_output_directories(out)

    n_filt = len(filters)
    if n_filt == 0:
        return
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

    plt.savefig(
        out / f"{output_basename}.{file_type}",
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
) -> Path | None:
    """Scatter of instrumental magnitude vs. uncertainty from a photometry table."""
    if "mags_fit" not in photometry.colnames or "mags_unc" not in photometry.colnames:
        return None
    _mf, _mu = photometry["mags_fit"], photometry["mags_unc"]
    mag = np.asarray(_mf.value if hasattr(_mf, "value") else _mf, dtype=float)
    err = np.asarray(_mu.value if hasattr(_mu, "value") else _mu, dtype=float)
    # ok = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    ok = np.isfinite(mag) & np.isfinite(err)
    if not np.any(ok):
        return None
    stem = filename_stem or (
        f"photometry_mag_vs_error_{band_label}" if band_label else "photometry_mag_vs_error"
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(mag[ok], err[ok], s=8, alpha=0.35, c="C0", edgecolors="none")
    ax.set_xlabel("mags_fit [mag]")
    ax.set_ylabel("mags_unc [mag]")
    ttl = "Magnitude vs. uncertainty"
    if band_label:
        ttl += f" ({band_label})"
    ax.set_title(ttl)
    path = _diagnostic_plot_path(output_dir, stem, file_type)
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


def plot_inter_filter_correlation_separations(
    separations_arcsec: np.ndarray,
    output_dir: str | Path,
    file_type: str = "pdf",
    *,
    reference_filter: str = "",
    other_filters: list[str] | None = None,
    filename_stem: str = "inter_filter_correlation_separations",
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
