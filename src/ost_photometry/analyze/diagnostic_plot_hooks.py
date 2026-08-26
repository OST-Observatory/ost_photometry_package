"""
Optional pipeline hooks for :class:`~ost_photometry.analyze.pipeline.config.DiagnosticPlots`.

Figures go to ``<output_dir>/diagnostics/`` using ``PipelineConfig.file_type_plots``.
Failures are logged and do not abort the pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

from .. import checks, terminal_output
from . import correlate, plots

_QC_PLOT_COLUMNS = (
    "is_comparison",
    "is_calibrator",
    "qfit",
    "cfit",
    "sharpness",
    "roundness",
    "roundness1",
    "roundness2",
    "flags",
    "fwhm",
)


def _copy_qc_plot_columns(src: Table, dest: Table) -> None:
    """Copy extraction/catalog quality columns onto a mag–error plot table."""
    for name in _QC_PLOT_COLUMNS:
        if name in src.colnames and name not in dest.colnames:
            dest[name] = src[name]


def diagnostics_subdirectory(output_dir: str | Path) -> Path:
    base = Path(output_dir)
    d = base / "diagnostics"
    checks.check_output_directories(base, d)
    return d


def _phase_requests_plots(dp: Any, phase: str) -> bool:
    """True if any diagnostic toggle for ``phase`` is enabled."""
    if phase == "extraction":
        return bool(
            dp.photometry_mag_vs_error_scatter
            or getattr(dp, "photometry_mag_vs_error_overview", False)
            or dp.photometry_radial_growth_curve
        )
    if phase == "correlation_inter":
        return bool(
            dp.correlation_inter_filter_separation_plot
            or getattr(dp, "exposure_pairing_overview", False)
        )
    if phase in ("calibration", "calibration_differential"):
        return bool(
            dp.calibration_crossmatch_separation_histogram
            or dp.photometry_mag_vs_error_scatter
            or dp.calibration_instrumental_vs_catalog
            or dp.calibration_zeropoint_residual_histogram
            or dp.calibration_zeropoint_residual_vs_color
            or dp.calibration_color_check_cal_stars
            or dp.combined_separation_histograms
        )
    return False


def _photometry_col_float(column) -> np.ndarray:
    if hasattr(column, "value"):
        return np.asarray(column.value, dtype=float)
    return np.asarray(column, dtype=float)


def _image_array_for_plot(image) -> np.ndarray | None:
    if image is None:
        return None
    try:
        data = image.get_data()
    except Exception:
        return None
    if data is None:
        return None
    arr = np.asarray(data, dtype=float)
    while arr.ndim > 2:
        arr = arr[0]
    return arr if arr.ndim == 2 else None


def _write_inter_filter_geometry(
    obs: Any,
    filter_list: list,
    images_by_filter: dict,
    *,
    reference_filter: str,
    output_dir: Path,
    file_type: str,
    filename_stem: str,
    title_suffix: str,
    write_figure: bool,
) -> list[dict]:
    frames = correlate.inter_filter_correlation_residual_frames(
        obs,
        filter_list,
        images_by_filter,
        reference_filter=reference_filter,
    )
    image_data = _image_array_for_plot(images_by_filter.get(reference_filter))
    summaries: list[dict] = []
    for fr in frames:
        summary = plots.residual_geometry_summary(
            fr["x"], fr["y"], fr["dx"], fr["dy"]
        )
        summary["other_filter"] = fr["other_filter"]
        summaries.append(summary)
        if not write_figure:
            continue
        plots.plot_inter_filter_correlation_geometry(
            fr["x"],
            fr["y"],
            fr["dx"],
            fr["dy"],
            output_dir,
            file_type,
            image_data=image_data,
            sep_arcsec=fr["sep_arcsec"],
            reference_filter=fr["reference_filter"],
            other_filter=fr["other_filter"],
            filename_stem=f"{filename_stem}_{fr['other_filter']}",
            title_suffix=title_suffix,
        )
    return summaries


def run_diagnostic_plots_phase(
    context: Any,
    config: Any,
    phase: str,
    *,
    calibration_epochs: Mapping[str, Table] | None = None,
) -> None:
    """Dispatch diagnostic figures by pipeline phase name."""
    dp = config.diagnostic_plots
    out_root = getattr(context, "output_dir", None)
    if out_root is None:
        return
    if not _phase_requests_plots(dp, phase):
        return

    # Create diagnostics/ only when at least one toggle for this phase is on.
    out_d = diagnostics_subdirectory(out_root)
    ft = getattr(config, "file_type_plots", "pdf")
    obs = getattr(context, "_observation", None)

    def _warn(msg: str) -> None:
        terminal_output.print_to_terminal(msg, style_name="WARNING")

    try:
        if phase == "extraction":
            ref_id = int(getattr(config, "reference_image_index", 0))
            for filter_ in context.filter_list:
                series = context.image_series_dict.get(filter_)
                if series is None:
                    continue
                if ref_id >= len(series.image_list):
                    continue
                img = series.image_list[ref_id]
                if img.photometry is None:
                    continue
                try:
                    if dp.photometry_mag_vs_error_scatter:
                        plots.plot_photometry_mag_vs_error(
                            img.photometry,
                            out_d,
                            ft,
                            band_label=filter_,
                            image_shape=img.get_shape(),
                        )
                    if getattr(dp, "photometry_mag_vs_error_overview", False):
                        mags: list[np.ndarray] = []
                        errs: list[np.ndarray] = []
                        labels: list[str] = []
                        jds: list[float] = []
                        airmasses: list[float] = []
                        for j, im in enumerate(series.image_list):
                            if im.photometry is None:
                                continue
                            ph = im.photometry
                            if (
                                "mags_fit" not in ph.colnames
                                or "mags_unc" not in ph.colnames
                            ):
                                continue
                            mags.append(_photometry_col_float(ph["mags_fit"]))
                            errs.append(_photometry_col_float(ph["mags_unc"]))
                            iid = getattr(im, "image_id", j)
                            labels.append(str(iid))
                            jd = getattr(im, "jd", None)
                            jds.append(float(jd) if jd is not None else np.nan)
                            am = getattr(im, "air_mass", None)
                            airmasses.append(float(am) if am is not None else np.nan)
                        if len(mags) > 1:
                            plots.plot_photometry_mag_vs_error_overview(
                                mags,
                                errs,
                                out_d,
                                ft,
                                band_label=filter_,
                                image_labels=labels,
                                image_jd=jds,
                                image_airmass=airmasses,
                            )
                    if dp.photometry_radial_growth_curve and "flux_fit" in (
                        img.photometry.colnames
                    ):
                        flux = _photometry_col_float(img.photometry["flux_fit"])
                        if flux.size == 0:
                            continue
                        iy = int(np.nanargmax(flux))
                        x0 = float(img.photometry["x_fit"][iy])
                        y0 = float(img.photometry["y_fit"][iy])
                        data = np.asarray(img.get_data(), dtype=float)
                        rmax = 0.45 * float(min(data.shape))
                        if rmax < 2.0:
                            continue
                        radii = np.linspace(0.75, min(25.0, rmax), 35)
                        plots.plot_aperture_growth_curve(
                            data,
                            x0,
                            y0,
                            radii,
                            out_d,
                            ft,
                            filename_stem=f"photometry_radial_growth_curve_{filter_}",
                        )
                except Exception as exc:
                    _warn(f"Diagnostic plot (extraction, {filter_}): {exc}")

        elif phase == "correlation_inter":
            if obs is None or len(context.filter_list) < 2:
                return
            try:
                max_pair_raw = getattr(
                    dp, "correlation_inter_filter_max_pair_plots", 25
                )
                # None → all pairs; negative treated as all
                if max_pair_raw is None or (
                    isinstance(max_pair_raw, int) and max_pair_raw < 0
                ):
                    max_pair_plots = None
                else:
                    max_pair_plots = int(max_pair_raw)

                ref_name = context.filter_list[0]
                others: list[str] = []

                from .pipeline.bridge import (
                    exposure_pairing_records_table,
                    list_exposure_image_groups,
                )

                groups = list_exposure_image_groups(context, config)
                pairing_mode = str(getattr(config, "exposure_pairing", ""))
                cfg_ref = getattr(config, "reference_filter", None) or ref_name
                if groups and cfg_ref in groups[0]:
                    ref_name = cfg_ref

                if getattr(dp, "exposure_pairing_overview", False) and groups:
                    pair_tbl = exposure_pairing_records_table(
                        groups,
                        reference_filter=ref_name,
                        pairing_mode=pairing_mode,
                    )
                    if len(pair_tbl) > 0:
                        ecsv_path = out_d / "exposure_pairing_pairs.ecsv"
                        pair_tbl.write(ecsv_path, format="ascii.ecsv", overwrite=True)
                        plots.plot_exposure_pairing_overview(
                            pair_tbl,
                            out_d,
                            ft,
                            reference_filter=ref_name,
                            pairing_mode=pairing_mode,
                        )

                if not dp.correlation_inter_filter_separation_plot:
                    return

                # Reference-image pair (series.reference_image_index per filter)
                ref_images = {}
                for f in context.filter_list:
                    ser = obs.image_series_dict.get(f)
                    if ser is None:
                        continue
                    ref_images[f] = ser.image_list[ser.reference_image_index]
                sep_ref, ref_name, others = (
                    correlate.inter_filter_correlation_separations_for_images(
                        obs,
                        context.filter_list,
                        ref_images,
                        reference_filter=ref_name,
                    )
                )
                if sep_ref.size:
                    plots.plot_inter_filter_correlation_separations(
                        sep_ref,
                        out_d,
                        ft,
                        reference_filter=ref_name,
                        other_filters=others,
                        filename_stem="inter_filter_correlation_separations_reference",
                        title_suffix=(
                            "reference images — "
                            + correlate.inter_filter_pair_title_suffix(ref_images)
                        ),
                    )
                    _write_inter_filter_geometry(
                        obs,
                        context.filter_list,
                        ref_images,
                        reference_filter=ref_name,
                        output_dir=out_d,
                        file_type=ft,
                        filename_stem="inter_filter_correlation_geometry_reference",
                        title_suffix=(
                            "reference images — "
                            + correlate.inter_filter_pair_title_suffix(ref_images)
                        ),
                        write_figure=True,
                    )

                seps_by_pair: list[np.ndarray] = []
                pair_labels: list[str] = []
                geometry_summaries: list[dict] = []
                geometry_labels: list[str] = []
                n_written = 0
                for i, group in enumerate(groups):
                    sep_i, ref_i, others_i = (
                        correlate.inter_filter_correlation_separations_for_images(
                            obs,
                            context.filter_list,
                            group,
                            reference_filter=ref_name,
                        )
                    )
                    if not sep_i.size:
                        continue
                    if not others:
                        others = others_i
                    label = correlate.inter_filter_pair_image_label(group)
                    seps_by_pair.append(sep_i)
                    pair_labels.append(f"{i:03d}")
                    write_pair = max_pair_plots is None or n_written < max_pair_plots
                    if write_pair and max_pair_plots != 0:
                        plots.plot_inter_filter_correlation_separations(
                            sep_i,
                            out_d,
                            ft,
                            reference_filter=ref_i,
                            other_filters=others_i,
                            filename_stem=(
                                f"inter_filter_correlation_separations_pair_{i:03d}_{label}"
                            ),
                            title_suffix=(
                                f"pair {i:03d} — "
                                + correlate.inter_filter_pair_title_suffix(group)
                            ),
                        )
                        n_written += 1
                    geom = _write_inter_filter_geometry(
                        obs,
                        context.filter_list,
                        group,
                        reference_filter=ref_name,
                        output_dir=out_d,
                        file_type=ft,
                        filename_stem=(
                            f"inter_filter_correlation_geometry_pair_{i:03d}_{label}"
                        ),
                        title_suffix=(
                            f"pair {i:03d} — "
                            + correlate.inter_filter_pair_title_suffix(group)
                        ),
                        write_figure=bool(write_pair and max_pair_plots != 0),
                    )
                    for g in geom:
                        geometry_summaries.append(g)
                        other = g.get("other_filter", "")
                        geometry_labels.append(
                            f"{i:03d}_{other}" if other else f"{i:03d}"
                        )

                if seps_by_pair:
                    plots.plot_inter_filter_correlation_separations_overview(
                        seps_by_pair,
                        pair_labels,
                        out_d,
                        ft,
                        reference_filter=ref_name,
                        other_filters=others,
                        pairing_mode=pairing_mode,
                    )
                    if geometry_summaries:
                        plots.plot_inter_filter_correlation_geometry_overview(
                            geometry_summaries,
                            out_d,
                            ft,
                            pair_labels=geometry_labels,
                            title_suffix=f"pairing={pairing_mode}",
                        )
                    if max_pair_plots is not None and len(seps_by_pair) > max_pair_plots:
                        _warn(
                            f"Inter-filter separation/geometry: wrote per-pair plots "
                            f"for the first {max_pair_plots} of {len(seps_by_pair)} "
                            f"pairs; see *_overview for all. "
                            f"(correlation_inter_filter_max_pair_plots={max_pair_plots})"
                        )
            except Exception as exc:
                _warn(f"Diagnostic plot (correlation_inter): {exc}")
        elif phase in ("calibration", "calibration_differential"):
            epochs = calibration_epochs or {}
            if not epochs:
                return
            try:
                first_eid = sorted(epochs.keys())[0]
                for eid, t in epochs.items():
                    sep_cal = np.array([])
                    if "match_sep_arcsec" in t.colnames:
                        sep_cal = np.asarray(t["match_sep_arcsec"], dtype=float)
                        sep_cal = sep_cal[np.isfinite(sep_cal)]
                    if dp.calibration_crossmatch_separation_histogram and sep_cal.size:
                        plots.plot_calibration_crossmatch_separations(
                            sep_cal,
                            out_d,
                            ft,
                            title=f"Catalog cross-match separations ({eid})",
                            filename_stem=(
                                "differential_catalog_crossmatch_separations"
                                if eid == first_eid
                                else f"differential_catalog_crossmatch_separations_{eid}"
                            ),
                        )
                    if (
                        dp.combined_separation_histograms
                        and eid == first_eid
                        and obs is not None
                        and sep_cal.size
                    ):
                        sep_inter = np.array([])
                        ref_name, others = "", []
                        if len(context.filter_list) >= 2:
                            sep_inter, ref_name, others = (
                                correlate.inter_filter_correlation_separations_arcsec(
                                    obs,
                                    context.filter_list,
                                    0,
                                )
                            )
                        plots.plot_combined_separation_histograms(
                            sep_inter,
                            sep_cal,
                            out_d,
                            ft,
                            reference_filter=ref_name,
                            other_filters=others,
                        )

                    if dp.photometry_mag_vs_error_scatter:
                        for f in context.filter_list:
                            mc, ec = f"mag_{f}", f"err_{f}"
                            if mc not in t.colnames or ec not in t.colnames:
                                continue
                            sub = Table()
                            sub["mags_fit"] = t[mc]
                            sub["mags_unc"] = t[ec]
                            _copy_qc_plot_columns(t, sub)
                            used_col = f"is_calibrator_{f}"
                            if used_col in t.colnames:
                                sub["is_calibrator"] = t[used_col]
                            if "x" in t.colnames:
                                sub["x_fit"] = t["x"]
                            elif "x_fit" in t.colnames:
                                sub["x_fit"] = t["x_fit"]
                            if "y" in t.colnames:
                                sub["y_fit"] = t["y"]
                            elif "y_fit" in t.colnames:
                                sub["y_fit"] = t["y_fit"]
                            plots.plot_photometry_mag_vs_error(
                                sub,
                                out_d,
                                ft,
                                band_label=f,
                                filename_stem=f"photometry_mag_vs_error_{f}_{eid}",
                            )

                    std_cols = [f"mag_std_{f}" for f in context.filter_list]
                    inst_cols = [f"mag_{f}" for f in context.filter_list]
                    if all(c in t.colnames for c in std_cols + inst_cols):
                        mask = np.ones(len(t), dtype=bool)
                        for c in std_cols:
                            v = np.asarray(t[c], dtype=float)
                            mask &= np.isfinite(v)

                        if (
                            dp.calibration_instrumental_vs_catalog
                            or dp.calibration_zeropoint_residual_histogram
                            or dp.calibration_zeropoint_residual_vs_color
                        ):
                            zp_residuals_diff: dict[str, np.ndarray] = {}
                            for f in context.filter_list:
                                m_inst = np.asarray(t[f"mag_{f}"], dtype=float)[mask]
                                m_std = np.asarray(t[f"mag_std_{f}"], dtype=float)[mask]
                                ok = np.isfinite(m_inst) & np.isfinite(m_std)
                                if not np.any(ok):
                                    continue
                                m_inst, m_std = m_inst[ok], m_std[ok]
                                if dp.calibration_instrumental_vs_catalog:
                                    plots.plot_instrumental_vs_catalog_magnitudes(
                                        m_inst,
                                        m_std,
                                        out_d,
                                        ft,
                                        band_label=f"{f}_{eid}",
                                    )
                                if dp.calibration_zeropoint_residual_histogram:
                                    zp = float(np.nanmedian(m_std - m_inst))
                                    zp_residuals_diff[f] = m_std - m_inst - zp
                            if (
                                dp.calibration_zeropoint_residual_histogram
                                and zp_residuals_diff
                            ):
                                plots.plot_zeropoint_residual_distribution(
                                    None,
                                    out_d,
                                    ft,
                                    residuals_by_band=zp_residuals_diff,
                                    filename_stem=f"zeropoint_residual_distribution_{eid}",
                                )

                        if (
                            (
                                dp.calibration_color_check_cal_stars
                                or dp.calibration_zeropoint_residual_vs_color
                            )
                            and {"B", "V"}.issubset(set(context.filter_list))
                            and "mag_std_B" in t.colnames
                            and "mag_std_V" in t.colnames
                            and "mag_B" in t.colnames
                            and "mag_V" in t.colnames
                        ):
                            mB = np.asarray(t["mag_B"], dtype=float)[mask]
                            mV = np.asarray(t["mag_V"], dtype=float)[mask]
                            sB = np.asarray(t["mag_std_B"], dtype=float)[mask]
                            sV = np.asarray(t["mag_std_V"], dtype=float)[mask]
                            ok2 = (
                                np.isfinite(mB)
                                & np.isfinite(mV)
                                & np.isfinite(sB)
                                & np.isfinite(sV)
                            )
                            if dp.calibration_color_check_cal_stars and np.any(ok2):
                                plots.plot_calibration_color_color_cal_stars(
                                    (sB - sV)[ok2],
                                    (mB - mV)[ok2],
                                    out_d,
                                    ft,
                                    filename_stem=f"calibration_color_color_cal_stars_{eid}",
                                )
                            if dp.calibration_zeropoint_residual_vs_color and np.any(ok2):
                                mBb, mVb = mB[ok2], mV[ok2]
                                sBb, sVb = sB[ok2], sV[ok2]
                                clit = sBb - sVb
                                zp_b = float(np.nanmedian(sBb - mBb))
                                zp_v = float(np.nanmedian(sVb - mVb))
                                res_b = sBb - mBb - zp_b
                                res_v = sVb - mVb - zp_v
                                ok3 = np.isfinite(clit) & np.isfinite(res_b) & np.isfinite(
                                    res_v
                                )
                                if np.any(ok3):
                                    plots.plot_zeropoint_residual_vs_color(
                                        clit[ok3],
                                        None,
                                        out_d,
                                        ft,
                                        residuals_by_band={
                                            "V": res_v[ok3],
                                            "B": res_b[ok3],
                                        },
                                        color_label=r"$(B-V)_\mathrm{std}$ [mag]",
                                        filename_stem=f"zeropoint_residual_vs_color_B_V_{eid}",
                                    )
            except Exception as exc:
                _warn(f"Diagnostic plot (calibration_differential): {exc}")

    except Exception as exc:
        _warn(f"Diagnostic plots phase {phase!r}: {exc}")
