"""
Optional pipeline hooks for :class:`~ost_photometry.analyze.pipeline.config.DiagnosticPlots`.

Figures go to ``<output_dir>/diagnostics/`` using ``PipelineConfig.file_type_plots``.
Failures are logged and do not abort the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from . import correlate, plots
from .. import checks, terminal_output


def diagnostics_subdirectory(output_dir: str | Path) -> Path:
    base = Path(output_dir)
    d = base / "diagnostics"
    checks.check_output_directories(base, d)
    return d


def _photometry_col_float(column) -> np.ndarray:
    if hasattr(column, "value"):
        return np.asarray(column.value, dtype=float)
    return np.asarray(column, dtype=float)


def run_diagnostic_plots_phase(
    context: Any,
    config: Any,
    phase: str,
    *,
    differential_epochs: Mapping[str, Table] | None = None,
) -> None:
    """Dispatch diagnostic figures by pipeline phase name."""
    dp = config.diagnostic_plots
    out_root = getattr(context, "output_dir", None)
    if out_root is None:
        return

    out_d = diagnostics_subdirectory(out_root)
    ft = getattr(config, "file_type_plots", "pdf")
    obs = getattr(context, "_observation", None)

    def _warn(msg: str) -> None:
        terminal_output.print_to_terminal(msg, style_name="WARNING")

    try:
        if phase == "extraction":
            if not (
                dp.photometry_mag_vs_error_scatter or dp.photometry_radial_growth_curve
            ):
                return
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
            if not dp.correlation_inter_filter_separation_plot:
                return
            if obs is None or len(context.filter_list) < 2:
                return
            try:
                sep, ref_name, others = correlate.inter_filter_correlation_separations_arcsec(
                    obs,
                    context.filter_list,
                    0,
                )
                if sep.size:
                    plots.plot_inter_filter_correlation_separations(
                        sep,
                        out_d,
                        ft,
                        reference_filter=ref_name,
                        other_filters=others,
                    )
            except Exception as exc:
                _warn(f"Diagnostic plot (correlation_inter): {exc}")

        elif phase == "calibration_data":
            if not (
                dp.calibration_crossmatch_separation_histogram
                or dp.combined_separation_histograms
            ):
                return
            if obs is None or obs.calib_parameters is None:
                return
            cp = obs.calib_parameters
            idx = cp.ids_calibration_objects
            if idx is None or len(idx) == 0:
                return
            ref = context.filter_list[0]
            if ref not in obs.image_series_dict:
                return
            try:
                series = obs.image_series_dict[ref]
                img = series.image_list[series.reference_image_index]
                phot = img.photometry
                if phot is None:
                    return
                idx = np.asarray(idx, dtype=int)
                obs_c = SkyCoord.from_pixel(
                    phot["x_fit"][idx],
                    phot["y_fit"][idx],
                    series.wcs,
                )
                cn = cp.column_names
                ra_k, dec_k = cn["ra"], cn["dec"]
                cat_c = SkyCoord(
                    cp.calib_tbl[ra_k].data,
                    cp.calib_tbl[dec_k].data,
                    unit=(cp.ra_unit, cp.dec_unit),
                    frame="icrs",
                )
                n = min(len(obs_c), len(cat_c))
                sep = np.asarray(obs_c[:n].separation(cat_c[:n]).arcsec, dtype=float)
                if dp.calibration_crossmatch_separation_histogram:
                    plots.plot_calibration_crossmatch_separations(sep, out_d, ft)
                if dp.combined_separation_histograms:
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
                        sep,
                        out_d,
                        ft,
                        reference_filter=ref_name,
                        other_filters=others,
                    )
            except Exception as exc:
                _warn(f"Diagnostic plot (calibration_data): {exc}")

        elif phase == "calibration_apply":
            if not (
                dp.calibration_instrumental_vs_catalog
                or dp.calibration_zeropoint_residual_histogram
                or dp.calibration_zeropoint_residual_vs_color
                or dp.calibration_color_check_cal_stars
            ):
                return
            if obs is None or obs.calib_parameters is None:
                return
            cp = obs.calib_parameters
            idx = cp.ids_calibration_objects
            if idx is None or len(idx) == 0:
                return
            idx = np.asarray(idx, dtype=int)
            cn = cp.column_names
            try:
                zp_residuals_by_band: dict[str, np.ndarray] = {}
                for f in context.filter_list:
                    mk = f"mag{f}"
                    if mk not in cn or cn[mk] not in cp.calib_tbl.colnames:
                        continue
                    if f not in obs.image_series_dict:
                        continue
                    series = obs.image_series_dict[f]
                    img = series.image_list[series.reference_image_index]
                    phot = img.photometry
                    if phot is None or "mags_fit" not in phot.colnames:
                        continue
                    mi_all = _photometry_col_float(phot["mags_fit"])
                    m_inst = mi_all[idx]
                    m_cat = np.asarray(cp.calib_tbl[cn[mk]], dtype=float)
                    k = min(m_inst.size, m_cat.size)
                    if k == 0:
                        continue
                    m_inst = m_inst[:k]
                    m_cat = m_cat[:k]
                    ok = np.isfinite(m_inst) & np.isfinite(m_cat)
                    if not np.any(ok):
                        continue
                    m_inst = m_inst[ok]
                    m_cat = m_cat[ok]
                    if dp.calibration_instrumental_vs_catalog:
                        plots.plot_instrumental_vs_catalog_magnitudes(
                            m_inst,
                            m_cat,
                            out_d,
                            ft,
                            band_label=f,
                        )
                    if dp.calibration_zeropoint_residual_histogram:
                        zp = float(np.nanmedian(m_cat - m_inst))
                        zp_residuals_by_band[f] = m_cat - m_inst - zp
                if dp.calibration_zeropoint_residual_histogram and zp_residuals_by_band:
                    plots.plot_zeropoint_residual_distribution(
                        None,
                        out_d,
                        ft,
                        residuals_by_band=zp_residuals_by_band,
                    )

                if (
                    dp.calibration_color_check_cal_stars
                    or dp.calibration_zeropoint_residual_vs_color
                ) and {"B", "V"}.issubset(set(context.filter_list)):
                    fB, fV = "B", "V"
                    need = [fB, fV]
                    if not all(f in obs.image_series_dict for f in need):
                        pass
                    elif not all(f"mag{f}" in cn for f in need):
                        pass
                    elif not all(cn[f"mag{f}"] in cp.calib_tbl.colnames for f in need):
                        pass
                    else:
                        miB = _photometry_col_float(
                            obs.image_series_dict[fB]
                            .image_list[
                                obs.image_series_dict[fB].reference_image_index
                            ]
                            .photometry["mags_fit"]
                        )[idx]
                        miV = _photometry_col_float(
                            obs.image_series_dict[fV]
                            .image_list[
                                obs.image_series_dict[fV].reference_image_index
                            ]
                            .photometry["mags_fit"]
                        )[idx]
                        cB = np.asarray(cp.calib_tbl[cn["magB"]], dtype=float)
                        cV = np.asarray(cp.calib_tbl[cn["magV"]], dtype=float)
                        kk = min(miB.size, miV.size, cB.size, cV.size)
                        miB, miV = miB[:kk], miV[:kk]
                        cB, cV = cB[:kk], cV[:kk]
                        cl = cB - cV
                        co = miB - miV
                        okc = np.isfinite(cl) & np.isfinite(co)
                        if dp.calibration_color_check_cal_stars and np.any(okc):
                            plots.plot_calibration_color_color_cal_stars(
                                cl[okc], co[okc], out_d, ft
                            )

                        if dp.calibration_zeropoint_residual_vs_color:
                            zp_b = float(np.nanmedian(cB - miB))
                            zp_v = float(np.nanmedian(cV - miV))
                            res_b = cB - miB - zp_b
                            res_v = cV - miV - zp_v
                            okr = (
                                np.isfinite(cl)
                                & np.isfinite(res_b)
                                & np.isfinite(res_v)
                            )
                            if np.any(okr):
                                plots.plot_zeropoint_residual_vs_color(
                                    cl[okr],
                                    None,
                                    out_d,
                                    ft,
                                    residuals_by_band={
                                        "V": res_v[okr],
                                        "B": res_b[okr],
                                    },
                                    color_label=r"$(B-V)_\mathrm{lit}$ [mag]",
                                    filename_stem="zeropoint_residual_vs_color_B_V",
                                )
            except Exception as exc:
                _warn(f"Diagnostic plot (calibration_apply): {exc}")

        elif phase == "calibration_differential":
            epochs = differential_epochs or {}
            if not epochs:
                return
            eid = sorted(epochs.keys())[0]
            t = epochs[eid]
            try:
                if dp.calibration_crossmatch_separation_histogram and (
                    "match_sep_arcsec" in t.colnames
                ):
                    sep = np.asarray(t["match_sep_arcsec"], dtype=float)
                    sep = sep[np.isfinite(sep)]
                    if sep.size:
                        plots.plot_calibration_crossmatch_separations(
                            sep,
                            out_d,
                            ft,
                            title=f"Catalog cross-match separations ({eid})",
                            filename_stem="differential_catalog_crossmatch_separations",
                        )

                if dp.photometry_mag_vs_error_scatter:
                    for f in context.filter_list:
                        mc, ec = f"mag_{f}", f"err_{f}"
                        if mc not in t.colnames or ec not in t.colnames:
                            continue
                        sub = Table()
                        sub["mags_fit"] = t[mc]
                        sub["mags_unc"] = t[ec]
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
