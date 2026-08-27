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


def _column_float_or_none(table: Table, name: str) -> np.ndarray | None:
    if name not in table.colnames:
        return None
    col = table[name]
    return np.asarray(col.value if hasattr(col, "value") else col, dtype=float)


def _epoch_transformation(context: Any, epoch_id: str, filter_name: str):
    results = getattr(context, "calibration_results", None) or {}
    result = results.get(epoch_id)
    if result is None:
        return None
    return (getattr(result, "transformation", None) or {}).get(filter_name)


def _color_pair_for_filter(
    context: Any, config: Any, epoch_id: str, filter_name: str
) -> tuple[str, str] | None:
    tc = _epoch_transformation(context, epoch_id, filter_name)
    if tc is not None:
        ci = getattr(tc, "color_index_filters", None)
        if ci and len(ci) == 2:
            return str(ci[0]), str(ci[1])
    indices = getattr(config, "color_indices", None) or {}
    pair = indices.get(filter_name)
    if pair is not None and len(pair) == 2:
        return str(pair[0]), str(pair[1])
    return None


def _catalog_color(table: Table, pair: tuple[str, str] | None) -> np.ndarray | None:
    if pair is None:
        return None
    a = _column_float_or_none(table, f"mag_std_{pair[0]}")
    b = _column_float_or_none(table, f"mag_std_{pair[1]}")
    if a is None or b is None:
        return None
    return a - b


def _color_pairs_for_epoch(
    context: Any, config: Any, epoch_id: str, filter_list
) -> list[tuple[str, str]]:
    have = {str(f) for f in filter_list}
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(pair) -> None:
        if not pair or len(pair) != 2:
            return
        item = (str(pair[0]), str(pair[1]))
        if item[0] in have and item[1] in have and item not in seen:
            seen.add(item)
            pairs.append(item)

    result = (getattr(context, "calibration_results", None) or {}).get(epoch_id)
    if result is not None:
        for tc in (getattr(result, "transformation", None) or {}).values():
            _add(getattr(tc, "color_index_filters", None))
    for pair in (getattr(config, "color_indices", None) or {}).values():
        _add(pair)
    if not pairs and "B" in have and "V" in have:
        pairs.append(("B", "V"))
    return pairs


def _fit_residual_for_filter(
    table: Table, context: Any, config: Any, epoch_id: str, filter_name: str
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    m_inst = _column_float_or_none(table, f"mag_{filter_name}")
    m_cat = _column_float_or_none(table, f"mag_std_{filter_name}")
    if m_inst is None or m_cat is None:
        return None, None, 0.0
    tc = _epoch_transformation(context, epoch_id, filter_name)
    t_coef = float(getattr(tc, "color_term", 0.0) or 0.0) if tc is not None else 0.0
    zp = float(getattr(tc, "zero_point", float("nan"))) if tc is not None else None
    pair = _color_pair_for_filter(context, config, epoch_id, filter_name)
    color = _catalog_color(table, pair)
    residual = plots.catalog_fit_residual(
        m_inst,
        m_cat,
        color=color,
        color_term=t_coef,
        zero_point=zp,
    )
    return residual, m_inst, t_coef


def _used_mask_for_filter(table: Table, filter_name: str) -> np.ndarray | None:
    col = f"is_calibrator_{filter_name}"
    if col not in table.colnames:
        return None
    flag = np.asarray(table[col], dtype=bool)
    return flag


def _residual_vs_color_title(color_terms: list[float]) -> str:
    if any(abs(float(t)) > 1e-12 for t in color_terms):
        return r"Residuals after $T\cdot c+\mathrm{ZP}$"
    return "Residuals after pure ZP?"


def _plot_catalog_extraction_checks(
    context: Any,
    config: Any,
    table: Table,
    epoch_id: str,
    out_d: Path,
    file_type: str,
    dp: Any,
) -> None:
    filters = list(context.filter_list)
    zp_residuals: dict[str, np.ndarray] = {}
    used_by_band: dict[str, np.ndarray] = {}
    color_terms: list[float] = []
    for filt in filters:
        residual, m_inst, t_coef = _fit_residual_for_filter(
            table, context, config, epoch_id, filt
        )
        if residual is None or m_inst is None:
            continue
        m_cat = _column_float_or_none(table, f"mag_std_{filt}")
        if m_cat is None:
            continue
        used = _used_mask_for_filter(table, filt)
        color_terms.append(t_coef)
        zp_residuals[filt] = residual
        if used is not None:
            used_by_band[filt] = used
        if dp.calibration_instrumental_vs_catalog:
            plots.plot_instrumental_vs_catalog_magnitudes(
                m_inst,
                m_cat,
                out_d,
                file_type,
                band_label=f"{filt}_{epoch_id}",
                used_mask=used,
                err_obs=_column_float_or_none(table, f"err_{filt}"),
                err_cat=_column_float_or_none(table, f"err_std_{filt}"),
                residual=residual,
                residual_label=r"$m_\mathrm{cat}-m_\mathrm{inst}-T\cdot c-\mathrm{ZP}$ [mag]",
            )
    if dp.calibration_zeropoint_residual_histogram and zp_residuals:
        plots.plot_zeropoint_residual_distribution(
            None,
            out_d,
            file_type,
            residuals_by_band=zp_residuals,
            used_mask_by_band=used_by_band or None,
            filename_stem=f"zeropoint_residual_distribution_{epoch_id}",
        )

    pairs = _color_pairs_for_epoch(context, config, epoch_id, filters)
    if not pairs:
        return
    vs_color_title = _residual_vs_color_title(color_terms)
    for f1, f2 in pairs:
        color = _catalog_color(table, (f1, f2))
        if color is None:
            continue
        color_name = f"{f1}-{f2}"
        if dp.calibration_zeropoint_residual_vs_color:
            by_band = {
                f: zp_residuals[f]
                for f in (f1, f2)
                if f in zp_residuals
            }
            if by_band:
                used_masks = {
                    f: used_by_band[f] for f in by_band if f in used_by_band
                }
                plots.plot_zeropoint_residual_vs_color(
                    color,
                    None,
                    out_d,
                    file_type,
                    residuals_by_band=by_band,
                    used_mask_by_band=used_masks or None,
                    color_label=color_name,
                    title=vs_color_title,
                    filename_stem=f"zeropoint_residual_vs_color_{f1}_{f2}_{epoch_id}",
                )
        if not dp.calibration_color_check_cal_stars:
            continue
        m1 = _column_float_or_none(table, f"mag_{f1}")
        m2 = _column_float_or_none(table, f"mag_{f2}")
        s1 = _column_float_or_none(table, f"mag_std_{f1}")
        s2 = _column_float_or_none(table, f"mag_std_{f2}")
        if m1 is None or m2 is None or s1 is None or s2 is None:
            continue
        tc1 = _epoch_transformation(context, epoch_id, f1)
        tc2 = _epoch_transformation(context, epoch_id, f2)
        t1 = float(getattr(tc1, "color_term", 0.0) or 0.0) if tc1 is not None else 0.0
        t2 = float(getattr(tc2, "color_term", 0.0) or 0.0) if tc2 is not None else 0.0
        zp1 = float(getattr(tc1, "zero_point", float("nan"))) if tc1 is not None else None
        zp2 = float(getattr(tc2, "zero_point", float("nan"))) if tc2 is not None else None
        if zp1 is None or not np.isfinite(zp1):
            zp1 = float(np.nanmedian(s1 - m1))
        if zp2 is None or not np.isfinite(zp2):
            zp2 = float(np.nanmedian(s2 - m2))
        m1_cal = m1 + t1 * color + zp1
        m2_cal = m2 + t2 * color + zp2
        color_obs = m1_cal - m2_cal
        used1 = _used_mask_for_filter(table, f1)
        used2 = _used_mask_for_filter(table, f2)
        used = None
        if used1 is not None and used2 is not None:
            used = used1 & used2
        elif used1 is not None:
            used = used1
        elif used2 is not None:
            used = used2
        stem = f"calibration_color_color_cal_stars_{epoch_id}"
        if len(pairs) > 1:
            stem = f"calibration_color_color_cal_stars_{f1}_{f2}_{epoch_id}"
        plots.plot_calibration_color_color_cal_stars(
            color,
            color_obs,
            out_d,
            file_type,
            filename_stem=stem,
            used_mask=used,
            color_label=color_name,
        )



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
    getter = getattr(image, "get_data", None)
    if getter is None:
        return None
    try:
        arr = np.asarray(getter(), dtype=float)
    except Exception:
        return None
    while arr.ndim > 2:
        arr = arr[0]
    return arr if arr.ndim == 2 else None


def _reference_image(context: Any, config: Any):
    filters = list(getattr(context, "filter_list", []) or [])
    ref = getattr(config, "reference_filter", None) or (filters[0] if filters else None)
    if not ref:
        return None
    series = (getattr(context, "image_series_dict", None) or {}).get(ref)
    images = getattr(series, "image_list", None) if series is not None else None
    if not images:
        return None
    idx = int(getattr(config, "reference_image_index", 0) or 0)
    if idx < 0 or idx >= len(images):
        idx = 0
    return images[idx]


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
                        diag_stem = (
                            "calibration_crossmatch_diagnostics"
                            if eid == first_eid
                            else f"calibration_crossmatch_diagnostics_{eid}"
                        )
                        plots.plot_calibration_crossmatch_diagnostics(
                            t,
                            out_d,
                            ft,
                            filename_stem=diag_stem,
                            title=f"Catalog cross-match diagnostics ({eid})",
                        )
                        img = _reference_image(context, config)
                        vec = plots.catalog_match_pixel_residuals(
                            t, getattr(img, "wcs", None) if img is not None else None
                        )
                        if vec is not None:
                            x, y, dx, dy, sep_vec = vec
                            geom_stem = (
                                "calibration_crossmatch_geometry"
                                if eid == first_eid
                                else f"calibration_crossmatch_geometry_{eid}"
                            )
                            plots.plot_inter_filter_correlation_geometry(
                                x,
                                y,
                                dx,
                                dy,
                                out_d,
                                ft,
                                image_data=_image_array_for_plot(img),
                                sep_arcsec=sep_vec,
                                filename_stem=geom_stem,
                                title="Catalog cross-match residual geometry"
                                + (f" ({eid})" if eid else ""),
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

                    if (
                        dp.calibration_instrumental_vs_catalog
                        or dp.calibration_zeropoint_residual_histogram
                        or dp.calibration_zeropoint_residual_vs_color
                        or dp.calibration_color_check_cal_stars
                    ):
                        _plot_catalog_extraction_checks(
                            context, config, t, eid, out_d, ft, dp
                        )
            except Exception as exc:
                _warn(f"Diagnostic plot (calibration_differential): {exc}")

    except Exception as exc:
        _warn(f"Diagnostic plots phase {phase!r}: {exc}")
