"""Build ``light_curves.ecsv`` and plot views (OOI, QC, overview)."""

from __future__ import annotations

import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time

from .... import terminal_output
from ....output_layout import results_dir, tables_dir
from ... import calibration
from ...ooi_ids import ooi_photometry_id
from ...post_processing.light_curve import (
    CALIBRATOR_STATS_FILENAME,
    build_light_curves_table,
    build_light_curves_table_from_flux,
    calibrator_variability_stats,
    ids_excluding,
    is_epoch_native_photometry_table,
    plot_from_light_curves_table,
    top_variable_calibrator_ids,
    write_epoch_meta_json,
    write_light_curves_table,
)
from ...post_processing.magnitude_systems import table_magnitude_system
from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext


def _any_image_series_ready(obs) -> bool:
    for f in getattr(obs, "image_series_dict", {}) or {}:
        series = obs.image_series_dict.get(f)
        if series and getattr(series, "image_list", None) and len(series.image_list) > 0:
            return True
    return False


def _calibration_object_ids_from_table(
    tbl: Table | None,
    filter_: str | None = None,
) -> list[int] | None:
    """Object ``id``s with finite catalog ``mag_std_*`` (epoch-native tables)."""
    if tbl is None or len(tbl) == 0 or "id" not in tbl.colnames:
        return None
    mag_cols: list[str] = []
    if filter_ is not None:
        col = f"mag_std_{filter_}"
        if col in tbl.colnames:
            mag_cols = [col]
    if not mag_cols:
        mag_cols = [c for c in tbl.colnames if c.startswith("mag_std_")]
    if not mag_cols:
        return None
    mask = np.zeros(len(tbl), dtype=bool)
    for col in mag_cols:
        arr = np.asarray(tbl[col], dtype=float)
        mask |= np.isfinite(arr)
    if not np.any(mask):
        return None
    ids = np.asarray(tbl["id"][mask], dtype=int)
    return sorted({int(i) for i in ids})


def _ooi_id_name_pairs(objects_of_interest: list, filter_: str) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    seen: set[int] = set()
    for obj in objects_of_interest:
        oid = ooi_photometry_id(obj, filter_=filter_)
        if oid is None:
            continue
        oid_i = int(oid)
        if oid_i in seen:
            continue
        seen.add(oid_i)
        pairs.append((oid_i, str(getattr(obj, "name", oid_i))))
    return pairs


def _parse_period(raw) -> float | None:
    if raw is None or raw == "?":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0.0 else None


class LightCurveStep(base.PipelineStep):
    """
    After calibration, write ``tables/light_curves.ecsv`` and plot views.

    Primary path: epoch-native ``table_magnitudes`` with ``mag_cal_*`` /
    ``err_cal_*``. Flux fallback fills the same long table from normalized
    ``ImageSeries`` flux when no epoch-native mag/flux columns exist.
    """

    name = "light_curve"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_light_curve:
            return True
        if config.skip_calibration:
            return True
        obs = context.require_observation()
        if obs is None:
            return True

        tbl = obs.table_magnitudes
        epoch_meta = context.calibration_epoch_meta or {}
        has_tbl = isinstance(tbl, Table) and len(tbl) > 0
        jd_ok = False
        if has_tbl:
            jd_ok = (
                "observation_jd" in tbl.colnames
                or "jd" in tbl.colnames
                or bool(epoch_meta)
            )

        for filter_ in context.filter_list:
            if jd_ok and is_epoch_native_photometry_table(
                tbl,
                filter_,
                quantity=config.light_curve_quantity,
            ):
                return False

        if _any_image_series_ready(obs):
            return False

        return True

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        obs = context.require_observation()
        output_dir = context.output_dir
        assert obs is not None

        tbl = obs.table_magnitudes
        if tbl is not None and not isinstance(tbl, Table):
            terminal_output.print_to_terminal(
                "LightCurveStep: table_magnitudes is not an astropy Table; "
                "using flux fallback only where possible.",
                style_name="WARNING",
            )
            tbl = None

        epoch_meta = context.calibration_epoch_meta or {}
        has_tbl = isinstance(tbl, Table) and len(tbl) > 0
        jd_ok = False
        if has_tbl:
            jd_ok = (
                "observation_jd" in tbl.colnames
                or "jd" in tbl.colnames
                or bool(epoch_meta)
            )

        tables_dir(output_dir)
        results_dir(output_dir, "lightcurves")
        write_epoch_meta_json(output_dir, epoch_meta)

        terminal_output.print_to_terminal("Light curves", style_name="HEADER")

        objects_of_interest = context.objects_of_interest or []
        object_names: dict[int, str] = {}
        for obj in objects_of_interest:
            for filt in context.filter_list:
                oid = ooi_photometry_id(obj, filter_=filt)
                if oid is not None:
                    object_names[int(oid)] = str(getattr(obj, "name", oid))
        ooi_ids_all = set(object_names.keys())

        cal_ids_all: set[int] = set()
        if has_tbl:
            raw_cal = _calibration_object_ids_from_table(tbl)
            if raw_cal:
                cal_ids_all.update(int(i) for i in raw_cal)
        cal_ids_all = ids_excluding(cal_ids_all, ooi_ids_all)

        parts: list[Table] = []
        for filter_ in context.filter_list:
            use_table = (
                has_tbl
                and jd_ok
                and is_epoch_native_photometry_table(
                    tbl,
                    filter_,
                    quantity=config.light_curve_quantity,
                )
            )
            image_series = obs.image_series_dict.get(filter_)
            use_flux = bool(
                image_series
                and getattr(image_series, "image_list", None)
                and len(image_series.image_list) > 0
            )
            ids_cal = (
                _calibration_object_ids_from_table(tbl, filter_) if has_tbl else None
            )
            cal_set = ids_excluding(
                set(int(i) for i in ids_cal) if ids_cal else set(),
                ooi_ids_all,
            )
            cal_ids_all.update(cal_set)

            if use_table:
                terminal_output.print_to_terminal(
                    f"Light curves in filter: {filter_}",
                    style_name="OKBLUE",
                )
                part = build_light_curves_table(
                    tbl,
                    [filter_],
                    epoch_meta=epoch_meta,
                    quantity=config.light_curve_quantity,
                    calibration_rows=config.light_curve_calibration_rows,
                    object_names=object_names,
                    calibrator_ids=cal_set,
                    outlier_sigma=config.light_curve_outlier_sigma,
                    observatory_location=config.observatory_location,
                    color=None,
                )
                if len(part) > 0:
                    parts.append(part)
            elif use_flux:
                terminal_output.print_to_terminal(
                    f"Light curves in filter: {filter_}",
                    style_name="OKBLUE",
                )
                terminal_output.print_to_terminal(
                    "No ``mag_cal_*`` light-curve path for this filter. "
                    "Using normalized flux for the light-curve table.",
                    indent=2,
                    style_name="WARNING",
                )
                part = self._flux_fallback_table(
                    image_series,
                    filter_,
                    config,
                    object_names,
                    cal_set,
                    tbl if has_tbl else None,
                )
                if part is not None and len(part) > 0:
                    parts.append(part)
            else:
                terminal_output.print_to_terminal(
                    f"LightCurveStep: no epoch-native table columns for filter "
                    f"{filter_!r} and no usable image_series; skipping this band.",
                    style_name="WARNING",
                )

        if not parts:
            terminal_output.print_to_terminal(
                "LightCurveStep: no light-curve rows; skipping plots.",
                style_name="WARNING",
            )
            return context

        lc = vstack(parts, metadata_conflicts="silent")
        if config.light_curve_color:
            from ...post_processing.light_curve import add_color_index_rows

            lc = add_color_index_rows(lc, config.light_curve_color)
        path = write_light_curves_table(lc, output_dir)
        terminal_output.print_to_terminal(
            f"Wrote {path.name} ({len(lc)} rows)",
            indent=1,
            style_name="INFO",
        )

        self._plot_views(
            lc,
            context,
            config,
            output_dir,
            objects_of_interest,
            cal_ids_all,
        )
        return context

    def _flux_fallback_table(
        self,
        image_series,
        filter_: str,
        config: PipelineConfig,
        object_names: dict[int, str],
        cal_set: set[int],
        phot: Table | None,
    ) -> Table | None:
        dist_samples = config.distribution_samples
        quasi = calibration.quasi_flux_calibration_image_series(
            image_series,
            distribution_samples=dist_samples,
        )
        plot_quantity = calibration.flux_normalization_image_series(
            image_series,
            quasi_calibrated_flux=quasi,
            distribution_samples=dist_samples,
        )
        obs_times = Time(image_series.get_observation_time(), format="jd")
        airmasses = []
        for im in image_series.image_list:
            am = getattr(im, "air_mass", None)
            airmasses.append(float(am) if am is not None else np.nan)
        n_obj = int(plot_quantity.pdf_median().shape[1])
        source_ids = np.arange(n_obj, dtype=np.int64)
        ra = dec = None
        if phot is not None and "id" in phot.colnames:
            if "ra" in phot.colnames and "dec" in phot.colnames:
                ra_map: dict[int, float] = {}
                dec_map: dict[int, float] = {}
                ids = np.asarray(phot["id"]).astype(int)
                for sid in np.unique(ids):
                    m = ids == int(sid)
                    ra_map[int(sid)] = float(
                        np.nanmedian(np.asarray(phot["ra"][m], dtype=float))
                    )
                    dec_map[int(sid)] = float(
                        np.nanmedian(np.asarray(phot["dec"][m], dtype=float))
                    )
                ra = np.array([ra_map.get(int(i), np.nan) for i in source_ids])
                dec = np.array([dec_map.get(int(i), np.nan) for i in source_ids])
        return build_light_curves_table_from_flux(
            plot_quantity,
            obs_times,
            filter_,
            source_ids=source_ids,
            object_names=object_names,
            calibrator_ids=cal_set,
            airmasses=np.asarray(airmasses, dtype=float),
            ra=ra,
            dec=dec,
            outlier_sigma=config.light_curve_outlier_sigma,
            observatory_location=config.observatory_location,
        )

    def _plot_views(
        self,
        lc: Table,
        context: AnalysisContext,
        config: PipelineConfig,
        output_dir: str,
        objects_of_interest: list,
        cal_ids_all: set[int],
        ) -> None:
        from ...plots import lightcurves as lc_plots

        mag_sys = table_magnitude_system(lc)
        file_type = config.file_type_plots
        filters = [
            f
            for f in context.filter_list
            if f in set(np.asarray(lc["filter"]).astype(str))
        ]
        color = config.light_curve_color
        if color and color in set(np.asarray(lc["filter"]).astype(str)):
            filters = list(filters) + [color]

        rng = np.random.default_rng(0)
        all_ids = {int(i) for i in np.unique(np.asarray(lc["id"]).astype(int))}
        stats_parts: list[Table] = []

        for filter_ in filters:
            ooi_pairs = _ooi_id_name_pairs(objects_of_interest, filter_)
            if filter_ == color:
                ooi_pairs = _ooi_id_name_pairs(
                    objects_of_interest, context.filter_list[0]
                )

            if config.plot_light_curve_objects_of_interest:
                for oid, name in ooi_pairs:
                    obj = next(
                        (
                            o
                            for o in objects_of_interest
                            if int(ooi_photometry_id(o, filter_=filter_) or -1) == oid
                            or str(getattr(o, "name", "")) == name
                        ),
                        None,
                    )
                    tt = getattr(obj, "transit_time", None) if obj is not None else None
                    per = _parse_period(getattr(obj, "period", None) if obj else None)
                    plot_from_light_curves_table(
                        lc,
                        oid,
                        filter_,
                        output_dir,
                        name_object=name,
                        file_type=file_type,
                        transit_time=tt,
                        period=per,
                        binning_factor=config.light_curve_binning_factor,
                        time_scale=config.light_curve_time_scale,
                        phase_cycles=config.light_curve_phase_cycles,
                        show_airmass=config.light_curve_show_airmass,
                        magnitude_system=mag_sys,
                    )

            if config.plot_light_curve_calibration_objects and cal_ids_all:
                ooi_set = {oid for oid, _n in ooi_pairs}
                cal_for_qc = ids_excluding(cal_ids_all, ooi_set)
                stats = calibrator_variability_stats(lc, cal_for_qc, filter_)
                if len(stats) > 0:
                    stats_parts.append(stats)
                top = top_variable_calibrator_ids(
                    stats,
                    n=config.light_curve_calibrator_qc_n,
                    exclude=ooi_set,
                )
                lc_plots.plot_check_star_qc(
                    lc,
                    output_dir,
                    filter_=filter_,
                    ooi_ids=ooi_pairs,
                    calibrator_ids=top,
                    file_type=file_type,
                    time_scale=config.light_curve_time_scale,
                    show_airmass=config.light_curve_show_airmass,
                    magnitude_system=mag_sys,
                )
                lc_plots.plot_calibrator_variability(
                    lc,
                    stats,
                    output_dir,
                    filter_=filter_,
                    top_ids=top,
                    file_type=file_type,
                    time_scale=config.light_curve_time_scale,
                    magnitude_system=mag_sys,
                )

            if config.plot_light_curve_all_objects:
                ooi_set = {oid for oid, _n in ooi_pairs}
                pool = [i for i in sorted(all_ids) if i not in ooi_set]
                n_extra = min(int(config.light_curve_overview_n), len(pool))
                extra = (
                    rng.choice(pool, size=n_extra, replace=False).tolist()
                    if n_extra
                    else []
                )
                extra = [int(i) for i in extra]
                lc_plots.plot_light_curve_overview(
                    lc,
                    output_dir,
                    filter_=filter_,
                    ooi_ids=ooi_pairs,
                    extra_ids=extra,
                    file_type=file_type,
                    time_scale=config.light_curve_time_scale,
                    magnitude_system=mag_sys,
                )

        if stats_parts:
            vstack(stats_parts, metadata_conflicts="silent").write(
                str(tables_dir(output_dir) / CALIBRATOR_STATS_FILENAME),
                format="ascii.ecsv",
                overwrite=True,
            )


__all__ = ["LightCurveStep"]
