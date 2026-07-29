"""Plot and save light curves from calibrated tables or normalized flux (fallback)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.time import Time

from .... import checks, terminal_output
from ... import calibration
from ...post_processing.light_curve import (
    is_epoch_native_photometry_table,
    prepare_plot_time_series,
)
from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext


def _any_image_series_ready(obs) -> bool:
    for f in getattr(obs, "image_series_dict", {}) or {}:
        series = obs.image_series_dict.get(f)
        if series and getattr(series, "image_list", None) and len(series.image_list) > 0:
            return True
    return False


class LightCurveStep(base.PipelineStep):
    """
    After calibration, plot JD light curves per filter.

    Primary path: epoch-native ``table_magnitudes`` with **calibrated**
    ``mag_cal_*`` / ``err_cal_*`` and ``context.calibration_epoch_meta`` for JDs.

    When only instrumental ``mag_inst_*`` exists (e.g. uncalibrated Clear-filter
    run), light curves use the same quasi-flux calibration and per-object
    normalization as ``_run_flux_fallback_for_filter`` — ``NdarrayDistribution``
    and ``ImageSeries.get_observation_time()``, not raw ``mag_inst_*``.
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

        checks.check_output_directories(
            f"{output_dir}/lightcurve",
            f"{output_dir}/tables",
        )
        if config.plot_light_curve_all_objects:
            checks.clear_directory(Path(f"{output_dir}/lightcurve/by_id"))
        if config.plot_light_curve_calibration_objects:
            checks.clear_directory(Path(f"{output_dir}/lightcurve/calibration"))

        binning = config.light_curve_binning_factor
        dist_samples = config.distribution_samples

        ids_cal = None
        if obs.calib_parameters is not None:
            ids_cal = getattr(obs.calib_parameters, "ids_calibration_objects", None)

        terminal_output.print_to_terminal("Light curves", style_name="HEADER")

        objects_of_interest = context.objects_of_interest or []

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

            if use_table:
                self._run_table_path_for_filter(
                    tbl,
                    filter_,
                    epoch_meta,
                    obs,
                    output_dir,
                    config,
                    binning,
                    ids_cal,
                    objects_of_interest,
                )
            elif use_flux:
                self._run_flux_fallback_for_filter(
                    image_series,
                    filter_,
                    output_dir,
                    config,
                    binning,
                    dist_samples,
                    ids_cal,
                    objects_of_interest,
                )
            else:
                terminal_output.print_to_terminal(
                    f"LightCurveStep: no epoch-native table columns for filter "
                    f"{filter_!r} and no usable image_series; skipping this band.",
                    style_name="WARNING",
                )

        return context

    def _run_table_path_for_filter(
        self,
        tbl: Table,
        filter_: str,
        epoch_meta: dict,
        obs,
        output_dir: str,
        config: PipelineConfig,
        binning: float | None,
        ids_cal,
        objects_of_interest: list,
    ) -> None:
        terminal_output.print_to_terminal(
            f"Light curves in filter: {filter_}",
            style_name="OKBLUE",
        )

        ids_ooi: set[int] = set()
        if config.plot_light_curve_objects_of_interest:
            for object_ in objects_of_interest:
                oid = object_.id_in_image_series.get(filter_)
                if oid is None:
                    continue
                oid = int(oid)
                ids_ooi.add(oid)
                prepare_plot_time_series(
                    tbl,
                    None,
                    filter_,
                    object_.name,
                    oid,
                    output_dir,
                    binning_factor=binning,
                    transit_time=getattr(object_, "transit_time", None),
                    period=getattr(object_, "period", None),
                    file_type_plots=config.file_type_plots,
                    epoch_meta=epoch_meta,
                    light_curve_quantity=config.light_curve_quantity,
                    light_curve_calibration_rows=config.light_curve_calibration_rows,
                )

        if config.plot_light_curve_calibration_objects and ids_cal is not None:
            arr = np.asarray(ids_cal)
            if arr.size and np.any(arr):
                for index in arr.flatten().astype(int):
                    prepare_plot_time_series(
                        tbl,
                        None,
                        filter_,
                        str(int(index)),
                        int(index),
                        output_dir,
                        binning_factor=binning,
                        file_type_plots=config.file_type_plots,
                        subdirectory="/calibration",
                        epoch_meta=epoch_meta,
                        light_curve_quantity=config.light_curve_quantity,
                        light_curve_calibration_rows=config.light_curve_calibration_rows,
                    )

        if config.plot_light_curve_all_objects:
            uids = np.unique(np.asarray(tbl["id"]).astype(int))
            cal_set: set[int] = set()
            if ids_cal is not None and np.size(ids_cal):
                cal_set = set(
                    np.asarray(ids_cal).flatten().astype(int).tolist()
                )
            for sid in uids:
                if int(sid) in ids_ooi or int(sid) in cal_set:
                    continue
                prepare_plot_time_series(
                    tbl,
                    None,
                    filter_,
                    str(int(sid)),
                    int(sid),
                    output_dir,
                    binning_factor=binning,
                    file_type_plots=config.file_type_plots,
                    subdirectory="/by_id",
                    epoch_meta=epoch_meta,
                    light_curve_quantity=config.light_curve_quantity,
                    light_curve_calibration_rows=config.light_curve_calibration_rows,
                )

    def _run_flux_fallback_for_filter(
        self,
        image_series,
        filter_: str,
        output_dir: str,
        config: PipelineConfig,
        binning: float | None,
        dist_samples: int,
        ids_cal,
        objects_of_interest: list,
    ) -> None:
        terminal_output.print_to_terminal(
            f"Light curves in filter: {filter_}",
            style_name="OKBLUE",
        )
        terminal_output.print_to_terminal(
            "No ``mag_cal_*`` light-curve path for this filter (instrumental-only "
            "``mag_inst_*`` or no epoch-native table). Using normalized flux for "
            "light curves.",
            indent=2,
            style_name="WARNING",
        )

        quasi = calibration.quasi_flux_calibration_image_series(
            image_series,
            distribution_samples=dist_samples,
        )
        plot_quantity = calibration.flux_normalization_image_series(
            image_series,
            quasi_calibrated_flux=quasi,
            distribution_samples=dist_samples,
        )

        obs_times = Time(
            image_series.get_observation_time(),
            format="jd",
        )

        ids_ooi: set[int] = set()
        if config.plot_light_curve_objects_of_interest:
            for object_ in objects_of_interest:
                oid = object_.id_in_image_series.get(filter_)
                if oid is None:
                    continue
                oid = int(oid)
                ids_ooi.add(oid)
                prepare_plot_time_series(
                    plot_quantity,
                    obs_times,
                    filter_,
                    object_.name,
                    oid,
                    output_dir,
                    binning_factor=binning,
                    transit_time=getattr(object_, "transit_time", None),
                    period=getattr(object_, "period", None),
                    file_type_plots=config.file_type_plots,
                    calibration_type="simple",
                )

        if config.plot_light_curve_calibration_objects and ids_cal is not None:
            arr = np.asarray(ids_cal)
            if arr.size and np.any(arr):
                for index in arr.flatten().astype(int):
                    prepare_plot_time_series(
                        plot_quantity,
                        obs_times,
                        filter_,
                        str(int(index)),
                        int(index),
                        output_dir,
                        binning_factor=binning,
                        file_type_plots=config.file_type_plots,
                        subdirectory="/calibration",
                        calibration_type="simple",
                    )

        if config.plot_light_curve_all_objects:
            shape_n = plot_quantity.shape[1]
            cal_set: set[int] = set()
            if ids_cal is not None and np.size(ids_cal):
                cal_set = set(
                    np.asarray(ids_cal).flatten().astype(int).tolist()
                )
            for idx in range(shape_n):
                if idx in ids_ooi or idx in cal_set:
                    continue
                prepare_plot_time_series(
                    plot_quantity,
                    obs_times,
                    filter_,
                    str(idx),
                    idx,
                    output_dir,
                    binning_factor=binning,
                    file_type_plots=config.file_type_plots,
                    subdirectory="/by_id",
                    calibration_type="simple",
                )
