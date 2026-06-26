"""
Differential calibration step: PhotometryCalibrator + shared calibration_sources fetch.

Uses the same ``PipelineConfig.calibration_source`` / ``vizier_dict`` / ``path_calibration_file``
as the legacy calibration_data step; radius and mag range from ``calibration_catalog_*``.
"""

import warnings
from pathlib import Path

import numpy as np

from .... import checks, terminal_output
from ... import utilities
from ...post_processing.adapters import ensure_epoch_native_photometry_table
from ...post_processing.io import write_epoch_native_magnitudes
from ...post_processing.light_curve import attach_observation_jd_column
from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ..bridge import (
    instrumental_epoch_native_from_calibration_epochs,
    observation_to_calibration_epochs,
)
from ...extinction import CoefficientMode, ExtinctionOrder
from ...differential_photometry import PhotometryCalibrator
from ...warnings_types import OstPhotometryAnalyzeWarning


def _calibration_summary_jd_by_epoch_id(
    context: AnalysisContext, config: PipelineConfig
) -> dict[str, float]:
    """Map epoch_id -> JD for summary plot (reference filter, else first available)."""
    meta = context.calibration_epoch_meta or {}
    fl = context.filter_list
    if not fl:
        return {}
    ref = config.differential_reference_filter or fl[0]
    out: dict[str, float] = {}
    for eid, m in meta.items():
        jdf = m.get("jd_by_filter") or m.get("filter_jds") or {}
        v = jdf.get(ref)
        if v is None:
            for f in fl:
                if f in jdf and jdf[f] is not None:
                    v = jdf[f]
                    break
        if v is not None:
            vf = float(v)
            if np.isfinite(vf):
                out[str(eid)] = vf
    return out


def _attach_jd_from_epoch_meta(
    tbl,
    context: AnalysisContext,
    config: PipelineConfig,
    filter_list: list,
):
    """Match legacy ``CalibrationApplyStep``: add ``observation_jd`` for standalone ECSV."""
    if len(tbl) == 0 or not filter_list or not context.calibration_epoch_meta:
        return tbl
    ref = config.differential_reference_filter or filter_list[0]
    return attach_observation_jd_column(
        tbl,
        context.calibration_epoch_meta,
        ref,
    )


def _log_calibration_skips(skipped: list) -> None:
    for entry in skipped:
        reason = entry.get("reason", "?")
        if reason == "index_unequal_lengths":
            terminal_output.print_to_terminal(
                entry.get("message", str(entry)),
                style_name="WARNING",
            )
        elif reason in ("jd_no_partner", "jd_exceeds_tolerance"):
            terminal_output.print_to_terminal(
                f"Skipped calibration epoch: {reason} — ref_filter={entry.get('reference_filter')!r} "
                f"image_id={entry.get('reference_exposure_image_id')} jd={entry.get('reference_jd')} "
                f"failed_filter={entry.get('failed_filter')!r} "
                f"best_delta_jd={entry.get('best_delta_jd')} "
                f"tolerance={entry.get('jd_tolerance')}",
                style_name="WARNING",
            )
        else:
            terminal_output.print_to_terminal(
                f"Calibration epoch pairing note: {entry}",
                style_name="INFO",
            )


class DifferentialCalibrationStep(base.PipelineStep):
    """
    Differential photometry calibration using a standard calibration catalog
    (``config.calibration_source``, same sources as legacy) and PhotometryCalibrator.

    Replaces CalibrationDataStep + CalibrationApplyStep when
    config.calibration_module == "differential".
    """

    name = "calibration_differential"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        if config.skip_calibration:
            return True
        if config.calibration_module != "differential":
            return True
        return False

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        from astropy.coordinates import SkyCoord
        from astropy.time import Time

        obs = context.require_observation()
        if obs is None:
            raise RuntimeError(
                "DifferentialCalibrationStep requires context._observation"
            )

        terminal_output.print_to_terminal(
            "Differential calibration (PhotometryCalibrator)",
            style_name="HEADER",
        )

        observation_to_calibration_epochs(context, config)
        if context.calibration_epochs_skipped:
            _log_calibration_skips(context.calibration_epochs_skipped)

        epochs = context.calibration_epochs
        if not epochs:
            raise RuntimeError(
                "No calibration epochs from observation_to_calibration_epochs. "
                "Ensure extraction and correlation have run; check pairing (JD tolerance / image counts)."
            )

        # Coefficient mode
        mode_map = {
            "per_image": CoefficientMode.PER_IMAGE,
            "per_night": CoefficientMode.PER_NIGHT,
            "fixed": CoefficientMode.FIXED,
            "ensemble": CoefficientMode.ENSEMBLE,
        }
        coeff_mode = mode_map.get(
            config.differential_coefficient_mode.lower(),
            CoefficientMode.PER_NIGHT,
        )

        # Extinction order
        ext_map = {
            "none": ExtinctionOrder.NONE,
            "first": ExtinctionOrder.FIRST,
            "second": ExtinctionOrder.SECOND,
        }
        ext_order = ext_map.get(
            config.differential_extinction_order.lower(),
            ExtinctionOrder.FIRST,
        )

        color_indices = getattr(config, "differential_color_indices", None)
        calibrator = PhotometryCalibrator(
            mode=coeff_mode,
            extinction_order=ext_order,
            observatory_location=config.observatory_location,
            color_indices=color_indices,
        )

        # Field center from first epoch
        first_tbl = next(iter(epochs.values()))
        ra_mean = np.mean(first_tbl["ra"])
        dec_mean = np.mean(first_tbl["dec"])
        field_center = SkyCoord(ra_mean, dec_mean, unit="deg")

        calibrator.setup_calibration_source(
            field_center,
            context.filter_list,
            calibration_source=config.calibration_source,
            radius_arcmin=config.calibration_catalog_radius_arcmin,
            calibration_catalog_mag_range=config.calibration_catalog_mag_range,
            vizier_dict=config.vizier_dict,
            path_calibration_file=config.path_calibration_file,
        )

        for epoch_id, tbl in epochs.items():
            meta = context.calibration_epoch_meta.get(epoch_id, {})
            filter_obstimes = {}
            jd_map = meta.get("jd_by_filter") or meta.get("filter_jds") or {}
            for f, jd in jd_map.items():
                if jd is not None:
                    filter_obstimes[f] = Time(jd, format="jd")
            calibrator.add_epoch(
                epoch_id,
                tbl,
                filter_obstimes=filter_obstimes if filter_obstimes else None,
                ra_col="ra",
                dec_col="dec",
            )

        # Optionally fit extinction from APASS comparison stars
        if getattr(config, "differential_fit_extinction_from_data", False):
            fitted = calibrator.fit_extinction_from_epochs(
                output_dir=context.output_dir,
                file_type=getattr(config, "file_type_plots", "pdf"),
            )
            if fitted:
                terminal_output.print_to_terminal(
                    f"Fitted extinction from data: {list(fitted.keys())}",
                    style_name="INFO",
                )
            else:
                warnings.warn(
                    "differential_fit_extinction_from_data is True, but "
                    "fit_extinction_from_epochs returned no coefficients "
                    "(need >=3 epochs with valid mag_std_* and airmass spread). "
                    "Using preset/default extinction from ExtinctionCorrector.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=1,
                )

        # Fit T/ZP per mode, then get_calibrated_photometry applies them
        jd_map = _calibration_summary_jd_by_epoch_id(context, config)
        calibrator.fit_transformation_parameters(
            filters=context.filter_list,
            determine_color_terms=True,
            min_comparisons=5,
            sigma_clip=config.differential_fit_sigma_clip,
            output_dir=context.output_dir,
            file_type=getattr(config, "file_type_plots", "pdf"),
            per_image_rolling_median_color_term=config.differential_per_image_rolling_median_color_term,
            per_image_rolling_median_zero_point=config.differential_per_image_rolling_median_zero_point,
            per_image_rolling_mean_color_term=config.differential_per_image_rolling_mean_color_term,
            per_image_rolling_mean_zero_point=config.differential_per_image_rolling_mean_zero_point,
            per_image_rolling_window=config.differential_per_image_rolling_window,
            calibration_summary_x_jd=jd_map if jd_map else None,
            calibration_summary_use_jd_x=config.differential_calibration_summary_use_jd_x,
        )

        # Get calibrated photometry and write to observation
        calibrated = calibrator.get_calibrated_photometry(
            output_prefix="mag_cal_",
        )

        # Test dump: raw vstacked table (mag_cal_*, epoch_id, …) for debugging
        # if len(calibrated) > 0:
        #     out_base = Path(context.output_dir)
        #     tables_dir = out_base / "tables"
        #     checks.check_output_directories(out_base, tables_dir)
        #     dump_path = tables_dir / "calibrated_differential_vstack.ecsv"
        #     calibrated.write(
        #         str(dump_path), format="ascii.ecsv", overwrite=True
        #     )
        #     terminal_output.print_to_terminal(
        #         f"Test dump (differential calibrated vstack): {dump_path}",
        #         style_name="INFO",
        #     )

        filter_list = context.filter_list
        if len(calibrated) > 0:
            table_native = ensure_epoch_native_photometry_table(calibrated)
            table_native = _attach_jd_from_epoch_meta(
                table_native, context, config, filter_list
            )
            obs.table_magnitudes = table_native
            context.table_magnitudes = table_native
            if filter_list:
                if len(filter_list) == 1:
                    rts = ""
                elif len(filter_list) == 2:
                    rts = f"_{filter_list[0]}-{filter_list[1]}"
                else:
                    rts = ""
                write_epoch_native_magnitudes(
                    obs,
                    table_native,
                    object_id=config.object_id,
                    photometry_extraction_method=config.photometry_extraction_method,
                    rts=rts,
                )
                if config.write_differential_legacy_magnitudes_dat:
                    table_legacy = utilities.differential_calibrated_to_legacy_table(
                        calibrated, filter_list
                    )
                    utilities.save_magnitudes_ascii(
                        obs,
                        table_legacy,
                        object_id=config.object_id,
                        photometry_extraction_method=config.photometry_extraction_method,
                        rts=rts,
                    )
        else:
            inst = instrumental_epoch_native_from_calibration_epochs(epochs, filter_list)
            if len(inst) > 0:
                inst = _attach_jd_from_epoch_meta(
                    inst, context, config, filter_list
                )
                obs.table_magnitudes = inst
                context.table_magnitudes = inst
                if filter_list:
                    if len(filter_list) == 1:
                        rts_inst = ""
                    elif len(filter_list) == 2:
                        rts_inst = f"_{filter_list[0]}-{filter_list[1]}"
                    else:
                        rts_inst = ""
                else:
                    rts_inst = ""
                out_path = write_epoch_native_magnitudes(
                    obs,
                    inst,
                    object_id=config.object_id,
                    photometry_extraction_method=config.photometry_extraction_method,
                    rts=rts_inst,
                    file_stem="extracted_magnitudes",
                )
                terminal_output.print_to_terminal(
                    f"Differential calibration produced no rows; wrote instrumental "
                    f"epoch-native table: {out_path}",
                    style_name="INFO",
                )
            else:
                obs.table_magnitudes = calibrated
                context.table_magnitudes = calibrated
                terminal_output.print_to_terminal(
                    "Differential calibration produced no rows and epoch tables were empty; "
                    "no instrumental ECSV was written.",
                    style_name="WARNING",
                )

        # Fit results (T/ZP per epoch); do not assign to context.calib_parameters (legacy CalibParameters)
        context.differential_calib_parameters = getattr(
            calibrator, "calib_parameters", None
        )

        from ...diagnostic_plot_hooks import run_diagnostic_plots_phase

        run_diagnostic_plots_phase(
            context,
            config,
            "calibration_differential",
            differential_epochs=calibrator.epochs,
        )

        return context
