"""
Unified calibration step: CalibrationEngine + epoch bridge.

Replaces CalibrationDataStep, CalibrationApplyStep, and DifferentialCalibrationStep.
"""

from __future__ import annotations

import warnings
import numpy as np

from .... import terminal_output
from ... import utilities
from ...calibration import CalibrationEngine, prepare_calibration_check_plots
from ...calibration.backends.linear import build_calibrator
from ...calibration_sources import crossmatch_standard_catalog, fetch_standard_calibration_catalog
from ...differential_photometry import DifferentialPhotometer, PhotometryCalibrator
from ...post_processing.adapters import ensure_epoch_native_photometry_table
from ...post_processing.io import write_epoch_native_magnitudes
from ...post_processing.light_curve import attach_observation_jd_column
from ...warnings_types import OstPhotometryAnalyzeWarning
from .. import base
from ..bridge import (
    instrumental_epoch_native_from_calibration_epochs,
    observation_to_calibration_epochs,
)
from ..config import PipelineConfig
from ..context import AnalysisContext
from .calibration_differential import (
    _attach_jd_from_epoch_meta,
    _calibration_summary_jd_by_epoch_id,
    _log_calibration_skips,
)


def _crossmatch_epochs(
    epochs: dict,
    context: AnalysisContext,
    config: PipelineConfig,
) -> dict:
    """Attach ``mag_std_*`` from the calibration catalog to each epoch table."""
    from astropy.coordinates import SkyCoord

    first_tbl = next(iter(epochs.values()))
    ra_mean = np.mean(first_tbl["ra"])
    dec_mean = np.mean(first_tbl["dec"])
    field_center = SkyCoord(ra_mean, dec_mean, unit="deg")
    catalog = fetch_standard_calibration_catalog(
        context.filter_list,
        field_center,
        calibration_source=config.calibration_source,
        field_of_view_arcmin=config.calibration_catalog_radius_arcmin,
        calibration_catalog_mag_range=config.calibration_catalog_mag_range,
        vizier_dict=config.vizier_dict,
        path_calibration_file=config.path_calibration_file,
    )
    out = {}
    for epoch_id, tbl in epochs.items():
        if catalog is not None and len(catalog) > 0:
            out[epoch_id] = crossmatch_standard_catalog(tbl, catalog, "ra", "dec")
        else:
            out[epoch_id] = tbl.copy()
    return out


class CalibrationStep(base.PipelineStep):
    """Epoch-native calibration via :class:`~ost_photometry.analyze.calibration.CalibrationEngine`."""

    name = "calibration"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        return config.skip_calibration

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        from astropy.time import Time

        obs = context.require_observation()
        strategy = config.resolved_calibration_strategy()
        terminal_output.print_to_terminal(
            f"Calibration ({strategy}, grouping={config.resolved_calibration_grouping()})",
            style_name="HEADER",
        )

        observation_to_calibration_epochs(context, config)
        if context.calibration_epochs_skipped:
            _log_calibration_skips(context.calibration_epochs_skipped)

        epochs = dict(context.calibration_epochs)
        if not epochs:
            raise RuntimeError(
                "No calibration epochs from observation_to_calibration_epochs. "
                "Ensure extraction and correlation have run."
            )

        filter_list = list(context.filter_list)
        color_indices = config.color_indices or config.differential_color_indices
        epochs = _crossmatch_epochs(epochs, context, config)
        context.calibration_epochs = epochs

        jd_map = _calibration_summary_jd_by_epoch_id(context, config)
        file_type = getattr(config, "file_type_plots", "pdf")
        calibrator = None
        photometer = DifferentialPhotometer(color_indices=color_indices)

        if strategy == "linear_fit":
            calibrator = build_calibrator(config, color_indices=color_indices)
            for epoch_id, tbl in epochs.items():
                meta = context.calibration_epoch_meta.get(epoch_id, {})
                filter_obstimes = {}
                jd_by_f = meta.get("jd_by_filter") or meta.get("filter_jds") or {}
                for f, jd in jd_by_f.items():
                    if jd is not None:
                        filter_obstimes[f] = Time(jd, format="jd")
                calibrator.add_epoch(
                    epoch_id,
                    tbl,
                    filter_obstimes=filter_obstimes if filter_obstimes else None,
                )
            epochs = dict(calibrator.epochs)

        results = CalibrationEngine.fit(
            epochs,
            config,
            filter_list,
            calibrator=calibrator,
            color_indices=color_indices,
            output_dir=context.output_dir,
            file_type=file_type,
            calibration_summary_x_jd=jd_map if jd_map else None,
        )
        context.calibration_results = results
        context.differential_calib_parameters = results

        if context.output_dir:
            prepare_calibration_check_plots(
                context.output_dir,
                epochs,
                results,
                filter_list,
                file_type=file_type,
            )

        if strategy == "linear_fit" and calibrator is not None:
            photometer = calibrator.photometer
            calibrated = calibrator.get_calibrated_photometry(output_prefix="mag_cal_")
        else:
            calibrated = CalibrationEngine.apply(
                epochs,
                results,
                filter_list,
                photometer=photometer,
            )

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
                write_legacy = (
                    config.write_legacy_wide_magnitudes_dat
                    or config.write_differential_legacy_magnitudes_dat
                )
                if write_legacy:
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
                inst = _attach_jd_from_epoch_meta(inst, context, config, filter_list)
                obs.table_magnitudes = inst
                context.table_magnitudes = inst
                out_path = write_epoch_native_magnitudes(
                    obs,
                    inst,
                    object_id=config.object_id,
                    photometry_extraction_method=config.photometry_extraction_method,
                    file_stem="extracted_magnitudes",
                )
                terminal_output.print_to_terminal(
                    f"Calibration produced no rows; wrote instrumental table: {out_path}",
                    style_name="WARNING",
                )
            else:
                warnings.warn(
                    "Calibration produced no rows and no instrumental fallback table.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=1,
                )

        from ...diagnostic_plot_hooks import run_diagnostic_plots_phase

        run_diagnostic_plots_phase(
            context,
            config,
            "calibration",
            differential_epochs=epochs,
        )
        return context


__all__ = ["CalibrationStep"]
