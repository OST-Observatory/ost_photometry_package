"""Calibration apply step (apply_calibration)."""

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ... import calibration, utilities
from ...post_processing.adapters import ensure_epoch_native_photometry_table
from ...post_processing.light_curve import attach_observation_jd_column
from ..bridge import (
    instrumental_epoch_native_from_calibration_epochs,
    observation_to_calibration_epochs,
    populate_legacy_calibration_epoch_meta,
)
from ...post_processing.io import write_epoch_native_magnitudes
from .... import terminal_output


class CalibrationApplyStep(base.PipelineStep):
    """Apply zero points and magnitude transformation for each filter combination."""

    name = "calibration_apply"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        return config.skip_calibration or config.calibration_module == "differential"

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        obs = context.require_observation()
        if obs.calib_parameters is None:
            raise RuntimeError(
                "CalibrationApplyStep requires calib_parameters on the Observation"
            )

        terminal_output.print_to_terminal(
            "Default calibration (legacy)",
            style_name="HEADER",
        )
        import warnings

        warnings.warn(
            "Pipeline calibration_module='legacy' is deprecated; prefer 'differential' "
            "once comparison tests pass.",
            DeprecationWarning,
            stacklevel=2,
        )

        calibration_filters = obs.calib_parameters.column_names
        _, usable_filter_combinations = (
            utilities.find_filter_for_magnitude_transformation(
                context.filter_list,
                calibration_filters,
            )
        )

        for filter_combination in usable_filter_combinations:
            calibration.apply_calibration(
                obs,
                filter_combination,
                apply_transformation=config.apply_transformation,
                transformation_coefficients_dict=config.transformation_coefficients_dict,
                derive_transformation_coefficients=config.derive_transformation_coefficients,
                object_id=config.object_id,
                photometry_extraction_method=config.photometry_extraction_method,
                calculate_zero_point_statistic=config.calculate_zero_point_statistic,
                distribution_samples=config.distribution_samples,
                file_type_plots=config.file_type_plots,
                add_progress_bar=False,
            )

        calibrated_ready = (
            bool(usable_filter_combinations)
            and obs.table_magnitudes is not None
            and len(obs.table_magnitudes) > 0
        )

        if calibrated_ready:
            obs.table_magnitudes = ensure_epoch_native_photometry_table(
                obs.table_magnitudes
            )
            context.table_magnitudes = obs.table_magnitudes
            populate_legacy_calibration_epoch_meta(context)
            fl = list(context.filter_list)
            if fl and context.calibration_epoch_meta:
                obs.table_magnitudes = attach_observation_jd_column(
                    obs.table_magnitudes,
                    context.calibration_epoch_meta,
                    fl[0],
                )
                context.table_magnitudes = obs.table_magnitudes
        else:
            observation_to_calibration_epochs(context, config)
            inst = instrumental_epoch_native_from_calibration_epochs(
                context.calibration_epochs,
                list(context.filter_list),
            )
            if len(inst) > 0:
                obs.table_magnitudes = inst
                context.table_magnitudes = inst
                fl = context.filter_list
                if fl:
                    if len(fl) == 1:
                        rts = ""
                    elif len(fl) == 2:
                        rts = f"_{fl[0]}-{fl[1]}"
                    else:
                        rts = ""
                else:
                    rts = ""
                out_path = write_epoch_native_magnitudes(
                    obs,
                    inst,
                    object_id=config.object_id,
                    photometry_extraction_method=config.photometry_extraction_method,
                    rts=rts,
                    file_stem="extracted_magnitudes",
                )
                terminal_output.print_to_terminal(
                    f"No legacy calibration applied; wrote instrumental epoch-native table: {out_path}",
                    style_name="INFO",
                )
            else:
                terminal_output.print_to_terminal(
                    "No legacy calibration applied and no calibration epochs could be built "
                    "for an instrumental ECSV dump (check extraction, WCS, and pairing).",
                    style_name="WARNING",
                )

        from ...diagnostic_plot_hooks import run_diagnostic_plots_phase

        run_diagnostic_plots_phase(context, config, "calibration_apply")

        return context
