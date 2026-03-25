"""Differential calibration step (APASSCatalog, PhotometryCalibrator)."""

import warnings

import numpy as np

from .... import terminal_output
from ... import utilities
from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ..bridge import observation_to_calibration_epochs
from ...extinction import CoefficientMode, ExtinctionOrder
from ...calibration_differential_catalog import PhotometryCalibrator
from ...warnings_types import OstPhotometryAnalyzeWarning


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
                f"pd={entry.get('reference_pd')} jd={entry.get('reference_jd')} "
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
    Differential photometry calibration using APASS and PhotometryCalibrator.

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
        import astropy.units as u

        obs = context._observation
        if obs is None:
            raise RuntimeError(
                "DifferentialCalibrationStep requires context._observation"
            )

        terminal_output.print_to_terminal(
            "Differential calibration (APASS + PhotometryCalibrator)",
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

        calibrator.setup_apass(
            field_center,
            radius=config.differential_apass_radius * u.arcmin,
            mag_limit=config.differential_apass_mag_limit,
        )

        for epoch_id, tbl in epochs.items():
            meta = context.calibration_epoch_meta.get(epoch_id, {})
            filter_obstimes = {}
            for f, jd in meta.get("filter_jds", {}).items():
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

        # Calibrate
        calibrator.calibrate(
            filters=context.filter_list,
            determine_color_terms=True,
            min_comparisons=5,
            sigma_clip=2.5,
            output_dir=context.output_dir,
            file_type=getattr(config, "file_type_plots", "pdf"),
        )

        # Get calibrated photometry and write to observation
        calibrated = calibrator.get_calibrated_photometry(
            output_prefix="mag_cal_",
        )

        obs.table_magnitudes = calibrated

        # Save to file in legacy format for compatibility with existing scripts
        filter_list = context.filter_list
        if filter_list and len(calibrated) > 0:
            table_legacy = utilities.differential_calibrated_to_legacy_table(
                calibrated, filter_list
            )
            if len(filter_list) == 1:
                rts = ""
            elif len(filter_list) == 2:
                rts = f"_{filter_list[0]}-{filter_list[1]}"
            else:
                rts = ""
            utilities.save_magnitudes_ascii(
                obs,
                table_legacy,
                id_object=config.object_id,
                photometry_extraction_method=config.photometry_extraction_method,
                rts=rts,
            )

        # Store calibrator results in context for post-process if needed
        context.calib_parameters = getattr(
            calibrator, "_calibration_results", None
        )

        return context
