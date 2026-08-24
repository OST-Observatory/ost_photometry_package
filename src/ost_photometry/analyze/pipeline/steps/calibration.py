"""
Unified calibration step: CalibrationEngine + epoch bridge.
"""

from __future__ import annotations

import warnings

import numpy as np

from .... import terminal_output
from ...calibration import CalibrationEngine, prepare_calibration_check_plots
from ...calibration.backends.linear import build_calibrator
from ...calibration_sources import crossmatch_standard_catalog, fetch_standard_calibration_catalog
from ...differential_photometry import DifferentialPhotometer
from ...extinction_io import build_extinction_corrector
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


def _calibration_summary_jd_by_epoch_id(
    context: AnalysisContext, config: PipelineConfig
) -> dict[str, float]:
    """Map epoch_id -> JD for summary plot (reference filter, else first available)."""
    meta = context.calibration_epoch_meta or {}
    fl = context.filter_list
    if not fl:
        return {}
    ref = config.reference_filter or fl[0]
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


def _attach_jd_from_epoch_meta(tbl, context: AnalysisContext, config: PipelineConfig, filter_list: list):
    """Add ``observation_jd`` for standalone ECSV output."""
    if len(tbl) == 0 or not filter_list or not context.calibration_epoch_meta:
        return tbl
    ref = config.reference_filter or filter_list[0]
    return attach_observation_jd_column(tbl, context.calibration_epoch_meta, ref)


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


def _crossmatch_epochs(epochs: dict, context: AnalysisContext, config: PipelineConfig) -> dict:
    """Attach ``mag_std_*`` from the calibration catalog to each epoch table."""
    from astropy.coordinates import SkyCoord

    from ...post_processing.magnitude_systems import require_catalog_bands_for_filters

    first_tbl = next(iter(epochs.values()))
    field_center = SkyCoord(
        np.mean(first_tbl["ra"]),
        np.mean(first_tbl["dec"]),
        unit="deg",
    )
    catalog = fetch_standard_calibration_catalog(
        context.filter_list,
        field_center,
        calibration_source=config.calibration_source,
        field_of_view_arcmin=config.calibration_catalog_radius_arcmin,
        calibration_catalog_mag_range=config.calibration_catalog_mag_range,
        vizier_dict=config.vizier_dict,
        path_calibration_file=config.path_calibration_file,
    )
    if catalog is not None and len(catalog) > 0 and context.filter_list:
        require_catalog_bands_for_filters(catalog, context.filter_list)
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

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        return config.skip_calibration

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        from astropy.time import Time

        obs = context.require_observation()
        strategy = config.calibration_strategy
        terminal_output.print_to_terminal(
            f"Calibration ({strategy}, grouping={config.calibration_grouping})",
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
        color_indices = config.color_indices
        epochs = _crossmatch_epochs(epochs, context, config)
        context.calibration_epochs = epochs

        jd_map = _calibration_summary_jd_by_epoch_id(context, config)
        file_type = config.file_type_plots
        calibrator = None
        extinction_corrector = build_extinction_corrector(
            config,
            fitted=context.extinction_coefficients,
        )
        photometer = DifferentialPhotometer(
            color_indices=color_indices,
            extinction_corrector=extinction_corrector,
        )

        if strategy == "linear_fit" and not config.derive_transform_from_data:
            ext_coeffs = None
            if config.extinction_mode == "from_value_airmass":
                ext_coeffs = context.extinction_coefficients
                if not ext_coeffs:
                    warnings.warn(
                        "extinction_mode='from_value_airmass' but no coefficients "
                        "from ExtinctionFitStep; using tabulated defaults.",
                        category=OstPhotometryAnalyzeWarning,
                        stacklevel=1,
                    )
            calibrator = build_calibrator(
                config,
                color_indices=color_indices,
                extinction_coefficients=ext_coeffs,
            )
            photometer = calibrator.photometer
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

        # linear_fit writes per-epoch plots inside PhotometryCalibrator or derive_transform backend
        if context.output_dir and strategy != "linear_fit":
            prepare_calibration_check_plots(
                context.output_dir,
                epochs,
                results,
                filter_list,
                file_type=file_type,
            )

        if (
            strategy == "linear_fit"
            and config.derive_transform_from_data
            and len(filter_list) == 2
        ):
            from ...calibration.derive_transform import apply_derive_transform_epochs

            calibrated = apply_derive_transform_epochs(
                epochs,
                results,
                filter_list,
            )
        elif strategy == "linear_fit" and calibrator is not None:
            photometer = calibrator.photometer
            calibrated = calibrator.get_calibrated_photometry(output_prefix="mag_cal_")
        else:
            calibrated = CalibrationEngine.apply(
                epochs,
                results,
                filter_list,
                photometer=photometer,
            )

        if len(calibrated) > 0 and config.uncertainty_mode != "fit_errors":
            from ...calibration.uncertainty import apply_uncertainty_mode_to_calibrated_table

            calibrated = apply_uncertainty_mode_to_calibrated_table(
                calibrated,
                results,
                filter_list,
                uncertainty_mode=config.uncertainty_mode,
                distribution_samples=config.distribution_samples,
            )

        if len(calibrated) > 0:
            table_native = ensure_epoch_native_photometry_table(calibrated)
            table_native = _attach_jd_from_epoch_meta(
                table_native, context, config, filter_list
            )
            from ...post_processing.magnitude_systems import (
                annotate_table_magnitude_meta,
                infer_filter_set,
                log_magnitude_output,
                resolve_catalog_magnitude_system,
                resolve_effective_output,
            )

            cal_fs = infer_filter_set(filter_list)
            cat_ms = resolve_catalog_magnitude_system(config.calibration_source)
            if cat_ms == "unknown":
                cat_ms = (
                    "ab"
                    if cal_fs == "sdss"
                    else "vega"
                    if cal_fs == "bessell"
                    else "unknown"
                )
            effective = resolve_effective_output(
                output_filter_set=config.output_filter_set,
                output_magnitude_system=config.output_magnitude_system,
                calibrated_filter_set=cal_fs,
                catalog_magnitude_system=cat_ms,
                convert_magnitudes=False,
            )
            annotate_table_magnitude_meta(
                table_native,
                filter_set=cal_fs,
                magnitude_system=cat_ms,
                catalog_magnitude_system=cat_ms,
                calibration_source=config.calibration_source,
                conversion_note="calibrated",
            )
            log_magnitude_output(effective, config.calibration_source)
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
            calibration_epochs=epochs,
        )
        return context


__all__ = ["CalibrationStep"]
