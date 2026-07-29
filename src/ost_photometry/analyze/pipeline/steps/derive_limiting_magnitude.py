"""Derive limiting magnitude after calibration (Image path or epoch-native tables)."""

import numpy as np

from .... import terminal_output
from ... import utilities
from ...post_processing.light_curve import epoch_native_mag_err_columns
from ...post_processing.imaging import ImagingPlotContext
from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig


def _find_image_by_image_id(image_series, image_id):
    for im in image_series.image_list:
        if getattr(im, "image_id", None) == image_id:
            return im
    return None


def _image_depth_calibrated_offset(
    context: AnalysisContext,
    tbl,
    epoch_id: str,
    filter_: str,
) -> float:
    """
    Map ImageDepth (instrumental -2.5 log10 flux at ZP=0) toward calibrated mags.

    Prefer median(mag_cal - mag_inst) over stars in this epoch; else differential
    transformation zero point for this epoch/filter.
    """
    inst_col = f"mag_{filter_}"
    cal_col = f"mag_cal_{filter_}"
    if "epoch_id" not in tbl.colnames:
        return 0.0
    mask = np.asarray(tbl["epoch_id"]).astype(str) == str(epoch_id)
    sub = tbl[mask]
    if len(sub) == 0:
        return 0.0
    if inst_col in sub.colnames and cal_col in sub.colnames:
        a = np.asarray(sub[cal_col], dtype=float)
        b = np.asarray(sub[inst_col], dtype=float)
        d = a - b
        m = np.isfinite(a) & np.isfinite(b)
        if np.any(m):
            return float(np.nanmedian(d[m]))
    dc = context.calibration_results
    if dc and epoch_id in dc:
        tc = dc[epoch_id].transformation.get(filter_)
        if tc is not None:
            return float(tc.zero_point)
    return 0.0


def _imaging_context_from_pipeline_image(image, filter_: str) -> ImagingPlotContext:
    arr = np.asarray(image.get_data())
    return ImagingPlotContext(
        wcs=image.wcs,
        reference_image=arr,
        out_path_stub=image.out_path.name,
        filter_name=filter_,
        image_shape=tuple(arr.shape),
        plot_reference_image_id=getattr(image, "image_id", None),
    )


class DeriveLimitingMagnitudeStep(base.PipelineStep):
    """Run ``derive_limiting_magnitude`` per filter (legacy) or per epoch×filter (differential)."""

    name = "derive_limiting_magnitude"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_calibration:
            return True
        if config.skip_derive_limiting_magnitude:
            return True
        return False

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        obs = context.require_observation()
        if obs is None:
            raise RuntimeError(
                "DeriveLimitingMagnitudeStep requires context._observation"
            )

        tbl = context.table_magnitudes
        has_epoch_native = (
            tbl is not None
            and len(tbl) > 0
            and "epoch_id" in tbl.colnames
            and bool(context.calibration_epoch_meta)
        )

        if has_epoch_native:
            eids = np.unique(np.asarray(tbl["epoch_id"]).astype(str))
            for eid in eids:
                eid_s = str(eid)
                meta = context.calibration_epoch_meta.get(eid_s, {})
                image_id_by_filter = (
                    meta.get("image_id_by_filter")
                    or meta.get("image_pd_by_filter")
                    or meta.get("filter_pds")
                    or {}
                )
                for filter_ in context.filter_list:
                    mag_pair = epoch_native_mag_err_columns(tbl, filter_)
                    if mag_pair is None:
                        continue
                    epoch_image_id = image_id_by_filter.get(filter_)
                    if epoch_image_id is None:
                        terminal_output.print_to_terminal(
                            f"DeriveLimitingMagnitudeStep: epoch {eid_s!r} has no "
                            f"image_id_by_filter entry for filter {filter_!r}; skipping "
                            "this band.",
                            style_name="WARNING",
                        )
                        continue
                    series = obs.image_series_dict.get(filter_)
                    if series is None:
                        continue
                    image = _find_image_by_image_id(series, epoch_image_id)
                    if image is None:
                        terminal_output.print_to_terminal(
                            f"DeriveLimitingMagnitudeStep: no image with image_id={epoch_image_id} "
                            f"for filter {filter_!r}; skipping.",
                            style_name="WARNING",
                        )
                        continue
                    if image.pixel_scale is None:
                        terminal_output.print_to_terminal(
                            f"DeriveLimitingMagnitudeStep: image image_id={epoch_image_id} "
                            f"({filter_!r}) has no pixel_scale; skipping.",
                            style_name="WARNING",
                        )
                        continue
                    offset = _image_depth_calibrated_offset(
                        context, tbl, eid_s, filter_
                    )
                    ictx = _imaging_context_from_pipeline_image(image, filter_)
                    utilities.derive_limiting_magnitude(
                        photometry_table=tbl,
                        filter_list=[filter_],
                        epoch_id=eid_s,
                        imaging_context=ictx,
                        pixel_scale=float(image.pixel_scale),
                        zeropoint=0.0,
                        image_depth_mag_offset=offset,
                        aperture_radius=config.aperture_radius,
                        radii_unit=config.radii_unit,
                        file_type_plots=config.file_type_plots,
                        use_wcs_projection_for_star_maps=(
                            config.use_wcs_projection_for_star_maps
                        ),
                    )
            return context

        if tbl is None or len(tbl) == 0:
            terminal_output.print_to_terminal(
                "DeriveLimitingMagnitudeStep: no epoch-native table_magnitudes; skipping.",
                style_name="WARNING",
            )
            return context

        if obs.calib_parameters is None:
            terminal_output.print_to_terminal(
                "DeriveLimitingMagnitudeStep: no calib_parameters and no "
                "epoch-native calibration table; skipping.",
                style_name="WARNING",
            )
            return context

        _, usable_filter_combinations = (
            utilities.find_filter_for_magnitude_transformation(
                context.filter_list,
                obs.calib_parameters.column_names,
            )
        )

        for filter_combination in usable_filter_combinations:
            utilities.derive_limiting_magnitude(
                obs,
                filter_combination,
                config.reference_image_index,
                aperture_radius=config.aperture_radius,
                radii_unit=config.radii_unit,
                file_type_plots=config.file_type_plots,
                use_wcs_projection_for_star_maps=(
                    config.use_wcs_projection_for_star_maps
                ),
            )

        return context
