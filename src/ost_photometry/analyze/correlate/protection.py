"""Unified protected-object resolution for correlation steps."""

from __future__ import annotations

import typing

import numpy as np
import astropy.units as u
from astropy.table import Table

if typing.TYPE_CHECKING:
    from .. import analyze
    from ..pipeline.config import PipelineConfig

from ... import style, terminal_output


def merge_protected_object_ids(
    *,
    protected_object_ids: list[int] | None = None,
    reference_object_ids: list[int] | None = None,
    calibration_object_ids: list[int] | None = None,
    protect_ooi: bool = True,
    protect_calibration_objects: bool = False,
) -> list[int]:
    """
    Build a deduplicated list of reference-image row indices to keep during correlation.

    Explicit ``protected_object_ids`` are always included. Objects-of-interest and
    calibration-star IDs are added when the corresponding ``protect_*`` flags are set.
    """
    merged: list[int] = []
    seen: set[int] = set()

    def _add_values(values: list[int] | None) -> None:
        if values is None:
            return
        for val in values:
            if val is None:
                continue
            idx = int(val)
            if idx not in seen:
                seen.add(idx)
                merged.append(idx)

    _add_values(protected_object_ids)
    if protect_ooi:
        _add_values(reference_object_ids)
    if protect_calibration_objects:
        _add_values(calibration_object_ids)

    return merged


def _drop_rows_without_standard_mags(std_tbl: Table, filter_list: list[str]) -> Table:
    """Remove catalog rows that lack finite ``mag_std_*`` / ``err_std_*`` for requested filters."""
    if len(std_tbl) == 0:
        return std_tbl
    keep = np.ones(len(std_tbl), dtype=bool)
    for filter_ in filter_list:
        mag_col = f"mag_std_{filter_}"
        err_col = f"err_std_{filter_}"
        if mag_col in std_tbl.colnames:
            mag = np.asarray(std_tbl[mag_col], dtype=float)
            keep &= np.isfinite(mag)
            if hasattr(std_tbl[mag_col], "mask"):
                keep &= ~np.asarray(std_tbl[mag_col].mask, dtype=bool)
        if err_col in std_tbl.colnames:
            err = np.asarray(std_tbl[err_col], dtype=float)
            keep &= np.isfinite(err)
            if hasattr(std_tbl[err_col], "mask"):
                keep &= ~np.asarray(std_tbl[err_col].mask, dtype=bool)
    return std_tbl[keep]


def resolve_calibration_object_ids(
    image_series: "analyze.ImageSeries",
    filter_list: list[str],
    *,
    calibration_source: str = "APASS",
    calibration_catalog_mag_range: tuple[float, float] = (0.0, 18.5),
    vizier_dict: dict[str, str] | None = None,
    path_calibration_file: str | None = None,
    reference_image_index: int | None = None,
    max_pixel_between_objects: int = 3,
    ooi_correlation_strategy: int = 1,
    verbose: bool = False,
    indent: int = 3,
) -> tuple[list[int], list[float], list[float]]:
    """
    Match calibration catalog stars to detections in the reference image.

    Returns IDs (row indices in correlated photometry) and reference-image pixel
    positions for plotting.
    """
    from ..calibration_sources import fetch_standard_calibration_catalog
    from .inter import determine_object_position

    if reference_image_index is None:
        reference_image_index = image_series.reference_image_index

    image = image_series.image_list[reference_image_index]
    center = image.coordinates_image_center
    fov_x = image.field_of_view_x
    if calibration_source in ("vsp", "simbad"):
        field_of_view_arcmin = 1.5 * fov_x
    else:
        field_of_view_arcmin = fov_x

    std_tbl = fetch_standard_calibration_catalog(
        filter_list,
        center,
        calibration_source=calibration_source,
        field_of_view_arcmin=field_of_view_arcmin,
        calibration_catalog_mag_range=calibration_catalog_mag_range,
        vizier_dict=vizier_dict,
        path_calibration_file=path_calibration_file,
        indent=indent + 1,
    )
    std_tbl = _drop_rows_without_standard_mags(std_tbl, filter_list)

    n_calib_stars = len(std_tbl)
    if n_calib_stars == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo match between calibrations stars and "
            f"the\n extracted stars detected. -> EXIT {style.Bcolors.ENDC}"
        )

    calib_stars_ids: list[int] = []
    calib_x_pixel_positions: list[float] = []
    calib_y_pixel_positions: list[float] = []

    ra = np.asarray(std_tbl["ra"], dtype=float)
    dec = np.asarray(std_tbl["dec"], dtype=float)

    for k in range(n_calib_stars):
        id_calib_star, ref_count, x_calib_star, y_calib_star = determine_object_position(
            image,
            ra[k],
            dec[k],
            image_series.wcs,
            maximal_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            ra_unit=u.deg,
            verbose=verbose,
        )
        if verbose:
            terminal_output.print_to_terminal("")

        if ref_count != 0:
            calib_stars_ids.append(id_calib_star[1][0])
            calib_x_pixel_positions.append(x_calib_star)
            calib_y_pixel_positions.append(y_calib_star)

    terminal_output.print_to_terminal(
        f"{len(calib_stars_ids):d} matches",
        indent=indent,
        style_name="OKBLUE",
    )
    terminal_output.print_to_terminal("")

    return calib_stars_ids, calib_x_pixel_positions, calib_y_pixel_positions


def resolve_protected_object_ids_for_intra(
    observation: "analyze.Observation",
    image_series: "analyze.ImageSeries",
    filter_: str,
    filter_list: list[str],
    *,
    protected_object_ids: list[int] | None = None,
    protect_ooi: bool = True,
    protect_calibration_objects: bool = False,
    calibration_source: str = "APASS",
    calibration_catalog_mag_range: tuple[float, float] = (0.0, 18.5),
    vizier_dict: dict[str, str] | None = None,
    path_calibration_file: str | None = None,
    reference_image_index: int | None = None,
    max_pixel_between_objects: int = 3,
    ooi_correlation_strategy: int = 1,
    verbose: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    """
    Collect all protected row indices for intra-filter correlation.

    Returns merged IDs plus optional x/y overlay positions (calibration stars only;
    OOI positions are derived separately for plotting).
    """
    ooi_ids = None
    if protect_ooi and observation.objects_of_interest:
        ooi_ids = observation.get_ids_object_of_interest(filter_=filter_)

    cal_ids: list[int] | None = None
    cal_x: list[float] = []
    cal_y: list[float] = []
    if protect_calibration_objects:
        cal_ids, cal_x, cal_y = resolve_calibration_object_ids(
            image_series,
            filter_list,
            calibration_source=calibration_source,
            calibration_catalog_mag_range=calibration_catalog_mag_range,
            vizier_dict=vizier_dict,
            path_calibration_file=path_calibration_file,
            reference_image_index=reference_image_index,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            verbose=verbose,
        )

    merged = merge_protected_object_ids(
        protected_object_ids=protected_object_ids,
        reference_object_ids=ooi_ids,
        calibration_object_ids=cal_ids,
        protect_ooi=protect_ooi,
        protect_calibration_objects=protect_calibration_objects,
    )
    return merged, cal_x, cal_y


def resolve_protected_object_ids_for_inter(
    observation: "analyze.Observation",
    filter_list: list[str],
    image_series_dict: dict[str, "analyze.ImageSeries"],
    config: "PipelineConfig",
) -> list[int]:
    """Collect protected row indices on the reference filter for inter-filter correlation."""
    reference_filter = filter_list[0]
    reference_series = image_series_dict[reference_filter]

    ooi_ids = None
    if config.protect_ooi and observation.objects_of_interest:
        ooi_ids = observation.get_ids_object_of_interest(filter_=reference_filter)

    cal_ids: list[int] | None = None
    if config.protect_calibration_objects:
        cal_ids, _, _ = resolve_calibration_object_ids(
            reference_series,
            filter_list,
            calibration_source=config.calibration_source,
            calibration_catalog_mag_range=config.calibration_catalog_mag_range,
            vizier_dict=config.vizier_dict,
            path_calibration_file=config.path_calibration_file,
            reference_image_index=config.reference_image_index,
            max_pixel_between_objects=config.max_pixel_between_objects,
            ooi_correlation_strategy=config.ooi_correlation_strategy,
            verbose=config.verbose,
        )

    return merge_protected_object_ids(
        protected_object_ids=config.protected_object_ids,
        reference_object_ids=ooi_ids,
        calibration_object_ids=cal_ids,
        protect_ooi=config.protect_ooi,
        protect_calibration_objects=config.protect_calibration_objects,
    )
