"""Intra-series correlation: align tables within one filter series."""

from __future__ import annotations

import warnings

import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .. import analyze

import astropy.units as u

from .. import utilities
from ..warnings_types import OstPhotometryAnalyzeWarning
from ... import terminal_output

from .core import correlate_datasets
from .ooi import identify_object_of_interest_in_dataset

def assign_correlated_object_ids_single_series(
    image_series: "analyze.ImageSeries",
) -> None:
    """Same as :func:`assign_global_correlated_object_ids` for one ImageSeries (intra only)."""
    if not image_series.image_list:
        return
    ref_im = image_series.reference_image_index
    phot0 = image_series.image_list[ref_im].photometry
    if phot0 is None:
        return
    n = len(phot0)
    ids = np.arange(n, dtype=np.int64)
    for image in image_series.image_list:
        if image.photometry is None:
            continue
        ni = len(image.photometry)
        if ni != n:
            warnings.warn(
                "assign_correlated_object_ids_single_series: "
                f"image has {ni} rows, reference has {n}.",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )
            image.photometry["id"] = np.arange(ni, dtype=np.int64)
        else:
            image.photometry["id"] = ids.copy()


def correlate_image_series_images(
        image_series: 'analyze.ImageSeries',
        max_pixel_between_objects: float = 3.,
        ooi_correlation_strategy: int = 1,
        cross_identification_limit: int = 1,
        protected_object_ids: list[int] | None = None,
        reference_obj_ids: list[int] | None = None,
        protect_ooi: bool = True,
        calibration_object_ids: list[int] | None = None,
        protect_calibration_objects: bool = False,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        correlation_method: str = 'astropy',
        separation_limit: u.Quantity = 2. * u.arcsec) -> None:
    """
    Correlate object positions from all stars in an image series to
    identify those objects that are visible on all images

    Parameters
    ----------
    image_series
        Image series of images, e.g., taken in one filter

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    ooi_correlation_strategy
        Option for the srcor correlation function
        Default is ``1``.

    cross_identification_limit
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    protected_object_ids
        Explicit row indices on the reference image to keep during correlation,
        independent of object type (variables, calibration stars, etc.).
        Default is ``None``.

    reference_obj_ids
        Legacy alias for objects-of-interest row indices (merged when
        ``protect_ooi`` is ``True``).
        Default is ``None``.

    protect_ooi
        If ``True``, include ``reference_obj_ids`` in the protected set.
        Default is ``True``.

    calibration_object_ids
        Legacy alias for calibration-star row indices (merged when
        ``protect_calibration_objects`` is ``True``).
        Default is ``None``.

    protect_calibration_objects
        If ``True``, include ``calibration_object_ids`` in the protected set.
        Default is ``False``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i``.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.
    """
    #   Number of images
    n_images = len(image_series.image_list)

    #   Set proxy image position IDs
    image_ids_arr = np.arange(n_images)

    terminal_output.print_to_terminal(
        f"Correlate results from images: {image_ids_arr}",
        indent=1,
    )

    #   Get WCS
    current_wcs = image_series.wcs

    #   Extract pixel positions of the objects
    x, y, n_objects = image_series.get_object_positions_pixel()

    from .protection import merge_protected_object_ids

    merged_protected_ids = merge_protected_object_ids(
        protected_object_ids=protected_object_ids,
        reference_object_ids=reference_obj_ids,
        calibration_object_ids=calibration_object_ids,
        protect_ooi=protect_ooi,
        protect_calibration_objects=protect_calibration_objects,
    )

    correlation_index, new_reference_image_index, rejected_images, _ = correlate_datasets(
        x,
        y,
        current_wcs,
        n_objects,
        n_images,
        reference_dataset_id=image_series.reference_image_index,
        protected_object_ids=merged_protected_ids,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        separation_limit=separation_limit,
        max_pixel_between_objects=max_pixel_between_objects,
        expected_bad_image_fraction=expected_bad_image_fraction,
        ooi_correlation_strategy=ooi_correlation_strategy,
        cross_identification_limit=cross_identification_limit,
        correlation_method=correlation_method,
    )

    #   Remove "bad" images from image IDs
    image_ids_arr = np.delete(image_ids_arr, rejected_images, 0)

    #   Remove images that are rejected (bad images) during the correlation process.
    image_series.image_list = [image_series.image_list[i] for i in image_ids_arr]
    image_series.reference_image_index = new_reference_image_index

    #   Limit the photometry tables to common objects.
    for j, image in enumerate(image_series.image_list):
        image.photometry = image.photometry[correlation_index[j, :]]

    assign_correlated_object_ids_single_series(image_series)


def correlate_preserve_objects(
        observation: 'analyze.Observation',
        filter_: str,
        filter_list: list[str],
        max_pixel_between_objects: int = 3,
        ooi_correlation_strategy: int = 1,
        cross_identification_limit: int = 1,
        reference_image_index: int = 0,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protected_object_ids: list[int] | None = None,
        protect_ooi: bool = True,
        protect_calibration_objects: bool = False,
        calibration_source: str = 'APASS',
        calibration_catalog_mag_range: tuple[float, float] = (0., 18.5),
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None,
        correlation_method: str = 'astropy',
        separation_limit: u.Quantity = 2. * u.arcsec,
        verbose: bool = False,
        duplicate_handling_object_identification: dict[str, str] | None = None,
        plots_for_all_images: bool = False,
        plot_only_reference_starmap: bool = True,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf',
) -> None:
    """
    Correlate exposures within one filter while keeping protected objects.

    Protection sources (combined, deduplicated):

    * ``protected_object_ids`` — explicit reference-image row indices
    * ``protect_ooi`` — objects of interest (variable stars, etc.)
    * ``protect_calibration_objects`` — catalog-matched calibration stars
    """
    from .protection import resolve_protected_object_ids_for_intra

    image_series = observation.image_series_dict[filter_]
    objects_of_interest = observation.objects_of_interest

    if protect_ooi and objects_of_interest:
        terminal_output.print_to_terminal(
            "Identify the variable objects",
            indent=1,
        )
        identify_object_of_interest_in_dataset(
            image_series.image_list[reference_image_index].photometry['x_fit'],
            image_series.image_list[reference_image_index].photometry['y_fit'],
            image_series.image_list[reference_image_index].photometry['flux_fit'],
            objects_of_interest,
            filter_,
            image_series.wcs,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            duplicate_handling=duplicate_handling_object_identification,
            verbose=verbose,
        )

    merged_ids, calib_x, calib_y = resolve_protected_object_ids_for_intra(
        observation,
        image_series,
        filter_,
        filter_list,
        protected_object_ids=protected_object_ids,
        protect_ooi=protect_ooi,
        protect_calibration_objects=protect_calibration_objects,
        calibration_source=calibration_source,
        calibration_catalog_mag_range=calibration_catalog_mag_range,
        vizier_dict=vizier_dict,
        path_calibration_file=path_calibration_file,
        reference_image_index=reference_image_index,
        max_pixel_between_objects=max_pixel_between_objects,
        ooi_correlation_strategy=ooi_correlation_strategy,
        verbose=verbose,
    )

    correlate_image_series_images(
        image_series,
        max_pixel_between_objects=max_pixel_between_objects,
        ooi_correlation_strategy=ooi_correlation_strategy,
        cross_identification_limit=cross_identification_limit,
        protected_object_ids=merged_ids,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        expected_bad_image_fraction=expected_bad_image_fraction,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
    )

    if protect_ooi and objects_of_interest:
        terminal_output.print_to_terminal(
            "Re-identify the variable star",
            indent=1,
        )
        identify_object_of_interest_in_dataset(
            image_series.image_list[image_series.reference_image_index].photometry['x_fit'],
            image_series.image_list[image_series.reference_image_index].photometry['y_fit'],
            image_series.image_list[image_series.reference_image_index].photometry['flux_fit'],
            objects_of_interest,
            filter_,
            image_series.wcs,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            duplicate_handling=duplicate_handling_object_identification,
            verbose=verbose,
        )

    overlay_x: list[float] = []
    overlay_y: list[float] = []
    if protect_calibration_objects and calib_x:
        overlay_x.extend(calib_x)
        overlay_y.extend(calib_y)
    if protect_ooi and objects_of_interest:
        coordinates_objects_of_interest = observation.objects_of_interest_coordinates
        x_position_object, y_position_object = image_series.wcs.all_world2pix(
            coordinates_objects_of_interest.ra,
            coordinates_objects_of_interest.dec,
            0,
        )
        overlay_x.extend(x_position_object.tolist())
        overlay_y.extend(y_position_object.tolist())

    if overlay_x:
        utilities.prepare_and_plot_starmap_from_image_series(
            image_series,
            overlay_x,
            overlay_y,
            plots_for_all_images=plots_for_all_images and not plot_only_reference_starmap,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            file_type_plots=file_type_plots,
        )


def correlate_preserve_variable(
        observation: 'analyze.Observation', filter_: str,
        max_pixel_between_objects: int = 3, ooi_correlation_strategy: int = 1,
        cross_identification_limit: int = 1, reference_image_index: int = 0,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_ooi: bool = True,
        correlation_method: str = 'astropy',
        separation_limit: u.Quantity = 2. * u.arcsec, verbose: bool = False,
        duplicate_handling_object_identification: dict[str, str] | None = None,
        plots_for_all_images: bool = False,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf') -> None:
    """Correlate while preserving objects of interest (legacy wrapper)."""
    correlate_preserve_objects(
        observation,
        filter_,
        [filter_],
        max_pixel_between_objects=max_pixel_between_objects,
        ooi_correlation_strategy=ooi_correlation_strategy,
        cross_identification_limit=cross_identification_limit,
        reference_image_index=reference_image_index,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        expected_bad_image_fraction=expected_bad_image_fraction,
        protect_ooi=protect_ooi,
        protect_calibration_objects=False,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
        verbose=verbose,
        duplicate_handling_object_identification=duplicate_handling_object_identification,
        plots_for_all_images=plots_for_all_images,
        plot_only_reference_starmap=not plots_for_all_images,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )
