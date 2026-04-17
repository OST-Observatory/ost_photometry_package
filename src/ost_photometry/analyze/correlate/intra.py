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
        reference_obj_ids: list[int] | None = None,
        protect_reference_obj: bool = True,
        calibration_object_ids: list[int] | None = None,
        protect_calibration_objects: bool = True,
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

    reference_obj_ids
        IDs of the reference objects. The reference objects will not be
        removed from the list of objects.
        Default is ``None``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    calibration_object_ids
        IDs of the calibration objects.
        Default is ``None``.

    protect_calibration_objects
        If ``False`` calibration objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``False``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

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

    # #   Correlate the object positions from the images
    # #   -> find common objects
    correlation_index, new_reference_image_index, rejected_images, _ = correlate_datasets(
        x,
        y,
        current_wcs,
        n_objects,
        n_images,
        reference_dataset_id=image_series.reference_image_index,
        reference_object_ids=reference_obj_ids,
        protect_reference_objects=protect_reference_obj,
        calibration_object_ids=calibration_object_ids,
        protect_calibration_objects=protect_calibration_objects,
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

def correlate_preserve_variable(
        observation: 'analyze.Observation', filter_: str,
        max_pixel_between_objects: int = 3, ooi_correlation_strategy: int = 1,
        cross_identification_limit: int = 1, reference_image_index: int = 0,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        correlation_method: str = 'astropy',
        separation_limit: u.Quantity = 2. * u.arcsec, verbose: bool = False,
        duplicate_handling_object_identification: dict[str, str] | None = None,
        plots_for_all_images: bool = False,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf') -> None:
    """
    Correlate results from all images, while preserving the variable
    star

    Parameters
    ----------
    observation
        Container object with image series and object of interest properties

    filter_
        Filter

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

    reference_image_index
        ID of the reference image
        Default is ``0``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    duplicate_handling_object_identification
        Specifies how to handle multiple object identification filtering.
        There are two options for each 'correlation_method':
            'own':     'first_in_list' and 'flux'.  The 'first_in_list'
                        filtering just takes the first obtained result.
            'astropy': 'distance' and 'flux'. The 'distance' filtering is
                        based on the distance between the correlated objects.
                        In this case, the one with the smallest distance is
                        used.
        The second option for both correlation method is based on the measure
        flux values. In this case the largest one is used.
        Default is ``None``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    plots_for_all_images
        If True star map plots for all stars are created
        Default is ``False``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Get image series
    image_series = observation.image_series_dict[filter_]

    #   Get object of interests
    objects_of_interest = observation.objects_of_interest

    #   Find position of the variable star I
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

    #   Check if variable star was detected I
    #
    #   Get object of interests ID list
    object_of_interest_ids = observation.get_ids_object_of_interest(filter_=filter_)

    #   Correlate the stellar positions from the different filter
    correlate_image_series_images(
        image_series,
        max_pixel_between_objects=max_pixel_between_objects,
        ooi_correlation_strategy=ooi_correlation_strategy,
        cross_identification_limit=cross_identification_limit,
        reference_obj_ids=object_of_interest_ids,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        expected_bad_image_fraction=expected_bad_image_fraction,
        protect_reference_obj=protect_reference_obj,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
    )

    #   Find position of the variable star II
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

    #   Convert ra & dec to pixel coordinates
    coordinates_objects_of_interest = observation.objects_of_interest_coordinates
    x_position_object, y_position_object = image_series.wcs.all_world2pix(
        coordinates_objects_of_interest.ra,
        coordinates_objects_of_interest.dec,
        0,
    )

    #   Plot image with the final positions overlaid (final version)
    utilities.prepare_and_plot_starmap_from_image_series(
        image_series,
        x_position_object,
        y_position_object,
        plots_for_all_images=plots_for_all_images,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )

