"""Inter-filter and calibration-related correlation."""

from __future__ import annotations

import multiprocessing as mp
import warnings

import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .. import analyze

from astropy.coordinates import SkyCoord, matching
import astropy.units as u
from astropy.table import Table, Column
from astropy import wcs

from .. import calibration_data, plots, utilities
from ..warnings_types import OstPhotometryAnalyzeWarning
from ... import style, terminal_output
from ... import utilities as base_utilities

from .core import correlate_datasets, correlation_own
from .intra import correlate_image_series_images
from .ooi import (
    identify_object_of_interest_in_dataset,
    verify_objects_of_interest_global_correlated_ids,
)

def assign_global_correlated_object_ids(
    observation: "analyze.Observation",
    filter_list: list[str],
) -> None:
    """
    Set photometry ``id`` to 0 .. N-1 in correlated row order on all images.

    After intra- and inter-filter correlation, row ``k`` refers to the same
    physical object across filters (and across images within each filter).
    The table ``id`` column is aligned to that index so downstream code
    (e.g. differential calibration on epoch-native vstack tables)
    can match objects by ``id``.

    Object-of-interest ``id_in_image_series`` values are row indices in these
    tables; they stay valid when ``id`` equals the row index.

    Parameters
    ----------
    observation
        Observation whose ``image_series_dict`` entries hold correlated photometry.
    filter_list
        Filters to update (typically the filters that participated in correlation).
    """
    n_expect: int | None = None
    ref_filter: str | None = None
    for filter_ in filter_list:
        series = observation.image_series_dict.get(filter_)
        if series is None or not series.image_list:
            continue
        ref_im = series.reference_image_index
        phot0 = series.image_list[ref_im].photometry
        if phot0 is None:
            continue
        n = len(phot0)
        if n_expect is None:
            n_expect = n
            ref_filter = filter_
        elif n != n_expect:
            warnings.warn(
                "assign_global_correlated_object_ids: "
                f"filter {filter_!r} has {n} correlated objects, "
                f"{ref_filter!r} has {n_expect}. "
                "Assigning 0..n-1 per filter; cross-filter id matching may be wrong.",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )
        ids = np.arange(n, dtype=np.int64)
        for image in series.image_list:
            if image.photometry is None:
                continue
            ni = len(image.photometry)
            if ni != n:
                warnings.warn(
                    "assign_global_correlated_object_ids: "
                    f"filter {filter_!r} image has {ni} rows, reference has {n}; "
                    "assigning id = 0 .. ni-1 for this image only.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )
                image.photometry["id"] = np.arange(ni, dtype=np.int64)
            else:
                image.photometry["id"] = ids.copy()


def correlate_image_series(
        observation: 'analyze.Observation', filter_list: list[str] | set[str],
        max_pixel_between_objects: int = 3,
        ooi_correlation_strategy: int = 1, cross_identification_limit: int = 1,
        reference_image_series_id: int = 0,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        protect_calibration_objects: bool = False,
        correlation_method: str = 'astropy',
        separation_limit: u.quantity.Quantity = 2. * u.arcsec,
        force_correlation_calibration_objects: bool = False,
        verbose: bool = False, file_type_plots: str = 'pdf',
        duplicate_handling_object_identification: dict[str, str] | None = None,
        indent: int = 1,
        debug_verify_ooi_global_ids: bool = False,
) -> None:
    """
    Correlate star lists from the stacked images of all filters to find
    those stars that are visible on all images

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        List with filter identifiers.

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

    reference_image_series_id
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

    protect_calibration_objects
        If ``False`` calibration objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``False``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    force_correlation_calibration_objects
        If ``True`` the correlation between the already correlated
        series and the calibration data will be enforced.
        Default is ``False``

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    duplicate_handling_object_identification
        Specifies how to handle multiple object identification filtering during
        object identification.
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

    indent
        Indentation for the console output lines
        Default is ``1``.

    debug_verify_ooi_global_ids
        If True, after :func:`assign_global_correlated_object_ids` run
        :func:`verify_objects_of_interest_global_correlated_ids` (re-match sky
        positions per image vs. stored ``id_in_image_series``).
    """
    terminal_output.print_to_terminal(
        "Correlate image series",
        indent=indent,
    )

    #   Get image series
    image_series_dict = observation.get_image_series(filter_list)
    image_series_keys = list(image_series_dict.keys())

    #   Get Reference filter
    reference_filter = list(filter_list)[reference_image_series_id]

    #   Define variables
    n_object_all_images_list = []
    x_pixel_positions_all_images = []
    y_pixel_positions_all_images = []
    wcs_list_image_series = []

    for id_series, series in enumerate(image_series_dict.values()):
        #   Get number of objects in each table/image
        wcs_list_image_series.append(series.wcs)

        reference_image_index = series.reference_image_index
        _x = series.image_list[reference_image_index].photometry['x_fit']
        x_pixel_positions_all_images.append(_x)
        y_pixel_positions_all_images.append(
            series.image_list[reference_image_index].photometry['y_fit']
        )
        n_object_all_images_list.append(len(_x))

    #   Max. number of objects
    n_objects_max = np.max(n_object_all_images_list)

    #   Number of image series
    n_series = len(x_pixel_positions_all_images)

    #   Get calibration star IDs as a list such that it can be later
    #   easily combined with the object of interest IDs
    if (observation.calib_parameters is not None and
            observation.calib_parameters.ids_calibration_objects is not None):
        calibration_object_ids = observation.calib_parameters.ids_calibration_objects.tolist()
    else:
        calibration_object_ids = None

    reference_obj_ids = observation.get_ids_object_of_interest(
        filter_=reference_filter,
    )

    #   Correlate the object positions from the images
    #   -> find common objects
    correlation_index, _, rejected_series, _ = correlate_datasets(
        x_pixel_positions_all_images,
        y_pixel_positions_all_images,
        wcs_list_image_series[reference_image_series_id],
        n_objects_max,
        n_series,
        dataset_type='series',
        reference_dataset_id=reference_image_series_id,
        reference_object_ids=reference_obj_ids,
        protect_reference_objects=protect_reference_obj,
        calibration_object_ids=calibration_object_ids,
        protect_calibration_objects=protect_calibration_objects,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        separation_limit=separation_limit,
        advanced_cleanup=False,
        max_pixel_between_objects=max_pixel_between_objects,
        expected_bad_image_fraction=expected_bad_image_fraction,
        ooi_correlation_strategy=ooi_correlation_strategy,
        cross_identification_limit=cross_identification_limit,
        correlation_method=correlation_method,
    )

    #   Remove "bad"/rejected image series
    for series_rejected in rejected_series:
        image_series_dict.pop(image_series_keys[series_rejected])

    #   Limit the photometry tables object_ids to common objects.
    for j, series in enumerate(image_series_dict.values()):
        for image in series.image_list:
            image.photometry = image.photometry[correlation_index[j, :]]

    #   Re-identify position of objects of interest
    objects_of_interest = observation.objects_of_interest
    if objects_of_interest:
        terminal_output.print_to_terminal(
            "Identify objects of interest",
            indent=indent + 1,
        )

        series = image_series_dict[reference_filter]
        reference_image_index = series.reference_image_index
        identify_object_of_interest_in_dataset(
            series.image_list[reference_image_index].photometry['x_fit'],
            series.image_list[reference_image_index].photometry['y_fit'],
            series.image_list[reference_image_index].photometry['flux_fit'],
            objects_of_interest,
            reference_filter,
            series.wcs,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            verbose=verbose,
            correlation_method=correlation_method,
            duplicate_handling=duplicate_handling_object_identification,
            indent=indent + 1,
        )

        #   Replicate IDs for the objects of interest
        #   -> This is required, since the identification above is only for the
        #      reference filter / image series
        for object_ in objects_of_interest:
            object_id = object_.id_in_image_series[reference_filter]
            for filter_ in filter_list:
                if filter_ != reference_filter:
                    object_.id_in_image_series[filter_] = object_id

    terminal_output.print_to_terminal('')

    #   Correlate with calibration data if necessary
    calibration_parameters = observation.calib_parameters

    if calibration_parameters is not None and (calibration_parameters.ids_calibration_objects is None
                                               or force_correlation_calibration_objects):
        select_calibration_objects(
            observation,
            filter_list,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            file_type_plots=file_type_plots,
            indent=2,
        )

    # Global object id = row index after correlation (for cross-filter pipelines)
    assign_global_correlated_object_ids(observation, list(image_series_dict.keys()))

    if debug_verify_ooi_global_ids:
        verify_objects_of_interest_global_correlated_ids(
            observation,
            list(image_series_dict.keys()),
            separation_limit=separation_limit,
            correlation_method=correlation_method,
            duplicate_handling_object_identification=duplicate_handling_object_identification,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            indent=indent,
        )


def inter_filter_correlation_separations_arcsec(
    observation: "analyze.Observation",
    filter_list: list[str] | set[str],
    reference_filter_index: int = 0,
) -> tuple[np.ndarray, str, list[str]]:
    """
    On-sky separations (arcsec) between reference-filter positions and other
    filters' positions for the same row after inter-filter correlation.

    Rows are assumed aligned (same length per series reference image), as after
    :func:`correlate_image_series`.
    """
    fl = list(filter_list)
    if len(fl) < 2:
        return np.array([]), "", []
    if reference_filter_index < 0 or reference_filter_index >= len(fl):
        reference_filter_index = 0
    ref = fl[reference_filter_index]
    isd = observation.image_series_dict
    if ref not in isd:
        return np.array([]), ref, []
    sref = isd[ref]
    ref_img = sref.image_list[sref.reference_image_index]
    if ref_img.photometry is None or len(ref_img.photometry) == 0:
        return np.array([]), ref, []
    n = len(ref_img.photometry)
    c_ref = SkyCoord.from_pixel(
        ref_img.photometry["x_fit"],
        ref_img.photometry["y_fit"],
        sref.wcs,
    )
    chunks: list[np.ndarray] = []
    others: list[str] = []
    for f in fl:
        if f == ref:
            continue
        if f not in isd:
            continue
        ser = isd[f]
        img = ser.image_list[ser.reference_image_index]
        if img.photometry is None or len(img.photometry) != n:
            continue
        c_f = SkyCoord.from_pixel(
            img.photometry["x_fit"],
            img.photometry["y_fit"],
            ser.wcs,
        )
        chunks.append(np.asarray(c_ref.separation(c_f).arcsec, dtype=float))
        others.append(f)
    if not chunks:
        return np.array([]), ref, []
    return np.concatenate(chunks), ref, others

def determine_object_position(
        image: base_utilities.Image, ra_obj: float, dec_obj: float, w: wcs.WCS,
        maximal_pixel_between_objects: float = 3.,
        ooi_correlation_strategy: int = 1,
        ra_unit: u.quantity.Quantity = u.hourangle,
        dec_unit: u.quantity.Quantity = u.deg, verbose: bool = False
        ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Find the image coordinates of a star based on the stellar
    coordinates and the WCS of the image

    Parameters
    ----------
    image
        Object with all image specific properties

    ra_obj
        Right ascension of the object

    dec_obj
        Declination of the object

    w
        WCS infos

    maximal_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    ooi_correlation_strategy
        Option for the srcor correlation function
        Default is ``1``.

    ra_unit
        Right ascension unit
        Default is ``u.hourangle``.

    dec_unit
        Declination unit
        Default is ``u.deg``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    Returns
    -------
    indexes
        Index positions of matched objects in the origins. Is -1 is no
        objects were found.

    count
        Number of times the object has been identified on the image

    x_position_object
        X coordinates of the objects in pixel

    y_position_object
        Y coordinates of the objects in pixel
    """
    #   Make coordinates object
    coord_obj = SkyCoord(
        ra_obj,
        dec_obj,
        unit=(ra_unit, dec_unit),
        frame="icrs",
    )

    #   Convert ra & dec to pixel coordinates
    x_position_object, y_position_object = w.all_world2pix(
        coord_obj.ra,
        coord_obj.dec,
        0
    )

    #   Get photometry tabel
    tbl = image.photometry

    #   Number of objects
    count = len(tbl['x_fit'])

    #   Define and fill new arrays to allow correlation
    x_position_all = np.zeros((count, 2))
    y_position_all = np.zeros((count, 2))
    x_position_all[0, 0] = x_position_object
    x_position_all[0:count, 1] = tbl['x_fit']
    y_position_all[0, 0] = y_position_object
    y_position_all[0:count, 1] = tbl['y_fit']

    #   Correlate object with stars on the image
    indexes, reject, count, reject_obj = correlation_own(
        x_position_all,
        y_position_all,
        maximal_pixel_between_objects,
        ooi_correlation_strategy=ooi_correlation_strategy,
        silent=not verbose,
    )

    return indexes, count, x_position_object, y_position_object


#   TODO: Make the option to protect claibration objects from beeing eliminated in the correlation process 
#         available as a generall option
def correlate_preserve_calibration_objects(
        image_series: 'analyze.ImageSeries', filter_list: list[str],
        calibration_source: str = 'APASS',
        calibration_catalog_mag_range: tuple[float, float] = (0., 18.5),
        vizier_dict: dict[str, str] | None = None, calib_file=None,
        max_pixel_between_objects: int = 3, ooi_correlation_strategy: int = 1,
        verbose: bool = False, cross_identification_limit: int = 1,
        reference_image_index: int = 0, n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_calibration_objects: bool = True,
        plot_only_reference_starmap: bool = True,
        correlation_method: str = 'astropy',
        separation_limit: u.quantity.Quantity = 2. * u.arcsec,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf') -> None:
    """
    Correlate results from all images, while preserving the calibration
    stars

    Parameters
    ----------
    image_series
        Image series object with all image data taken in a specific
        filter

    filter_list
        Filter list

    calibration_source
        Catalog / lookup key (e.g. ``APASS``, ``simbad``, ``vsp``, or a ``vizier_dict`` key).
        Default is ``APASS``.

    calibration_catalog_mag_range
        Inclusive magnitude range for calibration catalog stars.
        Default is ``(0.,18.5)``.

    vizier_dict
        Identifiers of catalogs, containing calibration data
        Default is ``None``.

    calib_file
        Path to the calibration file
        Default is ``None``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    ooi_correlation_strategy
        Option for the srcor correlation function
        Default is ``1``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

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

    protect_calibration_objects
        If ``False`` calibration objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``False``.

    plot_only_reference_starmap
        If True only the starmap for the reference image will be created.
        Default is ``True``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Load calibration data
    calib_tbl, column_names, ra_unit = calibration_data.load_calibration_data_table(
        image_series.image_list[reference_image_index],
        filter_list,
        calibration_source=calibration_source,
        calibration_catalog_mag_range=calibration_catalog_mag_range,
        vizier_dict=vizier_dict,
        path_calibration_file=calib_file,
    )

    #   Number of calibration stars
    n_calib_stars = len(calib_tbl)

    if n_calib_stars == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo match between calibrations stars and "
            f"the\n extracted stars detected. -> EXIT {style.Bcolors.ENDC}"
        )

    #   Find IDs of calibration stars to ensure they are not deleted in
    #   the correlation process
    #
    #   Lists for IDs, and xy coordinates
    calib_stars_ids = []
    calib_x_pixel_positions = []
    calib_y_pixel_positions = []

    #   Loop over all calibration stars
    #   TODO: The determination of the calibration star IDs should not be
    #         needed anymore
    #   TODO: Rewrite this with correlate.correlate_with_calibration_objects
    for k in range(0, n_calib_stars):
        #   Find the calibration star
        id_calib_star, ref_count, x_calib_star, y_calib_star = determine_object_position(
            image_series.image_list[reference_image_index],
            calib_tbl[column_names['ra']].data[k],
            calib_tbl[column_names['dec']].data[k],
            image_series.wcs,
            maximal_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            ra_unit=ra_unit,
            verbose=verbose,
        )
        if verbose:
            terminal_output.print_to_terminal('')

        #   Add ID and coordinates of the calibration star to the lists
        if ref_count != 0:
            calib_stars_ids.append(id_calib_star[1][0])
            calib_x_pixel_positions.append(x_calib_star)
            calib_y_pixel_positions.append(y_calib_star)
    terminal_output.print_to_terminal(
        f"{len(calib_stars_ids):d} matches",
        indent=3,
        style_name='OKBLUE',
    )
    terminal_output.print_to_terminal('')

    #   Correlate the results from all images
    correlate_image_series_images(
        image_series,
        max_pixel_between_objects=max_pixel_between_objects,
        ooi_correlation_strategy=ooi_correlation_strategy,
        cross_identification_limit=cross_identification_limit,
        calibration_object_ids=calib_stars_ids,
        protect_calibration_objects=protect_calibration_objects,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        expected_bad_image_fraction=expected_bad_image_fraction,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
    )

    #   Plot image with the final positions overlaid (final version)
    utilities.prepare_and_plot_starmap_from_image_series(
        image_series,
        calib_x_pixel_positions,
        calib_y_pixel_positions,
        plot_reference_only=plot_only_reference_starmap,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )


def correlate_with_calibration_objects(
        image_series: 'analyze.ImageSeries',
        calibration_object_coordinates: SkyCoord,
        calibration_tbl: Table, filter_list: list[str],
        column_names: dict[str, str], correlation_method: str = 'astropy',
        separation_limit: u.Quantity = 2. * u.arcsec,
        max_pixel_between_objects: int = 3, ooi_correlation_strategy: int = 1,
        indent: int = 1, file_type_plots: str = 'pdf',
        use_wcs_projection_for_star_maps: bool = True,
        ) -> tuple[Table, np.ndarray]:
    """
    Correlate observed objects with calibration stars

    Parameters
    ----------
    image_series
        Class with all images of a specific image series

    calibration_object_coordinates
        Coordinates of the calibration objects

    calibration_tbl
        Table with calibration data

    filter_list
        Filter list

    column_names
        Actual names of the columns in calibration_tbl versus
        the internal default names

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    ooi_correlation_strategy
        Option for the srcor correlation function
        Default is ``1``.

    indent
        Indentation for the console output lines
        Default is ``1``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    Returns
    -------
    calibration_tbl_sort
        Sorted table with calibration data

    index_obj_instrument
        Index of the observed stars that correspond to the calibration stars
    """
    terminal_output.print_to_terminal(
        "Correlate observed objects with calibration stars",
        indent=indent,
    )

    #   Pixel positions of the observed object
    reference_image_index = image_series.reference_image_index
    pixel_position_obj_x = image_series.image_list[reference_image_index].photometry['x_fit'].value.ravel()
    pixel_position_obj_y = image_series.image_list[reference_image_index].photometry['y_fit'].value.ravel()

    #   Pixel positions of calibration object
    pixel_position_cali_x, pixel_position_cali_y = calibration_object_coordinates.to_pixel(image_series.wcs)

    if correlation_method == 'astropy':
        #   Create coordinates object
        object_coordinates = SkyCoord.from_pixel(
            pixel_position_obj_x,
            pixel_position_obj_y,
            image_series.wcs,
        )

        #   Find matches between the datasets
        index_obj_instrument, index_obj_literature, separation, _ = matching.search_around_sky(
            object_coordinates,
            calibration_object_coordinates,
            separation_limit,
        )
        separation_arcsec = np.asarray(separation.arcsec, dtype=float)

        index_obj_instrument, separation_arcsec, index_obj_literature = (
            utilities.clear_duplicates(
                index_obj_instrument,
                separation_arcsec,
                index_obj_literature,
            )
        )
        index_obj_literature, separation_arcsec, index_obj_instrument = (
            utilities.clear_duplicates(
                index_obj_literature,
                separation_arcsec,
                index_obj_instrument,
            )
        )

        n_identified_literature_objs = len(index_obj_literature)

    elif correlation_method == 'own':
        #   Max. number of objects
        n_obj_max = np.max(len(pixel_position_obj_x), len(pixel_position_cali_x))

        #   Define and fill new arrays
        pixel_position_all_x = np.zeros((n_obj_max, 2))
        pixel_position_all_y = np.zeros((n_obj_max, 2))
        pixel_position_all_x[0:len(pixel_position_obj_x), 0] = pixel_position_obj_x
        pixel_position_all_x[0:len(pixel_position_cali_x), 1] = pixel_position_cali_x
        pixel_position_all_y[0:len(pixel_position_obj_y), 0] = pixel_position_obj_y
        pixel_position_all_y[0:len(pixel_position_cali_y), 1] = pixel_position_cali_y

        #   Correlate calibration stars with stars on the image
        correlated_indexes, rejected_images, n_identified_literature_objs, rejected_obj = correlation_own(
            pixel_position_all_x,
            pixel_position_all_y,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
        )
        index_obj_instrument = correlated_indexes[0]
        index_obj_literature = correlated_indexes[1]

    else:
        raise ValueError(
            f'The correlation method needs to either "astropy" or "own". Got '
            f'{correlation_method} instead.'
        )

    if n_identified_literature_objs == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo calibration star was identified "
            f"-> EXIT {style.Bcolors.ENDC}"
        )
    if n_identified_literature_objs == 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}\nOnly one calibration star was identified\n"
            "Unfortunately, that is not enough at the moment\n"
            f"-> EXIT {style.Bcolors.ENDC}"
        )

    #   Limit calibration table to common objects
    calibration_tbl_sort = calibration_tbl[index_obj_literature]

    terminal_output.print_to_terminal(
        f"{len(calibration_tbl_sort)} calibration stars have been matched to"
        f" observed stars",
        indent=indent,
        style_name='OKBLUE',
    )

    #   Add calibration star indexes to the calibration table
    calibration_tbl_sort['index_instrument'] = index_obj_instrument

    #   Limit number of calibration stars to the 100 brightest
    if len(calibration_tbl_sort) > 100:
        #   Sort calibration table
        magnitude_name = None
        for column_name in column_names:
            if 'mag' in column_name:
                magnitude_name = column_name
                break

        calibration_tbl_sort.sort(column_names[magnitude_name])

        #   Limit to brightest 100 objects
        calibration_tbl_sort = calibration_tbl_sort[0:100]
        index_obj_instrument = calibration_tbl_sort['index_instrument']

        terminal_output.print_to_terminal(
            f"Number of calibration stars limited to 100 brightest objects"
            f" in filter {magnitude_name[3:]}",
            indent=indent,
            style_name='OKBLUE',
        )

    #   Plots
    #
    #   Make new arrays based on the correlation results
    pixel_position_common_objs_x = pixel_position_obj_x[list(index_obj_instrument)]
    pixel_position_common_objs_y = pixel_position_obj_y[list(index_obj_instrument)]
    index_common_new = np.arange(len(calibration_tbl_sort))

    #   Add pixel positions and object ids to the calibration table
    calibration_tbl_sort.add_columns(
        [np.intc(index_common_new), pixel_position_common_objs_x, pixel_position_common_objs_y],
        names=['id', 'x_centroid', 'y_centroid']
    )

    calibration_tbl.add_columns(
        [np.arange(0, len(pixel_position_cali_y)), pixel_position_cali_x, pixel_position_cali_y],
        names=['id', 'x_centroid', 'y_centroid']
    )

    #   Plot star map with calibration stars
    for filter_ in filter_list:
        if 'mag' + filter_ in column_names:
            p = mp.Process(
                target=plots.starmap,
                args=(
                    image_series.out_path.name,
                    image_series.image_list[image_series.reference_image_index].get_data(),
                    filter_,
                    calibration_tbl,
                ),
                kwargs={
                    'tbl_2': calibration_tbl_sort,
                    'label': 'downloaded calibration stars',
                    'label_2': 'matched calibration stars',
                    'rts': 'calibration',
                    # 'name_object': image_series.object_name,
                    'wcs_image': image_series.wcs,
                    'use_wcs_projection': use_wcs_projection_for_star_maps,
                    'file_type': file_type_plots,
                }
            )
            p.start()

    terminal_output.print_to_terminal('')

    if index_obj_instrument is Column:
        return calibration_tbl_sort, index_obj_instrument.value
    else:
        return calibration_tbl_sort, index_obj_instrument


def select_calibration_objects(
    observation: 'analyze.Observation',
    filter_list: list[str] | set[str],
    reference_image_series_id: int = 0,
    correlation_method: str = 'astropy',
    separation_limit: u.Quantity = 2. * u.arcsec,
    max_pixel_between_objects: int = 3,
    ooi_correlation_strategy: int = 1,
    file_type_plots: str = 'pdf',
    indent: int = 1
    ):
    """
    Select observations that have a counterpart identified in the calibration data

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        List with filter identifiers.

    reference_image_series_id
        ID of the reference image
        Default is ``0``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    ooi_correlation_strategy
        Option for the srcor correlation function
        Default is ``1``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    #   Get calibration data
    calibration_parameters = observation.calib_parameters

    calibration_tbl = calibration_parameters.calib_tbl
    column_names = calibration_parameters.column_names
    ra_unit_calibration = calibration_parameters.ra_unit
    dec_unit_calibration = calibration_parameters.dec_unit

    #   Convert coordinates of the calibration stars to SkyCoord object
    calibration_object_coordinates = SkyCoord(
        calibration_tbl[column_names['ra']].data,
        calibration_tbl[column_names['dec']].data,
        unit=(ra_unit_calibration, dec_unit_calibration),
        frame="icrs"
    )

    #   Correlate with calibration stars
    #   -> assumes that calibration stars are already cleared of any reference objects
    #      or variable stars
    calibration_tbl, index_obj_instrument = correlate_with_calibration_objects(
        list(observation.image_series_dict.values())[0],
        calibration_object_coordinates,
        calibration_tbl,
        filter_list,
        column_names,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
        max_pixel_between_objects=max_pixel_between_objects,
        ooi_correlation_strategy=ooi_correlation_strategy,
        file_type_plots=file_type_plots,
        indent=indent+1,
    )

    observation.calib_parameters.calib_tbl = calibration_tbl
    observation.calib_parameters.ids_calibration_objects = index_obj_instrument
