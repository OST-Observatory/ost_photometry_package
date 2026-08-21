"""Inter-filter and calibration-related correlation."""

from __future__ import annotations

import typing
import warnings

import numpy as np

if typing.TYPE_CHECKING:
    from .. import analyze

import astropy.units as u
from astropy import wcs
from astropy.coordinates import SkyCoord

from ... import terminal_output
from ... import utilities as base_utilities
from .. import utilities
from ..ooi_ids import set_ooi_correlated_ids_from_filter
from ..warnings_types import OstPhotometryAnalyzeWarning
from .core import correlate_datasets, correlation_own
from .intra import correlate_image_series_images
from .ooi import (
    identify_object_of_interest_in_dataset,
    verify_objects_of_interest_global_correlated_ids,
)
from .protection import resolve_calibration_object_ids


def assign_global_correlated_object_ids(
    observation: analyze.Observation,
    filter_list: list[str],
) -> None:
    """
    Set photometry ``id`` to 0 .. N-1 in correlated row order on all images.

    After intra- and inter-filter correlation, row ``k`` refers to the same
    physical object across filters (and across images within each filter).
    The table ``id`` column is aligned to that index so downstream code
    (e.g. differential calibration on epoch-native vstack tables)
    can match objects by ``id``.

    Objects of interest store the same index as ``correlated_id`` after
    alignment. Per-filter ``id_in_image_series`` is only the pre-alignment
    row map.

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
        observation: analyze.Observation, filter_list: list[str] | set[str],
        max_pixel_between_objects: int = 3,
        ooi_correlation_strategy: int = 1, cross_identification_limit: int = 1,
        reference_image_series_id: int = 0,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_ooi: bool = True,
        protect_calibration_objects: bool = False,
        protected_object_ids: list[int] | None = None,
        calibration_object_ids: list[int] | None = None,
        correlation_method: str = 'astropy',
        separation_limit: u.quantity.Quantity = 2. * u.arcsec,
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

    protect_ooi
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    protect_calibration_objects
        If ``True``, include catalog-resolved calibration-star IDs in the
        protected set.
        Default is ``False``.

    protected_object_ids
        Explicit row indices on the reference filter to keep during correlation,
        independent of object type.
        Default is ``None``.

    calibration_object_ids
        Calibration-star row indices (merged when
        ``protect_calibration_objects`` is ``True``).
        Default is ``None``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

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
        positions per image vs. stored ``correlated_id``).
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

    for series in image_series_dict.values():
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

    #   Build protected-object set for inter-filter correlation
    from .protection import merge_protected_object_ids

    reference_obj_ids = observation.get_ids_object_of_interest(
        filter_=reference_filter,
    )

    merged_protected_ids = merge_protected_object_ids(
        protected_object_ids=protected_object_ids,
        reference_object_ids=reference_obj_ids,
        calibration_object_ids=calibration_object_ids,
        protect_ooi=protect_ooi,
        protect_calibration_objects=protect_calibration_objects,
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
        protected_object_ids=merged_protected_ids,
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

        set_ooi_correlated_ids_from_filter(objects_of_interest, reference_filter)

    terminal_output.print_to_terminal('')

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


def inter_filter_correlation_separations_for_images(
    observation: analyze.Observation,
    filter_list: list[str] | set[str],
    images_by_filter: dict[str, base_utilities.Image],
    *,
    reference_filter: str | None = None,
) -> tuple[np.ndarray, str, list[str]]:
    """
    On-sky separations (arcsec) between a specific image per filter.

    Rows are assumed aligned (same length / correlated object order) as after
    :func:`correlate_image_series`.
    """
    fl = list(filter_list)
    if len(fl) < 2:
        return np.array([]), "", []
    ref = reference_filter or fl[0]
    if ref not in images_by_filter or ref not in observation.image_series_dict:
        return np.array([]), ref, []
    ref_img = images_by_filter[ref]
    sref = observation.image_series_dict[ref]
    if ref_img.photometry is None or len(ref_img.photometry) == 0:
        return np.array([]), ref, []
    if sref.wcs is None:
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
        img = images_by_filter.get(f)
        if img is None or f not in observation.image_series_dict:
            continue
        ser = observation.image_series_dict[f]
        if img.photometry is None or len(img.photometry) != n or ser.wcs is None:
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


def inter_filter_correlation_separations_arcsec(
    observation: analyze.Observation,
    filter_list: list[str] | set[str],
    reference_filter_index: int = 0,
) -> tuple[np.ndarray, str, list[str]]:
    """
    Separations for the **reference image** of each filter series.

    For light curves with many exposures, this is only one pair (the series
    ``reference_image_index`` frames). Use
    :func:`inter_filter_correlation_separations_for_images` for other pairs.
    """
    fl = list(filter_list)
    if len(fl) < 2:
        return np.array([]), "", []
    if reference_filter_index < 0 or reference_filter_index >= len(fl):
        reference_filter_index = 0
    ref = fl[reference_filter_index]
    isd = observation.image_series_dict
    images: dict[str, base_utilities.Image] = {}
    for f in fl:
        if f not in isd:
            continue
        ser = isd[f]
        images[f] = ser.image_list[ser.reference_image_index]
    return inter_filter_correlation_separations_for_images(
        observation,
        fl,
        images,
        reference_filter=ref,
    )


def inter_filter_pair_image_label(images_by_filter: dict[str, base_utilities.Image]) -> str:
    """Short label like ``B0_V3`` from filter + ``image_id``."""
    parts: list[str] = []
    for f in sorted(images_by_filter.keys()):
        img = images_by_filter[f]
        iid = getattr(img, "image_id", None)
        parts.append(f"{f}{iid}" if iid is not None else str(f))
    return "_".join(parts)


def inter_filter_pair_title_suffix(images_by_filter: dict[str, base_utilities.Image]) -> str:
    """Human-readable pair description for plot titles."""
    bits: list[str] = []
    for f, img in images_by_filter.items():
        iid = getattr(img, "image_id", "?")
        name = getattr(img, "filename", None) or ""
        if name:
            bits.append(f"{f}: id={iid} ({name})")
        else:
            bits.append(f"{f}: id={iid}")
    return "; ".join(bits)

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


def correlate_preserve_calibration_objects(
        image_series: analyze.ImageSeries, filter_list: list[str],
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
    calib_stars_ids, calib_x_pixel_positions, calib_y_pixel_positions = (
        resolve_calibration_object_ids(
            image_series,
            filter_list,
            calibration_source=calibration_source,
            calibration_catalog_mag_range=calibration_catalog_mag_range,
            vizier_dict=vizier_dict,
            path_calibration_file=calib_file,
            reference_image_index=reference_image_index,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            verbose=verbose,
        )
    )

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
        plots_for_all_images=not plot_only_reference_starmap,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )

