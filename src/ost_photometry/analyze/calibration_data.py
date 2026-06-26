############################################################################
#                               Libraries                                  #
############################################################################

import numpy as np

from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import uncertainty as unc

from regions import RectanglePixelRegion

from . import correlate
# All catalog download / normalization lives in calibration_sources; this module
# only adapts to the legacy (Table, column_dict, ra_unit) contract for derive_calibration.
from .calibration_sources import (
    fetch_standard_calibration_catalog,
    get_vizier_catalog,
    standard_catalog_to_legacy,
)
from .. import calibration_parameters, terminal_output

import typing
if typing.TYPE_CHECKING:
    from . import analyze


############################################################################
#                           Routines & definitions                         #
############################################################################


class CalibParameters:
    def __init__(
        self,
        index: np.ndarray | None,
        column_names: dict[str, str],
        calib_tbl: Table,
        *,
        ra_unit: u.core.Unit | None = None,
        dec_unit: u.core.Unit | None = None,
    ):
        self.ids_calibration_objects = index
        self.column_names = column_names
        self.calib_tbl = calib_tbl
        self.ra_unit = ra_unit if ra_unit is not None else u.deg
        self.dec_unit = dec_unit if dec_unit is not None else u.deg


def load_calibration_data_table(
        image_like_object: 'analyze.ImageSeries | analyze.Image',
        filter_list: list[str], calibration_source: str = 'APASS',
        calibration_catalog_mag_range: tuple[float, float] = (0., 18.5),
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None, indent: int = 1
        ) -> tuple[Table, dict[str, str], u.core.Unit]:
    """
    Load calibration information

    Parameters
    ----------
    image_like_object
        Class object with all image specific properties

    filter_list
        Filter list

    calibration_source
        Catalog / lookup key (e.g. ``APASS``, ``simbad``, ``vsp``, or a ``vizier_dict`` key).
        Default is ``APASS``.

    calibration_catalog_mag_range
        Inclusive magnitude range (faint/bright limits) for catalog stars.
        Default is ``(0.,18.5)``.

    vizier_dict
        Vizier identifiers of catalogs that can be used for calibration.
        Default is ``None``.

    path_calibration_file
        Path to the calibration file
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``1``.

    Returns
    -------
    calib_tbl
        Astropy table with the calibration data

    column_names
        Column names versus the internal default names

    ra_unit
        Returns also the right ascension unit in case it changed
    """
    if vizier_dict is None:
        vizier_dict = calibration_parameters.vizier_dict

    center = image_like_object.coordinates_image_center
    fov_x = image_like_object.field_of_view_x

    if calibration_source == 'vsp':
        field_of_view_arcmin = 1.5 * fov_x
    elif calibration_source == 'simbad':
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

    # Standard schema -> flat magB/magV names and self-consistent column_dict
    calib_tbl, column_names, ra_unit = standard_catalog_to_legacy(
        std_tbl, filter_list
    )

    #   Remove masked columns from calibration table, since those could cause
    #   problems during calibration
    for filter_ in filter_list:
        if f'mag{filter_}' in column_names:
            #   Remove objects without magnitudes from the calibration list
            arr = calib_tbl[column_names[f'mag{filter_}']]
            if hasattr(arr, 'mask'):
                ind_rm = np.where(arr.mask)
                calib_tbl.remove_rows(ind_rm)

            #   Remove objects without errors from the calibration list
            arr = calib_tbl[column_names[f'err{filter_}']]
            if hasattr(arr, 'mask'):
                ind_rm = np.where(arr.mask)
                calib_tbl.remove_rows(ind_rm)

    if not calib_tbl:
        raise RuntimeError(
            f"\nNo calibration star with {filter_list} magnitudes found. -> EXIT"
        )
    terminal_output.print_to_terminal(
        f"Of these {len(calib_tbl)} are useful",
        indent=indent + 2,
        style_name='OKBLUE',
    )

    return calib_tbl, column_names, ra_unit


def observed_magnitude_of_calibration_stars(
        magnitude_distribution: unc.core.NdarrayDistribution | u.quantity.Quantity,
        calibration_stars_ids: np.ndarray
        ) -> unc.core.NdarrayDistribution | u.quantity.Quantity:
    """
    Sort and rearrange the distribution of extracted magnitudes so that
    the returned distribution contains the extracted magnitudes of the
    calibration stars.

    Parameters
    ----------
    magnitude_distribution
        Distribution with image magnitudes

    calibration_stars_ids
        IDs of the stars for which calibration data is available

    Returns
    -------
    distribution_calibration_observed
        Rearrange distribution
    """
    #   Sort magnitudes
    distribution_calibration_observed = magnitude_distribution[
        calibration_stars_ids
    ]

    return distribution_calibration_observed


#   TODO: Rename to 'downloading_calibration_data' after 'correlate_calibrate'
#         of 'Observation' in analysis.py changed the order of correlation and
#         downloading of calibration data.
def derive_calibration(
        observation: 'analyze.Observation', filter_list: list[str],
        calibration_source: str = 'APASS', max_pixel_between_objects: int = 3,
        ooi_correlation_strategy: int = 1,
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None,
        calibration_catalog_mag_range: tuple[float, float] = (0., 18.5),
        coordinates_obj_to_rm: SkyCoord | None = None,
        correlation_method: str = 'astropy',
        separation_limit: u.quantity.Quantity = 2. * u.arcsec,
        reference_filter: str | None = None,
        region_to_select_calibration_stars: RectanglePixelRegion | None = None,
        correlate_with_observed_objects: bool = True,
        file_type_plots: str = 'pdf',
        use_wcs_projection_for_star_maps: bool = True,
        indent: int = 1
    ) -> None:
    """
    Find suitable calibration stars

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        Filter list

    calibration_source
        Catalog / lookup key (e.g. ``APASS``, ``simbad``, ``vsp``, or a ``vizier_dict`` key).
        Default is ``APASS``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    ooi_correlation_strategy
        Option for the srcor correlation function
        Default is ``1``.

    vizier_dict
        Dictionary with identifiers of the Vizier catalogs with valid
        calibration data
        Default is ``None``.

    path_calibration_file
        Path to the calibration file
        Default is ``None``.

    calibration_catalog_mag_range
        Inclusive magnitude range for catalog stars.
        Default is ``(0.,18.5)``.

    coordinates_obj_to_rm
        Coordinates of an object that should not be used for calibrating
        the data.
        Default is ``None``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    reference_filter
        Name of the reference filter
        Default is ``None`.

    region_to_select_calibration_stars
        Region in which to select calibration stars. This is a useful
        feature in instances where not the entire field of view can be
        utilized for calibration purposes.
        Default is ``None``.

    correlate_with_observed_objects
        If ``True`` the downloaded calibration objects will be correlated
        with the observed objects to get a valid set of calibration objects

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    terminal_output.print_to_terminal(
        f"Get calibration star magnitudes - Filter: {tuple(filter_list)}",
        indent=indent,
    )

    #   Get one of image series to extract wcs, positions, ect.
    if reference_filter is None:
        reference_filter = filter_list[0]
    image_series = observation.image_series_dict[reference_filter]

    #   Get wcs
    wcs = image_series.wcs

    #   Load calibration data
    calibration_tbl, column_names, ra_unit_calibration = load_calibration_data_table(
        image_series,
        filter_list,
        calibration_source=calibration_source,
        calibration_catalog_mag_range=calibration_catalog_mag_range,
        vizier_dict=vizier_dict,
        path_calibration_file=path_calibration_file,
        indent=indent,
    )

    #   Convert coordinates of the calibration stars to SkyCoord object
    calibration_object_coordinates = SkyCoord(
        calibration_tbl[column_names['ra']].data,
        calibration_tbl[column_names['dec']].data,
        unit=(ra_unit_calibration, u.deg),
        frame="icrs"
    )

    #   Get PixelRegion of the field of view and convert it SkyRegion
    fov_pixel_region = image_series.fov_pixel_region
    region_sky = fov_pixel_region.to_sky(wcs)

    #   Remove calibration stars that are not within the field of view
    mask = region_sky.contains(calibration_object_coordinates, wcs)
    calibration_object_coordinates = calibration_object_coordinates[mask]
    calibration_tbl = calibration_tbl[mask]

    #   Remove calibration stars that are not within the selection region
    if region_to_select_calibration_stars:
        if hasattr(region_to_select_calibration_stars, 'to_sky'):
            region_to_select_calibration_stars = region_to_select_calibration_stars.to_sky(wcs)
        mask = region_to_select_calibration_stars.contains(calibration_object_coordinates, wcs)
        calibration_object_coordinates = calibration_object_coordinates[mask]
        calibration_tbl = calibration_tbl[mask]

    #   Remove a specific star from the loaded calibration stars
    if coordinates_obj_to_rm is not None:
        mask = np.ones(len(calibration_object_coordinates), dtype=bool)
        for coordinate_object in coordinates_obj_to_rm:
            separation = calibration_object_coordinates.separation(coordinate_object)

            #   Calculate mask of all object closer than ``radius``
            mask = mask & np.invert(separation < 1 * u.arcsec)

        calibration_object_coordinates = calibration_object_coordinates[mask]
        calibration_tbl = calibration_tbl[mask]

    #   Calculate object positions in pixel coordinates
    pixel_position_cali_x, pixel_position_cali_y = calibration_object_coordinates.to_pixel(wcs)

    #   Remove nans that are caused by missing ra/dec entries
    # pixel_position_cali_x = pixel_position_cali_x[~np.isnan(pixel_position_cali_x)]
    pixel_position_cali_y = pixel_position_cali_y[~np.isnan(pixel_position_cali_y)]
    calibration_tbl = calibration_tbl[~np.isnan(pixel_position_cali_y)]

    #   VSX (Vizier): drop calibration stars that coincide with known variables
    #   (same shared get_vizier_catalog as catalog fetch; empty filter_list = positions only).
    variable_stars_tbl, column_dict_variable, ra_unit_variable = get_vizier_catalog(
        [],
        image_series.coordinates_image_center,
        field_of_view_arcmin=image_series.field_of_view_x,
        catalog_identifier='B/vsx/vsx',
        cleanup_magnitudes=False,
        print_infos=False,
    )
    variable_stars_coordinates = SkyCoord(
        variable_stars_tbl[column_dict_variable['ra']].data,
        variable_stars_tbl[column_dict_variable['dec']].data,
        unit=(ra_unit_variable, u.deg),
        frame="icrs"
    )

    mask = np.ones(len(calibration_object_coordinates), dtype=bool)
    for coordinate_object in variable_stars_coordinates:
        separation = calibration_object_coordinates.separation(coordinate_object)

        #   Calculate mask of all object closer than ``radius``
        mask = mask & np.invert(separation < 1 * u.arcsec)

    calibration_object_coordinates = calibration_object_coordinates[mask]
    calibration_tbl = calibration_tbl[mask]

    terminal_output.print_to_terminal(
        f"{len(calibration_tbl)} calibration stars remain after cleanup",
        indent=indent + 2,
        style_name='GOOD',
    )

    #   TODO: Remove the following after changing the order of correlation and
    #          download of calibration data in 'correlate_calibrate' of
    #          'observation' in analysis.py
    if correlate_with_observed_objects and len(column_names) > 2:
        calibration_tbl, index_obj_instrument = correlate.correlate_with_calibration_objects(
            image_series,
            calibration_object_coordinates,
            calibration_tbl,
            filter_list,
            column_names,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            indent=indent + 1,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )
    else:
        index_obj_instrument = None

    #   Add calibration data to observation container
    observation.calib_parameters = CalibParameters(
        index_obj_instrument,
        # None,
        column_names,
        calibration_tbl,
        ra_unit=ra_unit_calibration,
    )


def distribution_from_calibration_table(
        parameters_calibration: CalibParameters, filter_list: list[str],
        distribution_samples: int = 1000) -> list[u.quantity.Quantity]:
    """
        Arrange the literature values in a numpy array or uncertainty array.

        Parameters
        ----------
        parameters_calibration
            Class instance with calibration data

        filter_list
            Filter names

        distribution_samples
            Number of samples used for distributions
            Default is `1000`.

        Returns
        -------
        distribution_list
            Normal distribution representing literature magnitudes
    """
    #   Get column names
    calib_column_names = parameters_calibration.column_names

    #   Get calibration table
    calibration_data_table = parameters_calibration.calib_tbl

    distribution_list: list[u.quantity.Quantity] = []
    for filter_ in filter_list:
        calibration_magnitudes = calibration_data_table[
            calib_column_names[f'mag{filter_}']
        ]
        calibration_magnitudes_err = calibration_data_table[
            calib_column_names[f'err{filter_}']
        ]

        literature_magnitudes_distribution = unc.normal(
            calibration_magnitudes.value << u.mag,
            std=calibration_magnitudes_err.value << u.mag,
            n_samples=distribution_samples,
        )
        #   The '.distribution' below is currently necessary, because astropy
        #   QuantityDistribution cannot be prickled/serialized
        #   TODO: Check if this workaround is still necessary
        distribution_list.append(
            literature_magnitudes_distribution.distribution
        )

    return distribution_list
