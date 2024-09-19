############################################################################
#                               Libraries                                  #
############################################################################

import requests

import numpy as np

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import uncertainty as unc

from regions import RectanglePixelRegion

from . import correlate
from .. import style, calibration_parameters, terminal_output

import typing
if typing.TYPE_CHECKING:
    from . import analyze


############################################################################
#                           Routines & definitions                         #
############################################################################


class CalibParameters:
    def __init__(
            self, index: np.ndarray, column_names: dict[str, str],
            calib_tbl: Table, **kwargs):
        self.ids_calibration_objects = index
        self.column_names = column_names
        self.calib_tbl = calib_tbl

        #   Add additional keywords
        self.__dict__.update(kwargs)

        #   Check for right ascension and declination
        ra_unit: u.core.Unit | None = kwargs.get('ra_unit', None)
        dec_unit: u.core.Unit | None = kwargs.get('dec_unit', None)
        if ra_unit is not None:
            self.ra_unit = ra_unit
        else:
            self.ra_unit = u.deg
        if dec_unit is not None:
            self.dec_unit = dec_unit
        else:
            self.dec_unit = u.deg


def get_comp_stars_aavso(
        coordinates_sky: SkyCoord, filters: list[str] | None = None,
        field_of_view: float = 18.5,
        magnitude_range: tuple[float, float] = (0., 18.5), indent: int = 2
        ) -> tuple[Table, dict[str, str]]:
    """
    Download calibration info for variable stars from AAVSO

    Parameters
    ----------
    coordinates_sky
        Coordinates of the field of field_of_view

    filters
        Filter names
        Default is ``None``.

    field_of_view
        Field of view in arc minutes
        Default is ``18.5``.

    magnitude_range
        Magnitude range
        Default is ``(0.,18.5)``.

    indent
        Indentation for the console output
        Default is ``2``.

    Returns
    -------
    tbl
        Table with calibration information

    column_dict
        Dictionary with column names vs default names
    """
    terminal_output.print_to_terminal(
        "Downloading calibration data from www.aavso.org",
        indent=indent,
    )

    #   Sanitize filter list
    if filters is None:
        filters = ['B', 'V']

    #   Prepare url
    ra = coordinates_sky.ra.degree
    dec = coordinates_sky.dec.degree
    vsp_template = 'https://www.aavso.org/apps/vsp/api/chart/"\
        "?format=json&fov={}&maglimit={}&ra={}&dec={}&special=std_field'

    #   Download data
    r = requests.get(vsp_template.format(field_of_view, magnitude_range[1], ra, dec))

    #   Check status code
    status_code = r.status_code
    if status_code != 200:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nThe request of the AAVSO website was not "
            "successful.\nProbably no calibration stars found.\n -> EXIT"
            f"{style.Bcolors.ENDC}"
        )
    else:
        #   Prepare arrays and lists
        obj_id = []
        obj_ra = []
        obj_dec = []
        n_obj = len(r.json()['photometry'])
        n_filter = len(filters)
        mags = np.zeros((n_obj, n_filter))
        errs = np.zeros((n_obj, n_filter))

        #   Loop over stars
        for i, star in enumerate(r.json()['photometry']):
            #   Fill lists with ID, ra, & dec
            obj_id.append(star['auid'])
            obj_ra.append(star['ra'])
            obj_dec.append(star['dec'])
            #   Loop over required filters
            for j, filter_ in enumerate(filters):
                #   Loop over filter from AAVSO
                for band in star['bands']:
                    #   Check if AAVSO filter is the required filter
                    if band['band'][0] == filter_:
                        #   Fill magnitude and uncertainty arrays
                        mags[i, j] = band['mag']
                        errs[i, j] = band['error']

        #   Initialize dictionary with column names
        column_dict = {'id': 'id', 'ra': 'ra', 'dec': 'dec'}
        #   Initialize table
        tbl = Table(
            names=['id', 'ra', 'dec', ],
            data=[obj_id, obj_ra, obj_dec, ]
        )

        #   Complete table & dictionary
        for j, filter_ in enumerate(filters):
            tbl.add_columns([
                mags[:, j],
                errs[:, j],
            ],
                names=[
                    'mag' + filter_,
                    'err' + filter_,
                ]
            )
            column_dict['mag' + filter_] = 'mag' + filter_
            column_dict['err' + filter_] = 'err' + filter_

        #   Filter magnitudes: lower threshold
        mask = tbl['magV'] >= magnitude_range[0]
        tbl = tbl[mask]

        terminal_output.print_to_terminal(
            f"{len(tbl)} calibration objects remaining after magnitude "
            "filtering",
            indent=indent,
        )
    
        return tbl, column_dict


def get_comp_stars_simbad(
        coordinates_sky: SkyCoord, filters: list[str] | None = None,
        field_of_view: float = 18.5,
        magnitude_range: tuple[float, float] = (0., 18.5), indent: int = 2
        ) -> tuple[Table, dict[str, str]]:
    """
    Download calibration info from Simbad

    Parameters
    ----------
    coordinates_sky
        Coordinates of the field of field_of_view

    filters
        Filter names
        Default is ``None``.

    field_of_view
        Field of view in arc minutes
        Default is ``18.5``.

    magnitude_range
        Magnitude range
        Default is ``(0.,18.5)``.

    indent
        Indentation for the console output
        Default is ``2``.

    Returns
    -------
    tbl
        Table with calibration information

    column_dict
        Dictionary with column names vs default names
    """
    terminal_output.print_to_terminal(
        "Downloading calibration data from Simbad",
        indent=indent,
    )

    #   Sanitize filter list
    if filters is None:
        filters = ['B', 'V']

    #   Initialize Simbad instance
    my_simbad = Simbad(
        # ROW_LIMIT=1e6,
    )

    for filter_ in filters:
        my_simbad.add_votable_fields(f'flux({filter_})')
        my_simbad.add_votable_fields(f'flux_error({filter_})')

    simbad_table = my_simbad.query_region(
        coordinates_sky, 
        radius=field_of_view * 0.66 * u.arcmin,
    )
    terminal_output.print_to_terminal(
        f"Found {len(simbad_table)} with the SIMBAD query",
        indent=indent,
    )

    #   Stop here if Table is empty
    if not simbad_table:
        terminal_output.print_to_terminal(
            "No calibration data available",
            indent=indent + 1,
            style_name='WARNING',
        )
        return Table(), {}

    #   Rename columns to default names
    for filter_ in filters:
        simbad_table.rename_column(f'FLUX_{filter_}', f'{filter_}mag')
        simbad_table.rename_column(f'FLUX_ERROR_{filter_}', f'e_{filter_}mag')
    
    #   Restrict magnitudes to requested range
    if 'Vmag' in simbad_table.keys():
        preferred_filer = 'Vmag'
    elif 'Rmag' in simbad_table.keys():
        preferred_filer = 'Rmag'
    elif 'Bmag' in simbad_table.keys():
        preferred_filer = 'Bmag'
    elif 'Imag' in simbad_table.keys():
        preferred_filer = 'Imag'
    elif 'Umag' in simbad_table.keys():
        preferred_filer = 'Umag'
    else:
        #   This should never happen
        terminal_output.print_to_terminal(
            "Calibration issue: Threshold magnitude not recognized",
            indent=indent + 1,
            style_name='ERROR',
        )
        raise RuntimeError
    
    mask = (simbad_table[preferred_filer] <= magnitude_range[1]) & (simbad_table[preferred_filer] >= magnitude_range[0])
    simbad_table = simbad_table[mask]

    terminal_output.print_to_terminal(
        f"{len(simbad_table)} calibration objects remaining after magnitude "
        "filtering",
        indent=indent,
    )
    
    #   Define dict with column names
    column_dict = {'ra': 'RA', 'dec': 'DEC'}
    
    for filter_ in filters:
        if f'{filter_}mag' in simbad_table.colnames:
            column_dict[f'mag{filter_}'] = f'{filter_}mag'

            #   Check if catalog contains magnitude errors
            if f'e_{filter_}mag' in simbad_table.colnames:
                column_dict[f'err{filter_}'] = f'e_{filter_}mag'
        else:
            terminal_output.print_to_terminal(
                f"No calibration data for {filter_} band",
                indent=indent + 1,
                style_name='WARNING',
            )

    return simbad_table, column_dict


def get_vizier_catalog(
        filter_list: list[str], coordinates_image_center: SkyCoord,
        field_of_view: float, catalog_identifier: str,
        magnitude_range: tuple[float, float] = (0., 18.5),
        cleanup_magnitudes: bool = True, print_infos: bool = True,
        indent: int = 2) -> tuple[Table, dict[str, str], str]:
    """
    Download catalog with calibration info from Vizier

    Parameters
    ----------
    filter_list
        Filter names

    coordinates_image_center
        Coordinates of the field of field_of_view

    field_of_view
        Field of view in arc minutes

    catalog_identifier
        Catalog identifier

    magnitude_range
        Magnitude range
        Default is ``(0.,18.5)``.

    cleanup_magnitudes
        If ``True`` a first magnitude cleanup: restrict magnitude range, rename
        columns, etc. will be performed
        Default is ``True``.

    print_infos
        If ``True'', status information is printed to the terminal.
        Default is ``True``.

    indent
        Indentation for the console output
        Default is ``2``.

    Returns
    -------
    tbl
        Table with calibration information

    column_dict
        Dictionary with column names vs default names

    ra_unit
    """
    if print_infos:
        terminal_output.print_to_terminal(
            f"Downloading calibration data from Vizier: {catalog_identifier}",
            indent=indent,
        )

    #   Get catalog specific columns
    catalog_properties_dict = calibration_parameters.catalog_properties_dict[catalog_identifier]

    #   Combine columns
    columns = (catalog_properties_dict['ra_dec_columns']
               + catalog_properties_dict['columns']
               + catalog_properties_dict['err_columns']
               )

    #   Define astroquery instance
    v = Vizier(
        columns=columns,
        row_limit=1e6,
        catalog=catalog_identifier,
    )

    #   Get data from the corresponding catalog
    table_list = v.query_region(
        coordinates_image_center,
        radius=field_of_view * u.arcmin,
    )

    #   Chose first table
    if not table_list:
        if print_infos:
            terminal_output.print_to_terminal(
                "No calibration data available",
                indent=indent + 1,
                style_name='WARNING',
            )
        return Table(), {}, ''

    result = table_list[0]

    #   Define dict with column names
    column_dict = {
        'ra': catalog_properties_dict['ra_dec_columns'][0],
        'dec': catalog_properties_dict['ra_dec_columns'][1]
    }

    if cleanup_magnitudes:
        #   Rename columns to default names
        if 'column_rename' in catalog_properties_dict:
            for element in catalog_properties_dict['column_rename']:
                result.rename_column(element[0], element[1])

        #   Calculate B, U, etc. if only B-V, U-B, etc are given
        if 'magnitude_arithmetic' in catalog_properties_dict:
            for element in catalog_properties_dict['magnitude_arithmetic']:
                result[element[0]] = result[element[1]] + result[element[2]]

        #   Restrict magnitudes to requested range
        if 'Vmag' in result.keys():
            preferred_filer = 'Vmag'
        elif 'Rmag' in result.keys():
            preferred_filer = 'Rmag'
        elif 'Bmag' in result.keys():
            preferred_filer = 'Bmag'
        elif 'Imag' in result.keys():
            preferred_filer = 'Imag'
        elif 'Umag' in result.keys():
            preferred_filer = 'Umag'
        else:
            #   This should never happen
            if print_infos:
                terminal_output.print_to_terminal(
                    "Calibration issue: Threshold magnitude not recognized",
                    indent=indent + 1,
                    style_name='ERROR',
                )
            raise RuntimeError

        mask = (result[preferred_filer] <= magnitude_range[1]) & (result[preferred_filer] >= magnitude_range[0])
        result = result[mask]

        if print_infos:
            terminal_output.print_to_terminal(
                f"{len(result)} calibration objects remaining after magnitude "
                "filtering",
                indent=indent,
            )

        for filter_ in filter_list:
            if f'{filter_}mag' in result.colnames:
                column_dict[f'mag{filter_}'] = f'{filter_}mag'

                #   Check if catalog contains magnitude errors
                if f'e_{filter_}mag' in result.colnames:
                    column_dict[f'err{filter_}'] = f'e_{filter_}mag'
            else:
                if print_infos:
                    terminal_output.print_to_terminal(
                        f"No calibration data for {filter_} band",
                        indent=indent + 1,
                        style_name='WARNING',
                    )

    return result, column_dict, catalog_properties_dict['ra_unit']


def read_votable_simbad(
        path_calibration_file: str, filter_list: list[str],
        magnitude_range: tuple[float, float] = (0., 18.5),
        indent: int = 2) -> tuple[Table, dict[str, str]]:
    """
    Read table in VO format already downloaded from Simbad

    Parameters
    ----------
    path_calibration_file
        Path to the calibration file

    filter_list
        Filter names

    magnitude_range
        Magnitude range
        Default is ``(0.,18.5)``.

    indent
        Indentation for the console output
        Default is ``2``.

    Returns
    -------
    tbl
        Table with calibration information

    column_dict
        Dictionary with column names vs default names
    """
    terminal_output.print_to_terminal(
        f"Read calibration data from a VO table: {path_calibration_file}",
        indent=indent,
    )

    #   Read table
    calib_tbl = Table.read(path_calibration_file, format='votable')

    #   Filter magnitudes: lower and upper threshold
    mask = calib_tbl['FLUX_V'] >= magnitude_range[0]
    mask = mask * calib_tbl['FLUX_V'] <= magnitude_range[1]
    calib_tbl = calib_tbl[mask]

    #   Define dict with column names
    column_dict = {'ra': 'RA_d', 'dec': 'DEC_d'}

    for filter_ in filter_list:
        if 'FLUX_' + filter_ in calib_tbl.colnames:
            #   Clean calibration table based on variability and multiplicity flags
            index_bad_objects = np.where(calib_tbl['FLUX_MULT_' + filter_].mask)
            calib_tbl.remove_rows(index_bad_objects)
            index_bad_objects = np.nonzero(calib_tbl['FLUX_MULT_' + filter_])
            calib_tbl.remove_rows(index_bad_objects)
            index_bad_objects = np.where(calib_tbl['FLUX_VAR_' + filter_].mask)
            calib_tbl.remove_rows(index_bad_objects)
            index_bad_objects = np.nonzero(calib_tbl['FLUX_VAR_' + filter_])
            calib_tbl.remove_rows(index_bad_objects)

            if not calib_tbl:
                raise Exception(
                    f"{style.Bcolors.FAIL}\nAll calibration stars in the "
                    f"{filter_} removed because of variability and multiplicity "
                    f"citeria. -> EXIT {style.Bcolors.ENDC}"
                )

            column_dict['mag' + filter_] = 'FLUX_' + filter_
            column_dict['err' + filter_] = 'FLUX_ERROR_' + filter_
            column_dict['qua' + filter_] = 'FLUX_QUAL_' + filter_
        else:
            terminal_output.print_to_terminal(
                f"No calibration data for {filter_} band",
                indent=indent + 1,
                style_name='WARNING',
            )

    return calib_tbl, column_dict


def load_calibration_data_table(
        image_like_object: 'analyze.ImageSeries | analyze.Image',
        filter_list: list[str], calibration_method: str = 'APASS',
        magnitude_range: tuple[float, float] = (0., 18.5),
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

    calibration_method
        Calibration method
        Default is ``APASS``.

    magnitude_range
        Magnitude range
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
    #   Get identifiers of catalogs if no has been provided
    if vizier_dict is None:
        vizier_dict = calibration_parameters.vizier_dict

    #   Read calibration table
    if calibration_method == 'vsp':
        #   Load calibration info from AAVSO for variable stars
        calib_tbl, column_names = get_comp_stars_aavso(
            image_like_object.coordinates_image_center,
            filters=filter_list,
            field_of_view=1.5 * image_like_object.field_of_view_x,
            magnitude_range=magnitude_range,
            indent=indent + 1,
        )
        ra_unit = u.hourangle

    elif calibration_method == 'simbad_vot' and path_calibration_file is not None:
        #   Load info from data file in VO format downloaded from Simbad
        calib_tbl, column_names = read_votable_simbad(
            path_calibration_file,
            filter_list,
            magnitude_range=magnitude_range,
            indent=indent + 1,
        )
        ra_unit = u.hourangle

    elif calibration_method == 'simbad':
        calib_tbl, column_names = get_comp_stars_simbad(
            image_like_object.coordinates_image_center,
            filters=filter_list,
            field_of_view=1.5 * image_like_object.field_of_view_x,
            magnitude_range=magnitude_range,
            indent=indent + 1,
        )
        ra_unit = u.hourangle

    elif calibration_method in vizier_dict.keys():
        #   Load info from Vizier
        calib_tbl, column_names, ra_unit = get_vizier_catalog(
            filter_list,
            image_like_object.coordinates_image_center,
            image_like_object.field_of_view_x,
            vizier_dict[calibration_method],
            magnitude_range=magnitude_range,
            indent=indent + 1,
        )
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nCalibration method not recognized\n"
            "Check variable: calib_method and vizier_dict "
            f"-> EXIT {style.Bcolors.ENDC}"
        )

    terminal_output.print_to_terminal(
        f"{len(calib_tbl)} calibration stars downloaded",
        indent=indent + 2,
        style_name='OKBLUE',
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
            f"{style.Bcolors.FAIL} \nNo calibration star with {filter_list} "
            f"magnitudes found. -> EXIT {style.Bcolors.ENDC}"
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


def derive_calibration(
        observation: 'analyze.Observation', filter_list: list[str],
        calibration_method: str = 'APASS', max_pixel_between_objects: int = 3,
        own_correlation_option: int = 1,
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None,
        magnitude_range: tuple[float, float] = (0., 18.5),
        coordinates_obj_to_rm: SkyCoord | None = None,
        correlation_method: str = 'astropy',
        separation_limit: u.quantity.Quantity = 2. * u.arcsec,
        reference_filter: str | None = None,
        region_to_select_calibration_stars: RectanglePixelRegion | None = None,
        correlate_with_observed_objects: bool = True,
        file_type_plots: str = 'pdf',
        use_wcs_projection_for_star_maps: bool = True,
        indent: int = 1) -> None:
    """
    Find suitable calibration stars

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        Filter list

    calibration_method
        Calibration method
        Default is ``APASS``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    vizier_dict
        Dictionary with identifiers of the Vizier catalogs with valid
        calibration data
        Default is ``None``.

    path_calibration_file
        Path to the calibration file
        Default is ``None``.

    magnitude_range
        Magnitude range
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
        calibration_method=calibration_method,
        magnitude_range=magnitude_range,
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
    #   TODO: Rewrite such that multiple objects can be removed
    if coordinates_obj_to_rm is not None:
        # from astropy.coordinates import matching
        # _, index_obj_to_rm, _, _ = matching.search_around_sky(
        #     coordinates_obj_to_rm,
        #     calibration_object_coordinates,
        #     separation_limit,
        # )
        # calibration_tbl.remove_rows(index_obj_to_rm)

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

    #   Filter for known variable objects
    variable_stars_tbl, column_dict_variable, ra_unit_variable = get_vizier_catalog(
        [],
        image_series.coordinates_image_center,
        image_series.field_of_view_x,
        'B/vsx/vsx',
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
        style_name='OKBLUE',
    )

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
            own_correlation_option=own_correlation_option,
            indent=indent + 1,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )
    else:
        index_obj_instrument = None

    #   Add calibration data to observation container
    observation.calib_parameters = CalibParameters(
        index_obj_instrument,
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
            calibration_magnitudes.value * u.mag,
            std=calibration_magnitudes_err.value * u.mag,
            n_samples=distribution_samples,
        )
        #   The '.distribution' below is currently necessary, because astropy
        #   QuantityDistribution cannot be prickled/serialized
        #   TODO: Check if this workaround is still necessary
        distribution_list.append(
            literature_magnitudes_distribution.distribution
        )

    return distribution_list
