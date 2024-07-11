############################################################################
#                               Libraries                                  #
############################################################################

import requests

import numpy as np

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord, matching
from astropy import uncertainty as unc

import multiprocessing as mp

from .. import style, calibration_parameters, terminal_output

from . import correlate, plots

import typing
if typing.TYPE_CHECKING:
    from . import analyze


############################################################################
#                           Routines & definitions                         #
############################################################################


class CalibParameters:
    def __init__(self, index, column_names, calib_tbl, **kwargs):
        self.inds = index
        self.column_names = column_names
        self.calib_tbl = calib_tbl

        #   Add additional keywords
        self.__dict__.update(kwargs)

        #   Check for right ascension and declination
        ra_unit = kwargs.get('ra_unit', None)
        dec_unit = kwargs.get('dec_unit', None)
        if ra_unit is not None:
            self.ra_unit = ra_unit
        else:
            self.ra_unit = u.deg
        if dec_unit is not None:
            self.dec_unit = dec_unit
        else:
            self.dec_unit = u.deg


def get_comp_stars_aavso(coordinates_sky, filters=None, field_of_view=18.5,
                         magnitude_range=(0., 18.5), indent=2):
    """
        Download calibration info for variable stars from AAVSO

        Parameters
        ----------
        coordinates_sky  : `astropy.coordinates.SkyCoord`
            Coordinates of the field of field_of_view

        filters          : `list` of `string` or `None`, optional
            Filter names
            Default is ``None``.

        field_of_view   : `float`, optional
            Field of view in arc minutes
            Default is ``18.5``.

        magnitude_range : `tuple` of `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        indent          : `integer`, optional
            Indentation for the console output
            Default is ``2``.

        Returns
        -------
        tbl             : `astropy.table.Table`
            Table with calibration information

        column_dict     : `dictionary` - 'string':`string`
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


def get_comp_stars_simbad(coordinates_sky, filters=None, field_of_view=18.5,
                          magnitude_range=(0., 18.5), indent=2):
    """
        Download calibration info from Simbad

        Parameters
        ----------
        coordinates_sky  : `astropy.coordinates.SkyCoord`
            Coordinates of the field of field_of_view

        filters          : `list` of `string` or `None`, optional
            Filter names
            Default is ``None``.

        field_of_view   : `float`, optional
            Field of view in arc minutes
            Default is ``18.5``.

        magnitude_range : `tuple` of `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        indent          : `integer`, optional
            Indentation for the console output
            Default is ``2``.

        Returns
        -------
        tbl             : `astropy.table.Table`
            Table with calibration information

        column_dict     : `dictionary` - 'string':`string`
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


def get_vizier_catalog(filter_list, coordinates_image_center, field_of_view,
                       catalog_identifier, magnitude_range=(0., 18.5),
                       indent=2):
    """
        Download catalog with calibration info from Vizier

        Parameters
        ----------
        filter_list                 : `list` of `string`
            Filter names

        coordinates_image_center    : `astropy.coordinates.SkyCoord`
            Coordinates of the field of field_of_view

        field_of_view               : `float`
            Field of view in arc minutes

        catalog_identifier          : `string`
            Catalog identifier

        magnitude_range             : `tuple` of `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        indent                      : `integer`, optional
            Indentation for the console output
            Default is ``2``.

        Returns
        -------
        tbl                         : `astropy.table.Table`
            Table with calibration information

        column_dict                 : `dictionary` - 'string':`string`
            Dictionary with column names vs default names

        ra_unit
    """
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
        terminal_output.print_to_terminal(
            "No calibration data available",
            indent=indent + 1,
            style_name='WARNING',
        )
        return Table(), {}

    result = table_list[0]

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
        terminal_output.print_to_terminal(
            "Calibration issue: Threshold magnitude not recognized",
            indent=indent + 1,
            style_name='ERROR',
        )
        raise RuntimeError

    mask = (result[preferred_filer] <= magnitude_range[1]) & (result[preferred_filer] >= magnitude_range[0])
    result = result[mask]

    terminal_output.print_to_terminal(
        f"{len(result)} calibration objects remaining after magnitude "
        "filtering",
        indent=indent,
    )

    #   Define dict with column names
    column_dict = {
        'ra': catalog_properties_dict['ra_dec_columns'][0],
        'dec': catalog_properties_dict['ra_dec_columns'][1]
    }
    
    for filter_ in filter_list:
        if f'{filter_}mag' in result.colnames:
            column_dict[f'mag{filter_}'] = f'{filter_}mag'

            #   Check if catalog contains magnitude errors
            if f'e_{filter_}mag' in result.colnames:
                column_dict[f'err{filter_}'] = f'e_{filter_}mag'
        else:
            terminal_output.print_to_terminal(
                f"No calibration data for {filter_} band",
                indent=indent + 1,
                style_name='WARNING',
            )

    return result, column_dict, catalog_properties_dict['ra_unit']


def read_votable_simbad(path_calibration_file, filter_list, magnitude_range=(0., 18.5),
                        indent=2):
    """
        Read table in VO format already downloaded from Simbad

        Parameters
        ----------
        path_calibration_file   : `string`
            Path to the calibration file

        filter_list             : `list` of `string`
            Filter names

        magnitude_range         : `tuple` of `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        indent                  : `integer`, optional
            Indentation for the console output
            Default is ``2``.

        Returns
        -------
        tbl                     : `astropy.table.Table`
            Table with calibration information

        column_dict             : `dictionary` - 'string':`string`
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
        image_like_object: 'analyze.ImageSeries | analyze.Image', filter_list, calibration_method='APASS',
        magnitude_range=(0., 18.5), vizier_dict: dict[str, str] | None = None,
        path_calibration_file=None, indent=1):
    """
        Load calibration information

        Parameters
        ----------
        image_like_object
            Class object with all image specific properties

        filter_list             : `list` with `strings`
            Filter list

        calibration_method      : `string`, optional
            Calibration method
            Default is ``APASS``.

        magnitude_range         : `tuple` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        vizier_dict
            Vizier identifiers of catalogs that can be used for calibration.
            Default is ``None``.

        path_calibration_file   : `string`, optional
            Path to the calibration file
            Default is ``None``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.

        Returns
        -------
        calib_tbl       : `astropy.table.Table`
            Astropy table with the calibration data

        column_names    : `dictionary`
            Column names versus the internal default names

        ra_unit         : `astropy.unit`
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
        calibration_stars_ids: np.ndarray) -> unc.core.NdarrayDistribution | u.quantity.Quantity:
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
        observation: 'analyze.Observation', filter_list, calibration_method='APASS',
        max_pixel_between_objects=3., own_correlation_option=1,
        vizier_dict: dict[str, str] | None = None, path_calibration_file=None, id_object=None,
        magnitude_range=(0., 18.5), coordinates_obj_to_rm=None,
        correlation_method='astropy', separation_limit: u.quantity.Quantity = 2. * u.arcsec,
        reference_filter=None, region_to_select_calibration_stars=None,
        correlate_with_observed_objects=True, reference_image_id=0, indent=1):
    """
    Find suitable calibration stars

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list                         : `list` or set` of `string`
        Filter list

    calibration_method                  : `string`, optional
        Calibration method
        Default is ``APASS``.

    max_pixel_between_objects           : `float`, optional
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option              : `integer`, optional
        Option for the srcor correlation function
        Default is ``1``.

    vizier_dict
        Dictionary with identifiers of the Vizier catalogs with valid
        calibration data
        Default is ``None``.

    path_calibration_file               : `string`, optional
        Path to the calibration file
        Default is ``None``.

    id_object                           : `integer`, optional
        ID of the object
        Default is ``None``.

    magnitude_range                     : `tuple` or `float`, optional
        Magnitude range
        Default is ``(0.,18.5)``.

    coordinates_obj_to_rm               : `astropy.coordinates.SkyCoord`, optional
        Coordinates of an object that should not be used for calibrating
        the data.
        Default is ``None``.

    correlation_method                  : `string`, optional
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit                    : `astropy.units`, optional
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    reference_filter                    : `string` or `None`, optional
        Name of the reference filter
        Default is ``None`.

    region_to_select_calibration_stars  : `regions.RectanglePixelRegion`, optional
        Region in which to select calibration stars. This is a useful
        feature in instances where not the entire field of view can be
        utilized for calibration purposes.
        Default is ``None``.

    correlate_with_observed_objects     : `boolean`, optional
        If ``True`` the downloaded calibration objects will be correlated
        with the observed objects to get a valid set of calibration objects

    reference_image_id                  : `integer`, optional
        ID of the reference image
        Default is ``0``.

    indent                              : `integer`, optional
        Indentation for the console output lines
        Default is ``1``.
    """
    terminal_output.print_to_terminal(
        f"Get calibration star magnitudes (filter: {tuple(filter_list)})",
        indent=indent,
    )

    #   Get one of image series to extract wcs, positions, ect.
    if reference_filter is None:
        reference_filter = filter_list[0]
    image_series = observation.image_series_dict[reference_filter]

    #   Get wcs
    wcs = image_series.wcs

    #   Load calibration data
    #   TODO: Check this routine - It gets and returns ra_unit
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
        # mask = calibration_object_coordinates.separation(coordinates_obj_to_rm) < 1 * u.arcsec
        # mask = np.invert(mask)
        # calibration_object_coordinates = calibration_object_coordinates[mask]
        # calibration_tbl = calibration_tbl[mask]

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

    #   TODO: Add a filter for known variable objects and non stellar objects

    terminal_output.print_to_terminal(
        f"{len(calibration_tbl)} calibration stars remain after the cleanup.",
        indent=indent + 2,
        style_name='OKBLUE',
    )

    if correlate_with_observed_objects:
        calibration_tbl, index_obj_instrument = correlate_with_calibration_objects(
            image_series,
            calibration_object_coordinates,
            calibration_tbl,
            filter_list,
            column_names,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            id_object=id_object,
            reference_image_id=reference_image_id,
            indent=indent,
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


#   TODO: Move to correlate
def correlate_with_calibration_objects(
        image_series: 'analyze.ImageSeries', calibration_object_coordinates,
        calibration_tbl,
        filter_list, column_names, correlation_method='astropy',
        separation_limit=2. * u.arcsec, max_pixel_between_objects=3.,
        own_correlation_option=1, id_object=None, reference_image_id=0,
        indent=1):
    """
    Correlate observed objects with calibration stars

    Parameters
    ----------
    image_series
        Class with all images of a specific image series

    calibration_object_coordinates      : `astropy.coordinates.SkyCoord`
        Coordinates of the calibration objects

    calibration_tbl                     : `astropy.table.Table`
        Table with calibration data

    filter_list                         : `list` or set` of `string`
        Filter list

    column_names                        : `dictionary`
        Actual names of the columns in calibration_tbl versus
        the internal default names

    correlation_method                  : `string`, optional
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit                    : `astropy.units`, optional
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    max_pixel_between_objects           : `float`, optional
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option              : `integer`, optional
        Option for the srcor correlation function
        Default is ``1``.

    id_object                           : `integer`, optional
        ID of the object
        Default is ``None``.

    reference_image_id                  : `integer`, optional
        ID of the reference image
        Default is ``0``.

    indent                              : `integer`, optional
        Indentation for the console output lines
        Default is ``1``.

    Returns
    -------
    calibration_tbl_sort                : `astropy.table.Table`
        Sorted table with calibration data

    index_obj_instrument                : `numpy.ndarray`
        Index of the observed stars that correspond to the calibration stars
    """
    #   Pixel positions of the observed object
    pixel_position_obj_x = image_series.image_list[reference_image_id].photometry['x_fit']
    pixel_position_obj_y = image_series.image_list[reference_image_id].photometry['y_fit']

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
        index_obj_instrument, index_obj_literature, _, _ = matching.search_around_sky(
            object_coordinates,
            calibration_object_coordinates,
            separation_limit,
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
        correlated_indexes, rejected_images, n_identified_literature_objs, rejected_obj = correlate.correlation_own(
            pixel_position_all_x,
            pixel_position_all_y,
            max_pixel_between_objects=max_pixel_between_objects,
            option=own_correlation_option,
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

    ###
    #   Plots
    #
    #   Make new arrays based on the correlation results
    pixel_position_common_objs_x = pixel_position_obj_x[list(index_obj_instrument)]
    pixel_position_common_objs_y = pixel_position_obj_y[list(index_obj_instrument)]
    index_common_new = np.arange(n_identified_literature_objs)

    #   Add pixel positions and object ids to the calibration table
    calibration_tbl_sort.add_columns(
        [np.intc(index_common_new), pixel_position_common_objs_x, pixel_position_common_objs_y],
        names=['id', 'xcentroid', 'ycentroid']
    )

    calibration_tbl.add_columns(
        [np.arange(0, len(pixel_position_cali_y)), pixel_position_cali_x, pixel_position_cali_y],
        names=['id', 'xcentroid', 'ycentroid']
    )

    #   Plot star map with calibration stars
    if id_object is not None:
        rts = f'calibration - object: {id_object}'
    else:
        rts = 'calibration'
    for filter_ in filter_list:
        if 'mag' + filter_ in column_names:
            p = mp.Process(
                target=plots.starmap,
                args=(
                    image_series.out_path.name,
                    #   Replace with reference image in the future
                    image_series.image_list[0].get_data(),
                    filter_,
                    calibration_tbl,
                ),
                kwargs={
                    'tbl_2': calibration_tbl_sort,
                    'label': 'downloaded calibration stars',
                    'label_2': 'matched calibration stars',
                    'rts': rts,
                    # 'name_object': image_series.object_name,
                    'wcs_image': image_series.wcs,
                }
            )
            p.start()

    terminal_output.print_to_terminal(
        f"{len(calibration_tbl_sort)} calibration stars have been matched to"
        f" observed stars",
        indent=indent + 2,
        style_name='OKBLUE',
    )

    return calibration_tbl_sort, index_obj_instrument


def distribution_from_calibration_table(
        calibration_parameters: CalibParameters, filter_list: list[str],
        distribution_samples: int = 1000) -> list[u.quantity.Quantity]:
    """
        Arrange the literature values in a numpy array or uncertainty array.

        Parameters
        ----------
        calibration_parameters
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
    calib_column_names = calibration_parameters.column_names

    #   Get calibration table
    calibration_data_table = calibration_parameters.calib_tbl

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
