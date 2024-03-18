############################################################################
#                               Libraries                                  #
############################################################################

import requests

import numpy as np

from uncertainties import unumpy

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord, matching

import multiprocessing as mp

from .. import style, calibration_data, terminal_output

from . import correlate, plot


############################################################################
#                           Routines & definitions                         #
############################################################################


class CalibParameters:
    def __init__(self, index, column_names, calib_tbl):
        self.inds = index
        self.column_names = column_names
        self.calib_tbl = calib_tbl


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
    catalog_properties_dict = calibration_data.catalog_properties_dict[catalog_identifier]

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

    #   TODO: Remove comment lines when changes have proven successful.
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


def load_calib(image, filter_list, calibration_method='APASS', magnitude_range=(0., 18.5),
               vizier_dict=None, path_calibration_file=None, ra_unit=u.deg, indent=1):
    """
        Load calibration information

        Parameters
        ----------
        image                   : `image.class` or `image.ensemble`
            Class object with all image specific properties

        filter_list             : `list` with `strings`
            Filter list

        calibration_method      : `string`, optional
            Calibration method
            Default is ``APASS``.

        magnitude_range         : `tuple` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        vizier_dict             : `dictionary` or None, optional
            Vizier identifiers of catalogs that can be used for calibration.
            Default is ``None``.

        path_calibration_file   : `string`, optional
            Path to the calibration file
            Default is ``None``.

        ra_unit                 : `astropy.unit`, optional
            Right ascension unit
            Default is ``u.deg``.

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
        vizier_dict = calibration_data.vizier_dict

    #   Read calibration table
    if calibration_method == 'vsp':
        #   Load calibration info from AAVSO for variable stars
        calib_tbl, column_names = get_comp_stars_aavso(
            image.coord,
            filters=filter_list,
            field_of_view=1.5 * image.fov,
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
            image.coord,
            filters=filter_list,
            field_of_view=1.5 * image.fov,
            magnitude_range=magnitude_range,
            indent=indent + 1,
        )
        ra_unit = u.hourangle

    elif calibration_method in vizier_dict.keys():
        #   Load info from Vizier
        calib_tbl, column_names, ra_unit = get_vizier_catalog(
            filter_list,
            image.coord,
            image.fov,
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


def get_observed_magnitudes_of_calibration_stars(image, magnitude_array, img_container):
    """
        Sort and rearrange input numpy array with extracted magnitude
        data, such that the returned numpy array contains the extracted
        magnitudes of the calibration stars

        Parameters
        ----------
        image                           : `image class`
            Image class object

        magnitude_array                 : `numpy.ndarray` or `unumpy.uarray`
            Array with image magnitudes

        img_container                   : `image.container`
            Container object with image ensemble objects for each filter

        Returns
        -------
        magnitudes_calibration_observed : `numpy.ndarray` or `unumpy.uarray`
            Rearrange array with magnitudes
    """
    #   Get calibration data
    index_calibration_stars = img_container.CalibParameters.inds
    col_names = img_container.CalibParameters.column_names

    #   Convert index array of the calibration stars to a list
    ind_list = list(index_calibration_stars)

    #   Calculate number of calibration stars
    count_cali = len(ind_list)

    #   Get required type for magnitude array. If ``True`` an unumpy array
    #   will be used. Otherwise, a structured numpy array will be created.
    unc = getattr(img_container, 'unc', True)

    ###
    #   Sort and add magnitudes
    #
    #   unumpy.uarray
    if unc:
        #   Check if we have calibration data for the current filter/image
        if f'mag{getattr(image, "filt", "?")}' in col_names:
            #   Sort
            magnitudes_calibration_observed = magnitude_array[ind_list]
        else:
            magnitudes_calibration_observed = unumpy.uarray(
                np.zeros(count_cali),
                np.zeros(count_cali)
            )

    #   numpy structured array
    else:
        #   Define array for the magnitudes of the calibration stars
        magnitudes_calibration_observed = np.zeros(
            count_cali,
            dtype=[('mag', 'f8'), ('err', 'f8')],
        )

        #   Check if we have calibration data for the current filter/image
        if f'mag{getattr(image, "filt", "?")}' in col_names:
            #   Sort
            magnitudes_calibration_observed['mag'] = magnitude_array['mag'][ind_list]
            magnitudes_calibration_observed['err'] = magnitude_array['err'][ind_list]

    #   Add array with magnitudes to the image
    return magnitudes_calibration_observed


def derive_calibration(img_container, filter_list, calibration_method='APASS',
                       max_pixel_between_objects=3., own_correlation_option=1,
                       vizier_dict=None, path_calibration_file=None,
                       id_object=None, ra_unit=u.deg, dec_unit=u.deg,
                       magnitude_range=(0., 18.5), coordinates_obj_to_rm=None,
                       correlation_method='astropy',
                       separation_limit=2. * u.arcsec, reference_filter=None,
                       region_to_select_calibration_stars=None, indent=1):
    """
        Determine calibration information, find suitable calibration stars
        and determine calibration factors

        Parameters
        ----------
        img_container                       : `image.container`
            Container object with image ensemble objects for each filter

        filter_list                         : `list` of `string`
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

        vizier_dict                         : `dictionary` or `None`, optional
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``None``.

        path_calibration_file               : `string`, optional
            Path to the calibration file
            Default is ``None``.

        id_object                           : `integer`, optional
            ID of the object
            Default is ``None``.

        ra_unit                             : `astropy.unit`, optional
            Right ascension unit
            Default is ``u.deg``.

        dec_unit                            : `astropy.unit`, optional
            Declination unit
            Default is ``u.deg``.

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

        indent                              : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    """
    terminal_output.print_to_terminal(
        f"Get calibration star magnitudes (filter: {tuple(filter_list)})",
        indent=indent,
    )

    #   Get one of image ensembles to extract wcs, positions, ect.
    if reference_filter is None:
        reference_filter = filter_list[0]
    img_ensemble = img_container.ensembles[reference_filter]

    #   Get wcs
    wcs = img_ensemble.wcs

    #   Load calibration data
    #   TODO: Check this routine - It gets and returns ra_unit
    calib_tbl, column_names, ra_unit = load_calib(
        img_ensemble,
        filter_list,
        calibration_method=calibration_method,
        magnitude_range=magnitude_range,
        vizier_dict=vizier_dict,
        path_calibration_file=path_calibration_file,
        indent=indent,
        ra_unit=ra_unit,
    )

    #   Convert coordinates of the calibration stars to SkyCoord object
    calib_coordinates = SkyCoord(
        calib_tbl[column_names['ra']].data,
        calib_tbl[column_names['dec']].data,
        unit=(ra_unit, dec_unit),
        frame="icrs"
    )

    #   Get PixelRegion of the field of view and convert it SkyRegion
    region_pix = img_ensemble.region_pix
    region_sky = region_pix.to_sky(wcs)

    #   Remove calibration stars that are not within the field of view
    mask = region_sky.contains(calib_coordinates, wcs)
    calib_coordinates = calib_coordinates[mask]
    calib_tbl = calib_tbl[mask]

    #   Remove calibration stars that are not within the selection region
    if region_to_select_calibration_stars:
        if hasattr(region_to_select_calibration_stars, 'to_sky'):
            region_to_select_calibration_stars = region_to_select_calibration_stars.to_sky(wcs)
        mask = region_to_select_calibration_stars.contains(calib_coordinates, wcs)
        calib_coordinates = calib_coordinates[mask]
        calib_tbl = calib_tbl[mask]

    #   Remove a specific star from the loaded calibration stars
    if coordinates_obj_to_rm is not None:
        mask = calib_coordinates.separation(coordinates_obj_to_rm) < 1 * u.arcsec
        mask = np.invert(mask)
        calib_coordinates = calib_coordinates[mask]

    #   Calculate object positions in pixel coordinates
    pixel_position_cali_x, pixel_position_cali_y = calib_coordinates.to_pixel(wcs)

    #   Remove nans that are caused by missing ra/dec entries
    pixel_position_cali_x = pixel_position_cali_x[~np.isnan(pixel_position_cali_x)]
    pixel_position_cali_y = pixel_position_cali_y[~np.isnan(pixel_position_cali_y)]
    calib_tbl = calib_tbl[~np.isnan(pixel_position_cali_y)]

    #   X & Y pixel positions
    pixel_position_obj_x = img_ensemble.image_list[0].photometry['x_fit']
    pixel_position_obj_y = img_ensemble.image_list[0].photometry['y_fit']

    if correlation_method == 'astropy':
        #   Create coordinates object
        object_coordinates = SkyCoord.from_pixel(
            pixel_position_obj_x,
            pixel_position_obj_y,
            wcs,
        )

        #   Find matches between the datasets
        index_obj_instrument, index_obj_literature, _, _ = matching.search_around_sky(
            object_coordinates,
            calib_coordinates,
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
    calib_tbl_sort = calib_tbl[index_obj_literature]

    ###
    #   Plots
    #
    #   Make new arrays based on the correlation results
    pixel_position_common_objs_x = pixel_position_obj_x[list(index_obj_instrument)]
    pixel_position_common_objs_y = pixel_position_obj_y[list(index_obj_instrument)]
    index_common_new = np.arange(n_identified_literature_objs)

    #   Add pixel positions and object ids to the calibration table
    calib_tbl_sort.add_columns(
        [np.intc(index_common_new), pixel_position_common_objs_x, pixel_position_common_objs_y],
        names=['id', 'xcentroid', 'ycentroid']
    )

    calib_tbl.add_columns(
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
                target=plot.starmap,
                args=(
                    img_ensemble.outpath.name,
                    #   Replace with reference image in the future
                    img_ensemble.image_list[0].get_data(),
                    filter_,
                    calib_tbl,
                ),
                kwargs={
                    'tbl_2': calib_tbl_sort,
                    'label': 'downloaded calibration stars',
                    'label_2': 'matched calibration stars',
                    'rts': rts,
                    'name_obj': img_ensemble.objname,
                    'wcs': img_ensemble.wcs,
                }
            )
            p.start()

    #   Add calibration data to image container
    img_container.CalibParameters = CalibParameters(
        index_obj_instrument,
        column_names,
        calib_tbl_sort,
    )


def magnitude_array_from_calibration_table(img_container, filter_list):
    """
        Arrange the literature values in a numpy array or uncertainty array.

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        filter_list             : `list` of `string`
            Filter names

        Returns
        -------
        literature_magnitudes   : `numpy.ndarray` or `uncertainties.unumpy.uarray`
            Array with literature magnitudes
    """
    #   Number of filter
    n_filter = len(filter_list)

    #   Get calibration table
    calib_tbl = img_container.CalibParameters.calib_tbl
    calib_column_names = img_container.CalibParameters.column_names

    n_calib_stars = len(calib_tbl)

    #   unumpy.array or default numpy.ndarray
    unc = getattr(img_container, 'unc', True)
    if unc:
        #   Create uncertainties array with the literature magnitudes
        literature_magnitudes = unumpy.uarray(
            np.zeros((n_filter, n_calib_stars)),
            np.zeros((n_filter, n_calib_stars))
        )

        #
        for z, filter_ in enumerate(filter_list):
            if f'mag{filter_}' in calib_column_names:
                #   Check if errors for the calibration magnitudes exist
                if f'err{filter_}' in calib_column_names:
                    err = np.array(
                        calib_tbl[calib_column_names[f'err{filter_}']]
                    )

                    #   Check if errors are nice floats
                    if err.dtype in (float, np.float32, np.float64):
                        err_value = err
                    else:
                        err_value = 0.
                else:
                    err_value = 0.

                #   Extract magnitudes
                literature_magnitudes[z] = unumpy.uarray(
                    calib_tbl[calib_column_names[f'mag{filter_}']],
                    err_value
                )

    #   Default numpy.ndarray
    else:
        #   Define new arrays
        literature_magnitudes = np.zeros(n_filter, dtype=[('mag', 'f8', n_calib_stars),
                                                          ('err', 'f8', n_calib_stars),
                                                          ('qua', 'U1', n_calib_stars),
                                                          ]
                                         )

        #
        for z, filter_ in enumerate(filter_list):
            if f'mag{filter_}' in calib_column_names:
                #   Extract magnitudes
                col_mags = np.array(
                    calib_tbl[calib_column_names[f'mag{filter_}']]
                )
                literature_magnitudes['mag'][z] = col_mags

                #   Check if errors for the calibration magnitudes exist
                if f'err{filter_}' in calib_column_names:
                    err_value = np.array(
                        calib_tbl[calib_column_names[f'err{filter_}']]
                    )
                else:
                    err_value = np.zeros(n_calib_stars)

                #   Check if errors are nice floats
                if err_value.dtype in (np.float, np.float32, np.float64):
                    literature_magnitudes['err'][z] = err_value

                #   Add quality flag, if it exists
                if f'qua{filter_}' in calib_column_names:
                    quality_value = np.array(
                        calib_tbl[calib_column_names[f'qua{filter_}']]
                    )
                    literature_magnitudes['qua'][z] = quality_value

    return literature_magnitudes
