############################################################################
#                               Libraries                                  #
############################################################################

import requests

import numpy as np

from uncertainties import unumpy

from astroquery.vizier import Vizier

from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord, matching

import multiprocessing as mp

from .. import style, calibration_data, terminal_output

from . import correlate, plot


############################################################################
#                           Routines & definitions                         #
############################################################################


#   TODO: Remove this class?
class CalibParameters:
    def __init__(self, inds, column_names, calib_tbl):
        self.inds = inds
        self.column_names = column_names
        self.calib_tbl = calib_tbl


def get_comp_stars(coord, filters=['B', 'V'], field_of_view=18.5,
                   mag_range=(0., 18.5), indent=2):
    """
        Download calibration info for variable stars from AAVSO

        Parameters
        ----------
        coord           : `astropy.coordinates.SkyCoord`
            Coordinates of the field of field_of_view

        filters         : `list` of `string`, optional
            Filter names
            Default is ``['B','V']``.

        field_of_view   : `float`, optional
            Field of view in arc minutes
            Default is ``18.5``.

        mag_range       : `tupel` of `float`, optional
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
    terminal_output.print_terminal(
        indent=indent,
        string="Downloading calibration data from www.aavso.org",
    )

    #   Prepare url
    ra = coord.ra.degree
    dec = coord.dec.degree
    vsp_template = 'https://www.aavso.org/apps/vsp/api/chart/"\
        "?format=json&fov={}&maglimit={}&ra={}&dec={}&special=std_field'

    #   Download data
    r = requests.get(vsp_template.format(field_of_view, mag_range[1], ra, dec))

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
        nfilt = len(filters)
        mags = np.zeros((n_obj, nfilt))
        errs = np.zeros((n_obj, nfilt))

        #   Loop over stars
        for i, star in enumerate(r.json()['photometry']):
            #   Fill lists with ID, ra, & dec
            obj_id.append(star['auid'])
            obj_ra.append(star['ra'])
            obj_dec.append(star['dec'])
            #   Loop over required filters
            for j, filt in enumerate(filters):
                #   Loop over filter from AAVSO
                for band in star['bands']:
                    #   Check if AAVSO filter is the required filter
                    if band['band'][0] == filt:
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
        for j, filt in enumerate(filters):
            tbl.add_columns([
                mags[:, j],
                errs[:, j],
            ],
                names=[
                    'mag' + filt,
                    'err' + filt,
                ]
            )
            column_dict['mag' + filt] = 'mag' + filt
            column_dict['err' + filt] = 'err' + filt

        #   Filter magnitudes: lower threshold
        mask = tbl['magV'] >= mag_range[0]
        tbl = tbl[mask]

        return tbl, column_dict


def get_catalog(bands, center, fov, catalog, mag_range=(0., 18.5),
                indent=2):
    """
        Download catalog with calibration info from Vizier

        Parameters
        ----------
        bands           : `list` of `string`
            Filter names

        center          : `astropy.coordinates.SkyCoord`
            Coordinates of the field of field_of_view

        fov             : `float`
            Field of view in arc minutes

        catalog         : `string`
            Catalog identifier

        mag_range       : `tupel` of `float`, optional
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
    terminal_output.print_terminal(
        catalog,
        indent=indent,
        string="Downloading calibration data from Vizier: {}",
    )

    #   Define and combine columns
    if catalog == 'II/168/ubvmeans':
        radec_col = ['_RA', '_DE']
    else:
        radec_col = ['RAJ2000', 'DEJ2000']
    default_col = {
        'columns': ["Bmag", "Vmag", "rmag", "imag"],
        'err_columns': ["e_Bmag", "e_Vmag", "e_rmag", "e_imag"],
    }
    catalog_col = {
        'I/329': default_col,
        'I/322A': default_col,
        'II/336/apass9': {
            'columns': ["Bmag", "Vmag", "r'mag", "i'mag"],
            'err_columns': ["e_Bmag", "e_Vmag", "e_r'mag", "e_i'mag"],
        },
        'I/297': {'columns': ["Bmag", "Vmag", "Rmag"], 'err_columns': []},
        'I/305': {
            'columns': ["Umag", "Bmag", "Vmag"],
            'err_columns': ["e_Umag", "e_Bmag", "e_Vmag"],
        },
        'II/168/ubvmeans': {
            'columns': ["Vmag", "B-V", "U-B"],
            'err_columns': ["e_Vmag", "e_B-V", "e_U-B"],
        },
        'II/272/gspc24': {
            'columns': ["Bmag", "Vmag", "Rmag"],
            'err_columns': ["e_Bmag", "e_Vmag", "e_Rmag"],
        },
        'II/339/uvotssc1': {
            'columns': ["U-AB", "B-AB", "V-AB"],
            'err_columns': [],
        },
        'II/370/xmmom5s': {
            'columns': ["UmAB", "BmAB", "VmAB"],
            'err_columns': ["e_UmAB", "e_BmAB", "e_VmAB"],
        },
        'J/MNRAS/443/725/catalog': {
            'columns': ["Vmag", "Rmag", "Imag"],
            'err_columns': ["e_Vmag", "e_Rmag", "e_Imag"],
        },
        'I/284/out': {
            'columns': ["B1mag", "R1mag", "Imag"],
            'err_columns': [],
        },
    }
    columns = radec_col + catalog_col[catalog]['columns']
    columns = columns + catalog_col[catalog]['err_columns']

    #   Define astroquery instance
    v = Vizier(
        columns=columns,
        row_limit=1e6,
        catalog=catalog,
    )

    #   Get data from the corresponding catalog
    tablelist = v.query_region(
        center,
        radius=fov * u.arcmin,
    )

    #   Chose first table
    if not tablelist:
        terminal_output.print_terminal(
            indent=indent + 1,
            string="No calibration data available",
            style_name='WARNING',
        )
        return Table(), {}

    result = tablelist[0]

    #   Rename columns to default names
    if catalog == 'II/370/xmmom5s':
        result.rename_column("UmAB", "Umag")
        result.rename_column("BmAB", "Bmag")
        result.rename_column("VmAB", "Vmag")
        result.rename_column("e_UmAB", "e_Umag")
        result.rename_column("e_BmAB", "e_Bmag")
        result.rename_column("e_VmAB", "e_Vmag")
    if catalog == 'II/339/uvotssc1':
        result.rename_column("U-AB", "Umag")
        result.rename_column("B-AB", "Bmag")
        result.rename_column("V-AB", "Vmag")
    if catalog == 'I/284/out':
        result.rename_column("B1mag", "Bmag")
        result.rename_column("R1mag", "Rmag")
    if catalog == 'II/336/apass9':
        result.rename_column("r_mag", "Rmag")
        result.rename_column("i_mag", "Imag")
        result.rename_column("e_r_mag", "e_Rmag")
        result.rename_column("e_i_mag", "e_Imag")

    #   Calculate B, U, etc. if only B-V, U-B, etc are given
    if catalog in ['II/168/ubvmeans']:
        result['Bmag'] = result['B-V'] + result['Vmag']
        result['e_Bmag'] = result['e_B-V'] + result['e_Vmag']
        result['Umag'] = result['U-B'] + result['Bmag']
        result['e_Umag'] = result['e_U-B'] + result['e_Bmag']

    #   Filter magnitudes: upper and lower threshold
    if 'Vmag' in result.keys():
        filmag = 'Vmag'
    elif 'Rmag' in result.keys():
        filmag = 'Rmag'
    elif 'Bmag' in result.keys():
        filmag = 'Bmag'
    elif 'Imag' in result.keys():
        filmag = 'Imag'
    elif 'Umag' in result.keys():
        filmag = 'Umag'
    else:
        #   This should never happen
        terminal_output.print_to_terminal(
            "Calibration issue: Threshold magnitude not recognized",
            indent=indent + 1,
            style_name='ERROR',
        )
        raise RuntimeError

    mask = (result[filmag] <= mag_range[1]) & (result[filmag] >= mag_range[0])
    result = result[mask]

    #   Define dict with column names
    if catalog == 'II/168/ubvmeans':
        column_dict = {'ra': '_RA', 'dec': '_DE'}
    else:
        column_dict = {'ra': 'RAJ2000', 'dec': 'DEJ2000'}
    for band in bands:
        if band + 'mag' in result.colnames:
            column_dict['mag' + band] = band + 'mag'

            #   Check if catalog contains magnitude errors
            if 'e_' + band + 'mag' in result.colnames:
                column_dict['err' + band] = 'e_' + band + 'mag'
        else:
            terminal_output.print_terminal(
                band,
                indent=indent + 1,
                string="No calibration data for {} band",
                style_name='WARNING',
            )

    return result, column_dict


def read_votable_simbad(calib_file, band_list, mag_range=(0., 18.5),
                        indent=2):
    """
        Read table in VO format already downloaded from Simbad

        Parameters
        ----------
        calib_file      : `string`
            Path to the calibration file

        band_list       : `list` of `string`
            Filter names

        mag_range       : `tupel` of `float`, optional
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
    terminal_output.print_terminal(
        calib_file,
        indent=indent,
        string="Read calibration data from a VO table: {}",
    )

    #   Read table
    calib_tbl = Table.read(calib_file, format='votable')

    #   Filter magnitudes: lower and upper threshold
    mask = calib_tbl['FLUX_V'] >= mag_range[0]
    mask = mask * calib_tbl['FLUX_V'] <= mag_range[1]
    calib_tbl = calib_tbl[mask]

    #   Define dict with column names
    column_dict = {'ra': 'RA_d', 'dec': 'DEC_d'}

    for band in band_list:
        if 'FLUX_' + band in calib_tbl.colnames:
            #   Check for variability and multiplicity flags
            ind_rm = np.where(calib_tbl['FLUX_MULT_' + band].mask)
            calib_tbl.remove_rows(ind_rm)
            ind_rm = np.nonzero(calib_tbl['FLUX_MULT_' + band])
            calib_tbl.remove_rows(ind_rm)
            ind_rm = np.where(calib_tbl['FLUX_VAR_' + band].mask)
            calib_tbl.remove_rows(ind_rm)
            ind_rm = np.nonzero(calib_tbl['FLUX_VAR_' + band])
            calib_tbl.remove_rows(ind_rm)

            if not calib_tbl:
                raise Exception(
                    f"{style.Bcolors.FAIL}\nAll calibration stars in the "
                    f"{band} removed because of variability and multiplicity "
                    f"citeria. -> EXIT {style.Bcolors.ENDC}"
                )

            column_dict['mag' + band] = 'FLUX_' + band
            column_dict['err' + band] = 'FLUX_ERROR_' + band
            column_dict['qua' + band] = 'FLUX_QUAL_' + band
        else:
            terminal_output.print_terminal(
                band,
                indent=indent + 1,
                string="No calibration data for {} band",
                style_name='WARNING',
            )

    return calib_tbl, column_dict


def load_calib(image, band_list, calib_method='APASS', mag_range=(0., 18.5),
               vizier_dict=None, calib_file=None, ra_unit=u.deg, indent=1):
    """
        Load calibration information

        Parameters
        ----------
        image           : `image.class` or `image.ensemble`
            Class object with all image specific properties

        band_list       : `list` with `strings`
            Filter list

        calib_method       : `string`, optional
            Calibration method
            Default is ``APASS``.

        mag_range       : `tupel` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        vizier_dict     : `dictionary` or None, optional
            Identifiers of catalogs, containing calibration data
            Default is ``None``.

        calib_file      : `string`, optional
            Path to the calibration file
            Default is ``None``.

        ra_unit         : `astropy.unit`, optional
            Right ascension unit
            Default is ``u.deg``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.

        Returns
        -------
        calib_tbl       : `astropy.table.Table`
            Astropy table with the calibration data

        col_names       : `dictionary`
            Column names versus the internal default names

        ra_unit         : `astropy.unit`
            Returns also the right ascension unit in case it changed
    """
    #   Get identifiers of catalogs if no has been provided
    if vizier_dict is None:
        vizier_dict = calibration_data.vizier_dict

    #   Read calibration table
    if calib_method == 'vsp':
        #   Load calibration info from AAVSO for variable stars
        calib_tbl, col_names = get_comp_stars(
            image.coord,

            filters=band_list,
            field_of_view=1.5 * image.fov,
            mag_range=mag_range,
            indent=indent + 1,
        )
        ra_unit = u.hourangle
    elif calib_method == 'simbad_vot' and calib_file is not None:
        #   Load info from data file in VO format downloaded from Simbad
        calib_tbl, col_names = read_votable_simbad(
            calib_file,
            band_list,
            mag_range=mag_range,
            indent=indent + 1,
        )
    elif calib_method in vizier_dict.keys():
        #   Load info from Vizier
        calib_tbl, col_names = get_catalog(
            band_list,
            image.coord,
            image.fov,
            vizier_dict[calib_method],
            mag_range=mag_range,
            indent=indent + 1,
        )
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nCalibration method not recognized\n"
            "Check variable: calib_method and vizier_dict "
            f"-> EXIT {style.Bcolors.ENDC}"
        )

    terminal_output.print_terminal(
        len(calib_tbl),
        indent=indent + 2,
        string="{} calibration stars downloaded",
        style_name='OKBLUE',
    )

    #   The next block is probably necessary for the magnitude
    #   transformation but impedes calibration without -> needs to be rewritten
    for band in band_list:
        if 'mag' + band in col_names:
            #   Remove objects without magnitudes from the calibration list
            arr = calib_tbl[col_names['mag' + band]]
            if hasattr(arr, 'mask'):
                ind_rm = np.where(arr.mask)
                #   TODO: Add a test to check whether the following line is working or not.
                calib_tbl.remove_rows(ind_rm)

    if not calib_tbl:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo calibration star with {band_list} "
            f"magnitudes found. -> EXIT {style.Bcolors.ENDC}"
        )
    terminal_output.print_terminal(
        len(calib_tbl),
        indent=indent + 2,
        string="Of these {} are useful",
        style_name='OKBLUE',
    )

    return calib_tbl, col_names, ra_unit


def get_observed_magnitudes_of_calibration_stars(img, mags, img_container):
    """
        Sort and rearrange input numpy array with extracted magnitude
        data, such that the returned numpy array contains the extracted
        magnitudes of the calibration stars

        Parameters
        ----------
        img             : `image class`
            Image class object

        mags            : `numpy.ndarray` or `unumpy.uarray`
            Array with image magnitudes

        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        Returns
        -------
        mags_fit        : `numpy.ndarray` or `unumpy.uarray`
            Rearrange array with magnitudes
    """
    #   Get calibration data
    ind_fit = img_container.CalibParameters.inds
    col_names = img_container.CalibParameters.column_names

    #   Convert index array of the calibration stars to a list
    ind_list = list(ind_fit)

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
        if f'mag{getattr(img, "filt", "?")}' in col_names:
            #   Sort
            mags_fit = mags[ind_list]
        else:
            mags_fit = unumpy.uarray(
                np.zeros(count_cali),
                np.zeros(count_cali)
            )

    #   numpy structured array
    else:
        #   Define array for the magnitudes of the calibration stars
        mags_fit = np.zeros(
            count_cali,
            dtype=[('mag', 'f8'), ('err', 'f8')],
        )

        #   Check if we have calibration data for the current filter/image
        if f'mag{getattr(img, "filt", "?")}' in col_names:
            #   Sort
            mags_fit['mag'] = mags['mag'][ind_list]
            mags_fit['err'] = mags['err'][ind_list]

    #   Add array with magnitudes to the image
    return mags_fit


def deter_calib(img_container, band_list, calib_method='APASS',
                dcr=3., option=1, vizier_dict=None,
                calib_file=None, id_obj=None, ra_unit=u.deg, dec_unit=u.deg,
                mag_range=(0., 18.5), rm_obj_coord=None,
                correl_method='astropy', seplimit=2. * u.arcsec, indent=1):
    """
        Determine calibration information, find suitable calibration stars
        and determine calibration factors

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        band_list               : `list` of `string`
            Filter list

        calib_method           : `string`, optional
            Calibration method
            Default is ``APASS``.

        dcr                     : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option                  : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        vizier_dict             : `dictionary` or `None`, optional
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``None``.

        calib_file              : `string`, optional
            Path to the calibration file
            Default is ``None``.

        id_obj                  : `integer`, optional
            ID of the object
            Default is ``None``.

        ra_unit                 : `astropy.unit`, optional
            Right ascension unit
            Default is ``u.deg``.

        dec_unit                : `astropy.unit`, optional
            Declination unit
            Default is ``u.deg``.

        mag_range               : `tupel` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        rm_obj_coord            : `astropy.coordinates.SkyCoord`, optional
            Coordinates of an object that should not be used for calibrating
            the data.
            Default is ``None``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        indent                  : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    """
    terminal_output.print_to_terminal(
        f"Get calibration star magnitudes (filter: {tuple(band_list)})",
        indent=indent,
    )

    #   Get one of image ensembles to extract wcs, positions, ect.
    #   TODO: replace with reference filter or at all relevant infos to image container?
    img_ensemble = img_container.ensembles[band_list[0]]

    #   Get wcs
    w = img_ensemble.wcs

    #   Load calibration data
    calib_tbl, col_names, ra_unit = load_calib(
        img_ensemble,
        band_list,
        calib_method=calib_method,
        mag_range=mag_range,
        vizier_dict=vizier_dict,
        calib_file=calib_file,
        indent=indent,
        ra_unit=ra_unit,
    )

    #   Convert coordinates of the calibration stars to SkyCoord object
    coord_calib = SkyCoord(
        calib_tbl[col_names['ra']].data,
        calib_tbl[col_names['dec']].data,
        unit=(ra_unit, dec_unit),
        frame="icrs"
    )

    #   Get PixelRegion of the field of view and convert it SkyRegion
    region_pix = img_ensemble.region_pix
    region_sky = region_pix.to_sky(w)

    #   Remove calibration stars that are not within the field of view
    mask = region_sky.contains(coord_calib, w)
    coord_calib = coord_calib[mask]
    calib_tbl = calib_tbl[mask]

    #   Remove a specific star from the loaded calibration stars
    if rm_obj_coord is not None:
        mask = coord_calib.separation(rm_obj_coord) < 1 * u.arcsec
        mask = np.invert(mask)
        coord_calib = coord_calib[mask]

    #   Calculate object positions in pixel coordinates
    x_cali, y_cali = coord_calib.to_pixel(w)

    #   Remove nans that are caused by missing ra/dec entries
    x_cali = x_cali[~np.isnan(x_cali)]
    y_cali = y_cali[~np.isnan(y_cali)]
    calib_tbl = calib_tbl[~np.isnan(y_cali)]

    #   X & Y pixel positions
    x = img_ensemble.image_list[0].photometry['x_fit']
    y = img_ensemble.image_list[0].photometry['y_fit']

    if correl_method == 'astropy':
        #   Create coordinates object
        coords_objs = SkyCoord.from_pixel(
            x,
            y,
            w,
        )

        #   Find matches between the datasets
        ind_fit, ind_lit, _, _ = matching.search_around_sky(
            coords_objs,
            coord_calib,
            seplimit,
        )

        count_cali = len(ind_lit)

    elif correl_method == 'own':
        #   Max. number of objects
        nmax = np.max(len(x), len(x_cali))

        #   Define and fill new arrays
        xall = np.zeros((nmax, 2))
        yall = np.zeros((nmax, 2))
        xall[0:len(x), 0] = x
        xall[0:len(x_cali), 1] = x_cali
        yall[0:len(y), 0] = y
        yall[0:len(y_cali), 1] = y_cali

        #   Correlate calibration stars with stars on the image
        inds, reject, count_cali, reject_obj = correlate.newsrcor(
            xall,
            yall,
            dcr=dcr,
            option=option,
        )
        ind_fit = inds[0]
        ind_lit = inds[1]

    else:
        raise ValueError(
            f'The correlation method needs to either "astropy" or "own". Got {correl_method} instead.'
        )

    if count_cali == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo calibration star was identified "
            f"-> EXIT {style.Bcolors.ENDC}"
        )
    if count_cali == 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}\nOnly one calibration star was identified\n"
            "Unfortunately, that is not enough at the moment\n"
            f"-> EXIT {style.Bcolors.ENDC}"
        )

    #   Limit calibration table to common objects
    calib_tbl_sort = calib_tbl[ind_lit]

    ###
    #   Plots
    #
    #   Make new arrays based on the correlation results
    x_fit = x[list(ind_fit)]
    y_fit = y[list(ind_fit)]
    indnew_fit = np.arange(count_cali)

    #   Add pixel positions and object ids to the calibration table
    calib_tbl_sort.add_columns(
        [np.intc(indnew_fit), x_fit, y_fit],
        names=['id', 'xcentroid', 'ycentroid']
    )

    calib_tbl.add_columns(
        [np.arange(0, len(y_cali)), x_cali, y_cali],
        names=['id', 'xcentroid', 'ycentroid']
    )

    #   Plot star map with calibration stars
    if id_obj is not None:
        rts = 'calib_' + str(id_obj)
    else:
        rts = 'calib'
    for band in band_list:
        if 'mag' + band in col_names:
            p = mp.Process(
                target=plot.starmap,
                args=(
                    img_ensemble.outpath.name,
                    #   Replace with reference image in the future
                    img_ensemble.image_list[0].get_data(),
                    band,
                    calib_tbl,
                ),
                kwargs={
                    'tbl_2': calib_tbl_sort,
                    'label': 'downloaded calibration stars',
                    'label_2': 'matched calibration stars',
                    'rts': rts,
                    'nameobj': img_ensemble.objname,
                }
            )
            p.start()

    #   Add calibration data to image container
    img_container.CalibParameters = CalibParameters(
        ind_fit,
        col_names,
        calib_tbl_sort,
    )


def magnitude_array_from_calibration_table(img_container, filter_list):
    """
        Arrange the literature values in a numpy array or uncertainty array.

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        filter_list     : `list` of `string`
            Filter names

        Returns
        -------
        lit_mags        : `numpy.ndarray` or `uncertainties.unumpy.uarray`
            Array with literature magnitudes
    """
    #   Number of filter
    n_filter = len(filter_list)

    #   Get calibration table
    calib_tbl = img_container.CalibParameters.calib_tbl
    calib_column_names = img_container.CalibParameters.column_names

    count_cali = len(calib_tbl)

    #   unmpy.array or default numpy.ndarray
    unc = getattr(img_container, 'unc', True)
    if unc:
        #   Create uncertainties array with the literature magnitudes
        lit_mags = unumpy.uarray(
            np.zeros((n_filter, count_cali)),
            np.zeros((n_filter, count_cali))
        )

        #
        for z, band in enumerate(filter_list):
            if 'mag' + band in calib_column_names:
                #   Check if errors for the calibration magnitudes exist
                if 'err' + band in calib_column_names:
                    err = np.array(
                        calib_tbl[calib_column_names['err' + band]]
                    )

                    #   Check if errors are nice floats
                    if err.dtype in (float, np.float32, np.float64):
                        valerr = err
                    else:
                        valerr = 0.
                else:
                    valerr = 0.

                #   Extract magnitudes
                lit_mags[z] = unumpy.uarray(
                    calib_tbl[calib_column_names['mag' + band]],
                    valerr
                )

    #   Default numpy.ndarray
    else:
        #   Define new arrays
        lit_mags = np.zeros(n_filter, dtype=[('mag', 'f8', count_cali),
                                             ('err', 'f8', count_cali),
                                             ('qua', 'U1', count_cali),
                                             ]
                            )

        #
        for z, band in enumerate(filter_list):
            if 'mag' + band in calib_column_names:
                #   Extract magnitudes
                col_mags = np.array(
                    calib_tbl[calib_column_names['mag' + band]]
                )
                lit_mags['mag'][z] = col_mags

                #   Check if errors for the calibration magnitudes exist
                if 'err' + band in calib_column_names:
                    valerr = np.array(
                        calib_tbl[calib_column_names['err' + band]]
                    )
                else:
                    valerr = np.zeros(count_cali)

                #   Check if errors are nice floats
                if valerr.dtype in (np.float, np.float32, np.float64):
                    lit_mags['err'][z] = valerr

                #   Add quality flag, if it exists
                if 'qua' + band in calib_column_names:
                    valqua = np.array(
                        calib_tbl[calib_column_names['qua' + band]]
                    )
                    lit_mags['qua'][z] = valqua

    return lit_mags
