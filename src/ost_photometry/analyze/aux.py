
############################################################################
####                            Libraries                               ####
############################################################################

import sys

import numpy as np

from uncertainties import unumpy

from pathlib import Path

import pandas as pd

from pytimedinput import timedInput

from astropy.table import Table
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.io import fits
from astropy.coordinates import SkyCoord, matching
from astropy.timeseries import TimeSeries
from astropy.modeling import models, fitting
import astropy.units as u
from astropy import wcs

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from regions import (
    RectangleSkyRegion,
    RectanglePixelRegion,
    PixCoord,
    CirclePixelRegion,
    Regions,
    )

from sklearn.cluster import SpectralClustering

import twirl

import multiprocessing as mp

import scipy.optimize as optimization

from .. import aux as base_aux

from .. import checks, style, terminal_output

from . import plot

############################################################################
####                        Routines & definitions                      ####
############################################################################

def err_prop(*args):
    '''
        Calculate error propagation

        Parameters
        ----------
        args        : `list`
            List of errors that should be added

        Returns
        -------
        u           : `float`
            Summed up error
    '''
    u = 0
    #   Adding up the errors
    for x in args:
        u = np.sqrt(np.square(u) + np.square(x))
    return u


def mk_cmd_table(ind_sort, x, y, mags, list_bands):
    '''
        Create and export the CMD

        Parameters
        ----------
        ind_sort        : `numpy.ndarray`
            IDs of the stars

        x               : `numpy.ndarray`
            Position of the stars on the image in pixel in X direction

        y               : `numpy.ndarray`
            Position of the stars on the image in pixel in X direction

        mags            : `numpy.ndarray`
            Magnitudes of all stars

        list_bands      : `list`
            Filter

        Returns
        -------
        tbl_cmd         : `astropy.table.Table`
            Table with CMD data
    '''
    #   Number of filter
    nfilter    = len(list_bands)

    #   Dimensions of magnitude array & number of images
    shape = mags['err'].shape
    dim = len(shape)
    if dim == 2:
        nimg = 1
    else:
        nimg = shape[1]

    # Make CMD table
    tbl_cmd = Table(
        names=['i','x', 'y',],
        data=[
            np.intc(ind_sort),
            x,
            y,
            ]
        )

    #   Set name of the magnitude field
    if 'med' in mags.dtype.names:
        name_mag = 'med'
    else:
        name_mag = 'mag'

    #   Add magnitude columns to CMD table
    for i in range(0, nfilter):
        if dim == 2:
            if 'err' in mags.dtype.names:
                tbl_cmd.add_columns(
                    [
                        mags[name_mag][i],
                        mags['err'][i],
                        ],
                    names=[
                        list_bands[i]+' [mag]',
                        list_bands[i]+'_err [mag]',
                        ]
                    )
            else:
                tbl_cmd.add_column(
                    mags[name_mag][i],
                    name=list_bands[i]+' [mag]'
                    )

            if i != 0:
                tbl_cmd.add_column(
                    mags[name_mag][i-1] - mags[name_mag][i],
                    name=list_bands[i-1]+'-'+list_bands[i]+' [mag]'
                    )
        else:
            for j in range(0, nimg):
                if nimg == 1:
                    if 'err' in mags.dtype.names:
                        tbl_cmd.add_columns(
                            [
                                mags[name_mag][i][j],
                                mags['err'][i][j],
                                ],
                            names=[
                                list_bands[i]+' [mag]',
                                list_bands[i]+'_err [mag]',
                                ]
                            )
                        if i != 0:
                            tbl_cmd.add_columns(
                                [
                                    mags[name_mag][i-1][j]-mags[name_mag][i][j],
                                    err_prop(
                                        mags['err'][i-1][j],
                                        mags['err'][i][j],
                                        ),
                                    ],
                                names=[
                                    list_bands[i-1]+'-'+list_bands[i]+' [mag]',
                                    list_bands[i-1]+'-'\
                                        +list_bands[i]+'_err [mag]',
                                    ]
                                )
                    else:
                        tbl_cmd.add_column(
                            mags[name_mag][i][j],
                            name=list_bands[i]+' [mag]'
                            )

                        if i != 0:
                            tbl_cmd.add_column(
                                mags[name_mag][i-1][j] - mags[name_mag][i][j],
                                name=list_bands[i-1]+'-'+list_bands[i]+' [mag]'
                                )

                else:
                    if 'err' in mags.dtype.names:
                        tbl_cmd.add_columns(
                            [
                                mags[name_mag][i][j],
                                mags['err'][i][j],
                                ],
                            names=[
                                list_bands[i]+' [mag] ('+str(j)+')',
                                list_bands[i]+'_err [mag] ('+str(j)+')',
                                ]
                            )
                        if i != 0:
                            tbl_cmd.add_columns(
                                [
                                    mags[name_mag][i-1][j]-mags[name_mag][i][j],
                                    err_prop(
                                        mags['err'][i-1][j],
                                        mags['err'][i][j],
                                        ),
                                    ],
                                names=[
                                    list_bands[i-1]+'-'+list_bands[i]\
                                        +' [mag] ('+str(j)+')',
                                    list_bands[i-1]+'-'+list_bands[i]\
                                        +'_err [mag] ('+str(j)+')',
                                    ]
                                )

                    else:
                        tbl_cmd.add_column(
                            mags[name_mag][i][j],
                            name=list_bands[i]+' [mag] ('+str(j)+')'
                            )
                        if i != 0:
                            tbl_cmd.add_column(
                                mags[name_mag][i-1][j] - mags[name_mag][i][j],
                                name=list_bands[i-1]+'-'+list_bands[i]\
                                        +' [mag] ('+str(j)+')'
                                )

    #   Sort CMD table
    if nimg == 1:
        tbl_cmd = tbl_cmd.group_by(list_bands[0]+' [mag]')
    else:
        tbl_cmd = tbl_cmd.group_by(list_bands[0]+' [mag] (0)')

    return tbl_cmd


def mk_mag_table(*args, **kwargs):
    '''
        Create and export astropy table with object positions and magnitudes

        Distinguishes between different input magnitude array types.
        Possibilities: unumpy.uarray & numpy structured ndarray
    '''
    #   Get type of the magnitude arrays
    unc = checks.check_unumpy_array(args[3])

    if unc:
        return mk_mag_table_unc(*args, **kwargs)
    else:
        return mk_mag_table_str(*args, **kwargs)


def mk_mag_table_str(ind_sort, x, y, mags, list_bands, id_tupels):
    '''
        Create and export astropy table with object positions and magnitudes
        Input magnitude array is expected to be a numpy structured array.

        Parameters
        ----------
        ind_sort        : `numpy.ndarray`
            IDs of the stars

        x               : `numpy.ndarray`
            Position of the stars on the image in pixel in X direction

        y               : `numpy.ndarray`
            Position of the stars on the image in pixel in X direction

        mags            : `numpy.ndarray`
            Magnitudes of all stars

        list_bands      : `list` of `string`
            Filter

        id_tupels       : `list` of `tupel`
            FORMAT = (Filter IDs, ID of the images to position 0, Filter IDs
                      for the color calculation, ID of the images to
                      position 2)

        Returns
        -------
        tbl_cmd         : `astropy.table.Table`
            Table with CMD data
    '''
    #   Number of filter
    nfilter    = len(list_bands)

    # Make CMD table
    tbl_cmd = Table(
        names=['i','x', 'y',],
        data=[
            np.intc(ind_sort),
            x,
            y,
            ]
        )

    #   Set name of the magnitude field
    name_mag = 'mag'

    #   Add magnitude columns to CMD table
    for ids in id_tupels:
        if 'err' in mags.dtype.names:
            tbl_cmd.add_columns(
                [
                    mags[name_mag][ids[0]][ids[1]],
                    mags['err'][ids[0]][ids[1]],
                    ],
                names=[
                    list_bands[ids[0]]+' [mag] ('+str(ids[1])+')',
                    list_bands[ids[0]]+'_err [mag] ('+str(ids[1])+')',
                    ]
                )
            if len(id_tupels) == 4:
                tbl_cmd.add_columns(
                    [
                    mags[name_mag][ids[2]][ids[3]]\
                        - mags[name_mag][ids[0]][ids[1]],
                    err_prop(
                        mags['err'][ids[2]][ids[3]],
                        mags['err'][ids[0]][ids[1]],
                        ),
                    ],
                    names=[
                        list_bands[ids[2]]+'-'+list_bands[ids[0]]\
                            +' [mag] ('+str(ids[1])+')',
                        list_bands[ids[2]]+'-'+list_bands[ids[0]]\
                            +'_err [mag] ('+str(ids[1])+')',
                        ]
                    )

        else:
            tbl_cmd.add_column(
                mags[name_mag][ids[0]][ids[1]],
                name=list_bands[ids[0]]+' [mag] ('+str(ids[1])+')'
                )
            tbl_cmd.add_column(
                mags[name_mag][ids[2]][ids[3]]-mags[name_mag][ids[0]][ids[1]],
                name=list_bands[ids[2]]+'-'+list_bands[ids[0]]\
                        +' [mag] ('+str(ids[1])+')'
                )

    #   Sort CMD table
    tbl_cmd = tbl_cmd.group_by(list_bands[0]+' [mag] (0)')

    return tbl_cmd


def mk_mag_table_unc(ind_sort, x, y, mags, list_bands, id_tupels):
    '''
        Create and export astropy table with object positions and magnitudes
        Input magnitude array is expected to be a unumpy uarray.

        Parameters
        ----------
        ind_sort        : `numpy.ndarray`
            IDs of the stars

        x               : `numpy.ndarray`
            Position of the stars on the image in pixel in X direction

        y               : `numpy.ndarray`
            Position of the stars on the image in pixel in X direction

        mags            : `unumpy.ndarray`
            Magnitudes of all stars

        list_bands      : `list` of `string`
            Filter

        id_tupels       : `list` of `tupel`
            FORMAT = (Filter IDs, ID of the images to position 0, Filter IDs
                      for the color calculation, ID of the images to
                      position 2)

        Returns
        -------
        tbl_cmd         : `astropy.table.Table`
            Table with CMD data
    '''
    #   Number of filter
    nfilter    = len(list_bands)

    # Make CMD table
    tbl_cmd = Table(
        names=['i','x', 'y',],
        data=[
            np.intc(ind_sort),
            x,
            y,
            ]
        )

    #   Add magnitude columns to CMD table
    for ids in id_tupels:
        tbl_cmd.add_columns(
            [
                unumpy.nominal_values(mags[ids[0]][ids[1]]),
                unumpy.std_devs(mags[ids[0]][ids[1]]),
                ],
            names=[
                list_bands[ids[0]]+' [mag] ('+str(ids[1])+')',
                list_bands[ids[0]]+'_err [mag] ('+str(ids[1])+')',
                ]
            )
        if len(id_tupels) == 4:
            color = mags[ids[2]][ids[3]] - mags[ids[0]][ids[1]]
            tbl_cmd.add_columns(
                [
                    unumpy.nominal_values(color),
                    unumpy.std_devs(color),
                ],
                names=[
                    list_bands[ids[2]]+'-'+list_bands[ids[0]]\
                        +' [mag] ('+str(ids[1])+')',
                    list_bands[ids[2]]+'-'+list_bands[ids[0]]\
                        +'_err [mag] ('+str(ids[1])+')',
                    ]
                )


    #   Sort CMD table
    tbl_cmd = tbl_cmd.group_by(
        list_bands[id_tupels[0][0]]+' [mag] ('+str(id_tupels[0][1])+')'
        )

    return tbl_cmd


def mk_cmd_table_u(ind_sort, x, y, mags, list_bands):
    '''
        Create and export the CMD

        Parameters
        ----------
        ind_sort        : `numpy.ndarray`
            IDs of the stars

        x               : `numpy.ndarray`
            Position of the stars on the image in pixel in X direction

        y               : `numpy.ndarray`
            Position of the stars on the image in pixel in X direction

        mags            : `unumpy.ndarray`
            Magnitudes of all stars

        list_bands      : `list`
            Filter

        Returns
        -------
        tbl_cmd         : `astropy.table.Table`
            Table with CMD data
    '''
    #   Number of filter
    nfilter    = len(list_bands)

    #   Dimensions of magnitude array & number of images
    shape = mags.shape
    dim = len(shape)
    if dim == 2:
        nimg = 1
    else:
        nimg = shape[1]

    # Make CMD table
    tbl_cmd = Table(
        names=['i','x', 'y',],
        data=[
            np.intc(ind_sort),
            x,
            y,
            ]
        )

    #   Add magnitude columns to CMD table
    for i in range(0, nfilter):
        if dim == 2:
            tbl_cmd.add_columns(
                [
                    unumpy.nominal_values(mags)[i],
                    unumpy.std_devs(mags)[i],
                    ],
                names=[
                    list_bands[i]+' [mag]',
                    list_bands[i]+'_err [mag]',
                    ]
                )

            if i != 0:
                tbl_cmd.add_column(
                    unumpy.nominal_values(mags[i-1] - mags[i]),
                    name=list_bands[i-1]+'-'+list_bands[i]+' [mag]'
                    )
        else:
            for j in range(0, nimg):
                if nimg == 1:
                    tbl_cmd.add_columns(
                        [
                            unumpy.nominal_values(mags)[i][j],
                            unumpy.std_devs(mags)[i][j],
                            ],
                        names=[
                            list_bands[i]+' [mag]',
                            list_bands[i]+'_err [mag]',
                            ]
                        )
                    if i != 0:
                        tbl_cmd.add_columns(
                            [
                                unumpy.nominal_values(mags[i-1][j]-mags[i][j]),
                                unumpy.std_devs(mags[i-1][j]-mags[i][j]),
                                ],
                            names=[
                                list_bands[i-1]+'-'+list_bands[i]+' [mag]',
                                list_bands[i-1]+'-'\
                                    +list_bands[i]+'_err [mag]',
                                ]
                            )
                else:
                    tbl_cmd.add_columns(
                        [
                            unumpy.nominal_values(mags[i][j]),
                            unumpy.std_devs(mags[i][j]),
                            ],
                        names=[
                            list_bands[i]+' [mag] ('+str(j)+')',
                            list_bands[i]+'_err [mag] ('+str(j)+')',
                            ]
                        )
                    if i != 0:
                        tbl_cmd.add_columns(
                            [
                                unumpy.nominal_values(mags[i-1][j]-mags[i][j]),
                                unumpy.std_devs(mags[i-1][j]-mags[i][j]),
                                ],
                            names=[
                                list_bands[i-1]+'-'+list_bands[i]\
                                    +' [mag] ('+str(j)+')',
                                list_bands[i-1]+'-'+list_bands[i]\
                                    +'_err [mag] ('+str(j)+')',
                                ]
                            )


    #   Sort CMD table
    if nimg == 1:
        tbl_cmd = tbl_cmd.group_by(list_bands[0]+' [mag]')
    else:
        tbl_cmd = tbl_cmd.group_by(list_bands[0]+' [mag] (0)')

    return tbl_cmd


def find_wcs(image_ensemble, ref_id=None, method='astrometry', rmcos=False,
             path_cos=None, x=None, y=None, force_wcs_determ=False, indent=2):
    '''
        Meta function for finding image WCS

        Parameters
        ----------
        image_ensemble  : `image.ensemble.class`
            Image class with all images taken in a specific filter

        ref_id          : `integer`, optional
            ID of the reference image
            Default is ``None``.

        method          : `string`, optional
            Method to use for the WCS determination
            Options: 'astrometry', 'astap', or 'twirl'
            Default is ``astrometry``.

        rmcos           : `boolean`, optional
            If True the function assumes that the cosmic ray reduction
            function was run before this function
            Default is ``False``.

        path_cos        : `string`
            Path to the image in case 'rmcos' is True
            Default is ``None``.

        x, y            : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    '''
    if ref_id is not None:
        #   Image
        img = image_ensemble.image_list[ref_id]

        #   Test if the image contains already a WCS
        cal_wcs, wcs_file = base_aux.check_wcs_exists(img)

        if not cal_wcs or force_wcs_determ:
            #   Calculate WCS -> astrometry.net
            if method == 'astrometry':
                image_ensemble.set_wcs(
                    base_aux.find_wcs_astrometry(
                        img,
                        rmcos=rmcos,
                        path_cos=path_cos,
                        indent=indent,
                        )
                    )

            #   Calculate WCS -> ASTAP program
            elif method == 'astap':
                image_ensemble.set_wcs(
                    base_aux.find_wcs_astap(img, indent=indent)
                    )

            #   Calculate WCS -> twirl libary
            elif method == 'twirl':
                if x is None or y is None:
                    raise RuntimeError(
                        f"{style.bcolors.FAIL} \nException in find_wcs(): '"
                        f"\n'x' or 'y' is None -> Exit {style.bcolors.ENDC}"
                        )
                image_ensemble.set_wcs(
                    base_aux.find_wcs_twirl(img, x, y, indent=indent)
                    )
            #   Raise exception
            else:
                raise RuntimeError(
                    f"{style.bcolors.FAIL} \nException in find_wcs(): '"
                    f"\nWCS method not known -> Supplied method was {method}"
                    f"{style.bcolors.ENDC}"
                    )
        else:
            image_ensemble.set_wcs(extract_wcs(wcs_file))
    else:
        for i, img in enumerate(image_ensemble.image_list):
            #   Test if the image contains already a WCS
            cal_wcs = base_aux.check_wcs_exists(img)

            if not cal_wcs or force_wcs_determ:
                #   Calculate WCS -> astrometry.net
                if method == 'astrometry':
                    w = base_aux.find_wcs_astrometry(
                        img,
                        rmcos=rmcos,
                        path_cos=path_cos,
                        indent=indent,
                        )

                #   Calculate WCS -> ASTAP program
                elif method == 'astap':
                    w = base_aux.find_wcs_astap(img, indent=indent)

                #   Calculate WCS -> twirl libary
                elif method == 'twirl':
                    if x is None or y is None:
                        raise RuntimeError(
                            f"{style.bcolors.FAIL} \nException in "
                            "find_wcs(): ' \n'x' or 'y' is None -> Exit"
                            f"{style.bcolors.ENDC}"
                            )
                    w = base_aux.find_wcs_twirl(img, x, y, indent=indent)

                #   Raise exception
                else:
                    raise RuntimeError(
                        f"{style.bcolors.FAIL} \nException in find_wcs(): '"
                        "\nWCS method not known -> Supplied method was "
                        f"{method} {style.bcolors.ENDC}"
                        )

            if i == 0:
                image_ensemble.set_wcs(w)


def extract_wcs(wcs_path, image_wcs=None, rmcos=False, filters=None):
    '''
        Load wcs from FITS file

        Parameters
        ----------
        wcs_path         : `string`
            Path to the image with the WCS or path to the directory that
            contains this image

        image_wcs       : `string`, optional
            WCS image name. Needed in case 'wcs_path` is only the path to
            the image directory.
            Default is ``None``.

        rmcos           : `boolean`, optional
            If True cosmic rays will be removed.
            Default is ``False``.

        filters         : `list` of `string`, optional
            Filter list
            Default is ``None``.
    '''
    #   Open the image with the WCS solution
    if image_wcs is not None:
        if rmcos:
            if filters is None:
                raise Exception(
                    f"{style.bcolors.FAIL} \nException in extract_wcs(): '"
                    "\n'rmcos=True' buit no 'filters' given -> Exit"
                    f"{style.bcolors.ENDC}"
                    )
            basename = f'img_cut_{filters[0]}_lacosmic'
        else:
            basename = image_wcs.split('/')[-1].split('.')[0]
        hdulist = fits.open(f'{wcs_path}/{basename}.new')
    else:
        hdulist = fits.open(wcs_path)

    #   Extract the WCS
    w = wcs.WCS(hdulist[0].header)

    return w


def mk_ts(obs_time, cali_mags, filt, objID):
    '''
        Make a time series object

        Parameters
        ----------
        obs_time        : `numpy.ndarray`
            Observation times

        cali_mags       : `numpy.ndarray`
            Magnitudes and uncertainties

        filt            : `filter`
            Filte

        objID           : `integer`
            ID/Number of the object for with the time series should be
            created

        Returns
        -------
        ts              : `astropy.timeseries.TimeSeries`
    '''
    #   Extract magnitudes of the object 'objID' depending on array dtype
    if checks.check_unumpy_array(cali_mags):
        umags = cali_mags[:,objID]
        mags_obj = unumpy.nominal_values(umags)
        errs_obj = unumpy.std_devs(umags)

    else:
        try:
            mags_obj = cali_mags['mag'][:,objID]
            errs_obj = cali_mags['err'][:,objID]
        except:
            mags_obj = cali_mags['flux'][:,objID]
            errs_obj = cali_mags['err'][:,objID]

    #   Create mask for time series to remove images without entries
    mask = np.isin(
        mags_obj,
        [0.],
        invert=True
        )

    #   Remove images without entries
    mags_obj = mags_obj[mask]
    errs_obj = errs_obj[mask]

    # Make time series and use reshape to get a justified array
    ts = TimeSeries(
            time=obs_time,
            data={
                filt: mags_obj.reshape(mags_obj.size,) * u.mag,
                filt+'_err': errs_obj.reshape(errs_obj.size,) * u.mag,
                }
            )
    return ts


def lin_func(x, a, b):
    '''
        Linear function
    '''
    return a + b*x


def fit_curve(fit_func, x, y, x0, sigma):
    '''
        Fit curve with supplied fit function

        Parameters
        ----------
        fit_func        : `function`
            Function used in the fitting process

        x               : `nump.ndarray`
            Abscissa values

        y               : `numpy.ndarray`
            Ordinate values

        x0              : `numpy.ndarray`
            Initial guess for the fit parameters

        sigma           : `numpy.ndarray`
            Uncertainty of the ordinate values

        Returns
        -------
        a               : `float`
            Parameter I

        a_err           : `float`
            Error parameter I

        b               : `float`
            Parameter II

        b_err           : `float`
            Error parameter II
    '''

    #   Fit curve
    if np.any(sigma == 0.):
        para, coma = optimization.curve_fit(fit_func, x, y, x0)
    else:
        para, coma = optimization.curve_fit(fit_func, x, y, x0, sigma)
    a = para[0]
    b = para[1]
    a_err = coma[0,0]
    b_err = coma[1,1]

    return a, a_err, b, b_err


def fit_data_oneD(x, y, order):
    '''
        Fit polynomial to the provided data.

        Parameters
        ----------
        x               : `nump.ndarray` or `unmapy.uarray`
            X data values

        y               : `nump.ndarray` or `unmapy.uarray`
            Y data values

        order           : `integer`
            Order of the polynomial to be fitted to the data
    '''
    #   Check array type
    unc = checks.check_unumpy_array(x)

    #   Set model
    model = models.Polynomial1D(degree=order)

    #   Set fitter
    fitter_poly = fitting.LevMarLSQFitter()

    #   Fit data
    if unc:
        if np.all(unumpy.nominal_values(x) == 0.):
            fit_poly = None
        else:
            fit_poly = fitter_poly(
                model,
                unumpy.nominal_values(x),
                unumpy.nominal_values(y),
                )
    else:
        if np.all(x == 0.):
            fit_poly = None
        else:
            fit_poly = fitter_poly(
                model,
                x,
                y,
                )

    return fit_poly


def prepare_arrays(img_container, nfilter, count):
    '''
        Prepare arrays for magnitude calibration

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        nfilter         : `integer`
            Number of filter

        count           : `integer`
            Number of stars
    '''
    #   Get image ensembles
    img_ensembles = img_container.ensembles

    #   Get maximum number of images
    nimgs = []
    for ensemble in img_ensembles.values():
        nimgs.append(len(ensemble.image_list))

    #   Maximum number of images
    nimg_max = np.max(nimgs)

    #   Get required array type
    unc = getattr(img_container, 'unc', True)

    #   Define magnitude arrays
    if unc:
        cali = unumpy.uarray(
            np.zeros((nfilter,nimg_max,count)),
            np.zeros((nfilter,nimg_max,count))
            )
    else:
        #   Define arrays
        cali = np.zeros(nfilter, dtype=[('mag', 'f8', (nimg_max, count)),
                                        ('std', 'f8', (nimg_max, count)),
                                        ('err', 'f8', (nimg_max, count)),
                                    ]
            )

    img_container.cali = cali
    img_container.noT = np.copy(cali)

    #   Define flux arrays
    if unc:
        img_container.flux = np.copy(cali)
    else:
        img_container.flux = np.zeros(
            nfilter,
            dtype=[('flux', 'f8', (nimg_max, count)),
                   ('err', 'f8', (nimg_max, count)),
                  ]
            )


def cal_mag(*args, **kwargs):
    '''
        Wrapper function: distinguish between astropy table
                          and pandas data frame
    '''
    if base_aux.np_vs_df(args[0]):
        return mag_df(*args, **kwargs)
    else:
        return mag_arr(*args, **kwargs)


#@timeis
def mag_arr(flux_arr):
    '''
        Calculate magnitudes from flux

        Parameters
        ----------
        flux_arr        : `numpy.ndarray`
            Numpy structured array containing flux values and corresponding
            uncertainties

        Returns
        -------
        mags            : `numpy.ndarray`
            Numpy structured array containing magnitudes and corresponding
            errors
    '''
    #   Get dimensions
    shape = flux_arr['flux_fit'].shape
    if len(shape) == 1:
        nobj = shape[0]

        #   Prepare array for the magnitudes and uncertainty
        mags = np.zeros(nobj, dtype=[('mag', 'f8'), ('err', 'f8')])

    elif len(shape) == 2:
        nimg = shape[0]
        nobj = shape[1]

        #   Prepare array for the magnitudes and uncertainty
        mags = np.zeros(
            nimg,
            dtype=[('mag', 'f8', (nobj)), ('err', 'f8', (nobj))],
            )

    else:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nDimension of the flux array > 2. This "
            f"is not supported. -> Exit {style.bcolors.ENDC}"
            )

    ###
    #   Calculate magnitudes
    #
    #   Extract flux
    flux        = flux_arr['flux_fit']
    #   Calculate magnitudes
    mags['mag'] = -2.5 * np.log10(flux)


    ###
    #   Calculate magnitudes and error
    #
    #   Error propagation also used by DAOPHOT -> see 'compute_phot_error'
    mags['err'] = 1.0857 * flux_arr['flux_unc'] / flux_arr['flux_fit']

    return mags


def mag_u_arr(flux):
    '''
        Calculate magnitudes from flux

        Parameters
        ----------
        flux            : `unumpy.ndarray`
            Numpy structured array containing flux values and corresponding
            uncertainties

        Returns
        -------
        mags            : `unumpy.ndarray`
            Numpy structured array containing magnitudes and corresponding
            errors
    '''
    #   Get dimensions
    shape = flux.shape
    dim = len(shape)
    if 0 == dim or dim > 2:
        raise ValueError(
            f"{style.bcolors.FAIL} \nDimension of the flux array > 2. This "
            f"is not supported. -> Exit {style.bcolors.ENDC}"
            )

    #   Calculate magnitudes
    mags = -2.5 * unumpy.log10(flux)

    return mags


#@timeis
def mag_df(flux_df, flux_id='flux_fit', unc_id='flux_unc'):
    '''
        Calculate magnitudes from flux

        Parameters
        ----------
        flux_df         : `pandas.DataFrame`
            Flux values and corresponding uncertainties

        flux_id         : `string`, optional
            Name for the flux column
            Default is ``flux_fit``.

        flux_unc        : `string`, optional
            Name for the flux uncertainty column
            Default is ``flux_unc``.

        Returns
        -------
        flux_df         : `pandas.DataFrame`
            Magnitudes and orresponding uncertainties
    '''
    #   Extract flux and flux uncertainty
    flux          = flux_df[flux_id]
    flux_err      = flux_df[unc_id]
    flux_err      = np.absolute(flux_err)

    #   Calculate magnitudes
    df_temp = -2.5 * np.log10(flux)

    #   Check if we are dealing with a Series or DataFrame
    if isinstance(df_temp, pd.Series):
        df_temp.name = 'mag'
    elif isinstance(df_temp, pd.DataFrame):
        #   Add 'mag' to the column index (make multi index), so that the
        #   magnitudes can be combined with the input data frame
        #   'flux_df'
        df_columns = df_temp.columns
        index = pd.MultiIndex.from_product([['mag'], df_columns])
        df_temp.columns = index
    else:
        raise Exception(
            f"{style.bcolors.FAIL} \nInput object is neither a Pandas Series "
            f"nor Dataframe -> EXIT {style.bcolors.ENDC}"
            )

    #   Add magnitudes to input data frame
    flux_df = pd.concat([flux_df, df_temp], axis=1)

    #   Prepare array with difference between flux and flux error
    #   Sanitize -> ensure that the difference is > 0
    pre_arr = np.where(
        flux_err < flux,
        flux - flux_err,
        1E-20
        )

    #   Calculate Errors
    mag_err = 1.0857 * flux_err / flux

    #   Branch according to Series or DataFrame
    if isinstance(mag_err_m, pd.DataFrame):
        #   New index
        index = pd.MultiIndex.from_product([['err'], df_columns])

        #   Convert numpy array to data frame
        err_df  = pd.DataFrame(
            mag_err,
            index=flux_df.index,
            columns=index,
            )
    else:
        #   Convert numpy array to data frame
        err_df  = pd.DataFrame({'err':mag_err}, index=flux_df.index)


    #   Attach errors to data frame
    flux_df = pd.concat([flux_df, err_df], axis=1)

    return flux_df


def mk_posi_tbl(img_container, ensemble_IDs):
    '''
        Make position tables

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        ensemble_IDs            : `list`
            List with image IDs

        Returns
        -------
        tbl_xy          : `dictionary` of `astropy.table.Table` objects
            X and Y position in pixel among other data
    '''
    #   Get image ensembles
    ensembles = img_container.ensembles

    tbl_xy = {}
    for ID in ensemble_IDs:
        #   Ensure ID is `string`
        ID = str(ID)

        #   Get ensemble
        ensem = ensembles[ID]

        #   Get reference image
        img = ensem.image_list[ensem.ref_id]

        #   Get magnitudes and positions
        try:
            mags = cal_mag(img.flux_es)['mag']
            ids = ensem.id_es
            xs = ensem.x_es
            ys = ensem.y_es
        except:
            mags = cal_mag(img.flux)['mag']
            ids = ensem.id_s
            xs = ensem.x_s
            ys = ensem.y_s

        #   Fill astropy table
        tbl_xy[ID] = Table(
            names=['id','xcentroid', 'ycentroid', 'mag'],
            data=[np.intc(ids), xs, ys, mags]
            )

    return tbl_xy


def mk_posi_tbl_ensem(ensemble):
    '''
        Make position tables

        Parameters
        ----------
        ensemble                : `image ensemble`
            Image image ensemble class object

        Returns
        -------
        tbl_xy          : `dictionary` of `astropy.table.Table` objects
            X and Y position in pixel among other data
    '''
    #   Define astropy table
    tbl_xy = {}

    #   Get image
    for img in ensemble.image_list:
        #   Get magnitudes and positions
        try:
            mags = cal_mag(img.flux_es)['mag']
            ids = ensemble.id_es
            xs = ensemble.x_es
            ys = ensemble.y_es
        except:
            mags = cal_mag(img.flux)['mag']
            ids = ensemble.id_s
            xs = ensemble.x_s
            ys = ensemble.y_s

        #   Fill astropy table
        tbl_xy[img.pd] = Table(
            names=['id','xcentroid', 'ycentroid', 'mag'],
            data=[np.intc(ids), xs, ys, mags]
            )

    return tbl_xy


def mk_posi_tbl_pd(ind_sort, img_IDs, x, y, mags):
    '''
        Make position tables

        Parameters
        ----------
        ind_sort        : `numpy.ndarray`
            IDs of the stars

        img_IDs         : `flist` of `float`
            List with image IDs

        x               : `numpy.ndarray` of `floats`
            Position of the objects on the image in pixel in X direction

        y               : `numpy.ndarray` of `floats`
            Position of the objects on the image in pixel in Y direction

        mags            : `pandas.DataFrame`
            Magnitude array of the stars

        Returns
        -------
        tbl_xy          : `astropy.table.Table`
            X and Y position in pixel among other data
    '''
    #   Drop level > 0, if column levels > 1
    if mags.columns.nlevels > 1:
        cols = mags.columns.droplevel(1)
    else:
        cols = mags.columns
    #   Check whether median values of the magnitudes are available
    #   in 'mags'
    if 'mag' in cols.values:
        col_name = 'mag'
    elif 'median' in cols.values:
        col_name = 'median'
    else:
        raise Exception(
            f"{style.bcolors.FAIL} \nMagnitude column not recognize in 'mags' "
            f"DataFrame -> EXIT {style.bcolors.ENDC}"
            )

    tbl_xy = {}
    for j, img_ID in enumerate(img_IDs):
        #   Restrict input data frame to current image
        mask = mags['type'] == img_ID

        img_ID = str(img_ID)
        tbl_xy[img_ID] = Table(
            names=['id','xcentroid', 'ycentroid', 'mag'],
            data=[np.intc(ind_sort), x, y, mags[mask][col_name]]
            )

    return tbl_xy


def find_filt(filt_list, in_dict, filt, camera, verbose=False, indent=2):
    '''
        Find the position of the filter from the dictionary 'filt'
        in the dictionary 'in_dict' with reference to 'filt_list'

        Parameters
        ----------
        filt_list       : `list` - `string`
            List of available filter, e.g., ['U', 'B', 'V', ...]

        in_dict         : `dictionary` - `string`:`dictionary`
            Calibration information. Keys:  camera identifier

        filt            : `string`
            Filter for which calibration data will be selected

        camera          : `string`
            Camera used

        verbose         : `boolean`, optional
            If ``True`` additional information will be printed to the console.
            Default is ``False``.

        indent          : `integer`, optional
            Indentation for the console output
            Default is ``2``.

        Returns
        -------
                        : `boolean`
            True if the filter 'filt` was successfully identified.

                        : `dictionary`
            Entry from dictionary 'in_dict' corresponding to filter 'filt'

                        : `integer`
            ID of filter 1

                        : `integer`
            ID of filter 2
    '''
    #   Initialize list of bools
    cam_bools = []

    #   Loop over outer dictionary: 'in_dict'
    for key_outer, value_outer in in_dict.items():
        #   Check if calibration data fits to the camera
        if camera == key_outer:
            #   Loop over inner dictionary
            for key_inner, value_inner in value_outer.items():
                #   Check if calibration data is available for the current
                #   filter 'filt'.
                if filt == key_inner:
                    f1 = value_inner['Filter 1']
                    f2 = value_inner['Filter 2']
                    #   Check if the filter used to calculate the
                    #   calibration data is also available in the filter
                    #   list 'filt_list'
                    if f1 in filt_list and f2 in filt_list:
                        #   Determine indexes of the filter
                        id_1 = filt_list.index(f1)
                        id_2 = filt_list.index(f2)
                        return value_inner, id_1, id_2
                    else:
                        if verbose:
                            terminal_output.print_terminal(
                                f1,
                                f2,
                                filt_list,
                                indent=indent,
                                string='Magnitude transformation coefficients'\
                                    ' do not apply. Wrong filter combination:'\
                                    ' {} & {} vs. {}',
                                style_name='WARNING',
                                )

            cam_bools.append(True)
        else:
            cam_bools.append(False)

    if not any(cam_bools):
        terminal_output.print_terminal(
            camera,
            indent=indent,
            string='Determined camera {} not consistent with the' \
            +' one given in the dictionary with the transformation' \
            +' coefficients.',
            style_name='WARNING',
            )

    return None, None, None


def check_variable(filename, filetype, filt_1, filt_2, cali, ISOcolumntype,
                   ISOcolumn):
    '''
        Check variables and set defaults

        Parameters
        ----------
        filename            : `string`
            Specified file name - can also be empty -> set default


        filetype            : `string`
            Specified file type - can also be empty -> set default

        filt_1              : `string`
            First filter

        filt_2              : `string`
            Second filter

        cali                : `dictionary`
            Keys = filter - Values = zero points

        ISOcolumntype       : `dictionary`
            Keys = filter - Values = type

        ISOcolumn           : `dictionary`
            Keys = filter - Values = column
    '''
    #   Set figure type
    if filename == "?" or filename =="":
        terminal_output.print_terminal(
            indent=1,
            string='[Warning] No filename given, us default (cmd)',
            style_name='WARNING',
            )
        filename = 'cmd'

    if filetype == '?' or filetype == '':
        terminal_output.print_terminal(
            indent=1,
            string='[Warning] No filetype given, use default (pdf)',
            style_name='WARNING',
            )
        filetype ='pdf'

    #   Check if file type is valid and set default
    filetype_list = ['pdf', 'png', 'eps', 'ps', 'svg']
    if filetype not in filetype_list:
        terminal_output.print_terminal(
            indent=1,
            string='[Warning] Unknown filetype given, use default instead '\
                  +'(pdf)',
            style_name='WARNING',
            )
        filetype = 'pdf'

    return filename, filetype

    #   Check if calibration parameter is consistent with the number of
    #   filter
    if len(filt_2) != len(cali):
        if len(filt_2) > len(cali):
            terminal_output.print_terminal(
                indent=1,
                string="[Error] More filter ('filt_2') specified than zero"\
                +" points ('cali')",
                style_name='WARNING',
            )
            sys.exit()
        else:
            terminal_output.print_terminal(
                indent=1,
                string="[Error] More zero points ('cali') specified than "\
                +"filter ('filt_2')",
                style_name='WARNING',
            )
            sys.exit()

    #   Check if the column declaration for the isochrones fits to the
    #   specified filter
    for fil in filt_2:
        if fil not in ISOcolumntype.keys():
            terminal_output.print_terminal(
                fil,
                indent=1,
                string="[Error] No entry for filter {:d} specified in"\
                +" 'ISOcolumntype'",
                style_name='WARNING',
            )
            sys.exit()
        if fil not in ISOcolumn.keys():
            terminal_output.print_terminal(
                fil,
                indent=1,
                string="[Error] No entry for filter {:d} specified in"\
                +" 'ISOcolumn'",
                style_name='WARNING',
            )
            sys.exit()
    if filt_1 not in ISOcolumn.keys():
        terminal_output.print_terminal(
            filt_1,
            indent=1,
            string="[Error] No entry for filter {:d} specified in"\
                    +" 'ISOcolumn'",
            style_name='WARNING',
        )
        sys.exit()


class Executor:
    '''
        Class that handels the multiprocessing, using apply_async.
        -> allows for easy catch of exceptions
    '''
    def __init__(self, process_num):
        #   Init multiprocessing pool
        self.pool = mp.Pool(process_num)
        #   Init variables
        self.res = []
        self.err = None

    def collect_results(self, result):
        '''
            Uses apply_async's callback to setup up a separate Queue
            for each process
        '''
        #   Catch all results
        self.res.append(result)

    def callback_error(self, e):
        '''
            Handles axceptions by apply_async's error callback
        '''
        #   Termninate pool
        self.pool.terminate()
        #   Raise exceptions
        self.err = e
        raise e

    def schedule(self, function, args, kwargs):
        '''
            Call to apply_async
        '''
        self.pool.apply_async(function, args, kwargs,
                              callback=self.collect_results,
                              error_callback=self.callback_error)

    def wait(self):
        '''
            Close pool and wait for completion
        '''
        self.pool.close()
        self.pool.join()


def mk_ds9_region(x, y, r, filename, wcs):
    '''
        Make and write a ds9 region file

        Parameters
        ----------
        x               : `numpy.ndarray`
            X coordinates in pixel

        y               : `numpy.ndarray`
            Y coordinates in pixel

        r               : `float`
            Radius in pixel

        filename        : `string`
            File name

        wcs             : `astropy.wcs.WCS`
            WCS infos
    '''
    #   Create the region
    c_regs = []

    for x_i, y_i in zip(x, y):
        #   Make a pixel coordinates object
        center = PixCoord(x=x_i, y=y_i)

        #   Create the region
        c = CirclePixelRegion(center, radius=r)

        #   Append region and convert to sky coordinates
        c_regs.append(c.to_sky(wcs))

    #   Convert to Regions that contain all individual regions
    reg = Regions(c_regs)

    #   Write the region file
    reg.write(filename, format='ds9', overwrite=True)


def prepare_and_plot_starmap(image, condense=False, tbl=None,
                             x_name='x_fit', y_name='y_fit', rts_pre='img',
                             label='Stars with photometric extractions',
                             add_image_id=True):
    '''
        Prepare table for star map and plot star map

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        condense        : `boolean`, optional
            If True the terminal output will be returned to the calling
            function.
            Default is ``False``.

        tbl             : `astropy.table.Table` or `None`, optional
            Table with position information.
            Default is ``None``.

        x_name          : `string`, optional
            Name of the X column in ``tbl``.
            Default is ``x_fit``.

        y_name          : `string`, optional
            Name of the Y column in ``tbl``.
            Default is ``y_fit``.

        rts_pre         : `string`, optional
            Expression used in the file name to characterizing the plot

        label           : `string`, optional
            Label that characterizes the star map.
            Default is ``Stars with photometric extractions``.

        add_image_id    : `boolean`, optional
            If ``True`` the image ID will be added to the file name.
            Default is ``True``.
    '''
    #   Get table, data, filter, & object name
    if tbl is None:
        tbl  = image.photometry
    data = image.get_data()
    filt = image.filt
    name = image.objname

    #   Prepare table
    nstars = len(tbl)
    tbl_xy = Table(
        names=['id', 'xcentroid', 'ycentroid'],
        data=[np.arange(0,nstars), tbl[x_name], tbl[y_name]],
        )

    #   Prepare string for file name
    if add_image_id:
        rts_pre += '-'+str(image.pd)

    #   Plot star map
    out_str = plot.starmap(
        image.outpath.name,
        data,
        filt,
        tbl_xy,
        label=label,
        rts=rts_pre,
        nameobj=name,
        condense=condense,
        )
    if condense:
        return out_str


def prepare_and_plot_starmap_final(img_container, filt_list):
    '''
        Prepare table for star map and plot star map

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        filt_list       : `list` of `strings`
            List with filter names
    '''
    terminal_output.print_terminal(
            indent=1,
            string="Plot star maps with positions from the final "\
                  +"correlation",
            )
    #   Make position table
    tbl_xy_final = mk_posi_tbl(
        img_container,
        filt_list,
        )

    for filt in filt_list:
        if filt == filt_list[0]:
            rts = str(filt_list[1])+'_final'
        else:
            rts = str(filt_list[0])+'_final'

        #   Get reference image
        image = img_container.ensembles[filt].ref_img

        #   Using multiprocessing to create the plot
        p = mp.Process(
            target=plot.starmap,
            args=(
                str(image.outpath / 'final'),
                image.get_data(),
                filt,
                tbl_xy_final[filt],
                ),
            kwargs={
                'rts':rts,
                'label':'Stars identified in '+str(filt_list[0])
                +' and '+str(filt_list[1])+' filter',
                'nameobj':image.objname,
                }
            )
        p.start()
    terminal_output.print_terminal()


def prepare_and_plot_starmap_final_3(img_ensemble, calib_xs, calib_ys,
                                     plot_test=True):
    '''
        Prepare table for star map and plot star map

        Parameters
        ----------
        img_ensemble    : `image ensemble`
            Image img_ensemble class object

        calib_xs        : `numpy.ndarray` of `floats`
            Position of the claibration objects on the image in pixel
            in X direction

        calib_ys        : `numpy.ndarray` of `floats`
            Position of the claibration objects on the image in pixel
            in Y direction

        plot_test       : `boolean`, optional
            If True only the masterplot for the reference image will
            be created.
            Default is ``True``.
    '''
    terminal_output.print_terminal(
        indent=1,
        string="Plot star map with the objects identified on all images",
        )

    #   Get image IDs, IDs of the objects, and pixel coordinates
    img_ids = img_ensemble.get_image_ids()

    #   Make position table
    tbl_xy_final = mk_posi_tbl_ensem(img_ensemble)

    #   Make new table with the position of the calibration stars
    tbl_xy_calib = Table(
        names=['xcentroid', 'ycentroid'],
        data=[[calib_xs], [calib_ys]]
        )

    #   Make the plot using multiprocessing
    for j, ID in enumerate(img_ids):
        key = str(ID)
        if plot_test and j != img_ensemble.ref_id: continue
        p = mp.Process(
            target=plot.starmap,
            args=(
                img_ensemble.outpath.name,
                img_ensemble.image_list[j].get_data(),
                img_ensemble.filt,
                tbl_xy_final[ID],
                ),
            kwargs={
                'tbl_2':tbl_xy_calib,
                'rts':key+'_final',
                'label':'Stars identified in all images',
                'label_2':'Calibration stars',
                'nameobj':img_ensemble.objname,
                }
            )
        p.start()
        terminal_output.print_terminal()


def add_median_table(img_ensemble, meanb=False):
    '''
        Calculate flux median or flux mean and add it to a new table

        Parameters
        ----------
        img_ensemble        : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        meanb               : `boolean`
            If True the mean instead of the median flux will be added to the
            table.
    '''
    #   Get flux:
    try:
        flux = img_ensemble.flux_es['flux_fit']
        xs = img_ensemble.x_es
        ys = img_ensemble.y_es
    except:
        flux = img_ensemble.flux['flux_fit']
        xs = img_ensemble.x_s
        ys = img_ensemble.y_s

    ###
    #   Calculate mean, median, and standard deviation of the
    #   object fluxes, using sigma clipping
    #
    mean, median, std = sigma_clipped_stats(
        flux,
        sigma=1.5,
        axis=0,
        )


    ###
    #   Make new table to add the median of the flux
    #
    #   New table
    if meanb:
        _tbl = Table(
            names=['x_fit', 'y_fit', 'flux_fit', 'flux_unc',],
            data=[xs, ys, mean, std,]
            )
    else:
        _tbl = Table(
            names=['x_fit', 'y_fit', 'flux_fit', 'flux_unc',],
            data=[xs, ys, median, std,]
            )

    #   Add to the overall dictionary
    img_ensemble.results = _tbl.group_by('x_fit')


def derive_limiting_mag(img_container, filt_list, ref_img, indent=1):
    '''
        Determine limiting magnitude

        Parameters
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        filt_list           : `list` of `strings`
            List with filter names

        ref_img             : `integer`, optional
            ID of the reference image
            Default is ``0``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    '''
    #   Get image ensembles
    img_ensembles = img_container.ensembles

    #   Get type of the magnitude arrays
    #   Possibilities: unumpy.uarray & numpy structured ndarray
    unc = getattr(img_container, 'unc', True)

    #   Get calibrated magnitudes
    cali_mags = getattr(img_container, 'cali', None)
    if unc:
        if (cali_mags is None or
            np.all(unumpy.nominal_values(cali_mags) == 0.)):
                #   If array with magnitude transformation is not available
                #   or if it is empty get the array without magnitude
                #   transformation
                cali_mags = getattr(img_container, 'noT', None)
                if cali_mags is not None:
                    #   Get only the magnitude values
                    cali_mags = unumpy.nominal_values(cali_mags)
        else:
            #   Get only the magnitude values
            cali_mags = unumpy.nominal_values(cali_mags)

    #   numpy structured ndarray type:
    else:
        if (cali_mags is None or np.all(cali_mags['mag'] == 0.)):
            #   If array with magnitude transformation is not available
            #   or if it is empty get the array without magnitude
            #   transformation
            cali_mags = getattr(img_container, 'noT', None)
            if cali_mags is not None:
                cali_mags = cali_mags['mag']
        else:
            cali_mags = cali_mags['mag']

    #   Get magnitudes of reference image
    for i, filt in enumerate(filt_list):
        #   Get image ensemble
        ensemble = img_ensembles[filt]

        #   Get object indices and X & Y pixel positions
        try:
            ind = ensemble.id_es
            x   = ensemble.x_es
            y   = ensemble.y_es
        except:
            ind = ensemble.id_s
            x   = ensemble.x_s
            y   = ensemble.y_s

        #   Get reference image
        image = ensemble.image_list[ref_img]

        mag_arr = cali_mags[i][ref_img]

        #   Make astropy table
        tbl_mag = Table(
            names=['xcentroid', 'ycentroid','mags'],
            data=[x, y, mag_arr]
            )
        tbl_mag = tbl_mag.group_by('mags')

        #   Remove implausible dark results
        mask = tbl_mag['mags'] < 30
        tbl_mag = tbl_mag[mask]

        #   Plot star map
        if ref_img != '':
            rts = 'mags_'+str(ref_img)
        else:
            rts = 'mags'
        p = mp.Process(
            target=plot.starmap,
            args=(
                image.outpath.name,
                image.get_data(),
                filt,
                tbl_mag[:][-10:],
                ),
            kwargs={
                'label':'10 faintest stars',
                'rts':rts,
                'mode':'mags',
                'nameobj':image.objname,
                }
            )
        p.start()

        #   Print result
        terminal_output.print_terminal(
            filt,
            indent=indent,
            string="Determine limiting magnitude for filter: {}",
            )
        terminal_output.print_terminal(
            np.median(tbl_mag['mags'][-10:]),
            indent=indent*2,
            string="Median of the 10 faintest objects: {}",
            )
        terminal_output.print_terminal(
            np.mean(tbl_mag['mags'][-10:]),
            indent=indent*2,
            string="Mean of the 10 faintest objects: {}",
            )


def rm_edge_objects(table, data, border=10, condense=False, indent=3):
    '''
        Remove detected objects that are too close to the image edges

        Parameters
        ----------
        table               : `astropy.table.Table` object
            Table with the object data

        data                : `numpy.ndarray`
            Image data (2D)

        border              : `integer`, optional
            Distance to the edge of the image where objects may be
            incomplete and should therefore be discarded.
            Default is ``10``.

        condense            : `boolean`, optional
            If True the terminal output will be returned to the calling
            function.
            Default is ``False``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``3``.
    '''
    #   Border range
    hsize = border + 1

    #   Get position data
    x = table['x_fit'].value
    y = table['y_fit'].value

    #   Calculate mask of objects to be removed
    mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] -1 - hsize)))

    outstr = terminal_output.print_terminal(
        np.count_nonzero(np.invert(mask)),
        indent=indent,
        string='{} objects removed because they are too close to the '\
              +'image edges',
        condense=condense,
        )
    if condense:
        return table[mask], outstr
    else:
        return table[mask], ''


def proper_motion_selection(ensemble, tbl, catalog="I/355/gaiadr3",
                            Gmag_limit=20, seplimit=1., sigma=3.,
                            maxiters_sigma=3):
    '''
        Select a sub set of the objects that are close to the median
        proper motion

        Parameters
        ----------
        ensemble            : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        tbl                 : `astropy.table.Table`
            Table with position information

        catalog             : `string`, optional
            Identifier for the catalog to download.
            Default is ``I/350/gaiaedr3``.

        Gmag_limit          : `float`, optional
            Limiting magnitude in the G band. Fainter objects will not be
            downloaded.

        seplimit            : `float`, optional
            Maximal allowed separation between objects in arcsec.
            Default is ``1``.

        sigma               : `float`, optional
            Sigma value used in the sigma clipping of the proper motion
            values.
            Default is ``3``.

        maxiters_sigma      : `integer`, optional
            Maximal number of iteration of the sigma clipping.
            Default is ``3``.
    '''
    #   Get wcs
    w = ensemble.wcs

    #   Convert pixel coordinates to ra & dec
    coords = w.all_pix2world(tbl['x'], tbl['y'], 0)

    #   Create SkyCoord object with coordinates of all objects
    coords_img = SkyCoord(
        coords[0],
        coords[1],
        unit=(u.degree, u.degree),
        frame="icrs"
        )


    ###
    #   Get Gaia data from Vizier
    #
    #   Columns to download
    columns = [
        'RA_ICRS',
        'DE_ICRS',
        'Gmag',
        'Plx',
        'e_Plx',
        'pmRA',
        'e_pmRA',
        'pmDE',
        'e_pmDE',
        'RUWE',
        ]

    #   Define astroquery instance
    v = Vizier(
        columns=columns,
        row_limit=1e6,
        catalog=catalog,
        column_filters={'Gmag':'<'+str(Gmag_limit)},
        )

    #   Get data from the corresponding catalog for the objects in the FOV
    result = v.query_region(
        ensemble.coord,
        radius=ensemble.fov*u.arcmin,
        )

    #   Create SkyCoord object with coordinates of all Gaia objects
    coords_calib = SkyCoord(
            result[0]['RA_ICRS'],
            result[0]['DE_ICRS'],
            unit=(u.degree, u.degree),
            frame="icrs"
            )

    ###
    #   Correlate own objects with Gaia objects
    #
    #   Set maximal separation between objects
    seplimit = seplimit*u.arcsec

    #   Correlate data
    id_img, id_calib, d2ds, d3ds = matching.search_around_sky(
        coords_img,
        coords_calib,
        seplimit,
        )


    ###
    #   Sigma clipping of the proper motion values
    #

    #   Proper motion of the common objects
    pmDE = result[0]['pmDE'][id_calib]
    pmRA = result[0]['pmRA'][id_calib]

    #   Parallax
    parallax = result[0]['Plx'][id_calib].data/1000*u.arcsec

    #   Distance
    distance = parallax.to_value(u.kpc, equivalencies=u.parallax())

    #   Sigma clipping
    sigma_clip_DE = sigma_clip(
        pmDE,
        sigma=sigma,
        maxiters=maxiters_sigma,
        )
    sigma_clip_RA = sigma_clip(
        pmRA,
        sigma=sigma,
        maxiters=maxiters_sigma,
        )

    #   Create mask from sigma clipping
    mask = sigma_clip_RA.mask | sigma_clip_DE.mask

    ###
    #   Make plots
    #
    #   Restrict Gaia table to the common objects
    result_cut = result[0][id_calib][mask]

    #   Convert ra & dec to pixel coordinates
    x_obj, y_obj = w.all_world2pix(
        result_cut['RA_ICRS'],
        result_cut['DE_ICRS'],
        0,
        )

    #   Get image
    image = ensemble.ref_img

    #   Star map
    prepare_and_plot_starmap(
        image,
        tbl=Table(names=['x_fit', 'y_fit'], data=[x_obj, y_obj]),
        rts_pre='img-pmGaia-',
        label='Stars selected according to proper motion',
        )

    #   2D and 3D plot of the proper motion and the distance
    plot.comp_scatter(
        pmRA,
        pmDE,
        'pm_RA * cos(DEC) (mas/yr)',
        'pm_DEC (mas/yr)',
        '_pm_',
        image.outpath.name,
        oneTOone=False,
        )
    plot.D3_scatter(
        [pmRA],
        [pmDE],
        [distance],
        image.outpath.name,
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
        )

    #   Apply mask
    return tbl[id_img][mask]


def region_selection(ensemble, coord, tbl, radius=600.):
    '''
        Select a sub set of the objects that are close to the median
        proper motion

        Parameters
        ----------
        ensemble            : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        coord               : `astropy.coordinates.SkyCoord` object
            Coordinate of the observed object such as a star cluster

        tbl                 : `astropy.table.Table`
            Table with object position information

        radius              : `float`, optional
            Radius around the object in arcsec.
            Default is ``600``.

        Returns
        -------
        tbl                 : `astropy.table.Table`
            Table with object position information

        mask                : `boolean numpy.ndarray`
            Mask that needs to be applied to the table.
    '''
    #   Get wcs
    w = ensemble.wcs

    #   Convert pixel coordinates to ra & dec
    coords = w.all_pix2world(tbl['x'], tbl['y'], 0)

    #   Create SkyCoord object with coordinates of all objects
    coords_img = SkyCoord(
        coords[0],
        coords[1],
        unit=(u.degree, u.degree),
        frame="icrs"
        )

    #   Calculate separation between the coordinates defined in ``coord``
    #   the objects in ``tbl``
    sep = coords_img.separation(coord)

    #   Calculate mask of all object closer than ``radius``
    mask = sep.arcsec <= radius

    #   Limit objects to those within radius
    tbl = tbl[mask]

    #   Plot starmap
    prepare_and_plot_starmap(
        ensemble.ref_img,
        tbl=Table(names=['x_fit', 'y_fit'], data=[tbl['x'], tbl['y']]),
        rts_pre='img-selection-',
        label='Stars selected within {}'.format(radius),
        )

    return tbl, mask


def find_cluster(ensemble, tbl, catalog="I/355/gaiadr3", Gmag_limit=20,
                 seplimit=1., max_distance=6., n_clusters=10,
                 random_state=2, n_neighbors=4, parameter_set=1):
    '''
        Identify cluster in data

        Parameters
        ----------
        ensemble            : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        tbl                 : `astropy.table.Table`
            Table with position information

        catalog             : `string`, optional
            Identifier for the catalog to download.
            Default is ``I/350/gaiaedr3``.

        Gmag_limit          : `float`, optional
            Limiting magnitude in the G band. Fainter objects will not be
            downloaded.

        seplimit            : `float`, optional
            Maximal allowed separation between objects in arcsec.
            Default is ``1``.

        max_distance        : `float`, optional
            Maximal distance of the star cluster.
            Default is ``6.``.

        n_clusters          : `integer`, optional
            Number of expected cluster in the data. Should be at least 2.
            One for the actual cluster and one for the filed stars.
            Default is ``10``.

        random_state        : `integer`, optional
            Random state parameter used to identify the star cluster.
            Default is ``2``.

        n_neighbors         : `integer`, optional
            Neighbors parameter used to identify the star cluster.
            Default is ``4``.

        parameter_set       : `integer`, optional
            Predefined parameter sets can be used.
            Possibilities: ``1``, ``2``, ``3``
            Default is ``1``.

        Returns
        -------
        tbl                 : `astropy.table.Table`
            Table with object position information

        id_img              :

        mask                : `boolean numpy.ndarray`
            Mask that needs to be applied to the table.

        cluster_mask        : `boolean numpy.ndarray`
            Mask that identifies cluster members according to the user
            input.

    '''
    #   Get wcs
    w = ensemble.wcs

    #   Convert pixel coordinates to ra & dec
    coords = w.all_pix2world(tbl['x'], tbl['y'], 0)

    #   Create SkyCoord object with coordinates of all objects
    coords_img = SkyCoord(
        coords[0],
        coords[1],
        unit=(u.degree, u.degree),
        frame="icrs"
        )

    #   Get reference image
    image = ensemble.ref_img


    ###
    #   Get Gaia data from Vizier
    #
    #   Columns to download
    columns = [
        'RA_ICRS',
        'DE_ICRS',
        'Gmag',
        'Plx',
        'e_Plx',
        'pmRA',
        'e_pmRA',
        'pmDE',
        'e_pmDE',
        'RUWE',
        ]

    #   Define astroquery instance
    v = Vizier(
        columns=columns,
        row_limit=1e6,
        catalog=catalog,
        column_filters={'Gmag':'<'+str(Gmag_limit)},
        )

    #   Get data from the corresponding catalog for the objects in the FOV
    result = v.query_region(
        ensemble.coord,
        radius=ensemble.fov*u.arcmin,
        )[0]

    #   Restrict proper motion to Simbad value plus some margin
    customSimbad = Simbad()
    customSimbad.add_votable_fields('pm')

    result_simbad = customSimbad.query_object(ensemble.objname)
    pmra = result_simbad['PMRA'].value[0]
    pmde = result_simbad['PMDEC'].value[0]
    if pmra != '--' and pmde != '--':
        pm_m = 3.
        mask_DE = (result['pmDE'] <= pmde-pm_m) | (result['pmDE'] >= pmde+pm_m)
        mask_RA = (result['pmRA'] <= pmra-pm_m) | (result['pmRA'] >= pmra+pm_m)
        mask = np.invert(mask_DE | mask_RA)
        result = result[mask]

    #   Create SkyCoord object with coordinates of all Gaia objects
    coords_calib = SkyCoord(
            result['RA_ICRS'],
            result['DE_ICRS'],
            unit=(u.degree, u.degree),
            frame="icrs"
            )


    ###
    #   Correlate own objects with Gaia objects
    #
    #   Set maximal separation between objects
    seplimit = seplimit*u.arcsec

    #   Correlate data
    id_img, id_calib, d2ds, d3ds = matching.search_around_sky(
        coords_img,
        coords_calib,
        seplimit,
        )


    ###
    #   Find cluster in proper motion and distance data
    #

    #   Proper motion of the common objects
    pmDE = result['pmDE'][id_calib]
    pmRA = result['pmRA'][id_calib]

    #   Parallax
    parallax = result['Plx'][id_calib].data/1000*u.arcsec

    #   Distance
    distance = parallax.to_value(u.kpc, equivalencies=u.parallax())

    #   Restrict sample to objects closer than 'max_distance'
    #   and remove nans and infs
    if max_distance is not None:
        max_mask = np.invert(distance <= max_distance)
        distance_mask = np.isnan(distance) | np.isinf(distance) | max_mask
    else:
        distance_mask = np.isnan(distance) | np.isinf(distance)

    #   Calculate a mask accounting for NaNs in proper motion and the
    #   distance estimates
    mask = np.invert(pmRA.mask | pmDE.mask | distance_mask)

    #   Convert astropy table to pandas data frame and add distance
    pd_result = result[id_calib].to_pandas()
    pd_result['distance'] = distance
    pd_result = pd_result[mask]

    #   Prepare SpectralClustering object to identify the "cluster" in the
    #   proper motion and distance data sets
    if parameter_set == 1:
        n_clusters = 2
        random_state = 25
        n_neighbors = 20
        affinity='nearest_neighbors'
    elif parameter_set == 2:
        n_clusters = 10
        random_state = 2
        n_neighbors = 4
        affinity='nearest_neighbors'
    elif parameter_set == 3:
        n_clusters = 2
        random_state = 25
        n_neighbors = 20
        affinity='rbf'
    spectral_cluster_model= SpectralClustering(
        #eigen_solver='lobpcg',
        n_clusters=n_clusters,
        random_state=random_state,
        #gamma=2.,
        #gamma=5.,
        n_neighbors=n_neighbors,
        affinity=affinity,
    )

    #   Find "cluster" in the data
    pd_result['cluster'] = spectral_cluster_model.fit_predict(
        pd_result[['pmDE', 'pmRA', 'distance']],
        )

    #   3D plot of the proper motion and the distance
    #   -> select the star cluster by eye
    groups = pd_result.groupby('cluster')
    pmRA_group = []
    pmDE_group = []
    distance_group = []
    for name, group in groups:
        pmRA_group.append(group.pmRA.values)
        pmDE_group.append(group.pmDE.values)
        distance_group.append(group.distance.values)
    plot.D3_scatter(
        pmRA_group,
        pmDE_group,
        distance_group,
        image.outpath.name,
        #color=np.unique(pd_result['cluster']),
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
        string='_3D_cluster_',
        pmra=pmra,
        pmde=pmde,
        )
    plot.D3_scatter(
        pmRA_group,
        pmDE_group,
        distance_group,
        image.outpath.name,
        #color=np.unique(pd_result['cluster']),
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
        string='_3D_cluster_',
        pmra=pmra,
        pmde=pmde,
        display=True,
        )

    #plot.D3_scatter(
        #[pd_result['pmRA']],
        #[pd_result['pmDE']],
        #[pd_result['distance']],
        #image.outpath.name,
        #color=[pd_result['cluster']],
        #name_x='pm_RA * cos(DEC) (mas/yr)',
        #name_y='pm_DEC (mas/yr)',
        #name_z='d (kpc)',
        #string='_3D_cluster_',
        #)

    #   Get user input
    cluter_id, timedOut = timedInput(
            style.bcolors.OKBLUE+
            "   Which one is the correct cluster (id)? "
            +style.bcolors.ENDC,
            timeout=300,
            )
    if timedOut or cluter_id == '':
        cluter_id = 0
    else:
        cluter_id = int(cluter_id)

    #   Calculated mask according to user input
    cluster_mask = pd_result['cluster']==cluter_id

    #   Apply correlation results and masks to the input table
    tbl = tbl[id_img][mask][cluster_mask.values]


    ###
    #   Make star map
    #
    prepare_and_plot_starmap(
        image,
        tbl=tbl,
        x_name='x',
        y_name='y',
        rts_pre='img-pmGaia-distance-cluster',
        label='Cluster members based on proper motion and distance',
        add_image_id=False,
        )

    #   Return table
    return tbl, id_img, mask, cluster_mask.values


def save_mags_ascii(container, tbl, trans=False, ID='', rts='',
                    photo_type='', doadd=True):
    '''
        Save magnitudes as ASCII files

        Parameters
        ----------
        container       : `image.container`
            Image container object with image ensemble objects for each
            filter

        tbl             : `astropy.table.Table`
            Table with magnitudes

        trans           : `boolean`, optional
            If True a magnitude transformation was performed
            Default is ``False``.

        ID              : `string`, optional
            ID of the object
            Default is ``''``.

        rts             : `string`, optional
            Additional string characterizing that should be included in the
            file name.
            Default is ``''``.

        photo_type      : `string`, optional
            Applied extraction method. Possibilities: ePSF or APER`
            Default is ``''``.

        doadd          : `boolean`, optional
            If True the file path will be added to the container object.
            Default is ``True``.
    '''
    #   Check output directories
    outdir = list(container.ensembles.values())[0].outpath
    checks.check_out(
        outdir,
        outdir / 'tables',
        )

    #   Define file name specifier
    if ID != '':
        ID = '_img_'+str(ID)
    if photo_type != '':
        photo_type = '_'+photo_type

    #   Define output formats for the table columns
    formats = {
        'x': '%12.2f',
        'y': '%12.2f',
        "B [mag]": '%12.3f',
        "B_err [mag]": '%12.3f',
        "V [mag]": '%12.3f',
        "V_err [mag]": '%12.3f',
        "B-V [mag]": '%12.3f',
        "B-V_err [mag]": '%12.3f',
        }

    #   Check if ``container`` object contains already entries
    #   for file names/paths. If not add dictionary.
    photo_filepath = getattr(container, 'photo_filepath', None)
    if photo_filepath is None or not isinstance(photo_filepath, dict):
        container.photo_filepath = {}

    #   Set file name
    if trans:
        #   Set file name for file with magnitude transformation
        filename = 'mags_TRANS_calibrated'+photo_type+ID+rts+'.dat'
    else:
        #   File name for file without magnitude transformation
        filename = 'mags_calibrated'+photo_type+ID+rts+'.dat'

    #   Combine to a path
    out_path = outdir / 'tables' / filename

    #   Add to object
    if doadd:
        container.photo_filepath[out_path] = trans

    #   Write file
    tbl.write(
        str(out_path),
        format='ascii',
        overwrite=True,
        formats=formats,
        )


def postprocess_results(img_container, filter_list, ID='', photo_type='',
                        region=False, radius=600, data_cluster=False,
                        pm_median=False, n_clusters=10,
                        max_distance_cluster=6., find_cluster_para_set=1):
    '''
        Restrict results to specific areas of the image and filter by means
        of proper motion and distance using Gaia

        Parameters
        ----------
        img_container           : `image.container`
            Image container object with image ensemble objects for each
            filter

        filter_list             : `list` of `string`
            Filter names

        ID                      : `string`, optional
            ID of the object
            Default is ``''``.

        photo_type              : `string`, optional
            Applied extraction method. Possibilities: ePSF or APER`
            Default is ``''``.

        region                  : `boolean`, optional
            If True the extracted objects will be filtered such that only
            objects with ``radius`` will be returned.
            Default is ``False``.

        radius                  : `float`, optional
            Radius around the object in arcsec.
            Default is ``600``.

        data_cluster            : `boolean`, optional
            If True cluster in the Gaia distance and proper motion data
            will be identified.
            Default is ``False``.

        pm_median               : `boolean`, optional
            If True only the objects that are close to the median
            proper motion will be returned.
            Default is ``False``.

        n_clusters              : `integer`, optional
            Number of expected cluster in the data. Should be at least 2.
            One for the actual cluster and one for the filed stars.
            Default is ``10``.

        max_distance_cluster    : `float`, optional
            Expected maximal distance of the cluster in kpc. Used to
            restrict the parameter space to facilitate an easy
            identification of the star cluster.
            Default is ``6``.

        find_cluster_para_set   : `integer`, optional
            Parameter set used to identify the star cluster in proper
            motion and distance data.
    '''
    #   Do nothing if no post process method were defined
    if not region and not pm_median and not data_cluster:
        return

    #   Get image ensembles
    img_ensembles = img_container.ensembles

    #   Get paths to the tables with the data
    paths = img_container.photo_filepath

    #   Loop over all Tables
    first = True
    for path, trans in paths.items():
        #   Read Table
        tbl = Table.read(path, format='ascii')


        ###
        #   Postprocess data
        #

        #   Extract circular region around a certain object
        #   such as a star cluster
        if region:
            if first:
                tbl, mask_region = region_selection(
                    img_ensembles[filter_list[0]],
                    img_container.coord,
                    tbl,
                    radius=radius
                    )
            else:
                tbl = tbl[mask_region]

        #   Find a cluster in the Gaia data that could be the star cluster
        if data_cluster:
            if first:
                tbl, img_id_cluster, mask_cluster, mask_objects = find_cluster(
                    img_ensembles[filter_list[0]],
                    tbl,
                    n_clusters=n_clusters,
                    max_distance=max_distance_cluster,
                    parameter_set=find_cluster_para_set,
                    )
            else:
                tbl = tbl[img_id_cluster][mask_cluster][mask_objects]

        #   Clean objects according to proper motion (Gaia)
        if pm_median:
            if first:
                tbl, img_id_pm, mask_pm = proper_motion_selection(
                    img_ensembles[filter_list[0]],
                    tbl,
                    )
            else:
                tbl = tbl[img_id_pm][mask_pm]

        ###
        #   Save results as ASCII files
        #
        save_mags_ascii(
            img_container,
            tbl,
            trans=trans,
            ID=ID,
            rts='_selection',
            photo_type=photo_type,
            doadd=False,
            )

        first = False
