############################################################################
#                               Libraries                                  #
############################################################################

import sys

import numpy as np

# from uncertainties import unumpy

from pytimedinput import timedInput

from astropy.table import Table
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy.coordinates import SkyCoord, matching
from astropy.timeseries import TimeSeries
from astropy.modeling import models, fitting
from astropy import uncertainty as unc
import astropy.units as u
from astropy import wcs

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from photutils.utils import ImageDepth

from regions import (
    # RectangleSkyRegion,
    # RectanglePixelRegion,
    PixCoord,
    CirclePixelRegion,
    Regions,
)

from sklearn.cluster import SpectralClustering

import multiprocessing as mp

import scipy.optimize as optimization

from .. import utilities as base_aux

from .. import checks, style, terminal_output, calibration_data

from . import plot

import typing
if typing.TYPE_CHECKING:
    from . import analyze


############################################################################
#                           Routines & definitions                         #
############################################################################


def err_prop(*args):
    """
        Calculate error propagation

        Parameters
        ----------
        args        : `list` of `float`s or `numpy.ndarray`s
            Sources of error that should be added up

        Returns
        -------
        sum_error   : `float` or `numpy.ndarray`
            Accumulated error
    """
    #   Adding up the errors
    for i, x in enumerate(args):
        if i == 0:
            sum_error = x
        else:
            sum_error = np.sqrt(np.square(sum_error) + np.square(x))
    return sum_error


def mk_magnitudes_table_and_array(
        observation: 'analyze.Observation', filter_list, photometry_column_keyword):
    """
        Create and export astropy table with object positions and magnitudes

        Parameters
        ----------
        observation
            Container object with image series objects for each filter

        filter_list                 : `list` of `string`
            Filter

        photometry_column_keyword   : `string`
            String used to identify the magnitude column in the
            photometry tables

        Returns
        -------
        tbl                         : `astropy.table.Table`
            Table with CMD data

        stacked_magnitudes          : `dict` of `dict` of `numpy.ndarray`
            Array with magnitudes and magnitude errors of all images in an
            image series
    """
    #   Dictionary for stacked magnitudes
    stacked_magnitudes = {}

    #   Get object indices, X & Y pixel positions and wcs
    #   Assumes that the image series are already correlated
    image_wcs = observation.image_series_dict[filter_list[0]].wcs
    index_objects = observation.image_series_dict[filter_list[0]].image_list[0].photometry['id']
    x_positions = observation.image_series_dict[filter_list[0]].image_list[0].photometry['x_fit']
    y_positions = observation.image_series_dict[filter_list[0]].image_list[0].photometry['y_fit']

    # Make CMD table
    tbl = Table(
        names=['i', 'x', 'y', ],
        data=[
            np.intc(index_objects),
            x_positions,
            y_positions,
        ]
    )

    #   Convert Pixel to sky coordinates
    sky = image_wcs.pixel_to_world(x_positions, y_positions)

    #   Add sky coordinates to table
    tbl['ra (deg)'] = sky.ra
    tbl['dec (deg)'] = sky.dec

    #   Add magnitude columns to table
    for filter_ in filter_list:
        #   Get image list
        image_series = observation.image_series_dict[filter_]
        image_list = image_series.image_list

        #   Lists for magnitudes and errors
        magnitude_list = []
        magnitude_error_list = []

        for image_id, image in enumerate(image_list):
            photometry_table = image.photometry
            magnitudes = photometry_table[photometry_column_keyword]
            magnitude_errors = photometry_table[f'{photometry_column_keyword}_unc']

            #   Add to table
            tbl.add_columns(
                [
                    magnitudes,
                    magnitude_errors,
                ],
                names=[
                    f'{filter_} ({image_id})',
                    f'{filter_}_err ({image_id})',
                ]
            )

            #   Add magnitudes and error to corresponding lists
            magnitude_list.append(magnitudes)
            magnitude_error_list.append(magnitude_errors)

        #   Make numpy array with magnitudes from all images in an imaging
        #   series and add this to the magnitude dictionary
        stacked_magnitudes[filter_] = {
            'values':np.stack(magnitude_list),
            'errors':np.stack(magnitude_error_list)
        }

    # for ids in filter_image_ids:
    #     tbl.add_columns(
    #         [
    #             magnitudes[filter_list[ids[0]]][ids[1]].pdf_median(),
    #             magnitudes[filter_list[ids[0]]][ids[1]].pdf_std(),
    #         ],
    #         names=[
    #             f'{filter_list[ids[0]]} ({ids[1]})',
    #             f'{filter_list[ids[0]]}_err ({ids[1]})',
    #         ]
    #     )

    #   Sort table
    tbl = tbl.group_by(
        f'{filter_list[0]} (0)'
    )

    return tbl, stacked_magnitudes


#   TODO: Check where this function is used and whether it is safe to rename the parameters.
def find_wcs(image_series: 'analyze.ImageSeries', reference_image_id=None, method='astrometry',
             rmcos=False, path_cos=None, x=None, y=None,
             force_wcs_determ=False, indent=2):
    """
        Meta function for finding image WCS

        Parameters
        ----------
        image_series
            Image class with all images taken in a specific filter

        reference_image_id  : `integer`, optional
            ID of the reference image
            Default is ``None``.

        method              : `string`, optional
            Method to use for the WCS determination
            Options: 'astrometry', 'astap', or 'twirl'
            Default is ``astrometry``.

        rmcos               : `boolean`, optional
            If True the function assumes that the cosmic ray reduction
            function was run before this function
            Default is ``False``.

        path_cos            : `string`
            Path to the image in case 'rmcos' is True
            Default is ``None``.

        x, y                : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    """
    if reference_image_id is not None:
        #   Image
        img = image_series.image_list[reference_image_id]

        #   Test if the image contains already a WCS
        cal_wcs, wcs_file = base_aux.check_wcs_exists(img)

        if not cal_wcs or force_wcs_determ:
            #   Calculate WCS -> astrometry.net
            if method == 'astrometry':
                image_series.set_wcs(
                    base_aux.find_wcs_astrometry(
                        img,
                        cosmic_rays_removed=rmcos,
                        path_cosmic_cleaned_image=path_cos,
                        indent=indent,
                    )
                )

            #   Calculate WCS -> ASTAP program
            elif method == 'astap':
                image_series.set_wcs(
                    base_aux.find_wcs_astap(img, indent=indent)
                )

            #   Calculate WCS -> twirl library
            elif method == 'twirl':
                if x is None or y is None:
                    raise RuntimeError(
                        f"{style.Bcolors.FAIL} \nException in find_wcs(): '"
                        f"\n'x' or 'y' is None -> Exit {style.Bcolors.ENDC}"
                    )
                image_series.set_wcs(
                    base_aux.find_wcs_twirl(img, x, y, indent=indent)
                )
            #   Raise exception
            else:
                raise RuntimeError(
                    f"{style.Bcolors.FAIL} \nException in find_wcs(): '"
                    f"\nWCS method not known -> Supplied method was {method}"
                    f"{style.Bcolors.ENDC}"
                )
        else:
            image_series.set_wcs(extract_wcs(wcs_file))
    else:
        for i, img in enumerate(image_series.image_list):
            #   Test if the image contains already a WCS
            cal_wcs = base_aux.check_wcs_exists(img)

            if not cal_wcs or force_wcs_determ:
                #   Calculate WCS -> astrometry.net
                if method == 'astrometry':
                    w = base_aux.find_wcs_astrometry(
                        img,
                        cosmic_rays_removed=rmcos,
                        path_cosmic_cleaned_image=path_cos,
                        indent=indent,
                    )

                #   Calculate WCS -> ASTAP program
                elif method == 'astap':
                    w = base_aux.find_wcs_astap(img, indent=indent)

                #   Calculate WCS -> twirl library
                elif method == 'twirl':
                    if x is None or y is None:
                        raise RuntimeError(
                            f"{style.Bcolors.FAIL} \nException in "
                            "find_wcs(): ' \n'x' or 'y' is None -> Exit"
                            f"{style.Bcolors.ENDC}"
                        )
                    w = base_aux.find_wcs_twirl(img, x, y, indent=indent)

                #   Raise exception
                else:
                    raise RuntimeError(
                        f"{style.Bcolors.FAIL} \nException in find_wcs(): '"
                        "\nWCS method not known -> Supplied method was "
                        f"{method} {style.Bcolors.ENDC}"
                    )
            else:
                w = wcs.WCS(fits.open(img.path)[0].header)

            if i == 0:
                image_series.set_wcs(w)


def extract_wcs(wcs_path, image_wcs=None, rm_cosmics=False, filters=None):
    """
        Load wcs from FITS file

        Parameters
        ----------
        wcs_path         : `string`
            Path to the image with the WCS or path to the directory that
            contains this image

        image_wcs       : `string`, optional
            WCS image name. Needed in case `wcs_path` is only the path to
            the image directory.
            Default is ``None``.

        rm_cosmics      : `boolean`, optional
            If True cosmic rays will be removed.
            Default is ``False``.

        filters         : `list` of `string`, optional
            Filter list
            Default is ``None``.
    """
    #   Open the image with the WCS solution
    if image_wcs is not None:
        #   TODO: Check whether it is better to remove the following
        if rm_cosmics:
            if filters is None:
                raise Exception(
                    f"{style.Bcolors.FAIL} \nException in extract_wcs(): '"
                    "\n'rmcos=True' buit no 'filters' given -> Exit"
                    f"{style.Bcolors.ENDC}"
                )
            basename = f'img_cut_{filters[0]}_lacosmic'
        else:
            basename = image_wcs.split('/')[-1].split('.')[0]
        hdu_list = fits.open(f'{wcs_path}/{basename}.new')
    else:
        hdu_list = fits.open(wcs_path)

    #   Extract the WCS
    w = wcs.WCS(hdu_list[0].header)

    return w


def mk_time_series(observation_times, magnitudes, filter_, object_id):
    """
        Make a time series object

        Parameters
        ----------
        observation_times   : `astropy.time.Time`
            Observation times

        magnitudes          : `list` of `astropy.uncertainty.core.QuantityDistribution`
            Magnitudes and uncertainties

        filter_             : `string`
            Filter

        object_id           : `integer`
            ID/Number of the object for with the time series should be
            created

        Returns
        -------
        ts                  : `astropy.timeseries.TimeSeries`
    """
    #   Get magnitude and error
    mags_obj = magnitudes[filter_]['values'][:, object_id]
    errs_obj = magnitudes[filter_]['errors'][:, object_id]

    #   Make time series and use reshape to get a justified array
    #   TODO: Check if the reshape below is necessary
    ts = TimeSeries(
        time=observation_times,
        data={
            filter_: mags_obj.reshape(mags_obj.size, ) * u.mag,
            filter_ + '_err': errs_obj.reshape(errs_obj.size, ) * u.mag,
        }
    )
    return ts


def lin_func(x, a, b):
    """
        Linear function
    """
    return a + b * x


def fit_curve(fit_func, x, y, x0, sigma):
    """
        Fit curve with supplied fit function

        Parameters
        ----------
        fit_func        : `function`
            Function used in the fitting process

        x               : `numpy.ndarray`
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
    """

    #   Fit curve
    if np.any(sigma == 0.):
        para, coma = optimization.curve_fit(
            fit_func, 
            np.ravel(x), 
            np.ravel(y), 
            x0,
        )
    else:
        para, coma = optimization.curve_fit(fit_func, x, y, x0, sigma)
    a = para[0]
    b = para[1]
    a_err = coma[0, 0]
    b_err = coma[1, 1]

    return a, a_err, b, b_err


def fit_data_one_d(x, y, order):
    """
        Fit polynomial to the provided data.

        Parameters
        ----------
        x               : `numpy.ndarray`
            X data values

        y               : `numpy.ndarray`
            Y data values

        order           : `integer`
            Order of the polynomial to be fitted to the data
    """
    #   Set model
    model = models.Polynomial1D(degree=order)

    #   Set fitter
    fitter_poly = fitting.LevMarLSQFitter()

    #   Fit data
    if np.all(x == 0.):
        fit_poly = None
    else:
        fit_poly = fitter_poly(
            model,
            x,
            y,
        )

    return fit_poly


def flux_to_magnitudes(flux, flux_error):
    """
        Calculate magnitudes from flux

        Parameters
        ----------
        flux            : `numpy.ndarray` or `astropy.table.Column`
            Flux values

        flux_error      : `numpy.ndarray` or `astropy.table.Column`
            Flux uncertainties

        Returns
        -------
        mags            : `astropy.uncertainty.core.QuantityDistribution`
            Numpy structured array containing magnitudes and corresponding
            errors
    """
    #   Sanitize input parameters
    #   TODO: Check if the following is necessary
    # if np.ma.isMaskedArray(flux):
    #     flux = flux.filled()
    # if np.ma.isMaskedArray(flux_error):
    #     flux_error = flux_error.filled()

    #   Calculate magnitudes
    magnitudes = -2.5 * np.log10(flux)
    magnitudes_error = -2.5 * flux_error / flux

    return magnitudes, magnitudes_error


def find_transformation_coefficients(filter_list, tsc_parameter_dict, filter_,
                                     camera, verbose=False, indent=2):
    """
        Find the position of the filter from the 'tsc_parameter_dict'
        dictionary with reference to 'filter_list'

        Parameters
        ----------
        filter_list         : `list` - `string`
            List of available filter, e.g., ['U', 'B', 'V', ...]

        tsc_parameter_dict  : `dictionary` - `string`:`dictionary`
            Magnitude transformation coefficients for different cameras.
            Keys:  camera identifier

        filter_             : `string`
            Filter for which calibration data will be selected

        camera              : `string`
            Camera used

        verbose             : `boolean`, optional
            If ``True`` additional information will be printed to the console.
            Default is ``False``.

        indent              : `integer`, optional
            Indentation for the console output
            Default is ``2``.

        Returns
        -------
        variable_1          : `dictionary`
            Entry from dictionary 'in_dict' corresponding to filter 'filter_'
    """
    #   Initialize list of bools
    cam_bools = []

    #   Loop over outer dictionary: 'in_dict'
    for key_outer, value_outer in tsc_parameter_dict.items():
        #   Check if calibration data fits to the camera
        if camera == key_outer:
            #   Loop over inner dictionary
            for key_inner, value_inner in value_outer.items():
                #   Check if calibration data is available for the current
                #   filter 'filter_'.
                if filter_ == key_inner:
                    f1 = value_inner['Filter 1']
                    f2 = value_inner['Filter 2']
                    #   Check if the filter used to calculate the
                    #   calibration data is also available in the filter
                    #   list 'filter_list'
                    if f1 == filter_list[0] and f2 == filter_list[1]:
                        return value_inner
                    else:
                        if verbose:
                            terminal_output.print_to_terminal(
                                'Magnitude transformation coefficients'
                                ' do not apply. Wrong filter '
                                'combination: {f1} & {f2} vs. {filter_list}',
                                indent=indent,
                                style_name='WARNING',
                            )

            cam_bools.append(True)
        else:
            cam_bools.append(False)

    if not any(cam_bools):
        terminal_output.print_to_terminal(
            f'Determined camera ({camera}) not consistent with the'
            ' one given in the dictionary with the transformation'
            ' coefficients.',
            indent=indent,
            style_name='WARNING',
        )

    return None


def check_variable(filename, filetype, filter_1, filter_2, iso_column_type,
                   iso_column):
    """
        Check variables and set defaults for CMDs and isochrone plots

        This function exists for backwards compatibility.

        Parameters
        ----------
        filename            : `string`
            Specified file name - can also be empty -> set default


        filetype            : `string`
            Specified file type - can also be empty -> set default

        filter_1            : `string`
            First filter

        filter_2            : `string`
            Second filter

        iso_column_type     : `dictionary`
            Keys = filter - Values = type

        iso_column          : `dictionary`
            Keys = filter - Values = column
    """

    filename, filetype = check_variable_apparent_cmd(
        filename,
        filetype,
    )

    check_variable_absolute_cmd(
        [filter_1, filter_2],
        iso_column_type,
        iso_column,
    )

    return filename, filetype


def check_variable_apparent_cmd(filename, filetype):
    """
        Check variables and set defaults for CMDs and isochrone plots

        Parameters
        ----------
        filename                : `string`
            Specified file name - can also be empty -> set default

        filetype                : `string`
            Specified file type - can also be empty -> set default
    """
    #   Set figure type
    if filename == "?" or filename == "":
        terminal_output.print_to_terminal(
            '[Warning] No filename given, us default (cmd)',
            indent=1,
            style_name='WARNING',
        )
        filename = 'cmd'

    if filetype == '?' or filetype == '':
        terminal_output.print_to_terminal(
            '[Warning] No filetype given, use default (pdf)',
            indent=1,
            style_name='WARNING',
        )
        filetype = 'pdf'

    #   Check if file type is valid and set default
    filetype_list = ['pdf', 'png', 'eps', 'ps', 'svg']
    if filetype not in filetype_list:
        terminal_output.print_to_terminal(
            '[Warning] Unknown filetype given, use default instead (pdf)',
            indent=1,
            style_name='WARNING',
        )
        filetype = 'pdf'

    # #   Check if calibration parameter is consistent with the number of
    # #   filter
    # if zero_points_dict:
    #     if len(filter_list) != len(zero_points_dict):
    #         if len(filter_list) > len(zero_points_dict):
    #             terminal_output.print_to_terminal(
    #                 "[Error] More filter ('filter') specified than zero"
    #                 " points ('zero_points_dict')",
    #                 indent=1,
    #                 style_name='ERROR',
    #             )
    #             sys.exit()
    #         else:
    #             terminal_output.print_to_terminal(
    #                 "[Error] More zero points ('zero_points_dict') "
    #                 "specified than filter ('filter')",
    #                 indent=1,
    #                 style_name='ERROR',
    #             )
    #             sys.exit()

    # #   Valid filter combinations
    # valid_filter_combination = {
    #     'U': 'B',
    #     'B': 'V',
    #     'V': 'R',
    #     'R': 'I',
    #     'H': 'J',
    #     'J': 'K',
    # }
    # if filter_1 in valid_filter_combination.keys():
    #     second_filter = valid_filter_combination[filter_1]
    #     if second_filter in filter_list:
    #         return filename, filetype, second_filter
    #     else:
    #         index_filter_1 = filter_list.index(filter_1)
    #         if index_filter_1 + 1 < len(filter_list):
    #             return filename, filetype, filter_list[index_filter_1 + 1]
    #
    # return filename, filetype, False
    return filename, filetype


def check_variable_absolute_cmd(filter_list, iso_column_type,
                                iso_column):
    """
        Check variables and set defaults for CMDs and isochrone plots

        Parameters
        ----------
        filter_list           : `list` of `string`
            Filter list

        iso_column_type       : `dictionary`
            Keys = filter - Values = type

        iso_column            : `dictionary`
            Keys = filter - Values = column
    """
    #   Check if the column declaration for the isochrones fits to the
    #   specified filter
    for filter_ in filter_list:
        if filter_ not in iso_column_type.keys():
            terminal_output.print_to_terminal(
                f"[Error] No entry for filter {filter_} specified in "
                f"'ISOcolumntype'",
                indent=1,
                style_name='FAIL',
            )
            sys.exit()
        if filter_ not in iso_column.keys():
            terminal_output.print_to_terminal(
                f"[Error] No entry for filter {filter_} specified in"
                " 'ISOcolumn'",
                indent=1,
                style_name='FAIL',
            )
            sys.exit()


class Executor:
    """
        Class that handles the multiprocessing, using apply_async.
        -> allows for easy catch of exceptions
    """

    def __init__(self, process_num):
        #   Init multiprocessing pool
        self.pool = mp.Pool(process_num, maxtasksperchild=6)
        #   Init variables
        self.res = []
        self.err = None

    def collect_results(self, result):
        """
            Uses apply_async's callback to set up a separate Queue
            for each process
        """
        #   Catch all results
        self.res.append(result)

    def callback_error(self, e):
        """
            Handles exceptions by apply_async's error callback
        """
        terminal_output.print_to_terminal(
            'Exception detected: Try to terminate the multiprocessing Pool',
            style_name='ERROR',
        )
        terminal_output.print_to_terminal(
            f'The exception is: {e}',
            style_name='ERROR',
        )
        #   Terminate pool
        self.pool.terminate()
        #   Raise exceptions
        self.err = e
        raise e

    def schedule(self, function, args, kwargs):
        """
            Call to apply_async
        """
        self.pool.apply_async(
            function,
            args,
            kwargs,
            callback=self.collect_results,
            error_callback=self.callback_error
        )

    def wait(self):
        """
            Close pool and wait for completion
        """
        self.pool.close()
        self.pool.join()


def mk_ds9_region(x_pixel_positions, y_pixel_positions, pixel_radius, filename,
                  wcs_object):
    """
        Make and write a ds9 region file

        Is this function still useful?


        Parameters
        ----------
        x_pixel_positions   : `numpy.ndarray`
            X coordinates in pixel

        y_pixel_positions   : `numpy.ndarray`
            Y coordinates in pixel

        pixel_radius        : `float`
            Radius in pixel

        filename            : `string`
            File name

        wcs_object          : `astropy.wcs.WCS`
            WCS information
    """
    #   Create the region
    c_regs = []

    for x_i, y_i in zip(x_pixel_positions, y_pixel_positions):
        #   Make a pixel coordinates object
        center = PixCoord(x=x_i, y=y_i)

        #   Create the region
        c = CirclePixelRegion(center, radius=pixel_radius)

        #   Append region and convert to sky coordinates
        c_regs.append(c.to_sky(wcs_object))

    #   Convert to Regions that contain all individual regions
    reg = Regions(c_regs)

    #   Write the region file
    reg.write(filename, format='ds9', overwrite=True)


def prepare_and_plot_starmap(image, terminal_logger=None, tbl=None,
                             x_name='x_fit', y_name='y_fit', rts_pre='image',
                             label='Stars with photometric extractions',
                             add_image_id=True):
    """
        Creates a star map using information from an Image object

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

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
    """
    #   Get table, data, filter, & object name
    if tbl is None:
        tbl = image.photometry
    data = image.get_data()
    filter_ = image.filter_
    # name = image.object_name

    #   Prepare table
    n_stars = len(tbl)
    tbl_xy = Table(
        names=['id', 'xcentroid', 'ycentroid'],
        data=[np.arange(0, n_stars), tbl[x_name], tbl[y_name]],
    )

    #   Prepare string for file name
    if add_image_id:
        rts_pre += f': {image.pd}'

    #   Plot star map
    plot.starmap(
        image.out_path.name,
        data,
        filter_,
        tbl_xy,
        label=label,
        rts=rts_pre,
        # name_object=name,
        wcs_image=image.wcs,
        terminal_logger=terminal_logger,
    )


def prepare_and_plot_starmap_from_observation(
        observation: 'analyze.Observation', filter_list):
    """
        Creates a star map using information from an observation container

        Parameters
        ----------
        observation     
            Container object with image series objects for each filter

        filter_list       : `list` of `strings`
            List with filter names
    """
    terminal_output.print_to_terminal(
        "Plot star maps with positions from the final correlation",
        indent=1,
    )

    for filter_ in filter_list:
        # if filter_ == filter_list[0]:
        #     rts = f'{filter_list[1]} [final version]'
        # else:
        #     rts = f'{filter_list[0]} [final version]'
        rts = 'final version'

        #   Get reference image
        image = observation.image_series_dict[filter_].reference_image

        #   Using multiprocessing to create the plot
        p = mp.Process(
            target=plot.starmap,
            args=(
                image.out_path.name,
                image.get_data(),
                filter_,
                image.photometry,
            ),
            kwargs={
                'rts': rts,
                'label': f'Stars identified in {filter_list[0]} and '
                         f'{filter_list[1]} filter',
                # 'name_object': image.object_name,
                'wcs': image.wcs,
            }
        )
        p.start()
    terminal_output.print_to_terminal('')


def prepare_and_plot_starmap_from_image_series(
        image_series: 'analyze.ImageSeries', calib_xs,
        calib_ys,
        plot_reference_only=True):
    """
        Creates a star map using information from an image series

        Parameters
        ----------
        image_series
            Image image_series class object

        calib_xs        : `numpy.ndarray` or `list` of `floats`
            Position of the calibration objects on the image in pixel
            in X direction

        calib_ys        : `numpy.ndarray` or `list`  of `floats`
            Position of the calibration objects on the image in pixel
            in Y direction

        plot_reference_only       : `boolean`, optional
            If True only the starmap for the reference image will
            be created.
            Default is ``True``.
    """
    terminal_output.print_to_terminal(
        "Plot star map with the objects identified on all images",
        indent=1,
    )

    #   Get image IDs, IDs of the objects, and pixel coordinates
    img_ids = image_series.get_image_ids()

    #   Make new table with the position of the calibration stars
    tbl_xy_calib = Table(
        names=['xcentroid', 'ycentroid'],
        data=[[calib_xs], [calib_ys]]
    )

    #   Make the plot using multiprocessing
    for j, image_id in enumerate(img_ids):
        if plot_reference_only and j != image_series.reference_image_id:
            continue
        p = mp.Process(
            target=plot.starmap,
            args=(
                image_series.out_path.name,
                image_series.image_list[j].get_data(),
                image_series.filter_,
                image_series.image_list[j].photometry,
            ),
            kwargs={
                'tbl_2': tbl_xy_calib,
                'rts': f'image: {image_id}, final version',
                'label': 'Stars identified in all images',
                # 'label_2': 'Calibration stars',
                'label_2': 'Variable object',
                # 'name_object': image_series.object_name,
                'wcs': image_series.wcs,
            }
        )
        p.start()
        terminal_output.print_to_terminal('')


def calibration_check_plots(
        filter_, out_dir, image_id, filter_list,
        color_observed, color_literature, ids_calibration_stars,
        literature_magnitudes, magnitudes, uncalibrated_magnitudes,
        color_observed_err=None, color_literature_err=None,
        literature_magnitudes_err=None, magnitudes_err=None,
        uncalibrated_magnitudes_err=None, plot_sigma_switch=False,
        multiprocessing: bool = True):
    """
        Useful plots to check the quality of the calibration process.

        Parameters
        ----------
        filter_                     : `string`
            Filter used

        out_dir                     : `string`
            Output directory

        image_id                    : `integer`
                Expression characterizing the plot

        filter_list                 : `list` - `string`
            Filter list

        mask:                       : `numpy.ndarray`
            Mask of stars that should be excluded

        color_observed              : `numpy.ndarray` - `numpy.float64`
            Instrument color of the calibration stars

        color_literature            : `numpy.ndarray` - `numpy.float64`
            Literature color of the calibration stars

        ids_calibration_stars       : `numpy.ndarray`
            IDs of the calibration stars

        literature_magnitudes       : `numpy.ndarray`
            Literature magnitudes of the objects that are used in the
            calibration process

        magnitudes                  : `numpy.ndarray`
            Magnitudes of all observed objects

        uncalibrated_magnitudes     : `numpy.ndarray`
            Magnitudes of all observed objects but not calibrated yet

        color_observed_err          : `numpy.ndarray' or ``None``, optional
            Uncertainty in the instrument color of the calibration stars

        color_literature_err        : `numpy.ndarray' or ``None``, optional
            Uncertainty in the literature color of the calibration stars

        literature_magnitudes_err   : `numpy.ndarray`
            Uncertainty in the literature magnitudes of the objects that are
            used in the calibration process

        magnitudes_err              : `numpy.ndarray` or `None`, optional
            Uncertainty in the magnitudes of the observed objects

        uncalibrated_magnitudes_err : `numpy.ndarray` or `None`, optional
            Uncertainty in the uncalibrated magnitudes of the observed objects

        plot_sigma_switch           : `boolean`, optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        multiprocessing
            If ``True'', multicore processing is allowed, otherwise not.
    """
    #   TODO: Cleanup & add new plots
    #   Comparison observed vs. literature magnitudes
    if multiprocessing:
        p = mp.Process(
            target=plot.scatter,
            args=(
                [magnitudes],
                f'{filter_}_calibration [mag]',
                [uncalibrated_magnitudes],
                f'{filter_}_no-calibration [mag]',
                f'mag-cali_mags_{filter_}_img_{image_id}',
                out_dir,
            ),
            kwargs={
                # 'name_object': name_object,
                'x_errors': [magnitudes_err],
                'y_errors': [uncalibrated_magnitudes_err],
            }
        )
        p.start()
    else:
        plot.scatter(
            [magnitudes],
            f'{filter_}_calibration [mag]',
            [uncalibrated_magnitudes],
            f'{filter_}_no-calibration [mag]',
            f'mag-cali_mags_{filter_}_img_{image_id}',
            out_dir,
            # name_object=name_object,
            x_errors=[magnitudes_err],
            y_errors=[uncalibrated_magnitudes_err],
        )

    #   Illustration of sigma clipping on calibration magnitudes
    # if plot_sigma_switch:
        #   Make fit
        # fit = fit_data_one_d(
        #     uncalibrated_magnitudes[ids_calibration_stars][mask],
        #     literature_magnitudes[mask],
        #     1,
        # )
        #
        # p = mp.Process(
        #     target=plot.scatter,
        #     args=(
        #         [
        #             uncalibrated_magnitudes[ids_calibration_stars],
        #             uncalibrated_magnitudes[ids_calibration_stars][mask]
        #         ],
        #         f'{filter_}_measured [mag]',
        #         [literature_magnitudes, literature_magnitudes[mask]],
        #         f'{filter_}_literature [mag]',
        #         f'mags_sigma_{filter_}_img_{image_id}',
        #         out_dir,
        #     ),
        #     kwargs={
        #         'name_object': name_object,
        #         'fits': [None, fit],
        #         'x_errors': [
        #             uncalibrated_magnitudes_err[ids_calibration_stars],
        #             uncalibrated_magnitudes_err[ids_calibration_stars][mask]
        #         ],
        #         'y_errors': [
        #             literature_magnitudes_err,
        #             literature_magnitudes_err[mask]
        #         ],
        #         'dataset_label': [
        #             'without sigma clipping',
        #             'with sigma clipping',
        #         ],
        #     }
        # )
        # p.start()

        # #   Make fit for test purposes TODO: Check that the settings are correct.
        # fit = fit_data_one_d(
        #     color_literature[mask],
        #     color_observed[mask],
        #     1,
        # )
        # p = mp.Process(
        #     target=plot.scatter,
        #     args=(
        #         [color_literature, color_literature[mask]],
        #         f'{filter_list[0]}-{filter_list[1]}_literature [mag]',
        #         [color_observed, color_observed[mask]],
        #         f'{filter_list[0]}-{filter_list[1]}_measured [mag]',
        #         f'color_sigma_{filter_}_img_{image_id}',
        #         out_dir,
        #     ),
        #     kwargs={
        #         'name_object': name_object,
        #         'x_errors': [color_literature_err, color_literature_err[mask]],
        #         'y_errors': [color_observed_err, color_observed_err[mask]],
        #         'dataset_label': [
        #             'without sigma clipping',
        #             'with sigma clipping',
        #         ],
        #         'fits': [fit, fit],
        #     }
        # )
        # p.start()

        # p = mp.Process(
        #     target=plot.scatter,
        #     args=(
        #         [color_literature, color_literature[mask]],
        #         f'{filter_list[id_filter_1]}-{filter_list[id_filter_2]}_literature [mag]',
        #         [color_literature - color_observed, color_literature[mask] - color_observed[mask]],
        #         f'{filter_list[id_filter_1]}-{filter_list[id_filter_2]}_literature - '
        #         f'{filter_list[id_filter_1]}-{filter_list[id_filter_2]}_measured [mag]',
        #         f'delta_color_sigma_{filter_}_img_{image_id}',
        #         out_dir,
        #     ),
        #     kwargs={
        #         'name_object': name_object,
        #         'x_errors': [color_lit_err, color_lit_err[mask]],
        #         'y_errors': [
        #             err_prop(color_fit_err, color_lit_err),
        #             err_prop(color_fit_err[mask], color_lit_err[mask])
        #         ],
        #         'dataset_label': [
        #             'without sigma clipping',
        #             'with sigma clipping',
        #         ],
        #     }
        # )
        # p.start()

        # p = mp.Process(
        #     target=plot.scatter,
        #     args=(
        #         [color_literature, color_literature[mask]],
        #         f'{filter_list[0]}-{filter_list[1]}_literature [mag]',
        #         [
        #             2 * literature_magnitudes - color_literature -
        #             2 * magnitudes[ids_calibration_stars] + color_observed,
        #             2 * literature_magnitudes[mask] - color_literature[mask] -
        #             2 * magnitudes[ids_calibration_stars][mask] + color_observed[mask]
        #         ],
        #         f'{filter_list[0]} + {filter_list[1]}_measured [mag]'
        #         f' - {filter_list[0]} - {filter_list[1]}_literature',
        #         f'delta_magnitudes_sigma_{filter_}_img_{image_id}',
        #         out_dir,
        #     ),
        #     kwargs={
        #         'name_object': name_object,
        #         'x_errors': [color_literature_err, color_literature_err[mask]],
        #         'y_errors': [
        #             err_prop(color_observed_err, color_literature_err),
        #             err_prop(color_observed_err[mask], color_literature_err[mask])
        #         ],
        #         'dataset_label': [
        #             'without sigma clipping',
        #             'with sigma clipping',
        #         ],
        #     }
        # )
        # p.start()

    # #   Difference between literature values and calibration results
    # #   TODO: Add image ID to plot file name
    # p = mp.Process(
    #     target=plot.scatter,
    #     args=(
    #         [literature_magnitudes, literature_magnitudes[mask]],
    #         f'{filter_}_literature [mag]',
    #         [
    #             magnitudes[ids_calibration_stars] - literature_magnitudes,
    #             magnitudes[ids_calibration_stars][mask] - literature_magnitudes[mask]
    #         ],
    #         f'{filter_}_observed - {filter_}_literature [mag]',
    #         'magnitudes_literature-vs-observed',
    #         out_dir,
    #     ),
    #     kwargs={
    #         'x_errors': [literature_magnitudes_err, literature_magnitudes_err[mask]],
    #         'y_errors': [
    #             err_prop(magnitudes_err[ids_calibration_stars], literature_magnitudes_err),
    #             err_prop(magnitudes_err[ids_calibration_stars][mask], literature_magnitudes_err[mask]),
    #         ],
    #         'dataset_label': [
    #             'without sigma clipping',
    #             'with sigma clipping',
    #         ],
    #     },
    # )
    # p.start()


def derive_limiting_magnitude(
        observation: 'analyze.Observation', filter_list, reference_img,
        aperture_radius=4., radii_unit='arcsec',
        indent=1):
    """
        Determine limiting magnitude

        Parameters
        ----------
        observation
            Container object with image series objects for each filter

        filter_list         : `list` of `strings`
            List with filter names

        reference_img       : `integer`, optional
            ID of the reference image
            Default is ``0``.

        aperture_radius     : `float`, optional
            Radius of the aperture used to derive the limiting magnitude
            Default is ``4``.

        radii_unit          : `string`, optional
            Unit of the radii above. Permitted are ``pixel`` and ``arcsec``.
            Default is ``arcsec``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    """
    #   Get image series
    image_series_dict = observation.image_series_dict

    #   Get magnitudes of reference image
    for i, filter_ in enumerate(filter_list):
        #   Get image series
        image_series = image_series_dict[filter_]

        #   Get reference image
        image = image_series.image_list[reference_img]

        #   Get object position and magnitudes
        photo = image_series.image_list[reference_img].photometry

        try:
            magnitude_type = 'mag_cali_trans'
            tbl_mag = photo.group_by(magnitude_type)
        except KeyError:
            magnitude_type = 'mag_cali_no-trans'
            tbl_mag = photo.group_by(magnitude_type)

        #   Remove implausible dark results
        mask = tbl_mag[magnitude_type] < 30 * u.mag
        tbl_mag = tbl_mag[mask]

        #   Plot star map
        if reference_img != '':
            rts = f'faintest objects, image: {reference_img}'
        else:
            rts = 'faintest objects'
        p = mp.Process(
            target=plot.starmap,
            args=(
                image.out_path.name,
                image.get_data(),
                filter_,
                tbl_mag[:][-10:],
            ),
            kwargs={
                'label': '10 faintest objects',
                'rts': rts,
                'mode': 'mags',
                # 'name_object': image.object_name,
                'wcs': image.wcs,
            }
        )
        p.start()

        #   Print result
        terminal_output.print_to_terminal(
            f"Determine limiting magnitude for filter: {filter_}",
            indent=indent,
        )
        terminal_output.print_to_terminal(
            "Based on detected objects:",
            indent=indent * 2,
        )
        median_faintest_objects = np.median(tbl_mag[magnitude_type][-10:])
        terminal_output.print_to_terminal(
            f"Median of the 10 faintest objects: "
            f"{median_faintest_objects:.1f} mag",
            indent=indent * 3,
        )
        mean_faintest_objects = np.mean(tbl_mag[magnitude_type][-10:])
        terminal_output.print_to_terminal(
            f"Mean of the 10 faintest objects: "
            f"{mean_faintest_objects:.1f} mag",
            indent=indent * 3,
        )

        #   Convert object positions to pixel index values
        index_x = np.rint(tbl_mag['x_fit']).astype(int)
        index_y = np.rint(tbl_mag['y_fit']).astype(int)

        #   Convert object positions to mask
        mask = np.zeros(image.get_shape(), dtype=bool)
        mask[index_y, index_x] = True

        #   Set radius for the apertures
        radius = aperture_radius
        if radii_unit == 'arcsec':
            radius = radius / image.pixel_scale

        #   Setup ImageDepth object from the photutils package
        depth = ImageDepth(
            radius,
            nsigma=5.0,
            napers=500,
            niters=2,
            overlap=False,
            # seed=123,
            zeropoint=np.median(image.zp).value,
            progress_bar=False,
        )

        #   Derive limits
        flux_limit, mag_limit = depth(image.get_data(), mask)

        #   Plot sky apertures
        p = mp.Process(
            target=plot.plot_limiting_mag_sky_apertures,
            args=(image.out_path.name, image.get_data(), mask, depth),
        )
        p.start()

        #   Print results
        terminal_output.print_to_terminal(
            "Based on the ImageDepth (photutils) routine:",
            indent=indent * 2,
        )
        #   Remark: the error is only based on the zero point error
        terminal_output.print_to_terminal(
            f"500 apertures, 5 sigma, 2 iterations: "
            f"{mag_limit.pdf_median():6.2f} +/- "
            f"{mag_limit.pdf_std():6.2f} mag",
            indent=indent * 3,
        )


def rm_edge_objects(table, data_array, border=10, terminal_logger=None,
                    indent=3):
    """
        Remove detected objects that are too close to the image edges

        Parameters
        ----------
        table               : `astropy.table.Table` object
            Table with the object data

        data_array          : `numpy.ndarray`
            Image data (2D)

        border              : `integer`, optional
            Distance to the edge of the image where objects may be
            incomplete and should therefore be discarded.
            Default is ``10``.

        terminal_logger     : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``3``.
    """
    #   Border range
    hsize = border + 1

    #   Get position data
    x = table['x_fit'].value
    y = table['y_fit'].value

    #   Calculate mask of objects to be removed
    mask = ((x > hsize) & (x < (data_array.shape[1] - 1 - hsize)) &
            (y > hsize) & (y < (data_array.shape[0] - 1 - hsize)))

    out_str = f'{np.count_nonzero(np.invert(mask))} objects removed because ' \
              'they are too close to the image edges'
    if terminal_logger is not None:
        terminal_logger.add_to_cache(out_str, indent=indent)
    else:
        terminal_output.print_to_terminal(out_str, indent=indent)

    return table[mask]


def proper_motion_selection(
        image_series: 'analyze.ImageSeries', tbl, catalog="I/355/gaiadr3",
        g_mag_limit=20, separation_limit=1., sigma=3.,
        max_n_iterations_sigma_clipping=3):
    """
        Select a subset of objects based on their proper motion

        Parameters
        ----------
        image_series
            Image series object with all image data taken in a specific
            filter

        tbl                             : `astropy.table.Table`
            Table with position information

        catalog                         : `string`, optional
            Identifier for the catalog to download.
            Default is ``I/350/gaiaedr3``.

        g_mag_limit                     : `float`, optional
            Limiting magnitude in the G band. Fainter objects will not be
            downloaded.

        separation_limit                : `float`, optional
            Maximal allowed separation between objects in arcsec.
            Default is ``1``.

        sigma                           : `float`, optional
            Sigma value used in the sigma clipping of the proper motion
            values.
            Default is ``3``.

        max_n_iterations_sigma_clipping : `integer`, optional
            Maximal number of iteration of the sigma clipping.
            Default is ``3``.
    """
    #   Get wcs
    w = image_series.wcs

    #   Convert pixel coordinates to ra & dec
    coordinates = w.all_pix2world(tbl['x'], tbl['y'], 0)

    #   Create SkyCoord object with coordinates of all objects
    obj_coordinates = SkyCoord(
        coordinates[0],
        coordinates[1],
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
        column_filters={'Gmag': '<' + str(g_mag_limit)},
    )

    #   Get data from the corresponding catalog for the objects in
    #   the field of view
    result = v.query_region(
        image_series.coordinates_image_center,
        radius=image_series.field_of_view_x * u.arcmin,
    )

    #   Create SkyCoord object with coordinates of all Gaia objects
    calib_coordinates = SkyCoord(
        result[0]['RA_ICRS'],
        result[0]['DE_ICRS'],
        unit=(u.degree, u.degree),
        frame="icrs"
    )

    ###
    #   Correlate own objects with Gaia objects
    #
    #   Set maximal separation between objects
    separation_limit = separation_limit * u.arcsec

    #   Correlate data
    id_img, id_calib, d2ds, d3ds = matching.search_around_sky(
        obj_coordinates,
        calib_coordinates,
        separation_limit,
    )

    ###
    #   Sigma clipping of the proper motion values
    #

    #   Proper motion of the common objects
    pm_de = result[0]['pmDE'][id_calib]
    pm_ra = result[0]['pmRA'][id_calib]

    #   Parallax
    parallax = result[0]['Plx'][id_calib].data / 1000 * u.arcsec

    #   Distance
    distance = parallax.to_value(u.kpc, equivalencies=u.parallax())

    #   Sigma clipping
    sigma_clip_de = sigma_clip(
        pm_de,
        sigma=sigma,
        maxiters=max_n_iterations_sigma_clipping,
    )
    sigma_clip_ra = sigma_clip(
        pm_ra,
        sigma=sigma,
        maxiters=max_n_iterations_sigma_clipping,
    )

    #   Create mask from sigma clipping
    mask = sigma_clip_ra.mask | sigma_clip_de.mask

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
    image = image_series.reference_image

    #   Star map
    prepare_and_plot_starmap(
        image,
        tbl=Table(names=['x_fit', 'y_fit'], data=[x_obj, y_obj]),
        rts_pre='proper motion [Gaia]',
        label='Objects selected based on proper motion',
    )

    #   2D and 3D plot of the proper motion and the distance
    plot.scatter(
        [pm_ra],
        'pm_RA * cos(DEC) (mas/yr)',
        [pm_de],
        'pm_DEC (mas/yr)',
        'compare_pm_',
        image.out_path.name,
    )
    plot.d3_scatter(
        [pm_ra],
        [pm_de],
        [distance],
        image.out_path.name,
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
    )

    #   Apply mask
    return tbl[id_img][mask]


def region_selection(
        image_series: 'analyze.ImageSeries', coordinates_target, tbl, radius=600.):
    """
        Select a subset of objects based on a target coordinate and a radius

        Parameters
        ----------
        image_series
            Image series object with all image data taken in a specific
            filter

        coordinates_target  : `astropy.coordinates.SkyCoord` object or `list` of `astropy.coordinates.SkyCoord` object
            Coordinates of the observed object such as a star cluster

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
    """
    #   Get wcs
    w = image_series.wcs

    #   Convert pixel coordinates to ra & dec
    coordinates = w.all_pix2world(tbl['x'], tbl['y'], 0)

    #   Create SkyCoord object with coordinates of all objects
    obj_coordinates = SkyCoord(
        coordinates[0],
        coordinates[1],
        unit=(u.degree, u.degree),
        frame="icrs"
    )

    #   Calculate separation between the coordinates defined in ``coord``
    #   the objects in ``tbl``
    if isinstance(coordinates_target, list):
        mask = np.zeros(len(obj_coordinates), dtype=bool)
        for target_coordinates in coordinates_target:
            sep = obj_coordinates.separation(target_coordinates)

            #   Calculate mask of all object closer than ``radius``
            mask = mask | (sep.arcsec <= radius)
    else:
        sep = obj_coordinates.separation(coordinates_target)

        #   Calculate mask of all object closer than ``radius``
        mask = sep.arcsec <= radius

    #   Limit objects to those within radius
    tbl = tbl[mask]

    #   Plot starmap
    prepare_and_plot_starmap(
        image_series.reference_image,
        tbl=Table(names=['x_fit', 'y_fit'], data=[tbl['x'], tbl['y']]),
        rts_pre='radius selection, image',
        label=f"Objects selected within {radius}'' of the target",
    )

    return tbl, mask


def find_cluster(
        image_series: 'analyze.ImageSeries', tbl: Table, object_names: list[str],
        catalog: str = "I/355/gaiadr3", g_mag_limit: float = 20.,
        separation_limit: float = 1., max_distance: float = 6.,
        parameter_set: int = 1) -> tuple[Table, int, np.ndarray, np.ndarray]:
    """
        Identify cluster in data

        Parameters
        ----------
        image_series
            Image series object with all image data taken in a specific
            filter

        tbl
            Table with position information

        object_names
            Names of the objects. This first entry in the list is assumed to
            be the custer of interest.

        catalog
            Identifier for the catalog to download.
            Default is ``I/350/gaiaedr3``.

        g_mag_limit
            Limiting magnitude in the G band. Fainter objects will not be
            downloaded.

        separation_limit
            Maximal allowed separation between objects in arcsec.
            Default is ``1``.

        max_distance
            Maximal distance of the star cluster.
            Default is ``6.``.

        parameter_set
            Predefined parameter sets can be used.
            Possibilities: ``1``, ``2``, ``3``
            Default is ``1``.

        Returns
        -------
        tbl
            Table with object position information

        id_img

        mask
            The mask that needs to be applied to the table.

        cluster_mask
            Mask that identifies cluster members according to the user
            input.

    """
    #   Get wcs
    w = image_series.wcs

    #   Convert pixel coordinates to ra & dec
    coordinates = w.all_pix2world(tbl['x'], tbl['y'], 0)

    #   Create SkyCoord object with coordinates of all objects
    obj_coordinates = SkyCoord(
        coordinates[0],
        coordinates[1],
        unit=(u.degree, u.degree),
        frame="icrs"
    )

    #   Get reference image
    image = image_series.reference_image

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
        column_filters={'Gmag': '<' + str(g_mag_limit)},
    )

    #   Get data from the corresponding catalog for the objects in
    #   the field of view
    result = v.query_region(
        image_series.coordinates_image_center,
        radius=image_series.field_of_view_x * u.arcmin,
    )[0]

    #   Multiple objects can be specified. The first object is assumed to
    #   be the cluster of interest.
    object_name = object_names[0]

    #   Restrict proper motion to Simbad value plus some margin
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('pm')

    result_simbad = custom_simbad.query_object(object_name)
    pm_ra_object = result_simbad['PMRA'].value[0]
    pm_de_object = result_simbad['PMDEC'].value[0]
    if pm_ra_object != '--' and pm_de_object != '--':
        pm_m = 3.
        mask_de = ((result['pmDE'] <= pm_de_object - pm_m) |
                   (result['pmDE'] >= pm_de_object + pm_m))
        mask_ra = ((result['pmRA'] <= pm_ra_object - pm_m) |
                   (result['pmRA'] >= pm_ra_object + pm_m))
        mask = np.invert(mask_de | mask_ra)
        result = result[mask]

    #   Create SkyCoord object with coordinates of all Gaia objects
    calib_coordinates = SkyCoord(
        result['RA_ICRS'],
        result['DE_ICRS'],
        unit=(u.degree, u.degree),
        frame="icrs"
    )

    ###
    #   Correlate own objects with Gaia objects
    #
    #   Set maximal separation between objects
    separation_limit = separation_limit * u.arcsec

    #   Correlate data
    id_img, id_calib, d2ds, d3ds = matching.search_around_sky(
        obj_coordinates,
        calib_coordinates,
        separation_limit,
    )

    ###
    #   Find cluster in proper motion and distance data
    #

    #   Proper motion of the common objects
    pm_de_common_objects = result['pmDE'][id_calib]
    pm_ra_common_objects = result['pmRA'][id_calib]

    #   Parallax
    parallax = result['Plx'][id_calib].data / 1000 * u.arcsec

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
    mask = np.invert(pm_de_common_objects.mask | pm_ra_common_objects.mask
                     | distance_mask)

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
        affinity = 'nearest_neighbors'
    elif parameter_set == 2:
        n_clusters = 10
        random_state = 2
        n_neighbors = 4
        affinity = 'nearest_neighbors'
    elif parameter_set == 3:
        n_clusters = 2
        random_state = 25
        n_neighbors = 20
        affinity = 'rbf'
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo valid parameter set defined: "
            f"Possibilities are 1, 2, or 3. {style.Bcolors.ENDC}"
        )
    spectral_cluster_model = SpectralClustering(
        # eigen_solver='lobpcg',
        n_clusters=n_clusters,
        random_state=random_state,
        # gamma=2.,
        # gamma=5.,
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
    pm_ra_group = []
    pm_de_group = []
    distance_group = []
    for name, group in groups:
        pm_ra_group.append(group.pmRA.values)
        pm_de_group.append(group.pmDE.values)
        distance_group.append(group.distance.values)
    plot.d3_scatter(
        pm_ra_group,
        pm_de_group,
        distance_group,
        image.out_path.name,
        # color=np.unique(pd_result['cluster']),
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
        string='_3D_cluster_',
        pm_ra=pm_ra_object,
        pm_dec=pm_de_object,
    )
    plot.d3_scatter(
        pm_ra_group,
        pm_de_group,
        distance_group,
        image.out_path.name,
        # color=np.unique(pd_result['cluster']),
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
        string='_3D_cluster_',
        pm_ra=pm_ra_object,
        pm_dec=pm_de_object,
        display=True,
    )

    # plot.D3_scatter(
    # [pd_result['pmRA']],
    # [pd_result['pmDE']],
    # [pd_result['distance']],
    # image.outpath.name,
    # color=[pd_result['cluster']],
    # name_x='pm_RA * cos(DEC) (mas/yr)',
    # name_y='pm_DEC (mas/yr)',
    # name_z='d (kpc)',
    # string='_3D_cluster_',
    # )

    #   Get user input
    cluster_id, timed_out = timedInput(
        style.Bcolors.OKBLUE +
        "   Which one is the correct cluster (id)? "
        + style.Bcolors.ENDC,
        timeout=300,
    )
    if timed_out or cluster_id == '':
        cluster_id = 0
    else:
        cluster_id = int(cluster_id)

    #   Calculated mask according to user input
    cluster_mask = pd_result['cluster'] == cluster_id

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
        rts_pre='selected cluster members',
        label='Cluster members based on proper motion and distance evaluation',
        add_image_id=False,
    )

    #   Return table
    return tbl, id_img, mask, cluster_mask.values


def save_magnitudes_ascii(
        observation: 'analyze.Observation', tbl, trans=False, id_object=None, rts='',
        photometry_extraction_method='',
        add_file_path_to_observation_object=True):
    """
        Save magnitudes as ASCII files

        Parameters
        ----------
        observation
            Image container object with image series objects for each
            filter

        tbl                             : `astropy.table.Table`
            Table with magnitudes

        trans                           : `boolean`, optional
            If True a magnitude transformation was performed
            Default is ``False``.

        id_object                       : `integer` or `None`, optional
            ID of the object
            Default is ``None``.

        rts                             : `string`, optional
            Additional string characterizing that should be included in the
            file name.
            Default is ``''``.

        photometry_extraction_method    : `string`, optional
            Applied extraction method. Possibilities: ePSF or APER`
            Default is ``''``.

        add_file_path_to_observation_object      : `boolean`, optional
            If True the file path will be added to the observation object.
            Default is ``True``.
    """
    #   Check output directories
    output_dir = list(observation.image_series_dict.values())[0].out_path
    checks.check_output_directories(
        output_dir,
        output_dir / 'tables',
    )

    #   Define file name specifier
    if id_object is not None:
        id_object = f'_img_{id_object}'
    else:
        id_object = ''
    if photometry_extraction_method != '':
        photometry_extraction_method = f'_{photometry_extraction_method}'

    #   Check if observation object contains already entries
    #   for file names/paths. If not add dictionary.
    photo_filepath = getattr(observation, 'photo_filepath', None)
    if photo_filepath is None or not isinstance(photo_filepath, dict):
        observation.photo_filepath = {}

    #   Set file name
    if trans:
        #   Set file name for file with magnitude transformation
        filename = f'mags_TRANS_calibrated{photometry_extraction_method}{id_object}{rts}.dat'
    else:
        #   File name for file without magnitude transformation
        filename = f'mags_calibrated{photometry_extraction_method}{id_object}{rts}.dat'

    #   Combine to a path
    out_path = output_dir / 'tables' / filename

    #   Add to object
    if add_file_path_to_observation_object:
        observation.photo_filepath[out_path] = trans

    ###
    #   Define output formats for the table columns
    #
    #   Get column names
    column_names = tbl.colnames

    #   Set default
    for column_name in column_names:
        if column_name not in ['ra (deg)', 'dec (deg)']:
            tbl[column_name].info.format = '{:12.3f}'

    #   Reset for x and y column
    formats = {
        'i': '{:5.0f}',
        'x': '{:12.2f}',
        'y': '{:12.2f}',
    }

    #   Write file
    tbl.write(
        str(out_path),
        format='ascii',
        overwrite=True,
        formats=formats,
    )


def post_process_results(
        observation: 'analyze.Observation', filter_list, id_object=None, extraction_method='',
        extract_only_circular_region=False, region_radius=600,
        data_cluster=False, clean_objects_using_proper_motion=False,
        max_distance_cluster=6., find_cluster_para_set=1,
        convert_magnitudes=False, target_filter_system='SDSS', tbl_list=None,
        distribution_samples=1000):
    """
        Restrict results to specific areas of the image and filter by means
        of proper motion and distance using Gaia

        Parameters
        ----------
        observation
            Container object with image series objects for each
            filter

        filter_list                         : `list` of `string`
            Filter names

        id_object                           : `integer` or `None`, optional
            ID of the object
            Default is ``None``.

        extraction_method                   : `string`, optional
            Applied extraction method. Possibilities: ePSF or APER`
            Default is ``''``.

        extract_only_circular_region        : `boolean`, optional
            If True the extracted objects will be filtered such that only
            objects with ``radius`` will be returned.
            Default is ``False``.

        region_radius                       : `float`, optional
            Radius around the object in arcsec.
            Default is ``600``.

        data_cluster                        : `boolean`, optional
            If True cluster in the Gaia distance and proper motion data
            will be identified.
            Default is ``False``.

        clean_objects_using_proper_motion   : `boolean`, optional
            If True only the object list will be clean based on their
            proper motion.
            Default is ``False``.

        max_distance_cluster                : `float`, optional
            Expected maximal distance of the cluster in kpc. Used to
            restrict the parameter space to facilitate an easy
            identification of the star cluster.
            Default is ``6``.

        find_cluster_para_set               : `integer`, optional
            Parameter set used to identify the star cluster in proper
            motion and distance data.

        convert_magnitudes                  : `boolean`, optional
            If True the magnitudes will be converted to another
            filter systems specified in `target_filter_system`.
            Default is ``False``.

        target_filter_system                : `string`, optional
            Photometric system the magnitudes should be converted to
            Default is ``SDSS``.

        tbl_list                            : `[astropy.table.Table]` or None, optional
            List with Tables containing magnitudes etc. If None are provided,
            the tables will be read from the observation container.
            Default is ``None``.

        distribution_samples  : `integer`, optional
            Number of samples used for distributions
            Default is `1000`.
    """
    #   Do nothing if no post process method were defined
    if (not extract_only_circular_region and not clean_objects_using_proper_motion
            and not data_cluster and not convert_magnitudes):
        return

    #   Get image series
    image_series_dict = observation.image_series_dict

    #   Get astropy tables with positions and magnitudes
    if tbl_list is None or not tbl_list:
        tbl_list = [
            (getattr(observation, 'table_mags_transformed', None), True),
            (getattr(observation, 'table_mags_not_transformed', None), False),
        ]

    #   Loop over all Tables
    mask_region = None
    img_id_cluster = None
    mask_cluster = None
    mask_objects = None
    img_id_pm = None
    mask_pm = None
    for (tbl, trans) in tbl_list:
        ###
        #   Post process data
        #

        #   Extract circular region around a certain object
        #   such as a star cluster
        if extract_only_circular_region:
            if mask_region is None:
                tbl, mask_region = region_selection(
                    image_series_dict[filter_list[0]],
                    observation.coordinates_objects,
                    tbl,
                    radius=region_radius
                )
            else:
                tbl = tbl[mask_region]

        #   Find a cluster in the Gaia data that could be the star cluster
        if data_cluster:
            if any(x is None for x in [img_id_cluster, mask_cluster, mask_objects]):
                tbl, img_id_cluster, mask_cluster, mask_objects = find_cluster(
                    image_series_dict[filter_list[0]],
                    tbl,
                    observation.get_object_of_interest_names(),
                    max_distance=max_distance_cluster,
                    parameter_set=find_cluster_para_set,
                )
            else:
                tbl = tbl[img_id_cluster][mask_cluster][mask_objects]

        #   Clean objects according to proper motion (Gaia)
        #   TODO: Check if this is still a useful option
        if clean_objects_using_proper_motion:
            if any(x is None for x in [img_id_pm, mask_pm]):
                tbl, img_id_pm, mask_pm = proper_motion_selection(
                    image_series_dict[filter_list[0]],
                    tbl,
                )
            else:
                tbl = tbl[img_id_pm][mask_pm]

        #   Convert magnitudes to a different filter system
        if convert_magnitudes:
            tbl = convert_magnitudes_to_other_system(
                tbl,
                target_filter_system,
                distribution_samples=distribution_samples,
            )

        ###
        #   Save results as ASCII files
        #
        #   TODO: Fix this dirty hack to fix the file names, if magnitude
        #         transformation is applied
        if trans:
            rts = f'_{filter_list[0]}-{filter_list[1]}_post_processed'
        else:
            rts = '_post_processed'
        save_magnitudes_ascii(
            observation,
            tbl,
            trans=trans,
            id_object=id_object,
            rts=rts,
            photometry_extraction_method=extraction_method,
            add_file_path_to_observation_object=False,
        )


def add_column_to_table(tbl, column_name, data, column_id):
    """
        Adds data from a distribution to an astropy Table

        Parameters
        ----------
        tbl                 : `astropy.table.Table`
            Table that already contains some data

        column_name         : `string`
            Name of the column to add

        data                : `astropy.uncertainty.core.QuantityDistribution`
            Data to add

        column_id           : `integer`
            Additional ID that identifies the column. If the
            ID is not -1, it will be added to the column header.

        Returns
        -------
        tbl                 : `astropy.table.Table`
            Table with the added column
    """
    if column_id == -1:
        tbl.add_columns(
            [data.pdf_median(), data.pdf_std()],
            names=[column_name, f'{column_name}_err']
        )
    else:
        tbl.add_columns(
            [data.pdf_median(), data.pdf_std()],
            names=[
                f'{column_name} ({column_id})',
                f'{column_name}_err ({column_id})',
            ]
        )

    return tbl


def distribution_from_table(
        image: 'analyze.Image',
        distribution_samples: int = 1000) -> unc:
    """
    Arrange the literature values in a numpy array or uncertainty array.

    Parameters
    ----------
    image
        Object with image data

    distribution_samples
        Number of samples used for distributions
        Default is `1000`

    Returns
    -------
    distribution
        Normal distribution representing observed magnitudes
    """
    #   Build normal distribution
    magnitude_distribution = unc.normal(
        image.photometry['mags_fit'].value * u.mag,
        std=image.photometry['mags_unc'].value * u.mag,
        n_samples=distribution_samples,
    )

    return magnitude_distribution


def convert_magnitudes_to_other_system(
        tbl: Table, target_filter_system: str, distribution_samples=1000
        ) -> Table:
    """
        Convert magnitudes from one magnitude system to another

        Parameters
        ----------
        tbl                     : `astropy.table.Table`
            Table with magnitudes

        target_filter_system    : `string`
            Photometric system the magnitudes should be converted to

        distribution_samples    : `integer`, optional
            Number of samples used for distributions
            Default is `1000`.
    """
    #   Get column names
    column_names = tbl.colnames

    #   Checks
    if target_filter_system not in ['SDSS', 'AB', 'BESSELL']:
        terminal_output.print_to_terminal(
            f'Magnitude conversion not possible. Unfortunately, '
            f'there is currently no conversion formula for this '
            f'photometric system: {target_filter_system}.',
            style_name='WARNING',
        )

    #   Select magnitudes and errors and corresponding filter
    available_image_ids = []
    available_filter_image_error = []

    #   Loop over column names
    for column_name in column_names:
        #   Detect color: 'continue in this case, since colors are not yet
        #   supported'
        if len(column_name) > 1 and column_name[1] == '-':
            continue

        #   Get filter
        column_filter = column_name[0]
        if column_filter in ['i', 'x', 'y']:
            continue

        #   Get the image ID
        image_id = column_name.split('(')[1].split(')')[0]

        #   Is an image ID available?
        if image_id != '':
            #   Check for error column
            error = any(x == f'{column_filter}_err ({image_id})' for x in column_names)

            #   Combine derived info -> (ID of the image, Filter,
            #                            boolean: error available?)
            info = (image_id, column_filter, error)
        else:
            #   Set dummy image ID
            image_id = -1

            #   Check for error column
            error = any(x == f'{column_filter}_err' for x in column_names)

            #   Combine derived info -> (ID of the image, Filter,
            #                            boolean: error available?)
            info = (-1, column_filter, error)

        #   Check if image and filter combination is already known.
        #   If yes continue.
        if info in available_filter_image_error:
            continue

        #   Save image, filter, & error info
        available_filter_image_error.append(info)

        if image_id not in available_image_ids:
            available_image_ids.append(image_id)

    #   Make conversion for each image ID individually
    for image_id in available_image_ids:
        #   Reset dictionary with data
        data_dict = {}

        #   Get image ID, filter and error combination
        for (current_image_id, column_filter, error) in available_filter_image_error:
            #   Restrict to current image ID
            if current_image_id != image_id:
                continue

            #   Fill data dictionary, branch according to error and image
            #   ID availability
            if image_id == -1:
                if error:
                    data_dict[column_filter] = unc.normal(
                        tbl[f'{column_filter}'].value * u.mag,
                        std=tbl[f'{column_filter}_err'].value * u.mag,
                        n_samples=distribution_samples,
                    )
                    # unumpy.uarray(
                    #     tbl[f'{column_filter}'].value,
                    #     tbl[f'{column_filter}_err'].value
                    # )
                else:
                    # data_dict[column_filter] = tbl[f'{column_filter}'].value
                    data_dict[column_filter] = unc.normal(
                        tbl[f'{column_filter}'].value * u.mag,
                        n_samples=distribution_samples,
                    )
            else:
                if error:
                    # data_dict[column_filter] = unumpy.uarray(
                    #     tbl[f'{column_filter} ({image_id})'].value,
                    #     tbl[f'{column_filter}_err ({image_id})'].value
                    # )
                    data_dict[column_filter] = unc.normal(
                        tbl[f'{column_filter} ({image_id})'].value * u.mag,
                        std=tbl[f'{column_filter}_err ({image_id})'].value * u.mag,
                        n_samples=distribution_samples,
                    )
                else:
                    # data_dict[column_filter] = tbl[f'{column_filter} ({image_id})'].value
                    data_dict[column_filter] = unc.normal(
                        tbl[f'{column_filter} ({image_id})'].value * u.mag,
                        n_samples=distribution_samples,
                    )

        if target_filter_system == 'AB':
            #   TODO: Fix this
            print('Will be available soon...')

        elif target_filter_system == 'SDSS':
            #   Get conversion function - only Jordi et a. (2005) currently
            #   available:
            calib_functions = calibration_data \
                .filter_system_conversions['SDSS']['Jordi_et_al_2005']

            #   Convert magnitudes and add those to data dictionary and the Table
            g = calib_functions['g'](
                **data_dict,
                distribution_samples=distribution_samples,
            )
            if g is not None:
                data_dict['g'] = g
                tbl = add_column_to_table(tbl, 'g', g, image_id)

            u_mag = calib_functions['u'](
                **data_dict,
                distribution_samples=distribution_samples,
            )
            if u_mag is not None:
                data_dict['u'] = u_mag
                tbl = add_column_to_table(tbl, 'u', u_mag, image_id)

            r = calib_functions['r'](
                **data_dict,
                distribution_samples=distribution_samples,
            )
            if r is not None:
                data_dict['r'] = r
                tbl = add_column_to_table(tbl, 'r', r, image_id)

            i = calib_functions['i'](
                **data_dict,
                distribution_samples=distribution_samples,
            )
            if i is not None:
                data_dict['i'] = i
                tbl = add_column_to_table(tbl, 'i', i, image_id)

            z = calib_functions['z'](
                **data_dict,
                distribution_samples=distribution_samples,
            )
            if z is not None:
                data_dict['z'] = z
                tbl = add_column_to_table(tbl, 'z', z, image_id)

        elif target_filter_system == 'BESSELL':
            #   TODO: Fix this
            print('Will be available soon...')

        return tbl


def find_filter_for_magnitude_transformation(
        filter_list, calibration_filters, valid_filter_combinations: list[list[str]] | None = None):
    """
    Identifies filter that can be used for magnitude transformation

    Parameters
    ----------
    filter_list                 : `list` of `strings`
        List with observed filter names

    calibration_filters         : `list` of `strings`
        Names of the available filter with calibration data

    valid_filter_combinations
        Valid filter combinations to calculate magnitude transformation
        Default is ``None``.

    Returns
    -------
    valid_filter                : `list` of `string`
        Filter for which magnitude transformation is possible

    usable_filter_combinations  : `list` of `list` od `string`
        Filter combinations for which magnitude transformation
        can be applied
    """
#   Load valid filter combinations, if none are supplied
    if valid_filter_combinations is None:
        valid_filter_combinations = calibration_data.valid_filter_combinations_for_transformation

    #   Setup list for valid filter etc.
    valid_filter = []
    usable_filter_combinations = []

    #   Determine usable filter combinations -> Filters must be in a valid
    #   filter combination for the magnitude transformation and calibration
    #   data must be available for the filter.
    for filter_combination in valid_filter_combinations:
        if filter_combination[0] in filter_list and filter_combination[1] in filter_list:
            faulty_filter = None
            if f'mag{filter_combination[0]}' not in calibration_filters:
                faulty_filter = filter_combination[0]
            if f'mag{filter_combination[1]}' not in calibration_filters:
                faulty_filter = filter_combination[1]
            if faulty_filter is not None:
                terminal_output.print_to_terminal(
                    "Magnitude transformation not possible because "
                    "no calibration data available for filter "
                    f"{faulty_filter}",
                    indent=2,
                    style_name='WARNING',
                )
                continue

            valid_filter.append(filter_combination[0])
            valid_filter.append(filter_combination[1])
            usable_filter_combinations.append(filter_combination)
    valid_filter = set(valid_filter)

    return valid_filter, usable_filter_combinations
