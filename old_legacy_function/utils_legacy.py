############################################################################
#                               Libraries                                  #
############################################################################
import sys

import numpy as np

from pathlib import Path

from astropy.table import Table, Column
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy.coordinates import SkyCoord, matching
from astropy.modeling import models, fitting, polynomial
import astropy.units as u
from astropy import wcs
from astropy import uncertainty as unc


from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.exceptions import TableParseError

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

from ... import utilities as base_utilities
from ... import wcs as wcs_utilities

from ... import checks, style, terminal_output, calibration_parameters

from .. import plots
from ..post_processing.adapters import ensure_epoch_native_photometry_table
from ..post_processing.io import write_epoch_native_magnitudes
from ..post_processing.coords import (
    plot_starmap_from_imaging_context,
    table_object_sky_coords,
)
from ..post_processing.imaging import ImagingPlotContext, imaging_context_from_image_series
from ..post_processing.light_curve import attach_observation_jd_column

from .cluster_selection import (  # noqa: F401
    find_cluster,
    proper_motion_selection,
    region_selection,
)
from .cmd_defaults import (  # noqa: F401
    check_variable_absolute_cmd,
    check_variable_apparent_cmd,
)
from .duplicates import clear_duplicates  # noqa: F401
from .errors import err_prop  # noqa: F401
from .legacy_magnitudes import (  # noqa: F401
    calibrated_epochs_to_legacy_wide_table,
    find_filter_for_magnitude_transformation,
    mk_magnitudes_table,
    save_calibration,
    save_magnitudes_ascii,
    transformation_keys_for_table_magnitudes,
)
from .limiting_magnitude import derive_limiting_magnitude  # noqa: F401
from .photometry import flux_to_magnitudes, rm_edge_objects  # noqa: F401
from .series_wcs import find_wcs  # noqa: F401
from .simbad_annotate import (  # noqa: F401
    mark_simbad_objects_on_image,
    query_simbad_objects,
)
from .starmaps import (  # noqa: F401
    prepare_and_plot_starmap,
    prepare_and_plot_starmap_from_observation,
    prepare_and_plot_starmap_from_image_series,
)

import typing
if typing.TYPE_CHECKING:
    from .. import analyze


############################################################################
#                           Routines & definitions                         #
############################################################################


def distribution_from_table(
        image: 'analyze.Image',
        distribution_samples: int = 1000) -> unc.core.NdarrayDistribution:
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
    #   Return if no photometry information are available
    if image.photometry is None:
        terminal_output.print_to_terminal(
            "Photometric data not yet available. Distribution cannot be "
            "created. -> returns 'None'.",
            style_name='WARNING',
        )
        return

    #   Build normal distribution
    magnitude_distribution = unc.normal(
        image.photometry['mags_fit'].value * u.mag,
        std=image.photometry['mags_unc'].value * u.mag,
        n_samples=distribution_samples,
    )

    return magnitude_distribution


def lin_func(x, a, b):
    """
        Linear function
    """
    return a + b * x


#   TODO: Add type hint for 'fit_func'
def fit_curve(
        fit_func, x: np.ndarray, y: np.ndarray, x0: np.ndarray, sigma: np.ndarray
        ) -> tuple[float, float, float, float]:
    """
    Fit curve with supplied fit function

    Parameters
    ----------
    fit_func
        Function used in the fitting process

    x
        Abscissa values

    y
        Ordinate values

    x0
        Initial guess for the fit parameters

    sigma
        Uncertainty of the ordinate values

    Returns
    -------
    a
        Parameter I

    a_err
        Error parameter I

    b
        Parameter II

    b_err
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


def fit_data_one_d(
        x: np.ndarray, y: np.ndarray, order: int) -> polynomial.Polynomial1D:
    """
    Fit polynomial to the provided data.

    Parameters
    ----------
    x
        abscissa data values

    y
        ordinate data values

    order
        Polynomial order to be fitted to the data

    Returns
    -------
    fit_poly
        The fitted polynomial
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


def find_transformation_coefficients(
        filter_list: list[str],
        tsc_parameter_dict: dict[str, dict[str, dict[str, float | str | list[str]]]] | None,
        filter_: str, camera: str, verbose: bool = False, indent: int = 2
        ) -> dict[str, float | str | list[str]] | None:
    """
    Find the position of the filter from the 'tsc_parameter_dict'
    dictionary with reference to 'filter_list'

    Parameters
    ----------
    filter_list
        List of available filter, e.g., ['U', 'B', 'V', ...]

    tsc_parameter_dict
        Magnitude transformation coefficients for different cameras.
        Keys:  camera identifier

    filter_
        Filter for which calibration data will be selected

    camera
        Instrument used

    verbose
        If ``True`` additional information will be printed to the console.
        Default is ``False``.

    indent
        Indentation for the console output
        Default is ``2``.

    Returns
    -------
    variable_1
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


def prepare_calibration_check_plots(
        filter_: str, out_dir: str, image_id: int,
        ids_calibration_stars: np.ndarray, literature_magnitudes: np.ndarray,
        magnitudes: np.ndarray, uncalibrated_magnitudes: np.ndarray,
        plot_type: str, filter_list: list[str] | None = None,
        color_observed: np.ndarray | None = None,
        color_literature: np.ndarray | None = None, color_observed_err=None,
        color_literature_err=None, literature_magnitudes_err=None,
        magnitudes_err: np.ndarray | None = None,
        uncalibrated_magnitudes_err: np.ndarray | None = None,
        multiprocessing: bool = True, file_type_plots: str = 'pdf') -> None:
    """
    Useful plots to check the quality of the calibration process.

    Parameters
    ----------
    filter_
        Filter used

    out_dir
        Output directory

    image_id
            Expression characterizing the plot

    ids_calibration_stars
        IDs of the calibration stars

    literature_magnitudes
        Literature magnitudes of the objects that are used in the
        calibration process

    magnitudes
        Array with magnitudes of all observed objects

    uncalibrated_magnitudes
        Magnitudes of all observed objects but not calibrated yet

    plot_type
        String that characterize the plot and calibration method used

    filter_list
        Filter list
        Default is ``None``.

    color_observed
        Instrument color of the calibration stars
        Default is ``None``.

    color_literature
        Literature color of the calibration stars
        Default is ``None``.

    color_observed_err
        Uncertainty in the instrument color of the calibration stars
        Default is ``None``.

    color_literature_err
        Uncertainty in the literature color of the calibration stars
        Default is ``None``.

    literature_magnitudes_err
        Uncertainty in the literature magnitudes of the objects that are
        used in the calibration process
        Default is ``None``.

    magnitudes_err
        Uncertainty in the magnitudes of the observed objects
        Default is ``None``.

    uncalibrated_magnitudes_err
        Uncertainty in the uncalibrated magnitudes of the observed objects
        Default is ``None``.

    multiprocessing
        If ``True'', multicore processing is allowed, otherwise not.
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Comparison calibrated vs. uncalibrated magnitudes
    if multiprocessing:
        p = mp.Process(
            target=plots.scatter,
            args=(
                [magnitudes],
                f'{filter_}_calibration [mag]',
                [uncalibrated_magnitudes],
                f'{filter_}_no-calibration [mag]',
                f'mag-cali_mags_{filter_}_img_{image_id}_{plot_type}',
                out_dir,
            ),
            kwargs={
                # 'name_object': name_object,
                'x_errors': [magnitudes_err],
                'y_errors': [uncalibrated_magnitudes_err],
                'file_type': file_type_plots,
            }
        )
        p.start()
    else:
        plots.scatter(
            [magnitudes],
            f'{filter_}_calibration [mag]',
            [uncalibrated_magnitudes],
            f'{filter_}_no-calibration [mag]',
            f'mag-cali_mags_{filter_}_img_{image_id}_{plot_type}',
            out_dir,
            # name_object=name_object,
            x_errors=[magnitudes_err],
            y_errors=[uncalibrated_magnitudes_err],
            file_type=file_type_plots,
        )

    #   Comparison observed vs. literature magnitudes
    #   Make fit
    fit = fit_data_one_d(
        uncalibrated_magnitudes[ids_calibration_stars],
        literature_magnitudes,
        1,
    )

    if uncalibrated_magnitudes_err is not None:
        if multiprocessing:
            p = mp.Process(
                target=plots.scatter,
                args=(
                    [uncalibrated_magnitudes[ids_calibration_stars]],
                    f'{filter_}_measured [mag]',
                    [literature_magnitudes],
                    f'{filter_}_literature [mag]',
                    f'mags_{filter_}_img_{image_id}_{plot_type}',
                    out_dir,
                ),
                kwargs={
                    'fits': [None, fit],
                    'x_errors': [
                        uncalibrated_magnitudes_err[ids_calibration_stars]
                    ],
                    'y_errors': [
                        literature_magnitudes_err
                    ],
                    'file_type': file_type_plots,
                }
            )
            p.start()
        else:
            plots.scatter(
                [uncalibrated_magnitudes[ids_calibration_stars]],
                f'{filter_}_measured [mag]',
                [literature_magnitudes],
                f'{filter_}_literature [mag]',
                f'mags_{filter_}_img_{image_id}_{plot_type}',
                out_dir,
                fits=[None, fit],
                x_errors=[uncalibrated_magnitudes_err[ids_calibration_stars]],
                y_errors=[literature_magnitudes_err],
                file_type=file_type_plots,
            )

    #   Comparison observed vs. literature color
    if (color_observed is not None and color_literature is not None
            and filter_list is not None):
        #   Make fit
        fit = fit_data_one_d(
            color_literature,
            color_observed,
            1,
        )

        if multiprocessing:
            p = mp.Process(
                target=plots.scatter,
                args=(
                    [color_literature],
                    f'{filter_list[0]}-{filter_list[1]}_literature [mag]',
                    [color_observed],
                    f'{filter_list[0]}-{filter_list[1]}_measured [mag]',
                    f'color_{filter_}_img_{image_id}_{plot_type}',
                    out_dir,
                ),
                kwargs={
                    'x_errors': [color_literature_err],
                    'y_errors': [color_observed_err],
                    'fits': [fit, fit],
                    'file_type': file_type_plots,
                }
            )
            p.start()
        else:
            plots.scatter(
                [color_literature],
                f'{filter_list[0]}-{filter_list[1]}_literature [mag]',
                [color_observed],
                f'{filter_list[0]}-{filter_list[1]}_measured [mag]',
                f'color_{filter_}_img_{image_id}_{plot_type}',
                out_dir,
                x_errors=[color_literature_err],
                y_errors=[color_observed_err],
                fits=[fit, fit],
                file_type=file_type_plots,
            )

    #   Difference between literature values and calibration results
    if magnitudes_err is not None:
        if multiprocessing:
            p = mp.Process(
                target=plots.scatter,
                args=(
                    [literature_magnitudes],
                    f'{filter_}_literature [mag]',
                    [
                        magnitudes[ids_calibration_stars] - literature_magnitudes,
                    ],
                    f'{filter_}_observed - {filter_}_literature [mag]',
                    f'magnitudes_literature-vs-observed_{image_id}_{filter_}_{plot_type}',
                    out_dir,
                ),
                kwargs={
                    'x_errors': [literature_magnitudes_err],
                    'y_errors': [
                        err_prop(magnitudes_err[ids_calibration_stars], literature_magnitudes_err),
                    ],
                    'file_type': file_type_plots,
                },
            )
            p.start()
        else:
            plots.scatter(
                [literature_magnitudes],
                f'{filter_}_literature [mag]',
                [magnitudes[ids_calibration_stars] - literature_magnitudes],
                f'{filter_}_observed - {filter_}_literature [mag]',
                f'magnitudes_literature-vs-observed_{image_id}_{filter_}_{plot_type}',
                out_dir,
                x_errors=[literature_magnitudes_err],
                y_errors=[
                    err_prop(magnitudes_err[ids_calibration_stars], literature_magnitudes_err),
                ],
                file_type=file_type_plots,
            )


