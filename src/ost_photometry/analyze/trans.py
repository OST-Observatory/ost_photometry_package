############################################################################
#                               Libraries                                  #
############################################################################
import sys
import time

import copy

import numpy as np

import astropy.units as u
from astropy import uncertainty as unc
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.stats import sigma_clip as sigma_clipping

from . import calib, analyze, utilities, plot

import typing
if typing.TYPE_CHECKING:
    from . import analyze

from .. import checks, style, calibration_data, terminal_output


############################################################################
#                           Routines & definitions                         #
############################################################################


def progress_bar(count_value, total, suffix=''):
    """
    A progress bar. The code is from: https://www.geeksforgeeks.org/progress-bars-in-python/

    Parameters
    ----------
    count_value
    total
    suffix
    """
    bar_length = 100
    filled_up_length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_length + '-' * (bar_length - filled_up_length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()


def find_best_comparison_image_second_filter(
        image_series: dict[str:'analyze.ImageEnsemble'], current_image: 'analyze.ImageEnsemble.Image',
        id_second_filter: int, filter_list: list[str]
        ) -> 'analyze.ImageEnsemble.Image':
    """
        Prepare variables for magnitude transformation

        Parameters
        ----------
        image_series
            Object that encompasses all image objects for a filter and relevant
            information

        current_image
            Object with all image specific properties

        id_second_filter
            ID of the second filter


        filter_list
            List of filter names

        Returns
        -------
        best_image_second_filter
            Image class with all image specific properties
    """
    #   Get observation time of current image and all images of the
    #   second filter
    obs_time_current_image = current_image.jd
    obs_times_images_second_filter = image_series[
        filter_list[id_second_filter]
    ].get_obs_time()

    #   Find ID of the image with the nearest exposure time
    id_best_image_second_filter = np.argmin(
        np.abs(obs_times_images_second_filter - obs_time_current_image)
    )

    #   Get image corresponding to this exposure time
    best_img_second_filter = image_series[
        filter_list[id_second_filter]
    ].image_list[id_best_image_second_filter]

    return best_img_second_filter


def check_transformation_requirements(
        img_container, trans_coefficients, filter_list, current_filter_id,
        derive_transformation_coefficients):
    """
        Prepare magnitude transformation: find filter combination,
        get calibration parameters, prepare variables, ...

        Parameters
        ----------
        img_container                   : `image.container`
            Container object with image ensemble objects for each filter

        trans_coefficients              : `dictionary` or ``None``
            Calibration coefficients for magnitude transformation

        filter_list                     : `list` of `string`
            List of filter names

        current_filter_id               : `integer`
            ID of the current filter

        derive_transformation_coefficients       : `boolean`
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.

        Returns
        -------
        type_transformation             : `string`
            Type of magnitude transformation to be performed

        second_filter_id                : `integer`
            ID of the second filter

        trans_coefficients_selection    : `dictionary`
            Dictionary with validated calibration parameters from Tcs.
    """
    #   Get filter name
    filter_ = filter_list[current_filter_id]

    #   Get second filter id -> assumes that filter list contains
    #   only two filter
    #   TODO: Generalize for more filter -> matrix calculations
    if len(filter_list) == 1:
        return None, None, None
    elif len(filter_list) == 2:
        if current_filter_id == 0:
            second_filter_id = 1
        else:
            second_filter_id = 0
    else:
        #   This should currently not happen
        terminal_output.print_to_terminal(
            f"Magnitude transformation currently only possible with two filter "
            f"but {len(filter_list)} filter given in filter_list",
            style_name='ERROR',
        )
        raise RuntimeError

    if derive_transformation_coefficients:
        type_transformation = 'derive'
        trans_coefficients_selection = None
    else:
        #   Load coefficients coefficients
        if trans_coefficients is None:
            trans_coefficients = calibration_data.get_transformation_calibration_values(
                img_container.ensembles[filter_].start_jd
            )
            trans_coefficients_selection = utilities.find_transformation_coefficients(
                filter_list,
                trans_coefficients,
                filter_,
                img_container.ensembles[filter_].instrument,
            )
            #   If no valid transformation coefficients can be loaded switch to
            #   automatic determination of these coefficients
            if trans_coefficients_selection is None:
                type_transformation = 'derive'
                terminal_output.print_to_terminal(
                    'Transformation coefficients cannot be loaded, switching'
                    ' to automatic determination of these coefficients.',
                    indent=3,
                    style_name='WARNING',
                )
            else:
                type_transformation = trans_coefficients_selection['type']
        else:
            trans_coefficients_selection = trans_coefficients
            type_transformation = trans_coefficients_selection['type']

    message_type = 'BOLD'
    if type_transformation == 'simple':
        string = "Apply simple magnitude transformation"
    elif type_transformation == 'air_mass':
        string = "Apply magnitude transformation accounting for air_mass"
    elif type_transformation == 'derive':
        string = f"Derive and apply magnitude transformation based on " \
                 f"{filter_} image"
    else:
        string = f"Magnitude transformation is not possible because some " \
                 f"prerequisites, such as a second filter, are not met."
        message_type = 'WARNING'
        # raise RuntimeError(
        #     f"{style.Bcolors.FAIL} \nNo valid transformation type. Got "
        #     f"{type_transformation}, but allowed are only: simple, "
        #     f"air_mass, and derive  {style.Bcolors.ENDC}"
        # )

    terminal_output.print_to_terminal(string, indent=3, style_name=message_type)

    return type_transformation, second_filter_id, trans_coefficients_selection


def derive_transformation_onthefly(
        image: 'analyze.ImageEnsemble.Image', filter_list: list[str],
        id_current_filter: int, color_literature: unc,
        magnitudes_literature_filter_1: unc,
        magnitudes_literature_filter_2: unc,
        magnitudes_observed_filter_1: unc, magnitudes_observed_filter_2: unc,
        distribution_samples: int = 1000) -> tuple[unc, unc]:
    """
        Determine the parameters for the color term used in the magnitude
        calibration. This corresponds to a magnitude transformation without
        considering the dependence on the air mass.

        Parameters
        ----------
        image
            Object with all image specific properties

        filter_list
            List of filter

        id_current_filter
            ID of the current filter

        color_literature
            Literature color of the calibration stars

        magnitudes_literature_filter_1
            Magnitudes of calibration stars from the literature
            for filter 1.

        magnitudes_literature_filter_2
            Magnitudes of calibration stars from the literature
            for filter 1.

        magnitudes_observed_filter_1
            Extracted magnitudes of the calibration stars from filter 1

        magnitudes_observed_filter_2
            Extracted magnitudes of the calibration stars from filter 2

        distribution_samples
            Number of samples used for distributions
            Default is `1000`

        Returns
        -------
        color_correction_filter_1
            Color correction term for filter 1.

        color_correction_filter_2
            Color correction term for filter 2.
    """
    #   Initial guess for the parameters
    # x0    = np.array([0.0, 0.0])
    x0: np.ndarray = np.array([1.0, 1.0])

    #   Fit function
    fit_func = utilities.lin_func

    #   Get variables
    diff_mag_1 = magnitudes_literature_filter_1 - magnitudes_observed_filter_1
    diff_mag_2 = magnitudes_literature_filter_2 - magnitudes_observed_filter_2

    #   TODO: Test with distributions
    color_literature_plot = color_literature.pdf_median()
    color_literature_err_plot = color_literature.pdf_std()
    diff_mag_plot_1 = diff_mag_1.pdf_median()
    diff_mag_plot_2 = diff_mag_2.pdf_median()

    #   Set
    sigma: np.ndarray = np.array(color_literature_err_plot)

    #   Fit
    z_1, z_1_err, color_correction_filter_1, color_correction_filter_1_err = utilities.fit_curve(
        fit_func,
        color_literature_plot,
        diff_mag_plot_1,
        x0,
        sigma,
    )
    z_2, z_2_err, color_correction_filter_2, color_correction_filter_2_err = utilities.fit_curve(
        fit_func,
        color_literature_plot,
        diff_mag_plot_2,
        x0,
        sigma,
    )
    if np.isinf(z_1_err):
        z_1_err = None
    if np.isinf(z_2_err):
        z_2_err = None

    #   Plots magnitude difference (literature vs. measured) vs. color
    plot.plot_transform(
        image.outpath.name,
        filter_list[0],
        filter_list[1],
        color_literature_plot.value,
        diff_mag_plot_1.value,
        z_1,
        color_correction_filter_1,
        color_correction_filter_1_err,
        fit_func,
        image.air_mass,
        filter_=filter_list[id_current_filter],
        color_literature_err=color_literature_err_plot.value,
        fit_variable_err=z_1_err,
        name_object=image.object_name,
        image_id=image.pd,
    )

    if id_current_filter == 0:
        other_filter = filter_list[1]
    else:
        other_filter = filter_list[0]
    plot.plot_transform(
        image.outpath.name,
        filter_list[0],
        filter_list[1],
        color_literature_plot.value,
        diff_mag_plot_2.value,
        z_2,
        color_correction_filter_2,
        color_correction_filter_2_err,
        fit_func,
        image.air_mass,
        filter_=other_filter,
        color_literature_err=color_literature_err_plot.value,
        fit_variable_err=z_2_err,
        name_object=image.object_name,
        image_id=image.pd,
    )

    color_correction_filter_1 = unc.normal(
        color_correction_filter_1 * u.mag,
        std=color_correction_filter_1_err * u.mag,
        n_samples=distribution_samples,
    )
    color_correction_filter_2 = unc.normal(
        color_correction_filter_2 * u.mag,
        std=color_correction_filter_2_err * u.mag,
        n_samples=distribution_samples,
    )

    return color_correction_filter_1, color_correction_filter_2


# def transformation_core(
#         image, magnitudes_literature_filter_1, magnitudes_literature_filter_2,
#         calib_magnitudes_observed_filter_1, calib_magnitudes_observed_filter_2,
#         magnitudes_filter_1, magnitudes_filter_2, magnitudes, tc_c, tc_color,
#         tc_t1, tc_k1, tc_t2, tc_k2, id_current_filter, filter_list,
#         transformation_type='derive', distribution_samples=1000):
#     """
#         Routine that performs the actual magnitude transformation.
#
#         Parameters
#         ----------
#         image                                : `image.class`
#             Image class with all image specific properties
#
#         magnitudes_literature_filter_1      : `astropy.uncertainty.core.QuantityDistribution`
#             Magnitudes of calibration stars from the literature
#             for filter 1.
#
#         magnitudes_literature_filter_2      : `astropy.uncertainty.core.QuantityDistribution`
#             Magnitudes of calibration stars from the literature
#             for filter 1.
#
#         calib_magnitudes_observed_filter_1   : `astropy.uncertainty.core.QuantityDistribution`
#             Extracted magnitudes of the calibration stars from filter 1
#
#         calib_magnitudes_observed_filter_2   : `astropy.uncertainty.core.QuantityDistribution`
#             Extracted magnitudes of the calibration stars from filter 2
#
#         magnitudes_filter_1                  : `astropy.uncertainty.core.QuantityDistribution`
#             Extracted magnitudes of objects from filter 1
#
#         magnitudes_filter_2                  : `astropy.uncertainty.core.QuantityDistribution`
#             Extracted magnitudes of objects from filter 2
#
#         magnitudes                           : `astropy.uncertainty.core.QuantityDistribution`
#             Extracted magnitudes for the current filter
#
#         tc_c                                 : `float` or `ufloat`
#             Calibration parameter for the magnitude transformation
#
#         tc_color                             : `float` or `ufloat`
#             Calibration parameter for the magnitude transformation
#
#         tc_t1                                : `float` or `ufloat`
#             Calibration parameter for the magnitude transformation
#
#         tc_k1                                : `float` or `ufloat`
#             Calibration parameter for the magnitude transformation
#
#         tc_t2                                : `float` or `ufloat`
#             Calibration parameter for the magnitude transformation
#
#         tc_k2                                : `float` or `ufloat`
#             Calibration parameter for the magnitude transformation
#
#         id_current_filter                    : `integer`
#             ID of the current filter
#
#         filter_list                          : `list` - `string`
#             List of filter
#
#         transformation_type                  : `string`, optional
#             Type of magnitude transformation.
#             Possibilities: simple, air_mass, or derive
#             Default is ``derive``.
#
#         distribution_samples                : `integer`, optional
#             Number of samples used for distributions
#             Default is `1000`
#
#         Returns
#         -------
#         color_observed                      : `astropy.uncertainty.core.QuantityDistribution`
#             Observed color of the calibration stars
#
#         color_literature                    : `astropy.uncertainty.core.QuantityDistribution`
#             Literature color of the calibration stars
#     """
#     #   Get clipped zero points
#     zp = image.zp
#
#     #   Instrument color of the calibration objects
#     color_observed = (calib_magnitudes_observed_filter_1 -
#                       calib_magnitudes_observed_filter_2)
#
#     #   Literature color of the calibration objects
#     color_literature = (magnitudes_literature_filter_1 -
#                         magnitudes_literature_filter_2)
#
#     ###
#     #   Apply magnitude transformation and calibration
#     #
#     #   Color
#     color = magnitudes_filter_1 - magnitudes_filter_2
#
#     #   Distinguish between versions
#     if transformation_type == 'simple':
#         #   Calculate calibration factor
#         c = tc_c * tc_color * u.mag
#     elif transformation_type == 'air_mass':
#         #   Calculate calibration factor
#         c_1 = tc_t1 * u.mag - tc_k1 * image.air_mass * u.mag
#         c_2 = tc_t2 * u.mag - tc_k2 * image.air_mass * u.mag
#
#     elif transformation_type == 'derive':
#         #   Calculate color correction coefficients
#         c_1, c_2 = derive_transformation_onthefly(
#             image,
#             filter_list,
#             id_current_filter,
#             color_literature,
#             magnitudes_literature_filter_1,
#             magnitudes_literature_filter_2,
#             calib_magnitudes_observed_filter_1,
#             calib_magnitudes_observed_filter_2,
#             distribution_samples=distribution_samples,
#         )
#
#     else:
#         raise Exception(
#             f"{style.Bcolors.FAIL}\nType of magnitude transformation not known"
#             "\n\t-> Check calibration coefficients \n\t-> Exit"
#             f"{style.Bcolors.ENDC}"
#         )
#
#     if transformation_type in ['air_mass', 'derive']:
#         #   Calculate C or more precise C'
#         denominator = 1. * u.mag - c_1 + c_2
#
#         if id_current_filter == 0:
#             c = c_1 / denominator
#         elif id_current_filter == 1:
#             c = c_2 / denominator
#         else:
#             raise Exception(
#                 f"{style.Bcolors.FAIL} \nMagnitude transformation: filter "
#                 "combination not valid \n\t-> This should never happen. The "
#                 f"current filter ID is {id_current_filter}{style.Bcolors.ENDC}"
#             )
#
#     #   Calculate calibrated magnitudes
#     phase_1_calibrated_magnitudes = magnitudes + c * color
#     phase_1_calibrated_magnitudes = phase_1_calibrated_magnitudes.reshape(
#         phase_1_calibrated_magnitudes.size,
#         1,
#     )
#     calibrated_magnitudes_zero_point = zp - c * color_observed
#     calibrated_magnitudes = phase_1_calibrated_magnitudes + calibrated_magnitudes_zero_point
#
#     # color_term = c * color
#     # color_term = color_term.reshape(color_term.size, 1)
#     # color_term_calibration = c * color_observed
#     # calibrated_magnitudes = calibrated_magnitudes_array + color_term - color_term_calibration
#
#     #   Sigma clipping to rm outliers
#     _, median, stddev = sigma_clipped_stats(
#         calibrated_magnitudes.distribution,
#         sigma=1.5,
#         axis=(1,2),
#     )
#
#     #   Add calibrated photometry to table of Image object
#     #   TODO: Add the photometry with filter_list information such that it is
#     #         clear how the magnitudes are derived
#     image.photometry['mag_cali_trans'] = median
#     image.photometry['mag_cali_trans_unc'] = stddev
#
#     return color_observed, color_literature


def transformation_core(
        image: 'analyze.ImageEnsemble.Image',
        magnitudes_literature_filter_1: unc,
        magnitudes_literature_filter_2: unc,
        calib_magnitudes_observed_filter_1: unc,
        calib_magnitudes_observed_filter_2: unc,
        magnitudes_filter_1: unc, magnitudes_filter_2: unc, tc_c: float,
        tc_color: float, tc_t1: float, tc_k1: float, tc_t2: float,
        tc_k2: float, id_current_filter: int, filter_list: list[str],
        transformation_type: str = 'derive',
        distribution_samples: int = 1000) -> tuple[unc, unc]:
    """
        Routine that performs the actual magnitude transformation.

        Parameters
        ----------
        image
            Object with all image specific properties

        magnitudes_literature_filter_1
            Magnitudes of calibration stars from the literature
            for filter 1.

        magnitudes_literature_filter_2
            Magnitudes of calibration stars from the literature
            for filter 1.

        calib_magnitudes_observed_filter_1
            Extracted magnitudes of the calibration stars from filter 1

        calib_magnitudes_observed_filter_2
            Extracted magnitudes of the calibration stars from filter 2

        magnitudes_filter_1
            Extracted magnitudes of objects from filter 1

        magnitudes_filter_2
            Extracted magnitudes of objects from filter 2

        tc_c
            Calibration parameter for the magnitude transformation

        tc_color
            Calibration parameter for the magnitude transformation

        tc_t1
            Calibration parameter for the magnitude transformation

        tc_k1
            Calibration parameter for the magnitude transformation

        tc_t2
            Calibration parameter for the magnitude transformation

        tc_k2
            Calibration parameter for the magnitude transformation

        id_current_filter
            ID of the current filter

        filter_list
            List of filter

        transformation_type
            Type of magnitude transformation.
            Possibilities: simple, air_mass, or derive
            Default is ``derive``.

        distribution_samples
            Number of samples used for distributions
            Default is `1000`

        Returns
        -------
        color_observed
            Observed color of the calibration stars

        color_literature
            Literature color of the calibration stars
    """
    #   Instrument color of the calibration objects
    color_observed = (calib_magnitudes_observed_filter_1 -
                      calib_magnitudes_observed_filter_2)

    #   Literature color of the calibration objects
    color_literature = (magnitudes_literature_filter_1 -
                        magnitudes_literature_filter_2)

    ###
    #   Apply magnitude transformation and calibration
    #
    #   Color
    color = magnitudes_filter_1 - magnitudes_filter_2

    #   Distinguish between versions
    if transformation_type == 'simple':
        #   Calculate calibration factor
        c = tc_c * tc_color * u.mag

    elif transformation_type in ['air_mass', 'derive']:
        if transformation_type == 'air_mass':
            #   Calculate calibration factor
            c_1 = tc_t1 * u.mag - tc_k1 * image.air_mass * u.mag
            c_2 = tc_t2 * u.mag - tc_k2 * image.air_mass * u.mag

        elif transformation_type == 'derive':
            #   Calculate color correction coefficients
            c_1, c_2 = derive_transformation_onthefly(
                image,
                filter_list,
                id_current_filter,
                color_literature,
                magnitudes_literature_filter_1,
                magnitudes_literature_filter_2,
                calib_magnitudes_observed_filter_1,
                calib_magnitudes_observed_filter_2,
                distribution_samples=distribution_samples,
            )
        else:
            raise Exception(
                f"{style.Bcolors.FAIL} \nThis should never happen. Error "
                f"in transformation calculation. {style.Bcolors.ENDC}"
            )

        #   Calculate C or more precise C'
        denominator = 1. * u.mag - c_1 + c_2

        if id_current_filter == 0:
            c = c_1 / denominator
        elif id_current_filter == 1:
            c = c_2 / denominator
        else:
            raise Exception(
                f"{style.Bcolors.FAIL} \nMagnitude transformation: filter "
                "combination not valid \n\t-> This should never happen. The "
                f"current filter ID is {id_current_filter}{style.Bcolors.ENDC}"
            )
    else:
        raise Exception(
            f"{style.Bcolors.FAIL}\nType of magnitude transformation not known"
            "\n\t-> Check calibration coefficients \n\t-> Exit"
            f"{style.Bcolors.ENDC}"
        )

    #   Calculate calibrated magnitudes
    color_term_all = c * color
    color_term_all = color_term_all.reshape(color_term_all.size, 1)
    color_term_calibration = c * color_observed
    calibrated_magnitudes = unc.Distribution(image.magnitudes_with_zp) + color_term_all - color_term_calibration

    #   Sigma clipping to rm outliers
    _, median, stddev = sigma_clipped_stats(
        calibrated_magnitudes.distribution,
        sigma=1.5,
        axis=(1,2),
    )

    #   Add calibrated photometry to table of Image object
    #   TODO: Add the photometry with filter_list information such that it is
    #         clear how the magnitudes are derived
    image.photometry['mag_cali_trans'] = median
    image.photometry['mag_cali_trans_unc'] = stddev

    return color_observed, color_literature


def apply_magnitude_transformation(
        calibration_stars_ids: np.ndarray, image: 'analyze.ImageEnsemble.Image',
        calib_magnitudes_literature: list[u.quantity.Quantity],
        magnitudes_calibration_current_image: u.quantity.Quantity,
        magnitudes_calibration_comparison_image: u.quantity.Quantity,
        magnitudes_current_image: u.quantity.Quantity,
        magnitudes_comparison_image: u.quantity.Quantity, filter_id: int,
        filter_list: list[str],
        transformation_coefficients: dict[str, (float | str)],
        plot_sigma: bool = False, transformation_type: str = 'derive',
        distribution_samples: int = 1000, multiprocessing: bool = False
    ) -> tuple[int, Table] | None:
    """
    Apply transformation

    Parameters
    ----------
    calibration_stars_ids
        IDs of the stars for which calibration data is available

    image
        Object with all image specific properties

    calib_magnitudes_literature
        Literature magnitudes for the calibration stars

    magnitudes_calibration_current_image
        Observed magnitudes of the calibration stars in the current filter

    magnitudes_calibration_comparison_image
        Observed magnitudes of the calibration stars in comparison filter

    magnitudes_current_image
        Observed magnitudes in the current filter

    magnitudes_comparison_image
        Observed magnitudes in the comparison filter

    filter_id
        ID of the current filter

    filter_list
        List of filter

    transformation_coefficients
        Calibration coefficients for magnitude transformation

    plot_sigma
        If True sigma clipped magnitudes will be plotted.
        Default is ``False``.

    transformation_type
        Type of magnitude transformation.
        Possibilities: simple, air_mass, or derive
        Default is ``derive``.

    distribution_samples
        Number of samples used for distributions
        Default is `1000`.

    multiprocessing
        Switch to distinguish between single and multicore processing
        Default is ``False``.

    """
    #   Restore magnitudes as distributions
    #   -> This is necessary since astropy QuantityDistribution cannot be
    #      prickled/serialized
    #   TODO: Check if this workaround is still necessary
    tmp_list = []
    for magnitudes in calib_magnitudes_literature:
        tmp_list.append(unc.Distribution(magnitudes))
    calib_magnitudes_literature = tmp_list

    magnitudes_calibration_current_image = unc.Distribution(magnitudes_calibration_current_image)
    magnitudes_calibration_comparison_image = unc.Distribution(magnitudes_calibration_comparison_image)
    magnitudes_current_image = unc.Distribution(magnitudes_current_image)
    magnitudes_comparison_image = unc.Distribution(magnitudes_comparison_image)

    #   Sort magnitudes for color computations. The current and comparison
    #   magnitudes can be different, but for color, a specific combination such
    #   as B-V is required.
    if filter_id == 0:
        magnitudes_calibration_first_filter = magnitudes_calibration_current_image
        magnitudes_calibration_second_filter = magnitudes_calibration_comparison_image
        magnitudes_first_filter = magnitudes_current_image
        magnitudes_second_filter = magnitudes_comparison_image
    else:
        magnitudes_calibration_first_filter = magnitudes_calibration_comparison_image
        magnitudes_calibration_second_filter = magnitudes_calibration_current_image
        magnitudes_first_filter = magnitudes_comparison_image
        magnitudes_second_filter = magnitudes_current_image

    #   Prepare calibration parameters
    tc_t1: float | None = None
    tc_k1: float | None = None
    tc_t2: float | None = None
    tc_k2: float | None = None
    tc_c: float | None = None
    tc_color: float | None = None
    if transformation_type == 'simple':
        tc_c = transformation_coefficients['C']
        tc_color = transformation_coefficients['color']
    elif transformation_type == 'air_mass':
        tc_t1 = transformation_coefficients['T_1']
        tc_k1 = transformation_coefficients['k_1']
        tc_t2 = transformation_coefficients['T_2']
        tc_k2 = transformation_coefficients['k_2']

    color_observed, color_literature = transformation_core(
        image,
        calib_magnitudes_literature[0],
        calib_magnitudes_literature[1],
        magnitudes_calibration_first_filter,
        magnitudes_calibration_second_filter,
        magnitudes_first_filter,
        magnitudes_second_filter,
        tc_c,
        tc_color,
        tc_t1,
        tc_k1,
        tc_t2,
        tc_k2,
        filter_id,
        filter_list,
        transformation_type=transformation_type,
        distribution_samples=distribution_samples,
    )

    #   Quality control plots
    #   TODO: Rename to transformation_check_plots?
    #   TODO: Move to a different place to allow also plot using simple calibration
    utilities.calibration_check_plots(
        filter_list[filter_id],
        image.outpath.name,
        image.object_name,
        image.pd,
        filter_list,
        # image.zp_mask,
        color_observed.pdf_median(),
        color_literature.pdf_median(),
        calibration_stars_ids,
        calib_magnitudes_literature[filter_id].pdf_median(),
        image.photometry['mag_cali_trans'],
        magnitudes_current_image.pdf_median(),
        color_observed_err=color_observed.pdf_std(),
        color_literature_err=color_literature.pdf_std(  ),
        literature_magnitudes_err=calib_magnitudes_literature[filter_id].pdf_std(),
        magnitudes_err=image.photometry['mag_cali_trans_unc'],
        uncalibrated_magnitudes_err=magnitudes_current_image.pdf_std(),
        plot_sigma_switch=plot_sigma,
    )

    if multiprocessing:
        return copy.deepcopy(image.pd), copy.deepcopy(image.photometry)


def calibrate_simple(
        image: 'analyze.ImageEnsemble.Image' , not_calibrated_magnitudes: unc,
        zp: unc,
        ) -> tuple[Table, unc]:
    """
        Calibrate magnitudes without magnitude transformation

        Parameters
        ----------
        image
            Object with all image specific properties

        not_calibrated_magnitudes
            Distribution of uncalibrated magnitudes

        zp
            Zero pint of the photometric calibration
    """
    #   Get photometry table
    photometry_table = image.photometry

    #   Reshape the magnitudes to allow broadcasting because zp is an array
    reshaped_magnitudes = not_calibrated_magnitudes.reshape(
        not_calibrated_magnitudes.size,
        1,
    )

    #   Calculate calibrated magnitudes
    calibrated_magnitudes = reshaped_magnitudes + zp

    #   Sigma clipping to rm outliers
    _, median, stddev = sigma_clipped_stats(
        calibrated_magnitudes.distribution,
        sigma=1.5,
        axis=(1,2),
    )

    #   TODO: Rewrite & test this
    #   If ZP is 0, calibrate with the median of all magnitudes
    # if np.all(zp == 0.):
    # if zp == 0.:
    #     calibrated_magnitudes = not_calibrated_magnitudes - np.median(not_calibrated_magnitudes)

    #   Add calibrated photometry to table of Image object
    photometry_table['mag_cali_no-trans'] = median
    photometry_table['mag_cali_no-trans_unc'] = stddev

    return photometry_table, calibrated_magnitudes


#   TODO: combine the next two functions?
def flux_calibration_ensemble(image_ensemble, distribution_samples=1000):
    """
        Simple calibration for flux values. Assuming the median over all
        objects in an image as a quasi ZP.

        Parameters
        ----------
        image_ensemble        : `image.ensemble`
            Image ensemble object with flux and magnitudes of all objects in
            all images within the ensemble

        distribution_samples  : `integer`, optional
            Number of samples used for distributions
            Default is `1000`.
    """
    #   Get flux as numpy array
    flux, flux_error = image_ensemble.get_flux_array()

    #   Derive median of flux in individual images
    _, median, _ = sigma_clipped_stats(
        flux,
        axis=1,
        sigma=1.5,
        mask_value=0.0,
    )

    #   Normalize the flux of all objects with the median flux in the
    #   corresponding images
    flux_distribution = unc.normal(
        flux,
        std=flux_error,
        n_samples=distribution_samples,
    )
    #   TODO: Check if a distribution would make sense here
    normalization_factor = median[:, np.newaxis]
    flux_calibrated = flux_distribution / normalization_factor

    #   Add to ensemble
    image_ensemble.quasi_calibrated_flux = flux_calibrated


#   TODO: Check if this can be improved based on distributions
def flux_normalization_ensemble(image_ensemble, distribution_samples=1000):
    """
        Normalize flux of each object

        Parameters
        ----------
        image_ensemble        : `image.ensemble`
            Image ensemble object with flux and magnitudes of all objects in
            all images within the ensemble

        distribution_samples  : `integer`, optional
            Number of samples used for distributions
            Default is `1000`.
    """
    #   Get list with flux distributions for the individual images
    # try:
    #     flux_list = image_ensemble.quasi_calibrated_flux
    # except AttributeError:
    #     flux_list = image_ensemble.get_flux_distribution()
    try:
        flux_distribution = image_ensemble.quasi_calibrated_flux
        flux = flux_distribution.pdf_median()
        flux_error = flux_distribution.pdf_std()
    except AttributeError:
        flux, flux_error = image_ensemble.get_flux_array()
        flux_distribution = unc.normal(
            flux,
            std=flux_error,
            n_samples=distribution_samples,
        )

    # normalized_flux = []
    # for flux in flux_list:
    #     flux_values = flux.pdf_median()
    #     if isinstance(flux_values, u.quantity.Quantity):
    #         flux_values = flux_values.value

    #     #   Calculated sigma clipped magnitudes
    #     _, median, std = sigma_clipped_stats(
    #         flux_values,
    #         axis=0,
    #         sigma=1.5,
    #         mask_value=0.0,
    #     )

    #     #   Add axis so that broadcasting to original array is possible
    #     # median_reshape = median[np.newaxis, :]
    #     # std_dev_reshape = stddev[np.newaxis, :]

    #     #   Normalized magnitudes
    #     normalization_factor = unc.normal(
    #         # median_reshape * u.mag,
    #         # std=std_dev_reshape * u.mag,
    #         # median * u.mag,
    #         # std=std * u.mag,
    #         median,
    #         std=std,
    #         n_samples=distribution_samples,
    #     )

    #     normalized_flux.append(flux / normalization_factor)

    #   Calculated sigma clipped magnitudes
    _, median, stddev = sigma_clipped_stats(
        flux,
        axis=0,
        sigma=1.5,
        mask_value=0.0,
    )
    # print(median)
    # print(stddev)
    # print(median.shape)
    # print(flux_distribution)
    # print(flux_distribution.shape)

    #   Prepare distributions 
    normalization_factor = unc.normal(
        median,
        std=stddev,
        n_samples=distribution_samples,
    )
    normalized_flux = flux_distribution / normalization_factor
    # print('+++++++++++++++++++++')
    # print(normalized_flux)

    image_ensemble.quasi_calibrated_flux_normalized = normalized_flux


def prepare_zero_point(
        image: 'analyze.ImageEnsemble.Image', id_filter: int,
        literature_magnitude_list: list[unc], magnitudes_calibration_stars: unc,
        calculate_zero_point_statistic: bool = True,
        distribution_samples: int = 1000) -> unc:
    """
        Calculate zero point values based on calibration stars and
        sigma clip these values before calculating median

        Parameters
        ----------
        image
            Class with all image specific properties

        id_filter
            ID of the current filter

        literature_magnitude_list
            Literature magnitudes

        magnitudes_calibration_stars
            Observed magnitudes of the objects that were used for the
            calibration from the image of filter 1

        calculate_zero_point_statistic
            If `True` a statistic on the zero points will be calculated.
            Default is ``True``.

        distribution_samples
            Number of samples used for distributions
            Default is `1000`.

    """
    #   Calculate zero point
    zp = literature_magnitude_list[id_filter] - magnitudes_calibration_stars
    # terminal_output.print_to_terminal(
    #     f"It is not possible to calculate the zero point.",
    #     style_name='ERROR',
    # )
    # image.zp = zp

    #   Plot zero point statistics
    plot.histogram_statistic(
        [zp.pdf_median()],
        f'Zero point ({image.filt})',
        '',
        f'histogram_zero_point_{image.filt}',
        image.outpath,
        dataset_label=[
            ['All calibration objects'],
        ],
        name_object=image.object_name,
    )

    #   TODO: Replace with distribution properties?
    #   TODO: Add random selection of calibration stars -> calculate variance
    n_calibration_objects = zp.shape[0]
    if n_calibration_objects > 20 and calculate_zero_point_statistic:
        #   Create samples using numpy's random number generator to generate
        #   an index array
        n_objects_sample = int(n_calibration_objects * 0.6)
        rng = np.random.default_rng()
        random_index = rng.integers(
            0,
            high=n_calibration_objects,
            size=(distribution_samples, n_objects_sample),
        )

        samples = zp.pdf_median()[random_index]

        #   Get statistic
        # mean_samples = np.mean(sample_values, axis=1)
        median_samples = np.median(samples, axis=1)
        median_over_samples = np.median(median_samples)
        standard_deviation_over_samples = np.std(median_samples)

        terminal_output.print_to_terminal(
            f"Based on {distribution_samples} randomly selected sub-samples, ",
            indent=3,
            style_name='UNDERLINE'
        )
        terminal_output.print_to_terminal(
            f"the following statistic is obtained for the zero points:",
            indent=3,
            style_name='UNDERLINE'
        )
        terminal_output.print_to_terminal(
            f"median = {median_over_samples:5.3f} - "
            f"standard deviation = {standard_deviation_over_samples:5.3f}",
            indent=3,
            style_name='UNDERLINE'
        )
        terminal_output.print_to_terminal(
            f"The sample size was {n_objects_sample}.",
            indent=3,
            style_name='UNDERLINE'
        )

    return zp


def calibrate_magnitudes_zero_point_core(
        current_image: 'analyze.ImageEnsemble.Image',
        index_calibration_stars: np.ndarray, current_filter_id: int,
        literature_magnitudes: list[u.quantity.Quantity],
        calculate_zero_point_statistic: bool = True,
        distribution_samples: int = 1000, multiprocessing: bool = False
        ) -> tuple[int, Table, u.quantity.Quantity]:
    """
    Core module for zero point calibration that allows also for multicore
    processing

    Parameters
    ----------
    current_image
        Image object of the image that is processed, containing the
        specific image properties

    index_calibration_stars
        IDs of the stars for which calibration data is available

    current_filter_id
        ID of the current filter

    literature_magnitudes
        Literature magnitudes of the calibration stars

    calculate_zero_point_statistic
        If `True` a statistic on the zero points will be calculated.
        Default is ``True``.

    distribution_samples
        Number of samples used for distributions
        Default is `1000`.

    multiprocessing
        Switch to distinguish between single and multicore processing
        Default is ``False``.

    Returns
    -------
    pd
        ID of the image

    zp
        Zero point for the image

    tbl
        Table with the photometric data

    magnitudes
        Array with calibrated magnitudes
    """
    #   Restore the literature magnitudes as distributions
    #   -> This is necessary since astropy QuantityDistribution cannot be
    #      prickled/serialized
    #   TODO: Check if this workaround is still necessary
    tmp_list = []
    for magnitudes in literature_magnitudes:
        tmp_list.append(unc.Distribution(magnitudes))
    literature_magnitudes = tmp_list

    #   Get magnitude array for first image
    magnitudes_current_image = utilities.distribution_from_table(
        current_image,
        distribution_samples=distribution_samples,
    )

    #   Get extracted magnitudes of the calibration stars for the
    #   current image
    magnitudes_calibration_current_image = calib.observed_magnitude_of_calibration_stars(
        magnitudes_current_image,
        index_calibration_stars,
    )

    #   Prepare ZP for the magnitude calibration
    zp = prepare_zero_point(
        current_image,
        current_filter_id,
        literature_magnitudes,
        magnitudes_calibration_current_image,
        calculate_zero_point_statistic=calculate_zero_point_statistic,
        distribution_samples=distribution_samples,
    )

    #   Calibration without transformation
    photometry_table, calibrated_magnitudes = calibrate_simple(
        current_image,
        magnitudes_current_image,
        zp,
    )

    if multiprocessing:
        pd = copy.deepcopy(current_image.pd)
        tbl = copy.deepcopy(photometry_table)
        magnitudes = copy.deepcopy(calibrated_magnitudes.distribution)

        return pd, tbl, magnitudes


def calibrate_magnitudes_zero_point(
        image_container: 'analyze.ImageContainer', filter_list: (list[str] | set[str]),
        distribution_samples: int = 1000, calculate_zero_point_statistic: bool = True,
        id_object: (int | None) = None, photometry_extraction_method: str = '',
        indent: int = 1) -> None:
    """
    Apply the zero points to the magnitudes

    Parameters
    ----------
    image_container
        Container object with image ensemble objects for each filter

    filter_list
        Filter names

    distribution_samples
        Number of samples used for distributions
        Default is `1000`.

    calculate_zero_point_statistic
        If `True` a statistic on the zero points will be calculated.
        Default is ``True``.

    id_object
        ID of the object
        Default is ``None``.

    photometry_extraction_method
        Applied extraction method. Possibilities: ePSF or APER`
        Default is ``''``.

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    terminal_output.print_to_terminal(
        "Apply zero point to magnitudes",
        indent=indent,
    )

    #   Get image ensembles
    image_ensembles = image_container.ensembles

    #   Get calibration magnitudes
    literature_magnitudes = calib.distribution_from_calibration_table(
        image_container.CalibParameters,
        filter_list,
        distribution_samples=distribution_samples,
    )

    #   TODO: Prepare this for multithreading
    for current_filter_id, filter_ in enumerate(filter_list):
        #   Get image ensemble
        image_ensemble = image_ensembles[filter_]

        #   Get image list
        image_list = image_ensemble.image_list

        #   Initialize multiprocessing object
        n_cores_multiprocessing = 12
        executor = utilities.Executor(n_cores_multiprocessing)

        #   Get IDs calibration data
        index_calibration_stars = getattr(
            image_container.CalibParameters,
            'inds',
            None,
        )

        #   Loop over images
        for current_image_id, current_image in enumerate(image_list):
            executor.schedule(
                calibrate_magnitudes_zero_point_core,
                args=(
                    current_image,
                    index_calibration_stars,
                    current_filter_id,
                    literature_magnitudes,
                ),
                kwargs={
                    'calculate_zero_point_statistic': calculate_zero_point_statistic,
                    'distribution_samples': distribution_samples,
                    'multiprocessing': True,
                }
            )

        #   Exit multiprocessing, if exceptions will occur
        if executor.err is not None:
            raise RuntimeError(
                f'\n{style.Bcolors.FAIL}Zero point calibration using '
                f' multiprocessing failed for {filter_} :({style.Bcolors.ENDC}'
            )

        #   Close multiprocessing pool and wait until it finishes
        executor.wait()

        #   Extract results
        res = executor.res

        #   Sort multiprocessing results
        tmp_list = []
        for image_ in image_ensemble.image_list:

            for pd, tbl, magnitudes in res:
                if pd == image_.pd:
                    image_.photometry = tbl
                    image_.magnitudes_with_zp = magnitudes
                    tmp_list.append(image_)

        # TODO: Check if this is necessary
        image_ensemble.image_list = tmp_list

    #   Save results as ASCII files
    #   Make astropy table
    table_not_transformed_magnitudes, array_not_transformed_magnitudes = utilities.mk_magnitudes_table_and_array(
        image_container,
        filter_list,
        'mag_cali_no-trans',
    )

    #   TODO: This is also messy and needs a cleanup
    #   Add table and array to container
    image_container.table_mags_not_transformed = table_not_transformed_magnitudes
    image_container.array_mags_not_transformed = array_not_transformed_magnitudes

    #   Save to file
    utilities.save_magnitudes_ascii(
        image_container,
        table_not_transformed_magnitudes,
        trans=False,
        id_object=id_object,
        photometry_extraction_method=photometry_extraction_method,
    )


def calibrate_magnitudes_transformation(
        image_container: 'analyze.ImageContainer', filter_list: (list[str] | set[str]),
        transformation_coefficients: dict[str, (float | str)] = None,
        derive_transformation_coefficients: bool = False, plot_sigma: bool = False,
        distribution_samples: int = 1000, calculate_zero_point_statistic: bool = True,
        id_object: (int | None) = None, photometry_extraction_method: str = '',
        indent: int = 1) -> None:
    """
    Apply magnitude transformation

    # Using:
    # Δ(b-v) = (b-v)obj - (b-v)cali
    # Δ(B-V) = Tbv * Δ(b-v)
    # Vobj = Δv + Tv_bv * Δ(B-V) + Vcomp or Vobj
           = v + Tv_bv*Δ(B-V) - v_cali

    Parameters
    ----------
    image_container
        Container object with image ensemble objects for each filter

    filter_list
        Filter names

    transformation_coefficients
        Calibration coefficients for the magnitude transformation
        Default is ``None``.

    derive_transformation_coefficients
        If True the magnitude transformation coefficients will be
        calculated from the current data even if calibration coefficients
        are available in the database.
        Default is ``False``

    plot_sigma
        If True sigma clipped magnitudes will be plotted.
        Default is ``False``.

    id_object
        ID of the object
        Default is ``None``.

    photometry_extraction_method
        Applied extraction method. Possibilities: ePSF or APER`
        Default is ``''``.

    calculate_zero_point_statistic
        If `True` a statistic on the zero points will be calculated.
        Default is ``True``.

    distribution_samples
        Number of samples used for distributions
        Default is `1000`

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    terminal_output.print_to_terminal(
        "Apply magnitude transformation",
        indent=indent,
    )

    #   Get image ensembles
    image_series_dict = image_container.ensembles

    #   Initialize list for
    transformation_type_list = []

    #   Get calibration magnitudes
    literature_magnitudes = calib.distribution_from_calibration_table(
        image_container.CalibParameters,
        filter_list,
        distribution_samples=distribution_samples,
    )

    #   Get IDs calibration data
    #   TODO: Check if these IDs apply to all filter/image series
    index_calibration_stars = getattr(
        image_container.CalibParameters,
        'inds',
        None,
    )

    for current_filter_id, filter_ in enumerate(filter_list):
        #   Get image ensemble
        current_image_series = image_series_dict[filter_]

        #   Get image list
        image_list = current_image_series.image_list
        n_images = len(image_list)

        #   Prepare transformation
        transformation_type, comparison_filter_id, trans_coefficients = check_transformation_requirements(
            image_container,
            transformation_coefficients,
            filter_list,
            current_filter_id,
            derive_transformation_coefficients,
        )
        transformation_type_list.append(transformation_type)

        if transformation_type is not None:

            #   Initialize multiprocessing object
            n_cores_multiprocessing = 12
            executor = utilities.Executor(n_cores_multiprocessing)

            for current_image_id, current_image in enumerate(image_list):
                #   Get magnitude array for first image
                magnitudes_current_image = utilities.distribution_from_table(
                    current_image,
                    distribution_samples=distribution_samples,
                )

                #   The '.distribution' below is currently necessary for the multicore
                #   processing below, because astropy QuantityDistribution cannot be
                #   prickled/serialized
                #   TODO: Check if this workaround is still necessary
                magnitudes_current_image = magnitudes_current_image.distribution

                #   Get extracted magnitudes of the calibration stars for the
                #   current image
                magnitudes_calibration_current_image = calib.observed_magnitude_of_calibration_stars(
                    magnitudes_current_image,
                    index_calibration_stars,
                )

                #   Prepare some variables and find corresponding image to
                #   current_image
                comparison_image = find_best_comparison_image_second_filter(
                    image_series_dict,
                    current_image,
                    comparison_filter_id,
                    filter_list,
                )

                #   Get magnitude array for comparison image
                magnitudes_comparison_image = utilities.distribution_from_table(
                    comparison_image,
                    distribution_samples=distribution_samples,
                )

                #   The '.distribution' below is currently necessary for the multicore
                #   processing below, because astropy QuantityDistribution cannot be
                #   prickled/serialized
                #   TODO: Check if this workaround is still necessary
                magnitudes_comparison_image = magnitudes_comparison_image.distribution

                #   Get extracted magnitudes of the calibration stars
                #   for the image in the comparison filter
                #   -> required for magnitude transformation
                magnitudes_calibration_comparison_image = calib.observed_magnitude_of_calibration_stars(
                    magnitudes_comparison_image,
                    index_calibration_stars,
                )

                executor.schedule(
                    apply_magnitude_transformation,
                    args=(
                        index_calibration_stars,
                        current_image,
                        literature_magnitudes,
                        magnitudes_calibration_current_image,
                        magnitudes_calibration_comparison_image,
                        magnitudes_comparison_image,
                        magnitudes_current_image,
                        current_filter_id,
                        filter_list,
                        transformation_coefficients,
                    ),
                    kwargs={
                        'plot_sigma': plot_sigma,
                        'transformation_type': transformation_type,
                        'distribution_samples': distribution_samples,
                        'multiprocessing': True,
                    }
                )

                #   Progress bar
                progress_bar(current_image_id, n_images)

            #   Exit multiprocessing, if exceptions will occur
            if executor.err is not None:
                raise RuntimeError(
                    f'\n{style.Bcolors.FAIL}Zero point calibration using '
                    f' multiprocessing failed for {filter_} :({style.Bcolors.ENDC}'
                )

            #   Close multiprocessing pool and wait until it finishes
            executor.wait()

            #   Extract results
            res = executor.res

            #   Sort multiprocessing results
            tmp_list = []
            for image_ in current_image_series.image_list:

                for pd, tbl in res:
                    if pd == image_.pd:
                        image_.photometry = tbl
                        tmp_list.append(image_)

            # TODO: Check if this is necessary
            current_image_series.image_list = tmp_list

            terminal_output.print_to_terminal('')

    ###
    #   Save results as ASCII files
    #
    #   TODO: Remove this from apply calibration and move it to a function called save_calibration
    #         -> put it one level up
    #   With transformation
    if not np.any(np.array(transformation_type_list) == None):
        #   Make astropy table
        table_transformed_magnitudes, array_transformed_magnitudes = utilities.mk_magnitudes_table_and_array(
            image_container,
            filter_list,
            'mag_cali_trans',
        )

        #   Add table to container
        image_container.table_mags_transformed = table_transformed_magnitudes
        image_container.array_mags_transformed = array_transformed_magnitudes

        #   Save to file
        utilities.save_magnitudes_ascii(
            image_container,
            table_transformed_magnitudes,
            trans=True,
            id_object=id_object,
            photometry_extraction_method=photometry_extraction_method,
            rts=f'_{filter_list[0]}-{filter_list[1]}'
        )
    else:
        terminal_output.print_to_terminal(
            "WARNING: No magnitude transformation possible",
            indent=indent,
            style_name='WARNING'
        )


def apply_calibration(
        image_container: 'analyze.ImageContainer', filter_list: (list[str] | set[str]),
        transformation_coefficients_dict: dict[str, (float | str)] = None,
        derive_transformation_coefficients: bool = False, plot_sigma: bool = False,
        id_object: (int | None) = None, photometry_extraction_method: str = '',
        calculate_zero_point_statistic: bool = True, distribution_samples: int = 1000,
        indent: int = 1) -> None:
    """
        Apply the zero points to the magnitudes and perform a magnitude
        transformation if possible

        Parameters
        ----------
        image_container
            Container object with image ensemble objects for each filter

        filter_list
            Filter names

        transformation_coefficients_dict
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_transformation_coefficients
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.
            Default is ``False``

        plot_sigma
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        id_object
            ID of the object
            Default is ``None``.

        photometry_extraction_method
            Applied extraction method. Possibilities: ePSF or APER`
            Default is ``''``.

        calculate_zero_point_statistic
            If `True` a statistic on the zero points will be calculated.
            Default is ``True``.

        distribution_samples
            Number of samples used for distributions
            Default is `1000`

        indent
            Indentation for the console output lines
            Default is ``1``.
    """
    #   Apply zero point calibration
    calibrate_magnitudes_zero_point(
        image_container=image_container,
        filter_list=filter_list,
        distribution_samples=distribution_samples,
        calculate_zero_point_statistic=calculate_zero_point_statistic,
        id_object=id_object,
        photometry_extraction_method=photometry_extraction_method,
        indent=indent,
    )

    #   Apply magnitude transformation
    calibrate_magnitudes_transformation(
        image_container=image_container,
        filter_list=filter_list,
        transformation_coefficients=transformation_coefficients_dict,
        derive_transformation_coefficients=derive_transformation_coefficients,
        distribution_samples=distribution_samples,
        calculate_zero_point_statistic=calculate_zero_point_statistic,
        id_object=id_object,
        photometry_extraction_method=photometry_extraction_method,
        indent=indent,
    )


# def apply_calibration(
#         image_container, filter_list, transformation_coefficients_dict=None,
#         derive_transformation_coefficients=False, plot_sigma=False,
#         id_object=None, photometry_extraction_method='',
#         calculate_zero_point_statistic=True, distribution_samples=1000,
#         indent=1):
#     """
#         Apply the zero points to the magnitudes and perform a magnitude
#         transformation if possible
#
#         # Using:
#         # Δ(b-v) = (b-v)obj - (b-v)cali
#         # Δ(B-V) = Tbv * Δ(b-v)
#         # Vobj = Δv + Tv_bv * Δ(B-V) + Vcomp or Vobj
#                = v + Tv_bv*Δ(B-V) - v_cali
#
#         Parameters
#         ----------
#         image_container                     : `image.container`
#             Container object with image ensemble objects for each filter
#
#         filter_list                         : `list` of `string`
#             Filter names
#
#         transformation_coefficients_dict    : `dictionary`, optional
#             Calibration coefficients for the magnitude transformation
#             Default is ``None``.
#
#         derive_transformation_coefficients  : `boolean`, optional
#             If True the magnitude transformation coefficients will be
#             calculated from the current data even if calibration coefficients
#             are available in the database.
#             Default is ``False``
#
#         plot_sigma                          : `boolean', optional
#             If True sigma clipped magnitudes will be plotted.
#             Default is ``False``.
#
#         id_object                           : `integer` or `None`, optional
#             ID of the object
#             Default is ``None``.
#
#         photometry_extraction_method        : `string`, optional
#             Applied extraction method. Possibilities: ePSF or APER`
#             Default is ``''``.
#
#         calculate_zero_point_statistic      : `boolean`, optional
#             If `True` a statistic on the zero points will be calculated.
#             Default is ``True``.
#
#         distribution_samples                : `integer`, optional
#             Number of samples used for distributions
#             Default is `1000`
#
#         indent                              : `integer`, optional
#             Indentation for the console output lines
#             Default is ``1``.
#     """
#     #   TODO: Separate simple calibration and magnitude transformation -> transform_magnitudes - simple_calibration
#     terminal_output.print_to_terminal(
#         "Apply calibration and perform magnitude transformation",
#         indent=indent,
#     )
#
#     #   Get image ensembles
#     image_ensembles = image_container.ensembles
#
#     #   Initialize list for
#     transformation_type_list = []
#
#     #   Get calibration magnitudes
#     literature_magnitudes = calib.distribution_from_calibration_table(
#         image_container.CalibParameters,
#         filter_list,
#         distribution_samples=distribution_samples,
#     )
#
#     #   TODO: Prepare this for multithreading
#     for current_filter_id, filter_ in enumerate(filter_list):
#         #   Get image ensemble
#         img_ensemble = image_ensembles[filter_]
#
#         #   Get image list
#         image_list = img_ensemble.image_list
#         n_images = len(image_list)
#
#         #   Prepare transformation
#         transformation_type, comparison_filter_id, trans_coefficients = check_transformation_requirements(
#             image_container,
#             transformation_coefficients_dict,
#             filter_list,
#             current_filter_id,
#             derive_transformation_coefficients,
#         )
#         transformation_type_list.append(transformation_type)
#
#         #   Loop over images
#         for current_image_id, current_image in enumerate(image_list):
#             #   Get magnitude array for first image
#             magnitudes_current_image = utilities.distribution_from_table(
#                 current_image,
#                 distribution_samples=distribution_samples,
#             )
#
#             #   Get extracted magnitudes of the calibration stars for the
#             #   current image
#             magnitudes_calibration_current_image = calib.observed_magnitude_of_calibration_stars(
#                 magnitudes_current_image,
#                 image_container,
#             )
#
#             #   Prepare ZP for the magnitude calibration
#             prepare_zero_point(
#                 current_image,
#                 current_filter_id,
#                 literature_magnitudes,
#                 magnitudes_calibration_current_image,
#                 calculate_zero_point_statistic=calculate_zero_point_statistic,
#                 distribution_samples=distribution_samples,
#             )
#
#             #   Calibration without transformation
#             calibrate_simple(current_image, magnitudes_current_image)
#
#             #   Prepare some variables and find corresponding image to
#             #   current_image
#             #   TODO: Replace current_image_id with current_image
#             if transformation_type is not None:
#                 comparison_image = find_best_comparison_image_second_filter(
#                     image_container,
#                     current_image_id,
#                     comparison_filter_id,
#                     current_filter_id,
#                     filter_list,
#                 )
#
#                 #   Get magnitude array for comparison image
#                 magnitudes_comparison_image = utilities.distribution_from_table(
#                     comparison_image,
#                     distribution_samples=distribution_samples,
#                 )
#
#                 #   Get extracted magnitudes of the calibration stars
#                 #   for the image in the comparison filter
#                 #   -> required for magnitude transformation
#                 magnitudes_calibration_comparison_image = calib.observed_magnitude_of_calibration_stars(
#                     magnitudes_comparison_image,
#                     image_container,
#                 )
#
#                 #   TODO: Move this to apply_transform
#                 if current_filter_id == 0:
#                     magnitudes_calibration_first_filter = magnitudes_calibration_current_image
#                     magnitudes_calibration_second_filter = magnitudes_calibration_comparison_image
#                     magnitudes_first_filter = magnitudes_current_image
#                     magnitudes_second_filter = magnitudes_comparison_image
#                 else:
#                     magnitudes_calibration_first_filter = magnitudes_calibration_comparison_image
#                     magnitudes_calibration_second_filter = magnitudes_calibration_current_image
#                     magnitudes_first_filter = magnitudes_comparison_image
#                     magnitudes_second_filter = magnitudes_current_image
#
#                 #   Calculate transformation
#                 apply_magnitude_transformation(
#                     image_container,
#                     current_image,
#                     literature_magnitudes,
#                     magnitudes_calibration_first_filter,
#                     magnitudes_calibration_second_filter,
#                     magnitudes_first_filter,
#                     magnitudes_second_filter,
#                     magnitudes_current_image,
#                     current_filter_id,
#                     filter_list,
#                     trans_coefficients,
#                     plot_sigma=plot_sigma,
#                     transformation_type=transformation_type,
#                     distribution_samples=distribution_samples,
#                 )
#
#             #   Progress bar
#             progress_bar(current_image_id, n_images)
#         terminal_output.print_to_terminal('')
#
#     ###
#     #   Save results as ASCII files
#     #
#     #   TODO: Remove this from apply calibration and move it to a function called save_calibration
#     #         -> put it one level up
#     #   With transformation
#     if not np.any(np.array(transformation_type_list) == None):
#         #   Make astropy table
#         table_transformed_magnitudes, array_transformed_magnitudes = utilities.mk_magnitudes_table_and_array(
#             image_container,
#             filter_list,
#             'mag_cali_trans',
#         )
#
#         #   Add table to container
#         image_container.table_mags_transformed = table_transformed_magnitudes
#         image_container.array_mags_transformed = array_transformed_magnitudes
#
#         #   Save to file
#         utilities.save_magnitudes_ascii(
#             image_container,
#             table_transformed_magnitudes,
#             trans=True,
#             id_object=id_object,
#             photometry_extraction_method=photometry_extraction_method,
#             rts=f'_{filter_list[0]}-{filter_list[1]}'
#         )
#     else:
#         terminal_output.print_to_terminal(
#             "WARNING: No magnitude transformation possible",
#             indent=indent,
#             style_name='WARNING'
#         )
#
#     #   Without transformation
#
#     #   Make astropy table
#     table_not_transformed_magnitudes, array_not_transformed_magnitudes = utilities.mk_magnitudes_table_and_array(
#         image_container,
#         filter_list,
#         'mag_cali_no-trans',
#     )
#
#     #   Add table and array to container
#     image_container.table_mags_not_transformed = table_not_transformed_magnitudes
#     image_container.array_mags_not_transformed = array_not_transformed_magnitudes
#
#     #   Save to file
#     utilities.save_magnitudes_ascii(
#         image_container,
#         table_not_transformed_magnitudes,
#         trans=False,
#         id_object=id_object,
#         photometry_extraction_method=photometry_extraction_method,
#     )


#   TODO: Rewrite with distributions
def determine_transformation(img_container, current_filter, filter_list,
                             tbl_transformation_coefficients,
                             fit_function=utilities.lin_func,
                             apply_uncertainty_weights=True, indent=2):
    """
        Determine the magnitude transformation factors

        Parameters
        ----------
        img_container                   : `image.container`
            Container object with image ensemble objects for each filter

        current_filter                  : `string`
            Current filter

        filter_list                     : `list` of `strings`
            List of filter

        tbl_transformation_coefficients : `astropy.table.Table`
            Astropy Table for the transformation coefficients

        fit_function                    : `function`, optional
            Fit function to use for determining the calibration factors
            Default is ``lin_func``

        apply_uncertainty_weights       : `boolean`, optional
            If True the transformation fit will be weighted by the
            uncertainties of the data points.

        indent                          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    """
    #   Get image ensembles
    ensemble_dict = img_container.ensembles

    #   Set filter key
    id_filter = filter_list.index(current_filter)

    #   Get calibration parameters
    calib_parameters = img_container.CalibParameters

    #   Get calibration data
    literature_magnitudes = calib_parameters.mags_lit

    #   Get required type for magnitude array.
    unc = checks.check_unumpy_array(literature_magnitudes)

    if unc:
        test_magnitudes_filter_1 = literature_magnitudes[0][0]
        test_magnitudes_filter_2 = literature_magnitudes[1][0]
    else:
        test_magnitudes_filter_1 = literature_magnitudes['mag'][0][0]
        test_magnitudes_filter_2 = literature_magnitudes['mag'][1][0]

    #   Check if magnitudes are not zero
    if test_magnitudes_filter_1 != 0. and test_magnitudes_filter_2 != 0.:
        image_1 = ensemble_dict[filter_list[0]].image_list[0]
        image_2 = ensemble_dict[filter_list[1]].image_list[0]
        image_key = ensemble_dict[filter_list[id_filter]].image_list[0]

        #   Extract values from a structured Numpy array
        #   TODO: The following does not work anymore: Check!
        # calib.get_observed_magnitudes_of_calibration_stars(image_1, img_container)
        # calib.get_observed_magnitudes_of_calibration_stars(image_2, img_container)
        # calib.get_observed_magnitudes_of_calibration_stars(image_key, img_container)

        #   TODO: This needs to be checked as well, since the mags_fit might not be a parameter of image_1 or image_2
        if unc:
            magnitudes_observed_filter_1 = image_1.mags_fit
            magnitudes_observed_filter_2 = image_2.mags_fit
            magnitudes_observed_filter_key = image_key.mags_fit

        else:
            magnitudes_observed_filter_1 = image_1.mags_fit['mag']
            magnitudes_observed_filter_2 = image_2.mags_fit['mag']
            magnitudes_observed_filter_key = image_key.mags_fit['mag']
            magnitudes_observed_filter_1_err = image_1.mags_fit['err']
            magnitudes_observed_filter_2_err = image_2.mags_fit['err']
            magnitudes_observed_filter_key_err = image_key.mags_fit['err']

            literature_magnitudes_errs = literature_magnitudes['err']
            literature_magnitudes = literature_magnitudes['mag']

            color_literature_err = utilities.err_prop(
                literature_magnitudes_errs[0],
                literature_magnitudes_errs[1],
            )
            color_observed_err = utilities.err_prop(
                magnitudes_observed_filter_1_err,
                magnitudes_observed_filter_2_err,
            )
            zero_err = utilities.err_prop(
                literature_magnitudes_errs[id_filter],
                magnitudes_observed_filter_key_err,
            )

        color_literature = literature_magnitudes[0] - literature_magnitudes[1]
        color_observed = (magnitudes_observed_filter_1 -
                          magnitudes_observed_filter_2)
        zero_point = (literature_magnitudes[id_filter] -
                      magnitudes_observed_filter_key)

        #   Initial guess for the parameters
        # x0    = np.array([0.0, 0.0])
        x0 = np.array([1.0, 1.0])

        ###
        #   Determine transformation coefficients
        #

        #   Plot variables
        if unc:
            color_literature_plot = unumpy.nominal_values(color_literature)
            color_literature_err_plot = unumpy.std_devs(color_literature)
            color_observed_plot = unumpy.nominal_values(color_observed)
            color_observed_err_plot = unumpy.std_devs(color_observed)
            zero_point_plot = unumpy.nominal_values(zero_point)
            zero_point_err_plot = unumpy.std_devs(zero_point)
        else:
            color_literature_plot = color_literature
            color_literature_err_plot = color_literature_err
            color_observed_plot = color_observed
            color_observed_err_plot = color_observed_err
            zero_point_plot = zero_point
            zero_point_err_plot = zero_err

        #   Color transform - Fit the data with fit_func
        #   Set sigma, using errors calculate above
        if apply_uncertainty_weights:
            sigma = np.array(color_observed_err_plot)
        else:
            sigma = 0.

        #   Fit
        a, _, b, tcolor_err = utilities.fit_curve(
            fit_function,
            color_literature_plot,
            color_observed_plot,
            x0,
            sigma,
        )

        tcolor = 1. / b

        #   Plot color transform
        terminal_output.print_to_terminal(
            f"Plot color transformation ({current_filter})",
            indent=indent,
        )
        plot.plot_transform(
            ensemble_dict[filter_list[0]].outpath.name,
            filter_list[0],
            filter_list[1],
            color_literature_plot,
            color_observed_plot,
            a,
            b,
            tcolor_err,
            fit_function,
            ensemble_dict[filter_list[0]].get_air_mass()[0],
            color_literature_err=color_literature_err_plot,
            fit_variable_err=color_observed_err_plot,
            name_object=ensemble_dict[filter_list[0]].object_name,
        )

        #  Mag transform - Fit the data with fit_func
        #   Set sigma, using errors calculate above
        if apply_uncertainty_weights:
            sigma = zero_point_err_plot
        else:
            sigma = 0.

        #   Fit
        z_dash, z_dash_err, t_mag, t_mag_err = utilities.fit_curve(
            fit_function,
            color_literature_plot,
            zero_point_plot,
            x0,
            sigma,
        )

        #   Plot mag transformation
        terminal_output.print_to_terminal(
            f"Plot magnitude transformation ({current_filter})",
            indent=indent,
        )

        plot.plot_transform(
            ensemble_dict[filter_list[0]].outpath.name,
            filter_list[0],
            filter_list[1],
            color_literature_plot,
            zero_point_plot,
            z_dash,
            t_mag,
            t_mag_err,
            fit_function,
            ensemble_dict[filter_list[0]].get_air_mass()[0],
            filter_=current_filter,
            color_literature_err=color_literature_err_plot,
            fit_variable_err=zero_point_err_plot,
            name_object=ensemble_dict[filter_list[0]].object_name,
        )

        #   Redefine variables -> shorter variables
        key_filter_l = current_filter.lower()
        f_0_l = filter_list[0].lower()
        f_1_l = filter_list[1].lower()
        f_0 = filter_list[0]
        f_1 = filter_list[1]

        #   Fill calibration table
        tbl_transformation_coefficients[f'C{key_filter_l}{f_0_l}{f_1_l}'] = [t_mag]
        tbl_transformation_coefficients[f'C{key_filter_l}{f_0_l}{f_1_l}_err'] = [t_mag_err]
        tbl_transformation_coefficients[f'z_dash{key_filter_l}{f_0_l}{f_1_l}'] = [z_dash]
        tbl_transformation_coefficients[f'z_dash{key_filter_l}{f_0_l}{f_1_l}_err'] = [z_dash_err]
        tbl_transformation_coefficients[f'T{f_0_l}{f_1_l}'] = [tcolor]
        tbl_transformation_coefficients[f'T{f_0_l}{f_1_l}_err'] = [tcolor_err]

        #   Print results
        terminal_output.print_to_terminal(
            f"Plot magnitude transformation ({current_filter})",
            indent=indent,
        )
        terminal_output.print_to_terminal(
            "###############################################",
            indent=indent,
        )
        terminal_output.print_to_terminal(
            f"Colortransform ({f_0_l}-{f_1_l} vs. {f_0}-{f_1}):",
            indent=indent,
        )
        terminal_output.print_to_terminal(
            f"T{f_0_l}{f_1_l} = {tcolor:.5f} +/- {tcolor_err:.5f}",
            indent=indent + 1,
        )
        terminal_output.print_to_terminal(
            f"{current_filter}-mag transform ({current_filter}-"
            f"{key_filter_l} vs. {f_0}-{f_1}):",
            indent=indent,
        )
        terminal_output.print_to_terminal(
            f"T{key_filter_l}_{f_0_l}{f_1_l} = {t_mag:.5f} "
            f"+/- {t_mag_err:.5f}",
            indent=indent + 1,
        )
        terminal_output.print_to_terminal(
            "###############################################",
            indent=indent,
        )


#   TODO: Rewrite with distributions
def calculate_trans(img_container, key_filter, filter_list,
                    tbl_transformation_coefficients,
                    apply_uncertainty_weights=True,
                    max_pixel_between_objects=3., own_correlation_option=1,
                    calibration_method='APASS', vizier_dict=None,
                    calibration_file=None, magnitude_range=(0., 18.5),
                    region_to_select_calibration_stars=None):
    """
        Calculate the transformation coefficients

        Parameters
        ----------
        img_container                   : `image.container`
            Container object with image ensemble objects for each filter

        key_filter                      : `string`
            Current filter

        filter_list                     : `list` of `strings`
            List of filter

        tbl_transformation_coefficients : `astropy.table.Table`
            Astropy Table for the transformation coefficients

        apply_uncertainty_weights       : `boolean`, optional
            If True the transformation fit will be weighted by the
            uncertainties of the data points.

        max_pixel_between_objects       : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option          : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        calibration_method              : `string`, optional
            Calibration method
            Default is ``APASS``.

        vizier_dict                     : `dictionary` or `None`, optional
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``None``.

        calibration_file                : `string`, optional
            Path to the calibration file
            Default is ``None``.

        magnitude_range                 : `tuple` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        region_to_select_calibration_stars  : `regions.RectanglePixelRegion`, optional
            Region in which to select calibration stars. This is a useful
            feature in instances where not the entire field of view can be
            utilized for calibration purposes.
            Default is ``None``.
    """
    #   Sanitize dictionary with Vizier catalog information
    if vizier_dict is None:
        vizier_dict = {'APASS': 'II/336/apass9'}

    ###
    #   Correlate the results from the different filter
    #
    #   TODO: Avoid circular imports
    analyze.correlate_ensembles(
        img_container,
        filter_list,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
    )

    ###
    #   Plot image with the final positions overlaid
    #   (final version)
    #
    utilities.prepare_and_plot_starmap_from_image_container(
        img_container,
        filter_list,
    )

    ###
    #   Calibrate transformation coefficients
    #
    calib.derive_calibration(
        img_container,
        filter_list,
        calibration_method=calibration_method,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        vizier_dict=vizier_dict,
        path_calibration_file=calibration_file,
        magnitude_range=magnitude_range,
        region_to_select_calibration_stars=region_to_select_calibration_stars,
    )
    terminal_output.print_to_terminal('')

    ###
    #   Determine transformation coefficients
    #   & Plot calibration plots
    #
    determine_transformation(
        img_container,
        key_filter,
        filter_list,
        tbl_transformation_coefficients,
        apply_uncertainty_weights=apply_uncertainty_weights,
    )
    terminal_output.print_to_terminal('')
