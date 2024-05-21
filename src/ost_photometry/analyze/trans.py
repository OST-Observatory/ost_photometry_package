############################################################################
#                               Libraries                                  #
############################################################################

import numpy as np

import astropy.units as u
from astropy import uncertainty as unc
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip as sigma_clipping

from . import calib, analyze, utilities, plot

from .. import checks, style, calibration_data, terminal_output


############################################################################
#                           Routines & definitions                         #
############################################################################

def prepare_transformation_variables(image_container, current_image_id, id_second_filter,
                                     id_current_filter, filter_list):
    """
        Prepare variables for magnitude transformation

        Parameters
        ----------
        image_container                 : `image.container`
            Container object with image ensemble objects for each filter

        current_image_id                : `integer`
            ID of the image

        id_second_filter                : `integer`
            ID of the second filter

        id_current_filter               : `integer`
            ID of the current filter

        filter_list                     : `list` of `string`
            List of filter names

        Returns
        -------
        best_image_second_filter        : `image.class`
            Image class with all image specific properties

        filter_image_ids_transformation : `list` of `tupel`
            See above
    """
    #   Get image ensemble
    image_ensembles = image_container.ensembles
    ensemble = image_ensembles[filter_list[id_current_filter]]

    #   Get image
    current_image = ensemble.image_list[current_image_id]

    #   Get observation time of current image and all images of the
    #   second filter
    obs_time_current_image = current_image.jd
    obs_times_images_second_filter = image_ensembles[
        filter_list[id_second_filter]
    ].get_obs_time()

    #   Find ID of the image with the nearest exposure time
    id_best_image_second_filter = np.argmin(
        np.abs(obs_times_images_second_filter - obs_time_current_image)
    )

    #   Get image corresponding to this exposure time
    best_img_second_filter = image_ensembles[
        filter_list[id_second_filter]
    ].image_list[id_best_image_second_filter]

    return best_img_second_filter


def prepare_transformation(img_container, trans_coefficients, filter_list,
                           current_filter, current_image_id, filter_image_ids,
                           derive_trans_coefficients=False):
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

        current_filter                  : `integer`
            ID of the current filter

        current_image_id                : `integer`
            ID of the image

        filter_image_ids                : `list` of `tuple` of `integer`
            Image and filter IDs

        derive_trans_coefficients       : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.
            Default is ``False``


        Returns
        -------
        type_transformation             : `string`
            Type of magnitude transformation to be performed

        second_filter_id                : `integer`
            ID of the second filter

        id_color_filter_1               : `integer`
            ID of the color filter 1. In B-V that would be B.

        id_color_filter_2               : `integer`
            ID of the color filter 2. In B-V that would be V.

        trans_coefficients_selection    : `dictionary`
            Dictionary with validated calibration parameters from Tcs.

        filter_image_ids                : `list` of `tuple` of `integer`
            Image and filter IDs
    """
    #   Get filter name
    filter_ = filter_list[current_filter]

    #   Get image
    current_img = img_container.ensembles[filter_].image_list[current_image_id]

    #   Load calibration coefficients
    if trans_coefficients is None:
        trans_coefficients = calibration_data.get_transformation_calibration_values(
            current_img.jd
        )

    #   Check if transformation is possible with the calibration
    #   coefficients.
    type_transformation = None
    second_filter_id = None
    id_color_filter_1 = None
    id_color_filter_2 = None
    trans_coefficients_selection = None
    if trans_coefficients is not None and not derive_trans_coefficients:
        trans_coefficients_selection, id_color_filter_1, id_color_filter_2 = utilities.find_filter(
            filter_list,
            trans_coefficients,
            filter_,
            current_img.instrument,
        )

        if trans_coefficients_selection is not None and 'type' in trans_coefficients_selection.keys():
            type_transformation = trans_coefficients_selection['type']

            #   Get correct filter order
            if id_color_filter_1 == current_filter:
                second_filter_id = id_color_filter_2
            else:
                second_filter_id = id_color_filter_1

        elif len(filter_list) >= 2:
            type_transformation = 'derive'

            #   Get correct filter ids: The first filter is the
            #   current filter, while the second filter is either
            #   the second in 'filter_list' or the one in 'filter_list'
            #    with the ID one below the first filter ID.
            id_color_filter_1 = current_filter

            if id_color_filter_1 == 0:
                id_color_filter_2 = 1
            else:
                id_color_filter_2 = id_color_filter_1 - 1

            second_filter_id = id_color_filter_2
        else:
            type_transformation = None

    elif len(filter_list) >= 2:
        type_transformation = 'derive'

        #   Check if calibration data is available for the
        #   filter in``filter_list`
        filter_calib = img_container.CalibParameters.column_names
        for second_filter_ in filter_list:
            if 'mag' + second_filter_ not in filter_calib:
                type_transformation = None

        if type_transformation is not None:
            #   Get correct filter ids: The first filter is the
            #   current filter, while the second filter is either
            #   the second in 'filter_list' or the one in 'filter_list'
            #    with the ID one below the first filter ID.
            id_color_filter_1 = current_filter

            if id_color_filter_1 == 0:
                id_color_filter_2 = 1
            else:
                id_color_filter_2 = id_color_filter_1 - 1

            second_filter_id = id_color_filter_2

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
        # raise RuntimeError(
        #     f"{style.Bcolors.FAIL} \nNo valid transformation type. Got "
        #     f"{type_transformation}, but allowed are only: simple, "
        #     f"air_mass, and derive  {style.Bcolors.ENDC}"
        # )

    # if type_transformation is not None:
    terminal_output.print_to_terminal(string, indent=3)

    #   Save filter and image ID configuration to allow
    #   for calculation later on
    filter_image_ids.append((current_filter, current_image_id))

    return (type_transformation, second_filter_id, id_color_filter_1,
            id_color_filter_2, trans_coefficients_selection, filter_image_ids)


def derive_transformation_onthefly(image, filter_list, id_current_filter,
                                   id_filter_1, id_filter_2, color_literature,
                                   magnitudes_literature_filter_1,
                                   magnitudes_literature_filter_2,
                                   magnitudes_observed_filter_1,
                                   magnitudes_observed_filter_2):
    """
        Determine the parameters for the color term used in the magnitude
        calibration. This corresponds to a magnitude transformation without
        considering the dependence on the air mass.

        Parameters
        ----------
        image                           : `image.class`
            Image class with all image specific properties

        filter_list                     : `list` - `string`
            List of filter

        id_current_filter               : `integer`
            ID of the current filter

        id_filter_1                     : `integer`
            ID of filter 1 for the color

        id_filter_2                     : `integer`
            ID of filter 2 for the color

        color_literature                : `astropy.uncertainty.core.QuantityDistribution`
            Literature color of the calibration stars

        magnitudes_literature_filter_1  : `astropy.uncertainty.core.QuantityDistribution`
            Magnitudes of calibration stars from the literature
            for filter 1.

        magnitudes_literature_filter_2  : `astropy.uncertainty.core.QuantityDistribution`
            Magnitudes of calibration stars from the literature
            for filter 1.

        magnitudes_observed_filter_1    : `astropy.uncertainty.core.QuantityDistribution`
            Extracted magnitudes of the calibration stars from filter 1

        magnitudes_observed_filter_2    : `astropy.uncertainty.core.QuantityDistribution`
            Extracted magnitudes of the calibration stars from filter 2



        Returns
        -------
        color_correction_filter_1       : `ufloat` or `float`
            Color correction term for filter 1.

        color_correction_filter_2       : `ufloat` or `float`
            Color correction term for filter 2.
    """
    #   Initial guess for the parameters
    # x0    = np.array([0.0, 0.0])
    x0 = np.array([1.0, 1.0])

    #   Fit function
    fit_func = utilities.lin_func

    #   Get variables
    diff_mag_1 = magnitudes_literature_filter_1 - magnitudes_observed_filter_1
    diff_mag_2 = magnitudes_literature_filter_2 - magnitudes_observed_filter_2

    #   TODO: Test median with std (should be worse!?)
    # color_literature_plot = color_literature.distribution
    # color_literature_err_plot = 0.
    # diff_mag_plot_1 = diff_mag_1.distribution
    # diff_mag_plot_2 = diff_mag_2.distribution
    color_literature_plot = color_literature.pdf_median()
    color_literature_err_plot = color_literature.pdf_std()
    diff_mag_plot_1 = diff_mag_1.pdf_median()
    diff_mag_plot_2 = diff_mag_2.pdf_median()

    #   Set
    sigma = np.array(color_literature_err_plot)

    #   Fit
    z_1, z_1_err, color_correction_filter_1, color_correction_filter_1_err = utilities.fit_curve(
        fit_func,
        color_literature_plot,
        diff_mag_plot_1,
        x0,
        sigma,
    )
    # z_1_ii, z_1_err_ii, color_correction_filter_1, color_correction_filter_1_err = utilities.fit_curve(
    #     fit_func,
    #     color_literature.pdf_median(),
    #     diff_mag_1.pdf_median(),
    #     x0,
    #     color_literature.pdf_std(),
    # )
    # print('Fit comparison: --------------------------')
    # print(z_1, z_1_err)
    # print(z_1_ii, z_1_err_ii)
    # print('------------------------------------------')
    # print(z_1)
    # print(color_correction_filter_1)
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

    #   TODO: Check that ravel and value is really necessary
    #         Check if color_literature_err_plot=0 can be improved!!!
    #   Plots magnitude difference (literature vs. measured) vs. color
    plot.plot_transform(
        image.outpath.name,
        filter_list[id_filter_1],
        filter_list[id_filter_2],
        # np.ravel(color_literature_plot.value),
        color_literature_plot.value,
        # np.ravel(diff_mag_plot_1.value),
        diff_mag_plot_1.value,
        z_1,
        color_correction_filter_1,
        color_correction_filter_1_err,
        fit_func,
        image.air_mass,
        filter_=filter_list[id_current_filter],
        color_literature_err=color_literature_err_plot.value,
        fit_variable_err=z_1_err,
        name_obj=image.objname,
    )

    if id_current_filter == id_filter_1:
        id_o = id_filter_2
    else:
        id_o = id_filter_1
    plot.plot_transform(
        image.outpath.name,
        filter_list[id_filter_1],
        filter_list[id_filter_2],
        color_literature_plot.value,
        diff_mag_plot_2.value,
        z_2,
        color_correction_filter_2,
        color_correction_filter_2_err,
        fit_func,
        image.air_mass,
        filter_=filter_list[id_o],
        color_literature_err=color_literature_err_plot.value,
        fit_variable_err=z_2_err,
        name_obj=image.objname,
    )

    color_correction_filter_1 = unc.normal(
        color_correction_filter_1 * u.mag,
        std=color_correction_filter_1_err * u.mag,
        n_samples=1000,
    )
    color_correction_filter_2 = unc.normal(
        color_correction_filter_2 * u.mag,
        std=color_correction_filter_2_err * u.mag,
        n_samples=1000,
    )

    return color_correction_filter_1, color_correction_filter_2


def transformation_core(image, calib_magnitudes_literature_filter_1,
                        calib_magnitudes_literature_filter_2,
                        calib_magnitudes_observed_filter_1,
                        calib_magnitudes_observed_filter_2, magnitudes_filter_1,
                        magnitudes_filter_2, magnitudes, tc_c, tc_color, tc_t1,
                        tc_k1, tc_t2, tc_k2, id_current_filter, id_filter_1,
                        id_filter_2, filter_list, transformation_type='derive'):
    """
        Routine that performs the actual magnitude transformation.

        Parameters
        ----------
        image                                : `image.class`
            Image class with all image specific properties

        calib_magnitudes_literature_filter_1 : `astropy.uncertainty.core.QuantityDistribution`
            Magnitudes of calibration stars from the literature
            for filter 1.

        calib_magnitudes_literature_filter_2 : `astropy.uncertainty.core.QuantityDistribution`
            Magnitudes of calibration stars from the literature
            for filter 1.

        calib_magnitudes_observed_filter_1   : `astropy.uncertainty.core.QuantityDistribution`
            Extracted magnitudes of the calibration stars from filter 1

        calib_magnitudes_observed_filter_2   : `astropy.uncertainty.core.QuantityDistribution`
            Extracted magnitudes of the calibration stars from filter 2

        magnitudes_filter_1                  : `astropy.uncertainty.core.QuantityDistribution`
            Extracted magnitudes of objects from filter 1

        magnitudes_filter_2                  : `astropy.uncertainty.core.QuantityDistribution`
            Extracted magnitudes of objects from filter 2

        magnitudes                           : `astropy.uncertainty.core.QuantityDistribution`
            Extracted magnitudes for the current filter

        tc_c                                 : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        tc_color                             : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        tc_t1                                : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        tc_k1                                : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        tc_t2                                : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        tc_k2                                : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        id_current_filter                    : `integer`
            ID of the current filter

        id_filter_1                          : `integer`
            ID of filter 1 for the color

        id_filter_2                          : `integer`
            ID of filter 2 for the color

        filter_list                          : `list` - `string`
            List of filter

        transformation_type                  : `string`, optional
            Type of magnitude transformation.
            Possibilities: simple, air_mass, or derive
            Default is ``derive``.

        Returns
        -------
                            : `astropy.uncertainty.core.QuantityDistribution`
            Calibrated magnitudes
    """
    #   Get clipped zero points
    zp_clipped = image.zp_clip

    #   Get mask from sigma clipping that needs to be applied to the data
    mask = np.where(image.zp_mask)

    #   Instrument color of the calibration objects
    color_observed = (calib_magnitudes_observed_filter_1 -
                      calib_magnitudes_observed_filter_2)
    #   Mask data according to sigma clipping
    color_observed_clipped = color_observed[mask]

    #   Literature color of the calibration objects
    color_literature = (calib_magnitudes_literature_filter_1 -
                        calib_magnitudes_literature_filter_2)
    #   Mask data according to sigma clipping
    color_literature_clipped = color_literature[mask]

    ###
    #   Apply magnitude transformation and calibration
    #
    #   Color
    color = magnitudes_filter_1 - magnitudes_filter_2
    image.color_mag = color

    #   Distinguish between versions
    if transformation_type == 'simple':
        #   Calculate calibration factor
        c = tc_c * tc_color * u.mag
    elif transformation_type == 'air_mass':
        #   Calculate calibration factor
        c_1 = tc_t1 * u.mag - tc_k1 * image.air_mass * u.mag
        c_2 = tc_t2 * u.mag - tc_k2 * image.air_mass * u.mag

    elif transformation_type == 'derive':
        #   Calculate color correction coefficients
        c_1, c_2 = derive_transformation_onthefly(
            image,
            filter_list,
            id_current_filter,
            id_filter_1,
            id_filter_2,
            color_literature_clipped,
            calib_magnitudes_literature_filter_1[mask],
            calib_magnitudes_literature_filter_2[mask],
            calib_magnitudes_observed_filter_1[mask],
            calib_magnitudes_observed_filter_2[mask],
        )

    else:
        raise Exception(
            f"{style.Bcolors.FAIL}\nType of magnitude transformation not known"
            "\n\t-> Check calibration coefficients \n\t-> Exit"
            f"{style.Bcolors.ENDC}"
        )

    if transformation_type in ['air_mass', 'derive']:
        #   Calculate C or more precise C'

        # print(c_1)
        # print(c_2)
        # print(transformation_type)
        denominator = 1. * u.mag - c_1 + c_2

        if id_current_filter == id_filter_1:
            c = c_1 / denominator
        elif id_current_filter == id_filter_2:
            c = c_2 / denominator
        else:
            raise Exception(
                f"{style.Bcolors.FAIL} \nMagnitude transformation: filter "
                "combination not valid \n\t-> This should never happen. The "
                f"current filter  ID is {id_current_filter}, while filter IDs"
                f"are {id_filter_1} and {id_filter_2} {style.Bcolors.ENDC}"
            )

    # print(c)
    #   Calculate calibrated magnitudes
    phase_1_calibrated_magnitudes = magnitudes + c * color
    # print(magnitudes)
    # print(c)
    # print(color)
    phase_1_calibrated_magnitudes = phase_1_calibrated_magnitudes.reshape(
        phase_1_calibrated_magnitudes.size,
        1,
    )
    calibrated_magnitudes_zero_point = zp_clipped - c * color_observed_clipped
    # print(zp_clipped)
    # print(c)
    # print(color_observed_clipped)
    # print(c * color_observed_clipped)
    # print(zp_clipped - c * color_observed_clipped)
    # print('-------------------')
    # print(phase_1_calibrated_magnitudes)
    # print(np.any(np.isnan(phase_1_calibrated_magnitudes)))
    # print(calibrated_magnitudes_zero_point)
    # print(np.any(np.isnan(calibrated_magnitudes_zero_point)))
    # print()
    calibrated_magnitudes = phase_1_calibrated_magnitudes + calibrated_magnitudes_zero_point
    # print(calibrated_magnitudes)
    # print(calibrated_magnitudes.dtype)
    # print(calibrated_magnitudes.shape)

    # calibrated_magnitudes = magnitudes.reshape(magnitudes.size, 1) + zp_clipped
    calibrated_magnitudes = np.median(calibrated_magnitudes, axis=1)

    #   Add calibrated photometry to table of Image object
    image.photometry['mag_cali_trans'] = calibrated_magnitudes.pdf_median()
    image.photometry['mag_cali_trans_unc'] = calibrated_magnitudes.pdf_std()

    # print(calibrated_magnitudes.shape)
    # print(color_observed.shape)
    # print(color_literature.shape)
    return calibrated_magnitudes, color_observed, color_literature


def apply_transformation(image_container, image, calib_magnitudes_literature,
                         calib_magnitudes_observed_first_filter,
                         calib_magnitudes_observed_second_filter,
                         magnitudes_first_filter, magnitudes_second_filter,
                         magnitudes, filter_id, id_filter_1, id_filter_2,
                         filter_list, transformation_coefficients,
                         plot_sigma=False, transformation_type='derive'):
    """
        Apply transformation

        Parameters
        ----------
        image_container                         : `image.container`
            Container object with image ensemble objects for each filter

        image                                   : `image.class`
            Image class with all image specific properties

        calib_magnitudes_literature             : `list` of `astropy.uncertainty.core.QuantityDistribution`
            Literature magnitudes for the calibration stars

        calib_magnitudes_observed_first_filter  : `astropy.uncertainty.core.QuantityDistribution`
            Observed magnitudes of the calibration stars in the first filter

        calib_magnitudes_observed_second_filter : `astropy.uncertainty.core.QuantityDistribution`
            Observed magnitudes of the calibration stars in second filter

        magnitudes_first_filter                 : `astropy.uncertainty.core.QuantityDistribution`
            Observed magnitudes in the first filter

        magnitudes_second_filter                : `astropy.uncertainty.core.QuantityDistribution`
            Observed magnitudes in the second filter

        magnitudes                              : `astropy.uncertainty.core.QuantityDistribution`
            Observed magnitudes for the current filter

        filter_id                               : `integer`
            ID of the current filter

        id_filter_1                             : `integer`
            ID of filter 1 for the color

        id_filter_2                             : `integer`
            ID of filter 2 for the color

        filter_list                             : `list` - `string`
            List of filter

        transformation_coefficients             : `dictionary`
            Calibration coefficients for magnitude transformation

        plot_sigma                              : `boolean`, optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        transformation_type                     : `string`, optional
            Type of magnitude transformation.
            Possibilities: simple, air_mass, or derive
            Default is ``derive``.
    """
    #   Prepare calibration parameters
    tc_t1 = None
    tc_k1 = None
    tc_t2 = None
    tc_k2 = None
    tc_c = None
    tc_color = None
    if transformation_type == 'simple':
        tc_c = transformation_coefficients['C']
        tc_color = transformation_coefficients['color']
    elif transformation_type == 'air_mass':
        tc_t1 = transformation_coefficients['T_1']
        tc_k1 = transformation_coefficients['k_1']
        tc_t2 = transformation_coefficients['T_2']
        tc_k2 = transformation_coefficients['k_2']

    #   Set values for mag_fit_1 and mag_fit_2 to allow
    #   calculation of the correct color later on
    if id_filter_1 == filter_id:
        calib_magnitudes_observed_filter_1 = calib_magnitudes_observed_first_filter
        calib_magnitudes_observed_filter_2 = calib_magnitudes_observed_second_filter

        magnitudes_filter_1 = magnitudes_first_filter
        magnitudes_filter_2 = magnitudes_second_filter
    else:
        calib_magnitudes_observed_filter_1 = calib_magnitudes_observed_second_filter
        calib_magnitudes_observed_filter_2 = calib_magnitudes_observed_first_filter

        magnitudes_filter_1 = magnitudes_second_filter
        magnitudes_filter_2 = magnitudes_first_filter

    magnitudes_calibrated, color_observed, color_literature = transformation_core(
        image,
        calib_magnitudes_literature[id_filter_1],
        calib_magnitudes_literature[id_filter_2],
        calib_magnitudes_observed_filter_1,
        calib_magnitudes_observed_filter_2,
        magnitudes_filter_1,
        magnitudes_filter_2,
        magnitudes,
        tc_c,
        tc_color,
        tc_t1,
        tc_k1,
        tc_t2,
        tc_k2,
        filter_id,
        id_filter_1,
        id_filter_2,
        filter_list,
        transformation_type=transformation_type,
    )

    #   Add calibrated magnitudes to image container
    image_container.calibrated_transformed_magnitudes[filter_list[filter_id]].append(
        magnitudes_calibrated
    )

    #   Quality control plots
    utilities.calibration_check_plots(
        filter_list[filter_id],
        image.outpath.name,
        image.objname,
        image.pd,
        filter_list,
        id_filter_1,
        id_filter_2,
        image.zp_mask,
        color_observed.pdf_median(),
        color_literature.pdf_median(),
        image_container.CalibParameters.inds,
        calib_magnitudes_literature[id_filter_1].pdf_median(),
        magnitudes_calibrated.pdf_median(),
        magnitudes.pdf_median(),
        color_observed_err=color_observed.pdf_std(),
        color_literature_err=color_literature.pdf_std(),
        literature_magnitudes_err=calib_magnitudes_literature[id_filter_1].pdf_std(),
        magnitudes_err=magnitudes_calibrated.pdf_std(),
        uncalibrated_magnitudes_err=magnitudes.pdf_std(),
        plot_sigma_switch=plot_sigma,
    )


def calibrate_simple_core(image, magnitudes):
    """
        Perform minimal calibration

        Parameters
        ----------
        image                   : `image.class`
            Image class with all image specific properties

        magnitudes              : `astropy.uncertainty.core.QuantityDistribution`
            Array with object magnitudes

        Returns
        -------
        calibrated_magnitudes   : `numpy.ndarray`
            Array with calibrated magnitudes
    """
    #   Get clipped zero points
    zp = image.zp_clip

    #   Reshape the magnitudes to allow broadcasting
    reshaped_magnitudes = magnitudes.reshape(magnitudes.size, 1)

    #   Calculate calibrated magnitudes
    calibrated_magnitudes = reshaped_magnitudes + zp

    #   If ZP is 0, calibrate with the median of all magnitudes
    #   TODO: Test this
    if np.all(zp == 0.):
        calibrated_magnitudes = reshaped_magnitudes - np.median(magnitudes)

    #   Add calibrated photometry to table of Image object
    image.photometry['mag_cali_no-trans'] = calibrated_magnitudes.pdf_median()
    image.photometry['mag_cali_no-trans_unc'] = calibrated_magnitudes.pdf_std()

    return calibrated_magnitudes


def calibrate_simple(image_container, image, not_calibrated_magnitudes,
                     filter_):
    """
        Calibrate magnitudes without magnitude transformation

        Parameters
        ----------
        image_container             : `image.container`
            Container object with image ensemble objects for each filter

        image                       : `image.class`
            Image class with all image specific properties

        not_calibrated_magnitudes   : `astropy.uncertainty.core.QuantityDistribution`
            Distribution of uncalibrated magnitudes

        filter_                     : `string`
            Current filter
    """
    #   Perform calibration
    calibrated_magnitudes = calibrate_simple_core(
        image,
        not_calibrated_magnitudes,
    )

    # #   Sigma clipping to rm outliers
    # mag_cali_sigma = sigma_clipping(
    #     calibrated_magnitudes.pdf_median(),
    #     sigma=1.5,
    #     axis=1,
    # )
    # mask = np.invert(mag_cali_sigma.mask)
    # mask = np.where(np.any(mask, axis=0))

    #   Calculate median
    # median = np.median(calibrated_magnitudes[:, mask], axis=1)
    median = np.median(calibrated_magnitudes, axis=1)

    #   Write data back to the image container
    image_container.calibrated_magnitudes[filter_].append(median)


#   TODO: combine the next two functions
def flux_calibration_ensemble(image_ensemble):
    """
        Simple calibration for flux values. Assuming the median over all
        objects in an image as a quasi ZP.

        Parameters
        ----------
        image_ensemble        : `image.ensemble`
            Image ensemble object with flux and magnitudes of all objects in
            all images within the ensemble
    """
    #   Get list with flux distributions for the individual images
    # flux_list = image_ensemble.get_flux_distribution()
    flux, flux_error = image_ensemble.get_flux_array()

    # flux_calibrated = []
    # for flux in flux_list:
    #     #   Calculate median flux in each image
    #     median_flux = np.median(flux)

    #     #   Calibrate
    #     flux_calibrated.append(flux / median_flux)
    # median_flux = np.median(flux, axis=1)
    _, median, stddev = sigma_clipped_stats(
        flux,
        axis=1,
        sigma=1.5,
        mask_value=0.0,
    )
    # print(median)
    # print(stddev)
    # print(median.shape)
    # print(flux.shape)
    flux_distribution = unc.normal(
        flux,
        std=flux_error,
        n_samples=1000,
    )
    # normalization_factor = unc.normal(
    #     median,
    #     std=stddev,
    #     n_samples=1000,
    # )[:, np.newaxis]
    normalization_factor = median[:, np.newaxis]
    flux_calibrated = flux_distribution / normalization_factor
    # print(flux_calibrated)
    # print(flux_calibrated.shape)
    # print('----------------------')

    #   Add to ensemble
    image_ensemble.quasi_calibrated_flux = flux_calibrated


#   TODO: Check if this can be improved based on distributions
def flux_normalization_ensemble(image_ensemble):
    """
        Normalize flux of each object

        Parameters
        ----------
        image_ensemble        : `image.ensemble`
            Image ensemble object with flux and magnitudes of all objects in
            all images within the ensemble
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
            n_samples=1000,
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
    #         n_samples=1000,
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
        n_samples=1000,
    )
    normalized_flux = flux_distribution / normalization_factor
    # print('+++++++++++++++++++++')
    # print(normalized_flux)

    image_ensemble.quasi_calibrated_flux_normalized = normalized_flux


def prepare_zero_point(image, id_filter_1, literature_magnitude_list,
                       observed_magnitude_filter_1, id_filter_2=None,
                       observed_magnitude_filter_2=None,
                       calculate_zero_point_statistic=True):
    """
        Prepare some values necessary for the magnitude calibration and add
        them to the image class

        Parameters
        ----------
        image                       : `image.class`
            Image class with all image specific properties

        id_filter_1                 : `integer`
            ID of the filter

        literature_magnitude_list   : list of `astropy.uncertainty.core.QuantityDistribution`
            Literature magnitudes

        observed_magnitude_filter_1 : `astropy.uncertainty.core.QuantityDistribution`
            Observed magnitudes of the objects that were used for the
            calibration from the image of filter 1

        id_filter_2                 : `integer`, optional
            ID of the `second` image/filter that is used for the magnitude
            transformation.
            Default is ``None``.

        observed_magnitude_filter_2 : `astropy.uncertainty.core.QuantityDistribution`, optional
            Observed magnitudes of the objects that were used for the
            calibration from the image of filter 2
            Default is ``None``.

        calculate_zero_point_statistic  : `boolean`, optional
            If `True` a statistic on the zero points will be calculated.
            Default is ``True``.
    """
    #   Calculated color. For two filter calculate delta color
    if id_filter_2 is not None:
        delta_color = (observed_magnitude_filter_1 +
                       observed_magnitude_filter_2 -
                       literature_magnitude_list[id_filter_1] -
                       literature_magnitude_list[id_filter_2]
                       )

    else:
        delta_color = (observed_magnitude_filter_1 -
                       literature_magnitude_list[id_filter_1])

    #   Calculate mask according to sigma clipping
    clip = sigma_clipping(delta_color.pdf_median(), sigma=1.5)
    image.zp_mask = np.invert(clip.recordmask)

    #   Calculate zero points and clip
    image.zp = (literature_magnitude_list[id_filter_1] -
                observed_magnitude_filter_1)
    image.zp_clip = image.zp[np.where(image.zp_mask)]

    #   TODO: Check if the following blocks can be improved, using
    #         distribution properties
    #   Plot zero point statistics
    plot.histogram_statistic(
        [image.zp.pdf_median()],
        [image.zp_clip.pdf_median()],
        f'Zero point ({image.filt})',
        '',
        f'histogram_zero_point_{image.filt}',
        image.outpath,
        dataset_label=[
            ['All calibration objects'],
            ['Sigma clipped calibration objects'],
        ],
        name_obj=image.objname,
    )

    #   TODO: Add random selection of calibration stars -> calculate variance
    n_calibration_objects = image.zp_clip.shape[0]
    if n_calibration_objects > 20 and calculate_zero_point_statistic:
        #   Number of samples
        n_samples = 10000

        #   Create samples using numpy's random number generator to generate
        #   an index array
        n_objects_sample = int(n_calibration_objects * 0.6)
        rng = np.random.default_rng()
        random_index = rng.integers(
            0,
            high=n_calibration_objects,
            size=(n_samples, n_objects_sample),
        )

        samples = image.zp_clip.pdf_median()[random_index]

        #   Get statistic
        # mean_samples = np.mean(sample_values, axis=1)
        median_samples = np.median(samples, axis=1)
        median_over_samples = np.median(median_samples)
        standard_deviation_over_samples = np.std(median_samples)

        terminal_output.print_to_terminal(
            f"Based on {n_samples} randomly selected sub-samples, the ",
            indent=3,
            style_name='UNDERLINE'
        )
        terminal_output.print_to_terminal(
            f"following statistic is obtained for the zero points:",
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


def apply_calibration(image_container, filter_list,
                      transformation_coefficients_dict=None,
                      derive_transformation_coefficients=False,
                      plot_sigma=False, id_object=None,
                      photometry_extraction_method='', 
                      calculate_zero_point_statistic=True, indent=1):
    """
        Apply the calibration to the magnitudes and perform a magnitude
        transformation if possible

        # Using:
        # Δ(b-v) = (b-v)obj - (b-v)cali
        # Δ(B-V) = Tbv * Δ(b-v)
        # Vobj = Δv + Tv_bv * Δ(B-V) + Vcomp or Vobj
               = v + Tv_bv*Δ(B-V) - v_cali


        Parameters
        ----------
        image_container                     : `image.container`
            Container object with image ensemble objects for each filter

        filter_list                         : `list` of `string`
            Filter names

        transformation_coefficients_dict    : `dictionary`, optional
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_transformation_coefficients  : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.
            Default is ``False``

        plot_sigma                          : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        id_object                           : `integer` or `None`, optional
            ID of the object
            Default is ``None``.

        photometry_extraction_method        : `string`, optional
            Applied extraction method. Possibilities: ePSF or APER`
            Default is ``''``.

        calculate_zero_point_statistic      : `boolean`, optional
            If `True` a statistic on the zero points will be calculated.
            Default is ``True``.

        indent                              : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    """
    terminal_output.print_to_terminal(
        "Apply calibration and perform magnitude transformation",
        indent=indent,
    )

    #   Get image ensembles
    img_ensembles = image_container.ensembles

    #   Prepare dictionary for calibrated magnitudes
    image_container.calibrated_transformed_magnitudes = {}
    image_container.calibrated_magnitudes = {}
    for filter_ in filter_list:
        image_container.calibrated_transformed_magnitudes[filter_] = []
        image_container.calibrated_magnitudes[filter_] = []

    #   Initialize list for tuple of filter and image ID for table construction
    filter_image_ids = []
    transformation_type_list = []

    #   Get calibration magnitudes
    literature_magnitudes = calib.distribution_from_calibration_table(
        image_container.CalibParameters,
        filter_list,
    )

    for current_filter_id, filter_ in enumerate(filter_list):
        #   Get image ensemble
        img_ensemble = img_ensembles[filter_]

        #   Get image list
        image_list = img_ensemble.image_list

        #   Prepare transformation
        (transformation_type, second_filter_id, id_color_filter_1,
         id_color_filter_2, trans_coefficients, filter_image_ids) = prepare_transformation(
            image_container,
            transformation_coefficients_dict,
            filter_list,
            current_filter_id,
            0,
            filter_image_ids,
            derive_trans_coefficients=derive_transformation_coefficients,
        )
        transformation_type_list.append(transformation_type)

        #   Loop over images
        for current_image_id, current_image in enumerate(image_list):
            #   Get magnitude array for first image
            magnitudes_current_image = utilities.distribution_from_table(
                current_image
            )

            #   Get extracted magnitudes of the calibration stars for the
            #   current image
            magnitudes_calibration_stars_current_image = calib.observed_magnitude_of_calibration_stars(
                current_image,
                magnitudes_current_image,
                image_container,
            )

            #   Prepare some variables and find corresponding image to
            #   current_image
            if transformation_type is not None:
                second_image = prepare_transformation_variables(
                    image_container,
                    current_image_id,
                    second_filter_id,
                    current_filter_id,
                    filter_list,
                )

                #   Get magnitude array for second image
                magnitudes_second_image = utilities.distribution_from_table(
                    second_image
                )

                #   Get extracted magnitudes of the calibration stars
                #   for the image in the second filter
                #   -> required for magnitude transformation
                magnitudes_calibration_stars_second_image = calib.observed_magnitude_of_calibration_stars(
                    second_image,
                    magnitudes_second_image,
                    image_container,
                )

            else:
                magnitudes_calibration_stars_second_image = None
                second_filter_id = None
                magnitudes_second_image = None

            #   Prepare ZP for the magnitude calibration and perform
            #   sigma clipping on the delta color or color, depending on
            #   whether magnitude transformation is possible or not.
            prepare_zero_point(
                current_image,
                current_filter_id,
                literature_magnitudes,
                magnitudes_calibration_stars_current_image,
                id_filter_2=second_filter_id,
                observed_magnitude_filter_2=magnitudes_calibration_stars_second_image,
                calculate_zero_point_statistic=calculate_zero_point_statistic,
            )

            #   Calculate transformation if possible
            if transformation_type is not None:
                apply_transformation(
                    image_container,
                    current_image,
                    literature_magnitudes,
                    magnitudes_calibration_stars_current_image,
                    magnitudes_calibration_stars_second_image,
                    magnitudes_current_image,
                    magnitudes_second_image,
                    magnitudes_current_image,
                    current_filter_id,
                    id_color_filter_1,
                    id_color_filter_2,
                    filter_list,
                    trans_coefficients,
                    plot_sigma=plot_sigma,
                    transformation_type=transformation_type,
                )

            #   Calibration without transformation
            calibrate_simple(
                image_container,
                current_image,
                magnitudes_current_image,
                filter_list[current_filter_id],
            )

        image_container.Tc_type = None

    ###
    #   Save results as ASCII files
    #
    #   Get object indices, X & Y pixel positions and wcs
    #   Assumes that the image ensembles are already correlated
    object_index = img_ensembles[filter_list[0]].image_list[0].photometry['id']
    pixel_position_x = img_ensembles[filter_list[0]].image_list[0].photometry['x_fit']
    pixel_position_y = img_ensembles[filter_list[0]].image_list[0].photometry['y_fit']
    wcs = img_ensembles[filter_list[0]].wcs

    #   If transformation is available
    if not np.any(np.array(transformation_type_list) == None):
        #   Make astropy table
        table_transformed_magnitudes = utilities.mk_magnitudes_table_distribution(
            object_index,
            pixel_position_x,
            pixel_position_y,
            image_container.calibrated_transformed_magnitudes,
            filter_list,
            filter_image_ids,
            wcs,
        )

        #   Add table to container
        image_container.table_mags_transformed = table_transformed_magnitudes

        #   Save to file
        utilities.save_magnitudes_ascii(
            image_container,
            table_transformed_magnitudes,
            trans=True,
            id_object=id_object,
            photometry_extraction_method=photometry_extraction_method,
        )
    else:
        terminal_output.print_to_terminal(
            "WARNING: No magnitude transformation possible",
            indent=indent,
            style_name='WARNING'
        )

    #   Without transformation

    #   Make astropy table
    table_mags_not_transformed = utilities.mk_magnitudes_table_distribution(
        object_index,
        pixel_position_x,
        pixel_position_y,
        image_container.calibrated_magnitudes,
        filter_list,
        filter_image_ids,
        wcs,
    )

    #   Add table to container
    image_container.table_mags_not_transformed = table_mags_not_transformed

    #   Save to file
    utilities.save_magnitudes_ascii(
        image_container,
        table_mags_not_transformed,
        trans=False,
        id_object=id_object,
        photometry_extraction_method=photometry_extraction_method,
    )


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
            name_obj=ensemble_dict[filter_list[0]].objname,
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
            name_obj=ensemble_dict[filter_list[0]].objname,
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
