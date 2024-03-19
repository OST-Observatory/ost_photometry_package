############################################################################
#                               Libraries                                  #
############################################################################

import numpy as np

from uncertainties import unumpy, ufloat

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip as sigma_clipping

from . import calib, analyze, utilities, plot

from .. import checks, style, calibration_data, terminal_output


############################################################################
#                           Routines & definitions                         #
############################################################################

def calculate_err_transformation(image, literature_magnitudes,
                                 color_magnitudes, trans_factors, id_filter_1,
                                 id_filter_2, id_current_filter,
                                 type_transformation='simple', air_mass=1.0):
    """
        Calculate errors in case of the simple magnitude transformation

        Parameters
        ----------
        image                       : `image.class`
            Image class with all image specific properties

        literature_magnitudes       : `numpy.ndarray`
            Literature magnitudes for the calibration stars

        color_magnitudes            :  `numpy.ndarray` of `numpy.float64`
            Magnitude difference -> color

        trans_factors               : `dictionary`
            Calibration data - magnitude transformation

        id_filter_1                 : `integer`
            ID of filter 1 for the color

        id_filter_2                 : `integer`
            ID of filter 2 for the color

        id_current_filter           : `integer`
            ID of the current filter

        type_transformation         : `string`
            Type of magnitude transformation

        air_mass                    : `float`
            Air mass

        Returns
        -------
        uncertainty           : `numpy.ndarray`
            Propagated uncertainty
    """
    #   Get mask from sigma clipping that needs to be applied to the data
    mask = image.ZP_mask

    #   Number of stars
    count = len(image.mags['err'])

    #   Define new array
    uncertainty = np.zeros(count, dtype=[('err', 'f8')])

    #   Zero point uncertainty
    uncertainty_zp = utilities.err_prop(
        image.mags_fit['err'],
        literature_magnitudes['err'][id_current_filter],
    )
    uncertainty_zp_clipped = np.median(uncertainty_zp[mask])

    #   Literature color errors
    uncertainty_color = utilities.err_prop(
        literature_magnitudes['err'][id_filter_1],
        literature_magnitudes['err'][id_filter_2],
    )
    uncertainty_color_clipped = np.median(uncertainty_color[mask])

    for i in range(0, count):
        #   Err: delta(color) [(inst_2 - inst_1) - (lit_2 - lit_1)]
        uncertainty_delta_color = utilities.err_prop(
            image.mags_1['err'][i],
            image.mags_2['err'][i],
            uncertainty_color_clipped,
        )

        #   Errors including magnitude transformation
        if type_transformation == 'simple':
            uncertainty_obj = utilities.err_prop(
                image.mags['err'][i],
                uncertainty_zp_clipped,
                trans_factors['color'] * color_magnitudes[i] * trans_factors['C_err'],
                trans_factors['C'] * color_magnitudes[i] * trans_factors['color_err'],
                trans_factors['C'] * trans_factors['color'] * uncertainty_delta_color,
            )
        elif type_transformation == 'air_mass':
            #   Calculate calibration factor
            c_1 = trans_factors['T_1'] - trans_factors['k_1'] * air_mass
            c_2 = trans_factors['T_2'] - trans_factors['k_2'] * air_mass

            #   c_1 & c_2 errors
            u_c_1 = utilities.err_prop(
                trans_factors['T_1_err'],
                air_mass * trans_factors['k_1_err']
            )
            u_c_2 = utilities.err_prop(
                trans_factors['T_2_err'],
                air_mass * trans_factors['k_2_err']
            )

        elif type_transformation == 'derive':
            c_1 = image.C_1
            c_2 = image.C_2
            u_c_1 = image.C_1_err
            u_c_2 = image.C_2_err
        else:
            raise Exception(
                f"{style.Bcolors.FAIL} \nType of magnitude transformation not "
                "known \n\t-> Check calibration coefficients \n\t-> Exit"
                f"{style.Bcolors.ENDC}"
            )

        if type_transformation in ['air_mass', 'derive']:
            #   Calculate the corresponding denominator
            d = 1. - c_1 + c_2

            #   Denominator error
            u_d = utilities.err_prop(u_c_1, u_c_2)

            #   C or more precise C'
            if id_current_filter == id_filter_1:
                c = c_1 / d
            elif id_current_filter == id_filter_2:
                c = c_2 / d

            #   C error
            if id_current_filter == id_filter_1:
                u_c = utilities.err_prop(u_c_1 * d, u_d * c_1 / d / d)
            elif id_current_filter == id_filter_2:
                u_c = utilities.err_prop(u_c_2 * d, u_d * c_2 / d / d)

            uncertainty_obj = utilities.err_prop(
                image.mags['err'][i],
                uncertainty_zp_clipped,
                u_c * color_magnitudes[i],
                c * uncertainty_delta_color,
            )

        uncertainty['err'][i] = np.mean(uncertainty_obj)

    return uncertainty['err']


def calculate_err(mask, calib_magnitudes_observed, calib_magnitudes_literature,
                  magnitudes):
    """
        Calculate errors in case of **no** magnitude transformation

        Parameters
        ----------
        mask                                : `numpy.ndarray` - `boolean`
            Mask of calibration stars that should be excluded

        calib_magnitudes_observed           : `numpy.ndarray`
            Extracted magnitudes for the calibration stars

        calib_magnitudes_literature         : `numpy.ndarray`
            Literature magnitudes for the calibration stars

        magnitudes                          : `numpy.ndarray`
            Magnitudes of all objects

        Returns
        -------
        u                                   : `numpy.ndarray`
            Propagated uncertainty
    """
    #   ZP errors
    u_zp = utilities.err_prop(
        calib_magnitudes_observed,
        calib_magnitudes_literature
    )
    u_zp_clip = np.median(u_zp[mask])

    #   Add up errors
    u = utilities.err_prop(
        magnitudes,
        u_zp_clip,
    )

    return u


def prepare_transformation_variables(img_container, current_image_id, id_second_filter,
                                     id_current_filter, filter_list, id_tuple_trans):
    """
        Prepare variables for magnitude transformation

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        current_image_id        : `integer`
            ID of the image

        id_second_filter        : `integer`
            ID of the second filter

        id_current_filter       : `integer`
            ID of the current filter

        filter_list             : `list` of `string`
            List of filter names

        id_tuple_trans          : `list` of `tuple` of `integer`
            Image and filter IDs

        Returns
        -------
        best_img_second_filter  : `image.class`
            Image class with all image specific properties
    """
    #   Get image ensemble
    img_ensembles = img_container.ensembles
    ensemble = img_ensembles[filter_list[id_current_filter]]

    #   Get image
    current_img = ensemble.image_list[current_image_id]

    #   Get observation time of current image and all images of the
    #   second filter
    obs_time_current_img = current_img.jd
    obs_times_images_second_filter = img_ensembles[
        filter_list[id_second_filter]
    ].get_obs_time()

    #   Find ID of the image with the nearest exposure time
    id_best_img_second_filter = np.argmin(
        np.abs(obs_times_images_second_filter - obs_time_current_img)
    )

    #   Save filter and image ID configuration to allow
    #   for a better color calculation later on
    id_tuple_trans.append((
        id_current_filter,
        current_image_id,
        id_second_filter,
        id_best_img_second_filter
    ))

    #   Get image corresponding to this exposure time
    best_img_second_filter = img_ensembles[
        filter_list[id_second_filter]
    ].image_list[id_best_img_second_filter]

    return best_img_second_filter


def prepare_transformation(img_container, trans_coefficients, filter_list, current_filter,
                           current_image_id, id_tuple_no_trans,
                           derive_trans_coefficients=False):
    """
        Prepare magnitude transformation: find filter combination,
        get calibration parameters, prepare variables, ...

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        trans_coefficients      : `dictionary` or ``None``
            Calibration coefficients for magnitude transformation

        filter_list             : `list` of `string`
            List of filter names

        current_filter          : `integer`
            ID of the current filter

        current_image_id        : `integer`
            ID of the image

        id_tuple_no_trans       : `list` of `tuple` of `integer`
            Image and filter IDs

        derive_trans_coefficients    : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.
            Default is ``False``


        Returns
        -------
        type_transformation     : `string`
            Type of magnitude transformation to be performed

        second_filter_id           : `integer`
            ID of the second filter

        id_color_filter_1             : `integer`
            ID of the color filter 1. In B-V that would be B.

        id_color_filter_2             : `integer`
            ID of the color filter 2. In B-V that would be V.

        trans_coefficients      : `dictionary`
            Dictionary with validated calibration parameters from Tcs.
    """
    #   Get filter name
    filter_ = filter_list[current_filter]

    #   Get image
    current_img = img_container.ensembles[filter_].image_list[current_image_id]

    #   Load calibration coefficients
    if trans_coefficients is None:
        trans_coefficients = calibration_data.get_transformation_calibration_values(current_img.jd)

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
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo valid transformation type. Got "
            f"{type_transformation}, but allowed are only: simple, "
            f"air_mass, and derive  {style.Bcolors.ENDC}"
        )

    if type_transformation is not None:
        terminal_output.print_to_terminal(string, indent=3)

    #   Save filter and image ID configuration to allow
    #   for a better color calculation later on
    id_tuple_no_trans.append((current_filter, current_image_id))

    return type_transformation, second_filter_id, id_color_filter_1, \
        id_color_filter_2, trans_coefficients_selection


def derive_transformation_onthefly(image, filter_list, id_current_filter, id_filter_1,
                                   id_filter_2, color_literature,
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

        color_literature                : `numpy.ndarray` or `unumpy.uarray`
            Literature color of the calibration stars

        magnitudes_literature_filter_1  : `numpy.ndarray` or `unumpy.uarray`
            Magnitudes of calibration stars from the literature
            for filter 1.

        magnitudes_literature_filter_2  : `numpy.ndarray` or `unumpy.uarray`
            Magnitudes of calibration stars from the literature
            for filter 1.

        magnitudes_observed_filter_1    : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of the calibration stars from filter 1

        magnitudes_observed_filter_2    : `numpy.ndarray` or `unumpy.uarray`
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

    #   Get required type for magnitude array.
    unc = checks.check_unumpy_array(color_literature)

    #   Get variables
    diff_mag_1 = magnitudes_literature_filter_1 - magnitudes_observed_filter_1
    diff_mag_2 = magnitudes_literature_filter_2 - magnitudes_observed_filter_2
    if unc:
        color_literature_plot = unumpy.nominal_values(color_literature)
        color_literature_err_plot = unumpy.std_devs(color_literature)
        diff_mag_plot_1 = unumpy.nominal_values(diff_mag_1)
        diff_mag_plot_2 = unumpy.nominal_values(diff_mag_2)
    else:
        color_literature_plot = color_literature
        color_literature_err_plot = 0.
        diff_mag_plot_1 = diff_mag_1
        diff_mag_plot_2 = diff_mag_2

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
        filter_list[id_filter_1],
        filter_list[id_filter_2],
        color_literature_plot,
        diff_mag_plot_1,
        z_1,
        color_correction_filter_1,
        color_correction_filter_1_err,
        fit_func,
        image.air_mass,
        filter_=filter_list[id_current_filter],
        color_literature_err=color_literature_err_plot,
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
        color_literature_plot,
        diff_mag_plot_2,
        z_2,
        color_correction_filter_2,
        color_correction_filter_2_err,
        fit_func,
        image.air_mass,
        filter_=filter_list[id_o],
        color_literature_err=color_literature_err_plot,
        fit_variable_err=z_2_err,
        name_obj=image.objname,
    )

    #   Return ufloat of normal float
    if unc:
        return ufloat(color_correction_filter_1, color_correction_filter_1_err), \
            ufloat(color_correction_filter_2, color_correction_filter_2_err)
    else:
        #   TODO: Check if this can be removed
        image.C_1 = color_correction_filter_1  # Dirty hack
        image.C_2 = color_correction_filter_2  # Dirty hack
        image.C_1_err = color_correction_filter_1_err  # Dirty hack
        image.C_2_err = color_correction_filter_2_err  # Dirty hack
        return color_correction_filter_1, color_correction_filter_2


def apply_transformation(*args, **kwargs):
    """
        Apply magnitude transformation and return calibrated magnitude array

        Distinguishes between different input array types.
        Possibilities: unumpy.uarray & numpy structured ndarray
    """
    #   Get type of the magnitude arrays
    unc = getattr(args[0], 'unc', True)

    if unc:
        apply_transformation_unumpy(*args, **kwargs)
    else:
        apply_transformation_structured(*args, **kwargs)


#   TODO: Convert arrays to index lists/arrays?
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

        calib_magnitudes_literature_filter_1 : `numpy.ndarray` or `unumpy.uarray`
            Magnitudes of calibration stars from the literature
            for filter 1.

        calib_magnitudes_literature_filter_2 : `numpy.ndarray` or `unumpy.uarray`
            Magnitudes of calibration stars from the literature
            for filter 1.

        calib_magnitudes_observed_filter_1    : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of the calibration stars from filter 1

        calib_magnitudes_observed_filter_2    : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of the calibration stars from filter 2

        magnitudes_filter_1                  : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of objects from filter 1

        magnitudes_filter_2                  : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of objects from filter 2

        magnitudes                           : `numpy.ndarray` or `unumpy.uarray`
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
                            : `numpy.ndarray` or `unumpy.uarray`
            Calibrated magnitudes
    """
    #   Get clipped zero points
    zp_clipped = image.ZP_clip

    #   Get mask from sigma clipping that needs to be applied to the data
    mask = image.ZP_mask

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
        c = tc_c * tc_color
    elif transformation_type == 'air_mass':
        #   Calculate calibration factor
        c_1 = tc_t1 - tc_k1 * image.air_mass
        c_2 = tc_t2 - tc_k2 * image.air_mass

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

        denominator = 1. - c_1 + c_2

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

    #   Calculate calibrated magnitudes
    calibrated_magnitudes = (
            magnitudes + c * color
            + np.median((zp_clipped - c * color_observed_clipped)))

    #   Add calibrated photometry to table of Image object
    image.photometry['mag_cali_trans'] = calibrated_magnitudes

    return calibrated_magnitudes, color_observed, color_literature


def apply_transformation_structured(img_container, image, calib_magnitudes_literature,
                                    id_current_filter, id_current_image, id_filter_1,
                                    id_filter_2, filter_list, transformation_coefficients,
                                    plot_sigma=False, transformation_type='derive'):
    """
        Apply transformation

        Parameters
        ----------
        img_container               : `image.container`
            Container object with image ensemble objects for each filter

        image                       : `image.class`
            Image class with all image specific properties

        calib_magnitudes_literature : `numpy.ndarray`
            Numpy structured array with literature magnitudes for the
            calibration stars

        id_current_filter           : `integer`
            ID of the current filter

        id_current_image            : `integer`
            ID of the current image

        id_filter_1                 : `integer`
            ID of filter 1 for the color

        id_filter_2                 : `integer`
            ID of filter 2 for the color

        filter_list                 : `list` - `string`
            List of filter

        transformation_coefficients : `dictionary`
            Calibration coefficients for magnitude transformation

        plot_sigma                  : `boolean`, optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        transformation_type         : `string`, optional
            Type of magnitude transformation.
            Possibilities: simple, air_mass, or derive
            Default is ``derive``.
    """
    #   Get necessary magnitudes arrays
    magnitudes_literature = calib_magnitudes_literature['mag']
    calib_magnitudes_observed_filter_1 = image.mag_fit_1['mag']
    calib_magnitudes_observed_filter_2 = image.mag_fit_2['mag']
    magnitudes_filter_1 = image.mags_1['mag']
    magnitudes_filter_2 = image.mags_2['mag']
    magnitudes = image.mags['mag']

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

    #   Apply magnitude transformation
    magnitudes_calibrated, color_observed, color_literature = transformation_core(
        image,
        magnitudes_literature[id_filter_1],
        magnitudes_literature[id_filter_2],
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
        id_current_filter,
        id_filter_1,
        id_filter_2,
        filter_list,
        transformation_type=transformation_type,
    )

    img_container.cali['mag'][id_current_filter][id_current_image] = magnitudes_calibrated

    #   Calculate uncertainties
    img_container.cali['err'][id_current_filter][id_current_image] = calculate_err_transformation(
        image,
        calib_magnitudes_literature,
        image.color_mag,
        transformation_coefficients,
        id_filter_1,
        id_filter_2,
        id_current_filter,
        type_transformation=transformation_type,
        air_mass=image.air_mass,
    )

    #   Quality control plots
    utilities.calibration_check_plots(
        filter_list[id_current_filter],
        image.outpath.name,
        image.objname,
        image.pd,
        filter_list,
        id_filter_1,
        id_filter_2,
        image.ZP_mask,
        color_observed,
        color_literature,
        img_container.CalibParameters.inds,
        magnitudes_literature[id_filter_1],
        magnitudes_calibrated,
        magnitudes,
        color_observed_err=utilities.err_prop(
            image.mag_fit_1['err'],
            image.mag_fit_2['err'],
        ),
        color_literature_err=utilities.err_prop(
            calib_magnitudes_literature['err'][id_filter_1],
            calib_magnitudes_literature['err'][id_filter_2],
        ),
        literature_magnitudes_err=calib_magnitudes_literature['err'][id_filter_1],
        magnitudes_err=img_container.cali['err'][id_current_filter][id_current_image],
        uncalibrated_magnitudes_err=image.mags['err'],
        plot_sigma_switch=plot_sigma,
    )


def apply_transformation_unumpy(img_container, image, calib_magnitudes_literature,
                                id_current_filter, id_current_image, id_filter_1,
                                id_filter_2, filter_list, transformation_coefficients,
                                plot_sigma=False, transformation_type='derive'):
    """
        Apply transformation

        Parameters
        ----------
        img_container               : `image.container`
            Container object with image ensemble objects for each filter

        image                       : `image.class`
            Image class with all image specific properties

        calib_magnitudes_literature : `numpy.ndarray`
            Unumpy array with literature magnitudes for the
            calibration stars

        id_current_filter           : `integer`
            ID of the current filter

        id_current_image            : `integer`
            ID of the current image

        id_filter_1                 : `integer`
            ID of filter 1 for the color

        id_filter_2                 : `integer`
            ID of filter 2 for the color

        filter_list                 : `list` - `string`
            List of filter

        transformation_coefficients : `dictionary`
            Calibration coefficients for magnitude transformation

        plot_sigma                  : `boolean`, optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        transformation_type         : `string`, optional
            Type of magnitude transformation.
            Possibilities: simple, air_mass, or derive
            Default is ``derive``.
    """
    #   Get necessary magnitudes arrays
    magnitudes_literature = calib_magnitudes_literature
    calib_magnitudes_observed_filter_1 = image.mag_fit_1
    calib_magnitudes_observed_filter_2 = image.mag_fit_2
    magnitudes_filter_1 = image.mags_1
    magnitudes_filter_2 = image.mags_2
    magnitudes = image.mags

    #   Prepare calibration parameters
    tc_t1 = None
    tc_k1 = None
    tc_t2 = None
    tc_k2 = None
    tc_c = None
    tc_color = None
    if transformation_type == 'simple':
        tc_c = ufloat(transformation_coefficients['C'], transformation_coefficients['C_err'])
        tc_color = ufloat(transformation_coefficients['color'], transformation_coefficients['color_err'])
    elif transformation_type == 'air_mass':
        tc_t1 = ufloat(transformation_coefficients['T_1'], transformation_coefficients['T_1_err'])
        tc_k1 = ufloat(transformation_coefficients['k_1'], transformation_coefficients['k_1_err'])
        tc_t2 = ufloat(transformation_coefficients['T_2'], transformation_coefficients['T_2_err'])
        tc_k2 = ufloat(transformation_coefficients['k_2'], transformation_coefficients['k_2_err'])

    #   Apply magnitude transformation
    mag_cali, color_observed, color_literature = transformation_core(
        image,
        magnitudes_literature[id_filter_1],
        magnitudes_literature[id_filter_2],
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
        id_current_filter,
        id_filter_1,
        id_filter_2,
        filter_list,
        transformation_type=transformation_type,
    )

    img_container.cali[id_current_filter][id_current_image] = mag_cali

    #   Quality control plots
    utilities.calibration_check_plots(
        filter_list[id_current_filter],
        image.outpath.name,
        image.objname,
        image.pd,
        filter_list,
        id_filter_1,
        id_filter_2,
        image.ZP_mask,
        unumpy.nominal_values(color_observed),
        unumpy.nominal_values(color_literature),
        img_container.CalibParameters.inds,
        unumpy.nominal_values(magnitudes_literature[id_filter_1]),
        unumpy.nominal_values(mag_cali),
        unumpy.nominal_values(magnitudes),
        color_observed_err=unumpy.std_devs(color_observed),
        color_literature_err=unumpy.std_devs(color_literature),
        literature_magnitudes_err=unumpy.std_devs(magnitudes_literature[id_filter_1]),
        magnitudes_err=unumpy.std_devs(mag_cali),
        uncalibrated_magnitudes_err=unumpy.std_devs(magnitudes),
        plot_sigma_switch=plot_sigma,
    )


def calibrate_simple(*args, **kwargs):
    """
        Apply minimal calibration: No magnitude transformation & no other
                                   kind of color corrections.
    """
    #   Get type of the magnitude arrays
    unc = getattr(args[0], 'unc', True)

    if unc:
        calibrate_unumpy(*args, **kwargs)
    else:
        calibrate_structured(*args, **kwargs)


def calibrate_simple_core(image, magnitudes_arr):
    """
        Perform minimal calibration

        Parameters
        ----------
        image                   : `image.class`
            Image class with all image specific properties

        magnitudes_arr          : `numpy.ndarray`
            Array with object magnitudes

        Returns
        -------
        calibrated_magnitudes  : `numpy.ndarray`
            Array with calibrated magnitudes
    """
    #   Get clipped zero points
    zp = image.ZP_clip

    #   Reshape the magnitude array to allow broadcasting
    reshaped_magnitudes_arr = magnitudes_arr.reshape(magnitudes_arr.size, 1)

    #   Calculate calibrated magnitudes
    calibrated_magnitudes = reshaped_magnitudes_arr + zp

    #   If ZP is 0, calibrate with the median of all magnitudes
    if np.all(zp == 0.):
        calibrated_magnitudes = reshaped_magnitudes_arr - np.median(magnitudes_arr)

    #   Add calibrated photometry to table of Image object
    image.photometry['mag_cali_no-trans'] = calibrated_magnitudes

    return calibrated_magnitudes


def calibrate_structured(img_container, image, literature_magnitudes,
                         id_filter, id_image):
    """
        Calibrate magnitudes without magnitude transformation

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        image                   : `image.class`
            Image class with all image specific properties

        literature_magnitudes   : `numpy.ndarray`
            Numpy structured array with literature magnitudes for the
            calibration stars

        id_filter               : `integer`
            ID of the current filter

        id_image:               : `integer`
            ID of the current image
    """
    #   Get mask from sigma clipping
    mask = image.ZP_mask

    #   Get magnitudes array
    magnitudes_cali_structured = img_container.noT

    #   Get extracted magnitudes for all objects
    not_calibrated_magnitudes = image.mags['mag']

    #   Perform calibration
    mag_cali = calibrate_simple_core(image, not_calibrated_magnitudes)

    #   Sigma clipping to rm outliers and calculate median, ...
    _, median, stddev = sigma_clipped_stats(mag_cali, axis=1, sigma=1.5)
    magnitudes_cali_structured['mag'][id_filter][id_image] = median
    magnitudes_cali_structured['std'][id_filter][id_image] = stddev

    magnitudes_cali_structured['err'][id_filter][id_image] = calculate_err(
        mask,
        image.mags_fit['err'],
        literature_magnitudes['err'][id_filter],
        image.mags['err'],
    )

    #   Write data back to the image container
    img_container.noT = magnitudes_cali_structured


def calibrate_unumpy(img_container, image, literature_magnitudes,
                     id_filter, id_image):
    """
        Calibrate magnitudes without magnitude transformation

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        image                   : `image.class`
            Image class with all image specific properties

        literature_magnitudes   : `numpy.ndarray`
            Numpy structured array with literature magnitudes for the
            calibration stars

        id_filter               : `integer`
            ID of the current filter

        id_image:               : `integer`
            ID of the current image
    """
    #   Get extracted magnitudes for all objects
    not_calibrated_magnitudes = image.mags

    #   Perform calibration
    calibrated_magnitudes = calibrate_simple_core(
        image,
        not_calibrated_magnitudes,
    )

    #   Sigma clipping to rm outliers
    mag_cali_sigma = sigma_clipping(
        unumpy.nominal_values(calibrated_magnitudes),
        sigma=1.5,
        axis=1,
    )
    mask = np.invert(mag_cali_sigma.mask)
    mask = np.any(mask, axis=0)

    #   Calculate median
    median = np.median(calibrated_magnitudes[:, mask], axis=1)

    #   Write data back to the image container
    img_container.noT[id_filter][id_image] = median


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
    #   Get flux
    flux = image_ensemble.get_flux_uarray()

    #   Calculate median flux in each image
    median_flux = np.median(flux, axis=1)

    #   Calibrate
    flux_cali = flux / median_flux[:, np.newaxis]

    #   Add to ensemble
    image_ensemble.uflux_cali = flux_cali


def flux_normalization_ensemble(image_ensemble):
    """
        Normalize flux

        Parameters
        ----------
        image_ensemble        : `image.ensemble`
            Image ensemble object with flux and magnitudes of all objects in
            all images within the ensemble
    """
    #   Get flux
    try:
        flux = image_ensemble.uflux_cali
    except:
        flux = image_ensemble.get_flux_uarray()

    flux_values = unumpy.nominal_values(flux)

    #   Calculated sigma clipped magnitudes
    _, median, stddev = sigma_clipped_stats(
        flux_values,
        axis=0,
        sigma=1.5,
        mask_value=0.0,
    )

    #   Add axis so that broadcasting to original array is possible
    median_reshape = median[np.newaxis, :]
    std_dev_reshape = stddev[np.newaxis, :]

    #   Normalized magnitudes
    normalization_factor = unumpy.uarray(median_reshape, std_dev_reshape)
    image_ensemble.uflux_norm = flux / normalization_factor


#   TODO: Check if image is here the correct property
def prepare_zero_point(img_container, image, id_filter_1,
                       magnitudes_literature, magnitudes_observed_filter_1,
                       id_filter_2=None, magnitudes_observed_filter_2=None):
    """
        Prepare some values necessary for the magnitude calibration and add
        them to the image class

        Parameters
        ----------
        img_container                   : `image.container`
            Container object with image ensemble objects for each filter

        image                           : `image.class`
            Image class with all image specific properties

        id_filter_1                     : `integer`
            ID of the filter

        magnitudes_literature           : `numpy.ndarray` or `unumpy.uarray`
            Literature magnitudes

        magnitudes_observed_filter_1    : `numpy.ndarray` or `unumpy.uarray`
            Observed magnitudes of the objects that were used for the
            calibration from the image of filter 1

        id_filter_2                     : `integer`, optional
            ID of the `second` image/filter that is used for the magnitude
            transformation.
            Default is ``None``.

        magnitudes_observed_filter_2    : `numpy.ndarray` or `unumpy.uarray`, optional
            Observed magnitudes of the objects that were used for the
            calibration from the image of filter 2
            Default is ``None``.
    """
    #   Get type of the magnitudes array used
    #   Possibilities: structured numpy array & unumpy uarray
    unc = getattr(img_container, 'unc', True)

    #   Set array with literature magnitudes for the calibration stars
    if not unc:
        magnitudes_literature = magnitudes_literature['mag']

        #   Get extracted magnitudes
        magnitudes_observed_filter_1 = magnitudes_observed_filter_1['mag']

        if id_filter_2 is not None:
            magnitudes_observed_filter_2 = magnitudes_observed_filter_2['mag']

    #   Calculated color. For two filter calculate delta color
    if id_filter_2 is not None:
        delta_color = (magnitudes_observed_filter_1 +
                       magnitudes_observed_filter_2 -
                       magnitudes_literature[id_filter_1] -
                       magnitudes_literature[id_filter_2]
                       )

    else:
        delta_color = (magnitudes_observed_filter_1 -
                       magnitudes_literature[id_filter_1])

    #   Calculate mask according to sigma clipping
    if unc:
        clip_values = unumpy.nominal_values(delta_color)
    else:
        clip_values = delta_color
    clip = sigma_clipping(clip_values, sigma=1.5)
    image.ZP_mask = np.invert(clip.recordmask)

    #   Calculate zero points and clip
    image.ZP = (magnitudes_literature[id_filter_1] -
                magnitudes_observed_filter_1)
    image.ZP_clip = image.ZP[image.ZP_mask]

    #   Plot zero point statistics
    plot.histogram_statistic(
        [image.ZP],
        [image.ZP_clip],
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
    n_calibration_objects = image.ZP_clip.shape[0]
    if n_calibration_objects > 20:
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

        samples = image.ZP_clip[random_index]

        #   Get nominal values if uncertainty package is used
        if unc:
            sample_values = unumpy.nominal_values(samples)
        else:
            sample_values = samples

        #   Get statistic
        # mean_samples = np.mean(sample_values, axis=1)
        median_samples = np.median(sample_values, axis=1)
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


def apply_calib(img_container, filter_list,
                transformation_coefficients_dict=None,
                derive_transformation_coefficients=False, plot_sigma=False,
                id_object=None, photometry_extraction_method='',
                indent=1):
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
        img_container                       : `image.container`
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

        indent                              : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    """
    terminal_output.print_to_terminal(
        "Apply calibration and perform magnitude transformation",
        indent=indent,
    )

    #   Get image ensembles
    img_ensembles = img_container.ensembles

    #   Get object indices, X & Y pixel positions and wcs
    #   Assumes that the image ensembles are already correlated
    object_index = img_ensembles[filter_list[0]].image_list[0].photometry['id']
    pixel_position_x = img_ensembles[filter_list[0]].image_list[0].photometry['x_fit']
    pixel_position_y = img_ensembles[filter_list[0]].image_list[0].photometry['y_fit']
    wcs = img_ensembles[filter_list[0]].wcs

    #   Number of filter
    n_filter = len(filter_list)

    #   Get number of objects
    n_objects = len(object_index)

    #   Prepare arrays
    utilities.prepare_arrays(img_container, n_filter, n_objects)

    #   Initialize bool and image ID for transformation
    tuple_transformation_ids = []
    tuple_no_transformation_ids = []

    #   Get calibration magnitudes
    literature_magnitudes = calib.magnitude_array_from_calibration_table(
        img_container,
        filter_list,
    )

    for current_filter, band in enumerate(filter_list):
        #   Get image ensemble
        img_ensemble = img_ensembles[band]

        #   Get image list
        image_list = img_ensemble.image_list

        #   Prepare transformation
        (transformation_type, second_filter_id, id_color_filter_1,
         id_color_filter_2, trans_coefficients) = prepare_transformation(
            img_container,
            transformation_coefficients_dict,
            filter_list,
            current_filter,
            0,
            tuple_no_transformation_ids,
            derive_trans_coefficients=derive_transformation_coefficients,
        )

        #   Loop over images
        for current_image_id, current_image in enumerate(image_list):
            #   Get magnitude array for image 1
            magnitudes_current_image = utilities.magnitude_array_from_table(
                img_container,
                current_image,
            )

            #   TODO: Remove later?
            #   Add magnitudes to image
            current_image.mags = magnitudes_current_image

            #   Get extracted magnitudes of the calibration stars for the
            #   current image
            magnitudes_calibration_stars_current_image = calib.get_observed_magnitudes_of_calibration_stars(
                current_image,
                magnitudes_current_image,
                img_container,
            )
            #   TODO: Remove later?
            current_image.mags_fit = magnitudes_calibration_stars_current_image

            #   Prepare some variables and find corresponding image to
            #   current_image
            if transformation_type is not None:
                second_image = prepare_transformation_variables(
                    img_container,
                    current_image_id,
                    second_filter_id,
                    current_filter,
                    filter_list,
                    tuple_transformation_ids,
                )

                #   Get magnitude array for image o
                magnitudes_second_image = utilities.magnitude_array_from_table(
                    img_container,
                    second_image,
                )

                #   Get extracted magnitudes of the calibration stars
                #   for the image in the second filter
                #   -> required for magnitude transformation
                magnitudes_calibration_stars_second_image = calib.get_observed_magnitudes_of_calibration_stars(
                    second_image,
                    magnitudes_second_image,
                    img_container,
                )

                #   Set values for mag_fit_1 and mag_fit_2 to allow
                #   calculation of the correct color later on
                #   TODO: Remove later?
                if id_color_filter_1 == current_filter:
                    current_image.mag_fit_1 = magnitudes_calibration_stars_current_image
                    current_image.mag_fit_2 = magnitudes_calibration_stars_second_image

                    current_image.mags_1 = magnitudes_current_image
                    current_image.mags_2 = magnitudes_second_image
                else:
                    current_image.mag_fit_1 = magnitudes_calibration_stars_second_image
                    current_image.mag_fit_2 = magnitudes_calibration_stars_current_image

                    current_image.mags_1 = magnitudes_second_image
                    current_image.mags_2 = magnitudes_current_image

            else:
                magnitudes_calibration_stars_second_image = None

            #   Prepare ZP for the magnitude calibration and perform
            #   sigma clipping on the delta color or color, depending on
            #   whether magnitude transformation is possible or not.
            prepare_zero_point(
                img_container,
                current_image,
                current_filter,
                literature_magnitudes,
                magnitudes_calibration_stars_current_image,
                id_filter_2=second_filter_id,
                magnitudes_observed_filter_2=magnitudes_calibration_stars_second_image,
            )

            ###
            #   Calculate transformation if possible
            #
            if transformation_type is not None:
                apply_transformation(
                    img_container,
                    current_image,
                    literature_magnitudes,
                    current_filter,
                    current_image_id,
                    id_color_filter_1,
                    id_color_filter_2,
                    filter_list,
                    trans_coefficients,
                    plot_sigma=plot_sigma,
                    transformation_type=transformation_type,
                )

            ###
            #   Calibration without transformation
            #
            calibrate_simple(
                img_container,
                current_image,
                literature_magnitudes,
                current_filter,
                current_image_id,
            )

        img_container.Tc_type = None

    ###
    #   Save results as ASCII files
    #
    calibrated_magnitudes = img_container.cali
    if not checks.check_unumpy_array(calibrated_magnitudes):
        calibrated_magnitudes = calibrated_magnitudes['mag']

    #   If transformation is available
    if np.any(calibrated_magnitudes != 0.):
        #   Make astropy table
        table_transformed_magnitudes = utilities.mk_magnitudes_table(
            object_index,
            pixel_position_x,
            pixel_position_y,
            img_container.cali,
            filter_list,
            tuple_transformation_ids,
            wcs,
        )

        #   Add table to container
        img_container.table_mags_transformed = table_transformed_magnitudes

        #   Save to file
        utilities.save_magnitudes_ascii(
            img_container,
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
    table_mags_not_transformed = utilities.mk_magnitudes_table(
        object_index,
        pixel_position_x,
        pixel_position_y,
        img_container.noT,
        filter_list,
        tuple_no_transformation_ids,
        wcs,
    )

    #   Add table to container
    img_container.table_mags_not_transformed = table_mags_not_transformed

    #   Save to file
    utilities.save_magnitudes_ascii(
        img_container,
        table_mags_not_transformed,
        trans=False,
        id_object=id_object,
        photometry_extraction_method=photometry_extraction_method,
    )


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
