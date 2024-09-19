############################################################################
#                               Libraries                                  #
############################################################################
import numpy as np

import astropy.units as u
from astropy import uncertainty as unc
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.table import Table

from regions import RectanglePixelRegion

from . import calibration_data, correlate, utilities, plots

import typing
if typing.TYPE_CHECKING:
    from . import analyze
    from .. import utilities

from .. import style, calibration_parameters, terminal_output


############################################################################
#                           Routines & definitions                         #
############################################################################


def find_best_comparison_image_second_filter(
        image_series: dict[str, 'analyze.ImageSeries'], current_image: 'utilities.Image',
        id_second_filter: int, filter_list: list[str]
        ) -> 'analyze.Image':
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
    ].get_observation_time()

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
        observation: 'analyze.Observation',
        trans_coefficients: dict[str, (float | str)] | None, filter_list: list[str],
        current_filter_id: int, derive_transformation_coefficients: bool
        ) -> tuple[str | None, int | None, None | dict[str, [float | str]]]:
    """
    Prepare magnitude transformation: find filter combination,
    get calibration parameters, prepare variables, ...

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    trans_coefficients
        Calibration coefficients for magnitude transformation

    filter_list
        List of filter names

    current_filter_id
        ID of the current filter

    derive_transformation_coefficients
        If True the magnitude transformation coefficients will be
        calculated from the current data even if calibration coefficients
        are available in the database.

    Returns
    -------
    type_transformation
        Type of magnitude transformation to be performed

    second_filter_id
        ID of the second filter

    trans_coefficients_selection
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
            second_filter_id: int = 1
        else:
            second_filter_id: int = 0
    else:
        #   This should currently not happen
        terminal_output.print_to_terminal(
            f"Magnitude transformation currently only possible with two filter "
            f"but {len(filter_list)} filter given in filter_list",
            style_name='ERROR',
        )
        raise RuntimeError

    if derive_transformation_coefficients:
        type_transformation: str = 'derive'
        trans_coefficients_selection = None
    else:
        #   Load coefficients coefficients
        if trans_coefficients is None:
            trans_coefficients = calibration_parameters.get_transformation_calibration_values(
                observation.image_series_dict[filter_].start_jd
            )
            trans_coefficients_selection = utilities.find_transformation_coefficients(
                filter_list,
                trans_coefficients,
                filter_,
                observation.image_series_dict[filter_].instrument,
            )
            #   If no valid transformation coefficients can be loaded switch to
            #   automatic determination of these coefficients
            if trans_coefficients_selection is None:
                type_transformation: str = 'derive'
                terminal_output.print_to_terminal(
                    'Transformation coefficients cannot be loaded, switching'
                    ' to automatic determination of these coefficients.',
                    indent=3,
                    style_name='WARNING',
                )
            else:
                type_transformation: str = trans_coefficients_selection['type']
        else:
            trans_coefficients_selection = trans_coefficients
            type_transformation: str = trans_coefficients_selection['type']

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

    terminal_output.print_to_terminal(string, indent=3, style_name=message_type)

    return type_transformation, second_filter_id, trans_coefficients_selection


def derive_transformation_onthefly(
        image: 'analyze.Image', filter_list: list[str],
        id_current_filter: int,
        magnitudes_literature_filter_1: unc.core.NdarrayDistribution,
        magnitudes_literature_filter_2: unc.core.NdarrayDistribution,
        magnitudes_observed_filter_1: unc.core.NdarrayDistribution,
        magnitudes_observed_filter_2: unc.core.NdarrayDistribution,
        file_type_plots: str = 'pdf',
        ) -> tuple[unc.core.NdarrayDistribution, unc.core.NdarrayDistribution]:
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

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

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

    #   Literature color of the calibration objects
    color_literature = (magnitudes_literature_filter_1 -
                        magnitudes_literature_filter_2)

    #   Perform sigma clipping on the difference between observed color
    #   and literature color values to remove outliers
    zp_sum = (magnitudes_observed_filter_2 - magnitudes_literature_filter_2 +
              magnitudes_observed_filter_1 - magnitudes_literature_filter_1
              )
    sigma_clip_mask = sigma_clip(
        zp_sum.pdf_median(),
        sigma=1.5,
    ).mask
    sigma_clip_mask = np.invert(sigma_clip_mask)

    #   Calculate magnitude differences
    diff_mag_1 = magnitudes_literature_filter_1 - magnitudes_observed_filter_1
    diff_mag_2 = magnitudes_literature_filter_2 - magnitudes_observed_filter_2

    #   Prepare fit variables
    #   -> apply sigma clipping mask
    #   -> calculate pdf_median() for now before fitting
    #   TODO: Test with distributions
    color_literature_err_plot = color_literature.pdf_std()[sigma_clip_mask]
    color_literature_plot = color_literature.pdf_median()[sigma_clip_mask]
    diff_mag_plot_1 = diff_mag_1.pdf_median()[sigma_clip_mask]
    diff_mag_plot_2 = diff_mag_2.pdf_median()[sigma_clip_mask]

    #   Plot illustrating the sigma clipping
    filter_str = ', '.join(filter_list)
    plots.scatter(
        [color_literature.pdf_median(), color_literature_plot],
        f'Color literature [mag]',
        [zp_sum.pdf_median(), zp_sum.pdf_median()[sigma_clip_mask]],
        f'Zero point sum - {filter_str} [mag]',
        f'zero_point_sum_sigma_clipped_image_{image.pd}',
        image.out_path.name,
        x_errors=[color_literature.pdf_std(), color_literature_err_plot],
        y_errors=[zp_sum.pdf_std(), zp_sum.pdf_std()[sigma_clip_mask]],
        dataset_label=[
            'without sigma clipping',
            'with sigma clipping',
        ],
        file_type=file_type_plots,
    )

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
    plots.plot_transform(
        image.out_path.name,
        filter_list[0],
        filter_list[1],
        filter_list[0],
        filter_list[id_current_filter],
        color_literature_plot.value,
        diff_mag_plot_1.value,
        z_1,
        color_correction_filter_1,
        color_correction_filter_1_err,
        fit_func,
        image.air_mass,
        color_literature_err=color_literature_err_plot.value,
        fit_variable_err=z_1_err,
        image_id=image.pd,
        x_data_original=color_literature.pdf_median(),
        y_data_original=diff_mag_1.pdf_median(),
        file_type=file_type_plots,
    )

    plots.plot_transform(
        image.out_path.name,
        filter_list[0],
        filter_list[1],
        filter_list[1],
        filter_list[id_current_filter],
        color_literature_plot.value,
        diff_mag_plot_2.value,
        z_2,
        color_correction_filter_2,
        color_correction_filter_2_err,
        fit_func,
        image.air_mass,
        color_literature_err=color_literature_err_plot.value,
        fit_variable_err=z_2_err,
        image_id=image.pd,
        x_data_original=color_literature.pdf_median(),
        y_data_original=diff_mag_2.pdf_median(),
        file_type=file_type_plots,
    )

    return color_correction_filter_1 * u.mag, color_correction_filter_2 * u.mag


def transformation_core(
        image: 'analyze.Image',
        magnitudes_current_filter: unc.core.NdarrayDistribution,
        magnitudes_literature_filter_1: unc.core.NdarrayDistribution,
        magnitudes_literature_filter_2: unc.core.NdarrayDistribution,
        calib_magnitudes_observed_filter_1: unc.core.NdarrayDistribution,
        calib_magnitudes_observed_filter_2: unc.core.NdarrayDistribution,
        magnitudes_filter_1: unc.core.NdarrayDistribution,
        magnitudes_filter_2: unc.core.NdarrayDistribution, tc_c: float,
        tc_color: float, tc_t1: float, tc_k1: float, tc_t2: float,
        tc_k2: float, id_current_filter: int, filter_list: list[str],
        transformation_type: str = 'derive',
        file_type_plots: str = 'pdf',
        ) -> unc.core.NdarrayDistribution:
    """
    Routine that performs the actual magnitude transformation.

    Parameters
    ----------
    image
        Object with all image specific properties

    magnitudes_current_filter
        Magnitudes of the filter that should be transformed

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

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    Returns
    -------
    color_observed
        Observed color of the calibration stars

    """
    #   Instrument color of the calibration objects
    color_observed = (calib_magnitudes_observed_filter_1 -
                      calib_magnitudes_observed_filter_2)

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
                magnitudes_literature_filter_1,
                magnitudes_literature_filter_2,
                calib_magnitudes_observed_filter_1,
                calib_magnitudes_observed_filter_2,
                file_type_plots=file_type_plots,
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

        #   Reshape the magnitudes to allow broadcasting because zp is an array
    magnitudes_current_filter = magnitudes_current_filter.reshape(
        magnitudes_current_filter.size,
        1,
    )

    #   Apply zero point to magnitudes
    magnitudes_with_zp = magnitudes_current_filter + unc.Distribution(image.zp)

    #   Calculate calibrated magnitudes
    color_term_all = c * color
    color_term_all = color_term_all.reshape(color_term_all.size, 1)
    color_term_calibration = c * color_observed
    calibrated_magnitudes = magnitudes_with_zp + color_term_all - color_term_calibration

    #   Sigma clipping to rm outliers
    _, median, stddev = sigma_clipped_stats(
        calibrated_magnitudes.distribution,
        sigma=1.5,
        axis=(1, 2),
    )

    #   Add calibrated photometry to table of Image object
    #   TODO: Add the photometry with filter_list information such that it is
    #         clear how the magnitudes are derived?
    image.photometry['mag_cali_trans'] = median
    image.photometry['mag_cali_trans_unc'] = stddev

    return color_observed


def apply_magnitude_transformation(
        calibration_stars_ids: np.ndarray, image: 'analyze.Image',
        calib_magnitudes_literature: list[u.quantity.Quantity],
        magnitudes_calibration_current_image: u.quantity.Quantity,
        magnitudes_calibration_comparison_image: u.quantity.Quantity,
        magnitudes_current_image: u.quantity.Quantity,
        magnitudes_comparison_image: u.quantity.Quantity, filter_id: int,
        filter_list: list[str],
        transformation_coefficients: dict[str, (float | str)],
        plot_sigma: bool = False, transformation_type: str = 'derive',
        multiprocessing: bool = False,
        file_type_plots: str = 'pdf') -> tuple[int, Table] | None:
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

    multiprocessing
        Switch to distinguish between single and multicore processing
        Default is ``False``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
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

    color_observed = transformation_core(
        image,
        magnitudes_current_image,
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
        file_type_plots=file_type_plots,
    )

    #   Quality control plots
    color_literature = (magnitudes_calibration_first_filter -
                        magnitudes_calibration_second_filter)
    utilities.prepare_calibration_check_plots(
        filter_list[filter_id],
        image.out_path.name,
        image.pd,
        calibration_stars_ids,
        calib_magnitudes_literature[filter_id].pdf_median(),
        image.photometry['mag_cali_trans'],
        magnitudes_current_image.pdf_median(),
        'magnitude_transformation',
        filter_list=filter_list,
        color_observed=color_observed.pdf_median(),
        color_literature=color_literature.pdf_median(),
        color_observed_err=color_observed.pdf_std(),
        color_literature_err=color_literature.pdf_std(),
        literature_magnitudes_err=calib_magnitudes_literature[filter_id].pdf_std(),
        magnitudes_err=image.photometry['mag_cali_trans_unc'],
        uncalibrated_magnitudes_err=magnitudes_current_image.pdf_std(),
        multiprocessing=not multiprocessing,
        file_type_plots=file_type_plots,
    )

    if multiprocessing:
        return image.pd, image.photometry


def calibrate_simple(
        image: 'analyze.Image',
        not_calibrated_magnitudes: unc.core.NdarrayDistribution,
        zp: unc.core.NdarrayDistribution,
        ) -> None:
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

    #    Sigma clipping to rm outliers
    _, median, stddev = sigma_clipped_stats(
        calibrated_magnitudes.distribution,
        sigma=1.5,
        axis=(1, 2),
    )

    #   Add calibrated photometry to table of Image object
    photometry_table['mag_cali_no-trans'] = median
    photometry_table['mag_cali_no-trans_unc'] = stddev


def quasi_flux_calibration_image_series(
        image_series: 'analyze.ImageSeries',
        distribution_samples: int = 1000) -> unc.core.NdarrayDistribution:
    """
        Simple calibration for flux values. Assuming the median over all
        objects in an image as a quasi ZP.

        Parameters
        ----------
        image_series
            image series object with flux and magnitudes of all objects in
            all images within the image series

        distribution_samples
            Number of samples used for distributions
            Default is `1000`.

        Returns
        -------
        flux_calibrated
            Quasi calibrated flux
    """
    #   Get flux as numpy array
    flux, flux_error = image_series.get_flux_array()

    #   Derive median of flux in individual images
    _, median, stddev = sigma_clipped_stats(
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
    flux_calibrated = flux_distribution / median[:, np.newaxis]

    return flux_calibrated


def flux_normalization_image_series(
        image_series: 'analyze.ImageSeries',
        quasi_calibrated_flux: unc.core.NdarrayDistribution | None = None,
        distribution_samples: int = 1000) -> unc.core.NdarrayDistribution:
    """
        Normalize flux of each object

        Parameters
        ----------
        image_series
            Object with flux and magnitudes of all objects in
            all images within the image series

        quasi_calibrated_flux
            Quasi-calibrated object flux: The median over all objects is
            used as the quasi ZP.

        distribution_samples
            Number of samples used for distributions
            Default is `1000`.

        Returns
        -------
        normalized_flux
            Normalized flux
    """
    if quasi_calibrated_flux is not None:
        flux_distribution = quasi_calibrated_flux
        flux = flux_distribution.pdf_median()
    else:
        flux, flux_error = image_series.get_flux_array()
        flux_distribution = unc.normal(
            flux,
            std=flux_error,
            n_samples=distribution_samples,
        )

    #   Calculated sigma clipped magnitudes
    _, median, stddev = sigma_clipped_stats(
        flux,
        axis=0,
        sigma=1.5,
        mask_value=0.0,
    )

    #   Prepare distributions
    # normalization_factor = unc.normal(
    #     median,
    #     std=stddev,
    #     n_samples=distribution_samples,
    # )
    normalized_flux = flux_distribution / median

    return normalized_flux


def prepare_zero_point(
        image: 'analyze.Image', id_filter: int,
        literature_magnitude_list: list[unc.core.NdarrayDistribution],
        magnitudes_calibration_stars: unc.core.NdarrayDistribution,
        calculate_zero_point_statistic: bool = True,
        sub_samples_zp_statistic: int = 1000, file_type_plots: str = 'pdf'
        ) -> unc.core.NdarrayDistribution:
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

    sub_samples_zp_statistic
        Number of randomly selected subsamples used for calculating zero
        point statistic
        Default is `1000`.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Calculate zero point
    zp = literature_magnitude_list[id_filter] - magnitudes_calibration_stars

    #   Plot zero point statistics
    plots.histogram_statistic(
        [zp.pdf_median()],
        f'Zero point ({image.filter_})',
        '',
        f'histogram_zero_point_{image.filter_}',
        str(image.out_path),
        dataset_label=[
            ['All calibration objects'],
        ],
        file_type=file_type_plots,
        # name_object=image.object_name,
    )

    #   Calculate zero point statistic
    n_calibration_objects = zp.shape[0]
    if n_calibration_objects > 20 and calculate_zero_point_statistic:
        terminal_output.print_to_terminal(
            f"Zero point statistic:",
            indent=2,
        )
        #   Create samples using numpy random number generator to generate
        #   an index array
        n_objects_sample = int(n_calibration_objects * 0.6)
        rng = np.random.default_rng()
        random_index = rng.integers(
            0,
            high=n_calibration_objects,
            size=(sub_samples_zp_statistic, n_objects_sample),
        )

        samples = zp.pdf_median()[random_index]

        #   Get statistic
        median_samples = np.median(samples, axis=1)
        median_over_samples = np.median(median_samples)
        standard_deviation_over_samples = np.std(median_samples)

        terminal_output.print_to_terminal(
            f"Based on {sub_samples_zp_statistic} randomly selected sub-samples, ",
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
        current_image: 'analyze.Image',
        index_calibration_stars: np.ndarray, current_filter_id: int,
        literature_magnitudes: list[u.quantity.Quantity],
        calculate_zero_point_statistic: bool = True,
        distribution_samples: int = 1000, file_type_plots: str = 'pdf'
        ) -> tuple[int, Table, np.ndarray]:
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

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    Returns
    -------
    pd
        ID of the image

    zp
        Zero point for the image
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
    magnitudes_calibration_current_image = calibration_data.observed_magnitude_of_calibration_stars(
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
        file_type_plots=file_type_plots,
    )

    #   Calibration without transformation
    calibrate_simple(
        current_image,
        magnitudes_current_image,
        zp,
    )

    #   Quality control plots
    utilities.prepare_calibration_check_plots(
        current_image.filter_,
        current_image.out_path.name,
        current_image.pd,
        index_calibration_stars,
        literature_magnitudes[current_filter_id].pdf_median(),
        current_image.photometry['mag_cali_no-trans'],
        magnitudes_current_image.pdf_median(),
        'simple_calibration',
        literature_magnitudes_err=literature_magnitudes[current_filter_id].pdf_std(),
        magnitudes_err=current_image.photometry['mag_cali_no-trans_unc'],
        uncalibrated_magnitudes_err=magnitudes_current_image.pdf_std(),
        multiprocessing=False,
        file_type_plots=file_type_plots,
    )

    return current_image.pd, current_image.photometry, zp.distribution


def calibrate_magnitudes_zero_point(
        observation: 'analyze.Observation', filter_list: (list[str] | set[str]),
        distribution_samples: int = 1000,
        calculate_zero_point_statistic: bool = True,
        n_cores_multiprocessing: int | None = None,
        file_type_plots: str = 'pdf', add_progress_bar: bool = True, indent: int = 1) -> None:
    """
    Apply the zero points to the magnitudes

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        Filter names

    distribution_samples
        Number of samples used for distributions
        Default is `1000`.

    calculate_zero_point_statistic
        If `True` a statistic on the zero points will be calculated.
        Default is ``True``.

    n_cores_multiprocessing
        Number of core used for multicore processing
        Default is ``None``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    add_progress_bar
        If ``True`` a progress bar will be shown illustrating the progress in
        the calibration based on the zero point.
        Default is ``True``.

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    terminal_output.print_to_terminal(
        "Apply zero point to magnitudes",
        indent=indent,
    )

    #   Get image series
    image_series_dict = observation.image_series_dict

    #   Get calibration magnitudes
    literature_magnitudes = calibration_data.distribution_from_calibration_table(
        observation.calib_parameters,
        filter_list,
        distribution_samples=distribution_samples,
    )

    for current_filter_id, filter_ in enumerate(filter_list):
        #   Get image series
        image_series = image_series_dict[filter_]

        #   Get image list
        image_list = image_series.image_list

        #   Initialize multiprocessing object
        executor = utilities.Executor(
            n_cores_multiprocessing,
            add_progress_bar=add_progress_bar,
            n_tasks=len(image_series.image_list),
        )

        #   Get IDs calibration data
        index_calibration_stars = observation.calib_parameters.ids_calibration_objects

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
                    'file_type_plots': file_type_plots,
                }
            )

        #   Exit multiprocessing, if exceptions will occur
        if executor.err is not None:
            #   TODO: Add some code that on requests tries to heal errors,
            #       such that crashed extractions on specific images will be
            #       ignored. -> Those images need to be removed from the
            #       image list at a later point.
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
        for image_ in image_series.image_list:
            for pd, tbl, zp in res:
                if pd == image_.pd:
                    image_.zp = zp
                    image_.photometry = tbl
                    tmp_list.append(image_)

        image_series.image_list = tmp_list


def calibrate_magnitudes_transformation(
        observation: 'analyze.Observation', filter_list: (list[str] | set[str]),
        transformation_coefficients: dict[str, (float | str)] | None = None,
        derive_transformation_coefficients: bool = False,
        plot_sigma: bool = False, distribution_samples: int = 1000,
        n_cores_multiprocessing: int | None = None,
        file_type_plots: str = 'pdf', add_progress_bar: bool = True,
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
    observation
        Container object with image series objects for each filter

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

    distribution_samples
        Number of samples used for distributions
        Default is ``1000``.

    n_cores_multiprocessing
        Number of core used for multicore processing
        Default is ``None``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    add_progress_bar
        If ``True`` a progress bar will be shown illustrating the progress of
        the magnitude transformation.
        Default is ``True``

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    terminal_output.print_to_terminal(
        "Apply magnitude transformation",
        indent=indent,
    )

    #   Get image series
    image_series_dict = observation.image_series_dict

    #   Initialize list for
    transformation_type_list: list[str | None] = []

    #   Get calibration magnitudes
    literature_magnitudes = calibration_data.distribution_from_calibration_table(
        observation.calib_parameters,
        filter_list,
        distribution_samples=distribution_samples,
    )

    #   Get IDs calibration data
    index_calibration_stars = observation.calib_parameters.ids_calibration_objects

    for current_filter_id, filter_ in enumerate(filter_list):
        #   Get image series
        current_image_series = image_series_dict[filter_]

        #   Get image list
        image_list = current_image_series.image_list

        #   Prepare transformation
        transformation_type, comparison_filter_id, trans_coefficients = check_transformation_requirements(
            observation,
            transformation_coefficients,
            filter_list,
            current_filter_id,
            derive_transformation_coefficients,
        )
        transformation_type_list.append(transformation_type)

        if transformation_type is not None:

            #   Initialize multiprocessing object
            executor = utilities.Executor(
                n_cores_multiprocessing,
                add_progress_bar=add_progress_bar,
                n_tasks=len(image_list),
            )

            for current_image in image_list:
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
                magnitudes_calibration_current_image = calibration_data.observed_magnitude_of_calibration_stars(
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
                magnitudes_calibration_comparison_image = calibration_data.observed_magnitude_of_calibration_stars(
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
                        magnitudes_current_image,
                        magnitudes_comparison_image,
                        current_filter_id,
                        filter_list,
                        trans_coefficients,
                    ),
                    kwargs={
                        'plot_sigma': plot_sigma,
                        'transformation_type': transformation_type,
                        'multiprocessing': True,
                        'file_type_plots': file_type_plots,
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
            for image_ in current_image_series.image_list:
                for pd, tbl in res:
                    if pd == image_.pd:
                        image_.photometry = tbl
                        tmp_list.append(image_)

            current_image_series.image_list = tmp_list

            terminal_output.print_to_terminal('')

    if not any(transformation_type_list):
        terminal_output.print_to_terminal(
            "WARNING: No magnitude transformation possible",
            indent=indent,
            style_name='WARNING'
        )


def apply_calibration(
        observation: 'analyze.Observation', filter_list: (list[str] | set[str]),
        transformation_coefficients_dict: dict[str, (float | str)] | None = None,
        derive_transformation_coefficients: bool = False, plot_sigma: bool = False,
        id_object: (int | None) = None, photometry_extraction_method: str = '',
        calculate_zero_point_statistic: bool = True, distribution_samples: int = 1000,
        n_cores_multiprocessing: int | None = None,
        file_type_plots: str = 'pdf', add_progress_bar: bool = True,
        indent: int = 1) -> None:
    """
    Apply the zero points to the magnitudes and perform a magnitude
    transformation if possible

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

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
        Default is ``1000``.

    n_cores_multiprocessing
        Number of core used for multicore processing
        Default is ``None``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    add_progress_bar
        If ``True`` a progress bar will be shown illustrating the progress in
        the calibration.
        Default is ``True``.

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    #   Apply zero point calibration
    calibrate_magnitudes_zero_point(
        observation,
        filter_list,
        distribution_samples=distribution_samples,
        calculate_zero_point_statistic=calculate_zero_point_statistic,
        n_cores_multiprocessing=n_cores_multiprocessing,
        file_type_plots=file_type_plots,
        add_progress_bar=add_progress_bar,
        indent=indent,
    )

    #   Apply magnitude transformation
    calibrate_magnitudes_transformation(
        observation,
        filter_list,
        transformation_coefficients=transformation_coefficients_dict,
        derive_transformation_coefficients=derive_transformation_coefficients,
        distribution_samples=distribution_samples,
        n_cores_multiprocessing=n_cores_multiprocessing,
        file_type_plots=file_type_plots,
        add_progress_bar=add_progress_bar,
        indent=indent,
    )

    if len(filter_list) == 1:
        rts: str = ''
    elif len(filter_list) == 2:
        rts: str = f'_{filter_list[0]}-{filter_list[1]}'
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \n{len(filter_list)} provided but only 1 or 2 are"
            f" supported => ABORT {style.Bcolors.ENDC}"
        )

    utilities.save_calibration(
        observation,
        filter_list,
        id_object,
        photometry_extraction_method=photometry_extraction_method,
        rts=rts,
    )


#   TODO: Test and cleanup
def determine_transformation_coefficients(
        observation: 'analyze.Observation', current_filter: str,
        filter_list: list[str], tbl_transformation_coefficients: Table,
        fit_function=utilities.lin_func,
        apply_uncertainty_weights: bool = True,
        distribution_samples: int = 1000, file_type_plots: str = 'pdf',
        indent: int = 2) -> None:
    """
    Determine the magnitude transformation factors

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    current_filter
        Current filter

    filter_list
        List of filter

    tbl_transformation_coefficients
        Astropy Table for the transformation coefficients

    fit_function
        Fit function to use for determining the calibration factors
        Default is ``lin_func``

    apply_uncertainty_weights
        If True the transformation fit will be weighted by the
        uncertainties of the data points.

    distribution_samples
        Number of samples used for distributions
        Default is `1000`.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    #   Get image series
    image_series_dict = observation.image_series_dict

    #   Set filter key
    id_filter = filter_list.index(current_filter)

    #   Get calibration parameters
    parameters_calibration = observation.calib_parameters

    #   Get index positions of calibration stars
    index_calibration_stars = observation.calib_parameters.ids_calibration_objects

    literature_magnitudes = calibration_data.distribution_from_calibration_table(
        parameters_calibration,
        filter_list,
        distribution_samples=distribution_samples,
    )
    #   Convert literature magnitudes to distributions
    #   -> This is necessary since astropy QuantityDistribution cannot be
    #      prickled/serialized. Therefore, literature_magnitudes is only the
    #      underlying numpy array
    #   TODO: Check if this workaround is still necessary
    tmp_list = []
    for magnitudes in literature_magnitudes:
        tmp_list.append(unc.Distribution(magnitudes))
    literature_magnitudes = tmp_list

    #   Check if literature magnitudes are not zero
    image_0 = image_series_dict[filter_list[0]].image_list[0]
    image_1 = image_series_dict[filter_list[1]].image_list[0]
    image_key = image_series_dict[current_filter].image_list[0]

    #   Get magnitude array images
    magnitudes_image_0 = utilities.distribution_from_table(
        image_0,
        distribution_samples=distribution_samples,
    )
    magnitudes_image_1 = utilities.distribution_from_table(
        image_1,
        distribution_samples=distribution_samples,
    )
    magnitudes_image_key = utilities.distribution_from_table(
        image_key,
        distribution_samples=distribution_samples,
    )

    magnitudes_calibration_image_0 = calibration_data.observed_magnitude_of_calibration_stars(
        magnitudes_image_0,
        index_calibration_stars,
    )
    magnitudes_calibration_image_1 = calibration_data.observed_magnitude_of_calibration_stars(
        magnitudes_image_1,
        index_calibration_stars,
    )
    magnitudes_calibration_image_key = calibration_data.observed_magnitude_of_calibration_stars(
        magnitudes_image_key,
        index_calibration_stars,
    )

    color_literature = literature_magnitudes[0] - literature_magnitudes[1]
    color_observed = (magnitudes_calibration_image_0 -
                      magnitudes_calibration_image_1)
    zero_point = (literature_magnitudes[id_filter] -
                  magnitudes_calibration_image_key)

    #   Initial guess for the parameters
    # x0    = np.array([0.0, 0.0])
    x0 = np.array([1.0, 1.0])

    #   Determine transformation coefficients
    #
    #   Plot variables
    color_literature_plot = color_literature.pdf_median()
    color_literature_err_plot = color_literature.pdf_std()
    color_observed_plot = color_observed.pdf_median()
    color_observed_err_plot = color_observed.pdf_std()
    zero_point_plot = zero_point.pdf_median()
    zero_point_err_plot = zero_point.pdf_std()

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
    plots.plot_transform(
        image_series_dict[filter_list[0]].out_path.name,
        filter_list[0],
        filter_list[1],
        current_filter,
        current_filter,
        color_literature_plot,
        color_observed_plot,
        a,
        b,
        tcolor_err,
        fit_function,
        image_series_dict[filter_list[0]].get_air_mass()[0],
        color_literature_err=color_literature_err_plot,
        fit_variable_err=color_observed_err_plot,
        file_type=file_type_plots,
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

    plots.plot_transform(
        image_series_dict[filter_list[0]].out_path.name,
        filter_list[0],
        filter_list[1],
        current_filter,
        current_filter,
        color_literature_plot,
        zero_point_plot,
        z_dash,
        t_mag,
        t_mag_err,
        fit_function,
        image_series_dict[filter_list[0]].get_air_mass()[0],
        color_literature_err=color_literature_err_plot,
        fit_variable_err=zero_point_err_plot,
        file_type=file_type_plots,
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


def calculate_trans(
        observation: 'analyze.Observation', key_filter: str,
        filter_list: list[str],
        tbl_transformation_coefficients: Table,
        apply_uncertainty_weights: bool = True,
        max_pixel_between_objects: int = 3, own_correlation_option: int = 1,
        calibration_method: str = 'APASS',
        vizier_dict: dict[str, str] | None = None,
        calibration_file: str | None = None,
        magnitude_range: tuple[float, float] = (0., 18.5),
        region_to_select_calibration_stars: RectanglePixelRegion | None = None,
        distribution_samples: int = 1000,
        duplicate_handling_object_identification: dict[str, str] | None = None,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf'
        ) -> None:
    """
    Calculate the transformation coefficients

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    key_filter
        Current filter

    filter_list
        List of filter

    tbl_transformation_coefficients
        Astropy Table for the transformation coefficients

    apply_uncertainty_weights
        If True the transformation fit will be weighted by the
        uncertainties of the data points.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    calibration_method
        Calibration method
        Default is ``APASS``.

    vizier_dict
        Dictionary with identifiers of the Vizier catalogs with valid
        calibration data
        Default is ``None``.

    calibration_file
        Path to the calibration file
        Default is ``None``.

    magnitude_range
        Magnitude range
        Default is ``(0.,18.5)``.

    region_to_select_calibration_stars
        Region in which to select calibration stars. This is a useful
        feature in instances where not the entire field of view can be
        utilized for calibration purposes.
        Default is ``None``.

    distribution_samples
        Number of samples used for distributions
        Default is `1000`.

    duplicate_handling_object_identification
        Specifies how to handle multiple object identification filtering during
        object identification.
        There are two options for each 'correlation_method':
            'own':     'first_in_list' and 'flux'.  The 'first_in_list'
                        filtering just takes the first obtained result.
            'astropy': 'distance' and 'flux'. The 'distance' filtering is
                        based on the distance between the correlated objects.
                        In this case, the one with the smallest distance is
                        used.
        The second option for both correlation method is based on the measure
        flux values. In this case the largest one is used.
        Default is ``None``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Sanitize dictionary with Vizier catalog information
    if vizier_dict is None:
        vizier_dict = {'APASS': 'II/336/apass9'}

    #   TODO: Add download of calibration data?

    #   Correlate the results from the different filter
    correlate.correlate_image_series(
        observation,
        filter_list,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        file_type_plots=file_type_plots,
        duplicate_handling_object_identification=duplicate_handling_object_identification,
    )

    #   Plot image with the final positions overlaid
    #   (final version)
    utilities.prepare_and_plot_starmap_from_observation(
        observation,
        filter_list,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots
    )

    #   Calibrate transformation coefficients
    calibration_data.derive_calibration(
        observation,
        filter_list,
        calibration_method=calibration_method,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        vizier_dict=vizier_dict,
        path_calibration_file=calibration_file,
        magnitude_range=magnitude_range,
        region_to_select_calibration_stars=region_to_select_calibration_stars,
        file_type_plots=file_type_plots,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
    )
    terminal_output.print_to_terminal('')

    #   Determine transformation coefficients & plot calibration plots
    determine_transformation_coefficients(
        observation,
        key_filter,
        filter_list,
        tbl_transformation_coefficients,
        apply_uncertainty_weights=apply_uncertainty_weights,
        distribution_samples=distribution_samples,
        file_type_plots=file_type_plots,
    )
    terminal_output.print_to_terminal('')
