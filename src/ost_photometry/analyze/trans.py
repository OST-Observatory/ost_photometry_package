############################################################################
####                            Libraries                               ####
############################################################################

import sys

import numpy as np

from uncertainties import unumpy, ufloat

import multiprocessing as mp

from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip as sigma_clipping

from . import calib, analyze, aux, plot

from .. import checks, style, calibration_data, terminal_output

############################################################################
####                        Routines & definitions                      ####
############################################################################

def cal_err_T(image, lit_mags, color_mag, Tc, cali, id_1, id_2, id_f,
              ttype='simple', air_mass=1.0):
    '''
        Calculate errors in case of the simple magnitude transformation

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        lit_mags    : `numpy.ndarray`
            Literature magnitudes for the calibration stars

        color_mag   :  `numpy.ndarray` of `numpy.float64`
            Magnitude difference -> color

        Tc          : `dictionary`
            Calibration data - magnitude transformation

        cali        : `numpy.ndarray` - `numpy.float64`
            Standard deviation from the calibration

        id_1        : `integer`
            ID of filter 1 for the color

        id_2        : `integer`
            ID of filter 2 for the color

        id_f        : `integer`
            ID of the current filter

        ttype       : `string`
            Type of magnitude transformation

        air_mass    : `float`
            Air mass

        Returns
        -------
        u           : `numpy.ndarray`
            Propagated uncertainty
    '''

    #   Get mask from sigma clipping that needs to be applied to the data
    mask = image.ZP_mask

    #   Number of stars
    count = len(image.mags['err'])

    #   Define new array
    u = np.zeros(count, dtype=[('err', 'f8')])

    #   ZP errors
    uZP      = aux.err_prop(image.mags_fit['err'], lit_mags['err'][id_f])
    uZP_clip = np.median(uZP[mask])

    #   Literature color errors
    ucl      = aux.err_prop(lit_mags['err'][id_1], lit_mags['err'][id_2])
    ucl_clip = np.median(ucl[mask])

    for i in range(0,count):
        #   Err: delta(color) [(inst_2 - inst_1) - (lit_2 - lit_1)]
        udc = aux.err_prop(
            image.mags_1['err'][i],
            image.mags_2['err'][i],
            ucl_clip,
            )

        #   Color
        color = color_mag[i]

        #   Errors including magnitude transformation
        if ttype == 'simple':
            u_obj = aux.err_prop(
                image.mags['err'][i],
                uZP_clip,
                Tc['color'] * color * Tc['C_err'],
                Tc['C'] * color * Tc['color_err'],
                Tc['C'] * Tc['color'] * udc,
                )
        elif ttype == 'airmass':
            #   Calculate calibration factor
            C_1 = Tc['T_1'] - Tc['k_1'] * air_mass
            C_2 = Tc['T_2'] - Tc['k_2'] * air_mass

            #   C_1 & C_2 errors
            uC_1 = aux.err_prop(Tc['T_1_err'], air_mass*Tc['k_1_err'])
            uC_2 = aux.err_prop(Tc['T_2_err'], air_mass*Tc['k_2_err'])

        elif ttype == 'derive':
            C_1 = image.C_1
            C_2 = image.C_2
            uC_1 = image.C_1_err
            uC_2 = image.C_2_err
        else:
            raise Exception(
                f"{style.bcolors.FAIL} \nType of magnitude transformation not "
                "known \n\t-> Check calibration coefficients \n\t-> Exit"
                f"{style.bcolors.ENDC}"
                )

        if ttype in ['airmass', 'derive']:
            #   Calculate the corresponding denominator
            d    = 1. - C_1 + C_2

            #   Denominator error
            u_d  = aux.err_prop(uC_1, uC_2)

            #   C or more precise C'
            if id_f == id_1:
                C = C_1 / d
            elif id_f == id_2:
                C = C_2 / d

            #   C error
            if id_f == id_1:
                u_C = aux.err_prop(uC_1*d, u_d*C_1/d/d)
            elif id_f == id_2:
                u_C = aux.err_prop(uC_2*d, u_d*C_2/d/d)

            u_obj = aux.err_prop(
                image.mags['err'][i],
                uZP_clip,
                u_C * color,
                C * udc,
                )

        u['err'][i] = np.mean(u_obj)

    return u['err']


def cal_err(mask, mags_fit, lit_mags, mags, cali):
    '''
        Calculate errors in case of **no** magnitude transformation

        Parameters
        ----------
        mask        : `numpy.ndarray` - `boolean`
            Mask of calibration stars that should be excluded

        mags_fit    : `numpy.ndarray`
            Extracted magnitudes for the calibration stars

        lit_mags    : `numpy.ndarray`
            Literature magnitudes for the calibration stars

        mags        : `numpy.ndarray`
            Magnitudes of all objects

        cali        : `numpy.ndarray`
            Standard deviation from the calibration

        Returns
        -------
        u           : `numpy.ndarray`
            Propagated uncertainty
    '''
    #   ZP errors
    uZP      = aux.err_prop(mags_fit, lit_mags)
    uZP_clip = np.median(uZP[mask])

    #   Add up errors
    u = aux.err_prop(
        mags,
        uZP_clip,
        )

    return u


def cal_sigma_plot(m_fit, masked, filt, m_lit, outdir, nameobj, rts,
                   fit=None, m_fit_err=None, m_lit_err=None):
    '''
        Set up multiprocessing for sigma clipped magnitude plots

    Parameters
    ----------
        m_fit           : `numpy.ndarray` - `numpy.float64`
            Numpy structured array with extracted magnitudes for the
            calibration stars

        masked          : `numpy.ndarray` - `boolean`
            Mask of calibration stars that should be excluded

        filt            : `string`
            Filter used

        m_lit           : `numpy.ndarray` - `numpy.float64`
            Numpy structured array with literature magnitudes for the
            calibration stars

        outdir          : `string`
            Output directory

        nameobj         : `string`
            Name of the object

        rts             : `string`
                Expression characterizing the plot

        fit             : ` astropy.modeling.fitting` instance, optional
            Fit to plot
            Default is ``None``.

        m_fit_err       : `numpy.ndarray' or ``None``, optional
            Error of the filter 1 magnitudes

        m_lit_err       : `numpy.ndarray' or ``None``, optional
            Error of the filter 1 magnitudes

    '''
    p = mp.Process(
        target=plot.plot_mags,
        args=(
            m_fit[masked],
            filt+'_inst',
            m_lit[masked],
            filt+'_lit',
            'mags_sigma_'+filt+rts,
            outdir,
            ),
        kwargs={
            'nameobj':nameobj,
            'fit':fit,
            'err1':m_fit_err,
            'err2':m_lit_err,
            }
        )
    p.start()
    p = mp.Process(
        target=plot.plot_mags,
        args=(
            m_fit,
            filt+'_inst',
            m_lit,
            filt+'_lit',
            'mags_no_sigma_'+filt+rts,
            outdir,
            ),
        kwargs={
            'nameobj':nameobj,
            }
        )
    p.start()


def cal_sigma_plot_color(filt, outdir, nameobj, f_list, id_1, id_2,
                         color_fit, color_lit, color_fit_clip,
                         color_lit_clip, rts):
    '''
        Set up multiprocessing for sigma plots

    Parameters
    ----------
        filt            : `string`
            Filter used

        outdir          : `string`
            Output directory

        nameobj         : `string`
            Name of the object

        f_list          : `list` - `string`
            Filter list

        id_1            : `integer`
            ID of filter 1

        id_2            : `integer`
            ID of filter 2

        color_fit       : `numpy.ndarray` - `numpy.float64`
            Instrument color of the calibration stars

        color_lit       : `numpy.ndarray` - `numpy.float64`
            Literature color of the calibration stars

        color_fit_clip  : `numpy.ndarray` - `numpy.float64`
            Clipped instrument color of the calibration stars

        color_lit_clip  : `numpy.ndarray` - `numpy.float64`
            Clipped literature color of the calibration stars

        rts             : `string`
                Expression characterizing the plot
    '''
    p = mp.Process(
        target=plot.plot_mags,
        args=(
            color_fit_clip,
            f_list[id_1]+'-'+f_list[id_2]+'_inst',
            color_lit_clip,
            f_list[id_1]+'-'+f_list[id_2]+'_lit',
            'color_sigma_'+filt+rts,
            outdir,
            ),
        kwargs={
            'nameobj':nameobj,
            }
        )
    p.start()
    p = mp.Process(
        target=plot.plot_mags,
        args=(
            color_fit,
            f_list[id_1]+'-'+f_list[id_2]+'_inst',
            color_lit,
            f_list[id_1]+'-'+f_list[id_2]+'_lit',
            'color_no_sigma_'+filt+rts,
            outdir,
            ),
        kwargs={
            'nameobj':nameobj,
            }
        )
    p.start()


def prepare_trans_variables(img_container, id_img_i, filt_o, filt_i,
                            filt_id_1, filter_list, id_tuple_trans):
    '''
        Prepare variables for magnitude transformation

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        id_img_i        : `integer`
            ID of the image

        filt_o          : `integer`
            ID of the second filter

        filt_i          : `integer`
            ID of the current filter

        filt_id_1       : `integer`
            ID of the first filter of the calibration color

        filter_list     : `list` of `string`
            List of filter names

        id_tuple_trans  : `list` of `tuple` of `integer`
            Image and filter IDs

        Returns
        -------
        img_o           : `image.class`
            Image class with all image specific properties
    '''
    #   Get image ensemble
    img_ensembles = img_container.ensembles
    ensemble = img_ensembles[filter_list[filt_i]]

    #   Get image
    img_i = ensemble.image_list[id_img_i]

    #   Get observation time of current image and all images of the
    #   second filter
    obs_t_i = img_i.jd
    obs_times_o = img_ensembles[filter_list[filt_o]].get_obs_time()

    #   Find ID of the image with the nearest exposure time
    id_img_o = np.argmin(np.abs(obs_times_o - obs_t_i))

    #   Save filter and image ID configuration to allow
    #   for a better color calculation later on
    id_tuple_trans.append((filt_i, id_img_i, filt_o, id_img_o))

    #   Get image corresponding to this exposure time
    img_o = img_ensembles[filter_list[filt_o]].image_list[id_img_o]

    #   Get extracted magnitudes of the calibration stars
    #   for the image in the ``o``ther filter
    #   -> required for magnitude transformation
    calib.get_calib_fit(img_o, img_container)

    #   Set values for mag_fit_1 and mag_fit_2 to allow
    #   calculation of the correct color later on
    if filt_id_1 == filt_i:
        img_i.mag_fit_1 = img_i.mags_fit
        img_i.mag_fit_2 = img_o.mags_fit

        img_i.mags_1    = img_i.mags
        img_i.mags_2    = img_o.mags
    else:
        img_i.mag_fit_1 = img_o.mags_fit
        img_i.mag_fit_2 = img_i.mags_fit

        img_i.mags_1    = img_o.mags
        img_i.mags_2    = img_i.mags

    return img_o


def prepare_trans(img_container, Tcs, filter_list, filt_i, id_img_i,
                  id_tuple_notrans, derive_Tcs=False):
    '''
        Prepare magnitude transformation: find filter combination,
        get calibration parameters, prepare variables, ...

        Parameters
        ----------
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        Tcs                 : `dictionary` or ``None``
            Calibration coefficients for magnitude transformation

        filter_list         : `list` of `string`
            List of filter names

        filt_i              : `integer`
            ID of the current filter

        id_img_i            : `integer`
            ID of the image

        id_tuple_notrans    : `list` of `tuple` of `integer`
            Image and filter IDs

        derive_Tcs      : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the data base.
            Default is ``False``


        Returns
        -------
        Tc_type             : `string`
            Type of magnitude transformation to be performed

        filt_o              : `integer`
            ID of the second filter

        filt_id_1           : `integer`
            ID of the color filter 1. In B-V that would be B.

        filt_id_2           : `integer`
            ID of the color filter 2. In B-V that would be V.

        Tc                  : `dictionary`
            Dictionary with validated calibration parameters from Tcs.
    '''
    #   Get filter name
    band = filter_list[filt_i]

    #   Get image
    img_i = img_container.ensembles[band].image_list[id_img_i]

    #   Load calibration coefficients
    if Tcs is None:
        Tcs = calibration_data.getTcs(img_i.jd)

    #   Check if transformation is possible with the calibration
    #   coefficients. If not, try to derive calibration coefficients.
    if Tcs is not None and not derive_Tcs:
        Tc, filt_id_1, filt_id_2 = aux.find_filt(
            filter_list,
            Tcs,
            band,
            img_i.instrument,
            )

        if Tc is not None and 'type' in Tc.keys():
            Tc_type = Tc['type']

            #   Get correct filter order
            if filt_id_1 == filt_i:
                filt_o = filt_id_2
            else:
                filt_o = filt_id_1

        elif len(filter_list) >= 2:
            Tc_type = 'derive'

            #   Get correct filter ids: The first filter is the
            #   current filter, while the second filter is either
            #   the second in 'filter_list' or the one in 'filter_list'
            #    with the ID one below the first filter ID.
            filt_id_1 = filt_i

            if filt_id_1 == 0:
                filt_id_2 = 1
            else:
                filt_id_2 = filt_id_1-1

            filt_o = filt_id_2
        else:
            Tc_type = None

    elif len(filter_list) >= 2:
        Tc_type = 'derive'
        Tc = None
        filt_o = None
        filt_id_1 = None
        filt_id_2 = None

        #   Check if calibration data is available for the
        #   filter in``filter_list`
        filter_calib = img_container.calib_parameters.column_names
        for band in filter_list:
            if 'mag'+band not in filter_calib:
                Tc_type = None


        if Tc_type is not None:
            #   Get correct filter ids: The first filter is the
            #   current filter, while the second filter is either
            #   the second in 'filter_list' or the one in 'filter_list'
            #    with the ID one below the first filter ID.
            filt_id_1 = filt_i

            if filt_id_1 == 0:
                filt_id_2 = 1
            else:
                filt_id_2 = filt_id_1-1

            filt_o = filt_id_2

    if Tc_type is None:
        filt_o = None
        filt_id_1 = None
        filt_id_2 = None
        Tc = None

    if Tc_type == 'simple':
        string = "Apply simple magnitude transformation"
    elif Tc_type == 'airmass':
        string = "Apply magnitude transformation accounting for airmass"
    elif Tc_type == 'derive':
        string = "Derive and apply magnitude transformation based on "\
                "current image"
    if Tc_type is not None:
        terminal_output.print_terminal(indent=3, string=string)

    #   Save filter and image ID configuration to allow
    #   for a better color calculation later on
    id_tuple_notrans.append((filt_i, id_img_i))

    return Tc_type, filt_o, filt_id_1, filt_id_2, Tc


def derive_trans_onthefly(image, f_list, id_f, id_1, id_2, color_lit_clip,
                          lit_mag_1, lit_mag_2, mag_cali_fit_1,
                          mag_cali_fit_2):
    '''
        Determine the parameters for the color term used in the magnitude
        calibration. This corresponds to a magnitude transformation without
        considering the dependence on the air mass.

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        f_list              : `list` - `string`
            List of filter

        id_f                : `integer`
            ID of the current filter

        id_1                : `integer`
            ID of filter 1 for the color

        id_2                : `integer`
            ID of filter 2 for the color

        color_lit_clip      : `numpy.ndarray` or `unumpy.uarray`
            Literature color of the calibration stars

        lit_mag_1           : `numpy.ndarray` or `unumpy.uarray`
            Magnitudes of calibration stars from the literature
            for filter 1.

        lit_mag_2           : `numpy.ndarray` or `unumpy.uarray`
            Magnitudes of calibration stars from the literature
            for filter 1.

        mag_cali_fit_1      : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of the calibration stars from filter 1

        mag_cali_fit_2      : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of the calibration stars from filter 2



        Returns
        -------
        T_1                 : `ufloat` or `float`
            Color correction term for filter 1.

        T_2                 : `ufloat` or `float`
            Color correction term for filter 2.
    '''
    #   Initial guess for the parameters
    #x0    = np.array([0.0, 0.0])
    x0    = np.array([1.0, 1.0])

    #   Fit function
    fit_func=aux.lin_func

    #   Get required type for magnitude array.
    unc = checks.check_unumpy_array(color_lit_clip)

    #   Get variables
    diff_mag_1 = lit_mag_1 - mag_cali_fit_1
    diff_mag_2 = lit_mag_2 - mag_cali_fit_2
    if unc:
        color_lit_plot = unumpy.nominal_values(color_lit_clip)
        color_lit_err_plot = unumpy.std_devs(color_lit_clip)
        diff_mag_plot_1 = unumpy.nominal_values(diff_mag_1)
        diff_mag_plot_2 = unumpy.nominal_values(diff_mag_2)
    else:
        color_lit_plot = color_lit_clip
        color_lit_err_plot = 0.
        diff_mag_plot_1 = diff_mag_1
        diff_mag_plot_2 = diff_mag_2

    #   Set
    sigma = np.array(color_lit_err_plot)

    #   Fit
    Z_1, Z_1_err, T_1, T_1_err = aux.fit_curve(
        fit_func,
        color_lit_plot,
        diff_mag_plot_1,
        x0,
        sigma,
        )
    Z_2, Z_2_err, T_2, T_2_err = aux.fit_curve(
        fit_func,
        color_lit_plot,
        diff_mag_plot_2,
        x0,
        sigma,
        )
    if np.isinf(Z_1_err):
        Z_1_err=None
    if np.isinf(Z_2_err):
        Z_2_err=None

    #   Plots magnitude difference (literature vs. measured) vs. color
    plot.plot_transform(
        image.outpath.name,
        f_list[id_1],
        f_list[id_2],
        color_lit_plot,
        diff_mag_plot_1,
        Z_1,
        T_1,
        T_1_err,
        fit_func,
        image.air_mass,
        filt=f_list[id_f],
        color_lit_err=color_lit_err_plot,
        fit_var_err=Z_1_err,
        nameobj=image.objname,
        )

    if id_f == id_1:
        id_o = id_2
    else:
        id_o = id_1
    plot.plot_transform(
        image.outpath.name,
        f_list[id_1],
        f_list[id_2],
        color_lit_plot,
        diff_mag_plot_2,
        Z_2,
        T_2,
        T_2_err,
        fit_func,
        image.air_mass,
        filt=f_list[id_o],
        color_lit_err=color_lit_err_plot,
        fit_var_err=Z_2_err,
        nameobj=image.objname,
        )

    #   Return ufloat of normal float
    if unc:
        return ufloat(T_1, T_1_err), ufloat(T_2, T_2_err)
    else:
        image.C_1 = T_1             #   Dirty hack
        image.C_2 = T_2             #   Dirty hack
        image.C_1_err = T_1_err     #   Dirty hack
        image.C_2_err = T_2_err     #   Dirty hack
        return T_1, T_2


def apply_trans(*args, **kwargs):
    '''
        Apply magnitude transformation and return calibrated magnitude array

        Distinguishes between different input array types.
        Possibilities: unumpy.uarray & numpy structured ndarray
    '''
    #   Get type of the magnitude arrays
    unc = getattr(args[0], 'unc', True)

    if unc:
        apply_trans_unc(*args, **kwargs)
    else:
        apply_trans_str(*args, **kwargs)


def trans_core(image, lit_mag_1, lit_mag_2, mag_cali_fit_1, mag_cali_fit_2,
               mags_1, mags_2, mags, Tc_C, Tc_color, Tc_T1, Tc_k1, Tc_T2,
               Tc_k2, id_f, id_1, id_2, f_list, plot_sigma=False,
               ttype='derive'):
    '''
        Routine that performs the actual magnitude transformation.

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        lit_mag_1           : `numpy.ndarray` or `unumpy.uarray`
            Magnitudes of calibration stars from the literature
            for filter 1.

        lit_mag_2           : `numpy.ndarray` or `unumpy.uarray`
            Magnitudes of calibration stars from the literature
            for filter 1.

        mag_cali_fit_1      : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of the calibration stars from filter 1

        mag_cali_fit_2      : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of the calibration stars from filter 2

        mags_1              : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of objects from filter 1

        mags_2              : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes of objects from filter 2

        mags                : `numpy.ndarray` or `unumpy.uarray`
            Extracted magnitudes for the current filter

        Tc_C                : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        Tc_color            : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        Tc_T1               : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        Tc_k1               : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        Tc_T2               : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        Tc_k2               : `float` or `ufloat`
            Calibration parameter for the magnitude transformation

        id_f                : `integer`
            ID of the current filter

        id_1                : `integer`
            ID of filter 1 for the color

        id_2                : `integer`
            ID of filter 2 for the color

        f_list              : `list` - `string`
            List of filter

        plot_sigma      : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        ttype               : `string`, optional
            Type of magnitude transformation.
            Possibilities: simple, airmass, or derive
            Default is ``derive``.

        Returns
        -------
                            : `numpy.ndarray` or `unumpy.uarray`
            Calibrated magnitudes
    '''
    #   Get clipped zero points
    ZP = image.ZP_clip

    #   Get mask from sigma clipping that needs to be applied to the data
    mask = image.ZP_mask

    #   Instrument color of the calibration objects
    color_fit = mag_cali_fit_1 - mag_cali_fit_2
    #   Mask data according to sigma clipping
    color_fit_clip = color_fit[mask]

    #   Literature color of the calibration objects
    color_lit = lit_mag_1 - lit_mag_2
    #   Mask data according to sigma clipping
    color_lit_clip = color_lit[mask]


    ###
    #   Apply magnitude transformation and calibration
    #
    #   Color
    color_mag   = mags_1 - mags_2
    image.color_mag = color_mag

    #   Distinguish between versions
    if ttype == 'simple':
        #   Calculate calibration factor
        C = Tc_C * Tc_color
    elif ttype == 'airmass':
        #   Calculate calibration factor
        C_1 = Tc_T1 - Tc_k1 * image.air_mass
        C_2 = Tc_T2 - Tc_k2 * image.air_mass

    elif ttype == 'derive':
        #   Calculate color correction coefficients
        C_1, C_2 = derive_trans_onthefly(
            image,
            f_list,
            id_f,
            id_1,
            id_2,
            color_lit_clip,
            lit_mag_1[mask],
            lit_mag_2[mask],
            mag_cali_fit_1[mask],
            mag_cali_fit_2[mask],
            )

    else:
        raise Exception(
            f"{style.bcolors.FAIL}\nType of magnitude transformation not known"
            "\n\t-> Check calibration coefficients \n\t-> Exit"
            f"{style.bcolors.ENDC}"
            )

    if ttype in ['airmass', 'derive']:
        #   Calculate C or more precise C'

        denominator = 1. - C_1 + C_2

        if id_f == id_1:
            C   = C_1 / denominator
        elif id_f == id_2:
            C   = C_2 / denominator
        else:
            raise Exception(
                f"{style.bcolors.FAIL} \nMagnitude transformation: filter "
                "combination not valid \n\t-> This should never happen. The "
                f"current filter  ID is {id_f}, while filter IDs are {id_1} "
                f"and {id_2} {style.bcolors.ENDC}"
                )

    #   Calculate calibrated magnitudes
    mag_cali = mags + np.median(ZP - C * color_fit_clip) + C * color_mag

    p = mp.Process(
        target=plot.plot_mags,
        args=(
            unumpy.nominal_values(mag_cali),
            image.filt+'_calib',
            unumpy.nominal_values(mags),
            image.filt+'_no-calib',
            'mag-cali_mags_'+image.filt+'_img_'+str(image.pd),
            image.outpath.name,
            ),
        kwargs={
            'nameobj':image.objname,
            }
        )
    p.start()

    #   Add sigma clipping plots based on the color
    if plot_sigma:
        cal_sigma_plot_color(
            f_list[id_f],
            image.outpath.name,
            image.objname,
            f_list,
            id_1,
            id_2,
            unumpy.nominal_values(color_fit),
            unumpy.nominal_values(color_lit),
            unumpy.nominal_values(color_fit_clip),
            unumpy.nominal_values(color_lit_clip),
            '_img_'+str(image.pd),
            )

    return mag_cali


def apply_trans_str(img_container, image, lit_m, id_f, id_i, id_1, id_2,
                    f_list, Tc, plot_sigma=False, ttype='derive'):
    '''
        Apply transformation

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        image           : `image.class`
            Image class with all image specific properties

        lit_m           : `numpy.ndarray`
            Numpy structured array with literature magnitudes for the
            calibration stars

        id_f            : `integer`
            ID of the current filter

        id_i            : `integer`
            ID of the current image

        id_1            : `integer`
            ID of filter 1 for the color

        id_2            : `integer`
            ID of filter 2 for the color

        f_list          : `list` - `string`
            List of filter

        Tc              : `dictionary`
            Calibration coefficients for magnitude transformation

        plot_sigma      : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        ttype               : `string`, optional
            Type of magnitude transformation.
            Possibilities: simple, airmass, or derive
            Default is ``derive``.
    '''
    #   Get current filter
    filt  = f_list[id_f]

    #   Get necessary magnitudes arrays
    lit_mag        = lit_m['mag']
    mag_cali_fit_1 = image.mag_fit_1['mag']
    mag_cali_fit_2 = image.mag_fit_2['mag']
    mags_1         = image.mags_1['mag']
    mags_2         = image.mags_2['mag']
    mags           = image.mags['mag']

    #   Prepare calibration parameters
    Tc_T1    = None
    Tc_k1    = None
    Tc_T2    = None
    Tc_k2    = None
    Tc_C     = None
    Tc_color = None
    if ttype == 'simple':
        Tc_C     = Tc['C']
        Tc_color = Tc['color']
    elif ttype == 'airmass':
        Tc_T1    = Tc['T_1']
        Tc_k1    = Tc['k_1']
        Tc_T2    = Tc['T_2']
        Tc_k2    = Tc['k_2']

    #   Apply magnitude transformation
    mag_cali = trans_core(
        image,
        lit_mag[id_1],
        lit_mag[id_2],
        mag_cali_fit_1,
        mag_cali_fit_2,
        mags_1,
        mags_2,
        mags,
        Tc_C,
        Tc_color,
        Tc_T1,
        Tc_k1,
        Tc_T2,
        Tc_k2,
        id_f,
        id_1,
        id_2,
        f_list,
        plot_sigma=plot_sigma,
        ttype=ttype,
        )

    img_container.cali['mag'][id_f][id_i] = mag_cali

    #   Calculate uncertainties
    img_container.cali['err'][id_f][id_i] = cal_err_T(
        image,
        lit_m,
        image.color_mag,
        Tc,
        mag_cali,
        id_1,
        id_2,
        id_f,
        ttype=ttype,
        air_mass=image.air_mass,
        )


def apply_trans_unc(img_container, image, lit_m, id_f, id_i, id_1, id_2,
                    f_list, Tc, plot_sigma=False, ttype='derive'):
    '''
        Apply transformation

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        image           : `image.class`
            Image class with all image specific properties

        lit_m          : `numpy.ndarray`
            Unumpy array with literature magnitudes for the
            calibration stars

        id_f            : `integer`
            ID of the current filter

        id_i            : `integer`
            ID of the current image

        id_1            : `integer`
            ID of filter 1 for the color

        id_2            : `integer`
            ID of filter 2 for the color

        f_list          : `list` - `string`
            List of filter

        Tc              : `dictionary`
            Calibration coefficients for magnitude transformation

        plot_sigma      : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        ttype               : `string`, optional
            Type of magnitude transformation.
            Possibilities: simple, airmass, or derive
            Default is ``derive``.
    '''
    #   Get necessary magnitudes arrays
    lit_mag        = lit_m
    mag_cali_fit_1 = image.mag_fit_1
    mag_cali_fit_2 = image.mag_fit_2
    mags_1         = image.mags_1
    mags_2         = image.mags_2
    mags           = image.mags

    #   Prepare calibration parameters
    Tc_T1    = None
    Tc_k1    = None
    Tc_T2    = None
    Tc_k2    = None
    Tc_C     = None
    Tc_color = None
    if ttype == 'simple':
        Tc_C     = ufloat(Tc['C'], Tc['C_err'])
        Tc_color = ufloat(Tc['color'], Tc['color_err'])
    elif ttype == 'airmass':
        Tc_T1    = ufloat(Tc['T_1'], Tc['T_1_err'])
        Tc_k1    = ufloat(Tc['k_1'], Tc['k_1_err'])
        Tc_T2    = ufloat(Tc['T_2'], Tc['T_2_err'])
        Tc_k2    = ufloat(Tc['k_2'], Tc['k_2_err'])

    #   Apply magnitude transformation
    mag_cali = trans_core(
        image,
        lit_mag[id_1],
        lit_mag[id_2],
        mag_cali_fit_1,
        mag_cali_fit_2,
        mags_1,
        mags_2,
        mags,
        Tc_C,
        Tc_color,
        Tc_T1,
        Tc_k1,
        Tc_T2,
        Tc_k2,
        id_f,
        id_1,
        id_2,
        f_list,
        plot_sigma=plot_sigma,
        ttype=ttype,
        )

    img_container.cali[id_f][id_i] = mag_cali


def calibrate_simple(*args, **kwargs):
    '''
        Apply minimal calibration: No magnitude transformation & no other
                                   kind of color corrections.
    '''
    #   Get type of the magnitude arrays
    unc = getattr(args[0], 'unc', True)

    if unc:
        calibrate_unc(*args, **kwargs)
    else:
        calibrate_str(*args, **kwargs)


def calibrate_simple_core(image, mag_arr):
    '''
        Perform minimal calibration

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        mag_arr         : `numpy.ndarray`
            Array with object magnitudes

        Returns
        -------
        mag_cali        : `numpy.ndarray`
            Array with calibrated magnitudes
    '''
    #   Get clipped zero points
    ZP = image.ZP_clip

    #   Reshape the magnitude array to allow broadcasting
    resha_mag = mag_arr.reshape(mag_arr.size, 1)

    #   Calculate calibrated magnitudes
    mag_cali = resha_mag + ZP

    #   If ZP is 0, calibrate with the median of all magnitudes
    if np.all(ZP == 0.):
        mag_cali = resha_mag - np.median(mag_arr)

    return mag_cali


def calibrate_str(img_container, image, lit_m, id_f, id_img):
    '''
        Calibrate magnitudes without magnitude transformation

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        image           : `image.class`
            Image class with all image specific properties

        lit_m           : `numpy.ndarray`
            Numpy structured array with literature magnitudes for the
            calibration stars

        id_f            : `integer`
            ID of the current filter

        id_img:         : `integer`
            ID of the current image
    '''
    #   Get mask from sigma clipping
    mask = image.ZP_mask

    #   Get magnitudes array
    m_out = img_container.noT

    #   Get extracted magnitudes for all objects
    mag_arr = image.mags['mag']

    #   Perform calibration
    mag_cali = calibrate_simple_core(image, mag_arr)

    #   Sigma clipping to rm outliers and calculate median, ...
    __scs = sigma_clipped_stats(mag_cali, axis=1, sigma=1.5)
    m_out['mag'][id_f][id_img] = __scs[1]
    m_out['std'][id_f][id_img] = __scs[2]

    m_out['err'][id_f][id_img] = cal_err(
        mask,
        image.mags_fit['err'],
        lit_m['err'][id_f],
        image.mags['err'],
        m_out['std'][id_f][id_img],
        )

    #   Write data back to the image container
    img_container.noT = m_out
    try:
        img_container.flux['flux'][id_f][id_img] = image.flux_es['flux_fit']
        img_container.flux['err'][id_f][id_img] = image.flux_es['flux_unc']
    except:
        img_container.flux['flux'][id_f][id_img] = image.flux['flux_fit']
        img_container.flux['err'][id_f][id_img] = image.flux['flux_unc']


def calibrate_unc(img_container, image, lit_m, id_f, id_img):
    '''
        Calibrate magnitudes without magnitude transformation

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        image           : `image.class`
            Image class with all image specific properties

        lit_m           : `numpy.ndarray`
            Numpy structured array with literature magnitudes for the
            calibration stars

        id_f            : `integer`
            ID of the current filter

        id_img:         : `integer`
            ID of the current image
    '''
    #   Get extracted magnitudes for all objects
    mag_arr = image.mags

    #   Perform calibration
    mag_cali = calibrate_simple_core(image, mag_arr)

    #   Sigma clipping to rm outliers
    mag_cali_sigma = sigma_clipping(
        unumpy.nominal_values(mag_cali),
        sigma=1.5,
        axis=1,
        )
    mask = np.invert(mag_cali_sigma.mask)
    mask = np.any(mask, axis=0)

    #   Calculate median etc ...
    median = np.median(mag_cali[:, mask], axis=1)

    #   Write data back to the image container
    img_container.noT[id_f][id_img] = median
    try:
        img_container.flux[id_f][id_img] = image.uflux_es
    except:
        img_container.flux[id_f][id_img] = image.uflux


def flux_calibrate_ensemble(ensemble):
    '''
        Simple calibration for flux values. Assuming the median over all
        objects in an image as a quasi ZP.

        Parameters
        ----------
        ensemble        : `image.ensemble`
            Image ensemble object with flux and magnitudes of all objects in
            all images within the ensemble
    '''
    #   Get flux
    flux = ensemble.uflux

    #   Calculate median flux in each image
    median_flux = np.median(flux, axis=1)

    #   Calibrate
    flux_cali = flux / median_flux[:, np.newaxis]

    #   Add to ensemble
    ensemble.uflux_cali = flux_cali


def flux_normalize_ensemble(ensemble):
    '''
        Normalize flux

        Parameters
        ----------
        ensemble        : `image.ensemble`
            Image ensemble object with flux and magnitudes of all objects in
            all images within the ensemble
    '''
    #   Get flux
    try:
        flux = ensemble.uflux_cali
    except:
        flux = ensemble.uflux

    flux_values = unumpy.nominal_values(flux)

    #   Calculated sigma clipped magnitudes
    sigma_clipp_flux = sigma_clipped_stats(
        flux_values,
        axis=0,
        sigma=1.5,
        mask_value=0.0,
        )

    #   Get median values
    median = sigma_clipp_flux[1]
    std = sigma_clipp_flux[2]

    #   Add axis so that broadcasting to original array is possible
    median_reshape = median[np.newaxis,:]
    std_reshape = std[np.newaxis,:]

    #   Normalized magnitudes
    ensemble.uflux_norm = flux / unumpy.uarray(median_reshape,std_reshape)


def prepare_ZP(img_container, image, image_o, id_i, id_o=None):
    '''
        Prepare some values necessary for the magnitude calibration and add
        them to the image class

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        image               : `image.class`
            Image class with all image specific properties

        image_o             : `image.class`
            Second image class with all image specific properties

        id_i                : `integer`
            ID of the filter

        id_o                : `integer`, optional
            ID of the `second` image/filter that is used for the magnitude
            transformation.
            Default is ``None``.
    '''
    #   Get type of the magnitudes array used
    #   Possibilities: structured numpy array & unumpy uarray
    unc = getattr(img_container, 'unc', True)

    #   Set array with literature magnitudes for the calibration stars
    mag_lit  = img_container.calib_parameters.mags_lit
    if not unc:
        mag_lit = mag_lit['mag']

    #   Get extracted magnitudes
    mag_fit_i  = image.mags_fit
    if not unc:
        mag_fit_i = mag_fit_i['mag']

    if id_o is not None:
        mag_fit_o  = image_o.mags_fit
        if not unc:
            mag_fit_o = mag_fit_o['mag']

    #   Calculated color. For two filter calculate delta color
    if id_o is not None:
        del_col_calib = mag_fit_i + mag_fit_o - mag_lit[id_i] - mag_lit[id_o]

    else:
        del_col_calib = mag_fit_i - mag_lit[id_i]


    #   Calculate mask according to sigma clipping
    if unc:
        clip_values = unumpy.nominal_values(del_col_calib)
    else:
        clip_values = del_col_calib
    clip = sigma_clipping(clip_values, sigma=1.5)
    image.ZP_mask = np.invert(clip.recordmask)

    #   Plot sigma clipping if it makes sense
    if not np.all(mag_lit == 0.):
        #   Make fit
        ZP_fit = aux.fit_data_oneD(
            mag_fit_i[image.ZP_mask],
            mag_lit[id_i][image.ZP_mask],
            1,
            )

        #   Get plot variables
        if unc:
            mag_fit_i_plot = unumpy.nominal_values(mag_fit_i)
            mag_lit_plot = unumpy.nominal_values(mag_lit[id_i])
            m_fit_err_plot = unumpy.std_devs(mag_fit_i[image.ZP_mask])
            m_lit_err_plot = unumpy.std_devs(mag_lit[id_i][image.ZP_mask])
        else:
            mag_fit_i_plot = mag_fit_i
            mag_lit_plot = mag_lit[id_i]
            m_fit_err_plot = mag_fit_i[image.ZP_mask]
            m_lit_err_plot = mag_lit[id_i][image.ZP_mask]

        cal_sigma_plot(
            mag_fit_i_plot,
            image.ZP_mask,
            image.filt,
            mag_lit_plot,
            image.outpath.name,
            image.objname,
            '_img_'+str(image.pd),
            fit=ZP_fit,
            m_fit_err=m_fit_err_plot,
            m_lit_err=m_lit_err_plot,
            )

    #   Calculate zero points and clip
    image.ZP       = mag_lit[id_i] - mag_fit_i
    image.ZP_clip  = image.ZP[image.ZP_mask]


def apply_calib(img_container, filter_list, Tcs=None, derive_Tcs=False,
                plot_sigma=False, plot_mags=True, ID='', photo_type='',
                refid=0, indent=1):
    '''
        Apply the calibration to the magnitudes and perform a magnitude
        transformation if possible

        # Using:
        # Δ(b-v) = (b-v)obj - (b-v)cali
        # Δ(B-V) = Tbv * Δ(b-v)
        # Vobj = Δv + Tv_bv * Δ(B-V) + Vcomp or Vobj
               = v + Tv_bv*Δ(B-V) - v_cali


        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        filter_list     : `list` of `string`
            Filter names

        Tcs             : `dictionary`, optional
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_Tcs      : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the data base.
            Default is ``False``

        plot_sigma      : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        plot_mags       : `boolean', optional
            If True a star map plot with the 100 faintest objects will be
            created.
            Default is ``True``.

        ID              : `string`, optional
            ID of the object
            Default is ``''``.

        photo_type      : `string`, optional
            Applied extraction method. Possibilities: ePSF or APER`
            Default is ``''``.

        refid           : `integer`, optional
            ID of the reference image
            Default is ``0``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.

        Returns
        -------
        cali or noT     : `numpy.ndarray`
            Array with magnitudes and errors
    '''
    terminal_output.print_terminal(
        indent=indent,
        string="Apply calibration and perform magnitude transformation",
        )

    #   Get image ensembles
    img_ensembles = img_container.ensembles

    #   Get object indices and X & Y pixel positions
    try:
        ind = img_ensembles[filter_list[0]].id_es
        x   = img_ensembles[filter_list[0]].x_es
        y   = img_ensembles[filter_list[0]].y_es
    except:
        ind = img_ensembles[filter_list[0]].id_s
        x   = img_ensembles[filter_list[0]].x_s
        y   = img_ensembles[filter_list[0]].y_s

    #   Number of filter
    nfilter = len(filter_list)

    #   Get number of objects
    try:
        count = len(img_ensembles[filter_list[0]].id_es)
    except:
        count = len(img_ensembles[filter_list[0]].id_s)

    #   Prepare arrays
    aux.prepare_arrays(img_container, nfilter, count)

    #   Initialize bool and image ID for transformation
    id_tuple_trans = []
    id_tuple_notrans = []

    #   Get calibration magnitudes
    lit_mags    = img_container.calib_parameters.mags_lit

    for filt_i, band in enumerate(filter_list):
        #   Get image ensemble
        img_ensemble = img_ensembles[band]

        #   Get image list
        img_list = img_ensemble.image_list

        #   Prepare transformation
        Tc_type, filt_o, filt_id_1, filt_id_2, Tc = prepare_trans(
            img_container,
            Tcs,
            filter_list,
            filt_i,
            0,
            id_tuple_notrans,
            derive_Tcs=derive_Tcs
            )

        #   Loop over images
        for id_img_i, img_i in enumerate(img_list):
            #   Get extracted magnitudes of the calibration stars for the
            #   current image
            calib.get_calib_fit(img_i, img_container)

            #   Prepare some variables and find corresponding image to img_i
            if Tc_type is not None:
                img_o = prepare_trans_variables(
                    img_container,
                    id_img_i,
                    filt_o,
                    filt_i,
                    filt_id_1,
                    filter_list,
                    id_tuple_trans,
                    )
            else:
                img_o = None

            #   Prepare ZP for the magnitude calibration and perform
            #   sigma clipping on the delta color or color, depending on
            #   whether magnitude transformation is possible or not.
            prepare_ZP(
                img_container,
                img_i,
                img_o,
                filt_i,
                id_o=filt_o,
                )


            ###
            #   Calculate transformation if possible
            #
            if Tc_type is not None:
                apply_trans(
                    img_container,
                    img_i,
                    lit_mags,
                    filt_i,
                    id_img_i,
                    filt_id_1,
                    filt_id_2,
                    filter_list,
                    Tc,
                    plot_sigma=plot_sigma,
                    ttype=Tc_type,
                    )


            ###
            #   Calibration without transformation
            #
            calibrate_simple(img_container, img_i, lit_mags, filt_i, id_img_i)


            ####
            ##   Plot star map
            ##
            #if plot_mags:
                #if u == refid:
                    #if trans_key[i]:
                        #mags_plot = cali['med'][i][u]
                    #else:
                        #mags_plot = noT['med'][i][u]
                    #prepare_and_plot_starmap(
                        #img,
                        #x,
                        #y,
                        #mags_plot,
                        #ID,
                        #band,
                        #)

        img_container.Tc_type = None


    ###
    #   Save results as ASCII files
    #
    cali = img_container.cali
    if not checks.check_unumpy_array(cali):
        cali = cali['mag']

    if np.any(cali != 0.):
        #   If transformation is available
        aux.save_mags_ascii(
            img_container,
            aux.mk_mag_table(
                ind,
                x,
                y,
                img_container.cali,
                filter_list,
                id_tuple_trans,
                ),
            trans=True,
            ID=ID,
            photo_type=photo_type,
            )
    else:
        terminal_output.print_terminal(
            indent=indent,
            string="WARNING: No magnitude transformation possible",
            style_name='WARNING'
            )

    #   Without transformation
    aux.save_mags_ascii(
        img_container,
        aux.mk_mag_table(
            ind,
            x,
            y,
            img_container.noT,
            filter_list,
            id_tuple_notrans,
            ),
        trans=False,
        ID=ID,
        photo_type=photo_type,
        )


def deter_trans(img_container, key_filt, filter_list, tbl_trans,
                fit_func=aux.lin_func, weights=True, indent=2):
    '''
        Determine the magnitude transformation factors

        Parameters
        ----------
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        key_filt            : `string`
            Current filter

        filter_list         : `list` of `strings`
            List of filter

        tbl_trans           : `astropy.table.Table`
            Astropy Table for the transformation coefficients

        fit_func            : `function`, optional
            Fit function to use for determining the calibration factors
            Default is ``lin_func``

        weights             : `boolean`, optional
            If True the transformation fit will be weighted by the
            uncertainties of the data points.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    '''
    #   Get image ensembles
    ensemble_dict = img_container.ensembles

    #   Set filter key
    id_filt = filter_list.index(key_filt)

    #   Get calibration parameters
    calib_parameters = img_container.calib_parameters

    #   Get calibration data
    mags_lit = calib_parameters.mags_lit

    #   Get required type for magnitude array.
    unc = checks.check_unumpy_array(mags_lit)

    if unc:
        test_mag_filter1 = mags_lit[0][0]
        test_mag_filter2 = mags_lit[1][0]
    else:
        test_mag_filter1 = mags_lit['mag'][0][0]
        test_mag_filter2 = mags_lit['mag'][1][0]

    #   Check if magnitudes are not zero
    if test_mag_filter1 != 0. and test_mag_filter2 != 0.:
        img_1 = ensemble_dict[filter_list[0]].image_list[0]
        img_2 = ensemble_dict[filter_list[1]].image_list[0]
        img_key = ensemble_dict[filter_list[id_filt]].image_list[0]

        #   Extract values from a structured Numpy array
        calib.get_calib_fit(img_1, img_container)
        calib.get_calib_fit(img_2, img_container)
        calib.get_calib_fit(img_key, img_container)

        if unc:
            fit_mags_1   = img_1.mags_fit
            fit_mags_2   = img_2.mags_fit
            fit_mags_key = img_key.mags_fit
        else:
            fit_mags_1   = img_1.mags_fit['mag']
            fit_mags_2   = img_2.mags_fit['mag']
            fit_mags_key = img_key.mags_fit['mag']
            fit_err_1    = img_1.mags_fit['err']
            fit_err_2    = img_2.mags_fit['err']
            fit_err_key  = img_key.mags_fit['err']

        #   Calculate values
        if unc:
            lit_mags = mags_lit
        else:
            lit_mags = mags_lit['mag']
            lit_errs = mags_lit['err']

            color_lit_err = aux.err_prop(lit_errs[0], lit_errs[1])
            color_fit_err = aux.err_prop(fit_err_1, fit_err_2)
            zero_err      = aux.err_prop(lit_errs[id_filt], fit_err_key)

        color_lit = lit_mags[0] - lit_mags[1]
        color_fit = fit_mags_1 - fit_mags_2
        zero      = lit_mags[id_filt] - fit_mags_key

        #   Initial guess for the parameters
        #x0    = np.array([0.0, 0.0])
        x0    = np.array([1.0, 1.0])


        ###
        #   Determine transformation coefficients
        #

        #   Plot variables
        if unc:
            color_lit_plot = unumpy.nominal_values(color_lit)
            color_lit_err_plot = unumpy.std_devs(color_lit)
            color_fit_plot = unumpy.nominal_values(color_fit)
            color_fit_err_plot = unumpy.std_devs(color_fit)
            zero_plot = unumpy.nominal_values(zero)
            zero_err_plot = unumpy.std_devs(zero)
        else:
            color_lit_plot = color_lit
            color_lit_err_plot = color_lit_err
            color_fit_plot = color_fit
            color_fit_err_plot = color_fit_err
            zero_plot = zero
            zero_err_plot = zero_err


        #   Color transform - Fit the data with fit_func
        #   Set sigma, using errors calculate above
        if weights:
            sigma = np.array(color_fit_err_plot)
        else:
            sigma = 0.

        #   Fit
        a, _, b, Tcolor_err = aux.fit_curve(
            fit_func,
            color_lit_plot,
            color_fit_plot,
            x0,
            sigma,
            )

        Tcolor = 1./b

        #   Plot color transform
        terminal_output.print_terminal(
            key_filt,
            indent=indent,
            string="Plot color transformation ({})",
            )
        plot.plot_transform(
            ensemble_dict[filter_list[0]].outpath.name,
            filter_list[0],
            filter_list[1],
            color_lit_plot,
            color_fit_plot,
            a,
            b,
            Tcolor_err,
            fit_func,
            ensemble_dict[filter_list[0]].get_air_mass()[0],
            color_lit_err=color_lit_err_plot,
            fit_var_err=color_fit_err_plot,
            nameobj=ensemble_dict[filter_list[0]].objname,
            )


        ##  Mag transform - Fit the data with fit_func
        #   Set sigma, using errors calculate above
        if weights:
            sigma = zero_err_plot
        else:
            sigma = 0.


        #   Fit
        Zdash, Zdash_err, Tmag, Tmag_err = aux.fit_curve(
            fit_func,
            color_lit_plot,
            zero_plot,
            x0,
            sigma,
            )

        #   Plot mag transformation
        terminal_output.print_terminal(
            key_filt,
            indent=indent,
            string="Plot magnitude transformation ({})",
            )

        plot.plot_transform(
            ensemble_dict[filter_list[0]].outpath.name,
            filter_list[0],
            filter_list[1],
            color_lit_plot,
            zero_plot,
            Zdash,
            Tmag,
            Tmag_err,
            fit_func,
            ensemble_dict[filter_list[0]].get_air_mass()[0],
            filt=key_filt,
            color_lit_err=color_lit_err_plot,
            fit_var_err=zero_err_plot,
            nameobj=ensemble_dict[filter_list[0]].objname,
            )

        #   Redefine variables -> shorter variables
        key_filt_l = key_filt.lower()
        f_0_l = filter_list[0].lower()
        f_1_l = filter_list[1].lower()
        f_0   = filter_list[0]
        f_1   = filter_list[1]

        #   Fill calibration table
        tbl_trans['C'+key_filt_l+f_0_l+f_1_l]            = [Tmag]
        tbl_trans['C'+key_filt_l+f_0_l+f_1_l+'_err']     = [Tmag_err]
        tbl_trans['Zdash'+key_filt_l+f_0_l+f_1_l]        = [Zdash]
        tbl_trans['Zdash'+key_filt_l+f_0_l+f_1_l+'_err'] = [Zdash_err]
        tbl_trans['T'+f_0_l+f_1_l]                       = [Tcolor]
        tbl_trans['T'+f_0_l+f_1_l+'_err']                = [Tcolor_err]

        #   Print results
        terminal_output.print_terminal(
            key_filt,
            indent=indent,
            string="Plot magnitude transformation ({})",
            )
        terminal_output.print_terminal(
            indent=indent,
            string="###############################################",
            )
        terminal_output.print_terminal(
            f_0_l,
            f_1_l,
            f_0,
            f_1,
            indent=indent,
            string="Colortransform ({}-{} vs. {}-{}):",
            )
        terminal_output.print_terminal(
            f_0_l,
            f_1_l,
            Tcolor,
            Tcolor_err,
            indent=indent+1,
            string="T{}{} = {:.5f} +/- {:.5f}",
            )
        terminal_output.print_terminal(
            key_filt,
            key_filt,
            key_filt_l,
            f_0,
            f_1,
            indent=indent,
            string="{}-mag transform ({}-{} vs. {}-{}):",
            )
        terminal_output.print_terminal(
            key_filt_l,
            f_0_l,
            f_1_l,
            Tmag,
            Tmag_err,
            indent=indent+1,
            string="T{}_{}{} = {:.5f} +/- {:.5f}",
            )
        terminal_output.print_terminal(
            indent=indent,
            string="###############################################",
            )


def calculate_trans(img_container, key_filt, filt_list, tbl_trans,
                    weights=True, dcr=3., option=1, calib_method='APASS',
                    vizier_dict={'APASS':'II/336/apass9'}, calib_file=None,
                    mag_range=(0.,18.5), rm_obj_coord=None, indent='      '):
    '''
        Calculate the transformation coefficients

        Parameters
        ----------
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        key_filt            : `string`
            Current filter

        filt_list         : `list` of `strings`
            List of filter

        tbl_trans           : `astropy.table.Table`
            Astropy Table for the transformation coefficients

        weights             : `boolean`, optional
            If True the transformation fit will be weighted by the
            uncertainties of the data points.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        calib_method           : `string`, optional
            Calibration method
            Default is ``APASS``.

        vizier_dict             : `dictionary`, optional
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``{'APASS':'II/336/apass9'}``.

        calib_file              : `string`, optional
            Path to the calibration file
            Default is ``None``.

        mag_range               : `tupel` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        rm_obj_coord            : `astropy.coordinates.SkyCoord`, optional
            Coordinates of an object that should not be used for calibrating
            the data.
            Default is ``None``.

        indent              : `string`, optional
            Indentation for the console output lines
            Default is ``'      '``.
    '''
    ###
    #   Correlate the results from the different filter
    #
    analyze.correlate_ensemble(
        img_container,
        filt_list,
        dcr=dcr,
        option=option,
        )


    ###
    #   Plot image with the final positions overlaid
    #   (final version)
    #
    aux.prepare_and_plot_starmap_final(
        img_container,
        filt_list,
        )


    ###
    #   Calibrate transformation coefficients
    #
    calib.deter_calib(
        img_container,
        filt_list,
        calib_method=calib_method,
        dcr=dcr,
        option=option,
        vizier_dict=vizier_dict,
        calib_file=calib_file,
        mag_range=mag_range,
        )
    terminal_output.print_terminal()


    ###
    #   Determine transformation coefficients
    #   & Plot calibration plots
    #
    deter_trans(
        img_container,
        key_filt,
        filt_list,
        tbl_trans,
        weights=weights,
        )
    terminal_output.print_terminal()
