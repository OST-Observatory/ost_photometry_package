############################################################################
####                            Libraries                               ####
############################################################################

from astropy.time import Time

import scipy.interpolate as interpolate

from . import terminal_output

############################################################################
####                        Routines & definitions                      ####
############################################################################

def get_img_types():
    '''
        Get image type designator: The terms that are used to identify bias,
        darks, flats, etc. in the image Headers.

        Returns
        -------
                            : `dictionary` of `string`
            Dictionary with the image type.
    '''
    #   Define default image types
    default_img_type = {
        'bias':['Bias Frame', 'Bias', 'BIAS'],
        'dark':['Dark Frame', 'Dark', 'DARK'],
        'flat':['Flat Field', 'Flat', 'FLAT'],
        'light':['Light Frame', 'Light', 'LIGHT'],
        }

    return default_img_type


def camera_info(camera, redout_mode, gain_setting=None):
    '''
        Camera specific parameters

        Parameters
        ----------
        camera          : `string`
            Camera or camera type used to obtain the data

        redout_mode     : `string`
            Mode used to readout the data from the camera chip.

        gain_setting    : `integer` or `None`, optional
            Gain used in the camera setting for cameras such as the QHYs.
            This is not the system gain, but it can be calculated from this
            value. See below.
            Default is ``None``.
    '''
    #   STF8300
    if camera in ['SBIG STF-8300 CCD Camera']:
        readnoise = 9.3
        gain      = None
        dark_rate = {0:0.18, -10:0.04, -15.8:0.02}
        d = 17.96
        h = 13.52

    #   QHY600M:
    elif camera in ['QHY600M']:
        readnoise = 7.904

        try:
            gain_fit_parameters = gain_fits['QHY600M'][redout_mode]
            spline = interpolate.BSpline(
                gain_fit_parameters['t'],
                gain_fit_parameters['c'],
                gain_fit_parameters['k'],
                extrapolate=False,
                )
            gain = spline(gain_setting)
        except:
            terminal_output.print_terminal(
                string="The true gain factor could not be determined. " \
                       "Use default value: 1.292.",
                indent=1,
                style_name='WARNING'
                )
            gain = 1.292

        dark_rate = {-20:0.0022, -10:0.0046}
        d = 32.00
        h = 24.00

    else:
        #   Default: modern CMOS camera assumption
        terminal_output.print_terminal(
            string="Camera not recognized. Assuming a modern CMOS camera ... ",
            indent=1,
            style_name='WARNING'
            )
        readnoise = 7.
        gain      = 1.
        dark_rate = {-20:0.0025, -10:0.005}
        d = 32.00
        h = 24.00

    return readnoise, gain, dark_rate, d, h


def get_chip_dimensions(instrument):
    '''
        Return camera chip dimensions in mm

        Parameters
        ----------
        instrument          : `string`
            Camera type or came driver name

        Returns
        -------
            d               : `float`
                Length of the camera chip

            h               : `float`
                Height of the camera chip
    '''
    info_camera = camera_info(instrument)
    return info_camera[4], info_camera[5]


###
#   Internal gain vs. system gain -> Spline fits to QHY data
gain_fits = {
    'QHY600M': {
        'Photography Mode': {
            't': [ 0.,          0.,         4.96183206,   10.,
                  14.96183206, 20.,         24.96183206,  25.95419847,
                  30.,         34.96183206, 40.,          44.96183206,
                  50.,         54.96183206, 55.95419847,  60.,
                  64.96183206, 70.,         74.96183206,  80.,
                  84.96183206, 90.,         94.96183206, 100.,           100.],
            'c': [1.29774436, 1.1387218,  0.98308271, 0.83082707, 0.68195489,
                  0.52969925, 0.4112782,  0.36390977, 0.32330827, 0.27255639,
                  0.22180451, 0.17443609, 0.12368421, 0.11015038, 0.07293233,
                  0.02218045, 0.01541353, 0.01203008, 0.00864662, 0.00526316,
                  0.00864662, 0.00526316, 0.00526316, 0.,         0.        ],
            'k': 1,
            },
        'High Gain Mode': {
            't': [ 0.,           0.,           4.96183206,  10.,
                  14.96183206,  20.,          24.96183206,  25.95419847,
                  30.,          34.96183206,  40.,          44.96183206,
                  50.,          54.96183206,  55.95419847,  60.,
                  64.96183206,  70.,          74.96183206,  80.,
                  85.03816794,  90.,          94.96183206, 100.,         100.],
            'c': [0.77330827, 0.73609023, 0.69887218, 0.66165414, 0.61766917,
                  0.58721805, 0.57706767, 0.54661654, 0.50601504, 0.47218045,
                  0.43496241, 0.3943609,  0.3537594,  0.33007519, 0.30300752,
                  0.26578947, 0.22518797, 0.19135338, 0.15413534, 0.11691729,
                  0.08308271, 0.04586466, 0.00864662, 0.,         0.        ],
            'k': 1,
            },
        'Extend Fullwell': {
            't': [ 0.,           0.,           4.96183206,  10.,
                  14.96183206,  20.,          24.96183206,  25.95419847,
                  30.,          35.03816794,  40.,          44.96183206,
                  50.,          54.96183206,  55.95419847,  60.,
                  64.96183206,  70.,          74.96183206,  80.,
                  85.03816794,  90.,          94.96183206, 100.,         100.],
            'c': [1.2943609,  1.23007519, 1.17255639, 1.10827068, 1.05413534,
                  0.98308271, 0.96616541, 0.92556391, 0.8612782,  0.80037594,
                  0.7462406,  0.68195489, 0.62443609, 0.6075188,  0.56015038,
                  0.50263158, 0.43834586, 0.37744361, 0.31654135, 0.2556391,
                  0.19473684, 0.13383459, 0.06954887, 0.,         0.        ],
            'k': 1,
            },
        'Extend Fullwell 2CMS': {
            't': [ 0.,           0.,           4.96183206,  10.,
                  14.96183206,  19.92366412,  24.96183206,  25.95419847,
                  30.,          34.96183206,  40.,          44.96183206,
                  50.,          54.96183206,  55.95419847,  59.92366412,
                  64.96183206,  69.92366412,  74.96183206,  80.,
                  85.03816794,  89.92366412,  95.03816794, 100.,          100.],
            'c': [1.25714286, 1.19962406, 1.15902256, 1.11503759, 1.05413534,
                  0.99323308, 0.9593985,  0.91879699, 0.84097744, 0.76654135,
                  0.70902256, 0.67180451, 0.62443609, 0.60075188, 0.55,
                  0.47894737, 0.44511278, 0.36390977, 0.31315789, 0.25225564,
                  0.1981203,  0.13383459, 0.07631579, 0.,         0.        ],
            'k': 1,
            },
        },
    }


###
#   Dictionary with Vizier catalog identifiers
#
vizier_dict = {
    'UCAC4':'I/322A',
    'GSC2.3':'I/305',
    'URAT1':'I/329',
    'NOMAD':'I/297',
    'HMUBV':'II/168/ubvmeans',
    'GSPC2.4':'II/272/gspc24',
    'APASS':'II/336/apass9',
    'Swift/UVOT':'II/339/uvotssc1',
    'XMM-OM':'II/370/xmmom5s',
    'VRI-NCC':'J/MNRAS/443/725/catalog',
    'USNO-B1.0':'I/284/out',
    }


###
#   Valid filter combinations to calculate magnitude transformation
#   dict -> key = filter, value = list(first color, second color)
#
valid_calibs = [['U','V'], ['B','V'], ['V','R'], ['V','I']]


###
#   Filter denomination vs. filter systems
#
filter_sytems = {
    'U':'bessell',
    'B':'bessell',
    'V':'bessell',
    'R':'bessell',
    'I':'bessell',
    'u`':'sdssu',
    'g`':'sdssg',
    'r`':'sdssr',
    'i`':'sdssi',
    'z-s`':'sdssz',
    'y`':'sdssy',
    # 'Blue':
    # 'Green':
    # 'Red':
    }


###
#   Magnitude calibration parameters
#   (Need to be ordered by date. Newest needs to be first.)
#
Tcs_qhy600m_20220420 = {
    'B':{
        'Filter 1':'B',
        #   Tbbv
        'T_1':0.085647,
        'T_1_err':1.3742e-05,
        'k_1':-0.048222,
        'k_1_err':7.1522e-06,
        'Filter 2':'V',
        #   Tvbv
        'T_2':0.0084589,
        'T_2_err':9.7904e-06,
        'k_2':-0.010063,
        'k_2_err':5.0955e-06,
        'type':'airmass',
        #   QHY600M
        'camera':['QHY600M'],
    },
    'V':{
        'Filter 1':'B',
        #   Tbbv
        'T_1':0.085858,
        'T_1_err':1.3649e-05,
        'k_1':-0.047997,
        'k_1_err':7.0814e-06,
        'Filter 2':'V',
        #   Tvbv
        'T_2':0.008503,
        'T_2_err':9.7294e-06,
        'k_2':-0.010016,
        'k_2_err':5.0477e-06,
        'type':'airmass',
        #   QHY600M
        'camera':['QHY600M'],
    },
}
Tcs_qhy600m_20080101 = {
    'B':{
        'Filter 1':'B',
        #   Tbbv
        'T_1':-0.11545,
        'T_1_err':0.020803,
        'k_1':-0.19031,
        'k_1_err':0.0088399,
        'Filter 2':'V',
        #   Tvbv
        'T_2':-0.32843,
        'T_2_err':0.0080104,
        'k_2':-0.1143,
        'k_2_err':0.0034039,
        'type':'airmass',
        #   QHY600M
        'camera':['QHY600M'],
    },
    'V':{
        'Filter 1':'B',
        #   Tbbv
        'T_1':-0.10083,
        'T_1_err':0.020197,
        'k_1':-0.17973,
        'k_1_err':0.0084819,
        'Filter 2':'V',
        #   Tvbv
        'T_2':-0.32454,
        'T_2_err':0.0075941,
        'k_2':-0.11125,
        'k_2_err':0.0031892,
        'type':'airmass',
        #   QHY600M
        'camera':['QHY600M'],
    },
}

Tcs = {
    '2022-04-20T00:00:00':{
        #   QHY600M
        # 'QHYCCD-Cameras-Capture':Tcs_qhy600m_20220420,
        # 'QHYCCD-Cameras2-Capture':Tcs_qhy600m_20220420,
        'QHY600M':Tcs_qhy600m_20220420,
    },
    '2008-01-01T00:00:00':{
        #   QHY600M
        # 'QHYCCD-Cameras-Capture': Tcs_qhy600m_20080101,
        # 'QHYCCD-Cameras2-Capture':Tcs_qhy600m_20080101,
        'QHY600M':Tcs_qhy600m_20080101,
    },
}


def getTcs(obsJD):
    '''
        Get the Tcs calibration values for the provided JD

        Parameters
        ----------
        obsJD           : `float`
            JD of the observation

        Returns
        -------
        Tcs             : `dictionary`
            Tcs calibration factors
    '''
    if obsJD is not None:
        for key in Tcs.keys():
            t = Time(key, format='isot', scale='utc')
            if obsJD >= t.jd:
                return Tcs[key]

    return None
