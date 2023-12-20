############################################################################
#                               Libraries                                  #
############################################################################

from astropy.time import Time
import astropy.units as u

from uncertainties import ufloat

import scipy.interpolate as interpolate

from . import terminal_output

############################################################################
#                           Routines & definitions                         #
############################################################################


def get_image_types():
    """
        Get image type designator: The terms that are used to identify bias,
        darks, flats, etc. in the image Headers.

        Returns
        -------
                            : `dictionary` of `string`
            Dictionary with the image type.
    """
    #   Define default image types
    default_img_type = {
        'bias': ['Bias Frame', 'Bias', 'BIAS'],
        'dark': ['Dark Frame', 'Dark', 'DARK'],
        'flat': ['Flat Field', 'Flat', 'FLAT'],
        'light': ['Light Frame', 'Light', 'LIGHT'],
    }

    return default_img_type


def chip_dimensions(camera):
    """
        Parameters
        ----------
        camera          : `string`
            Camera or camera type used to obtain the data

        Returns
        -------
        d               : `integer`
            Width in pixel

        h               : `integer`
            Height in pixel
    """
    #   STF8300
    if camera in ['SBIG STF-8300 CCD Camera']:
        d = 17.96
        h = 13.52

    elif camera == 'QHY600M':
        d = 32.00
        h = 24.00

    elif camera == 'QHY268M':
        d = 23.45
        h = 15.7

    else:
        d = 32.00
        h = 24.00

    return d, h


def camera_info(camera, readout_mode, temperature, gain_setting=None):
    """
        Camera specific parameters

        Parameters
        ----------
        camera          : `string`
            Camera or camera type used to obtain the data

        readout_mode    : `string`
            Mode used to read out the data from the camera chip.

        temperature     : `float`
            Temperature of the camera chip

        gain_setting    : `integer` or `None`, optional
            Gain used in the camera setting for cameras such as the QHYs.
            This is not the system gain, but it can be calculated from this
            value. See below.
            Default is ``None``.

        Returns
        -------
        read_noise      : `float`
            Read noise

        gain            : `float`
            Gain factor

        dark_rate       : `float`
            Dark current

        d               : `integer`
            Width in pixel

        h               : `integer`
            Height in pixel
    """
    #   STF8300
    if camera in ['SBIG STF-8300 CCD Camera']:
        read_noise = 9.3
        gain = None
        # dark_rate = {0:0.18, -10:0.04, -15.8:0.02}

    #   QHYs
    elif camera in ['QHY600M', 'QHY268M']:
        try:
            read_noise_fit_parameters = gain_qhy[camera][readout_mode]
            spline = interpolate.BSpline(
                read_noise_fit_parameters['t'],
                read_noise_fit_parameters['c'],
                read_noise_fit_parameters['k'],
                extrapolate=False,
            )
            read_noise = spline(gain_setting)
        except Exception as e:
            terminal_output.print_to_terminal(
                f'Camera: {camera}\n'
                "   The read noise could not be determined... \n"
                "   Use default value: 7.904\n"
                f"  Readout mode was {e}\n",
                indent=1,
                style_name='WARNING'
            )
            read_noise = 7.904

        try:
            gain_fit_parameters = gain_qhy[camera][readout_mode]
            spline = interpolate.BSpline(
                gain_fit_parameters['t'],
                gain_fit_parameters['c'],
                gain_fit_parameters['k'],
                extrapolate=False,
            )
            gain = spline(gain_setting)
        except Exception as e:
            terminal_output.print_to_terminal(
                f'Camera: {camera}\n'
                "   The true gain factor could not be determined... \n"
                "   Use default value: 1.292\n"
                f"   Readout mode was {e}\n",
                indent=1,
                style_name='WARNING'
            )
            gain = 1.292

    else:
        #   Default: modern CMOS camera assumption
        terminal_output.print_to_terminal(
            "Camera not recognized. Assuming a modern CMOS camera ... ",
            indent=1,
            style_name='WARNING'
        )
        read_noise = 7.
        gain = 1.

    #   Dark current
    try:
        dark_current_fit_parameters = dark_current[camera]
        spline = interpolate.BSpline(
            dark_current_fit_parameters['t'],
            dark_current_fit_parameters['c'],
            dark_current_fit_parameters['k'],
            extrapolate=True,
        )
        dark_rate = spline(temperature)
    except Exception as e:
        terminal_output.print_to_terminal(
            f'Camera: {camera}\n'
            "   The dark current could not be determined... \n"
            "   Use default value: 0.002 e/s"
            f"   {e}\n",
            indent=1,
            style_name='WARNING'
        )
        if camera == 'QHY600M':
            dark_rate = 0.002
        elif camera == 'QHY600M':
            dark_rate = 0.0005
        elif camera == 'SBIG STF-8300 CCD Camera':
            dark_rate = 0.02
        #   Default
        else:
            dark_rate = 0.003

    #   Chip dimensions
    d, h = chip_dimensions(camera)

    return read_noise, gain, dark_rate, d, h


def get_chip_dimensions(instrument):
    """
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
    """
    info_camera = chip_dimensions(instrument)
    return info_camera[3], info_camera[4]


###
#   Internal gain vs. system gain -> QHY data
#
gain_qhy = {
    'QHY600M': {
        'Photography Mode': {
            't': [0., 0., 4.96183206, 10.,
                  14.96183206, 20., 24.96183206, 25.95419847,
                  30., 34.96183206, 40., 44.96183206,
                  50., 54.96183206, 55.95419847, 60.,
                  64.96183206, 70., 74.96183206, 80.,
                  84.96183206, 90., 94.96183206, 100., 100.],
            'c': [1.29774436, 1.1387218, 0.98308271, 0.83082707, 0.68195489,
                  0.52969925, 0.4112782, 0.36390977, 0.32330827, 0.27255639,
                  0.22180451, 0.17443609, 0.12368421, 0.11015038, 0.07293233,
                  0.02218045, 0.01541353, 0.01203008, 0.00864662, 0.00526316,
                  0.00864662, 0.00526316, 0.00526316, 0., 0.],
            'k': 1,
        },
        'High Gain Mode': {
            't': [0., 0., 4.96183206, 10.,
                  14.96183206, 20., 24.96183206, 25.95419847,
                  30., 34.96183206, 40., 44.96183206,
                  50., 54.96183206, 55.95419847, 60.,
                  64.96183206, 70., 74.96183206, 80.,
                  85.03816794, 90., 94.96183206, 100., 100.],
            'c': [0.77330827, 0.73609023, 0.69887218, 0.66165414, 0.61766917,
                  0.58721805, 0.57706767, 0.54661654, 0.50601504, 0.47218045,
                  0.43496241, 0.3943609, 0.3537594, 0.33007519, 0.30300752,
                  0.26578947, 0.22518797, 0.19135338, 0.15413534, 0.11691729,
                  0.08308271, 0.04586466, 0.00864662, 0., 0.],
            'k': 1,
        },
        'Extend Fullwell': {
            't': [0., 0., 4.96183206, 10.,
                  14.96183206, 20., 24.96183206, 25.95419847,
                  30., 35.03816794, 40., 44.96183206,
                  50., 54.96183206, 55.95419847, 60.,
                  64.96183206, 70., 74.96183206, 80.,
                  85.03816794, 90., 94.96183206, 100., 100.],
            'c': [1.2943609, 1.23007519, 1.17255639, 1.10827068, 1.05413534,
                  0.98308271, 0.96616541, 0.92556391, 0.8612782, 0.80037594,
                  0.7462406, 0.68195489, 0.62443609, 0.6075188, 0.56015038,
                  0.50263158, 0.43834586, 0.37744361, 0.31654135, 0.2556391,
                  0.19473684, 0.13383459, 0.06954887, 0., 0.],
            'k': 1,
        },
        'Extend Fullwell 2CMS': {
            't': [0., 0., 4.96183206, 10.,
                  14.96183206, 19.92366412, 24.96183206, 25.95419847,
                  30., 34.96183206, 40., 44.96183206,
                  50., 54.96183206, 55.95419847, 59.92366412,
                  64.96183206, 69.92366412, 74.96183206, 80.,
                  85.03816794, 89.92366412, 95.03816794, 100., 100.],
            'c': [1.25714286, 1.19962406, 1.15902256, 1.11503759, 1.05413534,
                  0.99323308, 0.9593985, 0.91879699, 0.84097744, 0.76654135,
                  0.70902256, 0.67180451, 0.62443609, 0.60075188, 0.55,
                  0.47894737, 0.44511278, 0.36390977, 0.31315789, 0.25225564,
                  0.1981203, 0.13383459, 0.07631579, 0., 0.],
            'k': 1,
        },
    },
    'QHY268M': {
        'Photography Mode': {
            't': [0.000, 5.251, 10.046, 15.068, 20.091, 24.886,
                  25.799, 26.941, 28.082, 28.995, 29.909, 34.932,
                  39.954, 44.977, 50.000, 55.023, 60.046, 64.840,
                  69.863, 74.886, 79.909, 299.315],
            'c': [1.542, 1.331, 1.150, 1.019, 0.858, 0.627,
                  0.523, 0.519, 0.504, 0.500, 0.492, 0.427,
                  0.365, 0.300, 0.227, 0.162, 0.096, 0.031,
                  0.015, 0.012, 0.008, 0.002],
            'k': 1,
        },
        'Photography Mode 2CMS': {
            't': [0.000, 5.023, 10.046, 15.068, 20.091, 25.114,
                  26.027, 26.941, 28.082, 28.995, 29.909, 34.932,
                  39.954, 44.977, 50.000, 55.023, 60.046, 64.840,
                  69.863, 74.886, 79.909, 299.315],
            'c': [1.619, 1.400, 1.192, 1.046, 0.892, 0.654,
                  0.535, 0.519, 0.504, 0.500, 0.492, 0.427,
                  0.365, 0.300, 0.227, 0.162, 0.096, 0.031,
                  0.015, 0.012, 0.008, 0.002],
            'k': 1,
        },
        'High Gain Mode': {
            't': [0.000, 5.023, 10.046, 15.068, 20.091, 25.114,
                  29.909, 34.932, 39.954, 44.977, 50.000, 55.023,
                  55.936, 56.849, 57.991, 58.904, 60.046, 64.840,
                  69.863, 74.886, 79.909, 84.932, 89.954, 94.977,
                  99.772, 289.269],
            'c': [1.012, 0.965, 0.904, 0.850, 0.788, 0.738,
                  0.700, 0.662, 0.619, 0.558, 0.504, 0.473,
                  0.435, 0.423, 0.412, 0.404, 0.392, 0.350,
                  0.304, 0.254, 0.204, 0.158, 0.108, 0.058,
                  0.008, 0.002],
            'k': 1,
        },
        'High Gain Mode 2CMS': {
            't': [0.000, 5.023, 10.046, 15.068, 20.091, 25.114,
                  29.909, 34.932, 39.954, 44.977, 50.000, 55.023,
                  55.936, 56.849, 57.991, 58.904, 60.046, 64.840,
                  69.863, 74.886, 79.909, 84.932, 89.954, 94.977,
                  99.772, 289.269],
            'c': [1.012, 0.965, 0.904, 0.850, 0.788, 0.738,
                  0.700, 0.662, 0.619, 0.558, 0.504, 0.473,
                  0.435, 0.423, 0.412, 0.404, 0.392, 0.350,
                  0.304, 0.254, 0.204, 0.158, 0.108, 0.058,
                  0.008, 0.002],
            'k': 1,
        },
        'Extend Fullwell': {
            't': [0.000, 5.023, 10.046, 15.068, 20.091, 25.114, 29.909,
                  34.932, 39.954, 44.977, 50.000, 55.023, 59.817, 64.840,
                  69.863, 74.886, 79.909, 84.932, 89.726, 94.749, 99.772,
                  104.795, 109.817, 114.840, 119.635, 124.886, 129.909, 134.703,
                  139.726, 144.749, 149.772, 154.795, 159.817, 164.612, 169.635,
                  174.658, 299.315],
            'c': [1.573, 1.485, 1.400, 1.315, 1.235, 1.162, 1.088,
                  1.035, 0.988, 0.938, 0.865, 0.785, 0.685, 0.592,
                  0.542, 0.473, 0.388, 0.319, 0.246, 0.169, 0.092,
                  0.054, 0.038, 0.031, 0.023, 0.019, 0.019, 0.015,
                  0.015, 0.012, 0.012, 0.012, 0.012, 0.012, 0.008,
                  0.008, 0.004],
            'k': 1,
        },
        'Extend Fullwell 2CMS': {
            't': [0.000, 5.023, 10.046, 15.068, 20.091, 24.886, 29.909,
                  34.932, 39.954, 44.977, 50.000, 54.795, 59.817, 64.840,
                  69.863, 74.886, 79.909, 84.932, 89.726, 94.749, 99.772,
                  104.795, 109.817, 114.840, 119.635, 124.886, 129.909, 134.703,
                  139.726, 144.749, 149.772, 154.795, 159.817, 164.612, 169.635,
                  174.658, 299.315],
            'c': [1.635, 1.542, 1.438, 1.358, 1.269, 1.200, 1.127,
                  1.073, 1.019, 0.962, 0.892, 0.804, 0.704, 0.612,
                  0.562, 0.492, 0.396, 0.327, 0.246, 0.169, 0.092,
                  0.054, 0.038, 0.031, 0.023, 0.019, 0.019, 0.015,
                  0.015, 0.012, 0.012, 0.012, 0.012, 0.012, 0.008,
                  0.008, 0.004],
            'k': 1,
        },
    },
}


###
#   Readout noise vs. gain -> QHY data
#
read_noise_qhy = {
    'QHY600M': {
        'Photography Mode': {
            't': [0.000, 4.962, 9.924, 14.962, 20.000, 24.962, 25.954,
                  30.000, 34.962, 40.000, 44.962, 50.000, 54.962, 55.954,
                  60.000, 64.962, 70.000, 74.962, 80.000, 84.962, 90.000,
                  94.962, 100.000],
            'c': [7.856, 7.703, 7.466, 7.314, 7.161, 6.992, 2.754,
                  2.720, 2.653, 2.602, 2.534, 2.466, 2.449, 2.415,
                  2.314, 1.992, 2.008, 2.008, 2.025, 1.992, 1.992,
                  1.992, 1.975],
            'k': 1,
        },
        'High Gain Mode': {
            't': [0.000, 4.962, 10.000, 14.962, 20.000, 24.962, 25.954,
                  30.000, 34.962, 40.000, 44.962, 50.000, 54.962, 55.954,
                  60.000, 65.038, 70.000, 74.962, 80.000, 85.038, 90.000,
                  95.038, 100.000],
            'c': [3.703, 3.653, 3.703, 3.619, 3.585, 3.602, 3.551,
                  3.534, 3.551, 3.517, 3.517, 3.500, 3.331, 1.686,
                  1.669, 1.653, 1.619, 1.619, 1.551, 1.500, 1.483,
                  1.398, 1.110],
            'k': 1,
        },
        'Extend Fullwell': {
            't': [-0.076, 4.962, 9.924, 14.962, 20.000, 24.962, 25.954,
                  30.000, 34.962, 40.000, 44.962, 50.000, 54.962, 55.954,
                  60.000, 64.962, 70.000, 74.962, 80.000, 85.038, 90.000,
                  95.038, 100.000],
            'c': [7.907, 7.822, 7.805, 7.720, 7.737, 7.602, 7.568,
                  7.517, 7.415, 7.314, 7.364, 7.212, 7.144, 7.110,
                  7.025, 7.025, 6.975, 6.958, 6.839, 6.534, 6.449,
                  5.975, 5.432],
            'k': 1,
        },
        'Extend Fullwell 2CMS': {
            't': [0.000, 4.962, 10.000, 14.962, 20.000, 24.962, 25.954,
                  30.000, 34.962, 40.000, 44.962, 50.000, 54.962, 55.954,
                  60.000, 64.962, 70.000, 75.038, 80.000, 84.962, 90.000,
                  95.038, 100.000],
            'c': [5.941, 5.890, 5.958, 5.992, 5.958, 5.890, 5.856,
                  5.805, 5.636, 5.500, 5.483, 5.636, 5.703, 5.636,
                  5.534, 5.432, 5.754, 5.415, 5.331, 5.347, 5.229,
                  5.144, 4.975],
            'k': 1,
        },
    },
    'QHY268M': {
        'Photography Mode': {
            't': [0.000, 4.868, 9.908, 14.948, 19.989, 25.029, 25.716,
                  26.861, 28.007, 28.923, 29.840, 34.880, 39.920, 44.960,
                  50.000, 55.040, 60.080, 64.891, 69.931, 74.971, 80.011,
                  85.052, 90.092, 94.903, 99.943, 104.983, 110.023, 115.063,
                  120.103, 124.914, 129.954, 134.994, 140.034, 145.074, 150.115,
                  154.926, 159.966, 165.006, 170.046, 175.086, 180.126, 184.937,
                  189.977, 195.017, 200.057, 205.097, 210.137, 215.178, 219.989,
                  225.029, 230.069, 235.109, 240.149, 245.189, 250.229, 255.040,
                  260.080, 265.120, 270.160, 275.200, 280.241, 285.052, 290.092,
                  295.132, 300.172],
            'c': [7.723, 7.470, 7.271, 7.506, 7.524, 6.873, 3.873,
                  3.873, 3.892, 3.928, 3.964, 3.819, 3.837, 3.729,
                  3.639, 3.620, 3.458, 3.367, 2.934, 2.681, 2.536,
                  2.410, 2.337, 2.283, 2.229, 2.175, 2.139, 2.102,
                  2.066, 2.048, 2.030, 1.994, 1.976, 1.958, 1.922,
                  1.922, 1.886, 1.886, 1.849, 1.831, 1.813, 1.795,
                  1.777, 1.759, 1.723, 1.687, 1.669, 1.669, 1.633,
                  1.596, 1.560, 1.542, 1.506, 1.470, 1.416, 1.361,
                  1.343, 1.289, 1.217, 1.235, 1.127, 1.127, 1.018,
                  1.018, 0.892],
            'k': 1,
        },
        'Photography Mode 2CMS': {
            't': [0.000, 4.868, 9.908, 14.948, 19.989, 24.800, 25.945,
                  26.861, 28.007, 28.923, 30.069, 34.880, 40.149, 44.960,
                  50.000, 55.040, 60.080, 65.120, 69.931, 74.971, 80.011,
                  85.052, 90.092, 94.903, 99.943, 104.983, 110.023, 115.063,
                  120.103, 125.143, 130.183, 134.994, 140.034, 145.074, 150.115,
                  155.155, 159.966, 165.006, 170.046, 175.086, 180.126, 185.166,
                  190.206, 195.017, 200.057, 205.097, 210.137, 215.178, 220.218,
                  225.029, 230.069, 235.109, 240.149, 245.189, 250.000, 255.040,
                  260.080, 265.120, 270.160, 275.200, 280.011, 285.052, 290.092,
                  295.132, 300.172],
            'c': [5.753, 5.554, 5.373, 5.464, 5.554, 5.120, 2.193,
                  2.157, 2.139, 2.193, 2.193, 2.175, 2.175, 2.120,
                  2.048, 2.048, 1.904, 1.723, 1.705, 1.723, 1.723,
                  1.687, 1.687, 1.687, 1.687, 1.687, 1.669, 1.669,
                  1.651, 1.651, 1.651, 1.633, 1.614, 1.614, 1.596,
                  1.596, 1.578, 1.578, 1.560, 1.542, 1.542, 1.542,
                  1.524, 1.524, 1.506, 1.488, 1.452, 1.452, 1.416,
                  1.416, 1.380, 1.398, 1.325, 1.325, 1.253, 1.253,
                  1.271, 1.163, 1.163, 1.163, 1.018, 1.018, 1.018,
                  1.018, 0.765],
            'k': 1,
        },
        'High Gain Mode': {
            't': [0.057, 5.097, 10.137, 14.948, 19.989, 25.029, 30.069,
                  34.880, 39.920, 44.960, 50.000, 54.811, 55.956, 56.873,
                  58.018, 58.935, 60.080, 64.891, 69.931, 74.971, 80.011,
                  85.052, 89.863, 95.132, 99.943, 104.983, 110.023, 115.063,
                  120.103, 124.914, 129.954, 134.994, 140.034, 145.074, 150.115,
                  155.155, 159.966, 165.006, 170.046, 175.086, 180.126, 185.166,
                  190.206, 195.017, 200.057, 205.097, 209.908, 214.948, 220.218,
                  225.029, 230.069, 235.109, 240.149, 245.189, 250.229, 255.269,
                  260.309, 265.120, 270.160, 274.971, 280.011, 285.052, 290.092,
                  295.132, 300.172],
            'c': [3.693, 3.675, 3.620, 3.566, 3.530, 3.476, 3.548,
                  3.602, 3.584, 3.458, 3.386, 3.512, 1.705, 1.705,
                  1.687, 1.669, 1.687, 1.669, 1.687, 1.669, 1.633,
                  1.596, 1.578, 1.560, 1.199, 1.199, 1.199, 1.199,
                  1.199, 1.199, 1.181, 1.145, 1.145, 1.145, 1.108,
                  1.108, 1.127, 1.090, 1.108, 1.072, 1.072, 1.054,
                  1.054, 1.072, 1.018, 1.018, 1.036, 0.964, 0.964,
                  0.946, 0.964, 0.964, 0.910, 0.910, 0.910, 0.855,
                  0.855, 0.855, 0.747, 0.747, 0.747, 0.747, 0.747,
                  0.819, 0.765],
            'k': 1,
        },
        'High Gain Mode 2CMS': {
            't': [0.057, 5.097, 9.908, 14.948, 19.989, 25.029, 29.840,
                  34.880, 39.920, 44.960, 50.000, 55.040, 55.956, 57.102,
                  58.018, 58.935, 60.080, 65.120, 69.931, 74.971, 80.011,
                  85.052, 90.092, 94.903, 99.943, 104.983, 110.023, 115.063,
                  120.103, 125.143, 129.954, 134.994, 140.034, 145.074, 150.115,
                  155.155, 159.966, 165.006, 170.046, 175.086, 180.126, 185.166,
                  189.977, 195.017, 200.286, 205.097, 210.137, 215.178, 219.989,
                  225.029, 230.069, 235.109, 240.149, 244.960, 250.000, 255.040,
                  260.080, 265.120, 270.160, 274.971, 280.241, 285.052,
                  290.092],
            'c': [3.386, 3.349, 3.331, 3.277, 3.223, 3.205, 3.223,
                  3.277, 3.313, 3.223, 3.133, 3.205, 1.578, 1.560,
                  1.560, 1.560, 1.560, 1.560, 1.560, 1.542, 1.524,
                  1.488, 1.470, 1.434, 1.163, 1.163, 1.145, 1.145,
                  1.127, 1.127, 1.127, 1.127, 1.108, 1.108, 1.108,
                  1.090, 1.072, 1.072, 1.036, 1.054, 1.018, 1.018,
                  1.000, 1.000, 1.000, 0.946, 0.946, 0.964, 0.964,
                  0.855, 0.892, 0.892, 0.892, 0.892, 0.892, 0.729,
                  0.747, 0.747, 0.747, 0.747, 0.747, 0.747,
                  0.747],
            'k': 1,
        },
        'Extend Fullwell': {
            't': [0.000, 4.868, 9.908, 14.948, 19.989, 25.029, 29.840,
                  34.880, 39.920, 44.960, 50.000, 55.040, 59.851, 64.891,
                  69.931, 74.971, 80.011, 85.052, 90.092, 94.903, 99.943,
                  104.983, 110.023, 115.063, 120.103, 125.143, 129.954, 134.994,
                  140.034, 145.074, 150.115, 155.155, 159.966, 165.006, 170.046,
                  175.086, 180.126, 185.166, 190.206, 195.017, 200.057, 205.097,
                  210.137, 215.178, 220.218, 225.029, 230.069, 235.109, 240.149,
                  245.189, 250.000, 255.040, 260.309, 265.120, 270.160, 275.200,
                  280.241, 285.052, 290.092, 295.132, 300.172],
            'c': [7.416, 7.307, 7.181, 7.054, 6.982, 6.873, 6.801,
                  6.873, 7.018, 7.108, 7.090, 6.964, 6.675, 6.440,
                  6.729, 6.837, 6.404, 6.187, 6.169, 5.843, 5.410,
                  5.373, 5.392, 5.392, 5.355, 5.319, 5.247, 5.193,
                  5.139, 5.048, 5.048, 4.958, 4.886, 4.813, 4.759,
                  4.669, 4.578, 4.506, 4.434, 4.380, 4.289, 4.217,
                  4.163, 4.072, 4.000, 3.910, 3.855, 3.747, 3.675,
                  3.566, 3.476, 3.386, 3.241, 3.151, 3.060, 2.934,
                  2.825, 2.771, 2.627, 2.500, 2.392],
            'k': 1,
        },
        'Extend Fullwell 2CMS': {
            't': [0.057, 4.868, 9.908, 14.948, 19.989, 25.029, 30.069,
                  34.880, 39.920, 44.960, 50.000, 55.040, 60.080, 65.120,
                  69.931, 74.971, 80.011, 85.052, 90.092, 94.903, 100.172,
                  105.212, 110.023, 115.063, 120.103, 125.143, 130.183, 134.994,
                  140.034, 145.074, 150.115, 155.155, 159.966, 165.006, 170.046,
                  175.086, 179.897, 184.937, 190.206, 195.017, 200.057, 205.097,
                  210.137, 215.178, 219.989, 225.258, 230.069, 235.109, 240.149,
                  245.189, 250.229, 255.040, 260.309, 265.120, 270.160, 275.200,
                  280.011, 285.281, 290.092, 295.132, 300.172],
            'c': [5.843, 5.753, 5.608, 5.536, 5.446, 5.410, 5.373,
                  5.428, 5.482, 5.554, 5.554, 5.446, 5.229, 5.066,
                  5.319, 5.392, 5.030, 4.867, 4.849, 4.542, 4.307,
                  4.271, 4.253, 4.289, 4.271, 4.235, 4.199, 4.199,
                  4.145, 4.108, 4.090, 4.036, 4.018, 3.982, 3.946,
                  3.892, 3.855, 3.855, 3.801, 3.765, 3.675, 3.639,
                  3.602, 3.566, 3.512, 3.494, 3.422, 3.349, 3.277,
                  3.223, 3.205, 3.114, 3.024, 2.988, 2.880, 2.807,
                  2.753, 2.681, 2.518, 2.482, 2.355],
            'k': 1,
        },
    },
}


###
#   Dark current vs. temperature in C -> QHY data
#
dark_current = {
    'QHY600M': {
        't': [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        'c': [0.0022, 0.0032, 0.0046, 0.0068, 0.0105, 0.0357, 0.0675, 0.1231,
              0.2208],
        'k': 1,
    },
    'QHY268M': {
        't': [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        'c': [0.00053145, 0.00062832, 0.001309, 0.0018326, 0.0036652, 0.0059756,
              0.010472, 0.019111, 0.036913],
        'k': 1,
    },
    'SBIG STF-8300 CCD Camera': {
        't': [-15, -10, 0],
        'c': [0.02, 0.04, 0.18],
        'k': 1,
    },
}


###
#   Catalog specific definitions
#

#   Dictionary with Vizier catalog identifiers
vizier_dict = {
    'UCAC4': 'I/322A',
    'GSC2.3': 'I/305',
    'URAT1': 'I/329',
    'NOMAD': 'I/297',
    'HMUBV': 'II/168/ubvmeans',
    'GSPC2.4': 'II/272/gspc24',
    'APASS': 'II/336/apass9',
    'Swift/UVOT': 'II/339/uvotssc1',
    'XMM-OM': 'II/370/xmmom5s',
    'VRI-NCC': 'J/MNRAS/443/725/catalog',
    'USNO-B1.0': 'I/284/out',
    'Stetson_2019': 'J/MNRAS/485/3042/table4',
    'Pancino_2022': 'J/A+A/664/A109/table5',
    'APOP_Qi_2015': 'I/331/apop',
}

default_columns = {
    'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
    'columns': ["Bmag", "Vmag", "rmag", "imag"],
    'err_columns': ["e_Bmag", "e_Vmag", "e_rmag", "e_imag"],
}

default_ra_unit = u.deg

#   Catalog properties
catalog_properties_dict = {
    'I/329': default_columns | {'ra_unit': default_ra_unit},
    'I/322A': default_columns | {'ra_unit': default_ra_unit},
    'II/336/apass9': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["Bmag", "Vmag", "r'mag", "i'mag"],
        'err_columns': ["e_Bmag", "e_Vmag", "e_r'mag", "e_i'mag"],
        'ra_unit': default_ra_unit,
        'column_rename': [
            ("r_mag", "Rmag"),
            ("i_mag", "Imag"),
            ("e_r_mag", "e_Rmag"),
            ("e_i_mag", "e_Imag")
        ],
    },
    'I/297': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["Bmag", "Vmag", "Rmag"],
        'err_columns': [],
        'ra_unit': default_ra_unit,
    },
    'I/305': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["Umag", "Bmag", "Vmag"],
        'err_columns': ["e_Umag", "e_Bmag", "e_Vmag"],
        'ra_unit': default_ra_unit,
    },
    'II/168/ubvmeans': {
        'ra_dec_columns': ['_RA', '_DE'],
        'columns': ["Vmag", "B-V", "U-B"],
        'err_columns': ["e_Vmag", "e_B-V", "e_U-B"],
        'ra_unit': default_ra_unit,
        'magnitude_arithmetic': [
            ('Bmag', 'B-V', 'Vmag'),
            ('e_Bmag', 'e_B-V', 'e_Vmag'),
            ('Umag', 'U-B', 'Bmag'),
            ('e_Umag', 'e_U-B', 'e_Bmag')
        ],
    },
    'II/272/gspc24': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["Bmag", "Vmag", "Rmag"],
        'err_columns': ["e_Bmag", "e_Vmag", "e_Rmag"],
        'ra_unit': default_ra_unit,
    },
    'II/339/uvotssc1': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["U-AB", "B-AB", "V-AB"],
        'err_columns': [],
        'ra_unit': default_ra_unit,
        'column_rename': [
            ("U-AB", "Umag"),
            ("B-AB", "Bmag"),
            ("V-AB", "Vmag")
        ],
    },
    'II/370/xmmom5s': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["UmAB", "BmAB", "VmAB"],
        'err_columns': ["e_UmAB", "e_BmAB", "e_VmAB"],
        'ra_unit': default_ra_unit,
        'column_rename': [
            ("UmAB", "Umag"),
            ("BmAB", "Bmag"),
            ("VmAB", "Vmag"),
            ("e_UmAB", "e_Umag"),
            ("e_BmAB", "e_Bmag"),
            ("e_VmAB", "e_Vmag")
        ],
    },
    'J/MNRAS/443/725/catalog': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["Vmag", "Rmag", "Imag"],
        'err_columns': ["e_Vmag", "e_Rmag", "e_Imag"],
        'ra_unit': default_ra_unit,
    },
    'I/284/out': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["B1mag", "R1mag", "Imag"],
        'err_columns': [],
        'ra_unit': default_ra_unit,
        'column_rename': [
            ("B1mag", "Bmag"),
            ("R1mag", "Rmag")
        ],
    },
    'J/MNRAS/485/3042/table4': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["Umag", "Bmag", "Vmag", "Rmag", "Imag"],
        'err_columns': ["e_Umag", "e_Bmag", "e_Vmag", "e_Rmag", "e_Imag"],
        'ra_unit': u.hourangle,
    },
    'J/A+A/664/A109/table5': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["Umag", "Bmag", "Vmag", "Rmag", "Imag"],
        'err_columns': ["e_Umag", "e_Bmag", "e_Vmag", "e_Rmag", "e_Imag"],
        'ra_unit': default_ra_unit,
    },
    'I/331/apop': {
        'ra_dec_columns': ['RAICRS', 'DEICRS'],
        'columns': ["Bmag", "Vmag", "Rmag"],
        'err_columns': [],
        'ra_unit': default_ra_unit,
    }
}


###
#   Valid filter combinations to calculate magnitude transformation
#   dict -> key = filter, value = list(first color, second color)
#
valid_filter_combinations_for_transformation = [
    ['U', 'V'],
    ['B', 'V'],
    ['V', 'R'],
    ['V', 'I'],
]


###
#   Filter denomination vs. filter systems
#
filter_systems = {
    'U': 'bessell',
    'B': 'bessell',
    'V': 'bessell',
    'R': 'bessell',
    'I': 'bessell',
    'u`': 'sdss',
    'g`': 'sdss',
    'r`': 'sdss',
    'i`': 'sdss',
    'z-s`': 'sdss',
    'y`': 'sdss',
    'Halpha`': 'narrow_band',
    'SII': 'narrow_band',
    'OIII': 'narrow_band',
    # 'Blue':
    # 'Green':
    # 'Red':
}


###
#   Filter effective wavelength
#
filter_effective_wavelength = {
    'U': 3659.88,
    'B': 4380.74,
    'V': 5445.43,
    'R': 6411.47,
    'I': 7982.09,
}


def fitzpatrick_extinction_curve(r):
    """
        Fitzpatrick's extinction curve - A(lambda)/E(B-V) vs. 1/lambda [1/mym]
        This version is not valid for wavelengths below 2600AA.

        Parameters
        ----------
        r               : `float`
        Ration between absolute and relative extinction in the V band.

        Returns
        -------
        cubic_spline    : `scipy.interpolate.CubicSpline`
        Cubic spline to the Fitzpatrick anchor points

    """
    #   Spline anchor points (Fitzpatrick 1999)
    #   x = 1/lambda
    x = [0., 0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846]

    #   y = A(lambda)/E(B-V)
    #   Coefficients for UV anchor points (fitting function: Fitzpatrick & Massa 1990)
    c2 = -0.824 + 4.717 / r
    c1 = 2.030 - 3.007 * c2
    y = [
        0.,
        0.265 * r/3.1,
        0.829 * r/3.1,
        -0.426 + 1.0044 * r,
        -0.050 + 1.0016 * r,
        0.701 + 1.0016 * r,
        # -1.208 + 1.0032 * r - 0.00033 * r * r,
        1.208 + 1.0032 * r - 0.00033 * r * r,
        r + c1 + c2 * 3.704 + 0.6492006,
        r + c1 + c2 * 3.846 + 0.8752775,
    ]

    return interpolate.CubicSpline(x, y)


###
#   Filter system conversion functions
#
def jordi_u(**kwargs):
    if all(filter_ in kwargs for filter_ in ['U', 'B', 'V', 'g']):
        U = kwargs.get("U")
        B = kwargs.get("B")
        V = kwargs.get("V")
        g = kwargs.get("g")

        return ufloat(0.750, 0.050) * (U - B) + ufloat(0.770, 0.070) * (B - V) \
            + ufloat(0.720, 0.040) + g
    return None


def jordi_g(**kwargs):
    if all(filter_ in kwargs for filter_ in ['B', 'V']):
        B = kwargs.get("B")
        V = kwargs.get("V")

        return ufloat(0.630, 0.002) * (B - V) - ufloat(0.124, 0.002) + V

    if all(filter_ in kwargs for filter_ in ['V', 'R', 'r']):
        V = kwargs.get("V")
        R = kwargs.get("R")
        r = kwargs.get("r")

        return ufloat(1.646, 0.008) * (V - R) - ufloat(0.139, 0.004) + r

    if all(filter_ in kwargs for filter_ in ['V', 'I', 'i']):
        V = kwargs.get("V")
        I = kwargs.get("I")
        i = kwargs.get("i")

        if V - I <= 1.8:
            return ufloat(1.481, 0.004) * (V - I) - ufloat(0.536, 0.004) + i
        else:
            return ufloat(0.83, 0.01) * (V - I) + ufloat(0.60, 0.03) + i

    return None


def jordi_r(**kwargs):
    if all(filter_ in kwargs for filter_ in ['V', 'R']):
        V = kwargs.get("V")
        R = kwargs.get("R")

        if V - R <= 0.93:
            return ufloat(0.267, 0.005) * (V - R) + ufloat(0.088, 0.003) + R
        else:
            return ufloat(0.77, 0.04) * (V - R) - ufloat(0.37, 0.04)

    if all(filter_ in kwargs for filter_ in ['V', 'R', 'g']):
        V = kwargs.get("V")
        R = kwargs.get("R")
        g = kwargs.get("g")

        return g - ufloat(1.646, 0.008) * (V - R) + ufloat(0.139, 0.004)

    if all(filter_ in kwargs for filter_ in ['I', 'R', 'i']):
        I = kwargs.get("I")
        R = kwargs.get("R")
        i = kwargs.get("i")

        return ufloat(1.007, 0.005) * (R - I) - ufloat(0.236, 0.003) + i

    if all(filter_ in kwargs for filter_ in ['I', 'R', 'z']):
        I = kwargs.get("I")
        R = kwargs.get("R")
        z = kwargs.get("z")

        return ufloat(1.584, 0.008) * (R - I) - ufloat(0.386, 0.005) + z

    return None


def jordi_i(**kwargs):
    if all(filter_ in kwargs for filter_ in ['R', 'I']):
        R = kwargs.get("R")
        I = kwargs.get("I")

        return ufloat(0.247, 0.003) * (R - I) + ufloat(0.329, 0.002) + I

    if all(filter_ in kwargs for filter_ in ['V', 'I', 'g']):
        V = kwargs.get("V")
        I = kwargs.get("I")
        g = kwargs.get("g")

        if V - I <= 1.8:
            return g - ufloat(1.481, 0.004) * (V - I) + ufloat(0.536, 0.004)
        else:
            return g - ufloat(0.83, 0.01) * (V - I) - ufloat(0.60, 0.03)

    if all(filter_ in kwargs for filter_ in ['I', 'R', 'r']):
        I = kwargs.get("I")
        R = kwargs.get("R")
        r = kwargs.get("r")

        return r - ufloat(1.007, 0.005) * (R - I) + ufloat(0.236, 0.003)

    return None


def jordi_z(**kwargs):
    if all(filter_ in kwargs for filter_ in ['I', 'R', 'r']):
        I = kwargs.get("I")
        R = kwargs.get("R")
        r = kwargs.get("r")

        return r - ufloat(1.584, 0.008) * (R - I) + ufloat(0.386, 0.005)

    return None


###
#   Filter system conversions
#
filter_system_conversions = {
    'SDSS': {
        'Jordi_et_al_2005': {
            'g': jordi_g,
            'u': jordi_u,
            'r': jordi_r,
            'i': jordi_i,
            'z': jordi_z,
        }
    }
}

###
#   Magnitude calibration parameters
#   (Need to be ordered by date. Newest needs to be first.)
#
Tcs_qhy600m_20220420 = {
    'B': {
        'Filter 1': 'B',
        #   Tbbv
        'T_1': 0.085647,
        'T_1_err': 1.3742e-05,
        'k_1': -0.048222,
        'k_1_err': 7.1522e-06,
        'Filter 2': 'V',
        #   Tvbv
        'T_2': 0.0084589,
        'T_2_err': 9.7904e-06,
        'k_2': -0.010063,
        'k_2_err': 5.0955e-06,
        'type': 'air_mass',
        #   QHY600M
        'camera': ['QHY600M'],
    },
    'V': {
        'Filter 1': 'B',
        #   Tbbv
        'T_1': 0.085858,
        'T_1_err': 1.3649e-05,
        'k_1': -0.047997,
        'k_1_err': 7.0814e-06,
        'Filter 2': 'V',
        #   Tvbv
        'T_2': 0.008503,
        'T_2_err': 9.7294e-06,
        'k_2': -0.010016,
        'k_2_err': 5.0477e-06,
        'type': 'air_mass',
        #   QHY600M
        'camera': ['QHY600M'],
    },
}
Tcs_qhy600m_20080101 = {
    'B': {
        'Filter 1': 'B',
        #   Tbbv
        'T_1': -0.11545,
        'T_1_err': 0.020803,
        'k_1': -0.19031,
        'k_1_err': 0.0088399,
        'Filter 2': 'V',
        #   Tvbv
        'T_2': -0.32843,
        'T_2_err': 0.0080104,
        'k_2': -0.1143,
        'k_2_err': 0.0034039,
        'type': 'air_mass',
        #   QHY600M
        'camera': ['QHY600M'],
    },
    'V': {
        'Filter 1': 'B',
        #   Tbbv
        'T_1': -0.10083,
        'T_1_err': 0.020197,
        'k_1': -0.17973,
        'k_1_err': 0.0084819,
        'Filter 2': 'V',
        #   Tvbv
        'T_2': -0.32454,
        'T_2_err': 0.0075941,
        'k_2': -0.11125,
        'k_2_err': 0.0031892,
        'type': 'air_mass',
        #   QHY600M
        'camera': ['QHY600M'],
    },
}

Tcs = {
    '2022-04-20T00:00:00': {
        #   QHY600M
        # 'QHYCCD-Cameras-Capture':Tcs_qhy600m_20220420,
        # 'QHYCCD-Cameras2-Capture':Tcs_qhy600m_20220420,
        'QHY600M': Tcs_qhy600m_20220420,
    },
    '2008-01-01T00:00:00': {
        #   QHY600M
        # 'QHYCCD-Cameras-Capture': Tcs_qhy600m_20080101,
        # 'QHYCCD-Cameras2-Capture':Tcs_qhy600m_20080101,
        'QHY600M': Tcs_qhy600m_20080101,
    },
}


def get_transformation_calibration_values(observation_jd):
    """
        Get the Tcs calibration values for the provided JD

        Parameters
        ----------
        observation_jd   : `float`
            JD of the observation

        Returns
        -------
        Tcs             : `dictionary`
            Tcs calibration factors
    """
    if observation_jd is not None:
        for key in Tcs.keys():
            t = Time(key, format='isot', scale='utc')
            if observation_jd >= t.jd:
                return Tcs[key]

    return None
