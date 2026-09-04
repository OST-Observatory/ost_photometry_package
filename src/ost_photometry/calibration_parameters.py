############################################################################
#                               Libraries                                  #
############################################################################

import astropy.units as u
import scipy.interpolate as interpolate
from astropy import uncertainty as unc
from astropy.time import Time

from . import terminal_output
from .camera_specs import camera_defaults, chip_size, interpolate_camera_curve

############################################################################
#                           Routines & definitions                         #
############################################################################


def get_image_types() -> dict[str, list[str]]:
    """
        Get image type designator: The terms that are used to identify bias,
        darks, flats, etc. in the image Headers.

        Returns
        -------
                            : `dictionary` of `string`
            Dictionary with the image type.
    """
    #   Define default image types
    default_img_type: dict[str, list[str]] = {
        'bias': ['Bias Frame', 'Bias', 'BIAS'],
        'dark': ['Dark Frame', 'Dark', 'DARK'],
        'flat': ['Flat Field', 'Flat', 'FLAT', 'Flat Frame'],
        'light': ['Light Frame', 'Light', 'LIGHT'],
    }

    return default_img_type


def chip_dimensions(camera: str) -> tuple[float, float]:
    """
    Parameters
    ----------
    camera
        The camera or camera type used to obtain the data

    Returns
    -------
    d
        Width in mm

    h
        Height in mm
    """
    size = chip_size(camera)
    if size is not None:
        return size

    terminal_output.print_to_terminal(
        "Camera not recognized. Assuming a modern CMOS camera: "
        "chip length = 32.0mm & chip height = 24.0 mm",
        indent=1,
        style_name='WARNING'
    )
    return 32.00, 24.00


def camera_info(
        camera: str, readout_mode: str, temperature: float,
        gain_setting: int | float | None = None
    ) -> tuple[float, float, float, int, int]:
    """
    Camera specific parameters

    Parameters
    ----------
    camera
        The camera or camera type used to obtain the data

    readout_mode
        Mode used to read out the data from the camera chip.

    temperature
        The temperature of the camera chip

    gain_setting
        Gain used in the camera setting for cameras such as the QHYs.
        This is not the system gain, but it can be calculated from this
        value. See below.
        Default is ``None``.

    Returns
    -------
    read_noise
        Read noise

    gain
        The gain factor

    dark_rate
        Dark current

    d
        Width in mm

    h
        Height in mm
    """
    defaults = camera_defaults(camera)

    read_noise = interpolate_camera_curve(
        camera,
        "readout_noise",
        gain_setting,
        readout_mode,
        for_camera_info=True,
    )
    if read_noise is None:
        read_noise = defaults.get("readout_noise")
    if read_noise is None:
        terminal_output.print_to_terminal(
            f'Camera: {camera}\n'
            "   The read noise could not be determined... \n"
            "   Use default value: 7.904\n"
            f"   Readout mode was {readout_mode}",
            indent=1,
            style_name='WARNING'
        )
        read_noise = 7.904

    gain = interpolate_camera_curve(
        camera,
        "system_gain",
        gain_setting,
        readout_mode,
        for_camera_info=True,
    )
    if gain is None and "system_gain" in defaults:
        # Explicit catalog default, including ``None`` (STF-8300 has no CMOS GAIN).
        gain = defaults["system_gain"]
    elif gain is None:
        if camera in ['QHY600M', 'QHY268M']:
            terminal_output.print_to_terminal(
                f'Camera: {camera}\n'
                "   The true gain factor could not be determined... \n"
                "   Use default value: 1.292\n"
                f"   Readout mode was {readout_mode}",
                indent=1,
                style_name='WARNING'
            )
            gain = 1.292
        else:
            terminal_output.print_to_terminal(
                "Camera not recognized. Assuming a modern CMOS camera: "
                "read noise kept; gain = 1.",
                indent=1,
                style_name='WARNING'
            )
            gain = 1.

    dark_rate = interpolate_camera_curve(
        camera,
        "dark_current",
        temperature,
        None,
        for_camera_info=True,
    )
    if dark_rate is None:
        dark_rate = defaults.get("dark_current")
    if dark_rate is None:
        terminal_output.print_to_terminal(
            f'Camera: {camera}\n'
            "   The dark current could not be determined... \n"
            "   Use default value: 0.002 e/s",
            indent=1,
            style_name='WARNING'
        )
        if camera == 'QHY600M':
            dark_rate = 0.002
        elif camera == 'QHY268M':
            dark_rate = 0.0005
        elif camera in ['SBIG STF-8300 CCD Camera']:
            dark_rate = 0.02
        else:
            terminal_output.print_to_terminal(
                "Camera not recognized. Assuming a modern CMOS camera: "
                "dark rate = 0.003 e-/s",
                indent=1,
                style_name='WARNING'
            )
            dark_rate = 0.003

    d, h = chip_dimensions(camera)

    return read_noise, gain, dark_rate, d, h


def get_chip_dimensions(instrument: str) -> tuple[float, float]:
    """
    Return camera chip dimensions in mm

    Parameters
    ----------
    instrument
        Camera type or came driver name

    Returns
    -------
        d
            Length of the camera chip

        h
            Height of the camera chip
    """
    return chip_dimensions(instrument)


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
    'SDSS_Release_16': 'V/147/sdss12',
}

default_columns = {
    'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
    'columns': ["Bmag", "Vmag", "rmag", "imag"],
    'err_columns': ["e_Bmag", "e_Vmag", "e_rmag", "e_imag"],
}

default_ra_unit = u.deg

#   Catalog properties
catalog_properties_dict = {
    'V/147/sdss12': {
        'ra_unit': default_ra_unit,
        'ra_dec_columns': ['RA_ICRS', 'DE_ICRS'],
        'columns': ['upmag', 'gpmag', 'rpmag', 'ipmag', 'zpmag'],
        'err_columns': ['e_upmag', 'e_gpmag', 'e_rpmag', 'e_ipmag', 'e_zpmag'],
        'column_rename': [
            ("upmag", "umag"),
            ("gpmag", "gmag"),
            ("rpmag", "rmag"),
            ("ipmag", "imag"),
            ("zpmag", "zmag"),
            ("e_upmag", "e_umag"),
            ("e_gpmag", "e_gmag"),
            ("e_rpmag", "e_rmag"),
            ("e_ipmag", "e_imag"),
            ("e_zpmag", "e_zmag"),
        ],
    },
    'B/vsx/vsx': {
        'ra_unit': default_ra_unit,
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': [],
        'err_columns': [],
    },
    'I/329': default_columns | {'ra_unit': default_ra_unit},
    'I/322A': default_columns | {'ra_unit': default_ra_unit},
    'II/336/apass9': {
        'ra_dec_columns': ['RAJ2000', 'DEJ2000'],
        'columns': ["Bmag", "Vmag", "r'mag", "i'mag"],
        'err_columns': ["e_Bmag", "e_Vmag", "e_r'mag", "e_i'mag"],
        'ra_unit': default_ra_unit,
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


def fitzpatrick_extinction_curve(r: float) -> interpolate.CubicSpline:
    """
    Fitzpatrick's extinction curve - A(lambda)/E(B-V) vs. 1/lambda [1/mym]
    This version is not valid for wavelengths below 2600AA.

    Parameters
    ----------
    r
        Ration between absolute and relative extinction in the V band.

    Returns
    -------
    cubic_spline
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
        1.208 + 1.0032 * r - 0.00033 * r * r,
        r + c1 + c2 * 3.704 + 0.6492006,
        r + c1 + c2 * 3.846 + 0.8752775,
    ]

    return interpolate.CubicSpline(x, y)


###
#   Filter system conversion functions
#
def jordi_u(**kwargs) -> unc.core.NdarrayDistribution | None:
    distribution_samples = kwargs.get("distribution_samples")

    if all(filter_ in kwargs for filter_ in ['U', 'B', 'V', 'g']):
        U = kwargs.get("U")
        B = kwargs.get("B")
        V = kwargs.get("V")
        g = kwargs.get("g")

        conversation_constant_1 = unc.normal(
            0.750,
            std=0.050,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.770,
            std=0.070,
            n_samples=distribution_samples,
        )
        conversation_constant_3 = unc.normal(
            0.720 * u.mag,
            std=0.040 * u.mag,
            n_samples=distribution_samples,
        )
        return conversation_constant_1 * (U - B) + conversation_constant_2 * (B - V) \
            + conversation_constant_3 + g
    return None


def jordi_g(**kwargs) -> unc.core.NdarrayDistribution | None:
    distribution_samples = kwargs.get("distribution_samples")

    if all(filter_ in kwargs for filter_ in ['B', 'V']):
        B = kwargs.get("B")
        V = kwargs.get("V")

        conversation_constant_1 = unc.normal(
            0.630,
            std=0.002,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.124 * u.mag,
            std=0.002 * u.mag,
            n_samples=distribution_samples,
        )
        return conversation_constant_1 * (B - V) - conversation_constant_2 + V

    if all(filter_ in kwargs for filter_ in ['V', 'R', 'r']):
        V = kwargs.get("V")
        R = kwargs.get("R")
        r = kwargs.get("r")

        conversation_constant_1 = unc.normal(
            1.646,
            std=0.008,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.139 * u.mag,
            std=0.004 * u.mag,
            n_samples=distribution_samples,
        )
        return conversation_constant_1 * (V - R) - conversation_constant_2 + r

    if all(filter_ in kwargs for filter_ in ['V', 'I', 'i']):
        V = kwargs.get("V")
        I = kwargs.get("I")
        i = kwargs.get("i")

        conversation_constant_1_a = unc.normal(
            1.481,
            std=0.004,
            n_samples=distribution_samples,
        )
        conversation_constant_2_a = unc.normal(
            0.536 * u.mag,
            std=0.004 * u.mag,
            n_samples=distribution_samples,
        )
        conversation_constant_1_b = unc.normal(
            0.83,
            std=0.01,
            n_samples=distribution_samples,
        )
        conversation_constant_2_b = unc.normal(
            0.6 * u.mag,
            std=0.03 * u.mag,
            n_samples=distribution_samples,
        )
        if V - I <= 1.8:
            return conversation_constant_1_a * (V - I) - conversation_constant_2_a + i
        else:
            return conversation_constant_1_b * (V - I) + conversation_constant_2_b + i

    return None


def jordi_r(**kwargs) -> unc.core.NdarrayDistribution | None:
    distribution_samples = kwargs.get("distribution_samples")

    if all(filter_ in kwargs for filter_ in ['V', 'R']):
        V = kwargs.get("V")
        R = kwargs.get("R")

        conversation_constant_1_a = unc.normal(
            0.267,
            std=0.005,
            n_samples=distribution_samples,
        )
        conversation_constant_2_a = unc.normal(
            0.088 * u.mag,
            std=0.003 * u.mag,
            n_samples=distribution_samples,
        )
        conversation_constant_1_b = unc.normal(
            0.77,
            std=0.04,
            n_samples=distribution_samples,
        )
        conversation_constant_2_b = unc.normal(
            0.37 * u.mag,
            std=0.04 * u.mag,
            n_samples=distribution_samples,
        )
        if V - R <= 0.93:
            return conversation_constant_1_a * (V - R) + conversation_constant_2_a + R
        else:
            return conversation_constant_1_b * (V - R) - conversation_constant_2_b

    if all(filter_ in kwargs for filter_ in ['V', 'R', 'g']):
        V = kwargs.get("V")
        R = kwargs.get("R")
        g = kwargs.get("g")

        conversation_constant_1 = unc.normal(
            1.646,
            std=0.008,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.139 * u.mag,
            std=0.004 * u.mag,
            n_samples=distribution_samples,
        )
        return g - conversation_constant_1 * (V - R) + conversation_constant_2

    if all(filter_ in kwargs for filter_ in ['I', 'R', 'i']):
        I = kwargs.get("I")
        R = kwargs.get("R")
        i = kwargs.get("i")

        conversation_constant_1 = unc.normal(
            1.007,
            std=0.005,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.236 * u.mag,
            std=0.003 * u.mag,
            n_samples=distribution_samples,
        )
        return conversation_constant_1 * (R - I) - conversation_constant_2 + i

    if all(filter_ in kwargs for filter_ in ['I', 'R', 'z']):
        I = kwargs.get("I")
        R = kwargs.get("R")
        z = kwargs.get("z")

        conversation_constant_1 = unc.normal(
            1.584,
            std=0.008,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.386 * u.mag,
            std=0.005 * u.mag,
            n_samples=distribution_samples,
        )
        return conversation_constant_1 * (R - I) - conversation_constant_2 + z

    return None


def jordi_i(**kwargs) -> unc.core.NdarrayDistribution | None:
    distribution_samples = kwargs.get("distribution_samples")

    if all(filter_ in kwargs for filter_ in ['R', 'I']):
        R = kwargs.get("R")
        I = kwargs.get("I")

        conversation_constant_1 = unc.normal(
            0.247,
            std=0.003,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.329 * u.mag,
            std=0.002 * u.mag,
            n_samples=distribution_samples,
        )
        return conversation_constant_1 * (R - I) + conversation_constant_2 + I

    if all(filter_ in kwargs for filter_ in ['V', 'I', 'g']):
        V = kwargs.get("V")
        I = kwargs.get("I")
        g = kwargs.get("g")

        conversation_constant_1_a = unc.normal(
            1.481,
            std=0.004,
            n_samples=distribution_samples,
        )
        conversation_constant_2_a = unc.normal(
            0.536 * u.mag,
            std=0.004 * u.mag,
            n_samples=distribution_samples,
        )
        conversation_constant_1_b = unc.normal(
            0.83,
            std=0.01,
            n_samples=distribution_samples,
        )
        conversation_constant_2_b = unc.normal(
            0.60 * u.mag,
            std=0.03 * u.mag,
            n_samples=distribution_samples,
        )
        if V - I <= 1.8:
            return g - conversation_constant_1_a * (V - I) + conversation_constant_2_a
        else:
            return g - conversation_constant_1_b * (V - I) - conversation_constant_2_b

    if all(filter_ in kwargs for filter_ in ['I', 'R', 'r']):
        I = kwargs.get("I")
        R = kwargs.get("R")
        r = kwargs.get("r")

        conversation_constant_1 = unc.normal(
            1.007,
            std=0.005,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.236 * u.mag,
            std=0.003 * u.mag,
            n_samples=distribution_samples,
        )
        return r - conversation_constant_1 * (R - I) + conversation_constant_2

    return None


def jordi_z(**kwargs) -> unc.core.NdarrayDistribution | None:
    distribution_samples = kwargs.get("distribution_samples")

    if all(filter_ in kwargs for filter_ in ['I', 'R', 'r']):
        I = kwargs.get("I")
        R = kwargs.get("R")
        r = kwargs.get("r")

        conversation_constant_1 = unc.normal(
            1.584,
            std=0.008,
            n_samples=distribution_samples,
        )
        conversation_constant_2 = unc.normal(
            0.386 * u.mag,
            std=0.005 * u.mag,
            n_samples=distribution_samples,
        )
        return r - conversation_constant_1 * (R - I) + conversation_constant_2

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


def get_transformation_calibration_values(
        observation_jd: float
        ) -> dict[str, dict[str, dict[str, float | str | list[str]]]] | None:
    """
    Get the Magnitude transformation calibration values for the provided JD

    Parameters
    ----------
    observation_jd
        JD of the observation

    Returns
    -------
    Tcs
        Magnitude transformation calibration factors
    """
    if observation_jd is not None:
        for key in Tcs.keys():
            t = Time(key, format='isot', scale='utc')
            if observation_jd >= t.jd:
                return Tcs[key]

    return None
