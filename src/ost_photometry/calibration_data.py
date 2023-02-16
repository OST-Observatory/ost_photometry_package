############################################################################
####                            Libraries                               ####
############################################################################

from astropy.time import Time

from . import terminal_output

############################################################################
####                        Routines & definitions                      ####
############################################################################

def get_img_types(software):
    '''
        Get image type designator: The terms that are used to identify bias,
        darks, flats, etc. in the image Headers.

        Parameters
        ----------
        software            : `string`
            String that characterizes the imaging software.

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

    if 'MaxIm DL' in software:
        return default_img_type
    else:
        return default_img_type


def camera_info(camera):
    '''
        Camera specific parameters

        Parameters
        ----------
        camera          : `string`
            Camera or camera type used to obtain the data
    '''
    #   STF8300
    if camera in ['SBIG STF-8300 CCD Camera']:
        readnoise = 9.3
        gain      = None
        dark_rate = {0:0.18, -10:0.04, -15.8:0.02}
        satlevel  = 65535.
        d = 17.96
        h = 13.52
    #   QHY600M:
    elif camera in [
        'QHYCCD-Cameras-Capture',
        'QHYCCD-Cameras2-Capture',
        'QHY600M',
        ]:
        readnoise = 7.904
        gain      = 1.292
        dark_rate = {-20:0.0022, -10:0.0046}
        satlevel  = 65535.
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
        satlevel  = 65535.
        d = 32.00
        h = 24.00

    return readnoise, gain, dark_rate, satlevel, d, h


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
        'camera':[
            'QHY600M',
            'QHYCCD-Cameras-Capture',
            'QHYCCD-Cameras2-Capture',
            ],
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
        'camera':[
            'QHY600M',
            'QHYCCD-Cameras-Capture',
            'QHYCCD-Cameras2-Capture',
            ],
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
        'camera':[
            'QHY600M',
            'QHYCCD-Cameras-Capture',
            'QHYCCD-Cameras2-Capture',
            ],
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
        'camera':[
            'QHY600M',
            'QHYCCD-Cameras-Capture',
            'QHYCCD-Cameras2-Capture',
            ],
    },
}

Tcs = {
    '2022-04-20T00:00:00':{
        #   QHY600M
        'QHYCCD-Cameras-Capture':Tcs_qhy600m_20220420,
        'QHYCCD-Cameras2-Capture':Tcs_qhy600m_20220420,
        'QHY600M':Tcs_qhy600m_20220420,
    },
    '2008-01-01T00:00:00':{
        #   QHY600M
        'QHYCCD-Cameras-Capture': Tcs_qhy600m_20080101,
        'QHYCCD-Cameras2-Capture':Tcs_qhy600m_20080101,
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
