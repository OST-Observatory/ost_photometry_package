############################################################################
####                            Libraries                               ####
############################################################################

import os

import time

import random
import string

import subprocess

import pandas as pd

import json
import yaml

import numpy as np

from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy import wcs

from regions import PixCoord, RectangleSkyRegion, RectanglePixelRegion

from pathlib import Path

from . import checks, terminal_output, style, calibration_data

############################################################################
####                        Routines & definitions                      ####
############################################################################

class image:
    '''
        Image object used to store and transport some data
    '''

    def __init__(self, pd, filt, name, path, outdir):
        self.pd       = pd
        self.filt     = filt
        self.objname  = name
        if isinstance(path, Path):
            self.filename = path.name
            self.path     = path
        else:
            self.filename = path.split('/')[-1]
            self.path     = Path(path)
        if isinstance(outdir, Path):
            self.outpath = outdir
        else:
            self.outpath = Path(outdir)

    #   Read image
    def read_image(self):
        return CCDData.read(self.path)

    #   Get header
    def get_header(self):
        return CCDData.read(self.path).meta

    #   Get data
    def get_data(self):
        return CCDData.read(self.path).data


def cal_fov(image, indent=2, verbose=True):
    '''
        Calculate field of view, pixel scale, etc. ...

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        indent          : `integer`, optional
            Indentation for the console output
            Default is ``2``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    '''
    if verbose:
        terminal_output.print_terminal(
            indent=indent,
            string="Calculating FOV, PIXEL scale, etc. ... ",
            )

    #   Get header
    header = image.get_header()

    #   Read focal length - set default to 3454. mm
    f = header.get('FOCALLEN', 3454.)

    #   Read ra and dec of image center
    ra  = header.get('OBJCTRA', '00 00 00')
    dec = header.get('OBJCTDEC', '+00 00 00')

    #   Convert ra & dec to degrees
    coord_fov = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")

    #   Set instrument
    instrument = header.get('INSTRUME', '')

    #   Number of pixels
    npixX = header.get('NAXIS1', 0)
    npixY = header.get('NAXIS2', 0)

    if npixX == 0:
        raise ValueError(
            f"{style.bcolors.FAIL}\nException in cal_fov(): X dimension of "
            f"the image is 0 {style.bcolors.ENDC}"
            )
    if npixY == 0:
        raise ValueError(
            f"{style.bcolors.FAIL}\nException in cal_fov(): Y dimension of "
            f"the image is 0 {style.bcolors.ENDC}"
            )

    #   Calculate chip size in mm
    if 'XPIXSZ' in header:
        pixwidth = header['XPIXSZ']
        d = npixX*pixwidth/1000
        h = npixY*pixwidth/1000
    else:
        d, h = calibration_data.get_chip_dimensions(instrument)

    #   Calculate field of view
    fov_x = 2 * np.arctan(d/2/f)
    fov_y = 2 * np.arctan(h/2/f)

    #   Convert to arc min
    fov = fov_x*360./2./np.pi*60.
    fov_y = fov_y*360./2./np.pi*60.

    #   Calculate pixel scale
    pixscale = fov*60/npixX

    #   Create RectangleSkyRegion that covers the field of view
    #region_sky = RectangleSkyRegion(
        #center=coord_fov,
        #width=fov_x * u.rad,
        #height=fov_y * u.rad,
        #angle=0 * u.deg,
        #)
    #   Create RectanglePixelRegion that covers the field of view
    region_pix = RectanglePixelRegion(
        center=PixCoord(x=int(npixX/2), y=int(npixY/2)),
        width=npixX,
        height=npixY,
        )

    #   Add to image class
    image.coord      = coord_fov
    image.fov        = fov
    image.fov_y      = fov_y
    image.instrument = instrument
    image.pixscale   = pixscale
    #image.region_sky  = region_sky
    image.region_pix  = region_pix

    #   Add JD (observation time) and air mass from Header to image class
    jd = header.get('JD', None)
    if jd is None:
        obs_time = header.get('DATE-OBS', None)
        if not obs_time:
            raise ValueError(
                f"{style.bcolors.FAIL} \tERROR: No information about the "
                "observation time was found in the header"
                f"{style.bcolors.ENDC}"
                )
        jd = Time(obs_time, format='fits').jd

    image.jd = jd
    image.air_mass = header.get('AIRMASS', 1.0)

    #  Add instrument to image class
    image.instrument = instrument


def mkfilelist(path, formats=[".FIT",".fit",".FITS",".fits"], addpath=False,
               sort=False):
    '''
        Fill the file list

        Parameters
        ----------
        path        : `string`
            Path to the files

        formats     : `list` of `string`
            List of allowed Formats

        addpath     : `boolean`, optional
            If `True` the path will be added to the file names.
            Default is ``False``.

        sort        : `boolean`, optional
            If `True the file list will be sorted.
            Default is ``False``.

        Returns
        -------
        fileList    : `list` of `string`
            List with file names

        nfiles      : `interger`
            Number of files
    '''
    fileList = os.listdir(path)
    if sort:
        fileList.sort()

    #   Remove not TIFF entries
    tempList = []
    for file_i in fileList:
        for j, form in enumerate(formats):
            if file_i.find(form)!=-1:
                if addpath:
                    tempList.append(os.path.join(path, file_i))
                else:
                    tempList.append(file_i)

    return tempList, int(len(fileList))


def random_string_generator(str_size):
    '''
        Generate random string

        Parameters
        ----------
        str_size        : `integer`
            Length of the string

        Returns
        -------
                        : `string`
            Random string of length ``str_size``.
    '''
    allowed_chars = string.ascii_letters

    return ''.join(random.choice(allowed_chars) for x in range(str_size))


def get_basename(path):
    '''
        Determine basename without ending from a file path. Accounts for
        multiple dots in the file name.

        Parameters
        ----------
        path            : `string` or `pathlib.Path` object
            Path to the file

        Returns
        -------
        basename        : `string`
            Basename without ending
    '''
    name_parts = str(path).split('/')[-1].split('.')[0:-1]
    if len(name_parts) == 1:
        basename = name_parts[0]
    else:
        basename = name_parts[0]
        for part in name_parts[1:]:
            basename = basename+'.'+part

    return basename


def timeis(func):
    '''
        Decorator that reports the execution time
    '''

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(func.__name__, end-start)
        return result
    return wrap


def startProgress(title):
    '''
        Start progress bar
    '''
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0


def progress(x):
    '''
        Update progress bar
    '''
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x


def endProgress():
    '''
        End progress bar
    '''
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()


def dict_vs_df(var):
    '''
        Distinguish between dictionary and pandas data frame
    '''
    if isinstance(var, dict):
        return False
    elif isinstance(var, pd.DataFrame):
        return True
    else:
        raise Exception(
            f"{style.bcolors.FAIL} \nType of varaible not recognized"
            f"{style.bcolors.ENDC}"
            )


def np_vs_df(var):
    '''
        Distinguish between numpy array and pandas data frame
    '''
    if isinstance(var, np.ndarray):
        return False
    elif isinstance(var, pd.DataFrame):
        return True
    else:
        raise Exception(
            f"{style.bcolors.FAIL} \nType of varaible not recognized"
            f"{style.bcolors.ENDC}"
            )


def indices_to_slices(a):
    '''
        Convert a list of indices to slices for an array

        Parameters
        ----------
        a               : `list`
            List of indices

        Returns
        -------
        slices          : `list`
            List of slices
    '''
    it = iter(a)
    start = next(it)
    slices = []
    for i, x in enumerate(it):
        if x - a[i] != 1:
            end = a[i]
            if start == end:
                slices.append([start])
            else:
                slices.append([start, end])
            start = x
    if a[-1] == start:
        slices.append([start])
    else:
        slices.append([start, a[-1]])

    return slices


def link_files(output_path, file_list):
    '''
        Links files from a list (`file_list`) to a target directory

        Parameters
        ----------
        output_path         : `pathlib.Path`
            Target path

        file_list           : `list` of `string`
            List with file paths that should be linked to the target directory
    '''
    #   Check and if necessary create output directory
    checks.check_out(output_path)

    for path in file_list:
        #   Make a Path object
        p = Path(path)

        #   Set target
        target_path = output_path / p.name

        #   Remove stuff from previous runs
        target_path.unlink(missing_ok=True)

        #   Set link
        target_path.symlink_to(p.absolute())


def find_wcs_astrometry(image, rmcos=False, path_cos=None, indent=2,
                        force_wcs_determ=False, wcs_dir=None):
    '''
        Find WCS (using astrometry.net)

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        rmcos               : `boolean`, optional (obsolete)
            If True the function assumes that the cosmic ray reduction
            function was run before this function
            Default is ``False``.

        path_cos            : `string` (obsolete)
            Path to the image in case 'rmcos' is True
            Default is ``None``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        wcs_dir             : `string` or `None`
            Path to the working directory, where intermediate data will be
            saved. If `None` a 'wcs_imgs` directory will be created in the
            output directory.
            Default is ``None``.

        Returns
        -------
        w                   : `astropy.wcs.WCS`
            WCS information
    '''
    terminal_output.print_terminal(
        indent=indent,
        string="Searching for a WCS solution (pixel to ra/dec conversion)",
        )

    #   Define WCS dir
    if wcs_dir is None:
        wcs_dir = (image.outpath / 'wcs_imgs')
    else:
        wcs_dir = checks.check_pathlib_Path(wcs_dir)
        wcs_dir = wcs_dir / random_string_generator(7)
        checks.check_out(wcs_dir)


    #   Check output directories
    checks.check_out(image.outpath, wcs_dir)

    #   RA & DEC
    coord = image.coord
    ra  = coord.ra.deg
    dec = coord.dec.deg

    #   Select file depending on whether cosmics were rm or not
    if rmcos:
        wcsFILE = path_cos
    else:
        wcsFILE = image.path

    #   Get image base name
    basename = get_basename(wcsFILE)

    #   Compose file name
    filename = basename+'.new'
    filepath = Path(wcs_dir / filename)

    #   String passed to the shell
    #command=('solve-field --overwrite --scale-units arcsecperpix '
            #+'--scale-low '+str(image.pixscale-0.1)+' --scale-high '
            #+str(image.pixscale+0.1)+' --ra '+str(ra)+' --dec '+str(dec)
            #+' --radius 1.0 --dir '+str(wcs_dir)+' --resort '+str(wcsFILE).replace(' ', '\ ')
            #+' --fits-image'
            #)
    command=(
        f'solve-field --overwrite --scale-units arcsecperpix --scale-low '
        f'{image.pixscale-0.1} --scale-high {image.pixscale+0.1} --ra {ra} '
        f'--dec {dec} --radius 1.0 --dir {wcs_dir} --resort '
        '{} --fits-image'.format(str(wcsFILE).replace(" ", "\ "))
        )

    #   Running the command
    cmd_output = subprocess.run(
        [command],
        shell=True,
        text=True,
        capture_output=True,
        )

    rcode = cmd_output.returncode
    rfind = cmd_output.stdout.find('Creating new FITS file')
    if rcode != 0 or rfind == -1:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNo wcs solution could be found for "
            f"the images!\n {style.bcolors.ENDC}{style.bcolors.BOLD}"
            f"The command was:\n {command} \nDetailed error output:\n"
            f"{style.bcolors.ENDC}{cmd_output.stdout}{cmd_output.stderr}"
            f"{style.bcolors.FAIL}Exit{style.bcolors.ENDC}"
            )

    terminal_output.print_terminal(
        indent=indent,
        string="WCS solution found :)",
        style_name='OKGREEN',
        )

    #   Get image hdu list
    hdulist = fits.open(filepath)

    #   Extract the WCS
    w = wcs.WCS(hdulist[0].header)

    image.wcs = w
    return w


def find_wcs_twirl(image, x=None, y=None, indent=2):
    '''
        Calculate WCS information from star positions
        -> use twirl libary

        Parameters:
        -----------
        image           : `image.class`
            Image class with all image specific properties

        x, y            : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        indent          : `string`, optional
            Indentation for the console output lines
            Default is ``2``.
    '''
    terminal_output.print_terminal(
        indent=indent,
        string="Searching for a WCS solution (pixel to ra/dec conversion)",
        )

    #   Arrange object positions
    x = np.array(x)
    y = np.array(y)
    objects = np.column_stack((x,y))

    #   Limit the number of objects to 50
    if len(objects) > 50:
        n = 50
    else:
        n = len(objects)
    objects = objects[0:n]

    coord = image.coord
    fov   = image.fov
    print('n', n, 'fov', fov, coord.ra.deg, coord.dec.deg)
    #   Calculate WCS
    gaias = twirl.gaia_radecs(
        [coord.ra.deg, coord.dec.deg],
        fov/60,
        #limit=n,
        limit=300,
        )
    wcs = twirl._compute_wcs(objects, gaias, n=n)

    gaias_pixel = np.array(SkyCoord(gaias, unit="deg").to_pixel(wcs)).T
    print('gaias_pixel')
    print(gaias_pixel)
    print(gaias_pixel.T)
    print('objects')
    print(objects)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(8,8))
    plt.plot(*objects.T, "o", fillstyle="none", c="b", ms=12)
    plt.plot(*gaias_pixel.T, "o", fillstyle="none", c="C1", ms=18)
    plt.savefig('/tmp/test_twirl.pdf', bbox_inches='tight',format='pdf')
    plt.show()

    ##wcs = twirl.compute_wcs(
        #objects,
        #(coord.ra.deg, coord.dec.deg),
        #fov/60,
        #n=n,
        #)

    print(wcs)

    terminal_output.print_terminal(
        indent=indent,
        string="WCS solution found :)",
        style_name='OKGREEN',
        )

    image.wcs = w
    return wcs


def find_wcs_astap(image, indent=2, force_wcs_determ=False):
    '''
        Find WCS (using ASTAP)

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        Returns
        -------
        w                   : `astropy.wcs.WCS`
            WCS information
    '''
    terminal_output.print_terminal(
        indent=indent,
        string="Searching for a WCS solution (pixel to ra/dec conversion)" \
               " for image {}".format(image.pd),
        )

    #   FOV in degrees
    fov = image.fov_y / 60.

    #   Path to image
    wcsFILE = image.path

    #   String passed to the shell
    command=(
        'astap_cli -f {} -r 1 -fov {} -update'.format(wcsFILE, fov)
        )

    #   Running the command
    cmd_output = subprocess.run(
        [command],
        shell=True,
        text=True,
        capture_output=True,
        )

    rcode = cmd_output.returncode
    rfind = cmd_output.stdout.find('Solution found:')
    if rcode != 0 or rfind == -1:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNo wcs solution could be found for "
            f"the images!\n {style.bcolors.ENDC}{style.bcolors.BOLD}"
            f"The command was:\n{command} \nDetailed error output:\n"
            f"{style.bcolors.ENDC}{cmd_output.stdout}{cmd_output.stderr}"
            f"{style.bcolors.FAIL}Exit{style.bcolors.ENDC}"
            )

    terminal_output.print_terminal(
        indent=indent,
        string="WCS solution found :)",
        style_name='OKGREEN',
        )

    #   Get image hdu list
    hdulist = fits.open(wcsFILE)

    #   Extract the WCS
    w = wcs.WCS(hdulist[0].header)

    image.wcs = w
    return w


def check_wcs_exists(image, wcs_dir=None, indent=2):
    '''
        Checks if the image contains already a valid WCS.

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        wcs_dir             : `string` or `None`, optional
            Path to the working directory, where intermediate data will be
            saved. If `None` a 'wcs_imgs` directory will be created in the
            output directory.
            Default is ``None``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
                        : `boolean`
            Is `True` if the image header contains valid WCS information.

        wcsFILE         : `string`
            Path to the image with the WCS
    '''
    #   Path to image
    wcsFILE = image.path

    #   Get WCS of the original image
    wcs_original = wcs.WCS(fits.open(wcsFILE)[0].header)

    #   Determine wcs type of original WCS
    wcs_original_type = wcs_original.get_axis_types()[0]['coordinate_type']


    if wcs_original_type == 'celestial':
        terminal_output.print_terminal(
            indent=indent,
            string="Image contains already a valid WCS.",
            style_name='OKGREEN',
            )
        return True, wcsFILE
    else:
        #   Check if a image with a WCS in the astronomy.net format exists
        #   in the wcs directory (`wcs_dir`)

        #   Set WCS dir
        if wcs_dir is None:
            wcs_dir = (image.outpath / 'wcs_imgs')

        #   Get image base name
        basename = get_basename(image.path)

        #   Compose file name
        filename = basename+'.new'
        filepath = Path(wcs_dir / filename)

        if filepath.is_file():
            #   Get WCS
            wcs_astronomy_net = wcs.WCS(fits.open(filepath)[0].header)

            #   Determine wcs type
            wcs_astronomy_net_type = wcs_astronomy_net.get_axis_types()[0][
                'coordinate_type'
                ]

            if wcs_astronomy_net_type == 'celestial':
                terminal_output.print_terminal(
                    indent=indent,
                    string="Image in the wcs_dir with a valid WCS found.",
                    style_name='OKGREEN',
                    )
                return True, filepath

        return False, ''


def read_params_from_json(jsonfile):
    '''
        Read data from JSON file

        Parameters
        ----------
        jsonfile        : `string`
            Path to the JSON file

        Returns
        -------
                        : `dictionary`
            Dictionary with the data from the JSON file
    '''
    try:
        with open(jsonfile) as file:
            data = json.load(file)
    except:
        data = {}

    return data


def read_params_from_yaml(yamlfile):
    '''
        Read data from YAML file

        Parameters
        ----------
        jsonfile        : `string`
            Path to the YAML file

        Returns
        -------
                        : `dictionary`
            Dictionary with the data from the YAML file
    '''
    try:
        with open(yamlfile, 'r') as file:
            data = yaml.safe_load(file)
    except:
        data = {}

    return data
