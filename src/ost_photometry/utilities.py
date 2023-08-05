############################################################################
#                               Libraries                                  #
############################################################################

import os
import sys

import time

import random
import string

import subprocess

import json
import yaml

import numpy as np

from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy import wcs

import twirl

from regions import PixCoord, RectanglePixelRegion

from pathlib import Path

from . import checks, terminal_output, style, calibration_data

############################################################################
#                           Routines & definitions                         #
############################################################################


class Image:
    """
        Image object used to store and transport some data
    """

    def __init__(self, pd, filt, name, path, outdir):
        self.pd = pd
        self.filt = filt
        self.objname = name
        if isinstance(path, Path):
            self.filename = path.name
            self.path = path
        else:
            self.filename = path.split('/')[-1]
            self.path = Path(path)
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
    """
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
    """
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
    ra = header.get('OBJCTRA', '00 00 00')
    dec = header.get('OBJCTDEC', '+00 00 00')

    #   Convert ra & dec to degrees
    coord_fov = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")

    #   Number of pixels
    n_pix_x = header.get('NAXIS1', 0)
    n_pix_y = header.get('NAXIS2', 0)

    if n_pix_x == 0:
        raise ValueError(
            f"{style.Bcolors.FAIL}\nException in cal_fov(): X dimension of "
            f"the image is 0 {style.Bcolors.ENDC}"
        )
    if n_pix_y == 0:
        raise ValueError(
            f"{style.Bcolors.FAIL}\nException in cal_fov(): Y dimension of "
            f"the image is 0 {style.Bcolors.ENDC}"
        )

    #   Get binning
    bin_x = header.get('XBINNING', 1)
    bin_y = header.get('YBINNING', 1)

    #   Set instrument
    instrument = header.get('INSTRUME', '')

    if instrument in ['QHYCCD-Cameras-Capture', 'QHYCCD-Cameras2-Capture']:
        #   Physical chip dimensions in pixel
        xdim_phy = n_pix_x * bin_x
        ydim_phy = n_pix_y * bin_y

        #   Set instrument
        if xdim_phy == 9576 and ydim_phy == 6388:
            instrument = 'QHY600M'
        elif xdim_phy == 6280 and ydim_phy == 4210:
            instrument = 'QHY268M'
        elif xdim_phy == 3864 and ydim_phy == 2180:
            instrument = 'QHY485C'
        else:
            instrument = ''

    #   Calculate chip size in mm
    if 'XPIXSZ' in header:
        pixwidth = header['XPIXSZ']
        d = n_pix_x * pixwidth / 1000
        h = n_pix_y * pixwidth / 1000
    else:
        d, h = calibration_data.get_chip_dimensions(instrument)

    #   Calculate field of view
    fov_x = 2 * np.arctan(d / 2 / f)
    fov_y = 2 * np.arctan(h / 2 / f)

    #   Convert to arc min
    fov = fov_x * 360. / 2. / np.pi * 60.
    fov_y = fov_y * 360. / 2. / np.pi * 60.

    #   Calculate pixel scale
    pixscale = fov * 60 / n_pix_x

    #   Create RectangleSkyRegion that covers the field of view
    # region_sky = RectangleSkyRegion(
    # center=coord_fov,
    # width=fov_x * u.rad,
    # height=fov_y * u.rad,
    # angle=0 * u.deg,
    # )
    #   Create RectanglePixelRegion that covers the field of view
    region_pix = RectanglePixelRegion(
        center=PixCoord(x=int(n_pix_x / 2), y=int(n_pix_y / 2)),
        width=n_pix_x,
        height=n_pix_y,
    )

    #   Add to image class
    image.coord = coord_fov
    image.fov = fov
    image.fov_y = fov_y
    image.instrument = instrument
    image.pixscale = pixscale
    # image.region_sky  = region_sky
    image.region_pix = region_pix

    #   Add JD (observation time) and air mass from Header to image class
    jd = header.get('JD', None)
    if jd is None:
        obs_time = header.get('DATE-OBS', None)
        if not obs_time:
            raise ValueError(
                f"{style.Bcolors.FAIL} \tERROR: No information about the "
                "observation time was found in the header"
                f"{style.Bcolors.ENDC}"
            )
        jd = Time(obs_time, format='fits').jd

    image.jd = jd
    image.air_mass = header.get('AIRMASS', 1.0)

    #  Add instrument to image class
    image.instrument = instrument


def mkfilelist(path, formats=[".FIT", ".fit", ".FITS", ".fits"], addpath=False,
               sort=False):
    """
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
    """
    file_list = os.listdir(path)
    if sort:
        file_list.sort()

    #   Remove not TIFF entries
    temp_list = []
    for file_i in file_list:
        for j, form in enumerate(formats):
            if file_i.find(form) != -1:
                if addpath:
                    temp_list.append(os.path.join(path, file_i))
                else:
                    temp_list.append(file_i)

    return temp_list, int(len(file_list))


def random_string_generator(str_size):
    """
        Generate random string

        Parameters
        ----------
        str_size        : `integer`
            Length of the string

        Returns
        -------
                        : `string`
            Random string of length ``str_size``.
    """
    allowed_chars = string.ascii_letters

    return ''.join(random.choice(allowed_chars) for x in range(str_size))


def get_basename(path):
    """
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
    """
    name_parts = str(path).split('/')[-1].split('.')[0:-1]
    if len(name_parts) == 1:
        basename = name_parts[0]
    else:
        basename = name_parts[0]
        for part in name_parts[1:]:
            basename = basename + '.' + part

    return basename


def timeis(func):
    """
        Decorator that reports the execution time
    """

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(func.__name__, end - start)
        return result

    return wrap


#   TODO: Remove unused functions?
def start_progress(title):
    """
        Start progress bar
    """
    global progress_x
    sys.stdout.write(title + ": [" + "-" * 40 + "]" + chr(8) * 41)
    sys.stdout.flush()
    progress_x = 0


def progress(x):
    """
        Update progress bar
    """
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x


def end_progress():
    """
        End progress bar
    """
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()


def indices_to_slices(a):
    """
        Convert a list of indices to slices for an array

        Parameters
        ----------
        a               : `list`
            List of indices

        Returns
        -------
        slices          : `list`
            List of slices
    """
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
    """
        Links files from a list (`file_list`) to a target directory

        Parameters
        ----------
        output_path         : `pathlib.Path`
            Target path

        file_list           : `list` of `string`
            List with file paths that should be linked to the target directory
    """
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
                        wcs_dir=None):
    """
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

        wcs_dir             : `string` or `None`
            Path to the working directory, where intermediate data will be
            saved. If `None` a wcs_imgs` directory will be created in the
            output directory.
            Default is ``None``.

        Returns
        -------
        w                   : `astropy.wcs.WCS`
            WCS information
    """
    terminal_output.print_terminal(
        indent=indent,
        string="Searching for a WCS solution (pixel to ra/dec conversion)",
    )

    #   Define WCS dir
    if wcs_dir is None:
        wcs_dir = (image.outpath / 'wcs_imgs')
    else:
        wcs_dir = checks.check_pathlib_path(wcs_dir)
        wcs_dir = wcs_dir / random_string_generator(7)
        checks.check_out(wcs_dir)

    #   Check output directories
    checks.check_out(image.outpath, wcs_dir)

    #   RA & DEC
    coord = image.coord
    ra = coord.ra.deg
    dec = coord.dec.deg

    #   Select file depending on whether cosmics were rm or not
    if rmcos:
        wcs_file = path_cos
    else:
        wcs_file = image.path

    #   Get image base name
    basename = get_basename(wcs_file)

    #   Compose file name
    filename = basename + '.new'
    filepath = Path(wcs_dir / filename)

    #   String passed to the shell
    # command=('solve-field --overwrite --scale-units arcsecperpix '
    # +'--scale-low '+str(image.pixscale-0.1)+' --scale-high '
    # +str(image.pixscale+0.1)+' --ra '+str(ra)+' --dec '+str(dec)
    # +' --radius 1.0 --dir '+str(wcs_dir)+' --resort '+str(wcsFILE).replace(' ', '\ ')
    # +' --fits-image'
    # )
    command = (
        f'solve-field --overwrite --scale-units arcsecperpix --scale-low '
        f'{image.pixscale - 0.1} --scale-high {image.pixscale + 0.1} --ra {ra} '
        f'--dec {dec} --radius 1.0 --dir {wcs_dir} --resort '
        '{} --fits-image'.format(str(wcs_file).replace(" ", "\ "))
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
            f"{style.Bcolors.FAIL} \nNo wcs solution could be found for "
            f"the images!\n {style.Bcolors.ENDC}{style.Bcolors.BOLD}"
            f"The command was:\n {command} \nDetailed error output:\n"
            f"{style.Bcolors.ENDC}{cmd_output.stdout}{cmd_output.stderr}"
            f"{style.Bcolors.FAIL}Exit{style.Bcolors.ENDC}"
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
    """
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
    """
    terminal_output.print_terminal(
        indent=indent,
        string="Searching for a WCS solution (pixel to ra/dec conversion)",
    )

    #   Arrange object positions
    x = np.array(x)
    y = np.array(y)
    objects = np.column_stack((x, y))

    #   Limit the number of objects to 50
    if len(objects) > 50:
        n = 50
    else:
        n = len(objects)
    objects = objects[0:n]

    coord = image.coord
    fov = image.fov
    print('n', n, 'fov', fov, coord.ra.deg, coord.dec.deg)
    #   Calculate WCS
    gaias = twirl.gaia_radecs(
        [coord.ra.deg, coord.dec.deg],
        fov / 60,
        # limit=n,
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
    plt.figure(figsize=(8, 8))
    plt.plot(*objects.T, "o", fillstyle="none", c="b", ms=12)
    plt.plot(*gaias_pixel.T, "o", fillstyle="none", c="C1", ms=18)
    plt.savefig('/tmp/test_twirl.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    # #wcs = twirl.compute_wcs(
    # objects,
    # (coord.ra.deg, coord.dec.deg),
    # fov/60,
    # n=n,
    # )

    print(wcs)

    terminal_output.print_terminal(
        indent=indent,
        string="WCS solution found :)",
        style_name='OKGREEN',
    )

    image.wcs = wcs
    return wcs


def find_wcs_astap(image, indent=2):
    """
        Find WCS (using ASTAP)

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        w                   : `astropy.wcs.WCS`
            WCS information
    """
    terminal_output.print_to_terminal(
        "Searching for a WCS solution (pixel to ra/dec conversion)"
        f" for image {image.pd}",
        indent=indent,
    )

    #   FOV in degrees
    fov = image.fov_y / 60.

    #   Path to image
    wcs_file = image.path

    #   String passed to the shell
    command = (
        'astap_cli -f {} -r 1 -fov {} -update'.format(wcs_file, fov)
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
            f"{style.Bcolors.FAIL} \nNo wcs solution could be found for "
            f"the images!\n {style.Bcolors.ENDC}{style.Bcolors.BOLD}"
            f"The command was:\n{command} \nDetailed error output:\n"
            f"{style.Bcolors.ENDC}{cmd_output.stdout}{cmd_output.stderr}"
            f"{style.Bcolors.FAIL}Exit{style.Bcolors.ENDC}"
        )

    terminal_output.print_terminal(
        indent=indent,
        string="WCS solution found :)",
        style_name='OKGREEN',
    )

    #   Get image hdu list
    hdulist = fits.open(wcs_file)

    #   Extract the WCS
    w = wcs.WCS(hdulist[0].header)

    image.wcs = w
    return w


def check_wcs_exists(image, wcs_dir=None, indent=2):
    """
        Checks if the image contains already a valid WCS.

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        wcs_dir             : `string` or `None`, optional
            Path to the working directory, where intermediate data will be
            saved. If `None` a `wcs_imgs` directory will be created in the
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
    """
    #   Path to image
    wcs_file = image.path

    #   Get WCS of the original image
    wcs_original = wcs.WCS(fits.open(wcs_file)[0].header)

    #   Determine wcs type of original WCS
    wcs_original_type = wcs_original.get_axis_types()[0]['coordinate_type']

    if wcs_original_type == 'celestial':
        terminal_output.print_terminal(
            indent=indent,
            string="Image contains already a valid WCS.",
            style_name='OKGREEN',
        )
        return True, wcs_file
    else:
        #   Check if an image with a WCS in the astronomy.net format exists
        #   in the wcs directory (`wcs_dir`)

        #   Set WCS dir
        if wcs_dir is None:
            wcs_dir = (image.outpath / 'wcs_imgs')

        #   Get image base name
        basename = get_basename(image.path)

        #   Compose file name
        filename = basename + '.new'
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
    """
        Read data from JSON file

        Parameters
        ----------
        jsonfile        : `string`
            Path to the JSON file

        Returns
        -------
                        : `dictionary`
            Dictionary with the data from the JSON file
    """
    try:
        with open(jsonfile) as file:
            data = json.load(file)
    except:
        data = {}

    return data


def read_params_from_yaml(yamlfile):
    """
        Read data from YAML file

        Parameters
        ----------
        yamlfile        : `string`
            Path to the YAML file

        Returns
        -------
                        : `dictionary`
            Dictionary with the data from the YAML file
    """
    try:
        with open(yamlfile, 'r') as file:
            data = yaml.safe_load(file)
    except:
        data = {}

    return data
