############################################################################
#                               Libraries                                  #
############################################################################

import os

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
from astropy.table import Table

from photutils import psf

# import twirl

from regions import PixCoord, RectanglePixelRegion

from pathlib import Path

from . import checks, terminal_output, style, calibration_parameters

############################################################################
#                           Routines & definitions                         #
############################################################################


# class Image:
#     """
#         Image object used to store and transport some data
#     """
#
#     def __init__(self, pd, filter_, object_name, file_path, output_dir):
#         self.pd = pd
#         self.filter_ = filter_
#         # self.object_name = object_name
#         if isinstance(file_path, Path):
#             self.filename = file_path.name
#             self.path = file_path
#         else:
#             self.filename = file_path.split('/')[-1]
#             self.path = Path(file_path)
#         if isinstance(output_dir, Path):
#             self.out_path = output_dir
#         else:
#             self.out_path = Path(output_dir)
#
#     #   Read image
#     def read_image(self):
#         return CCDData.read(self.path)
#
#     #   Get header
#     def get_header(self):
#         return CCDData.read(self.path).meta
#
#     #   Get data
#     def get_data(self):
#         return CCDData.read(self.path).data


class Image:
    """
        Image class: Provides relevant image information and some methods for
                     handling image data.
    """
    def __init__(
            self, pd: int, filter_: str, path: str | Path,
            output_dir: str | Path) -> None:
        #   Set image ID
        self.pd: int = pd

        #   Set filter
        self.filter_: str = filter_

        #   Set file name and complete path
        if isinstance(path, Path):
            self.filename: str = path.name
            self.path: Path = path
        else:
            self.filename = path.split('/')[-1]
            self.path: Path = Path(path)

        #   Set path to output directory
        if isinstance(output_dir, Path):
            self.out_path: Path = output_dir
        else:
            self.out_path: Path = Path(output_dir)

        #   Set wcs default
        self.wcs: wcs.WCS | None = None

        #   Add and calculate further image parameters
        self.instrument: str | None = None
        self.field_of_view_y: float | None = None
        self.field_of_view_x: float | None = None
        self.coordinates_image_center: SkyCoord | None = None
        self.pixel_scale: float | None = None
        self.fov_pixel_region: RectanglePixelRegion | None = None
        self.air_mass: float | None = None
        self.jd: float | None = None
        self.calculate_field_of_view_etc()

        #   Prepare variables for later use
        self.epsf: psf.EPSFModel | None = None
        self.residual_image: np.ndarray | None = None
        self.photometry: Table | None = None
        self.positions: Table | None = None
        # self.magnitudes_with_zp: u.quantity.Quantity | None = None
        self.zp: np.ndarray | None = None

    #   Read image
    def read_image(self) -> CCDData:
        return CCDData.read(self.path)

    #   Get header
    def get_header(self) -> dict[str, str]:
        return CCDData.read(self.path).meta

    #   TODO: Add unit check for error and data
    #   Get data
    # def get_data(self, check_unit: bool = False) -> np.ndarray:
    #     data = CCDData.read(self.path).data
    #     #   If no unit is available, use electron / s
    #     if check_unit and 'unit' not in dir(data):
    #         data = data * u.electron / u.s
    #     return data
    def get_data(self) -> np.ndarray:
        return CCDData.read(self.path).data

    #   Get uncertainty
    def get_error(self) -> np.ndarray:
        return CCDData.read(self.path).uncertainty.array

    # Get mask
    def get_mask(self) -> np.ndarray:
        return CCDData.read(self.path).mask

    #   Get shape
    def get_shape(self) -> tuple[int, int]:
        return CCDData.read(self.path).data.shape

    def calculate_field_of_view_etc(self):
        #   Get header
        header = self.get_header()

        #   Read focal length - set default to 3454. mm
        focal_length = header.get('FOCALLEN', 3454.)

        #   Read ra and dec of image center
        ra = header.get('OBJCTRA', '00 00 00')
        dec = header.get('OBJCTDEC', '+00 00 00')

        #   Convert ra & dec to degrees
        coordinates_sky = SkyCoord(
            ra,
            dec,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )

        #   Number of pixels
        n_pixel_x = header.get('NAXIS1', 0)
        n_pixel_y = header.get('NAXIS2', 0)

        if n_pixel_x == 0:
            raise ValueError(
                f"{style.Bcolors.FAIL}\nException in calculate_field_of_view(): X "
                f"dimension of the image is 0 {style.Bcolors.ENDC}"
            )
        if n_pixel_y == 0:
            raise ValueError(
                f"{style.Bcolors.FAIL}\nException in calculate_field_of_view(): Y "
                f"dimension of the image is 0 {style.Bcolors.ENDC}"
            )

        #   Get binning
        x_binning = header.get('XBINNING', 1)
        y_binning = header.get('YBINNING', 1)

        #   Set instrument
        instrument = header.get('INSTRUME', '')

        if instrument in ['QHYCCD-Cameras-Capture', 'QHYCCD-Cameras2-Capture']:
            #   Physical chip dimensions in pixel
            physical_dimension_x = n_pixel_x * x_binning
            physical_dimension_y = n_pixel_y * y_binning

            #   Set instrument
            if physical_dimension_x == 9576 and physical_dimension_y in [6387, 6388]:
                instrument = 'QHY600M'
            elif physical_dimension_x in [6280, 6279] and physical_dimension_y in [4210, 4209]:
                instrument = 'QHY268M'
            elif physical_dimension_x == 3864 and physical_dimension_y in [2180, 2178]:
                instrument = 'QHY485C'
            else:
                instrument = ''

        #   Calculate chip size in mm
        if 'XPIXSZ' in header:
            pixel_width = header['XPIXSZ']
            chip_length = n_pixel_x * float(pixel_width) / 1000
            chip_height = n_pixel_y * float(pixel_width) / 1000
        else:
            chip_length, chip_height = calibration_parameters.get_chip_dimensions(
                instrument
            )

        #   Calculate field of view
        field_of_view_x = 2 * np.arctan(chip_length / 2 / focal_length)
        field_of_view_y = 2 * np.arctan(chip_height / 2 / focal_length)

        #   Convert to arc min
        field_of_view_x = field_of_view_x * 360. / 2. / np.pi * 60.
        field_of_view_y = field_of_view_y * 360. / 2. / np.pi * 60.

        #   Calculate pixel scale
        pixel_scale = field_of_view_x * 60 / n_pixel_x

        #   Create RectangleSkyRegion that covers the field of view
        # region_sky = RectangleSkyRegion(
        # center=coordinates_sky,
        # width=field_of_view_x * u.rad,
        # height=field_of_view_y * u.rad,
        # angle=0 * u.deg,
        # )
        #   Create RectanglePixelRegion that covers the field of view
        pixel_region = RectanglePixelRegion(
            center=PixCoord(x=int(n_pixel_x / 2), y=int(n_pixel_y / 2)),
            width=n_pixel_x,
            height=n_pixel_y,
        )

        #   Add to image class
        self.coordinates_image_center = coordinates_sky
        self.field_of_view_x = field_of_view_x
        self.field_of_view_y = field_of_view_y
        self.instrument = instrument
        self.pixel_scale = pixel_scale
        # image.region_sky  = region_sky
        self.fov_pixel_region = pixel_region

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

        self.jd = jd
        self.air_mass = header.get('AIRMASS', 1.0)

        #  Add instrument to image class
        self.instrument = instrument


#   TODO: Maybe remove?!
# def calculate_field_of_view(image, indent=2, verbose=True):
#     """
#         Calculate field of view, pixel scale, etc. ...
#
#         Parameters
#         ----------
#         image           : `image.class`
#             Image class with all image specific properties
#
#         indent          : `integer`, optional
#             Indentation for the console output
#             Default is ``2``.
#
#         verbose         : `boolean`, optional
#             If True additional output will be printed to the command line.
#             Default is ``False``.
#     """
#     if verbose:
#         terminal_output.print_to_terminal(
#             "Calculating field of view, PIXEL scale, etc. ... ",
#             indent=indent,
#         )
#
#     #   Get header
#     header = image.get_header()
#
#     #   Read focal length - set default to 3454. mm
#     focal_length = header.get('FOCALLEN', 3454.)
#
#     #   Read ra and dec of image center
#     ra = header.get('OBJCTRA', '00 00 00')
#     dec = header.get('OBJCTDEC', '+00 00 00')
#
#     #   Convert ra & dec to degrees
#     coordinates_sky = SkyCoord(
#         ra,
#         dec,
#         unit=(u.hourangle, u.deg),
#         frame="icrs",
#     )
#
#     #   Number of pixels
#     n_pixel_x = header.get('NAXIS1', 0)
#     n_pixel_y = header.get('NAXIS2', 0)
#
#     if n_pixel_x == 0:
#         raise ValueError(
#             f"{style.Bcolors.FAIL}\nException in calculate_field_of_view(): X "
#             f"dimension of the image is 0 {style.Bcolors.ENDC}"
#         )
#     if n_pixel_y == 0:
#         raise ValueError(
#             f"{style.Bcolors.FAIL}\nException in calculate_field_of_view(): Y "
#             f"dimension of the image is 0 {style.Bcolors.ENDC}"
#         )
#
#     #   Get binning
#     x_binning = header.get('XBINNING', 1)
#     y_binning = header.get('YBINNING', 1)
#
#     #   Set instrument
#     instrument = header.get('INSTRUME', '')
#
#     if instrument in ['QHYCCD-Cameras-Capture', 'QHYCCD-Cameras2-Capture']:
#         #   Physical chip dimensions in pixel
#         physical_dimension_x = n_pixel_x * x_binning
#         physical_dimension_y = n_pixel_y * y_binning
#
#         #   Set instrument
#         if physical_dimension_x == 9576 and physical_dimension_y in [6387, 6388]:
#             instrument = 'QHY600M'
#         elif physical_dimension_x in [6280, 6279] and physical_dimension_y in [4210, 4209]:
#             instrument = 'QHY268M'
#         elif physical_dimension_x == 3864 and physical_dimension_y in [2180, 2178]:
#             instrument = 'QHY485C'
#         else:
#             instrument = ''
#
#     #   Calculate chip size in mm
#     if 'XPIXSZ' in header:
#         pixel_width = header['XPIXSZ']
#         chip_length = n_pixel_x * pixel_width / 1000
#         chip_height = n_pixel_y * pixel_width / 1000
#     else:
#         chip_length, chip_height = calibration_parameters.get_chip_dimensions(
#             instrument
#         )
#
#     #   Calculate field of view
#     field_of_view_x = 2 * np.arctan(chip_length / 2 / focal_length)
#     field_of_view_y = 2 * np.arctan(chip_height / 2 / focal_length)
#
#     #   Convert to arc min
#     field_of_view_x = field_of_view_x * 360. / 2. / np.pi * 60.
#     field_of_view_y = field_of_view_y * 360. / 2. / np.pi * 60.
#
#     #   Calculate pixel scale
#     pixel_scale = field_of_view_x * 60 / n_pixel_x
#
#     #   Create RectangleSkyRegion that covers the field of view
#     # region_sky = RectangleSkyRegion(
#     # center=coordinates_sky,
#     # width=field_of_view_x * u.rad,
#     # height=field_of_view_y * u.rad,
#     # angle=0 * u.deg,
#     # )
#     #   Create RectanglePixelRegion that covers the field of view
#     pixel_region = RectanglePixelRegion(
#         center=PixCoord(x=int(n_pixel_x / 2), y=int(n_pixel_y / 2)),
#         width=n_pixel_x,
#         height=n_pixel_y,
#     )
#
#     #   Add to image class
#     image.coordinates_image_center = coordinates_sky
#     image.field_of_view_x = field_of_view_x
#     image.field_of_view_y = field_of_view_y
#     image.instrument = instrument
#     image.pixel_scale = pixel_scale
#     # image.region_sky  = region_sky
#     image.fov_pixel_region = pixel_region
#
#     #   Add JD (observation time) and air mass from Header to image class
#     jd = header.get('JD', None)
#     if jd is None:
#         obs_time = header.get('DATE-OBS', None)
#         if not obs_time:
#             raise ValueError(
#                 f"{style.Bcolors.FAIL} \tERROR: No information about the "
#                 "observation time was found in the header"
#                 f"{style.Bcolors.ENDC}"
#             )
#         jd = Time(obs_time, format='fits').jd
#
#     image.jd = jd
#     image.air_mass = header.get('AIRMASS', 1.0)
#
#     #  Add instrument to image class
#     image.instrument = instrument


def mk_file_list(
        file_path: str, formats: list[str] | None = None,
        add_path_to_file_names: bool = False, sort: bool = False
    ) -> tuple[list[str], int]:
    """
    Fill the file list

    Parameters
    ----------
    file_path
        Path to the files

    formats
        List of allowed Formats
        Default is ``None``.

    add_path_to_file_names
        If `True` the path will be added to the file names.
        Default is ``False``.

    sort
        If `True the file list will be sorted.
        Default is ``False``.

    Returns
    -------
    file_list
        List with file names

    n_files
        Number of files
    """
    #   Sanitize formats
    if formats is None:
        formats = [".FIT", ".fit", ".FITS", ".fits"]

    file_list = os.listdir(file_path)
    if sort:
        file_list.sort()

    #   Remove not TIFF entries
    temp_list = []
    for file_i in file_list:
        for j, format_ in enumerate(formats):
            if file_i.find(format_) != -1:
                if add_path_to_file_names:
                    temp_list.append(os.path.join(file_path, file_i))
                else:
                    temp_list.append(file_i)

    return temp_list, int(len(file_list))


def random_string_generator(str_size: int) -> str:
    """
    Generate random string

    Parameters
    ----------
    str_size
        Length of the string

    Returns
    -------

        Random string of length ``str_size``.
    """
    allowed_chars = string.ascii_letters

    return ''.join(random.choice(allowed_chars) for x in range(str_size))


def get_basename(path: str | Path) -> str:
    """
    Determine basename without ending from a file path. Accounts for
    multiple dots in the file name.

    Parameters
    ----------
    path
        The path to the file

    Returns
    -------
    basename
        The basename without ending
    """
    name_parts = str(path).split('/')[-1].split('.')[0:-1]
    if len(name_parts) == 1:
        basename = name_parts[0]
    else:
        basename = name_parts[0]
        for part in name_parts[1:]:
            basename = basename + '.' + part

    return basename


def execution_time(function):
    """
        Decorator that reports the execution time

        Parameters
        ----------
        function        : `function`
    """

    def wrap(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()

        print(function.__name__, end - start)
        return result

    return wrap


def indices_to_slices(index_list: list[int]) -> list[list[int]]:
    """
    Convert a list of indices to slices for an array

    Parameters
    ----------
    index_list
        List of indices

    Returns
    -------
    slices
        List of slices
    """
    index_iterator = iter(index_list)
    start = next(index_iterator)
    slices = []
    for i, x in enumerate(index_iterator):
        if x - index_list[i] != 1:
            end = index_list[i]
            if start == end:
                slices.append([start])
            else:
                slices.append([start, end])
            start = x
    if index_list[-1] == start:
        slices.append([start])
    else:
        slices.append([start, index_list[-1]])

    return slices


def link_files(output_path: Path, file_list: list[str]) -> None:
    """
    Links files from a list (`file_list`) to a target directory

    Parameters
    ----------
    output_path
        Target path

    file_list
        List with file paths that should be linked to the target directory
    """
    #   Check and if necessary create output directory
    checks.check_output_directories(output_path)

    for path in file_list:
        #   Make a Path object
        p = Path(path)

        #   Set target
        target_path = output_path / p.name

        #   Remove stuff from previous runs
        target_path.unlink(missing_ok=True)

        #   Set link
        target_path.symlink_to(p.absolute())


def find_wcs_astrometry(
        image: Image, cosmic_rays_removed: bool = False,
        path_cosmic_cleaned_image: str | None = None, indent: int = 2,
        wcs_working_dir: str | None = None) -> wcs.WCS:
    """
    Find WCS (using astrometry.net)

    Parameters
    ----------
    image
        An image class with all image specific properties

    cosmic_rays_removed
        If True the function assumes that the cosmic ray reduction
        function was run before this function
        Default is ``False``.

    path_cosmic_cleaned_image
        Path to the image in case 'cosmic_rays_removed' is True
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``2``.

    wcs_working_dir            
        Path to the working directory, where intermediate data will be
        saved. If `None` a wcs_images directory will be created in the
        output directory.
        Default is ``None``.

    Returns
    -------
    derived_wcs
        WCS information
    """
    terminal_output.print_to_terminal(
        "Searching for a WCS solution (pixel to ra/dec conversion)",
        indent=indent,
    )

    #   Define WCS dir
    if wcs_working_dir is None:
        wcs_working_dir = (image.out_path / 'wcs_images')
    else:
        wcs_working_dir = checks.check_pathlib_path(wcs_working_dir)
        wcs_working_dir = wcs_working_dir / random_string_generator(7)
        checks.check_output_directories(wcs_working_dir)

    #   Check output directories
    checks.check_output_directories(image.out_path, wcs_working_dir)

    #   RA & DEC
    coordinates = image.coordinates_image_center
    ra = coordinates.ra.deg
    dec = coordinates.dec.deg

    #   Select file depending on whether cosmics were rm or not
    if cosmic_rays_removed:
        wcs_file = path_cosmic_cleaned_image
    else:
        wcs_file = image.path

    #   Get image base name
    basename = get_basename(wcs_file)

    #   Compose file name
    filename = basename + '.new'
    filepath = Path(wcs_working_dir / filename)

    #   String passed to the shell
    # command=('solve-field --overwrite --scale-units arcsecperpix '
    # +'--scale-low '+str(image.pixscale-0.1)+' --scale-high '
    # +str(image.pixscale+0.1)+' --ra '+str(ra)+' --dec '+str(dec)
    # +' --radius 1.0 --dir '+str(wcs_dir)+' --resort '+str(wcsFILE).replace(' ', '\ ')
    # +' --fits-image'
    # )
    pixel_scale = image.pixel_scale
    pixel_scale_low = pixel_scale - 0.1
    pixel_scale_up = pixel_scale + 0.1
    command: str = (
        f'solve-field --overwrite --scale-units arcsecperpix --scale-low ' +
        f'{pixel_scale_low} --scale-high {pixel_scale_up} --ra {ra} ' +
        f'--dec {dec} --radius 1.0 --dir {wcs_working_dir} --resort ' +
        '{} --fits-image -z 2'.format(str(wcs_file).replace(" ", "\ "))
    )

    #   Running the command
    command_result = subprocess.run(
        [command],
        shell=True,
        text=True,
        capture_output=True,
    )

    return_code = command_result.returncode
    fits_created = command_result.stdout.find('Creating new FITS file')
    if return_code != 0 or fits_created == -1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo wcs solution could be found for "
            f"the images!\n {style.Bcolors.ENDC}{style.Bcolors.BOLD}"
            f"The command was:\n {command} \nDetailed error output:\n"
            f"{style.Bcolors.ENDC}{command_result.stdout}{command_result.stderr}"
            f"{style.Bcolors.FAIL}Exit{style.Bcolors.ENDC}"
        )

    terminal_output.print_to_terminal(
        "WCS solution found :)",
        indent=indent,
        style_name='OKGREEN',
    )

    #   Get image hdu list
    hdu_list = fits.open(filepath)

    #   Extract the WCS
    derived_wcs = wcs.WCS(hdu_list[0].header)

    image.wcs = derived_wcs
    return derived_wcs


#   TODO: Make this work
def find_wcs_twirl(
        image: Image, object_pixel_position_x: np.ndarray | None = None,
        object_pixel_position_y: np.ndarray = None, indent: int = 2) -> wcs.WCS:
    """
    Calculate WCS information from star positions
    -> use twirl library

    Parameters:
    -----------
    image
        The image class with all image specific properties

    object_pixel_position_x
        Pixel coordinates of the objects
        Default is ``None``.

    object_pixel_position_y
        Pixel coordinates of the objects
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``2``.

    Returns
    -------
    derived_wcs
        WCS information
    """
    terminal_output.print_to_terminal(
        "Searching for a WCS solution (pixel to ra/dec conversion)",
        indent=indent,
    )

    #   Arrange object positions
    object_pixel_position_x = np.array(object_pixel_position_x)
    object_pixel_position_y = np.array(object_pixel_position_y)
    objects = np.column_stack(
        (object_pixel_position_x, object_pixel_position_y)
    )

    #   Limit the number of objects to 50
    if len(objects) > 50:
        n = 50
    else:
        n = len(objects)
    objects = objects[0:n]

    coordinates = image.coordinates_image_center
    field_of_view = image.field_of_view_x
    print('n', n, 'field_of_view', field_of_view, coordinates.ra.deg, coordinates.dec.deg)
    #   Calculate WCS
    gaia_twirl = twirl.gaia_radecs(
        [coordinates.ra.deg, coordinates.dec.deg],
        field_of_view / 60,
        # limit=n,
        limit=300,
    )
    derived_wcs = twirl._compute_wcs(objects, gaia_twirl, n=n)

    gaia_twirl_pixel = np.array(
        SkyCoord(gaia_twirl, unit="deg").to_pixel(derived_wcs)
    ).T
    print('gaia_twirl_pixel')
    print(gaia_twirl_pixel)
    print(gaia_twirl_pixel.T)
    print('objects')
    print(objects)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.plot(*objects.T, "o", fillstyle="none", c="b", ms=12)
    plt.plot(*gaia_twirl_pixel.T, "o", fillstyle="none", c="C1", ms=18)
    plt.savefig('/tmp/test_twirl.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    # #derived_wcs = twirl.compute_wcs(
    # objects,
    # (coordinates.ra.deg, coordinates.dec.deg),
    # field_of_view/60,
    # n=n,
    # )

    print(derived_wcs)

    terminal_output.print_to_terminal(
        "WCS solution found :)",
        indent=indent,
        style_name='OKGREEN',
    )

    image.wcs = derived_wcs
    return derived_wcs


def find_wcs_astap(image: Image, indent: int = 2) -> wcs.WCS:
    """
    Find WCS (using ASTAP)

    Parameters
    ----------
    image
        The image class with all image specific properties

    indent
        Indentation for the console output lines
        Default is ``2``.

    Returns
    -------
    derived_wcs
        WCS information
    """
    terminal_output.print_to_terminal(
        "Searching for a WCS solution (pixel to ra/dec conversion)"
        f" for image {image.pd}",
        indent=indent,
    )

    #   Field of view in degrees
    field_of_view = image.field_of_view_y / 60.

    #   Path to image
    wcs_file = image.path

    #   String passed to the shell
    command = (
        'astap_cli -f {} -r 1 -fov {} -update'.format(wcs_file, field_of_view)
    )

    #   Running the command
    command_result = subprocess.run(
        [command],
        shell=True,
        text=True,
        capture_output=True,
    )

    return_code = command_result.returncode
    solution_found = command_result.stdout.find('Solution found:')
    if return_code != 0 or solution_found == -1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo wcs solution could be found for "
            f"the images!\n {style.Bcolors.ENDC}{style.Bcolors.BOLD}"
            f"The command was:\n{command} \nDetailed error output:\n"
            f"{style.Bcolors.ENDC}{command_result.stdout}{command_result.stderr}"
            f"{style.Bcolors.FAIL}Exit{style.Bcolors.ENDC}"
        )

    terminal_output.print_to_terminal(
        "WCS solution found :)",
        indent=indent,
        style_name='OKGREEN',
    )

    #   Get image hdu list
    hdu_list = fits.open(wcs_file)

    #   Extract the WCS
    derived_wcs = wcs.WCS(hdu_list[0].header)

    image.wcs = derived_wcs
    return derived_wcs


def check_wcs_exists(
        image: Image, wcs_dir: str | None = None, indent: int = 2
    ) -> tuple[bool, Path | str]:
    """
    Checks if the image contains already a valid WCS.

    Parameters
    ----------
    image
        The image class with all image specific properties

    wcs_dir
        Path to the working directory, where intermediate data will be
        saved. If `None` a wcs_images directory will be created in the
        output directory.
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``2``.

    Returns
    -------

        Is `True` if the image header contains valid WCS information.

    wcs_file
        Path to the image with the WCS
    """
    #   Path to image
    wcs_file = image.path

    #   Get WCS of the original image
    wcs_original = wcs.WCS(fits.open(wcs_file)[0].header)

    #   Determine wcs type of original WCS
    wcs_original_type = wcs_original.get_axis_types()[0]['coordinate_type']

    if wcs_original_type == 'celestial':
        terminal_output.print_to_terminal(
            "Image contains already a valid WCS.",
            indent=indent,
            style_name='OKGREEN',
        )
        return True, wcs_file
    else:
        #   Check if an image with a WCS in the astronomy.net format exists
        #   in the wcs directory (`wcs_dir`)

        #   Set WCS dir
        if wcs_dir is None:
            wcs_dir = (image.out_path / 'wcs_images')

        #   Get image base name
        basename = get_basename(image.path)

        #   Compose file name
        filename = f'{basename}.new'
        filepath = Path(wcs_dir / filename)

        if filepath.is_file():
            #   Get WCS
            wcs_astronomy_net = wcs.WCS(fits.open(filepath)[0].header)

            #   Determine wcs type
            wcs_astronomy_net_type = wcs_astronomy_net.get_axis_types()[0][
                'coordinate_type'
            ]

            if wcs_astronomy_net_type == 'celestial':
                terminal_output.print_to_terminal(
                    "Image found in wcs_dir with a valid WCS.",
                    indent=indent,
                    style_name='OKGREEN',
                )
                return True, filepath

        return False, ''


def read_params_from_json(json_file: str) -> dict:
    """
    Read data from JSON file

    Parameters
    ----------
    json_file
        Path to the JSON file

    Returns
    -------

        Dictionary with the data from the JSON file
    """
    try:
        with open(json_file) as file:
            data = json.load(file)
            #   TODO: Check data datatype
    except:
        #   TODO: Test this to specify the exception
        data = {}

    return data


def read_params_from_yaml(yaml_file: str) -> dict:
    """
    Read data from YAML file

    Parameters
    ----------
    yaml_file
        Path to the YAML file

    Returns
    -------

        Dictionary with the data from the YAML file
    """
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            #   TODO: Check data datatype
    except:
        #   TODO: Test this to specify the exception
        data = {}

    return data
