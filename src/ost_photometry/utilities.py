############################################################################
#                               Libraries                                  #
############################################################################

import json
import os
import random
import string
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

try:
    from pytimedinput import timedInput

    use_timed_input = True
except ImportError:
    use_timed_input = False

import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.time import Time
from photutils.psf import ImagePSF

# import twirl
from regions import PixCoord, RectanglePixelRegion

from . import calibration_parameters, checks, style, terminal_output
from .wcs import (
    check_wcs_exists,
    find_wcs_astap,
    find_wcs_astrometry,
    find_wcs_twirl,
    persist_wcs_to_fits,
    sync_image_coordinates_from_wcs,
)

############################################################################
#                           Routines & definitions                         #
############################################################################


#   TODO: Split into a base class and a derived class for analysis
class Image:
    """
    Image class: metadata + on-demand FITS access (no persistent pixel cache).

    Use ``with image.open() as ccd:`` inside a pipeline step to avoid repeated
    disk reads without holding all images in RAM across the full run.
    """

    def __init__(
        self, image_id: int, filter_: str, path: str | Path, output_dir: str | Path
    ) -> None:
        self.image_id: int = image_id
        self.filter_: str = filter_
        if isinstance(path, Path):
            self.filename: str = path.name
            self.path: Path = path
        else:
            self.filename = path.split("/")[-1]
            self.path = Path(path)
        if isinstance(output_dir, Path):
            self.out_path: Path = output_dir
        else:
            self.out_path = Path(output_dir)

        self.wcs: wcs.WCS | None = None
        self.instrument: str | None = None
        self.field_of_view_y: float | None = None
        self.field_of_view_x: float | None = None
        self.coordinates_image_center: SkyCoord | None = None
        self.pixel_scale: float | None = None
        self.fov_pixel_region: RectanglePixelRegion | None = None
        self.air_mass: float | None = None
        self.jd: float | None = None
        self.fwhm: float = 4.0
        self.epsf: ImagePSF | None = None
        self.residual_image: np.ndarray | None = None
        self.photometry: Table | None = None
        self.positions: Table | None = None
        self.zp: np.ndarray | None = None

        self._header_cache: fits.Header | None = None
        self._active_ccd: CCDData | None = None
        self.ensure_metadata()

    def ensure_metadata(self) -> None:
        """Load header-derived metadata once (small memory footprint)."""
        if self.field_of_view_x is None:
            self.calculate_field_of_view_etc()

    @contextmanager
    def open(self, memmap: bool = True):
        """Short-lived FITS session for a processing step (released on exit)."""
        ccd = CCDData.read(self.path, memmap=memmap)
        self._active_ccd = ccd
        try:
            yield ccd
        finally:
            self._active_ccd = None

    def read_image(self, memmap: bool = True) -> CCDData:
        return self._read_ccd(memmap=memmap)

    def get_header(self) -> fits.Header:
        if self._header_cache is None:
            self._header_cache = fits.getheader(self.path)
        return self._header_cache

    def _read_ccd(self, memmap: bool = True) -> CCDData:
        if self._active_ccd is not None:
            return self._active_ccd
        return CCDData.read(self.path, memmap=memmap)

    def get_data(self) -> np.ndarray:
        return self._read_ccd().data

    def get_error(self) -> np.ndarray:
        ccd = self._read_ccd()
        if ccd.uncertainty is None:
            raise ValueError(f"No uncertainty extension in {self.path}")
        return ccd.uncertainty.array

    def get_mask(self) -> np.ndarray | None:
        return self._read_ccd().mask

    def get_shape(self) -> tuple[int, int]:
        if self._active_ccd is not None:
            return self._active_ccd.data.shape
        header = self.get_header()
        return int(header.get("NAXIS1", 0)), int(header.get("NAXIS2", 0))

    def calculate_field_of_view_etc(self):
        #   Get header
        header = self.get_header()

        #   Read focal length - set default to 3454. mm
        focal_length = header.get("FOCALLEN", 3454.0)

        #   Read ra and dec of image center
        ra = header.get("OBJCTRA", "00 00 00")
        dec = header.get("OBJCTDEC", "+00 00 00")

        #   Convert ra & dec to degrees
        coordinates_sky = SkyCoord(
            ra,
            dec,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )

        #   Number of pixels
        n_pixel_x = header.get("NAXIS1", 0)
        n_pixel_y = header.get("NAXIS2", 0)

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
        x_binning = header.get("XBINNING", 1)
        y_binning = header.get("YBINNING", 1)

        #   Set instrument
        instrument = header.get("INSTRUME", "")

        if instrument in ["QHYCCD-Cameras-Capture", "QHYCCD-Cameras2-Capture"]:
            #   Physical chip dimensions in pixel
            physical_dimension_x = n_pixel_x * x_binning
            physical_dimension_y = n_pixel_y * y_binning

            #   Set instrument
            if physical_dimension_x == 9576 and physical_dimension_y in [6387, 6388]:
                instrument = "QHY600M"
            elif physical_dimension_x in [6280, 6279] and physical_dimension_y in [
                4210,
                4209,
            ]:
                instrument = "QHY268M"
            elif physical_dimension_x == 3864 and physical_dimension_y in [2180, 2178]:
                instrument = "QHY485C"
            else:
                instrument = ""

        #   Calculate chip size in mm
        if "XPIXSZ" in header:
            pixel_width = header["XPIXSZ"]
            chip_length = n_pixel_x * float(pixel_width) / 1000
            chip_height = n_pixel_y * float(pixel_width) / 1000
        elif "PIXSIZE1" in header:
            pixel_width = header["PIXSIZE1"]
            chip_length = n_pixel_x * float(pixel_width) / 1000
            chip_height = n_pixel_y * float(pixel_width) / 1000
        else:
            terminal_output.print_to_terminal(
                "Warning chip dimension could not be determined from Header. "
                "Use default values, assuming the image has not been cropped. "
                "This may be completely wrong. ",
                indent=1,
                style_name="WARNING",
            )
            chip_length, chip_height = calibration_parameters.get_chip_dimensions(
                instrument
            )

        #   Calculate field of view
        field_of_view_x = 2 * np.arctan(chip_length / 2 / focal_length)
        field_of_view_y = 2 * np.arctan(chip_height / 2 / focal_length)

        #   Convert to arc min
        field_of_view_x = field_of_view_x * 360.0 / 2.0 / np.pi * 60.0
        field_of_view_y = field_of_view_y * 360.0 / 2.0 / np.pi * 60.0

        #   Calculate pixel scale [arcsec/pixel]
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
        jd = header.get("JD", None)
        if jd is None:
            obs_time = header.get("DATE-OBS", None)
            if not obs_time:
                raise ValueError(
                    f"{style.Bcolors.FAIL} \tERROR: No information about the "
                    "observation time was found in the header"
                    f"{style.Bcolors.ENDC}"
                )
            jd = Time(obs_time, format="fits").jd

        self.jd = jd
        self.air_mass = header.get("AIRMASS", 1.0)

        #  Add instrument to image class
        self.instrument = instrument


def mk_file_list(
    file_path: str,
    formats: list[str] | None = None,
    add_path_to_file_names: bool = False,
    sort: bool = False,
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

    return "".join(random.choice(allowed_chars) for x in range(str_size))


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
    name_parts = str(path).split("/")[-1].split(".")[0:-1]
    if len(name_parts) == 1:
        basename = name_parts[0]
    else:
        basename = name_parts[0]
        for part in name_parts[1:]:
            basename = basename + "." + part

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
    except (json.JSONDecodeError, FileNotFoundError):
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
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
            #   TODO: Check data datatype
    except (yaml.YAMLError, FileNotFoundError):
        data = {}

    return data


def get_input(prompt: str, timeout: int = 30) -> tuple[str | None, bool]:
    """
    Prompt the user for input. Uses pytimedinput with a timeout if available,
    otherwise falls back to the built-in input function.

    Parameters
    ----------
    prompt (str):
        The message displayed to the user.

    timeout (int, optional):
        Timeout in seconds for timed input. Only applies if pytimedinput is
        installed.
        Default is ``30``.

    Returns
    -------
    str | None:
        The user's input as a string, or None if input timed out (only possible
        with pytimedinput).

    boolean:
        Returns `True` if the prompt timed out (only possible with
        pytimedinput). When using the built-in input() function, `False` is
        always returned.
    """
    if use_timed_input:
        user_input, timed_out = timedInput(prompt, timeout=timeout)
        if timed_out:
            terminal_output.print_to_terminal(
                "The prompt timed out!",
                indent=2,
                style_name="WARNING",
            )
            user_input: str = "no"
        return user_input, timed_out
    else:
        return input(prompt), False
