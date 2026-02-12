"""WCS (World Coordinate System) determination utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits

from . import checks, style, terminal_output

if TYPE_CHECKING:
    from .utilities import Image  # noqa: F401


def find_wcs_astrometry(
    image: Image,
    cosmic_rays_removed: bool = False,
    path_cosmic_cleaned_image: str | None = None,
    indent: int = 2,
    wcs_working_dir: str | None = None,
) -> wcs.WCS:
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
    from .utilities import get_basename, random_string_generator

    terminal_output.print_to_terminal(
        "Searching for a WCS solution (pixel to ra/dec conversion)",
        indent=indent,
    )

    #   Define WCS dir
    if wcs_working_dir is None:
        wcs_working_dir = image.out_path / "wcs_images"
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
    filename = basename + ".new"
    filepath = Path(wcs_working_dir / filename)

    #   String passed to the shell
    pixel_scale = image.pixel_scale
    pixel_scale_low = pixel_scale - 0.1
    pixel_scale_up = pixel_scale + 0.1
    command: str = (
        f"solve-field --overwrite --scale-units arcsecperpix --scale-low "
        + f"{pixel_scale_low} --scale-high {pixel_scale_up} --ra {ra} "
        + f"--dec {dec} --radius 1.0 --dir {wcs_working_dir} --resort "
        + '"{}" --fits-image -z 2'.format(str(wcs_file).replace(" ", "\ "))
    )

    #   Running the command
    command_result = subprocess.run(
        [command],
        shell=True,
        text=True,
        capture_output=True,
    )

    return_code = command_result.returncode
    fits_created = command_result.stdout.find("Creating new FITS file")
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
        style_name="OKGREEN",
    )

    #   Get image hdu list
    hdu_list = fits.open(filepath)

    #   Extract the WCS
    derived_wcs = wcs.WCS(hdu_list[0].header)

    image.wcs = derived_wcs
    return derived_wcs


def find_wcs_twirl(
    image: Image,
    object_pixel_position_x: np.ndarray | None = None,
    object_pixel_position_y: np.ndarray = None,
    indent: int = 2,
) -> wcs.WCS:
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
    import twirl

    terminal_output.print_to_terminal(
        "Searching for a WCS solution (pixel to ra/dec conversion)",
        indent=indent,
    )

    #   Arrange object positions
    object_pixel_position_x = np.array(object_pixel_position_x)
    object_pixel_position_y = np.array(object_pixel_position_y)
    objects = np.column_stack((object_pixel_position_x, object_pixel_position_y))

    #   Limit the number of objects to 50
    if len(objects) > 50:
        n = 50
    else:
        n = len(objects)
    objects = objects[0:n]

    coordinates = image.coordinates_image_center
    field_of_view = image.field_of_view_x
    print(
        "n", n, "field_of_view", field_of_view, coordinates.ra.deg, coordinates.dec.deg
    )
    #   Calculate WCS
    gaia_twirl = twirl.gaia_radecs(
        [coordinates.ra.deg, coordinates.dec.deg],
        field_of_view / 60,
        limit=300,
    )
    derived_wcs = twirl._compute_wcs(objects, gaia_twirl, n=n)

    gaia_twirl_pixel = np.array(
        SkyCoord(gaia_twirl, unit="deg").to_pixel(derived_wcs)
    ).T
    print("gaia_twirl_pixel")
    print(gaia_twirl_pixel)
    print(gaia_twirl_pixel.T)
    print("objects")
    print(objects)

    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.plot(*objects.T, "o", fillstyle="none", c="b", ms=12)
    plt.plot(*gaia_twirl_pixel.T, "o", fillstyle="none", c="C1", ms=18)
    plt.savefig("/tmp/test_twirl.pdf", bbox_inches="tight", format="pdf")
    plt.show()

    print(derived_wcs)

    terminal_output.print_to_terminal(
        "WCS solution found :)",
        indent=indent,
        style_name="OKGREEN",
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
    field_of_view = image.field_of_view_x / 60.0

    #   Path to image
    wcs_file = image.path

    #   String passed to the shell
    command = 'astap_cli -f "{}" -r 3 -fov {} -update'.format(wcs_file, field_of_view)

    #   Running the command
    command_result = subprocess.run(
        [command],
        shell=True,
        text=True,
        capture_output=True,
    )

    return_code = command_result.returncode
    solution_found = command_result.stdout.find("Solution found:")
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
        style_name="OKGREEN",
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
    bool
        Is `True` if the image header contains valid WCS information.

    wcs_file
        Path to the image with the WCS
    """
    from .utilities import get_basename

    #   Path to image
    wcs_file = image.path

    #   Get WCS of the original image
    wcs_original = wcs.WCS(fits.open(wcs_file)[0].header)

    #   Determine wcs type of original WCS
    wcs_original_type = wcs_original.get_axis_types()[0]["coordinate_type"]

    if wcs_original_type == "celestial":
        terminal_output.print_to_terminal(
            "Image contains already a valid WCS.",
            indent=indent,
            style_name="OKGREEN",
        )
        return True, wcs_file
    else:
        #   Check if an image with a WCS in the astronomy.net format exists
        #   in the wcs directory (`wcs_dir`)

        #   Set WCS dir
        if wcs_dir is None:
            wcs_dir = image.out_path / "wcs_images"

        #   Get image base name
        basename = get_basename(image.path)

        #   Compose file name
        filename = f"{basename}.new"
        filepath = Path(wcs_dir / filename)

        if filepath.is_file():
            #   Get WCS
            wcs_astronomy_net = wcs.WCS(fits.open(filepath)[0].header)

            #   Determine wcs type
            wcs_astronomy_net_type = wcs_astronomy_net.get_axis_types()[0][
                "coordinate_type"
            ]

            if wcs_astronomy_net_type == "celestial":
                terminal_output.print_to_terminal(
                    "Image found in wcs_dir with a valid WCS.",
                    indent=indent,
                    style_name="OKGREEN",
                )
                return True, filepath

        return False, ""
