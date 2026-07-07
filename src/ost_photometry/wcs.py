"""WCS (World Coordinate System) determination utilities."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits

from . import checks, style, terminal_output

if TYPE_CHECKING:
    from .utilities import Image  # noqa: F401

_ASTAP_UINT16_MAX = 60000


def _astap_field_of_view_degrees(image: Image) -> float:
    """Return the image height in degrees for ASTAP's ``-fov`` option."""
    if image.field_of_view_y is not None:
        return image.field_of_view_y / 60.0

    if image.field_of_view_x is not None:
        terminal_output.print_to_terminal(
            "WARNING: field_of_view_y not available; using field_of_view_x "
            "for ASTAP.",
            style_name="WARNING",
            indent=2,
        )
        return image.field_of_view_x / 60.0

    raise RuntimeError(
        f"{style.Bcolors.FAIL}Field of view could not be determined for "
        f"ASTAP -> EXIT{style.Bcolors.ENDC}"
    )


def _needs_astap_preprocessing(data: np.ndarray, bitpix: int) -> bool:
    """Return whether the FITS data should be converted for ASTAP."""
    if bitpix in (-32, -64):
        return True
    if np.issubdtype(data.dtype, np.floating):
        return True
    if not np.all(np.isfinite(data)):
        return True
    return bool(np.any(data < 0))


def _scale_image_for_astap(data: np.ndarray) -> np.ndarray:
    """Convert image data to unsigned 16-bit values suitable for ASTAP."""
    clean = np.nan_to_num(np.asarray(data, dtype=np.float64))
    clean = np.where(np.isfinite(clean), clean, 0.0)

    positive = clean[clean > 0]
    if positive.size == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Image contains no positive pixels; "
            f"ASTAP preprocessing failed -> EXIT{style.Bcolors.ENDC}"
        )

    background = float(np.percentile(positive, 5)) if positive.size > 10 else 0.0
    scaled = np.clip(clean - background, 0, None)
    maximum = float(scaled.max())
    if maximum <= 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Image contains no usable signal after "
            f"ASTAP preprocessing -> EXIT{style.Bcolors.ENDC}"
        )

    return (scaled / maximum * _ASTAP_UINT16_MAX).astype(np.uint16)


def _prepare_astap_fits(source_path: Path, working_dir: Path) -> tuple[Path, bool]:
    """
    Return an ASTAP-compatible FITS path.

    Calibrated or stacked images are written as a temporary unsigned 16-bit
    copy when required. The boolean indicates whether a temporary file was
    created and should be removed afterwards.
    """
    checks.check_output_directories(working_dir)

    with fits.open(source_path) as hdul:
        data = hdul[0].data
        if data is None:
            raise RuntimeError(
                f"{style.Bcolors.FAIL}FITS file contains no image data: "
                f"{source_path}{style.Bcolors.ENDC}"
            )

        bitpix = int(hdul[0].header.get("BITPIX", 0))
        if not _needs_astap_preprocessing(data, bitpix):
            return source_path, False

        header = hdul[0].header.copy()
        astap_data = _scale_image_for_astap(data)

    for keyword in ("BZERO", "BSCALE", "BLANK"):
        if keyword in header:
            del header[keyword]

    header["BITPIX"] = 16
    header.add_history(
        "Temporary unsigned 16-bit copy for ASTAP plate solving.",
        before=0,
    )

    temp_file = tempfile.NamedTemporaryFile(
        prefix=f"{source_path.stem}_astap_",
        suffix=".fit",
        dir=working_dir,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    fits.writeto(temp_path, astap_data, header, overwrite=True)
    return temp_path, True


def _strip_wcs_keywords(header: fits.Header) -> None:
    """Remove existing WCS keywords before writing a new solution."""
    remove_keys = [
        key
        for key in header
        if key.startswith(
            (
                "CTYPE",
                "CRVAL",
                "CRPIX",
                "CDELT",
                "CROTA",
                "CUNIT",
                "PC",
                "CD",
                "PV",
            )
        )
        or key in ("RADESYS", "EQUINOX", "LONPOLE", "LATPOLE")
    ]
    for key in remove_keys:
        del header[key]


def _apply_wcs_to_fits(target_path: Path, solved_wcs: wcs.WCS) -> wcs.WCS:
    """Write an ASTAP WCS solution to ``target_path`` and return it."""
    wcs_header = solved_wcs.to_header(relax=True)
    with fits.open(target_path, mode="update") as hdul:
        _strip_wcs_keywords(hdul[0].header)
        hdul[0].header.update(wcs_header)
        hdul.flush()

    return solved_wcs


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

    #   Invoke solve-field without shell=True so paths with spaces, plus signs,
    #   or odd quoting never need manual escaping or optional quotes.
    pixel_scale = image.pixel_scale
    pixel_scale_low = pixel_scale - 0.1
    pixel_scale_up = pixel_scale + 0.1
    cmd = [
        "solve-field",
        "--overwrite",
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        str(pixel_scale_low),
        "--scale-high",
        str(pixel_scale_up),
        "--ra",
        str(ra),
        "--dec",
        str(dec),
        "--radius",
        "1.0",
        "--dir",
        str(wcs_working_dir),
        "--resort",
        str(wcs_file),
        "--fits-image",
        "-z",
        "2",
    ]

    command_result = subprocess.run(
        cmd,
        shell=False,
        text=True,
        capture_output=True,
    )

    return_code = command_result.returncode
    fits_created = command_result.stdout.find("Creating new FITS file")
    if return_code != 0 or fits_created == -1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo wcs solution could be found for "
            f"the images!\n {style.Bcolors.ENDC}{style.Bcolors.BOLD}"
            f"The command was:\n {shlex.join(cmd)} \nDetailed error output:\n"
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
        f" for image {image.image_id}",
        indent=indent,
    )

    field_of_view = _astap_field_of_view_degrees(image)
    source_path = Path(image.path)
    working_dir = image.out_path / "wcs_images"
    astap_path, is_temporary = _prepare_astap_fits(source_path, working_dir)

    if is_temporary:
        terminal_output.print_to_terminal(
            "Created temporary 16-bit FITS copy for ASTAP.",
            indent=indent,
            style_name="WARNING",
        )

    cmd = [
        "astap_cli",
        "-f",
        str(astap_path),
        "-r",
        "3",
        "-fov",
        f"{field_of_view:.10g}",
        "-update",
    ]

    try:
        command_result = subprocess.run(
            cmd,
            shell=False,
            text=True,
            capture_output=True,
        )

        return_code = command_result.returncode
        solution_found = command_result.stdout.find("Solution found:")
        if return_code != 0 or solution_found == -1:
            raise RuntimeError(
                f"{style.Bcolors.FAIL} \nNo wcs solution could be found for "
                f"the images!\n {style.Bcolors.ENDC}{style.Bcolors.BOLD}"
                f"The command was:\n{shlex.join(cmd)} \nDetailed error output:\n"
                f"{style.Bcolors.ENDC}{command_result.stdout}{command_result.stderr}"
                f"{style.Bcolors.ENDC}{style.Bcolors.FAIL}Exit{style.Bcolors.ENDC}"
            )

        terminal_output.print_to_terminal(
            "WCS solution found :)",
            indent=indent,
            style_name="OKGREEN",
        )

        with fits.open(astap_path) as solved_hdul:
            solved_wcs = wcs.WCS(solved_hdul[0].header)

        if is_temporary:
            derived_wcs = _apply_wcs_to_fits(source_path, solved_wcs)
        else:
            derived_wcs = solved_wcs
    finally:
        if is_temporary:
            astap_path.unlink(missing_ok=True)

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
