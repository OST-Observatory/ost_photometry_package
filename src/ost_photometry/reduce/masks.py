"""Pixel mask utilities for data reduction."""

import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.nddata import CCDData

from .. import checks, terminal_output
from .image_collection import image_file_collection as make_image_file_collection


def get_pixel_mask(out_path: Path, shape: np.ndarray) -> tuple[bool, CCDData]:
    """
    Calculates or loads a pixel mask highlighting bad and hot pixel.

    Tries to load a precalculated bad pixel mask. If that fails tries to
    load pixel masks calculated by the 'master_dark' and 'master_flat'
    routine and combine those. Assumes default names for the individual
    masks.

    Parameters
    ----------
    out_path
        Path pointing to the main storage location

    shape
        2D array with image dimensions. Is used to check if a
        precalculate mask fits to the image.

    Returns
    -------
    success
        True if either a precalculate bad pixel mask has been found or
        if masks calculated by the 'master_dark' and 'master_flat' have
        been found.

    mask
        Precalculated or combined pixel mask
    """
    #   Load pixel mask
    try:
        mask = CCDData.read(out_path / "bad_pixel_mask.fit")
        if mask.shape == shape:
            #   If shape is the same, set success to True.
            success = True
        else:
            terminal_output.print_to_terminal(
                "No default bad pixel mask available. Try to use "
                "the mask calculated in the data reduction...",
                indent=1,
                style_name="WARNING",
            )
            #   Raise RuntimeError to trigger except.
            raise RuntimeError("")
    except (FileNotFoundError, RuntimeError):
        #   If no precalculated mask are available, try to load masks
        #   calculated by 'master_dark' and 'master_flat'

        try:
            #   Set default masks
            mask_hot_pixel = np.zeros(shape, dtype=bool)
            mask_bad_pixel = np.zeros(shape, dtype=bool)

            #   New image collection
            image_file_collection = make_image_file_collection(out_path)

            #   Get hot pixel masks
            ifc_hot_pixel = image_file_collection.filter(imagetyp="dark mask")

            #   Get correct mask in terms of binning
            for mask_data, _file_name in ifc_hot_pixel.data(return_fname=True):
                if mask_data.shape == shape:
                    mask_hot_pixel = mask_data.astype("bool")

            #   Get bad pixel masks
            ifc_bad_pixel = image_file_collection.filter(imagetyp="flat mask")

            #   Get correct mask in terms of binning
            for mask_data, _file_name in ifc_bad_pixel.data(return_fname=True):
                if mask_data.shape == shape:
                    mask_bad_pixel = mask_data.astype("bool")

            #   Combine mask
            mask = np.logical_or(mask_hot_pixel, mask_bad_pixel)
            success = True
        except ValueError:
            terminal_output.print_to_terminal(
                "No bad pixel mask available. Skip adding bad pixel mask.",
                indent=1,
                style_name="WARNING",
            )
            mask = np.zeros(shape, dtype=bool)
            success = False

    return success, mask


def make_hot_pixel_mask(
    dark_image: CCDData,
    gain: float | None,
    output_dir: str | Path,
    verbose: bool = False,
) -> None:
    """
    Make a hot pixel mask from a dark frame

    Parameters
    ----------
    dark_image
        Dark image

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.

    output_dir
        Path to the directory where the master files should be saved to

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.
    """
    #   Sanitize the provided paths
    out_path = checks.check_pathlib_path(output_dir)

    #   Get exposure time
    exposure_time = dark_image.header["EXPTIME"]

    #   Get image shape
    image_dimension_x = dark_image.meta["naxis1"]
    image_dimension_y = dark_image.meta["naxis2"]

    #   Scale image with exposure time and gain
    dark_image = dark_image.multiply(gain * u.electron / u.adu)
    dark_image = dark_image.divide(exposure_time * u.second)

    #   Number of pixel
    n_pixel = dark_image.shape[1] * dark_image.shape[0]

    #   Calculate the hot pixel mask. Increase the threshold if the number of
    #   hot pixels is unrealistically high
    threshold_hot_pixel = 2
    hot_pixel_sum = 0
    hot_pixels = np.zeros(dark_image.shape)
    for _ in range(0, 100):
        hot_pixels = dark_image.data > threshold_hot_pixel
        hot_pixel_sum = hot_pixels.sum()
        #   Check if number of hot pixel is realistic
        if hot_pixel_sum / n_pixel <= 0.03:
            break
        threshold_hot_pixel += 1

    if verbose:
        sys.stdout.write(f"\r\tNumber of hot pixels: {hot_pixel_sum}\n")
        sys.stdout.write(f"\r\tLimit (e-/s/pix) used: {threshold_hot_pixel}\n")
        sys.stdout.flush()

    #   Save mask with hot pixels
    mask_as_ccd_data_object = CCDData(
        data=hot_pixels.astype("uint8"),
        unit=u.dimensionless_unscaled,
    )
    mask_as_ccd_data_object.header["imagetyp"] = "dark mask"
    file_name = f"mask_from_dark_{image_dimension_x}x{image_dimension_y}.fit"
    mask_as_ccd_data_object.write(out_path / file_name, overwrite=True)


def make_bad_pixel_mask(
    bad_pixel_mask_list: list[np.ndarray], output_dir: str | Path, verbose: bool = False
) -> None:
    """
    Calculate a bad pixel mask from a list of bad pixel masks

    Parameters
    ----------
    bad_pixel_mask_list
        List with bad pixel masks

    output_dir
        Path to the directory where the master files should be saved to

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.
    """
    #   Sanitize the provided paths
    out_path = checks.check_pathlib_path(output_dir)

    #   Get information on the image dimensions/binning
    mask_shape_list = []
    for bad_pixel_mask in bad_pixel_mask_list:
        mask_shape_list.append(bad_pixel_mask.shape)
    mask_shape_set = set(mask_shape_list)

    #   Loop over all image shapes (binning options)
    for shape in mask_shape_set:
        #   Calculate overall bad pixel mask
        combined_mask = np.zeros(shape)
        for bad_pixel_mask in bad_pixel_mask_list:
            if bad_pixel_mask.shape == shape:
                combined_mask = np.logical_or(combined_mask, bad_pixel_mask)

        if verbose:
            terminal_output.print_to_terminal(
                f"Number of bad pixels ({shape}): {combined_mask.sum()}",
                indent=1,
            )

        #   Save mask
        mask_as_ccd_data_object = CCDData(
            data=combined_mask.astype("uint8"),
            unit=u.dimensionless_unscaled,
        )
        mask_as_ccd_data_object.header["imagetyp"] = "flat mask"
        file_name = f"mask_from_ccdmask_{shape[1]}x{shape[0]}.fit"
        mask_as_ccd_data_object.write(out_path / file_name, overwrite=True)
