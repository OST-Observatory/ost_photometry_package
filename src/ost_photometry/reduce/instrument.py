"""Instrument and camera-related utilities for data reduction."""

import numpy as np
import ccdproc as ccdp
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.stats import sigma_clip

from .. import style, terminal_output


def get_instruments(image_file_collection: ccdp.ImageFileCollection) -> set[str] | None:
    """
    Extract instrument information.

    Parameters
    ----------
    image_file_collection
        Image file collection with all images

    Returns
    -------
    instruments
        List of instruments
    """
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No images found -> EXIT\n"
            f"\t=> Check paths to the images!{style.Bcolors.ENDC}"
        )

    if image_file_collection.summary is not None:
        instruments: set[str] = set(image_file_collection.summary["instrume"])
    else:
        terminal_output.print_to_terminal(
            "WARNING: Instruments could not be determined because the image "
            "file collection does not contain a summery -> Returning None",
            style_name="WARNING",
        )
        return None

    return instruments


def get_instrument_info(
    image_file_collection: ccdp.ImageFileCollection,
    temperature_tolerance: float,
    ignore_readout_mode_mismatch: bool = False,
    ignore_instrument_mismatch: bool = False,
) -> tuple[str, str, int | None, int, float]:
    """
    Extract information regarding the instruments and readout mode.
    Currently the instrument and readout mode need to be unique. An
    exception will be raised in case multiple readout modes or
    instruments are detected.
    """
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No images found -> EXIT\n"
            f"\t=> Check paths to the images!{style.Bcolors.ENDC}"
        )

    instrument_mask = image_file_collection.summary["instrume"].mask
    files_without_instrument = np.array(image_file_collection.files)[instrument_mask]
    for file_name in files_without_instrument:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without instrument information: \n "
            f"{file_name} \n Skip file.",
            style_name="WARNING",
            indent=2,
        )

    instruments = set(
        image_file_collection.summary["instrume"][np.invert(instrument_mask)]
    )

    if len(instruments) > 1:
        if ignore_instrument_mismatch:
            terminal_output.print_to_terminal(
                f"Multiple instruments detected: {instruments} Will use first one.",
                style_name="WARNING",
            )
        else:
            raise RuntimeError(
                f"{style.Bcolors.FAIL}Multiple instruments detected.\n"
                f"This is currently not supported -> EXIT \n{style.Bcolors.ENDC}"
            )
    instrument = list(instruments)[0]

    if "QHY268M" in instrument:
        instrument = "QHY268M"
    if "QHY600M" in instrument:
        instrument = "QHY600M"

    if instrument in ["QHYCCD-Cameras-Capture", "QHYCCD-Cameras2-Capture"]:
        x_dimensions = set(image_file_collection.summary["naxis1"])
        if len(x_dimensions) > 1:
            raise RuntimeError(
                f"{style.Bcolors.FAIL}Multiple image dimensions detected.\n"
                f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
            )
        x_dimension = list(x_dimensions)[0]

        y_dimensions = set(image_file_collection.summary["naxis2"])
        if len(y_dimensions) > 1:
            raise RuntimeError(
                f"{style.Bcolors.FAIL}Multiple image dimensions detected.\n"
                f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
            )
        y_dimension = list(y_dimensions)[0]

        x_bins = set(image_file_collection.summary["xbinning"])
        if len(x_bins) > 1:
            raise RuntimeError(
                f"{style.Bcolors.FAIL}Multiple binning values detected.\n"
                f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
            )
        x_bin = list(x_bins)[0]

        y_bins = set(image_file_collection.summary["ybinning"])
        if len(y_bins) > 1:
            raise RuntimeError(
                f"{style.Bcolors.FAIL}Multiple binning values detected.\n"
                f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
            )
        y_bin = list(y_bins)[0]

        x_dimension_physical = x_dimension * x_bin
        y_dimension_physical = y_dimension * y_bin

        if x_dimension_physical == 9576 and y_dimension_physical in [6388, 6387]:
            instrument = "QHY600M"
        elif x_dimension_physical in [6280, 6279] and y_dimension_physical in [
            4210,
            4209,
        ]:
            instrument = "QHY268M"
        elif x_dimension_physical == 3864 and y_dimension_physical in [2180, 2178]:
            instrument = "QHY485C"
        else:
            instrument = ""

    readout_mode = "default"

    if isinstance(image_file_collection.summary, Table):
        if "readoutm" in image_file_collection.summary.colnames:
            readout_mode_keyword = "readoutm"
        elif "readmode" in image_file_collection.summary.colnames:
            readout_mode_keyword = "readmode"
        else:
            raise KeyError(
                f"{style.Bcolors.FAIL} \nReadout mode keyword for FITS Header could not"
                f" be determined -> ABORT {style.Bcolors.ENDC}"
            )
    else:
        raise ValueError(
            f"{style.Bcolors.FAIL} \nReadout mode keyword for FITS Header "
            "could notbe determined. Summary table of image file collectiont "
            f"is not available. -> ABORT {style.Bcolors.ENDC}"
        )

    readout_mode_mask = image_file_collection.summary[readout_mode_keyword].mask
    files_without_readout_mode = np.array(image_file_collection.files)[
        readout_mode_mask
    ]
    for file_name in files_without_readout_mode:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without readout mode information: \n "
            f"{file_name} \n Skip file.",
            style_name="WARNING",
            indent=2,
        )

    readout_modes = list(
        set(
            image_file_collection.summary[readout_mode_keyword][
                np.invert(readout_mode_mask)
            ]
        )
    )

    if len(readout_modes) > 1:
        if ignore_readout_mode_mismatch:
            readout_mode = readout_modes[0]
            terminal_output.print_to_terminal(
                f"Multiple readout modes detected. Use first one "
                f"detected: {readout_mode}",
                style_name="WARNING",
            )
        else:
            raise RuntimeError(
                f"{style.Bcolors.FAIL}Multiple readout modes detected.\n"
                f"This is currently not supported -> EXIT \n{style.Bcolors.ENDC}"
            )

    if instrument in ["QHY600M", "QHY268M"]:
        if not readout_modes:
            readout_mode = "Extend Fullwell 2CMS"
        elif len(readout_modes) == 1:
            readout_mode = list(readout_modes)[0]
            if readout_mode in ["Fast", "Slow", "Normal"]:
                readout_mode = "Extend Fullwell 2CMS"
            if readout_mode == 0:
                readout_mode = "PhotoGraphic DSO"
            elif readout_mode == 1:
                readout_mode = "High Gain Mode"
            elif readout_mode == 2:
                readout_mode = "Extend Fullwell"
            elif readout_mode == 3:
                readout_mode = "Extend Fullwell 2CMS"
        elif ignore_readout_mode_mismatch:
            terminal_output.print_to_terminal(
                "WARNING: Multiple readout modes detected. Assume Extend Fullwell 2CMS",
                style_name="WARNING",
                indent=2,
            )
            readout_mode = "Extend Fullwell 2CMS"

    gain_mask = image_file_collection.summary["gain"].mask
    files_without_gain = np.array(image_file_collection.files)[gain_mask]
    for file_name in files_without_gain:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without gain information: \n "
            f"{file_name} \n Skip file.",
            style_name="WARNING",
            indent=2,
        )

    gain_settings = set(image_file_collection.summary["gain"][np.invert(gain_mask)])
    if len(gain_settings) > 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Multiple gain values detected.\n"
            f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
        )
    gain_setting = list(gain_settings)[0]

    offset_mask = image_file_collection.summary["offset"].mask
    files_without_offset = np.array(image_file_collection.files)[offset_mask]
    for file_name in files_without_offset:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without offset information: \n "
            f"{file_name} \n Skip file.",
            style_name="WARNING",
            indent=2,
        )

    offset_settings = set(
        image_file_collection.summary["offset"][np.invert(offset_mask)]
    )
    if len(offset_settings) > 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Multiple offset values detected.\n"
            f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
        )

    pixel_bit_mask = image_file_collection.summary["bitpix"].mask
    files_without_pixel_bit = np.array(image_file_collection.files)[pixel_bit_mask]
    for file_name in files_without_pixel_bit:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without pixel bit information: \n "
            f"{file_name} \n Skip file.",
            style_name="WARNING",
            indent=2,
        )

    pixel_bit_set = set(
        image_file_collection.summary["bitpix"][np.invert(pixel_bit_mask)]
    )
    if len(pixel_bit_set) > 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Multiple bit values detected.\n"
            f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
        )
    pixel_bit_value = list(pixel_bit_set)[0]

    mask = image_file_collection.summary["ccd-temp"].mask
    files_without_ccd_temperature = np.array(image_file_collection.files)[mask]
    for file_name in files_without_ccd_temperature:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without temperature information: "
            f"{file_name} -> Skip file.",
            style_name="WARNING",
            indent=2,
        )

    files_with_ccd_temperature = np.array(image_file_collection.files)[np.invert(mask)]
    temperatures = image_file_collection.summary["ccd-temp"][np.invert(mask)]

    if temperatures.fill_value == "?":
        temperatures.fill_value = 999.0
    if temperatures.dtype == "object":
        temperatures = temperatures.astype(float)

    median_temperature = np.median(temperatures)
    std_temperature = np.std(temperatures)

    if std_temperature > temperature_tolerance:
        clipped_temperatures_mask = sigma_clip(temperatures, sigma=2.0).mask
        clipped_temperatures = temperatures[clipped_temperatures_mask]
        clipped_images = files_with_ccd_temperature[clipped_temperatures_mask]
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Significant temperature difference "
            f"detected. The median temperature is {median_temperature}°C."
            f"The following images have temperatures (°C) of: \n"
            f"{clipped_temperatures.value} \n {clipped_images} \n{style.Bcolors.ENDC}"
        )

    return instrument, readout_mode, gain_setting, pixel_bit_value, median_temperature


def get_imaging_software(image_file_collection: ccdp.ImageFileCollection) -> set[str]:
    """Extract imaging software version."""
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No images found -> EXIT\n"
            f"\t=> Check paths to the images!{style.Bcolors.ENDC}"
        )
    return set(image_file_collection.summary["swcreate"])
