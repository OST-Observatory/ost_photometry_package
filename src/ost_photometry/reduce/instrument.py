"""Instrument and camera-related utilities for data reduction."""

import numpy as np
import ccdproc as ccdp
from astropy.stats import sigma_clip
from astropy.table import Table

from .. import style, terminal_output

_QHY_CAMERAS = frozenset({"QHY600M", "QHY268M"})


def _get_summary_column(
    image_file_collection: ccdp.ImageFileCollection, column_name: str
) -> np.ma.MaskedArray | None:
    """Return a summary column if present, otherwise ``None``."""
    summary = image_file_collection.summary
    if summary is None or not isinstance(summary, Table):
        return None
    if column_name not in summary.colnames:
        return None
    return summary[column_name]


def _get_unique_header_value(
    image_file_collection: ccdp.ImageFileCollection,
    column_name: str,
    label: str,
    *,
    required: bool = True,
) -> int | float | None:
    """
    Extract a single consistent FITS header value from the file collection.

    Missing values are reported per file and excluded. When ``required`` is
    ``False`` and the keyword is absent from all headers, ``None`` is returned.
    """
    column = _get_summary_column(image_file_collection, column_name)
    if column is None:
        if required:
            raise KeyError(
                f"{style.Bcolors.FAIL}{label} keyword ({column_name!r}) not found "
                f"in FITS headers -> ABORT{style.Bcolors.ENDC}"
            )
        return None

    value_mask = column.mask
    files_without_value = np.array(image_file_collection.files)[value_mask]
    for file_name in files_without_value:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without {label} information: \n "
            f"{file_name} \n Skip file.",
            style_name="WARNING",
            indent=2,
        )

    values = set(column[np.invert(value_mask)])
    if len(values) > 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Multiple {label} values detected.\n"
            f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
        )
    if not values:
        return None
    return list(values)[0]


def _get_readout_mode_keyword(summary: Table) -> str | None:
    """Return the FITS keyword used for readout mode, if present."""
    if "readoutm" in summary.colnames:
        return "readoutm"
    if "readmode" in summary.colnames:
        return "readmode"
    return None


def resolve_readout_mode(
    image_file_collection: ccdp.ImageFileCollection,
    instrument: str,
    *,
    ignore_readout_mode_mismatch: bool = False,
) -> str:
    """
    Extract the camera readout mode from FITS headers.

    Older cameras (e.g. SBIG STF-8300) often lack a readout-mode keyword; in
    that case ``"default"`` is returned. QHY cameras without the keyword
    default to ``"Extend Fullwell 2CMS"``.
    """
    summary = image_file_collection.summary
    if not isinstance(summary, Table):
        raise ValueError(
            f"{style.Bcolors.FAIL} \nReadout mode keyword for FITS Header "
            "could not be determined. Summary table of image file collection "
            f"is not available. -> ABORT {style.Bcolors.ENDC}"
        )

    readout_mode_keyword = _get_readout_mode_keyword(summary)
    if readout_mode_keyword is None:
        if instrument in _QHY_CAMERAS:
            terminal_output.print_to_terminal(
                "No readout mode keyword in FITS headers; "
                "assuming Extend Fullwell 2CMS for QHY camera.",
                style_name="WARNING",
                indent=1,
            )
            return "Extend Fullwell 2CMS"

        terminal_output.print_to_terminal(
            "No readout mode keyword in FITS headers; using default.",
            style_name="WARNING",
            indent=1,
        )
        return "default"

    readout_mode_mask = summary[readout_mode_keyword].mask
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
        set(summary[readout_mode_keyword][np.invert(readout_mode_mask)])
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
    else:
        readout_mode = "default"

    if instrument in _QHY_CAMERAS:
        if not readout_modes:
            readout_mode = "Extend Fullwell 2CMS"
        elif len(readout_modes) == 1:
            readout_mode = readout_modes[0]
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
    elif len(readout_modes) == 1:
        readout_mode = readout_modes[0]

    return readout_mode


def get_egain_from_collection(
    image_file_collection: ccdp.ImageFileCollection,
) -> float | None:
    """Return the electronic gain in e-/ADU from FITS ``EGAIN`` headers."""
    return _get_unique_header_value(
        image_file_collection,
        "egain",
        "EGAIN",
        required=False,
    )


def resolve_system_gain(
    instrument: str,
    gain_setting: int | float | None,
    egain: float | None,
    calibration_gain: float | None,
    user_gain: float | None = None,
) -> float | None:
    """
    Resolve the system gain in e-/ADU.

    For QHY cameras the internal ``GAIN`` setting is mapped to the true gain
    via calibration curves. When ``EGAIN`` is present and differs from the
    driver default of 1.0, it is preferred because it is usually reliable.
    For older cameras without a ``GAIN`` keyword, ``EGAIN`` is used directly.
    """
    if user_gain is not None:
        return user_gain

    if instrument in _QHY_CAMERAS and gain_setting is not None:
        if egain is not None and egain != 1.0:
            return float(egain)
        if calibration_gain is not None:
            return float(calibration_gain)

    if egain is not None:
        return float(egain)

    if calibration_gain is not None:
        return float(calibration_gain)

    return None


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
) -> tuple[str, str, int | float | None, int, float]:
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

    readout_mode = resolve_readout_mode(
        image_file_collection,
        instrument,
        ignore_readout_mode_mismatch=ignore_readout_mode_mismatch,
    )

    gain_setting = _get_unique_header_value(
        image_file_collection,
        "gain",
        "gain",
        required=False,
    )
    if gain_setting is None:
        terminal_output.print_to_terminal(
            "No GAIN keyword in FITS headers; will rely on EGAIN where available.",
            style_name="WARNING",
            indent=1,
        )

    _get_unique_header_value(
        image_file_collection,
        "offset",
        "offset",
        required=False,
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
