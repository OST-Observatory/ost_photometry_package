"""Instrument and camera-related utilities for data reduction."""

import ccdproc as ccdp
import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table

from .. import style, terminal_output
from ..camera_specs import (
    GENERIC_INSTRUMENT_NAMES,
    UNKNOWN_READOUT_MODE,
    is_unknown_readout_mode,
    normalize_instrument_name,
)

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


def _unique_summary_value(
    image_file_collection: ccdp.ImageFileCollection, column_name: str, label: str
) -> int | float | None:
    """Single value of a summary column; raise if several values are present."""
    column = _get_summary_column(image_file_collection, column_name)
    if column is None:
        return 1 if "bin" in column_name else None
    values = set(column[np.invert(np.ma.getmaskarray(column))])
    if len(values) > 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Multiple {label} values detected.\n"
            f"This is not supported -> EXIT \n{style.Bcolors.ENDC}"
        )
    return next(iter(values)) if values else None


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
    that case ``"default"`` is returned. For QHY cameras a missing keyword or
    a placeholder (MaxIm DL writes its exposure-tab setting "Fast" /
    "Normal" / "Slow", not the driver mode) gives ``"unknown"``: gain and
    read noise are then measured or taken from the OST measurements
    (``camera_noise_source="auto"``), not from a guessed mode. Numeric QHY
    modes 0-3 are translated to their names.
    """
    summary = image_file_collection.summary
    if not isinstance(summary, Table):
        raise ValueError(
            f"{style.Bcolors.FAIL} \nReadout mode keyword for FITS Header "
            "could not be determined. Summary table of image file collection "
            f"is not available. -> ABORT {style.Bcolors.ENDC}"
        )
    qhy = instrument in _QHY_CAMERAS

    readout_mode_keyword = _get_readout_mode_keyword(summary)
    if readout_mode_keyword is None:
        if qhy:
            terminal_output.print_to_terminal(
                "No readout mode keyword in the FITS headers: readout mode unknown; "
                "gain and read noise are measured or taken from the OST measurements.",
                style_name="WARNING",
                indent=1,
            )
            return UNKNOWN_READOUT_MODE
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
    if qhy:
        numeric = {0: "PhotoGraphic DSO", 1: "High Gain Mode", 2: "Extend Fullwell",
                   3: "Extend Fullwell 2CMS"}
        readout_modes = [numeric.get(m, m) if not isinstance(m, str) else m
                         for m in readout_modes]
        placeholders = [m for m in readout_modes if is_unknown_readout_mode(m)]
        if not readout_modes or len(placeholders) == len(readout_modes):
            labels = ", ".join(repr(str(m)) for m in placeholders) or "none"
            terminal_output.print_to_terminal(
                f"Readout mode unknown (READOUTM {labels} is the MaxIm DL exposure setting, "
                "not the QHY driver mode); gain and read noise are measured or taken "
                "from the OST measurements.",
                style_name="WARNING",
                indent=1,
            )
            return UNKNOWN_READOUT_MODE
        readout_modes = [m for m in readout_modes if not is_unknown_readout_mode(m)]

    if len(readout_modes) > 1:
        if ignore_readout_mode_mismatch:
            readout_mode = readout_modes[0]
            terminal_output.print_to_terminal(
                f"Multiple readout modes detected. Use first one "
                f"detected: {readout_mode}",
                style_name="WARNING",
            )
            return readout_mode
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Multiple readout modes detected.\n"
            f"This is currently not supported -> EXIT \n{style.Bcolors.ENDC}"
        )
    if len(readout_modes) == 1:
        return readout_modes[0]
    return "default"


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

    if instrument in GENERIC_INSTRUMENT_NAMES:
        x_dimension = _unique_summary_value(image_file_collection, "naxis1", "image dimension")
        y_dimension = _unique_summary_value(image_file_collection, "naxis2", "image dimension")
        x_bin = _unique_summary_value(image_file_collection, "xbinning", "binning")
        y_bin = _unique_summary_value(image_file_collection, "ybinning", "binning")
        instrument = normalize_instrument_name(
            instrument, x_dimension, y_dimension, x_bin, y_bin
        )
    else:
        instrument = normalize_instrument_name(instrument)

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
            f"Found file without temperature information: "
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
