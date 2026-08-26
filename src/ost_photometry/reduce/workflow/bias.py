"""Reduction workflow: bias module."""

from pathlib import Path

import ccdproc as ccdp
import numpy as np
from astropy.stats import mad_std

from ... import checks
from .. import utilities


def master_bias(
    bias_path: str | Path,
    output_dir: str | Path,
    image_type: dict[str, list[str]],
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
    dtype: str | np.dtype | None = None,
) -> None:
    """
    This function calculates master biases from individual bias images
    located in one directory.

    Parameters
    ----------
    bias_path            : `string` or `pathlib.Path`
        Path to the images

    output_dir           : `string` or `pathlib.Path`
        Path to the directory where the master files should be saved to

    image_type           : `dictionary`
        Image types of the images. Possibilities: bias, dark, flat,
        light

    trim_x_start
        Number of pixels to trim from the start of the X direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_x_end
        Number of pixels to trim from the end of the X direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_y_start
        Number of pixels to trim from the start of the Y direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_y_end
        Number of pixels to trim from the end of the Y direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    dtype
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64
    """
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(bias_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Create image collection
    image_file_collection = utilities.image_file_collection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        return

    #   Get bias frames
    bias_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class="bias",
    )
    bias_frames = image_file_collection.files_filtered(
        imagetyp=bias_image_type,
        include_path=True,
    )

    #   Combine biases: Average images + sigma clipping to remove outliers,
    #                   set memory limit to 15GB, set unit to 'adu' since
    #                   this is not set in our images -> find better
    #                   solution
    combined_bias = ccdp.combine(
        bias_frames,
        method="average",
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        mem_limit=15e9,
        unit="adu",
        dtype=dtype,
    )

    #   Trimming the image, for example to remove an overscan region
    image_shape = combined_bias.data.shape
    combined_bias = combined_bias[
        trim_y_start : image_shape[0] - trim_y_end,
        trim_x_start : image_shape[1] - trim_x_end,
    ]

    #   Add header keyword to mark the file as a Master
    combined_bias.meta["combined"] = True

    #   Write file to disk
    combined_bias.write(out_path / "combined_bias.fit", overwrite=True)


