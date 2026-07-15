"""Reduction workflow: stack module."""

import shutil
from pathlib import Path

import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import mad_std

from ... import checks, style, terminal_output
from ...core.parallel import Executor
from .. import utilities

def stack_filter_images(
    images_to_combine: list[str],
    stacking_method: str,
    dtype: str | np.dtype | None,
    filter_: str,
    out_path: Path,
    new_target_name: str | None,
) -> str:
    """Combine images for one filter and write the stacked file."""
    combined_image = ccdp.combine(
        images_to_combine,
        method=stacking_method,
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        mem_limit=15e9,
        dtype=dtype,
    )
    utilities.update_header_information(
        combined_image,
        len(images_to_combine),
        new_target_name,
    )
    file_name = "combined_filter_{}.fit".format(filter_.replace("''", "p"))
    combined_image.write(out_path / file_name, overwrite=True)
    return file_name


def stack_image(
    image_path: Path,
    output_dir: Path,
    image_type_list: list[str],
    stacking_method: str = "average",
    dtype: str | np.dtype | None = None,
    new_target_name: str | None = None,
    debug: bool = False,
    n_cores_multiprocessing: int | None = None,
) -> None:
    """
    Combine images

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type_list
        Header keyword characterizing the image type for which the
        shifts shall be determined

    stacking_method
        Method used for combining the images.
        Possibilities: ``median`` or ``average`` or ``sum``
        Default is ``average`.

    dtype
        The dtype that should be used while combining the images.
        Default is ''None''. -> None is equivalent to float64

    new_target_name
        Name of the target. If not None, this target name will be written
        to the FITS header.
        Default is ``None``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.
    """
    terminal_output.print_to_terminal("Stack light images...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   New image collection for the images
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Check if image_file_collection is not empty
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No FITS files found in {file_path}. "
            f"=> EXIT {style.Bcolors.ENDC}"
        )

    #   Determine filter
    image_type = utilities.get_image_type(
        image_file_collection,
        image_type_list,
    )
    filters: set[str] = set(
        h["filter"] for h in image_file_collection.headers(imagetyp=image_type)
    )

    filter_jobs: list[tuple[str, list[str]]] = []
    for filter_ in filters:
        images_to_combine = image_file_collection.files_filtered(
            imagetyp=image_type,
            filter=filter_,
            include_path=True,
        )
        if images_to_combine:
            filter_jobs.append((filter_, images_to_combine))

    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(filter_jobs),
        add_progress_bar=True,
    )
    for filter_, images_to_combine in filter_jobs:
        executor.schedule(
            stack_filter_images,
            args=(images_to_combine, stacking_method, dtype, filter_, out_path, new_target_name),
        )

    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Stacking light images using multiprocessing"
            f" failed :({style.Bcolors.ENDC}"
        )
    executor.wait()

    #   Remove individual reduced images
    if not debug:
        shutil.rmtree(file_path, ignore_errors=True)
