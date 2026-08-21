"""Reduction workflow: dispatch module."""

from pathlib import Path

from ... import style
from .dark import master_dark
from .flat import master_flat
from .science import reduce_light


def master_image_list(*args, **kwargs):
    """
    Wrapper function to create a master calibration image for the files
    in the directories given in the path list 'paths'
    """
    if kwargs["calib_type"] == "dark":
        master_dark(*args, **kwargs)
    elif kwargs["calib_type"] == "flat":
        master_flat(*args, **kwargs)


def reduce_master(paths, *args, **kwargs):
    """
    Wrapper function for reduction of the science images

    Parameters
    ----------
    paths           : `list of strings`
        List with paths to the images
    """
    if isinstance(paths, list):
        for path in paths:
            reduce_light(path, *args, **kwargs)
    elif isinstance(paths, str) or isinstance(paths, Path):
        reduce_light(paths, *args, **kwargs)
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Supplied path is neither str nor list"
            f"{style.Bcolors.ENDC}"
        )


