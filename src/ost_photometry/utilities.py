############################################################################
#                               Libraries                                  #
############################################################################

import json
import os
import random
import string
import time
from pathlib import Path

import yaml

try:
    from pytimedinput import timedInput

    use_timed_input = True
except ImportError:
    use_timed_input = False

from . import checks, terminal_output
from .image import Image
from .wcs import (
    check_wcs_exists,
    find_wcs_astap,
    find_wcs_astrometry,
    find_wcs_twirl,
    persist_wcs_to_fits,
    sync_image_coordinates_from_wcs,
)

# Backward-compatible re-export; analysis state lives on AnalysisImage.
__all__ = [
    "Image",
    "check_wcs_exists",
    "find_wcs_astap",
    "find_wcs_astrometry",
    "find_wcs_twirl",
    "persist_wcs_to_fits",
    "sync_image_coordinates_from_wcs",
    "mk_file_list",
    "random_string_generator",
    "get_basename",
    "execution_time",
    "indices_to_slices",
    "link_files",
    "read_params_from_json",
    "read_params_from_yaml",
    "get_input",
    "parse_cluster_selection_id",
]

############################################################################
#                           Routines & definitions                         #
############################################################################


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
        for format_ in formats:
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


def _mapping_or_empty(data: object) -> dict:
    """Return ``data`` if it is a mapping, otherwise an empty dict."""
    return data if isinstance(data, dict) else {}


def read_params_from_json(json_file: str) -> dict:
    """
    Read data from JSON file

    Parameters
    ----------
    json_file
        Path to the JSON file

    Returns
    -------

        Dictionary with the data from the JSON file. Missing, invalid, or
        non-mapping payloads yield ``{}``.
    """
    try:
        with open(json_file) as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return {}

    return _mapping_or_empty(data)


def read_params_from_yaml(yaml_file: str) -> dict:
    """
    Read data from YAML file

    Parameters
    ----------
    yaml_file
        Path to the YAML file

    Returns
    -------

        Dictionary with the data from the YAML file. Missing, invalid, empty,
        or non-mapping payloads yield ``{}``.
    """
    try:
        with open(yaml_file) as file:
            data = yaml.safe_load(file)
    except (yaml.YAMLError, FileNotFoundError, OSError):
        return {}

    return _mapping_or_empty(data)


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


def parse_cluster_selection_id(raw: str | None) -> int | None:
    """Extract a non-negative cluster label from terminal input (tolerates control chars)."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)
