############################################################################
#                               Libraries                                  #
############################################################################

import io
import json
import os
import random
import select
import string
import sys
import time
from pathlib import Path

import yaml

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


def _stdin_fileno() -> int | None:
    stdin = sys.stdin
    if stdin is None or not hasattr(stdin, "fileno"):
        return None
    try:
        return int(stdin.fileno())
    except (ValueError, OSError, io.UnsupportedOperation):
        return None


def _normalize_input(raw: str | None) -> str:
    return "" if raw is None else str(raw).strip().lower()


def _timeout_reply() -> tuple[str, bool]:
    terminal_output.print_to_terminal(
        "The prompt timed out!",
        indent=2,
        style_name="WARNING",
    )
    return "no", True


def get_input(prompt: str, timeout: int = 30) -> tuple[str, bool]:
    """Ask for a line of input and give up after ``timeout`` seconds.

    Uses :func:`select.select` on stdin (stdlib). On timeout, or if stdin
    cannot be read, returns ``("no", True)`` so callers that treat only
    ``yes``/``y`` as affirmative keep the previous default (rebuild masters /
    re-reduce science frames). ``timeout <= 0`` waits indefinitely.

    The old optional ``pytimedinput`` dependency is not required; without it
    this used to fall back to blocking :func:`input` and never timed out.
    """
    try:
        import termios
        import tty
    except ImportError:
        termios = None  # type: ignore[assignment]
        tty = None  # type: ignore[assignment]

    shown = prompt
    if timeout > 0 and "(timeout" not in prompt:
        shown = f"{prompt.rstrip()} (timeout {int(timeout)}s, default no) "
    sys.stdout.write(shown)
    sys.stdout.flush()

    if timeout <= 0:
        try:
            return _normalize_input(sys.stdin.readline()), False
        except EOFError:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return _timeout_reply()

    fd = _stdin_fileno()
    if fd is None:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return _timeout_reply()

    deadline = time.monotonic() + float(timeout)
    old_term: list | None = None
    use_cbreak = False
    if termios is not None and tty is not None:
        try:
            old_term = termios.tcgetattr(fd)
            tty.setcbreak(fd, termios.TCSADRAIN)
            use_cbreak = True
        except (termios.error, OSError, ValueError, AttributeError):
            use_cbreak = False
            old_term = None

    try:
        chars: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return _timeout_reply()
            ready, _, _ = select.select([sys.stdin], [], [], remaining)
            if not ready:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return _timeout_reply()
            if not use_cbreak:
                line = sys.stdin.readline()
                if not line:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return _timeout_reply()
                return _normalize_input(line), False
            chunk = sys.stdin.read(1)
            if chunk == "":
                sys.stdout.write("\n")
                sys.stdout.flush()
                return _timeout_reply()
            if chunk in "\n\r":
                sys.stdout.write("\n")
                sys.stdout.flush()
                return _normalize_input("".join(chars)), False
            if chunk == "\x03":
                raise KeyboardInterrupt
            if chunk in ("\x7f", "\b"):
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            chars.append(chunk)
            sys.stdout.write(chunk)
            sys.stdout.flush()
    finally:
        if old_term is not None and termios is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            except (termios.error, OSError):
                pass


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
