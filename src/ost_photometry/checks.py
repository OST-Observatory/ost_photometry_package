############################################################################
#                               Libraries                                  #
############################################################################

import os

from pathlib import Path

from .style import Bcolors


############################################################################
#                           Routines & definitions                         #
############################################################################

def check_path(path: str) -> None:
    """
    Check if paths are valid

    Parameters
    ----------
    path
        The path to check
    """
    if not os.path.isdir(path):
        raise RuntimeError(
            f"{Bcolors.FAIL} \n{path} not found -> Check file name! "
            f"ABORT {Bcolors.ENDC}"
        )


def check_file(file_path: str) -> None:
    """
    Check if file exists

    Parameters
    ----------
    file_path
        File to check
    """
    if not os.path.isfile(file_path):
        raise RuntimeError(
            f"{Bcolors.FAIL} \n{file_path} not found -> Check file name! "
            f"ABORT {Bcolors.ENDC}"
        )


def list_subdirectories(path: str) -> list[str]:
    """
    List subdirectories

    Parameters
    ----------
    path
        The path to directory with subdirectories


    Returns
    -------

        List with the original path and paths to the subdirectories
    """
    #   List sub directories
    subdirectories = os.listdir(path)

    # result = [os.path.join(path,element) for element in subdirectories]
    result = []
    for element in subdirectories:
        new_path = os.path.join(path, element)
        if os.path.isdir(new_path):
            result.append(new_path)
    return [path] + result


def check_dir(path_dict: dict[str, str]) -> None:
    """
    Check whether the directories exist

    Parameters
    ----------
    path_dict
        Dictionary with : Keys - Path identifier; values - Path
    """
    missing = ""
    fail = False
    for var_name, path in path_dict.items():
        if not os.path.isdir(path):
            missing += f"{var_name} ({path}), "
            fail = True
    if fail:
        raise RuntimeError(
            f"{Bcolors.FAIL}\nNo valid {missing} files found "
            f"-> Check directory! {Bcolors.ENDC}"
        )


def check_pathlib_path(path: str | Path) -> Path:
    """
    Check if the provided path is a pathlib.Path object

    Parameters
    ----------
    path
        The path to the images

    Returns
    -------

        Return `Path`` object.
    """
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        raise RuntimeError(
            f'{Bcolors.FAIL}The provided path ({path}) is neither a String nor'
            f' a pathlib.Path object. {Bcolors.ENDC}'
        )


def check_output_directories(*args) -> None:
    """
        Check whether the provided paths exist
            -> Create new directories if not
    """
    for arg in args:
        if isinstance(arg, str):
            path = Path(arg)
            Path.mkdir(path, exist_ok=True)
        elif isinstance(arg, Path):
            Path.mkdir(arg, exist_ok=True)
        else:
            raise RuntimeError(
                f'{Bcolors.FAIL}The provided path ({arg}) is neither a String '
                f'nor a pathlib.Path object. {Bcolors.ENDC}'
            )


def clear_directory(path: Path) -> None:
    """
    Check if path is a directory and if it is empty. If the path does not
    exist, create it. If the directory is not empty, remove all files in
    this directory.

    Parameters
    ----------
    path
        Directory path
    """
    if path.is_dir():
        file_list = [x for x in path.iterdir()]
        for fil in file_list:
            fil.unlink()
    else:
        path.mkdir(exist_ok=True)


def check_if_directory_is_empty(path: Path) -> bool:
    """
    Check if path is a directory and if it is empty.

    Parameters
    ----------
    path
        The path to the directory.

    Returns
    -------

        `False` if the directory is not empty
    """

    if path.is_dir():
        file_list = [x for x in path.iterdir()]
        if file_list:
            return False
    return True
