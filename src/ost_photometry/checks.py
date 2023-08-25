############################################################################
#                               Libraries                                  #
############################################################################

import os

from pathlib import Path

from .style import Bcolors


############################################################################
#                           Routines & definitions                         #
############################################################################

def check_path(path):
    """
        Check if paths are valid

        Parameters
        ----------
        path        : `string`
            Path to check
    """
    if not os.path.isdir(path):
        raise RuntimeError(
            f"{Bcolors.FAIL} \n{path} not found -> Check file name! "
            f"ABORT {Bcolors.ENDC}"
        )


def check_file(file_path):
    """
        Check if file exists

        Parameters
        ----------
        file_path        : `string`
            File to check
    """
    if not os.path.isfile(file_path):
        raise RuntimeError(
            f"{Bcolors.FAIL} \n{file_path} not found -> Check file name! "
            f"ABORT {Bcolors.ENDC}"
        )


def list_subdirectories(path):
    """
        List subdirectories

        Parameters
        ----------
        path            : `string`
            Path to directory with subdirectories


        Returns
        -------
                        : `list`
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


def check_dir(path_dict):
    """
        Check whether the directories exist

        Parameters
        ----------
        path_dict       : `dictionary`
            Keys - Path identifier : `string`; values - Path : `string`
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


def check_unumpy_array(arr):
    """
        Check if an array is an unumpy array. Since those arrays are also
        numpy arrays, the for dtype. The dtype of unumpy arrays is always
        ``object``.

        Parameters
        ----------
        arr         : `numpy.ndarray`

        Returns
        -------
                    : `boolean`
            ``True`` if unumpy array, ``False`` otherwise.
    """
    if arr.dtype == "object":
        return True

    return False


def check_pathlib_path(path):
    """
        Check if the provided path is a pathlib.Path object

        Parameters
        ----------
        path            : `string` or `pathlib.Path`
            Path to the images

        Returns
        -------
                        : `pathlib.Path`
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


def check_output_directories(*args):
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


def clear_directory(path):
    """
        Check if path is a directory and if it is empty. If the path not
        exists, create it. If the directory is not empty, remove all files in
        this directory.

        Parameters
        ----------
        path            : `pathlib.Path`
            Path to the directory.
    """

    if path.is_dir():
        file_list = [x for x in path.iterdir()]
        for fil in file_list:
            fil.unlink()
    else:
        path.mkdir(exist_ok=True)


def check_if_directory_is_empty(path):
    """
        Check if path is a directory and if it is empty.

        Parameters
        ----------
        path            : `pathlib.Path`
            Path to the directory.

        Returns
        -------
                        : `boolean`
            `False` if the directory is not empty
    """

    if path.is_dir():
        file_list = [x for x in path.iterdir()]
        if file_list:
            return False
    return True
