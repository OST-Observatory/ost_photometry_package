"""Terminal output helpers (ANSI formatting + logging bridge)."""

from . import style
from .logging_config import get_logger

_LEVEL = {
    "FAIL": 40,
    "ERROR": 40,
    "WARNING": 30,
    "HEADER": 20,
    "BOLD": 20,
    "OK": 20,
    "OKBLUE": 20,
    "OKGREEN": 20,
    "GOOD": 20,
    "UNDERLINE": 20,
    "ITALIC": 20,
    "NORMAL": 20,
}


def print_to_terminal(
        string: str, indent: int = 1, style_name: str = 'BOLD') -> None:
    """
    Print output to terminal after formatting

    Parameters
    ----------
    string
        Output string.
        Default is ````.

    indent
        Indentation level of the terminal output.
        Default is ``1``.

    style_name
        Style type of the output.
        Default is ``BOLD``.
    """
    msg = format_string(string, indent=indent, style_name=style_name)
    get_logger("terminal").log(_LEVEL.get(style_name, 20), msg)


def format_string(
        string: str, indent: int = 1, style_name: str = 'BOLD') -> str:
    """
    Formats string

    Parameters
    ----------
    string
        Output string.

    indent
        Indentation level of the terminal output.
        Default is ``1``.

    style_name
        Style type of the output.
        Default is ``BOLD``.

    Returns
    -------
    string_out
    """
    string_out = "".rjust(3 * indent)
    if style_name == 'HEADER':
        string_out += style.Bcolors.HEADER

    elif style_name in ['FAIL', 'ERROR']:
        string_out += style.Bcolors.FAIL

    elif style_name == 'WARNING':
        string_out += style.Bcolors.WARNING

    elif style_name in ['OKBLUE', 'OK']:
        string_out += style.Bcolors.OKBLUE

    elif style_name in ['OKGREEN', 'GOOD']:
        string_out += style.Bcolors.OKGREEN

    elif style_name == 'UNDERLINE':
        string_out += style.Bcolors.UNDERLINE

    elif style_name == 'ITALIC':
        string_out += style.Bcolors.ITALIC

    elif style_name == 'NORMAL':
        string_out += style.Bcolors.NORMAL

    else:
        string_out += style.Bcolors.BOLD

    string_out += string

    string_out += style.Bcolors.ENDC

    return string_out


class TerminalLog:
    """
        Logging system to the terminal
    """

    def __init__(self):
        self.cache = ""

    def add_to_cache(
            self, string: str, indent: int = 1, style_name: str = 'BOLD'
        ) -> None:
        """
        Add string to cache after formatting

        Parameters
        ----------
        string
            Output string.

        indent
            Indentation level of the terminal output.
            Default is ``1``.

        style_name
            Style type of the output.
            Default is ``BOLD``.

        """
        self.cache += format_string(string, indent=indent, style_name=style_name)
        self.cache += "\n"

    def print_to_terminal(
            self, string: str, indent: int = 1, style_name: str = 'BOLD'
        ) -> None:
        """
        Print output to terminal after formatting

        Parameters
        ----------
        string
            Output string.

        indent
            Indentation level of the terminal output.
            Default is ``1``.

        style_name
            Style type of the output.
            Default is ``BOLD``.
        """
        #   Add string to cache
        self.add_to_cache(string, indent=indent, style_name=style_name)

        print(self.cache)

        #   Reset cache after printing this block
        self.cache = ""
