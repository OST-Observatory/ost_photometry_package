from . import style

############################################################################
#                           Routines & definitions                         #
############################################################################


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
    #   Print to terminal
    print(format_string(string, indent=indent, style_name=style_name))


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

        #   Reset cache
        # self.cache = ""
