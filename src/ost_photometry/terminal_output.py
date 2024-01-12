from . import style

############################################################################
#                           Routines & definitions                         #
############################################################################


#   TODO: Remove this function is the future
def print_terminal(*args, string='', condense=False, indent=1,
                   style_name='BOLD'):
    """
        Creates formatted output for the terminal

        Parameters
        ----------
        *args           :
            Variables to be inserted in the ``string``.

        string          : `string`, optional
            Output string.
            Default is ````.

        condense        : `boolean`, optional
            If True the terminal output will be returned to the calling
            function.
            Default is ``False``.

        indent          : `integer`, optional
            Indentation level of the terminal output.
            Default is ``1``.

        style_name      : `string`, optional
            Style type of the output.
            Default is ``BOLD``.
    """
    out_string = "".rjust(3 * indent)
    if style_name == 'HEADER':
        out_string += style.Bcolors.HEADER

    elif style_name in ['FAIL', 'ERROR']:
        out_string += style.Bcolors.FAIL

    elif style_name == 'WARNING':
        out_string += style.Bcolors.WARNING

    elif style_name == 'OKBLUE':
        out_string += style.Bcolors.OKBLUE

    elif style_name == 'OKGREEN':
        out_string += style.Bcolors.OKGREEN

    elif style_name == 'UNDERLINE':
        out_string += style.Bcolors.UNDERLINE

    else:
        out_string += style.Bcolors.BOLD

    out_string += string.format(*args)
    out_string += style.Bcolors.ENDC

    if condense:
        out_string += '\n'
        return out_string
    else:
        print(out_string)


def print_to_terminal(string, indent=1, style_name='BOLD'):
    """
        Print output to terminal after formatting

        Parameters
        ----------
        string          : `string`, optional
            Output string.
            Default is ````.

        indent          : `integer`, optional
            Indentation level of the terminal output.
            Default is ``1``.

        style_name      : `string`, optional
            Style type of the output.
            Default is ``BOLD``.
    """
    #   Print to terminal
    print(format_string(string, indent=indent, style_name=style_name))


def format_string(string, indent=1, style_name='BOLD'):
    """
        Formats string

        Parameters
        ----------
        string          : `string`, optional
            Output string.
            Default is ````.

        indent          : `integer`, optional
            Indentation level of the terminal output.
            Default is ``1``.

        style_name      : `string`, optional
            Style type of the output.
            Default is ``BOLD``.

        Returns
        -------
        string_out      : `string`
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

    def add_to_cache(self, string, indent=1, style_name='BOLD'):
        """
            Add string to cache after formatting

            Parameters
            ----------
            string          : `string`, optional
                Output string.
                Default is ````.

            indent          : `integer`, optional
                Indentation level of the terminal output.
                Default is ``1``.

            style_name      : `string`, optional
                Style type of the output.
                Default is ``BOLD``.

        """
        self.cache += format_string(string, indent=indent, style_name=style_name)
        self.cache += "\n"

    def print_to_terminal(self, string, indent=1, style_name='BOLD'):
        """
            Print output to terminal after formatting

            Parameters
            ----------
            string          : `string`, optional
                Output string.
                Default is ````.

            indent          : `integer`, optional
                Indentation level of the terminal output.
                Default is ``1``.

            style_name      : `string`, optional
                Style type of the output.
                Default is ``BOLD``.
        """
        #   Add string to cache
        self.add_to_cache(string, indent=indent, style_name=style_name)

        print(self.cache)

        #   Reset cache
        # self.cache = ""
