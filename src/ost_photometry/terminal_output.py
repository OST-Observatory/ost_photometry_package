from . import style

############################################################################
#                           Routines & definitions                         #
############################################################################


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
        out_string += style.bcolors.HEADER

    elif style_name == 'FAIL':
        out_string += style.bcolors.FAIL

    elif style_name == 'WARNING':
        out_string += style.bcolors.WARNING

    elif style_name == 'OKBLUE':
        out_string += style.bcolors.OKBLUE

    elif style_name == 'OKGREEN':
        out_string += style.bcolors.OKGREEN

    elif style_name == 'UNDERLINE':
        out_string += style.bcolors.UNDERLINE

    else:
        out_string += style.bcolors.BOLD

    out_string += string.format(*args)
    out_string += style.bcolors.ENDC

    if condense:
        out_string += '\n'
        return out_string
    else:
        print(out_string)


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
        string_out += style.bcolors.HEADER

    elif style_name == 'FAIL':
        string_out += style.bcolors.FAIL

    elif style_name == 'WARNING':
        string_out += style.bcolors.WARNING

    elif style_name in ['OKBLUE', 'OK']:
        string_out += style.bcolors.OKBLUE

    elif style_name in ['OKGREEN', 'GOOD']:
        string_out += style.bcolors.OKGREEN

    elif style_name == 'UNDERLINE':
        string_out += style.bcolors.UNDERLINE

    else:
        string_out += style.bcolors.BOLD

    string_out += string

    string_out += style.bcolors.ENDC

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

    def print_to_terminal(self, string, indent=1, style_name='BOLD', print_cache=False):
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

            print_cache     : `boolean`, optional
                Print complete cache instead of only the string
                Default is ``False``
        """
        #   Add to cache
        if print_cache:
            self.add_to_cache(string)

        #   Format string
        string = format_string(string, indent=indent, style_name=style_name)

        #   Add cache or string
        if print_cache:
            out_string = self.cache + string
        else:
            out_string = string

        #   Print to terminal
        print(out_string)

        #   Reset cache
        if print_cache:
            self.cache = ""
