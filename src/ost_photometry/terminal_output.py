from . import style

############################################################################
####                        Routines & definitions                      ####
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
