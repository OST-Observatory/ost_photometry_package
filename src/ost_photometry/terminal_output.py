from . import style, checks

############################################################################
####                        Routines & definitions                      ####
############################################################################

def print_terminal(*args, string='', condense=False, indent=1,
                   style_name='BOLD'):
    '''
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
    '''
    outstr = "".rjust(3*indent)
    if style_name == 'HEADER':
        outstr += style.bcolors.HEADER

    elif style_name == 'FAIL':
        outstr += style.bcolors.FAIL

    elif style_name == 'WARNING':
        outstr += style.bcolors.WARNING

    elif style_name == 'OKBLUE':
        outstr += style.bcolors.OKBLUE

    elif style_name == 'OKGREEN':
        outstr += style.bcolors.OKGREEN

    elif style_name == 'UNDERLINE':
        outstr += style.bcolors.UNDERLINE

    else:
        outstr += style.bcolors.BOLD

    outstr += string.format(*args)
    outstr += style.bcolors.ENDC

    if condense:
        outstr += '\n'
        return outstr
    else:
        print(outstr)

