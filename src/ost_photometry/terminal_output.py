"""Terminal output helpers (ANSI formatting + logging bridge).

Styles: ``HEADER`` (magenta, script titles), ``OKBLUE`` (sections),
``OKGREEN`` (results), ``WARNING``, ``FAIL``, ``BOLD`` / ``NORMAL`` (text)
and ``DETAIL``: per-frame chatter, shown only with the environment variable
``OST_PHOTOMETRY_VERBOSE=1`` (see :mod:`.logging_config`).

The indentation of the last message is remembered so that progress bars
(:func:`progress_bar`, the ``Executor``) start one level deeper instead of
at the beginning of the line.
"""

import re

from . import style
from .logging_config import get_logger

#: Indentation level of the last message (progress bars go one deeper).
_STATE = {"indent": 0, "quiet": 0}

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
    "DETAIL": 10,
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
    logger = get_logger("terminal")
    level = _LEVEL.get(style_name, 20)
    if _STATE["quiet"] and level < 30:
        level = 10  # inside quiet(): a detail
    if logger.isEnabledFor(level):
        _STATE["indent"] = int(indent)
    logger.log(level, msg)


class quiet:
    """Context in which ordinary messages become details (shown only with
    ``OST_PHOTOMETRY_VERBOSE=1``); warnings and errors stay visible. For
    per-frame steps of a long run, e.g. plate solving every frame."""

    def __enter__(self):
        _STATE["quiet"] += 1
        return self

    def __exit__(self, *_exc):
        _STATE["quiet"] -= 1
        return False


def current_indent() -> int:
    """Indentation level of the last printed message."""
    return int(_STATE["indent"])


#: Words that mark a line as a problem (yellow) or a result (green).
_WARNING_WORDS = re.compile(
    r"\b(warning|failed|failure|error|incomplete|missing|not reduced|not found|"
    r"no usable|no bias|no darks?|no flats?|no reliable|rejected|retry|skipped|"
    r"impossible|still to take|unknown)\b",
    re.IGNORECASE,
)
_GOOD_WORDS = re.compile(r"^(done|finished|written|saved)\b", re.IGNORECASE)


def auto_style(text: str) -> str:
    """Style for a plain log line: ``WARNING`` for problems, ``OKGREEN`` for
    completions, ``OKBLUE`` for headings (``Something:`` alone), else
    ``NORMAL``."""
    stripped = text.strip()
    if _WARNING_WORDS.search(stripped):
        return "WARNING"
    if _GOOD_WORDS.search(stripped):
        return "OKGREEN"
    if stripped.endswith(":") and len(stripped) < 60:
        return "OKBLUE"
    return "NORMAL"


def print_auto(text: str, indent: int = 2) -> None:
    """Print a plain log line at ``indent`` with :func:`auto_style`."""
    print_to_terminal(text, indent=indent, style_name=auto_style(text))


def print_report(lines, indent: int = 1) -> None:
    """Report lines with their own leading spaces (two per level) as
    indentation levels below ``indent``; colours by :func:`auto_style`,
    unindented ``Heading: ...`` lines as section headings."""
    for line in lines:
        text = str(line)
        spaces = len(text) - len(text.lstrip(" "))
        level = indent + spaces // 2
        stripped = text.strip()
        if not stripped:
            continue
        style_name = auto_style(stripped)
        if spaces == 0 and style_name == "NORMAL" and ":" in stripped[:40]:
            style_name = "OKBLUE"
        print_to_terminal(stripped, indent=level, style_name=style_name)


def print_header(title: str) -> None:
    """Title line of a script (magenta, framed)."""
    rule = "=" * max(10, len(title) + 4)
    print_to_terminal(rule, indent=0, style_name="HEADER")
    print_to_terminal(f"  {title}", indent=0, style_name="HEADER")
    print_to_terminal(rule, indent=0, style_name="HEADER")


def print_section(title: str, indent: int = 1) -> None:
    """A step of a script (blue)."""
    print_to_terminal(title, indent=indent, style_name="OKBLUE")


def print_done(start_time: float | None = None, files=None, next_step: str | None = None) -> None:
    """End of a script: "Done" (green), the written files, the next step and
    the run time."""
    import time

    print_to_terminal("Done", indent=1, style_name="OKGREEN")
    for label, value in dict(files or {}).items():
        print_to_terminal(f"{label + ':':<11} {value}", indent=2, style_name="NORMAL")
    if next_step:
        print_to_terminal(f"Next: {next_step}", indent=1, style_name="OKBLUE")
    if start_time is not None:
        minutes = (time.time() - float(start_time)) / 60.0
        print_to_terminal(f"Run time {minutes:.1f} min", indent=1, style_name="NORMAL")


def progress_prefix(extra: int = 1) -> str:
    """Spaces that put a progress bar ``extra`` levels below the last line."""
    return " " * (3 * (current_indent() + extra))


def progress_bar(total: int, desc: str = "", extra: int = 1, **kwargs):
    """A tqdm bar indented below the last message (``desc`` before it)."""
    from tqdm import tqdm

    label = f"{desc}: " if desc else ""
    return tqdm(total=total, bar_format=progress_prefix(extra) + label + "{l_bar}{bar}{r_bar}",
                **kwargs)


def progress_callback(desc: str = "", extra: int = 1):
    """``callback(i, n, name)`` that drives one indented bar (for the
    ``progress=`` arguments of the archive functions)."""
    state: dict = {"bar": None}

    def callback(i: int, n: int, _name: str = "") -> None:
        if state["bar"] is None or state["bar"].total != n:
            if state["bar"] is not None:
                state["bar"].close()
            state["bar"] = progress_bar(n, desc, extra)
        state["bar"].n = int(i)
        state["bar"].refresh()
        if i >= n:
            state["bar"].close()
            state["bar"] = None

    return callback


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

    elif style_name in ('NORMAL', 'DETAIL'):
        string_out += style.Bcolors.NORMAL

    else:
        string_out += style.Bcolors.BOLD

    string_out += string

    string_out += style.Bcolors.ENDC

    return string_out


class TerminalLog:
    """Block of messages of one task (e.g. one image in a worker process),
    printed together.

    With ``quiet=True`` (many images in parallel) only warnings and errors
    are shown, prefixed with ``label``; the rest are details, shown only with
    ``OST_PHOTOMETRY_VERBOSE=1``.
    """

    def __init__(self, quiet: bool = False, label: str = ""):
        self.cache = ""
        self.quiet = bool(quiet)
        self.label = str(label)
        self.entries: list[tuple[str, int, str]] = []

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
        self.entries.append((str(string), int(indent), str(style_name)))
        self.cache += format_string(string, indent=indent, style_name=style_name)
        self.cache += "\n"

    def print_to_terminal(
            self, string: str, indent: int = 1, style_name: str = 'BOLD'
        ) -> None:
        """
        Print the block (plus ``string``) and reset it.

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
        if string:
            self.add_to_cache(string, indent=indent, style_name=style_name)
        if self.quiet:
            logger = get_logger("terminal")
            prefix = f"{self.label}: " if self.label else ""
            for text, level_indent, name in self.entries:
                if not text.strip():
                    continue
                if _LEVEL.get(name, 20) >= 30:
                    logger.log(_LEVEL[name], format_string(prefix + text.strip(), indent=2,
                                                           style_name=name))
                else:
                    logger.log(10, format_string(text, indent=level_indent, style_name=name))
        else:
            print(self.cache)
        self.entries = []
        self.cache = ""
