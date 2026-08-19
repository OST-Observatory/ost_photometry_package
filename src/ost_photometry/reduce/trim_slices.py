"""Pure CCD trim slice helpers (no ccdproc dependency)."""

from __future__ import annotations

import math

import numpy as np


def ccd_trim_slices(
    shape: tuple[int, int],
    *,
    x_start: int = 0,
    x_end: int = 0,
    y_start: int = 0,
    y_end: int = 0,
    end_as_positive_margin: bool = True,
) -> tuple[slice, slice]:
    """
    Build ``(y_slice, x_slice)`` for trimming a 2-D CCD array.

    Parameters
    ----------
    shape
        Image shape as ``(ny, nx)``.

    x_start, y_start
        Number of pixels to remove from the start of each axis.

    x_end, y_end
        End trim. Interpretation depends on ``end_as_positive_margin``.

    end_as_positive_margin
        If ``True`` (fixed-margin trim, e.g. N1), ``x_end``/``y_end`` are
        positive pixel counts removed from the end.
        If ``False`` (alignment trim), ``x_end``/``y_end`` are offsets added
        to the axis length (typically ``0`` or negative).
    """
    ny, nx = int(shape[0]), int(shape[1])
    x0 = int(x_start)
    y0 = int(y_start)

    if end_as_positive_margin:
        x1 = nx - int(x_end) if x_end else nx
        y1 = ny - int(y_end) if y_end else ny
    else:
        x1 = nx + int(x_end)
        y1 = ny + int(y_end)

    if not (0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny):
        raise ValueError(
            f"Invalid trim window for shape {(ny, nx)}: "
            f"y=[{y0}:{y1}], x=[{x0}:{x1}]"
        )

    return slice(y0, y1), slice(x0, x1)


def aa_common_trim_margins(
        image_shift: np.ndarray,
) -> tuple[int, int, int, int]:
    """Pixel trim margins shared by all images after astroalign xy-shift.

    ``image_shift`` uses python/image axis order: row 0 is Y, row 1 is X.
    """
    min_shift_x = float(np.nanmin(image_shift[1, :]))
    max_shift_x = float(np.nanmax(image_shift[1, :]))
    min_shift_y = float(np.nanmin(image_shift[0, :]))
    max_shift_y = float(np.nanmax(image_shift[0, :]))

    if min_shift_x > 0:
        x_start = int(math.ceil(max_shift_x))
        x_end = 0
    elif min_shift_x < 0 and max_shift_x < 0:
        x_start = 0
        x_end = int(math.ceil(np.abs(min_shift_x))) * -1
    else:
        x_start = int(math.ceil(max_shift_x))
        x_end = int(math.ceil(np.abs(min_shift_x))) * -1

    if min_shift_y > 0:
        y_start = int(math.ceil(max_shift_y))
        y_end = 0
    elif min_shift_y < 0 and max_shift_y < 0:
        y_start = 0
        y_end = int(math.ceil(np.abs(min_shift_y))) * -1
    else:
        y_start = int(math.ceil(max_shift_y))
        y_end = int(math.ceil(np.abs(min_shift_y))) * -1

    return x_start, x_end, y_start, y_end
