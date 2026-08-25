"""Photometry helpers used by extraction."""

from __future__ import annotations

import numpy as np
from astropy.table import Column, Table

from ... import terminal_output


def flux_to_magnitudes(
    flux: np.ndarray | Column,
    flux_error: np.ndarray | Column,
) -> tuple[np.ndarray | Column, np.ndarray | Column]:
    """
    Calculate magnitudes from flux.

    Parameters
    ----------
    flux
        Flux values.
    flux_error
        Flux uncertainties.

    Returns
    -------
    magnitudes, magnitudes_error
        Object magnitudes (``-2.5 log10``) and **positive** 1σ uncertainties
        ``(2.5 / ln(10)) · σ(F)/|F|``.
    """
    magnitudes = -2.5 * np.log10(flux)
    # Pogson: σ(m) = (2.5 / ln(10)) · σ(F)/|F|.
    magnitudes_error = (2.5 / np.log(10)) * np.abs(flux_error / flux)
    return magnitudes, magnitudes_error


def rm_edge_objects(
    table: Table,
    data_array: np.ndarray,
    border: int = 10,
    terminal_logger: terminal_output.TerminalLog | None = None,
    indent: int = 3,
) -> Table:
    """
    Remove detected objects that are too close to the image edges.

    Parameters
    ----------
    table
        Object data with ``x_fit`` / ``y_fit`` columns.
    data_array
        Image data (2D).
    border
        Distance to the edge (pixels) inside which objects are discarded.
        Default is ``10``.
    terminal_logger
        Optional logger; otherwise prints to the terminal.
    indent
        Indentation for console output. Default is ``3``.
    """
    hsize = border + 1
    x = table["x_fit"].value
    y = table["y_fit"].value
    mask = (
        (x > hsize)
        & (x < (data_array.shape[1] - 1 - hsize))
        & (y > hsize)
        & (y < (data_array.shape[0] - 1 - hsize))
    )

    out_str = (
        f"Removed {np.count_nonzero(np.invert(mask))} objects "
        f"that were too close to the edges of the image."
    )
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            out_str,
            style_name="ITALIC",
            indent=indent + 1,
        )
    else:
        terminal_output.print_to_terminal(
            out_str,
            style_name="ITALIC",
            indent=indent + 1,
        )

    return table[mask]


__all__ = ["flux_to_magnitudes", "rm_edge_objects"]
