"""Small error-propagation helpers."""

from __future__ import annotations

import numpy as np


def err_prop(*args: float | np.ndarray) -> float | np.ndarray:
    """
    Combine independent error terms in quadrature.

    Parameters
    ----------
    args
        Sources of error (floats or arrays) to add in RSS form.

    Returns
    -------
    sum_error
        Accumulated error with the same broadcast shape as the inputs.
    """
    sum_error: float | np.ndarray = 0.0
    for i, x in enumerate(args):
        if i == 0:
            sum_error = x
        else:
            sum_error = np.sqrt(np.square(sum_error) + np.square(x))
    return sum_error


__all__ = ["err_prop"]
