"""Deduplicate array rows by a selection metric."""

from __future__ import annotations

import numpy as np


def clear_duplicates(
    data_array: np.ndarray,
    selection_quantity: np.ndarray,
    additional_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find duplicates in an array (``data_array``). Select the best of the
    duplicates based on a selection criterium (``selection_quantity``)
    such as the distance between two points.

    The resulting changes will be applied to a second array
    (``additional_array``) of the same length.

    Parameters
    ----------
    data_array
        Array from which the duplicates should be removed.
    selection_quantity
        Quantities used to choose the best duplicate. The duplicate with
        the lowest value is kept; the remaining ones are removed.
    additional_array
        Additional array cleared in the same way as ``data_array``.

    Returns
    -------
    data_array, selection_quantity, additional_array
        Copies cleared of duplicates (same order as input survivors).
    """
    data_array = np.asarray(data_array)
    selection_quantity = np.asarray(selection_quantity)
    additional_array = np.asarray(additional_array)

    if data_array.size == 0:
        return data_array, selection_quantity, additional_array

    keep = np.ones(data_array.shape[0], dtype=bool)
    sort_order = np.argsort(data_array, kind="stable")

    group_start = 0
    while group_start < sort_order.size:
        group_end = group_start + 1
        while (
            group_end < sort_order.size
            and data_array[sort_order[group_end]]
            == data_array[sort_order[group_start]]
        ):
            group_end += 1

        if group_end - group_start > 1:
            group_indices = sort_order[group_start:group_end]
            best_index = group_indices[np.argmin(selection_quantity[group_indices])]
            keep[group_indices] = False
            keep[best_index] = True

        group_start = group_end

    return (
        data_array[keep],
        selection_quantity[keep],
        additional_array[keep],
    )


__all__ = ["clear_duplicates"]
