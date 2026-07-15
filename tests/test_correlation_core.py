"""Tests for correlate.core helpers and correlation_astropy edge cases."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import astropy.units as u
from astropy import wcs

from helpers import load_module_from_path, pkg_src


def _clear_duplicates(
    data_array: np.ndarray,
    selection_quantity: np.ndarray,
    additional_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        group_indices = sort_order[group_start:group_end]
        best = group_indices[np.argmin(selection_quantity[group_indices])]
        keep[group_indices] = False
        keep[best] = True
        group_start = group_end

    return data_array[keep], selection_quantity[keep], additional_array[keep]


def _core_module():
    src = pkg_src()
    load_module_from_path(
        "ost_photometry.style",
        src / "ost_photometry" / "style.py",
    )
    load_module_from_path(
        "ost_photometry.terminal_output",
        src / "ost_photometry" / "terminal_output.py",
    )

    analyze_pkg = types.ModuleType("ost_photometry.analyze")
    sys.modules.setdefault("ost_photometry.analyze", analyze_pkg)

    utilities_mod = types.ModuleType("ost_photometry.analyze.utilities")
    utilities_mod.clear_duplicates = _clear_duplicates
    sys.modules["ost_photometry.analyze.utilities"] = utilities_mod

    return load_module_from_path(
        "ost_photometry.analyze.correlate.core",
        src / "ost_photometry" / "analyze" / "correlate" / "core.py",
    )


@pytest.fixture
def core():
    return _core_module()


def test_drop_protected_from_rejected_object_ids_scalar_and_vector_special(
    core,
):
    rejected = np.array([1, 5, 3, 7])
    assert np.array_equal(
        core._drop_protected_from_rejected_object_ids(rejected, [3]),
        np.array([1, 5, 7]),
    )
    assert np.array_equal(
        core._drop_protected_from_rejected_object_ids(rejected, [3, 5]),
        np.array([1, 7]),
    )
    assert np.array_equal(
        core._drop_protected_from_rejected_object_ids(rejected, np.array([5])),
        np.array([1, 3, 7]),
    )


def test_drop_protected_from_rejected_object_ids_empty_inputs(core):
    rejected = np.array([2, 4])
    assert np.array_equal(
        core._drop_protected_from_rejected_object_ids(rejected, []),
        rejected,
    )
    assert np.array_equal(
        core._drop_protected_from_rejected_object_ids(np.array([]), [1]),
        np.array([]),
    )


def test_dataset_positions_identical(core):
    x_ref = np.array([1.0, 2.0])
    y_ref = np.array([3.0, 4.0])
    assert core._dataset_positions_identical(x_ref, y_ref, x_ref.copy(), y_ref.copy())
    assert not core._dataset_positions_identical(
        x_ref,
        y_ref,
        np.array([1.0, 2.1]),
        y_ref,
    )


def _simple_wcs() -> wcs.WCS:
    return wcs.WCS(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CRPIX1": 50.0,
            "CRPIX2": 50.0,
            "CDELT1": -0.001,
            "CDELT2": 0.001,
        }
    )


def _pixel_positions(values: list[float]) -> u.Quantity:
    return u.Quantity(values, unit=u.pixel)


def test_correlation_astropy_identical_positions_between_datasets(core):
    wcs_obj = _simple_wcs()
    positions_x = _pixel_positions([10.0, 20.0, 30.0])
    positions_y = _pixel_positions([11.0, 21.0, 31.0])

    index_array, rejected_images = core.correlation_astropy(
        [positions_x, positions_x],
        [positions_y, positions_y],
        wcs_obj,
        advanced_cleanup=False,
    )

    assert rejected_images.size == 0
    assert index_array.shape == (2, 3)
    np.testing.assert_array_equal(index_array[0], [0, 1, 2])
    np.testing.assert_array_equal(index_array[1], [0, 1, 2])


def test_correlation_astropy_advanced_cleanup_keeps_protected_object(core):
    wcs_obj = _simple_wcs()
    ref_x = _pixel_positions([10.0, 20.0, 30.0, 40.0])
    ref_y = _pixel_positions([10.0, 20.0, 30.0, 40.0])
    cur_x = _pixel_positions([10.0, 20.0, 30.5, 40.0])
    cur_y = _pixel_positions([10.0, 20.0, 30.5, 40.0])

    index_array, rejected_images = core.correlation_astropy(
        [ref_x, cur_x],
        [ref_y, cur_y],
        wcs_obj,
        special_object_ids=[2],
        expected_bad_image_fraction=1,
        protect_special_objects=True,
        advanced_cleanup=True,
        separation_limit=2.0 * u.arcsec,
    )

    assert index_array.shape[1] == 4
    assert index_array[0, 2] == 2
    assert 2 in index_array[0]


def test_correlation_astropy_afterburner_removes_dataset_for_protected_miss(
    core,
):
    wcs_obj = _simple_wcs()
    ref_x = _pixel_positions([10.0, 20.0, 30.0])
    ref_y = _pixel_positions([10.0, 20.0, 30.0])
    cur_x = _pixel_positions([10.0, 20.0, 30.0])
    cur_y = _pixel_positions([10.0, 20.0, 30.0])
    miss_x = _pixel_positions([10.0, 20.0, 99.0])
    miss_y = _pixel_positions([10.0, 20.0, 99.0])

    index_array, rejected_images = core.correlation_astropy(
        [ref_x, cur_x, miss_x],
        [ref_y, cur_y, miss_y],
        wcs_obj,
        special_object_ids=[2],
        protect_special_objects=True,
        advanced_cleanup=False,
        separation_limit=2.0 * u.arcsec,
    )

    assert index_array.shape == (2, 3)
    assert 2 in rejected_images
    assert index_array.shape[1] == 3
    assert index_array[0, 2] == 2
