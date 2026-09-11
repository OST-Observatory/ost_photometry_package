"""Tests for correlate.core helpers and correlation_astropy edge cases."""

from __future__ import annotations

import sys
import types

import astropy.units as u
import numpy as np
import pytest
from astropy import wcs

from helpers import ensure_stub_package, isolated_sys_modules, load_module_from_path, pkg_src


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


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

    analyze_dir = src / "ost_photometry" / "analyze"
    ensure_stub_package("ost_photometry.analyze", path=analyze_dir)
    ensure_stub_package("ost_photometry.analyze.correlate", path=analyze_dir / "correlate")

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


def test_correlation_astropy_sparse_keeps_incomplete_track(core):
    wcs_obj = _simple_wcs()
    ref_x = _pixel_positions([10.0, 20.0, 30.0])
    ref_y = _pixel_positions([10.0, 20.0, 30.0])
    cur_x = _pixel_positions([10.0, 20.0])
    cur_y = _pixel_positions([10.0, 20.0])
    miss_x = _pixel_positions([10.0, 20.0, 99.0])
    miss_y = _pixel_positions([10.0, 20.0, 99.0])

    index_array, rejected_images = core.correlation_astropy(
        [ref_x, cur_x, miss_x],
        [ref_y, cur_y, miss_y],
        wcs_obj,
        special_object_ids=[2],
        protect_special_objects=True,
        advanced_cleanup=False,
        require_complete_intersection=False,
        n_allowed_non_detections_object=5,
        min_detection_fraction=0.3,
        separation_limit=2.0 * u.arcsec,
    )

    assert rejected_images.size == 0
    assert index_array.shape == (3, 3)
    assert index_array[1, 2] == -1
    assert index_array[0, 2] == 2
    assert index_array[2, 2] == -1


def test_correlation_astropy_uses_per_frame_wcs(core):
    wcs_a = _simple_wcs()
    wcs_b = wcs.WCS(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CRPIX1": 60.0,
            "CRPIX2": 50.0,
            "CDELT1": -0.001,
            "CDELT2": 0.001,
        }
    )
    ref_x = _pixel_positions([50.0, 40.0])
    ref_y = _pixel_positions([50.0, 40.0])
    cur_x = _pixel_positions([60.0, 50.0])
    cur_y = _pixel_positions([50.0, 40.0])

    index_array, rejected = core.correlation_astropy(
        [ref_x, cur_x],
        [ref_y, cur_y],
        wcs_a,
        wcs_list=[wcs_a, wcs_b],
        advanced_cleanup=False,
        require_complete_intersection=True,
        separation_limit=2.0 * u.arcsec,
    )
    assert rejected.size == 0
    np.testing.assert_array_equal(index_array[1], [0, 1])


def test_correlation_astropy_sequential_adds_new_track(core):
    wcs_obj = _simple_wcs()
    x0 = _pixel_positions([10.0, 20.0, 30.0])
    y0 = _pixel_positions([10.0, 20.0, 30.0])
    x1 = _pixel_positions([10.0, 20.0, 40.0])
    y1 = _pixel_positions([10.0, 20.0, 40.0])
    x2 = _pixel_positions([10.0, 40.0])
    y2 = _pixel_positions([10.0, 40.0])

    index_array, rejected = core.correlation_astropy(
        [x0, x1, x2],
        [y0, y1, y2],
        wcs_obj,
        advanced_cleanup=False,
        require_complete_intersection=False,
        n_allowed_non_detections_object=5,
        min_detection_fraction=0.3,
        correlation_link_mode="sequential",
        separation_limit=2.0 * u.arcsec,
    )
    assert rejected.size == 0
    assert index_array.shape[0] == 3
    assert index_array.shape[1] >= 4
    # Star at 40 appears first on frame 1 and continues on frame 2.
    new_cols = np.flatnonzero(index_array[0] == -1)
    assert new_cols.size >= 1
    col = int(new_cols[0])
    assert index_array[1, col] >= 0
    assert index_array[2, col] >= 0


def test_sequential_seeds_from_reference_dataset(core):
    wcs_obj = _simple_wcs()
    x0 = _pixel_positions([10.0])
    y0 = _pixel_positions([10.0])
    x1 = _pixel_positions([10.0, 20.0, 30.0])
    y1 = _pixel_positions([10.0, 20.0, 30.0])
    x2 = _pixel_positions([10.0, 20.0])
    y2 = _pixel_positions([10.0, 20.0])

    index_array, rejected = core.correlation_astropy(
        [x0, x1, x2],
        [y0, y1, y2],
        wcs_obj,
        reference_dataset_id=1,
        advanced_cleanup=False,
        require_complete_intersection=False,
        n_allowed_non_detections_object=5,
        min_detection_fraction=0.3,
        correlation_link_mode="sequential",
        separation_limit=2.0 * u.arcsec,
    )
    assert rejected.size == 0
    np.testing.assert_array_equal(index_array[1, :3], [0, 1, 2])
    assert index_array.shape[1] >= 3


def test_sequential_bridges_single_frame_miss(core):
    """A star missing on one frame keeps its track on later frames."""
    wcs_obj = _simple_wcs()
    x0 = _pixel_positions([10.0, 20.0, 30.0])
    y0 = _pixel_positions([10.0, 20.0, 30.0])
    x1 = _pixel_positions([10.0, 20.0])  # star at 30 missing
    y1 = _pixel_positions([10.0, 20.0])
    x2 = _pixel_positions([30.2, 10.0, 20.0])
    y2 = _pixel_positions([30.1, 10.0, 20.0])
    x3 = _pixel_positions([20.0, 30.0, 10.0])
    y3 = _pixel_positions([20.0, 30.0, 10.0])

    index_array, rejected = core.correlation_astropy(
        [x0, x1, x2, x3],
        [y0, y1, y2, y3],
        wcs_obj,
        advanced_cleanup=False,
        require_complete_intersection=False,
        n_allowed_non_detections_object=5,
        min_detection_fraction=0.3,
        correlation_link_mode="sequential",
        coordinate_frame="pixel",
        pixel_separation=3.0,
    )
    assert rejected.size == 0
    # exactly three tracks: no fragment was started for the returning star
    assert index_array.shape == (4, 3)
    assert index_array[1, 2] == -1
    assert index_array[2, 2] == 0
    assert index_array[3, 2] == 1


def test_sequential_backward_walk_restarts_from_reference(core):
    wcs_obj = _simple_wcs()
    # reference is frame 2; frame 1 misses star B, frame 0 has it again
    xa = _pixel_positions([10.0, 20.0])
    ya = _pixel_positions([10.0, 20.0])
    xb = _pixel_positions([10.0])
    yb = _pixel_positions([10.0])
    index_array, _ = core.correlation_astropy(
        [xa, xb, xa, xa],
        [ya, yb, ya, ya],
        wcs_obj,
        reference_dataset_id=2,
        advanced_cleanup=False,
        require_complete_intersection=False,
        n_allowed_non_detections_object=5,
        min_detection_fraction=0.3,
        correlation_link_mode="sequential",
        coordinate_frame="pixel",
        pixel_separation=3.0,
    )
    assert index_array.shape == (4, 2)
    np.testing.assert_array_equal(index_array[:, 1], [1, -1, 1, 1])


def test_sequential_drops_chain_walk_off_anchor(core):
    """Neighbour-to-neighbour links that walk > separation_limit from origin are cut."""
    wcs_obj = _simple_wcs()
    # 0.001 deg/pixel ≈ 3.6"/px; 5" keeps a 1-pixel hop, not a 2-pixel hop.
    x0 = _pixel_positions([10.0])
    y0 = _pixel_positions([10.0])
    x1 = _pixel_positions([11.0])
    y1 = _pixel_positions([10.0])
    x2 = _pixel_positions([12.0])
    y2 = _pixel_positions([10.0])

    index_array, rejected = core.correlation_astropy(
        [x0, x1, x2],
        [y0, y1, y2],
        wcs_obj,
        advanced_cleanup=False,
        require_complete_intersection=False,
        n_allowed_non_detections_object=5,
        min_detection_fraction=0.3,
        correlation_link_mode="sequential",
        separation_limit=5.0 * u.arcsec,
    )
    assert rejected.size == 0
    orig = int(np.flatnonzero(index_array[0] >= 0)[0])
    assert index_array[0, orig] == 0
    assert index_array[1, orig] == 0
    assert index_array[2, orig] == -1


def test_pixel_matching_keeps_identity_when_wcs_differs(core):
    wcs_a = _simple_wcs()
    wcs_b = wcs.WCS(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CRPIX1": 60.0,
            "CRPIX2": 50.0,
            "CDELT1": -0.001,
            "CDELT2": 0.001,
        }
    )
    x = _pixel_positions([50.0, 40.0, 30.0])
    y = _pixel_positions([50.0, 40.0, 30.0])
    index_px, rejected = core.correlation_astropy(
        [x, x],
        [y, y],
        wcs_a,
        wcs_list=[wcs_a, wcs_b],
        advanced_cleanup=False,
        require_complete_intersection=True,
        separation_limit=2.0 * u.arcsec,
        coordinate_frame="pixel",
        pixel_separation=3.0,
    )
    assert rejected.size == 0
    np.testing.assert_array_equal(index_px[1], [0, 1, 2])


def test_auto_correlation_coordinates_choose_pixel_when_aligned(core):
    x0 = _pixel_positions([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    y0 = _pixel_positions([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    x1 = _pixel_positions([10.2, 20.1, 30.0, 40.3, 49.9, 60.1, 70.0, 80.2])
    y1 = _pixel_positions([10.1, 20.0, 30.2, 40.0, 50.1, 60.0, 69.9, 80.0])
    frame, shift = core.resolve_correlation_coordinates(
        "auto",
        [x0, x1, x1],
        [y0, y1, y1],
        0,
    )
    assert frame == "pixel"
    assert shift is not None and shift < 1.0


def test_resolve_ooi_separation_limit_floors_at_five_arcsec(core):
    src = pkg_src()
    load_module_from_path(
        "ost_photometry.analyze.ooi_ids",
        src / "ost_photometry" / "analyze" / "ooi_ids.py",
    )
    ooi = load_module_from_path(
        "ost_photometry.analyze.correlate.ooi",
        src / "ost_photometry" / "analyze" / "correlate" / "ooi.py",
    )
    tight = ooi.resolve_ooi_separation_limit(2.0 * u.arcsec)
    assert tight.to(u.arcsec).value == pytest.approx(5.0)
    wide = ooi.resolve_ooi_separation_limit(8.0 * u.arcsec)
    assert wide.to(u.arcsec).value == pytest.approx(8.0)
    custom = 3.0 * u.arcsec
    assert ooi.resolve_ooi_separation_limit(2.0 * u.arcsec, custom) is custom
