"""Tests for shared CCD trim helpers."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import load_module_from_path, pkg_src


def _trim_slices_mod():
    return load_module_from_path(
        "ost_photometry.reduce.trim_slices",
        pkg_src() / "ost_photometry" / "reduce" / "trim_slices.py",
    )


def test_ccd_trim_slices_positive_margins():
    mod = _trim_slices_mod()
    y_sl, x_sl = mod.ccd_trim_slices(
        (100, 200),
        x_start=10,
        x_end=20,
        y_start=5,
        y_end=15,
        end_as_positive_margin=True,
    )
    assert y_sl == slice(5, 85)
    assert x_sl == slice(10, 180)


def test_ccd_trim_slices_zero_end_margin_keeps_full_axis():
    mod = _trim_slices_mod()
    y_sl, x_sl = mod.ccd_trim_slices(
        (50, 50),
        x_start=2,
        x_end=0,
        y_start=3,
        y_end=0,
        end_as_positive_margin=True,
    )
    assert y_sl == slice(3, 50)
    assert x_sl == slice(2, 50)


def test_ccd_trim_slices_alignment_offsets():
    mod = _trim_slices_mod()
    y_sl, x_sl = mod.ccd_trim_slices(
        (100, 100),
        x_start=4,
        x_end=-6,
        y_start=2,
        y_end=-3,
        end_as_positive_margin=False,
    )
    assert y_sl == slice(2, 97)
    assert x_sl == slice(4, 94)


def test_ccd_trim_slices_rejects_empty_window():
    mod = _trim_slices_mod()
    with pytest.raises(ValueError):
        mod.ccd_trim_slices(
            (10, 10),
            x_start=8,
            x_end=5,
            y_start=0,
            y_end=0,
            end_as_positive_margin=True,
        )


def test_trim_ccd_positive_margins():
    pytest.importorskip("ccdproc")
    from astropy.nddata import CCDData
    from ost_photometry.reduce.registration import trim_ccd

    data = np.arange(100, dtype=float).reshape(10, 10)
    ccd = CCDData(data, unit="adu")
    trimmed = trim_ccd(
        ccd,
        x_start=1,
        x_end=2,
        y_start=1,
        y_end=1,
        end_as_positive_margin=True,
    )
    assert trimmed.shape == (8, 7)
    np.testing.assert_array_equal(trimmed.data, data[1:9, 1:8])


def test_trim_ccd_alignment_convention_matches_legacy_slice():
    pytest.importorskip("ccdproc")
    from astropy.nddata import CCDData
    from ost_photometry.reduce.registration import trim_ccd

    data = np.arange(100, dtype=float).reshape(10, 10)
    ccd = CCDData(data, unit="adu")
    x_start, x_end, y_start, y_end = 1, -2, 2, -1
    trimmed = trim_ccd(
        ccd,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        end_as_positive_margin=False,
    )
    expected = data[y_start : 10 + y_end, x_start : 10 + x_end]
    assert trimmed.shape == expected.shape
    np.testing.assert_array_equal(trimmed.data, expected)


def test_aa_common_trim_margins_mixed_and_one_sided_shifts():
    mod = _trim_slices_mod()
    mixed = np.array(
        [
            [-1.2, 0.0, 2.4],
            [-3.1, 1.0, 4.7],
        ]
    )
    assert mod.aa_common_trim_margins(mixed) == (5, -4, 3, -2)

    positive = np.array([[0.2, 1.1], [0.4, 2.2]])
    assert mod.aa_common_trim_margins(positive) == (3, 0, 2, 0)

    negative = np.array([[-2.2, -0.1], [-3.3, -0.4]])
    assert mod.aa_common_trim_margins(negative) == (0, -4, 0, -3)
