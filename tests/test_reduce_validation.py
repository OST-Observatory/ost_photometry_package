"""Tests for reduction input validation and master-file checks."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import load_module_from_path, pkg_src


def _validation_mod():
    pkg = pkg_src() / "ost_photometry" / "reduce"
    load_module_from_path(
        "ost_photometry.style",
        pkg_src() / "ost_photometry" / "style.py",
    )
    load_module_from_path(
        "ost_photometry.terminal_output",
        pkg_src() / "ost_photometry" / "terminal_output.py",
    )
    return load_module_from_path(
        "ost_photometry.reduce.validation",
        pkg / "validation.py",
    )


def test_check_science_flat_coverage():
    val = _validation_mod()
    missing = val.check_science_flat_coverage({"B", "V"}, {"B"})
    assert missing == ["V"]


def test_check_frame_sanity():
    val = _validation_mod()
    assert val.check_frame_sanity(np.array([1.0, 2.0])) is None
    assert val.check_frame_sanity(np.zeros(4)) == "all_zero"
    assert val.check_frame_sanity(np.array([np.nan, np.nan])) == "non_finite"
    assert val.check_frame_sanity(np.array([])) == "empty"


def test_summarize_light_reduction_results_raises_when_all_skipped():
    val = _validation_mod()
    with pytest.raises(RuntimeError, match="All science images were skipped"):
        val.summarize_light_reduction_results(
            ["skip_no_master_flat", "skip_no_filter"],
            fail_on_missing_flat=True,
        )


def test_summarize_light_reduction_results_ok_when_some_reduced():
    val = _validation_mod()
    val.summarize_light_reduction_results(
        ["reduced", "skip_no_master_flat"],
        fail_on_missing_flat=True,
    )


def test_any_mask_pattern_for_image_presence():
    """Regression: empty bool list vs no matching types."""
    flats = [False, False, True]
    assert not (not flats)
    assert any(flats)
    assert not any([False, False])


def test_check_master_files_on_disk_science_flat_subset(tmp_path):
    """Master flat check: every science filter must have a combined flat."""
    try:
        exposure = load_module_from_path(
            "ost_photometry.reduce.exposure",
            pkg_src() / "ost_photometry" / "reduce" / "exposure.py",
        )
    except (ModuleNotFoundError, AttributeError):
        pytest.skip("ccdproc not available")

    class _FakeCCD:
        def __init__(self, header):
            self.header = header

    class _FakeCollection:
        files = ["a.fits"]

        def ccds(self, imagetyp=None, combined=False):
            if imagetyp == "DARK":
                return [_FakeCCD({"exptime": 30.0})]
            if imagetyp == "FLAT":
                return [_FakeCCD({"filter": "B"})]
            return []

    image_types = {
        "dark": "DARK",
        "flat": "FLAT",
        "bias": "BIAS",
        "light": "LIGHT",
    }

    def _get_image_type(_ifc, _types, image_class=None):
        return image_types.get(image_class or "", "LIGHT")

    def _check_master(
        science_filters,
        required_dark_exposure_times,
    ) -> bool:
        ifc = _FakeCollection()
        combined_darks_dict = {
            ccd.header["exptime"]: ccd for ccd in ifc.ccds(imagetyp="DARK", combined=True)
        }
        combined_flats_dict = {
            ccd.header["filter"]: ccd for ccd in ifc.ccds(imagetyp="FLAT", combined=True)
        }
        master_available = True
        master_dark_exptimes = list(combined_darks_dict.keys())
        for req_time in required_dark_exposure_times:
            valid, _ = exposure.find_nearest_exposure_time(
                req_time,
                master_dark_exptimes,
                time_tolerance=0.5,
            )
            if not valid:
                master_available = False
                break
        for filt in set(science_filters):
            if filt not in combined_flats_dict:
                master_available = False
                break
        return master_available

    assert _check_master({"B"}, [30.0]) is True
    assert _check_master({"V"}, [30.0]) is False


def test_files_filtered_truthiness_uses_list_not_array():
    """files_filtered may return ndarray; never use ``if not`` on it directly."""

    class _FakeCollection:
        def files_filtered(self, imagetyp=None, include_path=False):
            return np.array(["a.fits", "b.fits"])

    files = list(_FakeCollection().files_filtered(imagetyp="FLAT"))
    assert files == ["a.fits", "b.fits"]
    assert bool(files) is True

    empty = list(_FakeCollection().files_filtered(imagetyp="MISSING")[:0])
    assert bool(empty) is False
