"""Legacy wide tables remain readable via the epoch-native adapter."""

from __future__ import annotations

import pytest
from astropy.table import Table

from helpers import ensure_stub_package, isolated_sys_modules, load_module_from_path, pkg_src


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _adapters():
    root = pkg_src() / "ost_photometry"
    analyze = root / "analyze"
    ensure_stub_package("ost_photometry", path=root)
    ensure_stub_package("ost_photometry.analyze", path=analyze)
    ensure_stub_package(
        "ost_photometry.analyze.post_processing",
        path=analyze / "post_processing",
    )

    load_module_from_path(
        "ost_photometry.analyze.post_processing.schema",
        pkg_src() / "ost_photometry" / "analyze" / "post_processing" / "schema.py",
    )
    return load_module_from_path(
        "ost_photometry.analyze.post_processing.adapters",
        pkg_src() / "ost_photometry" / "analyze" / "post_processing" / "adapters.py",
    )


def test_legacy_wide_column_i_converts_to_epoch_native_id():
    mod = _adapters()
    wide = Table(
        {
            "i": [7, 8],
            "x": [10.0, 11.0],
            "y": [20.0, 21.0],
            "ra (deg)": [1.0, 1.1],
            "dec (deg)": [2.0, 2.1],
            "V (transformed, image=0)": [12.0, 12.5],
            "V_err (transformed, image=0)": [0.01, 0.02],
        }
    )
    native = mod.legacy_wide_table_to_epoch_native(wide)
    assert "id" in native.colnames
    assert list(native["id"]) == [7, 8]
    assert "mag_cal_V" in native.colnames
    assert native["epoch_id"][0] == "epoch_0"
