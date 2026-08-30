"""CMD table loading and colour-magnitude series extraction."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from helpers import isolated_sys_modules, load_module_from_path, pkg_src, stub_analyze_package


def _prepare():
    stub_analyze_package()
    return load_module_from_path(
        "ost_photometry.analyze.cmd_prepare",
        pkg_src() / "ost_photometry" / "analyze" / "cmd_prepare.py",
    )


def test_slice_keeps_first_epoch():
    with isolated_sys_modules():
        mod = _prepare()
        tbl = Table(
            {
                "id": [1, 1, 2, 2],
                "epoch_id": ["epoch_000", "epoch_001", "epoch_000", "epoch_001"],
                "mag_cal_B": [15.0, 15.1, 16.0, 16.1],
                "mag_cal_V": [14.0, 14.1, 15.0, 15.1],
            }
        )
        out = mod.slice_cmd_table_to_single_epoch(tbl)
        assert len(out) == 2
        assert set(np.asarray(out["epoch_id"])) == {"epoch_000"}


def test_native_series_and_cali_and_errors():
    with isolated_sys_modules():
        mod = _prepare()
        tbl = Table(
            {
                "id": [1, 2],
                "epoch_id": ["epoch_000", "epoch_000"],
                "mag_cal_B": [16.0, 17.0],
                "mag_cal_V": [15.0, 16.0],
                "err_cal_B": [0.03, 0.04],
                "err_cal_V": [0.02, 0.05],
            }
        )
        tbl.meta["photometry_schema"] = "ost_photometry.epoch_native.v1"
        assert mod.table_is_epoch_native_cmd(tbl)
        series = mod.cmd_series_from_table(
            tbl,
            "B",
            "V",
            do_error_bars=True,
            cali={"B": 0.1, "V": 0.2},
        )
        np.testing.assert_allclose(series.magnitude_filter_2, [15.2, 16.2])
        np.testing.assert_allclose(series.color, [0.9, 0.9])
        np.testing.assert_allclose(
            series.color_err, np.hypot([0.03, 0.04], [0.02, 0.05])
        )
        np.testing.assert_allclose(series.magnitude_filter_2_err, [0.02, 0.05])


def test_student_bracket_mag_columns():
    with isolated_sys_modules():
        mod = _prepare()
        tbl = Table(
            {
                "B [mag]": [16.0, 17.0],
                "V [mag]": [15.0, 16.0],
            }
        )
        assert not mod.table_is_epoch_native_cmd(tbl)
        series = mod.cmd_series_from_table(tbl, "B", "V")
        np.testing.assert_allclose(series.color, [1.0, 1.0])
        np.testing.assert_allclose(series.magnitude_filter_2, [15.0, 16.0])


def test_legacy_wide_transformed_vs_simple():
    with isolated_sys_modules():
        mod = _prepare()
        tbl = Table(
            {
                "B (transformed, image=0)": [16.5],
                "V (transformed, image=0)": [15.5],
                "B (simple, image=0)": [16.0],
                "V (simple, image=0)": [15.0],
            }
        )
        trans = mod.cmd_series_from_table(
            tbl, "B", "V", magnitude_transformation=True
        )
        simple = mod.cmd_series_from_table(
            tbl, "B", "V", magnitude_transformation=False
        )
        np.testing.assert_allclose(trans.color, [1.0])
        np.testing.assert_allclose(trans.magnitude_filter_2, [15.5])
        np.testing.assert_allclose(simple.color, [1.0])
        np.testing.assert_allclose(simple.magnitude_filter_2, [15.0])


def test_distance_modulus():
    with isolated_sys_modules():
        mod = _prepare()
        assert mod.distance_modulus("?", "?") == 0.0
        assert mod.distance_modulus(10.0, "?") == pytest.approx(10.0)
        assert mod.distance_modulus("?", 1.0) == pytest.approx(10.0)


def test_load_cmd_table_ecsv(tmp_path):
    with isolated_sys_modules():
        mod = _prepare()
        path = tmp_path / "mags.ecsv"
        tbl = Table(
            {
                "id": [1, 1],
                "epoch_id": ["epoch_000", "epoch_001"],
                "mag_cal_V": [14.0, 14.5],
            }
        )
        tbl.write(path, format="ascii.ecsv")
        loaded = mod.load_cmd_table(path)
        assert len(loaded) == 1
        assert str(loaded["epoch_id"][0]) == "epoch_000"


def test_load_isochrone_config_blank():
    with isolated_sys_modules():
        mod = _prepare()
        cfg = mod.load_isochrone_config("", ["B", "V"])
        assert cfg.isochrones == ""
        cfg_missing = mod.load_isochrone_config("no_such_isochrones.yaml", ["B", "V"])
        assert cfg_missing.isochrones == ""
