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
        assert cfg.isochrone_set is None
        assert cfg.feh is None
        with pytest.raises(FileNotFoundError, match="not found"):
            mod.load_isochrone_config("no_such_isochrones.yaml", ["B", "V"])


def _minimal_isochrone_yaml(**extra: object) -> str:
    body = """
isochrones: "/tmp/iso"
isochrone_type: file
isochrone_column_type:
  B: [single, 0, "-"]
  V: [single, 0, "-"]
isochrone_column:
  B: 1
  V: 2
  AGE: 0
isochrone_keyword: age
isochrone_log_age: true
isochrone_legend: true
"""
    for key, value in extra.items():
        body += f"{key}: {value}\n"
    return body


def test_load_isochrone_config_without_optional_metadata(tmp_path):
    with isolated_sys_modules():
        mod = _prepare()
        path = tmp_path / "iso.yaml"
        path.write_text(_minimal_isochrone_yaml())
        cfg = mod.load_isochrone_config(path, ["B", "V"])
        assert cfg.isochrones == "/tmp/iso"
        assert cfg.isochrone_set is None
        assert cfg.feh is None
        assert cfg.z is None
        assert cfg.y is None
        assert cfg.alpha_fe is None


def test_load_isochrone_config_optional_metadata(tmp_path):
    with isolated_sys_modules():
        mod = _prepare()
        path = tmp_path / "iso.yaml"
        path.write_text(
            _minimal_isochrone_yaml(
                isochrone_set='"BaSTI-IAC"',
                FeH=-1.55,
                Z=0.0004,
                Y=0.2476,
                alpha_Fe=0.0,
            )
        )
        cfg = mod.load_isochrone_config(path, ["B", "V"])
        assert cfg.isochrone_set == "BaSTI-IAC"
        assert cfg.feh == pytest.approx(-1.55)
        assert cfg.z == pytest.approx(0.0004)
        assert cfg.y == pytest.approx(0.2476)
        assert cfg.alpha_fe == pytest.approx(0.0)


def test_format_isochrone_annotation_omits_missing_fields():
    with isolated_sys_modules():
        mod = _prepare()
        text = mod.format_isochrone_annotation(
            apply_corrections_to="observation",
        )
        assert "BaSTI" not in text
        assert r"[\mathrm{Fe}/\mathrm{H}]" not in text
        assert "$Z=" not in text
        assert "Best age" not in text
        assert "Corrections: stars" in text


def test_format_isochrone_annotation_includes_present_fields():
    with isolated_sys_modules():
        mod = _prepare()
        text = mod.format_isochrone_annotation(
            isochrone_set="BaSTI-IAC",
            feh=-1.55,
            z=0.0004,
            y=0.2476,
            alpha_fe=0.0,
            e_b_v=0.12,
            rv=3.1,
            m_m=14.5,
            apply_corrections_to="isochrone",
            best_age=12.0,
            best_age_unit="Gyr",
            chi_square=1.23,
        )
        assert "BaSTI-IAC" in text
        assert r"$[\mathrm{Fe}/\mathrm{H}]=-1.55$" in text
        assert "$Z=0.0004$" in text
        assert "$Y=0.2476$" in text
        assert r"$[\alpha/\mathrm{Fe}]=0$" in text
        assert r"$E(B-V)=0.12$" in text
        assert r"$R_V=3.1$" in text
        assert r"$(m-M)=14.5$" in text
        assert "Corrections: isochrones" in text
        assert r"Best age: $12$ Gyr" in text
        assert r"$\chi^2=1.23$" in text


_GRID_SHARED = """
isochrone_type: file
isochrone_column_type:
  B: [single, 0, "-"]
  V: [single, 0, "-"]
isochrone_column:
  B: 1
  V: 2
  AGE: 0
isochrone_keyword: age
isochrone_log_age: true
isochrone_legend: true
"""


def test_load_isochrone_config_grids_selects_use_and_overrides_metadata(tmp_path):
    with isolated_sys_modules():
        mod = _prepare()
        path = tmp_path / "iso.yaml"
        path.write_text(
            _GRID_SHARED
            + """
isochrone_set: "YY"
Z: 0.99
FeH: 0.0
use: z02
grids:
  z01:
    isochrones: "/tmp/z01"
    Z: 0.01
  z02:
    isochrones: "/tmp/z02"
    Z: 0.02
    Y: 0.27
"""
        )
        cfg = mod.load_isochrone_config(path, ["B", "V"])
        assert cfg.isochrones == "/tmp/z02"
        assert cfg.isochrone_set == "YY"
        assert cfg.z == pytest.approx(0.02)
        assert cfg.y == pytest.approx(0.27)
        assert cfg.feh == pytest.approx(0.0)


def test_load_isochrone_config_grids_inherit_file_level_composition(tmp_path):
    with isolated_sys_modules():
        mod = _prepare()
        path = tmp_path / "iso.yaml"
        path.write_text(
            _GRID_SHARED
            + """
isochrone_set: "PARSEC 3.6"
FeH: 0.0
use: solar_1Gyr
grids:
  solar_0p2Gyr:
    isochrones: "/tmp/0p2"
  solar_1Gyr:
    isochrones: "/tmp/1gyr"
"""
        )
        cfg = mod.load_isochrone_config(path, ["B", "V"])
        assert cfg.isochrones == "/tmp/1gyr"
        assert cfg.feh == pytest.approx(0.0)
        assert cfg.z is None


def test_load_isochrone_config_grids_unknown_use(tmp_path):
    with isolated_sys_modules():
        mod = _prepare()
        path = tmp_path / "iso.yaml"
        path.write_text(
            _GRID_SHARED
            + """
use: missing
grids:
  solar_1Gyr:
    isochrones: "/tmp/1gyr"
"""
        )
        with pytest.raises(ValueError, match="Unknown isochrone grid"):
            mod.load_isochrone_config(path, ["B", "V"])


def test_load_isochrone_config_grids_requires_use(tmp_path):
    with isolated_sys_modules():
        mod = _prepare()
        path = tmp_path / "iso.yaml"
        path.write_text(
            _GRID_SHARED
            + """
grids:
  solar_1Gyr:
    isochrones: "/tmp/1gyr"
"""
        )
        with pytest.raises(ValueError, match="no 'use' key"):
            mod.load_isochrone_config(path, ["B", "V"])


def test_load_isochrone_config_grid_requires_path(tmp_path):
    with isolated_sys_modules():
        mod = _prepare()
        path = tmp_path / "iso.yaml"
        path.write_text(
            _GRID_SHARED
            + """
use: solar_1Gyr
grids:
  solar_1Gyr:
    Z: 0.02
"""
        )
        with pytest.raises(ValueError, match="has no 'isochrones' path"):
            mod.load_isochrone_config(path, ["B", "V"])


def test_flag_cluster_members_and_series_mask():
    with isolated_sys_modules():
        mod = _prepare()
        tbl = Table(
            {
                "id": [1, 2, 3],
                "epoch_id": ["epoch_000", "epoch_000", "epoch_000"],
                "mag_cal_B": [16.0, 17.0, 18.0],
                "mag_cal_V": [15.0, 16.0, 17.0],
                "err_cal_B": [0.03, 0.04, 0.5],
                "err_cal_V": [0.02, 0.05, 0.4],
            }
        )
        tbl.meta["photometry_schema"] = "ost_photometry.epoch_native.v1"
        flagged = mod.flag_cluster_members(
            tbl, np.array([1, 3]), p_mem_by_id={1: 0.95, 3: 0.2}
        )
        np.testing.assert_array_equal(
            flagged["is_cluster_member"], [True, False, True]
        )
        np.testing.assert_allclose(flagged["cluster_p_mem"], [0.95, 0.0, 0.2])
        series = mod.cmd_series_from_table(flagged, "B", "V", do_error_bars=True)
        np.testing.assert_array_equal(
            series.is_cluster_member, [True, False, True]
        )
        clipped = mod.mask_cmd_series(series, max_photometric_err=0.2)
        np.testing.assert_allclose(clipped.color, [1.0, 1.0])
        np.testing.assert_array_equal(clipped.is_cluster_member, [True, False])
        assert mod.cluster_member_flags(tbl) is None


def test_mask_cmd_series_drops_nan_and_large_errors():
    with isolated_sys_modules():
        mod = _prepare()
        series = mod.CmdSeries(
            "B",
            "V",
            np.array([0.5, np.nan, 0.6, 0.7]),
            np.array([15.0, 15.1, 15.2, 15.3]),
            color_err=np.array([0.02, 0.03, 0.04, 0.5]),
            magnitude_filter_2_err=np.array([0.01, 0.02, 0.03, 0.04]),
        )
        finite = mod.mask_cmd_series(series)
        np.testing.assert_allclose(finite.color, [0.5, 0.6, 0.7])
        clipped = mod.mask_cmd_series(series, max_photometric_err=0.2)
        np.testing.assert_allclose(clipped.color, [0.5, 0.6])
        np.testing.assert_allclose(clipped.magnitude_filter_2, [15.0, 15.2])


def test_weighted_chi_square_downweights_large_sigma():
    with isolated_sys_modules():
        mod = _prepare()
        residual = np.array([1.0, 1.0])
        assert mod.weighted_chi_square(residual) == pytest.approx(2.0)
        assert mod.weighted_chi_square(
            residual, np.array([1.0, 10.0])
        ) == pytest.approx(1.01)
        assert mod.weighted_chi_square(residual, np.array([0.0, np.nan])) == (
            pytest.approx(2.0)
        )


def test_fiducial_fit_sigma_photometry_or_scatter():
    with isolated_sys_modules():
        mod = _prepare()
        ivw = mod.fiducial_fit_sigma(np.array([0.1, 0.1]), scatter=0.5, n=2)
        assert ivw == pytest.approx(0.1 / np.sqrt(2))
        fallback = mod.fiducial_fit_sigma(None, scatter=0.4, n=4)
        assert fallback == pytest.approx(0.2)
        assert mod.fiducial_fit_sigma(None, scatter=0.0, n=1) == 1.0
