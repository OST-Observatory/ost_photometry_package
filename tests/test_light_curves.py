"""Light-curve table builder, ranking, and plot helpers."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time

from helpers import (
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
    stub_analyze_package,
)


def _load_light_curve():
    stub_analyze_package("post_processing")
    root = pkg_src() / "ost_photometry"
    load_module_from_path("ost_photometry.terminal_output", root / "terminal_output.py")
    load_module_from_path("ost_photometry.output_layout", root / "output_layout.py")
    return load_module_from_path(
        "ost_photometry.analyze.post_processing.light_curve",
        pkg_src() / "ost_photometry" / "analyze" / "post_processing" / "light_curve.py",
    )


def _load_plots():
    stub_analyze_package("post_processing", "plots")
    root = pkg_src() / "ost_photometry"
    load_module_from_path("ost_photometry.terminal_output", root / "terminal_output.py")
    load_module_from_path("ost_photometry.output_layout", root / "output_layout.py")
    load_module_from_path(
        "ost_photometry.analyze.warnings_types",
        pkg_src() / "ost_photometry" / "analyze" / "warnings_types.py",
    )
    load_module_from_path(
        "ost_photometry.calibration_parameters",
        root / "calibration_parameters.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.post_processing.magnitude_systems",
        pkg_src()
        / "ost_photometry"
        / "analyze"
        / "post_processing"
        / "magnitude_systems.py",
    )
    return load_module_from_path(
        "ost_photometry.analyze.plots.lightcurves",
        pkg_src() / "ost_photometry" / "analyze" / "plots" / "lightcurves.py",
    )


def _mini_photometry() -> Table:
    """Four sources × two bands × six epochs."""
    n_ep = 6
    jd0 = 2459000.4
    rows: dict[str, list] = {
        "id": [],
        "epoch_id": [],
        "ra": [],
        "dec": [],
        "observation_jd": [],
        "airmass_V": [],
        "airmass_B": [],
        "mag_cal_V": [],
        "err_cal_V": [],
        "mag_cal_B": [],
        "err_cal_B": [],
        "mag_std_V": [],
    }
    rng = np.random.default_rng(1)
    for eid in range(n_ep):
        jd = jd0 + eid * 0.02
        for sid in range(4):
            mag_v = 12.0 + 0.02 * sid
            if sid == 0:
                mag_v += 0.20 * np.sin(2.0 * np.pi * eid / n_ep)
            if sid == 1:
                mag_v += 0.12 * np.sin(2.0 * np.pi * eid / 3.0)
            err = 0.01 if sid < 3 else 0.45
            mag_v = mag_v + rng.normal(0.0, err)
            mag_b = mag_v + 0.35
            rows["id"].append(sid)
            rows["epoch_id"].append(f"epoch_{eid:03d}")
            rows["ra"].append(290.0)
            rows["dec"].append(54.8)
            rows["observation_jd"].append(jd)
            rows["airmass_V"].append(1.1 + 0.05 * eid)
            rows["airmass_B"].append(1.12 + 0.05 * eid)
            rows["mag_cal_V"].append(mag_v)
            rows["err_cal_V"].append(err)
            rows["mag_cal_B"].append(mag_b)
            rows["err_cal_B"].append(err)
            rows["mag_std_V"].append(12.0 if sid in (1, 2, 3) else np.nan)
    tbl = Table(rows)
    idx = int(np.flatnonzero((tbl["id"] == 2) & (tbl["epoch_id"] == "epoch_002"))[0])
    tbl["mag_cal_V"][idx] = 14.5
    return tbl


def test_night_id_groups_evening_and_morning():
    with isolated_sys_modules():
        lc = _load_light_curve()
        jd = np.array([2459000.8, 2459001.2])
        nights = lc.night_id_from_jd(jd)
        assert nights[0] == nights[1]
        assert int(nights[0]) == 2459000


def test_build_light_curves_table_columns_airmass_and_nights():
    with isolated_sys_modules():
        mod = _load_light_curve()
        phot = _mini_photometry()
        tbl = mod.build_light_curves_table(
            phot,
            ["B", "V"],
            object_names={0: "var star"},
            calibrator_ids={1, 2, 3},
            outlier_sigma=5.0,
            color="B-V",
        )
        for col in (
            "id",
            "filter",
            "jd",
            "bjd_tdb",
            "airmass",
            "night_id",
            "mag",
            "mag_err",
            "flag_outlier",
            "quantity",
        ):
            assert col in tbl.colnames
        v = tbl[np.asarray(tbl["filter"]).astype(str) == "V"]
        assert np.all(np.isfinite(np.asarray(v["airmass"], dtype=float)))
        assert "B-V" in set(np.asarray(tbl["filter"]).astype(str))
        names = np.asarray(tbl["object_name"]).astype(str)
        assert np.any(names == "var star")
        flagged = tbl[
            (np.asarray(tbl["id"]).astype(int) == 2)
            & (np.asarray(tbl["filter"]).astype(str) == "V")
        ]
        assert np.any(np.asarray(flagged["flag_outlier"], dtype=bool))


def test_y_limits_follow_quantity_not_median():
    with isolated_sys_modules():
        plots = _load_plots()
        mag = np.array([0.4, 0.45, 0.5, 0.42])
        lo_hi = plots.y_limits_for_quantity(mag, quantity="magnitude")
        assert lo_hi[0] > lo_hi[1]
        flux = np.array([0.95, 1.0, 1.05, 0.98])
        flo, fhi = plots.y_limits_for_quantity(flux, quantity="flux")
        assert flo < fhi


def test_fold_phase_in_unit_interval():
    with isolated_sys_modules():
        plots = _load_plots()
        t0 = 2459000.0
        p = 1.234
        t = np.array([t0, t0 + 0.25 * p, t0 + 1.25 * p, t0 + 3.0 * p])
        ph = plots.fold_phase(t, t0, p)
        assert np.all((ph >= 0.0) & (ph < 1.0))
        assert ph[1] == pytest.approx(0.25, abs=1e-9)
        assert ph[2] == pytest.approx(0.25, abs=1e-9)
        assert ph[3] == pytest.approx(0.0, abs=1e-9)


def test_bjd_tdb_offset_is_light_travel_time():
    with isolated_sys_modules():
        mod = _load_light_curve()
        n = 3
        jd = np.full(n, 2459000.5)
        tbl = Table(
            {
                "jd": jd,
                "ra": np.full(n, 0.0),
                "dec": np.full(n, 0.0),
            }
        )
        loc = EarthLocation(
            lat=52.409184 * u.deg, lon=12.973185 * u.deg, height=39 * u.m
        )
        out = mod.add_bjd_tdb_column(tbl, loc)
        bjd = np.asarray(out["bjd_tdb"], dtype=float)
        delta = bjd - jd
        assert np.all(np.isfinite(bjd))
        assert np.all(np.abs(delta) < 0.01)
        assert np.all(np.abs(delta) > 1e-6)
        t = Time(jd[0], format="jd", scale="utc")
        ltt = t.light_travel_time(
            SkyCoord(0 * u.deg, 0 * u.deg, frame="icrs"),
            kind="barycentric",
            location=loc,
        )
        assert bjd[0] == pytest.approx(float((t.tdb + ltt).jd), rel=0, abs=1e-8)


def test_color_rows_from_matched_epoch_id():
    with isolated_sys_modules():
        mod = _load_light_curve()
        phot = _mini_photometry()
        tbl = mod.build_light_curves_table(phot, ["B", "V"], outlier_sigma=None)
        tbl = mod.add_color_index_rows(tbl, "B-V")
        color = tbl[np.asarray(tbl["filter"]).astype(str) == "B-V"]
        assert len(color) > 0
        one = color[np.asarray(color["id"]).astype(int) == 2][0]
        eid = str(one["epoch_id"])
        sid = 2
        b = tbl[
            (np.asarray(tbl["id"]).astype(int) == sid)
            & (np.asarray(tbl["filter"]).astype(str) == "B")
            & (np.asarray(tbl["epoch_id"]).astype(str) == eid)
        ]
        v = tbl[
            (np.asarray(tbl["id"]).astype(int) == sid)
            & (np.asarray(tbl["filter"]).astype(str) == "V")
            & (np.asarray(tbl["epoch_id"]).astype(str) == eid)
        ]
        assert float(one["mag"]) == pytest.approx(
            float(b["mag"][0]) - float(v["mag"][0]), abs=1e-9
        )


def test_outlier_flag_on_injected_spike():
    with isolated_sys_modules():
        mod = _load_light_curve()
        phot = _mini_photometry()
        tbl = mod.build_light_curves_table(phot, ["V"], outlier_sigma=None)
        tbl = mod.flag_outliers_in_light_curves(tbl, sigma=5.0)
        v2 = tbl[
            (np.asarray(tbl["id"]).astype(int) == 2)
            & (np.asarray(tbl["filter"]).astype(str) == "V")
        ]
        assert np.any(np.asarray(v2["flag_outlier"], dtype=bool))


def test_excess_rms_ranking_not_raw_rms():
    with isolated_sys_modules():
        mod = _load_light_curve()
        phot = _mini_photometry()
        tbl = mod.build_light_curves_table(
            phot,
            ["V"],
            calibrator_ids={1, 2, 3},
            outlier_sigma=5.0,
        )
        stats = mod.calibrator_variability_stats(tbl, {1, 2, 3}, "V")
        top = mod.top_variable_calibrator_ids(stats, n=1)
        assert top[0] == 1
        ids = list(np.asarray(stats["id"]).astype(int))
        exc = {
            int(i): float(e) for i, e in zip(ids, stats["excess_rms"], strict=True)
        }
        assert exc[1] > exc[3]
        noisy = tbl[
            (np.asarray(tbl["id"]).astype(int) == 3)
            & (np.asarray(tbl["filter"]).astype(str) == "V")
            & (~np.asarray(tbl["flag_outlier"], dtype=bool))
        ]
        y = np.asarray(noisy["mag"], dtype=float)
        raw = float(np.sqrt(np.mean((y - np.median(y)) ** 2)))
        assert raw > exc[3]


def test_ids_excluding_drops_ooi_from_calibrator_pool():
    with isolated_sys_modules():
        mod = _load_light_curve()
        assert mod.ids_excluding({1, 38, 40}, {38}) == {1, 40}
        assert mod.ids_excluding([38], [38]) == set()
        assert mod.ids_excluding({1, 2}, None) == {1, 2}


def test_top_variable_calibrator_ids_skips_excluded_ooi():
    with isolated_sys_modules():
        mod = _load_light_curve()
        phot = _mini_photometry()
        tbl = mod.build_light_curves_table(
            phot,
            ["V"],
            calibrator_ids={1, 2, 3},
            outlier_sigma=5.0,
        )
        stats = mod.calibrator_variability_stats(tbl, {1, 2, 3}, "V")
        # id 1 ranks first; excluding it should promote the next calibrator.
        top_all = mod.top_variable_calibrator_ids(stats, n=2)
        assert top_all[0] == 1
        top = mod.top_variable_calibrator_ids(stats, n=2, exclude={1})
        assert 1 not in top
        assert len(top) == 2


def test_check_star_qc_panels_do_not_repeat_ooi():
    with isolated_sys_modules():
        mod = _load_light_curve()
        phot = _mini_photometry()
        tbl = mod.build_light_curves_table(
            phot,
            ["V"],
            object_names={1: "V* demo"},
            calibrator_ids={1, 2, 3},
        )
        panels = mod.build_check_star_qc_panels(
            tbl,
            "V",
            ooi_ids=[(1, "V* demo")],
            calibrator_ids=[1, 2, 3],
        )
        titles = [title for title, _sub in panels]
        assert titles[0] == "object of interest V* demo (id=1)"
        assert sum("id=1" in t for t in titles) == 1
        assert [t for t in titles if t.startswith("catalog calibrator")] == [
            "catalog calibrator id=2 (#1 by excess RMS)",
            "catalog calibrator id=3 (#2 by excess RMS)",
        ]

def test_excess_rms_helper():
    with isolated_sys_modules():
        mod = _load_light_curve()
        y = np.array([1.0, 1.0, 1.0, 1.2])
        e = np.array([0.01, 0.01, 0.01, 0.01])
        assert mod.excess_rms(y, e) > 0.05
        y_n = np.array([10.0, 10.4, 9.6, 10.1])
        e_n = np.array([0.5, 0.5, 0.5, 0.5])
        assert mod.excess_rms(y_n, e_n) < mod.excess_rms(y, e)


def test_jd_and_folded_plots_write_files(tmp_path):
    with isolated_sys_modules():
        lc_mod = _load_light_curve()
        plots = _load_plots()
        phot = _mini_photometry()
        tbl = lc_mod.build_light_curves_table(phot, ["V"], object_names={0: "star"})
        sub = lc_mod.slice_light_curve(tbl, 0, "V")
        path = plots.light_curve_jd_from_table(
            sub, str(tmp_path), name_object="star", filter_="V"
        )
        assert path.is_file()
        folded = plots.light_curve_fold_from_table(
            sub,
            str(tmp_path),
            transit_time="2020-05-31T12:00:00",
            period=0.12,
            name_object="star",
            filter_="V",
            phase_cycles=2,
            binning_factor=0.2,
        )
        assert folded.is_file()


def test_epoch_has_catalog_calibrated_mags_ignores_instrumental():
    with isolated_sys_modules():
        mod = _load_light_curve()
        cal = Table(
            {
                "id": [0],
                "epoch_id": ["epoch_000"],
                "mag_cal_V": [12.0],
                "err_cal_V": [0.01],
            }
        )
        inst = Table(
            {
                "id": [0],
                "epoch_id": ["epoch_000"],
                "mag_inst_Clear": [14.0],
                "err_inst_Clear": [0.02],
            }
        )
        assert mod.epoch_has_catalog_calibrated_mags(cal, "V")
        assert not mod.epoch_has_catalog_calibrated_mags(inst, "Clear")
        assert not mod.epoch_has_catalog_calibrated_mags(cal, "Clear")
