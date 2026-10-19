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
    load_module_from_path(
        "ost_photometry.analyze.post_processing.light_curve",
        pkg_src() / "ost_photometry" / "analyze" / "post_processing" / "light_curve.py",
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
        evening = Time("2022-01-06T19:00:00", scale="utc").jd
        after_midnight = Time("2022-01-07T00:30:00", scale="utc").jd
        morning = Time("2022-01-07T05:00:00", scale="utc").jd
        next_evening = Time("2022-01-07T19:00:00", scale="utc").jd
        nights = lc.night_id_from_jd([evening, after_midnight, morning, next_evening])
        assert nights[0] == nights[1] == nights[2]
        assert nights[3] == nights[0] + 1
        # Integer JD of the noon UT before the night.
        assert Time(float(nights[0]), format="jd").iso.startswith("2022-01-06 12:00")
        assert lc.night_id_from_jd([np.nan])[0] == -1


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


def test_y_limits_include_flux_eclipse_dip():
    with isolated_sys_modules():
        plots = _load_plots()
        flux = np.concatenate([np.ones(20), np.full(5, 0.55)])
        lo, hi = plots.y_limits_for_quantity(flux, quantity="flux")
        assert lo < 0.55 < hi
        assert lo < 1.0 < hi
        mag = np.concatenate([np.full(20, 12.0), np.full(5, 12.8)])
        hi_m, lo_m = plots.y_limits_for_quantity(mag, quantity="magnitude")
        assert lo_m < 12.0 < hi_m
        assert lo_m < 12.8 < hi_m


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


def test_point_error_and_common_offset_from_split_columns():
    """mag_err = rand ⊕ epoch, mag_err_sys = common; colour rows use B-V columns."""
    with isolated_sys_modules():
        mod = _load_light_curve()
        phot = _mini_photometry()
        n = len(phot)
        for f in ("B", "V"):
            phot[f"err_cal_rand_{f}"] = np.full(n, 0.03)
            phot[f"err_cal_epoch_{f}"] = np.full(n, 0.04)
            phot[f"err_cal_common_{f}"] = np.full(n, 0.02)
        phot["err_cal_rand_B-V"] = np.full(n, 0.01)
        phot["err_cal_epoch_B-V"] = np.full(n, 0.0)
        phot["err_cal_common_B-V"] = np.full(n, 0.005)
        tbl = mod.build_light_curves_table(phot, ["B", "V"], outlier_sigma=None, color="B-V")
        filt = np.asarray(tbl["filter"]).astype(str)
        v = tbl[filt == "V"]
        assert np.allclose(v["mag_err"], 0.05)
        assert np.allclose(v["mag_err_sys"], 0.02)
        color = tbl[filt == "B-V"]
        assert len(color) > 0
        assert np.allclose(color["mag_err"], 0.01)
        assert np.allclose(color["mag_err_sys"], 0.005)
        # without the colour columns: quadrature of the components
        phot.remove_columns(["err_cal_rand_B-V", "err_cal_epoch_B-V", "err_cal_common_B-V"])
        tbl = mod.build_light_curves_table(phot, ["B", "V"], outlier_sigma=None, color="B-V")
        color = tbl[np.asarray(tbl["filter"]).astype(str) == "B-V"]
        assert np.allclose(color["mag_err"], np.hypot(0.05, 0.05))


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


def test_eclipse_dip_not_flagged_isolated_spike_is():
    with isolated_sys_modules():
        mod = _load_light_curve()
        n = 20
        rng = np.random.default_rng(0)
        mag = 12.0 + rng.normal(0.0, 0.01, n)
        mag[8:13] = 12.6 + rng.normal(0.0, 0.01, 5)
        mag[16] = 14.5
        tbl = Table(
            {
                "id": np.zeros(n, dtype=np.int64),
                "filter": np.full(n, "V", dtype="U8"),
                "quantity": np.full(n, "magnitude", dtype="U16"),
                "mag": mag,
                "flux": np.full(n, np.nan),
                "jd": 2459000.4 + np.arange(n) * 0.01,
            }
        )
        out = mod.flag_outliers_in_light_curves(tbl, sigma=5.0)
        flag = np.asarray(out["flag_outlier"], dtype=bool)
        assert not np.any(flag[8:13])
        assert flag[16]
        assert int(np.count_nonzero(flag)) == 1


def test_two_consecutive_spikes_flagged_eclipse_kept():
    with isolated_sys_modules():
        mod = _load_light_curve()
        plots = _load_plots()
        n = 24
        rng = np.random.default_rng(1)
        mag = 12.0 + rng.normal(0.0, 0.01, n)
        mag[6:11] = 12.7 + rng.normal(0.0, 0.01, 5)
        mag[18:20] = 15.0
        tbl = Table(
            {
                "id": np.zeros(n, dtype=np.int64),
                "filter": np.full(n, "Clear", dtype="U8"),
                "quantity": np.full(n, "magnitude", dtype="U16"),
                "mag": mag,
                "flux": np.full(n, np.nan),
                "jd": 2459000.4 + np.arange(n) * 0.01,
            }
        )
        out = mod.flag_outliers_in_light_curves(tbl, sigma=5.0)
        flag = np.asarray(out["flag_outlier"], dtype=bool)
        assert not np.any(flag[6:11])
        assert np.all(flag[18:20])
        good = mag[~flag]
        hi, lo = plots.y_limits_for_quantity(good, quantity="magnitude")
        assert lo < 12.0 < hi
        assert lo < 12.7 < hi
        assert not (lo < 15.0 < hi)


def test_bright_spike_run_is_flagged():
    with isolated_sys_modules():
        mod = _load_light_curve()
        n = 20
        rng = np.random.default_rng(2)
        mag = 12.0 + rng.normal(0.0, 0.01, n)
        mag[4:9] = 10.5
        tbl = Table(
            {
                "id": np.zeros(n, dtype=np.int64),
                "filter": np.full(n, "V", dtype="U8"),
                "quantity": np.full(n, "magnitude", dtype="U16"),
                "mag": mag,
                "flux": np.full(n, np.nan),
                "jd": 2459000.4 + np.arange(n) * 0.01,
            }
        )
        out = mod.flag_outliers_in_light_curves(tbl, sigma=5.0)
        flag = np.asarray(out["flag_outlier"], dtype=bool)
        assert np.all(flag[4:9])
        assert not np.any(flag[:4])
        assert not np.any(flag[9:])


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


def test_position_scatter_flags_mixed_track():
    with isolated_sys_modules():
        mod = _load_light_curve()
        ra = np.array([120.0, 120.0, 120.0 + 5.0 / 3600.0 / np.cos(np.deg2rad(30.0))])
        dec = np.array([30.0, 30.0 + 0.1 / 3600.0, 30.0])
        rms, worst = mod.position_scatter_arcsec(ra, dec)
        assert 4.9 < worst < 5.1
        assert rms < worst
        assert np.isnan(mod.position_scatter_arcsec([1.0], [1.0])[1])


def test_calibrator_stats_report_position_scatter():
    with isolated_sys_modules():
        mod = _load_light_curve()
        phot = _mini_photometry()
        tbl = mod.build_light_curves_table(phot, ["V"], calibrator_ids={1, 2, 3})
        stats = mod.calibrator_variability_stats(tbl, {1, 2, 3}, "V")
        assert "pos_max_arcsec" in stats.colnames
        assert "pos_rms_arcsec" in stats.colnames
        assert len(stats) == 3


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


def test_phase_bins_weighted_skip_flags_and_never_zero_error():
    with isolated_sys_modules():
        plots = _load_plots()
        phase = np.array([0.05, 0.06, 0.07, 0.08, 0.55, 0.56, 0.57])
        y = np.array([1.0, 1.0, 1.0, 5.0, 2.0, 2.0, 2.0])
        e = np.array([0.1, 0.1, 0.2, 0.1, 0.1, np.nan, 0.1])
        flagged = np.array([False, False, False, True, False, False, False])
        cx, cy, ce = plots._phase_bin_centers_and_means(phase, y, e, 2, flagged=flagged)
        # The flagged 5.0 is left out; inverse-variance mean of 1,1,1 is 1.
        assert cy[0] == pytest.approx(1.0)
        assert ce[0] == pytest.approx(1.0 / np.sqrt(100 + 100 + 25))
        # NaN-error point gets no weight; n = 2 -> propagated error only.
        assert cy[1] == pytest.approx(2.0)
        assert ce[1] == pytest.approx(0.1 / np.sqrt(2))
        assert np.all(ce > 0)


def test_outlier_clip_uses_errors_and_ignores_zero_mad():
    with isolated_sys_modules():
        mod = _load_light_curve()
        rng = np.random.default_rng(4)
        n = 200
        err = np.full(n, 0.01)
        err[50] = 0.2  # faint point with a big error: 0.3 mag off is 1.5 σ
        y = rng.normal(0.0, 1.0, n) * err
        y[50] = 0.3
        y[120] = 0.08  # 8 σ with a 0.01 error
        cand = mod._clip_candidates(y, 5.0, err)
        assert not cand[50] and cand[120]
        # without errors the big-error point would be clipped
        assert mod._clip_candidates(y, 5.0)[50]
        # constant (quantised) series: nothing flagged
        q = np.zeros(20)
        q[3] = 0.001
        assert not mod._clip_candidates(q, 5.0).any()


def test_rows_get_their_own_filters_time():
    with isolated_sys_modules():
        lc = _load_light_curve()
        meta = {
            "e1": {"jd_by_filter": {"B": 2459000.10, "V": 2459000.12}},
            "e2": {"jd_by_filter": {"B": 2459000.20, "V": 2459000.23}},
        }
        tbl = Table({"id": [1, 1], "epoch_id": ["e1", "e2"]})
        out = lc.attach_observation_jd_column(tbl, meta, "B")
        assert np.allclose(out["observation_jd"], [2459000.10, 2459000.20])
        assert np.allclose(lc._row_jds_from_table(out, meta, "V"), [2459000.12, 2459000.23])
        assert np.allclose(lc._row_jds_from_table(out, meta, "B"), [2459000.10, 2459000.20])
        # old table: only the reference filter's column, epoch_meta still knows V
        old = Table({"id": [1, 1], "epoch_id": ["e1", "e2"],
                     "observation_jd": [2459000.10, 2459000.20]})
        assert np.allclose(lc._row_jds_from_table(old, meta, "V"), [2459000.12, 2459000.23])


def test_transit_time_is_read_as_bjd_tdb():
    with isolated_sys_modules():
        plots = _load_plots()
        assert plots.transit_time_bjd_tdb(2459111.0417) == pytest.approx(2459111.0417)
        assert plots.transit_time_bjd_tdb("2459111.0417") == pytest.approx(2459111.0417)
        iso = plots.transit_time_bjd_tdb("2020-09-18T01:00:00")
        assert iso == pytest.approx(Time("2020-09-18T01:00:00", scale="tdb").jd, abs=1e-9)
        with pytest.raises(ValueError, match="full BJD_TDB"):
            plots.transit_time_bjd_tdb(9111.0417)


def test_fold_never_mixes_bjd_and_jd():
    with isolated_sys_modules():
        plots = _load_plots()
        tbl = Table({"jd": [2459000.0, 2459000.1, 2459000.2],
                     "bjd_tdb": [2459000.004, np.nan, 2459000.204]})
        t = plots._fold_times(tbl, "bjd_tdb")
        assert np.isnan(t[1])
        assert t[0] == pytest.approx(2459000.004)
        no_bjd = Table({"jd": [2459000.0, 2459000.1]})
        assert np.allclose(plots._fold_times(no_bjd, "bjd_tdb"), [2459000.0, 2459000.1])


def test_transit_points_are_not_outliers_and_do_not_set_the_continuum():
    with isolated_sys_modules():
        lc = _load_light_curve()
        n = 120
        t0, period = 2460000.60, 3.0
        bjd = t0 - 0.10 + np.arange(n) * (0.2 / n)          # one night around a transit
        mag = np.full(n, 12.0) + np.random.default_rng(1).normal(0, 0.002, n)
        in_transit = np.abs(bjd - t0) < 0.01               # 29 min: shorter than the window
        mag[in_transit] += 0.05
        tbl = Table({"id": np.ones(n, dtype=int), "filter": np.full(n, "V"), "quantity": np.full(n, "magnitude"),
                     "jd": bjd, "bjd_tdb": bjd, "mag": mag, "mag_err": np.full(n, 0.002),
                     "flux": np.full(n, np.nan), "flux_err": np.full(n, np.nan), "night_id": np.zeros(n, dtype=int)})
        plain = lc.flag_outliers_in_light_curves(tbl, sigma=5.0, max_faint_minutes=60.0)
        assert np.any(np.asarray(plain["flag_outlier"])[in_transit])
        kept = lc.flag_outliers_in_light_curves(tbl, sigma=5.0, max_faint_minutes=60.0,
                                                ephemerides={1: (period, t0)})
        assert not np.any(np.asarray(kept["flag_outlier"])[in_transit])
        norm = lc.apply_nightly_zero_point(kept, ephemerides={1: (period, t0)})
        assert abs(np.median(np.asarray(norm["mag"])[~in_transit])) < 0.001
