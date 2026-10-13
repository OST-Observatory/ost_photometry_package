"""Differential light curves (quiet comparison ensemble) and epoch flags."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from helpers import (
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
    stub_analyze_package,
)

_PP = pkg_src() / "ost_photometry" / "analyze" / "post_processing"

N_EP = 40
N_SRC = 30
OOI = 0
VARIABLES = (1, 2, 3)
ECLIPSE = range(12, 18)
CLOUD_EPOCH = 22
PATCHY_EPOCH = 31


def _load():
    stub_analyze_package("post_processing")
    root = pkg_src() / "ost_photometry"
    load_module_from_path("ost_photometry.terminal_output", root / "terminal_output.py")
    load_module_from_path("ost_photometry.output_layout", root / "output_layout.py")
    lc = load_module_from_path(
        "ost_photometry.analyze.post_processing.light_curve", _PP / "light_curve.py"
    )
    diff = load_module_from_path(
        "ost_photometry.analyze.post_processing.differential_light_curve",
        _PP / "differential_light_curve.py",
    )
    return lc, diff


def _load_plots():
    stub_analyze_package("post_processing", "plots")
    root = pkg_src() / "ost_photometry"
    load_module_from_path("ost_photometry.analyze.warnings_types", root / "analyze" / "warnings_types.py")
    load_module_from_path("ost_photometry.calibration_parameters", root / "calibration_parameters.py")
    load_module_from_path(
        "ost_photometry.analyze.post_processing.magnitude_systems", _PP / "magnitude_systems.py"
    )
    return load_module_from_path(
        "ost_photometry.analyze.plots.lightcurves",
        root / "analyze" / "plots" / "lightcurves.py",
    )


def _field(seed: int = 3) -> Table:
    """One night, 30 sources × 40 epochs of a calibrated epoch-native table.

    Instrumental ``mag_V`` carry extinction (0.2 mag / airmass), a cloud at
    one epoch and patchy clouds at another. ``mag_cal_V`` / ``err_cal_V``
    have a large catalog error on purpose.
    """
    rng = np.random.default_rng(seed)
    base = np.linspace(-11.0, -8.0, N_SRC)
    base[OOI] = -9.5
    noise = 0.004 + 0.004 * (base + 11.0)
    airmass = np.linspace(1.1, 1.8, N_EP)
    rows: dict[str, list] = {
        k: []
        for k in (
            "id", "epoch_id", "ra", "dec", "observation_jd", "airmass_V",
            "mag_V", "err_V", "mag_cal_V", "err_cal_V",
        )
    }
    patchy = rng.normal(0.0, 0.15, N_SRC)
    for e in range(N_EP):
        common = 0.2 * airmass[e] + (0.4 if e == CLOUD_EPOCH else 0.0)
        for s in range(N_SRC):
            m = base[s] + common + rng.normal(0.0, noise[s])
            if s in VARIABLES:
                m += 0.15 * np.sin(2.0 * np.pi * e / (5.0 + s))
            if s == OOI and e in ECLIPSE:
                m += 0.3
            if e == PATCHY_EPOCH:
                m += patchy[s]
            rows["id"].append(s)
            rows["epoch_id"].append(f"epoch_{e:03d}")
            rows["ra"].append(10.0 + 0.001 * s)
            rows["dec"].append(20.0)
            rows["observation_jd"].append(2459000.6 + 0.008 * e)
            rows["airmass_V"].append(airmass[e])
            rows["mag_V"].append(m)
            rows["err_V"].append(noise[s])
            rows["mag_cal_V"].append(m + 25.0)
            rows["err_cal_V"].append(0.08)
    return Table(rows)


def _lc(lc_mod, tbl=None):
    return lc_mod.build_light_curves_table(
        _field() if tbl is None else tbl,
        ["V"],
        object_names={OOI: "eclipser"},
        outlier_sigma=None,
    )


def _ooi_rows(tbl):
    sub = tbl[np.asarray(tbl["id"]).astype(int) == OOI]
    return sub[np.argsort(np.asarray(sub["jd"], dtype=float))]


def test_rows_carry_instrumental_mags_not_calibrated():
    with isolated_sys_modules():
        lc_mod, _diff = _load()
        phot = _field()
        tbl = _lc(lc_mod, phot)
        assert "mag_inst" in tbl.colnames
        row = tbl[(tbl["id"] == 5) & (tbl["epoch_id"] == "epoch_007")][0]
        src = phot[(phot["id"] == 5) & (phot["epoch_id"] == "epoch_007")][0]
        assert row["mag_inst"] == src["mag_V"]
        assert row["mag_inst_err"] == src["err_V"]
        assert row["mag"] == src["mag_cal_V"]


def test_instrumental_columns_prefer_mag_inst_then_mag_then_flux():
    with isolated_sys_modules():
        lc_mod, _diff = _load()
        t = Table({"mag_inst_V": [1.0], "err_inst_V": [0.1], "mag_V": [2.0], "err_V": [0.2]})
        assert lc_mod.epoch_native_instrumental_columns(t, "V")[0] == "mag_inst_V"
        t.remove_columns(["mag_inst_V", "err_inst_V"])
        assert lc_mod.epoch_native_instrumental_columns(t, "V")[0] == "mag_V"
        f = Table({"flux_inst_C": [100.0], "flux_err_inst_C": [1.0]})
        assert lc_mod.epoch_native_instrumental_columns(f, "C")[2] == "flux"
        mag, err = lc_mod.instrumental_mag_from_flux([100.0, -1.0], [1.0, 1.0])
        assert np.isclose(mag[0], -5.0)
        assert np.isclose(err[0], 2.5 / np.log(10.0) * 0.01)
        assert np.isnan(mag[1])


def test_quiet_ensemble_excludes_variables_and_ooi():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        out, stats = diff.add_differential_columns(_lc(lc_mod), {OOI}, verbose=False)
        members = set(np.asarray(stats["id"][stats["is_ensemble"]]).astype(int))
        assert len(members) == 10
        assert not members & {OOI, *VARIABLES}
        ens_rows = out[np.asarray(out["is_ensemble"], dtype=bool)]
        assert set(np.asarray(ens_rows["id"]).astype(int)) == members
        # Similar brightness: every member inside the (possibly widened) window.
        assert np.all(np.asarray(stats["in_window"][stats["is_ensemble"]], dtype=bool))


def test_differential_recovers_eclipse_and_removes_cloud():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        out, _stats = diff.add_differential_columns(_lc(lc_mod), {OOI}, verbose=False)
        ooi = _ooi_rows(out)
        dmag = np.asarray(ooi["dmag"], dtype=float)
        in_ecl = np.isin(np.arange(N_EP), list(ECLIPSE))
        clean = ~in_ecl & ~np.asarray(ooi["flag_epoch"], dtype=bool)
        assert abs(np.median(dmag[in_ecl]) - 0.3) < 0.02
        assert np.std(dmag[clean]) < 0.02
        # The cloud is common mode: gone from the differential curve ...
        assert abs(dmag[CLOUD_EPOCH]) < 0.03
        # ... but visible in the calibrated-mag-free ensemble offset.
        off = np.asarray(ooi["ens_offset"], dtype=float)
        assert off[CLOUD_EPOCH] - np.median(off) > 0.3
        dflux = np.asarray(ooi["dflux"], dtype=float)
        assert np.allclose(dflux, 10.0 ** (-0.4 * dmag))
        assert abs(np.median(dflux[clean]) - 1.0) < 0.01


def test_flag_epoch_marks_cloud_and_patchy_epochs_only():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        out, _stats = diff.add_differential_columns(_lc(lc_mod), {OOI}, verbose=False)
        ooi = _ooi_rows(out)
        flag = np.asarray(ooi["flag_epoch"], dtype=bool)
        reason = np.asarray(ooi["flag_epoch_reason"]).astype(str)
        assert set(np.flatnonzero(flag)) == {CLOUD_EPOCH, PATCHY_EPOCH}
        assert "transparency" in reason[CLOUD_EPOCH]
        assert "ensemble_scatter" in reason[PATCHY_EPOCH]
        # Flag is per epoch: identical for every source of that epoch.
        cloud = out[out["epoch_id"] == f"epoch_{CLOUD_EPOCH:03d}"]
        assert np.all(np.asarray(cloud["flag_epoch"], dtype=bool))
        # Per-source outlier flags are left alone.
        assert not np.any(np.asarray(out["flag_outlier"], dtype=bool))


def test_extinction_trend_alone_is_not_flagged():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        phot = _field()
        keep = ~np.isin(
            np.asarray(phot["epoch_id"]).astype(str),
            [f"epoch_{CLOUD_EPOCH:03d}", f"epoch_{PATCHY_EPOCH:03d}"],
        )
        out, _stats = diff.add_differential_columns(
            _lc(lc_mod, phot[keep]), {OOI}, verbose=False
        )
        assert not np.any(np.asarray(out["flag_epoch"], dtype=bool))


def test_incomplete_ensemble_epoch_is_flagged():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        _out, stats = diff.add_differential_columns(_lc(lc_mod), {OOI}, verbose=False)
        members = np.asarray(stats["id"][stats["is_ensemble"]]).astype(int)
        phot = _field()
        drop = np.isin(np.asarray(phot["id"]).astype(int), members[:8]) & (
            np.asarray(phot["epoch_id"]).astype(str) == "epoch_005"
        )
        phot["mag_V"][drop] = np.nan
        out, _stats = diff.add_differential_columns(_lc(lc_mod, phot), {OOI}, verbose=False)
        ooi = _ooi_rows(out)
        assert bool(ooi["flag_epoch"][5])
        assert "ensemble_incomplete" in str(ooi["flag_epoch_reason"][5])


def test_differential_errors_have_no_catalog_term():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        out, _stats = diff.add_differential_columns(_lc(lc_mod), {OOI}, verbose=False)
        ooi = _ooi_rows(out)
        dmag_err = np.asarray(ooi["dmag_err"], dtype=float)
        inst_err = np.asarray(ooi["mag_inst_err"], dtype=float)
        ens_err = np.asarray(ooi["ens_offset_err"], dtype=float)
        assert np.allclose(dmag_err, np.hypot(inst_err, ens_err))
        assert np.all(dmag_err < 0.03)  # err_cal_V is 0.08
        assert np.all(np.asarray(ooi["mag_err"], dtype=float) == 0.08)


def test_two_star_ensemble_uses_leave_one_out():
    with isolated_sys_modules():
        _lc_mod, diff = _load()
        rng = np.random.default_rng(0)
        mag = rng.normal(0.0, 0.01, (20, 2)) + np.array([-10.0, -9.8])
        err = np.full_like(mag, 0.01)
        valid = np.ones_like(mag, dtype=bool)
        centered = mag - np.median(mag, axis=0)
        cm = diff.ensemble_common_mode(centered, err, valid, np.array([0, 1]), np.zeros(2))
        # Star 0 is compared with star 1 alone, and vice versa.
        assert np.allclose(cm["loo"][:, 0], centered[:, 1])
        assert np.allclose(cm["loo"][:, 1], centered[:, 0])
        assert np.allclose(cm["offset"], centered.mean(axis=1))


def test_colour_rows_have_no_differential_values():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        phot = _field()
        phot["mag_B"] = phot["mag_V"] + 0.5
        phot["err_B"] = phot["err_V"]
        phot["mag_cal_B"] = phot["mag_cal_V"] + 0.5
        phot["err_cal_B"] = phot["err_cal_V"]
        phot["airmass_B"] = phot["airmass_V"]
        tbl = lc_mod.build_light_curves_table(
            phot, ["B", "V"], outlier_sigma=None, color="B-V"
        )
        colour = tbl[np.asarray(tbl["filter"]).astype(str) == "B-V"]
        assert len(colour) > 0
        assert np.all(np.isnan(np.asarray(colour["mag_inst"], dtype=float)))
        out, _stats = diff.add_differential_columns(tbl, {OOI}, verbose=False)
        colour = out[np.asarray(out["filter"]).astype(str) == "B-V"]
        assert np.all(np.isnan(np.asarray(colour["dflux"], dtype=float)))
        assert not np.any(np.asarray(colour["flag_epoch"], dtype=bool))
        v = out[np.asarray(out["filter"]).astype(str) == "V"]
        assert np.all(np.isfinite(np.asarray(v["dflux"], dtype=float)))


def test_table_without_mag_inst_gets_empty_columns():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        tbl = _lc(lc_mod)
        tbl.remove_columns(["mag_inst", "mag_inst_err"])
        out, stats = diff.add_differential_columns(tbl, {OOI}, verbose=False)
        assert len(stats) == 0
        assert np.all(np.isnan(np.asarray(out["dflux"], dtype=float)))
        assert not np.any(np.asarray(out["flag_epoch"], dtype=bool))


def test_differential_and_ensemble_plots_write_files(tmp_path):
    with isolated_sys_modules():
        lc_mod, diff = _load()
        out, _stats = diff.add_differential_columns(_lc(lc_mod), {OOI}, verbose=False)
        plots = _load_plots()
        sub = lc_mod.slice_light_curve(out, OOI, "V")
        assert plots.has_differential_product(sub)
        path = plots.light_curve_jd_from_table(
            sub, str(tmp_path), name_object="eclipser", filter_="V",
            product="differential",
        )
        assert path.name.startswith("lightcurve_diff_jd_")
        assert path.is_file()
        folded = plots.light_curve_fold_from_table(
            sub, str(tmp_path), transit_time="2020-05-31T12:00:00", period=0.1,
            name_object="eclipser", filter_="V", product="differential",
        )
        assert folded.name.startswith("lightcurve_diff_folded_")
        qc = plots.plot_ensemble_qc(
            out, str(tmp_path), filter_="V", ooi_ids=[(OOI, "eclipser")]
        )
        assert qc is not None and qc.is_file()
        # Catalog-transformed view still works and marks bad epochs.
        assert plots.light_curve_jd_from_table(
            sub, str(tmp_path), name_object="eclipser", filter_="V"
        ).is_file()


def test_colour_rows_inherit_epoch_and_outlier_flags():
    with isolated_sys_modules():
        lc_mod, diff = _load()
        phot = _field()
        phot["mag_B"] = phot["mag_V"] + 0.5
        phot["err_B"] = phot["err_V"]
        phot["mag_cal_B"] = phot["mag_cal_V"] + 0.5
        phot["err_cal_B"] = phot["err_cal_V"]
        phot["airmass_B"] = phot["airmass_V"]
        tbl = lc_mod.build_light_curves_table(phot, ["B", "V"], outlier_sigma=None)
        out, _stats = diff.add_differential_columns(tbl, {OOI}, verbose=False)
        # One per-source outlier in B only.
        b_row = np.flatnonzero(
            (np.asarray(out["filter"]).astype(str) == "B")
            & (np.asarray(out["id"]) == 5)
            & (np.asarray(out["epoch_id"]).astype(str) == "epoch_007")
        )[0]
        out["flag_outlier"][b_row] = True
        with_colour = lc_mod.add_color_index_rows(out, "B-V")
        colour = with_colour[np.asarray(with_colour["filter"]).astype(str) == "B-V"]
        cloud = colour[np.asarray(colour["epoch_id"]).astype(str) == f"epoch_{CLOUD_EPOCH:03d}"]
        assert np.all(np.asarray(cloud["flag_epoch"], dtype=bool))
        assert all("transparency" in r for r in np.asarray(cloud["flag_epoch_reason"]).astype(str))
        hit = colour[
            (np.asarray(colour["id"]) == 5)
            & (np.asarray(colour["epoch_id"]).astype(str) == "epoch_007")
        ]
        assert bool(hit["flag_outlier"][0])
        assert int(np.count_nonzero(np.asarray(colour["flag_outlier"], dtype=bool))) == 1
