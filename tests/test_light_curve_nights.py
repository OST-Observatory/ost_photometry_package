"""Combine and plot light curves from several nights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ost_photometry.analyze.plots.lightcurves import (
    light_curve_nights_fold_from_table,
    light_curve_nights_jd_from_table,
    light_curve_nights_panels_from_table,
    night_date_label,
    unique_night_ids,
)
from ost_photometry.analyze.post_processing.light_curve import (
    apply_nightly_zero_point,
    combine_light_curve_night_tables,
    night_id_from_jd,
    plot_nights_compare_from_table,
    resolve_light_curves_ecsv_path,
)


def _long_table(*, source_id: int, jd0: float, mag0: float, n: int = 4) -> Table:
    jd = jd0 + np.arange(n) * 0.015
    return Table(
        {
            "id": np.full(n, source_id, dtype=np.int64),
            "object_name": np.array(["RZCas"] * n),
            "filter": np.array(["V"] * n),
            "epoch_id": np.array([f"e{i}" for i in range(n)]),
            "jd": jd,
            "bjd_tdb": jd,
            "airmass": np.full(n, 1.2),
            "night_id": night_id_from_jd(jd),
            "mag": mag0 + 0.02 * np.arange(n),
            "mag_err": np.full(n, 0.01),
            "flux": np.ones(n),
            "flux_err": np.full(n, 0.01),
            "quantity": np.array(["magnitude"] * n),
            "flag_outlier": np.zeros(n, dtype=bool),
            "ra": np.full(n, 20.0),
            "dec": np.full(n, 35.0),
            "is_calibrator": np.zeros(n, dtype=bool),
        }
    )


def test_night_date_label_and_unique_ids():
    jd = np.array([2459000.4, 2459000.45, 2459001.4])
    nid = night_id_from_jd(jd)
    assert len(np.unique(nid)) == 2
    tbl = Table({"night_id": nid, "jd": jd})
    nights = unique_night_ids(tbl)
    assert nights.size == 2
    lab = night_date_label(jd[:2])
    assert len(lab) == 10 and lab[4] == "-" and lab[7] == "-"


def test_apply_nightly_zero_point_subtracts_median():
    a = _long_table(source_id=1, jd0=2459000.4, mag0=12.0)
    b = _long_table(source_id=1, jd0=2459001.4, mag0=12.4)
    tbl = Table()
    for col in a.colnames:
        tbl[col] = np.concatenate([np.asarray(a[col]), np.asarray(b[col])])
    out = apply_nightly_zero_point(tbl)
    n1 = np.asarray(out["night_id"]) == int(a["night_id"][0])
    n2 = np.asarray(out["night_id"]) == int(b["night_id"][0])
    assert np.isclose(np.median(out["mag"][n1]), 0.0, atol=1e-12)
    assert np.isclose(np.median(out["mag"][n2]), 0.0, atol=1e-12)


def test_combine_matches_by_sky_across_different_ids(tmp_path: Path):
    t1 = _long_table(source_id=7, jd0=2459000.4, mag0=12.0)
    t2 = _long_table(source_id=99, jd0=2459001.4, mag0=12.1)
    p1 = tmp_path / "n1" / "tables"
    p2 = tmp_path / "n2" / "tables"
    p1.mkdir(parents=True)
    p2.mkdir(parents=True)
    t1.write(p1 / "light_curves.ecsv", format="ascii.ecsv")
    t2.write(p2 / "light_curves.ecsv", format="ascii.ecsv")
    sky = SkyCoord(ra=20.0, dec=35.0, unit="deg")
    combined = combine_light_curve_night_tables(
        [tmp_path / "n1", tmp_path / "n2"],
        "V",
        coord=sky,
    )
    assert len(combined) == 8
    assert set(np.unique(combined["id"])) == {0}
    assert unique_night_ids(combined).size == 2
    assert resolve_light_curves_ecsv_path(tmp_path / "n1").name == "light_curves.ecsv"


def test_nights_plots_write_files(tmp_path: Path):
    a = _long_table(source_id=1, jd0=2459000.4, mag0=12.0)
    b = _long_table(source_id=1, jd0=2459001.4, mag0=12.1)
    tbl = Table()
    for col in a.colnames:
        tbl[col] = np.concatenate([np.asarray(a[col]), np.asarray(b[col])])
    out = tmp_path / "out"
    jd_path = light_curve_nights_jd_from_table(
        tbl, str(out), name_object="RZCas", filter_="V", file_type="pdf"
    )
    panel_path = light_curve_nights_panels_from_table(
        tbl, str(out), name_object="RZCas", filter_="V", file_type="pdf"
    )
    fold_path = light_curve_nights_fold_from_table(
        tbl,
        str(out),
        transit_time="2020-05-30T21:00:00",
        period=1.2,
        name_object="RZCas",
        filter_="V",
        file_type="pdf",
    )
    assert jd_path.is_file()
    assert panel_path.is_file()
    assert fold_path.is_file()
    plot_nights_compare_from_table(
        tbl,
        1,
        "V",
        str(out / "cmp"),
        name_object="RZCas",
        transit_time="2020-05-30T21:00:00",
        period=1.2,
    )
    lc_dir = out / "cmp" / "results" / "lightcurves"
    assert (lc_dir / "lightcurve_nights_jd_RZCas_V.pdf").is_file()
    assert (lc_dir / "lightcurve_nights_folded_RZCas_V.pdf").is_file()
    assert (lc_dir / "lightcurve_nights_panels_RZCas_V.pdf").is_file()
