"""Bias / dark choice, flat probabilities and the full calibration plan."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from ost_photometry.archive.local import manifest_from_directory
from ost_photometry.archive.manifest import manifest_from_rows
from ost_photometry.reduce.grouping.darks import (
    add_electronic_ids,
    assign_bias_dark,
    calibration_sets,
    night_of,
)
from ost_photometry.reduce.grouping.flats import (
    CERTAIN,
    REJECTED,
    FlatImages,
    FlatSet,
    dust_similarity,
    score_flat_set,
    vignetting_difference,
)
from ost_photometry.reduce.grouping.plan import (
    Overrides,
    PlanSettings,
    build_calibration_plan,
    load_plan,
    read_overrides,
    write_frames,
    write_plan,
)
from ost_photometry.reduce.grouping.sessions import Session
from test_grouping_classify import _bias, _dark, _flat, _light

M104 = (189.998, -11.623)
M57 = (283.396, 33.029)
JD_NIGHT1 = 2459647.0  # 2022-03-08 12:00 UT; +0.4 is the evening


# ---------------------------------------------------------------------------
# Bias / darks
# ---------------------------------------------------------------------------


def _calib_row(fid, kind, jd, exptime, offset=5, temp=-20.0, **extra):
    row = {"frame_id": str(fid), "frame_type": kind, "jd": jd, "exptime": exptime,
           "instrume": "QHY600M", "xbinning": 3, "ybinning": 3, "readoutm": "Normal",
           "gain": 0, "offset": offset, "set_temp": temp, "naxis1": 3192, "naxis2": 2129}
    row.update(extra)
    return row


def _with_types(rows):
    table = manifest_from_rows(rows)
    table["frame_type"] = np.array([r["frame_type"] for r in sorted(
        rows, key=lambda r: r["jd"])], dtype=str)
    return add_electronic_ids(table)


def test_electronic_ids_split_offset_and_temperature():
    table = _with_types([
        _calib_row(1, "bias", JD_NIGHT1, 0.0),
        _calib_row(2, "bias", JD_NIGHT1 + 0.01, 0.0, offset=2),
        _calib_row(3, "bias", JD_NIGHT1 + 0.02, 0.0, temp=-10.0),
        _calib_row(4, "bias", JD_NIGHT1 + 0.03, 0.0, temp=-19.0),
    ])
    ids = dict(zip(table["frame_id"], table["electronic_id"], strict=True))
    assert ids["1"] == ids["4"]
    assert len({ids["1"], ids["2"], ids["3"]}) == 3
    assert set(table["night"]) == {"20220308"}


def test_assign_bias_dark_nearest_covering_night():
    frames = _with_types([
        _calib_row(1, "bias", JD_NIGHT1, 0.0),
        _calib_row(2, "dark", JD_NIGHT1, 60.0),
        _calib_row(3, "dark", JD_NIGHT1 + 5, 60.0),
        _calib_row(4, "dark", JD_NIGHT1 + 5, 120.0),
        _calib_row(5, "dark", JD_NIGHT1 + 40, 120.0),  # outside the window
    ])
    consumers = _with_types([
        _calib_row(10, "light", JD_NIGHT1 + 0.2, 60.0),
        _calib_row(11, "light", JD_NIGHT1 + 0.3, 120.0),
    ])
    (result,) = assign_bias_dark(consumers, frames, window_days=30)
    # Night 1 darks only cover 60 s exactly, but 120 s is covered by scaling? No:
    # scaling never goes up, so the night with both exposures (day 5) wins.
    assert result.darks.night == night_of(JD_NIGHT1 + 5)
    assert result.bias.night == "20220308"
    assert result.missing_exptimes == []
    assert any("darks from night" in n for n in result.notes)


def test_assign_bias_dark_reports_missing():
    frames = _with_types([_calib_row(2, "dark", JD_NIGHT1, 60.0)])
    consumers = _with_types([_calib_row(10, "light", JD_NIGHT1 + 0.2, 300.0)])
    (result,) = assign_bias_dark(consumers, frames)
    assert result.bias is None and result.missing_exptimes == [300.0]
    assert sets_ok(calibration_sets(frames, "dark"))


def test_assign_bias_dark_uses_covering_darks_outside_the_window():
    frames = _with_types([
        _calib_row(2, "dark", JD_NIGHT1 + 3, 60.0, run="2022-03-11"),
        _calib_row(3, "dark", JD_NIGHT1 + 45, 300.0, run="2022-04-22"),  # dark finder
        _calib_row(4, "dark", JD_NIGHT1 + 45.01, 300.0, run="2022-04-22"),
    ])
    consumers = _with_types([_calib_row(10, "light", JD_NIGHT1 + 0.2, 300.0)])
    (result,) = assign_bias_dark(consumers, frames, window_days=30)
    assert result.darks.night == night_of(JD_NIGHT1 + 45) and result.missing_exptimes == []
    assert result.darks.runs == ["2022-04-22"]
    assert any("outside the 30-day window" in n and "run 2022-04-22" in n
               for n in result.notes)
    # Without matching exposures anywhere: nearest night, reported with a hint.
    (result,) = assign_bias_dark(consumers, frames[:1], window_days=30)
    assert result.missing_exptimes == [300.0]
    assert any("dark finder" in n for n in result.notes)
    assert any("run 2022-03-11" in n for n in result.notes)


def test_flat_reports_name_sets_runs_and_reasons():
    from ost_photometry.reduce.grouping.flats import (
        FlatCandidate,
        _other_candidates,
        flat_sets,
    )

    table = _with_types([
        _calib_row(1, "flat", JD_NIGHT1 + 0.1, 2.0, filter="V", run="2022-03-08"),
        _calib_row(2, "flat", JD_NIGHT1 + 0.11, 2.0, filter="V", run="2022-03-08"),
    ])
    (fs,) = flat_sets(table)
    assert fs.runs == ["2022-03-08"]
    assert fs.describe() == f"{fs.set_id} (2 frames, run 2022-03-08)"
    text = _other_candidates([FlatCandidate(fs.set_id, 0.0, REJECTED,
                                            "other session(s) in between: S2")], {fs.set_id: fs})
    assert "not used" in text and "run 2022-03-08" in text and "S2" in text


def sets_ok(sets):
    return all(s.set_id.startswith("D_") for s in sets)


# ---------------------------------------------------------------------------
# Flat scoring and image metrics
# ---------------------------------------------------------------------------


def _session(sid="S1", start=JD_NIGHT1 + 0.1, end=JD_NIGHT1 + 0.4, camera="qhy600m"):
    return Session(sid, camera, "CDK20", start, end)


def _flatset(jd, camera="qhy600m", binning="3x3", filt="V"):
    return FlatSet("FS", camera, binning, filt, "CDK20", "e", jd, jd)


def test_score_flat_set_rules():
    session = _session()
    others = [session, _session("S2", JD_NIGHT1 + 1.1, JD_NIGHT1 + 1.3)]
    kw = {"session_binnings": {"3x3"}}
    assert score_flat_set(_flatset(JD_NIGHT1 + 0.2), session, others, **kw).category == CERTAIN
    assert score_flat_set(_flatset(JD_NIGHT1 + 0.2, camera="qhy268"), session, others,
                          **kw).probability == 0.0
    assert score_flat_set(_flatset(JD_NIGHT1 + 0.2, binning="2x2"), session, others,
                          **kw).probability == 0.0
    # Flats after S2 cannot belong to S1 (S2 in between).
    later = score_flat_set(_flatset(JD_NIGHT1 + 1.5), session, others, **kw)
    assert later.probability == 0.0 and "S2" in later.reason
    # Same night at dawn: high but not certain; dust evidence raises it.
    dawn = score_flat_set(_flatset(JD_NIGHT1 + 0.6), session, others, **kw)
    assert 0.7 < dawn.probability < 0.95
    dawn_dust = score_flat_set(_flatset(JD_NIGHT1 + 0.6), session, others,
                               dust_evidence=(0.6, 0.95), **kw)
    assert dawn_dust.probability > 0.95
    changed = score_flat_set(_flatset(JD_NIGHT1 + 0.6), session, others,
                             dust_evidence=(0.02, 0.03), vignetting_rms=0.05, **kw)
    assert changed.probability < 0.4
    far = score_flat_set(_flatset(JD_NIGHT1 - 20), session, others, **kw)
    assert far.category == REJECTED


def test_dust_similarity_and_vignetting_metrics():
    rng = np.random.default_rng(3)
    shape = (300, 400)
    yy, xx = np.indices(shape)

    def flat(donuts, noise=0.003, centre=(180, 140)):
        base = 1.0 - 0.15 * (((xx - centre[0]) / 200) ** 2 + ((yy - centre[1]) / 150) ** 2)
        for x, y in donuts:
            r = np.hypot(xx - x, yy - y)
            base *= 1.0 - 0.03 * ((r > 6) & (r < 12))
        return base * (1 + rng.normal(0, noise, shape))

    donuts = [(100, 100), (250, 200), (320, 80), (60, 220)]
    a = FlatImages(flat(donuts), flat(donuts), flat(donuts))
    b = FlatImages(flat(donuts), flat(donuts), flat(donuts))
    moved = [(x + 40, y + 20) for x, y in donuts]
    c = FlatImages(flat(moved), flat(moved), flat(moved))
    r_same, n_same = dust_similarity(a, b)
    r_diff, n_diff = dust_similarity(a, c)
    assert r_same > 0.6 and n_same > 0.9
    assert abs(r_diff) < 0.2
    assert vignetting_difference(a.master, b.master) < 0.004
    shifted = flat(donuts, centre=(300, 60))
    assert vignetting_difference(a.master, shifted) > 0.008


# ---------------------------------------------------------------------------
# Full plan on a synthetic data set
# ---------------------------------------------------------------------------


def _tan_header(ra, dec, pa, shape=(300, 400), scale=0.674):
    """TAN WCS cards for a synthetic light (+y axis at position angle ``pa``)."""
    t = np.radians(pa)
    s = scale / 3600
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = s * np.array([[-np.cos(t), np.sin(t)], [np.sin(t), np.cos(t)]])
    return w.to_header()


def _write(path, data, wcs_header=None, **cards):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    for key, value in cards.items():
        header[key] = value
    if wcs_header is not None:
        header.update(wcs_header)
    fits.writeto(path, np.clip(data, 0, 65535).astype(np.uint16), header, overwrite=True)


def _date(jd):
    from astropy.time import Time

    return Time(jd, format="jd").isot


def _camera_cards(offset=5, temp=-20.0):
    return {"INSTRUME": "QHYCCD-Cameras-Capture", "XBINNING": 3, "YBINNING": 3,
            "READOUTM": "Normal", "GAIN": 0, "OFFSET": offset, "SET-TEMP": temp,
            "CCD-TEMP": temp + 0.1, "EGAIN": 1.0,
            "TELESCOP": "Planewave CDK20", "FOCALLEN": 3454.0, "XPIXSZ": 11.28}


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    night1 = root / "2022-03-08"
    cards = _camera_cards()
    for i in range(3):
        _write(night1 / "bias" / f"z{i}.fit", _bias(), IMAGETYP="Bias Frame", EXPTIME=0.0,
               **{"DATE-OBS": _date(JD_NIGHT1 + 0.70 + i * 1e-3)}, **cards)
        _write(night1 / "darks" / f"d{i}.fit", _dark(), IMAGETYP="Dark Frame", EXPTIME=60.0,
               **{"DATE-OBS": _date(JD_NIGHT1 + 0.71 + i * 1e-3)}, **cards)
        _write(night1 / "darks" / f"d5_{i}.fit", _dark(), IMAGETYP="Dark Frame", EXPTIME=5.0,
               **{"DATE-OBS": _date(JD_NIGHT1 + 0.72 + i * 1e-3)}, **cards)
    for i in range(4):
        _write(night1 / "flats" / f"f{i}.fit", _flat(), IMAGETYP="Flat Field", EXPTIME=5.0,
               FILTER="V", **{"DATE-OBS": _date(JD_NIGHT1 + 0.65 + i * 1e-3)}, **cards)
    for i in range(4):
        _write(night1 / "m104" / f"a{i}.fit", _light(), _tan_header(*M104, 151.6),
               IMAGETYP="Light Frame", EXPTIME=60.0,
               FILTER="V", OBJECT="M104", OBJCTRA="12 39 59.4", OBJCTDEC="-11 37 23",
               **{"DATE-OBS": _date(JD_NIGHT1 + 0.40 + i * 0.01)}, **cards)
        _write(night1 / "m57" / f"b{i}.fit", _light(), _tan_header(*M57, 331.6),
               IMAGETYP="Light Frame", EXPTIME=60.0,
               FILTER="V", OBJECT="m57", OBJCTRA="18 53 35.1", OBJCTDEC="+33 01 45",
               **{"DATE-OBS": _date(JD_NIGHT1 + 0.50 + i * 0.01)}, **cards)
    night2 = root / "2022-03-20"
    cards2 = _camera_cards(offset=2)
    for i in range(3):
        _write(night2 / "m104" / f"c{i}.fit", _light(), _tan_header(*M104, 159.9),
               IMAGETYP="Light Frame", EXPTIME=60.0,
               FILTER="V", OBJECT="M 104", OBJCTRA="12 39 59.4", OBJCTDEC="-11 37 23",
               **{"DATE-OBS": _date(JD_NIGHT1 + 12.40 + i * 0.01)}, **cards2)
        _write(night2 / "darks" / f"e{i}.fit", _dark(), IMAGETYP="Dark Frame", EXPTIME=60.0,
               **{"DATE-OBS": _date(JD_NIGHT1 + 12.7 + i * 1e-3)}, **cards2)
    return root


class PlanSolver:
    """PA 151.6 in night 1, 159.9 after the remount; centre from the file name."""

    def __call__(self, path, **kwargs):
        name = path.rsplit("/", 1)[-1]
        pa = 159.9 if name.startswith("c") else (331.6 if name.startswith("b") else 151.6)
        ra, dec = M57 if name.startswith("b") else M104
        t = np.radians(pa)
        s = 0.674 / 3600
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ra, dec]
        w.wcs.crpix = [200, 150]
        w.wcs.cd = s * np.array([[-np.cos(t), np.sin(t)], [np.sin(t), np.cos(t)]])
        return w


def test_build_calibration_plan_end_to_end(dataset, tmp_path):
    manifest = manifest_from_directory(dataset)
    plan = build_calibration_plan(manifest, work_dir=tmp_path / "work", solver=PlanSolver(),
                                  log=lambda *_: None)
    frames = plan.frames
    kinds = dict(zip(frames["file_name"], frames["frame_type"], strict=True))
    assert kinds["z0.fit"] == "bias" and kinds["d0.fit"] == "dark"
    assert kinds["f0.fit"] == "flat" and kinds["b0.fit"] == "light"
    sessions = {s.session_id: s for s in plan.sessions}
    assert len(sessions) == 2  # pier flip (b*) stays, remount (c*) splits
    night1 = next(s for s in plan.sessions if s.session_id.startswith("S20220308"))
    assert night1.pa_mod180 == pytest.approx(151.6, abs=0.01)
    assert len(night1.frame_ids) == 8

    targets = {str(r["target_name"]).lower().replace(" ", ""): r for r in plan.targets}
    assert set(targets) == {"m104", "m57"}
    assert targets["m104"]["n_frames"] == 7  # both nights
    assert all(plan.targets["stack"])

    units = {u.session_id: u for u in plan.units}
    u1 = units[night1.session_id]
    assert len(u1.light_ids) == 8  # both targets in one unit, same masters
    assert u1.bias_id and u1.dark_id and u1.flats["V"]
    flat_master = plan.masters[u1.flats["V"]]
    assert flat_master.category in {"certain", "likely"}
    assert flat_master.dark_id  # 5 s darks for the flats
    u2 = next(u for u in plan.units if u is not u1)
    assert u2.dark_id and not u2.bias_id  # offset 2: own darks, no bias
    assert any("flat" in n for n in u2.notes)  # night-1 flat is unlikely after the remount

    plan_path = write_plan(plan, tmp_path / "calibration_plan.yaml")
    write_frames(plan, tmp_path / "calibration_groups.ecsv")
    data = load_plan(plan_path)
    assert set(data["units"]) == {u.unit_id for u in plan.units}
    assert "Edit only the 'overrides' block" in plan_path.read_text()

    # Overrides: exclude a frame, do not stack M57, rename M104.
    overrides = Overrides(exclude_frames=[str(frames["frame_id"][list(frames["file_name"]).index("a0.fit")])],
                          no_stack_targets=["m57"], rename_targets={"M104": "Sombrero"})
    plan2 = build_calibration_plan(manifest, work_dir=tmp_path / "work", solver=PlanSolver(),
                                   overrides=overrides, log=lambda *_: None)
    names = {str(r["target_name"]): bool(r["stack"]) for r in plan2.targets}
    assert names.get("Sombrero") is True and names.get("m57") is False
    assert sum(len(u.light_ids) for u in plan2.units) == 10
    write_plan(plan2, plan_path)
    assert read_overrides(plan_path).rename_targets == {"M104": "Sombrero"}


def test_load_plan_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nmasters: {}\nunits: {U1: {bias: MB_x}}\n")
    with pytest.raises(ValueError, match="unknown master"):
        load_plan(bad)
    bad.write_text("version: 1\noverrides: {nonsense: 1}\n")
    with pytest.raises(ValueError, match="Unknown override keys"):
        load_plan(bad)
    with pytest.raises(ValueError, match="version"):
        bad.write_text("version: 99\n")
        load_plan(bad)
    assert PlanSettings().pa_tolerance == 0.5


def test_timeline_plots_are_written(dataset, tmp_path):
    from ost_photometry.reduce.grouping.plots import plot_night_timelines

    manifest = manifest_from_directory(dataset)
    plan = build_calibration_plan(manifest, work_dir=tmp_path / "work", solver=PlanSolver(),
                                  log=lambda *_: None)
    paths = plot_night_timelines(plan, tmp_path / "out")
    assert len(paths) == 2
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)
    assert paths[0].parent == tmp_path / "out" / "diagnostics" / "calibration_groups"
