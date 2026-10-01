"""Targets, orientation sampling / solving and mount sessions."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from ost_photometry.archive.manifest import manifest_from_rows
from ost_photometry.reduce.grouping import orientation as orient_mod
from ost_photometry.reduce.grouping.orientation import (
    OrientationResult,
    orientation_from_server,
    refine_orientation_changes,
    sample_frames_for_solving,
    solve_orientations,
)
from ost_photometry.reduce.grouping.sessions import UNKNOWN_SESSION, segment_sessions
from ost_photometry.reduce.grouping.targets import (
    assign_targets,
    cluster_positions,
    normalize_name,
    target_summary,
)

QHY = {"instrume": "QHY600M", "telescop": "CDK20", "naxis1": 3192, "naxis2": 2129,
       "xbinning": 3, "ybinning": 3, "xpixsz": 11.28, "focallen": 3454.0}
M57 = (283.396, 33.029)
M104 = (189.998, -11.623)


def _light(fid, jd, *, pos=M57, obj="m57", camera=QHY, **extra):
    row = dict(camera)
    row.update({"frame_id": str(fid), "jd": jd, "ra": pos[0], "dec": pos[1], "object": obj,
                "exptime": 60.0, "filter": "V", "file_name": f"{fid}.fit"})
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_normalize_name():
    assert normalize_name("M 57") == normalize_name("m57") == normalize_name("M_57")
    assert normalize_name("-") == ""


def test_cluster_positions_dither_and_separate_fields():
    labels = cluster_positions([10.0, 10.01, 10.02, 50.0], [20.0, 20.0, 20.01, 20.0],
                               [0.3, 0.3, 0.3, 0.3])
    assert labels[0] == labels[1] == labels[2] != labels[3]


def test_assign_targets_two_targets_names_and_unknown_positions():
    rows = [
        _light(1, 1.00, obj="M104", pos=M104),
        _light(2, 1.01, obj="m104", pos=(M104[0] + 0.01, M104[1])),  # dither
        _light(3, 1.10, obj="M 57"),
        _light(4, 1.11, obj="m57", pos=(np.nan, np.nan)),  # no pointing: by name
        _light(5, 1.12, obj="", pos=(np.nan, np.nan)),  # no name: by time neighbours
        _light(6, 1.50, obj="NGC 7000", pos=(np.nan, np.nan)),  # name-only target
    ]
    table = assign_targets(manifest_from_rows(rows))
    ids = dict(zip(table["frame_id"], table["target_id"], strict=True))
    names = dict(zip(table["frame_id"], table["target_name"], strict=True))
    assert ids["1"] == ids["2"] != ids["3"]
    assert ids["3"] == ids["4"] == ids["5"]
    assert ids["6"] not in {ids["1"], ids["3"]}
    assert names["1"] == "M104" and names["3"] in {"M 57", "m57"}
    assert names["6"] == "NGC 7000"
    notes = dict(zip(table["frame_id"], table["target_note"], strict=True))
    assert "object name" in notes["4"] and "time neighbours" in notes["5"]
    summary = target_summary(table)
    assert len(summary) == 3


def test_same_name_different_field_and_merge_rename():
    rows = [_light(1, 1.0, obj="Mosaic", pos=(10.0, 20.0)),
            _light(2, 1.1, obj="Mosaic", pos=(12.0, 20.0))]
    table = assign_targets(manifest_from_rows(rows))
    assert set(table["target_name"]) == {"Mosaic", "Mosaic_2"}
    merged = assign_targets(manifest_from_rows(rows), merge_targets=[["Mosaic", "Mosaic_2"]],
                            rename={"Mosaic": "Mosaic M31"})
    assert len(set(merged["target_id"])) == 1
    assert set(merged["target_name"]) == {"Mosaic M31"}


def test_solved_centres_override_header_and_archive_grouping():
    rows = [_light(1, 1.0, pos=(10.0, 20.0)), _light(2, 1.1, pos=(10.0, 20.0))]
    centers = {"2": (40.0, 20.0)}
    table = assign_targets(manifest_from_rows(rows), centers=centers)
    assert table["target_id"][0] != table["target_id"][1]
    rows = [_light(1, 1.0, pos=(10.0, 20.0), main_object_id=5, main_object_name="Comet X"),
            _light(2, 2.0, pos=(30.0, 25.0), main_object_id=5, main_object_name="Comet X")]
    moving = assign_targets(manifest_from_rows(rows), group_by="archive_object")
    assert moving["target_id"][0] == moving["target_id"][1]
    assert moving["target_name"][0] == "Comet X"


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def test_sampling_block_ends_interval_and_target_changes():
    jd0 = 2459647.5
    rows = [_light(i, jd0 + i * 5 / 1440) for i in range(20)]  # one frame per 5 min
    rows += [_light(100 + i, jd0 + 0.5 + i * 5 / 1440) for i in range(3)]  # after a gap
    table = manifest_from_rows(rows)
    labels = ["A"] * 10 + ["B"] * 10 + ["A"] * 3
    picked = sample_frames_for_solving(table, interval_minutes=30, gap_hours=1,
                                       target_labels=labels)
    ids = [str(table["frame_id"][i]) for i in picked]
    assert {"0", "19", "100", "102"} <= set(ids)  # block ends
    assert {"9", "10"} <= set(ids)  # target change
    assert "6" in ids  # 30-minute interval


def test_orientation_from_server_cd_matrix():
    s = 0.677 / 3600
    row = {"frame_id": "1", "plate_solved": True, "wcs_cd1_1": -s, "wcs_cd1_2": 0.0,
           "wcs_cd2_1": 0.0, "wcs_cd2_2": s, "wcs_ra": 10.0, "wcs_dec": 20.0}
    result = orientation_from_server(row)
    assert result.solved and result.pa_deg == pytest.approx(0.0)
    assert result.scale_arcsec == pytest.approx(0.677)
    assert orientation_from_server({**row, "plate_solved": False}) is None


class FakeSolver:
    """Returns a TAN WCS with the PA stored for each file name."""

    def __init__(self, pa_by_name, fail=()):
        self.pa_by_name = pa_by_name
        self.fail = set(fail)
        self.calls = []

    def __call__(self, path, **kwargs):
        from astropy.wcs import WCS

        name = path.rsplit("/", 1)[-1]
        self.calls.append(name)
        if name in self.fail:
            return None
        t = np.radians(self.pa_by_name[name])
        s = 0.677 / 3600
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [283.4, 33.0]
        w.wcs.crpix = [100, 100]
        w.wcs.cd = s * np.array([[-np.cos(t), np.sin(t)], [np.sin(t), np.cos(t)]])
        return w


def _files(tmp_path, rows):
    for row in rows:
        path = tmp_path / row["file_name"]
        fits.writeto(path, np.zeros((4, 4), dtype=np.uint16), overwrite=True)
        row["local_path"] = str(path)
        row["sha256"] = f"{int(row['frame_id']):064d}"
    return rows


def test_solve_with_cache_and_bisection(tmp_path):
    jd0 = 2459647.5
    rows = _files(tmp_path, [_light(i, jd0 + i / 1440) for i in range(17)])
    table = manifest_from_rows(rows)
    # Camera remounted between frame 11 and 12 (151.6 -> 159.9 deg).
    pas = {f"{i}.fit": (151.6 if i <= 11 else 159.9) for i in range(17)}
    solver = FakeSolver(pas)
    cache = tmp_path / "orientation_cache.ecsv"
    results = solve_orientations(table, [0, 16], work_dir=tmp_path / "w", cache_path=cache,
                                 solver=solver)
    assert results["0"].pa_deg == pytest.approx(151.6)
    results = refine_orientation_changes(table, results, work_dir=tmp_path / "w",
                                         cache_path=cache, solver=solver)
    solved = {k for k, r in results.items() if r.solved}
    assert {"11", "12"} <= solved
    assert len(solver.calls) <= 7  # bisection, not every frame
    # Second run: everything from the cache, no solver calls.
    solver2 = FakeSolver(pas)
    again = solve_orientations(table, [0, 16], work_dir=tmp_path / "w", cache_path=cache,
                               solver=solver2)
    assert solver2.calls == [] and again["16"].pa_deg == pytest.approx(159.9)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def _orient(fid, pa, parity="normal", scale=0.677):
    return OrientationResult(str(fid), pa, parity, scale, 0.0, 0.0, "astap")


def test_pier_flip_same_session_remount_new_session():
    jd0 = 2459647.5
    rows = [_light(i, jd0 + i * 0.01) for i in range(6)]
    orient = {"0": _orient(0, 331.6), "2": _orient(2, 151.6),  # pier flip: same mod 180
              "4": _orient(4, 159.9)}  # remount
    table, sessions = segment_sessions(manifest_from_rows(rows), orient)
    sid = dict(zip(table["frame_id"], table["session_id"], strict=True))
    assert sid["0"] == sid["1"] == sid["2"]
    assert sid["3"] == UNKNOWN_SESSION  # between two different sessions, unsolved
    assert sid["4"] == sid["5"] != sid["0"]
    assert any("orientation changed by 8.30" in e for s in sessions for e in s.evidence)
    first = next(s for s in sessions if s.session_id == sid["0"])
    assert first.pa_mod180 == pytest.approx(151.6, abs=1e-6)
    assert first.pa_spread == pytest.approx(0.0, abs=1e-6)


def test_target_change_keeps_session_and_camera_change_breaks():
    jd0 = 2459647.5
    st8 = {**QHY, "instrume": "SBIG ST-8", "naxis1": 1530, "naxis2": 1020}
    rows = [
        _light(0, jd0, obj="M104", pos=M104),
        _light(1, jd0 + 0.01, obj="M104", pos=M104),
        _light(2, jd0 + 0.02, obj="m57"),
        _light(3, jd0 + 0.03, camera=st8),
        _light(4, jd0 + 0.04),
    ]
    orient = {"0": _orient(0, 151.6), "2": _orient(2, 331.62), "4": _orient(4, 151.6)}
    table, sessions = segment_sessions(manifest_from_rows(rows), orient)
    sid = dict(zip(table["frame_id"], table["session_id"], strict=True))
    assert sid["0"] == sid["1"] == sid["2"]
    assert sid["3"] != sid["2"] and "sbig-st-8" in sid["3"]
    # Same orientation after the ST-8 was mounted: still a new session (hard break).
    assert sid["4"] not in {sid["0"], sid["3"]}


def test_context_frames_of_other_camera_break_sessions():
    jd0 = 2459647.5
    rows = [_light(0, jd0), _light(1, jd0 + 2.0)]
    context = manifest_from_rows([{"frame_id": "c", "jd": jd0 + 1.0, "telescop": "CDK20",
                                   "instrument_archive": "QHY 268M"}])
    orient = {"0": _orient(0, 151.6), "1": _orient(1, 151.6)}
    table, _ = segment_sessions(manifest_from_rows(rows), orient, context=context)
    assert table["session_id"][0] != table["session_id"][1]
    table, _ = segment_sessions(manifest_from_rows(rows), orient)
    assert table["session_id"][0] == table["session_id"][1]


def test_unsolved_session_is_unverified():
    rows = [_light(0, 2459647.5), _light(1, 2459647.6)]
    table, sessions = segment_sessions(manifest_from_rows(rows), {})
    assert len(sessions) == 1 and not sessions[0].verified
    assert "unverified" in sessions[0].evidence[-1]
    assert sessions[0].session_id.startswith("S20220308_01_qhy600m")


def test_orient_module_exports():
    assert hasattr(orient_mod, "solve_frame")


def test_header_pointing_off_joins_solved_field_by_name():
    rows = [
        _light(1, 1.00, obj="M67", pos=(132.85, 11.73)),
        _light(2, 1.01, obj="M67", pos=(137.45, 11.73)),  # mount pointing 4.5 deg off
        _light(3, 1.02, obj="M67", pos=(133.90, 11.73)),
        _light(4, 1.30, obj="NGC 2420", pos=(114.6, 21.6)),
    ]
    centers = {"1": (132.83, 11.71)}
    table = assign_targets(manifest_from_rows(rows), centers=centers)
    ids = dict(zip(table["frame_id"], table["target_id"], strict=True))
    assert ids["1"] == ids["2"] == ids["3"] != ids["4"]
    notes = dict(zip(table["frame_id"], table["target_note"], strict=True))
    assert "matched to the solved field" in notes["2"]


def test_implausible_scale_is_rejected():
    from ost_photometry.reduce.grouping.orientation import check_plausible

    row = {"xpixsz": 9.0, "focallen": 3454.0}  # predicts 0.537"/px
    good = check_plausible(_orient(1, 135.0, scale=0.54), row)
    bad = check_plausible(_orient(1, 135.0, scale=2.06), row)
    assert good.solved and not bad.solved and "implausible" in bad.note
    assert check_plausible(_orient(1, 135.0, scale=2.06), {}).solved  # no prediction


def test_cached_failure_is_retried_with_larger_radius(tmp_path):
    rows = _files(tmp_path, [_light(0, 2459647.5)])
    table = manifest_from_rows(rows)
    cache = tmp_path / "cache.ecsv"
    failing = FakeSolver({}, fail={"0.fit"})
    solve_orientations(table, [0], work_dir=tmp_path / "w", cache_path=cache, solver=failing,
                       radius_deg=3.0)
    working = FakeSolver({"0.fit": 151.6})
    again = solve_orientations(table, [0], work_dir=tmp_path / "w", cache_path=cache,
                               solver=working, radius_deg=15.0)
    assert working.calls == ["0.fit"] and again["0"].solved
