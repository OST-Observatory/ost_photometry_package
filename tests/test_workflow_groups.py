"""Reduction of a calibration plan (masters per group, lights per unit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from ost_photometry.archive.local import manifest_from_directory
from ost_photometry.reduce.grouping.plan import (
    build_calibration_plan,
    load_plan,
    write_frames,
    write_plan,
)
from test_grouping_plan import PlanSolver, dataset  # noqa: F401  (pytest fixture)


def test_reduce_planned_end_to_end(dataset, tmp_path):  # noqa: F811
    pytest.importorskip("ccdproc")
    from astropy.table import Table

    from ost_photometry.reduce.workflow.groups import ReductionSettings, reduce_planned

    manifest = manifest_from_directory(dataset)
    plan = build_calibration_plan(manifest, work_dir=tmp_path / "work", solver=PlanSolver(),
                                  log=lambda *_: None)
    plan_path = write_plan(plan, tmp_path / "calibration_plan.yaml")
    frames_path = write_frames(plan, tmp_path / "calibration_groups.ecsv")
    data = load_plan(plan_path)
    frames = Table.read(frames_path, format="ascii.ecsv")

    settings = ReductionSettings(rm_cosmic_rays=False, gain=1.0, read_noise=5.0, dark_rate=0.1,
                                 saturation_level=65535.0, n_cores_multiprocessing=2)
    out = tmp_path / "out"
    reports, table = reduce_planned(data, frames, out, settings, log=lambda *_: None)

    assert sum(len(r.reduced) for r in reports.values()) == 11
    assert all(s == "reduced" for s in table["status"])
    assert (out / "reduction_report.ecsv").is_file()
    # Masters and reduced frames are written as float32 (storage_dtype).
    assert fits.getheader(table["reduced_path"][0])["BITPIX"] == -32
    assert all(fits.getheader(p)["BITPIX"] == -32
               for p in (out / "masters").rglob("combined_*.fit"))
    masters = {p.name for p in (out / "masters").iterdir()}
    assert any(m.startswith("MF_") for m in masters)
    assert any(m.startswith("MB_") for m in masters) and any(m.startswith("MD_") for m in masters)

    by_unit = {}
    for path_str, unit in zip(table["reduced_path"], table["unit_id"], strict=True):
        header = fits.getheader(path_str)
        assert header["CALUNIT"] == unit
        assert header["IMAGETYP"] == "Light Frame"
        assert header["SESSID"] and header["TARGET"]
        assert header["FLATGRP"].startswith("MF_")
        from astropy.nddata import CCDData

        ccd = CCDData.read(path_str)
        sky = float(np.nanmedian(ccd.data))
        # Synthetic sky: 200 ADU in 60 s, gain 1 -> about 3.3 e-/s (bias and
        # dark removed exactly once; a double bias subtraction goes negative).
        assert 2.5 < sky < 4.5
        # Uncertainty from the bias-free signal: sqrt(~205 + 5^2) ADU / 60 s
        # ~ 0.25 e-/s; counting the 730 ADU bias as photons gave ~ 0.52.
        assert 0.2 < float(np.nanmedian(ccd.uncertainty.array)) < 0.35
        assert ccd.mask is None or ccd.mask.mean() < 0.1
        # For L.A.Cosmic in the analysis: read noise and the saturation of the
        # calibrated electrons ((65535 - 730 bias) x gain 1 / brightest flat).
        assert header["RDNOISE"] == pytest.approx(5.0)
        assert 55000 < header["SATLEVEL"] < 64805
        by_unit.setdefault(unit, set()).add(header["SESSID"])
    assert len(by_unit) == 2 and all(len(v) == 1 for v in by_unit.values())

    # Second run reuses the masters (no rebuild needed).
    mtime = {p: p.stat().st_mtime for p in (out / "masters").rglob("combined_*.fit")}
    reduce_planned(data, frames, out, settings, log=lambda *_: None)
    assert all(p.stat().st_mtime == t for p, t in mtime.items())


def test_stage_frames_links_or_corrects(tmp_path):
    from ost_photometry.reduce.workflow.groups import stage_frames

    rows = []
    for i, imagetyp in enumerate(("Flat Field", "Flat Field", "Light Frame")):
        path = tmp_path / f"raw{i}.fit"
        header = fits.Header()
        header["IMAGETYP"] = imagetyp
        fits.writeto(path, np.zeros((4, 4), dtype=np.uint16), header)
        rows.append({"frame_id": str(i), "file_name": path.name, "local_path": str(path),
                     "imagetyp": imagetyp})
    staged = stage_frames(rows, tmp_path / "staged", "flat")
    assert len(staged) == 3
    assert staged[0].is_symlink() and staged[1].is_symlink()
    assert not staged[2].is_symlink()
    assert fits.getheader(staged[2])["IMAGETYP"] == "Flat Field"
    assert fits.getheader(tmp_path / "raw2.fit")["IMAGETYP"] == "Light Frame"  # raw untouched


def test_stack_planned_per_target(dataset, tmp_path):  # noqa: F811
    pytest.importorskip("ccdproc")
    pytest.importorskip("reproject")
    from astropy.table import Table

    from ost_photometry.reduce.workflow.combine import StackSettings, stack_planned
    from ost_photometry.reduce.workflow.groups import ReductionSettings, reduce_planned

    manifest = manifest_from_directory(dataset)
    plan = build_calibration_plan(manifest, work_dir=tmp_path / "work", solver=PlanSolver(),
                                  log=lambda *_: None)
    data = load_plan(write_plan(plan, tmp_path / "plan.yaml"))
    frames = Table.read(write_frames(plan, tmp_path / "groups.ecsv"), format="ascii.ecsv")
    out = tmp_path / "out"
    _, report = reduce_planned(
        data, frames, out,
        ReductionSettings(rm_cosmic_rays=False, gain=1.0, read_noise=5.0, dark_rate=0.1,
                          saturation_level=65535.0, n_cores_multiprocessing=2),
        log=lambda *_: None,
    )
    settings = StackSettings(stack_weighting="fwhm", shift_method="wcs",
                             camera_combination="combine", n_cores_multiprocessing=2,
                             keep_reduced_lights=True)
    summary = stack_planned(data, frames, report, out, settings, log=lambda *_: None)
    assert all(Path(p).is_file() for p in report["reduced_path"])
    per_target = {}
    for row in summary:
        per_target.setdefault(str(row["target_name"]).lower().replace(" ", ""), []).append(row)
    assert set(per_target) == {"m104", "m57"}
    separate = [r for r in per_target["m104"] if r["camera"] != "combined"]
    assert len(separate) == 1 and separate[0]["n_images"] == 7  # both nights in one stack
    assert any(r["camera"] == "combined" for r in per_target["m104"])
    m104 = fits.getheader(separate[0]["path"])
    assert m104["BITPIX"] == -32  # same float type as the aligned frames
    assert m104["N-IMAGES"] == 7 and m104["EXPTIME"] == pytest.approx(420.0)
    assert m104["WEIGHTNG"] == "fwhm"
    # Read noise of the stack at its total exposure: 5 e- x sqrt(7) for
    # equal weights, a bit more for unequal FWHM weights.
    assert 5.0 * np.sqrt(7) * 0.999 < m104["RDNOISE"] < 5.0 * 7
    assert m104["SATLEVEL"] > 7 * 55000
    assert m104["CRIDENT"] and m104["CRCLIP"]  # the analysis skips L.A.Cosmic
    combined = [r for r in per_target["m104"] if r["camera"] == "combined"][0]
    assert fits.getheader(combined["path"])["RDNOISE"] > 0
    m57 = [r for r in per_target["m57"] if r["camera"] != "combined"][0]
    assert fits.getheader(m57["path"])["N-IMAGES"] == 4
    assert (out / "stacks" / "summary.ecsv").is_file()

    # Restrict to one target; by default the reduced frames of the aligned
    # (kept) frames are removed, those of the other target stay.
    from dataclasses import replace

    only = stack_planned(data, frames, report, tmp_path / "out2",
                         replace(settings, keep_reduced_lights=False), targets=["m57"],
                         log=lambda *_: None)
    assert {str(n).lower() for n in only["target_name"]} == {"m57"}
    target_of = {str(r["frame_id"]): str(r["target_name"]).lower() for r in frames}
    for fid, path in zip(report["frame_id"], report["reduced_path"], strict=True):
        assert Path(path).is_file() == (target_of[str(fid)] != "m57")


def test_combine_camera_stacks_noise_weighting(tmp_path):
    pytest.importorskip("ccdproc")
    from astropy.nddata import CCDData, StdDevUncertainty

    from ost_photometry.reduce.workflow.combine import combine_camera_stacks

    paths = []
    for name, value, sigma in (("a", 10.0, 1.0), ("b", 20.0, 3.0)):
        ccd = CCDData(np.full((10, 10), value), unit="electron/s",
                      uncertainty=StdDevUncertainty(np.full((10, 10), sigma)))
        ccd.meta["EXPTIME"] = 100.0
        ccd.meta["N-IMAGES"] = 5
        path = tmp_path / f"{name}.fit"
        ccd.write(path)
        paths.append(path)
    out = combine_camera_stacks(paths, tmp_path / "c.fit", cameras=["qhy600m", "qhy268"])
    data, header = fits.getdata(out), fits.getheader(out)
    assert np.allclose(data, (10 * 1 + 20 / 9) / (1 + 1 / 9))  # inverse-variance mean
    assert header["NCAMERAS"] == 2 and header["N-IMAGES"] == 10
    assert header["EXPTIME"] == pytest.approx(200.0)
