"""Tests for weighted stacking (reduce.workflow.stack)."""

from __future__ import annotations

import numpy as np
import pytest


def _stack():
    pytest.importorskip("ccdproc")
    from ost_photometry.reduce.workflow import stack

    return stack


def _write_frame(path, value, *, weight=None, filt="V", jd=2460000.0):
    from astropy.nddata import CCDData

    from ost_photometry.fits_headers import set_frame_weight

    ccd = CCDData(np.full((8, 8), float(value), dtype=np.float32), unit="electron/s")
    ccd.meta["IMAGETYP"] = "LIGHT"
    ccd.meta["FILTER"] = filt
    ccd.meta["EXPTIME"] = 10.0
    ccd.meta["JD"] = jd
    ccd.meta["DATE-OBS"] = "2024-01-01T00:00:00"
    ccd.meta["EGAIN"] = 1.0
    ccd.meta["OBJECT"] = "test"
    if weight is not None:
        set_frame_weight(ccd.meta, weight)
    ccd.write(path, overwrite=True)


def test_prepare_stack_weights_rules():
    stack = _stack()
    assert stack.prepare_stack_weights(None, "average", 3) is None
    assert stack.prepare_stack_weights([1.0, 1.0, 1.0], "average", 3) is None
    w = stack.prepare_stack_weights([2.0, 1.0, 1.0], "average", 3)
    assert np.allclose(w, [2.0, 1.0, 1.0])
    assert stack.prepare_stack_weights([2.0, 1.0, 1.0], "median", 3) is None
    w_sum = stack.prepare_stack_weights([3.0, 1.0], "sum", 2)
    assert w_sum.sum() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="stack weights"):
        stack.prepare_stack_weights([1.0, 2.0], "average", 3)
    with pytest.raises(ValueError, match="finite"):
        stack.prepare_stack_weights([1.0, np.nan], "average", 2)


def test_weighted_average_and_header(tmp_path):
    stack = _stack()
    from astropy.io import fits

    files = []
    for i, (value, weight) in enumerate(((1.0, 3.0), (2.0, 0.0), (3.0, 0.0))):
        path = tmp_path / f"f{i}.fit"
        _write_frame(path, value, weight=weight)
        files.append(str(path))

    weights = stack.weights_for_files(files)
    assert np.allclose(weights, [3.0, 0.0, 0.0])

    name = stack.stack_filter_images(
        files,
        "average",
        None,
        "V",
        tmp_path,
        None,
        weights=weights,
        stack_meta={"weighting": "fwhm", "n_frames_total": 4, "n_rejected": 1,
                    "fwhm_median": 3.2},
    )
    data, header = fits.getdata(tmp_path / name, header=True)
    assert np.allclose(data, 1.0)
    assert header["N-IMAGES"] == 3
    assert header["WEIGHTNG"] == "fwhm"
    assert header["NFRAMES0"] == 4
    assert header["NREJECT"] == 1
    assert header["FWHMMED"] == pytest.approx(3.2)

    #   Median ignores weights (with a warning) -> plain median.
    name = stack.stack_filter_images(files, "median", None, "V", tmp_path, None, weights=weights)
    data = fits.getdata(tmp_path / name)
    assert np.allclose(data, 2.0)

    #   Unweighted average for comparison.
    name = stack.stack_filter_images(files, "average", None, "V", tmp_path, None)
    data, header = fits.getdata(tmp_path / name, header=True)
    assert np.allclose(data, 2.0)
    assert header["WEIGHTNG"] == "none"


def test_weights_for_files_fallbacks(tmp_path):
    stack = _stack()
    from ost_photometry.reduce.frame_selection import quality_table_from_rows

    a = tmp_path / "a.fit"
    b = tmp_path / "b.fit"
    c = tmp_path / "c.fit"
    _write_frame(a, 1.0, weight=2.0)
    _write_frame(b, 1.0)
    _write_frame(c, 1.0)
    table = quality_table_from_rows(
        [{"file": "b.fit", "filter": "V", "stack_weight": 0.5, "fwhm_px": 3.0}]
    )
    weights = stack.weights_for_files([str(a), str(b), str(c)], quality_table=table)
    assert np.allclose(weights, [2.0, 0.5, 1.0])


def test_stack_image_reads_frmwght_and_keeps_inputs(tmp_path):
    stack = _stack()
    from astropy.io import fits

    aligned = tmp_path / "aligned_lights"
    aligned.mkdir()
    for i, (value, weight) in enumerate(((1.0, 3.0), (5.0, 1.0))):
        _write_frame(aligned / f"v{i}.fit", value, weight=weight, jd=2460000.0 + i)
    _write_frame(aligned / "b0.fit", 7.0, weight=1.0, filt="B")

    stack.stack_image(
        aligned,
        tmp_path,
        ["LIGHT"],
        stacking_method="average",
        n_cores_multiprocessing=1,
        stack_weighting="fwhm",
        keep_input_frames=True,
    )
    data, header = fits.getdata(tmp_path / "combined_filter_V.fit", header=True)
    assert np.allclose(data, 2.0)  # (3*1 + 1*5) / 4
    assert header["N-IMAGES"] == 2
    assert header["WEIGHTNG"] == "fwhm"
    assert aligned.is_dir()
    assert (tmp_path / "combined_filter_B.fit").exists()

    with pytest.raises(ValueError, match="stack_weighting"):
        stack.stack_image(aligned, tmp_path, ["LIGHT"], stack_weighting="seeing")


def test_total_exposure_time_sums_mixed_exposures(tmp_path):
    stack = _stack()
    from astropy.io import fits

    files = []
    for i, exptime in enumerate((30.0, 60.0, 90.0)):
        path = tmp_path / f"e{i}.fit"
        _write_frame(path, 1.0, jd=2460000.0 + i)
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["EXPTIME"] = exptime
            hdul[0].header["INSTRUME"] = "QHY600M"
            hdul.flush()
        files.append(str(path))
    assert stack.total_exposure_time(files) == pytest.approx(180.0)
    name = stack.stack_filter_images(files, "average", None, "V", tmp_path, None)
    header = fits.getheader(tmp_path / name)
    assert header["EXPTIME"] == pytest.approx(180.0)
    assert header["INSTRU"] == "QHY600M"


def test_group_columns_for_weights_and_references():
    pytest.importorskip("ccdproc")
    from ost_photometry.reduce.frame_selection import (
        quality_table_from_rows,
        resolve_reference_frames,
        stack_weights,
    )

    rows = []
    for target in ("M57", "M104"):
        for i, fwhm in enumerate((2.0, 4.0)):
            rows.append({"file": f"{target}_{i}.fit", "filter": "V", "target": target,
                         "fwhm_px": fwhm, "n_stars": 40})
    table = quality_table_from_rows(rows)
    table["target"] = np.array([str(f).split("_")[0] for f in table["file"]])
    refs = resolve_reference_frames(table, group_columns=("target", "filter"))
    assert refs == {"M57|V": "M57_0.fit", "M104|V": "M104_0.fit"}
    w = stack_weights(table, "fwhm", group_columns=("target", "filter"))
    for target in ("M57", "M104"):
        sel = np.asarray(table["target"]) == target
        assert w[sel].mean() == pytest.approx(1.0)


def test_fwhm_weights_use_arcsec_for_mixed_pixel_scales():
    from ost_photometry.reduce.frame_selection import quality_table_from_rows, stack_weights

    # Same seeing (2 arcsec) seen by two cameras with different pixel scales.
    rows = [
        {"file": "a.fit", "filter": "V", "fwhm_px": 4.0, "pixel_scale": 0.5, "fwhm_arcsec": 2.0},
        {"file": "b.fit", "filter": "V", "fwhm_px": 2.0, "pixel_scale": 1.0, "fwhm_arcsec": 2.0},
    ]
    w = stack_weights(quality_table_from_rows(rows), "fwhm")
    assert w[0] == pytest.approx(w[1])


def test_stack_mid_time_and_airmass_are_weighted_means(tmp_path):
    from astropy.io import fits
    from astropy.nddata import CCDData
    from astropy.time import Time

    stack = _stack()
    files = []
    for i, (jd, am) in enumerate(((2460000.30, 1.2), (2460000.40, 1.5), (2460000.50, 2.0))):
        path = tmp_path / f"f{i}.fit"
        _write_frame(path, 1.0, jd=jd)
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["AIRMASS"] = am
        files.append(str(path))
    combined = CCDData(np.ones((8, 8)), unit="electron/s", meta=fits.Header({"JD": 2460000.30}))
    w = np.array([1.0, 2.0, 1.0])
    stack._write_stack_epoch(combined, files, w)
    mid = (0.30 + 2 * 0.40 + 0.50) / 4 + 2460000.0 + 5.0 / 86400.0
    assert combined.meta["MJD_MID"] == pytest.approx(mid - 2400000.5, abs=1e-9)
    assert Time(combined.meta["DATE-MID"], format="fits").jd == pytest.approx(mid, abs=1e-6)
    # effective airmass of the averaged flux (V: k = 0.2), slightly below the mean
    k = combined.meta["AIRMSEXK"]
    x = np.array([1.2, 1.5, 2.0])
    x_eff = -2.5 / k * np.log10(np.sum(w * 10 ** (-0.4 * k * x)) / w.sum())
    assert combined.meta["AIRMASS"] == pytest.approx(x_eff)
    assert x_eff < (1.2 + 2 * 1.5 + 2.0) / 4
    assert combined.meta["AIRMSMAX"] == 2.0


def test_stack_marks_frames_without_flat(tmp_path):
    from astropy.io import fits
    from astropy.nddata import CCDData

    stack = _stack()
    files = []
    for i, state in enumerate(("applied", "none")):
        path = tmp_path / f"f{i}.fit"
        _write_frame(path, 1.0)
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["FLATCORR"] = state
        files.append(str(path))
    combined = CCDData(np.ones((8, 8)), unit="electron/s", meta=fits.Header({"EXPTIME": 20.0}))
    stack._write_stack_noise(combined, files, None)
    assert combined.meta["FLATCORR"] == "partial"
    combined = CCDData(np.ones((8, 8)), unit="electron/s", meta=fits.Header({"EXPTIME": 20.0}))
    stack._write_stack_noise(combined, files[1:], None)
    assert combined.meta["FLATCORR"] == "none"
