"""Stacks keep relative star fluxes with different transmission and seeing."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty

SHAPE = (240, 240)
SKY = 20.0          # e/s per pixel
EXPTIME = 60.0
READ_NOISE = 5.0    # e


def _stars(rng):
    xs, ys = np.meshgrid(np.arange(30, 211, 30), np.arange(30, 211, 30))
    xs = xs.ravel() + rng.uniform(-3, 3, xs.size)
    ys = ys.ravel() + rng.uniform(-3, 3, ys.size)
    flux = 10 ** rng.uniform(1.5, 4.0, xs.size)  # e/s, 2.5 decades
    flux[:4] = 2e4  # bright
    return xs, ys, flux


def _frame(rng, xs, ys, flux, fwhm, transmission):
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    s = fwhm / 2.3548
    model = np.full(SHAPE, SKY)
    for x, y, f in zip(xs, ys, flux, strict=True):
        model += transmission * f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)
    counts = model * EXPTIME
    var = counts + READ_NOISE**2
    data = (rng.poisson(counts) + rng.normal(0, READ_NOISE, SHAPE)) / EXPTIME
    return data, np.sqrt(var) / EXPTIME


def _aperture(img, xs, ys, r):
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    out = []
    for x, y in zip(xs, ys, strict=True):
        d = np.hypot(xx - x, yy - y)
        ann = (d > r + 3) & (d < r + 9)
        out.append(np.sum(img[d <= r]) - np.median(img[ann]) * np.count_nonzero(d <= r))
    return np.asarray(out)


@pytest.mark.parametrize("seed", [1, 2])
def test_stack_keeps_relative_fluxes(tmp_path, seed):
    pytest.importorskip("ccdproc")
    from ost_photometry.reduce.workflow.stack import stack_filter_images

    rng = np.random.default_rng(seed)
    xs, ys, flux = _stars(rng)
    setups = [(3.0, 1.0), (3.0, 1.0), (3.2, 0.8), (3.0, 1.0), (3.9, 1.0)]  # (FWHM px, transmission)
    paths = []
    for i, (fwhm, trans) in enumerate(setups):
        data, sigma = _frame(rng, xs, ys, flux, fwhm, trans)
        ccd = CCDData(data.astype(np.float32), unit="electron/s",
                      uncertainty=StdDevUncertainty(sigma.astype(np.float32)))
        ccd.meta.update({"EXPTIME": EXPTIME, "FILTER": "V", "JD": 2459647.5 + i / 1440,
                         "DATE-OBS": "2022-03-08T12:00:00", "FWHM": fwhm})
        path = tmp_path / f"f{i}.fit"
        ccd.write(path)
        paths.append(str(path))
    name = stack_filter_images(paths, "average", None, "V", tmp_path, None)
    stacked = fits.getdata(tmp_path / name).astype(float)
    header = fits.getheader(tmp_path / name)
    assert header["STKCLIP"] == "ccdclip"
    assert header["TRANSMIN"] == pytest.approx(0.8, abs=0.01)
    # clipping must not change any star: the stack equals the plain mean of the
    # scaled frames (no clipping), for bright and faint stars alike
    from ost_photometry.reduce.weighted_combine import combine_images

    plain = combine_images(paths, scale=_scales(header, len(paths)), sigma_clip=False,
                           match_background=True).data
    r_ap = 4.0 * 3.9
    usable = flux > 300
    ratio = _aperture(stacked, xs, ys, r_ap)[usable] / _aperture(plain, xs, ys, r_ap)[usable]
    assert np.max(np.abs(ratio - 1.0)) < 2e-3
    bright = flux[usable] > 1e4
    assert np.max(np.abs(ratio[bright] - 1.0)) < 2e-4
    # the old clipping (MAD of the frames) cuts the bright stars' cores and
    # wings of the frames with the best / worst seeing: their flux changes
    mad = combine_images(paths, scale=_scales(header, len(paths)), sigma_clip=True,
                         match_background=True).data
    loss = _aperture(mad, xs, ys, r_ap)[usable][bright] / _aperture(plain, xs, ys, r_ap)[usable][bright]
    assert abs(np.median(loss) - 1.0) > 0.002


def _scales(header, n):
    text = " ".join(str(h) for h in header["HISTORY"] if "flux factors" in str(h))
    values = [float(v) for v in text.split(":", 1)[1].replace("...", "").split(",")]
    assert len(values) == n
    return values


def test_two_frames_are_never_clipped():
    from ost_photometry.reduce.weighted_combine import clip_against_uncertainty

    data = np.ma.masked_array(np.array([[0.0], [100.0]]))
    sigma = np.ones_like(data.data)
    assert not np.any(clip_against_uncertainty(data, sigma, 5.0, 5.0))
    data3 = np.ma.masked_array(np.array([[0.0], [0.5], [100.0]]))
    reject = clip_against_uncertainty(data3, np.ones((3, 1)), 5.0, 5.0)
    assert reject[:, 0].tolist() == [False, False, True]
