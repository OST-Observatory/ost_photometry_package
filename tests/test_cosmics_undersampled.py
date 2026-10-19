"""L.A.Cosmic keeps the cores of undersampled stars."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.nddata import CCDData, StdDevUncertainty


def _frame(rng, fwhm=1.8, shape=(400, 400)):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    s = fwhm / 2.3548
    xs = rng.uniform(15, shape[1] - 15, 60)
    ys = rng.uniform(15, shape[0] - 15, 60)
    flux = 10 ** rng.uniform(3.3, 5.0, 60)
    model = np.full(shape, 200.0)
    for x, y, f in zip(xs, ys, flux, strict=True):
        # pixel-integrated Gaussian (5x5 subsampling)
        acc = np.zeros(shape)
        for dy in (np.arange(5) - 2) / 5:
            for dx in (np.arange(5) - 2) / 5:
                acc += np.exp(-((xx + dx - x) ** 2 + (yy + dy - y) ** 2) / (2 * s * s))
        model += f * acc / 25 / (2 * np.pi * s * s)
    data = rng.poisson(model).astype(float) + rng.normal(0, 5, shape)
    cr = (rng.integers(5, shape[0] - 5, 40), rng.integers(5, shape[1] - 5, 40))
    data[cr] += 3000.0
    return data, xs, ys, flux, cr, model


def _apflux(img, xs, ys, r=5.0):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    return np.array([np.sum(img[np.hypot(xx - x, yy - y) <= r]) - 200.0 * np.count_nonzero(np.hypot(xx - x, yy - y) <= r)
                     for x, y in zip(xs, ys, strict=True)])


def test_undersampled_star_cores_survive_lacosmic():
    ccdp = pytest.importorskip("ccdproc")
    pytest.importorskip("astroscrappy")
    from ost_photometry.reduce.workflow.science import _protect_undersampled_stars

    rng = np.random.default_rng(3)
    data, xs, ys, flux, cr, model = _frame(rng)
    ccd = CCDData(data, unit="electron", uncertainty=StdDevUncertainty(np.sqrt(np.abs(data) + 25)))
    cr_input, protected, objlim = _protect_undersampled_stars(ccd, 4.0, "test.fit")
    assert protected is not None and objlim >= 15
    out = ccdp.cosmicray_lacosmic(cr_input, objlim=objlim, readnoise=5.0, sigclip=4.5, satlevel=1e9)
    # stars well away from the injected cosmics
    yy, xx = cr
    far = np.array([np.min(np.hypot(xx - x, yy - y)) > 8 for x, y in zip(xs, ys, strict=True)])
    truth = _apflux(model, xs, ys)
    kept = _apflux(np.asarray(out.data), xs, ys)[far] / truth[far]
    raw = _apflux(data, xs, ys)[far] / truth[far]
    assert np.max(np.abs(kept / raw - 1.0)) < 1e-3
    # the cosmics outside the protected cores are still cleaned
    hit = ~protected[cr]
    assert np.mean(np.asarray(out.data)[cr][hit] - model[cr][hit] < 300.0) > 0.9
