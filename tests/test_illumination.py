"""Illumination (sky flat) correction from the science frames."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty

from ost_photometry.reduce.illumination import (
    ILLUMINATION_KEYWORD,
    correct_groups,
    illumination_map,
)

SHAPE = (512, 768)
SKY = 200.0
NEBULA = (380, 250)  # x, y


def _away_from_nebula(margin=150):
    """Pixels where the map is measured, not interpolated under the nebula."""
    yy, xx = np.indices(SHAPE)
    edge = (yy > 40) & (yy < SHAPE[0] - 40) & (xx > 40) & (xx < SHAPE[1] - 40)
    return edge & (np.hypot(xx - NEBULA[0], yy - NEBULA[1]) > margin)


def _response():
    """A flat error: rings (600 px period, as on the OST flats) plus tilt, ±3 %."""
    yy, xx = np.indices(SHAPE)
    r = np.hypot(yy - 300, xx - 420)
    return 1.0 + 0.025 * np.cos(2 * np.pi * r / 600.0) + 0.01 * (xx / SHAPE[1] - 0.5)


def _scene(rng, offset):
    """Sky + stars + a nebula; the field shifts by ``offset`` px per frame."""
    yy, xx = np.indices(SHAPE)
    image = np.full(SHAPE, SKY)
    stars = []
    for _ in range(60):
        x, y = rng.uniform(20, SHAPE[1] - 20), rng.uniform(20, SHAPE[0] - 20)
        flux = rng.uniform(2e4, 1e5)
        stars.append((x + offset[0], y + offset[1], flux))
    for x, y, flux in stars:
        image += flux / (2 * np.pi * 2.0**2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 8.0)
    # nebula at the same sky position, i.e. (almost) the same detector position
    image += 40.0 * np.exp(-((xx - NEBULA[0] - offset[0]) ** 2
                             + (yy - NEBULA[1] - offset[1]) ** 2) / (2 * 50.0**2))
    return image, stars


def _write_frames(tmp_path, n=14, seed=3):
    rng = np.random.default_rng(seed)
    response = _response()
    paths, truth = [], []
    for k in range(n):
        offset = rng.uniform(-8, 8, 2)
        scene, stars = _scene(rng, offset)
        observed = rng.normal(scene * response, np.sqrt(scene * response))
        ccd = CCDData(observed.astype(np.float32), unit="electron",
                      uncertainty=StdDevUncertainty(np.sqrt(scene * response).astype(np.float32)))
        ccd.meta.update({"FILTER": "H alpha", "INSTRUME": "QHY600M", "XBINNING": 3,
                         "FLATGRP": "MF_test"})
        path = tmp_path / f"light_{k:02d}.fits"
        ccd.write(path)
        paths.append(path)
        truth.append(stars)
    return paths, response, truth


def test_map_recovers_the_flat_error(tmp_path):
    paths, response, _truth = _write_frames(tmp_path)
    illum = illumination_map(paths, min_frames=10)
    assert illum is not None and illum.n_frames == len(paths)
    expected = response / np.median(response)
    error = (illum.data / expected - 1)[_away_from_nebula()]
    error -= np.median(error)  # normalisation (median of map and response differ)
    assert np.percentile(np.abs(error), 95) < 0.0075  # pattern ±3 %
    # the nebula is masked, not taken for illumination
    assert 0.02 < illum.masked_fraction < 0.3
    # too few frames: no map
    assert illumination_map(paths[:5], min_frames=10) is None


def test_correction_flattens_the_sky_and_keeps_star_fluxes(tmp_path):
    paths, response, truth = _write_frames(tmp_path)
    before = [fits.getdata(p).astype(float) for p in paths]
    result = correct_groups(paths, lambda h: (h["FLATGRP"], h["FILTER"]), tmp_path / "maps",
                            min_frames=10)
    (illum,) = result.values()
    assert (tmp_path / "maps" / "ILLUM_MF_test_H_alpha.fits").is_file()
    with fits.open(paths[0]) as hdul:
        assert hdul[0].header[ILLUMINATION_KEYWORD].startswith("ILLUM_")
        assert "UNCERT" in [h.name for h in hdul]
    corrected = fits.getdata(paths[0]).astype(float)
    sky_before = np.median(before[0][::16, ::16].reshape(-1, 1), axis=1)
    # sky spread: was the response pattern (±3 %), now noise-limited
    sky = _away_from_nebula()

    def spread(img):
        blocks = np.where(sky, img, np.nan)[: 512 // 32 * 32, : 768 // 32 * 32].reshape(16, 32, 24, 32)
        med = np.nanmedian(blocks, axis=(1, 3))
        return np.nanpercentile(med, 90) / np.nanpercentile(med, 10) - 1
    deep_before = np.median(before, axis=0)
    deep_after = np.median([fits.getdata(p).astype(float) for p in paths], axis=0)
    assert spread(deep_before) > 0.025 and spread(deep_after) < 0.01
    assert sky_before.size > 0
    # stars: flux / true flux varied with the response, now constant
    yy, xx = np.indices(SHAPE)
    ratios_before, ratios_after, at_star = [], [], []
    stars = truth[0]
    for x, y, flux in stars:
        if not (20 < x < SHAPE[1] - 20 and 20 < y < SHAPE[0] - 20):
            continue
        if np.hypot(x - NEBULA[0], y - NEBULA[1]) < 200:
            continue
        if min(np.hypot(x - a, y - b) for a, b, _f in stars if (a, b) != (x, y)) < 25:
            continue
        at_star.append(response[int(round(y)), int(round(x))])
        aperture = (xx - x) ** 2 + (yy - y) ** 2 <= 8.0**2
        annulus = ((xx - x) ** 2 + (yy - y) ** 2 > 12.0**2) & ((xx - x) ** 2 + (yy - y) ** 2 <= 16.0**2)
        for img, out in ((before[0], ratios_before), (corrected, ratios_after)):
            net = img[aperture].sum() - np.median(img[annulus]) * aperture.sum()
            out.append(net / flux)
    # measured / true flux follows the response before (slope 1), not after
    assert len(ratios_after) >= 15
    slope_before = np.polyfit(at_star, ratios_before, 1)[0] / np.median(ratios_before)
    slope_after = np.polyfit(at_star, ratios_after, 1)[0] / np.median(ratios_after)
    assert 0.7 < slope_before < 1.5
    assert abs(slope_after) < 0.35 * slope_before  # at least three times smaller
    assert np.median(ratios_after) == pytest.approx(np.median(ratios_before) / np.median(
        illum.data), rel=0.02)
    # a second run leaves corrected frames alone
    again = correct_groups(paths, lambda h: (h["FLATGRP"], h["FILTER"]), tmp_path / "maps")
    assert again == {}
