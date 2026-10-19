"""Hot pixels relative to the typical dark rate, also with a bias pedestal."""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData


def _dark(pedestal_adu: float, rng) -> tuple[CCDData, np.ndarray]:
    shape = (300, 300)
    data = pedestal_adu + rng.normal(0.0, 0.5, shape)  # ADU, 60 s, gain 1
    hot = np.zeros(shape, dtype=bool)
    hot[rng.integers(0, 300, 200), rng.integers(0, 300, 200)] = True
    data[hot] += 600.0  # 10 e-/s
    ccd = CCDData(data, unit="adu", meta=fits.Header())
    ccd.meta.update({"EXPTIME": 60.0, "NAXIS1": 300, "NAXIS2": 300})
    return ccd, hot


def test_hot_pixels_found_with_and_without_pedestal(tmp_path):
    from ost_photometry.reduce.masks import make_hot_pixel_mask

    rng = np.random.default_rng(1)
    for pedestal in (0.0, 500.0):  # 500 ADU / 60 s = 8.3 e-/s on every pixel
        ccd, hot = _dark(pedestal, rng)
        out = tmp_path / f"p{int(pedestal)}"
        out.mkdir()
        make_hot_pixel_mask(ccd, 1.0, out)
        mask = fits.getdata(out / "mask_from_dark_300x300.fit").astype(bool)
        assert np.array_equal(mask, hot)
