"""WCS-based image alignment (reproject onto a reference solution)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.nddata import CCDData
from astropy.wcs import WCS

pytest.importorskip("ccdproc")
pytest.importorskip("reproject")


def _tan_wcs(*, crpix: tuple[float, float], crval: tuple[float, float] = (10.0, 20.0)) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = list(crval)
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.cdelt = [-0.0002, 0.0002]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.radesys = "ICRS"
    return wcs


def _gaussian_stamp(shape: tuple[int, int], peak_xy: tuple[float, float]) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x0, y0 = peak_xy
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * 1.5**2))


def _write_ccd(path: Path, data: np.ndarray, wcs: WCS) -> None:
    ccd = CCDData(data.astype(np.float32), unit="adu", wcs=wcs)
    ccd.meta["DATE-OBS"] = "2024-01-01T00:00:00"
    ccd.meta["MJD-OBS"] = 60310.0
    ccd.meta["FILTER"] = "V"
    ccd.meta["IMAGETYP"] = "LIGHT"
    ccd.write(path, overwrite=True)


def test_pixel_offset_on_reference_crpix_shift():
    from ost_photometry.reduce.registration.wcs_align import pixel_offset_on_reference

    src = _tan_wcs(crpix=(32.0, 32.0))
    dst = _tan_wcs(crpix=(35.0, 34.0))
    dx, dy = pixel_offset_on_reference(src, (64, 64), dst)
    # Same CRVAL: source centre maps to dest CRPIX, offset is dest-src CRPIX.
    assert dx == pytest.approx(3.0, abs=0.05)
    assert dy == pytest.approx(2.0, abs=0.05)


def test_reproject_moves_peak_onto_reference_grid():
    from ost_photometry.reduce.registration.wcs_align import reproject_ccd_onto_wcs

    shape = (48, 48)
    src_peak = (20.0, 18.0)
    dst_crpix = (28.0, 26.0)
    src_wcs = _tan_wcs(crpix=src_peak)
    dst_wcs = _tan_wcs(crpix=dst_crpix)
    src = CCDData(_gaussian_stamp(shape, src_peak), unit="adu", wcs=src_wcs)

    out = reproject_ccd_onto_wcs(src, dst_wcs, shape)
    y_peak, x_peak = np.unravel_index(np.nanargmax(out.data), out.data.shape)
    assert x_peak == pytest.approx(dst_crpix[0], abs=1.5)
    assert y_peak == pytest.approx(dst_crpix[1], abs=1.5)
    assert out.wcs is not None
    assert out.wcs.wcs.crpix[0] == pytest.approx(dst_crpix[0])


def test_apply_wcs_align_writes_reference_copy_and_shifted_frame(tmp_path: Path):
    from ost_photometry.reduce.registration.wcs_align import apply_wcs_align

    shape = (48, 48)
    ref_peak = (24.0, 24.0)
    src_peak = (20.0, 18.0)
    ref_wcs = _tan_wcs(crpix=ref_peak)
    src_wcs = _tan_wcs(crpix=src_peak)
    ref_path = tmp_path / "ref.fit"
    src_path = tmp_path / "src.fit"
    _write_ccd(ref_path, _gaussian_stamp(shape, ref_peak), ref_wcs)
    _write_ccd(src_path, _gaussian_stamp(shape, src_peak), src_wcs)
    out_dir = tmp_path / "aligned"
    trans_dir = tmp_path / "trans"
    out_dir.mkdir()
    trans_dir.mkdir()

    apply_wcs_align(str(ref_path), str(ref_path), out_dir, trans_dir)
    apply_wcs_align(str(src_path), str(ref_path), out_dir, trans_dir)

    assert (out_dir / "ref.fit").is_file()
    assert (out_dir / "src.fit").is_file()
    aligned = CCDData.read(out_dir / "src.fit")
    y_peak, x_peak = np.unravel_index(np.nanargmax(aligned.data), aligned.data.shape)
    assert x_peak == pytest.approx(ref_peak[0], abs=1.5)
    assert y_peak == pytest.approx(ref_peak[1], abs=1.5)
    assert aligned.meta.get("ALIGNMTH") == "wcs"
    assert (trans_dir / "src.yaml").is_file()
