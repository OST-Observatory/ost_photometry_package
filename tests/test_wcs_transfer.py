"""WCS of an unsolved frame from a solved frame of the same camera (star pattern)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from skimage.transform import SimilarityTransform, warp

from ost_photometry.reduce.registration.wcs_align import (
    transfer_wcs,
    transfer_wcs_from_solved_frames,
)

NY, NX = 300, 400
SCALE = 0.4385 / 3600.0


def _stars(rng, n=80):
    xs = rng.uniform(15, NX - 15, n)
    ys = rng.uniform(15, NY - 15, n)
    flux = rng.uniform(500, 20000, n)
    yy, xx = np.indices((NY, NX))
    image = np.full((NY, NX), 100.0)
    for x, y, f in zip(xs, ys, flux, strict=True):
        image += f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.8**2))
    return image + rng.normal(0, 3, image.shape)


def _donor_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [359.33, 56.72]
    w.wcs.crpix = [NX / 2 + 0.5, NY / 2 + 0.5]
    angle = np.radians(97.0)
    w.wcs.cd = SCALE * np.array([[-np.cos(angle), np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return w


def _write(path, data, header=None):
    fits.writeto(path, data.astype(np.float32), header or fits.Header(), overwrite=True)


def test_transferred_wcs_matches_the_true_one(tmp_path):
    rng = np.random.default_rng(3)
    donor_image = _stars(rng)
    donor_wcs = _donor_wcs()
    _write(tmp_path / "donor.fit", donor_image, donor_wcs.to_header())

    #   this frame: the donor's field shifted and rotated (another dither)
    to_donor = SimilarityTransform(rotation=np.radians(1.5), translation=(12.3, -7.8))
    frame = warp(donor_image, to_donor, output_shape=(NY, NX), preserve_range=True,
                 mode="constant", cval=100.0)
    _write(tmp_path / "frame.fit", frame + rng.normal(0, 3, frame.shape),
           fits.Header({"DATE-OBS": "2013-03-04T19:07:00"}))
    assert transfer_wcs(tmp_path / "frame.fit", tmp_path / "donor.fit")

    got = WCS(fits.getheader(tmp_path / "frame.fit")).celestial
    xs = np.array([0, NX - 1, 0, NX - 1, NX / 2])
    ys = np.array([0, 0, NY - 1, NY - 1, NY / 2])
    #   truth: a frame pixel p sits at donor pixel to_donor(p)
    donor_xy = to_donor(np.column_stack([xs, ys]))
    expected = donor_wcs.pixel_to_world(donor_xy[:, 0], donor_xy[:, 1])
    separation_px = expected.separation(got.pixel_to_world(xs, ys)).arcsec / (SCALE * 3600)
    assert separation_px.max() < 0.1
    assert fits.getheader(tmp_path / "frame.fit")["WCSSRC"] == "donor.fit"


def test_donors_of_the_same_camera_only(tmp_path):
    rng = np.random.default_rng(4)
    image = _stars(rng)
    setup = {"INSTRUME": "SBIG ST-8", "XBINNING": 1, "YBINNING": 1}
    _write(tmp_path / "donor.fit", image, fits.Header({**setup, "JD": 2456356.3,
                                                        **dict(_donor_wcs().to_header())}))
    _write(tmp_path / "same.fit", image, fits.Header({**setup, "JD": 2456356.31}))
    _write(tmp_path / "other.fit", image, fits.Header({**setup, "INSTRUME": "QHY600M",
                                                        "JD": 2456356.31}))
    rescued, failed = transfer_wcs_from_solved_frames(
        [str(tmp_path / "same.fit"), str(tmp_path / "other.fit")], [str(tmp_path / "donor.fit")])
    assert rescued == [str(tmp_path / "same.fit")] and failed == [str(tmp_path / "other.fit")]


@pytest.mark.parametrize("scale", [1.05])
def test_a_scaled_match_is_rejected(tmp_path, scale):
    """Two frames of one camera setup have the same pixel scale: a match
    with another scale is a wrong one."""
    rng = np.random.default_rng(5)
    donor_image = _stars(rng)
    _write(tmp_path / "donor.fit", donor_image, _donor_wcs().to_header())
    to_donor = SimilarityTransform(scale=scale, translation=(5.0, 3.0))
    frame = warp(donor_image, to_donor, output_shape=(NY, NX), preserve_range=True,
                 mode="constant", cval=100.0)
    _write(tmp_path / "frame.fit", frame)
    assert not transfer_wcs(tmp_path / "frame.fit", tmp_path / "donor.fit")


def test_alignment_rescues_unsolved_frames_and_skips_the_rest(tmp_path, monkeypatch):
    """shift_method='wcs': a frame ASTAP cannot solve gets the WCS of a solved
    frame of the same camera; one without such a frame is skipped, and
    neither is solved a second time in the workers."""
    from astropy.nddata import CCDData

    from ost_photometry.reduce.registration import align as align_mod
    from ost_photometry.reduce.registration.wcs_align import fits_has_celestial_wcs

    rng = np.random.default_rng(6)
    sky = _stars(rng)
    base = _donor_wcs()
    light = tmp_path / "light"
    light.mkdir()

    def write(name, shift, *, solved, jd, instrument="SBIG ST-8"):
        dx, dy = shift
        ccd = CCDData(np.roll(sky, (dy, dx), axis=(0, 1)).astype(np.float32), unit="adu")
        ccd.meta.update({"IMAGETYP": "LIGHT", "FILTER": "V", "INSTRUME": instrument,
                         "XBINNING": 1, "YBINNING": 1, "JD": jd, "EXPTIME": 60.0})
        if solved:
            w = base.deepcopy()
            w.wcs.crpix = [base.wcs.crpix[0] + dx, base.wcs.crpix[1] + dy]
            ccd.wcs = w
        ccd.write(light / name)

    write("ref.fit", (0, 0), solved=True, jd=2456356.30)
    write("solved.fit", (6, -4), solved=True, jd=2456356.31)
    write("unsolved.fit", (-7, 5), solved=False, jd=2456356.32)
    write("lonely.fit", (3, 3), solved=False, jd=2456356.33, instrument="QHY600M")

    calls = []

    def no_astap(path, *args, **kwargs):
        calls.append(str(path))
        return fits_has_celestial_wcs(path)

    monkeypatch.setattr(align_mod, "ensure_celestial_wcs_on_fits", no_astap)
    (tmp_path / "out").mkdir()
    result = align_mod.align_images(
        light, tmp_path / "out", ["LIGHT"], reference_image_index=0, shift_method="wcs",
        n_cores_multiprocessing=1, image_output_directory="aligned",
    )
    assert sorted(result.aligned_files()) == ["ref.fit", "solved.fit", "unsolved.fit"]
    assert [name for name, _note in result.skipped_files()] == ["lonely.fit"]
    assert "no plate solution" in result.skipped_files()[0][1]
    assert sorted(Path(c).name for c in calls).count("lonely.fit") == 1  # not solved again

    def brightest(path):
        data = np.nan_to_num(CCDData.read(path).data)
        return np.unravel_index(np.argmax(data), data.shape)

    ref_peak = brightest(tmp_path / "out" / "aligned" / "ref.fit")
    assert np.allclose(brightest(tmp_path / "out" / "aligned" / "unsolved.fit"), ref_peak, atol=1)
