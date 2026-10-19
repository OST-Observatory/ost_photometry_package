"""Mask-aware weighted average (workaround for ccdproc's weighted average)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.nddata import CCDData, StdDevUncertainty

from ost_photometry.reduce.weighted_combine import (
    CCDPROC_CHECKED_VERSION,
    masked_weighted_mean,
    weighted_average_combine,
)

WEIGHTS = np.array([0.5, 1.0, 1.5, 2.0])


def _frames(shape=(20, 30), seed=2):
    rng = np.random.default_rng(seed)
    data = 8.0 + rng.normal(0, 0.5, (len(WEIGHTS), *shape))
    mask = np.zeros(data.shape, dtype=bool)
    mask[1, 5, :] = True          # a bad row in frame 1
    mask[2, :, 7] = True          # a bad column in frame 2
    mask[:, 0, 0] = True          # masked everywhere
    data[mask] = -100.0           # garbage under the mask
    return data, mask


def test_masked_values_do_not_bias_the_mean():
    data, mask = _frames()
    mean, uncertainty, empty = masked_weighted_mean(np.ma.masked_array(data, mask), WEIGHTS,
                                                    sigma_clip_thresholds=None)
    valid = np.ma.masked_array(data, mask)
    expected = np.ma.average(valid, axis=0, weights=np.broadcast_to(WEIGHTS[:, None, None],
                                                                    data.shape))
    np.testing.assert_allclose(mean[~empty], expected.filled(np.nan)[~empty])
    lo, hi = valid.min(axis=0).filled(np.nan), valid.max(axis=0).filled(np.nan)
    assert np.all((mean[~empty] >= lo[~empty] - 1e-12) & (mean[~empty] <= hi[~empty] + 1e-12))
    assert empty[0, 0] and np.isnan(mean[0, 0]) and empty.sum() == 1
    assert np.all(np.isfinite(uncertainty[~empty]))


def test_sigma_clipping_rejects_outliers():
    data = np.full((8, 3, 3), 10.0) + np.random.default_rng(0).normal(0, 0.1, (8, 3, 3))
    data[3, 1, 1] = 500.0  # cosmic ray
    mean, _, _ = masked_weighted_mean(np.ma.masked_array(data), np.arange(1.0, 9.0))
    assert abs(mean[1, 1] - 10.0) < 0.2


def test_weighted_average_combine_files_and_blocks(tmp_path):
    data, mask = _frames()
    paths = []
    for i in range(len(WEIGHTS)):
        ccd = CCDData(data[i].astype(np.float32), unit="electron/s", mask=mask[i],
                      uncertainty=StdDevUncertainty(np.ones(data[i].shape)))
        ccd.meta["EXPTIME"] = 30.0
        ccd.meta["FILTER"] = "V"
        path = tmp_path / f"f{i}.fit"
        ccd.write(path)
        paths.append(path)
    whole = weighted_average_combine(paths, WEIGHTS)
    blocks = weighted_average_combine(paths, WEIGHTS, mem_limit=1)  # one row per block
    np.testing.assert_allclose(blocks.data, whole.data, equal_nan=True)
    np.testing.assert_array_equal(blocks.mask, whole.mask)
    assert whole.unit == "electron / s" and whole.meta["FILTER"] == "V"
    assert whole.mask[0, 0] and whole.mask.sum() == 1
    expected, _, _ = masked_weighted_mean(np.ma.masked_array(data, mask), WEIGHTS)
    np.testing.assert_allclose(whole.data, expected, rtol=1e-6, equal_nan=True)
    # The same from CCDData objects
    ccds = [CCDData.read(p) for p in paths]
    np.testing.assert_allclose(weighted_average_combine(ccds, WEIGHTS).data, whole.data,
                               equal_nan=True)
    with pytest.raises(ValueError, match="weights"):
        weighted_average_combine(paths, WEIGHTS[:2])


def test_stack_filter_images_weighted_is_unbiased(tmp_path):
    pytest.importorskip("ccdproc")
    from astropy.io import fits

    from ost_photometry.reduce.workflow.stack import stack_filter_images

    data, mask = _frames()
    paths = []
    for i in range(len(WEIGHTS)):
        ccd = CCDData(data[i].astype(np.float32), unit="electron/s", mask=mask[i])
        ccd.meta.update({"EXPTIME": 30.0, "FILTER": "V", "JD": 2459647.5,
                         "DATE-OBS": "2022-03-08T12:00:00"})
        path = tmp_path / f"f{i}.fit"
        ccd.write(path)
        paths.append(str(path))
    name = stack_filter_images(paths, "average", None, "V", tmp_path, None, weights=WEIGHTS)
    stacked = fits.getdata(tmp_path / name).astype(float)
    # the stack matches the frames' sky levels first (match_background)
    from ost_photometry.reduce.weighted_combine import _frame_pass

    offsets, _sky, _peak = _frame_pass(paths, np.ones(len(paths)), np.asarray(WEIGHTS, float), None,
                                       data.shape[1:], match_background=True, peak_radius=None)
    valid = np.ma.masked_array(data + offsets[:, None, None], mask)
    lo, hi = valid.min(axis=0).filled(np.nan), valid.max(axis=0).filled(np.nan)
    ok = np.isfinite(lo)
    assert np.all((stacked[ok] >= lo[ok] - 1e-4) & (stacked[ok] <= hi[ok] + 1e-4))
    # Rows / columns masked in one frame are not darker than the rest.
    assert abs(np.nanmedian(stacked[5]) - np.nanmedian(stacked[10])) < 0.3


@pytest.mark.xfail(
    strict=True,
    reason=(
        f"ccdproc <= {CCDPROC_CHECKED_VERSION}: the weighted average divides by the weights "
        "of masked / clipped values too. If this test XPASSes, the installed ccdproc is "
        "fixed: switch stack_filter_images / combine_camera_stacks back to ccdproc.combine "
        "and drop ost_photometry.reduce.weighted_combine (docs/TODO.md)."
    ),
)
def test_ccdproc_weighted_average_bug():
    ccdp = pytest.importorskip("ccdproc")
    data = np.full((3, 4, 4), 10.0)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 1, 1] = True
    ccds = [CCDData(d, unit="electron", mask=m) for d, m in zip(data, mask, strict=True)]
    combined = ccdp.combine(ccds, method="average", weights=np.array([1.0, 2.0, 3.0]))
    assert combined.data[1, 1] == pytest.approx(10.0)  # ccdproc 2.5.1 gives 10 * 5/6


def _noisy_frames(n, shape=(60, 60), sigma=2.0, seed=0, truth=100.0):
    from astropy.nddata import CCDData, StdDevUncertainty

    rng = np.random.default_rng(seed)
    frames = []
    sig = np.full(shape, sigma)
    for i in range(n):
        s = sig * (1.0 + 0.5 * i)  # frames of different quality
        data = truth + rng.normal(0.0, s)
        frames.append(CCDData(data, unit="electron", uncertainty=StdDevUncertainty(s)))
    return frames


@pytest.mark.parametrize("method", ["average", "median"])
@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_stack_uncertainty_matches_scatter(method, n):
    """Pull of the stacked value against the truth has σ ≈ 1."""
    from ost_photometry.reduce.weighted_combine import combine_images

    if method == "median" and n == 1:
        pytest.skip("median of one frame is the frame")
    pulls = []
    for seed in range(40):
        frames = _noisy_frames(n, seed=seed)
        out = combine_images(frames, method=method, sigma_clip=False)
        pulls.append((out.data - 100.0) / out.uncertainty.array)
    pulls = np.concatenate([p.ravel() for p in pulls])
    assert np.all(np.isfinite(pulls))
    tolerance = 0.06 if method == "average" else 0.1
    assert np.std(pulls) == pytest.approx(1.0, abs=tolerance)


def test_weighted_stack_uncertainty_uses_weights():
    from ost_photometry.reduce.weighted_combine import combine_images

    pulls = []
    for seed in range(40):
        frames = _noisy_frames(4, seed=seed)
        w = 1.0 / np.array([f.uncertainty.array[0, 0] ** 2 for f in frames])
        out = combine_images(frames, method="average", weights=w, sigma_clip=False)
        pulls.append(((out.data - 100.0) / out.uncertainty.array).ravel())
    assert np.std(np.concatenate(pulls)) == pytest.approx(1.0, abs=0.06)


def test_single_coverage_pixel_keeps_its_sigma():
    """Where only one frame covers a pixel the stack has that frame's σ (was 0)."""
    from ost_photometry.reduce.weighted_combine import combine_images

    frames = _noisy_frames(3)
    for f in frames[1:]:
        f.mask = np.zeros(f.shape, dtype=bool)
        f.mask[:, :10] = True
    out = combine_images(frames, method="average", sigma_clip=False)
    assert np.allclose(out.uncertainty.array[:, :10], frames[0].uncertainty.array[:, :10])
    assert np.all(out.uncertainty.array > 0)


def test_frames_without_uncertainty_use_scatter():
    from astropy.nddata import CCDData

    from ost_photometry.reduce.weighted_combine import combine_images

    rng = np.random.default_rng(5)
    frames = [CCDData(100.0 + rng.normal(0, 3.0, (80, 80)), unit="adu") for _ in range(20)]
    out = combine_images(frames, method="average", sigma_clip=False)
    assert np.median(out.uncertainty.array) == pytest.approx(3.0 / np.sqrt(20), rel=0.05)


def test_match_background_for_frames_from_two_nights():
    """3 frames with bright sky + 10 with dark sky, 5σ clipping: the stack noise
    must follow its propagated σ only after the sky levels are matched."""
    from ost_photometry.reduce.weighted_combine import combine_images

    rng = np.random.default_rng(4)
    shape = (200, 200)
    frames = []
    for sky, sigma, count in ((3.8, 0.6, 3), (1.8, 0.27, 10)):
        for _ in range(count):
            data = sky + rng.normal(0.0, sigma, shape)
            frames.append(CCDData(data, unit="electron/s", uncertainty=StdDevUncertainty(np.full(shape, sigma))))

    def pull_sigma(stack):
        sky = np.median(stack.data)
        return np.std((stack.data - sky) / stack.uncertainty.array)

    plain = combine_images(frames, method="average", sigma_clip_low_thresh=5, sigma_clip_high_thresh=5)
    matched = combine_images(frames, method="average", sigma_clip_low_thresh=5,
                             sigma_clip_high_thresh=5, match_background=True)
    assert pull_sigma(plain) > 1.5
    assert pull_sigma(matched) == pytest.approx(1.0, abs=0.08)
