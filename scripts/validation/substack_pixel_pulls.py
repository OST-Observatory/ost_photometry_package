"""Pixel-level pulls between two independent stacks of the same field.

Usage: python scripts/validation/substack_pixel_pulls.py <run_dir> <filter>

Expects ``<run_dir>/output_a`` and ``output_b`` with ``combined_filter_<f>.fit``.
Stack B is registered onto A; on sky pixels the difference divided by
√(σ_A² + σ_B²) must scatter by ~1 if the propagated stack errors are right
(slightly below 1 where registration interpolation smooths the pixels). The
same on 4×4 pixel sums shows correlated / structured noise on larger scales.
"""

from __future__ import annotations

import sys

import astroalign as aa
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter


def _robust_sigma(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _bin_sum(values: np.ndarray, k: int = 4) -> np.ndarray:
    ny, nx = values.shape
    ny, nx = ny - ny % k, nx - nx % k
    return values[:ny, :nx].reshape(ny // k, k, nx // k, k).sum(axis=(1, 3))


def main(run_dir: str, filter_: str) -> None:
    a = CCDData.read(f"{run_dir}/output_a/combined_filter_{filter_}.fit")
    b = CCDData.read(f"{run_dir}/output_b/combined_filter_{filter_}.fit")
    data_a, data_b = np.asarray(a.data, float), np.asarray(b.data, float)
    sigma_a, sigma_b = np.asarray(a.uncertainty.array, float), np.asarray(b.uncertainty.array, float)

    transform, _ = aa.find_transform(np.nan_to_num(data_b), np.nan_to_num(data_a))
    data_b_reg, _ = aa.apply_transform(transform, np.nan_to_num(data_b), data_a, fill_value=np.nan)
    var_b_reg, _ = aa.apply_transform(transform, np.nan_to_num(sigma_b**2), data_a, fill_value=np.nan)

    sky_a = sigma_clipped_stats(data_a, sigma=3)[1]
    sky_b = sigma_clipped_stats(data_b_reg[np.isfinite(data_b_reg)], sigma=3)[1]
    smooth = gaussian_filter(np.nan_to_num(data_a), 2.0)
    sky_pixels = smooth < sky_a + 2 * np.nanmedian(sigma_a)

    diff = (data_a - sky_a) - (data_b_reg - sky_b)
    predicted = np.sqrt(sigma_a**2 + var_b_reg)
    use = sky_pixels & np.isfinite(diff) & np.isfinite(predicted) & (predicted > 0)
    print(f"{filter_}: pixel pull robust σ {_robust_sigma(diff[use] / predicted[use]):.3f} "
          f"(n={int(use.sum())})")

    sums = _bin_sum(np.where(use, diff, np.nan))
    variances = _bin_sum(np.where(use, predicted**2, np.nan))
    ok = np.isfinite(sums) & np.isfinite(variances) & (variances > 0)
    print(f"   4x4 sums: pull robust σ {_robust_sigma(sums[ok] / np.sqrt(variances[ok])):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
