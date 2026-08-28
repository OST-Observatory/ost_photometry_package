"""Pre-fit calibrator quality cuts and median-ZP residual clip."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from helpers import isolated_sys_modules, load_module_from_path, stub_analyze_package


def _load_quality_modules():
    analyze = stub_analyze_package("calibration")
    load_module_from_path(
        "ost_photometry.analyze.warnings_types",
        analyze / "warnings_types.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.extinction",
        analyze / "extinction.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.calibration.result",
        analyze / "calibration" / "result.py",
    )
    zp = load_module_from_path(
        "ost_photometry.analyze.calibration.zp",
        analyze / "calibration" / "zp.py",
    )
    quality = load_module_from_path(
        "ost_photometry.analyze.calibration.quality",
        analyze / "calibration" / "quality.py",
    )
    return quality, zp


def _epoch_table(
    *,
    n_field: int = 40,
    n_cal: int = 8,
    mag_lo: float = 12.0,
    mag_hi: float = 16.0,
    cal_mag: float = 13.2,
) -> Table:
    mag_field = np.linspace(mag_lo, mag_hi, n_field)
    mag_cal = cal_mag + np.linspace(-0.3, 0.3, n_cal)
    mag = np.concatenate([mag_field, mag_cal])
    err = 0.015 * 10.0 ** (0.2 * (mag - mag_lo))
    n = mag.size
    qfit = np.full(n, 0.05)
    sharpness = np.full(n, 0.5)
    roundness = np.full(n, 0.1)
    mag_std = np.full(n, np.nan)
    mag_std[n_field:] = mag[n_field:] + 0.15
    tbl = Table()
    tbl["id"] = np.arange(n)
    tbl["mag_V"] = mag
    tbl["err_V"] = err
    tbl["mag_std_V"] = mag_std
    tbl["qfit"] = qfit
    tbl["cfit"] = np.full(n, 0.04)
    tbl["sharpness"] = sharpness
    tbl["roundness"] = roundness
    return tbl


def test_high_photometric_error_is_dropped():
    with isolated_sys_modules():
        quality, _zp = _load_quality_modules()
        tbl = _epoch_table()
        n_field = 40
        tbl["err_V"][n_field:] *= 0.7
        tbl["err_V"][n_field] = 0.8
        mask = quality.calibrator_candidate_mask(
            tbl,
            ["V"],
            photon_factor=None,
            qfit_max=None,
            cfit_max=None,
            qfit_bright_percentile=None,
            cfit_bright_percentile=None,
            apply_finder_shape_cuts=False,
        )
        assert int(np.sum(mask)) == 7
        assert not mask[n_field]
        assert mask[n_field + 1]


def test_high_qfit_is_dropped():
    with isolated_sys_modules():
        quality, _zp = _load_quality_modules()
        tbl = _epoch_table()
        n_field = 40
        tbl["qfit"][n_field] = 0.9
        mask = quality.calibrator_candidate_mask(
            tbl,
            ["V"],
            error_p84_clip=False,
            photon_factor=None,
            qfit_max=0.2,
            cfit_max=None,
            qfit_bright_percentile=None,
            cfit_bright_percentile=None,
            apply_finder_shape_cuts=False,
        )
        assert int(np.sum(mask)) == 7
        assert not mask[n_field]


def test_bad_sharpness_and_roundness_are_dropped():
    with isolated_sys_modules():
        quality, _zp = _load_quality_modules()
        tbl = _epoch_table()
        n_field = 40
        tbl["sharpness"][n_field] = 2.0
        tbl["roundness"][n_field + 1] = 1.5
        mask = quality.calibrator_candidate_mask(
            tbl,
            ["V"],
            error_p84_clip=False,
            photon_factor=None,
            qfit_max=None,
            cfit_max=None,
            qfit_bright_percentile=None,
            cfit_bright_percentile=None,
            apply_finder_shape_cuts=True,
            sharpness_range=(0.2, 1.0),
            roundness_range=(-1.0, 1.0),
        )
        assert int(np.sum(mask)) == 6
        assert not mask[n_field]
        assert not mask[n_field + 1]


def test_quality_cut_falls_back_when_too_few_remain():
    with isolated_sys_modules():
        quality, _zp = _load_quality_modules()
        tbl = _epoch_table(n_field=5, n_cal=4)
        tbl["qfit"][:] = 0.9
        mask = quality.calibrator_candidate_mask(
            tbl,
            ["V"],
            error_p84_clip=False,
            photon_factor=None,
            qfit_max=0.2,
            cfit_max=None,
            qfit_bright_percentile=None,
            cfit_bright_percentile=None,
            apply_finder_shape_cuts=False,
            min_keep=3,
        )
        assert int(np.sum(mask)) == 4


def test_median_zp_residual_clip_drops_outlier():
    with isolated_sys_modules():
        _, zp = _load_quality_modules()
        n = 10
        mag = np.full(n, 13.0)
        mag_std = mag + 0.15
        mag_std[0] = mag[0] + 2.0
        tbl = Table()
        tbl["mag_V"] = mag
        tbl["mag_std_V"] = mag_std
        cand = np.ones(n, dtype=bool)
        unclipped = zp.fit_median_zero_point_epoch(
            tbl, "epoch_000", ["V"], cand, sigma_clip=None
        )
        clipped = zp.fit_median_zero_point_epoch(
            tbl, "epoch_000", ["V"], cand, sigma_clip=2.5
        )
        assert unclipped.transformation["V"].n_stars_used == 10
        assert clipped.transformation["V"].n_stars_used == 9
        assert not clipped.calibrator_mask_by_filter["V"][0]
        assert np.all(clipped.calibrator_mask_by_filter["V"][1:])


def test_missing_qfit_column_is_not_a_cut():
    with isolated_sys_modules():
        quality, _zp = _load_quality_modules()
        tbl = _epoch_table()
        tbl.remove_column("qfit")
        tbl.remove_column("cfit")
        mask = quality.calibrator_candidate_mask(
            tbl,
            ["V"],
            error_p84_clip=False,
            photon_factor=None,
            apply_finder_shape_cuts=False,
        )
        assert int(np.sum(mask)) == 8
