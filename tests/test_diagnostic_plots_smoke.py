"""Smoke test: diagnostic plot helpers and config (no real photometry data)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from astropy import wcs
from astropy.table import Table

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from helpers import (  # noqa: E402
    ensure_stub_package,
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
)


def _plotting_stack_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        import photutils  # noqa: F401
    except ImportError:
        return False
    return True


def _load_pipeline_config_module():
    """Load ``pipeline.config`` without importing ``ost_photometry.analyze`` (heavy deps)."""
    import importlib.util

    path = _PKG_SRC / "ost_photometry" / "analyze" / "pipeline" / "config.py"
    spec = importlib.util.spec_from_file_location(
        "ost_photometry.analyze.pipeline.config",
        path,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses looks up the class module in sys.modules during decoration
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestDiagnosticPlotsSmoke(unittest.TestCase):
    def test_diagnostic_plots_config_dataclass(self):
        cfg_mod = _load_pipeline_config_module()
        DiagnosticPlots = cfg_mod.DiagnosticPlots
        PipelineConfig = cfg_mod.PipelineConfig

        self.assertTrue(DiagnosticPlots().photometry_mag_vs_error_scatter)
        self.assertFalse(DiagnosticPlots().photometry_radial_growth_curve)
        cfg = PipelineConfig()
        self.assertIsInstance(cfg.diagnostic_plots, DiagnosticPlots)
        self.assertTrue(cfg.diagnostic_plots.calibration_instrumental_vs_catalog)

    @unittest.skipUnless(_plotting_stack_available(), "requires matplotlib and photutils")
    def test_synthetic_inter_filter_separation_plot(self):
        import matplotlib

        matplotlib.use("Agg")

        from ost_photometry.analyze.plots import (
            plot_inter_filter_correlation_separations,
            plot_inter_filter_correlation_separations_overview,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = plot_inter_filter_correlation_separations(
                np.array([0.05, 0.12, 0.08, 0.2]),
                tmp,
                "pdf",
                reference_filter="V",
                other_filters=["B"],
                title_suffix="reference images — V: id=0",
            )
            self.assertTrue(path.is_file())
            path2 = plot_inter_filter_correlation_separations_overview(
                [np.array([0.05, 0.12]), np.array([0.08, 0.2, 0.1])],
                ["000", "001"],
                tmp,
                "pdf",
                reference_filter="V",
                other_filters=["B"],
                pairing_mode="jd_nearest",
            )
            self.assertTrue(path2.is_file())


def _load_wcs_residuals():
    root = pkg_src() / "ost_photometry"
    analyze = root / "analyze"
    ensure_stub_package("ost_photometry", path=root)
    ensure_stub_package("ost_photometry.analyze", path=analyze)
    ensure_stub_package("ost_photometry.analyze.correlate", path=analyze / "correlate")
    return load_module_from_path(
        "ost_photometry.analyze.correlate.wcs_residuals",
        analyze / "correlate" / "wcs_residuals.py",
    )


def _load_calibration_qc():
    root = pkg_src() / "ost_photometry"
    analyze = root / "analyze"
    ensure_stub_package("ost_photometry", path=root)
    ensure_stub_package("ost_photometry.analyze", path=analyze)
    ensure_stub_package("ost_photometry.analyze.plots", path=analyze / "plots")
    return load_module_from_path(
        "ost_photometry.analyze.plots.calibration_qc",
        analyze / "plots" / "calibration_qc.py",
    )


def test_mag_vs_error_qc_plots(tmp_path):
    pytest.importorskip("matplotlib")
    with isolated_sys_modules():
        qc = _load_calibration_qc()
        mag = np.linspace(10.0, 18.0, 80)
        err = -0.01 * 10 ** (0.2 * (mag - 10.0))
        err[-1] = -5.0
        mag_p, err_p = qc._finite_mag_and_positive_err(mag, err)
        assert mag_p.size == 80
        assert np.all(err_p > 0)

        basic = Table({"mags_fit": mag, "mags_unc": -err_p})
        path = qc.plot_photometry_mag_vs_error(basic, tmp_path, "pdf")
        assert path is not None and path.is_file()

        rich = Table(
            {
                "mags_fit": mag,
                "mags_unc": err_p,
                "is_comparison": np.arange(80) < 10,
                "x_fit": np.linspace(20.0, 180.0, 80),
                "y_fit": np.linspace(15.0, 140.0, 80),
                "qfit": np.linspace(0.02, 0.4, 80),
                "sharpness": np.linspace(0.3, 0.8, 80),
                "roundness": np.linspace(-0.2, 0.4, 80),
            }
        )
        path_q = qc.plot_photometry_mag_vs_error(
            rich,
            tmp_path,
            "pdf",
            filename_stem="photometry_mag_vs_error_quality",
            image_shape=(200, 160),
        )
        assert path_q is not None and path_q.is_file()

        calib = Table(
            {
                "mags_fit": mag,
                "mags_unc": err_p,
                "is_comparison": np.arange(80) < 18,
                "is_calibrator": np.arange(80) < 10,
                "x_fit": np.linspace(20.0, 180.0, 80),
                "y_fit": np.linspace(15.0, 140.0, 80),
                "qfit": np.linspace(0.02, 0.4, 80),
                "sharpness": np.linspace(0.3, 0.8, 80),
                "roundness": np.linspace(-0.2, 0.4, 80),
            }
        )
        path_c = qc.plot_photometry_mag_vs_error(
            calib,
            tmp_path,
            "pdf",
            filename_stem="photometry_mag_vs_error_calibrators",
        )
        assert path_c is not None and path_c.is_file()

        overview = qc.plot_photometry_mag_vs_error_overview(
            [mag, mag + 0.1, mag - 0.05],
            [-err_p, -err_p * 1.1, err_p * 0.9],
            tmp_path,
            "pdf",
            image_jd=[2460000.1, 2460000.2, 2460000.35],
            image_airmass=[1.05, 1.4, 1.9],
        )
        assert overview is not None and overview.is_file()


def test_binned_error_percentiles_and_photon_model():
    pytest.importorskip("numpy")
    with isolated_sys_modules():
        qc = _load_calibration_qc()
        rng = np.random.default_rng(0)
        mag = np.linspace(10.0, 18.0, 200)
        true = qc._photon_noise_sigma(mag, 0.008, 2e-9)
        err = np.abs(true * (1.0 + 0.05 * rng.normal(size=mag.size)))
        centers, p16, p50, p84 = qc._binned_error_percentiles(mag, err)
        assert centers.size >= 3
        assert np.all(np.diff(centers) > 0)
        assert np.all(p16 <= p50)
        assert np.all(p50 <= p84)
        assert p50[-1] > p50[0]
        params = qc._fit_photon_noise_envelope(mag, err)
        assert params is not None
        floor, faint_scale = params
        assert floor > 0
        assert faint_scale > 0
        faint = qc._photon_noise_sigma(np.array([18.0]), floor, faint_scale)[0]
        bright = qc._photon_noise_sigma(np.array([10.0]), floor, faint_scale)[0]
        assert faint > bright


def test_quality_series_lists_all_available_columns():
    pytest.importorskip("numpy")
    with isolated_sys_modules():
        qc = _load_calibration_qc()
        n = 24
        table = Table(
            {
                "qfit": np.linspace(0.01, 0.3, n),
                "sharpness": np.linspace(0.2, 0.9, n),
                "roundness": np.linspace(-0.4, 0.4, n),
                "x_fit": np.arange(n, dtype=float),
                "y_fit": np.arange(n, dtype=float),
            }
        )
        ok = np.ones(n, dtype=bool)
        series = qc._quality_series_for_plot(table, ok)
        labels = [label for _values, label in series]
        assert len(series) == 3
        assert any("qfit" in label for label in labels)
        assert any("sharpness" in label for label in labels)
        assert any(label == "roundness" for label in labels)


def test_snr_guide_styles_are_distinct():
    pytest.importorskip("numpy")
    with isolated_sys_modules():
        qc = _load_calibration_qc()
        s10 = qc._SNR_GUIDE_STYLE[10.0]
        s5 = qc._SNR_GUIDE_STYLE[5.0]
        assert s10["color"] != s5["color"]
        assert s10["ls"] != s5["ls"]
        assert qc._MEDIAN_COLOR != qc._PHOTON_COLOR


def test_snr_guides_stay_inside_axes():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    with isolated_sys_modules():
        qc = _load_calibration_qc()
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.set_ylim(0.004, 0.04)
        qc._draw_snr_guides(ax)
        qc._expand_ylim_for_snr_guides(ax)
        _y0, y1 = ax.get_ylim()
        plt.close(fig)
        assert y1 > qc._snr_sigma(5.0)
        assert y1 > qc._snr_sigma(10.0)


def test_residual_geometry_rotation_vs_scale():
    pytest.importorskip("numpy")
    with isolated_sys_modules():
        qc = _load_calibration_qc()
        rng = np.random.default_rng(0)
        x0, y0 = 100.0, 80.0
        phi = rng.uniform(0.0, 2.0 * np.pi, 120)
        radius = rng.uniform(10.0, 90.0, 120)
        x = x0 + radius * np.cos(phi)
        y = y0 + radius * np.sin(phi)

        theta = np.radians(0.3)
        dx = -theta * (y - y0) + 1.5
        dy = theta * (x - x0) - 0.8
        rot = qc.residual_geometry_summary(x, y, dx, dy, x0=x0, y0=y0)
        assert rot["n"] >= 100
        assert abs(rot["median_dx_pix"] - 1.5) < 0.05
        assert abs(rot["median_dy_pix"] + 0.8) < 0.05
        assert rot["rms_tangential_pix"] > 5.0 * rot["rms_radial_pix"]
        assert abs(rot["rotation_arcmin"] - 18.0) < 1.0
        assert abs(rot["scale_fraction"]) < 0.002

        scale = 0.004
        dx_s = scale * (x - x0) + 0.2
        dy_s = scale * (y - y0) - 0.3
        scl = qc.residual_geometry_summary(x, y, dx_s, dy_s, x0=x0, y0=y0)
        assert scl["rms_radial_pix"] > 5.0 * scl["rms_tangential_pix"]
        assert abs(scl["scale_fraction"] - scale) < 5e-4
        assert abs(scl["rotation_arcmin"]) < 1.0

        dx_t = np.full_like(x, 2.0)
        dy_t = np.full_like(y, -1.0)
        trn = qc.residual_geometry_summary(x, y, dx_t, dy_t, x0=x0, y0=y0)
        assert abs(trn["median_dx_pix"] - 2.0) < 1e-9
        assert abs(trn["median_dy_pix"] + 1.0) < 1e-9
        assert trn["rms_radial_pix"] < 1e-9
        assert trn["rms_tangential_pix"] < 1e-9


def test_residual_geometry_plot_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    with isolated_sys_modules():
        qc = _load_calibration_qc()
        x = np.linspace(10.0, 90.0, 40)
        y = np.linspace(12.0, 70.0, 40)
        dx = 0.05 * (x - 50.0)
        dy = 0.05 * (y - 40.0)
        image = np.linspace(0.0, 1.0, 80 * 100).reshape(80, 100)
        path = qc.plot_inter_filter_correlation_geometry(
            x,
            y,
            dx,
            dy,
            tmp_path,
            "pdf",
            image_data=image,
            reference_filter="V",
            other_filter="B",
        )
        assert path is not None and path.is_file()
        overview = qc.plot_inter_filter_correlation_geometry_overview(
            [qc.residual_geometry_summary(x, y, dx, dy, x0=50.0, y0=40.0)],
            tmp_path,
            "pdf",
            pair_labels=["000_B"],
        )
        assert overview is not None and overview.is_file()


def test_catalog_crossmatch_diagnostic_plots(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("astropy.wcs")
    from astropy.wcs import WCS

    with isolated_sys_modules():
        qc = _load_calibration_qc()
        n = 60
        rng = np.random.default_rng(2)
        mag = np.linspace(11.0, 16.5, n)
        sep = np.abs(0.15 + 0.05 * rng.normal(size=n))
        sep[-8:] = 1.4 + 0.2 * rng.random(8)
        table = Table(
            {
                "x": np.linspace(20.0, 180.0, n),
                "y": np.linspace(15.0, 150.0, n),
                "mag_V": mag,
                "mag_std_V": mag + 0.2,
                "match_sep_arcsec": sep,
                "match_sep2_arcsec": sep + 0.4,
                "is_calibrator": np.arange(n) < 12,
                "ra_cat": np.full(n, 359.35),
                "dec_cat": np.full(n, 56.73),
            }
        )
        path = qc.plot_calibration_crossmatch_diagnostics(table, tmp_path, "pdf")
        assert path is not None and path.is_file()

        wcs_image = WCS(
            {
                "NAXIS": 2,
                "NAXIS1": 200,
                "NAXIS2": 160,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 359.35,
                "CRVAL2": 56.73,
                "CRPIX1": 100.5,
                "CRPIX2": 80.5,
                "CDELT1": -0.001,
                "CDELT2": 0.001,
                "CUNIT1": "deg",
                "CUNIT2": "deg",
            }
        )
        table["x"] = np.full(n, 100.5)
        table["y"] = np.full(n, 80.5)
        table["ra_cat"] = np.full(n, 359.35)
        table["dec_cat"] = np.full(n, 56.73 + 0.0003)
        vec = qc.catalog_match_pixel_residuals(table, wcs_image)
        assert vec is not None
        x, y, dx, dy, sep_v = vec
        assert x.size == n
        assert np.all(np.isfinite(dx) & np.isfinite(dy) & np.isfinite(sep_v))
        geom = qc.plot_inter_filter_correlation_geometry(
            x,
            y,
            dx,
            dy,
            tmp_path,
            "pdf",
            sep_arcsec=sep_v,
            filename_stem="calibration_crossmatch_geometry",
            title="Catalog cross-match residual geometry",
        )
        assert geom is not None and geom.is_file()


def test_residual_vectors_on_rotated_wcs():
    pytest.importorskip("astropy")
    with isolated_sys_modules():
        residuals = _load_wcs_residuals()
        qc = _load_calibration_qc()
        header = {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CRPIX1": 50.0,
            "CRPIX2": 50.0,
            "CDELT1": -0.001,
            "CDELT2": 0.001,
        }
        wcs_ref = wcs.WCS(header)
        theta = np.radians(0.2)
        cdelt1, cdelt2 = -0.001, 0.001
        header_rot = dict(header)
        header_rot.pop("CDELT1")
        header_rot.pop("CDELT2")
        header_rot["CD1_1"] = cdelt1 * np.cos(theta)
        header_rot["CD1_2"] = -cdelt2 * np.sin(theta)
        header_rot["CD2_1"] = cdelt1 * np.sin(theta)
        header_rot["CD2_2"] = cdelt2 * np.cos(theta)
        wcs_rot = wcs.WCS(header_rot)

        yy, xx = np.mgrid[15:85:8, 15:85:8]
        x = xx.ravel().astype(float)
        y = yy.ravel().astype(float)
        dx, dy, sep = residuals.residual_vectors_on_reference_wcs(
            x, y, wcs_ref, x, y, wcs_rot
        )
        assert dx.shape == x.shape
        assert np.all(np.isfinite(sep))
        summary = qc.residual_geometry_summary(x, y, dx, dy, x0=49.5, y0=49.5)
        assert summary["rms_tangential_pix"] > 3.0 * summary["rms_radial_pix"]
        assert abs(summary["rotation_arcmin"]) > 5.0


def test_catalog_fit_residual_median_zp_vs_color_term():
    pytest.importorskip("numpy")
    with isolated_sys_modules():
        qc = _load_calibration_qc()
        rng = np.random.default_rng(0)
        n = 80
        color = np.linspace(-0.2, 1.4, n)
        t_coef, zp = 0.25, 1.5
        m_inst = np.linspace(12.0, 16.0, n)
        m_cat = m_inst + t_coef * color + zp + rng.normal(0.0, 0.008, n)
        r_med = qc.catalog_fit_residual(m_inst, m_cat, color=color, color_term=0.0)
        r_fit = qc.catalog_fit_residual(
            m_inst, m_cat, color=color, color_term=t_coef, zero_point=zp
        )
        slope_med = qc._theil_sen_slope(color, r_med)
        slope_fit = qc._theil_sen_slope(color, r_fit)
        assert abs(slope_med - t_coef) < 0.04
        assert abs(slope_fit) < 0.03
        assert abs(float(np.median(r_fit))) < 0.01


def test_calibrated_color_removes_delta_zp_offset():
    pytest.importorskip("numpy")
    with isolated_sys_modules():
        qc = _load_calibration_qc()
        n = 40
        zp_b, zp_v = 2.0, 1.4
        color_lit = np.linspace(0.2, 1.1, n)
        m_inst_v = np.linspace(12.0, 15.0, n)
        m_inst_b = m_inst_v + color_lit - (zp_b - zp_v)
        m_cat_v = m_inst_v + zp_v
        m_cat_b = m_inst_b + zp_b
        color_inst = m_inst_b - m_inst_v
        color_cal = qc.calibrated_color(m_inst_b, m_inst_v, zp_b, zp_v)
        assert abs(float(np.median(color_inst - (m_cat_b - m_cat_v))) + (zp_b - zp_v)) < 1e-12
        assert abs(float(np.median(color_cal - (m_cat_b - m_cat_v)))) < 1e-12


def test_catalog_check_plots_used_mask_and_student_call(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    with isolated_sys_modules():
        qc = _load_calibration_qc()
        rng = np.random.default_rng(1)
        n = 50
        color = np.linspace(-0.1, 1.2, n)
        t_coef, zp = 0.18, 1.1
        m_inst = np.linspace(11.5, 16.0, n)
        m_cat = m_inst + t_coef * color + zp + rng.normal(0.0, 0.01, n)
        used = np.zeros(n, dtype=bool)
        used[:30] = True
        err_obs = np.full(n, 0.02)
        err_cat = np.full(n, 0.03)
        residual = qc.catalog_fit_residual(
            m_inst, m_cat, color=color, color_term=t_coef, zero_point=zp
        )
        fig, ax = plt.subplots()
        qc._scatter_catalog_sample(ax, m_inst, m_cat, used)
        assert len(ax.collections) >= 2
        plt.close(fig)

        path_mask = qc.plot_instrumental_vs_catalog_magnitudes(
            m_inst,
            m_cat,
            tmp_path,
            "pdf",
            band_label="V",
            used_mask=used,
            err_obs=err_obs,
            err_cat=err_cat,
            residual=residual,
        )
        assert path_mask is not None and path_mask.is_file()

        m_cal = m_inst + zp
        path_student = qc.plot_instrumental_vs_catalog_magnitudes(
            m_cal,
            m_cat,
            tmp_path,
            "pdf",
            band_label="V",
            show_one_to_one=True,
            x_label=r"$m_\mathrm{cal}$ [mag]",
            filename_stem="calibrated_vs_catalog_V",
            residual=qc.catalog_fit_residual(m_inst, m_cat, zero_point=zp),
        )
        assert path_student is not None and path_student.is_file()

        path_hist = qc.plot_zeropoint_residual_distribution(
            residual,
            tmp_path,
            "pdf",
            band_label="V",
            used_mask=used,
        )
        assert path_hist is not None and path_hist.is_file()

        path_col = qc.plot_zeropoint_residual_vs_color(
            color,
            residual,
            tmp_path,
            "pdf",
            band_label="V",
            color_label="B-V",
            used_mask=used,
            title=r"Rest nach $T\cdot c+\mathrm{ZP}$",
        )
        assert path_col is not None and path_col.is_file()

        color_cal = qc.calibrated_color(m_inst + color, m_inst, zp + 0.2, zp)
        path_cc = qc.plot_calibration_color_color_cal_stars(
            color,
            color_cal,
            tmp_path,
            "pdf",
            used_mask=used,
            color_label="B-V",
        )
        assert path_cc is not None and path_cc.is_file()
