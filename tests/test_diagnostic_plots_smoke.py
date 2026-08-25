"""Smoke test: diagnostic plot helpers and config (no real photometry data)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from astropy import wcs

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


if __name__ == "__main__":
    unittest.main()
