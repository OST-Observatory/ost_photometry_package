"""Smoke test: diagnostic plot helpers and config (no real photometry data)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


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

        self.assertFalse(DiagnosticPlots().photometry_mag_vs_error_scatter)
        cfg = PipelineConfig()
        self.assertIsInstance(cfg.diagnostic_plots, DiagnosticPlots)

    @unittest.skipUnless(_plotting_stack_available(), "requires matplotlib and photutils")
    def test_synthetic_inter_filter_separation_plot(self):
        import matplotlib

        matplotlib.use("Agg")

        from ost_photometry.analyze.plots import plot_inter_filter_correlation_separations

        with tempfile.TemporaryDirectory() as tmp:
            path = plot_inter_filter_correlation_separations(
                np.array([0.05, 0.12, 0.08, 0.2]),
                tmp,
                "pdf",
                reference_filter="V",
                other_filters=["B"],
            )
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
