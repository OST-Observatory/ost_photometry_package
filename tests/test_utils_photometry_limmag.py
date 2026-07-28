"""Tests for utils photometry and limiting-magnitude helpers."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from helpers import load_module_from_path, pkg_src


def _photometry():
    # Stub terminal_output for rm_edge_objects without package init.
    import sys
    import types

    if "ost_photometry.terminal_output" not in sys.modules:
        term = types.ModuleType("ost_photometry.terminal_output")
        term.print_to_terminal = lambda *a, **k: None
        term.TerminalLog = object
        sys.modules["ost_photometry"] = types.ModuleType("ost_photometry")
        sys.modules["ost_photometry.terminal_output"] = term
    return load_module_from_path(
        "ost_photometry.analyze.utils.photometry",
        pkg_src() / "ost_photometry" / "analyze" / "utils" / "photometry.py",
    )


def _limmag_helpers():
    """Load limiting_magnitude with stubs so pure helpers need no photutils."""
    import sys
    import types

    for name in (
        "ost_photometry",
        "ost_photometry.terminal_output",
        "ost_photometry.analyze",
        "ost_photometry.analyze.plots",
        "ost_photometry.analyze.post_processing",
        "ost_photometry.analyze.post_processing.adapters",
        "ost_photometry.analyze.post_processing.imaging",
        "photutils",
        "photutils.utils",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    term = sys.modules["ost_photometry.terminal_output"]
    term.print_to_terminal = lambda *a, **k: None

    plots = sys.modules["ost_photometry.analyze.plots"]
    plots.starmap = object
    plots.plot_limiting_mag_sky_apertures = object

    adapters = sys.modules["ost_photometry.analyze.post_processing.adapters"]
    adapters.ensure_epoch_native_photometry_table = lambda t: t

    imaging = sys.modules["ost_photometry.analyze.post_processing.imaging"]

    class ImagingPlotContext:
        pass

    imaging.ImagingPlotContext = ImagingPlotContext
    sys.modules["photutils.utils"].ImageDepth = object

    return load_module_from_path(
        "ost_photometry.analyze.utils.limiting_magnitude",
        pkg_src() / "ost_photometry" / "analyze" / "utils" / "limiting_magnitude.py",
    )


def test_flux_to_magnitudes():
    mod = _photometry()
    m, e = mod.flux_to_magnitudes(np.array([100.0]), np.array([1.0]))
    np.testing.assert_allclose(m, [-5.0], rtol=1e-12)
    np.testing.assert_allclose(e, [-0.025], rtol=1e-12)


def test_rm_edge_objects_drops_border_sources():
    mod = _photometry()
    tbl = Table(
        {
            "x_fit": np.array([5.0, 50.0, 95.0]),
            "y_fit": np.array([50.0, 50.0, 50.0]),
        }
    )
    data = np.zeros((100, 100))
    out = mod.rm_edge_objects(tbl, data, border=10)
    assert len(out) == 1
    assert out["x_fit"][0] == 50.0


def test_resolve_limiting_mag_column_prefers_calibrated():
    mod = _limmag_helpers()
    tbl = Table({"mag_cal_V": [12.0], "mag_inst_V": [11.0]})
    assert mod._resolve_limiting_mag_column(tbl, "V") == "mag_cal_V"


def test_subset_photometry_by_epoch():
    mod = _limmag_helpers()
    tbl = Table({"epoch_id": ["a", "b", "a"], "mag_cal_V": [1.0, 2.0, 3.0]})
    sub = mod._subset_photometry_by_epoch(tbl, "a")
    assert len(sub) == 2
    np.testing.assert_array_equal(sub["mag_cal_V"], [1.0, 3.0])
