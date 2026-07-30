"""Tests for ePSF star-count selection (min / fraction / max)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from helpers import load_module_from_path, pkg_src  # noqa: E402


class TestNEpsfStarsToSelect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module_from_path(
            "ost_photometry.analyze.utils.epsf_selection",
            pkg_src()
            / "ost_photometry"
            / "analyze"
            / "utils"
            / "epsf_selection.py",
        )

    def test_clamps_to_maximum_in_dense_fields(self):
        # 500 stars * 0.2 = 100 → capped at 50
        self.assertEqual(
            self.mod.n_epsf_stars_to_select(
                500,
                fraction_epsf_stars=0.2,
                minimum_n_stars=15,
                maximum_n_stars=50,
            ),
            50,
        )

    def test_raises_to_minimum_when_fraction_small(self):
        self.assertEqual(
            self.mod.n_epsf_stars_to_select(
                40,
                fraction_epsf_stars=0.2,
                minimum_n_stars=15,
                maximum_n_stars=50,
            ),
            15,
        )

    def test_none_maximum_keeps_fraction(self):
        self.assertEqual(
            self.mod.n_epsf_stars_to_select(
                500,
                fraction_epsf_stars=0.2,
                minimum_n_stars=15,
                maximum_n_stars=None,
            ),
            100,
        )

    def test_invalid_max_below_min(self):
        with self.assertRaises(ValueError):
            self.mod.n_epsf_stars_to_select(
                100, minimum_n_stars=20, maximum_n_stars=10
            )

    def test_config_default_and_kwargs(self):
        cfg_mod = load_module_from_path(
            "ost_photometry.analyze.pipeline.config",
            pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
        )
        ext = cfg_mod.ExtractionConfig()
        self.assertEqual(ext.minimum_n_eps_stars, 15)
        self.assertEqual(ext.maximum_n_eps_stars, 100)
        self.assertEqual(ext.fraction_epsf_stars, 0.2)
        kw = ext.main_extract_kwargs()
        self.assertEqual(kw["maximum_n_eps_stars"], 100)
        self.assertEqual(kw["minimum_n_eps_stars"], 15)


if __name__ == "__main__":
    unittest.main()
