"""Tests for epoch-native ASCII / ECSV format helpers."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from helpers import ensure_stub_package, load_module_from_path, pkg_src  # noqa: E402


class TestAsciiFormatsId(unittest.TestCase):
    def setUp(self):
        self._modules = sys.modules.copy()

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._modules)

    def test_ascii_write_formats_for_columns_filters_missing_keys(self):
        schema = load_module_from_path(
            "ost_photometry.analyze.post_processing.schema",
            pkg_src() / "ost_photometry" / "analyze" / "post_processing" / "schema.py",
        )
        self.assertEqual(
            schema.ascii_write_formats_for_columns(["id", "x", "y", "ra"]),
            {"id": "{:5.0f}", "x": "{:12.2f}", "y": "{:12.2f}"},
        )
        self.assertEqual(
            schema.ascii_write_formats_for_columns(["x", "y"]),
            {"x": "{:12.2f}", "y": "{:12.2f}"},
        )
        self.assertNotIn("i", schema.ascii_write_formats_for_columns(["i", "x"]))
        self.assertNotIn("id", schema.ascii_write_formats_for_columns(["i", "x"]))

    def test_write_epoch_native_no_formats_key_warning(self):
        import tempfile
        from unittest.mock import patch

        root = pkg_src() / "ost_photometry"
        analyze = root / "analyze"
        ensure_stub_package("ost_photometry", path=root)
        ensure_stub_package("ost_photometry.analyze", path=analyze)
        ensure_stub_package(
            "ost_photometry.analyze.post_processing",
            path=analyze / "post_processing",
        )
        ensure_stub_package("ost_photometry.checks")
        load_module_from_path(
            "ost_photometry.analyze.post_processing.schema",
            analyze / "post_processing" / "schema.py",
        )
        io_mod = load_module_from_path(
            "ost_photometry.analyze.post_processing.io",
            analyze / "post_processing" / "io.py",
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            class _OutPath:
                def __truediv__(self, other):
                    return tmp_path / other

                @property
                def name(self):
                    return str(tmp_path)

            obs = SimpleNamespace(
                image_series_dict={"V": SimpleNamespace(out_path=_OutPath())}
            )
            tbl = Table(
                {
                    "id": np.arange(3, dtype=int),
                    "x": np.array([1.0, 2.0, 3.0]),
                    "y": np.array([4.0, 5.0, 6.0]),
                    "ra": np.zeros(3),
                    "dec": np.zeros(3),
                    "epoch_id": np.array(["epoch_000"] * 3),
                }
            )
            (tmp_path / "tables").mkdir(parents=True, exist_ok=True)
            with patch(
                "ost_photometry.checks.check_output_directories",
                lambda *a, **k: None,
                create=True,
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    path = io_mod.write_epoch_native_magnitudes(obs, tbl)
                    format_warnings = [
                        w
                        for w in caught
                        if issubclass(w.category, AstropyUserWarning)
                        and "formats" in str(w.message).lower()
                    ]
            self.assertTrue(path.is_file())
            self.assertEqual(format_warnings, [])


if __name__ == "__main__":
    unittest.main()
