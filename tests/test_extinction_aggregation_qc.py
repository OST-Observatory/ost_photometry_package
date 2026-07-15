"""Tests for site-extinction aggregation QC plots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import load_module_from_path, pkg_src


def _extinction_io_mod():
    pkg = pkg_src() / "ost_photometry" / "analyze"
    load_module_from_path(
        "ost_photometry.analyze.warnings_types",
        pkg / "warnings_types.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.extinction",
        pkg / "extinction.py",
    )
    return load_module_from_path(
        "ost_photometry.analyze.extinction_io",
        pkg / "extinction_io.py",
    )


def _write_night_json(path: Path, filt: str, k: float, k_err: float) -> None:
    payload = {
        "coefficients": {
            filt: {
                "filter_name": filt,
                "k_prime": k,
                "k_prime_err": k_err,
                "k_second": 0.0,
                "k_second_err": 0.0,
                "color_filter_1": "B",
                "color_filter_2": "V",
                "valid": True,
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_per_night_extinction_samples(tmp_path):
    io = _extinction_io_mod()
    n1 = tmp_path / "night_2024_01.json"
    n2 = tmp_path / "night_2024_02.json"
    _write_night_json(n1, "V", 0.14, 0.02)
    _write_night_json(n2, "V", 0.16, 0.03)

    samples = io.collect_per_night_extinction_samples([n1, n2])
    assert "V" in samples
    assert len(samples["V"]) == 2
    assert samples["V"][0][0] == "night_2024_01"


def test_write_extinction_aggregation_qc_plots(tmp_path):
    pytest.importorskip("matplotlib")
    io = _extinction_io_mod()
    n1 = tmp_path / "night_a.json"
    n2 = tmp_path / "night_b.json"
    _write_night_json(n1, "V", 0.14, 0.02)
    _write_night_json(n2, "V", 0.18, 0.03)

    coeffs, meta = io.aggregate_extinction_coefficients([n1, n2], site="TestSite")
    plot_dir = tmp_path / "qc"
    written = io.write_extinction_aggregation_qc_plots(
        [n1, n2],
        coeffs,
        meta,
        plot_dir,
        site="TestSite",
    )
    assert any(p.name == "extinction_nights_V.pdf" for p in written)
    assert any(p.name == "extinction_site_summary.pdf" for p in written)
    assert all(p.is_file() for p in written)
