"""Legacy vs differential calibration comparison harness (Phase 0)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from helpers import load_module_from_path, pkg_src

_PKG_SRC = pkg_src()

pytestmark = pytest.mark.comparison


def _import_submodule(dotted: str):
    return importlib.import_module(dotted)


def _deps_available() -> bool:
    try:
        import photutils  # noqa: F401
        import regions  # noqa: F401
        return True
    except ImportError:
        return False


def _legacy_linear_fit_zp_t(
    inst: np.ndarray, std: np.ndarray, color: np.ndarray
) -> tuple[float, float]:
    """Simple legacy-style fit: delta = T * color + ZP."""
    delta = std - inst
    a = np.column_stack([color, np.ones_like(color)])
    coeffs, _, _, _ = np.linalg.lstsq(a, delta, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _differential_fit_zp_t(
    tbl: Table, filter_: str, color_filters: tuple[str, str]
) -> tuple[float, float]:
    diff_mod = _import_submodule("ost_photometry.analyze.differential_photometry")
    DifferentialPhotometer = diff_mod.DifferentialPhotometer

    mask = np.asarray(tbl["is_comparison"], dtype=bool)
    phot = DifferentialPhotometer(color_indices={filter_: color_filters})
    result = phot.fit_transformation_epoch(
        tbl,
        epoch_id="epoch_000",
        filters=[filter_],
        comparison_mask=mask,
        mag_col_prefix="mag_",
        std_col_prefix="mag_std_",
        determine_color_terms=True,
        output_dir=None,
    )
    tc = result.transformation[filter_]
    return tc.color_term, tc.zero_point


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_legacy_vs_differential_zp_color_term(
    synthetic_calibration_epoch_table, comparison_tolerance
):
    """Both calibration paths should yield similar T/ZP on synthetic data."""
    tbl = synthetic_calibration_epoch_table
    mask = np.asarray(tbl["is_comparison"], dtype=bool)
    color = np.asarray(tbl["mag_B"][mask] - tbl["mag_V"][mask])
    legacy_t, legacy_zp = _legacy_linear_fit_zp_t(
        np.asarray(tbl["mag_V"][mask]),
        np.asarray(tbl["mag_std_V"][mask]),
        color,
    )
    diff_t, diff_zp = _differential_fit_zp_t(tbl, "V", ("B", "V"))

    tol = comparison_tolerance
    assert abs(legacy_t - diff_t) < tol["color_term_abs"], (
        f"color term mismatch: legacy={legacy_t}, differential={diff_t}"
    )
    assert abs(legacy_zp - diff_zp) < tol["zp_abs"], (
        f"zero point mismatch: legacy={legacy_zp}, differential={diff_zp}"
    )


@pytest.mark.comparison
def test_pipeline_config_calibration_module_switch():
    """Both calibration modules must remain selectable via PipelineConfig."""
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        _PKG_SRC / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    legacy = PipelineConfig(calibration_module="legacy")
    differential = PipelineConfig(calibration_module="differential")
    assert legacy.calibration_module == "legacy"
    assert differential.calibration_module == "differential"


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_calibration_epochs_schema_roundtrip(synthetic_calibration_epoch_table):
    """Bridge-style epoch tables keep required columns for both paths."""
    schema_mod = _import_submodule("ost_photometry.analyze.post_processing.schema")
    ensure_epoch_native = schema_mod.ensure_epoch_native_photometry_table

    tbl = synthetic_calibration_epoch_table
    required = {"id", "ra", "dec", "mag_B", "mag_V"}
    assert required.issubset(set(tbl.colnames))
    native = ensure_epoch_native(tbl)
    assert len(native) == len(tbl)
