"""Smoke tests for modular reduce workflow imports."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from helpers import load_module_from_path, pkg_src


def test_reduce_workflow_config_and_constants_without_ccdproc():
    wf = pkg_src() / "ost_photometry" / "reduce" / "workflow"
    constants = load_module_from_path(
        "ost_photometry.reduce.workflow.constants",
        wf / "constants.py",
    )
    config = load_module_from_path(
        "ost_photometry.reduce.workflow.config",
        wf / "config.py",
    )
    assert constants.REDUCE_STATUS_REDUCED == "reduced"
    assert hasattr(config, "ReduceConfig")


@pytest.mark.parametrize(
    "symbol",
    [
        "reduce_main",
        "ReduceConfig",
        "master_bias",
        "master_dark",
        "reduce_light",
        "stack_image",
    ],
)
def test_reduce_workflow_public_api(symbol):
    pytest.importorskip("ccdproc")
    wf = importlib.import_module("ost_photometry.reduce.workflow")
    assert symbol in wf.__all__
    assert hasattr(wf, symbol)


def test_reduce_config_is_dataclass():
    from helpers import load_module_from_path, pkg_src

    config = load_module_from_path(
        "ost_photometry.reduce.workflow.config",
        pkg_src() / "ost_photometry" / "reduce" / "workflow" / "config.py",
    )
    cfg = config.ReduceConfig(
        image_path=Path("/tmp/in"),
        output_dir=Path("/tmp/out"),
        image_type_dir={"light": ["LIGHT"]},
        shift_all=True,
    )
    assert cfg.shift_all is True
    assert cfg.shift_method == "aa_true"
    assert cfg.wcs_method == "astap"
    assert cfg.image_path == Path("/tmp/in")


def test_redu_facade_reexports_reduce_main():
    pytest.importorskip("ccdproc")
    redu = importlib.import_module("ost_photometry.reduce.redu")
    assert hasattr(redu, "reduce_main")
    assert hasattr(redu, "ReduceConfig")


@pytest.mark.parametrize(
    "symbol",
    [
        "get_exposure_times",
        "check_exposure_times",
        "find_nearest_exposure_time_to_reference_image",
        "get_instrument_info",
        "determine_wcs_all_images",
        "get_pixel_mask",
        "make_bad_pixel_mask",
        "make_hot_pixel_mask",
        "get_image_type",
        "check_master_files_on_disk",
        "image_file_collection",
    ],
)
def test_reduce_utilities_reexports_workflow_helpers(symbol):
    pytest.importorskip("ccdproc")
    utilities = importlib.import_module("ost_photometry.reduce.utilities")
    assert hasattr(utilities, symbol)
