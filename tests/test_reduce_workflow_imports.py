"""Smoke tests for modular reduce workflow imports."""

from __future__ import annotations

import importlib

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


def test_redu_facade_reexports_reduce_main():
    pytest.importorskip("ccdproc")
    redu = importlib.import_module("ost_photometry.reduce.redu")
    assert hasattr(redu, "reduce_main")
    assert hasattr(redu, "ReduceConfig")
