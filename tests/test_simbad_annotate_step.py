"""Simbad annotation as a post-processing pipeline step."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from helpers import ensure_stub_package, isolated_sys_modules, load_module_from_path, pkg_src

_PRESENT_WCS = object()
_SRC = pkg_src() / "ost_photometry"


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _stub_if_dummy(mod, **attrs) -> None:
    if getattr(mod, "__file__", None) is None:
        for key, val in attrs.items():
            setattr(mod, key, val)


def _stub_simbad_deps(*, annotate=None) -> None:
    analyze = _SRC / "analyze"
    ensure_stub_package("ost_photometry", path=_SRC)
    ensure_stub_package("ost_photometry.analyze", path=analyze)
    ensure_stub_package("ost_photometry.analyze.pipeline", path=analyze / "pipeline")
    ensure_stub_package(
        "ost_photometry.analyze.pipeline.steps",
        path=analyze / "pipeline" / "steps",
    )
    ensure_stub_package(
        "ost_photometry.analyze.post_processing",
        path=analyze / "post_processing",
    )
    ensure_stub_package("ost_photometry.analyze.plots")
    ensure_stub_package("ost_photometry.terminal_output")
    ensure_stub_package("astroquery")
    ensure_stub_package("astroquery.exceptions")
    ensure_stub_package("astroquery.simbad")

    _stub_if_dummy(
        sys.modules["ost_photometry.terminal_output"],
        print_to_terminal=lambda *a, **k: None,
    )
    _stub_if_dummy(
        sys.modules["astroquery.exceptions"],
        TableParseError=type("TableParseError", (Exception,), {}),
    )
    _stub_if_dummy(sys.modules["astroquery.simbad"], Simbad=MagicMock())
    if annotate is not None:
        helper = ensure_stub_package(
            "ost_photometry.analyze.post_processing.simbad_annotate"
        )
        helper.annotate_reference_image_with_simbad = annotate


def _load_config_context_step(*, annotate=None):
    _stub_simbad_deps(annotate=annotate or (lambda *a, **k: None))
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        _SRC / "analyze" / "pipeline" / "config.py",
    )
    ctx_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.context",
        _SRC / "analyze" / "pipeline" / "context.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.pipeline.base",
        _SRC / "analyze" / "pipeline" / "base.py",
    )
    step_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.steps.simbad_annotate",
        _SRC / "analyze" / "pipeline" / "steps" / "simbad_annotate.py",
    )
    return cfg_mod, ctx_mod, step_mod


def _load_helper():
    _stub_simbad_deps()
    return load_module_from_path(
        "ost_photometry.analyze.post_processing.simbad_annotate",
        _SRC / "analyze" / "post_processing" / "simbad_annotate.py",
    )


def _image(*, wcs=_PRESENT_WCS, filter_="V"):
    image = SimpleNamespace(
        wcs=wcs,
        out_path=Path("/tmp/out"),
        filter_=filter_,
        image_id=0,
    )
    image.get_data = MagicMock(return_value=np.ones((8, 8)))
    return image


def _series(*images):
    return SimpleNamespace(image_list=list(images), reference_image_index=0)


def test_default_pipeline_includes_simbad_annotate_after_extraction():
    text = (_SRC / "analyze" / "pipeline" / "orchestrator.py").read_text()
    extract_at = text.find("ExtractionStep(),")
    simbad_at = text.find("SimbadAnnotateStep(),")
    intra_at = text.find("CorrelationIntraStep(),")
    assert 0 <= extract_at < simbad_at < intra_at


def test_skip_single_mode_when_annotate_image_false():
    cfg_mod, ctx_mod, step_mod = _load_config_context_step()
    ctx = ctx_mod.AnalysisContext(
        image_series_dict={"V": _series(_image())},
        filter_list=["V"],
        output_dir="/tmp/out",
    )
    cfg = cfg_mod.PipelineConfig(extraction_mode="single", annotate_image=False)
    assert step_mod.SimbadAnnotateStep().skip(ctx, cfg) is True


def test_does_not_skip_by_default():
    cfg_mod, ctx_mod, step_mod = _load_config_context_step()
    ctx = ctx_mod.AnalysisContext(
        image_series_dict={"V": _series(_image())},
        filter_list=["V"],
        output_dir="/tmp/out",
    )
    cfg = cfg_mod.PipelineConfig()
    assert step_mod.SimbadAnnotateStep().skip(ctx, cfg) is False


def test_skip_multi_mode_when_annotate_reference_image_false():
    cfg_mod, ctx_mod, step_mod = _load_config_context_step()
    ctx = ctx_mod.AnalysisContext(
        image_series_dict={"V": _series(_image(), _image())},
        filter_list=["V"],
        output_dir="/tmp/out",
    )
    cfg = cfg_mod.PipelineConfig(
        extraction_mode="multi",
        annotate_reference_image=False,
    )
    assert step_mod.SimbadAnnotateStep().skip(ctx, cfg) is True


def test_run_annotates_each_filter_reference_image():
    called = []

    def _fake(image, **kwargs):
        called.append(image.filter_)

    cfg_mod, ctx_mod, step_mod = _load_config_context_step(annotate=_fake)
    ctx = ctx_mod.AnalysisContext(
        image_series_dict={
            "V": _series(_image(filter_="V")),
            "B": _series(_image(filter_="B")),
        },
        filter_list=["V", "B"],
        output_dir="/tmp/out",
    )
    step_mod.SimbadAnnotateStep().run(ctx, cfg_mod.PipelineConfig())
    assert called == ["V", "B"]


def test_annotate_skips_when_wcs_missing():
    helper = _load_helper()
    marked = MagicMock()
    helper.mark_simbad_objects_on_image = marked
    helper.annotate_reference_image_with_simbad(_image(wcs=None))
    marked.assert_not_called()


def test_annotate_swallows_query_errors():
    helper = _load_helper()
    helper.mark_simbad_objects_on_image = MagicMock(
        side_effect=RuntimeError("simbad down")
    )
    helper.annotate_reference_image_with_simbad(_image())
