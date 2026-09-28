"""Deprecated ``estimate_fwhm`` alias and frame-quality config validation."""

from __future__ import annotations

from pathlib import Path

import pytest


def _main_module():
    pytest.importorskip("ccdproc")
    from ost_photometry.reduce.workflow import main

    return main


def test_estimate_fwhm_alias_warns_and_enables_quality(monkeypatch, tmp_path):
    main = _main_module()
    captured = {}

    def fake_run(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr(main, "_run_reduction", fake_run)
    with pytest.warns(DeprecationWarning, match="estimate_fwhm"):
        main.reduce_main(
            str(tmp_path), str(tmp_path / "out"), estimate_fwhm=True,
            measure_frame_quality=False,
        )
    cfg = captured["cfg"]
    assert cfg.measure_frame_quality is True
    assert cfg.reference_image_index is None
    assert cfg.reference_image_selection == "best_fwhm"
    assert cfg.stack_weighting == "none"
    assert cfg.keep_aligned_lights is False


def test_run_reduction_rejects_inconsistent_quality_options(tmp_path):
    main = _main_module()
    from ost_photometry.reduce.workflow.config import ReduceConfig

    base = dict(
        image_path=Path(tmp_path),
        output_dir=Path(tmp_path / "out"),
        image_type_dir={"light": ["LIGHT"], "dark": ["DARK"], "flat": ["FLAT"],
                        "bias": ["BIAS"]},
    )
    with pytest.raises(ValueError, match="frame_selection requires"):
        main._run_reduction(
            ReduceConfig(**base, measure_frame_quality=False,
                         frame_selection={"fwhm_max": 3.0})
        )
    with pytest.raises(ValueError, match="stack_weighting requires"):
        main._run_reduction(
            ReduceConfig(**base, measure_frame_quality=False, stack_weighting="fwhm")
        )
    with pytest.raises(ValueError, match="stack_weighting must be one of"):
        main._run_reduction(ReduceConfig(**base, stack_weighting="seeing"))
    with pytest.raises(ValueError, match="reference_image_selection must be one of"):
        main._run_reduction(ReduceConfig(**base, reference_image_selection="sharpest"))
    with pytest.raises(ValueError, match="Unknown frame_selection keys"):
        main._run_reduction(ReduceConfig(**base, frame_selection={"fwhm_limit": 3.0}))
