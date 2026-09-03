"""Exposure pairing: N2 stacks ignore ΔJD; skip messages are user-facing."""

from __future__ import annotations

from types import SimpleNamespace

from helpers import isolated_sys_modules, load_module_from_path, pkg_src, stub_analyze_package


def _load_bridge():
    stub_analyze_package("pipeline", "post_processing")
    root = pkg_src() / "ost_photometry"
    load_module_from_path("ost_photometry.terminal_output", root / "terminal_output.py")
    load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        root / "analyze" / "pipeline" / "config.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.pipeline.context",
        root / "analyze" / "pipeline" / "context.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.post_processing.schema",
        root / "analyze" / "post_processing" / "schema.py",
    )
    return load_module_from_path(
        "ost_photometry.analyze.pipeline.bridge",
        root / "analyze" / "pipeline" / "bridge.py",
    )


def test_format_duration_and_jd_skip_message():
    with isolated_sys_modules():
        bridge = _load_bridge()
        assert "32 min" in bridge.format_duration_days(0.022233796305954456)
        assert "29 min" in bridge.format_duration_days(0.02)
        msg = bridge.describe_calibration_epoch_skip(
            {
                "reason": "jd_exceeds_tolerance",
                "reference_filter": "B",
                "reference_exposure_image_id": 0,
                "failed_filter": "V",
                "best_delta_jd": 0.022233796305954456,
                "jd_tolerance": 0.02,
            }
        )
        assert "Could not pair B image 0 with V" in msg
        assert "32 min" in msg
        assert "29 min" in msg
        assert "jd_exceeds_tolerance" not in msg
        err = bridge.no_calibration_epochs_message(
            [
                {
                    "reason": "jd_exceeds_tolerance",
                    "reference_filter": "B",
                    "reference_exposure_image_id": 0,
                    "failed_filter": "V",
                    "best_delta_jd": 0.0222,
                    "jd_tolerance": 0.02,
                }
            ],
            pairing="jd_nearest",
            jd_tolerance=0.02,
            filter_list=["B", "V"],
        )
        assert "Cannot calibrate: no B+V epoch" in err
        assert "exposure_pairing='index'" in err
        assert "observation_to_calibration_epochs" not in err


def test_one_image_per_filter_uses_index_not_jd_window():
    with isolated_sys_modules():
        bridge = _load_bridge()
        from ost_photometry.analyze.pipeline.config import PipelineConfig
        from ost_photometry.analyze.pipeline.context import AnalysisContext

        def _im(image_id: int, jd: float):
            return SimpleNamespace(image_id=image_id, jd=jd, photometry=object())

        ctx = AnalysisContext(
            image_series_dict={
                "B": SimpleNamespace(image_list=[_im(0, 2459174.3016666668)]),
                "V": SimpleNamespace(image_list=[_im(0, 2459174.323900463)]),
            },
            filter_list=["B", "V"],
            output_dir="/tmp",
        )
        cfg = PipelineConfig(exposure_pairing="jd_nearest", exposure_jd_tolerance=0.02)
        assert bridge._effective_exposure_pairing(ctx, cfg, log=False) == "index"
        groups = bridge.list_exposure_image_groups(ctx, cfg)
        assert len(groups) == 1
        assert groups[0]["B"].image_id == 0
        assert groups[0]["V"].image_id == 0


def test_multi_epoch_still_enforces_jd_tolerance():
    with isolated_sys_modules():
        bridge = _load_bridge()
        from ost_photometry.analyze.pipeline.config import PipelineConfig
        from ost_photometry.analyze.pipeline.context import AnalysisContext

        def _im(image_id: int, jd: float):
            return SimpleNamespace(image_id=image_id, jd=jd, photometry=object())

        ctx = AnalysisContext(
            image_series_dict={
                "B": SimpleNamespace(
                    image_list=[
                        _im(0, 2459174.30),
                        _im(1, 2459174.40),
                    ]
                ),
                "V": SimpleNamespace(
                    image_list=[
                        _im(0, 2459174.3223),
                        _im(1, 2459174.4223),
                    ]
                ),
            },
            filter_list=["B", "V"],
            output_dir="/tmp",
        )
        cfg = PipelineConfig(exposure_pairing="jd_nearest", exposure_jd_tolerance=0.02)
        assert bridge._effective_exposure_pairing(ctx, cfg, log=False) == "jd_nearest"
        skipped: list[dict] = []
        pairs = bridge._pairing_jd_nearest(
            ctx, ["B", "V"], "B", 0.02, skipped, debug=False
        )
        assert pairs == []
        assert skipped and skipped[0]["reason"] == "jd_exceeds_tolerance"
