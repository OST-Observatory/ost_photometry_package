"""Tests for unified protected-object correlation."""

from helpers import load_module_from_path, pkg_src


def _protection_module():
    return load_module_from_path(
        "ost_photometry.analyze.correlate.protection",
        pkg_src() / "ost_photometry" / "analyze" / "correlate" / "protection.py",
    )


def test_merge_protected_object_ids_explicit_always_included():
    prot = _protection_module()
    merged = prot.merge_protected_object_ids(
        protected_object_ids=[7, 3],
        reference_object_ids=[1, 7],
        calibration_object_ids=[3, 9],
        protect_ooi=True,
        protect_calibration_objects=True,
    )
    assert merged == [7, 3, 1, 9]


def test_merge_protected_object_ids_respects_protect_flags():
    prot = _protection_module()
    merged = prot.merge_protected_object_ids(
        protected_object_ids=[5],
        reference_object_ids=[1, 2],
        calibration_object_ids=[3, 4],
        protect_ooi=False,
        protect_calibration_objects=False,
    )
    assert merged == [5]


def test_merge_protected_object_ids_combines_ooi_and_calibration():
    prot = _protection_module()
    merged = prot.merge_protected_object_ids(
        reference_object_ids=[10],
        calibration_object_ids=[20, 10],
        protect_ooi=True,
        protect_calibration_objects=True,
    )
    assert merged == [10, 20]


def test_protected_object_ids_in_pipeline_config():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(
        protected_object_ids=[4, 8],
        protect_ooi=True,
        protect_calibration_objects=True,
    )
    assert cfg.protected_object_ids == [4, 8]
    assert cfg.correlation.protected_object_ids == [4, 8]


def test_protect_reference_obj_alias():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(protect_reference_obj=False)
    assert cfg.protect_ooi is False
