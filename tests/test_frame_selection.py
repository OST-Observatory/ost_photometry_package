"""Tests for the ccdproc-free frame selection / weighting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import load_module_from_path, pkg_src


def _mod():
    src = pkg_src() / "ost_photometry"
    load_module_from_path("ost_photometry.style", src / "style.py")
    load_module_from_path("ost_photometry.terminal_output", src / "terminal_output.py")
    return load_module_from_path(
        "ost_photometry.reduce.frame_selection",
        src / "reduce" / "frame_selection.py",
    )


def _rows(
    fwhm: list[float],
    *,
    filt: str = "V",
    n_stars: list[int] | None = None,
    prefix: str = "img",
    **extra,
):
    n = len(fwhm)
    n_stars = n_stars or [50] * n
    rows = []
    for i in range(n):
        row = {
            "file": f"{prefix}_{filt}_{i:03d}.fit",
            "filter": filt,
            "jd": 2460000.0 + i * 0.001,
            "fwhm_px": fwhm[i],
            "n_stars": n_stars[i],
            "status": "ok" if np.isfinite(fwhm[i]) else "no_stars",
            "background": 10.0,
            "background_rms": 1.0,
            "roundness": 0.1,
        }
        for key, values in extra.items():
            row[key] = values[i]
        rows.append(row)
    return rows


def test_from_mapping_and_is_active():
    fs = _mod()
    default = fs.FrameSelection.from_mapping(None)
    assert not default.is_active()
    assert default.describe() == "no frame selection"
    sel = fs.FrameSelection.from_mapping({"fwhm_max": 4.0, "best_fraction": 0.8})
    assert sel.is_active()
    assert "fwhm <= 4" in sel.describe()
    assert "best 80%" in sel.describe()
    same = fs.FrameSelection.from_mapping(sel)
    assert same is sel
    with pytest.raises(ValueError, match="Unknown frame_selection keys"):
        fs.FrameSelection.from_mapping({"fwhm_maximum": 3})
    with pytest.raises(ValueError, match="best_fraction"):
        fs.FrameSelection(best_fraction=1.5)
    with pytest.raises(ValueError, match="fwhm_unit"):
        fs.FrameSelection(fwhm_unit="deg")
    with pytest.raises(ValueError, match="min_frames"):
        fs.FrameSelection(min_frames=0)


def test_quality_table_from_rows_sorts_and_fills_defaults():
    fs = _mod()
    rows = _rows([3.0, 2.5], filt="B") + _rows([4.0], filt="V")
    rows[0]["jd"] = 2460000.5  # B frame observed later than the second B frame
    table = fs.quality_table_from_rows(rows)
    assert list(table["filter"]) == ["B", "B", "V"]
    assert table["jd"][0] < table["jd"][1]
    assert list(table["reject_reason"]) == ["", "", ""]
    assert table["stack_weight"].dtype.kind == "f"
    assert np.all(table["stack_weight"] == 1.0)
    assert table["rejected"].dtype == np.bool_
    assert len(fs.empty_quality_table()) == 0


def test_fwhm_weighted_penalises_frames_with_fewer_stars():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([3.0, 3.0, np.nan], n_stars=[100, 50, 10]))
    weighted = np.asarray(table["fwhm_weighted"])
    assert weighted[0] == pytest.approx(3.0)
    assert weighted[1] == pytest.approx(6.0)
    assert np.isnan(weighted[2])


def test_fwhm_max_in_pixels_and_arcsec():
    fs = _mod()
    table = fs.quality_table_from_rows(
        _rows([2.5, 3.5, 4.5], fwhm_arcsec=[1.0, 1.4, 1.8])
    )
    keep, reasons = fs.select_frames(table, fs.FrameSelection(fwhm_max=4.0))
    assert keep.tolist() == [True, True, False]
    assert reasons[2].startswith("fwhm 4.50 > 4.00 px")
    keep, reasons = fs.select_frames(
        table, fs.FrameSelection(fwhm_max=1.2, fwhm_unit="arcsec")
    )
    assert keep.tolist() == [True, False, False]
    assert "arcsec" in reasons[1]


def test_fwhm_max_arcsec_without_pixel_scale_is_not_applied():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([2.5, 3.5, 4.5]))
    keep, reasons = fs.select_frames(
        table, fs.FrameSelection(fwhm_max=1.0, fwhm_unit="arcsec")
    )
    assert keep.all()
    assert reasons == ["", "", ""]


def test_best_fraction_keeps_ceil_per_filter():
    fs = _mod()
    rows = _rows([3.0, 2.0, 4.0, 2.5, 3.5], filt="V") + _rows([5.0, 4.0, 6.0], filt="B")
    table = fs.quality_table_from_rows(rows)
    keep, reasons = fs.select_frames(table, fs.FrameSelection(best_fraction=0.5))
    v = np.asarray(table["filter"]) == "V"
    assert np.count_nonzero(keep & v) == 3  # ceil(0.5 * 5)
    assert np.count_nonzero(keep & ~v) == 2  # ceil(0.5 * 3)
    kept_v = np.asarray(table["fwhm_px"])[keep & v]
    assert sorted(kept_v.tolist()) == [2.0, 2.5, 3.0]
    assert all("best 50%" in r for r, k in zip(reasons, keep, strict=True) if not k)


def test_rank_by_weighted_fwhm():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([3.0, 3.1], n_stars=[10, 100]))
    keep_px, _ = fs.select_frames(table, fs.FrameSelection(best_fraction=0.5))
    keep_w, _ = fs.select_frames(
        table, fs.FrameSelection(best_fraction=0.5, rank_by="fwhm_weighted")
    )
    assert keep_px.tolist() == [True, False]
    assert keep_w.tolist() == [False, True]


def test_sigma_clip_rejects_outlier_only():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([3.0, 3.1, 2.9, 3.05, 9.0]))
    keep, reasons = fs.select_frames(table, fs.FrameSelection(fwhm_sigma_clip=3.0))
    assert keep.tolist() == [True, True, True, True, False]
    assert "MAD" in reasons[4]


def test_threshold_criteria_and_reasons_are_joined():
    fs = _mod()
    table = fs.quality_table_from_rows(
        _rows(
            [3.0, 3.0, 3.0, 3.0],
            n_stars=[50, 5, 50, 50],
            roundness=[0.1, 0.1, 0.6, 0.1],
            background=[10.0, 10.0, 10.0, 100.0],
            masked_fraction=[0.01, 0.01, 0.01, 0.5],
        )
    )
    sel = fs.FrameSelection(
        n_stars_min=10, roundness_max=0.3, background_max=50.0, masked_fraction_max=0.2
    )
    keep, reasons = fs.select_frames(table, sel)
    assert keep.tolist() == [True, False, False, False]
    assert reasons[1] == "n_stars 5 < 10"
    assert reasons[2].startswith("roundness 0.60 > 0.30")
    assert "background" in reasons[3] and "masked_fraction" in reasons[3]
    assert "; " in reasons[3]


def test_reject_no_stars_only_when_active():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([3.0, np.nan]))
    keep, _ = fs.select_frames(table, fs.FrameSelection())
    assert keep.tolist() == [True, True]
    keep, reasons = fs.select_frames(table, fs.FrameSelection(fwhm_max=10.0))
    assert keep.tolist() == [True, False]
    assert reasons[1] == "no_stars"
    keep, _ = fs.select_frames(
        table, fs.FrameSelection(fwhm_max=10.0, reject_no_stars=False)
    )
    assert keep.tolist() == [True, True]


def test_min_frames_floor_restores_best_rejected():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([2.0, 5.0, 6.0, 7.0]))
    keep, reasons = fs.select_frames(
        table, fs.FrameSelection(fwhm_max=3.0, min_frames=3)
    )
    assert keep.tolist() == [True, True, True, False]
    assert reasons[1] == "" and reasons[2] == ""
    assert reasons[3].startswith("fwhm 7.00")


def test_filters_are_independent():
    fs = _mod()
    rows = _rows([2.0, 2.1], filt="B") + _rows([6.0, 6.1], filt="V")
    table = fs.quality_table_from_rows(rows)
    keep, _ = fs.select_frames(table, fs.FrameSelection(fwhm_sigma_clip=2.0))
    assert keep.all()
    # V fails the cut entirely; min_frames=1 restores its sharpest frame (6.0).
    keep, _ = fs.select_frames(table, fs.FrameSelection(fwhm_max=3.0, min_frames=1))
    assert keep.tolist() == [True, True, True, False]


def test_rank_frames_nan_last_and_tie_break_on_n_stars():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([3.0, np.nan, 3.0, 2.0], n_stars=[10, 50, 40, 5]))
    order = fs.rank_frames(table, np.arange(4))
    files = [str(f) for f in table["file"][order]]
    assert files[0].endswith("003.fit")
    assert files[1].endswith("002.fit")
    assert files[2].endswith("000.fit")
    assert files[3].endswith("001.fit")


def test_resolve_reference_per_filter_and_global():
    fs = _mod()
    rows = _rows([3.0, 2.2], filt="B") + _rows([4.0, np.nan], filt="V")
    table = fs.quality_table_from_rows(rows)
    table["rejected"] = np.array([False, True, False, False])
    refs = fs.resolve_reference_frames(table, per_filter=True)
    assert refs == {"B": "img_B_000.fit", "V": "img_V_000.fit"}
    table["rejected"][:] = False
    refs = fs.resolve_reference_frames(table, per_filter=False)
    assert refs == {fs.GLOBAL_REFERENCE_KEY: "img_B_001.fit"}
    fs.mark_reference_frames(table, refs)
    assert np.asarray(table["is_reference"]).tolist() == [False, True, False, False]
    only_nan = fs.quality_table_from_rows(_rows([np.nan, np.nan]))
    assert fs.resolve_reference_frames(only_nan) == {}


def test_stack_weights_formulas_and_normalisation():
    fs = _mod()
    table = fs.quality_table_from_rows(
        _rows([2.0, 4.0, 4.0], n_stars=[100, 50, 50], background_rms=[1.0, 2.0, 2.0])
    )
    assert np.allclose(fs.stack_weights(table, "none"), 1.0)

    w = fs.stack_weights(table, "fwhm")
    assert w.mean() == pytest.approx(1.0)
    assert w[0] / w[1] == pytest.approx(4.0)  # (4/2)^2 vs (4/4)^2
    assert w[1] == pytest.approx(w[2])

    w = fs.stack_weights(table, "n_stars")
    assert w[0] / w[1] == pytest.approx(2.0)
    assert w.mean() == pytest.approx(1.0)

    w = fs.stack_weights(table, "noise")
    assert w[0] / w[1] == pytest.approx(4.0)

    with pytest.raises(ValueError, match="stack_weighting"):
        fs.stack_weights(table, "seeing")


def test_stack_weights_clip_nan_and_rejected():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([1.0, 10.0, 10.0, np.nan, 10.0]))
    table["rejected"] = np.array([False, False, False, False, True])
    w = fs.stack_weights(table, "fwhm")
    assert np.isnan(w[4])
    kept = w[:4]
    assert kept.mean() == pytest.approx(1.0)
    assert kept[0] / kept[1] <= fs.WEIGHT_CLIP[1] / fs.WEIGHT_CLIP[0] + 1e-9
    assert kept[3] == pytest.approx(kept[1])  # nan metric -> median weight


class _FakeAlignment:
    def __init__(self, aligned, skipped):
        self._aligned = aligned
        self._skipped = skipped

    def aligned_files(self):
        return list(self._aligned)

    def skipped_files(self):
        return list(self._skipped)


def test_merge_alignment_result_sets_flags_and_notes():
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([3.0, 3.0, 3.0, 3.0]))
    table["rejected"] = np.array([False, False, False, True])
    files = [str(f) for f in table["file"]]
    result = _FakeAlignment(
        aligned=[f"/tmp/x/{files[0]}"],
        skipped=[(files[1], "MaxIterError: no match")],
    )
    fs.merge_alignment_result(table, result)
    assert np.asarray(table["aligned"]).tolist() == [True, False, False, False]
    assert list(table["align_note"]) == ["", "MaxIterError: no match", "not processed", ""]


def test_mark_selection_and_ecsv_roundtrip(tmp_path):
    fs = _mod()
    table = fs.quality_table_from_rows(
        _rows([2.0, 5.0, np.nan], fwhm_arcsec=[1.0, 2.5, np.nan])
    )
    fs.mark_selection(table, fs.FrameSelection(fwhm_max=3.0))
    fs.mark_reference_frames(table, fs.resolve_reference_frames(table))
    table["stack_weight"] = fs.stack_weights(table, "fwhm")
    path = fs.write_quality_table(table, tmp_path / "sub" / "frame_quality.ecsv")
    back = fs.read_quality_table(path)
    assert back.colnames == table.colnames
    assert np.asarray(back["rejected"]).tolist() == [False, True, True]
    assert list(back["reject_reason"]) == ["", "fwhm 5.00 > 3.00 px", "no_stars"]
    assert back["rejected"].dtype == np.bool_
    assert back["is_reference"][0]
    assert np.isnan(back["fwhm_px"][2])
    assert np.isnan(back["stack_weight"][1])


def test_summarize_selection_runs(capsys):
    fs = _mod()
    table = fs.quality_table_from_rows(_rows([2.0, 5.0]))
    fs.mark_selection(table, fs.FrameSelection(fwhm_max=3.0))
    fs.summarize_selection(table)
    fs.summarize_selection(fs.empty_quality_table())
