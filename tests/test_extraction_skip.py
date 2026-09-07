"""Skip unusable frames in multi-image extraction instead of aborting the pool."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from astropy.table import Table

from ost_photometry.analyze.extraction import (
    ExtractionSkipError,
    _merge_extraction_results,
    _skip_extraction_result,
    check_epsf_stars,
)


def _photometry_table() -> Table:
    return Table({"flux_fit": [1.0], "flux_err": [0.1]})


def _fake_series(ids: list[int], ref_idx: int = 0, filter_: str = "Clear"):
    images = [
        SimpleNamespace(image_id=i, photometry=None, filter_=filter_) for i in ids
    ]
    return SimpleNamespace(
        filter_=filter_,
        image_list=images,
        reference_image_index=ref_idx,
        reference_image=images[ref_idx],
    )


def test_check_epsf_stars_empty_positions_is_skip_error():
    image = SimpleNamespace(image_id=186, filter_="Clear", positions=None)
    with pytest.raises(ExtractionSkipError, match="No sources identified"):
        check_epsf_stars(image)


def test_skip_result_returns_none_table_in_multiprocessing():
    image = SimpleNamespace(image_id=186, filter_="Clear")
    err = ExtractionSkipError("Not enough stars (12) for the ePSF determination")
    result = _skip_extraction_result(image, err, terminal_logger=None, multiprocessing=True)
    assert result == (186, None)


def test_skip_result_reraises_without_multiprocessing():
    image = SimpleNamespace(image_id=186, filter_="Clear")
    err = ExtractionSkipError("Not enough stars (12) for the ePSF determination")
    with pytest.raises(ExtractionSkipError, match="Not enough stars"):
        _skip_extraction_result(image, err, terminal_logger=None, multiprocessing=False)


def test_merge_drops_skipped_frame_and_keeps_photometry():
    series = _fake_series([0, 1, 186], ref_idx=0)
    tbl0 = _photometry_table()
    tbl1 = _photometry_table()
    skipped = _merge_extraction_results(
        series,
        [(0, tbl0), (186, None), (1, tbl1)],
    )
    assert skipped == [186]
    assert [img.image_id for img in series.image_list] == [0, 1]
    assert series.image_list[0].photometry is tbl0
    assert series.reference_image_index == 0
    assert series.reference_image.image_id == 0


def test_merge_remaps_reference_when_reference_skipped():
    series = _fake_series([0, 1, 2], ref_idx=0)
    tbl1 = _photometry_table()
    tbl2 = _photometry_table()
    skipped = _merge_extraction_results(
        series,
        [(0, None), (1, tbl1), (2, tbl2)],
    )
    assert skipped == [0]
    assert [img.image_id for img in series.image_list] == [1, 2]
    assert series.reference_image_index == 0
    assert series.reference_image.image_id == 1


def test_merge_raises_when_all_frames_skipped():
    series = _fake_series([0, 1], ref_idx=0)
    with pytest.raises(RuntimeError, match="Extraction failed for all"):
        _merge_extraction_results(series, [(0, None), (1, None)])
