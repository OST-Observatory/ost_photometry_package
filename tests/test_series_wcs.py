"""Series WCS: per-image solve without broadcasting onto every frame."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from ost_photometry.analyze.utils import series_wcs


def test_find_wcs_solve_all_does_not_broadcast(monkeypatch):
    solved: list = []

    def fake_find(img, **_kw):
        w = MagicMock(name=f"wcs_{len(solved)}")
        solved.append(w)
        return w

    monkeypatch.setattr(series_wcs.wcs_utilities, "find_wcs_for_image", fake_find)
    monkeypatch.setattr(
        series_wcs.wcs_utilities,
        "sync_image_coordinates_from_wcs",
        lambda *_a, **_k: None,
    )

    img0 = SimpleNamespace(wcs=None)
    img1 = SimpleNamespace(wcs=None)
    series = SimpleNamespace(
        image_list=[img0, img1],
        reference_image_index=0,
        wcs=None,
    )
    broadcasts = []

    def set_wcs(w, *, broadcast=True):
        series.wcs = w
        broadcasts.append(broadcast)

    series.set_wcs = set_wcs
    series_wcs.find_wcs(series, reference_image_index=0, solve_all_images=True)
    assert len(solved) == 2
    assert img0.wcs is solved[0]
    assert img1.wcs is solved[1]
    assert broadcasts == [False]
    assert series.wcs is solved[0]


def test_ensure_image_wcs_copies_series_wcs_when_aligned(monkeypatch):
    def _should_not_solve(*_a, **_k):
        raise AssertionError("should not solve")

    monkeypatch.setattr(
        series_wcs.wcs_utilities, "find_wcs_for_image", _should_not_solve
    )
    synced = []
    monkeypatch.setattr(
        series_wcs.wcs_utilities,
        "sync_image_coordinates_from_wcs",
        lambda img, w: synced.append((img, w)),
    )
    series_w = MagicMock(name="series_wcs")
    img = SimpleNamespace(wcs=None)
    out = series_wcs.ensure_image_wcs(img, series_wcs=series_w, aligned_grid=True)
    assert out is series_w
    assert img.wcs is series_w
    assert synced == [(img, series_w)]
