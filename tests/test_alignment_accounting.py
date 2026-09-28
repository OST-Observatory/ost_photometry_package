"""Tests for alignment bookkeeping (registration.accounting)."""

from __future__ import annotations

import pytest

from helpers import load_module_from_path, pkg_src


def _acc():
    return load_module_from_path(
        "ost_photometry.reduce.registration.accounting",
        pkg_src() / "ost_photometry" / "reduce" / "registration" / "accounting.py",
    )


def test_resolve_reference_index_index_file_and_errors():
    acc = _acc()
    files = ["/x/a.fit", "/x/b.fit", "/x/c.fit"]
    assert acc.resolve_reference_index(files, None, None) == 0
    assert acc.resolve_reference_index(files, 2, None) == 2
    assert acc.resolve_reference_index(files, 0, "c.fit") == 2  # file wins
    assert acc.resolve_reference_index(files, None, "/other/dir/b.fit") == 1
    with pytest.raises(ValueError, match="not among"):
        acc.resolve_reference_index(files, None, "missing.fit")
    with pytest.raises(ValueError, match="out of range"):
        acc.resolve_reference_index(files, 3, None)
    with pytest.raises(ValueError, match="no files"):
        acc.resolve_reference_index([], None, None)


def test_apply_results_and_filter_alignment_from_results():
    acc = _acc()
    ok = acc.apply_ok("/x/a.fit")
    assert ok == ("a.fit", True, "")
    skipped = acc.apply_skipped("/x/b.fit", acc.exception_note(ValueError("boom")))
    assert skipped == ("b.fit", False, "ValueError: boom")

    outcome = acc.FilterAlignment.from_results(
        "filter V",
        "/x/a.fit",
        ["/x/a.fit", "/x/b.fit", "/x/c.fit", "/x/d.fit"],
        [ok, skipped, None],
        pre_skipped=[("/x/d.fit", "shift outlier")],
    )
    assert outcome.n_total == 4
    assert outcome.n_aligned == 1
    assert outcome.aligned == ["a.fit"]
    assert outcome.skipped == [
        ("b.fit", "ValueError: boom"),
        ("d.fit", "shift outlier"),
        ("c.fit", "no result"),
    ]
    line = outcome.summary_line()
    assert line.startswith("Alignment (filter V): 1 of 4 frames aligned (reference a.fit)")
    assert "b.fit (ValueError: boom)" in line


def test_alignment_result_aggregates_groups():
    acc = _acc()
    result = acc.AlignmentResult()
    result.add(
        acc.FilterAlignment.from_results("filter B", "b1.fit", ["b1.fit", "b2.fit"],
                                         [acc.apply_ok("b1.fit"), acc.apply_ok("b2.fit")])
    )
    result.add(
        acc.FilterAlignment.from_results("filter V", "v1.fit", ["v1.fit", "v2.fit"],
                                         [acc.apply_ok("v1.fit"), acc.apply_skipped("v2.fit", "x")])
    )
    assert result.n_total == 4
    assert result.n_aligned == 3
    assert result.aligned_files() == ["b1.fit", "b2.fit", "v1.fit"]
    assert result.skipped_files() == [("v2.fit", "x")]
    assert len(result.summary_lines()) == 2
