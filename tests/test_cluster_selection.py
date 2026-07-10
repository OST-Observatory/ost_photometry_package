"""Tests for cluster selection input parsing."""


def _parse_cluster_selection_id(raw: str | None) -> int | None:
    """Mirror of ost_photometry.utilities.parse_cluster_selection_id."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def test_parse_cluster_selection_id_strips_control_characters():
    assert _parse_cluster_selection_id("\x180") == 0
    assert _parse_cluster_selection_id("  2 \n") == 2
    assert _parse_cluster_selection_id("") is None
    assert _parse_cluster_selection_id("abc") is None
