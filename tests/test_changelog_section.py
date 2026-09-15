"""CHANGELOG section extractor used by the GitHub Release workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "changelog_section.py"


def _load():
    spec = importlib.util.spec_from_file_location("changelog_section", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_SAMPLE = """# Changelog

## [Unreleased]

- pending

## [0.4.5] - 2026-09-14

### Added

- track QC

## [0.4.4] - 2026-07-07

baseline

[Unreleased]: https://example.test/compare/v0.4.5...HEAD
"""


def test_changelog_section_strips_heading_and_footer_links():
    body = _load().changelog_section(_SAMPLE, "0.4.5")
    assert body.startswith("### Added")
    assert "track QC" in body
    assert "0.4.4" not in body
    assert "Unreleased" not in body
    assert body.endswith("\n")


def test_changelog_section_missing_version():
    with pytest.raises(KeyError, match="no CHANGELOG section"):
        _load().changelog_section(_SAMPLE, "0.9.9")


def test_main_accepts_v_prefix(tmp_path: Path, capsys):
    mod = _load()
    path = tmp_path / "CHANGELOG.md"
    path.write_text(_SAMPLE, encoding="utf-8")
    assert mod.main(["v0.4.5", "--file", str(path)]) == 0
    assert "track QC" in capsys.readouterr().out
    assert mod.main(["0.9.9", "--file", str(path)]) == 1
