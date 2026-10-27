"""Terminal output helpers: styles, report levels, quiet details, bar indentation."""

from __future__ import annotations

import pytest

from ost_photometry import terminal_output as t


@pytest.fixture
def captured(monkeypatch):
    seen: list[tuple[int, str, str]] = []
    original = t.print_to_terminal

    def record(text, indent=1, style_name="BOLD"):
        seen.append((indent, str(text), style_name))
        original(text, indent=indent, style_name=style_name)

    monkeypatch.setattr(t, "print_to_terminal", record)
    return seen


@pytest.mark.parametrize(
    ("text", "style"),
    [
        ("note: no bias frames (dark scaling impossible)", "WARNING"),
        ("INCOMPLETE: 8 of 32 lights not reduced", "WARNING"),
        ("GET https://x: ConnectionError, retry 1/5 in 1 s", "WARNING"),
        ("Done", "OKGREEN"),
        ("Targets:", "OKBLUE"),
        ("Building master MF_x (flat, 30 frames)...", "NORMAL"),
    ],
)
def test_auto_style(text, style):
    assert t.auto_style(text) == style


def test_report_levels_follow_the_leading_spaces(captured):
    t.print_report(["Sessions: 2", "  S1: 199 lights", "    note: no bias frames"], indent=2)
    assert [(i, s) for i, _txt, s in captured] == [(2, "OKBLUE"), (3, "NORMAL"), (4, "WARNING")]


def test_quiet_turns_messages_into_details(monkeypatch):
    levels: list[int] = []
    logger = t.get_logger("terminal")
    monkeypatch.setattr(logger, "log", lambda level, msg, *a, **k: levels.append(level))
    with t.quiet():
        t.print_to_terminal("WCS solution found :)", indent=2, style_name="OKGREEN")
        t.print_to_terminal("WCS solve failed", indent=2, style_name="WARNING")
    t.print_to_terminal("normal again", indent=2)
    assert levels == [10, 30, 20]


def test_progress_bars_start_below_the_last_line():
    t.print_to_terminal("Stack flats...", indent=2)
    assert t.current_indent() == 2
    assert t.progress_prefix() == " " * 9
    bar = t.progress_bar(3, "frames", disable=True)
    bar.close()
