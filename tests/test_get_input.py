"""Timed yes/no prompts for reduce (reuse masters / previous science frames)."""

from __future__ import annotations

import os
import sys
import time

from ost_photometry.utilities import get_input


def test_get_input_times_out_when_stdin_stays_idle(monkeypatch):
    read_fd, write_fd = os.pipe()
    stdin = open(read_fd)  # noqa: SIM115 — kept open for the duration of the test
    monkeypatch.setattr(sys, "stdin", stdin)
    start = time.monotonic()
    text, timed_out = get_input("reuse masters? ", timeout=0.25)
    elapsed = time.monotonic() - start
    os.close(write_fd)
    stdin.close()
    assert timed_out is True
    assert text == "no"
    assert 0.15 <= elapsed < 2.0


def test_get_input_reads_yes_from_a_pipe(monkeypatch):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"YES\n")
    os.close(write_fd)
    stdin = open(read_fd)  # noqa: SIM115
    monkeypatch.setattr(sys, "stdin", stdin)
    text, timed_out = get_input("reuse masters? ", timeout=2)
    stdin.close()
    assert timed_out is False
    assert text == "yes"
