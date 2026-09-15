#!/usr/bin/env python3
"""Print the Keep-a-Changelog section for a version or git tag.

Usage::

    python scripts/changelog_section.py v0.4.5
    python scripts/changelog_section.py 0.4.5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CHANGELOG = _ROOT / "CHANGELOG.md"
_HEADING = re.compile(r"^## \[([^\]]+)\](?:\s+-\s+\S+)?\s*$")


def _version(ref: str) -> str:
    text = ref.strip()
    if text.lower().startswith("v") and text[1:2].isdigit():
        return text[1:]
    return text


def changelog_section(changelog: str, version: str) -> str:
    """Return the body under ``## [version]`` (no heading), or raise ``KeyError``."""
    lines = changelog.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        match = _HEADING.match(line)
        if match and match.group(1) == version:
            start = i + 1
            break
    if start is None:
        raise KeyError(f"no CHANGELOG section for [{version}]")
    end = len(lines)
    for i in range(start, len(lines)):
        if _HEADING.match(lines[i]):
            end = i
            break
        if lines[i].startswith("[") and "]: " in lines[i]:
            end = i
            break
    body = "\n".join(lines[start:end]).strip()
    if not body:
        raise KeyError(f"CHANGELOG section [{version}] is empty")
    return body + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", help="version or git tag, e.g. v0.4.5")
    parser.add_argument(
        "--file",
        type=Path,
        default=_CHANGELOG,
        help="changelog path (default: repo CHANGELOG.md)",
    )
    args = parser.parse_args(argv)
    version = _version(args.ref)
    try:
        text = changelog_section(args.file.read_text(encoding="utf-8"), version)
    except (OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
