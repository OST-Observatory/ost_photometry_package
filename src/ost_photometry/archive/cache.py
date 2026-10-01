"""Content-addressed file cache and per-group symlink farms."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._+-]+")


def safe_name(text: str, *, max_length: int = 120) -> str:
    """File-system friendly version of ``text`` (no separators or spaces)."""
    cleaned = _SAFE.sub("_", str(text).strip()).strip("._")
    return (cleaned or "unnamed")[:max_length]


class FileCache:
    """Files stored as ``<root>/<sha[:2]>/<sha><suffix>``.

    Cached files are made read-only so that no pipeline step (for example a
    plate solver that writes WCS keywords) can modify a raw frame.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, sha256: str, suffix: str = ".fit") -> Path:
        sha = str(sha256).lower()
        if len(sha) < 8:
            raise ValueError(f"Not a SHA-256 checksum: {sha256!r}")
        return self.root / sha[:2] / f"{sha}{suffix}"

    def has(self, sha256: str, suffix: str = ".fit") -> bool:
        return self.path_for(sha256, suffix).is_file()

    def protect(self, path: str | Path) -> None:
        path = Path(path)
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def frame_link_name(frame_id: str, file_name: str) -> str:
    """Unique link name: two archive files may share a basename."""
    return f"{safe_name(frame_id, max_length=40)}_{safe_name(file_name)}"


def link_frames(
    rows: Iterable[Mapping[str, object]],
    dest_dir: str | Path,
    *,
    path_column: str = "local_path",
) -> dict[str, Path]:
    """Symlink the frames of ``rows`` into ``dest_dir``.

    Returns ``{frame_id: link_path}``. Existing links with the same name are
    replaced; rows without a local file are skipped.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    links: dict[str, Path] = {}
    for row in rows:
        source = str(row.get(path_column, "") or "")
        if not source or not Path(source).is_file():
            continue
        frame_id = str(row.get("frame_id", ""))
        link = dest / frame_link_name(frame_id, str(row.get("file_name") or Path(source).name))
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.abspath(source), link)
        links[frame_id] = link
    return links


__all__ = ["FileCache", "frame_link_name", "link_frames", "safe_name"]
