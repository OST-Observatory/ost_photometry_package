"""Find the stacks of a target: astro_imaging output or archive summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits
from astropy.table import Table


@dataclass
class StackSet:
    """The per-filter stacks of one colour image."""

    target: str
    camera: str
    stacks: dict[str, str] = field(default_factory=dict)


def stacks_from_directory(directory: str | Path) -> dict[str, str]:
    """``{FILTER: path}`` of the stacks in an astro_imaging output directory.

    Uses ``combined_trimmed_filter_*.fit`` (filters aligned afterwards)
    when present, else ``combined_filter_*.fit``; the filter comes from the
    header. Subdirectories (restack, diagnostics) are not searched.
    """
    directory = Path(directory)
    paths = sorted(directory.glob("combined_trimmed_filter_*.fit")) or sorted(
        directory.glob("combined_filter_*.fit"))
    out: dict[str, str] = {}
    for path in paths:
        name = str(fits.getheader(path).get("FILTER", "")).strip()
        out[name or path.stem.split("filter_", 1)[-1]] = str(path)
    return out


def stacks_from_summary(summary: Table, camera: str = "auto") -> list[StackSet]:
    """Stack sets per target from an archive ``summary.ecsv``.

    ``camera="auto"``: the camera combination when it exists, else every
    camera on its own; ``"combined"`` or a camera id: only that one.
    """
    groups: dict[tuple[str, str], StackSet] = {}
    by_target: dict[str, set[str]] = {}
    for row in summary:
        by_target.setdefault(str(row["target_name"]), set()).add(str(row["camera"]))
    for row in summary:
        target, cam = str(row["target_name"]), str(row["camera"])
        cams = by_target[target]
        if camera == "auto":
            wanted = {"combined"} if "combined" in cams else cams
        else:
            wanted = {camera}
        if cam not in wanted:
            continue
        entry = groups.setdefault((target, cam), StackSet(target, cam))
        entry.stacks[str(row["filter"])] = str(row["path"])
    return list(groups.values())


__all__ = ["StackSet", "stacks_from_directory", "stacks_from_summary"]
