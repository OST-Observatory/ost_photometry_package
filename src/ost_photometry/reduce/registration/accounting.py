"""Alignment bookkeeping: which frames were aligned, which were skipped, and why.

Kept free of ``ccdproc`` / ``astroalign`` imports so the accounting can be
unit-tested and reused by scripts without the registration backends.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: ``(basename, success, note)`` returned by every ``apply_*`` worker.
ApplyResult = tuple[str, bool, str]

_NOTE_LIMIT = 80


def apply_ok(file_name: str | Path) -> ApplyResult:
    """Worker result for a successfully aligned frame."""
    return (Path(file_name).name, True, "")


def apply_skipped(file_name: str | Path, note: str) -> ApplyResult:
    """Worker result for a frame that could not be aligned."""
    return (Path(file_name).name, False, str(note)[:_NOTE_LIMIT])


def exception_note(exc: BaseException) -> str:
    """Short ``Type: message`` note for a skipped frame."""
    return f"{type(exc).__name__}: {exc}"[:_NOTE_LIMIT]


def resolve_reference_index(
    files: Sequence[str],
    index: int | None,
    file_name: str | None,
) -> int:
    """Index of the reference frame in ``files``.

    A ``file_name`` (matched by basename) takes precedence over ``index``;
    ``index=None`` means the first frame. Raises ``ValueError`` when the
    file is not in the collection or the index is out of range.
    """
    if not files:
        raise ValueError("Cannot resolve a reference frame: no files.")
    if file_name is not None:
        wanted = Path(str(file_name)).name
        for i, name in enumerate(files):
            if Path(str(name)).name == wanted:
                return i
        raise ValueError(
            f"Reference frame {wanted!r} is not among the {len(files)} frames to align."
        )
    idx = 0 if index is None else int(index)
    if idx < 0 or idx >= len(files):
        raise ValueError(
            f"reference_image_index {index} is out of range for {len(files)} images"
        )
    return idx


@dataclass
class FilterAlignment:
    """Outcome of aligning one group of frames (one filter, or all frames)."""

    label: str
    reference_file: str
    n_total: int
    aligned: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_aligned(self) -> int:
        return len(self.aligned)

    @classmethod
    def from_results(
        cls,
        label: str,
        reference_file: str | Path,
        files: Iterable[str | Path],
        results: Iterable[ApplyResult | None],
        *,
        pre_skipped: Iterable[tuple[str | Path, str]] = (),
    ) -> FilterAlignment:
        """Build from worker results; frames without a result count as skipped."""
        names = [Path(str(f)).name for f in files]
        outcome = cls(label=str(label), reference_file=Path(str(reference_file)).name,
                      n_total=len(names))
        seen: set[str] = set()
        for entry in results:
            if not entry:
                continue
            name, success, note = entry
            name = Path(str(name)).name
            seen.add(name)
            if success:
                outcome.aligned.append(name)
            else:
                outcome.skipped.append((name, str(note)))
        for name, note in pre_skipped:
            name = Path(str(name)).name
            if name not in seen:
                seen.add(name)
                outcome.skipped.append((name, str(note)))
        for name in names:
            if name not in seen:
                outcome.skipped.append((name, "no result"))
        return outcome

    def summary_line(self, *, max_listed: int = 6) -> str:
        text = (
            f"Alignment ({self.label}): {self.n_aligned} of {self.n_total} frames aligned"
            f" (reference {self.reference_file})"
        )
        if self.skipped:
            listed = ", ".join(
                f"{name} ({note})" if note else name for name, note in self.skipped[:max_listed]
            )
            more = f", … +{len(self.skipped) - max_listed}" if len(self.skipped) > max_listed else ""
            text += f"; skipped: {listed}{more}"
        return text


@dataclass
class AlignmentResult:
    """Alignment outcome for every processed group."""

    per_group: dict[str, FilterAlignment] = field(default_factory=dict)

    def add(self, outcome: FilterAlignment) -> None:
        self.per_group[outcome.label] = outcome

    @property
    def n_total(self) -> int:
        return sum(g.n_total for g in self.per_group.values())

    @property
    def n_aligned(self) -> int:
        return sum(g.n_aligned for g in self.per_group.values())

    def aligned_files(self) -> list[str]:
        return [name for g in self.per_group.values() for name in g.aligned]

    def skipped_files(self) -> list[tuple[str, str]]:
        return [entry for g in self.per_group.values() for entry in g.skipped]

    def summary_lines(self) -> list[str]:
        return [g.summary_line() for g in self.per_group.values()]


__all__ = [
    "AlignmentResult",
    "ApplyResult",
    "FilterAlignment",
    "apply_ok",
    "apply_skipped",
    "exception_note",
    "resolve_reference_index",
]
