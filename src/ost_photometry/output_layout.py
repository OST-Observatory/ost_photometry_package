"""Canonical analysis output layout (plots, tables, scratch).

``<output>/``

- ``diagnostics/<step>/`` — QC (extraction, correlation, calibration, extinction, cluster, cmds, lightcurves)
- ``results/<kind>/`` — science figures (lightcurves, cmds, starmaps)
- ``tables/`` — ECSV / ASCII
- ``work/<kind>/`` — scratch (wcs_images, extraction galleries, subtract)
"""

from __future__ import annotations

from pathlib import Path

DIAGNOSTICS_ROOT = "diagnostics"
RESULTS_ROOT = "results"
WORK_ROOT = "work"
TABLES_ROOT = "tables"

DIAGNOSTIC_STEPS = frozenset(
    {
        "extraction",
        "correlation",
        "calibration",
        "extinction",
        "cluster",
        "cmds",
        "lightcurves",
    }
)
RESULT_KINDS = frozenset({"lightcurves", "cmds", "starmaps"})
WORK_KINDS = frozenset({"wcs_images", "extraction", "subtract"})


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _root(output_dir: str | Path) -> Path:
    return Path(output_dir)


def diagnostics_dir(output_dir: str | Path, step: str) -> Path:
    """QC directory for one pipeline step (created if missing)."""
    if step not in DIAGNOSTIC_STEPS:
        raise ValueError(
            f"Unknown diagnostic step {step!r}; expected one of {sorted(DIAGNOSTIC_STEPS)}"
        )
    return _ensure(_root(output_dir) / DIAGNOSTICS_ROOT / step)


def results_dir(output_dir: str | Path, kind: str, *parts: str) -> Path:
    """Science-figure directory (created if missing)."""
    if kind not in RESULT_KINDS:
        raise ValueError(
            f"Unknown result kind {kind!r}; expected one of {sorted(RESULT_KINDS)}"
        )
    path = _root(output_dir) / RESULTS_ROOT / kind
    for part in parts:
        cleaned = str(part).strip("/")
        if cleaned:
            path = path / cleaned
    return _ensure(path)


def work_dir(output_dir: str | Path, kind: str) -> Path:
    """Scratch / bulky gallery directory (created if missing)."""
    if kind not in WORK_KINDS:
        raise ValueError(
            f"Unknown work kind {kind!r}; expected one of {sorted(WORK_KINDS)}"
        )
    return _ensure(_root(output_dir) / WORK_ROOT / kind)


def tables_dir(output_dir: str | Path) -> Path:
    """ECSV / ASCII tables (created if missing)."""
    return _ensure(_root(output_dir) / TABLES_ROOT)


def extraction_plot_dir(output_dir: str | Path, *, gallery: bool = False) -> Path:
    """Reference extraction QC vs all-image galleries under ``work/extraction``."""
    if gallery:
        return work_dir(output_dir, "extraction")
    return diagnostics_dir(output_dir, "extraction")


__all__ = [
    "DIAGNOSTIC_STEPS",
    "DIAGNOSTICS_ROOT",
    "RESULT_KINDS",
    "RESULTS_ROOT",
    "TABLES_ROOT",
    "WORK_KINDS",
    "WORK_ROOT",
    "diagnostics_dir",
    "extraction_plot_dir",
    "results_dir",
    "tables_dir",
    "work_dir",
]
