"""Input validation helpers for the data reduction workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .. import style, terminal_output

IssueSeverity = Literal["error", "warning"]
IssueCode = Literal[
    "missing_flat_for_science",
    "all_zero_science",
    "non_finite_science",
    "empty_science",
]


@dataclass(frozen=True)
class ValidationIssue:
  """One validation finding from :func:`validate_raw_collection`."""

  code: IssueCode | str
  message: str
  severity: IssueSeverity = "error"
  file_name: str | None = None


def collect_filters_for_image_class(
    image_file_collection,
    image_type_dict: dict[str, list[str]],
    image_class: str,
) -> set[str]:
    """Return deduplicated ``FILTER`` values for one image class (e.g. light)."""
    type_list = image_type_dict.get(image_class, [])
    if not type_list:
        return set()
    imagetyp = np.asarray(image_file_collection.summary["imagetyp"])
    mask = np.isin(imagetyp, type_list)
    if not np.any(mask):
        return set()
    filters = image_file_collection.summary["filter"][mask]
    out: set[str] = set()
    for filt in filters:
        if filt is np.ma.masked or filt in (None, ""):
            continue
        out.add(str(filt))
    return out


def collect_science_filters(
    image_file_collection,
    image_type_dict: dict[str, list[str]],
) -> set[str]:
    """Filters used by science (light) frames."""
    return collect_filters_for_image_class(
        image_file_collection, image_type_dict, "light"
    )


def collect_flat_filters(
    image_file_collection,
    image_type_dict: dict[str, list[str]],
) -> set[str]:
    """Filters used by raw flat frames."""
    return collect_filters_for_image_class(
        image_file_collection, image_type_dict, "flat"
    )


def check_science_flat_coverage(
    science_filters: set[str],
    flat_filters: set[str],
) -> list[str]:
    """Return science filters that have no matching raw flat."""
    return sorted(science_filters - flat_filters)


def check_frame_sanity(data: np.ndarray) -> str | None:
    """
    Return ``None`` if frame data looks usable, else a short reason code.

    Reason codes: ``empty``, ``all_zero``, ``non_finite``.
    """
    arr = np.asarray(data)
    if arr.size == 0:
        return "empty"
    if not np.any(np.isfinite(arr)):
        return "non_finite"
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "non_finite"
    if np.max(np.abs(finite)) == 0:
        return "all_zero"
    return None


def _sample_file_paths(
    image_file_collection,
    image_type_dict: dict[str, list[str]],
    image_class: str,
    sample_size: int,
) -> list[str]:
    from .image_types import get_image_type

    image_type = get_image_type(
        image_file_collection, image_type_dict, image_class=image_class
    )
    if not image_type:
        return []
    paths = list(
        image_file_collection.files_filtered(
            imagetyp=image_type,
            include_path=True,
        )
    )
    if not paths:
        return []
    step = max(1, len(paths) // max(sample_size, 1))
    return paths[::step][:sample_size]


def validate_raw_collection(
    image_file_collection,
    image_type_dict: dict[str, list[str]],
    *,
    sanity_check_sample_size: int = 3,
) -> list[ValidationIssue]:
    """Aggregate pre-reduction validation issues."""
    from astropy.nddata import CCDData

    issues: list[ValidationIssue] = []

    science_filters = collect_science_filters(image_file_collection, image_type_dict)
    flat_filters = collect_flat_filters(image_file_collection, image_type_dict)
    missing = check_science_flat_coverage(science_filters, flat_filters)
    for filt in missing:
        issues.append(
            ValidationIssue(
                code="missing_flat_for_science",
                message=(
                    f"Science filter '{filt}' has no flat frame in the input directory."
                ),
                severity="error",
            )
        )

    sample_paths = _sample_file_paths(
        image_file_collection,
        image_type_dict,
        "light",
        sanity_check_sample_size,
    )
    for path in sample_paths:
        try:
            ccd = CCDData.read(path, unit="adu")
            reason = check_frame_sanity(np.asarray(ccd.data))
        except (OSError, ValueError, TypeError) as exc:
            issues.append(
                ValidationIssue(
                    code="read_failed",
                    message=f"Could not read science frame for sanity check: {exc}",
                    severity="warning",
                    file_name=str(path),
                )
            )
            continue
        if reason is None:
            continue
        code_map = {
            "all_zero": "all_zero_science",
            "non_finite": "non_finite_science",
            "empty": "empty_science",
        }
        issues.append(
            ValidationIssue(
                code=code_map.get(reason, reason),
                message=f"Science frame failed sanity check ({reason}): {path}",
                severity="error",
                file_name=str(path),
            )
        )

    return issues


def emit_validation_warnings(issues: list[ValidationIssue]) -> None:
    """Print warning-severity validation issues."""
    for issue in issues:
        if issue.severity != "warning":
            continue
        terminal_output.print_to_terminal(issue.message, style_name="WARNING", indent=2)


def raise_on_fatal_validation_issues(
    issues: list[ValidationIssue],
    *,
    fail_on_missing_flat: bool = True,
) -> None:
    """Raise :class:`RuntimeError` on error-severity issues."""
    fatal = []
    for issue in issues:
        if issue.severity != "error":
            continue
        if issue.code == "missing_flat_for_science" and not fail_on_missing_flat:
            continue
        fatal.append(issue)
    if not fatal:
        return
    lines = [issue.message for issue in fatal]
    raise RuntimeError(
        f"{style.Bcolors.FAIL}Input validation failed:\n"
        + "\n".join(f"  - {line}" for line in lines)
        + f"{style.Bcolors.ENDC}"
    )


def summarize_light_reduction_results(
    results: list[str | None],
    *,
    fail_on_missing_flat: bool = True,
) -> None:
    """Print science reduction summary and optionally raise if all were skipped."""
    counts: dict[str, int] = {}
    for status in results:
        key = status or "unknown"
        counts[key] = counts.get(key, 0) + 1
    reduced = counts.get("reduced", 0)
    skipped = sum(v for k, v in counts.items() if k.startswith("skip_"))
    terminal_output.print_to_terminal(
        f"Science reduction summary: {reduced} reduced, {skipped} skipped",
        indent=2,
    )
    for key, n in sorted(counts.items()):
        if key == "reduced":
            continue
        terminal_output.print_to_terminal(f"  {key}: {n}", indent=3)
    if reduced == 0 and skipped > 0 and fail_on_missing_flat:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}All science images were skipped during reduction."
            f"{style.Bcolors.ENDC}"
        )


__all__ = [
    "ValidationIssue",
    "check_frame_sanity",
    "check_science_flat_coverage",
    "collect_filters_for_image_class",
    "collect_flat_filters",
    "collect_science_filters",
    "emit_validation_warnings",
    "raise_on_fatal_validation_issues",
    "summarize_light_reduction_results",
    "validate_raw_collection",
]
