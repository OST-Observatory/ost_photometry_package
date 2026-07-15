"""Load, save, merge, and aggregate atmospheric extinction coefficient tables."""

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .extinction import DEFAULT_EXTINCTION, ExtinctionCoefficients
from .warnings_types import OstPhotometryAnalyzeWarning

def bundled_site_extinction_path() -> Path:
    """Path to the packaged OST site extinction table."""
    from importlib.resources import files

    return Path(str(files("ost_photometry.data") / "ost_potsdam_extinction.json"))


def extinction_coefficient_to_dict(ec: ExtinctionCoefficients) -> dict[str, Any]:
    """Serialize one :class:`ExtinctionCoefficients` instance."""
    return {
        "filter_name": ec.filter_name,
        "k_prime": ec.k_prime,
        "k_prime_err": ec.k_prime_err,
        "k_second": ec.k_second,
        "k_second_err": ec.k_second_err,
        "color_filter_1": ec.color_filter_1,
        "color_filter_2": ec.color_filter_2,
        "valid": ec.valid,
    }


def extinction_coefficients_to_dict(
    coeffs: dict[str, ExtinctionCoefficients],
) -> dict[str, dict[str, Any]]:
    """Serialize a filter → coefficient mapping."""
    return {k: extinction_coefficient_to_dict(v) for k, v in coeffs.items()}


def extinction_coefficient_from_dict(
    filter_name: str,
    data: dict[str, Any],
) -> ExtinctionCoefficients:
    """Deserialize one coefficient record."""
    return ExtinctionCoefficients(
        filter_name=str(data.get("filter_name", filter_name)),
        k_prime=float(data["k_prime"]),
        k_prime_err=float(data.get("k_prime_err", 0.0)),
        k_second=float(data.get("k_second", 0.0)),
        k_second_err=float(data.get("k_second_err", 0.0)),
        color_filter_1=str(data.get("color_filter_1", "")),
        color_filter_2=str(data.get("color_filter_2", "")),
        valid=bool(data.get("valid", True)),
    )


def extinction_coefficients_from_dict(
    data: dict[str, Any],
) -> dict[str, ExtinctionCoefficients]:
    """Deserialize a flat or wrapped coefficient mapping."""
    if "coefficients" in data and isinstance(data["coefficients"], dict):
        raw = data["coefficients"]
    else:
        raw = {k: v for k, v in data.items() if isinstance(v, dict) and "k_prime" in v}
    return {
        str(k): extinction_coefficient_from_dict(str(k), v)
        for k, v in raw.items()
    }


def _copy_default_extinction() -> dict[str, ExtinctionCoefficients]:
    out: dict[str, ExtinctionCoefficients] = {}
    for k, v in DEFAULT_EXTINCTION.items():
        out[k] = ExtinctionCoefficients(
            v.filter_name,
            v.k_prime,
            v.k_prime_err,
            v.k_second,
            v.k_second_err,
            v.color_filter_1,
            v.color_filter_2,
            v.valid,
        )
    return out


def merge_extinction_coefficients(
    base: dict[str, ExtinctionCoefficients],
    overlay: dict[str, ExtinctionCoefficients],
    *,
    missing_only: bool = False,
) -> dict[str, ExtinctionCoefficients]:
    """Merge ``overlay`` into ``base`` (shallow copy of base)."""
    merged = deepcopy(base)
    for filt, coeff in overlay.items():
        if missing_only and filt in merged:
            continue
        merged[filt] = deepcopy(coeff)
    return merged


def save_extinction_coefficients(
    path: str | Path,
    coeffs: dict[str, ExtinctionCoefficients],
    *,
    meta: dict[str, Any] | None = None,
    wrapped: bool = True,
) -> None:
    """Write coefficients to JSON (wrapped with ``meta`` by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    if wrapped:
        payload = {
            "meta": meta or {},
            "coefficients": extinction_coefficients_to_dict(coeffs),
        }
    else:
        payload = extinction_coefficients_to_dict(coeffs)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_extinction_coefficients(path: str | Path) -> dict[str, ExtinctionCoefficients]:
    """Load coefficients from JSON (flat or wrapped format)."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return extinction_coefficients_from_dict(data)


def resolve_tabulated_extinction_coefficients(
    path: str | Path | None = None,
    *,
    warn_on_fallback: bool = True,
) -> dict[str, ExtinctionCoefficients]:
    """
    Resolve coefficients for ``extinction_mode="tabulated"``.

    Order: explicit ``path`` → bundled site file → ``DEFAULT_EXTINCTION``.
    Missing filters in a partial table are filled from ``DEFAULT_EXTINCTION``.
    """
    candidates: list[tuple[str, Path | None]] = []
    if path is not None:
        candidates.append(("custom", Path(path)))
    else:
        candidates.append(("bundled", bundled_site_extinction_path()))

    loaded: dict[str, ExtinctionCoefficients] = {}
    for label, candidate in candidates:
        if candidate is None or not candidate.is_file():
            if label == "bundled" and warn_on_fallback:
                warnings.warn(
                    f"Bundled site extinction file not found ({candidate}); "
                    "using DEFAULT_EXTINCTION until site table is installed.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )
            continue
        try:
            loaded = load_extinction_coefficients(candidate)
            break
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Could not load extinction coefficients from {candidate}: {exc}",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )

    if not loaded:
        return _copy_default_extinction()

    merged = merge_extinction_coefficients(_copy_default_extinction(), loaded, missing_only=False)
    return merged


def _aggregate_filter_values(
    values: list[tuple[float, float]],
    *,
    statistic: Literal["median", "weighted_median"] = "median",
    sigma_clip: float | None = 2.5,
) -> tuple[float, float, int]:
    """Return (k_prime, k_prime_err, n_used) from (k, k_err) samples."""
    if not values:
        return float("nan"), float("nan"), 0
    k = np.asarray([v[0] for v in values], dtype=float)
    e = np.asarray([max(v[1], 1e-6) for v in values], dtype=float)
    mask = np.isfinite(k)
    k = k[mask]
    e = e[mask]
    if len(k) == 0:
        return float("nan"), float("nan"), 0
    if sigma_clip is not None and len(k) > 2:
        med = float(np.median(k))
        mad = float(np.median(np.abs(k - med)))
        if mad > 0:
            z = np.abs(k - med) / (1.4826 * mad)
            keep = z <= sigma_clip
            k = k[keep]
            e = e[keep]
    if len(k) == 0:
        return float("nan"), float("nan"), 0
    if statistic == "weighted_median":
        order = np.argsort(k)
        k_sorted = k[order]
        w = 1.0 / e[order] ** 2
        cum = np.cumsum(w) / np.sum(w)
        idx = int(np.searchsorted(cum, 0.5))
        k_out = float(k_sorted[min(idx, len(k_sorted) - 1)])
    else:
        k_out = float(np.median(k))
    k_err = float(np.sqrt(np.mean(e**2)) / max(len(k), 1) ** 0.5)
    return k_out, k_err, int(len(k))


def aggregate_extinction_coefficients(
    paths: list[str | Path],
    *,
    statistic: Literal["median", "weighted_median"] = "median",
    sigma_clip: float | None = 2.5,
    site: str = "OST_Potsdam",
    method: str = "value_airmass",
) -> tuple[dict[str, ExtinctionCoefficients], dict[str, Any]]:
    """
    Aggregate per-night JSON files into one site table.

    Returns ``(coefficients, meta)`` with meta describing contributing nights.
    """
    by_filter: dict[str, list[tuple[float, float]]] = {}
    templates: dict[str, ExtinctionCoefficients] = {}
    night_labels: list[str] = []

    for path in paths:
        path = Path(path)
        night_labels.append(path.name)
        coeffs = load_extinction_coefficients(path)
        for filt, ec in coeffs.items():
            if not ec.valid or not np.isfinite(ec.k_prime):
                continue
            by_filter.setdefault(filt, []).append((ec.k_prime, ec.k_prime_err))
            templates.setdefault(filt, ec)

    result: dict[str, ExtinctionCoefficients] = {}
    per_filter_meta: dict[str, dict[str, Any]] = {}
    for filt, samples in by_filter.items():
        k_out, k_err, n_used = _aggregate_filter_values(
            samples,
            statistic=statistic,
            sigma_clip=sigma_clip,
        )
        if not np.isfinite(k_out):
            continue
        tpl = templates[filt]
        result[filt] = ExtinctionCoefficients(
            filter_name=filt,
            k_prime=k_out,
            k_prime_err=k_err,
            k_second=tpl.k_second,
            k_second_err=tpl.k_second_err,
            color_filter_1=tpl.color_filter_1,
            color_filter_2=tpl.color_filter_2,
            valid=True,
        )
        per_filter_meta[filt] = {
            "n_nights": n_used,
            "k_prime_spread": float(np.std([s[0] for s in samples])) if len(samples) > 1 else 0.0,
        }

    meta = {
        "site": site,
        "updated": date.today().isoformat(),
        "method": method,
        "statistic": statistic,
        "sigma_clip": sigma_clip,
        "n_input_nights": len(paths),
        "input_files": night_labels,
        "filters": sorted(result.keys()),
        "per_filter": per_filter_meta,
    }
    return result, meta


def collect_per_night_extinction_samples(
    paths: list[str | Path],
) -> dict[str, list[tuple[str, float, float]]]:
    """Gather per-night ``(label, k_prime, k_prime_err)`` samples per filter."""
    by_filter: dict[str, list[tuple[str, float, float]]] = {}
    for path in paths:
        path = Path(path)
        coeffs = load_extinction_coefficients(path)
        for filt, ec in coeffs.items():
            if not ec.valid or not np.isfinite(ec.k_prime):
                continue
            by_filter.setdefault(filt, []).append(
                (path.stem, float(ec.k_prime), float(ec.k_prime_err))
            )
    return by_filter


def write_extinction_aggregation_qc_plots(
    paths: list[str | Path],
    coeffs: dict[str, ExtinctionCoefficients],
    meta: dict[str, Any],
    plot_dir: str | Path,
    *,
    site: str = "OST_Potsdam",
) -> list[Path]:
    """
    Write QC PDFs for a site-extinction aggregation run.

    Produces ``extinction_nights_<filter>.pdf`` (per-night scatter) and
    ``extinction_site_summary.pdf`` (aggregated coefficients).
    """
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_per_night_extinction_samples(paths)
    written: list[Path] = []

    for filt in sorted(samples):
        nights = samples[filt]
        if not nights:
            continue
        labels = [n[0] for n in nights]
        k_vals = np.asarray([n[1] for n in nights], dtype=float)
        k_errs = np.asarray([n[2] for n in nights], dtype=float)
        agg = coeffs.get(filt)
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 5))
        x = np.arange(len(labels))
        ax.errorbar(
            x,
            k_vals,
            yerr=k_errs,
            fmt="o",
            color="steelblue",
            ecolor="gray",
            capsize=3,
            label="per night",
        )
        if agg is not None and np.isfinite(agg.k_prime):
            ax.axhline(
                agg.k_prime,
                color="crimson",
                linestyle="-",
                linewidth=1.2,
                label=f"aggregated ({meta.get('statistic', 'median')})",
            )
            ax.axhspan(
                agg.k_prime - agg.k_prime_err,
                agg.k_prime + agg.k_prime_err,
                color="crimson",
                alpha=0.12,
                label="aggregated ± err",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("k' [mag/airmass]")
        ax.set_title(f"{site} — filter {filt} ({len(labels)} nights)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = plot_dir / f"extinction_nights_{filt}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    if coeffs:
        fig, ax = plt.subplots(figsize=(max(6, len(coeffs) * 0.8), 5))
        filters = sorted(coeffs)
        k_out = np.asarray([coeffs[f].k_prime for f in filters], dtype=float)
        k_err = np.asarray([coeffs[f].k_prime_err for f in filters], dtype=float)
        x = np.arange(len(filters))
        ax.bar(x, k_out, yerr=k_err, color="steelblue", capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(filters)
        ax.set_ylabel("k' [mag/airmass]")
        n_nights = meta.get("n_input_nights", "?")
        ax.set_title(f"{site} site table — {n_nights} input nights")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = plot_dir / "extinction_site_summary.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    return written


__all__ = [
    "aggregate_extinction_coefficients",
    "bundled_site_extinction_path",
    "collect_per_night_extinction_samples",
    "extinction_coefficient_from_dict",
    "extinction_coefficient_to_dict",
    "extinction_coefficients_from_dict",
    "extinction_coefficients_to_dict",
    "load_extinction_coefficients",
    "merge_extinction_coefficients",
    "resolve_tabulated_extinction_coefficients",
    "save_extinction_coefficients",
    "write_extinction_aggregation_qc_plots",
]
