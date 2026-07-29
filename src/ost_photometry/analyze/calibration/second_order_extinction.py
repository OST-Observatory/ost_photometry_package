"""
Second-order extinction analysis from mk_calib field transformation records.

Fits field ``C`` coefficients vs airmass: ``C = T + k' * X`` with
second-order coefficient ``k" = -k'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .mk_calib import (
    FieldTransformationRecord,
    LEGACY_C_COLUMNS,
    LEGACY_C_ERR_COLUMNS,
    load_field_transformation_records,
)


def _lin_func(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return intercept + slope * x


def _fit_c_vs_airmass(
    airmass: list[float],
    c_values: list[float],
    c_errors: list[float],
    *,
    apply_weights: bool,
) -> tuple[float, float, float, float]:
    """Fit ``C = T + k' * X``; return ``T, T_err, k', k'_err``."""
    x = np.asarray(airmass, dtype=float)
    y = np.asarray(c_values, dtype=float)
    if apply_weights:
        sigma = np.asarray(c_errors, dtype=float)
        weights = 1.0 / np.maximum(sigma**2, 1e-12)
    else:
        weights = np.ones(len(x), dtype=float)
    sw = np.sqrt(weights)
    design = np.column_stack([np.ones(len(x)), x]) * sw[:, None]
    target = y * sw
    coeffs, residuals, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    intercept, slope = float(coeffs[0]), float(coeffs[1])
    if len(x) > 2 and rank == 2:
        dof = max(len(x) - 2, 1)
        if len(residuals):
            s_sq = float(residuals[0]) / dof
        else:
            fit_res = target - design @ coeffs
            s_sq = float(np.sum(fit_res**2)) / dof
        cov = s_sq * np.linalg.inv(design.T @ design)
        intercept_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        slope_err = float(np.sqrt(max(cov[1, 1], 0.0)))
    else:
        intercept_err = slope_err = 0.0
    return intercept, intercept_err, slope, slope_err


@dataclass
class SecondOrderFitResult:
    """Result of fitting one ``C`` column vs airmass across fields."""

    filter: str
    coefficient_column: str
    intercept_t: float
    intercept_t_err: float
    slope_k_prime: float
    slope_k_prime_err: float
    k_second_order: float
    k_second_order_err: float
    field_names: list[str]
    airmass: list[float]
    c_values: list[float]
    c_errors: list[float]
    plot_path: str | None = None

    @property
    def legacy_t_label(self) -> str:
        return self.coefficient_column[1:4]


def collect_points_for_fit(
    records: list[FieldTransformationRecord],
    filt: str,
    coefficient_column: str,
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Gather airmass, C, C_err, and field names for one coefficient column."""
    airmass: list[float] = []
    c_vals: list[float] = []
    c_errs: list[float] = []
    names: list[str] = []
    am_col = f"airmass_{filt}"
    for rec in records:
        coeff = rec.coefficient_by_column(coefficient_column)
        if coeff is None:
            continue
        if filt not in rec.airmass:
            continue
        airmass.append(rec.airmass[filt])
        c_vals.append(coeff.c)
        c_errs.append(coeff.c_err)
        names.append(rec.name)
    return airmass, c_vals, c_errs, names


def fit_second_order_extinction(
    airmass: list[float],
    c_values: list[float],
    c_errors: list[float],
    *,
    apply_weights: bool = True,
) -> tuple[float, float, float, float]:
    """
    Fit ``C = T + k' * X`` and return ``T, T_err, k', k'_err``.

    ``k" = -k'`` (mk_calib convention).
    """
    if len(airmass) < 2:
        raise ValueError("Need at least two fields for a second-order extinction fit")
    return _fit_c_vs_airmass(airmass, c_values, c_errors, apply_weights=apply_weights)


def fit_and_plot_second_order(
    records: list[FieldTransformationRecord],
    filt: str,
    coefficient_column: str,
    outdir: str | Path,
    *,
    apply_weights: bool = True,
    annotate_fields: bool = True,
) -> SecondOrderFitResult | None:
    """Fit one coefficient column and write ``C_vs_x_*.pdf`` diagnostic plot."""
    airmass, c_vals, c_errs, names = collect_points_for_fit(
        records, filt, coefficient_column
    )
    if len(airmass) < 2:
        return None

    t_val, t_err, k_prime, k_err = fit_second_order_extinction(
        airmass, c_vals, c_errs, apply_weights=apply_weights
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_label = coefficient_column[1:4]
    plot_path = outdir / f"C_vs_x_{filt}_T{t_label}.pdf"

    x_lin = np.sort(np.asarray(airmass, dtype=float))
    y_lin = _lin_func(x_lin, t_val, k_prime)

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle("C values vs. air mass", fontsize=30)
    plt.errorbar(
        airmass,
        c_vals,
        yerr=c_errs,
        color="blue",
        marker=".",
        mew=0.0,
        linestyle="none",
    )
    if annotate_fields:
        for i, name in enumerate(names):
            plt.annotate(
                name,
                (airmass[i], c_vals[i]),
                xytext=(airmass[i], c_vals[i]),
                textcoords="offset points",
            )
    plabel = (
        f"slope = {k_prime}, k = {-k_prime} +/- {k_err}; "
        f"T{t_label} = {t_val} +/- {t_err}"
    )
    plt.plot(x_lin, y_lin, linestyle="-", color="red", linewidth=0.8, label=plabel)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=4,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.xlabel("x [air mass]", fontsize=20)
    plt.ylabel(coefficient_column, fontsize=20)
    plt.savefig(plot_path, bbox_inches="tight", format="pdf")
    plt.close(fig)

    return SecondOrderFitResult(
        filter=filt,
        coefficient_column=coefficient_column,
        intercept_t=t_val,
        intercept_t_err=t_err,
        slope_k_prime=k_prime,
        slope_k_prime_err=k_err,
        k_second_order=-k_prime,
        k_second_order_err=k_err,
        field_names=names,
        airmass=list(airmass),
        c_values=list(c_vals),
        c_errors=list(c_errs),
        plot_path=str(plot_path),
    )


def run_second_order_campaign(
    table_paths: list[str | Path],
    filter_list: list[str],
    outdir: str | Path,
    *,
    c_columns: dict[str, list[str]] | None = None,
    c_err_columns: dict[str, list[str]] | None = None,
    apply_weights: bool = True,
    annotate_fields: bool = True,
) -> list[SecondOrderFitResult]:
    """
    Run the full second-order extinction analysis for all filters and C columns.

    Parameters match the ``mk_calib_photometry/3_second_order_extinction*.py`` scripts.
    ``table_paths`` may point to ``trans_para_*.dat`` or ``trans_para_*.json``.
    """
    records = load_field_transformation_records(table_paths)
    c_columns = c_columns or LEGACY_C_COLUMNS
    c_err_columns = c_err_columns or LEGACY_C_ERR_COLUMNS
    # c_err_columns kept for API symmetry; errors come from records
    _ = c_err_columns

    results: list[SecondOrderFitResult] = []
    for filt in filter_list:
        columns = c_columns.get(filt, [])
        for col in columns:
            fit = fit_and_plot_second_order(
                records,
                filt,
                col,
                outdir,
                apply_weights=apply_weights,
                annotate_fields=annotate_fields,
            )
            if fit is not None:
                results.append(fit)
    return results


__all__ = [
    "SecondOrderFitResult",
    "collect_points_for_fit",
    "fit_and_plot_second_order",
    "fit_second_order_extinction",
    "run_second_order_campaign",
]
