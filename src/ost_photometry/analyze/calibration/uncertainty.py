"""Uncertainty propagation for calibrated magnitudes."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy.table import Table

if TYPE_CHECKING:
    from .result import CalibrationResult

UncertaintyMode = Literal["fit_errors", "flux_monte_carlo", "both"]


def _analysis_warning_category():
    from ..warnings_types import OstPhotometryAnalyzeWarning

    return OstPhotometryAnalyzeWarning


def propagate_fit_errors(
    table: Table,
    filter_: str,
    *,
    inst_err_col: str | None = None,
    cal_err_col: str | None = None,
    zp_err: float = 0.0,
    color_term_err: float = 0.0,
    color_term: float = 0.0,
    cov_tz: float = 0.0,
    color: np.ndarray | None = None,
    sigma_color_sq: np.ndarray | float = 0.0,
) -> None:
    """
    Set calibrated magnitude errors from instrumental errors + fit parameter errors.

    When ``color`` is given, uses the full first-order variance including
    ``2·color·cov(T, ZP)``. Without ``color``, falls back to
    ``√(σ_inst² + σ_ZP² + σ_T²)`` (no color dependence; ``cov_tz`` ignored).
    """
    from .transform import calibrated_magnitude_variance

    inst_err_col = inst_err_col or f"err_{filter_}"
    cal_err_col = cal_err_col or f"err_cal_{filter_}"
    if inst_err_col not in table.colnames:
        return
    inst_err = np.asarray(table[inst_err_col], dtype=float)
    if color is None:
        total = np.sqrt(inst_err**2 + zp_err**2 + color_term_err**2)
    else:
        var = calibrated_magnitude_variance(
            inst_err,
            color,
            color_term=color_term,
            color_term_err=color_term_err,
            zero_point_err=zp_err,
            cov_tz=cov_tz,
            sigma_color_sq=sigma_color_sq,
        )
        total = np.sqrt(np.maximum(var, 0.0))
    table[cal_err_col] = total


def propagate_flux_monte_carlo(
    flux: np.ndarray,
    flux_err: np.ndarray,
    zp: float,
    *,
    n_samples: int = 1000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo magnitude errors from flux uncertainties.

    Returns (mag_cal, mag_cal_err) arrays.
    """
    rng = np.random.default_rng(seed)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    valid = np.isfinite(flux) & (flux > 0) & np.isfinite(flux_err)
    mag = np.full(flux.shape, np.nan)
    mag_err = np.full(flux.shape, np.nan)
    if not np.any(valid):
        return mag, mag_err

    n = int(np.sum(valid))
    samples = rng.normal(
        flux[valid],
        np.maximum(flux_err[valid], 1e-12),
        size=(n_samples, n),
    )
    samples = np.maximum(samples, 1e-12)
    mag_samples = -2.5 * np.log10(samples)
    mag[valid] = -2.5 * np.log10(flux[valid])
    mag_err[valid] = np.nanstd(mag_samples, axis=0)
    mag[valid] = mag[valid] + zp
    return mag, mag_err


def _resolve_calibration_result(
    results: dict[str, "CalibrationResult"],
    epoch_id: str,
) -> "CalibrationResult | None":
    if epoch_id in results:
        return results[epoch_id]
    if "ensemble" in results:
        return results["ensemble"]
    if len(results) == 1:
        return next(iter(results.values()))
    return None


def _apply_uncertainty_mode_slice(
    table: Table,
    row_mask: np.ndarray,
    calibration: "CalibrationResult",
    filters: list[str],
    *,
    uncertainty_mode: UncertaintyMode,
    distribution_samples: int,
    random_seed: int | None,
) -> None:
    """Update ``err_cal_*`` for one epoch slice in place."""
    if uncertainty_mode == "fit_errors":
        return

    for filter_ in filters:
        tc = calibration.transformation.get(filter_)
        if tc is None:
            continue
        cal_err_col = f"err_cal_{filter_}"
        if cal_err_col not in table.colnames:
            continue

        fit_err = np.asarray(table[cal_err_col][row_mask], dtype=float)
        flux_col = f"flux_{filter_}"
        ferr_col = f"flux_err_{filter_}"

        if uncertainty_mode in ("flux_monte_carlo", "both"):
            if flux_col not in table.colnames or ferr_col not in table.colnames:
                if uncertainty_mode == "flux_monte_carlo":
                    warnings.warn(
                        f"uncertainty_mode='flux_monte_carlo' but {flux_col}/{ferr_col} "
                        f"missing for filter {filter_}; keeping fit_errors.",
                        category=_analysis_warning_category(),
                        stacklevel=3,
                    )
                continue
            flux = np.asarray(table[flux_col][row_mask], dtype=float)
            ferr = np.asarray(table[ferr_col][row_mask], dtype=float)
            has_flux = np.isfinite(flux) & (flux > 0) & np.isfinite(ferr)
            if not np.any(has_flux):
                if uncertainty_mode == "flux_monte_carlo":
                    warnings.warn(
                        f"uncertainty_mode='flux_monte_carlo' but no valid flux rows "
                        f"for filter {filter_}; keeping fit_errors.",
                        category=_analysis_warning_category(),
                        stacklevel=3,
                    )
                continue
            _, mc_err = propagate_flux_monte_carlo(
                flux,
                ferr,
                tc.zero_point,
                n_samples=distribution_samples,
                seed=random_seed,
            )
            if uncertainty_mode == "flux_monte_carlo":
                new_err = mc_err
            else:
                new_err = np.sqrt(np.maximum(fit_err**2 + mc_err**2, 0.0))
            updated = np.asarray(table[cal_err_col][row_mask], dtype=float)
            updated[np.isfinite(new_err)] = new_err[np.isfinite(new_err)]
            table[cal_err_col][row_mask] = updated


def apply_uncertainty_mode_to_calibrated_table(
    calibrated: Table,
    results: dict[str, "CalibrationResult"],
    filters: list[str],
    *,
    uncertainty_mode: UncertaintyMode = "fit_errors",
    distribution_samples: int = 1000,
    random_seed: int | None = None,
) -> Table:
    """
    Refine ``err_cal_*`` on a calibrated epoch-native table.

  ``fit_errors`` leaves errors unchanged (already set by :meth:`apply_transform_to_table`
    or derive-transform apply). ``flux_monte_carlo`` replaces them with flux MC spread
    (+ ZP). ``both`` combines fit and MC errors in quadrature.
    """
    if uncertainty_mode == "fit_errors" or len(calibrated) == 0:
        return calibrated

    out = calibrated.copy()
    if "epoch_id" in out.colnames:
        for epoch_id in np.unique(out["epoch_id"]):
            cal = _resolve_calibration_result(results, str(epoch_id))
            if cal is None:
                warnings.warn(
                    f"No calibration result for epoch_id={epoch_id!r}; "
                    "skipping uncertainty refinement.",
                    category=_analysis_warning_category(),
                    stacklevel=2,
                )
                continue
            mask = np.asarray(out["epoch_id"] == epoch_id)
            _apply_uncertainty_mode_slice(
                out,
                mask,
                cal,
                filters,
                uncertainty_mode=uncertainty_mode,
                distribution_samples=distribution_samples,
                random_seed=random_seed,
            )
    else:
        cal = _resolve_calibration_result(results, "")
        if cal is None and results:
            cal = next(iter(results.values()))
        if cal is not None:
            mask = np.ones(len(out), dtype=bool)
            _apply_uncertainty_mode_slice(
                out,
                mask,
                cal,
                filters,
                uncertainty_mode=uncertainty_mode,
                distribution_samples=distribution_samples,
                random_seed=random_seed,
            )
    return out


__all__ = [
    "UncertaintyMode",
    "apply_uncertainty_mode_to_calibrated_table",
    "propagate_fit_errors",
    "propagate_flux_monte_carlo",
]
