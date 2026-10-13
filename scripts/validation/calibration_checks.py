"""Checks of calibrated magnitudes and their errors on a pipeline run.

Usage: python scripts/validation/calibration_checks.py <output_dir>

Reads ``tables/calibrated_magnitudes_*.ecsv`` (the first one not marked
``post_processed``) and ``tables/calibration_epochs.ecsv``.

1. Fit statistics per filter: intrinsic scatter σ_int, χ²_ν, number of
   calibrators, quality flags.
2. Hold-out catalog stars (catalog magnitude, not used in the fit):
   robust σ of ``(mag_cal - mag_std) / √(err_cal² + (s·err_std)² + σ_int²)``
   with the fitted ``catalog_error_scale`` s (σ_int: a star's own deviation
   from the colour-term model). These stars are biased: most were rejected
   by the clipping or the quality cuts, so the value is above 1.
3. Constant stars over epochs (several epochs only): robust scatter of
   ``mag_cal`` about its median divided by ``√(err_cal_rand² + σ_ZP,phot²)``
   (the parts that change from epoch to epoch) should be ~1.
4. Random cross-validation: the candidate calibrators (``is_comparison``,
   catalog magnitudes, not saturated) of an epoch are split in random
   halves; ``fit_zero_point`` on one half predicts the other. The fraction
   of ``|(Δ - ZP - T·C) / √(σ_inst² + s²(σ_cat² + T²σ_C²) + σ_int² + σ_pred²)|``
   below 1 should be 0.683. This tests the whole error model without the
   selection bias of check 2 (the robust σ underrates heavy-tailed real
   residuals, use the coverage).
"""

from __future__ import annotations

import glob
import sys

import numpy as np
from astropy.table import Table


def _robust(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _cross_validation(tbl: Table, f: str, color_filters: tuple[str, str], *, n_splits: int = 40,
                      max_epochs: int = 20, seed: int = 0) -> np.ndarray:
    """Hold-out pulls over random half splits (check 4)."""
    from ost_photometry.analyze.calibration.statistics import catalog_errors, fit_zero_point

    c1, c2 = color_filters
    needed = [f"mag_{f}", f"err_{f}", f"mag_std_{f}", f"mag_std_{c1}", f"mag_std_{c2}"]
    if any(c not in tbl.colnames for c in needed):
        return np.array([])
    rng = np.random.default_rng(seed)
    eids = np.asarray(tbl["epoch_id"]).astype(str) if "epoch_id" in tbl.colnames else np.zeros(len(tbl), str)
    epochs = np.unique(eids)
    if epochs.size > max_epochs:
        epochs = rng.choice(epochs, max_epochs, replace=False)
    pulls = []
    for eid in epochs:
        t = tbl[eids == eid]
        delta = np.asarray(t[f"mag_std_{f}"], dtype=float) - np.asarray(t[f"mag_{f}"], dtype=float)
        color = np.asarray(t[f"mag_std_{c1}"], dtype=float) - np.asarray(t[f"mag_std_{c2}"], dtype=float)
        s_inst = np.asarray(t[f"err_{f}"], dtype=float)
        def cat_err(name, t=t):
            return catalog_errors(np.asarray(t[name], dtype=float) if name in t.colnames else None, len(t))

        s_cat = cat_err(f"err_std_{f}")
        s_color = np.hypot(cat_err(f"err_std_{c1}"), cat_err(f"err_std_{c2}"))
        pool = np.isfinite(delta) & np.isfinite(color) & np.isfinite(s_inst) & (s_inst > 0)
        if "is_comparison" in t.colnames:
            pool &= np.asarray(t["is_comparison"], dtype=bool)
        if f"saturated_{f}" in t.colnames:
            pool &= ~np.asarray(t[f"saturated_{f}"], dtype=bool)
        idx = np.flatnonzero(pool)
        if idx.size < 20:
            continue
        for _ in range(max(1, n_splits // max(1, epochs.size // 5))):
            perm = rng.permutation(idx)
            train, test = perm[: idx.size // 2], perm[idx.size // 2:]
            fit = fit_zero_point(delta[train], color[train], s_inst[train], s_cat[train],
                                 fit_color=True, sigma_color=s_color[train])
            if not np.isfinite(fit.zero_point):
                continue
            sc = fit.catalog_error_scale
            ct = color[test]
            var_pred = fit.zero_point_err**2 + 2 * ct * fit.cov_tz + ct**2 * fit.color_term_err**2
            var = (s_inst[test] ** 2 + sc**2 * (s_cat[test] ** 2 + fit.color_term**2 * s_color[test] ** 2)
                   + fit.sigma_int**2 + var_pred)
            pulls.append((delta[test] - fit.zero_point - fit.color_term * ct) / np.sqrt(var))
    return np.concatenate(pulls) if pulls else np.array([])


def main(out_dir: str) -> None:
    files = sorted(f for f in glob.glob(f"{out_dir}/tables/calibrated_magnitudes_*.ecsv")
                   if "post_processed" not in f)
    tbl = Table.read(files[0])
    epochs = Table.read(f"{out_dir}/tables/calibration_epochs.ecsv")
    filters = sorted({c[len("mag_cal_"):] for c in tbl.colnames if c.startswith("mag_cal_")})
    print(f"{files[0].split('/')[-1]}: {len(tbl)} rows, filters {filters}")
    for f in filters:
        e = epochs[np.asarray(epochs["filter"]).astype(str) == f]
        print(f"\n[{f}] {len(e)} epoch fit(s): σ_int median {np.nanmedian(e['sigma_int']):.4f}, "
              f"χ²_ν median {np.nanmedian(e['chi2_nu']):.2f}, "
              + (f"catalog error scale median {np.nanmedian(e['catalog_error_scale']):.2f}, "
                 if "catalog_error_scale" in e.colnames else "")
              + f"n_used median {int(np.nanmedian(e['n_stars_used']))}, "
              f"ZP err {np.nanmedian(e['zero_point_err']):.4f} "
              f"(phot {np.nanmedian(e['zero_point_err_phot']):.4f}, cat {np.nanmedian(e['zero_point_err_cat']):.4f}), "
              f"flags {sorted(set(np.asarray(e['quality_flag']).astype(str)))}")
        sigma_int = float(np.nanmedian(e["sigma_int"]))
        std = f"mag_std_{f}"
        if std in tbl.colnames:
            used = (
                np.asarray(tbl[f"is_calibrator_{f}"], dtype=bool)
                if f"is_calibrator_{f}" in tbl.colnames
                else np.zeros(len(tbl), dtype=bool)
            )
            holdout = np.isfinite(np.asarray(tbl[std], dtype=float)) & ~used
            if f"saturated_{f}" in tbl.colnames:
                holdout &= ~np.asarray(tbl[f"saturated_{f}"], dtype=bool)
            err_std = (
                np.asarray(tbl[f"err_std_{f}"], dtype=float)
                if f"err_std_{f}" in tbl.colnames else np.full(len(tbl), 0.02)
            )
            err_std = np.where(np.isfinite(err_std) & (err_std > 0), err_std, 0.02)
            if "catalog_error_scale" in e.colnames:
                err_std = err_std * float(np.nanmedian(e["catalog_error_scale"]))
            resid = np.asarray(tbl[f"mag_cal_{f}"], dtype=float) - np.asarray(tbl[std], dtype=float)
            denom = np.sqrt(np.asarray(tbl[f"err_cal_{f}"], dtype=float) ** 2 + err_std**2 + sigma_int**2)
            pull = (resid / denom)[holdout]
            print(f"  hold-out catalog stars: n={int(np.isfinite(pull).sum())}, "
                  f"robust σ(pull) {_robust(pull):.2f}, median residual {np.nanmedian(resid[holdout]):+.4f}")
            cal_pull = (resid / denom)[used]
            print(f"  calibrators (for comparison): n={int(used.sum())}, robust σ(pull) {_robust(cal_pull):.2f}")
        color_index = str(e["color_index"][0]).split("-") if len(e) and "color_index" in e.colnames else ["B", "V"]
        cv = _cross_validation(tbl, f, (color_index[0], color_index[1]))
        cv = cv[np.isfinite(cv)]
        if cv.size:
            print(f"  random cross-validation (half splits): n={cv.size}, robust σ(pull) {_robust(cv):.2f}, "
                  f"|pull| < 1: {np.mean(np.abs(cv) < 1):.3f} (0.683), < 2: {np.mean(np.abs(cv) < 2):.3f} (0.954)")
        if "epoch_id" in tbl.colnames and len(set(tbl["epoch_id"])) > 2 and f"err_cal_rand_{f}" in tbl.colnames:
            zp_phot = {str(r["epoch_id"]): float(r["zero_point_err_phot"]) for r in e}
            ids = np.asarray(tbl["id"])
            eids = np.asarray(tbl["epoch_id"]).astype(str)
            mags = np.asarray(tbl[f"mag_cal_{f}"], dtype=float)
            rand = np.asarray(tbl[f"err_cal_rand_{f}"], dtype=float)
            zp = np.array([zp_phot.get(x, np.nan) for x in eids])
            ratios, errs = [], []
            for sid in np.unique(ids):
                m = (ids == sid) & np.isfinite(mags)
                if m.sum() < 20:
                    continue
                pred = np.sqrt(rand[m] ** 2 + zp[m] ** 2)
                ratios.append(_robust(mags[m]) / np.nanmedian(pred))
                errs.append(np.nanmedian(pred))
            ratios, errs = np.asarray(ratios), np.asarray(errs)
            if ratios.size:
                order = np.argsort(errs)
                for part in np.array_split(order, 4):
                    print(f"  epoch scatter / (rand ⊕ ZP_phot): err ~{np.median(errs[part]):.4f}: "
                          f"{np.nanmedian(ratios[part]):.2f} (n={part.size})")


if __name__ == "__main__":
    main(sys.argv[1])
