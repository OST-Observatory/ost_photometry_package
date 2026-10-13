"""Differential light curves against a quiet comparison ensemble.

The catalog-transformed curve (``mag`` = ``mag_cal_*``) keeps the zero point,
colour term and catalog uncertainty on purpose. The differential product here
uses only **instrumental** magnitudes (``mag_inst``): per filter, the quietest
field stars of similar brightness to the objects of interest form an ensemble;
their mean offset per epoch is the common mode (clouds, extinction, tracking).
Every source is divided by that common mode in flux:

``dmag = mag_inst - ens_offset`` (normalized to median 0),
``dflux = 10 ** (-0.4 * dmag)`` (continuum ≈ 1).

Ensemble members are measured against the ensemble without themselves
(leave-one-out), so their ``dflux`` scatter is an honest check of the method.
Catalog uncertainties never enter ``dmag_err`` / ``dflux_err``.

The same common mode decides ``flag_epoch``: an epoch is bad when too few
ensemble stars were measured, when the common mode leaves the extinction trend
of its night (clouds), or when the ensemble stars disagree with each other
(patchy clouds). Flags stay in the table; no rows are deleted.
"""

from __future__ import annotations

import warnings

import numpy as np
from astropy.table import Table, vstack

from ... import terminal_output
from .light_curve import DIFFERENTIAL_FLOAT_COLUMNS, excess_rms

ENSEMBLE_STATS_FILENAME = "light_curve_ensemble.ecsv"

_MAD_TO_SIGMA = 1.4826
_MIN_ENSEMBLE = 2
_MIN_EPOCHS = 3


def _nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmedian`` without the warning for all-NaN slices (-> NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _robust_scale(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    return float(_MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def _epoch_matrix(
    sub: Table,
    *,
    use_outlier_flags: bool = True,
) -> dict:
    """``(n_epochs, n_sources)`` matrices of ``mag_inst`` for one filter.

    Epochs are ordered by time. ``valid`` excludes non-finite values and, by
    default, per-source outliers (``flag_outlier``).
    """
    ids = np.asarray(sub["id"]).astype(np.int64)
    eids = np.asarray(sub["epoch_id"]).astype(str)
    jd = np.asarray(sub["jd"], dtype=float)
    mag = np.asarray(sub["mag_inst"], dtype=float)
    err = np.abs(np.asarray(sub["mag_inst_err"], dtype=float))
    if use_outlier_flags and "flag_outlier" in sub.colnames:
        outlier = np.asarray(sub["flag_outlier"], dtype=bool)
    else:
        outlier = np.zeros(len(sub), dtype=bool)

    epoch_keys = np.unique(eids)
    epoch_jd = np.array(
        [
            np.nanmedian(jd[eids == k]) if np.any(np.isfinite(jd[eids == k])) else np.inf
            for k in epoch_keys
        ]
    )
    epoch_keys = epoch_keys[np.argsort(epoch_jd, kind="stable")]
    source_ids = np.unique(ids)
    e_index = {k: i for i, k in enumerate(epoch_keys)}
    s_index = {int(s): j for j, s in enumerate(source_ids)}

    n_ep, n_src = len(epoch_keys), len(source_ids)
    m = np.full((n_ep, n_src), np.nan)
    e = np.full((n_ep, n_src), np.nan)
    bad = np.zeros((n_ep, n_src), dtype=bool)
    row = np.full((n_ep, n_src), -1, dtype=np.int64)
    for r in range(len(sub)):
        i = e_index[eids[r]]
        j = s_index[int(ids[r])]
        if row[i, j] >= 0:
            continue
        row[i, j] = r
        m[i, j] = mag[r]
        e[i, j] = err[r]
        bad[i, j] = outlier[r]
    # Missing or non-positive errors: use the source's median error; a source
    # without any error stays NaN (no weight in the ensemble, NaN dmag_err).
    for j in range(n_src):
        col = e[:, j]
        ok = np.isfinite(col) & (col > 0)
        fill = float(np.median(col[ok])) if np.any(ok) else np.nan
        e[~ok, j] = fill
    valid = np.isfinite(m) & ~bad
    return {
        "epoch_keys": epoch_keys,
        "source_ids": source_ids,
        "mag": m,
        "err": e,
        "valid": valid,
        "row": row,
    }


def _centered(mag: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Each source minus its median over valid epochs."""
    return mag - _nanmedian(np.where(valid, mag, np.nan), axis=0)[None, :]


def select_quiet_ensemble(
    mag: np.ndarray,
    err: np.ndarray,
    valid: np.ndarray,
    candidates: np.ndarray,
    *,
    reference_mag: float | None,
    size: int = 10,
    mag_window: float = 1.0,
    robust: bool = False,
) -> tuple[np.ndarray, dict]:
    """Pick the quietest candidates of similar brightness.

    Quietness is the measured scatter of each candidate after removing the
    median common mode of all candidates (raw instrumental scatter is
    dominated by the common mode itself). It is ranked by the scatter itself,
    not the excess RMS: inside the brightness window the scatter is what the
    ensemble adds to the differential curve, and the excess RMS is 0 for
    every star when the photometric errors are overestimated.
    ``robust=True`` uses the MAD scatter, so one bad epoch (patchy clouds)
    does not decide the selection.

    The magnitude window around ``reference_mag`` doubles until at least
    ``min(size, number of candidates)`` stars are inside (at most three
    times); ``reference_mag=None`` uses all candidates.

    Returns the member indices (quietest first) and per-source statistics
    (``med_mag``, ``rms``, ``excess_rms``, ``in_window``, ``window``).
    """
    n_src = mag.shape[1]
    cand = np.asarray(candidates, dtype=bool)
    r = _centered(mag, valid)
    if np.any(cand):
        common = _nanmedian(np.where(valid[:, cand], r[:, cand], np.nan), axis=1)
    else:
        common = np.zeros(mag.shape[0])
    q = r - np.nan_to_num(common, nan=0.0)[:, None]

    med_mag = np.full(n_src, np.nan)
    rms = np.full(n_src, np.nan)
    exc = np.full(n_src, np.nan)
    for j in range(n_src):
        ok = valid[:, j]
        if int(np.count_nonzero(ok)) < 2:
            continue
        med_mag[j] = float(np.median(mag[ok, j]))
        if cand[j]:
            y = q[ok, j] - np.median(q[ok, j])
            rms[j] = _robust_scale(y) if robust else float(np.sqrt(np.mean(y**2)))
            exc[j] = excess_rms(q[ok, j], err[ok, j])
    cand &= np.isfinite(rms)

    window = np.inf
    in_window = cand.copy()
    if reference_mag is not None and np.isfinite(reference_mag):
        want = min(int(size), int(np.count_nonzero(cand)))
        window = float(mag_window)
        for _ in range(3):
            in_window = cand & (np.abs(med_mag - float(reference_mag)) <= window)
            if int(np.count_nonzero(in_window)) >= want:
                break
            window *= 2.0
        else:
            in_window = cand.copy()
            window = np.inf

    pool = np.flatnonzero(in_window)
    order = pool[np.argsort(rms[pool], kind="stable")]
    members = order[: int(size)]
    return members, {
        "med_mag": med_mag,
        "rms": rms,
        "excess_rms": exc,
        "in_window": in_window,
        "window": window,
    }


def ensemble_common_mode(
    centered: np.ndarray,
    err: np.ndarray,
    valid: np.ndarray,
    members: np.ndarray,
    member_excess: np.ndarray,
) -> dict:
    """Weighted mean offset of the ensemble per epoch, plus leave-one-out.

    Weights are ``1 / (err**2 + excess_rms**2)`` so a slightly variable
    member counts less. ``offset_err`` is the photometric error of that mean
    (no catalog term, no excess floor).
    """
    n_ep = centered.shape[0]
    idx = np.asarray(members, dtype=int)
    r = centered[:, idx]
    v = valid[:, idx]
    exc = np.nan_to_num(np.asarray(member_excess, dtype=float), nan=0.0)
    w = np.where(v, 1.0 / (err[:, idx] ** 2 + exc[None, :] ** 2), 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    rw = np.where(v, r, 0.0) * w
    sw = w.sum(axis=1)
    swr = rw.sum(axis=1)
    sw2e2 = (w**2 * np.where(v, err[:, idx], 0.0) ** 2).sum(axis=1)
    n = v.sum(axis=1)
    with np.errstate(all="ignore"):
        offset = np.where(sw > 0, swr / sw, np.nan)
        offset_err = np.where(sw > 0, np.sqrt(sw2e2) / sw, np.nan)
        # Leave-one-out per member.
        sw_loo = sw[:, None] - w
        loo = np.where(sw_loo > 0, (swr[:, None] - rw) / sw_loo, np.nan)
        loo_var = sw2e2[:, None] - (w * np.where(v, err[:, idx], 0.0)) ** 2
        loo_err = np.where(sw_loo > 0, np.sqrt(np.maximum(loo_var, 0.0)) / sw_loo, np.nan)
    loo = np.where(v, loo, offset[:, None])
    loo_err = np.where(v, loo_err, offset_err[:, None])

    scatter = np.full(n_ep, np.nan)
    for i in range(n_ep):
        if n[i] >= 3:
            scatter[i] = _robust_scale(r[i, v[i]] - offset[i])
    return {
        "offset": offset,
        "offset_err": offset_err,
        "n": n.astype(np.int64),
        "scatter": scatter,
        "loo": loo,
        "loo_err": loo_err,
    }


def _transparency_model(
    offset: np.ndarray,
    airmass: np.ndarray,
    usable: np.ndarray,
) -> np.ndarray:
    """Expected common mode of one night: linear in airmass, else the median.

    The fit is clipped once at 3 robust sigma so cloudy epochs do not pull
    the extinction slope.
    """
    ok = usable & np.isfinite(offset)
    model = np.full(offset.shape, np.nan)
    if not np.any(ok):
        return model
    x = np.asarray(airmass, dtype=float)
    fit_ok = ok & np.isfinite(x)
    if int(np.count_nonzero(fit_ok)) >= 4 and float(np.ptp(x[fit_ok])) >= 0.05:
        coef = np.polyfit(x[fit_ok], offset[fit_ok], 1)
        res = offset[fit_ok] - np.polyval(coef, x[fit_ok])
        scale = _robust_scale(res)
        if np.isfinite(scale) and scale > 0:
            keep = np.abs(res) <= 3.0 * scale
            if int(np.count_nonzero(keep)) >= 4:
                coef = np.polyfit(x[fit_ok][keep], offset[fit_ok][keep], 1)
        model = np.where(np.isfinite(x), np.polyval(coef, x), np.median(offset[ok]))
        return model
    model[:] = float(np.median(offset[ok]))
    return model


def flag_bad_epochs(
    offset: np.ndarray,
    n_ensemble: np.ndarray,
    scatter: np.ndarray,
    airmass: np.ndarray,
    night_id: np.ndarray,
    *,
    ensemble_size: int,
    sigma: float | None = 5.0,
    min_offset: float = 0.05,
    min_ensemble_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Global epoch flags from the ensemble common mode.

    Reasons (``;``-joined when several apply):

    ``ensemble_incomplete``
        Fewer than ``max(2, min_ensemble_fraction * ensemble_size)`` members
        were measured.
    ``transparency``
        The common mode leaves the night's extinction trend (linear in
        airmass, else the night median) by more than
        ``max(sigma * robust scatter, min_offset)`` mag.
    ``ensemble_scatter``
        The members disagree: their scatter about the common mode exceeds the
        median scatter by more than ``max(sigma * robust scale, min_offset)``.

    ``sigma=None`` disables the last two tests.
    """
    n_ep = offset.size
    reasons: list[list[str]] = [[] for _ in range(n_ep)]
    need = max(_MIN_ENSEMBLE, int(np.ceil(float(min_ensemble_fraction) * ensemble_size)))
    incomplete = np.asarray(n_ensemble) < need
    for i in np.flatnonzero(incomplete):
        reasons[i].append("ensemble_incomplete")

    if sigma is not None:
        sig = float(sigma)
        nights = np.asarray(night_id)
        usable = ~incomplete & np.isfinite(offset)
        for night in np.unique(nights):
            m = nights == night
            model = _transparency_model(offset[m], np.asarray(airmass)[m], usable[m])
            res = offset[m] - model
            scale = _robust_scale(res[usable[m]])
            limit = max(sig * scale if np.isfinite(scale) else 0.0, float(min_offset))
            hit = usable[m] & (np.abs(res) > limit)
            for i in np.flatnonzero(m)[hit]:
                reasons[i].append("transparency")

        s = np.asarray(scatter, dtype=float)
        s_ok = usable & np.isfinite(s)
        if int(np.count_nonzero(s_ok)) >= _MIN_EPOCHS:
            med = float(np.median(s[s_ok]))
            scale = _robust_scale(s[s_ok])
            limit = max(sig * scale if np.isfinite(scale) else 0.0, float(min_offset))
            for i in np.flatnonzero(s_ok & (s - med > limit)):
                reasons[i].append("ensemble_scatter")

    flag = np.array([bool(r) for r in reasons], dtype=bool)
    text = np.array([";".join(r) for r in reasons], dtype="U64")
    return flag, text


def _common_mode_and_flags(
    centered: np.ndarray,
    err: np.ndarray,
    valid: np.ndarray,
    members: np.ndarray,
    excess: np.ndarray,
    airmass: np.ndarray,
    night: np.ndarray,
    flag_kw: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    cm = ensemble_common_mode(centered, err, valid, members, excess)
    flag, reason = flag_bad_epochs(
        cm["offset"],
        cm["n"],
        cm["scatter"],
        airmass,
        night,
        ensemble_size=int(members.size),
        **flag_kw,
    )
    return cm, flag, reason


def _empty_ensemble_stats() -> Table:
    return Table(
        {
            "id": np.array([], dtype=np.int64),
            "filter": np.array([], dtype="U32"),
            "med_mag_inst": np.array([], dtype=float),
            "coverage": np.array([], dtype=float),
            "rms": np.array([], dtype=float),
            "excess_rms": np.array([], dtype=float),
            "in_window": np.array([], dtype=bool),
            "is_ensemble": np.array([], dtype=bool),
        }
    )


def _init_differential_columns(out: Table) -> None:
    n = len(out)
    for col in DIFFERENTIAL_FLOAT_COLUMNS:
        out[col] = np.full(n, np.nan)
    out["ens_n"] = np.zeros(n, dtype=np.int64)
    out["is_ensemble"] = np.zeros(n, dtype=bool)
    out["flag_epoch"] = np.zeros(n, dtype=bool)
    out["flag_epoch_reason"] = np.full(n, "", dtype="U64")


def add_differential_columns(
    lc: Table,
    ooi_ids: set[int] | list[int] | None = None,
    *,
    ensemble_size: int = 10,
    mag_window: float = 1.0,
    min_coverage: float = 0.8,
    epoch_flag_sigma: float | None = 5.0,
    epoch_flag_min_mag: float = 0.05,
    min_ensemble_fraction: float = 0.5,
    verbose: bool = True,
) -> tuple[Table, Table]:
    """Add the differential product and ``flag_epoch`` to the long table.

    Parameters
    ----------
    lc
        Long light-curve table with ``mag_inst`` / ``mag_inst_err``.
    ooi_ids
        Objects of interest: never ensemble members; their median
        instrumental magnitude centres the magnitude window.
    ensemble_size
        Number of quiet comparison stars per filter.
    mag_window
        Half width (mag) of the brightness window around the OOI; widened
        when too few candidates are inside.
    min_coverage
        Minimum fraction of epochs a candidate must be measured in.
    epoch_flag_sigma, epoch_flag_min_mag, min_ensemble_fraction
        See :func:`flag_bad_epochs`.

    Returns
    -------
    The table with new columns (see module docstring) and per-candidate
    ensemble statistics (``light_curve_ensemble.ecsv``).
    """
    out = lc.copy()
    _init_differential_columns(out)
    if len(out) == 0 or "mag_inst" not in out.colnames:
        if verbose and len(out) > 0:
            terminal_output.print_to_terminal(
                "Differential light curves: no mag_inst column; skipped.",
                indent=1,
                style_name="WARNING",
            )
        return out, _empty_ensemble_stats()

    ooi = {int(i) for i in (ooi_ids or ())}
    filts = np.asarray(out["filter"]).astype(str)
    stats_parts: list[Table] = []
    for filter_ in np.unique(filts):
        idx = np.flatnonzero(filts == filter_)
        sub = out[idx]
        if not np.any(np.isfinite(np.asarray(sub["mag_inst"], dtype=float))):
            continue
        mat = _epoch_matrix(sub)
        mag, err, valid = mat["mag"], mat["err"], mat["valid"]
        src = mat["source_ids"]
        n_ep = mag.shape[0]
        if n_ep < _MIN_EPOCHS:
            if verbose:
                terminal_output.print_to_terminal(
                    f"Differential light curve ({filter_}): only {n_ep} epochs; skipped.",
                    indent=1,
                    style_name="WARNING",
                )
            continue

        is_ooi = np.array([int(s) in ooi for s in src], dtype=bool)
        coverage = valid.sum(axis=0) / float(n_ep)
        has_err = np.all(np.isfinite(err), axis=0)
        candidates = ~is_ooi & (coverage >= float(min_coverage)) & has_err
        ooi_mags = [
            float(np.median(mag[valid[:, j], j]))
            for j in np.flatnonzero(is_ooi)
            if np.any(valid[:, j])
        ]
        ref = float(np.median(ooi_mags)) if ooi_mags else None
        row = mat["row"]
        first = np.array([r[r >= 0][0] for r in row], dtype=np.int64)
        airmass = np.asarray(sub["airmass"], dtype=float)[first]
        night = np.asarray(sub["night_id"]).astype(np.int64)[first]
        centered = _centered(mag, valid)
        flag_kw = {
            "sigma": epoch_flag_sigma,
            "min_offset": epoch_flag_min_mag,
            "min_ensemble_fraction": min_ensemble_fraction,
        }

        # Pass 1: robust preselection, so a bad epoch cannot hide itself by
        # selecting the stars it happened to spare. Pass 2: plain RMS
        # without the flagged epochs, so stars with short eclipses drop out.
        members, sel = select_quiet_ensemble(
            mag,
            err,
            valid,
            candidates,
            reference_mag=ref,
            size=ensemble_size,
            mag_window=mag_window,
            robust=True,
        )
        if members.size >= _MIN_ENSEMBLE:
            _cm, flag, _reason = _common_mode_and_flags(
                centered, err, valid, members, sel["excess_rms"][members],
                airmass, night, flag_kw,
            )
            members, sel = select_quiet_ensemble(
                mag,
                err,
                valid & ~flag[:, None],
                candidates,
                reference_mag=ref,
                size=ensemble_size,
                mag_window=mag_window,
            )
        stats = Table(
            {
                "id": src[candidates].astype(np.int64),
                "filter": np.full(int(np.count_nonzero(candidates)), filter_, dtype="U32"),
                "med_mag_inst": sel["med_mag"][candidates],
                "coverage": coverage[candidates],
                "rms": sel["rms"][candidates],
                "excess_rms": sel["excess_rms"][candidates],
                "in_window": sel["in_window"][candidates],
                "is_ensemble": np.isin(np.flatnonzero(candidates), members),
            }
        )
        stats.meta[f"reference_mag_inst_{filter_}"] = np.nan if ref is None else ref
        stats.meta[f"mag_window_{filter_}"] = float(sel["window"])
        stats_parts.append(stats)

        if members.size < _MIN_ENSEMBLE:
            if verbose:
                terminal_output.print_to_terminal(
                    f"Differential light curve ({filter_}): {members.size} quiet "
                    f"comparison star(s) found, need {_MIN_ENSEMBLE}; skipped.",
                    indent=1,
                    style_name="WARNING",
                )
            continue

        cm, flag, reason = _common_mode_and_flags(
            centered, err, valid, members, sel["excess_rms"][members],
            airmass, night, flag_kw,
        )

        # Differential magnitude of every source; members use leave-one-out.
        ref_offset = np.repeat(cm["offset"][:, None], mag.shape[1], axis=1)
        ref_err = np.repeat(cm["offset_err"][:, None], mag.shape[1], axis=1)
        ref_offset[:, members] = cm["loo"]
        ref_err[:, members] = cm["loo_err"]
        d = mag - ref_offset
        norm_ok = valid & ~flag[:, None]
        zero = _nanmedian(np.where(norm_ok, d, np.nan), axis=0)
        dmag = d - zero[None, :]
        dmag_err = np.sqrt(err**2 + ref_err**2)
        dflux = 10.0 ** (-0.4 * dmag)
        dflux_err = 0.4 * np.log(10.0) * dflux * dmag_err

        member_set = set(int(src[j]) for j in members)
        cols = {
            "ens_offset": np.repeat(cm["offset"][:, None], mag.shape[1], axis=1),
            "ens_offset_err": np.repeat(cm["offset_err"][:, None], mag.shape[1], axis=1),
            "ens_scatter": np.repeat(cm["scatter"][:, None], mag.shape[1], axis=1),
            "dmag": dmag,
            "dmag_err": dmag_err,
            "dflux": dflux,
            "dflux_err": dflux_err,
        }
        i_ep, j_src = np.nonzero(row >= 0)
        rows_local = row[i_ep, j_src]
        rows_global = idx[rows_local]
        for name, arr in cols.items():
            out[name][rows_global] = arr[i_ep, j_src]
        out["ens_n"][rows_global] = cm["n"][i_ep]
        out["flag_epoch"][rows_global] = flag[i_ep]
        out["flag_epoch_reason"][rows_global] = reason[i_ep]
        out["is_ensemble"][rows_global] = np.array(
            [int(src[j]) in member_set for j in j_src], dtype=bool
        )

        if verbose:
            rms = sel["rms"][members]
            win = sel["window"]
            win_txt = "all magnitudes" if not np.isfinite(win) else f"±{win:.2g} mag of the OOI"
            terminal_output.print_to_terminal(
                f"Differential light curve ({filter_}): ensemble of {members.size} "
                f"quiet stars ({win_txt}; RMS "
                f"{np.min(rms):.4f}–{np.max(rms):.4f} mag), "
                f"{int(np.count_nonzero(flag))} of {n_ep} epochs flagged",
                indent=1,
                style_name="INFO" if not np.any(flag) else "WARNING",
            )

    if not stats_parts:
        return out, _empty_ensemble_stats()
    return out, vstack(stats_parts, metadata_conflicts="silent")


__all__ = [
    "DIFFERENTIAL_FLOAT_COLUMNS",
    "ENSEMBLE_STATS_FILENAME",
    "add_differential_columns",
    "ensemble_common_mode",
    "flag_bad_epochs",
    "select_quiet_ensemble",
]
