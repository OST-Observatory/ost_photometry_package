"""Robust scatter / error of constant stars in a light-curve run, by magnitude.

Usage: python scripts/validation/scatter_over_error.py <run_dir> [max_airmass]

Reads ``<run_dir>/output/tables/light_curves.ecsv``. For every source with at
least 60 good epochs (no ``flag_epoch``, ``flag_outlier``, ``flag_saturated``,
airmass below ``max_airmass``) it compares the robust point-to-point scatter
of the differential magnitude ``dmag`` with the median ``dmag_err``. A ratio
of 1 means the errors describe the measured scatter. Use all stars, not only
the ensemble: the ensemble is chosen for low scatter and is biased low.
"""

import sys

import numpy as np
from astropy.table import Table


def main(run_dir: str, max_airmass: float = 1.3) -> None:
    lc = Table.read(f"{run_dir}/output/tables/light_curves.ecsv")
    for f in sorted(set(np.asarray(lc["filter"]).astype(str))):
        if "-" in f:
            continue
        s = lc[np.asarray(lc["filter"]).astype(str) == f]
        good = (
            ~np.asarray(s["flag_epoch"], bool)
            & ~np.asarray(s["flag_outlier"], bool)
            & ~np.asarray(s["flag_saturated"], bool)
            & (np.asarray(s["airmass"], float) < max_airmass)
        )
        s = s[good]
        ids = np.asarray(s["id"])
        rows = []
        for sid in np.unique(ids):
            m = ids == sid
            if m.sum() < 60:
                continue
            order = np.argsort(np.asarray(s["jd"][m], float))
            d = np.asarray(s["dmag"][m], float)[order]
            e = np.asarray(s["dmag_err"][m], float)[order]
            dd = np.diff(d) / np.sqrt(2)
            dd = dd[np.isfinite(dd)]
            if dd.size < 30 or not np.any(np.isfinite(e)):
                continue
            p2p = 1.4826 * np.median(np.abs(dd - np.median(dd)))
            rows.append((np.nanmedian(np.asarray(s["mag_inst"][m], float)), p2p / np.nanmedian(e), np.nanmedian(e)))
        rows = np.array(rows)
        print(f"{f}: {len(rows)} stars")
        edges = np.quantile(rows[:, 0], np.linspace(0, 1, 7))
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
            print(
                f"  mag_inst {lo:6.2f}..{hi:6.2f}: n={m.sum():3d}  "
                f"median err {np.median(rows[m, 2]):.4f}  scatter/err {np.median(rows[m, 1]):.2f}"
            )


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.3)
