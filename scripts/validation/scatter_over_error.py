"""Robust scatter / error of constant stars in a light-curve run, by magnitude.

Usage: python scripts/validation/scatter_over_error.py <output_dir> [max_airmass] [quantity]

Reads ``<output_dir>/tables/light_curves.ecsv``. For every source with at
least 60 good epochs (no ``flag_epoch``, ``flag_outlier``, ``flag_saturated``,
airmass below ``max_airmass``) it compares the robust point-to-point scatter
of ``quantity`` with its median error:

- ``dmag`` (default): differential magnitude / ``dmag_err``;
- ``mag``: catalog-transformed magnitude / ``mag_err`` (point-to-point part);
- ``flux``: relative flux of the fallback (filters without catalog) /
  ``flux_err``.

A ratio of 1 means the errors describe the measured scatter. Use all stars,
not only the ensemble: the ensemble is chosen for low scatter and is biased
low.
"""

import sys

import numpy as np
from astropy.table import Table

_ERR = {"dmag": "dmag_err", "mag": "mag_err", "flux": "flux_err"}


def main(out_dir: str, max_airmass: float = 1.3, quantity: str = "dmag") -> None:
    lc = Table.read(f"{out_dir}/tables/light_curves.ecsv")
    err_col = _ERR[quantity]
    for f in sorted(set(np.asarray(lc["filter"]).astype(str))):
        if "-" in f:
            continue
        s = lc[np.asarray(lc["filter"]).astype(str) == f]
        good = ~np.asarray(s["flag_outlier"], bool) & ~np.asarray(s["flag_saturated"], bool)
        if "flag_epoch" in s.colnames:
            good &= ~np.asarray(s["flag_epoch"], bool)
        airmass = np.asarray(s["airmass"], float)
        good &= ~(airmass >= max_airmass)  # NaN airmass is kept
        s = s[good]
        ids = np.asarray(s["id"])
        rows = []
        for sid in np.unique(ids):
            m = ids == sid
            if m.sum() < 60:
                continue
            order = np.argsort(np.asarray(s["jd"][m], float))
            d = np.asarray(s[quantity][m], float)[order]
            e = np.asarray(s[err_col][m], float)[order]
            dd = np.diff(d) / np.sqrt(2)
            dd = dd[np.isfinite(dd)]
            if dd.size < 30 or not np.any(np.isfinite(e)):
                continue
            p2p = 1.4826 * np.median(np.abs(dd - np.median(dd)))
            rows.append((np.nanmedian(np.asarray(s["mag_inst"][m], float)), p2p / np.nanmedian(e), np.nanmedian(e)))
        if not rows:
            print(f"{f}: no star with enough epochs")
            continue
        rows = np.array(rows)
        print(f"{f} ({quantity}): {len(rows)} stars")
        edges = np.quantile(rows[:, 0], np.linspace(0, 1, 7))
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
            print(
                f"  mag_inst {lo:6.2f}..{hi:6.2f}: n={m.sum():3d}  "
                f"median err {np.median(rows[m, 2]):.4f}  scatter/err {np.median(rows[m, 1]):.2f}"
            )


if __name__ == "__main__":
    main(
        sys.argv[1],
        float(sys.argv[2]) if len(sys.argv) > 2 else 1.3,
        sys.argv[3] if len(sys.argv) > 3 else "dmag",
    )
