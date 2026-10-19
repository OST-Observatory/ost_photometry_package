"""Calibrated magnitudes against Gaia DR3 synthetic Johnson photometry.

Usage: python scripts/validation/gaia_synthetic_comparison.py <output_dir> [radius_arcmin] [gaia|stetson]

Reads ``<output_dir>/tables/calibrated_magnitudes_*.ecsv`` (several epochs
are reduced to the per-star median), queries the Gaia DR3 Synthetic
Photometry Catalogue (Vizier ``I/360/syntphot``, Johnson-Kron-Cousins B/V,
standardised on Landolt) around the field centre and matches within 1".

``stetson`` uses Stetson's ground-based standard photometry instead
(Pancino et al. 2022 collection, Vizier ``J/A+A/664/A109/table5``, entries
with ``Collect == Stetson``; e.g. the NGC 7789 standard field). Gaia XP
synthetic photometry of faint stars in crowded fields is contaminated by
neighbours, Stetson's is not.

For B and V it prints the median of ``mag_cal - gaia`` and its trends with
magnitude, colour, distance from the image centre and neighbours; the same
for the calibration catalog (``mag_std - gaia``) as a reference: a trend in
ours that the catalog does not have comes from the photometry. Only stars
with Gaia flag 1, no saturation and Gaia error < 0.03 mag are used.
"""

from __future__ import annotations

import glob
import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.vizier import Vizier


def _robust(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.nan, np.nan, v.size
    med = np.median(v)
    return float(med), float(1.4826 * np.median(np.abs(v - med)) / np.sqrt(v.size) * 1.2533), int(v.size)


def _per_star(tbl: Table) -> Table:
    if "epoch_id" not in tbl.colnames or len(set(tbl["epoch_id"])) <= 1:
        return tbl
    ids = np.asarray(tbl["id"])
    keep = [c for c in tbl.colnames if c.startswith(("mag_cal_", "mag_std_", "x", "y", "ra", "dec", "saturated_"))]
    rows = []
    for sid in np.unique(ids):
        m = ids == sid
        row = {"id": sid}
        for c in keep:
            v = np.asarray(tbl[c][m])
            if c.startswith("saturated_"):
                row[c] = bool(np.any(v))
            else:
                v = v.astype(float)
                row[c] = float(np.nanmedian(v)) if np.any(np.isfinite(v)) else np.nan
        rows.append(row)
    return Table(rows=rows)


def _trend(label, x, d, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x < hi)
        med, err, n = _robust(d[m])
        if n >= 5:
            out.append(f"{lo:.2f}-{hi:.2f}: {1000 * med:+5.0f}±{1000 * err:.0f} (n={n})")
    print(f"    vs {label}: " + " | ".join(out))


def _stetson(center, radius_arcmin):
    """Stetson B/V as a table with the Gaia column names used below."""
    from astropy.table import Table as _T

    viz = Vizier(columns=["RAJ2000", "DEJ2000", "Bmag", "e_Bmag", "Vmag", "e_Vmag", "Collect", "VarFlag"],
                 row_limit=200_000)
    tbl = viz.query_region(center, radius=radius_arcmin * u.arcmin, catalog="J/A+A/664/A109/table5")[0]
    tbl = tbl[np.asarray(tbl["Collect"]).astype(str) == "Stetson"]
    out = _T()
    out["RA_ICRS"] = np.asarray(tbl["RAJ2000"], float)
    out["DE_ICRS"] = np.asarray(tbl["DEJ2000"], float)
    for band in ("B", "V"):
        mag = np.asarray(np.ma.filled(tbl[f"{band}mag"], np.nan), float)
        err = np.asarray(np.ma.filled(tbl[f"e_{band}mag"], np.nan), float)
        out[f"{band}mag"] = mag
        # flux and its error so that 1.0857·e_F/F = e_mag
        out[f"F{band}"] = np.ones(len(tbl))
        out[f"e_F{band}"] = err / 1.0857
        var = np.asarray(np.ma.filled(tbl["VarFlag"], 0)).astype(str) if "VarFlag" in tbl.colnames else None
        ok = np.isfinite(mag) & np.isfinite(err)
        if var is not None:
            ok &= np.isin(np.char.strip(var), ["", "0", "--"])
        out[f"{band}Flag"] = ok.astype(int)
    return out


def main(out_dir: str, radius_arcmin: float = 15.0, reference: str = "gaia") -> None:
    f = sorted(x for x in glob.glob(f"{out_dir}/tables/calibrated_magnitudes_*.ecsv") if "post_processed" not in x)[0]
    tbl = _per_star(Table.read(f))
    ra, dec = np.asarray(tbl["ra"], float), np.asarray(tbl["dec"], float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    center = SkyCoord(np.nanmedian(ra) * u.deg, np.nanmedian(dec) * u.deg)
    if reference == "stetson":
        gaia = _stetson(center, radius_arcmin)
    else:
        viz = Vizier(columns=["RA_ICRS", "DE_ICRS", "Bmag", "Vmag", "FB", "e_FB", "FV", "e_FV", "BFlag", "VFlag"],
                     row_limit=200_000)
        gaia = viz.query_region(center, radius=radius_arcmin * u.arcmin, catalog="I/360/syntphot")[0]
    g = SkyCoord(np.asarray(gaia["RA_ICRS"], float) * u.deg, np.asarray(gaia["DE_ICRS"], float) * u.deg)
    ours = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg)
    idx, sep, _ = ours.match_to_catalog_sky(g)
    good = sep < 1 * u.arcsec
    rows = np.flatnonzero(ok)[good]
    gi = idx[good]
    # neighbours within 10" in Gaia (crowding)
    _i1, _i2, s2, _ = g.search_around_sky(g[gi], 10 * u.arcsec)
    n_neigh = np.bincount(_i1, minlength=gi.size) - 1
    xs = np.asarray(tbl["x"], float)[rows] if "x" in tbl.colnames else np.full(rows.size, np.nan)
    ys = np.asarray(tbl["y"], float)[rows] if "y" in tbl.colnames else np.full(rows.size, np.nan)
    r_center = np.hypot(xs - np.nanmedian(xs), ys - np.nanmedian(ys))
    gb, gv = np.asarray(gaia["Bmag"], float)[gi], np.asarray(gaia["Vmag"], float)[gi]
    print(f"{f.split('/')[-1]}: {len(tbl)} stars, {rows.size} matched to Gaia synthetic photometry")
    for flt, gmag, flux, eflux, flag in (("B", gb, "FB", "e_FB", "BFlag"), ("V", gv, "FV", "e_FV", "VFlag")):
        col = f"mag_cal_{flt}"
        if col not in tbl.colnames:
            continue
        cal = np.asarray(tbl[col], float)[rows]
        std = np.asarray(tbl[f"mag_std_{flt}"], float)[rows] if f"mag_std_{flt}" in tbl.colnames else np.full(rows.size, np.nan)
        gerr = 1.0857 * np.asarray(gaia[eflux], float)[gi] / np.asarray(gaia[flux], float)[gi]
        use = np.isfinite(cal) & np.isfinite(gmag) & (np.asarray(gaia[flag])[gi] == 1) & (gerr < 0.03)
        sat = f"saturated_{flt}"
        if sat in tbl.colnames:
            use &= ~np.asarray(tbl[sat], bool)[rows]
        d = (cal - gmag)[use]
        d_cat = (std - gmag)[use]
        med, err, n = _robust(d)
        medc, errc, nc = _robust(d_cat)
        print(f"\n[{flt}] ours - Gaia: {1000 * med:+.0f}±{1000 * err:.0f} mmag (n={n}, robust σ "
              f"{1000 * 1.4826 * np.nanmedian(np.abs(d - med)):.0f} mmag) | catalog - Gaia: "
              f"{1000 * medc:+.0f}±{1000 * errc:.0f} (n={nc})")
        mag = gmag[use]
        colour = (gb - gv)[use]
        iso = n_neigh[use] == 0
        has_cat = np.isfinite(std[use])
        m_iso, e_iso, n_iso = _robust(d[iso])
        m_cs, e_cs, n_cs = _robust(d[has_cat])
        print(f"  ours - Gaia: isolated stars {1000 * m_iso:+.0f}±{1000 * e_iso:.0f} (n={n_iso}); "
              f"catalog stars {1000 * m_cs:+.0f}±{1000 * e_cs:.0f} (n={n_cs}) vs catalog - Gaia "
              f"{1000 * medc:+.0f}±{1000 * errc:.0f}")
        di = d[iso] - m_iso
        print("  ours, isolated stars only (no Gaia neighbour < 10\"), offset removed, mmag:")
        _trend("Gaia mag", mag[iso], di, np.quantile(mag[iso], np.linspace(0, 1, 6)))
        rci = r_center[use][iso]
        if np.any(np.isfinite(rci)):
            _trend("r from centre [px]", rci, di, np.quantile(rci[np.isfinite(rci)], np.linspace(0, 1, 5)))
        for name, dd in (("ours", d - med), ("catalog", d_cat - medc)):
            print(f"  {name} (offset removed), mmag:")
            _trend("Gaia mag", mag, dd, np.quantile(mag[np.isfinite(mag)], np.linspace(0, 1, 6)))
            _trend("B-V", colour, dd, np.quantile(colour[np.isfinite(colour)], np.linspace(0, 1, 6)))
            if name == "ours":
                rc = r_center[use]
                if np.any(np.isfinite(rc)):
                    _trend("r from centre [px]", rc, dd, np.quantile(rc[np.isfinite(rc)], np.linspace(0, 1, 5)))
                _trend("Gaia neighbours <10\"", n_neigh[use].astype(float), dd, np.array([0, 0.5, 1.5, 3.5, 99]))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 15.0,
         sys.argv[3] if len(sys.argv) > 3 else "gaia")
