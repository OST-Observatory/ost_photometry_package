"""
Magnitude system transforms for calibration catalog tables.

**Lupton (2005)** empirical relations map SDSS/Sloan *ugriz* to Johnson-Cousins
*UBVRI*. R/I from *r'*, *i'* are widely used for APASS and SDSS-based Vizier
catalogs that lack native Johnson R/I. B/V (and optional U) use the same
Lupton / Jester-style relations for post-calibration SDSS→Bessell conversion.

References
----------
* https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Lupton2005
* AAVSO summary of SDSS transformations (same coefficients).

Error propagation uses first-order partial derivatives (Gaussian error model);
missing errors default to 0.02 mag per band.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

# Coefficients from Lupton (2005):
#   B = g + 0.3130*(g - r) + 0.2271
#   V = g - 0.5784*(g - r) - 0.0038
#   R = r - 0.2936*(r - i) - 0.1439
#   I = r - 1.2444*(r - i) - 0.3820
# U from Jester et al. (2005) all-stars: U-B = 0.78*(u - g) - 0.88  →  U = B + …


def _default_err(arr: np.ndarray, err: np.ndarray | None) -> np.ndarray:
    if err is None:
        return np.full_like(arr, 0.02, dtype=float)
    return np.asarray(err, dtype=float)


def johnson_ri_from_sloan_ri(
    r_sloan: np.ndarray,
    i_sloan: np.ndarray,
    err_r: np.ndarray | None = None,
    err_i: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Johnson R, I and propagated errors from Sloan r', i'.

    Parameters
    ----------
    r_sloan, i_sloan
        Magnitude arrays (finite where valid).
    err_r, err_i
        Uncertainties; if None, 0.02 mag is used where values are finite.

    Returns
    -------
    R_johnson, I_johnson, err_R, err_I
        Same shape as inputs; NaN where r or i not finite.
    """
    r_sloan = np.asarray(r_sloan, dtype=float)
    i_sloan = np.asarray(i_sloan, dtype=float)
    n = r_sloan.shape
    R_johnson = np.full(n, np.nan)
    I_johnson = np.full(n, np.nan)
    err_R = np.full(n, np.nan)
    err_I = np.full(n, np.nan)

    err_r = _default_err(r_sloan, err_r)
    err_i = _default_err(i_sloan, err_i)

    valid = np.isfinite(r_sloan) & np.isfinite(i_sloan)
    dR_dr, dR_di = 0.7064, 0.2936
    dI_dr, dI_di = -0.2444, 1.2444
    ri = r_sloan[valid] - i_sloan[valid]
    R_johnson[valid] = r_sloan[valid] - 0.2936 * ri - 0.1439
    I_johnson[valid] = r_sloan[valid] - 1.2444 * ri - 0.3820
    er, ei = err_r[valid], err_i[valid]
    err_R[valid] = np.sqrt((dR_dr * er) ** 2 + (dR_di * ei) ** 2)
    err_I[valid] = np.sqrt((dI_dr * er) ** 2 + (dI_di * ei) ** 2)

    return R_johnson, I_johnson, err_R, err_I


def johnson_bv_from_sloan_gr(
    g_sloan: np.ndarray,
    r_sloan: np.ndarray,
    err_g: np.ndarray | None = None,
    err_r: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Johnson B, V from Sloan g, r (Lupton 2005).

    B = g + 0.3130*(g - r) + 0.2271
    V = g - 0.5784*(g - r) - 0.0038
    """
    g_sloan = np.asarray(g_sloan, dtype=float)
    r_sloan = np.asarray(r_sloan, dtype=float)
    n = g_sloan.shape
    B = np.full(n, np.nan)
    V = np.full(n, np.nan)
    eB = np.full(n, np.nan)
    eV = np.full(n, np.nan)
    err_g = _default_err(g_sloan, err_g)
    err_r = _default_err(r_sloan, err_r)
    valid = np.isfinite(g_sloan) & np.isfinite(r_sloan)
    gr = g_sloan[valid] - r_sloan[valid]
    B[valid] = g_sloan[valid] + 0.3130 * gr + 0.2271
    V[valid] = g_sloan[valid] - 0.5784 * gr - 0.0038
    eg, er = err_g[valid], err_r[valid]
    eB[valid] = np.sqrt((1.3130 * eg) ** 2 + (0.3130 * er) ** 2)
    eV[valid] = np.sqrt((0.4216 * eg) ** 2 + (0.5784 * er) ** 2)
    return B, V, eB, eV


def johnson_u_from_sloan_ug_and_b(
    u_sloan: np.ndarray,
    g_sloan: np.ndarray,
    b_johnson: np.ndarray,
    err_u: np.ndarray | None = None,
    err_g: np.ndarray | None = None,
    err_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Johnson U from Sloan u, g and Johnson B (Jester et al. 2005 all-stars).

    U - B = 0.78*(u - g) - 0.88  →  U = B + 0.78*(u - g) - 0.88
    """
    u_sloan = np.asarray(u_sloan, dtype=float)
    g_sloan = np.asarray(g_sloan, dtype=float)
    b_johnson = np.asarray(b_johnson, dtype=float)
    n = b_johnson.shape
    U = np.full(n, np.nan)
    eU = np.full(n, np.nan)
    err_u = _default_err(u_sloan, err_u)
    err_g = _default_err(g_sloan, err_g)
    err_b = _default_err(b_johnson, err_b)
    valid = np.isfinite(u_sloan) & np.isfinite(g_sloan) & np.isfinite(b_johnson)
    ug = u_sloan[valid] - g_sloan[valid]
    U[valid] = b_johnson[valid] + 0.78 * ug - 0.88
    eu, eg, eb = err_u[valid], err_g[valid], err_b[valid]
    eU[valid] = np.sqrt(eb**2 + (0.78 * eu) ** 2 + (0.78 * eg) ** 2)
    return U, eU


def johnson_ubvri_from_sloan_arrays(
    *,
    u: np.ndarray | None = None,
    g: np.ndarray | None = None,
    r: np.ndarray | None = None,
    i: np.ndarray | None = None,
    err_u: np.ndarray | None = None,
    err_g: np.ndarray | None = None,
    err_r: np.ndarray | None = None,
    err_i: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build Johnson UBVRI (+ errors) from available Sloan arrays.

    Returns a dict ``{filter: (mag, err)}`` for bands that can be formed.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if g is not None and r is not None:
        B, V, eB, eV = johnson_bv_from_sloan_gr(g, r, err_g, err_r)
        out["B"] = (B, eB)
        out["V"] = (V, eV)
        if u is not None:
            U, eU = johnson_u_from_sloan_ug_and_b(u, g, B, err_u, err_g, eB)
            out["U"] = (U, eU)
    if r is not None and i is not None:
        R, I, eR, eI = johnson_ri_from_sloan_ri(r, i, err_r, err_i)
        out["R"] = (R, eR)
        out["I"] = (I, eI)
    return out


def add_johnson_ri_from_sloan(
    table: Table,
    *,
    r_col: str = "r'mag",
    i_col: str = "i'mag",
    err_r_col: str | None = "e_r'mag",
    err_i_col: str | None = "e_i'mag",
    inplace: bool = False,
    skip_if_johnson_present: bool = True,
) -> Table:
    """
    Add ``Rmag``/``Imag`` and ``e_Rmag``/``e_Imag`` columns from Sloan r'/i'.

    Intended for **raw Vizier column names** (e.g. APASS ``r'mag``) *before*
    conversion to the standard ``mag_std_*`` schema. If ``skip_if_johnson_present``
    and ``Rmag`` already exists, returns unchanged (avoid double transformation).
    """
    if not inplace:
        table = table.copy()
    if skip_if_johnson_present and "Rmag" in table.colnames:
        return table
    if r_col not in table.colnames or i_col not in table.colnames:
        return table

    r_sloan = np.array(table[r_col], dtype=float)
    i_sloan = np.array(table[i_col], dtype=float)
    err_r = (
        np.array(table[err_r_col], dtype=float)
        if err_r_col and err_r_col in table.colnames
        else None
    )
    err_i = (
        np.array(table[err_i_col], dtype=float)
        if err_i_col and err_i_col in table.colnames
        else None
    )

    R_j, I_j, eR, eI = johnson_ri_from_sloan_ri(r_sloan, i_sloan, err_r, err_i)
    table["Rmag"] = R_j
    table["Imag"] = I_j
    table["e_Rmag"] = eR
    table["e_Imag"] = eI
    return table


def add_johnson_ri_to_standard_table(
    table: Table,
    *,
    inplace: bool = False,
    skip_if_present: bool = True,
) -> Table:
    """
    Add ``mag_std_R``/``err_std_R`` and ``mag_std_I``/``err_std_I`` from Sloan bands.

    Expects ``mag_std_r``, ``mag_std_i`` (Sloan) and optional ``err_std_r``/``err_std_i``.
    """
    if not inplace:
        table = table.copy()
    if skip_if_present and "mag_std_R" in table.colnames:
        return table
    if "mag_std_r" not in table.colnames or "mag_std_i" not in table.colnames:
        return table

    r_sloan = np.array(table["mag_std_r"], dtype=float)
    i_sloan = np.array(table["mag_std_i"], dtype=float)
    err_r = (
        np.array(table["err_std_r"], dtype=float)
        if "err_std_r" in table.colnames
        else None
    )
    err_i = (
        np.array(table["err_std_i"], dtype=float)
        if "err_std_i" in table.colnames
        else None
    )

    R_j, I_j, eR, eI = johnson_ri_from_sloan_ri(r_sloan, i_sloan, err_r, err_i)
    table["mag_std_R"] = R_j
    table["mag_std_I"] = I_j
    table["err_std_R"] = eR
    table["err_std_I"] = eI
    return table
