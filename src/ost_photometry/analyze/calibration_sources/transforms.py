"""
Magnitude system transforms for calibration catalog tables.

**Lupton (2005)** empirical relations map SDSS/Sloan *r′*, *i′* to Johnson–Cousins
*R*, *I*. They are widely used for APASS and SDSS-based Vizier catalogs that lack
native Johnson R/I.

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
#   R = r - 0.2936 * (r - i) - 0.1439
#   I = r - 1.2444 * (r - i) - 0.3820
# Partial derivatives ∂R/∂r, ∂R/∂i, ∂I/∂r, ∂I/∂i for σ_R, σ_I from σ_r, σ_i.


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

    if err_r is None:
        err_r = np.full_like(r_sloan, 0.02)
    else:
        err_r = np.asarray(err_r, dtype=float)
    if err_i is None:
        err_i = np.full_like(i_sloan, 0.02)
    else:
        err_i = np.asarray(err_i, dtype=float)

    valid = np.isfinite(r_sloan) & np.isfinite(i_sloan)
    # Analytic Jacobian entries (constants for Lupton linear-in-(r-i) form)
    dR_dr, dR_di = 0.7064, 0.2936
    dI_dr, dI_di = -0.2444, 1.2444
    ri = r_sloan[valid] - i_sloan[valid]
    R_johnson[valid] = r_sloan[valid] - 0.2936 * ri - 0.1439
    I_johnson[valid] = r_sloan[valid] - 1.2444 * ri - 0.3820
    er, ei = err_r[valid], err_i[valid]
    # σ_f² ≈ (∂f/∂r σ_r)² + (∂f/∂i σ_i)² (uncorrelated)
    err_R[valid] = np.sqrt((dR_dr * er) ** 2 + (dR_di * ei) ** 2)
    err_I[valid] = np.sqrt((dI_dr * er) ** 2 + (dI_di * ei) ** 2)

    return R_johnson, I_johnson, err_R, err_I


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

    Parameters
    ----------
    table : Table
        Must contain ``r_col`` and ``i_col``.
    inplace : bool
        If False, copy first.
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
    Used after APASS (or similar) is mapped into the standard schema with lowercase
    Sloan keys for r′/i′.
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
