"""Cluster membership helpers (no live Gaia / Vizier)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from helpers import isolated_sys_modules, load_module_from_path, pkg_src, stub_analyze_package


def _parse_cluster_selection_id(raw: str | None) -> int | None:
    """Mirror of ost_photometry.utilities.parse_cluster_selection_id."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def test_parse_cluster_selection_id_strips_control_characters():
    assert _parse_cluster_selection_id("\x180") == 0
    assert _parse_cluster_selection_id("  2 \n") == 2
    assert _parse_cluster_selection_id("") is None
    assert _parse_cluster_selection_id("abc") is None


def _membership():
    stub_analyze_package("utils")
    src = pkg_src() / "ost_photometry" / "analyze" / "utils"
    load_module_from_path(
        "ost_photometry.analyze.utils.duplicates",
        src / "duplicates.py",
    )
    return load_module_from_path(
        "ost_photometry.analyze.utils.cluster_membership",
        src / "cluster_membership.py",
    )


def test_gaia_quality_mask_ruwe_and_snr():
    with isolated_sys_modules():
        mod = _membership()
        pm = np.array([1.0, 1.0, 1.0, np.nan])
        plx = np.array([0.5, 0.5, 0.5, 0.5])
        ruwe = np.array([1.0, 2.0, 1.1, 1.0])
        err = np.array([0.05, 0.05, 0.5, 0.05])
        keep = mod.gaia_quality_mask(
            pm_ra=pm,
            pm_de=pm,
            plx=plx,
            ruwe=ruwe,
            plx_err=err,
            ruwe_max=1.4,
            plx_snr_min=5.0,
        )
        np.testing.assert_array_equal(keep, [True, False, False, False])


def test_plx_min_from_distance_and_epoch_shift():
    with isolated_sys_modules():
        mod = _membership()
        assert mod.plx_min_mas_from_distance_kpc(5.0) == pytest.approx(0.2)
        assert mod.plx_min_mas_from_distance_kpc(None) is None
        ra = np.array([180.0])
        dec = np.array([0.0])
        pm_ra = np.array([1000.0])  # 1 arcsec / yr
        pm_de = np.array([0.0])
        ra_new, dec_new = mod.propagate_gaia_positions(ra, dec, pm_ra, pm_de, 1.0)
        assert ra_new[0] == pytest.approx(180.0 + 1.0 / 3600.0)
        assert dec_new[0] == pytest.approx(0.0)
        ra0, _ = mod.propagate_gaia_positions(ra, dec, pm_ra, pm_de, 0.0)
        assert ra0[0] == pytest.approx(180.0)


def test_match_photometry_keeps_nearest_unique():
    with isolated_sys_modules():
        mod = _membership()
        obj = SkyCoord([10.0, 10.0 + 1.0 / 3600.0], [0.0, 0.0], unit="deg")
        gaia = SkyCoord([10.0], [0.0], unit="deg")
        id_img, id_gaia, sep = mod.match_photometry_to_gaia(
            obj, gaia, separation_arcsec=2.0
        )
        assert id_gaia.size == 1
        assert id_img[0] == 0
        assert sep[0] < 0.1


def test_gmm_membership_picks_tight_component():
    pytest.importorskip("sklearn")
    with isolated_sys_modules():
        rng = np.random.default_rng(0)
        n_c, n_f = 80, 120
        pm_ra = np.concatenate(
            [
                rng.normal(-1.0, 0.08, n_c),
                rng.normal(4.0, 3.0, n_f),
            ]
        )
        pm_de = np.concatenate(
            [
                rng.normal(-2.0, 0.08, n_c),
                rng.normal(-1.0, 3.0, n_f),
            ]
        )
        plx = np.concatenate(
            [
                rng.normal(0.5, 0.03, n_c),
                rng.normal(0.2, 0.4, n_f),
            ]
        )
        mod = _membership()
        result = mod.membership_from_astrometry(
            pm_ra,
            pm_de,
            plx,
            method="gmm",
            random_state=0,
        )
        assert result.method == "gmm"
        assert "tighter" in result.reason.lower() or "simbad" in result.reason.lower()
        assert result.p_mem.size == n_c + n_f
        core = result.p_mem[:n_c]
        field = result.p_mem[n_c:]
        assert np.median(core) > 0.7
        assert np.median(field) < 0.4
        simbad = mod.membership_from_astrometry(
            pm_ra,
            pm_de,
            plx,
            method="gmm",
            simbad_pm_ra=-1.0,
            simbad_pm_de=-2.0,
            simbad_plx=0.5,
            random_state=0,
        )
        assert np.median(simbad.p_mem[:n_c]) > 0.7
        assert "simbad" in simbad.reason.lower()


def test_flag_cluster_members_writes_p_mem():
    from helpers import stub_analyze_package as stub

    with isolated_sys_modules():
        stub()
        cmd = load_module_from_path(
            "ost_photometry.analyze.cmd_prepare",
            pkg_src() / "ost_photometry" / "analyze" / "cmd_prepare.py",
        )
        tbl = Table({"id": [1, 2, 3]})
        out = cmd.flag_cluster_members(
            tbl, np.array([1, 3]), p_mem_by_id={1: 0.9, 3: 0.6}
        )
        np.testing.assert_array_equal(out["is_cluster_member"], [True, False, True])
        np.testing.assert_allclose(out["cluster_p_mem"], [0.9, 0.0, 0.6])


def test_membership_diagnostics_write_files(tmp_path):
    pytest.importorskip("matplotlib")
    with isolated_sys_modules():
        stub_analyze_package("plots")
        load_module_from_path(
            "ost_photometry.output_layout",
            pkg_src() / "ost_photometry" / "output_layout.py",
        )
        mod = load_module_from_path(
            "ost_photometry.analyze.plots.cluster_membership",
            pkg_src() / "ost_photometry" / "analyze" / "plots" / "cluster_membership.py",
        )
        rng = np.random.default_rng(1)
        n_c, n_f = 40, 60
        pm_ra = np.concatenate([rng.normal(-1.0, 0.1, n_c), rng.normal(4.0, 2.0, n_f)])
        pm_de = np.concatenate([rng.normal(-2.0, 0.1, n_c), rng.normal(0.0, 2.0, n_f)])
        plx = np.concatenate([rng.normal(0.5, 0.04, n_c), rng.normal(0.2, 0.3, n_f)])
        gmag = rng.uniform(12.0, 18.0, n_c + n_f)
        p_mem = np.concatenate([np.full(n_c, 0.9), np.full(n_f, 0.1)])
        mod.plot_cluster_membership_diagnostics(
            output_dir=str(tmp_path),
            file_type="png",
            pm_ra=pm_ra,
            pm_de=pm_de,
            plx=plx,
            p_mem=p_mem,
            gmag=gmag,
            pmem_min=0.5,
            method="gmm",
            cluster_component=0,
            reason="tighter Gaussian",
            simbad_pm_ra=-1.0,
            simbad_pm_de=-2.0,
            simbad_plx=0.5,
        )
        cluster = tmp_path / "diagnostics" / "cluster"
        for stem in (
            "cluster_pm_members",
            "cluster_parallax",
            "cluster_pmem",
            "cluster_mu_plx_3d",
        ):
            assert (cluster / f"{stem}.png").is_file()
