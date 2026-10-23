"""Stetson standards, ``calibration_source="auto"`` and the confusion cut on calibrators."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table

CENTER = SkyCoord(359.33, 56.72, unit="deg")


def _stetson_raw() -> Table:
    n = 6
    t = Table()
    t["RAJ2000"] = 359.3 + 0.01 * np.arange(n)
    t["DEJ2000"] = np.full(n, 56.7)
    t["Collect"] = ["Stetson", "Landolt", "Stetson", "Stetson", "Stetson", "SPSS"]
    t["VarFlag"] = MaskedColumn([0, 0, 1, 0, 0, 0], mask=[False, False, False, True, False, False])
    t["BinFlag"] = [0, 0, 0, 0, 1, 0]
    t["Bmag"] = MaskedColumn([13.5, 14.0, 13.0, 15.5, 13.2, 13.0], mask=[False, True, False, False, False, False])
    t["e_Bmag"] = MaskedColumn([0.004, 0.0, 0.004, 0.006, 0.004, 0.01], mask=[False, True, False, False, False, False])
    t["Vmag"] = [12.9, 13.4, 12.4, 19.0, 12.6, 12.5]
    t["e_Vmag"] = [0.003] * n
    return t


def test_stetson_keeps_standards_and_drops_flagged_stars(monkeypatch):
    from ost_photometry.analyze.calibration_sources import fetch

    class FakeVizier:
        def __init__(self, columns=None, row_limit=None):
            self.columns = columns

        def query_region(self, center, radius=None, catalog=None):
            assert catalog == fetch.STETSON_VIZIER_ID
            return [_stetson_raw()]

    monkeypatch.setattr(fetch, "Vizier", FakeVizier)
    out = fetch._fetch_stetson_standard(CENTER, 15.0, (10.0, 18.0), 1)
    # row 2 variable, row 4 binary, row 5 not a Stetson/Landolt standard,
    # row 3 V = 19 outside the range; the masked VarFlag of row 3 is no flag
    assert np.allclose(out["mag_std_V"], [12.9, 13.4])
    assert out["mag_std_B"][0] == 13.5 and np.isnan(out["mag_std_B"][1])
    assert np.isnan(out["err_std_B"][1])
    assert "mag_std_U" not in out.colnames or np.all(np.isnan(out["mag_std_U"]))


def _std(filters: dict[str, int], n: int = 60) -> Table:
    t = Table({"ra": np.linspace(0, 1, n), "dec": np.zeros(n)})
    for f, k in filters.items():
        mag = np.full(n, np.nan)
        mag[:k] = 14.0
        t[f"mag_std_{f}"] = mag
        t[f"err_std_{f}"] = np.full(n, 0.01)
    return t


@pytest.fixture
def fake_sources(monkeypatch):
    from ost_photometry.analyze.calibration_sources import fetch

    catalogs = {
        "Stetson": _std({"U": 30, "B": 40, "V": 50, "I": 45}),
        "APASS": _std({"B": 50, "V": 55, "g": 50, "r": 50, "i": 50, "R": 50, "I": 50}),
        "GaiaXP": _std({"U": 10, "B": 50, "V": 50, "R": 50, "I": 50, "u": 25, "g": 50, "r": 50, "i": 50, "z": 15}),
        "SDSS_Release_16": _std({"u": 30, "g": 40, "r": 40, "i": 40, "z": 35}),
    }
    calls = []

    def fake_fetch(filter_list, center, *, calibration_source, **kwargs):
        calls.append(calibration_source)
        if calibration_source == "broken":
            raise RuntimeError("no network")
        return catalogs[calibration_source].copy()

    monkeypatch.setattr(fetch, "fetch_standard_calibration_catalog", fake_fetch)
    return fetch, calls


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (["B", "V"], "Stetson"),
        (["V", "R"], "APASS"),  # Stetson has no R here
        (["g", "r"], "APASS"),
        (["u", "g"], "GaiaXP"),
        (["g", "z"], "SDSS_Release_16"),  # too few z in Gaia XP here
        (["B", "V", "Ha"], "Stetson"),  # narrow band needs no catalog
    ],
)
def test_auto_takes_the_first_source_covering_every_filter(fake_sources, filters, expected):
    fetch, calls = fake_sources
    out = fetch._fetch_auto_standard(
        filters, CENTER, auto_sources=fetch.DEFAULT_AUTO_SOURCES, auto_min_stars=20, indent=1,
    )
    assert out.meta[fetch.META_CALIBRATION_SOURCE] == expected
    assert calls[-1] == expected


def test_auto_skips_unavailable_sources_and_falls_back_to_the_best(fake_sources):
    fetch, calls = fake_sources
    out = fetch._fetch_auto_standard(
        ["B", "V"], CENTER, auto_sources=("broken", "Stetson"), auto_min_stars=20, indent=1,
    )
    assert out.meta[fetch.META_CALIBRATION_SOURCE] == "Stetson"
    # too few everywhere: the source with the most stars in its poorest filter
    out = fetch._fetch_auto_standard(
        ["B", "V"], CENTER, auto_sources=("Stetson", "APASS"), auto_min_stars=100, indent=1,
    )
    assert out.meta[fetch.META_CALIBRATION_SOURCE] == "APASS"
    with pytest.raises(RuntimeError, match="covers"):
        fetch._fetch_auto_standard(
            ["z"], CENTER, auto_sources=("Stetson", "APASS"), auto_min_stars=20, indent=1,
        )


def test_resolved_source_drives_the_magnitude_system():
    from types import SimpleNamespace

    from ost_photometry.analyze.pipeline.steps.calibration import effective_calibration_source
    from ost_photometry.analyze.post_processing.magnitude_systems import (
        resolve_catalog_magnitude_system,
    )

    cat = Table({"ra": [1.0], "dec": [2.0]})
    cat.meta["calibration_source"] = "SDSS_Release_16"
    config = SimpleNamespace(calibration_source="auto")
    assert effective_calibration_source(SimpleNamespace(calibration_catalog=cat), config) == "SDSS_Release_16"
    assert effective_calibration_source(SimpleNamespace(calibration_catalog=None), config) == "auto"
    assert resolve_catalog_magnitude_system("Stetson") == "vega"
    assert resolve_catalog_magnitude_system("SDSS_Release_16") == "ab"


def test_config_defaults_to_auto_with_stetson_first():
    from ost_photometry.analyze.pipeline import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.calibration_source == "auto"
    assert cfg.calibration_auto_sources == ("Stetson", "APASS", "GaiaXP", "SDSS_Release_16")
    assert cfg.calibrator_max_confusion_bias == 0.005


def test_confusion_biased_calibrators_are_dropped():
    from ost_photometry.analyze.calibration.quality import calibrator_candidate_mask

    n = 10
    bias = np.zeros(n)
    bias[[2, 7]] = 0.02  # faint calibrators in a crowded field
    bias[5] = np.nan  # no measurement: kept
    tbl = Table({"ra": 10.0 + 0.01 * np.arange(n), "dec": np.full(n, 20.0),
                 "mag_V": np.full(n, 13.0), "err_V": np.full(n, 0.01),
                 "mag_std_V": np.full(n, 12.5), "err_std_V": np.full(n, 0.02),
                 "confusion_bias_V": bias})
    kw = dict(error_p84_clip=False, photon_factor=None, apply_finder_shape_cuts=False,
              qfit_max=None, cfit_max=None)
    mask = calibrator_candidate_mask(tbl, ["V"], **kw)
    assert list(np.flatnonzero(~mask)) == [2, 7]
    assert np.all(calibrator_candidate_mask(tbl, ["V"], max_confusion_bias=None, **kw))


def test_gaia_xp_flags_excess_and_flux_errors(monkeypatch):
    from ost_photometry.analyze.calibration_sources import fetch

    raw = Table()
    raw["Source"] = [1, 2, 3, 4]
    raw["RA_ICRS"] = [359.3, 359.31, 359.32, 359.33]
    raw["DE_ICRS"] = [56.7] * 4
    raw["E(BP/RP)corr"] = MaskedColumn([0.0, 0.5, 0.01, 0.0], mask=[False, False, False, True])
    raw["Vmag"] = [14.0, 14.0, 15.0, 19.0]
    raw["FV"] = [1.0e-17] * 4
    raw["e_FV"] = [1.0e-19] * 4
    raw["VFlag"] = [1, 1, 1, 1]
    raw["umag"] = [16.0, 16.0, 17.0, 20.0]
    raw["Fu"] = [2.0e-30] * 4
    raw["e_Fu"] = [2.0e-31] * 4
    raw["uFlag"] = [1, 1, 0, 1]

    class FakeVizier:
        def __init__(self, columns=None, row_limit=None):
            pass

        def query_region(self, center, radius=None, catalog=None):
            assert catalog == fetch.GAIA_XP_VIZIER_ID
            return [raw]

    monkeypatch.setattr(fetch, "Vizier", FakeVizier)
    out = fetch._fetch_gaia_xp_standard(CENTER, 15.0, (10.0, 17.5), 1)
    # star 2: BP/RP excess 0.5 (blend) dropped; star 4: V = 19 outside the range
    assert list(out["gaia_xp_id"]) == ["1", "3"]
    assert out["err_std_V"][0] == pytest.approx(1.0857e-2)
    assert out["err_std_u"][0] == pytest.approx(1.0857e-1)
    # star 3: u outside the validated colour / magnitude range -> NaN
    assert np.isnan(out["mag_std_u"][1]) and out["mag_std_V"][1] == 15.0


def test_gaia_xp_sdss_bands_are_on_the_sdss_system():
    from ost_photometry.analyze.post_processing.magnitude_systems import (
        SDSS_CATALOGS,
        resolve_catalog_magnitude_system,
    )

    assert "GaiaXP" in SDSS_CATALOGS
    # UBVRI Vega, ugriz SDSS: the filter set decides, not the catalog
    assert resolve_catalog_magnitude_system("GaiaXP") == "unknown"
