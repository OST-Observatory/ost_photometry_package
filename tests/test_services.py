"""Mirror failover of the external services (no network: everything faked)."""

from __future__ import annotations

import io
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
import requests
from astropy.coordinates import SkyCoord, name_resolve
from astropy.table import Table

from ost_photometry.core import services

CENTER = SkyCoord(359.33, 56.72, unit="deg")


@pytest.fixture(autouse=True)
def _fresh_probe_state():
    services.reset()
    yield
    services.reset()


_PROBE_ANSWER = b'<TABLEDATA><TR><TD>1</TD></TR></TABLEDATA>\nmain_id\n"HD 1"'


def _fake_cds(monkeypatch, cds_answer):
    """Probe answers: ``cds_answer(url)`` for CDS, data for the mirrors."""
    monkeypatch.delenv(services.VIZIER_SERVER_ENV, raising=False)
    monkeypatch.delenv(services.SIMBAD_SERVER_ENV, raising=False)
    probed = []

    def fake_get(url, timeout=None):
        probed.append(url)
        if "unistra.fr" in url:
            return cds_answer(url)
        return SimpleNamespace(status_code=200, content=_PROBE_ANSWER)

    monkeypatch.setattr(services.requests, "get", fake_get)
    return probed


@pytest.fixture
def cds_down(monkeypatch):
    """CDS does not answer, the US mirrors do; returns the probed URLs."""
    def no_route(url):
        raise requests.ConnectionError("no route to host")

    return _fake_cds(monkeypatch, no_route)


def test_mirror_is_used_when_cds_does_not_answer(cds_down, monkeypatch):
    messages = []
    monkeypatch.setattr(services.terminal_output, "print_to_terminal",
                        lambda text, **kw: messages.append((text, kw.get("style_name"))))
    assert services.vizier_server() == "vizier.cfa.harvard.edu"
    assert services.simbad_server() == "simbad.harvard.edu"
    assert not services.cds_reachable()
    n_probes = len(cds_down)
    assert services.vizier_server() == "vizier.cfa.harvard.edu"
    assert len(cds_down) == n_probes  # probed once per process
    assert len(messages) == 2 and all(style == "WARNING" for _, style in messages)
    assert "APASS" in messages[0][0]  # large catalogs are not on the mirror

    vizier = services.use_vizier_server(SimpleNamespace(VIZIER_SERVER=services.CDS_VIZIER))
    assert vizier.VIZIER_SERVER == "vizier.cfa.harvard.edu"
    simbad = services.use_simbad_server(SimpleNamespace(server=services.CDS_SIMBAD))
    assert simbad.server == "simbad.harvard.edu"


def test_cds_answering_404_counts_as_down(monkeypatch):
    """CDS has answered every request with 404 (and a page that would pass
    a status check): only a probe answer with data counts."""
    _fake_cds(monkeypatch, lambda url: SimpleNamespace(status_code=404, content=b"<html>Not Found</html>"))
    monkeypatch.setattr(services.terminal_output, "print_to_terminal", lambda *a, **k: None)
    assert services.vizier_server() == "vizier.cfa.harvard.edu"
    assert services.simbad_server() == "simbad.harvard.edu"
    services.reset()
    _fake_cds(monkeypatch, lambda url: SimpleNamespace(status_code=200, content=b"<html>Maintenance</html>"))
    assert services.vizier_server() == "vizier.cfa.harvard.edu"
    services.reset()
    _fake_cds(monkeypatch, lambda url: SimpleNamespace(status_code=200, content=_PROBE_ANSWER))
    assert services.vizier_server() == services.CDS_VIZIER


def test_forced_server_is_not_probed(monkeypatch):
    monkeypatch.setenv(services.VIZIER_SERVER_ENV, "vizier.cfa.harvard.edu")

    def no_probe(*args, **kwargs):
        raise AssertionError("probe with a forced server")

    monkeypatch.setattr(services.requests, "get", no_probe)
    assert services.vizier_server() == "vizier.cfa.harvard.edu"


def test_catalog_not_on_the_mirror_is_unavailable(monkeypatch):
    monkeypatch.setattr(services, "vizier_server", lambda: "vizier.cfa.harvard.edu")
    checked = []

    def served(server, catalog):
        checked.append(catalog)
        return catalog == "J/A+A/664/A109/table5"

    monkeypatch.setattr(services, "_catalog_served", served)
    services.require_vizier_catalog("J/A+A/664/A109/table5")
    with pytest.raises(services.CatalogUnavailable, match="II/336/apass9"):
        services.require_vizier_catalog("II/336/apass9")
    with pytest.raises(services.CatalogUnavailable):
        services.require_vizier_catalog("II/336/apass9")
    assert checked == ["J/A+A/664/A109/table5", "II/336/apass9"]  # cached


def test_catalogs_on_cds_are_not_checked(monkeypatch):
    def no_check(server, catalog):
        raise AssertionError("catalog check on CDS")

    monkeypatch.setattr(services, "_catalog_served", no_check)
    services.require_vizier_catalog("II/336/apass9")


class _FakeSimbad:
    result = Table()

    def __init__(self):
        self.server = services.CDS_SIMBAD

    def query_object(self, name):
        return self.result


def _fake_ned(result=None, error=None):
    def query_object(name):
        if error is not None:
            raise error
        return result

    return SimpleNamespace(query_object=query_object)


def _no_sesame(*args, **kwargs):
    raise AssertionError("Sesame is CDS: not asked when CDS does not answer")


def test_names_resolve_via_simbad_then_ned_without_cds(monkeypatch):
    import astroquery.ipac.ned
    import astroquery.simbad

    monkeypatch.setattr(services, "cds_reachable", lambda: False)
    monkeypatch.setattr(SkyCoord, "from_name", _no_sesame)
    monkeypatch.setattr(astroquery.simbad, "Simbad", _FakeSimbad)
    monkeypatch.setattr(_FakeSimbad, "result", Table({"ra": [111.9164536], "dec": [24.3365323]}))
    position = services.resolve_name("HAT-P-20")
    assert position.ra.deg == pytest.approx(111.9164536) and position.dec.deg == pytest.approx(24.3365323)

    monkeypatch.setattr(_FakeSimbad, "result", Table({"ra": [], "dec": []}))
    monkeypatch.setattr(astroquery.ipac.ned, "Ned", _fake_ned(Table({"RA": [359.35025], "DEC": [56.70829]})))
    assert services.resolve_name("NGC 7789").ra.deg == pytest.approx(359.35025)

    monkeypatch.setattr(astroquery.ipac.ned, "Ned", _fake_ned(error=RuntimeError("line 1\nnot recognized")))
    with pytest.raises(name_resolve.NameResolveError, match="Simbad: unknown name; NED: not recognized"):
        services.resolve_name("no such object")


def _esa_gaia_xp_table() -> Table:
    """The four stars of the Vizier test, as the ESA archive returns them."""
    esa = Table()
    esa["source_id"] = [1, 2, 3, 4]
    esa["ra"] = [359.3, 359.31, 359.32, 359.33]
    esa["dec"] = [56.7] * 4
    esa["c_star"] = np.ma.MaskedArray([0.0, 0.5, 0.01, 0.0], mask=[False, False, False, True])
    for prefix in ("u_jkc", "b_jkc", "v_jkc", "r_jkc", "i_jkc", "u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss"):
        esa[f"{prefix}_mag"] = [np.nan] * 4
        esa[f"{prefix}_flux"] = [np.nan] * 4
        esa[f"{prefix}_flux_error"] = [np.nan] * 4
        esa[f"{prefix}_flag"] = [False] * 4
    esa["v_jkc_mag"] = [14.0, 14.0, 15.0, 19.0]
    esa["v_jkc_flux"] = [1.0e-17] * 4
    esa["v_jkc_flux_error"] = [1.0e-19] * 4
    esa["v_jkc_flag"] = [True] * 4
    esa["u_sdss_mag"] = [16.0, 16.0, 17.0, 20.0]
    esa["u_sdss_flux"] = [2.0e-30] * 4
    esa["u_sdss_flux_error"] = [2.0e-31] * 4
    esa["u_sdss_flag"] = [True, True, False, True]
    return esa


def test_gaia_xp_comes_from_the_esa_archive_when_vizier_cannot(monkeypatch):
    from ost_photometry.analyze.calibration_sources import fetch

    def unavailable(catalog):
        raise services.CatalogUnavailable(f"Vizier {catalog} is not available")

    queries = []

    def esa_query(adql, **kwargs):
        queries.append(adql)
        return _esa_gaia_xp_table()

    monkeypatch.setattr(services, "require_vizier_catalog", unavailable)
    monkeypatch.setattr(services, "gaia_archive_query", esa_query)
    out = fetch._fetch_gaia_xp_standard(CENTER, 15.0, (10.0, 17.5), 1)
    assert "gaiadr3.synthetic_photometry_gspc" in queries[0]
    assert "CIRCLE('ICRS', 359.33000000, 56.72000000, 0.25000000)" in queries[0]
    # as from Vizier: excess 0.5 dropped, V = 19 outside the range, u flag 0 -> NaN
    assert list(out["gaia_xp_id"]) == ["1", "3"]
    assert out["err_std_V"][0] == pytest.approx(1.0857e-2)
    assert out["err_std_u"][0] == pytest.approx(1.0857e-1)
    assert np.isnan(out["mag_std_u"][1]) and out["mag_std_V"][1] == 15.0
    assert np.all(np.isnan(out["mag_std_B"]))


def test_apass_is_unavailable_without_a_query(monkeypatch):
    from ost_photometry.analyze.calibration_sources import fetch

    def unavailable(catalog):
        raise services.CatalogUnavailable(f"Vizier {catalog} is not available")

    def no_vizier(*args, **kwargs):
        raise AssertionError("Vizier queried for an unavailable catalog")

    monkeypatch.setattr(services, "require_vizier_catalog", unavailable)
    monkeypatch.setattr(fetch, "Vizier", no_vizier)
    with pytest.raises(services.CatalogUnavailable, match="II/336/apass9"):
        fetch._fetch_apass9_standard(CENTER, 15.0, ["B", "V"], (10.0, 17.5), 1)


def test_gaia_dr3_cone_comes_from_the_esa_archive_when_vizier_cannot(monkeypatch):
    from ost_photometry.analyze.utils import cluster_selection

    def unavailable(catalog):
        raise services.CatalogUnavailable(f"Vizier {catalog} is not available")

    queries = []

    def esa_query(adql, **kwargs):
        queries.append(adql)
        return Table({name: [1.0, 2.0] for name in cluster_selection.GAIA_DR3_ESA_COLUMNS.values()})

    monkeypatch.setattr(services, "require_vizier_catalog", unavailable)
    monkeypatch.setattr(services, "gaia_archive_query", esa_query)
    monkeypatch.setattr(cluster_selection.terminal_output, "print_to_terminal", lambda *a, **k: None)
    tbl = cluster_selection.query_gaia_dr3_cone(CENTER, 30 * u.arcmin, g_mag_limit=19.0)
    assert tbl.colnames == list(cluster_selection.GAIA_DR3_COLUMNS)
    assert "phot_g_mean_mag < 19.0" in queries[0] and "0.50000000" in queries[0]

    with pytest.raises(services.CatalogUnavailable):
        cluster_selection.query_gaia_dr3_cone(CENTER, 30 * u.arcmin, catalog="I/350/gaiaedr3")


def test_known_variables_skip_xmatch_when_cds_does_not_answer(monkeypatch):
    from ost_photometry.analyze.calibration_sources import known_variables

    def no_xmatch(*args, **kwargs):
        raise AssertionError("xMatch is CDS")

    monkeypatch.setattr(services, "cds_reachable", lambda: False)
    monkeypatch.setattr(known_variables, "query_vsx_xmatch", no_xmatch)
    monkeypatch.setattr(known_variables, "_drop_via_cone_search", lambda *a, **k: np.array([1]))
    monkeypatch.setattr(known_variables.terminal_output, "print_to_terminal", lambda *a, **k: None)
    catalog = Table({"ra": [359.30, 359.31, 359.32], "dec": [56.7] * 3})
    out = known_variables.drop_catalog_rows_near_known_variables(catalog, CENTER, 15.0)
    assert list(out["ra"]) == [359.30, 359.32]


def _votable_bytes(table: Table, status: str | None = None) -> bytes:
    from astropy.io.votable import from_table
    from astropy.io.votable.tree import Info

    votable = from_table(table)
    if status is not None:
        votable.resources[0].infos.append(Info(name="QUERY_STATUS", value=status))
    buffer = io.BytesIO()
    votable.to_xml(buffer)
    return buffer.getvalue()


def test_gaia_archive_falls_back_to_the_next_service(monkeypatch):
    """ESA fails (timeout, or an error answer): ARI Heidelberg serves the
    same tables; synchronous queries with a timeout."""
    messages = []
    monkeypatch.setattr(services.terminal_output, "print_to_terminal",
                        lambda text, **kw: messages.append(text))
    posted = []

    def fake_post(url, data=None, timeout=None):
        posted.append((url, data["QUERY"], timeout))
        if "esac" in url:
            raise requests.ReadTimeout("read timed out")
        #   the ESA archive answers SOURCE_ID: lower case like ARI's columns
        return SimpleNamespace(status_code=200, content=_votable_bytes(Table({"SOURCE_ID": [1, 2]})))

    monkeypatch.setattr(services.requests, "post", fake_post)
    table = services.gaia_archive_query("SELECT source_id FROM gaiadr3.gaia_source")
    assert list(table["source_id"]) == [1, 2]
    assert [url for url, _, _ in posted] == [f"{u}/sync" for u in services.GAIA_TAP_URLS]
    assert all(timeout == services.GAIA_TAP_TIMEOUT for _, _, timeout in posted)
    assert len(messages) == 1 and "read timed out" in messages[0]

    def error_then_overflow(url, data=None, timeout=None):
        if "esac" in url:
            return SimpleNamespace(status_code=400, content=_votable_bytes(Table({"x": [0]}), "ERROR"))
        return SimpleNamespace(status_code=200, content=_votable_bytes(Table({"source_id": [5]}), "OVERFLOW"))

    messages.clear()
    monkeypatch.setattr(services.requests, "post", error_then_overflow)
    assert list(services.gaia_archive_query("SELECT ...")["source_id"]) == [5]
    assert "HTTP 400" in messages[0] and "truncated at 1 rows" in messages[1]


def test_gaia_archive_unavailable_when_no_service_answers(monkeypatch):
    def down(url, data=None, timeout=None):
        raise requests.ConnectionError(f"{url} unreachable")

    monkeypatch.setattr(services.requests, "post", down)
    monkeypatch.setattr(services.terminal_output, "print_to_terminal", lambda *a, **k: None)
    with pytest.raises(services.CatalogUnavailable, match="ari.uni-heidelberg"):
        services.gaia_archive_query("SELECT ...")


class _FakeVizierAnswer:
    """Vizier double: the answer text of ``query_region_async``."""

    VIZIER_SERVER = "vizier.cfa.harvard.edu"

    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def query_region_async(self, coordinates, **kwargs):
        return SimpleNamespace(status_code=self.status_code, text=self.text)

    def _parse_result(self, response):
        return ["parsed"]


def test_vizier_error_answer_is_not_an_empty_field():
    """The CfA mirror answered B/vsx/vsx with QUERY_STATUS ERROR; astroquery
    turns that into an empty result ("no variables in the field")."""
    error = (
        '<INFO name="Error" value="Couldn\'t connect with server"/>'
        '<INFO name="Error" value=" "/>'
        '<INFO name="Error" value="No server is available for connection"/>'
        '<INFO name="QUERY_STATUS" value="ERROR"></INFO>'
    )
    with pytest.raises(services.CatalogUnavailable,
                       match="cfa.harvard.edu.*Couldn't connect with server No server is available"):
        services.query_vizier_region(_FakeVizierAnswer(error), CENTER, radius=15 * u.arcmin)
    ok = '<INFO name="QUERY_STATUS" value="OK"/><TABLE></TABLE>'
    assert services.query_vizier_region(_FakeVizierAnswer(ok), CENTER, radius=15 * u.arcmin) == ["parsed"]
    #   CDS has answered with 404 pages, which astroquery reads as "no tables"
    with pytest.raises(services.CatalogUnavailable, match="HTTP 404"):
        services.query_vizier_region(_FakeVizierAnswer("<html>Not Found</html>", 404), CENTER)


def test_warning_when_no_server_passes_the_probe(monkeypatch):
    messages = []
    monkeypatch.setattr(services.terminal_output, "print_to_terminal",
                        lambda text, **kw: messages.append(text))
    _fake_cds(monkeypatch, lambda url: SimpleNamespace(status_code=404, content=b""))
    monkeypatch.setattr(services.requests, "get",
                        lambda url, timeout=None: SimpleNamespace(status_code=503, content=b""))
    assert services.vizier_server() == services.CDS_VIZIER
    assert len(messages) == 1 and "None of vizier.cds.unistra.fr, vizier.cfa.harvard.edu" in messages[0]
    assert not services.cds_reachable()  # CDS only as the last resort: no xMatch, no Sesame
