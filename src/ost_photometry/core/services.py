"""External astronomy services with mirror failover.

CDS in Strasbourg runs VizieR, Simbad, Sesame (name resolution) and xMatch.
When CDS does not answer, queries go to the US mirrors instead:

* VizieR: ``vizier.cfa.harvard.edu``. The mirror serves the smaller
  catalogs itself (Stetson standards, VSX) but returns nothing for the large
  ones (APASS, Gaia, Gaia XP, SDSS). :func:`require_vizier_catalog` raises
  :class:`CatalogUnavailable` for those, so a caller can fall back (Gaia:
  the Gaia archives, :func:`gaia_archive_query`) instead of taking the field
  as empty.
* Simbad: ``simbad.harvard.edu``, a full copy.
* Object names: Sesame, then Simbad, then NED (:func:`resolve_name`).
* Gaia DR3 tables: the ESA archive, else the copy at ARI Heidelberg.

Every server is probed once per process with a one-row query
(``PROBE_TIMEOUT`` seconds); only an answer with data counts (CDS has also
answered 404 for every request, which is not "up"). The
environment variables ``OST_PHOTOMETRY_VIZIER_SERVER`` and
``OST_PHOTOMETRY_SIMBAD_SERVER`` fix a server without probing; a server set
in astroquery's own configuration is tried first.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import warnings

import numpy as np
import requests

from .. import terminal_output

CDS_VIZIER = "vizier.cds.unistra.fr"
VIZIER_MIRRORS = ("vizier.cfa.harvard.edu",)
CDS_SIMBAD = "simbad.cds.unistra.fr"
SIMBAD_MIRRORS = ("simbad.harvard.edu",)
#: TAP services with the Gaia DR3 tables, in the order tried. Synchronous
#: queries: the asynchronous jobs of the ESA archive stop after 60 s of
#: execution, which a busy archive exceeds even for a small cone.
GAIA_TAP_URLS = (
    "https://gea.esac.esa.int/tap-server/tap",
    "https://gaia.ari.uni-heidelberg.de/tap",
)
#: Connect and read timeout of a Gaia archive query (seconds).
GAIA_TAP_TIMEOUT = (15.0, 300.0)
PROBE_TIMEOUT = 8.0
VIZIER_SERVER_ENV = "OST_PHOTOMETRY_VIZIER_SERVER"
SIMBAD_SERVER_ENV = "OST_PHOTOMETRY_SIMBAD_SERVER"

#: One-row queries, and the bytes the answer must contain.
_PROBE = {
    "vizier": ("/viz-bin/votable?-source=B/vsx/vsx&-out.max=1", b"<TR>"),
    "simbad": (
        "/simbad/sim-tap/sync?request=doQuery&lang=adql&format=csv"
        "&query=SELECT%20TOP%201%20main_id%20FROM%20basic",
        b"main_id",
    ),
}
_MIRROR_NOTE = {
    "vizier": " (large catalogs such as APASS, Gaia and SDSS are not served there)",
    "simbad": "",
}
_reachable: dict[str, bool] = {}
_chosen: dict[str, str] = {}
_served: dict[tuple[str, str], bool] = {}


@contextlib.contextmanager
def _quiet_numpy_formatting():
    """numpy >= 2.5 warns inside astropy's vectorised formatting (angles of
    the query, masked VOTable cells): harmless."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in do_format")
        yield


class CatalogUnavailable(RuntimeError):
    """The catalog cannot be queried now (CDS down, not on the mirror)."""


def reset() -> None:
    """Forget the probe results (next query probes again)."""
    _reachable.clear()
    _chosen.clear()
    _served.clear()


def _host_reachable(kind: str, host: str) -> bool:
    if host not in _reachable:
        path, expected = _PROBE[kind]
        try:
            response = requests.get(f"https://{host}{path}", timeout=PROBE_TIMEOUT)
            _reachable[host] = response.status_code == 200 and expected in response.content
        except requests.RequestException:
            _reachable[host] = False
    return _reachable[host]


def _server(kind: str, env: str, cds: str, mirrors: tuple[str, ...], configured: str) -> str:
    forced = os.environ.get(env, "").strip()
    if forced:
        return forced
    if kind not in _chosen:
        candidates = list(dict.fromkeys([configured, cds, *mirrors]))
        server = next((h for h in candidates if _host_reachable(kind, h)), None)
        if server is None:
            server = candidates[0]
            terminal_output.print_to_terminal(
                f"None of {', '.join(candidates)} answers a test query: trying {server}",
                style_name="WARNING",
            )
        elif server != candidates[0]:
            terminal_output.print_to_terminal(
                f"{candidates[0]} does not answer: using the mirror {server}{_MIRROR_NOTE[kind]}",
                style_name="WARNING",
            )
        _chosen[kind] = server
    return _chosen[kind]


def vizier_server() -> str:
    """The VizieR server queries go to (CDS, else a reachable mirror)."""
    from astroquery.vizier import conf

    return _server("vizier", VIZIER_SERVER_ENV, CDS_VIZIER, VIZIER_MIRRORS, conf.server)


def simbad_server() -> str:
    """The Simbad server queries go to (CDS, else a reachable mirror)."""
    from astroquery.simbad import conf

    return _server("simbad", SIMBAD_SERVER_ENV, CDS_SIMBAD, SIMBAD_MIRRORS, conf.server)


def cds_reachable() -> bool:
    """CDS answers (VizieR as the probe; xMatch and Sesame are CDS too).
    Not when CDS is only the last resort because no server passed the probe."""
    server = vizier_server()
    if os.environ.get(VIZIER_SERVER_ENV, "").strip():
        return server == CDS_VIZIER
    return _reachable.get(CDS_VIZIER, server == CDS_VIZIER)


def use_vizier_server(vizier):
    """Point a ``Vizier`` instance at :func:`vizier_server`; returns it."""
    vizier.VIZIER_SERVER = vizier_server()
    return vizier


def use_simbad_server(simbad):
    """Point a ``Simbad`` instance at :func:`simbad_server`; returns it."""
    server = simbad_server()
    if getattr(simbad, "server", server) != server:
        simbad.server = server
    return simbad


def _catalog_served(server: str, catalog: str) -> bool:
    """One row of ``catalog`` from ``server`` (mirrors return none for the
    catalogs they do not have)."""
    from astroquery.vizier import Vizier

    probe = Vizier(row_limit=1, columns=["*"])
    probe.VIZIER_SERVER = server
    try:
        with _quiet_numpy_formatting():
            tables = probe.get_catalogs(catalog)
    except Exception:
        return False
    return bool(tables) and len(tables[0]) > 0


def require_vizier_catalog(catalog: str) -> None:
    """Raise :class:`CatalogUnavailable` when the VizieR server in use (a
    mirror, because CDS does not answer) does not serve ``catalog``."""
    server = vizier_server()
    if server == CDS_VIZIER:
        return
    key = (server, catalog)
    if key not in _served:
        _served[key] = _catalog_served(server, catalog)
    if not _served[key]:
        raise CatalogUnavailable(
            f"Vizier {catalog} is not available: CDS ({CDS_VIZIER}) does not answer "
            f"and the mirror {server} does not serve it"
        )


def _tap_sync(url: str, adql: str, maxrec: int):
    """Table of a synchronous TAP query; raises on HTTP or query errors."""
    from astropy.io.votable import parse

    response = requests.post(
        f"{url}/sync",
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable",
              "MAXREC": str(int(maxrec)), "QUERY": adql},
        timeout=GAIA_TAP_TIMEOUT,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        votable = parse(io.BytesIO(response.content))
    status = {
        info.name: (info.value, info.content)
        for resource in votable.resources
        for info in resource.infos
    }.get("QUERY_STATUS", ("OK", ""))
    if str(status[0]).upper() == "ERROR" or response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {(status[1] or '').strip()[:200]}")
    with _quiet_numpy_formatting():
        table = votable.get_first_table().to_table()
    #   the ESA archive answers SOURCE_ID, ARI Heidelberg source_id (the Gaia
    #   columns are lower case)
    table.rename_columns(table.colnames, [name.lower() for name in table.colnames])
    if str(status[0]).upper() == "OVERFLOW":
        terminal_output.print_to_terminal(
            f"Gaia archive: result truncated at {len(table)} rows",
            style_name="WARNING",
        )
    return table


_VIZIER_ERROR = re.compile(r'<INFO name="Error" value="([^"]*)"')


def query_vizier_region(vizier, coordinates, **kwargs):
    """``vizier.query_region(coordinates, **kwargs)``, but an error answer of
    VizieR raises :class:`CatalogUnavailable`: astroquery returns an empty
    result for it, which reads as "no stars in the field" (CDS has answered
    404 pages, the CfA mirror "No server is available for connection")."""
    server = getattr(vizier, "VIZIER_SERVER", "?")
    with _quiet_numpy_formatting():
        response = vizier.query_region_async(coordinates, **kwargs)
    if response.status_code != 200:
        raise CatalogUnavailable(f"Vizier ({server}) answered HTTP {response.status_code}")
    text = response.text
    if 'name="QUERY_STATUS" value="ERROR"' in text:
        detail = " ".join(m.strip() for m in _VIZIER_ERROR.findall(text) if m.strip())
        raise CatalogUnavailable(
            f"Vizier ({server}) answered with an error: {detail or 'no detail'}"
        )
    with _quiet_numpy_formatting():
        return vizier._parse_result(response)


def gaia_archive_query(adql: str, *, maxrec: int = 1_000_000):
    """Table of an ADQL query on the Gaia DR3 tables: the ESA archive, else
    the next service of :data:`GAIA_TAP_URLS` (synchronous queries, a
    warning when the archive truncates the result).

    Raises :class:`CatalogUnavailable` when no service answers.
    """
    errors = []
    for url in GAIA_TAP_URLS:
        try:
            return _tap_sync(url, adql, maxrec)
        except Exception as err:
            errors.append(f"{url}: {_short(err)}")
            if url != GAIA_TAP_URLS[-1]:
                terminal_output.print_to_terminal(
                    f"Gaia archive {url} failed ({_short(err)}); trying the next one",
                    style_name="WARNING",
                )
    raise CatalogUnavailable(f"Gaia archives not available ({'; '.join(errors)})")


def _first_position(table, ra: str, dec: str):
    from astropy.coordinates import SkyCoord

    if table is None or len(table) == 0 or ra not in table.colnames:
        return None
    ra_value = float(np.ma.filled(np.ma.asarray(table[ra][0], dtype=float), np.nan))
    dec_value = float(np.ma.filled(np.ma.asarray(table[dec][0], dtype=float), np.nan))
    if not (np.isfinite(ra_value) and np.isfinite(dec_value)):
        return None
    return SkyCoord(ra_value, dec_value, unit="deg", frame="icrs")


def _short(err: Exception) -> str:
    lines = [line.strip() for line in str(err).splitlines() if line.strip()]
    return lines[-1] if lines else type(err).__name__


def resolve_name(name: str):
    """ICRS position of an object name: Sesame (CDS), else Simbad (the
    mirror when CDS does not answer), else NED.

    Raises ``astropy.coordinates.name_resolve.NameResolveError`` when none
    of them knows the name.
    """
    from astropy.coordinates import SkyCoord, name_resolve
    from astropy.utils import data

    errors = []
    if cds_reachable():
        try:
            with data.conf.set_temp("remote_timeout", PROBE_TIMEOUT):
                return SkyCoord.from_name(name)
        except name_resolve.NameResolveError as err:
            errors.append(f"Sesame: {_short(err)}")
    try:
        from astroquery.simbad import Simbad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            position = _first_position(use_simbad_server(Simbad()).query_object(name), "ra", "dec")
        if position is not None:
            return position
        errors.append("Simbad: unknown name")
    except Exception as err:
        errors.append(f"Simbad: {_short(err)}")
    try:
        from astroquery.ipac.ned import Ned

        position = _first_position(Ned.query_object(name), "RA", "DEC")
        if position is not None:
            return position
        errors.append("NED: unknown name")
    except Exception as err:
        errors.append(f"NED: {_short(err)}")
    raise name_resolve.NameResolveError(f"Could not resolve {name!r} ({'; '.join(errors)})")


__all__ = [
    "CDS_SIMBAD",
    "CDS_VIZIER",
    "CatalogUnavailable",
    "GAIA_TAP_URLS",
    "SIMBAD_MIRRORS",
    "VIZIER_MIRRORS",
    "cds_reachable",
    "gaia_archive_query",
    "query_vizier_region",
    "require_vizier_catalog",
    "reset",
    "resolve_name",
    "simbad_server",
    "use_simbad_server",
    "use_vizier_server",
    "vizier_server",
]
