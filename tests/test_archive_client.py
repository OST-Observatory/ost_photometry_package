"""Archive client and fetch logic against a fake HTTP session (no network)."""

from __future__ import annotations

import hashlib
import io
import json
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from astropy.io import fits

from ost_photometry.archive import client as client_mod
from ost_photometry.archive.cache import FileCache, frame_link_name, link_frames
from ost_photometry.archive.client import ArchiveClient, ArchiveError, RateLimiter
from ost_photometry.archive.fetch import (
    collect_records,
    fetch_dataset,
    is_calibration_candidate,
    is_light_record,
    record_to_row,
)

BASE = "https://example.org/data_archive"


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b"", headers=None):
        self.status_code = status
        self._payload = payload
        self._content = content
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def iter_content(self, chunk_size=1):
        stream = io.BytesIO(self._content)
        while True:
            block = stream.read(chunk_size)
            if not block:
                return
            yield block

    def close(self):
        pass


class FakeSession:
    """Routes ``path -> handler(method, params, json, headers)``."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None, stream=False):
        parts = urlsplit(url)
        path = parts.path
        query = dict(params or {})
        query.update({k: v[0] for k, v in parse_qs(parts.query).items()})
        self.calls.append((method, path, query, json, headers))
        handler = self.routes.get(path)
        if handler is None:
            return FakeResponse(404, {"detail": "Not found"})
        return handler(method, query, json, headers or {})


def _fits_bytes(imagetyp="Light Frame", filt="V", obj="M57", frame=0):
    header = fits.Header()
    header["FRAMENO"] = frame
    header["IMAGETYP"] = imagetyp
    header["FILTER"] = filt
    header["OBJECT"] = obj
    header["INSTRUME"] = "QHYCCD-Cameras-Capture"
    header["XBINNING"] = 3
    header["OFFSET"] = 5
    header["READOUTM"] = "Normal"
    header["DATE-OBS"] = "2022-03-09T01:00:00"
    header["EXPTIME"] = 60.0
    buffer = io.BytesIO()
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.uint16), header).writeto(buffer)
    return buffer.getvalue()


def _record(pk, run_pk, run_name, etype="LI", ml="UK", target="M57", content=b"x", **extra):
    record = {
        "pk": pk,
        "observation_run": run_pk,
        "observation_run_name": run_name,
        "file_name": f"f{pk}.fit",
        "file_type": "FITS",
        "exposure_type": etype,
        "effective_exposure_type": etype,
        "exposure_type_ml": ml,
        "exposure_type_ml_abstained": False,
        "exposure_type_user": None,
        "content_hash": hashlib.sha256(content).hexdigest(),
        "file_size": len(content),
        "hjd": 2459647.5 + pk * 0.001,
        "obs_date": "2022-03-09 01:00:00",
        "exptime": 60.0,
        "instrument": "QHY 600M",
        "telescope": "CDK20",
        "main_target": target,
        "main_object_name": target,
        "main_object_id": 7,
        "ra": 283.4,
        "dec": 33.0,
        "spectrograph": "N",
        "spectroscopy": False,
    }
    record.update(extra)
    return record


def test_url_handles_api_paths_and_returned_links():
    client = ArchiveClient(BASE, session=FakeSession({}), rate_per_minute=0)
    assert client.url("runs/runs/") == BASE + "/api/runs/runs/"
    assert client.url("/api/runs/datafiles/3/download/") == BASE + "/api/runs/datafiles/3/download/"
    assert client.url("/data_archive/api/x/") == BASE + "/api/x/"
    assert client.url("https://other/x") == "https://other/x"


def test_login_uses_csrf_and_raises_rate(monkeypatch):
    def csrf(method, params, body, headers):
        return FakeResponse(200, {"authenticated": False, "csrfToken": "tok"})

    def login(method, params, body, headers):
        assert method == "POST"
        assert headers["X-CSRFToken"] == "tok"
        assert headers["Referer"] == BASE + "/"
        if body["password"] != "secret":
            return FakeResponse(401, {"error": "Invalid credentials"})
        return FakeResponse(200, {"user": {"username": body["username"]}})

    session = FakeSession(
        {"/data_archive/api/users/auth/csrf/": csrf, "/data_archive/api/users/auth/login/": login}
    )
    client = ArchiveClient(BASE, session=session)
    with pytest.raises(ArchiveError, match="Invalid credentials"):
        client.login("alice", "wrong", prompt=False)
    assert not client.authenticated
    monkeypatch.setenv(client_mod.ENV_USER, "alice")
    monkeypatch.setenv(client_mod.ENV_PASSWORD, "secret")
    client.login(prompt=False)
    assert client.authenticated
    assert client._limiter.rate_per_minute == client_mod.RATE_AUTHENTICATED
    # The password never appears in anything the client keeps.
    assert "secret" not in repr(vars(client))


def test_login_without_credentials_and_prompt_disabled(monkeypatch):
    monkeypatch.delenv(client_mod.ENV_USER, raising=False)
    monkeypatch.delenv(client_mod.ENV_PASSWORD, raising=False)
    client = ArchiveClient(BASE, session=FakeSession({}))
    with pytest.raises(ArchiveError, match="No archive credentials"):
        client.login(prompt=False)


def test_paging_counts_pages_and_retry_after():
    pages = {1: [{"pk": 1}, {"pk": 2}], 2: [{"pk": 3}]}
    attempts = {"n": 0}

    def runs(method, params, body, headers):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResponse(429, {"detail": "slow down"}, headers={"Retry-After": "2"})
        page = int(params["page"])
        return FakeResponse(
            200,
            {"count": 3, "next": "http://wrong-scheme/next" if page == 1 else None,
             "results": pages[page]},
        )

    slept = []
    client = ArchiveClient(
        BASE, session=FakeSession({"/data_archive/api/runs/runs/": runs}),
        rate_per_minute=0, sleep=slept.append,
    )
    assert [r["pk"] for r in client.runs(ordering="name")] == [1, 2, 3]
    assert slept == [2.0]


def test_http_error_is_reported():
    client = ArchiveClient(BASE, session=FakeSession({}), rate_per_minute=0, max_retries=0)
    with pytest.raises(ArchiveError, match="HTTP 404"):
        client.run(99)


def test_rate_limiter_spaces_requests():
    now = {"t": 0.0}
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now["t"] += seconds

    limiter = RateLimiter(60, clock=lambda: now["t"], sleep=sleep)
    limiter.wait()
    limiter.wait()
    assert slept == [pytest.approx(1.0)]


def test_download_verifies_checksum(tmp_path):
    content = _fits_bytes()

    def download(method, params, body, headers):
        return FakeResponse(200, content=content)

    session = FakeSession({"/data_archive/api/runs/datafiles/5/download/": download})
    client = ArchiveClient(BASE, session=session, rate_per_minute=0)
    good = hashlib.sha256(content).hexdigest()
    path = client.download(5, tmp_path / "a.fit", expected_sha256=good)
    assert path.read_bytes() == content
    with pytest.raises(ArchiveError, match="Checksum mismatch"):
        client.download(5, tmp_path / "b.fit", expected_sha256="0" * 64)
    assert not (tmp_path / "b.fit").exists()
    assert not (tmp_path / "b.fit.part").exists()


def test_record_classification_helpers():
    assert is_light_record(_record(1, 1, "r", "LI", "LI"))
    assert is_light_record(_record(1, 1, "r", "LI", "UK"))
    assert not is_light_record(_record(1, 1, "r", "LI", "FL"))  # ML says flat
    assert is_calibration_candidate(_record(1, 1, "r", "LI", "FL"))
    assert is_calibration_candidate(_record(1, 1, "r", "UK", "UK"))
    assert not is_calibration_candidate(_record(1, 1, "r", "LI", "UK"))
    user_light = _record(1, 1, "r", "FL", "FL", exposure_type_user="LI")
    assert is_light_record(user_light) and not is_calibration_candidate(user_light)
    row = record_to_row(_record(1, 1, "r", ra=0.0, dec=0.0), role="target")
    assert np.isnan(row["ra"])  # archive placeholder 0/0 means unknown


def _archive_routes(files):
    """Two runs (science + neighbour) with lights, flats and a second target."""
    runs = [
        {"pk": 1, "name": "2022-03-08", "mid_observation_jd": 2459647.6},
        {"pk": 2, "name": "2022-03-10", "mid_observation_jd": 2459649.6},
        {"pk": 3, "name": "2022-05-01", "mid_observation_jd": 2459700.6},
    ]
    run_files = {
        1: [
            _record(10, 1, "2022-03-08", "LI", "LI", "M57", files[10]),
            _record(11, 1, "2022-03-08", "LI", "LI", "M104", files[11]),
            _record(12, 1, "2022-03-08", "FL", "FL", "-", files[12]),
            {**_record(13, 1, "2022-03-08"), "file_type": "JPG"},
        ],
        2: [
            _record(20, 2, "2022-03-10", "DA", "DA", "-", files[20]),
            _record(21, 2, "2022-03-10", "LI", "LI", "NGC 1", files[21]),
        ],
        3: [_record(30, 3, "2022-05-01", "FL", "FL", "-", files[30])],
    }

    def runs_view(method, params, body, headers):
        selected = [r for r in runs if not params.get("name") or params["name"] in r["name"]]
        return FakeResponse(200, {"count": len(selected), "next": None, "results": selected})

    def datafiles(method, params, body, headers):
        result = run_files[int(params["observation_run"])]
        return FakeResponse(200, {"count": len(result), "next": None, "results": result})

    routes = {
        "/data_archive/api/runs/runs/": runs_view,
        "/data_archive/api/runs/datafiles/": datafiles,
    }
    for pk, content in files.items():
        routes[f"/data_archive/api/runs/datafiles/{pk}/download/"] = (
            lambda m, p, b, h, c=content: FakeResponse(200, content=c)
        )
    return routes


def test_collect_records_run_mode_with_targets_and_neighbours():
    files = {pk: _fits_bytes(frame=pk) for pk in (10, 11, 12, 20, 21, 30)}
    client = ArchiveClient(BASE, session=FakeSession(_archive_routes(files)), rate_per_minute=0)
    rows = collect_records(client, run_name="2022-03-08", calib_window_days=3)
    roles = {r["frame_id"]: r["role"] for r in rows}
    assert roles == {
        "10": "target", "11": "target", "12": "calibration",
        "20": "calibration", "21": "context",
    }
    rows = collect_records(client, run_name="2022-03-08", targets=["m 57"], calib_window_days=3)
    roles = {r["frame_id"]: r["role"] for r in rows}
    assert roles["10"] == "target" and roles["11"] == "context"


def test_fetch_dataset_downloads_merges_headers_and_uses_cache(tmp_path):
    files = {
        pk: _fits_bytes(filt="B" if pk == 12 else "V", frame=pk)
        for pk in (10, 11, 12, 20, 21, 30)
    }
    session = FakeSession(_archive_routes(files))
    client = ArchiveClient(BASE, session=session, rate_per_minute=0)
    manifest, report = fetch_dataset(
        client, run_name="2022-03-08", calib_window_days=3, cache_dir=tmp_path / "cache"
    )
    assert report.failures == []
    assert report.n_downloaded == 4 and report.n_cached == 0
    assert report.skipped_non_fits == 1
    assert report.targets == {"M57": 1, "M104": 1}
    by_id = {str(r["frame_id"]): r for r in manifest}
    assert by_id["12"]["filter"] == "B"
    assert by_id["10"]["readoutm"] == "Normal"
    assert by_id["10"]["xbinning"] == 3
    assert by_id["21"]["downloaded"] == np.False_  # context: metadata only
    cached = FileCache(tmp_path / "cache").path_for(str(by_id["10"]["sha256"]))
    assert cached.is_file()
    _, report2 = fetch_dataset(
        client, run_name="2022-03-08", calib_window_days=3, cache_dir=tmp_path / "cache"
    )
    assert report2.n_downloaded == 0 and report2.n_cached == 4
    assert any("Targets" in line for line in report.lines())


def test_link_frames_names_are_unique(tmp_path):
    a = tmp_path / "a" / "img.fit"
    b = tmp_path / "b" / "img.fit"
    for p in (a, b):
        p.parent.mkdir()
        p.write_bytes(b"x")
    links = link_frames(
        [{"frame_id": "1", "file_name": "img.fit", "local_path": str(a)},
         {"frame_id": "2", "file_name": "img.fit", "local_path": str(b)},
         {"frame_id": "3", "file_name": "gone.fit", "local_path": str(tmp_path / "gone.fit")}],
        tmp_path / "links",
    )
    assert set(links) == {"1", "2"}
    assert links["1"].name == frame_link_name("1", "img.fit") != links["2"].name
    assert links["1"].resolve() == a.resolve()


def test_unreadable_download_is_reported_not_counted(tmp_path):
    files = {pk: _fits_bytes(frame=pk) for pk in (10, 11, 12, 20, 21, 30)}
    files[12] = b"this is not a FITS file"
    client = ArchiveClient(BASE, session=FakeSession(_archive_routes(files)), rate_per_minute=0)
    manifest, report = fetch_dataset(
        client, run_name="2022-03-08", calib_window_days=3, cache_dir=tmp_path / "c"
    )
    assert [name for name, _ in report.failures] == ["f12.fit"]
    assert report.n_downloaded == 3
    by_id = {str(r["frame_id"]): r for r in manifest}
    assert not by_id["12"]["downloaded"]
