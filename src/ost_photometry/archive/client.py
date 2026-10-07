"""HTTP client for the OST data archive (Django REST API).

Authentication uses the archive's session cookie and CSRF token (token API
clients are not supported by the archive). Public runs can be read
anonymously, but the anonymous throttle is 60 requests per minute, so a
login is recommended for downloads of single files.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

#: Production archive (``FORCE_SCRIPT_NAME=/data_archive``).
DEFAULT_URL = "https://polaris.astro.physik.uni-potsdam.de/data_archive"

#: Environment variables for credentials (never stored by the client).
ENV_USER = "OST_ARCHIVE_USER"
ENV_PASSWORD = "OST_ARCHIVE_PASSWORD"

#: Requests per minute the client allows itself (server: 60 anon / 600 user).
RATE_ANONYMOUS = 50
RATE_AUTHENTICATED = 500

_RETRY_STATUS = frozenset({429, 502, 503, 504})


class ArchiveError(RuntimeError):
    """An archive request failed (HTTP error, bad response, checksum mismatch)."""


class RateLimiter:
    """Keep at most ``rate_per_minute`` requests per minute (minimum spacing)."""

    def __init__(
        self,
        rate_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate_per_minute = float(rate_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self.rate_per_minute <= 0:
            return
        interval = 60.0 / self.rate_per_minute
        now = self._clock()
        if self._last is not None:
            remaining = self._last + interval - now
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


class ArchiveClient:
    """Read-only client for runs, objects, data files and downloads."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        session: requests.Session | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        rate_per_minute: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ) -> None:
        """``log`` receives a line for every retry (timeouts, HTTP 429 / 5xx)
        so that long waits are visible."""
        self.base_url = base_url.rstrip("/")
        self._log = log
        self.session = session if session is not None else requests.Session()
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._sleep = sleep
        self._fixed_rate = rate_per_minute
        self._limiter = RateLimiter(
            rate_per_minute if rate_per_minute is not None else RATE_ANONYMOUS,
            clock=clock,
            sleep=sleep,
        )
        self.authenticated = False
        self.username: str | None = None

    # ------------------------------------------------------------------
    # URL handling
    # ------------------------------------------------------------------

    def url(self, path: str) -> str:
        """Absolute URL for an API path (``runs/runs/``) or a returned link.

        Links returned by the API are relative to the server root without
        the ``/data_archive`` prefix (e.g. ``/api/runs/datafiles/1/download/``);
        both forms are accepted.
        """
        if path.startswith(("http://", "https://")):
            return path
        prefix = urlsplit(self.base_url).path.rstrip("/")
        if path.startswith("/"):
            if prefix and path.startswith(prefix + "/"):
                path = path[len(prefix):]
            return self.base_url + path
        return urljoin(self.base_url + "/api/", path)

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
        timeout_retries: int | None = None,
    ) -> requests.Response:
        """``timeout_retries`` limits the retries after connection errors /
        timeouts (default ``max_retries``); HTTP 429 / 5xx always get
        ``max_retries``."""
        url = self.url(path)
        retries = self.max_retries
        timeout_retries = retries if timeout_retries is None else int(timeout_retries)
        attempt = 0
        while True:
            self._limiter.wait()
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=dict(headers or {}),
                    timeout=self.timeout,
                    stream=stream,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= timeout_retries:
                    raise ArchiveError(f"{method} {url} failed: {exc}") from exc
                delay = min(2.0**attempt, 60.0)
                self._retry_note(method, url, type(exc).__name__, attempt, delay,
                                 timeout_retries)
                self._sleep(delay)
                attempt += 1
                continue
            if response.status_code in _RETRY_STATUS and attempt < retries:
                delay = _retry_after_seconds(response, default=min(2.0**attempt, 60.0))
                response.close()
                self._retry_note(method, url, f"HTTP {response.status_code}", attempt, delay,
                                 retries)
                self._sleep(delay)
                attempt += 1
                continue
            if response.status_code >= 400:
                detail = _response_detail(response)
                response.close()
                raise ArchiveError(
                    f"{method} {url} returned HTTP {response.status_code}: {detail}"
                )
            return response

    def _retry_note(self, method: str, url: str, reason: str, attempt: int,
                    delay: float, retries: int) -> None:
        if self._log is not None:
            self._log(f"{method} {url}: {reason}, retry {attempt + 1}/{retries} "
                      f"in {delay:.0f} s")

    def get_json(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout_retries: int | None = None,
    ) -> Any:
        response = self._request("GET", path, params=params, timeout_retries=timeout_retries)
        try:
            return response.json()
        except ValueError as exc:
            raise ArchiveError(f"GET {self.url(path)} did not return JSON") from exc

    def _paged(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        limit: int = 200,
        min_limit: int = 25,
    ) -> list[dict]:
        """All results of a DRF page-number list (``count`` / ``results``).

        The ``next`` links are not followed because behind the reverse proxy
        they may carry the wrong scheme; pages are counted instead. A page
        that times out is requested again with half the page size (down to
        ``min_limit``, then with the normal retries): slow servers answer
        smaller pages in time.
        """
        results: list[dict] = []
        page = 1
        while True:
            query = dict(params or {})
            query.update({"page": page, "limit": limit})
            try:
                payload = self.get_json(path, query,
                                        timeout_retries=0 if limit > min_limit else None)
            except ArchiveError as exc:
                if limit <= min_limit or not isinstance(exc.__cause__, requests.Timeout):
                    raise
                limit = max(min_limit, limit // 2)
                # the full pages read so far are a multiple of the new size
                page = len(results) // limit + 1
                if self._log is not None:
                    self._log(f"GET {self.url(path)}: timeout, retrying with "
                              f"{limit} entries per page")
                continue
            if isinstance(payload, list):
                return payload
            batch = list(payload.get("results", []))
            results.extend(batch)
            count = int(payload.get("count", len(results)))
            if not batch or len(results) >= count or not payload.get("next"):
                return results
            page += 1

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def login(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        prompt: bool = True,
    ) -> None:
        """Start an authenticated session.

        Credentials come from the arguments, then from ``OST_ARCHIVE_USER`` /
        ``OST_ARCHIVE_PASSWORD``, then (``prompt=True``) from an interactive
        prompt. The password is never stored or logged.
        """
        username = username or os.environ.get(ENV_USER)
        password = password or os.environ.get(ENV_PASSWORD)
        if not username and prompt:
            username = input("Archive user name: ").strip()
        if not password and prompt:
            password = getpass.getpass(f"Archive password for {username}: ")
        if not username or not password:
            raise ArchiveError(
                f"No archive credentials; set {ENV_USER} / {ENV_PASSWORD} or log in "
                "interactively."
            )
        token = self._csrf_token()
        try:
            self._request(
                "POST",
                "users/auth/login/",
                json={"username": username, "password": password},
                headers={"X-CSRFToken": token, "Referer": self.base_url + "/"},
            )
        except ArchiveError as exc:
            raise ArchiveError(f"Archive login failed for {username!r}: {exc}") from None
        self.authenticated = True
        self.username = username
        if self._fixed_rate is None:
            self._limiter.rate_per_minute = RATE_AUTHENTICATED

    def logout(self) -> None:
        if not self.authenticated:
            return
        token = self._csrf_token()
        try:
            self._request(
                "POST",
                "users/auth/logout/",
                headers={"X-CSRFToken": token, "Referer": self.base_url + "/"},
            )
        finally:
            self.authenticated = False
            self.username = None
            if self._fixed_rate is None:
                self._limiter.rate_per_minute = RATE_ANONYMOUS

    def _csrf_token(self) -> str:
        payload = self.get_json("users/auth/csrf/")
        token = payload.get("csrfToken") if isinstance(payload, dict) else None
        if not token:
            raise ArchiveError("Archive did not return a CSRF token.")
        return str(token)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def runs(self, **filters: Any) -> list[dict]:
        """Observation runs; filters as in the archive (``name``, ``target``, ``ordering``)."""
        params = {k: v for k, v in filters.items() if v is not None}
        return self._paged("runs/runs/", params)

    def run(self, pk: int) -> dict:
        return self.get_json(f"runs/runs/{int(pk)}/")

    def find_run(self, name: str) -> dict:
        """The run called ``name`` (exact, case-insensitive)."""
        candidates = self.runs(name=name)
        exact = [r for r in candidates if str(r.get("name", "")).lower() == name.lower()]
        if len(exact) == 1:
            return exact[0]
        if not exact and len(candidates) == 1:
            return candidates[0]
        names = ", ".join(sorted(str(r.get("name")) for r in candidates)[:10]) or "none"
        raise ArchiveError(f"Run {name!r} is ambiguous or unknown; candidates: {names}")

    def datafiles(self, *, run_pk: int | None = None, **filters: Any) -> list[dict]:
        """Data files, e.g. all files of one run (``run_pk``)."""
        params = {k: v for k, v in filters.items() if v is not None}
        if run_pk is not None:
            params["observation_run"] = int(run_pk)
        return self._paged("runs/datafiles/", params)

    def datafile(self, pk: int) -> dict:
        """One data-file record (with ``content_hash``)."""
        return self.get_json(f"runs/datafiles/{int(pk)}/")

    def find_calibration_frames(
        self,
        kind: str,
        *,
        ccd_temp: float,
        instrument: str,
        naxis1: int,
        naxis2: int,
        exptime: float | None = None,
        binning_x: int = 1,
        binning_y: int = 1,
        gain: float | None = None,
        offset: float | None = None,
        exptime_tolerance: float = 0.5,
        temp_tolerance: float = 2.0,
        limit: int = 100,
    ) -> list[dict]:
        """Dark or bias frames of public runs matching a camera setup.

        Uses the archive's dark finder (``frame_type`` ``dark`` / ``bias``;
        bias ignores the exposure time). Needs a login. The archive filters
        gain / offset only when they are positive and knows no readout
        mode; callers check those on the results / headers. At most
        ``limit`` (<= 100) results, newest first.
        """
        if kind not in ("dark", "bias"):
            raise ValueError(f"kind must be 'dark' or 'bias', got {kind!r}")
        if not self.authenticated:
            raise ArchiveError("The archive dark finder needs a login.")
        if kind == "dark" and not (exptime and exptime > 0):
            raise ValueError("darks need an exposure time")
        payload: dict[str, Any] = {
            "frame_type": kind,
            "exptime": float(exptime) if kind == "dark" else 0.0,
            "exptime_tolerance": float(exptime_tolerance),
            "ccd_temp": float(ccd_temp),
            "temp_tolerance": float(temp_tolerance),
            "instrument": str(instrument),
            "naxis1": int(naxis1),
            "naxis2": int(naxis2),
            "binning_x": int(binning_x),
            "binning_y": int(binning_y),
            "limit": int(min(max(limit, 1), 100)),
        }
        if gain is not None and gain > 0:
            payload["gain"] = float(gain)
        if offset is not None and offset > 0:
            payload["offset"] = int(offset)
        token = self._csrf_token()
        response = self._request(
            "POST",
            "runs/dark-finder/",
            json=payload,
            headers={"X-CSRFToken": token, "Referer": self.base_url + "/"},
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise ArchiveError("The dark finder did not return JSON") from exc
        results = list(body.get("results", [])) if isinstance(body, dict) else []
        # Archives without frame_type support answer every request with darks.
        return [r for r in results if r.get("frame_type", "dark") == kind]

    def find_darks(self, *, exptime: float, **kwargs: Any) -> list[dict]:
        """Dark frames matching a camera setup, see :meth:`find_calibration_frames`."""
        return self.find_calibration_frames("dark", exptime=exptime, **kwargs)

    def search_objects(self, text: str, *, limit: int = 50) -> list[dict]:
        """Objects whose name or identifiers match ``text``."""
        payload = self.get_json("objects/vuetify", {"search": text, "limit": limit, "page": 1})
        if isinstance(payload, dict):
            return list(payload.get("items", []))
        return list(payload or [])

    def object_runs(self, obj_pk: int) -> list[dict]:
        return list(self.get_json(f"objects/{int(obj_pk)}/observation_runs/") or [])

    def object_datafiles(self, obj_pk: int) -> list[dict]:
        return list(self.get_json(f"objects/{int(obj_pk)}/datafiles/") or [])

    def header(self, pk: int) -> dict:
        payload = self.get_json(f"runs/datafiles/{int(pk)}/header/")
        return dict(payload.get("header", {})) if isinstance(payload, dict) else {}

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        pk: int,
        dest: str | Path,
        *,
        expected_sha256: str | None = None,
        chunk_size: int = 1 << 20,
    ) -> Path:
        """Download one data file to ``dest`` and verify its SHA-256.

        The file is written to ``<dest>.part`` first and renamed only after
        the checksum matched, so an interrupted download never leaves a
        truncated file behind.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        digest = hashlib.sha256()
        response = self._request("GET", f"runs/datafiles/{int(pk)}/download/", stream=True)
        try:
            with open(part, "wb") as handle:
                for block in response.iter_content(chunk_size=chunk_size):
                    if block:
                        handle.write(block)
                        digest.update(block)
        finally:
            response.close()
        checksum = digest.hexdigest()
        if expected_sha256 and checksum.lower() != str(expected_sha256).lower():
            part.unlink(missing_ok=True)
            raise ArchiveError(
                f"Checksum mismatch for data file {pk}: expected {expected_sha256}, got {checksum}"
            )
        part.replace(dest)
        return dest


def _retry_after_seconds(response: requests.Response, *, default: float) -> float:
    value = response.headers.get("Retry-After")
    if value is None:
        return default
    try:
        return max(0.0, min(float(value), 300.0))
    except ValueError:
        return default


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "")[:200]
    if isinstance(payload, dict):
        for key in ("detail", "error", "message"):
            if key in payload:
                return str(payload[key])[:200]
    return str(payload)[:200]


__all__ = [
    "ArchiveClient",
    "ArchiveError",
    "DEFAULT_URL",
    "ENV_PASSWORD",
    "ENV_USER",
    "RATE_ANONYMOUS",
    "RATE_AUTHENTICATED",
    "RateLimiter",
    "sha256_of_file",
]
