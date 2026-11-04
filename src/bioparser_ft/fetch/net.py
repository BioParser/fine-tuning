"""This module does polite HTTP requests. It has these functions:

- a rate limit for each host, also when many threads send requests at the same time
- a limit for the number of open requests to each host
- a new request after a temporary error
- a cache on the disk for the API answers
- a size limit for downloads
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import os
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .. import __version__

# Many publisher sites refuse clients that are not browsers, also for open-access pages.
# For this reason, web requests (not API requests) identify as a desktop browser.
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
WEB_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_PDF_BYTES = 150 * 1024 * 1024
# httpx limits the time of each read, not the time of the full body. A server that sends one byte
# every 25 seconds can keep a worker for days. This is the limit for the full body.
MAX_DOWNLOAD_SECONDS = 300.0
# The minimum time in seconds between the starts of two requests to one host. The API values are
# below the published limits of the services. All other hosts (the publishers and the
# repositories) use the `delay` value, 1 second by default.
HOST_DELAYS = {
    "api.crossref.org": 0.2,
    "api.openalex.org": 0.2,
    "api.unpaywall.org": 0.2,
    "www.ebi.ac.uk": 0.2,
    "api.datacite.org": 0.2,
    "api.semanticscholar.org": 1.1,  # The shared pool of Semantic Scholar gives 1 request each second
    "api.core.ac.uk": 1.0,
    # doi.org only sends redirects. The publisher after the redirect has its own rate limit.
    "doi.org": 0.2,
    "dx.doi.org": 0.2,
}
RESOLVERS = {"doi.org", "dx.doi.org"}
MAX_PER_HOST = 2  # The maximum number of open requests to the first host of a URL at the same time


def host_of(url: httpx.URL) -> str:
    return url.host if url.port is None else f"{url.host}:{url.port}"


def is_public_host(host: str) -> bool:
    """Return False if the host is, or resolves to, a loopback, private or link-local address.

    A landing page, a redirect or an API record can send the client to an internal address,
    for example 169.254.169.254 (cloud metadata) or 192.168.1.1 (a router). If DNS cannot
    resolve the host, return True: httpx then gives the DNS error.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return True
    return all(ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]).is_global for info in infos)


@dataclass
class Fetched:
    url: str  # The last URL after all redirects
    status: int
    content_type: str
    body: bytes


class PoliteClient:
    """An HTTP client for many threads. All threads share the rate limits, the cache and the cookies."""

    def __init__(
        self,
        *,
        email: str | None = None,
        cache_dir: Path | None = None,
        delay: float = 1.0,
        timeout: float = 30.0,
        refresh: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_per_host: int = MAX_PER_HOST,
    ):
        self.email = email
        self.cache_dir = cache_dir
        self.delay = delay
        self.refresh = refresh
        self.max_per_host = max_per_host
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}  # The earliest start time of the next request to each host
        self._server_delay: dict[str, float] = {}  # From the rate-limit headers of the server
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        ua = f"bioparser-ft-fetch/{__version__}" + (f" (mailto:{email})" if email else "")
        self.api_headers = {"User-Agent": ua, "Accept": "application/json"}
        # One client has one cookie jar. For this reason, the PDF request sends the cookies of the
        # landing page. The connection pool and the cookie jar of httpx have locks for threads.
        hooks: list[Callable[[httpx.Request], None]] = [self._before_request]
        if transport is None:  # The real network. The tests use a mock transport and no DNS.
            hooks.insert(0, self._refuse_internal_host)
        self.http = httpx.Client(
            follow_redirects=True,
            max_redirects=10,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
            # The hooks run before each request and before each redirect. For this reason, the rate
            # limit of a publisher and the address check also apply to the requests through doi.org.
            event_hooks={"request": hooks},
        )

    def close(self) -> None:
        self.http.close()

    # ------------------------------------------------------------------ rate limits
    def delay_for(self, host: str) -> float:
        base = min(HOST_DELAYS[host], self.delay) if host in HOST_DELAYS else self.delay
        return max(base, self._server_delay.get(host, 0.0))

    def _before_request(self, request: httpx.Request) -> None:
        self._throttle(host_of(request.url))

    def _refuse_internal_host(self, request: httpx.Request) -> None:
        if not is_public_host(request.url.host):
            raise httpx.ConnectError(f"refused: {request.url.host} is not a public address", request=request)

    def _throttle(self, host: str) -> None:
        """Wait for the next free start time of the host.

        Each thread reserves its own start time. For this reason, two threads do not start
        requests to one host at the same moment.
        """
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next.get(host, 0.0))
            self._next[host] = start + self.delay_for(host)
        if start > now:
            self._sleep(start - now)

    def _pause_host(self, host: str, seconds: float) -> None:
        """Stop all requests to the host for some seconds, for example after an HTTP 429 answer."""
        with self._lock:
            self._next[host] = max(self._next.get(host, 0.0), time.monotonic() + seconds)

    def _slot(self, host: str) -> contextlib.AbstractContextManager[object]:
        if host in RESOLVERS:
            return contextlib.nullcontext()  # A redirect is fast: the publisher limits the open requests
        with self._lock:
            if host not in self._slots:
                self._slots[host] = threading.BoundedSemaphore(self.max_per_host)
            return self._slots[host]

    def _note_rate_limit(self, host: str, headers: httpx.Headers) -> None:
        """Obey the rate limit that the server gives, for example Crossref: 50 requests in 1 second."""
        limit = headers.get("x-rate-limit-limit", "").strip()
        interval = re.fullmatch(r"(\d+(?:\.\d+)?)s?", headers.get("x-rate-limit-interval", "").strip())
        if limit.isdigit() and int(limit) > 0 and interval:
            with self._lock:
                self._server_delay[host] = float(interval.group(1)) / int(limit)

    # ------------------------------------------------------------------ internal functions
    def _request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        attempts: int = 3,
    ) -> httpx.Response | None:
        host = host_of(httpx.URL(url))
        for attempt in range(attempts):
            try:
                with self._slot(host):  # The request hook does the rate limit
                    resp = self.http.get(url, params=params, headers=headers)
            except httpx.HTTPError:
                if attempt == attempts - 1:
                    return None
                self._sleep(2 ** (attempt + 1))
                continue
            final = host_of(resp.url)  # The host after the redirects
            self._note_rate_limit(final, resp.headers)
            if resp.status_code in RETRY_STATUS and attempt < attempts - 1:
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
                self._pause_host(final, min(wait, 60.0))  # All threads wait, not only this thread
                continue
            return resp
        return None

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps([url, sorted((params or {}).items())], default=str).encode()
        ).hexdigest()
        return self.cache_dir / "api" / f"{key[:2]}" / f"{key}.json"

    # ------------------------------------------------------------------ public functions
    def get_json(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> Any | None:
        """Send a GET request to a JSON API.

        Return None for a 404 answer or an error. The cache keeps the 200 and 404 answers.
        """
        path = self._cache_path(url, params)
        if path and path.exists() and not self.refresh:
            try:
                return json.loads(path.read_text(encoding="utf-8"))["data"]
            except (OSError, ValueError, KeyError):
                pass  # A damaged cache file: send the request again
        resp = self._request(url, params=params, headers={**self.api_headers, **(headers or {})})
        if resp is None or resp.status_code not in (200, 404):
            return None  # A temporary problem: do not keep it in the cache
        data = None
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                return None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
            # Do not write the API key to the disk. The URL in the file is only for persons to read.
            stored_url = resp.url.copy_remove_param("api_key")
            tmp.write_text(json.dumps({"url": str(stored_url), "data": data}), encoding="utf-8")
            tmp.replace(path)  # An atomic write: other threads read the old file or the new file
        return data

    def get_web(
        self,
        url: str,
        *,
        referer: str | None = None,
        max_bytes: int = MAX_PDF_BYTES,
        max_seconds: float = MAX_DOWNLOAD_SECONDS,
    ) -> Fetched | None:
        """Send a GET request like a browser, for landing pages and PDF files.

        The answer comes as a stream. The function stops when the answer is larger than max_bytes,
        or when the body takes more than max_seconds.
        """
        if urlsplit(url).scheme not in ("http", "https"):
            return None
        host = host_of(httpx.URL(url))
        headers = dict(WEB_HEADERS)
        if referer:
            headers["Referer"] = referer
        for attempt in range(2):
            try:
                # The request hook does the rate limit of each host, also after a redirect
                with self._slot(host), self.http.stream("GET", url, headers=headers) as resp:
                    deadline = time.monotonic() + max_seconds
                    chunks, size = [], 0
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes or time.monotonic() > deadline:
                            return None
                        chunks.append(chunk)
                    return Fetched(
                        url=str(resp.url),
                        status=resp.status_code,
                        content_type=resp.headers.get("content-type", "").split(";")[0].strip().lower(),
                        body=b"".join(chunks),
                    )
            except httpx.HTTPError:
                if attempt == 1:
                    return None
                self._sleep(2)
        return None
