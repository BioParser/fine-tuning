"""This module does polite HTTP requests. It has these functions:

- a rate limit for each host
- a new request after a temporary error
- a cache on the disk for the API answers
- a size limit for downloads
"""

from __future__ import annotations

import hashlib
import json
import time
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


@dataclass
class Fetched:
    url: str  # The last URL after all redirects
    status: int
    content_type: str
    body: bytes


class PoliteClient:
    def __init__(
        self,
        *,
        email: str | None = None,
        cache_dir: Path | None = None,
        delay: float = 1.0,
        timeout: float = 60.0,
        refresh: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ):
        self.email = email
        self.cache_dir = cache_dir
        self.delay = delay
        self.refresh = refresh
        self._sleep = sleep
        self._last: dict[str, float] = {}
        ua = f"bioparser-ft-fetch/{__version__}" + (f" (mailto:{email})" if email else "")
        self.api_headers = {"User-Agent": ua, "Accept": "application/json"}
        # One client has one cookie jar. For this reason, the PDF request sends the cookies of the landing page.
        self.http = httpx.Client(
            follow_redirects=True,
            max_redirects=10,
            timeout=httpx.Timeout(timeout, connect=20.0),
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    # ------------------------------------------------------------------ internal functions
    def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        wait = self._last.get(host, 0.0) + self.delay - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last[host] = time.monotonic()

    def _request(self, url: str, *, params=None, headers=None, attempts: int = 3) -> httpx.Response | None:
        for attempt in range(attempts):
            self._throttle(url)
            try:
                resp = self.http.get(url, params=params, headers=headers)
            except httpx.HTTPError:
                if attempt == attempts - 1:
                    return None
                self._sleep(2 ** (attempt + 1))
                continue
            if resp.status_code in RETRY_STATUS and attempt < attempts - 1:
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
                self._sleep(min(wait, 60.0))
                continue
            return resp
        return None

    def _cache_path(self, url: str, params: dict | None) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps([url, sorted((params or {}).items())], default=str).encode()
        ).hexdigest()
        return self.cache_dir / "api" / f"{key[:2]}" / f"{key}.json"

    # ------------------------------------------------------------------ public functions
    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None) -> Any | None:
        """Send a GET request to a JSON API.

        Return None for a 404 answer or an error. The cache keeps the 200 and 404 answers.
        """
        path = self._cache_path(url, params)
        if path and path.exists() and not self.refresh:
            cached = json.loads(path.read_text(encoding="utf-8"))
            return cached["data"]
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
            path.write_text(json.dumps({"url": str(resp.url), "data": data}), encoding="utf-8")
        return data

    def get_web(
        self, url: str, *, referer: str | None = None, max_bytes: int = MAX_PDF_BYTES
    ) -> Fetched | None:
        """Send a GET request like a browser, for landing pages and PDF files.

        The answer comes as a stream. The function stops when the answer is larger than max_bytes.
        """
        if urlsplit(url).scheme not in ("http", "https"):
            return None
        headers = dict(WEB_HEADERS)
        if referer:
            headers["Referer"] = referer
        for attempt in range(2):
            self._throttle(url)
            try:
                with self.http.stream("GET", url, headers=headers) as resp:
                    chunks, size = [], 0
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
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
