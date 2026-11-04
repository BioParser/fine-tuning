"""This module is the browser mode (Playwright) for sites that use JavaScript or a click by a person.

Auto mode (headless): the code opens the landing page and accepts the cookie banner. Then it
follows the citation_pdf_url of the page, and it clicks the PDF and Download controls like a person.

Manual mode (visible window): you click, solve a challenge, or log in with your own institutional
access. The code captures the PDF file (a download or the PDF viewer) and gives it the correct name.
The code keeps the browser profile on the disk. For this reason, a library login is still valid
for the next papers.

The code does not solve CAPTCHAs and it does not go around paywalls. A person makes these decisions.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .pdfcheck import is_pdf

COOKIE_BUTTONS = (
    "#onetrust-accept-btn-handler",
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "button:has-text('Accept all')",
    "button:has-text('Accept cookies')",
    "button:has-text('Allow all')",
    "button:has-text('I accept')",
    "button:has-text('I agree')",
)
PDF_TRIGGERS = (
    "a:has-text('Download PDF')",
    "a:has-text('View PDF')",
    "button:has-text('Download PDF')",
    "a[href$='.pdf']",
    "a[href*='/pdf']",
    "a:has-text('PDF')",
    "button:has-text('PDF')",
    "a:has-text('Download')",
)
_META_JS = """() => {
  for (const n of ['citation_pdf_url', 'bepress_citation_pdf_url', 'eprints.document_url']) {
    const m = document.querySelector(`meta[name="${n}"]`);
    if (m && m.content) return new URL(m.content, document.baseURI).href;
  }
  return null;
}"""


class _StdinLines:
    """One thread reads the keyboard input for the full run.

    When you push Enter, the code goes to the next paper.
    """

    def __init__(self) -> None:
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        try:
            for line in sys.stdin:
                self.q.put(line)
        except (OSError, ValueError):
            # No stdin to read (Jupyter, pytest or a closed stdin): only the time limit operates
            return

    def clear(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()

    def pressed(self) -> bool:
        try:
            self.q.get_nowait()
            return True
        except queue.Empty:
            return False


class BrowserFetcher:
    def __init__(
        self,
        *,
        profile_dir: Path,
        headless: bool,
        executable_path: str | None = None,
        channel: str | None = None,
        nav_timeout_s: float = 45.0,
    ):
        self.profile_dir = profile_dir
        self.headless = headless
        self.executable_path = executable_path
        self.channel = channel
        self.nav_timeout_ms = nav_timeout_s * 1000
        self._responses: list[Any] = []
        self._downloads: list[Any] = []
        self._stdin: _StdinLines | None = None

    def __enter__(self) -> BrowserFetcher:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise SystemExit(
                "Playwright is not installed: run `uv sync --group fetch` and `uv run playwright install chromium`"
            ) from exc
        self._pw = sync_playwright().start()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "user_data_dir": str(self.profile_dir),
            "headless": self.headless,
            "accept_downloads": True,
        }
        if self.executable_path:
            kwargs["executable_path"] = self.executable_path
        if self.channel:
            kwargs["channel"] = self.channel
        self.ctx = self._pw.chromium.launch_persistent_context(**kwargs)
        self.ctx.set_default_navigation_timeout(self.nav_timeout_ms)
        self.ctx.on("page", self._attach)
        for page in self.ctx.pages:
            self._attach(page)
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.ctx.close()
        finally:
            self._pw.stop()

    # ---------------------------------------------------------------- capture of the PDF file
    def _attach(self, page: Any) -> None:
        page.on("response", self._on_response)
        page.on("download", lambda d: self._downloads.append(d))

    def _on_response(self, resp: Any) -> None:
        content_type = (resp.headers.get("content-type") or "").lower()
        if "pdf" in content_type or resp.url.lower().split("?")[0].endswith(".pdf"):
            self._responses.append(resp)  # The code reads the body later, outside the event handler

    def _collect(self) -> tuple[bytes, str] | None:
        for dl in list(self._downloads):
            try:
                path = dl.path()  # This call waits until the download is complete
                data = Path(path).read_bytes() if path else b""
            except Exception:
                continue
            if is_pdf(data):
                return data, dl.url
        for resp in list(self._responses):
            try:
                data = resp.body() if resp.status == 200 else b""
            except Exception:
                continue
            if is_pdf(data):
                return data, resp.url
        return None

    def _pump(self, ms: int = 500) -> None:
        """Let Playwright handle its events.

        The sync API of Playwright handles events only during a Playwright call. If the code
        uses time.sleep() here, it does not see the download.
        """
        for page in reversed(self.ctx.pages):
            try:
                page.wait_for_timeout(ms)
                return
            except Exception:
                continue
        time.sleep(ms / 1000)

    def _wait(self, seconds: float) -> tuple[bytes, str] | None:
        deadline = time.monotonic() + seconds
        while True:
            if got := self._collect():
                return got
            if time.monotonic() >= deadline:
                return None
            self._pump()

    def _start(self, url: str) -> Any:
        from playwright.sync_api import Error as PlaywrightError

        self._responses, self._downloads = [], []
        page = self.ctx.new_page()
        # The error "Download is starting" (the URL is a file) or a slow page: keep the part that loaded
        with contextlib.suppress(PlaywrightError):
            page.goto(url, wait_until="domcontentloaded")
        return page

    def _close_new_pages(self, keep: int = 1) -> None:
        for page in self.ctx.pages[keep:]:
            with contextlib.suppress(Exception):
                page.close()

    # ---------------------------------------------------------------- modes
    def fetch_auto(self, url: str) -> tuple[bytes, str] | None:
        """Open the URL in the headless browser and get the PDF without help from a person.

        Return (bytes, url), or None.
        """
        from playwright.sync_api import Error as PlaywrightError

        page = self._start(url)
        try:
            if got := self._wait(3):
                return got
            for sel in COOKIE_BUTTONS:
                try:
                    loc = page.locator(sel)
                    if loc.count() and loc.first.is_visible():
                        loc.first.click(timeout=2000)
                        break
                except PlaywrightError:
                    continue
            try:
                meta = page.evaluate(_META_JS)
            except PlaywrightError:
                meta = None
            if meta:
                with contextlib.suppress(PlaywrightError):
                    page.goto(meta, wait_until="commit")
                if got := self._wait(10):
                    return got
                with contextlib.suppress(PlaywrightError):
                    page.go_back(wait_until="domcontentloaded")
            for sel in PDF_TRIGGERS:
                try:
                    loc = page.locator(sel)
                    if not loc.count() or not loc.first.is_visible():
                        continue
                    loc.first.click(timeout=4000)
                except PlaywrightError:
                    continue
                if got := self._wait(10):
                    return got
            return self._collect()
        finally:
            self._close_new_pages()

    def fetch_manual(self, url: str, label: str, timeout_s: float = 300) -> tuple[bytes, str] | None:
        """Open the URL in the visible browser and wait until the person gets the PDF.

        Return (bytes, url). Return None after the time limit, or when the person pushes Enter.
        """
        if self._stdin is None:
            self._stdin = _StdinLines()
        self._stdin.clear()
        self._start(url)
        print(
            f"\n>>> {label}\n    Opened: {url}\n    Get the PDF in the browser window. If it is necessary, "
            f"log in or click 'Download PDF'. Push Enter here to go to the next paper. "
            f"Time limit: {timeout_s:.0f} s.",
            flush=True,
        )
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline:
                if self._stdin.pressed():
                    print("    next paper", flush=True)
                    return None
                if got := self._collect():
                    print("    PDF captured", flush=True)
                    return got
                self._pump()
            print("    time limit: next paper", flush=True)
            return None
        finally:
            self._close_new_pages()
