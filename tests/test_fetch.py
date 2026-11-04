"""Offline tests for bioparser_ft.fetch. No request goes out of the computer.

An httpx.MockTransport gives the answers of the scholarly APIs and the publishers.
The browser tests use pages from a local HTTP server. They operate only when Playwright
and a Chromium binary are available (BIOPARSER_CHROMIUM=/path/to/chrome).
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import ets_row, write_ets

httpx = pytest.importorskip("httpx")
pytest.importorskip("bs4")
pytest.importorskip("pypdf")

from bioparser_ft.fetch.citations import group_papers, parse_citation  # noqa: E402
from bioparser_ft.fetch.cli import build_parser, run  # noqa: E402
from bioparser_ft.fetch.landing import pdf_links_from_html, publisher_pdf_urls  # noqa: E402
from bioparser_ft.fetch.matching import verify  # noqa: E402
from bioparser_ft.fetch.net import PoliteClient  # noqa: E402
from bioparser_ft.fetch.pdfcheck import check_pdf  # noqa: E402

FILLER = " The study reports measurements of skull and body size for many specimens." * 6


def make_pdf(text: str, pad: int = 6000) -> bytes:
    """Return a minimal PDF file with one page and a text layer that pypdf can read."""
    content = f"BT /F1 11 Tf 50 700 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n%" + b"x" * pad + b"\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


# ----------------------------------------------------------------------- citations
@pytest.mark.parametrize(
    ("citation", "author", "year", "title_start", "thesis"),
    [
        ("Doe, J. and Roe, R., 2020. Bats of the north (Chiroptera). J. Mamm., 101(2), pp.10-20.",
         "Doe", 2020, "Bats of the north", False),
        ("Lee, K., 2022. How heavy is a fossil? A new method for body mass. Evolution, 5(1), pp.1-9.",
         "Lee", 2022, "How heavy is a fossil? A new method", False),
        ("Roy, P. and Das, A. (2023). Sexual dimorphism in rats, Rattus rattus (Linnaeus, 1758). J. Zool., 5 (2): 3-4.",
         "Roy", 2023, "Sexual dimorphism in rats", False),
        ("Molinari, J., A global assessment of the island rule in bats. Journal of Biogeography.",
         "Molinari", None, "A global assessment of the island rule", False),
        ("Smith, P. and Owen, R.D., Notes on a mouse, Mus pearsoni (Myers, 1977) in the Chaco.",
         "Smith", None, "Notes on a mouse", False),
        ('Garcia, Ana, and Bo Li. "Systematics of sheath-tailed bats of the genus Saccopteryx." Zootaxa 12 (2023): 1-5.',
         "Garcia", 2023, "Systematics of sheath-tailed bats", False),
        ("Kim, A.C., 2023. Functional significance of the mandible (Doctoral dissertation, Ohio University).",
         "Kim", 2023, "Functional significance of the mandible", True),
        ("Park, J. and Wu, O., 2023. Long-fingered bat Myotis capaccinii. In Handbook of Bats (pp. 1-32). Cham: Springer.",
         "Park", 2023, "Long-fingered bat Myotis capaccinii", False),
        ("Khan, T.R. and Ali, F.A., 2022. Zaib Khan and Iftikhar Ahmad. Record of a pipistrelle from Pakistan. Pure Appl. Biol., 11(1), p.5.",
         "Khan", 2022, "Record of a pipistrelle from Pakistan", False),
    ],
)  # fmt: skip
def test_parse_citation_formats(citation, author, year, title_start, thesis):
    a, y, title, is_thesis, _ = parse_citation(citation)
    assert (a, y, is_thesis) == (author, year, thesis)
    assert title.startswith(title_start), title


def test_doi_in_citation():
    *_, doi = parse_citation(
        "Dahal D.R, Thapa S. First records of a bat. J. Bat Res. DOI: 10.14709/BarbJ.15.1.2022.05."
    )
    assert doi == "10.14709/barbj.15.1.2022.05"


def test_verify_levels():
    cit = "Doe, J. and Roe, R., 2020. Echolocation calls of cave bats in northern Somewhere. J. Mamm., 1(2), pp.1-2."
    title = "Echolocation calls of cave bats in northern Somewhere"
    assert verify(cit, title, "Doe", 2020, title, 2020, ["Doe", "Roe"]).confidence == "high"
    # The online year and the print year can be different
    assert verify(cit, title, "Doe", 2020, title, 2021, ["Doe"]).confidence == "high"
    assert verify(cit, title, "Doe", 2020, title, 2015, ["Doe"]).confidence == "medium"
    assert verify(cit, title, "Doe", 2020, "Chiroptera", 2020, ["Doe"]).confidence == "low"  # A general title
    assert (
        verify(cit, title, "Doe", 2020, "Echolocation of dolphins in the sea", 2020, ["Doe"]).confidence
        == "low"
    )


# ----------------------------------------------------------------------- landing pages
def test_pdf_links_from_html_priorities():
    html = """<html><head>
      <meta name="citation_pdf_url" content="/content/article.pdf">
      <link rel="alternate" type="application/pdf" href="https://cdn.example.org/alt.pdf"></head><body>
      <a href="javascript:void(0)">PDF</a>
      <a href="/suppl/table_S1.pdf">Supplementary PDF</a>
      <a href="/doi/pdf/10.1/x">Download PDF</a>
      <a href="/about">About</a></body></html>"""
    links = pdf_links_from_html(html, "https://journal.example.org/doi/10.1/x")
    assert links[0] == "https://journal.example.org/content/article.pdf"  # The meta tag comes first
    assert links[1] == "https://cdn.example.org/alt.pdf"
    assert links[-1].endswith("table_S1.pdf")  # The supplements come last
    assert not any("about" in u or "javascript" in u for u in links)


def test_bepress_and_figshare_links():
    html = (
        '<meta name="bepress_citation_pdf_url" content="https://docs.lib.edu/cgi/viewcontent.cgi?article=1">'
        '<a href="https://hammer.example.edu/ndownloader/files/123">Download (2 MB)</a>'
    )
    links = pdf_links_from_html(html, "https://docs.lib.edu/etd/1")
    assert links == [
        "https://docs.lib.edu/cgi/viewcontent.cgi?article=1",
        "https://hammer.example.edu/ndownloader/files/123",
    ]


def test_publisher_patterns():
    assert publisher_pdf_urls("10.1007/s42991-023-00001-1") == [
        "https://link.springer.com/content/pdf/10.1007/s42991-023-00001-1.pdf"
    ]
    assert "pdfdirect" in publisher_pdf_urls("10.1111/jbi.14000")[0]
    assert publisher_pdf_urls("10.7717/peerj.12345") == ["https://peerj.com/articles/12345.pdf"]
    elsevier = publisher_pdf_urls("10.1016/j.sjbs.2022.103490", {"alternative-id": ["S1319562X22007082"]})
    assert elsevier == [
        "https://www.sciencedirect.com/science/article/pii/S1319562X22007082/pdfft?isDTMRedir=true&download=true"
    ]
    assert publisher_pdf_urls("10.9999/unknown") == []


# ----------------------------------------------------------------------- PDF file checks
def test_check_pdf():
    title = "Echolocation calls of cave bats in northern Somewhere"
    good = check_pdf(make_pdf(title + FILLER), title, "10.1000/a")
    assert good.is_pdf and good.verified and good.pages == 1 and good.title_recall == 1.0
    wrong = check_pdf(
        make_pdf("Population genetics of marine snails in the Baltic Sea" + FILLER), title, None
    )
    assert wrong.usable and not wrong.verified
    assert not check_pdf(b"<html>Access denied</html>", title, None).is_pdf
    # A very small file is an error page
    assert "bytes" in check_pdf(make_pdf(title, pad=0), title, None).reason


# ----------------------------------------------------------------------- end to end
CIT_A = "Doe, J. and Roe, R., 2020. Echolocation calls of cave bats in northern Somewhere. Journal of Mammalogy, 101(2), pp.10-20."
CIT_B = "Smith, A., 2021. Skull morphology of field mice from Elsewhere. Zootaxa, 5000(1), pp.1-5."
CIT_C = "Nobody, Q., 2019. An article that no index knows about at all. Obscure Journal, 1(1), pp.1-2."
TITLE_A = "Echolocation calls of cave bats in northern Somewhere"
TITLE_B = "Skull morphology of field mice from Elsewhere"


def _crossref_item(doi, title, family, year):
    return {"DOI": doi, "title": [title], "author": [{"family": family}], "issued": {"date-parts": [[year]]}}


def fake_web(request: httpx.Request) -> httpx.Response:
    url = urlsplit(str(request.url))
    q = parse_qs(url.query)
    host, path = url.netloc, url.path
    if host == "api.crossref.org" and path == "/works":
        text = q["query.bibliographic"][0]
        items = (
            [_crossref_item("10.1000/A", TITLE_A, "Doe", 2020)] if "Echolocation" in text
            else [_crossref_item("10.1000/b", TITLE_B, "Smith", 2021)] if "Skull" in text
            else [_crossref_item("10.1000/zzz", "Chiroptera", "Other", 1990)]
        )  # fmt: skip
        return httpx.Response(200, json={"message": {"items": items}})
    if host == "api.unpaywall.org":
        if path.endswith("10.1000/a"):
            loc = {"url_for_pdf": "https://repo.example.org/a.pdf", "version": "publishedVersion"}
            return httpx.Response(
                200, json={"best_oa_location": loc, "oa_locations": [loc], "oa_status": "green"}
            )
        return httpx.Response(404, json={"error": True})
    if host == "doi.org" and path == "/10.1000/b":
        html = '<html><head><meta name="citation_pdf_url" content="https://publisher.example.com/b.pdf"></head></html>'
        return httpx.Response(200, html=html)
    if host == "repo.example.org" and path == "/a.pdf":
        return httpx.Response(
            200, content=make_pdf(TITLE_A + FILLER), headers={"content-type": "application/pdf"}
        )
    if host == "publisher.example.com" and path == "/b.pdf":
        return httpx.Response(
            200, content=make_pdf(TITLE_B + FILLER), headers={"content-type": "application/pdf"}
        )
    if host == "api.openalex.org" and path == "/works":
        return httpx.Response(200, json={"results": []})
    if host == "www.ebi.ac.uk":
        return httpx.Response(200, json={"resultList": {"result": []}})
    if host == "api.semanticscholar.org" and path.endswith("/search"):
        return httpx.Response(200, json={"data": []})
    if host == "api.datacite.org":
        return httpx.Response(200, json={"data": []})
    if host == "api.crossref.org":  # One work
        return httpx.Response(200, json={"message": {}})
    return httpx.Response(404, text="not found")


def _setup(tmp_path: Path) -> Path:
    tsv = tmp_path / "ets_tsv"
    tsv.mkdir()
    write_ets(tsv / "ETS_Doe and Roe, 2020.tsv", [ets_row(references=CIT_A, verbatimTraitValue="1")])
    write_ets(tsv / "ETS_Doe and Roe, 2020 part 2.tsv", [ets_row(references=CIT_A, verbatimTraitValue="2")])
    write_ets(tsv / "ETS_Smith, 2021.tsv", [ets_row(references=CIT_B)])
    write_ets(tsv / "ETS_Nobody, 2019.tsv", [ets_row(references=CIT_C)])
    return tsv


def _client(tmp_path: Path, calls: list) -> PoliteClient:
    def handler(request):
        calls.append(str(request.url))
        return fake_web(request)

    return PoliteClient(
        email="test@example.org",
        cache_dir=tmp_path / "cache",
        delay=0,
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )


def test_fetch_end_to_end(tmp_path: Path):
    tsv = _setup(tmp_path)
    out = tmp_path / "pdfs"
    args = build_parser().parse_args([
        "--tsv-dir", str(tsv), "--out-dir", str(out), "--report", str(tmp_path / "report.tsv"),
        "--manifest", str(tmp_path / "manifest.tsv"), "--overrides", str(tmp_path / "none.tsv"),
        "--email", "test@example.org",
    ])  # fmt: skip
    calls: list[str] = []
    outcomes = {o.paper.key: o for o in run(args, client=_client(tmp_path, calls))}

    assert outcomes["Doe and Roe, 2020"].status == "downloaded"
    assert outcomes["Doe and Roe, 2020"].got.source == "unpaywall"
    assert outcomes["Smith, 2021"].status == "downloaded"
    # The citation_pdf_url on the landing page of the DOI
    assert outcomes["Smith, 2021"].got.source == "landing"
    assert outcomes["Nobody, 2019"].status == "unresolved"  # The code refuses the incorrect Crossref result
    assert outcomes["Doe and Roe, 2020"].oa_status == "green"
    assert (out / "Doe and Roe, 2020.pdf").read_bytes().startswith(b"%PDF-")
    assert not any(
        "repo.example.org" in c for c in calls[calls.index("https://repo.example.org/a.pdf") + 1 :]
    )

    manifest = (tmp_path / "manifest.tsv").read_text(encoding="utf-8").splitlines()
    assert manifest == ["tsv\tmarkdown", "ETS_Doe and Roe, 2020 part 2.tsv\tDoe and Roe, 2020.zip"]
    with (tmp_path / "report.tsv").open(encoding="utf-8") as fh:
        report = {r["key"]: r for r in csv.DictReader(fh, delimiter="\t")}
    assert (
        report["Doe and Roe, 2020"]["doi"] == "10.1000/a" and report["Doe and Roe, 2020"]["verified"] == "yes"
    )
    assert len(report["Smith, 2021"]["sha256"]) == 64
    assert report["Doe and Roe, 2020"]["oa_status"] == "green"

    calls.clear()  # Second run: the PDF files are there and the cache has the API answers, so there is no network traffic
    again = {o.paper.key: o.status for o in run(args, client=_client(tmp_path, calls))}
    assert again["Doe and Roe, 2020"] == "exists" and again["Nobody, 2019"] == "unresolved"
    assert calls == []


def test_overrides_split_and_fix(tmp_path: Path):
    tsv = _setup(tmp_path)
    ov = tmp_path / "doi_overrides.tsv"
    ov.write_text("tsv\tdoi\turl\tnote\nETS_Nobody, 2019.tsv\t10.1000/b\t\tfound by hand\n", encoding="utf-8")
    papers = {p.key: p for p in group_papers(tsv, ov)}
    assert papers["Nobody, 2019"].doi == "10.1000/b" and papers["Nobody, 2019"].overridden
    assert papers["Doe and Roe, 2020"].tsv_files == [
        "ETS_Doe and Roe, 2020 part 2.tsv",
        "ETS_Doe and Roe, 2020.tsv",
    ]


# ----------------------------------------------------------------------- browser
CHROMIUM = os.environ.get("BIOPARSER_CHROMIUM")


class _Site(BaseHTTPRequestHandler):
    PAGES: ClassVar[dict[str, str]] = {
        "/landing.html": '<div id="ot"><button id="onetrust-accept-btn-handler" onclick="this.parentNode.remove()">'
                         'Accept</button></div><p>Article</p><a href="/files/paper.pdf">Download PDF</a>',
        "/js.html": "<button onclick=\"setTimeout(()=>{location.href='/files/paper.pdf'},300)\">PDF</button>",
        "/meta.html": '<html><head><meta name="citation_pdf_url" content="/files/paper.pdf"></head><body>x</body></html>',
        # This page does the click of the person in the --manual mode
        "/human.html": "<p>Log in</p><script>setTimeout(()=>{location.href='/files/paper.pdf'},1500)</script>",
        "/nothing.html": "<p>No PDF here</p>",
    }  # fmt: skip

    def do_GET(self):
        if self.path == "/files/paper.pdf":
            body = make_pdf("Echolocation calls of cave bats" + FILLER)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", 'attachment; filename="paper.pdf"')
        elif self.path in self.PAGES:
            body = f"<!doctype html><html><body>{self.PAGES[self.path]}</body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
        else:
            body = b"no"
            self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.mark.skipif(not CHROMIUM, reason="set BIOPARSER_CHROMIUM to a Chromium/Chrome binary")
@pytest.mark.parametrize("page", ["/landing.html", "/js.html", "/meta.html"])
def test_browser_auto_mode(tmp_path: Path, page: str):
    pytest.importorskip("playwright")
    from bioparser_ft.fetch.browser import BrowserFetcher

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with BrowserFetcher(profile_dir=tmp_path / "profile", headless=True, executable_path=CHROMIUM) as bf:
            got = bf.fetch_auto(f"http://127.0.0.1:{server.server_port}{page}")
        assert got is not None, f"no PDF captured from {page}"
        data, url = got
        assert data.startswith(b"%PDF-") and url.endswith("/files/paper.pdf")
    finally:
        server.shutdown()


@pytest.mark.skipif(not CHROMIUM, reason="set BIOPARSER_CHROMIUM to a Chromium/Chrome binary")
def test_browser_manual_mode_waits_and_times_out(tmp_path: Path):
    pytest.importorskip("playwright")
    from bioparser_ft.fetch.browser import BrowserFetcher

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with BrowserFetcher(profile_dir=tmp_path / "profile", headless=True, executable_path=CHROMIUM) as bf:
            got = bf.fetch_manual(f"{base}/human.html", "test paper", timeout_s=30)
            assert got is not None and got[0].startswith(b"%PDF-")
            assert bf.fetch_manual(f"{base}/nothing.html", "no pdf", timeout_s=3) is None
    finally:
        server.shutdown()


def test_proxy_only_in_manual_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The library terms do not let programs download articles automatically.

    Only the manual pass uses the proxy and the saved login. The headless pass uses an empty profile.
    """
    from bioparser_ft.fetch import browser
    from bioparser_ft.fetch.citations import Paper
    from bioparser_ft.fetch.cli import Outcome, browser_pass
    from bioparser_ft.fetch.scholarly import Resolution

    opened: list[tuple[bool, Path, str]] = []

    class FakeBrowser:
        def __init__(self, *, profile_dir: Path, headless: bool, **kwargs):
            self.profile_dir, self.headless = profile_dir, headless

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def fetch_auto(self, url: str):
            opened.append((self.headless, self.profile_dir, url))

        def fetch_manual(self, url: str, label: str, timeout_s: float):
            opened.append((self.headless, self.profile_dir, url))

    monkeypatch.setattr(browser, "BrowserFetcher", FakeBrowser)
    prefix = "https://login.libproxy.helsinki.fi/login?url="
    profile = tmp_path / "profile"
    args = build_parser().parse_args(["--proxy-prefix", prefix, "--browser-profile", str(profile)])
    paper = Paper("Doe, 2020", "Doe, J., 2020. A title. Journal, 1, 1-2.", ["ETS_Doe, 2020.tsv"])
    outcome = Outcome(paper, Resolution(doi="10.1000/a"), status="not_found")

    browser_pass([outcome], args, manual=False)
    browser_pass([outcome], args, manual=True)
    (auto_headless, auto_profile, auto_url), (manual_headless, manual_profile, manual_url) = opened
    assert auto_headless and auto_url == "https://doi.org/10.1000/a"
    assert auto_profile != profile and not auto_profile.exists()  # a new profile, removed after the pass
    assert not manual_headless and manual_profile == profile
    assert manual_url == prefix + "https://doi.org/10.1000/a"


def test_report_json_safe(tmp_path: Path):
    """A report cell never contains a tab or a line break, so each paper has one row."""
    from bioparser_ft.fetch.cli import _cell

    assert _cell("a\tb\nc") == "a b c" and json.dumps(_cell(None)) == '""'


def test_oa_candidates_parses_every_source(tmp_path: Path):
    """The code reads the correct fields from Unpaywall, OpenAlex, Europe PMC, Semantic Scholar and Crossref."""
    from bioparser_ft.fetch.scholarly import Scholarly

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == "api.unpaywall.org":
            return httpx.Response(
                200,
                json={
                    "best_oa_location": {
                        "url_for_pdf": None,
                        "url_for_landing_page": "https://pub.example/x",
                        "version": "publishedVersion",
                    },
                    "oa_locations": [
                        {"url_for_pdf": "https://repo.example/x-am.pdf", "version": "acceptedVersion"}
                    ],
                },
            )
        if host == "api.openalex.org":
            return httpx.Response(
                200,
                json={
                    "best_oa_location": {
                        "pdf_url": "https://pub.example/x.pdf",
                        "version": "publishedVersion",
                        "landing_page_url": "https://pub.example/x",
                    },
                    "locations": [],
                    "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456"},
                },
            )
        if host == "www.ebi.ac.uk":
            return httpx.Response(
                200,
                json={
                    "resultList": {
                        "result": [
                            {
                                "pmcid": "PMC123456",
                                "fullTextUrlList": {
                                    "fullTextUrl": [
                                        {
                                            "documentStyle": "pdf",
                                            "availabilityCode": "OA",
                                            "url": "https://epmc.example/x.pdf",
                                        },
                                        {
                                            "documentStyle": "pdf",
                                            "availabilityCode": "S",
                                            "url": "https://paywall.example/x",
                                        },
                                    ]
                                },
                            }
                        ]
                    }
                },
            )
        if host == "api.semanticscholar.org":
            return httpx.Response(200, json={"openAccessPdf": {"url": "https://s2.example/x.pdf"}})
        if host == "api.crossref.org" and path.startswith("/works/"):
            return httpx.Response(
                200,
                json={
                    "message": {
                        "link": [
                            {
                                "URL": "https://pub.example/tdm/x.pdf",
                                "content-type": "application/pdf",
                                "content-version": "vor",
                            },
                            {"URL": "https://pub.example/tdm/x.xml", "content-type": "text/xml"},
                        ]
                    }
                },
            )
        return httpx.Response(404)

    client = PoliteClient(
        email="t@example.org", delay=0, transport=httpx.MockTransport(handler), sleep=lambda s: None
    )
    stages = [
        [(c.source, c.url) for c in batch]
        for batch in Scholarly(client, email="t@example.org").oa_candidates("10.1000/x")
    ]
    assert stages[0] == [
        ("openalex", "https://pub.example/x.pdf"),
        ("unpaywall", "https://repo.example/x-am.pdf"),
    ]
    stage1 = dict((u, s) for s, u in stages[1])
    assert stage1["https://epmc.example/x.pdf"] == "europepmc" and "https://paywall.example/x" not in stage1
    assert "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC123456&blobtype=pdf" in stage1
    assert stage1["https://s2.example/x.pdf"] == "semanticscholar"
    assert (
        stage1["https://pub.example/tdm/x.pdf"] == "crossref-link"
        and "https://pub.example/tdm/x.xml" not in stage1
    )
    assert [u for _, u in stages[2]] == ["https://doi.org/10.1000/x", "https://pub.example/x"]


# ----------------------------------------------------------------------- concurrency
def _args(tmp_path: Path, tsv: Path, *extra: str):
    return build_parser().parse_args([
        "--tsv-dir", str(tsv), "--out-dir", str(tmp_path / "pdfs"), "--report", str(tmp_path / "report.tsv"),
        "--manifest", str(tmp_path / "manifest.tsv"), "--overrides", str(tmp_path / "none.tsv"),
        "--email", "test@example.org", *extra,
    ])  # fmt: skip


@pytest.mark.parametrize("workers", ["1", "4"])
def test_workers_give_the_same_results(tmp_path: Path, workers: str):
    """The papers run at the same time, but the results and the report sequence do not change."""
    tsv = _setup(tmp_path)
    outcomes = run(_args(tmp_path, tsv, "--workers", workers), client=_client(tmp_path, []))
    assert [(o.paper.key, o.status) for o in outcomes] == [
        ("Doe and Roe, 2020", "downloaded"),
        ("Nobody, 2019", "unresolved"),
        ("Smith, 2021", "downloaded"),
    ]
    with (tmp_path / "report.tsv").open(encoding="utf-8") as fh:
        assert [r["key"] for r in csv.DictReader(fh, delimiter="\t")] == [o.paper.key for o in outcomes]


def test_error_in_one_paper_does_not_stop_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from bioparser_ft.fetch import cli

    real = cli.fetch_http

    def fetch_http(client, sch, paper, res, info=None):
        if paper.key.startswith("Smith"):
            raise RuntimeError("broken PDF parser")
        return real(client, sch, paper, res, info)

    monkeypatch.setattr(cli, "fetch_http", fetch_http)
    tsv = _setup(tmp_path)
    outcomes = {o.paper.key: o for o in run(_args(tmp_path, tsv), client=_client(tmp_path, []))}
    assert outcomes["Doe and Roe, 2020"].status == "downloaded"
    assert outcomes["Smith, 2021"].status == "not_found"
    assert "broken PDF parser" in outcomes["Smith, 2021"].notes[0]


def test_threads_share_the_rate_limit_of_a_host():
    """Two requests to one host start 0.1 s apart or more. Two different hosts do not wait for each other."""
    starts: dict[str, list[float]] = {}
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            starts.setdefault(request.url.host, []).append(time.monotonic())
        return httpx.Response(200, json={})

    client = PoliteClient(delay=0.1, transport=httpx.MockTransport(handler))
    urls = [f"https://{host}/{i}" for host in ("a.example", "b.example") for i in range(4)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(client.get_json, urls)) == [{}] * 8
    for host in ("a.example", "b.example"):
        t = sorted(starts[host])
        assert all(later - earlier >= 0.09 for earlier, later in itertools.pairwise(t))
    assert abs(min(starts["a.example"]) - min(starts["b.example"])) < 0.15


def test_server_rate_limit_and_retry_after():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/busy" and calls.count("/busy") == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(
            200, json={}, headers={"X-Rate-Limit-Limit": "2", "X-Rate-Limit-Interval": "1s"}
        )

    slept: list[float] = []
    client = PoliteClient(delay=0.0, transport=httpx.MockTransport(handler), sleep=slept.append)
    assert client.get_json("https://api.crossref.org/works/10.1000/x") == {}
    assert client.delay_for("api.crossref.org") == 0.5  # The server gives 2 requests in 1 second
    assert client.get_json("https://slow.example/busy") == {}
    assert calls.count("/busy") == 2 and max(slept) > 6  # The client waited for the Retry-After time


def test_oa_candidates_parallel_stages_are_equal(tmp_path: Path):
    """With a thread pool, the API calls of one stage go out at the same time. The batches do not change."""
    from bioparser_ft.fetch.scholarly import Scholarly

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.unpaywall.org":
            loc = {"url_for_pdf": "https://repo.example/x.pdf", "version": "acceptedVersion"}
            return httpx.Response(200, json={"best_oa_location": loc, "oa_status": "green"})
        if host == "api.openalex.org":
            return httpx.Response(200, json={"best_oa_location": {"pdf_url": "https://pub.example/x.pdf"}})
        if host == "api.semanticscholar.org":
            return httpx.Response(200, json={"openAccessPdf": {"url": "https://s2.example/x.pdf"}})
        return httpx.Response(404)

    def stages(pool):
        client = PoliteClient(
            email="t@example.org", delay=0, transport=httpx.MockTransport(handler), sleep=lambda s: None
        )
        info: dict = {}
        sch = Scholarly(client, email="t@example.org", pool=pool)
        batches = [[(c.source, c.url) for c in b] for b in sch.oa_candidates("10.1000/x", info)]
        return batches, info["oa_status"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert stages(pool) == stages(None)
    assert stages(None)[1] == "green"


def test_headless_pass_uses_parallel_browsers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each headless browser has its own thread and its own empty profile."""
    from bioparser_ft.fetch import browser
    from bioparser_ft.fetch.citations import Paper
    from bioparser_ft.fetch.cli import Outcome, browser_pass
    from bioparser_ft.fetch.scholarly import Resolution

    seen: set[tuple[int, Path]] = set()
    lock = threading.Lock()

    class FakeBrowser:
        def __init__(self, *, profile_dir: Path, headless: bool, **kwargs):
            self.profile_dir = profile_dir

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def fetch_auto(self, url: str):
            with lock:
                seen.add((threading.get_ident(), self.profile_dir))
            time.sleep(0.05)  # The other threads take the next papers in this time

    monkeypatch.setattr(browser, "BrowserFetcher", FakeBrowser)
    args = build_parser().parse_args(["--browser-workers", "3", "--browser-profile", str(tmp_path / "p")])
    outcomes = [
        Outcome(
            Paper(f"P{i}, 2020", "c", [f"ETS_P{i}, 2020.tsv"]), Resolution(doi=f"10.1000/{i}"), "not_found"
        )
        for i in range(9)
    ]
    browser_pass(outcomes, args, manual=False)
    assert len({t for t, _ in seen}) == 3 and len({p for _, p in seen}) == 3
    assert tmp_path / "p" not in {p for _, p in seen}  # No headless browser uses the saved profile


@pytest.mark.skipif(not CHROMIUM, reason="set BIOPARSER_CHROMIUM to a Chromium/Chrome binary")
def test_parallel_headless_pass_with_chromium(tmp_path: Path):
    """Two real browsers, each in its own thread with its own Playwright, get four PDF files."""
    pytest.importorskip("playwright")
    from bioparser_ft.fetch.citations import Paper
    from bioparser_ft.fetch.cli import Outcome, browser_pass
    from bioparser_ft.fetch.scholarly import Resolution

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    title = "Echolocation calls of cave bats"
    try:
        args = build_parser().parse_args([
            "--browser-workers", "2", "--browser-executable", CHROMIUM,
            "--out-dir", str(tmp_path / "pdfs"), "--browser-profile", str(tmp_path / "profile"),
        ])  # fmt: skip
        outcomes = [
            Outcome(Paper(f"Bat {i}, 2020", "c", [f"ETS_Bat {i}, 2020.tsv"], url=f"{base}{page}"),
                    Resolution(title=title), "not_found")
            for i, page in enumerate(["/landing.html", "/js.html", "/meta.html", "/landing.html"])
        ]  # fmt: skip
        browser_pass(outcomes, args, manual=False)
    finally:
        server.shutdown()
    assert [o.status for o in outcomes] == ["downloaded"] * 4
    assert sorted(p.name for p in (tmp_path / "pdfs").glob("*.pdf")) == [
        f"Bat {i}, 2020.pdf" for i in range(4)
    ]
    assert not (tmp_path / "profile").exists()


def test_rate_limit_applies_to_each_redirect_hop():
    """doi.org only sends redirects. The publisher after the redirect keeps its own rate limit."""
    starts: dict[str, list[float]] = {}
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            starts.setdefault(request.url.host, []).append(time.monotonic())
        if request.url.host == "doi.org":
            return httpx.Response(302, headers={"Location": f"https://pub.example{request.url.path}"})
        return httpx.Response(200, html="<html>landing page</html>")

    client = PoliteClient(delay=0.3, transport=httpx.MockTransport(handler))
    with ThreadPoolExecutor(max_workers=3) as pool:
        pages = list(pool.map(client.get_web, [f"https://doi.org/10.1000/{i}" for i in range(3)]))
    assert [p.url for p in pages] == [f"https://pub.example/10.1000/{i}" for i in range(3)]
    for host, gap in (("doi.org", 0.18), ("pub.example", 0.28)):
        t = sorted(starts[host])
        assert all(later - earlier >= gap for earlier, later in itertools.pairwise(t)), host
