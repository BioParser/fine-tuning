"""This module finds the DOI of a citation and the open-access PDF locations of a DOI.

It uses scholarly APIs: Crossref, OpenAlex, Unpaywall, Europe PMC, Semantic Scholar,
DataCite and CORE.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from .citations import Paper, clean_doi
from .landing import publisher_pdf_urls
from .matching import CONFIDENCE_ORDER, Verdict, verify
from .net import PoliteClient

CROSSREF = "https://api.crossref.org/works"
OPENALEX = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
DATACITE = "https://api.datacite.org/dois"
CORE = "https://api.core.ac.uk/v3/search/works"

VERSION_RANK = {"publishedversion": 0, "acceptedversion": 1, "submittedversion": 2}


@dataclass(order=True)
class Candidate:
    """A URL that can give the PDF file.

    The kind is "pdf" for a direct file, or "landing" for an HTML page that the code parses.
    """

    rank: tuple = field(compare=True)
    url: str = field(compare=False)
    source: str = field(compare=False)
    kind: str = field(compare=False, default="pdf")


def make_candidate(url: str | None, source: str, *, kind: str = "pdf", version: str = "", stage: int = 0):
    if not url:
        return None
    # The client made the labels from the published article. For this reason, the version of record comes first.
    return Candidate((stage, VERSION_RANK.get(version.lower().replace(" ", ""), 1)), url, source, kind)


@dataclass
class Resolution:
    doi: str | None = None
    title: str = ""
    year: int | None = None
    source: str = ""  # One of: crossref, openalex, semanticscholar, datacite, core, citation, override
    verdict: Verdict | None = None
    candidates: list[Candidate] = field(default_factory=list)  # URLs from the search, for items without a DOI
    landing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        if self.source in ("citation", "override"):
            return "given"
        return self.verdict.confidence if self.verdict else "none"


# --------------------------------------------------------------------- functions that read API fields
def _crossref_year(item: dict) -> int | None:
    for k in ("issued", "published-print", "published-online", "created"):
        parts = (item.get(k) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _first(xs: Any) -> str:
    return xs[0] if isinstance(xs, list) and xs else (xs or "")


class Scholarly:
    def __init__(self, client: PoliteClient, *, email: str | None = None, pool: Executor | None = None):
        self.c = client
        self.email = email
        self.pool = pool  # With a thread pool, the API calls of one stage go out at the same time
        self.s2_key = os.environ.get("S2_API_KEY")
        self.core_key = os.environ.get("CORE_API_KEY")
        self.openalex_key = os.environ.get("OPENALEX_API_KEY")

    def _mailto(self) -> dict:
        return {"mailto": self.email} if self.email else {}

    def _openalex_params(self, **params) -> dict:
        if self.openalex_key:
            params["api_key"] = self.openalex_key
        return params | self._mailto()

    def _parallel(self, *calls: Callable[[], Any]) -> list[Any]:
        """Do the calls at the same time, and return the results in the sequence of the calls.

        The calls go to different hosts. The client keeps the rate limit of each host.
        """
        if self.pool is None:
            return [call() for call in calls]
        futures = [self.pool.submit(call) for call in calls]
        return [f.result() for f in futures]

    def _check(self, paper: Paper, title: str, year: int | None, authors: list[str]) -> Verdict:
        return verify(paper.citation, paper.title, paper.first_author, paper.year, title, year, authors)

    # --------------------------------------------------------------- find the DOI
    def resolve(self, paper: Paper) -> Resolution:
        """Return the DOI from the overrides or from the citation.

        If there is no DOI, return the best search result that agrees with the citation.
        The code always uses an override URL first.
        """
        if paper.doi:
            res = self._given_doi(paper)
        elif paper.url:
            res = Resolution(source="override", title=paper.title, year=paper.year, landing=[paper.url])
        else:
            res = self._search(paper)
        if paper.url and (c := make_candidate(paper.url, "override", stage=-1)):
            res.candidates.insert(0, c)
        return res

    def _given_doi(self, paper: Paper) -> Resolution:
        """For a known DOI, get the correct title from Crossref.

        Then the PDF check compares the PDF with the real title, not with the title guess
        from the citation.
        """
        item = (self.c.get_json(f"{CROSSREF}/{quote(paper.doi, safe='/')}", self._mailto()) or {}).get(
            "message"
        ) or {}
        return Resolution(
            doi=paper.doi,
            title=_first(item.get("title")) or paper.title,
            year=_crossref_year(item) or paper.year,
            source="override" if paper.overridden else "citation",
        )

    def _search(self, paper: Paper) -> Resolution:
        best: Resolution | None = None
        for finder in (self._crossref, self._openalex_search, self._s2_search, self._datacite, self._core):
            try:
                res = finder(paper)
            except Exception as exc:  # An error in one API must not stop the full run
                res = Resolution(notes=[f"{finder.__name__}: {exc!r}"])
            if res and res.verdict and res.verdict.accepted:
                if (
                    best is None
                    or CONFIDENCE_ORDER[res.verdict.confidence] > CONFIDENCE_ORDER[best.verdict.confidence]
                ):
                    best = res
                if best.verdict.confidence == "high" and (best.doi or best.candidates):
                    break
        return best or Resolution(
            notes=["no result agrees with the citation: add a DOI or URL to doi_overrides.tsv"]
        )

    def _crossref(self, paper: Paper) -> Resolution | None:
        data = self.c.get_json(CROSSREF, {"query.bibliographic": paper.citation, "rows": 5} | self._mailto())
        best = None
        for item in ((data or {}).get("message") or {}).get("items", []):
            title = _first(item.get("title"))
            authors = [a.get("family") or a.get("name") or "" for a in item.get("author", [])]
            v = self._check(paper, title, _crossref_year(item), authors)
            if v.accepted and (
                best is None or CONFIDENCE_ORDER[v.confidence] > CONFIDENCE_ORDER[best.verdict.confidence]
            ):
                best = Resolution(
                    doi=clean_doi(item["DOI"]),
                    title=title,
                    year=_crossref_year(item),
                    source="crossref",
                    verdict=v,
                )
        return best

    def _openalex_search(self, paper: Paper) -> Resolution | None:
        params = {"search": f"{paper.title} {paper.first_author}".strip(), "per-page": 5}
        if paper.year:
            params["filter"] = f"publication_year:{paper.year - 1}-{paper.year + 1}"
        data = self.c.get_json(OPENALEX, self._openalex_params(**params))
        best = None
        for w in (data or {}).get("results", []):
            authors = [(a.get("author") or {}).get("display_name", "") for a in w.get("authorships", [])]
            v = self._check(
                paper, w.get("display_name") or w.get("title") or "", w.get("publication_year"), authors
            )
            if not v.accepted or (
                best and CONFIDENCE_ORDER[v.confidence] <= CONFIDENCE_ORDER[best.verdict.confidence]
            ):
                continue
            res = Resolution(
                doi=clean_doi(w["doi"]) if w.get("doi") else None,
                title=w.get("display_name", ""),
                year=w.get("publication_year"),
                source="openalex",
                verdict=v,
            )
            res.candidates, res.landing = self._openalex_locations(w)
            best = res
        return best

    def _s2_search(self, paper: Paper) -> Resolution | None:
        headers = {"x-api-key": self.s2_key} if self.s2_key else None
        data = self.c.get_json(
            S2 + "search",
            {
                "query": paper.title[:300],
                "limit": 5,
                "fields": "title,year,authors,externalIds,openAccessPdf",
            },
            headers,
        )
        best = None
        for p in (data or {}).get("data", []) or []:
            v = self._check(
                paper,
                p.get("title") or "",
                p.get("year"),
                [a.get("name", "") for a in p.get("authors") or []],
            )
            if not v.accepted or (
                best and CONFIDENCE_ORDER[v.confidence] <= CONFIDENCE_ORDER[best.verdict.confidence]
            ):
                continue
            doi = (p.get("externalIds") or {}).get("DOI")
            pdf = (p.get("openAccessPdf") or {}).get("url")
            best = Resolution(
                doi=clean_doi(doi) if doi else None,
                title=p.get("title", ""),
                year=p.get("year"),
                source="semanticscholar",
                verdict=v,
                candidates=[c for c in [make_candidate(pdf, "semanticscholar", stage=1)] if c],
            )
        return best

    def _datacite(self, paper: Paper) -> Resolution | None:
        """Theses and repository items often have DataCite DOIs, for example in Figshare, Zenodo and DSpace.

        The code uses DataCite only when Crossref, OpenAlex and Semantic Scholar give no
        high-confidence result.
        """
        data = self.c.get_json(DATACITE, {"query": paper.title[:250], "page[size]": 5})
        best = None
        for d in (data or {}).get("data", []) or []:
            a = d.get("attributes") or {}
            title = _first([t.get("title", "") for t in a.get("titles") or []])
            authors = [c.get("familyName") or c.get("name", "") for c in a.get("creators") or []]
            year = int(a["publicationYear"]) if str(a.get("publicationYear", "")).isdigit() else None
            v = self._check(paper, title, year, authors)
            if v.accepted and (
                best is None or CONFIDENCE_ORDER[v.confidence] > CONFIDENCE_ORDER[best.verdict.confidence]
            ):
                best = Resolution(
                    doi=clean_doi(a.get("doi") or d.get("id", "")),
                    title=title,
                    year=year,
                    source="datacite",
                    verdict=v,
                    landing=[a["url"]] if a.get("url") else [],
                )
        return best

    def _core(self, paper: Paper) -> Resolution | None:
        """CORE collects the items of many institutional repositories, so it is good for theses.

        CORE operates only with the environment variable CORE_API_KEY.
        """
        if not self.core_key:
            return None
        data = self.c.get_json(
            CORE,
            {"q": f'title:"{paper.title[:200]}"', "limit": 5},
            {"Authorization": f"Bearer {self.core_key}"},
        )
        best = None
        for w in (data or {}).get("results", []) or []:
            authors = [a.get("name", "") for a in w.get("authors") or []]
            v = self._check(paper, w.get("title") or "", w.get("yearPublished"), authors)
            if v.accepted and (
                best is None or CONFIDENCE_ORDER[v.confidence] > CONFIDENCE_ORDER[best.verdict.confidence]
            ):
                best = Resolution(
                    doi=clean_doi(w["doi"]) if w.get("doi") else None,
                    title=w.get("title", ""),
                    year=w.get("yearPublished"),
                    source="core",
                    verdict=v,
                    candidates=[c for c in [make_candidate(w.get("downloadUrl"), "core", stage=1)] if c],
                )
        return best

    # --------------------------------------------------------------- PDF locations
    def _openalex_locations(self, work: dict) -> tuple[list[Candidate], list[str]]:
        cands, landing = [], []
        for loc in [work.get("best_oa_location"), *(work.get("locations") or [])]:
            if not loc:
                continue
            if c := make_candidate(loc.get("pdf_url"), "openalex", version=loc.get("version") or "", stage=0):
                cands.append(c)
            if loc.get("landing_page_url"):
                landing.append(loc["landing_page_url"])
        return cands, landing

    def oa_candidates(self, doi: str, info: dict | None = None) -> Iterator[list[Candidate]]:
        """Yield the candidates in batches, with the best sources first.

        When a batch gives a good PDF, the caller stops. Then the code does not call the other APIs.
        The function writes the open-access status of Unpaywall or OpenAlex to info["oa_status"]:
        gold, diamond, hybrid, bronze, green or closed.
        """
        info = {} if info is None else info
        landing: list[str] = []
        qdoi = quote(doi, safe="/")
        # Stage 0: Unpaywall and OpenAlex. They are the largest open-access indexes and they give the version.
        u, w = self._parallel(
            # Unpaywall operates only with a contact email
            lambda: self.c.get_json(UNPAYWALL + qdoi, {"email": self.email}) if self.email else None,
            lambda: self.c.get_json(f"{OPENALEX}/doi:{qdoi}", self._openalex_params()),
        )
        batch: list[Candidate] = []
        for loc in [(u or {}).get("best_oa_location"), *((u or {}).get("oa_locations") or [])]:
            if not loc:
                continue
            if c := make_candidate(loc.get("url_for_pdf"), "unpaywall", version=loc.get("version") or ""):
                batch.append(c)
            landing += [x for x in (loc.get("url_for_landing_page"), loc.get("url")) if x]
        info["oa_status"] = (
            (u or {}).get("oa_status") or ((w or {}).get("open_access") or {}).get("oa_status") or ""
        )
        if w:
            c2, l2 = self._openalex_locations(w)
            batch += c2
            landing += l2
            pmcid = ((w.get("ids") or {}).get("pmcid") or "").rstrip("/").rsplit("/", 1)[-1]
        else:
            pmcid = ""
        yield sorted(batch)

        # Stage 1: Europe PMC for PMC copies, Semantic Scholar, and the full-text links in Crossref.
        headers = {"x-api-key": self.s2_key} if self.s2_key else None
        e, s, cr = self._parallel(
            lambda: self.c.get_json(
                EUROPEPMC, {"query": f'DOI:"{doi}"', "format": "json", "resultType": "core", "pageSize": 1}
            ),
            lambda: self.c.get_json(S2 + f"DOI:{qdoi}", {"fields": "openAccessPdf"}, headers),
            lambda: self.c.get_json(f"{CROSSREF}/{qdoi}", self._mailto()),
        )
        batch = []
        for r in ((e or {}).get("resultList") or {}).get("result", [])[:1]:
            pmcid = pmcid or r.get("pmcid") or ""
            for f in (r.get("fullTextUrlList") or {}).get("fullTextUrl") or []:
                if f.get("documentStyle") == "pdf" and f.get("availabilityCode") in ("OA", "F"):
                    batch.append(make_candidate(f.get("url"), "europepmc", stage=1))
        if pmcid.upper().startswith("PMC"):
            batch.append(
                make_candidate(
                    f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf",
                    "europepmc",
                    stage=1,
                )
            )
            batch.append(
                make_candidate(f"https://europepmc.org/articles/{pmcid}?pdf=render", "europepmc", stage=1)
            )
        batch.append(
            make_candidate(((s or {}).get("openAccessPdf") or {}).get("url"), "semanticscholar", stage=1)
        )
        item = (cr or {}).get("message") or {}
        for link in item.get("link") or []:
            if link.get("content-type") == "application/pdf":
                batch.append(
                    make_candidate(
                        link.get("URL"),
                        "crossref-link",
                        version="publishedversion" if link.get("content-version") == "vor" else "",
                        stage=1,
                    )
                )
        for url in publisher_pdf_urls(doi, item):
            batch.append(make_candidate(url, "publisher-pattern", stage=1))
        yield sorted(c for c in batch if c)

        # Stage 2: landing pages. The doi.org page comes first, because it opens the page of the publisher.
        seen, pages = set(), []
        for url in [f"https://doi.org/{doi}", *landing]:
            if url not in seen:
                seen.add(url)
                pages.append(make_candidate(url, "landing", kind="landing", stage=2))
        yield [c for c in pages if c]
